from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.rollout.sglang_rollout import GenerateState, _extract_rollout_top_p_token_data
from slime.utils import http_utils
from slime.utils.prompt_equal import PROMPT_EQUAL_LOSS_ESTIMATORS
from slime.utils.types import Sample

from . import deepsearch_world as dsw
from .cut_bill import local_search_schema as cut_bill_search_schema
from .cut_bill import run_search as run_cut_bill_search
from .cut_bill import system_prompt as cut_bill_system_prompt
from .env import FusedEnvironment, _format_retrieval, normalize_task, resolve_task_mode
from .history import messages_without_historical_thinking as _messages_without_historical_thinking
from .history import strip_trailing_chat_template_stop as _strip_trailing_chat_template_stop
from .parser import Gemma4ToolParser, ToolCall, make_tool_parser
from .agentcpm_explore import build_messages as build_agentcpm_explore_messages
from .prompts import (
    COT_SYSTEM_PROMPT,
    COT_USER_PROMPT,
    RAG_USER_PROMPT,
    FUSED_MCP_SYSTEM_PROMPT,
    FUSED_MCP_USER_PROMPT,
    FUSED_CLI_SYSTEM_PROMPT,
    FUSED_CLI_USER_PROMPT,
    FUSED_ET_SYSTEM_PROMPT,
    FUSED_ET_USER_PROMPT,
    FUSED_SEARCH_LONG_USER_PROMPT,
    FUSED_SEARCH_SYSTEM_PROMPT,
    FUSED_SEARCH_USER_PROMPT,
    FUSED_UNIFIED_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    REACT_USER_PROMPT,
    build_system_prompt,
    finish_schema,
    normalize_harness,
)
from .rllm_deepresearch import SEARCH_SYSTEM_PROMPT as RLLM_DR_SEARCH_SYSTEM_PROMPT
from .rllm_deepresearch import local_search_schema, run_search as run_rllm_deepresearch_search
from .search_gym import SEARCH_GYM_SYSTEM_PROMPT, SEARCH_GYM_USER_PROMPT

logger = logging.getLogger(__name__)
DEFAULT_SGLANG_CONTEXT_LENGTH_MARGIN = 256
_LAST_SGLANG_REQUEST_LOG_TS = 0.0
_EVAL_ENGINE_POOL_LOOP: asyncio.AbstractEventLoop | None = None
_EVAL_ENGINE_POOL: EvalSessionEnginePool | None = None


class SGLangContextLengthExceededError(ValueError):
    pass


class SGLangWeightVersionError(RuntimeError):
    pass


def _routing_headers(args, session_id: str) -> dict[str, str] | None:
    if getattr(args, "router_policy", None) in {"consistent_hashing", "manual"}:
        return {"X-SMG-Routing-Key": session_id}
    return None


class EvalSessionEnginePool:
    def __init__(
        self,
        urls: list[str],
        *,
        max_sessions_per_engine: int | None = None,
        control_concurrency: int | None = None,
    ):
        self._active = {url.rstrip("/"): 0 for url in urls}
        self._blocked_until = {url.rstrip("/"): 0.0 for url in urls}
        self._tie_breaker = 0
        self._max_sessions_per_engine = max_sessions_per_engine
        default_control_concurrency = max(1, min(64, len(self._active) * 4))
        self.control_concurrency = control_concurrency or default_control_concurrency
        self.control_semaphore = asyncio.Semaphore(self.control_concurrency)
        self._control_client: httpx.AsyncClient | None = None
        self._background_tasks: set[asyncio.Task] = set()

    def control_client(self) -> httpx.AsyncClient:
        if self._control_client is None:
            keepalive_expiry = max(0.1, float(os.environ.get("SLIME_HTTP_KEEPALIVE_EXPIRY", "4")))
            self._control_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.control_concurrency,
                    # SGLang control calls are tiny and infrequent relative to
                    # generation. Avoid reusing a server-closed Uvicorn socket
                    # after long agent/tool gaps; concurrency remains unchanged.
                    max_keepalive_connections=0,
                    keepalive_expiry=keepalive_expiry,
                ),
                timeout=httpx.Timeout(None),
                trust_env=False,
            )
        return self._control_client

    async def post_control(self, url: str, payload: dict[str, Any], *, max_retries: int) -> Any:
        return await http_utils._post(
            self.control_client(),
            url,
            payload,
            max_retries=max_retries,
        )

    def run_control_in_background(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def acquire(self) -> str | None:
        if not self._active:
            return None
        now = time.monotonic()
        eligible = [
            url
            for url, count in self._active.items()
            if self._blocked_until[url] <= now
            and (self._max_sessions_per_engine is None or count < self._max_sessions_per_engine)
        ]
        if not eligible:
            return None
        minimum = min(self._active[url] for url in eligible)
        candidates = [url for url in eligible if self._active[url] == minimum]
        url = candidates[self._tie_breaker % len(candidates)]
        self._tie_breaker += 1
        self._active[url] += 1
        return url

    def release(self, url: str) -> None:
        normalized = url.rstrip("/")
        if normalized in self._active:
            self._active[normalized] = max(0, self._active[normalized] - 1)

    def record_transport_failure(self, url: str, cooldown: float = 15.0) -> bool:
        normalized = url.rstrip("/")
        if normalized not in self._blocked_until:
            return True
        now = time.monotonic()
        was_available = self._blocked_until[normalized] <= now
        self._blocked_until[normalized] = max(self._blocked_until[normalized], now + cooldown)
        return was_available


def _get_eval_engine_pool(args) -> EvalSessionEnginePool | None:
    global _EVAL_ENGINE_POOL_LOOP, _EVAL_ENGINE_POOL
    urls = list(getattr(args, "sglang_engine_urls", None) or [])
    if not urls:
        return None
    loop = asyncio.get_running_loop()
    normalized_urls = {url.rstrip("/") for url in urls}
    if (
        _EVAL_ENGINE_POOL_LOOP is not loop
        or _EVAL_ENGINE_POOL is None
        or set(_EVAL_ENGINE_POOL._active) != normalized_urls
    ):
        _EVAL_ENGINE_POOL_LOOP = loop
        max_sessions_per_engine = max(
            1,
            min(
                int(getattr(args, "sglang_server_concurrency", 48)),
                int(getattr(args, "sglang_max_running_requests", 1 << 30)),
            ),
        )
        _EVAL_ENGINE_POOL = EvalSessionEnginePool(
            urls,
            max_sessions_per_engine=max_sessions_per_engine,
        )
    return _EVAL_ENGINE_POOL


@dataclass
class EvalSGLangSession:
    args: Any
    session_id: str
    capacity: int
    enabled: bool
    opened: bool = False
    last_rid: str | None = None
    expected_prefix_ids: list[int] | None = None
    stable_prefix_length: int = 0
    session_turns: int = 0
    fallback_count: int = 0
    delta_tokens: int = 0
    full_tokens_avoided: int = 0
    engine_pool: EvalSessionEnginePool | None = None
    server_url: str | None = None

    @classmethod
    def for_eval(cls, args, session_id: str, capacity: int) -> EvalSGLangSession:
        engine_pool = _get_eval_engine_pool(args)
        enabled = _env_bool("SLIME_FUSED_EVAL_USE_SGLANG_SESSION", False) and engine_pool is not None
        return cls(
            args=args,
            session_id=session_id,
            capacity=max(1, capacity),
            enabled=enabled,
            engine_pool=engine_pool,
        )

    async def prepare_request(self, full_prompt_ids: list[int]) -> tuple[list[int], dict[str, Any] | None]:
        if not self.enabled:
            return full_prompt_ids, None
        if not self.opened and not await self._open():
            return full_prompt_ids, None

        if self.expected_prefix_ids is None:
            request_ids = full_prompt_ids
        elif _has_token_prefix(full_prompt_ids, self.expected_prefix_ids):
            prefix_length = len(self.expected_prefix_ids)
            request_ids = full_prompt_ids[prefix_length:]
            if not request_ids:
                await self.disable("empty prompt delta")
                return full_prompt_ids, None
            self.delta_tokens += len(request_ids)
            self.full_tokens_avoided += prefix_length
        else:
            # Chat templates commonly normalize the generated assistant stop
            # marker when rendering the next turn.  SGLang sessions support
            # replacing the unstable suffix from a verified offset, so retain
            # the session when the entire previous prompt is still unchanged.
            prefix_length = _common_token_prefix_length(full_prompt_ids, self.expected_prefix_ids)
            if prefix_length < self.stable_prefix_length or prefix_length == 0:
                await self.disable("rendered prompt changed before the previous prompt boundary")
                return full_prompt_ids, None
            request_ids = full_prompt_ids[prefix_length:]
            if not request_ids:
                await self.disable("empty prompt delta")
                return full_prompt_ids, None
            self.delta_tokens += len(request_ids)
            self.full_tokens_avoided += prefix_length
            return request_ids, {"id": self.session_id, "rid": self.last_rid, "offset": prefix_length}

        return request_ids, {"id": self.session_id, "rid": self.last_rid}

    async def record_response(
        self,
        *,
        full_prompt_ids: list[int],
        output_ids: list[int],
        output_text: str,
        rid: str | None,
    ) -> None:
        if not self.enabled:
            return
        if not rid:
            await self.disable("SGLang session response did not include a request id")
            return
        if output_text and not output_ids:
            await self.disable("SGLang session response did not include output token ids")
            return
        self.last_rid = rid
        self.expected_prefix_ids = [*full_prompt_ids, *output_ids]
        self.stable_prefix_length = len(full_prompt_ids)
        self.session_turns += 1

    async def disable(self, reason: str, *, log_warning: bool = True) -> None:
        if not self.enabled:
            return
        self.fallback_count += 1
        if log_warning:
            logger.warning("Disabling native SGLang session %s: %s", self.session_id, reason)
        await self.close()
        self.enabled = False

    async def _open(self) -> bool:
        assert self.engine_pool is not None
        self.server_url = self.engine_pool.acquire()
        if self.server_url is None:
            self.enabled = False
            return False
        url = f"{self.server_url}/open_session"
        try:
            async with self.engine_pool.control_semaphore:
                output = await asyncio.wait_for(
                    self.engine_pool.post_control(
                        url,
                        {
                            "capacity_of_str_len": self.capacity,
                            "session_id": self.session_id,
                            # Bound leaked server-side state if a transport
                            # failure prevents /close_session from arriving.
                            "timeout": float(os.environ.get("SLIME_FUSED_SESSION_IDLE_TIMEOUT", "600")),
                        },
                        max_retries=1,
                    ),
                    timeout=float(os.environ.get("SLIME_FUSED_SESSION_CONTROL_TIMEOUT", "120")),
                )
        except asyncio.CancelledError:
            self.engine_pool.release(self.server_url)
            self.server_url = None
            raise
        except Exception as exc:
            self.fallback_count += 1
            self.enabled = False
            should_log = True
            if isinstance(exc, httpx.TransportError):
                should_log = self.engine_pool.record_transport_failure(self.server_url)
            self.engine_pool.release(self.server_url)
            self.server_url = None
            if should_log:
                logger.warning(
                    "Native SGLang session unavailable for %s; using stateless generation: %s: %r",
                    self.session_id,
                    type(exc).__name__,
                    exc,
                )
            return False
        if output != self.session_id:
            self.fallback_count += 1
            self.enabled = False
            self.engine_pool.release(self.server_url)
            self.server_url = None
            logger.warning(
                "Native SGLang session returned unexpected id for %s; using stateless generation",
                self.session_id,
            )
            return False
        self.opened = True
        return True

    async def close(self, *, background: bool = False) -> None:
        if not self.opened:
            return
        self.opened = False
        assert self.engine_pool is not None and self.server_url is not None
        engine_pool = self.engine_pool
        server_url = self.server_url
        self.server_url = None
        engine_pool.release(server_url)

        async def close_server_session() -> None:
            try:
                async with engine_pool.control_semaphore:
                    await asyncio.wait_for(
                        engine_pool.post_control(
                            f"{server_url}/close_session",
                            {"session_id": self.session_id},
                            max_retries=2,
                        ),
                        timeout=float(os.environ.get("SLIME_FUSED_SESSION_CONTROL_TIMEOUT", "120")),
                    )
            except Exception as exc:
                should_log = True
                if isinstance(exc, httpx.TransportError):
                    should_log = engine_pool.record_transport_failure(server_url)
                if should_log:
                    logger.warning(
                        "Failed to close native SGLang session %s: %s: %r",
                        self.session_id,
                        type(exc).__name__,
                        exc,
                    )

        if background:
            engine_pool.run_control_in_background(close_server_session())
            return
        await close_server_session()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Run a fused-agent multi-turn workflow as a slime custom generate hook.

    This function is intentionally plugged into slime's stock
    ``sglang_rollout.generate_and_rm_group`` path, including the fully-async
    worker. It owns only per-sample agent interaction and returns train-ready
    ``Sample`` objects with token/logprob/loss-mask fields populated.
    """
    state = GenerateState(args)
    task = _task_from_sample(base_sample)
    harness = normalize_harness(os.environ.get("FUSED_HARNESS", getattr(args, "fused_harness", "gem")))
    rllm_deepresearch = harness == "rllm_deepresearch"
    cut_bill = harness == "cut_bill"
    research_search = rllm_deepresearch or cut_bill
    search_gym = harness == "search_gym"
    agentcpm_explore = harness == "agentcpm_explore"
    deepsearch_world = harness == "deepsearch_world"
    rag = harness == "rag"
    reasoning_only = harness in {"cot", "rag", "bare"}
    retrieval_max_results = int(os.environ.get("RETRIEVAL_MAX_RESULTS", "10" if research_search else "5"))
    env = FusedEnvironment(
        task,
        # Training and evaluation may intentionally use different retrieval services.
        retrieval_url=os.environ.get(
            "EVAL_RETRIEVAL_SERVER_URL" if evaluation else "TRAIN_RETRIEVAL_SERVER_URL",
            os.environ.get("RETRIEVAL_SERVER_URL"),
        ),
        retrieval_max_results=retrieval_max_results,
        enable_tools=not reasoning_only,
        search_gym=search_gym,
        agentcpm_explore=agentcpm_explore,
        deepsearch_world=deepsearch_world,
        rag=rag,
    )
    observation, info = env.reset()
    retrieved_context = ""
    rag_retrieval_info: dict[str, Any] = {}
    if rag:
        if env.mode != "web_search":
            env.close()
            raise ValueError(f"rag only supports web-search tasks, got task mode {env.mode!r}")
        retrieved_context, _, _, rag_retrieval_info = await env.step(
            ToolCall("web_search", {"query": observation, "max_results": retrieval_max_results})
        )
    deepsearch_task = observation
    if research_search and env.mode != "web_search":
        env.close()
        raise ValueError(f"{harness} only supports web-search tasks, got task mode {env.mode!r}")
    if deepsearch_world and env.mode != "web_search":
        env.close()
        raise ValueError(f"deepsearch_world only supports web-search tasks, got task mode {env.mode!r}")
    base_max_steps = int(
        os.environ.get(
            "DEEPSEARCH_WORLD_MAX_STEPS"
            if deepsearch_world
            else "CUT_BILL_MAX_TURNS" if cut_bill else "RLLM_DR_MAX_TURNS" if rllm_deepresearch else "FUSED_MAX_STEPS",
            "30" if deepsearch_world else "48" if research_search else getattr(args, "fused_max_steps", "16"),
        )
    )
    per_step_max_tokens = int(os.environ.get("PER_STEP_MAX_TOKENS", str(sampling_params.get("max_new_tokens", 2048))))
    disable_thinking = True if deepsearch_world else False if research_search else _env_bool("FUSED_DISABLE_THINKING", True)
    discard_historical_thinking = not disable_thinking and _env_bool("FUSED_DISCARD_HISTORICAL_THINKING", False)
    strict_tito = not evaluation and _env_bool("SLIME_FUSED_STRICT_TITO", False)
    if strict_tito and discard_historical_thinking:
        raise ValueError("Strict TiTO requires append-only history; historical thinking cannot be discarded")
    prompt_equal_loss = not evaluation and getattr(args, "advantage_estimator", "grpo") in PROMPT_EQUAL_LOSS_ESTIMATORS
    max_context_tokens = _effective_sglang_context_limit(args)
    global_max_tool_calls_per_turn = os.environ.get(
        "FUSED_MAX_TOOL_CALLS_PER_TURN", os.environ.get("MAX_TOOL_CALLS_PER_TURN")
    )
    if env.mode == "mcp":
        max_tool_calls_per_turn = int(
            os.environ.get("FUSED_MCP_MAX_TOOL_CALLS_PER_TURN", global_max_tool_calls_per_turn or "8")
        )
    else:
        max_tool_calls_per_turn = int(global_max_tool_calls_per_turn or "4")
    credit_assignment_enable = _env_bool("CREDIT_ASSIGNMENT_ENABLE", True)
    credit_assignment_tool_parser_error = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR", True
    )
    credit_assignment_think_parser_error = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_THINK_PARSER_ERROR", True
    )
    credit_assignment_repeated_search_query = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY", True
    )
    credit_assignment_too_many_tool_calls = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS", True
    )
    credit_assignment_search_bypass = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_SEARCH_BYPASS", True)
    credit_assignment_direct_submit_without_tool = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL", True
    )
    credit_assignment_mixed_tool_and_answer = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER", True
    )
    credit_assignment_tail_guard_early_stop = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_TAIL_GUARD_EARLY_STOP", True
    )
    credit_assignment_ngram_repetition = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_NGRAM_REPETITION", True
    )
    credit_assignment_max_turns = credit_assignment_enable and _env_bool("CREDIT_ASSIGNMENT_MAX_TURNS", True)
    credit_assignment_max_response_len = credit_assignment_enable and _env_bool(
        "CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN", True
    )
    credit_assignment_parser_error_token_window = int(
        os.environ.get("CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR_TOKEN_WINDOW", "256")
    )
    ngram_repetition_n = int(os.environ.get("CREDIT_ASSIGNMENT_NGRAM_REPETITION_N", "8"))
    ngram_repetition_threshold = float(os.environ.get("CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD", "0.35"))
    ngram_repetition_min_tokens = int(os.environ.get("CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS", "128"))
    repeated_search_max_strikes = max(1, int(os.environ.get("FUSED_REPEATED_SEARCH_MAX_STRIKES", "2")))
    detect_abnormal_trajectories = not evaluation and not research_search
    detect_eval_response_anomalies = evaluation and not deepsearch_world

    tools = [cut_bill_search_schema()] if cut_bill else [local_search_schema(), finish_schema()] if rllm_deepresearch else env.tools()
    model_name = getattr(state.tokenizer, "name_or_path", None) or getattr(args, "hf_checkpoint", None)
    parser = (
        dsw.DeepSearchWorldParser(valid_tools=_valid_tool_names(tools) | {"finish"})
        if deepsearch_world
        else make_tool_parser(
            model_name,
            valid_tools=_valid_tool_names(tools) | ({"finish"} if search_gym or cut_bill else set()),
        )
    )
    messages = (
        dsw.plan_messages(observation)
        if deepsearch_world
        else _initial_messages(
            harness,
            info.get("task_type", ""),
            observation,
            tools,
            model_name,
            tool_parser=parser,
            inline_tool_prompt=not _chat_template_accepts_native_tools(state.tokenizer),
            retrieved_context=retrieved_context,
        )
    )
    render_tools = None if deepsearch_world or cut_bill or agentcpm_explore else tools
    max_steps = (
        base_max_steps + 2
        if deepsearch_world
        else base_max_steps if research_search else _max_steps_for_mode(env.mode, base_max_steps)
    )
    manager = (
        None
        if evaluation
        else TrajectoryManager(
            fork_threshold_tokens=int(os.environ.get("SLIME_FUSED_FORK_THRESHOLD_TOKENS", "1024")),
            strict_append_only=strict_tito and _tito_model_type(model_name) == "gemma4",
        )
    )
    session_id = base_sample.session_id or uuid.uuid4().hex
    base_sample.session_id = session_id
    eval_sglang_session = (
        EvalSGLangSession.for_eval(
            args,
            session_id,
            max_context_tokens or int(getattr(args, "rollout_max_context_len", 0) or 65536),
        )
        if evaluation and not deepsearch_world
        else None
    )
    capture_eval_details = not evaluation or _should_capture_eval_trajectory(base_sample)

    final_reward = 0.0
    final_done = False
    last_info: dict[str, Any] = {}
    total_steps = 0
    total_tool_call_turns = 0
    trajectory_steps: list[dict[str, Any]] = []
    episode_start_time = time.time()
    episode_start_timestamp = _utc_timestamp()
    llm_time = 0.0
    env_time = 0.0
    pending_turns: list[dict[str, Any]] = []
    tito_prefix_ids: list[int] = []
    tito_messages_snapshot: list[dict[str, Any]] | None = None
    trajectory_weight_version: str | None = None
    require_weight_version = not evaluation and _env_bool("SLIME_FUSED_REQUIRE_WEIGHT_VERSION", False)
    last_search_query: str | None = None
    seen_rllm_calls: set[tuple[str, str]] = set()
    repeated_search_strikes = 0
    used_non_finish_tool = False
    final_response = ""
    last_finish_reason = "stop"
    compact_prompt_length = 0
    total_completion_tokens = 0
    credit_event: str | None = None
    credit_step_index: int | None = None
    recoverable_tool_parser_errors: list[str] = []
    recoverable_tool_parser_error_spans: list[tuple[int, int] | None] = []
    tool_parser_syntax_repairs: list[str] = []
    tool_parser_schema_coercions: list[str] = []
    eval_response_anomaly_info: dict[str, Any] = {}
    deepsearch_phase = "plan"
    deepsearch_state = dict(dsw.EMPTY_STATE)
    deepsearch_steps: list[dict[str, Any]] = []
    deepsearch_action_turns = 0
    try:
        for step_idx in range(max_steps):
            rollout_messages = (
                _messages_without_historical_thinking(messages, parser=parser)
                if discard_historical_thinking
                else messages
            )
            historical_thinking_discarded = rollout_messages != messages
            # Run the chat-template render off the event loop. The HF fast
            # tokenizer releases the GIL during tokenize, so offloading lets the
            # many concurrent trajectory coroutines actually overlap instead of
            # serializing on the single rollout event-loop thread (which
            # otherwise saturates one core and starves the SGLang engines).
            if evaluation:
                prompt_ids = await asyncio.to_thread(
                    _render_prompt_ids,
                    state.tokenizer,
                    rollout_messages,
                    tools=render_tools,
                    disable_thinking=disable_thinking,
                )
                prompt_context_start_idx = None
                tito_boundary_before = False
                context_delta_ids = []
                tito_context_reason = "evaluation"
            elif discard_historical_thinking and step_idx > 0:
                prompt_ids = await asyncio.to_thread(
                    _render_prompt_ids,
                    state.tokenizer,
                    rollout_messages,
                    tools=render_tools,
                    disable_thinking=disable_thinking,
                )
                prompt_context_start_idx = None
                if _has_token_prefix(prompt_ids, tito_prefix_ids):
                    # Nothing was discarded from the served prefix this step
                    # (e.g. no prior turn emitted a closed think block), so the
                    # current TiTO segment extends cleanly.
                    tito_boundary_before = False
                    context_delta_ids = list(prompt_ids[len(tito_prefix_ids) :])
                    tito_context_reason = "append_delta"
                else:
                    # The rewritten history is a new policy context. Start a fresh
                    # TiTO segment so its tokens/logprobs match the served prompt.
                    # Never downgrade to the assistant-tail recovery path here: the
                    # mismatch is a semantic history rewrite, not cosmetic drift.
                    tito_boundary_before = True
                    context_delta_ids = list(prompt_ids)
                    tito_context_reason = "historical_thinking_discard"
            elif tito_prefix_ids and tito_messages_snapshot is not None and _tito_model_type(model_name) == "gemma4":
                prompt_context_start_idx = None
                try:
                    context_delta_ids = await asyncio.to_thread(
                        _render_gemma4_tito_delta_ids,
                        state.tokenizer,
                        tito_messages_snapshot,
                        rollout_messages,
                        tito_prefix_ids,
                        tools=render_tools,
                        disable_thinking=disable_thinking,
                    )
                except Exception as exc:
                    if strict_tito:
                        raise RuntimeError("Strict Gemma4 TiTO incremental prompt construction failed") from exc
                    logger.exception("Gemma4 incremental TITO prompt construction failed; starting a new segment.")
                    prompt_ids = await asyncio.to_thread(
                        _render_prompt_ids,
                        state.tokenizer,
                        rollout_messages,
                        tools=render_tools,
                        disable_thinking=disable_thinking,
                    )
                    tito_boundary_before = True
                    context_delta_ids = list(prompt_ids)
                    tito_context_reason = "tito_incremental_tokenization_failed"
                else:
                    prompt_ids = list(tito_prefix_ids) + context_delta_ids
                    tito_boundary_before = False
                    tito_context_reason = "append_delta"
            elif tito_prefix_ids and tito_messages_snapshot is not None and _tito_model_type(model_name) in {
                "qwen3",
                "qwen3_5",
            }:
                prompt_context_start_idx = None
                try:
                    context_delta_ids = await asyncio.to_thread(
                        _render_qwen3_tito_delta_ids,
                        state.tokenizer,
                        tito_messages_snapshot,
                        rollout_messages,
                        tito_prefix_ids,
                        disable_thinking=disable_thinking,
                    )
                except Exception as exc:
                    if strict_tito:
                        raise RuntimeError("Strict TiTO incremental prompt construction failed") from exc
                    logger.exception("Qwen3-family incremental TITO prompt construction failed; starting a new segment.")
                    prompt_ids = await asyncio.to_thread(
                        _render_prompt_ids,
                        state.tokenizer,
                        rollout_messages,
                        tools=render_tools,
                        disable_thinking=disable_thinking,
                    )
                    tito_boundary_before = True
                    context_delta_ids = list(prompt_ids)
                    tito_context_reason = "tito_incremental_tokenization_failed"
                else:
                    prompt_ids = list(tito_prefix_ids) + context_delta_ids
                    tito_boundary_before = False
                    tito_context_reason = "append_delta"
            else:
                prompt_ids = await asyncio.to_thread(
                    _render_prompt_ids,
                    state.tokenizer,
                    rollout_messages,
                    tools=render_tools,
                    disable_thinking=disable_thinking,
                )
                prompt_context_start_idx = None
                tito_boundary_before = bool(tito_prefix_ids) and not _has_token_prefix(prompt_ids, tito_prefix_ids)
                if tito_boundary_before:
                    context_delta_ids = list(prompt_ids)
                    tito_context_reason = "prompt_prefix_mismatch"
                else:
                    context_delta_ids = list(prompt_ids[len(tito_prefix_ids) :])
                    tito_context_reason = "initial" if not tito_prefix_ids else "append_delta"
            if strict_tito and tito_boundary_before:
                raise RuntimeError(f"Strict TiTO rejected an unexpected segment boundary: {tito_context_reason}")
            if max_context_tokens and len(prompt_ids) >= max_context_tokens:
                final_done = True
                last_info = {
                    "termination_reason": "max_context_len_exceeded",
                    "prompt_tokens": len(prompt_ids),
                    "max_context_tokens": max_context_tokens,
                }
                break
            step_sampling = dict(sampling_params)
            if deepsearch_world:
                step_sampling.update(
                    {
                        "temperature": float(os.environ.get("DEEPSEARCH_WORLD_TEMPERATURE", "0.6")),
                        "top_p": float(os.environ.get("DEEPSEARCH_WORLD_TOP_P", "0.95")),
                        "max_new_tokens": min(
                            per_step_max_tokens,
                            int(os.environ.get("DEEPSEARCH_WORLD_MAX_TOKENS", "1024")),
                        ),
                    }
                )
            elif rllm_deepresearch:
                step_sampling.update(
                    {
                        "temperature": float(os.environ.get("RLLM_DR_TEMPERATURE", "1.0")),
                        "top_p": float(os.environ.get("RLLM_DR_TOP_P", "1.0")),
                        "top_k": int(os.environ.get("RLLM_DR_TOP_K", "-1")),
                        "repetition_penalty": 1.0,
                        "max_new_tokens": max(
                            0,
                            int(os.environ.get("RLLM_DR_MAX_TOKENS", "32768")) - len(prompt_ids),
                        ),
                    }
                )
            elif cut_bill:
                step_sampling.update(
                    {
                        "repetition_penalty": 1.0,
                        "max_new_tokens": per_step_max_tokens,
                    }
                )
            else:
                if evaluation and os.environ.get("FUSED_MODEL_SERIES", "").strip().lower() == "qwen3.5":
                    step_sampling.update(
                        {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "top_k": 20,
                            "min_p": 0.0,
                            "presence_penalty": 1.5,
                            "repetition_penalty": 1.0,
                        }
                    )
                step_sampling["max_new_tokens"] = max(
                    0, min(int(step_sampling.get("max_new_tokens", per_step_max_tokens)), per_step_max_tokens)
                )
            if max_context_tokens:
                step_sampling["max_new_tokens"] = min(
                    step_sampling["max_new_tokens"], max_context_tokens - len(prompt_ids)
                )
            if step_sampling["max_new_tokens"] <= 0:
                final_done = True
                last_info = {
                    "termination_reason": "max_context_len_exceeded",
                    "prompt_tokens": len(prompt_ids),
                    "max_context_tokens": max_context_tokens,
                }
                break

            llm_start = time.time()
            try:
                request_prompt_ids = prompt_ids
                session_params = None
                if eval_sglang_session is not None:
                    request_prompt_ids, session_params = await eval_sglang_session.prepare_request(prompt_ids)
                try:
                    output = await _call_sglang(
                        args,
                        request_prompt_ids,
                        step_sampling,
                        session_id=session_id,
                        evaluation=evaluation,
                        request_semaphore=state.semaphore if evaluation else None,
                        session_params=session_params,
                        context_token_count=len(prompt_ids),
                        server_url=eval_sglang_session.server_url if session_params is not None else None,
                        expected_weight_version=trajectory_weight_version,
                        require_weight_version=require_weight_version,
                    )
                except (asyncio.CancelledError, SGLangContextLengthExceededError):
                    raise
                except Exception as exc:
                    if eval_sglang_session is None or session_params is None:
                        raise
                    log_warning = True
                    if (
                        isinstance(exc, httpx.TransportError)
                        and eval_sglang_session.engine_pool is not None
                        and eval_sglang_session.server_url is not None
                    ):
                        log_warning = eval_sglang_session.engine_pool.record_transport_failure(
                            eval_sglang_session.server_url
                        )
                    await eval_sglang_session.disable(
                        f"session generation failed with {type(exc).__name__}: {exc!r}",
                        log_warning=log_warning,
                    )
                    output = await _call_sglang(
                        args,
                        prompt_ids,
                        step_sampling,
                        session_id=session_id,
                        evaluation=evaluation,
                        request_semaphore=state.semaphore if evaluation else None,
                        context_token_count=len(prompt_ids),
                        expected_weight_version=trajectory_weight_version,
                        require_weight_version=require_weight_version,
                    )
            except SGLangContextLengthExceededError as exc:
                final_done = True
                last_info = {
                    "termination_reason": "max_context_len_exceeded",
                    "prompt_tokens": len(prompt_ids),
                    "max_new_tokens": int(step_sampling.get("max_new_tokens", 0) or 0),
                    "max_context_tokens": max_context_tokens,
                    "error": str(exc),
                }
                break
            step_llm_time = time.time() - llm_start
            llm_time += step_llm_time
            if evaluation:
                output_ids = []
                output_logprobs = []
                response_loss_mask = None
                if eval_sglang_session is not None and session_params is not None:
                    await eval_sglang_session.record_response(
                        full_prompt_ids=prompt_ids,
                        output_ids=output.get("output_ids") or [],
                        output_text=output.get("text") or "",
                        rid=output.get("rid"),
                    )
                response = _strip_trailing_chat_template_stop(output["text"])
                parsed_actions = [] if reasoning_only else await asyncio.to_thread(parser.parse, response)
            else:
                output_ids = output["output_ids"]
                output_logprobs = output["output_logprobs"]
                observed_weight_version = output.get("weight_version")
                if trajectory_weight_version is None and observed_weight_version is not None:
                    trajectory_weight_version = str(observed_weight_version)
                # Decode, tool-call parsing, and the default loss mask are pure-Python
                # CPU work; bundle them into a single worker-thread hop so a wave of
                # trajectories finishing an LLM turn together cannot monopolize the
                # rollout event loop (which must stay free to dispatch env/LLM
                # requests for every other in-flight trajectory).
                decode_fn = _decode_step if reasoning_only else _decode_and_parse_step
                response, parsed_actions, response_loss_mask = await asyncio.to_thread(
                    decode_fn,
                    state.tokenizer,
                    None if reasoning_only else parser,
                    output_ids,
                    disable_thinking=disable_thinking,
                )
            if deepsearch_world:
                response = dsw.ensure_think_tags(response)
                parsed_actions = await asyncio.to_thread(parser.parse, response)
            _record_tool_parser_errors(
                args=args,
                base_sample=base_sample,
                session_id=session_id,
                rollout_step=step_idx,
                model_name=model_name,
                response=response,
                parser=parser,
                prompt_ids=prompt_ids,
                evaluation=evaluation,
            )
            if _has_recoverable_tool_schema_errors(parser, parsed_actions):
                recoverable_tool_parser_errors.extend(parser.last_schema_errors)
                recoverable_tool_parser_error_spans.extend(parser.last_schema_error_spans)
            tool_parser_syntax_repairs.extend(getattr(parser, "last_syntax_repairs", ()) or ())
            tool_parser_schema_coercions.extend(getattr(parser, "last_schema_coercions", ()) or ())
            final_response = response
            finish_reason = output["finish_reason"]
            last_finish_reason = finish_reason
            step_completion_tokens = output.get("completion_tokens")
            if step_completion_tokens is None:
                step_completion_tokens = await asyncio.to_thread(_encode_len, state.tokenizer, response)
            step_completion_tokens = int(step_completion_tokens or 0)
            total_completion_tokens += step_completion_tokens
            current_prompt_tokens = int(output.get("prompt_tokens", len(prompt_ids)))
            compact_prompt_length = max(
                0,
                current_prompt_tokens + step_completion_tokens - total_completion_tokens,
            )
            total_steps += 1
            if any(action.name != "finish" for action in parsed_actions):
                total_tool_call_turns += 1

            assistant_content = response
            if research_search and parsed_actions:
                assistant_content = re.sub(
                    r"<tool_call>.*?</tool_call>", "", assistant_content, flags=re.DOTALL
                ).strip()
                assistant_content = re.sub(
                    r"<function=[^>]+>.*?(?:</function>|$)", "", assistant_content, flags=re.DOTALL
                ).strip()
            assistant_msg = {"role": "assistant", "content": assistant_content}
            if research_search and parsed_actions:
                assistant_msg["tool_calls"] = [
                    {
                        "id": f"call_{step_idx}_{action_idx}",
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.arguments or {}, ensure_ascii=False),
                        },
                    }
                    for action_idx, action in enumerate(parsed_actions)
                ]
            if not evaluation:
                pending_turns.append(
                    {
                        "turn": TurnRecord(
                            prompt_ids=prompt_ids,
                            output_ids=output_ids,
                            finish_reason="tool_calls" if parsed_actions else finish_reason,
                            output_log_probs=output_logprobs,
                            weight_version=(
                                str(observed_weight_version) if observed_weight_version is not None else None
                            ),
                            require_rollout_logprobs=True,
                            require_weight_version=require_weight_version,
                            context_delta_ids=context_delta_ids,
                            tito_boundary_before=tito_boundary_before,
                            tito_model_type=_tito_model_type(model_name),
                            tito_context_reason=tito_context_reason,
                            disable_thinking=disable_thinking,
                            loss_mask=response_loss_mask,
                            rollout_top_p_token_ids=output.get("rollout_top_p_token_ids"),
                            rollout_top_p_token_offsets=output.get("rollout_top_p_token_offsets"),
                            prompt_context_start_idx=prompt_context_start_idx,
                        ),
                        "prompt_messages": list(messages),
                        "response_message": assistant_msg,
                        "raw_response": response,
                        "metadata": {
                            "sid": session_id,
                            "step": step_idx,
                            "tito_context_reason": tito_context_reason,
                            "tito_context_delta_tokens": len(context_delta_ids),
                            "tito_boundary_before": tito_boundary_before,
                            "disable_thinking": disable_thinking,
                            "weight_version": observed_weight_version,
                        },
                    }
                )
                if tito_boundary_before:
                    tito_prefix_ids = context_delta_ids + list(output_ids)
                else:
                    # Extend in place: rebuilding the full prefix each turn is an
                    # O(n^2) copy over the trajectory and runs on the event loop.
                    tito_prefix_ids.extend(context_delta_ids)
                    tito_prefix_ids.extend(output_ids)
            messages.append(assistant_msg)
            if not evaluation:
                # The exact generated output ids already live in tito_prefix_ids.
                # Keep only the semantic message snapshot needed to prove that the
                # next request is append-only; never re-tokenize this assistant turn.
                tito_messages_snapshot = copy.deepcopy(messages)

            if deepsearch_world and deepsearch_phase == "plan":
                deepsearch_state = dsw.parse_plan(response)
                trajectory_steps.append(
                    _episode_step(
                        observation="",
                        response=response,
                        action="planning",
                        reward=0.0,
                        done=False,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=False,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                deepsearch_phase = "action"
                messages = dsw.action_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
                continue

            if deepsearch_world and deepsearch_phase == "action":
                deepsearch_action_turns += 1
            elif deepsearch_world and deepsearch_phase == "end":
                if not parsed_actions:
                    parsed_actions = [ToolCall("finish", {"command": "submit", "result": response.strip()})]

            if detect_abnormal_trajectories and finish_reason == "length":
                final_done = True
                last_info = {"termination_reason": "max_response_len_exceeded"}
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break

            if reasoning_only:
                env.answer = response
                final_reward = env.compute_final_reward(require_tool_evidence=False)
                final_done = True
                last_info = {
                    "termination_reason": "reasoning_only",
                    "reward_debug": env.reward_debug,
                    **rag_retrieval_info,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=final_reward,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break

            actions = parsed_actions
            if detect_eval_response_anomalies:
                anomaly_info = await asyncio.to_thread(
                    _eval_response_anomaly_info,
                    response,
                    actions,
                    state.tokenizer,
                    allow_gemma4_native_tool_response_handoff=_tito_model_type(model_name) == "gemma4",
                    allow_leading_think_close=not disable_thinking,
                    allow_action_terminated_think=search_gym or agentcpm_explore,
                    ngram_n=ngram_repetition_n,
                    ngram_threshold=ngram_repetition_threshold,
                    ngram_min_tokens=ngram_repetition_min_tokens,
                )
                if anomaly_info:
                    eval_response_anomaly_info = anomaly_info
                    final_reward = 0.0
                    final_done = True
                    last_info = {"termination_reason": "ABNORMAL_EVAL_RESPONSE", **anomaly_info}
                    trajectory_steps.append(
                        _episode_step(
                            observation=observation,
                            response=response,
                            action="".join(parser.format_action(action) for action in actions),
                            reward=0.0,
                            done=True,
                            messages=rollout_messages if capture_eval_details else [],
                            tito_context_reason=tito_context_reason,
                            historical_thinking_discarded=historical_thinking_discarded,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    break
            if research_search:
                if not actions:
                    env.answer = response
                    final_reward = env.compute_final_reward(require_tool_evidence=False)
                    final_done = True
                    last_info = {"termination_reason": "rllm_dr_no_tool_call", "reward_debug": env.reward_debug}
                    trajectory_steps.append(
                        _episode_step(
                            observation=observation,
                            response=response,
                            action="",
                            reward=final_reward,
                            done=True,
                            messages=rollout_messages if capture_eval_details else [],
                            tito_context_reason=tito_context_reason,
                            historical_thinking_discarded=historical_thinking_discarded,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    break

                if all(action.name == "finish" for action in actions):
                    env.answer = str(
                        actions[-1].arguments.get("result") or actions[-1].arguments.get("answer") or ""
                    )
                    final_reward = env.compute_final_reward(require_tool_evidence=False)
                    final_done = True
                    last_info = {"termination_reason": "env_done", "reward_debug": env.reward_debug}
                    trajectory_steps.append(
                        _episode_step(
                            observation=observation,
                            response=response,
                            action="".join(parser.format_action(action) for action in actions),
                            reward=final_reward,
                            done=True,
                            messages=rollout_messages if capture_eval_details else [],
                            tito_context_reason=tito_context_reason,
                            historical_thinking_discarded=historical_thinking_discarded,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    break

                if len(actions) >= 9:
                    final_done = True
                    last_info = {
                        "termination_reason": "rllm_dr_excessive_parallel_calls",
                        "excessive_parallel_calls": True,
                    }
                    break

                call_keys = [
                    (action.name, json.dumps(action.arguments or {}, ensure_ascii=False, sort_keys=True))
                    for action in actions
                ]
                if any(key in seen_rllm_calls for key in call_keys) or len(set(call_keys)) != len(call_keys):
                    final_done = True
                    last_info = {
                        "termination_reason": "cut_bill_duplicate_search" if cut_bill else "rllm_dr_duplicate_search",
                        "duplicate_search_detected": True,
                    }
                    break
                seen_rllm_calls.update(call_keys)

                batch_start = time.time()
                results = await asyncio.gather(
                    *(
                        (run_cut_bill_search if cut_bill else run_rllm_deepresearch_search)(
                            action,
                            retrieval_url=env.retrieval_url,
                            max_results=env.retrieval_max_results,
                        )
                        for action in actions
                    )
                )
                step_env_time = time.time() - batch_start
                env_time += step_env_time
                env.tool_calls += len(actions)
                raw_observations = [result[0] for result in results]
                env_infos = [result[1] for result in results]
                formatted_observations = [
                    _format_tool_observation(parser, action.name, raw_observation)
                    for action, raw_observation in zip(actions, raw_observations, strict=True)
                ]
                formatted_obs = "\n".join(formatted_observations)
                tool_failed = any(info.get("tool_return_error") for info in env_infos)
                refine_failed = any(info.get("refine_error") for info in env_infos)
                final_done = tool_failed or refine_failed
                last_info = {
                    **_merge_tool_infos(env_infos),
                    "termination_reason": (
                        "rllm_dr_tool_error"
                        if tool_failed
                        else "rllm_dr_refine_error" if refine_failed else "rllm_dr_tools_complete"
                    ),
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=formatted_obs,
                        response=response,
                        action="".join(parser.format_action(action) for action in actions),
                        reward=0.0,
                        done=final_done,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=step_env_time,
                        disable_thinking=disable_thinking,
                    )
                )
                if research_search:
                    messages.extend(
                        {
                            "role": "tool",
                            "tool_call_id": f"call_{step_idx}_{action_idx}",
                            "name": action.name,
                            "content": str(raw_observation),
                        }
                        for action_idx, (action, raw_observation) in enumerate(
                            zip(actions, raw_observations, strict=True)
                        )
                    )
                else:
                    _append_tool_observation_messages(
                        messages,
                        parser,
                        actions,
                        formatted_observations,
                        raw_observations,
                    )
                observation = formatted_obs
                for action in actions:
                    if action.name == "local_search":
                        env.web_search_queries.add(_normalize_search_query((action.arguments or {}).get("query", "")))
                if final_done:
                    break
                continue

            if (
                detect_abnormal_trajectories
                and credit_assignment_think_parser_error
                and _has_unclosed_think(response)
                and not _response_has_malformed_tool_call(response)
            ):
                final_reward = 0.0
                final_done = True
                credit_event = "think_parser_error"
                credit_step_index = len(pending_turns) - 1
                last_info = {
                    "termination_reason": "ABNORMAL_THINK_PARSE_ERROR",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "think_parser_error_count": 1,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break

            recoverable_tool_schema_errors = _has_recoverable_tool_schema_errors(parser, parsed_actions)
            if (
                detect_abnormal_trajectories
                and credit_assignment_tool_parser_error
                and getattr(parser, "last_schema_errors", ())
                and not recoverable_tool_schema_errors
            ):
                final_reward = 0.0
                final_done = True
                credit_event = "tool_parser_error"
                credit_step_index = len(pending_turns) - 1
                if isinstance(parser, Gemma4ToolParser):
                    error_span = next((span for span in parser.last_schema_error_spans if span is not None), None)
                    if error_span is None:
                        error_span = _parser_error_action_span(response)
                    error_attribution = "localized" if error_span is not None else "unattributable"
                else:
                    # XML/Qwen parser diagnostics currently do not expose a
                    # token-precise span; do not blame the whole parsed call.
                    error_span = None
                    error_attribution = "unattributable"
                await _mark_pending_turn_error_span(
                    state.tokenizer,
                    pending_turns[-1],
                    response,
                    error_span,
                    output_len=len(output_ids),
                    attribution=error_attribution,
                )
                last_info = {
                    "termination_reason": "ABNORMAL_PARSE_ERROR",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "credit_assignment_error_attribution": error_attribution,
                    "tool_parser_error_count": 1,
                    "tool_parser_errors": list(parser.last_schema_errors),
                    "tool_parser_error_kinds": list(getattr(parser, "last_schema_error_kinds", ())),
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break

            if not actions:
                if deepsearch_world:
                    deepsearch_steps.append(dsw.step_record(response))
                    trajectory_steps.append(
                        _episode_step(
                            observation="",
                            response=response,
                            action="",
                            reward=0.0,
                            done=False,
                            messages=rollout_messages if capture_eval_details else [],
                            tito_context_reason=tito_context_reason,
                            historical_thinking_discarded=False,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    if deepsearch_action_turns >= base_max_steps:
                        deepsearch_phase = "end"
                        messages = dsw.end_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
                    else:
                        messages = dsw.action_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
                    continue
                if detect_abnormal_trajectories and credit_assignment_tool_parser_error:
                    final_reward = 0.0
                    final_done = True
                    credit_event = "tool_parser_error"
                    credit_step_index = len(pending_turns) - 1
                    _record_tool_parser_errors(
                        args=args, base_sample=base_sample, session_id=session_id,
                        rollout_step=step_idx, model_name=model_name, response=response,
                        parser=parser, prompt_ids=prompt_ids, evaluation=evaluation,
                        fallback_errors=["no tool call could be parsed from the model response"],
                    )
                    error_span = _parser_error_action_span(response)
                    error_attribution = "localized" if error_span is not None else "unattributable"
                    await _mark_pending_turn_error_span(
                        state.tokenizer,
                        pending_turns[-1],
                        response,
                        error_span,
                        output_len=len(output_ids),
                        attribution=error_attribution,
                    )
                    last_info = {
                        "termination_reason": "ABNORMAL_PARSE_ERROR",
                        "credit_assignment_event": credit_event,
                        "credit_assignment_error_step_index": credit_step_index,
                        "credit_assignment_error_attribution": error_attribution,
                        "tool_parser_error_count": 1,
                    }
                    trajectory_steps.append(
                        _episode_step(
                            observation=observation,
                            response=response,
                            action="",
                            reward=0.0,
                            done=True,
                            messages=rollout_messages if capture_eval_details else [],
                            tito_context_reason=tito_context_reason,
                            historical_thinking_discarded=historical_thinking_discarded,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    break
                actions = [ToolCall("finish", {"command": "submit", "result": response})]
            elif (
                isinstance(parser, Gemma4ToolParser)
                and detect_abnormal_trajectories
                and credit_assignment_tool_parser_error
                and _response_has_malformed_tool_call(response)
                and not getattr(parser, "last_syntax_repairs", ())
            ):
                final_reward = 0.0
                final_done = True
                credit_event = "tool_parser_error"
                credit_step_index = len(pending_turns) - 1
                _record_tool_parser_errors(
                    args=args, base_sample=base_sample, session_id=session_id,
                    rollout_step=step_idx, model_name=model_name, response=response,
                    parser=parser, prompt_ids=prompt_ids, evaluation=evaluation,
                    fallback_errors=["malformed Gemma4 tool-call marker"],
                )
                error_span = _parser_error_action_span(response)
                error_attribution = "localized" if error_span is not None else "unattributable"
                await _mark_pending_turn_error_span(
                    state.tokenizer,
                    pending_turns[-1],
                    response,
                    error_span,
                    output_len=len(output_ids),
                    attribution=error_attribution,
                )
                last_info = {
                    "termination_reason": "ABNORMAL_PARSE_ERROR",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "credit_assignment_error_attribution": error_attribution,
                    "tool_parser_error_count": 1,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            if detect_abnormal_trajectories and max_tool_calls_per_turn > 0 and len(actions) > max_tool_calls_per_turn:
                final_reward = 0.0
                final_done = True
                if credit_assignment_too_many_tool_calls:
                    credit_event = "too_many_tool_calls"
                    credit_step_index = len(pending_turns) - 1
                    await _mark_pending_turn_error_span(
                        state.tokenizer,
                        pending_turns[-1],
                        response,
                        _actions_span(actions),
                        output_len=len(output_ids),
                    )
                last_info = {
                    "termination_reason": "ABNORMAL_TOOL_BURST",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "too_many_tool_call_count": 1,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action="",
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            if (
                detect_abnormal_trajectories
                and credit_assignment_mixed_tool_and_answer
                and _has_mixed_tool_and_answer(response, actions)
            ):
                final_reward = 0.0
                final_done = True
                credit_event = "mixed_tool_and_answer"
                credit_step_index = len(pending_turns) - 1
                await _mark_pending_turn_error_span(
                    state.tokenizer,
                    pending_turns[-1],
                    response,
                    _mixed_tool_and_answer_span(response, actions),
                    output_len=len(output_ids),
                )
                last_info = {
                    "termination_reason": "ABNORMAL_MIXED_TOOL_AND_ANSWER",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "mixed_tool_and_answer": True,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=parser.format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            direct_submit_without_tool = (
                not used_non_finish_tool
                and _requires_non_finish_tool(tools)
                and all(action.name == "finish" for action in actions)
            )
            if (
                detect_abnormal_trajectories
                and direct_submit_without_tool
                and (step_idx == 0 or credit_assignment_direct_submit_without_tool)
            ):
                final_reward = 0.0
                final_done = True
                credit_event = "direct_submit_without_tool"
                credit_step_index = len(pending_turns) - 1
                last_info = {
                    "termination_reason": "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "direct_submit_without_tool": True,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=parser.format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            repeated_action_span = None
            for action in actions:
                if not _is_web_search_tool(action.name):
                    continue
                query = _normalize_search_query((action.arguments or {}).get("query") or "")
                if not query:
                    continue
                if query == last_search_query:
                    repeated_search_strikes += 1
                    repeated_action_span = _actions_span([action])
                else:
                    repeated_search_strikes = 0
                    repeated_action_span = None
                last_search_query = query
            repeated_query = repeated_action_span is not None
            if repeated_query and (detect_abnormal_trajectories or evaluation):
                duplicate_search_info = {
                    "duplicate_search_detected": True,
                    "duplicate_query_count": repeated_search_strikes,
                    "duplicate_query_max_strikes": repeated_search_max_strikes,
                }
                if repeated_search_strikes < repeated_search_max_strikes:
                    obs = (
                        "Repeated search query detected. Use different keywords, split the question into a new "
                        "sub-query, or submit only if the existing evidence is sufficient."
                    )
                    formatted_obs = _format_tool_observation(parser, actions[0].name, obs)
                    last_info = duplicate_search_info
                    trajectory_steps.append(
                        _episode_step(
                            observation=formatted_obs,
                            response=response,
                            action=parser.format_action(actions[0]),
                            reward=0.0,
                            done=False,
                            messages=rollout_messages if capture_eval_details else [],
                            tito_context_reason=tito_context_reason,
                            historical_thinking_discarded=historical_thinking_discarded,
                            llm_time=step_llm_time,
                            env_time=0.0,
                            disable_thinking=disable_thinking,
                        )
                    )
                    _append_tool_observation_message(messages, parser, actions[0], formatted_obs, obs)
                    observation = formatted_obs
                    continue
                final_reward = 0.0
                final_done = True
                if detect_abnormal_trajectories and credit_assignment_repeated_search_query:
                    credit_event = "repeated_search_query"
                    credit_step_index = len(pending_turns) - 1
                    await _mark_pending_turn_error_span(
                        state.tokenizer,
                        pending_turns[-1],
                        response,
                        repeated_action_span,
                        output_len=len(output_ids),
                    )
                last_info = {
                    "termination_reason": (
                        "ABNORMAL_REPEATED_QUERY" if detect_abnormal_trajectories else "repeated_query_early_stop"
                    ),
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    **duplicate_search_info,
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=parser.format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            repeated_output = (
                await asyncio.to_thread(
                    _ngram_repetition_stats,
                    output_ids,
                    n=ngram_repetition_n,
                    min_tokens=ngram_repetition_min_tokens,
                )
                if detect_abnormal_trajectories
                else {"score": 0.0, "n": ngram_repetition_n, "total": 0, "unique": 0}
            )
            repeated_action_span = _actions_span(actions)
            if (
                detect_abnormal_trajectories
                and repeated_output["score"] > ngram_repetition_threshold
                and repeated_action_span is not None
            ):
                final_reward = 0.0
                final_done = True
                if credit_assignment_ngram_repetition:
                    credit_event = "ngram_repetition"
                    credit_step_index = len(pending_turns) - 1
                    await _mark_pending_turn_error_span(
                        state.tokenizer,
                        pending_turns[-1],
                        response,
                        repeated_action_span,
                        output_len=len(output_ids),
                    )
                last_info = {
                    "termination_reason": "ABNORMAL_NGRAM_REPETITION",
                    "credit_assignment_event": credit_event,
                    "credit_assignment_error_step_index": credit_step_index,
                    "ngram_repetition_detected": True,
                    "ngram_repetition_score": repeated_output["score"],
                    "ngram_repetition_n": repeated_output["n"],
                    "ngram_repetition_total": repeated_output["total"],
                    "ngram_repetition_unique": repeated_output["unique"],
                }
                trajectory_steps.append(
                    _episode_step(
                        observation=observation,
                        response=response,
                        action=parser.format_action(actions[0]),
                        reward=0.0,
                        done=True,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=0.0,
                        disable_thinking=disable_thinking,
                    )
                )
                break
            if env.mode in {"mcp", "web_search"} and _can_append_batched_tool_results(parser, actions):
                batch_start = time.time()
                formatted_observations: list[str] = []
                raw_observations: list[Any] = []
                executed_actions: list[ToolCall] = []
                env_infos: list[dict[str, Any]] = []
                batch_done = False
                batch_reward = 0.0
                for action in actions:
                    if action.name != "finish":
                        used_non_finish_tool = True
                    obs, reward, done, env_info = await env.step(action)
                    executed_actions.append(action)
                    raw_observations.append(obs)
                    formatted_observations.append(_format_tool_observation(parser, action.name, obs))
                    env_infos.append(dict(env_info or {}))
                    batch_reward = float(reward)
                    batch_done = bool(done)
                    if batch_done:
                        break
                step_env_time = time.time() - batch_start
                env_time += step_env_time
                formatted_obs = "\n".join(formatted_observations)
                final_reward = batch_reward
                final_done = batch_done
                last_info = _merge_tool_infos(env_infos)
                if recoverable_tool_schema_errors:
                    last_info.update(
                        {
                            "tool_parser_errors": list(parser.last_schema_errors),
                            "tool_parser_error_kinds": list(parser.last_schema_error_kinds),
                            "tool_parser_error_recoverable": True,
                        }
                    )
                trajectory_steps.append(
                    _episode_step(
                        observation=formatted_obs,
                        response=response,
                        action="".join(parser.format_action(action) for action in executed_actions),
                        reward=final_reward if batch_done else 0.0,
                        done=batch_done,
                        messages=rollout_messages if capture_eval_details else [],
                        tito_context_reason=tito_context_reason,
                        historical_thinking_discarded=historical_thinking_discarded,
                        llm_time=step_llm_time,
                        env_time=step_env_time,
                        disable_thinking=disable_thinking,
                    )
                )
                _append_tool_observation_messages(
                    messages, parser, executed_actions, formatted_observations, raw_observations
                )
                observation = formatted_obs
                if batch_done:
                    break
                if deepsearch_world:
                    deepsearch_steps.append(dsw.step_record(response, executed_actions[0], str(raw_observations[0])))
                    if deepsearch_action_turns >= base_max_steps:
                        deepsearch_phase = "end"
                        messages = dsw.end_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
                    else:
                        messages = dsw.action_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
                continue

            action = actions[0]
            if action.name != "finish":
                used_non_finish_tool = True
            env_start = time.time()
            obs, reward, done, env_info = await env.step(action)
            step_env_time = time.time() - env_start
            env_time += step_env_time
            formatted_obs = _format_tool_observation(parser, action.name, obs)
            final_reward = float(reward)
            final_done = bool(done)
            last_info = dict(env_info or {})
            if recoverable_tool_schema_errors:
                last_info.update(
                    {
                        "tool_parser_errors": list(parser.last_schema_errors),
                        "tool_parser_error_kinds": list(parser.last_schema_error_kinds),
                        "tool_parser_error_recoverable": True,
                    }
                )
            if detect_abnormal_trajectories and last_info.get("credit_assignment") == "reasoning_step_only":
                final_reward = 0.0
                if last_info.get("termination_reason") == "ABNORMAL_SEARCH_BYPASS" and credit_assignment_search_bypass:
                    credit_event = "search_bypass"
                    credit_step_index = len(pending_turns) - 1
                else:
                    credit_event = "reasoning_step_only"
                    credit_step_index = len(pending_turns) - 1
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason") == "ABNORMAL_SEARCH_BYPASS"
                and credit_assignment_search_bypass
            ):
                final_reward = 0.0
                credit_event = "search_bypass"
                credit_step_index = len(pending_turns) - 1
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason")
                in {"ABNORMAL_PARSE_ERROR", "INVALID_REACT_STRUCTURE", "INVALID_FINAL_STEP"}
                and credit_assignment_tool_parser_error
            ):
                credit_event = "tool_parser_error"
                credit_step_index = len(pending_turns) - 1
                _record_tool_parser_errors(
                    args=args, base_sample=base_sample, session_id=session_id,
                    rollout_step=step_idx, model_name=model_name, response=response,
                    parser=parser, prompt_ids=prompt_ids, evaluation=evaluation,
                    fallback_errors=[str(last_info.get("termination_reason"))],
                )
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason") == "ABNORMAL_NESTED_FINISH_PAYLOAD"
                and credit_assignment_tool_parser_error
            ):
                credit_event = "tool_parser_error"
                credit_step_index = len(pending_turns) - 1
                _record_tool_parser_errors(
                    args=args, base_sample=base_sample, session_id=session_id,
                    rollout_step=step_idx, model_name=model_name, response=response,
                    parser=parser, prompt_ids=prompt_ids, evaluation=evaluation,
                    fallback_errors=["ABNORMAL_NESTED_FINISH_PAYLOAD"],
                )
                await _mark_pending_turn_error_span(
                    state.tokenizer,
                    pending_turns[-1],
                    response,
                    _actions_span(actions),
                    output_len=len(output_ids),
                )
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason") == "ABNORMAL_TOOL_BURST"
                and credit_assignment_too_many_tool_calls
            ):
                credit_event = "too_many_tool_calls"
                credit_step_index = len(pending_turns) - 1
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason") == "ABNORMAL_REPEATED_QUERY"
                and credit_assignment_repeated_search_query
            ):
                credit_event = "repeated_search_query"
                credit_step_index = len(pending_turns) - 1
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason") == "ABNORMAL_NGRAM_REPETITION"
                and credit_assignment_ngram_repetition
            ):
                credit_event = "ngram_repetition"
                credit_step_index = len(pending_turns) - 1
            elif (
                detect_abnormal_trajectories
                and last_info.get("termination_reason") == "ABNORMAL_MIXED_TOOL_AND_ANSWER"
                and credit_assignment_mixed_tool_and_answer
            ):
                credit_event = "mixed_tool_and_answer"
                credit_step_index = len(pending_turns) - 1
            trajectory_steps.append(
                _episode_step(
                    observation=formatted_obs,
                    response=response,
                    action=parser.format_action(action),
                    reward=final_reward if done else 0.0,
                    done=done,
                    messages=rollout_messages if capture_eval_details else [],
                    tito_context_reason=tito_context_reason,
                    historical_thinking_discarded=historical_thinking_discarded,
                    llm_time=step_llm_time,
                    env_time=step_env_time,
                    disable_thinking=disable_thinking,
                )
            )
            _append_tool_observation_message(messages, parser, action, formatted_obs, obs)
            observation = formatted_obs
            if done:
                break
            if deepsearch_world:
                deepsearch_steps.append(dsw.step_record(response, action, str(obs)))
                if deepsearch_action_turns >= base_max_steps:
                    deepsearch_phase = "end"
                    messages = dsw.end_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
                else:
                    messages = dsw.action_messages(deepsearch_task, deepsearch_state, deepsearch_steps)
        else:
            last_info = {**last_info, "termination_reason": "max_turns_exceeded"}
            if detect_abnormal_trajectories and credit_assignment_max_turns and pending_turns:
                credit_event = "max_turns_exceeded"
                credit_step_index = len(pending_turns) - 1
                response = str(pending_turns[-1].get("raw_response", ""))
                action_span = _actions_span(await asyncio.to_thread(parser.parse, response))
                if action_span is not None:
                    await _mark_pending_turn_error_span(
                        state.tokenizer,
                        pending_turns[-1],
                        response,
                        action_span,
                        output_len=len(pending_turns[-1]["turn"].output_ids),
                    )

        if not final_done and final_reward == 0.0:
            final_reward = env.compute_final_reward()
            last_info = {"reward_debug": env.reward_debug, **last_info}
    finally:
        if eval_sglang_session is not None:
            await eval_sglang_session.close(background=True)
        env.close()

    termination_reason = last_info.get("termination_reason", "env_done" if final_done else "unknown")
    if (
        detect_abnormal_trajectories
        and termination_reason == "TAIL_GUARD_EARLY_STOP"
        and credit_assignment_tail_guard_early_stop
    ):
        credit_event = "tail_guard_early_stop"
    elif (
        detect_abnormal_trajectories
        and termination_reason == "max_response_len_exceeded"
        and credit_assignment_max_response_len
        and pending_turns
    ):
        credit_event = "max_response_len_exceeded"
        credit_step_index = len(pending_turns) - 1
    if credit_event is not None and credit_step_index is not None:
        last_info["credit_assignment_event"] = credit_event
        last_info["credit_assignment_error_step_index"] = credit_step_index

    episode_end_time = time.time()
    episode_timing = {
        "start_timestamp": episode_start_timestamp,
        "end_timestamp": _utc_timestamp(),
        "llm_time": llm_time,
        "env_time": env_time,
        "reward_time": 0.0,
        "total_time": episode_end_time - episode_start_time,
    }
    reward_debug = env.reward_debug or last_info.get("reward_debug", {})
    tito_reasons = [
        str(item["metadata"].get("tito_context_reason"))
        for item in pending_turns
        if item.get("metadata", {}).get("tito_context_reason") is not None
    ]
    rollout_versions = sorted(
        {str(item["turn"].weight_version) for item in pending_turns if item["turn"].weight_version is not None}
    )
    common_metadata = {
        **dict(base_sample.metadata or {}),
        **last_info,
        "fused_task_type": env.mode,
        "fused_reward_debug": reward_debug,
        "fused_termination": termination_reason,
        "credit_assignment_event": credit_event,
        "credit_assignment_error_step_index": credit_step_index,
        "fused_traj_steps": total_steps,
        "fused_tool_call_turns": total_tool_call_turns,
        "fused_prompt_length_tokens": compact_prompt_length,
        "fused_completion_length_tokens": total_completion_tokens,
        "rollout_weight_version": trajectory_weight_version,
        "fused_rollout_weight_versions": rollout_versions,
        "fused_rollout_weight_version_count": len(rollout_versions),
        "fused_rollout_logprob_invalid_ratio": 0.0,
        "fused_tito_boundary_count": sum(bool(item["turn"].tito_boundary_before) for item in pending_turns),
        "fused_tito_exact_prefix_turns": sum(reason in {"initial", "append_delta"} for reason in tito_reasons),
        "fused_tito_incremental_turns": tito_reasons.count("append_delta"),
        "fused_tito_prompt_prefix_mismatch_turns": tito_reasons.count("prompt_prefix_mismatch"),
        "fused_tito_incremental_tokenization_failed_turns": tito_reasons.count("tito_incremental_tokenization_failed"),
    }
    if recoverable_tool_parser_errors:
        common_metadata.update(
            {
                "tool_parser_error_recoverable": True,
                "tool_parser_errors": recoverable_tool_parser_errors,
                "tool_parser_error_kinds": ["unknown_parameter"] * len(recoverable_tool_parser_errors),
                "tool_parser_error_spans": recoverable_tool_parser_error_spans,
            }
        )
    if tool_parser_syntax_repairs:
        common_metadata["tool_parser_syntax_repairs"] = tool_parser_syntax_repairs
    if tool_parser_schema_coercions:
        common_metadata["tool_parser_schema_coercions"] = tool_parser_schema_coercions
    if prompt_equal_loss:
        instance_id = (base_sample.metadata or {}).get("instance_id")
        if instance_id is None:
            instance_id = base_sample.group_index if base_sample.group_index is not None else base_sample.index
        common_metadata.update(
            {"prompt_equal_loss": True, "parent_traj_id": session_id, "instance_id": str(instance_id)}
        )
    if eval_sglang_session is not None:
        common_metadata.update(
            {
                "fused_sglang_session_turns": eval_sglang_session.session_turns,
                "fused_sglang_session_fallbacks": eval_sglang_session.fallback_count,
                "fused_sglang_session_delta_tokens": eval_sglang_session.delta_tokens,
                "fused_sglang_session_full_tokens_avoided": eval_sglang_session.full_tokens_avoided,
            }
        )

    if evaluation:
        common_metadata["segment_count"] = 1
        should_dump_episode = capture_eval_details or _is_failed_eval_termination(termination_reason)
        if should_dump_episode:
            common_metadata["rllm_episode"] = _rllm_episode_dict(
                base_sample=base_sample,
                task=task,
                session_id=session_id,
                reward=final_reward,
                termination_reason=termination_reason,
                reward_debug=reward_debug,
                credit_event=credit_event,
                credit_step_index=credit_step_index,
                response_anomaly_info=eval_response_anomaly_info,
                total_steps=total_steps,
                total_tool_call_turns=total_tool_call_turns,
                timing=episode_timing,
                steps=trajectory_steps,
                task_type=env.mode,
                discard_historical_thinking_enabled=discard_historical_thinking,
            )
            common_metadata["rllm_episode"]["metadata"]["segment_count"] = 1
        return [
            Sample(
                index=base_sample.index,
                group_index=base_sample.group_index,
                rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
                prompt=base_sample.prompt,
                label=base_sample.label,
                reward=final_reward,
                response=final_response,
                response_length=total_completion_tokens,
                status=Sample.Status.TRUNCATED if last_finish_reason == "length" else Sample.Status.COMPLETED,
                metadata=common_metadata,
                session_id=session_id,
            )
        ]

    assert manager is not None
    # Turn recording + trajectory assembly touch every token of the episode;
    # keep them off the event loop since whole waves of trajectories finish
    # (and hit this path) together.
    await asyncio.to_thread(
        _record_pending_turns,
        manager,
        session_id=session_id,
        pending_turns=pending_turns,
        credit_event=credit_event,
        credit_step_index=credit_step_index,
        parser_error_token_window=credit_assignment_parser_error_token_window,
    )

    episode_dict = _rllm_episode_dict(
        base_sample=base_sample,
        task=task,
        session_id=session_id,
        reward=final_reward,
        termination_reason=termination_reason,
        reward_debug=reward_debug,
        credit_event=credit_event,
        credit_step_index=credit_step_index,
        response_anomaly_info=eval_response_anomaly_info,
        total_steps=total_steps,
        total_tool_call_turns=total_tool_call_turns,
        timing=episode_timing,
        steps=trajectory_steps,
        task_type=env.mode,
        discard_historical_thinking_enabled=discard_historical_thinking,
    )
    samples = await asyncio.to_thread(
        manager.get_trajectory,
        session_id,
        base_sample=base_sample,
        reward=final_reward,
        allow_fully_masked=credit_event == "tail_guard_early_stop",
        extra_metadata={**common_metadata, "rllm_episode": episode_dict},
    )
    if not samples:
        episode_dict["metadata"]["segment_count"] = 0
        if prompt_equal_loss:
            episode_dict["metadata"].update(
                {"prompt_equal_loss": True, "parent_traj_id": session_id, "segment_count": 0}
            )
        failed = Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            reward=0.0,
            status=Sample.Status.FAILED,
            rollout_log_probs=[],
            # Dead-prompt contract (Dressage's mark_aborted_no_grad): a prompt
            # with zero trainable tokens must not count toward N_P in the
            # prompt-equal denominators nor contribute gradient.
            remove_sample=True,
            metadata={
                **dict(base_sample.metadata or {}),
                **(
                    {
                        "prompt_equal_loss": True,
                        "parent_traj_id": session_id,
                        "instance_id": common_metadata["instance_id"],
                        "segment_index": 0,
                    }
                    if prompt_equal_loss
                    else {}
                ),
                "fused_error": "empty_trajectory",
                "fused_task_type": env.mode,
                "fused_termination": termination_reason,
                "credit_assignment_event": credit_event,
                "credit_assignment_error_step_index": credit_step_index,
                "fused_traj_steps": total_steps,
                "fused_tool_call_turns": total_tool_call_turns,
                "segment_count": 0,
                "rllm_episode": episode_dict,
                **last_info,
            },
        )
        return failed
    segment_count = len(samples)
    episode_dict["metadata"]["segment_count"] = segment_count
    if prompt_equal_loss:
        # All sibling segments share this one episode dict (and the offline dump
        # dedupes by episode id), so mark the segment breakdown once, in place.
        episode_dict["metadata"].update(
            {
                "prompt_equal_loss": True,
                "parent_traj_id": session_id,
                "segment_count": segment_count,
            }
        )
    for segment_index, sample in enumerate(samples):
        sample.response = final_response
        # Segment lineage is stamped in every mode so filters/metrics can
        # aggregate at trajectory level.
        sample.metadata.update(
            {
                "parent_traj_id": session_id,
                "segment_index": segment_index,
                "segment_count": segment_count,
            }
        )
        if prompt_equal_loss:
            sample.metadata.update(
                {
                    "prompt_equal_loss": True,
                    "instance_id": common_metadata["instance_id"],
                }
            )
        # Keep raw rewards sparse for every segment mode. The reward processor
        # broadcasts the terminal anchor's advantage to all sibling segments.
        sample.reward = final_reward if segment_index == segment_count - 1 else 0.0
        sample.status = Sample.Status.COMPLETED
        if sample.rollout_log_probs is None:
            raise ValueError("fused training sample omitted rollout_log_probs")
    return samples


generate.manages_eval_request_concurrency = True


def _task_from_sample(sample: Sample) -> dict[str, Any]:
    task: dict[str, Any] = {}
    if isinstance(sample.metadata, dict):
        task.update(sample.metadata)
    task["prompt"] = sample.prompt
    reward_model = sample.label
    ground_truth = _sample_ground_truth(sample)
    if isinstance(reward_model, dict):
        reward_model = dict(reward_model)
        if ground_truth is not None and not reward_model.get("ground_truth"):
            reward_model["ground_truth"] = ground_truth
    elif ground_truth is not None and reward_model is None:
        reward_model = {"ground_truth": ground_truth}
    task["reward_model"] = reward_model
    if ground_truth is not None:
        task["ground_truth"] = ground_truth
    return normalize_task(task)


def _sample_ground_truth(sample: Sample) -> Any:
    def _first_non_empty(values: list[Any]) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, set, dict)) and not value:
                continue
            return value
        return None

    sources: list[Any] = []
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    extra_info = metadata.get("extra_info")
    if isinstance(extra_info, dict):
        sources.extend(extra_info.get(key) for key in ("ground_truth", "target", "answer", "answers"))
    sources.extend(metadata.get(key) for key in ("ground_truth", "target", "answer", "answers"))

    label = sample.label
    if isinstance(label, dict):
        sources.extend(label.get(key) for key in ("ground_truth", "target", "answer", "answers"))
    elif label is not None:
        sources.append(label)

    return _first_non_empty(list(sources))


def _initial_messages(
    harness: str,
    task_type: str,
    observation: str,
    tools: list[dict],
    model_name: str | None = None,
    *,
    tool_parser=None,
    inline_tool_prompt: bool = True,
    retrieved_context: str = "",
) -> list[dict[str, str]]:
    if harness == "rllm_deepresearch":
        system = build_system_prompt(RLLM_DR_SEARCH_SYSTEM_PROMPT, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": _strip_conflicting_answer_tag_instruction(observation)},
        ]
    if harness == "cut_bill":
        return [
            {"role": "system", "content": cut_bill_system_prompt()},
            {"role": "user", "content": _strip_conflicting_answer_tag_instruction(observation)},
        ]
    if harness == "search_gym":
        system = build_system_prompt(SEARCH_GYM_SYSTEM_PROMPT, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": SEARCH_GYM_USER_PROMPT.format(question=observation)},
        ]
    if harness == "agentcpm_explore":
        return build_agentcpm_explore_messages(observation, tools)
    if harness == "bare":
        return [{"role": "user", "content": observation}]
    if harness == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": COT_USER_PROMPT.format(problem_statement=observation)},
        ]
    if harness == "rag":
        return [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RAG_USER_PROMPT.format(
                    problem_statement=observation,
                    retrieved_context=retrieved_context,
                ),
            },
        ]
    if harness == "react":
        system = build_system_prompt(REACT_SYSTEM_PROMPT, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        user = REACT_USER_PROMPT.format(problem_statement=observation)
    elif task_type == "mcp":
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_MCP_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        user = FUSED_MCP_USER_PROMPT.format(problem_statement=observation)
    elif task_type == "cli":
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_CLI_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        user = FUSED_CLI_USER_PROMPT.format(problem_statement=observation)
    elif task_type == "et":
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_ET_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        user = FUSED_ET_USER_PROMPT.format(problem_statement=observation)
    else:
        base = FUSED_UNIFIED_SYSTEM_PROMPT if harness == "unified_gem" else FUSED_SEARCH_SYSTEM_PROMPT
        system = build_system_prompt(base, tools, model_name, tool_parser=tool_parser, inline_tool_prompt=inline_tool_prompt)
        observation = _strip_conflicting_answer_tag_instruction(observation)
        user_prompt = (
            FUSED_SEARCH_LONG_USER_PROMPT
            if os.environ.get("FUSED_WEB_SEARCH_USER_PROMPT", "short") == "long"
            else FUSED_SEARCH_USER_PROMPT
        )
        user = user_prompt.format(problem_statement=observation)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip_conflicting_answer_tag_instruction(text: str) -> str:
    return re.sub(
        r"\s*When ready, output the final answer enclosed in <answer> and </answer> tags\.\s*"
        r"Do not generate any content after the </answer> tag\.\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _append_tool_observation_message(
    messages: list[dict[str, Any]],
    parser,
    action: ToolCall,
    formatted_observation: str,
    raw_observation: Any,
) -> None:
    _append_tool_observation_messages(messages, parser, [action], [formatted_observation], [raw_observation])


def _append_tool_observation_messages(
    messages: list[dict[str, Any]],
    parser,
    actions: list[ToolCall],
    formatted_observations: list[str],
    raw_observations: list[Any],
) -> None:
    payloads = [
        _tool_response_payload(action.name, raw_observation)
        for action, raw_observation in zip(actions, raw_observations, strict=True)
    ]
    assistant_message = parser.assistant_tool_results_message(actions, payloads)
    if assistant_message is not None:
        messages[-1] = assistant_message
        return
    for formatted_observation in formatted_observations:
        messages.append({"role": "user", "content": formatted_observation})


def _can_append_batched_tool_results(parser, actions: list[ToolCall]) -> bool:
    if len(actions) <= 1 or any(action.name == "finish" for action in actions):
        return False
    return parser.assistant_tool_results_message(actions, [{} for _ in actions]) is not None


def _merge_tool_infos(infos: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for info in infos:
        merged.update(info)
    if len(infos) > 1:
        merged["tools/batched_calls"] = len(infos)
    return merged


def _valid_tool_names(tools: list[dict]) -> set[str]:
    names = set()
    for schema in tools:
        fn = schema.get("function", schema)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.add(name)
            names.add(str(name).replace("-", "_"))
    names.add("finish")
    return names


def _requires_non_finish_tool(tools: list[dict]) -> bool:
    return any(name not in {"finish", "submit"} for name in _declared_tool_names(tools))


def _has_token_prefix(token_ids: list[int], prefix_ids: list[int]) -> bool:
    return len(token_ids) >= len(prefix_ids) and token_ids[: len(prefix_ids)] == prefix_ids


def _common_token_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        length += 1
    return length


def _tito_model_type(model_name: str | None) -> str | None:
    explicit_series = os.environ.get("FUSED_MODEL_SERIES", "").strip().lower().replace("-", "_")
    if explicit_series in {"gemma4", "gemma_4"}:
        return "gemma4"
    if explicit_series in {"qwen3.5", "qwen3_5"}:
        return "qwen3_5"
    if explicit_series == "qwen3":
        return "qwen3"
    normalized = str(model_name or "").lower().replace("-", "_")
    if "gemma4" in normalized or "gemma_4" in normalized:
        return "gemma4"
    if "qwen3.5" in normalized or "qwen3_5" in normalized:
        return "qwen3_5"
    if "qwen3" in normalized:
        return "qwen3"
    return None


def _record_tool_parser_errors(
    *,
    args,
    base_sample: Sample,
    session_id: str,
    rollout_step: int,
    model_name: str | None,
    response: str,
    parser,
    prompt_ids: list[int],
    evaluation: bool,
    fallback_errors: list[str] | None = None,
) -> None:
    """Append every parser error to the current run's post-hoc debug log.

    Rollout workers are separate Ray processes, so use one O_APPEND write per
    JSON record rather than a process-local logger or shared file handle.
    """
    parser_errors = list(getattr(parser, "last_schema_errors", ()) or ())
    # Schema failures are recorded immediately after parsing.  The later
    # abnormal-termination branches pass fallback_errors for responses where
    # the parser had no more specific diagnosis; do not write the same schema
    # failure a second time from those branches.
    if fallback_errors is not None and parser_errors:
        return
    errors = parser_errors or list(fallback_errors or ())
    if not errors:
        return
    configured_path = os.environ.get("SLIME_TOOL_PARSER_ERROR_LOG")
    if configured_path:
        log_path = configured_path
    else:
        episode_dir = os.environ.get("SLIME_EPISODE_LOG_DIR")
        if episode_dir:
            root = Path(episode_dir)
            root = root.parent if root.name in {"episodes", "train", "evals"} else root
        else:
            root = Path(getattr(args, "dump_details", ".") or ".")
        log_path = str(root / "tool_parser_errors.log")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation": bool(evaluation),
        "rollout_id": getattr(base_sample, "rollout_id", None),
        "sample_index": getattr(base_sample, "index", None),
        "group_index": getattr(base_sample, "group_index", None),
        "session_id": session_id,
        "step": int(rollout_step),
        "model": _tito_model_type(model_name) or model_name,
        "parser": type(parser).__name__,
        "errors": errors,
        "error_kinds": list(getattr(parser, "last_schema_error_kinds", ()) or ()),
        "error_spans": list(getattr(parser, "last_schema_error_spans", ()) or ()),
        "response": response,
        "response_length": len(response),
        "prompt_tokens": len(prompt_ids),
    }
    line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError:
        logger.exception("Failed to append tool parser error log at %s", log_path)


def _declared_tool_names(tools: list[dict]) -> set[str]:
    names = set()
    for schema in tools:
        fn = schema.get("function", schema)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            normalized = str(name).strip().replace("-", "_")
            names.add(normalized)
    return names


def _has_recoverable_tool_schema_errors(parser, actions: list[ToolCall]) -> bool:
    """Return true when unknown optional fields were filtered from a valid call."""
    if not isinstance(parser, Gemma4ToolParser) or not actions:
        return False
    kinds = list(getattr(parser, "last_schema_error_kinds", ()) or ())
    return bool(kinds) and all(kind == "unknown_parameter" for kind in kinds)


def _actions_span(actions: list[ToolCall]) -> tuple[int, int] | None:
    spans = [(action.start, action.end) for action in actions if action.start is not None and action.end is not None]
    if not spans:
        return None
    return min(start for start, _ in spans), max(end for _, end in spans)


def _normalize_search_query(query: Any) -> str:
    return " ".join(str(query or "").strip().lower().split())


def _has_mixed_tool_and_answer(response: str, actions: list[ToolCall]) -> bool:
    return any(action.name != "finish" for action in actions) and (
        any(action.name == "finish" for action in actions)
        or _answer_span(response, excluded_spans=_actions_spans(actions)) is not None
    )


def _mixed_tool_and_answer_span(response: str, actions: list[ToolCall]) -> tuple[int, int] | None:
    spans = [
        (action.start, action.end)
        for action in actions
        if action.start is not None and action.end is not None and action.name != "finish"
    ]
    answer_span = _answer_span(response, excluded_spans=_actions_spans(actions))
    if answer_span is not None:
        spans.append(answer_span)
    else:
        spans.extend(
            (action.start, action.end)
            for action in actions
            if action.start is not None and action.end is not None and action.name == "finish"
        )
    if not spans:
        return _actions_span(actions)
    return min(start for start, _ in spans), max(end for _, end in spans)


def _actions_spans(actions: list[ToolCall]) -> list[tuple[int, int]]:
    return [(action.start, action.end) for action in actions if action.start is not None and action.end is not None]


def _answer_span(response: str, *, excluded_spans: list[tuple[int, int]] | None = None) -> tuple[int, int] | None:
    excluded_spans = excluded_spans or []
    answer_matches = list(re.finditer(r"<answer>\s*.*?\s*</answer>", response or "", flags=re.DOTALL | re.IGNORECASE))
    for match in reversed(answer_matches):
        if _span_overlaps_any((match.start(), match.end()), excluded_spans):
            continue
        return match.start(), match.end()
    boxed_start, boxed_end = _boxed_answer_span(response or "")
    if (
        boxed_start is not None
        and boxed_end is not None
        and not _span_overlaps_any((boxed_start, boxed_end), excluded_spans)
    ):
        return boxed_start, boxed_end
    return None


def _span_overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in spans)


def _boxed_answer_span(text: str) -> tuple[int | None, int | None]:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None, None
    start = idx + len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return idx, i + 1
    return None, None


def _ngram_repetition_stats(output_ids: list[int], *, n: int, min_tokens: int) -> dict[str, float | int]:
    if n <= 0 or len(output_ids) < max(min_tokens, n):
        return {"score": 0.0, "n": n, "total": 0, "unique": 0}
    ngrams = [tuple(output_ids[i : i + n]) for i in range(len(output_ids) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    score = 1.0 - unique / total if total else 0.0
    return {"score": score, "n": n, "total": total, "unique": unique}


def _eval_response_anomaly_info(
    response: str,
    actions: list[ToolCall],
    tokenizer,
    *,
    allow_gemma4_native_tool_response_handoff: bool = False,
    allow_leading_think_close: bool,
    allow_action_terminated_think: bool = False,
    ngram_n: int,
    ngram_threshold: float,
    ngram_min_tokens: int,
) -> dict[str, Any]:
    anomalies = []
    info: dict[str, Any] = {}

    protocol_response = (
        _strip_gemma4_native_tool_response_handoff(response, actions)
        if allow_gemma4_native_tool_response_handoff
        else response
    )
    normalized_response = protocol_response.lower()
    if any(
        marker in normalized_response
        for marker in ("<tool_response>", "</tool_response>", "<|tool_response>", "<tool_response|>")
    ):
        anomalies.append("forged_tool_response")
        info["forged_tool_response_detected"] = True

    tag_imbalances = _response_tag_imbalances(
        protocol_response,
        allow_leading_think_close=allow_leading_think_close,
    )
    if (
        allow_action_terminated_think
        and tag_imbalances.get("think") == {"open": 1, "close": 0, "misordered": False}
        and len(actions) == 1
        and actions[0].end is not None
        and not response[actions[0].end :].strip()
    ):
        tag_imbalances.pop("think")
    if tag_imbalances:
        anomalies.append("unbalanced_tags")
        info["unbalanced_response_tags"] = tag_imbalances

    try:
        encoded = tokenizer.encode(response, add_special_tokens=False)
        response_ids = list(encoded.ids if hasattr(encoded, "ids") else encoded)
    except Exception:
        response_ids = list(response.encode("utf-8"))
    repetition = _ngram_repetition_stats(response_ids, n=ngram_n, min_tokens=ngram_min_tokens)
    if repetition["score"] > ngram_threshold:
        anomalies.append("ngram_repetition")
        info.update(
            {
                "ngram_repetition_detected": True,
                "ngram_repetition_score": repetition["score"],
                "ngram_repetition_n": repetition["n"],
                "ngram_repetition_total": repetition["total"],
                "ngram_repetition_unique": repetition["unique"],
            }
        )

    if not anomalies:
        return {}
    return {"eval_response_anomalies": anomalies, **info}


def _strip_gemma4_native_tool_response_handoff(response: str, actions: list[ToolCall]) -> str:
    """Strip only Gemma4's empty trailing handoff marker from protocol checks.

    The native chat template asks the model to end a parsed tool call with an
    opening ``<|tool_response>`` marker. The runtime, not the model, supplies
    the response payload. A marker anywhere else, or anything after it, remains
    visible to the forged-response and tag-balance checks.
    """
    action_ends = [action.end for action in actions if action.end is not None]
    if not action_ends:
        return response
    last_action_end = max(action_ends)
    suffix = response[last_action_end:]
    if re.fullmatch(r"\s*<\|tool_response>\s*", suffix, flags=re.IGNORECASE) is None:
        return response
    return response[:last_action_end]


def _response_tag_imbalances(
    response: str,
    *,
    allow_leading_think_close: bool = False,
) -> dict[str, dict[str, int | bool]]:
    imbalances = {}
    for name, begin, end in (
        ("think", "<think>", "</think>"),
        ("answer", "<answer>", "</answer>"),
        ("tool_call", "<tool_call>", "</tool_call>"),
        ("tool_response", "<tool_response>", "</tool_response>"),
        ("native_tool_call", "<|tool_call>", "<tool_call|>"),
        ("native_tool_response", "<|tool_response>", "<tool_response|>"),
    ):
        markers = list(re.finditer(f"({re.escape(begin)}|{re.escape(end)})", response, flags=re.IGNORECASE))
        opens = sum(match.group(0).lower() == begin.lower() for match in markers)
        closes = len(markers) - opens
        implicit_opens = int(
            name == "think"
            and allow_leading_think_close
            and bool(markers)
            and markers[0].group(0).lower() == end.lower()
        )
        balance = implicit_opens
        misordered = False
        for match in markers:
            balance += 1 if match.group(0).lower() == begin.lower() else -1
            misordered = misordered or balance < 0
        if opens + implicit_opens != closes or misordered:
            imbalances[name] = {"open": opens, "close": closes, "misordered": misordered}
    return imbalances


def _has_unclosed_think(response: str) -> bool:
    """Return whether a response opens more ``<think>`` blocks than it closes."""
    markers = re.findall(r"<think>|</think>", response, flags=re.IGNORECASE)
    depth = 0
    for marker in markers:
        if marker.lower() == "<think>":
            depth += 1
        elif depth > 0:
            depth -= 1
    return depth > 0


def _parser_error_action_span(response: str) -> tuple[int, int] | None:
    span = _first_unclosed_tool_call_span(response)
    if span is not None:
        return span
    span = _first_malformed_tool_call_span(response)
    if span is not None:
        return span
    idx = response.rfind("</think>")
    if idx >= 0:
        start = idx + len("</think>")
        while start < len(response) and response[start].isspace():
            start += 1
        if start < len(response):
            return start, len(response)
    idx = response.rfind("<channel|>")
    if idx >= 0:
        start = idx + len("<channel|>")
        while start < len(response) and response[start].isspace():
            start += 1
        if start < len(response):
            return start, len(response)
    return (0, len(response)) if response else None


def _response_has_malformed_tool_call(response: str) -> bool:
    return (
        _first_unclosed_tool_call_span(response) is not None or _first_malformed_tool_call_span(response) is not None
    )


def _first_unclosed_tool_call_span(response: str) -> tuple[int, int] | None:
    spans = []
    for begin, end_marker in (("<tool_call>", "</tool_call>"), ("<|tool_call>", "<tool_call|>")):
        search_pos = 0
        while True:
            start = response.find(begin, search_pos)
            if start < 0:
                break
            end = response.find(end_marker, start + len(begin))
            if end < 0:
                spans.append((start, len(response)))
                break
            search_pos = end + len(end_marker)
    return min(spans, default=None)


def _first_malformed_tool_call_span(response: str) -> tuple[int, int] | None:
    import re

    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", response, flags=re.DOTALL):
        try:
            json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return match.start(), match.end()
    for match in re.finditer(r"<\|tool_call>\s*(.*?)\s*<tool_call\|>", response, flags=re.DOTALL):
        parsed = make_tool_parser("gemma4").parse(match.group(0))
        if not parsed:
            return match.start(), match.end()
    return None


async def _mark_pending_turn_error_span(
    tokenizer,
    item: dict[str, Any],
    response: str,
    char_span: tuple[int, int] | None,
    *,
    output_len: int,
    attribution: str = "localized",
) -> None:
    item["credit_assignment_error_attribution"] = attribution
    if attribution == "unattributable":
        return
    # The char->token span conversion may re-encode the response twice; run it
    # off the event loop since abnormal terminations cluster in waves.
    span = await asyncio.to_thread(
        _response_span_to_output_token_span,
        tokenizer,
        response,
        char_span,
        output_len=output_len,
    )
    _set_pending_turn_error_span(item, span, output_len=output_len)


def _set_pending_turn_error_span(item: dict[str, Any], span: tuple[int, int] | None, *, output_len: int) -> None:
    if span is None:
        return
    start, end = span
    start = max(0, min(output_len, int(start)))
    end = max(start, min(output_len, int(end)))
    if end <= start:
        return
    item["credit_assignment_action_span"] = (start, end)


def _response_span_to_output_token_span(
    tokenizer,
    response: str,
    char_span: tuple[int, int] | None,
    *,
    output_len: int,
) -> tuple[int, int] | None:
    if char_span is None:
        return None
    char_start, char_end = char_span
    if len(response) == output_len:
        return char_start, char_end
    start = _encode_len(tokenizer, response[:char_start])
    end = _encode_len(tokenizer, response[:char_end])
    if start is None or end is None:
        return (0, output_len)
    return start, end


def _encode_len(tokenizer, text: str) -> int | None:
    try:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=False))
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            return len(encoded["input_ids"])
        return len(encoded)
    except Exception:
        return None


def _record_pending_turns(
    manager: TrajectoryManager,
    *,
    session_id: str,
    pending_turns: list[dict[str, Any]],
    credit_event: str | None,
    credit_step_index: int | None,
    parser_error_token_window: int,
) -> None:
    for idx, item in enumerate(pending_turns):
        turn = item["turn"]
        policy_loss_mask = _credit_assignment_loss_mask(
            output_len=len(turn.output_ids),
            turn_index=idx,
            credit_event=credit_event,
            credit_step_index=credit_step_index,
            parser_error_token_window=parser_error_token_window,
            action_span=item.get("credit_assignment_action_span"),
            error_attribution=item.get("credit_assignment_error_attribution"),
            base_loss_mask=turn.loss_mask,
        )
        metadata = dict(item["metadata"])
        if credit_event is not None:
            metadata["credit_assignment_event"] = credit_event
            metadata["credit_assignment_error_step_index"] = credit_step_index
            if idx == credit_step_index and item.get("credit_assignment_error_attribution") is not None:
                metadata["credit_assignment_error_attribution"] = item["credit_assignment_error_attribution"]
            if idx == credit_step_index and item.get("credit_assignment_action_span") is not None:
                start, end = item["credit_assignment_action_span"]
                metadata["credit_assignment_action_start"] = start
                metadata["credit_assignment_action_end"] = end
        manager.record_turn(
            session_id,
            turn=TurnRecord(
                prompt_ids=turn.prompt_ids,
                output_ids=turn.output_ids,
                finish_reason=turn.finish_reason,
                output_log_probs=turn.output_log_probs,
                weight_version=turn.weight_version,
                require_rollout_logprobs=turn.require_rollout_logprobs,
                require_weight_version=turn.require_weight_version,
                context_delta_ids=turn.context_delta_ids,
                tito_boundary_before=turn.tito_boundary_before,
                tito_model_type=turn.tito_model_type,
                tito_context_reason=turn.tito_context_reason,
                disable_thinking=turn.disable_thinking,
                loss_mask=turn.loss_mask,
                policy_loss_mask=policy_loss_mask,
                prompt_context_start_idx=turn.prompt_context_start_idx,
                rollout_top_p_token_ids=turn.rollout_top_p_token_ids,
                rollout_top_p_token_offsets=turn.rollout_top_p_token_offsets,
                ill_formed=turn.ill_formed,
            ),
            prompt_messages=item["prompt_messages"],
            response_message=item["response_message"],
            metadata=metadata,
        )


def _credit_assignment_loss_mask(
    *,
    output_len: int,
    turn_index: int,
    credit_event: str | None,
    credit_step_index: int | None,
    parser_error_token_window: int = 256,
    action_span: tuple[int, int] | None = None,
    error_attribution: str | None = None,
    base_loss_mask: list[int] | None = None,
) -> list[int] | None:
    def apply_base(mask: list[int]) -> list[int]:
        if base_loss_mask is None:
            return mask
        assert (
            len(base_loss_mask) == output_len
        ), f"base_loss_mask length {len(base_loss_mask)} != output length {output_len}"
        return [int(policy) & int(base) for policy, base in zip(mask, base_loss_mask, strict=True)]

    if credit_event is None:
        return None
    if credit_event == "search_bypass":
        return None
    if credit_event == "direct_submit_without_tool":
        return apply_base([0] * output_len)
    if credit_event in {"tail_guard_early_stop"}:
        return apply_base([0] * output_len)
    if credit_step_index is None:
        return apply_base([0] * output_len)
    if credit_event == "tool_parser_error":
        if turn_index != credit_step_index:
            return apply_base([0] * output_len)
        if action_span is not None:
            start, end = action_span
            if 0 <= start < end <= output_len:
                return apply_base([0] * start + [1] * (end - start) + [0] * (output_len - end))
        # Missing delimiters and truncated calls do not always expose a precise
        # span. Penalize a bounded suffix of the failing turn so format errors
        # retain a learning signal without blaming prior tool/reasoning turns.
        penalized_len = max(0, min(output_len, parser_error_token_window))
        return apply_base([0] * (output_len - penalized_len) + [1] * penalized_len)
    # Other credit events keep their event-specific action-span behavior. Parser
    # failures are handled above because their negative signal additionally
    # depends on whether the emitted error can be localized reliably.
    if credit_event in {
        "think_parser_error",
        "max_response_len_exceeded",
        "max_turns_exceeded",
        "too_many_tool_calls",
        "repeated_search_query",
        "ngram_repetition",
        "mixed_tool_and_answer",
    }:
        if turn_index != credit_step_index:
            return apply_base([0] * output_len)
        if action_span is not None:
            start, end = action_span
            if 0 <= start < end <= output_len:
                return apply_base([0] * start + [1] * (end - start) + [0] * (output_len - end))
        # Without a character-derived span, only parser/length failures have a
        # meaningful tail location. Other events retain the whole turn rather
        # than guessing and accidentally deleting valid training signal.
        if credit_event not in {"tool_parser_error", "think_parser_error", "max_response_len_exceeded"}:
            return apply_base([1] * output_len)
        masked_len = max(0, min(output_len, parser_error_token_window))
        mask = [1] * output_len
        if masked_len:
            mask[-masked_len:] = [0] * masked_len
        return apply_base(mask)
    return apply_base([1] * output_len if turn_index == credit_step_index else [0] * output_len)


def _decode_and_parse_step(tokenizer, parser, output_ids: list[int], *, disable_thinking: bool):
    """Decode one LLM turn and derive its parsed actions + default loss mask.

    Everything here is CPU-bound pure-Python/tokenizer work; callers run it via
    asyncio.to_thread to keep the rollout event loop free.
    """
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=False) if output_ids else ""
    response = _strip_trailing_chat_template_stop(raw_response)
    parsed_actions = parser.parse(response)
    loss_mask = _default_response_loss_mask(
        tokenizer,
        response,
        output_len=len(output_ids),
        disable_thinking=disable_thinking,
        output_ids=output_ids,
    )
    return response, parsed_actions, loss_mask


def _decode_step(tokenizer, _parser, output_ids: list[int], *, disable_thinking: bool):
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=False) if output_ids else ""
    response = _strip_trailing_chat_template_stop(raw_response)
    loss_mask = _default_response_loss_mask(
        tokenizer,
        response,
        output_len=len(output_ids),
        disable_thinking=disable_thinking,
        output_ids=output_ids,
    )
    return response, [], loss_mask


def _default_response_loss_mask(
    tokenizer,
    response: str,
    *,
    output_len: int,
    disable_thinking: bool,
    output_ids: list[int] | None = None,
) -> list[int] | None:
    if output_len <= 0:
        return []
    if not disable_thinking:
        # enable-thinking: train the whole response (reasoning + answer). The
        # builder treats None as an all-ones mask.
        return None

    # disable-thinking: the empty think shell "<think>\n\n</think>\n\n" lives in
    # the prompt, so a well-behaved response carries no think block and trains in
    # full. Only when the model *mis-fires* a leading <think>...</think> despite
    # being told not to think do we mask that stray block out.
    #
    # Prefer locating </think> in token space: it is a single, non-mergeable
    # added token in both Qwen3 and Qwen3.5 tokenizers, so scanning output_ids
    # for its id gives an exact boundary and avoids the ±1 drift of re-encoding
    # a character substring (_encode_len(response[:think_end])).
    #
    # A mis-fired block need not open with <think> (Qwen3.5-shape: the model
    # continues reasoning right after the prompt's empty shell and closes with a
    # bare </think>), so a bare leading closer is masked too. But if real action
    # content (a tool call) precedes the first </think>, the closer is a stray
    # artifact -- masking through it would zero-mask legitimate action tokens.
    close_id = _think_close_token_id(tokenizer)
    if output_ids is not None and close_id is not None:
        try:
            j = output_ids.index(close_id)
        except ValueError:
            return [1] * output_len
        closer_pos = response.find("</think>")
        if closer_pos >= 0 and "<tool_call" in response[:closer_pos]:
            return [1] * output_len
        # Mask the mis-fired think block through </think> itself; the answer
        # (everything after </think>) stays trainable. Any trailing "\n\n"
        # separator sits in the trainable side but carries no real content.
        start = min(j + 1, output_len)
        return [0] * start + [1] * (output_len - start)

    # Fallback for tokenizers without a resolvable </think> id: character-level
    # boundary with the pre-existing ±1 re-encode behavior. Gemma4 uses a
    # native thought channel instead of <think>...</think>.
    think_end = _leading_thinking_block_end(response)
    if think_end is None:
        return [1] * output_len

    start = _encode_len(tokenizer, response[:think_end])
    if start is None:
        start = min(max(0, think_end), output_len) if len(response) == output_len else output_len
    start = max(0, min(output_len, start))
    return [0] * start + [1] * (output_len - start)


def _think_close_token_id(tokenizer) -> int | None:
    """Resolve the single-token id of ``</think>`` for this tokenizer, or None.

    Both Qwen3 (151668) and Qwen3.5 (248069) expose ``</think>`` as one added
    token; other tokenizers may lack it or split it, in which case we return None
    and callers fall back to character-level handling.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return None
    try:
        tid = convert("</think>")
    except Exception:
        return None
    if tid is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and tid == unk_id:
        return None
    return tid


def _leading_thinking_block_end(response: str) -> int | None:
    text = str(response or "")
    stripped = text.lstrip()
    prefix_len = len(text) - len(stripped)
    if stripped.startswith("<think>"):
        end = stripped.find("</think>")
        if end < 0:
            return None
        block_end = prefix_len + end + len("</think>")
    elif stripped.startswith("<|channel>thought\n"):
        end = stripped.find("<channel|>")
        if end < 0:
            return None
        block_end = prefix_len + end + len("<channel|>")
    else:
        return None
    while block_end < len(text) and text[block_end] in {"\n", "\r"}:
        block_end += 1
    return block_end


def _render_prompt_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    disable_thinking: bool = True,
) -> list[int]:
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        disable_thinking=disable_thinking,
    )
    if hasattr(rendered, "data") and isinstance(rendered.data, dict):
        rendered = rendered.data["input_ids"]
    elif isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    return list(rendered)


def _render_qwen3_tito_delta_ids(
    tokenizer,
    old_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    prefix_ids: list[int],
    *,
    disable_thinking: bool,
) -> list[int]:
    """Encode only the Qwen3-family chat-template suffix added since the last turn.

    The caller prepends these ids to the exact prompt/output ids served on the
    previous turn. This avoids decoding and re-tokenizing generated assistant
    tokens, whose thinking whitespace can be canonicalized by Qwen3's template.
    """
    if len(new_messages) < len(old_messages) or new_messages[: len(old_messages)] != old_messages:
        raise ValueError("Qwen3-family TITO requires append-only message history")

    appended_messages = new_messages[len(old_messages) :]
    if not appended_messages:
        raise ValueError("Qwen3-family TITO continuation has no appended messages")
    unsupported_roles = [message.get("role") for message in appended_messages if message.get("role") not in {"tool", "user"}]
    if unsupported_roles:
        raise ValueError(f"Qwen3-family TITO cannot append message roles {unsupported_roles!r}")

    # Qwen3.5's template validates that every rendered conversation contains a
    # user query.  Render the actual last user message in both anchors so it is
    # part of the common prefix and only the newly appended tool/user messages
    # become the delta.  Qwen3's older template did not require this, but the
    # same construction is valid for both model families.
    # Qwen3.5 distinguishes the actual query user turn from synthetic user
    # turns carrying prior tool responses.  The latter are valid history but
    # do not satisfy the template's ``last_query_index`` check.
    last_user = next(
        (
            message
            for message in reversed(old_messages)
            if message.get("role") == "user"
            and not str(message.get("content", "")).lstrip().startswith("<tool_response>")
        ),
        None,
    )
    if last_user is None:
        raise ValueError("Qwen3-family TITO requires an existing user query")
    dummy_system = {"role": "system", "content": "slime tito anchor"}
    rendered_base = _apply_chat_template(
        tokenizer,
        [dummy_system, last_user],
        tokenize=False,
        add_generation_prompt=False,
        disable_thinking=disable_thinking,
    )
    rendered_with_delta = _apply_chat_template(
        tokenizer,
        [dummy_system, last_user, *appended_messages],
        tokenize=False,
        add_generation_prompt=True,
        disable_thinking=disable_thinking,
    )
    rendered_base = str(rendered_base)
    rendered_with_delta = str(rendered_with_delta)
    if not rendered_with_delta.startswith(rendered_base):
        raise ValueError("Qwen3-family chat template did not produce an append-only rendered suffix")

    suffix = rendered_with_delta[len(rendered_base) :]
    encoded = tokenizer.encode(suffix, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    encoded = [int(token_id) for token_id in encoded]

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if hasattr(newline_ids, "tolist"):
        newline_ids = newline_ids.tolist()
    newline_ids = [int(token_id) for token_id in newline_ids]
    if im_end_id is None or im_end_id == getattr(tokenizer, "unk_token_id", None) or not newline_ids:
        raise ValueError("Qwen3-family tokenizer cannot encode the assistant turn boundary")

    # SGLang may include or omit the stop token from output_ids. Complete the
    # historical assistant boundary exactly once before appending tool/user text.
    im_end_id = int(im_end_id)
    if len(prefix_ids) >= len(newline_ids) + 1 and prefix_ids[-len(newline_ids) - 1 :] == [im_end_id, *newline_ids]:
        bridge_ids: list[int] = []
    elif prefix_ids and prefix_ids[-1] == im_end_id:
        bridge_ids = newline_ids
    else:
        bridge_ids = [im_end_id, *newline_ids]
    return bridge_ids + encoded


def _render_gemma4_tito_delta_ids(
    tokenizer,
    old_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    prefix_ids: list[int] | None = None,
    *,
    tools: list[dict] | None,
    disable_thinking: bool,
) -> list[int]:
    """Render only the Gemma4 tool-result suffix after a generated action.

    Gemma4 replaces the last raw assistant message with a structured assistant
    message containing both the parsed tool call and its result. Re-rendering
    that message would replay and possibly re-tokenize an action whose exact ids
    are already in the TiTO prefix. Build an otherwise identical assistant turn
    containing only the result, then take the suffix after the generation prompt.
    """
    if len(old_messages) != len(new_messages) or old_messages[:-1] != new_messages[:-1]:
        raise ValueError("Gemma4 TITO requires only the last assistant message to be replaced")
    if not old_messages or old_messages[-1].get("role") != "assistant":
        raise ValueError("Gemma4 TITO requires a previous assistant action")

    replacement = new_messages[-1]
    if replacement.get("role") != "assistant" or not replacement.get("tool_responses"):
        raise ValueError("Gemma4 TITO requires a structured assistant tool response")

    response_only = {
        "role": "assistant",
        "tool_responses": copy.deepcopy(replacement["tool_responses"]),
    }
    base_ids = _render_prompt_ids(
        tokenizer,
        new_messages[:-1],
        tools=tools,
        disable_thinking=disable_thinking,
    )
    with_response_ids = _render_prompt_ids(
        tokenizer,
        [*new_messages[:-1], response_only],
        tools=tools,
        disable_thinking=disable_thinking,
    )
    if not _has_token_prefix(with_response_ids, base_ids):
        raise ValueError("Gemma4 chat template did not produce an append-only tool-response suffix")
    delta_ids = list(with_response_ids[len(base_ids) :])

    # A native Gemma4 tool call normally stops after generating the opening
    # <|tool_response> handoff token.  The response-only chat-template suffix
    # starts with that same token.  Replaying it creates
    # ``<|tool_response><|tool_response>...`` and the model commonly answers
    # with one more empty handoff instead of another action.  SGLang may omit
    # the stop token from output_ids, so remove the suffix opener only when it
    # is already the last served token.
    tool_response_ids = tokenizer.encode("<|tool_response>", add_special_tokens=False)
    if hasattr(tool_response_ids, "tolist"):
        tool_response_ids = tool_response_ids.tolist()
    tool_response_ids = [int(token_id) for token_id in tool_response_ids]
    if (
        tool_response_ids
        and prefix_ids
        and prefix_ids[-len(tool_response_ids) :] == tool_response_ids
        and delta_ids[: len(tool_response_ids)] == tool_response_ids
    ):
        delta_ids = delta_ids[len(tool_response_ids) :]
    return delta_ids


def _last_assistant_context_start_idx(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    disable_thinking: bool = True,
) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return len(
                _render_messages_without_generation_prompt(
                    tokenizer,
                    messages[: idx + 1],
                    tools=tools,
                    disable_thinking=disable_thinking,
                )
            )
    return None


def _render_messages_without_generation_prompt(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    disable_thinking: bool = True,
) -> list[int]:
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        disable_thinking=disable_thinking,
    )
    if hasattr(rendered, "data") and isinstance(rendered.data, dict):
        rendered = rendered.data["input_ids"]
    elif isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    return list(rendered)


def _apply_chat_template(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    tokenize: bool,
    add_generation_prompt: bool,
    disable_thinking: bool,
):
    native_tools = _chat_template_accepts_native_tools(tokenizer)
    messages = _prepare_messages_for_chat_template(
        messages,
        disable_thinking=disable_thinking,
        add_empty_reasoning=not native_tools,
    )
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": not disable_thinking,
    }
    if tools and native_tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if tools and native_tools and _type_error_mentions_kwarg(exc, "tools"):
            kwargs.pop("tools", None)
            inline_messages = _messages_with_inline_gemma4_tools(messages, tools)
            try:
                return tokenizer.apply_chat_template(inline_messages, **kwargs)
            except TypeError as inline_exc:
                if "enable_thinking" not in str(inline_exc):
                    raise
                kwargs.pop("enable_thinking", None)
                return tokenizer.apply_chat_template(inline_messages, **kwargs)
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking", None)
        fallback_messages = (
            messages
            if native_tools
            else _prepare_messages_for_fallback_chat_template(messages, disable_thinking=disable_thinking)
        )
        try:
            rendered = tokenizer.apply_chat_template(fallback_messages, **kwargs)
        except TypeError as fallback_exc:
            if not (tools and native_tools and _type_error_mentions_kwarg(fallback_exc, "tools")):
                raise
            kwargs.pop("tools", None)
            fallback_messages = _messages_with_inline_gemma4_tools(messages, tools)
            rendered = tokenizer.apply_chat_template(fallback_messages, **kwargs)
        if native_tools or not (disable_thinking and add_generation_prompt):
            return rendered
        return _append_disabled_thinking_generation_prefix(tokenizer, rendered, tokenize=tokenize)


def _chat_template_accepts_native_tools(tokenizer) -> bool:
    name = str(getattr(tokenizer, "name_or_path", "") or "").lower().replace("-", "_")
    return "gemma4" in name or "gemma_4" in name or "qwen3.5" in name or "qwen3_5" in name


def _type_error_mentions_kwarg(exc: TypeError, kwarg: str) -> bool:
    message = str(exc)
    return kwarg in message and ("keyword" in message or "argument" in message)


def _messages_with_inline_gemma4_tools(messages: list[dict[str, Any]], tools: list[dict]) -> list[dict[str, Any]]:
    declarations = _inline_gemma4_tool_declarations(tools)
    if not declarations:
        return messages

    prepared = [dict(message) for message in messages]
    for idx, message in enumerate(prepared):
        if message.get("role") != "system":
            continue
        content = str(message.get("content") or "").strip()
        if "<|tool>declaration:" in content:
            return prepared
        prepared[idx] = {**message, "content": (content + "\n" + declarations).strip()}
        return prepared

    return [{"role": "system", "content": declarations}, *prepared]


def _inline_gemma4_tool_declarations(tools: list[dict]) -> str:
    declarations = []
    for schema in tools:
        declaration = Gemma4ToolParser._format_function_declaration(schema)
        if declaration:
            declarations.append(f"<|tool>{declaration}<tool|>")
    return "".join(declarations)


def _prepare_messages_for_chat_template(
    messages: list[dict[str, Any]],
    *,
    disable_thinking: bool,
    add_empty_reasoning: bool = True,
) -> list[dict[str, Any]]:
    if not disable_thinking or not add_empty_reasoning:
        return messages
    prepared = []
    for message in messages:
        if message.get("role") != "assistant":
            prepared.append(message)
            continue
        if message.get("tool_calls") or message.get("tool_responses"):
            prepared.append(message)
            continue
        content = str(message.get("content") or "")
        if "<think>" in content or message.get("reasoning_content") is not None:
            prepared.append(message)
            continue
        # Preserve an explicit empty think shell during prompt replay for
        # chat templates that suppress historical <think> blocks when the
        # reasoning content is falsy.
        prepared.append({**message, "reasoning_content": "\n"})
    return prepared


def _prepare_messages_for_fallback_chat_template(
    messages: list[dict[str, Any]],
    *,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    if not disable_thinking:
        return messages
    empty_thinking_prefix = "<think>\n\n</think>\n\n"
    prepared = []
    for message in messages:
        if message.get("role") != "assistant":
            prepared.append(message)
            continue
        content = str(message.get("content") or "")
        if content.startswith(empty_thinking_prefix):
            prepared.append({**message, "content": content})
            continue
        prepared.append({**message, "content": empty_thinking_prefix + content})
    return prepared


def _append_disabled_thinking_generation_prefix(tokenizer, rendered, *, tokenize: bool):
    empty_thinking_prefix = "<think>\n\n</think>\n\n"
    if not tokenize:
        return str(rendered) + empty_thinking_prefix

    prefix_ids = _tokenize_text(tokenizer, empty_thinking_prefix)
    if hasattr(rendered, "data") and isinstance(rendered.data, dict):
        data = dict(rendered.data)
        data["input_ids"] = list(data["input_ids"]) + prefix_ids
        if "attention_mask" in data:
            data["attention_mask"] = list(data["attention_mask"]) + [1] * len(prefix_ids)
        rendered.data = data
        return rendered
    if isinstance(rendered, dict):
        rendered = dict(rendered)
        rendered["input_ids"] = list(rendered["input_ids"]) + prefix_ids
        if "attention_mask" in rendered:
            rendered["attention_mask"] = list(rendered["attention_mask"]) + [1] * len(prefix_ids)
        return rendered
    return list(rendered) + prefix_ids


def _tokenize_text(tokenizer, text: str) -> list[int]:
    if not callable(tokenizer):
        return [ord(ch) for ch in text]
    encoded = tokenizer(text, add_special_tokens=False)
    if hasattr(encoded, "data") and isinstance(encoded.data, dict):
        encoded = encoded.data
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _episode_step(
    *,
    observation: str,
    response: str,
    action: str,
    reward: float,
    done: bool,
    messages: list[dict[str, Any]],
    llm_time: float,
    env_time: float,
    disable_thinking: bool = False,
    tito_context_reason: str | None = None,
    historical_thinking_discarded: bool = False,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "disable_thinking": bool(disable_thinking),
        "timing": {
            "start_timestamp": _utc_timestamp(),
            "end_timestamp": _utc_timestamp(),
            "llm_time": llm_time,
            "env_time": env_time,
        },
    }
    if tito_context_reason is not None:
        # How this step's served prompt relates to the TiTO accumulator.
        info["tito_context_reason"] = tito_context_reason
    info["historical_thinking_discarded"] = bool(historical_thinking_discarded)
    return {
        "observation": observation,
        "thought": _extract_thought(response),
        "action": action,
        "reward": float(reward),
        "done": bool(done),
        "model_response": response,
        "chat_completions": _chat_completions_for_step(messages, response),
        "info": info,
    }


def _chat_completions_for_step(messages: list[dict[str, Any]], response: str) -> list[dict[str, Any]]:
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == response:
        return list(messages)
    return [*messages, {"role": "assistant", "content": response}]


def _extract_thought(response: str) -> str:
    start = response.find("<think>")
    end = response.find("</think>")
    if start >= 0 and end > start:
        return response[start : end + len("</think>")]
    start = response.find("<|channel>thought")
    end = response.find("<channel|>", start + len("<|channel>thought")) if start >= 0 else -1
    if start >= 0 and end > start:
        return response[start : end + len("<channel|>")]
    return ""


def _format_tool_observation(parser_or_tool_name, tool_name_or_output: Any, output: Any | None = None) -> str:
    if output is None:
        parser = None
        tool_name = str(parser_or_tool_name)
        output_value = tool_name_or_output
    else:
        parser = parser_or_tool_name
        tool_name = str(tool_name_or_output)
        output_value = output
    output_text = _format_observation_output(tool_name, output_value)
    if parser is not None and hasattr(parser, "format_tool_observation"):
        return parser.format_tool_observation(tool_name, output_text)
    return "<tool_response>\n" f"Execution output of [{tool_name}]:\n" f"{output_text}\n" "</tool_response>"


def _format_observation_output(tool_name: str, output: Any) -> str:
    if _is_web_search_tool(tool_name):
        return _format_web_search_output(output)
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, default=str)


def _tool_response_payload(tool_name: str, output: Any) -> Any:
    if _is_web_search_tool(tool_name):
        return _format_observation_output(tool_name, output)
    if not isinstance(output, str):
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def _is_web_search_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip().lower().replace("-", "_") in {"web_search", "search", "webqa"}


def _format_web_search_output(output: Any) -> str:
    if not isinstance(output, str):
        return _format_retrieval(output)
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return " ".join(output.split())
    return _format_retrieval(parsed)


def _rllm_episode_dict(
    *,
    base_sample: Sample,
    task: dict[str, Any],
    session_id: str,
    reward: float,
    termination_reason: str,
    reward_debug: dict[str, Any],
    credit_event: str | None,
    credit_step_index: int | None,
    response_anomaly_info: dict[str, Any],
    total_steps: int,
    total_tool_call_turns: int,
    timing: dict[str, Any],
    steps: list[dict[str, Any]],
    task_type: str,
    discard_historical_thinking_enabled: bool,
) -> dict[str, Any]:
    task_for_dump = _task_for_dump(task)
    episode_id = _episode_id(base_sample, task_for_dump)
    benchmark = _benchmark_metric_name(task_for_dump, task_type)
    metrics = {
        f"{benchmark}/pass@1": float(reward > 0),
        "traj/steps": float(total_steps),
        "turn/tool_call_turn": float(total_tool_call_turns),
    }
    suffix = _task_metric_suffix(task_type)
    metrics[f"traj/steps/{suffix}"] = float(total_steps)
    metrics[f"turn/tool_call_turn/{suffix}"] = float(total_tool_call_turns)
    for key, value in (reward_debug or {}).items():
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    trajectory = {
        "uid": str(uuid.uuid4()),
        "name": f"{benchmark}_0",
        "task": task_for_dump,
        "steps": steps,
        "reward": float(reward),
        "info": {"timing": timing},
    }
    metadata = {
        "reward_debug": reward_debug or {},
        "timing": timing,
        "discard_historical_thinking_enabled": bool(discard_historical_thinking_enabled),
        "historical_thinking_discard_steps": sum(
            1 for step in steps if (step.get("info") or {}).get("historical_thinking_discarded")
        ),
        **response_anomaly_info,
    }
    if credit_event is not None:
        metadata["credit_assignment_event"] = credit_event
        metadata["credit_assignment_error_step_index"] = credit_step_index
    return {
        "id": episode_id,
        "task": task_for_dump,
        "termination_reason": termination_reason,
        "is_correct": bool(reward > 0),
        "session_id": session_id,
        "trajectories": [trajectory],
        "metrics": metrics,
        "metadata": metadata,
        "info": {"timing": timing},
    }


def _task_for_dump(task: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in task.items() if k not in {"image", "images", "prompt", "reward_model"}}
    if "question" not in cleaned and isinstance(task.get("prompt"), str):
        cleaned["question"] = task["prompt"]
    if "ground_truth" not in cleaned and task.get("reward_model") is not None:
        cleaned["ground_truth"] = task["reward_model"]
    if not cleaned.get("data_source"):
        mode = resolve_task_mode(task)
        if mode in {"mcp", "cli", "et"}:
            cleaned["data_source"] = mode
    return cleaned


def _episode_id(sample: Sample, task: dict[str, Any]) -> str:
    task_key = task.get("id") or task.get("uuid") or task.get("instance_id") or task.get("question") or task
    task_hash = hashlib.sha256(json.dumps(task_key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:32]
    rollout_idx = sample.rollout_id if sample.rollout_id is not None else sample.index
    if rollout_idx is None:
        rollout_idx = 0
    return f"{task_hash}:{rollout_idx}"


def _benchmark_metric_name(task_for_dump: dict[str, Any], task_type: str) -> str:
    """Metric/trajectory name keyed by benchmark source (e.g. ``medqa``).

    Falls back to the normalized task_type suffix, then ``unknown``.
    """
    raw = task_for_dump.get("data_source") or task_for_dump.get("benchmark") or task_for_dump.get("dataset") or ""
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    return normalized or _task_metric_suffix(task_type)


def _task_metric_suffix(task_type: str) -> str:
    normalized = str(task_type or "").lower().replace("-", "_").replace(" ", "_")
    if normalized in {"web_search", "search", "webqa"}:
        return "webqa"
    if normalized in {"mcp", "cli"}:
        return normalized
    if normalized in {"et", "endless_terminal", "endless_terminals", "swe"}:
        return "cli"
    return normalized or "unknown"


def _max_steps_for_mode(task_type: str, default: int) -> int:
    normalized = str(task_type or "").upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "WEB_SEARCH": ("FUSED_WEB_SEARCH_MAX_STEPS", "FUSED_WEBQA_MAX_STEPS"),
        "MCP": ("FUSED_MCP_MAX_STEPS",),
        "CLI": ("FUSED_CLI_MAX_STEPS",),
        "ET": ("FUSED_ET_MAX_STEPS", "FUSED_CLI_MAX_STEPS"),
    }
    for env_name in aliases.get(normalized, ()):
        value = os.environ.get(env_name)
        if value:
            return max(1, int(value))
    return default


def _effective_sglang_context_limit(args) -> int:
    limits = [
        int(value)
        for value in (
            getattr(args, "sglang_context_length", None),
            getattr(args, "rollout_max_context_len", None),
        )
        if value
    ]
    if not limits:
        return 0
    margin = max(0, int(os.environ.get("SGLANG_CONTEXT_LENGTH_MARGIN", DEFAULT_SGLANG_CONTEXT_LENGTH_MARGIN)))
    return max(0, min(limits) - margin)


async def _call_sglang(
    args,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    *,
    session_id: str,
    evaluation: bool = False,
    request_semaphore: asyncio.Semaphore | None = None,
    session_params: dict[str, Any] | None = None,
    context_token_count: int | None = None,
    server_url: str | None = None,
    expected_weight_version: str | None = None,
    require_weight_version: bool = False,
) -> dict[str, Any]:
    global _LAST_SGLANG_REQUEST_LOG_TS
    max_new_tokens = int(sampling_params.get("max_new_tokens", 0) or 0)
    request_sampling_params = {
        **sampling_params,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
    }
    top_p_replay_required = not evaluation and float(getattr(args, "rollout_top_p", 1.0)) != 1.0
    if top_p_replay_required:
        request_sampling_params["custom_params"] = {
            **dict(request_sampling_params.get("custom_params") or {}),
            "return_top_p_token_ids": True,
        }
    max_context_tokens = _effective_sglang_context_limit(args)
    effective_prompt_tokens = context_token_count if context_token_count is not None else len(prompt_ids)
    requested_tokens = effective_prompt_tokens + max_new_tokens
    if max_context_tokens and requested_tokens > max_context_tokens:
        raise SGLangContextLengthExceededError(
            f"SGLang request would use {requested_tokens} tokens "
            f"({effective_prompt_tokens} prompt + {max_new_tokens} new), exceeding local limit {max_context_tokens}."
        )
    url = (
        f"{server_url.rstrip('/')}/generate"
        if server_url
        else f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    )
    rid = uuid.uuid4().hex
    payload = {
        "rid": rid,
        "input_ids": prompt_ids,
        "sampling_params": request_sampling_params,
        "return_logprob": not evaluation,
    }
    if session_params is not None:
        payload["session_params"] = session_params
    headers = _routing_headers(args, session_id)
    if server_url is not None:
        # Native session requests bypass the router and can leave an origin
        # idle for an arbitrary tool/retrieval interval. A fresh local TCP
        # connection avoids ambiguous ReadError failures from stale Uvicorn
        # keepalive sockets without changing request or GPU concurrency.
        headers = {**(headers or {}), "Connection": "close"}
    started = time.time()
    now = started
    should_log = _env_bool("SLIME_FUSED_PROGRESS_LOGS", False) and now - _LAST_SGLANG_REQUEST_LOG_TS >= float(
        os.environ.get("SLIME_FUSED_SGLANG_LOG_INTERVAL", "10")
    )
    if should_log:
        _LAST_SGLANG_REQUEST_LOG_TS = now
        logger.info(
            "fused-agent sending SGLang generate request rid=%s prompt_tokens=%d max_new_tokens=%d url=%s",
            rid,
            effective_prompt_tokens,
            max_new_tokens,
            url,
        )
    try:
        post_kwargs = {"headers": headers}
        if session_params is not None:
            post_kwargs["max_retries"] = 1
        if request_semaphore is None:
            output = await http_utils.post(url, payload, **post_kwargs)
        else:
            async with request_semaphore:
                output = await http_utils.post(url, payload, **post_kwargs)
    except (asyncio.CancelledError, httpx.TransportError):
        await _abort_sglang_request(args, rid, server_url=server_url)
        raise
    meta = output.get("meta_info") or {}
    observed_weight_version = meta.get("weight_version")
    if observed_weight_version is not None:
        observed_weight_version = str(observed_weight_version)
    if require_weight_version and observed_weight_version is None:
        raise SGLangWeightVersionError(
            "SGLang response omitted weight_version while strict fused rollout versioning is enabled"
        )
    if expected_weight_version is not None and observed_weight_version != str(expected_weight_version):
        raise SGLangWeightVersionError(
            "SGLang weight version changed within one fused trajectory: "
            f"expected {expected_weight_version!r}, observed {observed_weight_version!r}"
        )
    finish_reason = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    if evaluation:
        completion_tokens = meta.get("completion_tokens")
        return {
            "text": output.get("text") or "",
            "finish_reason": finish_reason,
            "prompt_tokens": int(meta.get("prompt_tokens", effective_prompt_tokens)),
            "completion_tokens": int(completion_tokens) if completion_tokens is not None else None,
            "output_ids": output.get("output_ids") or [],
            "rid": meta.get("id") or output.get("rid") or rid,
        }
    token_logprobs = meta.get("output_token_logprobs") or []
    # Unpacking per-token logprob/top-p payloads is CPU work proportional to
    # the response length; offload long responses to keep the event loop free.
    if len(token_logprobs) >= 256:
        output_ids, output_logprobs, top_p_data = await asyncio.to_thread(_unpack_generate_meta, meta, token_logprobs)
    else:
        output_ids, output_logprobs, top_p_data = _unpack_generate_meta(meta, token_logprobs)
    if output.get("text") and not output_ids:
        raise ValueError("SGLang returned generated text without output token/logprob evidence")
    if top_p_replay_required and top_p_data is None:
        raise ValueError("SGLang response omitted top-p token replay metadata requested for fused-agent training")
    invalid_logprobs = [index for index, value in enumerate(output_logprobs) if not math.isfinite(value)]
    if invalid_logprobs:
        raise ValueError(f"SGLang returned non-finite output logprobs at indices {invalid_logprobs[:8]}")
    result = {
        "text": output.get("text") or "",
        "output_ids": output_ids,
        "output_logprobs": output_logprobs,
        "finish_reason": finish_reason,
        "weight_version": observed_weight_version,
    }
    if top_p_data is not None:
        result["rollout_top_p_token_ids"], result["rollout_top_p_token_offsets"] = top_p_data
    if should_log:
        logger.info(
            "fused-agent received SGLang generate response rid=%s output_tokens=%d finish_reason=%s elapsed=%.2fs",
            rid,
            len(output_ids),
            finish_reason,
            time.time() - started,
        )
    return result


def _should_capture_eval_trajectory(sample: Sample) -> bool:
    rate = min(1.0, max(0.0, float(os.environ.get("SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE", "0"))))
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    key = sample.index if sample.index is not None else sample.session_id or sample.prompt
    bucket = int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def _is_failed_eval_termination(termination_reason: str) -> bool:
    if not _env_bool("SLIME_FUSED_EVAL_DUMP_FAILURES", True):
        return False
    return termination_reason not in {"env_done", "reasoning_only"}


def _unpack_generate_meta(
    meta: dict[str, Any], token_logprobs: list
) -> tuple[list[int], list[float], tuple[list[int], list[int]] | None]:
    output_ids = [x[1] for x in token_logprobs]
    output_logprobs = [float(x[0]) for x in token_logprobs]
    top_p_data = _extract_rollout_top_p_token_data(meta, expected_num_tokens=len(output_ids))
    return output_ids, output_logprobs, top_p_data


async def _abort_sglang_request(args, rid: str, *, server_url: str | None = None) -> None:
    client = http_utils._http_client
    if client is None:
        return
    try:
        await client.post(
            (
                f"{server_url.rstrip('/')}/abort_request"
                if server_url
                else f"http://{args.sglang_router_ip}:{args.sglang_router_port}/abort_request"
            ),
            json={"rid": rid},
            timeout=5.0,
        )
    except Exception:
        pass
