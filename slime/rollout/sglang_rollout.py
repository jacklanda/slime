import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from argparse import Namespace
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.backends.sglang_utils.server_control import (
    ABORT_HTTP_TIMEOUT_SECONDS,
    ABORT_TIMEOUT_SECONDS,
    abort_servers_until_idle,
)
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import DynamicFilterOutput, MetricGatherer, call_dynamic_filter
from slime.rollout.failure_types import FailureClass
from slime.rollout.filter_hub.dynamic_sampling_filters import group_failure_class
from slime.rollout.task_family import (
    has_task_family_quota_candidates,
    parse_task_family_quotas,
    sample_group_mean_steps,
    sample_group_task_family,
    sample_task_family,
    select_task_family_quota_groups,
    task_family_quota_counts,
)
from slime.utils.async_utils import run
from slime.utils import http_utils
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, get_rollout_num_engines, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import build_sglang_meta_trace_attrs, trace_function, trace_span
from slime.utils.types import Sample
from slime.utils.visualization import maybe_print_rollout_group

from .rm_hub import async_rm, batched_async_rm
from .prefill_logprobs import recompute_rollout_logprobs_via_prefill

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}
_TOP_P_TOKEN_ID_META_KEYS = ("top_p_token_ids", "top_p_kept_token_ids")
_TOP_P_TOKEN_OFFSET_META_KEYS = ("top_p_token_offsets", "top_p_kept_token_offsets")
_ENGINE_METRIC_RE = re.compile(r"^(sglang[:_](?:num_running_reqs|num_queue_reqs|token_usage|cache_hit_rate))(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$")


class _EvalProgressLogStream:
    """Render tqdm carriage-return refreshes as complete Ray log lines."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        if text.startswith("\r"):
            text = text[1:] + "\n"
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class EvalEngineLoad:
    engine_count: int
    running_requests: float
    waiting_requests: float
    max_token_usage: float
    mean_cache_hit_rate: float


def _parse_eval_engine_metrics(text: str) -> EvalEngineLoad | None:
    values: dict[str, list[float]] = {
        "sglang:num_running_reqs": [],
        "sglang:num_queue_reqs": [],
        "sglang:token_usage": [],
        "sglang:cache_hit_rate": [],
    }
    workers: set[str] = set()
    for line in text.splitlines():
        match = _ENGINE_METRIC_RE.match(line.strip())
        if match is None:
            continue
        name, labels, raw_value = match.groups()
        name = name.replace("sglang_", "sglang:", 1)
        if labels:
            worker_match = re.search(r'worker_addr="([^"]+)"', labels)
            if worker_match is not None:
                workers.add(worker_match.group(1))
            if re.search(r'priority="[^"]+"', labels):
                continue
        try:
            values[name].append(float(raw_value))
        except ValueError:
            continue
    token_usage = values["sglang:token_usage"]
    if not token_usage:
        return None
    engine_count = len(workers) or len(token_usage)
    cache_hits = values["sglang:cache_hit_rate"]
    return EvalEngineLoad(
        engine_count=max(1, engine_count),
        running_requests=sum(values["sglang:num_running_reqs"]),
        waiting_requests=sum(values["sglang:num_queue_reqs"]),
        max_token_usage=max(token_usage),
        mean_cache_hit_rate=sum(cache_hits) / len(cache_hits) if cache_hits else 0.0,
    )


async def _fetch_eval_engine_load(args: Namespace) -> EvalEngineLoad | None:
    client = http_utils._http_client
    if client is None:
        return None
    try:
        response = await client.get(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/engine_metrics",
            timeout=float(getattr(args, "eval_concurrency_poll_interval", 5.0)),
        )
        response.raise_for_status()
        return _parse_eval_engine_metrics((await response.aread()).decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.debug("Unable to sample SGLang engine metrics for adaptive eval concurrency: %s", exc)
        return None


class EvalConcurrencyController:
    def __init__(self, args: Namespace, total: int):
        self.maximum = max(1, min(total, int(args.eval_max_inflight_tasks)))
        configured_initial = getattr(args, "eval_initial_inflight_tasks", None)
        self.target = max(1, min(self.maximum, int(configured_initial or self.maximum)))
        self.minimum = max(1, min(self.target, self.target // 2))
        self.step = max(1, int(getattr(args, "eval_concurrency_step", 32)))
        self.poll_interval = max(0.1, float(getattr(args, "eval_concurrency_poll_interval", 5.0)))
        self.server_concurrency = max(1, int(args.sglang_server_concurrency))
        self.next_poll_at = 0.0
        self.poll_task: asyncio.Task | None = None
        self.healthy_samples = 0

    def update(self, load: EvalEngineLoad) -> int:
        request_capacity = self.server_concurrency * load.engine_count
        queue_pressure = load.waiting_requests >= max(4 * load.engine_count, 0.25 * request_capacity)
        cache_pressure = load.max_token_usage >= 0.90 or (load.max_token_usage >= 0.84 and load.mean_cache_hit_rate < 0.50)
        if queue_pressure or cache_pressure:
            self.healthy_samples = 0
            self.target = max(self.minimum, self.target - self.step)
            return self.target

        underfed = load.max_token_usage < 0.82 and load.waiting_requests <= load.engine_count and load.running_requests < 0.75 * request_capacity
        self.healthy_samples = self.healthy_samples + 1 if underfed else 0
        if self.healthy_samples >= 2:
            self.target = min(self.maximum, self.target + self.step)
            self.healthy_samples = 0
        return self.target

    def poll(self, args: Namespace) -> int:
        if self.poll_task is not None and self.poll_task.done():
            try:
                load = self.poll_task.result()
            except Exception as exc:
                logger.debug("Adaptive eval concurrency metric task failed: %s", exc)
                load = None
            self.poll_task = None
            if load is not None:
                previous = self.target
                self.update(load)
                if self.target != previous:
                    logger.info(
                        "Adaptive concurrency changed %d -> %d " "(engines=%d running=%.0f waiting=%.0f kv=%.3f cache_hit=%.3f)",
                        previous,
                        self.target,
                        load.engine_count,
                        load.running_requests,
                        load.waiting_requests,
                        load.max_token_usage,
                        load.mean_cache_hit_rate,
                    )
        now = time.monotonic()
        if self.poll_task is None and now >= self.next_poll_at:
            self.poll_task = asyncio.create_task(_fetch_eval_engine_load(args))
            self.next_poll_at = now + self.poll_interval
        return self.target

    async def close(self) -> None:
        if self.poll_task is None:
            return
        self.poll_task.cancel()
        await asyncio.gather(self.poll_task, return_exceptions=True)
        self.poll_task = None


def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (sample.multimodal_train_inputs is not None or not has_multimodal_inputs)

    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {k: v for k, v in processor_output.items() if k not in _PROCESSOR_PROMPT_KEYS} or None
        return prompt_ids

    if reuse_existing_input_ids:
        return sample.tokens

    return tokenizer.encode(sample.prompt, add_special_tokens=False)


def _decode_int32_meta_array(meta_info: dict[str, Any], keys: tuple[str, ...]) -> list[int] | None:
    for key in keys:
        if key in meta_info:
            value = meta_info[key]
            break
    else:
        return None

    if value is None:
        return None
    if isinstance(value, str):
        value = pybase64.b64decode(value.encode("ascii"))
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype=np.int32).tolist()
    if isinstance(value, np.ndarray):
        return value.astype(np.int32, copy=False).tolist()
    return [int(x) for x in value]


def _extract_rollout_top_p_token_data(
    meta_info: dict[str, Any],
    *,
    expected_num_tokens: int | None = None,
) -> tuple[list[int], list[int]] | None:
    token_ids = _decode_int32_meta_array(meta_info, _TOP_P_TOKEN_ID_META_KEYS)
    offsets = _decode_int32_meta_array(meta_info, _TOP_P_TOKEN_OFFSET_META_KEYS)
    if token_ids is None and offsets is None:
        return None
    if token_ids is None or offsets is None:
        raise ValueError("SGLang top-p token replay must include both token ids and offsets.")
    if not offsets or offsets[0] != 0:
        raise ValueError(f"SGLang top-p token offsets must start with 0, got {offsets[:1]}.")
    if offsets[-1] != len(token_ids):
        raise ValueError(f"SGLang top-p token ids/offsets mismatch: offsets[-1]={offsets[-1]}, len(token_ids)={len(token_ids)}.")
    if expected_num_tokens is not None and len(offsets) != expected_num_tokens + 1:
        raise ValueError("SGLang top-p token offsets length must equal generated token count + 1: " f"len(offsets)={len(offsets)}, generated={expected_num_tokens}.")
    return token_ids, offsets


def _merge_rollout_top_p_token_data(
    base_token_ids: list[int] | None,
    base_offsets: list[int] | None,
    token_ids: list[int],
    offsets: list[int],
) -> tuple[list[int], list[int]]:
    base_token_ids = list(base_token_ids or [])
    base_offsets = list(base_offsets or [0])
    base_offset = base_offsets[-1]
    return base_token_ids + token_ids, base_offsets + [base_offset + offset for offset in offsets[1:]]


def _append_rollout_top_p_token_data(
    sample: Sample,
    meta_info: dict[str, Any],
    *,
    expected_num_tokens: int | None = None,
) -> None:
    top_p_data = _extract_rollout_top_p_token_data(meta_info, expected_num_tokens=expected_num_tokens)
    if top_p_data is None:
        return
    sample.rollout_top_p_token_ids, sample.rollout_top_p_token_offsets = _merge_rollout_top_p_token_data(
        sample.rollout_top_p_token_ids,
        sample.rollout_top_p_token_offsets,
        *top_p_data,
    )


def _should_use_grm(args: Namespace, sample: Sample, evaluation: bool) -> bool:
    flag = "enable_use_grm_evals" if evaluation else "enable_use_grm_train"
    if not bool(getattr(args, flag, False)):
        return False
    return sample_task_family(sample) == "webqa"


async def _score_samples_with_grm(args: Namespace, samples: list[Sample], *, evaluation: bool) -> None:
    if not samples:
        return
    grm_path = getattr(args, "grm_custom_rm_path", None)
    for sample in samples:
        sample.custom_rm_path = grm_path
        sample.reward = None
    trace_name = "grm_eval_reward_model" if evaluation else "grm_train_reward_model"
    with trace_span(samples, trace_name):
        rewards = await batched_async_rm(args, samples, evaluation=evaluation)
    for sample, reward in zip(samples, rewards, strict=False):
        sample.reward = reward


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(args.sglang_server_concurrency * get_rollout_num_engines(args))
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            min_p=getattr(args, "rollout_min_p", 0.0),
            presence_penalty=getattr(args, "rollout_presence_penalty", 0.0),
            repetition_penalty=getattr(args, "rollout_repetition_penalty", 1.0),
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )
        if args.rollout_top_p != 1.0:
            self.sampling_params["custom_params"] = {"return_top_p_token_ids": True}

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0
        self.quarantined_tasks: set[asyncio.Task] = set()
        self.retired = False

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        if self.retired:
            return
        self.remaining_batch_size = 0
        self.pendings = set()
        self.pending_groups = {}
        self.aborted = False

    def retire(self) -> None:
        """Prevent a failed rollout generation from sharing state with its successor."""
        self.aborted = True
        self.retired = True
        SingletonMeta._instances.pop(type(self), None)

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            task = asyncio.create_task(
                # submit a group of samples as a single task.
                generate_and_rm_group(
                    self.args,
                    group,
                    sampling_params=self.sampling_params.copy(),
                    evaluation=False,
                )
            )
            self.pendings.add(task)
            self.pending_groups[task] = group
        self.remaining_batch_size += len(samples)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED, f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    assert sampling_params["max_new_tokens"] >= 0, f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None
    if images:
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in images]
        # For single-turn multimodal requests, send text so SGLang expands the
        # image placeholders with its own processor rules.
        payload["text"] = sample.prompt
    else:
        payload["input_ids"] = prompt_ids

    if not sample.tokens:
        sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, max_retries=1, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )

    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    if _should_use_grm(args, sample, evaluation) and (
        sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED
    ):
        await _score_samples_with_grm(args, [sample], evaluation=evaluation)
        return sample

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)
    retry_base_sample = copy.deepcopy(sample)
    retry_times = max(0, int(getattr(args, "eval_termination_retry_times", 0))) if evaluation else 0
    base_sampling_seed = sampling_params.get("sampling_seed")
    retry_reasons: list[str] = []

    custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path
    custom_generate_func = load_function(custom_func_path) if custom_func_path is not None else None
    manages_request_concurrency = custom_generate_func is not None and (
        getattr(custom_generate_func, "manages_request_concurrency", False)
        or (evaluation and getattr(custom_generate_func, "manages_eval_request_concurrency", False))
    )

    for attempt in range(retry_times + 1):
        current_sampling_params = sampling_params
        if base_sampling_seed is not None and attempt > 0:
            current_sampling_params = sampling_params.copy()
            current_sampling_params["sampling_seed"] = int(base_sampling_seed) + attempt * 1_000_003

        if manages_request_concurrency:
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                return sample
            with state.dp_rank_context() as _:
                sample = await custom_generate_func(args, sample, current_sampling_params, evaluation=evaluation)
        else:
            async with state.semaphore:
                if state.aborted:
                    sample.status = Sample.Status.ABORTED
                    return sample

                with state.dp_rank_context() as _:
                    if custom_generate_func is not None:
                        # if signature has evaluation, pass evaluation
                        if "evaluation" in inspect.signature(custom_generate_func).parameters:
                            sample = await custom_generate_func(
                                args, sample, current_sampling_params, evaluation=evaluation
                            )
                        else:
                            sample = await custom_generate_func(args, sample, current_sampling_params)
                    else:
                        sample = await generate(args, sample, current_sampling_params)

        generated_samples = sample if isinstance(sample, list) else [sample]
        termination_reason = _eval_termination_reason(generated_samples)
        if not _is_retryable_eval_termination(termination_reason) or attempt >= retry_times:
            if evaluation:
                for generated_sample in generated_samples:
                    generated_sample.metadata = {
                        **dict(generated_sample.metadata or {}),
                        "eval_retry_count": attempt,
                        "eval_retry_termination_reasons": retry_reasons,
                    }
            break

        retry_reasons.append(termination_reason)
        logger.warning(
            "Retrying trace index=%s after termination_reason=%s (retry %d/%d)",
            retry_base_sample.index,
            termination_reason,
            attempt + 1,
            retry_times,
        )
        sample = copy.deepcopy(retry_base_sample)
        sample.session_id = None

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        grm_samples = [sample for sample in samples if _should_use_grm(args, sample, evaluation)]
        non_grm_samples = [sample for sample in samples if not _should_use_grm(args, sample, evaluation)]
        await _score_samples_with_grm(args, grm_samples, evaluation=evaluation)

        for sample in non_grm_samples:
            if _should_rescore_eval_sample(args, sample, evaluation):
                sample.reward = None
        samples_need_reward = [sample for sample in non_grm_samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward, evaluation=evaluation)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        if _should_use_grm(args, sample, evaluation):
            await _score_samples_with_grm(args, [sample], evaluation=evaluation)
            return sample
        if _should_rescore_eval_sample(args, sample, evaluation):
            sample.reward = None
        # Some custom generate paths may have already filled the reward.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample, evaluation=evaluation)

    return sample


def _eval_termination_reason(samples: list[Sample]) -> str | None:
    for sample in reversed(samples):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        reason = metadata.get("fused_termination") or metadata.get("termination_reason")
        if reason:
            return str(reason)
    return None


def _is_retryable_eval_termination(reason: str | None) -> bool:
    if reason in {None, "env_done", "reasoning_only"}:
        return False
    # Repeating the same parsed search action is a deterministic model behavior.
    # Retrying the whole trajectory only multiplies requests and warning logs.
    return not reason.endswith("_duplicate_search")


def _should_rescore_eval_sample(args: Namespace, sample: Sample, evaluation: bool) -> bool:
    if not evaluation:
        return False
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    return rm_type == "benchmark_verifier"


@trace_function(
    "generate_and_rm_group",
    target="group",
    attrs_getter=lambda args, group, sampling_params, evaluation=False: {"group_size": len(group)},
)
async def generate_and_rm_group(args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False) -> list[Sample] | list[list[Sample]]:
    # ``generate_and_rm`` may return either a ``Sample`` or a ``list[Sample]``
    # depending on whether the ``--custom-generate-function-path`` callable
    # emits one trainable sample or several (e.g. multi-turn agent rollouts
    # that fan out into multiple prefix-chained samples). The asyncio.gather
    # below preserves whichever shape each task produced, so the group is
    # ``list[Sample]`` for plain rollouts and ``list[list[Sample]]`` for
    # the fan-out case.
    state = GenerateState(args)
    group_started_at = time.time()

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    group_samples = group
    slot_bases = copy.deepcopy(group_samples)
    slot_sampling_params = []
    for idx in range(len(group_samples)):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = state.group_sampling_seeds[idx]
        slot_sampling_params.append(current_sampling_params)

    retry_times = max(0, int(getattr(args, "rollout_infra_retry_times", 2) or 0))
    group_timeout = _rollout_group_timeout(args, evaluation=evaluation)
    slot_results: list[Any | None] = [None] * len(group_samples)
    attempts = [0] * len(group_samples)
    pending_slots = list(range(len(group_samples)))

    while pending_slots:
        tasks: dict[asyncio.Task, tuple[int, Sample]] = {}
        for slot in pending_slots:
            sample = group_samples[slot] if attempts[slot] == 0 else copy.deepcopy(slot_bases[slot])
            if attempts[slot] > 0:
                sample.session_id = str(uuid.uuid4())
            sample.metadata = {
                **dict(sample.metadata or {}),
                "rollout_logical_slot": slot,
                "infra_retry_attempt": attempts[slot],
            }
            task = asyncio.create_task(
                generate_and_rm(args, sample, slot_sampling_params[slot].copy(), evaluation=evaluation)
            )
            tasks[task] = (slot, sample)

        try:
            done, unfinished = await asyncio.wait(tasks, timeout=group_timeout)
        except asyncio.CancelledError:
            # asyncio.wait does not propagate cancellation to the tasks it is
            # watching. Without this cleanup, per-sample producers survive a
            # cancelled group and can submit new SGLang requests after rollout
            # offload has started.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if unfinished:
            logger.warning(
                "Rollout group attempt timed out after %.1fs; completed=%d unfinished=%d; cancelling unfinished slots.",
                group_timeout,
                len(done),
                len(unfinished),
            )
            for task in unfinished:
                task.cancel()
            await asyncio.gather(*unfinished, return_exceptions=True)

        for task, (slot, sample) in tasks.items():
            if task in unfinished:
                slot_results[slot] = _timeout_sample(sample, evaluation=evaluation)
                continue
            try:
                slot_results[slot] = task.result()
            except Exception as exc:
                logger.error(
                    "Rollout task failed for sample index=%s; converting only this slot to a failed sample.",
                    sample.index,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                slot_results[slot] = _failed_task_sample(sample, exc, evaluation=evaluation)

        retry_slots = []
        for slot in pending_slots:
            result = slot_results[slot]
            if _result_failure_class(result) != FailureClass.RETRYABLE_INFRA or attempts[slot] >= retry_times:
                continue
            attempts[slot] += 1
            retry_slots.append(slot)
            logger.warning(
                "Retrying rollout logical slot=%d sample_index=%s after retryable infra failure (%d/%d).",
                slot,
                slot_bases[slot].index,
                attempts[slot],
                retry_times,
            )
        pending_slots = retry_slots

    group = slot_results
    for slot, (result, attempt) in enumerate(zip(group, attempts, strict=True)):
        for sample in _flatten_samples([result]):
            sample.metadata = {
                **dict(sample.metadata or {}),
                "rollout_logical_slot": slot,
                "infra_retry_attempt": attempt,
                "infra_retry_count": attempt,
            }

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group, evaluation=evaluation)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    group_finished_at = time.time()
    for sample in _flatten_samples(group):
        metadata = dict(sample.metadata or {})
        profile = dict(metadata.get("fused_profile") or {})
        profile.update(
            {
                "group_start_time_s": group_started_at,
                "group_end_time_s": group_finished_at,
                "group_total_time_s": group_finished_at - group_started_at,
            }
        )
        metadata["fused_profile"] = profile
        episode = metadata.get("rllm_episode")
        if isinstance(episode, dict):
            episode_metadata = episode.setdefault("metadata", {})
            if isinstance(episode_metadata, dict):
                episode_metadata["fused_profile"] = dict(profile)
        sample.metadata = metadata
    return group


def _rollout_group_timeout(args: Namespace, *, evaluation: bool) -> float:
    env_name = "SLIME_EVAL_ROLLOUT_GROUP_TIMEOUT" if evaluation else "SLIME_ROLLOUT_GROUP_TIMEOUT"
    if value := os.environ.get(env_name):
        return max(1.0, float(value))
    if value := os.environ.get("SLIME_ROLLOUT_GROUP_TIMEOUT"):
        return max(1.0, float(value))
    fused_env = "FUSED_EVAL_TRAJECTORY_TIMEOUT" if evaluation else "FUSED_TRAJECTORY_TIMEOUT"
    if value := os.environ.get(fused_env):
        return max(1.0, float(value))
    return float(getattr(args, "rollout_group_timeout", 3600.0))


def _timeout_sample(sample: Sample, *, evaluation: bool) -> Sample:
    timed_out = copy.deepcopy(sample)
    timed_out.status = Sample.Status.FAILED
    timed_out.reward = 0.0
    timed_out.rollout_log_probs = [0.0] * timed_out.response_length
    stage = str((sample.metadata or {}).get("rollout_stage") or "unknown")
    # A group deadline alone cannot prove that normal decode/tool work failed.
    # Only stages entered by a dedicated transport/lease failure are replaceable.
    failure_class = (
        FailureClass.RETRYABLE_INFRA.value
        if stage in {"mcp_lease", "sglang_transport"}
        else FailureClass.PERMANENT_TASK.value
        if stage == "verifier"
        else FailureClass.POLICY.value
    )
    timed_out.metadata = {
        **dict(sample.metadata or {}),
        "termination_reason": "timeout",
        "fused_termination": "timeout",
        "fused_error": "rollout_group_timeout",
        "rollout_timeout_stage": stage,
        "failure_class": failure_class,
        "evaluation": evaluation,
    }
    return timed_out


def _failed_task_sample(sample: Sample, exc: BaseException, *, evaluation: bool) -> Sample:
    failed = _timeout_sample(sample, evaluation=evaluation)
    failed.metadata.update(
        {
            "termination_reason": "rollout_task_exception",
            "fused_termination": "rollout_task_exception",
            "fused_error": "rollout_task_exception",
            "rollout_exception_type": type(exc).__name__,
            "rollout_exception": str(exc),
            "failure_class": _exception_failure_class(exc).value,
        }
    )
    return failed


def _exception_failure_class(exc: BaseException) -> FailureClass:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return FailureClass.RETRYABLE_INFRA
    if isinstance(exc, (ConnectionError, TimeoutError)) or type(exc).__name__ in {
        "BrokenProcessPool",
        "ConnectError",
        "ConnectionClosed",
        "MCPLeaseTimeout",
        "PoolClosed",
        "ReadError",
        "RemoteProtocolError",
        "TransportError",
    }:
        return FailureClass.RETRYABLE_INFRA
    return FailureClass.PERMANENT_TASK


def _result_failure_class(result: Any) -> FailureClass | None:
    return group_failure_class([result])


def _group_has_trainable_response(group: list[Sample] | list[list[Sample]]) -> bool:
    return any(sample.status == Sample.Status.COMPLETED and sample.response_length > 0 for sample in _flatten_samples(group))


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []
    count = 0

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    cancellation_incomplete = False

    # Stop multi-turn producers before aborting the servers. Otherwise an
    # aborted generate call can return to the agent, which immediately submits
    # its next turn and prevents the server from ever becoming idle.
    if state.pendings:
        stale_tasks = tuple(state.pendings)
        logger.info("Cancelling %d remaining rollout tasks before SGLang abort", len(stale_tasks))
        for task in stale_tasks:
            task.cancel()
        cancellation_done, cancellation_pending = await asyncio.wait(
            stale_tasks,
            timeout=ABORT_TIMEOUT_SECONDS,
        )
        if cancellation_done:
            await asyncio.gather(*cancellation_done, return_exceptions=True)
        if cancellation_pending:
            cancellation_incomplete = True
            state.quarantined_tasks.update(cancellation_pending)
            for task in cancellation_pending:
                task.add_done_callback(state.quarantined_tasks.discard)
            logger.error(
                "%d stale rollout tasks did not finish cancellation; quarantining them and requiring engine restart",
                len(cancellation_pending),
            )
        if args.partial_rollout:
            for task in stale_tasks:
                group = state.pending_groups.get(task)
                if group is None:
                    continue
                for sample in group:
                    if sample.response and "start_rollout_id" not in sample.metadata:
                        sample.metadata["start_rollout_id"] = rollout_id
                aborted_samples.append(group)
                count += len(group)
        state.pendings.clear()
        state.pending_groups.clear()

    urls = [url for url in getattr(args, "sglang_engine_urls", []) if url]
    if not urls:
        try:
            if parse(sglang_router.__version__) <= parse("0.2.1"):
                response = await asyncio.wait_for(
                    get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers"),
                    timeout=ABORT_HTTP_TIMEOUT_SECONDS,
                )
                urls = response["urls"]
            else:
                response = await asyncio.wait_for(
                    get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"),
                    timeout=ABORT_HTTP_TIMEOUT_SECONDS,
                )
                urls = [worker["url"] for worker in response["workers"]]
        except Exception:
            logger.exception("Failed to discover SGLang workers from the router during abort")

    restart_required = cancellation_incomplete
    if urls:
        engines_idle = await abort_servers_until_idle(urls)
        if not engines_idle or cancellation_incomplete:
            # The control-plane abort timed out, so the server-side requests
            # may outlive these Python tasks.  Tell RolloutManager to replace
            # the engine actors before any offload/onload cycle can reuse them.
            args._rollout_abort_engine_restart_required = True
            restart_required = True
            logger.error(
                "SGLang abort left cancellation or engine state incomplete; marking rollout engines for restart"
            )
    else:
        logger.error("No SGLang engine URLs are available during abort; local rollout tasks were cancelled")
        args._rollout_abort_engine_restart_required = True
        restart_required = True

    if restart_required:
        # Quarantined producers and HTTP requests can retain permits from the
        # current singleton semaphore/connection pool. Retire both generations
        # so replacement engines are never paired with polluted client state.
        state.retire()
        try:
            await http_utils.reset_http_client(args)
        except Exception:
            # Engine replacement is the authoritative recovery step. A broken
            # client transport must not prevent RolloutManager from reaching it.
            logger.exception("Failed to reset rollout HTTP transport during abort; engine restart will retry setup")

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(args: Namespace, rollout_id: int, data_source) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size
    quota_spec = getattr(args, "rollout_task_family_quotas", None)
    task_family_quotas = parse_task_family_quotas(quota_spec)
    admission_only_quotas = bool(
        task_family_quotas and getattr(args, "rollout_task_family_admission_only", False)
    )
    enforce_post_filter_quotas = bool(task_family_quotas and not admission_only_quotas)
    prefer_higher_mean_steps = bool(
        enforce_post_filter_quotas and getattr(args, "rollout_task_family_top_mean_steps", False)
    )

    data = []
    all_data = [] if args.rollout_all_samples_process_path is not None else None
    do_print = True
    filter_relax_after = int(getattr(args, "fully_async_filter_relax_after_groups", 0) or 0)
    completed_groups = 0
    dropped_groups = 0
    dropped_family_counts: Counter[str] = Counter()
    drop_reasons: Counter[str] = Counter()
    dropped_terminations: Counter[str] = Counter()
    dropped_rewards: Counter[str] = Counter()
    completed_family_counts: Counter[str] = Counter()
    accepted_family_counts: Counter[str] = Counter()
    submitted_family_counts: Counter[str] = Counter()
    started = time.time()
    last_log = started
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Trace collection")
    while len(data) < target_data_size or (
        enforce_post_filter_quotas
        and not has_task_family_quota_candidates(data, target_data_size, task_family_quotas)
    ):
        if task_family_quotas:
            pending_family_counts = Counter(
                sample_group_task_family(group) for group in state.pending_groups.values()
            )
            submit_limit = max(0, args.over_sampling_batch_size - len(state.pendings))
            min_pending_groups = min(
                args.over_sampling_batch_size,
                max(0, int(os.environ.get("SLIME_SYNC_MIN_PENDING_GROUPS", "0"))),
            )
            mcp_only_min_pending_groups = min(
                min_pending_groups,
                max(
                    0,
                    int(os.environ.get("SLIME_SYNC_MCP_ONLY_MIN_PENDING_GROUPS", str(min_pending_groups))),
                ),
            )
            family_min_pending_groups = {}
            for family in task_family_quotas:
                family_min_pending = min(
                    min_pending_groups,
                    max(
                        0,
                        int(os.environ.get(f"SLIME_SYNC_{family.upper()}_MIN_PENDING_GROUPS", "0")),
                    ),
                )
                if family_min_pending > 0:
                    family_min_pending_groups[family] = family_min_pending
            if admission_only_quotas:
                family_plan = _task_family_admission_plan(
                    submit_limit,
                    target_data_size - len(data),
                    task_family_quotas,
                    submitted_family_counts,
                    len(state.pendings),
                    min_pending_groups,
                )
            else:
                family_plan = _task_family_submission_plan(
                    submit_limit,
                    target_data_size,
                    task_family_quotas,
                    accepted_family_counts,
                    completed_family_counts,
                    pending_family_counts,
                    min_pending_groups,
                    mcp_only_min_pending_groups,
                    family_min_pending_groups,
                )
            if family_plan:
                if hasattr(data_source, "get_samples_by_family"):
                    samples = data_source.get_samples_by_family(family_plan)
                else:
                    get_samples = data_source.get_samples if hasattr(data_source, "get_samples") else data_source
                    samples = get_samples(sum(family_plan.values()))
                if not samples:
                    raise RuntimeError("Rollout data source returned no prompt groups")
                submitted_family_counts.update(sample_group_task_family(group) for group in samples)
                state.submit_generate_tasks(samples)
        elif state.remaining_batch_size < target_data_size:
            get_samples = data_source.get_samples if hasattr(data_source, "get_samples") else data_source
            samples = get_samples(args.over_sampling_batch_size)
            if not samples:
                raise RuntimeError("Rollout data source returned no prompt groups")
            submitted_family_counts.update(sample_group_task_family(group) for group in samples)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        collector_timeout = _rollout_group_timeout(args, evaluation=False) * (
            max(0, int(getattr(args, "rollout_infra_retry_times", 2) or 0)) + 1
        ) + 1.0
        done, state.pendings = await asyncio.wait(
            state.pendings,
            timeout=collector_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done and state.pendings:
            task = next(iter(state.pendings))
            logger.warning(
                "Rollout collector watchdog expired after %.1fs; cancelling one group that did not enforce its own deadline.",
                collector_timeout,
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            state.pendings.remove(task)
            done = {task}

        while done:
            task = done.pop()
            source_group = state.pending_groups.pop(task, None)
            if not task.done():
                if source_group is None:
                    raise RuntimeError("Timed-out rollout task is missing its source group.")
                group = [_timeout_sample(sample, evaluation=False) for sample in source_group]
            else:
                try:
                    group: list[Sample] = task.result()
                except (asyncio.CancelledError, Exception) as exc:
                    if source_group is None:
                        raise RuntimeError("Failed rollout task is missing its source group.") from exc
                    if isinstance(exc, asyncio.CancelledError):
                        group = [_timeout_sample(sample, evaluation=False) for sample in source_group]
                    else:
                        logger.exception(
                            "Rollout group task failed for sample indices=%s; converting the group to failed samples.",
                            [sample.index for sample in source_group],
                        )
                        group = [_failed_task_sample(sample, exc, evaluation=False) for sample in source_group]

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                del sample
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            if all_data is not None:
                all_data.append(group)
            completed_groups += 1
            flat_group = _flatten_samples(group)
            family = sample_group_task_family(group)
            completed_family_counts[family] += 1
            metric_gatherer.on_completed_group(args, flat_group)
            failure_class = group_failure_class(flat_group)
            if failure_class is not None:
                dynamic_filter_output = DynamicFilterOutput(keep=False, reason=failure_class.value)
            else:
                dynamic_filter_output = call_dynamic_filter(
                    dynamic_filter,
                    args,
                    flat_group,
                    rollout_id=rollout_id,
                )
            relax_filter = (
                failure_class is None
                and filter_relax_after > 0
                and completed_groups >= filter_relax_after
                and _group_has_trainable_response(group)
            )
            _append_webqa_prefilter_audit(
                rollout_id=rollout_id,
                completed_group_index=completed_groups - 1,
                samples=flat_group,
                keep=bool(dynamic_filter_output.keep or relax_filter),
                drop_reason=(
                    f"relaxed_{dynamic_filter_output.reason}"
                    if relax_filter and not dynamic_filter_output.keep
                    else dynamic_filter_output.reason
                    if not dynamic_filter_output.keep
                    else None
                ),
            )
            if not dynamic_filter_output.keep and not relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                drop_reasons[str(dynamic_filter_output.reason or "unspecified")] += 1
                for dropped_sample in _flatten_samples(group):
                    metadata = dropped_sample.metadata or {}
                    termination = metadata.get("fused_termination") or metadata.get("termination_reason") or dropped_sample.status
                    dropped_terminations[str(termination)] += 1
                    dropped_rewards[str(dropped_sample.reward)] += 1
                del dropped_sample
                dropped_groups += 1
                dropped_family_counts[family] += 1
                state.remaining_batch_size -= 1
                del source_group
                del task
                del group
                continue
            if not dynamic_filter_output.keep and relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=f"relaxed_{dynamic_filter_output.reason}")

            accepted_family_counts[family] += 1

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size or enforce_post_filter_quotas:
                # Rich-rendering a full episode (up to 128 step panels plus
                # per-token mask views) is seconds of pure-Python work; keep it
                # off the event loop so it cannot stall request dispatch for
                # every in-flight trajectory.
                await asyncio.to_thread(maybe_print_rollout_group, args, group, group_id=len(data))
                data.append(group)
                if pbar.n < pbar.total:
                    pbar.update(args.n_samples_per_prompt)
        now = time.time()
        if now - last_log > 30.0:
            valid_family_counts = Counter(sample_group_task_family(group) for group in data)
            reported_families = [
                family
                for family in ("webqa", "mcp")
                if family in task_family_quotas
                or valid_family_counts[family] > 0
                or dropped_family_counts[family] > 0
            ]
            reported_families.extend(
                sorted((valid_family_counts.keys() | dropped_family_counts.keys()) - set(reported_families))
            )
            valid_family_summary = ", ".join(
                f"{family}:{valid_family_counts[family]}" for family in reported_families
            )
            dropped_family_summary = ", ".join(
                f"{family}:{dropped_family_counts[family]}" for family in reported_families
            )
            logger.info(
                "rollout %d: valid=%d/%d (%s), dropped=%d/%d (%s), pending=%d, elapsed=%ds",
                rollout_id,
                len(data),
                target_data_size,
                valid_family_summary,
                dropped_groups,
                completed_groups,
                dropped_family_summary,
                len(state.pendings),
                int(now - started),
            )
            last_log = now

    pbar.close()
    candidate_family_counts = Counter(sample_group_task_family(group) for group in data)
    if enforce_post_filter_quotas:
        data = select_task_family_quota_groups(
            data,
            target_data_size,
            quota_spec,
            prefer_higher_mean_steps=prefer_higher_mean_steps,
        )
        required_family_counts = task_family_quota_counts(target_data_size, task_family_quotas)
        selected_family_counts = Counter(sample_group_task_family(group) for group in data)
        missing = {
            family: count - selected_family_counts[family]
            for family, count in required_family_counts.items()
            if selected_family_counts[family] < count
        }
        if missing:
            raise RuntimeError(
                "Synchronous rollout could not satisfy task family quotas: "
                f"missing={missing}, candidates={dict(candidate_family_counts)}"
            )
        if prefer_higher_mean_steps:
            selected_step_summary = {
                family: [
                    round(sample_group_mean_steps(group), 2)
                    for group in data
                    if sample_group_task_family(group) == family
                ]
                for family in task_family_quotas
            }
            logger.info("rollout %d: selected top mean-step groups=%s", rollout_id, selected_step_summary)
    else:
        data = data[:target_data_size]
        selected_family_counts = candidate_family_counts
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish group collection: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # Abort only when this rollout still owns unfinished requests.  With the
    # default over_sampling_batch_size == rollout_batch_size path, all submitted
    # groups are normally consumed before we get here, so a full server abort is
    # just noisy.
    aborted_samples = await abort(args, rollout_id) if state.pendings else []

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        assert all_data is not None
        all_samples = sorted(all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    await recompute_rollout_logprobs_via_prefill(
        args,
        _flatten_samples(data),
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        sampling_params=state.sampling_params,
    )

    metrics = metric_gatherer.collect()
    metrics["rollout/dynamic_filter/completed_groups"] = completed_groups
    metrics["rollout/dynamic_filter/dropped_groups"] = dropped_groups
    metrics["rollout/dynamic_filter/kept_groups"] = len(data)
    metrics["rollout/config/sync_min_pending_groups"] = min(
        args.over_sampling_batch_size,
        max(0, int(os.environ.get("SLIME_SYNC_MIN_PENDING_GROUPS", "0"))),
    )
    metrics["rollout/config/sync_mcp_only_min_pending_groups"] = min(
        metrics["rollout/config/sync_min_pending_groups"],
        max(
            0,
            int(
                os.environ.get(
                    "SLIME_SYNC_MCP_ONLY_MIN_PENDING_GROUPS",
                    str(metrics["rollout/config/sync_min_pending_groups"]),
                )
            ),
        ),
    )
    for family in task_family_quotas:
        metrics[f"rollout/config/sync_{family}_min_pending_groups"] = min(
            metrics["rollout/config/sync_min_pending_groups"],
            max(
                0,
                int(os.environ.get(f"SLIME_SYNC_{family.upper()}_MIN_PENDING_GROUPS", "0")),
            ),
        )
    metrics["rollout/config/task_family_admission_only"] = int(admission_only_quotas)
    metrics["rollout/config/task_family_top_mean_steps"] = int(prefer_higher_mean_steps)
    for family, count in candidate_family_counts.items():
        metrics[f"rollout/task_family_candidates/{family}"] = count
    for family, count in selected_family_counts.items():
        metrics[f"rollout/task_family_selected/{family}"] = count
        family_groups = [group for group in data if sample_group_task_family(group) == family]
        metrics[f"rollout/task_family_selected_mean_steps/{family}"] = sum(
            sample_group_mean_steps(group) for group in family_groups
        ) / len(family_groups)
    for family, count in submitted_family_counts.items():
        metrics[f"rollout/task_family_submitted/{family}"] = count
    for family, count in completed_family_counts.items():
        metrics[f"rollout/task_family_roi/{family}"] = accepted_family_counts[family] / count
    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


def _task_family_submission_plan(
    submit_limit: int,
    target: int,
    quotas: dict[str, float],
    accepted: Counter[str],
    completed: Counter[str],
    pending: Counter[str] | None = None,
    min_pending_groups: int = 0,
    mcp_only_min_pending_groups: int | None = None,
    family_min_pending_groups: dict[str, int] | None = None,
) -> dict[str, int]:
    """Size a quota-aware wave from expected yield and a GPU-work reservoir."""
    if not quotas or submit_limit <= 0:
        return {}

    pending = pending or Counter()
    family_min_pending_groups = family_min_pending_groups or {}
    targets = task_family_quota_counts(target, quotas)
    roi_prior = min(1.0, max(1e-3, float(os.environ.get("SLIME_TASK_FAMILY_ROI_PRIOR", "0.5"))))
    prior_strength = max(0.0, float(os.environ.get("SLIME_TASK_FAMILY_ROI_PRIOR_STRENGTH", "4")))
    weights = {}
    roi_candidate_counts = {}
    for family, family_target in targets.items():
        remaining_deficit = max(0, family_target - accepted[family])
        if remaining_deficit == 0:
            continue
        denominator = completed[family] + prior_strength
        roi = (
            (accepted[family] + roi_prior * prior_strength) / denominator
            if denominator > 0
            else roi_prior
        )
        roi = max(roi, 1e-3)
        expected_pending_valid = pending[family] * roi
        expected_deficit = max(0.0, remaining_deficit - expected_pending_valid)
        weights[family] = remaining_deficit / roi
        roi_candidate_counts[family] = expected_deficit / roi
    if not weights:
        return {}

    if (
        set(weights) == {"mcp"}
        and mcp_only_min_pending_groups is not None
        and not family_min_pending_groups
    ):
        min_pending_groups = min(min_pending_groups, max(0, mcp_only_min_pending_groups))

    total_weight = sum(weights.values())
    roi_submit_count = math.ceil(sum(roi_candidate_counts.values()))
    eligible_pending_count = sum(pending[family] for family in (quotas if family_min_pending_groups else weights))
    reservoir_submit_count = max(0, min_pending_groups - eligible_pending_count)
    floor_deficits = {
        family: max(0, family_min_pending_groups.get(family, 0) - pending[family])
        for family in quotas
    }
    floor_submit_count = sum(floor_deficits.values())
    submit_count = min(submit_limit, max(roi_submit_count, reservoir_submit_count, floor_submit_count))
    if submit_count == 0:
        return {}
    if floor_submit_count >= submit_count:
        raw = {family: submit_count * deficit / floor_submit_count for family, deficit in floor_deficits.items()}
        plan = {family: int(value) for family, value in raw.items()}
        remainders = {family: raw[family] - plan[family] for family in raw}
    else:
        plan = dict(floor_deficits)
        weighted_count = submit_count - floor_submit_count
        raw = {family: weighted_count * weight / total_weight for family, weight in weights.items()}
        for family, value in raw.items():
            plan[family] = plan.get(family, 0) + int(value)
        remainders = {family: raw[family] - int(raw[family]) for family in raw}
    remaining = submit_count - sum(plan.values())
    for family in sorted(remainders, key=lambda name: remainders[name], reverse=True):
        if remaining <= 0:
            break
        plan[family] += 1
        remaining -= 1
    return {family: count for family, count in plan.items() if count > 0}


def _task_family_admission_plan(
    submit_limit: int,
    accepted_deficit: int,
    quotas: dict[str, float],
    submitted: Counter[str],
    pending_count: int,
    min_pending_groups: int,
) -> dict[str, int]:
    """Keep the cumulative admitted prompt stream at the requested family mix."""
    if not quotas or submit_limit <= 0:
        return {}

    reservoir_deficit = max(0, min_pending_groups - pending_count)
    submit_count = min(submit_limit, max(accepted_deficit, reservoir_deficit))
    if submit_count <= 0:
        return {}

    projected = Counter({family: submitted[family] for family in quotas})
    plan: Counter[str] = Counter()
    for _ in range(submit_count):
        desired = task_family_quota_counts(sum(projected.values()) + 1, quotas)
        family = max(quotas, key=lambda name: desired[name] - projected[name])
        projected[family] += 1
        plan[family] += 1
    return dict(plan)


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    dataset_configs = getattr(args, "eval_datasets", []) or []
    if getattr(args, "eval_mix_datasets", False) and len(dataset_configs) > 1:
        return RolloutFnEvalOutput(data=await eval_rollout_mixed_datasets(args, dataset_configs)), []

    results = {}
    for dataset_cfg in dataset_configs:
        results.update(await eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    dataset, sampling_params = _prepare_eval_dataset(args, dataset_cfg)
    data = await _generate_eval_samples_bounded(args, dataset, dataset_cfg, sampling_params)
    return _format_eval_dataset_result(args, dataset_cfg, data)


def _prepare_eval_dataset(args: Namespace, dataset_cfg: EvalDatasetConfig) -> tuple[Dataset, dict[str, Any]]:
    global EVAL_PROMPT_DATASET

    eval_multimodal_keys = dataset_cfg.multimodal_keys if dataset_cfg.multimodal_keys is not None else args.multimodal_keys
    eval_apply_chat_template = dataset_cfg.apply_chat_template if dataset_cfg.apply_chat_template is not None else args.apply_chat_template
    eval_apply_chat_template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )
    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        eval_apply_chat_template,
        json.dumps(eval_multimodal_keys, sort_keys=True) if eval_multimodal_keys is not None else None,
        json.dumps(eval_apply_chat_template_kwargs, sort_keys=True) if eval_apply_chat_template_kwargs is not None else None,
    )
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=eval_multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=eval_apply_chat_template,
            apply_chat_template_kwargs=eval_apply_chat_template_kwargs,
        )
    sampling_params = {
        "temperature": dataset_cfg.temperature,
        "top_p": dataset_cfg.top_p,
        "top_k": dataset_cfg.top_k,
        "max_new_tokens": dataset_cfg.max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": (
            dataset_cfg.skip_special_tokens
            if dataset_cfg.skip_special_tokens is not None
            else args.rollout_skip_special_tokens
        ),
        "no_stop_trim": dataset_cfg.no_stop_trim if dataset_cfg.no_stop_trim is not None else True,
        "spaces_between_special_tokens": False,
    }
    if dataset_cfg.repetition_penalty is not None:
        sampling_params["repetition_penalty"] = dataset_cfg.repetition_penalty
    return EVAL_PROMPT_DATASET[cache_key], sampling_params


def _format_eval_dataset_result(
    args: Namespace,
    dataset_cfg: EvalDatasetConfig,
    data: list[Sample],
) -> dict[str, dict[str, list[Any]]]:
    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


async def _generate_one_eval_sample(
    args: Namespace,
    dataset_cfg: EvalDatasetConfig,
    base_sampling_params: dict[str, Any],
    sample_index: int,
    prompt_sample: Sample,
    sample_offset: int,
):
    sample = copy.deepcopy(prompt_sample)
    sample.index = sample_index
    sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
    sample.custom_rm_path = dataset_cfg.custom_rm_path
    sample.generate_function_path = dataset_cfg.custom_generate_function_path
    if getattr(args, "enable_use_grm_evals", False):
        sample.custom_rm_path = getattr(args, "grm_custom_rm_path", None)
    sampling_params = base_sampling_params
    if getattr(args, "sglang_enable_deterministic_inference", False):
        sampling_params = base_sampling_params.copy()
        sampling_params["sampling_seed"] = args.rollout_seed + sample_offset
    return await generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)


async def eval_rollout_mixed_datasets(
    args: Namespace,
    dataset_configs: list[EvalDatasetConfig],
) -> dict[str, dict[str, list[Any]]]:
    names = [dataset_cfg.name for dataset_cfg in dataset_configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Mixed eval dataset names must be unique, got {names}")
    contexts = [(*_prepare_eval_dataset(args, dataset_cfg), dataset_cfg) for dataset_cfg in dataset_configs]
    generated = await _generate_mixed_eval_samples_bounded(args, contexts)
    results: dict[str, dict[str, list[Any]]] = {}
    for dataset_cfg in dataset_configs:
        results.update(_format_eval_dataset_result(args, dataset_cfg, generated[dataset_cfg.name]))
    return results


async def _generate_mixed_eval_samples_bounded(
    args: Namespace,
    contexts: list[tuple[Dataset, dict[str, Any], EvalDatasetConfig]],
) -> dict[str, list[Sample]]:
    """Generate datasets round-robin under one global inflight budget."""
    from collections import deque

    total = sum(len(dataset.samples) * dataset_cfg.n_samples_per_eval_prompt for dataset, _, dataset_cfg in contexts)
    controller = None
    if total == 0:
        max_inflight = 0
    elif getattr(args, "eval_adaptive_concurrency", False):
        controller = EvalConcurrencyController(args, total)
        max_inflight = controller.target
    else:
        max_inflight = max(1, min(total, int(getattr(args, "eval_max_inflight_tasks", 384))))

    def specs_for(dataset: Dataset, sampling_params: dict[str, Any], dataset_cfg: EvalDatasetConfig):
        sample_index = 0
        for prompt_sample in dataset.samples:
            for sample_offset in range(dataset_cfg.n_samples_per_eval_prompt):
                yield dataset_cfg, sampling_params, sample_index, prompt_sample, sample_offset
                sample_index += 1

    remaining = deque(
        iter(specs_for(dataset, sampling_params, dataset_cfg))
        for dataset, sampling_params, dataset_cfg in contexts
        if dataset.samples and dataset_cfg.n_samples_per_eval_prompt
    )

    def next_spec():
        while remaining:
            iterator = remaining.popleft()
            try:
                spec = next(iterator)
            except StopIteration:
                continue
            remaining.append(iterator)
            return spec
        return None

    async def generate_one(spec):
        dataset_cfg, sampling_params, sample_index, prompt_sample, sample_offset = spec
        generated = await _generate_one_eval_sample(
            args,
            dataset_cfg,
            sampling_params,
            sample_index,
            prompt_sample,
            sample_offset,
        )
        return dataset_cfg.name, generated

    pending: set[asyncio.Task] = set()

    def schedule_until(target: int) -> None:
        while len(pending) < target:
            spec = next_spec()
            if spec is None:
                return
            pending.add(asyncio.create_task(generate_one(spec)))

    schedule_until(max_inflight)
    data = {dataset_cfg.name: [] for _, _, dataset_cfg in contexts}
    logged_datasets: set[str] = set()
    pbar = tqdm(total=total, desc=f"Eval mixed ({len(contexts)} datasets)", file=_EvalProgressLogStream(sys.stderr))
    try:
        while pending:
            completed, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                dataset_name, generated = task.result()
                generated_samples = generated if isinstance(generated, list) else [generated]
                data[dataset_name].extend(generated_samples)
                if dataset_name not in logged_datasets and generated_samples:
                    sample = generated_samples[0]
                    logger.info(
                        "eval_rollout_mixed_datasets example %s data: %s reward=%s",
                        dataset_name,
                        [str(sample.prompt) + sample.response],
                        sample.reward,
                    )
                    logged_datasets.add(dataset_name)
                pbar.update(1)
            target = controller.poll(args) if controller is not None else max_inflight
            schedule_until(target)
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    finally:
        if controller is not None:
            await controller.close()
        pbar.close()

    for samples in data.values():
        samples.sort(key=lambda sample: sample.index)
    return data


async def _generate_eval_samples_bounded(
    args: Namespace,
    dataset: Dataset,
    dataset_cfg: EvalDatasetConfig,
    base_sampling_params: dict[str, Any],
) -> list[Sample]:
    total = len(dataset.samples) * dataset_cfg.n_samples_per_eval_prompt
    controller = None
    if total == 0:
        max_inflight = 0
    elif getattr(args, "eval_adaptive_concurrency", False):
        controller = EvalConcurrencyController(args, total)
        max_inflight = controller.target
    else:
        max_inflight = max(1, min(total, int(getattr(args, "eval_max_inflight_tasks", 384))))

    def sample_specs():
        sample_index = 0
        for prompt_sample in dataset.samples:
            for sample_offset in range(dataset_cfg.n_samples_per_eval_prompt):
                yield sample_index, prompt_sample, sample_offset
                sample_index += 1

    async def generate_one(sample_index: int, prompt_sample: Sample, sample_offset: int):
        return await _generate_one_eval_sample(
            args,
            dataset_cfg,
            base_sampling_params,
            sample_index,
            prompt_sample,
            sample_offset,
        )

    specs = iter(sample_specs())
    pending: set[asyncio.Task] = set()
    specs_exhausted = False

    def schedule_until(target: int) -> None:
        nonlocal specs_exhausted
        while not specs_exhausted and len(pending) < target:
            try:
                sample_index, prompt_sample, sample_offset = next(specs)
            except StopIteration:
                specs_exhausted = True
                return
            pending.add(asyncio.create_task(generate_one(sample_index, prompt_sample, sample_offset)))

    schedule_until(max_inflight)

    data: list[Sample] = []
    log_example = True
    pbar = tqdm(total=total, desc=f"Eval {dataset_cfg.name}", file=_EvalProgressLogStream(sys.stderr))
    try:
        while pending:
            completed, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                generated = task.result()
                if log_example:
                    logged_sample = generated[0] if isinstance(generated, list) else generated
                    logger.info(
                        "eval_rollout_single_dataset example data: %s reward=%s",
                        [str(logged_sample.prompt) + logged_sample.response],
                        logged_sample.reward,
                    )
                    log_example = False
                if isinstance(generated, list):
                    data.extend(generated)
                else:
                    data.append(generated)
                pbar.update(1)
            target = controller.poll(args) if controller is not None else max_inflight
            schedule_until(target)
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    finally:
        if controller is not None:
            await controller.close()
        pbar.close()

    data.sort(key=lambda sample: sample.index)
    return data


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output


def _flatten_samples(group) -> list[Sample]:
    samples: list[Sample] = []
    stack = list(group)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack[:0] = item
        else:
            samples.append(item)
    return samples


def _append_webqa_prefilter_audit(
    *,
    rollout_id: int,
    completed_group_index: int,
    samples: list[Sample],
    keep: bool,
    drop_reason: str | None,
) -> None:
    audit_dir = os.environ.get("SLIME_ROLLOUT_PREFILTER_AUDIT_DIR", "").strip()
    if not audit_dir:
        return
    rows = []
    seen_trajectories = set()
    for sample in samples:
        if sample_task_family(sample) != "webqa":
            continue
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        trajectory_id = metadata.get("parent_traj_id", getattr(sample, "rollout_id", None))
        if trajectory_id is not None and trajectory_id in seen_trajectories:
            continue
        if trajectory_id is not None:
            seen_trajectories.add(trajectory_id)
        debug = metadata.get("fused_reward_debug") or metadata.get("reward_debug") or {}
        question = str(metadata.get("question") or "")
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16] if question else ""
        task_id = next(
            (
                str(metadata[key])
                for key in ("task_id", "instance_id", "id")
                if metadata.get(key) is not None
            ),
            question_hash or str(getattr(sample, "index", "")),
        )
        old_reward = float(debug.get("exact_reward", 0.0))
        try:
            new_reward = float(debug.get("alias_reward", sample.reward))
        except (TypeError, ValueError):
            new_reward = 0.0
        rows.append(
            {
                "rollout_id": rollout_id,
                "group_index": completed_group_index,
                "task_id": task_id,
                "prediction": str(debug.get("prediction", "")),
                "old_reward": old_reward,
                "new_reward": new_reward,
                "drop_reason": drop_reason,
                "keep": keep,
            }
        )
    if not rows:
        return
    path = Path(audit_dir) / f"rollout_{rollout_id:06d}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.writelines(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    except OSError:
        logger.exception("Failed to append WebQA pre-filter audit records to %s", path)
