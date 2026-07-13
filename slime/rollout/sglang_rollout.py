import asyncio
import copy
import inspect
import json
import logging
import os
import re
import time
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.backends.sglang_utils.server_control import abort_servers_until_idle
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
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

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}
_TOP_P_TOKEN_ID_META_KEYS = ("top_p_token_ids", "top_p_kept_token_ids")
_TOP_P_TOKEN_OFFSET_META_KEYS = ("top_p_token_offsets", "top_p_kept_token_offsets")
_ENGINE_METRIC_RE = re.compile(r"^(sglang[:_](?:num_running_reqs|num_queue_reqs|token_usage|cache_hit_rate))(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$")


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
                        "Adaptive eval concurrency changed %d -> %d " "(engines=%d running=%.0f waiting=%.0f kv=%.3f cache_hit=%.3f)",
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


def _should_use_grm_eval(args: Namespace, evaluation: bool) -> bool:
    return evaluation and bool(getattr(args, "enable_use_grm_evals", False))


async def _score_eval_samples_with_grm(args: Namespace, samples: list[Sample]) -> None:
    if not samples:
        return
    grm_path = getattr(args, "grm_custom_rm_path", None)
    for sample in samples:
        sample.custom_rm_path = grm_path
        sample.reward = None
    with trace_span(samples, "grm_eval_reward_model"):
        rewards = await batched_async_rm(args, samples, evaluation=True)
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
        self.remaining_batch_size = 0
        self.pendings = set()
        self.pending_groups = {}
        self.aborted = False

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
        output = await post(url, payload, headers=headers)
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

    if _should_use_grm_eval(args, evaluation) and (sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED):
        await _score_eval_samples_with_grm(args, [sample])
        return sample

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path
    custom_generate_func = load_function(custom_func_path) if custom_func_path is not None else None
    manages_eval_request_concurrency = evaluation and getattr(
        custom_generate_func, "manages_eval_request_concurrency", False
    )

    if manages_eval_request_concurrency:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample
        with state.dp_rank_context() as _:
            sample = await custom_generate_func(args, sample, sampling_params, evaluation=True)
    else:
        async with state.semaphore:
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                return sample

            with state.dp_rank_context() as _:
                if custom_generate_func is not None:
                    # if signature has evaluation, pass evaluation
                    if "evaluation" in inspect.signature(custom_generate_func).parameters:
                        sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                    else:
                        sample = await custom_generate_func(args, sample, sampling_params)
                else:
                    sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        if _should_use_grm_eval(args, evaluation):
            await _score_eval_samples_with_grm(args, samples)
            return samples

        for sample in samples:
            if _should_rescore_eval_sample(args, sample, evaluation):
                sample.reward = None
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward, evaluation=evaluation)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        if _should_use_grm_eval(args, evaluation):
            await _score_eval_samples_with_grm(args, [sample])
            return sample
        if _should_rescore_eval_sample(args, sample, evaluation):
            sample.reward = None
        # Some custom generate paths may have already filled the reward.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample, evaluation=evaluation)

    return sample


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

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation)))

    group_timeout = _rollout_group_timeout(args, evaluation=evaluation)
    try:
        group = await asyncio.wait_for(asyncio.gather(*tasks), timeout=group_timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "Rollout group timed out after %.1fs; cancelling %s unfinished sample tasks.",
            group_timeout,
            sum(not task.done() for task in tasks),
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        group = [_task_result_or_timeout(task, sample, evaluation=evaluation) for sample, task in zip(group, tasks, strict=True)]

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group, evaluation=evaluation)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

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
    timed_out.metadata = {
        **dict(sample.metadata or {}),
        "termination_reason": "timeout",
        "fused_termination": "timeout",
        "fused_error": "rollout_group_timeout",
        "evaluation": evaluation,
    }
    return timed_out


def _task_result_or_timeout(task: asyncio.Task, sample: Sample, *, evaluation: bool):
    if task.cancelled():
        return _timeout_sample(sample, evaluation=evaluation)
    try:
        result = task.result()
    except Exception:
        return _timeout_sample(sample, evaluation=evaluation)
    return result


def _group_has_trainable_response(group: list[Sample] | list[list[Sample]]) -> bool:
    return any(sample.status == Sample.Status.COMPLETED and sample.response_length > 0 for sample in _flatten_samples(group))


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    await abort_servers_until_idle(urls)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
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

    data = []
    all_data = []
    do_print = True
    filter_relax_after = int(getattr(args, "fully_async_filter_relax_after_groups", 0) or 0)
    completed_groups = 0
    dropped_groups = 0
    started = time.time()
    last_log = started
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Group collection")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(
            state.pendings,
            timeout=_rollout_group_timeout(args, evaluation=False),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done and state.pendings:
            task = next(iter(state.pendings))
            logger.warning(
                "Rollout collection timed out waiting for a finished group after %.1fs; cancelling one pending group.",
                _rollout_group_timeout(args, evaluation=False),
            )
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            state.pendings.remove(task)
            done = {task}

        for task in done:
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
                    group = [_timeout_sample(sample, evaluation=False) for sample in source_group]

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            completed_groups += 1
            dynamic_filter_output = call_dynamic_filter(
                dynamic_filter,
                args,
                _flatten_samples(group),
                rollout_id=rollout_id,
            )
            relax_filter = filter_relax_after > 0 and completed_groups >= filter_relax_after and _group_has_trainable_response(group)
            if not dynamic_filter_output.keep and not relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                dropped_groups += 1
                state.remaining_batch_size -= 1
                continue
            if not dynamic_filter_output.keep and relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=f"relaxed_{dynamic_filter_output.reason}")

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                # Rich-rendering a full episode (up to 128 step panels plus
                # per-token mask views) is seconds of pure-Python work; keep it
                # off the event loop so it cannot stall request dispatch for
                # every in-flight trajectory.
                await asyncio.to_thread(maybe_print_rollout_group, args, group, group_id=len(data))
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
        now = time.time()
        if now - last_log > 30.0:
            logger.info(
                "sync rollout %d: collected %d/%d, dropped=%d/%d, pending=%d, elapsed=%.1fs",
                rollout_id,
                len(data),
                target_data_size,
                dropped_groups,
                completed_groups,
                len(state.pendings),
                now - started,
            )
            last_log = now

    pbar.close()
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
    all_samples = sorted(all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    metrics = metric_gatherer.collect()
    metrics["rollout/dynamic_filter/completed_groups"] = completed_groups
    metrics["rollout/dynamic_filter/dropped_groups"] = dropped_groups
    metrics["rollout/dynamic_filter/kept_groups"] = len(data)
    return RolloutFnTrainOutput(samples=data, metrics=metrics), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    results = {}
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
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

    global EVAL_PROMPT_DATASET

    eval_multimodal_keys = (
        dataset_cfg.multimodal_keys if dataset_cfg.multimodal_keys is not None else args.multimodal_keys
    )
    eval_apply_chat_template = (
        dataset_cfg.apply_chat_template if dataset_cfg.apply_chat_template is not None else args.apply_chat_template
    )
    eval_apply_chat_template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )

    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        eval_apply_chat_template,
        json.dumps(eval_multimodal_keys, sort_keys=True) if eval_multimodal_keys is not None else None,
        (
            json.dumps(eval_apply_chat_template_kwargs, sort_keys=True)
            if eval_apply_chat_template_kwargs is not None
            else None
        ),
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
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=(
            dataset_cfg.skip_special_tokens
            if dataset_cfg.skip_special_tokens is not None
            else args.rollout_skip_special_tokens
        ),
        no_stop_trim=dataset_cfg.no_stop_trim if dataset_cfg.no_stop_trim is not None else True,
        spaces_between_special_tokens=False,
    )
    if dataset_cfg.repetition_penalty is not None:
        base_sampling_params["repetition_penalty"] = dataset_cfg.repetition_penalty

    data = await _generate_eval_samples_bounded(args, dataset, dataset_cfg, base_sampling_params)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


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
        return await generate_and_rm(
            args,
            sample,
            sampling_params=sampling_params,
            evaluation=True,
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
    pbar = tqdm(total=total, desc=f"Eval {dataset_cfg.name}", disable=not log_example)
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

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
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
