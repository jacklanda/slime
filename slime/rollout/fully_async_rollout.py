"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import httpx
import logging
import os
import queue
import json
import threading
import time
from collections import Counter

from slime.rollout.sglang_rollout import EvalConcurrencyController, GenerateState, _parse_eval_engine_metrics, generate_and_rm_group
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import (
    DynamicFilterOutput,
    MetricGatherer,
    call_dynamic_filter,
    is_valid_reward_group,
)
from slime.rollout.filter_hub.dynamic_sampling_filters import is_infra_failure
from slime.rollout.task_family import (
    has_task_family_quota_candidates as _has_task_family_quota_candidates,
    parse_task_family_quotas as _parse_task_family_quotas,
    sample_group_task_family as _sample_group_task_family,
    select_task_family_quota_groups,
)
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.utils.visualization import maybe_print_rollout_group

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
    "shutdown_fully_async_rollout_worker",
]

logger = logging.getLogger("slime.rollout.fully_async")


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            server_capacity = args.sglang_server_concurrency * get_rollout_num_engines(args)
            concurrency = max(
                1,
                min(
                    server_capacity,
                    _int_env("SLIME_FULLY_ASYNC_MAX_CONCURRENCY", server_capacity),
                ),
            )
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=concurrency)
            _global_worker.start()
        return _global_worker


def _create_adaptive_concurrency_controller(args, capacity: int) -> EvalConcurrencyController | None:
    if not _env_bool("SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY", False):
        return None

    maximum = max(1, min(capacity, _int_env("SLIME_FULLY_ASYNC_MAX_CONCURRENCY", capacity)))
    initial = max(1, min(maximum, _int_env("SLIME_FULLY_ASYNC_INITIAL_CONCURRENCY", maximum // 2)))
    controller_args = copy.copy(args)
    controller_args.eval_max_inflight_tasks = maximum
    controller_args.eval_initial_inflight_tasks = initial
    controller_args.eval_concurrency_step = max(
        1,
        _int_env("SLIME_FULLY_ASYNC_CONCURRENCY_STEP", max(1, maximum // 4)),
    )
    controller_args.eval_concurrency_poll_interval = max(
        0.1,
        _float_env("SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL", 5.0),
    )
    return EvalConcurrencyController(controller_args, total=maximum)


async def _fetch_engine_load(args, client: httpx.AsyncClient):
    try:
        response = await client.get(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/engine_metrics",
            timeout=max(0.1, _float_env("SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL", 10.0)),
        )
        response.raise_for_status()
        return _parse_eval_engine_metrics(response.text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to sample SGLang engine metrics for fully-async concurrency: %s", exc)
        return None


def shutdown_fully_async_rollout_worker(*, cancel_inflight: bool = False) -> None:
    """Stop the process-local worker and release its in-flight rollouts."""

    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop(cancel_inflight=cancel_inflight)
            _global_worker = None


atexit.register(shutdown_fully_async_rollout_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.accepting_work = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._cancel_inflight_on_stop = False
        self._work_limit: int | None = None
        self._submitted_since_resume = 0
        self._submitted_total = 0
        self._pending_completed: dict[int, list[Sample]] = {}
        self._source_exhausted = False

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self, *, cancel_inflight: bool = False) -> None:
        self.accepting_work = False
        self._cancel_inflight_on_stop = cancel_inflight
        self.running = False
        loop = self._event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._request_loop_stop)
        if self.worker_thread and self.worker_thread.is_alive():
            timeout = max(1.0, _float_env("SLIME_FULLY_ASYNC_SHUTDOWN_TIMEOUT", 60.0)) if cancel_inflight else 5.0
            self.worker_thread.join(timeout=timeout)
            if cancel_inflight and self.worker_thread.is_alive():
                raise RuntimeError(
                    f"fully-async rollout worker did not stop within {timeout:.1f}s; "
                    "refusing to tear down SGLang with active requests"
                )

    def get_completed_groups(self, limit: int | None = None) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while limit is None or len(completed) < limit:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_completed_groups_for_range(self, start: int, end: int) -> list[tuple[int, list[Sample]]]:
        while True:
            try:
                gid, group = self.output_queue.get_nowait()
                self._pending_completed[gid] = group
            except queue.Empty:
                break
        gids = sorted(gid for gid in self._pending_completed if start <= gid < end)
        return [(gid, self._pending_completed.pop(gid)) for gid in gids]

    def requeue_completed_groups(self, groups: list[tuple[int, list[Sample]]]) -> None:
        for item in groups:
            self.output_queue.put_nowait(item)

    def queue_size(self) -> int:
        return self.output_queue.qsize() + len(self._pending_completed)

    def exhausted_and_idle(self) -> bool:
        return self._source_exhausted and not self._active_tasks and self.queue_size() == 0

    def resume(self, *, work_limit: int | None = None, continuous: bool = False) -> None:
        self._work_limit = work_limit
        if not continuous:
            self._submitted_since_resume = 0
        self.accepting_work = True
        self._source_exhausted = False

    def pause(self) -> None:
        self.accepting_work = False

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._loop())
        finally:
            self._event_loop = None

    def _request_loop_stop(self) -> None:
        if self._cancel_inflight_on_stop:
            for task in tuple(self._active_tasks):
                task.cancel()

    async def _loop(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        active_tasks = self._active_tasks
        controller = _create_adaptive_concurrency_controller(self.args, self.concurrency)
        max_concurrent = controller.target if controller is not None else self.concurrency
        metrics_client = httpx.AsyncClient(timeout=None, trust_env=False) if controller is not None else None
        metrics_task: asyncio.Task | None = None
        next_metrics_poll_at = 0.0
        gid_counter = 0
        loop_poll_interval = 0.05 if getattr(self.args, "rollout_only_inference_fast_path", False) else 1.0

        try:
            while self.running:
                try:
                    if controller is not None:
                        if metrics_task is not None and metrics_task.done():
                            load = metrics_task.result()
                            metrics_task = None
                            if load is not None:
                                previous = controller.target
                                max_concurrent = controller.update(load)
                                if max_concurrent != previous:
                                    logger.info(
                                        "Fully-async concurrency changed %d -> %d "
                                        "(engines=%d running=%.0f waiting=%.0f kv=%.3f cache_hit=%.3f)",
                                        previous,
                                        max_concurrent,
                                        load.engine_count,
                                        load.running_requests,
                                        load.waiting_requests,
                                        load.max_token_usage,
                                        load.mean_cache_hit_rate,
                                    )
                        now = time.monotonic()
                        if metrics_task is None and now >= next_metrics_poll_at:
                            metrics_task = asyncio.create_task(_fetch_engine_load(self.args, metrics_client))
                            next_metrics_poll_at = now + controller.poll_interval

                    # Reap done tasks
                    if active_tasks:
                        done = {t for t in active_tasks if t.done()}
                        for t in done:
                            try:
                                t.result()  # results already handled in callback
                            except Exception as e:  # noqa: BLE001
                                logger.warning("fully-async task crashed: %r", e)
                        active_tasks -= done

                    # Top up.
                    while (
                        self.accepting_work
                        and len(active_tasks) < max_concurrent
                        and self.running
                        and (
                            self._work_limit is None
                            or (
                                self._submitted_total if self._cross_shard_prefetch_enabled() else self._submitted_since_resume
                            )
                            < self._work_limit
                        )
                    ):
                        groups = self.data_buffer.get_samples(1)
                        if not groups:
                            self._source_exhausted = True
                            break
                        self._source_exhausted = False
                        for group in groups:
                            self._submitted_since_resume += 1
                            self._submitted_total += 1
                            gid = gid_counter
                            gid_counter += 1
                            self._snapshot_rollout_cursor(gid + 1)
                            task = asyncio.create_task(
                                generate_and_rm_group(
                                    self.args,
                                    group,
                                    sampling_params=self.state.sampling_params.copy(),
                                    evaluation=False,
                                )
                            )
                            task.add_done_callback(self._make_done_cb(gid))
                            active_tasks.add(task)

                    await asyncio.sleep(loop_poll_interval)
                except Exception as e:  # noqa: BLE001
                    logger.exception("fully-async loop iteration error: %s", e)
                    await asyncio.sleep(loop_poll_interval)
        finally:
            if metrics_task is not None:
                metrics_task.cancel()
                await asyncio.gather(metrics_task, return_exceptions=True)
            if metrics_client is not None:
                await metrics_client.aclose()
            if active_tasks:
                if self._cancel_inflight_on_stop:
                    logger.info("fully-async: cancelling %d in-flight tasks", len(active_tasks))
                    for task in active_tasks:
                        task.cancel()
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                else:
                    logger.info("fully-async: waiting for %d in-flight tasks to drain", len(active_tasks))
                    try:
                        await asyncio.wait(active_tasks, timeout=30)
                    except Exception:  # noqa: BLE001
                        pass
            active_tasks.clear()

    def _cross_shard_prefetch_enabled(self) -> bool:
        return getattr(self.args, "rollout_only_inference_fast_path", False) and _env_bool(
            "SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH", False
        )

    def _snapshot_rollout_cursor(self, submitted: int) -> None:
        if not self._cross_shard_prefetch_enabled() or not hasattr(self.data_buffer, "snapshot_cursor_for_rollout"):
            return
        batch_size = int(self.args.rollout_batch_size)
        start_rollout_id = int(getattr(self.args, "start_rollout_id", 0) or 0)
        total_remaining = max(0, len(self.data_buffer) - start_rollout_id * batch_size)
        if submitted % batch_size and submitted != total_remaining:
            return
        rollout_id = start_rollout_id + (submitted - 1) // batch_size
        self.data_buffer.snapshot_cursor_for_rollout(rollout_id)

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if self._cancel_inflight_on_stop:
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            if any(getattr(s, "status", None) == Sample.Status.ABORTED for s in result):
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    keep_all_groups = _env_bool("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", False)
    metric_gatherer = MetricGatherer()

    target = args.rollout_batch_size
    valid_target = max(0, _int_env("SLIME_FULLY_ASYNC_VALID_GROUPS_PER_SHARD", 0))
    collect_valid_groups = valid_target > 0 and getattr(args, "rollout_only_inference_fast_path", False)
    if collect_valid_groups:
        target = valid_target
    cross_shard_prefetch = (
        keep_all_groups
        and not collect_valid_groups
        and getattr(args, "rollout_only_inference_fast_path", False)
        and _env_bool("SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH", False)
    )
    shard_offset = max(0, rollout_id - int(getattr(args, "start_rollout_id", 0) or 0)) * args.rollout_batch_size
    if (
        not collect_valid_groups
        and
        getattr(args, "rollout_only_inference_fast_path", False)
        and _env_bool("SLIME_FULLY_ASYNC_NO_DATASET_WRAP", False)
    ):
        dataset_size = len(data_buffer)
        consumed = rollout_id * args.rollout_batch_size
        target = min(target, max(0, dataset_size - consumed))
        if target <= 0:
            worker.pause()
            return RolloutFnTrainOutput(samples=[], metrics={})
    candidate_budget = target
    if collect_valid_groups:
        candidate_budget = max(target, _int_env("SLIME_FULLY_ASYNC_MAX_CANDIDATE_GROUPS_PER_SHARD", target * 16))
        worker.resume(work_limit=candidate_budget)
    elif cross_shard_prefetch:
        total_remaining = max(0, len(data_buffer) - int(getattr(args, "start_rollout_id", 0) or 0) * args.rollout_batch_size)
        prefetch_groups = max(0, _int_env("SLIME_FULLY_ASYNC_PREFETCH_GROUPS", args.rollout_batch_size))
        worker.resume(work_limit=min(total_remaining, shard_offset + target + prefetch_groups), continuous=True)
    else:
        worker.resume(
            work_limit=target
            if keep_all_groups and getattr(args, "rollout_only_inference_fast_path", False)
            else None
        )
    quotas = {} if keep_all_groups else _parse_task_family_quotas(getattr(args, "rollout_task_family_quotas", None) or "")
    quota_candidate_multiplier = max(1, _int_env("SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER", 4))
    candidate_limit = target * quota_candidate_multiplier if quotas else target
    logger.info(
        "fully-async rollout %d: target=%d candidate_limit=%d candidate_budget=%d valid_only=%s queue_warm=%d",
        rollout_id,
        target,
        candidate_limit,
        candidate_budget,
        collect_valid_groups,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    filter_relax_after = int(getattr(args, "fully_async_filter_relax_after_groups", 0) or 0)
    completed_groups = 0
    dropped_groups = 0
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target or (quotas and len(collected) < candidate_limit and not _has_task_family_quota_candidates(collected.values(), target, quotas)):
        # Pull whatever's done.
        drained = 0
        remaining = target - len(collected)
        completed = (
            worker.get_completed_groups_for_range(shard_offset, shard_offset + target)
            if cross_shard_prefetch
            else worker.get_completed_groups(limit=None if collect_valid_groups else remaining)
        )
        for completed_index, (gid, group) in enumerate(completed):
            if len(collected) >= target:
                worker.requeue_completed_groups(completed[completed_index:])
                break
            completed_groups += 1
            flat_group = _flatten_samples(group)
            metric_gatherer.on_completed_group(args, flat_group)
            valid_reward_group = is_valid_reward_group(args, flat_group)
            if collect_valid_groups:
                await asyncio.to_thread(
                    _write_trajectory_profile_shard,
                    args,
                    rollout_id,
                    gid,
                    flat_group,
                    valid_reward_group,
                )
            if collect_valid_groups and not valid_reward_group:
                metric_gatherer.on_dynamic_filter_drop(reason="zero_reward_variance")
                dropped_groups += 1
                continue
            if keep_all_groups:
                dynamic_filter_output = DynamicFilterOutput(keep=True, reason="keep_all")
            elif any(is_infra_failure(sample) for sample in flat_group):
                dynamic_filter_output = DynamicFilterOutput(keep=False, reason="infra_failure")
            else:
                dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, flat_group, rollout_id=rollout_id)
            relax_filter = (
                dynamic_filter_output.reason != "infra_failure"
                and filter_relax_after > 0
                and completed_groups >= filter_relax_after
            )
            if not dynamic_filter_output.keep and not relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                dropped_groups += 1
                continue
            if not dynamic_filter_output.keep and relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=f"relaxed_{dynamic_filter_output.reason}")
            # Keep the seconds-long rich episode render off the event loop
            # (same reasoning as the sync collection path).
            await asyncio.to_thread(maybe_print_rollout_group, args, group, group_id=gid)
            collected[gid] = group
            drained += 1

        if collect_valid_groups and completed_groups >= candidate_budget and len(collected) < target:
            logger.warning(
                "fully-async rollout %d exhausted %d candidate groups with only %d/%d valid groups",
                rollout_id,
                candidate_budget,
                len(collected),
                target,
            )
            break
        if collect_valid_groups and worker.exhausted_and_idle() and len(collected) < target:
            logger.warning(
                "fully-async rollout %d exhausted the dataset with only %d/%d valid groups",
                rollout_id,
                len(collected),
                target,
            )
            break

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if _env_bool("SLIME_FUSED_PROGRESS_LOGS", False) and now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, dropped=%d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                dropped_groups,
                completed_groups,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention). Some custom
    # generate functions return nested sample groups, so find the first real
    # Sample-like object instead of accidentally reading list.index.
    def _key(group: list[Sample]) -> int:
        stack = list(group)
        while stack:
            item = stack.pop(0)
            if isinstance(item, list):
                stack[:0] = item
                continue
            for attr in ("index", "group_index", "rollout_id"):
                idx = getattr(item, attr, None)
                if idx is None or callable(idx):
                    continue
                try:
                    return int(idx)
                except (TypeError, ValueError):
                    continue
        return 0

    out = _select_task_family_quota_groups(sorted(collected.values(), key=_key), target, args)
    if not cross_shard_prefetch:
        worker.pause()
    candidate_family_counts = Counter(_sample_group_task_family(group) for group in collected.values())
    selected_family_counts = Counter(_sample_group_task_family(group) for group in out)
    logger.info(
        "fully-async rollout %d: done in %.1fs, dropped=%d/%d, queue_left=%d, candidate_families=%s, selected_families=%s",
        rollout_id,
        time.time() - started,
        dropped_groups,
        completed_groups,
        worker.queue_size(),
        dict(candidate_family_counts),
        dict(selected_family_counts),
    )
    metrics = metric_gatherer.collect()
    metrics["rollout/dynamic_filter/completed_groups"] = completed_groups
    metrics["rollout/dynamic_filter/dropped_groups"] = dropped_groups
    metrics["rollout/dynamic_filter/kept_groups"] = len(out)
    metrics["rollout/dynamic_filter/target_valid_groups"] = valid_target
    metrics["rollout/dynamic_filter/valid_group_shortfall"] = max(0, valid_target - len(out))
    metrics["rollout/config/fused_webqa_min_unique_searches"] = _int_env("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", 2)
    metrics["rollout/config/fused_repeated_search_max_strikes"] = _int_env("FUSED_REPEATED_SEARCH_MAX_STRIKES", 2)
    for family, count in candidate_family_counts.items():
        metrics[f"rollout/task_family_candidates/{family}"] = count
    for family, count in selected_family_counts.items():
        metrics[f"rollout/task_family_selected/{family}"] = count
    return RolloutFnTrainOutput(samples=out, metrics=metrics)


def _select_task_family_quota_groups(groups: list[list[Sample]], target: int, args) -> list[list[Sample]]:
    if _env_bool("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", False):
        return groups[:target]
    return select_task_family_quota_groups(
        groups,
        target,
        getattr(args, "rollout_task_family_quotas", None),
    )


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _write_trajectory_profile_shard(
    args, rollout_id: int, gid: int, samples: list[Sample], selected_for_shard: bool
) -> None:
    profile_dir = os.environ.get("SLIME_FUSED_PROFILE_DIR")
    if not profile_dir:
        return
    profile_dir = os.path.abspath(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)
    records = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        records.append(
            {
                "rollout_id": rollout_id,
                "worker_group_id": gid,
                "group_index": sample.group_index,
                "sample_index": sample.index,
                "session_id": sample.session_id,
                "reward": sample.get_reward_value(args),
                "steps": metadata.get("fused_traj_steps"),
                "termination_reason": metadata.get("fused_termination"),
                "parser_error": metadata.get("credit_assignment_event")
                in {"tool_parser_error", "think_parser_error"},
                "selected_for_shard": selected_for_shard,
                "profile": metadata.get("fused_profile") or {},
            }
        )
    profile_path = os.path.join(profile_dir, f"rollout_{rollout_id:06d}_group_{gid:09d}.json")
    temporary_path = profile_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "rollout_id": rollout_id,
                "worker_group_id": gid,
                "selected_for_shard": selected_for_shard,
                "trajectories": records,
            },
            stream,
            ensure_ascii=False,
            indent=4,
            default=str,
        )
        stream.write("\n")
    os.replace(temporary_path, profile_path)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


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


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
