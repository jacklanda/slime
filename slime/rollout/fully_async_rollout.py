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
import logging
import queue
import math
import os
import threading
import time
from collections import Counter

from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample
from slime.utils.visualization import maybe_print_rollout_group

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
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
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


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

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def resume(self) -> None:
        self.accepting_work = True

    def pause(self) -> None:
        self.accepting_work = False

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
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
                while self.accepting_work and len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
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

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
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
    worker.resume()
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    target = args.rollout_batch_size
    quotas = _parse_task_family_quotas(getattr(args, "rollout_task_family_quotas", None) or "")
    quota_candidate_multiplier = max(1, _int_env("SLIME_FUSED_QUOTA_CANDIDATE_MULTIPLIER", 4))
    candidate_limit = target * quota_candidate_multiplier if quotas else target
    logger.info(
        "fully-async rollout %d: target=%d candidate_limit=%d queue_warm=%d",
        rollout_id,
        target,
        candidate_limit,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    candidate_metric_groups: list[list[Sample]] = []
    filter_relax_after = int(getattr(args, "fully_async_filter_relax_after_groups", 0) or 0)
    completed_groups = 0
    dropped_groups = 0
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target or (
        quotas and len(collected) < candidate_limit and not _has_task_family_quota_candidates(collected.values(), target, quotas)
    ):
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups():
            completed_groups += 1
            candidate_metric_groups.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, _flatten_samples(group), rollout_id=rollout_id)
            relax_filter = filter_relax_after > 0 and completed_groups >= filter_relax_after
            if not dynamic_filter_output.keep and not relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                dropped_groups += 1
                continue
            if not dynamic_filter_output.keep and relax_filter:
                metric_gatherer.on_dynamic_filter_drop(reason=f"relaxed_{dynamic_filter_output.reason}")
            maybe_print_rollout_group(args, group, group_id=gid)
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
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
    metrics.update(_fused_rollout_distribution_metrics("candidate", candidate_metric_groups))
    metrics.update(_fused_rollout_distribution_metrics("selected", out))
    metrics["rollout/config/fused_webqa_min_unique_searches"] = _int_env("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", 2)
    metrics["rollout/config/fused_repeated_search_max_strikes"] = _int_env("FUSED_REPEATED_SEARCH_MAX_STRIKES", 2)
    for family, count in candidate_family_counts.items():
        metrics[f"rollout/task_family_candidates/{family}"] = count
    for family, count in selected_family_counts.items():
        metrics[f"rollout/task_family_selected/{family}"] = count
    return RolloutFnTrainOutput(samples=out, metrics=metrics)


def _select_task_family_quota_groups(groups: list[list[Sample]], target: int, args) -> list[list[Sample]]:
    quota_spec = getattr(args, "rollout_task_family_quotas", None)
    if not quota_spec:
        return groups[:target]
    quotas = _parse_task_family_quotas(quota_spec)
    if not quotas:
        return groups[:target]

    selected: list[list[Sample]] = []
    selected_ids: set[int] = set()
    counts = {family: math.floor(target * fraction) for family, fraction in quotas.items()}
    remaining = target - sum(counts.values())
    for family, _fraction in sorted(quotas.items(), key=lambda item: item[1], reverse=True):
        if remaining <= 0:
            break
        counts[family] += 1
        remaining -= 1

    for family, count in counts.items():
        if count <= 0:
            continue
        taken = 0
        for idx, group in enumerate(groups):
            if idx in selected_ids or _sample_group_task_family(group) != family:
                continue
            selected.append(group)
            selected_ids.add(idx)
            taken += 1
            if taken >= count:
                break

    missing_families = [
        family
        for family, count in counts.items()
        if count > 0 and not any(_sample_group_task_family(group) == family for group in selected)
    ]
    for family in missing_families:
        for idx, group in enumerate(groups):
            if idx in selected_ids or _sample_group_task_family(group) != family:
                continue
            selected.append(group)
            selected_ids.add(idx)
            break

    for idx, group in enumerate(groups):
        if len(selected) >= target:
            break
        if idx not in selected_ids:
            selected.append(group)

    return selected[:target]


def _has_task_family_quota_candidates(groups, target: int, quotas: dict[str, float]) -> bool:
    counts = _task_family_quota_counts(target, quotas)
    available = Counter(_sample_group_task_family(group) for group in groups)
    return all(available[family] >= count for family, count in counts.items() if count > 0)


def _parse_task_family_quotas(spec: str | dict[str, float]) -> dict[str, float]:
    if isinstance(spec, dict):
        items = spec.items()
    else:
        items = []
        for part in str(spec).split(","):
            if not part.strip() or "=" not in part:
                continue
            name, value = part.split("=", 1)
            items.append((name, value))
    quotas: dict[str, float] = {}
    for name, value in items:
        family = _normalize_task_family(name)
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            continue
        if family and fraction > 0:
            quotas[family] = fraction
    total = sum(quotas.values())
    if total <= 0:
        return {}
    return {family: fraction / total for family, fraction in quotas.items()}


def _task_family_quota_counts(target: int, quotas: dict[str, float]) -> dict[str, int]:
    counts = {family: math.floor(target * fraction) for family, fraction in quotas.items()}
    remaining = target - sum(counts.values())
    for family, _fraction in sorted(quotas.items(), key=lambda item: item[1], reverse=True):
        if remaining <= 0:
            break
        counts[family] += 1
        remaining -= 1
    return counts


def _sample_group_task_family(group: list[Sample]) -> str:
    samples = _flatten_samples(group)
    families = [_sample_task_family(sample) for sample in samples]
    counts = Counter(family for family in families if family)
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def _sample_task_family(sample: Sample) -> str:
    metadata = getattr(sample, "metadata", {}) or {}
    for key in ("fused_task_type", "task_type", "data_source"):
        value = metadata.get(key)
        if value:
            return _normalize_task_family(value)
    if metadata.get("tools_py") or metadata.get("environment"):
        return "mcp"
    if metadata.get("docker_image"):
        return "cli"
    return "webqa"


def _fused_rollout_distribution_metrics(prefix: str, groups: list[list[Sample]]) -> dict[str, float]:
    steps: list[float] = []
    events: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    for group in groups:
        samples = _flatten_samples(group)
        if not samples:
            continue
        metadata = getattr(samples[0], "metadata", {}) or {}
        value = metadata.get("fused_traj_steps") or metadata.get("traj_steps")
        try:
            if value is not None:
                steps.append(float(value))
        except (TypeError, ValueError):
            pass
        event = metadata.get("credit_assignment_event")
        events[str(event if event is not None else "none")] += 1
        termination = metadata.get("fused_termination") or metadata.get("termination_reason") or "unknown"
        terminations[str(termination)] += 1

    metrics: dict[str, float] = {}
    if steps:
        metrics[f"rollout/{prefix}/steps/mean"] = sum(steps) / len(steps)
        metrics[f"rollout/{prefix}/steps/min"] = min(steps)
        metrics[f"rollout/{prefix}/steps/max"] = max(steps)
        for step, count in Counter(int(step) if float(step).is_integer() else step for step in steps).items():
            metrics[f"rollout/{prefix}/steps/count_{step}"] = count
    total_events = sum(events.values())
    if total_events:
        for event, count in events.items():
            metrics[f"rollout/{prefix}/credit_assignment_event/{_metric_key_part(event)}"] = count / total_events
    total_terminations = sum(terminations.values())
    if total_terminations:
        for termination, count in terminations.items():
            metrics[f"rollout/{prefix}/termination/{_metric_key_part(termination)}"] = count / total_terminations
    return metrics


def _metric_key_part(value: str) -> str:
    return str(value or "unknown").strip().lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def _normalize_task_family(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"web_search", "search", "webqa"}:
        return "webqa"
    if normalized in {"mcp", "tool", "tools"}:
        return "mcp"
    if normalized in {"cli", "swe", "et", "endless_terminal", "endless_terminals"}:
        return "cli"
    return normalized


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
