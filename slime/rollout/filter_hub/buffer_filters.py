import math

from slime.utils.types import Sample

__all__ = ["quota_bucket_by_steps"]


def quota_bucket_by_steps(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    if num_samples <= 0 or not buffer:
        return []

    bucket_order = ("short", "mid", "long")
    bucketed_indices: dict[str, list[int]] = {name: [] for name in bucket_order}
    for idx, group in enumerate(buffer):
        bucketed_indices[_group_step_bucket(group)].append(idx)

    counts = _quota_counts(num_samples)
    selected_indices: list[int] = []
    selected_index_set: set[int] = set()

    for bucket in bucket_order:
        need = counts[bucket]
        if need <= 0:
            continue
        taken = 0
        for idx in bucketed_indices[bucket]:
            if idx in selected_index_set:
                continue
            selected_indices.append(idx)
            selected_index_set.add(idx)
            taken += 1
            if taken >= need:
                break

    if len(selected_indices) < num_samples:
        for idx in _fallback_order(bucketed_indices):
            if idx in selected_index_set:
                continue
            selected_indices.append(idx)
            selected_index_set.add(idx)
            if len(selected_indices) >= num_samples:
                break

    selected_indices.sort()
    selected = [buffer[idx] for idx in selected_indices]
    for idx in reversed(selected_indices):
        del buffer[idx]
    return selected


def _group_step_bucket(group: list[Sample]) -> str:
    steps = _group_steps(group)
    if steps <= 3:
        return "short"
    if steps <= 5:
        return "mid"
    return "long"


def _group_steps(group: list[Sample]) -> int:
    max_steps = 0
    for sample in _iter_samples(group):
        metadata = getattr(sample, "metadata", {}) or {}
        value = metadata.get("fused_traj_steps") or metadata.get("traj_steps") or 0
        try:
            max_steps = max(max_steps, int(float(value)))
        except (TypeError, ValueError):
            continue
    return max_steps


def _iter_samples(samples):
    for sample in samples:
        if isinstance(sample, list):
            yield from _iter_samples(sample)
        else:
            yield sample


def _quota_counts(num_samples: int) -> dict[str, int]:
    quotas = {
        "short": 0.30,
        "mid": 0.45,
        "long": 0.25,
    }
    counts = {bucket: math.floor(num_samples * ratio) for bucket, ratio in quotas.items()}
    remaining = num_samples - sum(counts.values())
    for bucket in ("mid", "long", "short"):
        if remaining <= 0:
            break
        counts[bucket] += 1
        remaining -= 1
    return counts


def _fallback_order(bucketed_indices: dict[str, list[int]]) -> list[int]:
    ordered: list[int] = []
    for bucket in ("long", "mid", "short"):
        ordered.extend(bucketed_indices[bucket])
    return ordered
