import math
from collections import Counter

from slime.utils.types import Sample


def select_task_family_quota_groups(
    groups: list[list[Sample]],
    target: int,
    quota_spec,
    *,
    prefer_higher_mean_steps: bool = False,
) -> list[list[Sample]]:
    quotas = parse_task_family_quotas(quota_spec)
    if not quotas:
        return groups[:target]

    selected: list[list[Sample]] = []
    selected_ids: set[int] = set()
    counts = task_family_quota_counts(target, quotas)
    for family, count in counts.items():
        family_indices = [
            idx
            for idx, group in enumerate(groups)
            if idx not in selected_ids and sample_group_task_family(group) == family
        ]
        if prefer_higher_mean_steps:
            family_indices.sort(key=lambda idx: (-sample_group_mean_steps(groups[idx]), idx))
        for idx in family_indices[:count]:
            selected.append(groups[idx])
            selected_ids.add(idx)

    remaining_indices = [idx for idx in range(len(groups)) if idx not in selected_ids]
    if prefer_higher_mean_steps:
        remaining_indices.sort(key=lambda idx: (-sample_group_mean_steps(groups[idx]), idx))
    for idx in remaining_indices:
        if len(selected) >= target:
            break
        selected.append(groups[idx])
    return selected[:target]


def has_task_family_quota_candidates(groups, target: int, quotas: dict[str, float]) -> bool:
    counts = task_family_quota_counts(target, quotas)
    available = Counter(sample_group_task_family(group) for group in groups)
    return all(available[family] >= count for family, count in counts.items() if count > 0)


def parse_task_family_quotas(spec: str | dict[str, float] | None) -> dict[str, float]:
    if isinstance(spec, dict):
        items = spec.items()
    else:
        items = []
        for part in str(spec or "").split(","):
            if not part.strip() or "=" not in part:
                continue
            name, value = part.split("=", 1)
            items.append((name, value))
    quotas: dict[str, float] = {}
    for name, value in items:
        family = normalize_task_family(name)
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


def task_family_quota_counts(target: int, quotas: dict[str, float]) -> dict[str, int]:
    counts = {family: math.floor(target * fraction) for family, fraction in quotas.items()}
    remaining = target - sum(counts.values())
    for family, _fraction in sorted(quotas.items(), key=lambda item: item[1], reverse=True):
        if remaining <= 0:
            break
        counts[family] += 1
        remaining -= 1
    return counts


def sample_group_task_family(group: list[Sample]) -> str:
    families = [sample_task_family(sample) for sample in flatten_samples(group)]
    counts = Counter(family for family in families if family)
    return counts.most_common(1)[0][0] if counts else "unknown"


def sample_group_mean_steps(group: list[Sample]) -> float:
    steps: list[float] = []
    seen_trajectories = set()
    for sample in flatten_samples(group):
        metadata = getattr(sample, "metadata", {}) or {}
        parent_traj_id = metadata.get("parent_traj_id")
        if parent_traj_id is not None:
            if parent_traj_id in seen_trajectories:
                continue
            seen_trajectories.add(parent_traj_id)
        value = metadata.get("fused_traj_steps", metadata.get("traj_steps", 0))
        try:
            step_count = float(value)
        except (TypeError, ValueError):
            step_count = 0.0
        steps.append(step_count if math.isfinite(step_count) else 0.0)
    return sum(steps) / len(steps) if steps else 0.0


def sample_task_family(sample: Sample) -> str:
    metadata = getattr(sample, "metadata", {}) or {}
    for key in ("fused_task_type", "task_type", "data_source"):
        value = metadata.get(key)
        if value:
            return normalize_task_family(value)
    if metadata.get("tools_py") or metadata.get("environment"):
        return "mcp"
    if metadata.get("docker_image"):
        return "cli"
    return "webqa"


def normalize_task_family(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"web_search", "search", "webqa"}:
        return "webqa"
    if normalized in {"mcp", "tool", "tools"}:
        return "mcp"
    if normalized in {"cli", "swe", "et", "endless_terminal", "endless_terminals"}:
        return "cli"
    return normalized


def flatten_samples(group) -> list[Sample]:
    samples: list[Sample] = []
    stack = list(group)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack[:0] = item
        else:
            samples.append(item)
    return samples
