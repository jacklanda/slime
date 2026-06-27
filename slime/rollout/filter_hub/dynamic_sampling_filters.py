import os

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = [
    "check_reward_nonzero_std",
    "check_reward_nonzero_std_agentic",
    "check_reward_nonzero_std_and_fused_steps",
]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonzero_std_agentic(args, samples: list[Sample], **kwargs):
    flat_samples = list(_iter_samples(samples))
    rewards = [sample.get_reward_value(args) for sample in flat_samples]
    if torch.tensor(rewards, dtype=torch.float64).std() > 1e-6:
        return DynamicFilterOutput(keep=True)

    scores = [
        _agentic_score(args, sample, rollout_id=kwargs.get("rollout_id"))
        for sample in flat_samples
    ]
    keep = bool(torch.tensor(scores, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_agentic_score_{_reason_value(scores[0] if scores else 0.0)}",
    )


def check_reward_nonzero_std_and_fused_steps(args, samples: list[Sample], **kwargs):
    reward_filter_output = check_reward_nonzero_std(args, samples, **kwargs)
    if not reward_filter_output.keep:
        return reward_filter_output

    flat_samples = list(_iter_samples(samples))
    min_mean_steps = _float_env("FUSED_FILTER_MIN_MEAN_STEPS", 0.0)
    min_mcp_mean_steps = _float_env("FUSED_FILTER_MIN_MCP_MEAN_STEPS", 0.0)
    max_abnormal_ratio = _float_env("FUSED_FILTER_MAX_ABNORMAL_RATIO", 0.0)

    if max_abnormal_ratio > 0 and flat_samples:
        abnormal_ratio = _mean([1.0 if _is_abnormal(sample) else 0.0 for sample in flat_samples])
        if abnormal_ratio > max_abnormal_ratio:
            return DynamicFilterOutput(
                keep=False,
                reason=f"high_abnormal_ratio_{_reason_value(abnormal_ratio)}_gt_{_reason_value(max_abnormal_ratio)}",
            )

    if min_mean_steps > 0:
        mean_steps = _mean(_fused_steps(flat_samples))
        if mean_steps < min_mean_steps:
            return DynamicFilterOutput(
                keep=False,
                reason=f"low_steps_{_reason_value(mean_steps)}_lt_{_reason_value(min_mean_steps)}",
            )

    if min_mcp_mean_steps > 0:
        mcp_steps = _fused_steps(
            sample
            for sample in flat_samples
            if (sample.metadata or {}).get("fused_task_type") == "mcp"
        )
        if mcp_steps and _mean(mcp_steps) < min_mcp_mean_steps:
            return DynamicFilterOutput(
                keep=False,
                reason=f"low_mcp_steps_{_reason_value(_mean(mcp_steps))}_lt_{_reason_value(min_mcp_mean_steps)}",
            )

    return DynamicFilterOutput(keep=True)


def _iter_samples(samples):
    for sample in samples:
        if isinstance(sample, list):
            yield from _iter_samples(sample)
        else:
            yield sample


def _fused_steps(samples) -> list[float]:
    steps = []
    for sample in samples:
        steps.append(_metadata_float(sample, "fused_traj_steps", "traj_steps"))
    return steps


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_abnormal(sample: Sample) -> bool:
    metadata = sample.metadata or {}
    termination = metadata.get("fused_termination") or metadata.get("termination_reason") or ""
    termination = str(termination)
    return (
        termination.startswith("ABNORMAL")
        or "exceeded" in termination
        or termination in {"error", "timeout"}
    )


def _agentic_score(args, sample: Sample, *, rollout_id: int | None = None) -> float:
    reward = float(sample.get_reward_value(args))
    if reward <= 0 or _is_abnormal(sample):
        return reward

    target_steps = _agentic_target_steps(rollout_id)
    target_tool_calls = _float_env("FUSED_AGENTIC_FILTER_TARGET_TOOL_CALLS", max(1.0, target_steps - 1.0))
    step_bonus = _float_env("FUSED_AGENTIC_FILTER_STEP_BONUS", 0.05)
    tool_call_bonus = _float_env("FUSED_AGENTIC_FILTER_TOOL_CALL_BONUS", 0.03)
    steps = _metadata_float(sample, "fused_traj_steps", "traj_steps")
    tool_calls = _metadata_float(sample, "fused_tool_call_turns", "tool_call_turns", "tool_call_turn")
    return reward + step_bonus * min(max(steps, 0.0) / target_steps, 1.0) + tool_call_bonus * min(
        max(tool_calls, 0.0) / target_tool_calls,
        1.0,
    )


def _agentic_target_steps(rollout_id: int | None) -> float:
    start = _float_env("FUSED_AGENTIC_FILTER_TARGET_STEPS", 8.0)
    end = _float_env("FUSED_AGENTIC_FILTER_TARGET_STEPS_END", start)
    warmup_rollouts = _float_env("FUSED_AGENTIC_FILTER_TARGET_STEPS_WARMUP_ROLLOUTS", 0.0)
    if warmup_rollouts <= 0 or rollout_id is None:
        return max(1.0, start)
    progress = min(max(float(rollout_id), 0.0) / warmup_rollouts, 1.0)
    return max(1.0, start + (end - start) * progress)


def _metadata_float(sample: Sample, *keys: str) -> float:
    metadata = sample.metadata or {}
    for key in keys:
        if key not in metadata:
            continue
        try:
            return float(metadata[key] or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _reason_value(value: float) -> str:
    return str(round(value, 1)).replace(".", "p")
