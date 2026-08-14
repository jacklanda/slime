import os

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.failure_types import FailureClass
from slime.utils.credit_assignment import CreditAssignmentConfig, excluded_from_reward_baseline
from slime.utils.prompt_equal import has_multi_segment_trajectories, trajectory_level_samples, uses_prompt_equal_loss
from slime.utils.types import Sample

__all__ = [
    "check_reward_nonzero_std",
    "check_reward_nonzero_std_and_fused_steps",
    "is_infra_failure",
    "group_failure_class",
    "sample_failure_class",
]


_RETRYABLE_INFRA_TERMINATIONS = {
    "infra_failure",
    "mcp_lease_timeout",
    "mcp_process_failure",
    "sglang_transport_error",
}

_PERMANENT_TASK_TERMINATIONS = {
    "env_init_error",
    "error",
    "rollout_task_exception",
}


def sample_failure_class(sample: Sample) -> FailureClass | None:
    """Classify a failed trajectory without treating every FAILED status as infra."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    explicit = metadata.get("failure_class")
    if explicit:
        if isinstance(explicit, FailureClass):
            return explicit
        try:
            return FailureClass(str(explicit))
        except ValueError:
            pass

    for key in ("fused_reward_debug", "reward_debug"):
        reward_debug = metadata.get(key)
        if not isinstance(reward_debug, dict):
            continue
        debug_class = reward_debug.get("failure_class")
        if debug_class:
            try:
                return FailureClass(str(debug_class))
            except ValueError:
                pass
        if reward_debug.get("tools_load_error") or reward_debug.get("verifier_error"):
            return FailureClass.PERMANENT_TASK
        if reward_debug.get("infra_failure"):
            return FailureClass.RETRYABLE_INFRA

    if metadata.get("infra_failure") or metadata.get("fused_infra_failure"):
        return FailureClass.RETRYABLE_INFRA
    termination = str(metadata.get("fused_termination") or metadata.get("termination_reason") or "").lower()
    if termination in _RETRYABLE_INFRA_TERMINATIONS:
        return FailureClass.RETRYABLE_INFRA
    if termination in _PERMANENT_TASK_TERMINATIONS:
        return FailureClass.PERMANENT_TASK

    status = getattr(sample, "status", None)
    if status == Sample.Status.FAILED or getattr(status, "value", status) == Sample.Status.FAILED.value:
        return FailureClass.POLICY
    return None


def group_failure_class(samples) -> FailureClass | None:
    classes = {sample_failure_class(sample) for sample in _iter_samples(samples)}
    classes.discard(None)
    if FailureClass.PERMANENT_TASK in classes:
        return FailureClass.PERMANENT_TASK
    if FailureClass.POLICY in classes:
        return FailureClass.POLICY
    if FailureClass.RETRYABLE_INFRA in classes:
        return FailureClass.RETRYABLE_INFRA
    return None


def is_infra_failure(sample: Sample) -> bool:
    """Return whether a sample has an explicitly retryable infrastructure failure.

    Infrastructure failures must not be treated as ordinary zero-reward model
    outcomes: doing so creates artificial negative advantages and can collapse
    a prompt group's reward variance.
    """
    return sample_failure_class(sample) == FailureClass.RETRYABLE_INFRA


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    failure_class = group_failure_class(samples)
    if failure_class is not None:
        return DynamicFilterOutput(keep=False, reason=failure_class.value)
    if uses_prompt_equal_loss(samples) or has_multi_segment_trajectories(samples):
        # Judge variance on trajectory-level rewards so sparse placeholders or
        # legacy duplicated segment rewards cannot fake or dilute the std.
        reward_samples = trajectory_level_samples(samples)
    else:
        reward_samples = samples

    credit_config = CreditAssignmentConfig.from_args(args)
    if credit_config.enable:
        clean_samples = [
            sample
            for sample in reward_samples
            if not excluded_from_reward_baseline(sample.metadata, credit_config)
        ]
        if clean_samples:
            reward_samples = clean_samples

    rewards = [sample.get_reward_value(args) for sample in reward_samples]
    reward_values = torch.tensor(rewards, dtype=torch.float64)
    spread = reward_values.std() if reward_values.numel() > 1 else torch.tensor(0.0, dtype=reward_values.dtype)
    keep = bool(torch.isfinite(spread)) and bool(spread > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonzero_std_and_fused_steps(args, samples: list[Sample], **kwargs):
    reward_filter_output = check_reward_nonzero_std(args, samples, **kwargs)
    if not reward_filter_output.keep:
        return reward_filter_output

    # Segments of one multi-segment trajectory duplicate the trajectory-level
    # metadata (fused_traj_steps, fused_termination); dedupe to one vote per
    # trajectory so thinking-heavy (multi-segment) trajectories don't bias the
    # step/abnormal statistics.
    flat_samples = _dedupe_by_trajectory(_iter_samples(samples))
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
        mcp_steps = _fused_steps(sample for sample in flat_samples if (sample.metadata or {}).get("fused_task_type") == "mcp")
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


def _dedupe_by_trajectory(samples) -> list[Sample]:
    """Keep one representative per trajectory (first segment seen); samples
    without a parent_traj_id each stand alone."""
    result = []
    seen_trajectories = set()
    for sample in samples:
        parent_traj_id = (sample.metadata or {}).get("parent_traj_id")
        if parent_traj_id is not None:
            if parent_traj_id in seen_trajectories:
                continue
            seen_trajectories.add(parent_traj_id)
        result.append(sample)
    return result


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
    return termination.startswith("ABNORMAL") or "exceeded" in termination or termination in {"error", "timeout"}


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
