import os
from collections import defaultdict
from typing import Any

import torch

from slime.utils.types import Sample


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw_rewards = [_reward_value(args, sample) for sample in samples]
    shaped_rewards = [_horizon_shaped_reward(reward, sample) for reward, sample in zip(raw_rewards, samples, strict=True)]

    if not (
        args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, shaped_rewards

    normalized = [0.0] * len(samples)
    for indices in _group_sample_indices(samples).values():
        rewards = torch.tensor([shaped_rewards[i] for i in indices], dtype=torch.float32)
        rewards = rewards - rewards.mean()
        if args.advantage_estimator in ["grpo", "gspo", "cispo"] and args.grpo_std_normalization:
            std = rewards.std() if rewards.numel() > 1 else torch.tensor(0.0, dtype=rewards.dtype)
            rewards = rewards / (std + 1e-6)
        for index, reward in zip(indices, rewards.tolist(), strict=True):
            normalized[index] = reward

    return raw_rewards, normalized


def _horizon_shaped_reward(raw_reward: float, sample: Sample) -> float:
    reward = _clamp(float(raw_reward), 0.0, 1.0)
    if reward <= 0.0 or _is_abnormal(sample):
        return reward

    target_steps = max(_float_env("FUSED_HORIZON_REWARD_TARGET_STEPS", 8.0), 1.0)
    target_tool_calls = max(
        _float_env("FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS", target_steps - 1.0),
        1.0,
    )
    step_weight = max(_float_env("FUSED_HORIZON_REWARD_STEP_WEIGHT", 0.7), 0.0)
    tool_call_weight = max(_float_env("FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT", 0.3), 0.0)
    weight_sum = step_weight + tool_call_weight
    if weight_sum <= 0.0:
        step_weight = 1.0
        tool_call_weight = 0.0
        weight_sum = 1.0
    step_weight /= weight_sum
    tool_call_weight /= weight_sum

    min_multiplier = _clamp(_float_env("FUSED_HORIZON_REWARD_MIN_MULTIPLIER", 0.2), 0.0, 1.0)
    gamma = max(_float_env("FUSED_HORIZON_REWARD_GAMMA", 1.0), 1e-6)

    metadata = sample.metadata or {}
    steps = _metadata_float(metadata, "fused_traj_steps", "traj_steps")
    tool_calls = _metadata_float(metadata, "fused_tool_call_turns", "tool_call_turns", "tool_call_turn")
    step_progress = _clamp(steps / target_steps, 0.0, 1.0)
    tool_call_progress = _clamp(tool_calls / target_tool_calls, 0.0, 1.0)
    horizon_progress = _clamp(step_weight * step_progress + tool_call_weight * tool_call_progress, 0.0, 1.0)
    multiplier = min_multiplier + (1.0 - min_multiplier) * (horizon_progress**gamma)
    return _clamp(reward * multiplier, 0.0, 1.0)


def _reward_value(args, sample: Sample) -> float:
    value = sample.get_reward_value(args)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"horizon reward shaping requires scalar reward, got {type(value).__name__}")


def _group_sample_indices(samples: list[Sample]) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        group_id = sample.group_index if sample.group_index is not None else sample.rollout_id
        if group_id is None:
            group_id = index
        groups[group_id].append(index)
    return groups


def _is_abnormal(sample: Sample) -> bool:
    metadata = sample.metadata or {}
    termination = metadata.get("fused_termination") or metadata.get("termination_reason") or ""
    termination = str(termination)
    return termination.startswith("ABNORMAL") or "exceeded" in termination or termination in {"error", "timeout"}


def _metadata_float(metadata: dict[str, Any], *keys: str) -> float:
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
