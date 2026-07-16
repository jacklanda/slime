from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


PROMPT_EQUAL_LOSS_ESTIMATORS = {"grpo", "reinforce_plus_plus_baseline"}
_SEGMENT_REWARD_NORMALIZATION_ESTIMATORS = {
    "grpo",
    "gspo",
    "cispo",
    "reinforce_plus_plus_baseline",
}


def uses_prompt_equal_loss(samples: list[Any]) -> bool:
    return any(bool((getattr(sample, "metadata", None) or {}).get("prompt_equal_loss")) for sample in samples)


def has_multi_segment_trajectories(samples: list[Any]) -> bool:
    """True when any trajectory in the batch is split across multiple samples
    (TiTO fork segments), regardless of the loss scheme."""
    return any(int((getattr(sample, "metadata", None) or {}).get("segment_count", 1) or 1) > 1 for sample in samples)


def _is_segment_marked(metadata: dict[str, Any]) -> bool:
    """A sample participates in trajectory-level reward handling when it either
    opted into the prompt-equal loss scheme or is one segment of a forked
    trajectory (non-discard TiTO fork). Grouping both by parent_traj_id keeps a
    k-segment trajectory from voting k times in group normalization."""
    if metadata.get("prompt_equal_loss"):
        return True
    return metadata.get("parent_traj_id") is not None and int(metadata.get("segment_count", 1) or 1) > 1


def trajectory_level_rewards(args: Any, samples: list[Any]) -> list[float]:
    """Collapse per-segment rewards to one value per trajectory.

    The anchor segment (max segment_index) carries the trajectory's terminal
    reward. Current fused rollouts keep sibling rewards sparse, while legacy or
    external TiTO producers may duplicate the terminal reward on every segment;
    taking the anchor's value is correct for both. Samples without a
    parent_traj_id each count as their own trajectory.
    """
    anchor_rewards: dict[str, float] = {}
    anchor_segment_indices: dict[str, int] = {}
    order: list[str] = []
    for position, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", None) or {}
        parent_traj_id = metadata.get("parent_traj_id")
        key = str(parent_traj_id) if parent_traj_id is not None else f"__no_parent_{position}"
        segment_index = int(metadata.get("segment_index", 0) or 0)
        if key not in anchor_rewards:
            order.append(key)
            anchor_rewards[key] = float(sample.get_reward_value(args))
            anchor_segment_indices[key] = segment_index
        elif segment_index >= anchor_segment_indices[key]:
            anchor_rewards[key] = float(sample.get_reward_value(args))
            anchor_segment_indices[key] = segment_index
    return [anchor_rewards[key] for key in order]


def prompt_equal_mask_sums(
    args: Any,
    samples: list[Any],
    masks: list[list[int]],
    rollout_ids: list[int],
) -> list[float]:
    """Return step-local D_P = M_P * N_P / GBS for prompt-equal samples."""
    assert len(samples) == len(masks) == len(rollout_ids)
    global_batch_size = int(getattr(args, "global_batch_size", 0) or 0)
    assert global_batch_size > 0, "prompt-equal aggregation requires global_batch_size > 0"

    rollout_order = list(dict.fromkeys(rollout_ids))
    rollout_to_step = {rollout_id: position // global_batch_size for position, rollout_id in enumerate(rollout_order)}
    prompt_token_counts: dict[tuple[int, str], int] = {}
    live_prompt_ids: dict[int, set[str]] = defaultdict(set)

    unmarked_rollout_totals: dict[int, int] = defaultdict(int)
    for sample, mask, rollout_id in zip(samples, masks, rollout_ids, strict=True):
        if not (getattr(sample, "metadata", None) or {}).get("prompt_equal_loss"):
            unmarked_rollout_totals[rollout_id] += sum(mask)

    for sample, mask, rollout_id in zip(samples, masks, rollout_ids, strict=True):
        metadata = getattr(sample, "metadata", None) or {}
        if not metadata.get("prompt_equal_loss"):
            continue
        parent_traj_id = metadata.get("parent_traj_id")
        instance_id = metadata.get("instance_id")
        is_dead = bool(getattr(sample, "remove_sample", False))
        assert parent_traj_id is not None, f"sample at index {getattr(sample, 'index', '?')} has no metadata['parent_traj_id']; " "multi-segment training requires this"
        assert instance_id is not None, f"sample at index {getattr(sample, 'index', '?')} has no metadata['instance_id']; " "prompt-equal aggregation requires this"
        if not is_dead:
            assert getattr(sample, "group_index", None) is not None, f"live sample at index {getattr(sample, 'index', '?')} " f"(instance_id={instance_id!r}) has group_index=None"
            prompt_id = str(instance_id)
            step = rollout_to_step[rollout_id]
            key = (step, prompt_id)
            live_prompt_ids[step].add(prompt_id)
            prompt_token_counts[key] = prompt_token_counts.get(key, 0) + sum(mask)

    result = []
    for sample, rollout_id in zip(samples, rollout_ids, strict=True):
        metadata = getattr(sample, "metadata", None) or {}
        if not metadata.get("prompt_equal_loss"):
            result.append(float(unmarked_rollout_totals[rollout_id]))
            continue
        step = rollout_to_step[rollout_id]
        prompt_id = str(metadata["instance_id"])
        scale = len(live_prompt_ids[step]) / global_batch_size
        result.append(float(prompt_token_counts.get((step, prompt_id), 0)) * scale)
    return result


def process_segment_rewards(
    args: Any,
    samples: list[Any],
    raw_rewards: list[float],
    shaped_rewards: list[float],
) -> list[float]:
    """Normalize trajectory anchors per prompt group, then broadcast to sibling segments.

    Covers both segment flavors: prompt-equal (discard-historical-thinking)
    samples and plain multi-segment TiTO forks. Without this, a k-segment
    trajectory would enter group normalization k times and out-weigh its
    single-segment peers.
    """
    assert len(raw_rewards) == len(samples)
    assert len(shaped_rewards) == len(samples)

    parent_groups: dict[str, list[int]] = defaultdict(list)
    unmarked_indices = []
    for index, sample in enumerate(samples):
        metadata = getattr(sample, "metadata", None) or {}
        if not _is_segment_marked(metadata):
            unmarked_indices.append(index)
            continue
        parent_traj_id = metadata.get("parent_traj_id")
        assert parent_traj_id is not None, f"sample at index {getattr(sample, 'index', '?')} has no metadata['parent_traj_id']; " "multi-segment reward broadcasting requires this"
        parent_groups[str(parent_traj_id)].append(index)

    anchors = {}
    for parent_traj_id, indices in parent_groups.items():
        anchor = max(
            indices,
            key=lambda index: int((getattr(samples[index], "metadata", None) or {}).get("segment_index", 0)),
        )
        anchor_metadata = getattr(samples[anchor], "metadata", None) or {}
        segment_count = anchor_metadata.get("segment_count")
        if segment_count is not None:
            # The terminal segment carries the trajectory's reward; if a filter
            # hook physically dropped it while keeping siblings, silently
            # promoting a 0.0-placeholder segment to anchor would broadcast a
            # zero reward over the whole trajectory.
            assert int(anchor_metadata.get("segment_index", 0)) == int(segment_count) - 1, f"trajectory {parent_traj_id!r}: reward anchor segment " f"(segment_index={segment_count - 1}) is missing from the batch; " f"max present segment_index={anchor_metadata.get('segment_index', 0)}"
        anchors[parent_traj_id] = anchor

    estimator = getattr(args, "advantage_estimator", None)
    normalize = estimator in _SEGMENT_REWARD_NORMALIZATION_ESTIMATORS and bool(getattr(args, "rewards_normalization", False))
    processed = list(shaped_rewards)
    if normalize:
        prompt_groups: dict[Any, list[int]] = defaultdict(list)
        for representative in [*anchors.values(), *unmarked_indices]:
            group_index = getattr(samples[representative], "group_index", None)
            assert group_index is not None or bool(getattr(samples[representative], "remove_sample", False)), f"live reward representative at index {getattr(samples[representative], 'index', '?')} has group_index=None"
            prompt_groups[group_index].append(representative)

        for representative_indices in prompt_groups.values():
            values = torch.tensor([shaped_rewards[index] for index in representative_indices], dtype=torch.float32)
            normalized = values - values.mean()
            # Match the Dressage reference implementation's population std.
            # A singleton group has zero advantage after mean-centering, so skip
            # the std division rather than relying on epsilon alone.
            if values.numel() > 1 and estimator in {"grpo", "gspo", "cispo"} and bool(getattr(args, "grpo_std_normalization", False)):
                normalized = normalized / (normalized.std(correction=0) + 1e-6)
            for index, value in zip(representative_indices, normalized.tolist(), strict=True):
                processed[index] = value

    for parent_traj_id, indices in parent_groups.items():
        anchor_value = processed[anchors[parent_traj_id]]
        for index in indices:
            processed[index] = anchor_value
    return processed
