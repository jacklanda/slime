"""SR-PPO (sequential/shrinkage credit) advantage implementation.

This module is intentionally independent of the actor and rollout engines.  It
is loaded through slime's ``--custom-advantage-function-path`` hook and expects
the rollout producer to attach one pass-probability per prefix under
``prefix_probs`` (or one of the documented aliases below).
"""

from __future__ import annotations

import math
from typing import Any

import torch


MODES = {
    "shrinkage-critic-k-gradient-1": (True, False),
    "shrinkage-critic-1-gradient-1": (False, False),
    "shrinkage-critic-k-gradient-k": (True, True),
    "shrinkage-critic-1-gradient-k": (False, True),
    "raw": (False, False),
}


def _k(value: Any, default: float = 4.0) -> float:
    result = float(value if value is not None else default)
    if not math.isfinite(result) or result < 1:
        raise ValueError(f"SR-PPO pass_at_k must be finite and >= 1, got {result}")
    return result


def pass_at_k_to_pass_at_1(prob: torch.Tensor, k: float) -> torch.Tensor:
    if k == 1:
        return prob
    return 1.0 - (1.0 - prob.float()).clamp(0, 1).pow(1.0 / k)


def pass_at_k_gradient(prompt_prob: torch.Tensor, k: float, normalize: bool) -> torch.Tensor:
    weights = k * (1.0 - prompt_prob.detach().float()).clamp(0, 1).pow(k - 1.0)
    if normalize:
        weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return weights


def shrinkage_advantage(
    prefix_probs: torch.Tensor | list[torch.Tensor],
    rewards: list[float] | torch.Tensor,
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    *,
    critic_pass_at_k: float,
    gradient_pass_at_k: float,
    normalize_gradient_k: bool,
) -> list[torch.Tensor]:
    if isinstance(prefix_probs, (list, tuple)):
        if len(prefix_probs) != len(rewards):
            raise ValueError("SR-PPO rewards and prefix_probs batch sizes differ")
        probs_list = [pass_at_k_to_pass_at_1(torch.as_tensor(p), critic_pass_at_k) for p in prefix_probs]
        prompt = torch.stack([p[:1] for p in probs_list])
        weights = pass_at_k_gradient(prompt, gradient_pass_at_k, normalize_gradient_k) if gradient_pass_at_k != 1.0 or normalize_gradient_k else None
        result = []
        for i, (probs, length, mask) in enumerate(zip(probs_list, response_lengths, loss_masks, strict=True)):
            values = torch.as_tensor(rewards[i], dtype=probs.dtype, device=probs.device) - probs[:length].detach()
            if weights is not None:
                values = values * weights[i].to(values.dtype)
            result.append(values * mask[:length].to(device=values.device, dtype=values.dtype))
        return result
    if prefix_probs.ndim != 2 or prefix_probs.shape[1] < 1:
        raise ValueError("SR-PPO prefix_probs must have shape [batch, response_len + 1]")
    probs = pass_at_k_to_pass_at_1(prefix_probs, critic_pass_at_k)
    outcomes = torch.as_tensor(rewards, dtype=probs.dtype, device=probs.device).reshape(-1, 1)
    if outcomes.shape[0] != probs.shape[0]:
        raise ValueError("SR-PPO rewards and prefix_probs batch sizes differ")
    adv = outcomes - probs[:, :-1].detach()
    if gradient_pass_at_k != 1.0 or normalize_gradient_k:
        adv = adv * pass_at_k_gradient(probs[:, :1], gradient_pass_at_k, normalize_gradient_k)
    return [row[:length].clone() * mask[:length].to(device=row.device, dtype=row.dtype) for row, length, mask in zip(adv, response_lengths, loss_masks, strict=True)]


def custom_advantage_fn(args, rollout_data) -> None:
    """Populate ``rollout_data['advantages']`` and ``['returns']`` in-place."""
    mode = str(getattr(args, "srppo_mode", None) or "shrinkage-critic-1-gradient-k")
    if mode not in MODES:
        raise ValueError(f"Unknown SR-PPO mode {mode!r}; expected one of {sorted(MODES)}")
    rewards = rollout_data.get("rewards")
    lengths = rollout_data.get("response_lengths")
    masks = rollout_data.get("loss_masks")
    if rewards is None or lengths is None or masks is None:
        raise ValueError("SR-PPO requires rewards, response_lengths, and loss_masks")
    if mode == "raw":
        outcomes = torch.as_tensor(rewards, dtype=torch.float32)
        rollout_data["advantages"] = [
            torch.where(outcomes[i] == 1, mask.float(), -mask.float()) for i, mask in enumerate(masks)
        ]
        rollout_data["returns"] = list(rollout_data["advantages"])
        return

    prefix_probs = None
    for key in ("prefix_probs", "outcome_prefix_probs", "critic_prefix_probs"):
        if rollout_data.get(key) is not None:
            prefix_probs = rollout_data[key]
            break
    if prefix_probs is None:
        # The synchronous Megatron path supplies the independently trained
        # outcome critic as ``values``.  Values are response-aligned; the
        # first response value is the prompt-prefix estimate (s_0).
        values = rollout_data.get("values")
        if values is None:
            raise RuntimeError("SR-PPO needs prefix_probs or critic values from the outcome critic")
        prefix_probs = [torch.sigmoid(v.float()) for v in values]
    if not isinstance(prefix_probs, (torch.Tensor, list, tuple)):
        prefix_probs = torch.as_tensor(prefix_probs, dtype=torch.float32)
    critic_k, gradient_k = MODES[mode]
    k = _k(getattr(args, "srppo_pass_at_k", 4.0))
    normalize = bool(getattr(args, "srppo_normalize_gradient_k", False)) and gradient_k
    advantages = shrinkage_advantage(
        prefix_probs,
        rewards,
        lengths,
        masks,
        critic_pass_at_k=k if critic_k else 1.0,
        gradient_pass_at_k=k if gradient_k else 1.0,
        normalize_gradient_k=normalize,
    )
    rollout_data["advantages"] = advantages
    # The value critic is trained against the trajectory outcome, not against
    # the actor advantage. This is the Monte-Carlo outcome-critic target.
    rollout_data["returns"] = [
        torch.full_like(mask[:length], float(rewards[i]))
        for i, (length, mask) in enumerate(zip(lengths, masks, strict=True))
    ]
