from collections import defaultdict
from dataclasses import dataclass
import math

from slime.utils.credit_assignment import CreditAssignmentConfig, excluded_from_reward_baseline
from slime.utils.prompt_equal import has_multi_segment_trajectories, trajectory_level_samples, uses_prompt_equal_loss


@dataclass
class DynamicFilterOutput:
    keep: bool
    reason: str | None = None


def call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)

    output = fn(*args, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)

    return output


def is_valid_reward_group(args, samples) -> bool:
    rewards = []
    for sample in samples:
        status = getattr(sample, "status", None)
        if getattr(status, "value", status) == "failed":
            return False
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        if metadata.get("failure_class") or metadata.get("infra_failure") or metadata.get("fused_infra_failure"):
            return False
        for key in ("fused_reward_debug", "reward_debug"):
            debug = metadata.get(key)
            if isinstance(debug, dict) and (
                debug.get("failure_class")
                or debug.get("infra_failure")
                or debug.get("tools_load_error")
                or debug.get("verifier_error")
            ):
                return False
        try:
            reward = float(sample.get_reward_value(args))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if math.isfinite(reward):
            rewards.append(reward)
    return len(rewards) > 1 and max(rewards) - min(rewards) > 1e-6


def reward_baseline_samples(args, samples):
    if uses_prompt_equal_loss(samples) or has_multi_segment_trajectories(samples):
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
    return reward_samples


def values_have_nonzero_std(values) -> bool:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return len(finite_values) > 1 and max(finite_values) - min(finite_values) > 1e-6


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._completed_groups = 0
        self._valid_groups = 0
        self._webqa_reward_shadow = defaultdict(lambda: 0)

    def on_completed_group(self, args, samples):
        self._completed_groups += 1
        if is_valid_reward_group(args, samples):
            self._valid_groups += 1
        self._on_webqa_reward_shadow(args, samples)

    def _on_webqa_reward_shadow(self, args, samples):
        if not samples or not all(
            str((getattr(sample, "metadata", None) or {}).get("data_source", "")).lower() == "webqa"
            for sample in samples
        ):
            return
        reward_samples = reward_baseline_samples(args, samples)
        if not reward_samples:
            return
        debug_rows = [
            (sample.metadata or {}).get("fused_reward_debug")
            or (sample.metadata or {}).get("reward_debug")
            or {}
            for sample in reward_samples
        ]
        if not all("exact_reward" in debug and "span_reward" in debug and "alias_reward" in debug for debug in debug_rows):
            return
        exact_values = [float(debug["exact_reward"]) for debug in debug_rows]
        span_values = [float(debug["span_reward"]) for debug in debug_rows]
        alias_values = [float(debug["alias_reward"]) for debug in debug_rows]
        self._webqa_reward_shadow["completed_groups"] += 1
        self._webqa_reward_shadow["exact_would_keep"] += int(values_have_nonzero_std(exact_values))
        self._webqa_reward_shadow["span_would_keep"] += int(values_have_nonzero_std(span_values))
        self._webqa_reward_shadow["alias_would_keep"] += int(values_have_nonzero_std(alias_values))
        self._webqa_reward_shadow["new_zero_std_1"] += int(
            all(abs(value - 1.0) <= 1e-6 for value in alias_values)
            and not all(abs(value - 1.0) <= 1e-6 for value in exact_values)
        )

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        metrics["rollout/dynamic_filter/valid_groups"] = self._valid_groups
        metrics["rollout/dynamic_filter/roi"] = round(
            self._valid_groups / self._completed_groups, 2
        ) if self._completed_groups else 0.0
        metrics.update(
            {
                f"rollout/webqa_reward_shadow/{name}": count
                for name, count in self._webqa_reward_shadow.items()
            }
        )
        for name in ("completed_groups", "exact_would_keep", "span_would_keep", "alias_would_keep", "new_zero_std_1"):
            metrics.setdefault(f"rollout/webqa_reward_shadow/{name}", 0)
        return metrics
