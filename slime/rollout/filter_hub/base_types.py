from collections import defaultdict
from dataclasses import dataclass
import math


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
        try:
            reward = float(sample.get_reward_value(args))
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if math.isfinite(reward):
            rewards.append(reward)
    return len(rewards) > 1 and max(rewards) - min(rewards) > 1e-6


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._completed_groups = 0
        self._valid_groups = 0

    def on_completed_group(self, args, samples):
        self._completed_groups += 1
        if is_valid_reward_group(args, samples):
            self._valid_groups += 1

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
        return metrics
