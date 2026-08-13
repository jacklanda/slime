import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


class _DisplayFloat(float):
    def __repr__(self) -> str:
        return f"{self:.2f}"


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def format_metrics_for_display(value: Any) -> Any:
    """Return a display-only copy with floating-point values rounded to two decimals."""
    if isinstance(value, dict):
        return {key: format_metrics_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [format_metrics_for_display(item) for item in value]
    if isinstance(value, tuple):
        return tuple(format_metrics_for_display(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return _DisplayFloat(round(float(value), 2))
    return value


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def compute_pass_at_k_and_pass_all(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
) -> dict[str, float]:
    if group_size < 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size
    if num_groups < 1:
        return {}

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)
    correct_by_group = np.sum(rewards_of_group == 1, axis=1)
    pass_all = (correct_by_group == group_size).astype(float)
    episode_std_by_group = np.std((rewards_of_group == 1).astype(float), axis=1)
    episode_std = round(float(np.mean(episode_std_by_group)), 3)

    metrics = {}
    for k in _pass_at_k_report_sizes(group_size):
        num_samples = np.full(num_groups, group_size)
        pass_at_k = _estimate_pass_at_k(num_samples, correct_by_group, k)
        metrics |= {
            f"pass@{k}/mean": round(float(np.mean(pass_at_k)), 3),
            f"pass@{k}/std": episode_std,
        }

    metrics |= {
        f"pass^{group_size}/mean": round(float(np.mean(pass_all)), 3),
        f"pass^{group_size}/std": episode_std,
    }
    return metrics


def _pass_at_k_report_sizes(group_size: int) -> list[int]:
    sizes = [2**i for i in range(int(math.log2(group_size)) + 1)]
    if sizes[-1] != group_size:
        sizes.append(group_size)
    return sizes


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_train_step(args, rollout_id):
    return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return compute_train_step(args, rollout_id)
    return rollout_id
