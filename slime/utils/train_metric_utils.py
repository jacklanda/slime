import logging
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.timer import Timer

logger = logging.getLogger(__name__)


def log_perf_data_raw(
    rollout_id: int,
    args: Namespace,
    is_primary_rank: bool,
    compute_total_fwd_flops: Callable,
    extra_metrics: dict | None = None,
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    if extra_metrics:
        log_dict.update(extra_metrics)

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]
            log_dict["train/mfu"] = log_dict["perf/actor_train_tflops"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    log_dict.update(_compute_rllm_timing_metrics(log_dict, timer_instance.seq_lens))

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")


def _compute_rllm_timing_metrics(log_dict: dict, seq_lens: list[int]) -> dict[str, float]:
    timer_to_timing = {
        "perf/update_weights_time": "update_weights",
        "perf/actor_train_time": "update_actor",
        "perf/step_time": "step",
        "perf/log_probs_time": "old_log_probs",
        "perf/adv_time": "adv",
    }
    metrics = {}
    for src, name in timer_to_timing.items():
        if src in log_dict:
            metrics[f"timing_s/{name}"] = log_dict[src]

    num_tokens = sum(seq_lens)
    if num_tokens > 0:
        for name in ("update_actor", "adv"):
            timing_key = f"timing_s/{name}"
            if timing_key in metrics:
                metrics[f"timing_per_token_ms/{name}"] = metrics[timing_key] * 1000 / num_tokens
    return metrics
