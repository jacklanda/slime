from collections.abc import Mapping


def build_step_timing_metrics(
    *,
    step_time: float,
    train_time: float,
    phase_times: Mapping[str, float],
    useful_training_tokens: int | None = None,
) -> dict[str, float]:
    """Build synchronous-loop wall times; wait is all non-training critical-path time."""
    if step_time < 0 or train_time < 0:
        raise ValueError("step_time and train_time must be non-negative")
    if useful_training_tokens is not None and useful_training_tokens < 0:
        raise ValueError("useful_training_tokens must be non-negative")

    train_wait_time = max(step_time - train_time, 0.0)
    metrics = {
        "perf/step_time": step_time,
        "perf/train_time": train_time,
        "perf/train_wait_time": train_wait_time,
        "perf/wait_time_ratio": train_wait_time / step_time if step_time > 0 else 0.0,
    }
    for name, elapsed in phase_times.items():
        if elapsed < 0:
            raise ValueError(f"phase time must be non-negative: {name}={elapsed}")
        metric_name = f"perf/{name}_time"
        if metric_name in metrics:
            raise ValueError(f"phase name conflicts with a core timing metric: {name}")
        metrics[metric_name] = elapsed
    if useful_training_tokens is not None:
        metrics["perf/useful_tokens"] = float(useful_training_tokens)
        metrics["perf/useful_tokens_per_sec"] = useful_training_tokens / step_time if step_time > 0 else 0.0
    return metrics
