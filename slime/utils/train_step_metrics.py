from collections.abc import Mapping


def build_step_timing_metrics(
    *,
    step_time: float,
    train_time: float,
    phase_times: Mapping[str, float],
) -> dict[str, float]:
    """Build synchronous-loop wall times; wait is all non-training critical-path time."""
    if step_time < 0 or train_time < 0:
        raise ValueError("step_time and train_time must be non-negative")

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
    return metrics
