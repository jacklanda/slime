import pytest

from slime.utils.train_step_metrics import build_step_timing_metrics


NUM_GPUS = 0


def test_build_step_timing_metrics_uses_full_wall_clock_for_wait_time():
    metrics = build_step_timing_metrics(
        step_time=1200.0,
        train_time=120.0,
        phase_times={"rollout_phase": 1000.0, "trainer_create": 50.0},
        useful_training_tokens=2400,
    )

    assert metrics["perf/step_time"] == 1200.0
    assert metrics["perf/train_time"] == 120.0
    assert metrics["perf/train_wait_time"] == 1080.0
    assert metrics["perf/wait_time_ratio"] == pytest.approx(0.9)
    assert metrics["perf/rollout_phase_time"] == 1000.0
    assert metrics["perf/trainer_create_time"] == 50.0
    assert metrics["perf/useful_tokens"] == 2400.0
    assert metrics["perf/useful_tokens_per_sec"] == 2.0


def test_core_timing_metrics_cannot_be_overridden_by_phase_names():
    with pytest.raises(ValueError, match="conflicts with a core timing metric"):
        build_step_timing_metrics(step_time=10.0, train_time=4.0, phase_times={"train": 3.0})


def test_build_step_timing_metrics_handles_zero_and_clock_skew():
    zero = build_step_timing_metrics(
        step_time=0.0,
        train_time=0.0,
        phase_times={},
        useful_training_tokens=10,
    )
    skew = build_step_timing_metrics(step_time=1.0, train_time=1.1, phase_times={})

    assert zero["perf/wait_time_ratio"] == 0.0
    assert zero["perf/useful_tokens_per_sec"] == 0.0
    assert skew["perf/train_wait_time"] == 0.0
    assert skew["perf/wait_time_ratio"] == 0.0


def test_build_step_timing_metrics_rejects_negative_durations():
    with pytest.raises(ValueError, match="must be non-negative"):
        build_step_timing_metrics(step_time=-1.0, train_time=0.0, phase_times={})
    with pytest.raises(ValueError, match="phase time must be non-negative"):
        build_step_timing_metrics(step_time=1.0, train_time=0.5, phase_times={"eval": -0.1})
    with pytest.raises(ValueError, match="useful_training_tokens must be non-negative"):
        build_step_timing_metrics(
            step_time=1.0,
            train_time=0.5,
            phase_times={},
            useful_training_tokens=-1,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
