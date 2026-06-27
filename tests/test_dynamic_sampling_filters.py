from types import SimpleNamespace

import pytest

from slime.rollout.filter_hub.dynamic_sampling_filters import (
    _agentic_target_steps,
    check_reward_nonzero_std_agentic,
)
from slime.utils.types import Sample


NUM_GPUS = 0


def _sample(reward: float, *, steps: int, tool_calls: int, termination: str = "env_done"):
    return Sample(
        reward=reward,
        metadata={
            "fused_traj_steps": steps,
            "fused_tool_call_turns": tool_calls,
            "fused_termination": termination,
        },
    )


def test_agentic_nonzero_variance_filter_keeps_correct_groups_with_step_variance(monkeypatch):
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TARGET_STEPS", "8")
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_STEP_BONUS", "0.05")
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TOOL_CALL_BONUS", "0")

    output = check_reward_nonzero_std_agentic(
        SimpleNamespace(reward_key=None),
        [
            _sample(1.0, steps=2, tool_calls=1),
            _sample(1.0, steps=8, tool_calls=7),
        ],
    )

    assert output.keep is True


def test_agentic_nonzero_variance_filter_drops_uniform_correct_groups(monkeypatch):
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TARGET_STEPS", "8")

    output = check_reward_nonzero_std_agentic(
        SimpleNamespace(reward_key=None),
        [
            _sample(1.0, steps=8, tool_calls=7),
            _sample(1.0, steps=8, tool_calls=7),
        ],
    )

    assert output.keep is False
    assert output.reason == "zero_agentic_score_1p1"


def test_agentic_nonzero_variance_filter_does_not_bonus_abnormal_samples(monkeypatch):
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TARGET_STEPS", "8")

    output = check_reward_nonzero_std_agentic(
        SimpleNamespace(reward_key=None),
        [
            _sample(1.0, steps=2, tool_calls=1, termination="ABNORMAL_REPEATED_QUERY"),
            _sample(1.0, steps=8, tool_calls=7, termination="ABNORMAL_NGRAM_REPETITION"),
        ],
    )

    assert output.keep is False


def test_agentic_target_steps_curriculum(monkeypatch):
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TARGET_STEPS", "8")
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TARGET_STEPS_END", "16")
    monkeypatch.setenv("FUSED_AGENTIC_FILTER_TARGET_STEPS_WARMUP_ROLLOUTS", "100")

    assert _agentic_target_steps(0) == 8
    assert _agentic_target_steps(50) == 12
    assert _agentic_target_steps(100) == 16
    assert _agentic_target_steps(200) == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
