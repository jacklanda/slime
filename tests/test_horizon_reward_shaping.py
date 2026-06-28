from types import SimpleNamespace

import pytest

from slime.rollout.filter_hub.horizon_reward_shaping import _horizon_shaped_reward, post_process_rewards
from slime.utils.types import Sample


NUM_GPUS = 0


def _sample(reward: float, *, steps: int, tool_calls: int, group_index: int = 0, termination: str = "env_done"):
    return Sample(
        reward=reward,
        group_index=group_index,
        metadata={
            "fused_traj_steps": steps,
            "fused_tool_call_turns": tool_calls,
            "fused_termination": termination,
        },
    )


def _args(**overrides):
    values = {
        "reward_key": None,
        "advantage_estimator": "grpo",
        "rewards_normalization": True,
        "grpo_std_normalization": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_horizon_shaped_reward_stays_in_unit_interval(monkeypatch):
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TARGET_STEPS", "8")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS", "7")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_MIN_MULTIPLIER", "0.2")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_STEP_WEIGHT", "1")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT", "0")

    low = _horizon_shaped_reward(1.0, _sample(1.0, steps=0, tool_calls=0))
    mid = _horizon_shaped_reward(1.0, _sample(1.0, steps=4, tool_calls=0))
    high = _horizon_shaped_reward(1.0, _sample(1.0, steps=8, tool_calls=0))
    over = _horizon_shaped_reward(2.0, _sample(2.0, steps=16, tool_calls=0))

    assert low == pytest.approx(0.2)
    assert mid == pytest.approx(0.6)
    assert high == pytest.approx(1.0)
    assert over == pytest.approx(1.0)


def test_horizon_shaped_reward_does_not_boost_failed_or_abnormal_samples(monkeypatch):
    monkeypatch.setenv("FUSED_HORIZON_REWARD_MIN_MULTIPLIER", "0.2")

    assert _horizon_shaped_reward(0.0, _sample(0.0, steps=8, tool_calls=7)) == 0.0
    assert _horizon_shaped_reward(
        1.0,
        _sample(1.0, steps=8, tool_calls=7, termination="ABNORMAL_REPEATED_QUERY"),
    ) == 1.0


def test_post_process_rewards_uses_shaped_rewards_for_group_advantage(monkeypatch):
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TARGET_STEPS", "8")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TARGET_TOOL_CALLS", "7")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_MIN_MULTIPLIER", "0.2")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_STEP_WEIGHT", "1")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT", "0")

    raw_rewards, rewards = post_process_rewards(
        _args(),
        [
            _sample(1.0, steps=0, tool_calls=0, group_index=0),
            _sample(1.0, steps=8, tool_calls=7, group_index=0),
        ],
    )

    assert raw_rewards == [1.0, 1.0]
    assert rewards == pytest.approx([-0.4, 0.4])


def test_post_process_rewards_keeps_unnormalized_shaped_rewards_when_normalization_disabled(monkeypatch):
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TARGET_STEPS", "8")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_MIN_MULTIPLIER", "0.2")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_STEP_WEIGHT", "1")
    monkeypatch.setenv("FUSED_HORIZON_REWARD_TOOL_CALL_WEIGHT", "0")

    raw_rewards, rewards = post_process_rewards(
        _args(rewards_normalization=False),
        [
            _sample(1.0, steps=0, tool_calls=0),
            _sample(1.0, steps=8, tool_calls=0),
        ],
    )

    assert raw_rewards == [1.0, 1.0]
    assert rewards == pytest.approx([0.2, 1.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
