from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils import loss
from slime.utils import distributed_utils


NUM_GPUS = 0


def test_masked_whiten_returns_zero_when_no_policy_tokens_remain(monkeypatch):
    monkeypatch.setattr(distributed_utils.dist, "all_reduce", lambda *_args, **_kwargs: None)

    result = distributed_utils.distributed_masked_whiten(
        torch.tensor([3.0, -2.0]),
        torch.zeros(2, dtype=torch.int),
    )

    torch.testing.assert_close(result, torch.zeros(2))


@pytest.mark.parametrize(
    ("loss_type", "expected_mask"),
    [
        ("policy_loss", torch.tensor([1, 0, 0, 0, 1])),
        ("value_loss", torch.ones(5, dtype=torch.int)),
    ],
)
def test_advantage_whitening_uses_loss_specific_masks(monkeypatch, loss_type, expected_mask):
    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(loss.mpu, "get_data_parallel_group", lambda **_kwargs: None)

    captured = {}

    def masked_whiten(values, mask, **_kwargs):
        captured["mask"] = mask.clone()
        active = values[mask.bool()]
        return (values - active.mean()) / active.std(unbiased=False)

    monkeypatch.setattr(loss, "distributed_masked_whiten", masked_whiten)
    args = Namespace(
        use_rollout_logprobs=False,
        kl_coef=0,
        kl_loss_type="k1",
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        loss_type=loss_type,
        use_opd=False,
        normalize_advantages=True,
    )
    rollout_data = {
        "log_probs": None,
        "rollout_log_probs": None,
        "ref_log_probs": None,
        "rewards": [2.0, -1.0],
        "values": None,
        "response_lengths": [3, 2],
        "loss_masks": [torch.ones(3, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "policy_loss_masks": [torch.tensor([1, 0, 0]), torch.tensor([0, 1])],
        "total_lengths": [5, 4],
    }

    loss.compute_advantages_and_returns(args, rollout_data)

    torch.testing.assert_close(captured["mask"], expected_mask)
    advantages = torch.cat(rollout_data["advantages"])
    torch.testing.assert_close(advantages[expected_mask.bool()].mean(), torch.tensor(0.0))
    torch.testing.assert_close(advantages[expected_mask.bool()].std(unbiased=False), torch.tensor(1.0))


@pytest.mark.parametrize("cp_size", [1, 2])
def test_advantage_whitening_reduces_over_the_dp_group_including_cp(monkeypatch, cp_size):
    """The whitening population must span CP.

    Ranks inside one CP group hold the same samples but disjoint token halves.
    Reducing over DP-without-CP gives each CP rank its own mean/var, so one
    sequence would get two different affine transforms.
    """
    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(loss.mpu, "get_context_parallel_world_size", lambda: cp_size)

    captured = {}

    def get_data_parallel_group(**kwargs):
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr(loss.mpu, "get_data_parallel_group", get_data_parallel_group)
    # CP>1 slices each sequence by token offsets; keep the whole response local
    # so this test isolates the process-group choice.
    monkeypatch.setattr(
        loss,
        "get_logits_and_tokens_offset_with_cp",
        lambda total_len, response_len: (None, None, None, [(total_len - response_len, total_len), (0, 0)]),
    )
    monkeypatch.setattr(loss, "distributed_masked_whiten", lambda values, mask, **_kwargs: values)

    args = Namespace(
        use_rollout_logprobs=False,
        kl_coef=0,
        kl_loss_type="k1",
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        loss_type="policy_loss",
        use_opd=False,
        normalize_advantages=True,
    )
    rollout_data = {
        "log_probs": None,
        "rollout_log_probs": None,
        "ref_log_probs": None,
        "rewards": [2.0, -1.0],
        "values": None,
        "response_lengths": [3, 2],
        "loss_masks": [torch.ones(3, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "policy_loss_masks": [torch.tensor([1, 0, 0]), torch.tensor([0, 1])],
        "total_lengths": [5, 4],
    }

    loss.compute_advantages_and_returns(args, rollout_data)

    assert captured["kwargs"] == {"with_context_parallel": True}


def test_zero_kl_uses_loss_masks_when_log_probs_and_values_are_missing(monkeypatch):
    monkeypatch.setattr(loss.mpu, "is_pipeline_last_stage", lambda: True)

    args = Namespace(
        use_rollout_logprobs=False,
        kl_coef=0,
        kl_loss_type="k1",
        custom_advantage_function_path=None,
        advantage_estimator="grpo",
        use_opd=False,
        normalize_advantages=False,
    )
    rollout_data = {
        "log_probs": None,
        "rollout_log_probs": None,
        "ref_log_probs": None,
        "rewards": [1.5, -0.5],
        "values": None,
        "response_lengths": [3, 2],
        "loss_masks": [torch.ones(3, dtype=torch.int), torch.ones(2, dtype=torch.int)],
        "total_lengths": [5, 4],
    }

    loss.compute_advantages_and_returns(args, rollout_data)

    assert [x.dtype for x in rollout_data["kl"]] == [torch.float32, torch.float32]
    torch.testing.assert_close(rollout_data["kl"][0], torch.zeros(3))
    torch.testing.assert_close(rollout_data["kl"][1], torch.zeros(2))
    torch.testing.assert_close(rollout_data["returns"][0], torch.full((3,), 1.5))
    torch.testing.assert_close(rollout_data["returns"][1], torch.full((2,), -0.5))
    torch.testing.assert_close(rollout_data["advantages"][0], rollout_data["returns"][0])
    torch.testing.assert_close(rollout_data["advantages"][1], rollout_data["returns"][1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
