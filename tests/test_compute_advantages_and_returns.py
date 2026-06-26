from argparse import Namespace

import torch

from slime.backends.megatron_utils import loss


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
