from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils.tiled_policy_loss import install_tiled_policy_loss


def _args() -> Namespace:
    return Namespace(
        loss_type="policy_loss",
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        allgather_cp=False,
    )


def _batch() -> dict:
    mask = torch.ones(1)
    return {
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [mask],
        "policy_loss_masks": [mask],
        "rollout_mask_sums": [mask.sum()],
    }


@pytest.mark.unit
@pytest.mark.parametrize("tiled", [False, True])
def test_cache_is_cleared_only_at_tiled_policy_backward_boundary(monkeypatch, tiled):
    monkeypatch.setenv("SLIME_GEMMA4_CLEAR_CACHE_BEFORE_BACKWARD", "1")
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args: lambda value: value.sum())
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 1)

    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    tiled_result = {"log_probs": [torch.zeros(1)], "entropy": [torch.zeros(1)]}
    monkeypatch.setattr(loss_module, "get_tiled_log_probs_and_entropy", lambda *_args, **_kwargs: tiled_result)
    monkeypatch.setattr(loss_module, "policy_loss_function", lambda *_args, **_kwargs: (torch.tensor(1.0), {}))

    output_layer = object() if tiled else None
    loss_module.loss_function(
        _args(),
        _batch(),
        num_microbatches=1,
        step_global_batch_size=1,
        logits=torch.zeros(1),
        tiled_output_layer=output_layer,
    )

    assert len(empty_cache_calls) == int(tiled)


@pytest.mark.unit
def test_standard_gpt_tiled_policy_loss_bypasses_projection_only_while_active(monkeypatch):
    class FakeGPTModel:
        def _postprocess(self, *args, **kwargs):
            return "full-logits"

    model = FakeGPTModel()
    install_tiled_policy_loss(model)
    hidden_states = torch.arange(12).reshape(3, 1, 4)

    assert model._postprocess(hidden_states=hidden_states, labels=None) == "full-logits"

    monkeypatch.setenv("SLIME_TILED_POLICY_LOSS_ACTIVE", "1")
    result = model._postprocess(hidden_states=hidden_states, labels=None)

    assert result.shape == (1, 3, 4)
    torch.testing.assert_close(result, hidden_states.transpose(0, 1))
