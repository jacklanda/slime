from argparse import Namespace

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module


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
@pytest.mark.parametrize("gemma4_tiled", [False, True])
def test_cache_is_cleared_only_at_gemma4_tiled_policy_backward_boundary(monkeypatch, gemma4_tiled):
    monkeypatch.setenv("SLIME_GEMMA4_CLEAR_CACHE_BEFORE_BACKWARD", "1")
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", lambda *_args: lambda value: value.sum())
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 1)

    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    tiled_result = {"log_probs": [torch.zeros(1)], "entropy": [torch.zeros(1)]}
    monkeypatch.setattr(loss_module, "get_gemma4_tiled_log_probs_and_entropy", lambda *_args, **_kwargs: tiled_result)
    monkeypatch.setattr(loss_module, "policy_loss_function", lambda *_args, **_kwargs: (torch.tensor(1.0), {}))

    output_layer = object() if gemma4_tiled else None
    loss_module.loss_function(
        _args(),
        _batch(),
        num_microbatches=1,
        step_global_batch_size=1,
        logits=torch.zeros(1),
        gemma4_output_layer=output_layer,
    )

    assert len(empty_cache_calls) == int(gemma4_tiled)
