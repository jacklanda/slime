import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils.data import _align_response_loss_mask


def test_align_response_loss_mask_with_prompt():
    loss_mask = torch.tensor([1, 1, 0], dtype=torch.int)

    aligned = _align_response_loss_mask(loss_mask, total_length=5, response_length=3)

    torch.testing.assert_close(aligned, torch.tensor([0, 1, 1, 0, 0], dtype=torch.int))


def test_align_response_loss_mask_without_prompt_masks_unpredictable_first_token():
    loss_mask = torch.tensor([1, 1, 0], dtype=torch.int)

    aligned = _align_response_loss_mask(loss_mask, total_length=3, response_length=3)

    torch.testing.assert_close(aligned, torch.tensor([1, 0, 0], dtype=torch.int))


def test_align_response_loss_mask_rejects_inconsistent_lengths():
    with pytest.raises(ValueError, match="must be >= response_length"):
        _align_response_loss_mask(torch.ones(3, dtype=torch.int), total_length=2, response_length=3)

    with pytest.raises(ValueError, match="must match response_length"):
        _align_response_loss_mask(torch.ones(2, dtype=torch.int), total_length=3, response_length=3)
