import pytest
import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import BatchInvariantRMSNormFn


@pytest.mark.unit
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_gemma4_scale_rmsnorm_backward_matches_reference(zero_centered_gamma):
    """Gemma4 has trained RMSNorm weights up to roughly 300, not unit weights."""
    if not torch.cuda.is_available():
        pytest.skip("batch-invariant RMSNorm is CUDA-only")

    torch.manual_seed(17)
    x = torch.randn(13, 41, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.linspace(0.001, 298.0, 41, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    grad_output = torch.randn_like(x)
    eps = 1e-6

    output = BatchInvariantRMSNormFn.apply(x, weight, eps, zero_centered_gamma)
    weight_eff_ref = weight_ref + 1.0 if zero_centered_gamma else weight_ref
    output_ref = x_ref * torch.rsqrt(x_ref.square().mean(dim=-1, keepdim=True) + eps) * weight_eff_ref

    output.backward(grad_output)
    output_ref.backward(grad_output)

    torch.testing.assert_close(output, output_ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-5, atol=1e-5)
