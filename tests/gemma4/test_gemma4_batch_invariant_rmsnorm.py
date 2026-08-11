import pytest
import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import BatchInvariantRMSNormFn


@pytest.mark.unit
@pytest.mark.parametrize("zero_centered_gamma", [False, True])
@pytest.mark.parametrize("match_sglang", [False, True])
def test_gemma4_scale_rmsnorm_backward_matches_reference(monkeypatch, zero_centered_gamma, match_sglang):
    """Gemma4 has trained RMSNorm weights up to roughly 300, not unit weights."""
    if not torch.cuda.is_available():
        pytest.skip("batch-invariant RMSNorm is CUDA-only")

    if match_sglang:
        monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")

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


@pytest.mark.unit
@pytest.mark.parametrize("hidden_size", [256, 512, 2560])
def test_gemma4_batch_invariant_rmsnorm_forward_matches_sglang(monkeypatch, hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("batch-invariant RMSNorm is CUDA-only")

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm as sglang_rms_norm

    monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")
    torch.manual_seed(23)
    x = torch.randn(17, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    actual = BatchInvariantRMSNormFn.apply(x, weight, 1e-6, False)
    expected = sglang_rms_norm(x.detach(), weight.detach(), 1e-6)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_gemma4_ple_projection_norm_matches_sglang(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("batch-invariant RMSNorm is CUDA-only")

    from sglang.srt.batch_invariant_ops.batch_invariant_ops import rms_norm as sglang_rms_norm

    from slime_plugins.models.gemma4_provider import _Gemma4RMSNorm

    monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")
    torch.manual_seed(29)
    x = torch.randn(19, 42, 256, device="cuda", dtype=torch.bfloat16)
    norm = _Gemma4RMSNorm(256, eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data.normal_()

    actual = norm(x)
    expected = sglang_rms_norm(x, norm.weight.detach(), norm.eps)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
