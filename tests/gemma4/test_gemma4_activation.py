import pytest
import torch
import torch.nn.functional as F

from slime_plugins.models.gemma4_activation import Gemma4GeluAndMul


@pytest.mark.unit
def test_gemma4_geglu_has_empty_sharded_state_dict():
    activation = Gemma4GeluAndMul()

    assert activation.state_dict() == {}
    assert activation.sharded_state_dict("model.activation.", (), {}) == {}


@pytest.mark.unit
def test_gemma4_geglu_forward_matches_sglang_bitwise(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("SGLang fused GeGLU is CUDA-only")

    from sgl_kernel import gelu_tanh_and_mul

    monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")
    torch.manual_seed(37)
    inputs = torch.randn(19, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    actual = Gemma4GeluAndMul()(inputs)
    expected = gelu_tanh_and_mul(inputs.detach())

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma4_geglu_backward_matches_reference(monkeypatch, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Gemma4 fused GeGLU backward test requires CUDA")

    monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")
    torch.manual_seed(41)
    inputs = torch.randn(11, 512, device="cuda", dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().float().requires_grad_(True)
    grad_output = torch.randn(11, 256, device="cuda", dtype=dtype)

    actual = Gemma4GeluAndMul()(inputs)
    gate, up = torch.chunk(reference_inputs, 2, dim=-1)
    expected = F.gelu(gate, approximate="tanh") * up
    actual.backward(grad_output)
    expected.backward(grad_output.float())

    torch.testing.assert_close(inputs.grad, reference_inputs.grad.to(dtype), rtol=2e-6, atol=2e-6)
