import pytest
import torch

from tests.gemma4._standalone_imports import load_gemma4_provider_module

_logit_softcapping = load_gemma4_provider_module()._logit_softcapping


@pytest.mark.unit
def test_gemma4_softcap_forward_matches_sglang_bitwise(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("SGLang fused softcap is CUDA-only")

    from sglang.srt.layers.logits_processor import fused_softcap

    monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")
    torch.manual_seed(29)
    logits = (torch.randn(7, 4097, device="cuda", dtype=torch.bfloat16) * 40).requires_grad_()
    expected = logits.detach().clone()
    fused_softcap(expected, 30.0)

    actual = _logit_softcapping(logits + 0, 30.0)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_gemma4_softcap_backward_matches_reference(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Gemma4 softcap backward test requires CUDA")

    monkeypatch.setenv("SLIME_GEMMA4_BATCH_INVARIANT", "1")
    torch.manual_seed(31)
    logits = torch.randn(11, 257, device="cuda", dtype=torch.float32, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    grad = torch.randn_like(logits)

    actual = _logit_softcapping(logits + 0, 30.0)
    expected = torch.tanh(reference_logits / 30.0) * 30.0
    actual.backward(grad)
    expected.backward(grad)

    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-6)
