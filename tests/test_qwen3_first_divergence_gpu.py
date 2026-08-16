"""Cross-runtime parity test for Qwen3 decoder residual and RMSNorm boundaries."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


NUM_GPUS = 1
NUM_LAYERS = 36
HIDDEN_SIZE = 64
FFN_SIZE = 128
VOCAB_SIZE = 128
SEQ_LEN = 8


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _weights(shape: tuple[int, ...], scale: float = 0.02) -> torch.Tensor:
    return (torch.randn(shape, device="cuda") * scale).bfloat16()


def _build_norm_pair(weight: torch.Tensor):
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        SGLangDecoderRMSNorm,
    )
    from sglang.srt.layers.layernorm import RMSNorm

    config = SimpleNamespace(
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=False,
        params_dtype=torch.bfloat16,
        sequence_parallel=False,
    )
    megatron_norm = SGLangDecoderRMSNorm(config, HIDDEN_SIZE, eps=1e-6).cuda()
    sglang_norm = RMSNorm(
        HIDDEN_SIZE,
        eps=1e-6,
        weight_dtype=torch.float32,
        cast_x_before_out_mul=True,
        override_orig_dtype=torch.float32,
        fp32_residual=True,
    ).cuda()
    with torch.no_grad():
        megatron_norm.weight.copy_(weight)
        sglang_norm.weight.copy_(weight.float())
    return megatron_norm, sglang_norm


def _project_attention(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(x.bfloat16(), weight)


def _project_mlp(x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x.bfloat16(), gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down_weight)


def _megatron_residual_add(branch: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.transformer.transformer_layer import (
        _prepare_output_for_residual_add,
    )

    branch_with_bias = _prepare_output_for_residual_add((branch, None), True)
    return get_bias_dropout_add(training=False, fused=False)(branch_with_bias, residual, prob=0.0)


def _assert_first_divergence_equal(megatron_captures: dict[str, torch.Tensor], sglang_captures: dict[str, torch.Tensor]) -> None:
    for name in (
        "embedding",
        "layer_0_attention_residual",
        "layer_0_mlp_residual",
        "layer_1_pre_norm",
        "layer_35_residual",
        "final_norm",
        "logits",
    ):
        megatron_value = megatron_captures[name].float()
        sglang_value = sglang_captures[name].float()
        if not torch.equal(megatron_value, sglang_value):
            diff = (megatron_value - sglang_value).abs()
            pytest.fail(f"first Qwen3 train-rollout divergence at {name}: " f"mae={diff.mean().item():.9g}, max={diff.max().item():.9g}")


@torch.no_grad()
def test_qwen3_first_divergence_boundaries_match_sglang_forward_native():
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    torch.manual_seed(20260816)
    embedding_weight = _weights((VOCAB_SIZE, HIDDEN_SIZE))
    lm_head_weight = embedding_weight
    attention_weights = [_weights((HIDDEN_SIZE, HIDDEN_SIZE)) for _ in range(NUM_LAYERS)]
    gate_up_weights = [_weights((2 * FFN_SIZE, HIDDEN_SIZE)) for _ in range(NUM_LAYERS)]
    down_weights = [_weights((HIDDEN_SIZE, FFN_SIZE)) for _ in range(NUM_LAYERS)]
    input_norm_weights = [(1.0 + 0.02 * torch.randn(HIDDEN_SIZE, device="cuda")).bfloat16() for _ in range(NUM_LAYERS)]
    post_norm_weights = [(1.0 + 0.02 * torch.randn(HIDDEN_SIZE, device="cuda")).bfloat16() for _ in range(NUM_LAYERS)]
    final_norm_weight = (1.0 + 0.02 * torch.randn(HIDDEN_SIZE, device="cuda")).bfloat16()
    norm_pairs = [(_build_norm_pair(input_norm_weights[i]), _build_norm_pair(post_norm_weights[i])) for i in range(NUM_LAYERS)]
    final_megatron_norm, final_sglang_norm = _build_norm_pair(final_norm_weight)

    tokens = torch.randint(VOCAB_SIZE, (SEQ_LEN,), device="cuda")
    embedded = F.embedding(tokens, embedding_weight)
    megatron_captures = {"embedding": embedded.float()}
    sglang_captures = {"embedding": embedded.float()}

    enable_batch_invariant_mode()
    try:
        megatron_residual = embedded.float()
        sglang_hidden = embedded
        sglang_residual = None

        for layer_id in range(NUM_LAYERS):
            (
                (megatron_input_norm, sglang_input_norm),
                (
                    megatron_post_norm,
                    sglang_post_norm,
                ),
            ) = norm_pairs[layer_id]

            megatron_pre_norm = megatron_input_norm(megatron_residual)
            if sglang_residual is None:
                sglang_pre_norm = sglang_input_norm.forward_native(sglang_hidden)
                sglang_residual = sglang_hidden
            else:
                sglang_pre_norm, sglang_residual = sglang_input_norm.forward_native(sglang_hidden, sglang_residual)
            if layer_id == 1:
                megatron_captures["layer_1_pre_norm"] = megatron_pre_norm
                sglang_captures["layer_1_pre_norm"] = sglang_pre_norm

            megatron_attention = _project_attention(megatron_pre_norm, attention_weights[layer_id])
            sglang_attention = _project_attention(sglang_pre_norm, attention_weights[layer_id])
            megatron_residual = _megatron_residual_add(megatron_attention, megatron_residual)
            sglang_pre_mlp, sglang_residual = sglang_post_norm.forward_native(sglang_attention, sglang_residual)
            if layer_id == 0:
                megatron_captures["layer_0_attention_residual"] = megatron_residual
                sglang_captures["layer_0_attention_residual"] = sglang_residual

            megatron_pre_mlp = megatron_post_norm(megatron_residual)
            megatron_mlp = _project_mlp(megatron_pre_mlp, gate_up_weights[layer_id], down_weights[layer_id])
            sglang_hidden = _project_mlp(sglang_pre_mlp, gate_up_weights[layer_id], down_weights[layer_id])
            megatron_residual = _megatron_residual_add(megatron_mlp, megatron_residual)
            sglang_materialized_residual = sglang_hidden.float() + sglang_residual.float()

            if layer_id == 0:
                megatron_captures["layer_0_mlp_residual"] = megatron_residual
                sglang_captures["layer_0_mlp_residual"] = sglang_materialized_residual
            if layer_id == NUM_LAYERS - 1:
                megatron_captures["layer_35_residual"] = megatron_residual
                sglang_captures["layer_35_residual"] = sglang_materialized_residual

        megatron_final = final_megatron_norm(megatron_residual)
        sglang_final, _ = final_sglang_norm.forward_native(sglang_hidden, sglang_residual)
        megatron_captures["final_norm"] = megatron_final
        sglang_captures["final_norm"] = sglang_final
        megatron_captures["logits"] = F.linear(megatron_final.bfloat16(), lm_head_weight)
        sglang_captures["logits"] = F.linear(sglang_final.bfloat16(), lm_head_weight)
    finally:
        disable_batch_invariant_mode()

    _assert_first_divergence_equal(megatron_captures, sglang_captures)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
