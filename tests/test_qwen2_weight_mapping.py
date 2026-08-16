from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf

NUM_GPUS = 0


@pytest.mark.parametrize(
    ("fused_name", "unfused_name", "hf_name"),
    [
        (
            "self_attention.linear_qkv.layer_norm_weight",
            "input_layernorm.weight",
            "model.layers.3.input_layernorm.weight",
        ),
        (
            "mlp.linear_fc1.layer_norm_weight",
            "pre_mlp_layernorm.weight",
            "model.layers.3.post_attention_layernorm.weight",
        ),
    ],
)
def test_qwen2_norm_weight_mapping_accepts_fused_and_unfused_names(
    fused_name, unfused_name, hf_name
):
    args = SimpleNamespace(
        hidden_size=2560,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
    )
    weight = torch.randn(args.hidden_size)
    prefix = "module.module.decoder.layers.3."

    fused = convert_qwen2_to_hf(args, prefix + fused_name, weight)
    unfused = convert_qwen2_to_hf(args, prefix + unfused_name, weight)

    assert fused == [(hf_name, weight)]
    assert unfused == fused


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
