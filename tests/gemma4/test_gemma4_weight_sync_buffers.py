from argparse import Namespace

import torch

from slime.backends.megatron_utils.update_weight import common


class Gemma4TransformerConfig:
    pass


class Qwen3TransformerConfig:
    pass


def _model_with_layer_scalars(config, num_layers: int):
    model = torch.nn.Module()
    model.config = config
    model.module = torch.nn.Module()
    model.module.decoder = torch.nn.Module()
    layers = []
    for layer_idx in range(num_layers):
        layer = torch.nn.Module()
        layer.register_buffer("layer_scalar", torch.tensor([float(layer_idx)]))
        layer.register_buffer("rotary_cache", torch.tensor([float(layer_idx)]))
        layers.append(layer)
    model.module.decoder.layers = torch.nn.ModuleList(layers)
    return model


def _patch_parallel_state(monkeypatch, *, layer_offset: int = 0):
    monkeypatch.setattr(common.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common.mpu, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(common, "get_transformer_layer_offset", lambda _config: layer_offset)


def test_gemma4_global_iterator_emits_all_layer_scalars(monkeypatch):
    _patch_parallel_state(monkeypatch)
    model = _model_with_layer_scalars(Gemma4TransformerConfig(), num_layers=42)

    emitted = dict(common._named_params_and_buffers_global(Namespace(num_experts=None), [model]))

    scalar_names = [name for name in emitted if name.endswith(".layer_scalar")]
    assert scalar_names == [
        f"module.module.decoder.layers.{layer_idx}.layer_scalar" for layer_idx in range(42)
    ]
    assert not any("rotary_cache" in name for name in emitted)


def test_gemma4_layer_scalar_uses_global_pipeline_layer_index(monkeypatch):
    _patch_parallel_state(monkeypatch, layer_offset=17)
    model = _model_with_layer_scalars(Gemma4TransformerConfig(), num_layers=2)

    emitted = dict(common._named_params_and_buffers_global(Namespace(num_experts=None), [model]))

    assert list(emitted) == [
        "module.module.decoder.layers.17.layer_scalar",
        "module.module.decoder.layers.18.layer_scalar",
    ]


def test_non_gemma_layer_scalar_is_not_emitted(monkeypatch):
    _patch_parallel_state(monkeypatch)
    model = _model_with_layer_scalars(Qwen3TransformerConfig(), num_layers=2)

    emitted = dict(common._named_params_and_buffers_global(Namespace(num_experts=None), [model]))

    assert not any(name.endswith(".layer_scalar") for name in emitted)
    assert not any("rotary_cache" in name for name in emitted)


def test_gemma4_layer_scalar_needs_no_tensor_parallel_attributes():
    scalar = torch.tensor([0.75])

    assert common.all_gather_param("module.module.decoder.layers.3.layer_scalar", scalar) is scalar
