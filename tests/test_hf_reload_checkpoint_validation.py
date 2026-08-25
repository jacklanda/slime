import json

import pytest
import torch
from safetensors.torch import save_file

from slime.backends.megatron_utils.hf_checkpoint_saver import _validate_hf_checkpoint_for_reload


def _write_checkpoint(path, *, finite=True):
    tensors = {
        "model.embed_tokens.weight": torch.ones(4, 3),
        "lm_head.weight": torch.ones(4, 3),
        "model.norm.weight": torch.ones(3),
    }
    if not finite:
        tensors["lm_head.weight"][0, 0] = float("nan")
    save_file(tensors, path / "model-00001.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001.safetensors" for name in tensors}}),
        encoding="utf-8",
    )


def test_validate_hf_reload_checkpoint(tmp_path):
    _write_checkpoint(tmp_path)
    _validate_hf_checkpoint_for_reload(
        type("Args", (), {"vocab_size": 4, "hidden_size": 3})(), tmp_path
    )


def test_validate_hf_reload_checkpoint_rejects_non_finite(tmp_path):
    _write_checkpoint(tmp_path, finite=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        _validate_hf_checkpoint_for_reload(
            type("Args", (), {"vocab_size": 4, "hidden_size": 3})(), tmp_path
        )


def test_validate_hf_reload_checkpoint_allows_architecture_specific_names(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    tensors = {"language_model.embed_tokens.weight": torch.ones(4, 3)}
    save_file(tensors, tmp_path / "model-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {next(iter(tensors)): "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    _validate_hf_checkpoint_for_reload(
        type("Args", (), {"vocab_size": 4, "hidden_size": 3, "hf_checkpoint": str(tmp_path)})(), tmp_path
    )


def test_validate_tied_qwen_checkpoint_does_not_require_lm_head(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "tie_word_embeddings": True}), encoding="utf-8"
    )
    tensors = {
        "model.embed_tokens.weight": torch.ones(4, 3),
        "model.norm.weight": torch.ones(3),
    }
    save_file(tensors, tmp_path / "model-00001.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001.safetensors" for name in tensors}}), encoding="utf-8"
    )
    _validate_hf_checkpoint_for_reload(
        type("Args", (), {"vocab_size": 4, "hidden_size": 3, "hf_checkpoint": str(tmp_path)})(), tmp_path
    )
