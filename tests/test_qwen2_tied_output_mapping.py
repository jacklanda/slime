import torch

from slime.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf


def test_qwen3_tied_output_uses_canonical_hf_embedding_only():
    args = type("Args", (), {"untie_embeddings_and_output_weights": False})()
    out = convert_qwen2_to_hf(args, "module.module.embedding.word_embeddings.weight", torch.ones(4, 3))
    assert [name for name, _ in out] == ["model.embed_tokens.weight"]


def test_qwen3_untied_output_is_published():
    args = type("Args", (), {"untie_embeddings_and_output_weights": True})()
    out = convert_qwen2_to_hf(args, "module.module.output_layer.weight", torch.ones(4, 3))
    assert out[0][0] == "lm_head.weight"
