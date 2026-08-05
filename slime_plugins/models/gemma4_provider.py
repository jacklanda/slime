"""Custom model provider for Gemma4.

Installs Gemma4-specific behaviors that sit outside the transformer layer:
- embedding scaling (multiply embeddings by sqrt(hidden_size))
- logit softcapping (`final_logit_softcapping`)
- dual-RoPE (different rope_theta + partial-rotary for global vs sliding layers)
- layer_scalar buffers loaded from the HF checkpoint
"""

import json
import logging
import os
from dataclasses import dataclass, field
from types import MethodType

import torch.nn.functional as F
import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from slime_plugins.models.gemma4 import _load_hf_text_config

logger = logging.getLogger(__name__)


@dataclass
class _Gemma4RuntimeState:
    """Per-forward state shared by the GPT model and all cloned layer configs."""

    per_layer_inputs: torch.Tensor | None = None
    shared_kv_states: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.per_layer_inputs = None
        self.shared_kv_states.clear()


def _is_rank_zero() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
    )

    _install_hooks(model, args, config, pre_process, post_process)
    return model


class DualRotaryEmbedding(torch.nn.Module):
    """Wraps a (global, local) pair of RotaryEmbedding modules and emits a
    single concatenated tensor (global part first). ``Gemma4TransformerLayer``
    slices it per-layer based on ``is_sliding``. Concat (not tuple) because
    Megatron's ``SelfAttention.forward`` reads a 2-tuple as
    ``(self_attn, cross_attn)`` RoPE and would misread our pair.
    """

    def __init__(self, local_rope, global_rope, global_dim: int):
        super().__init__()
        self.local_rope = local_rope
        self.global_rope = global_rope
        self.global_dim = global_dim

    def get_rotary_seq_len(self, *args, **kwargs):
        return self.local_rope.get_rotary_seq_len(*args, **kwargs)

    def forward(self, seq_len, **kwargs):
        global_emb = self.global_rope(seq_len, **kwargs)
        local_emb = self.local_rope(seq_len, **kwargs)
        return torch.cat([global_emb, local_emb], dim=-1)


class _Gemma4LogitSoftcap(torch.autograd.Function):
    """Apply Gemma4 final logit softcapping without allocating new logits."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        ctx.mark_dirty(logits)
        logits.div_(scale)
        logits.tanh_()
        logits.mul_(scale)
        ctx.save_for_backward(logits)
        return logits

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (softcapped,) = ctx.saved_tensors
        scale = ctx.scale
        grad_logits = softcapped / scale
        grad_logits.pow_(2)
        grad_logits.neg_()
        grad_logits.add_(1.0)
        grad_logits.mul_(grad_output)
        return grad_logits, None


def _logit_softcapping(logits: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return logits
    return _Gemma4LogitSoftcap.apply(logits, float(scale))


class _Gemma4RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)


def _install_per_layer_input_modules(inner, args, config, hf_text):
    ple_dim = getattr(hf_text, "hidden_size_per_layer_input", 0) or 0
    if not ple_dim or not hasattr(inner, "embedding"):
        return

    from megatron.core import tensor_parallel

    num_layers = hf_text.num_hidden_layers
    packed_dim = num_layers * ple_dim
    vocab_size = getattr(hf_text, "vocab_size_per_layer_input", hf_text.vocab_size)
    inner.embedding.per_layer_embeddings = tensor_parallel.VocabParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=packed_dim,
        init_method=config.embedding_init_method,
        reduce_scatter_embeddings=False,
        config=config,
        tp_group=inner.embedding.tp_group,
    )
    inner.per_layer_model_projection = torch.nn.Linear(config.hidden_size, packed_dim, bias=False)
    inner.per_layer_projection_norm = _Gemma4RMSNorm(ple_dim, eps=config.layernorm_epsilon)

    inner._gemma4_ple_dim = ple_dim
    inner._gemma4_num_layers = num_layers
    inner._gemma4_per_layer_input_scale = 2.0**-0.5
    inner._gemma4_per_layer_model_projection_scale = config.hidden_size**-0.5


def _install_runtime_state(inner) -> _Gemma4RuntimeState:
    """Bind one runtime state to the model, layers, and attention modules.

    Global Gemma4 layers are built with ``dataclasses.replace(config, ...)``.
    Runtime tensors therefore cannot live on the config object: each cloned
    config would see different state. Module attributes preserve the intended
    model ownership and make that sharing explicit.
    """

    state = _Gemma4RuntimeState()
    inner._gemma4_runtime_state = state
    if hasattr(inner, "decoder"):
        for layer in inner.decoder.layers:
            layer._gemma4_runtime_state = state
            if hasattr(layer, "self_attention"):
                layer.self_attention._gemma4_runtime_state = state

    orig_forward = inner.forward

    def _forward_with_gemma4_state(self, *f_args, **f_kwargs):
        self._gemma4_runtime_state.reset()
        return orig_forward(*f_args, **f_kwargs)

    inner.forward = MethodType(_forward_with_gemma4_state, inner)
    return state


def _install_hooks(model, args, config, pre_process, post_process):
    """Install Gemma4-specific pre/post-process hooks on a built GPTModel.

    We use ``register_forward_hook`` rather than subclassing GPTModel
    because:
      - Two independent behaviors (embed scale, softcap) on two different
        submodules. Subclassing would require overriding
        ``GPTModel.forward`` and branching on pp/vp stage.
      - The hooks are shape- and dtype-preserving, so they compose cleanly
        with PP (only first-stage runs embedding, only last-stage runs
        output_layer) - we gate registration on ``pre_process`` /
        ``post_process`` accordingly.
      - Keeps the diff local to this plugin: we don't need to shadow any
        Megatron-maintained class.
    """
    hf_text = _load_hf_text_config(args.hf_checkpoint)
    hidden_size = config.hidden_size

    inner = model.module if hasattr(model, "module") else model
    _install_per_layer_input_modules(inner, args, config, hf_text)
    runtime_state = _install_runtime_state(inner)

    # Embedding scaling - HF applies this inside the embedding module.
    # See ``Gemma4TextScaledWordEmbedding``: the scale is stored as an fp32
    # tensor and cast to the embedding weight's dtype at forward time, so
    # the scale-as-applied depends on the current weight dtype (bf16 during
    # training, fp32 during some eval paths). We match that behavior here.
    if pre_process and hasattr(inner, "embedding"):
        embed_scale = torch.tensor(hidden_size**0.5)  # fp32

        def _embed_hook(module, inp, kwargs, output):
            scaled = output * embed_scale.to(output.dtype)
            if not hasattr(module, "per_layer_embeddings"):
                return scaled

            input_ids = kwargs.get("input_ids")
            if input_ids is None and inp:
                input_ids = inp[0]
            if input_ids is None:
                raise ValueError("Gemma4 per-layer embeddings require input_ids in embedding forward inputs")
            token_ple = module.per_layer_embeddings(input_ids)
            token_ple = token_ple.transpose(0, 1).contiguous()
            if token_ple.shape[0] != scaled.shape[0]:
                from megatron.core import tensor_parallel

                token_ple = tensor_parallel.scatter_to_sequence_parallel_region(
                    token_ple,
                    group=module.tp_group,
                )

            projected = inner.per_layer_model_projection(scaled) * inner._gemma4_per_layer_model_projection_scale
            projected = projected.reshape(
                scaled.shape[0],
                scaled.shape[1],
                inner._gemma4_num_layers,
                inner._gemma4_ple_dim,
            )
            projected = inner.per_layer_projection_norm(projected)

            token_ple = token_ple.reshape(
                scaled.shape[0],
                scaled.shape[1],
                inner._gemma4_num_layers,
                inner._gemma4_ple_dim,
            )
            runtime_state.per_layer_inputs = (projected + token_ple) * inner._gemma4_per_layer_input_scale
            return scaled

        inner.embedding.register_forward_hook(_embed_hook, with_kwargs=True)

    # Final logit softcapping - HF applies tanh(logits / cap) * cap.
    # Some Megatron output_layer variants (parallel_output paths) return
    # ``(logits, bias)``; we pass the non-logit tail through unchanged.
    softcap = getattr(hf_text, "final_logit_softcapping", None)
    if post_process and softcap and hasattr(inner, "output_layer"):

        def _softcap_hook(module, inp, output):
            if isinstance(output, tuple):
                return (_logit_softcapping(output[0], softcap),) + output[1:]
            return _logit_softcapping(output, softcap)

        inner.output_layer.register_forward_hook(_softcap_hook)

    # Dual RoPE: replace Megatron's single rotary_pos_emb with a wrapper that
    # produces (global, local) RoPE side-by-side. Gemma4 uses partial-rotary
    # on global layers (implemented here by zeroing the tail of inv_freq).
    if hasattr(inner, "rotary_pos_emb"):
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        rope_params = getattr(hf_text, "rope_parameters", {}) or {}
        full = rope_params.get("full_attention", {}) or {}
        sliding = rope_params.get("sliding_attention", {}) or {}
        global_theta = full.get("rope_theta", 1_000_000.0)
        local_theta = sliding.get("rope_theta", 10_000.0)
        global_head_dim = hf_text.global_head_dim
        global_partial = full.get("partial_rotary_factor", 0.25)

        local_rope = inner.rotary_pos_emb  # already built with args.rotary_base

        global_rope = RotaryEmbedding(
            kv_channels=global_head_dim,
            rotary_percent=1.0,
            rotary_base=global_theta,
        )
        # HF "proportional" RoPE: first (partial * head_dim // 2) inv_freq
        # entries are live, the rest are zero (no rotation on those dims).
        # Writing this to the existing buffer keeps device/dtype correct.
        rope_angles = int(global_partial * global_head_dim // 2)
        half = global_head_dim // 2
        # Guard the RoPE geometry: 0 means "no rotation" (nonsensical here);
        # > half would produce nope<0 and a shape-mismatched copy_. Both
        # should fail loudly rather than silently writing garbage.
        assert 0 < rope_angles <= half, (
            f"global_partial_rotary_factor={global_partial} with "
            f"global_head_dim={global_head_dim} produced rope_angles="
            f"{rope_angles}; must be in (0, {half}]."
        )
        inv_freq_live = 1.0 / (
            global_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / global_head_dim)
        )
        nope = half - rope_angles
        inv_freq = torch.cat([inv_freq_live, torch.zeros(nope)]) if nope > 0 else inv_freq_live
        assert inv_freq.shape == global_rope.inv_freq.shape, (
            f"inv_freq shape {tuple(inv_freq.shape)} doesn't match "
            f"global_rope.inv_freq shape {tuple(global_rope.inv_freq.shape)}; "
            "Megatron RotaryEmbedding layout may have changed."
        )
        global_rope.inv_freq.copy_(inv_freq.to(global_rope.inv_freq.device))

        inner.rotary_pos_emb = DualRotaryEmbedding(local_rope, global_rope, global_head_dim)
        config.dual_rope_global_dim = global_head_dim
        if _is_rank_zero():
            logger.info(
                "DualRotaryEmbedding: local_theta=%s global_theta=%s " "global_dim=%s rope_angles=%d (nope=%d)",
                local_theta,
                global_theta,
                global_head_dim,
                rope_angles,
                nope,
            )

    if hasattr(inner, "decoder") and args.hf_checkpoint:
        _load_layer_scalars(inner, args.hf_checkpoint, config)


def _read_layer_scalars_from_safetensors(hf_checkpoint: str) -> dict[int, float] | None:
    """Read all ``layer_scalar`` values from the HF safetensors checkpoint.

    Returns ``{global_layer_idx: scalar}`` or ``None`` if the checkpoint has
    no safetensors index (older HF layouts) or no layer_scalar weights. Only
    called on rank 0 - results are broadcast to the other ranks.
    """
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        single_path = os.path.join(hf_checkpoint, "model.safetensors")
        if not os.path.exists(single_path):
            logger.warning("No safetensors index at %s; skipping layer scalars", index_path)
            return None

        from safetensors import safe_open

        scalars: dict[int, float] = {}
        with safe_open(single_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if "layer_scalar" not in key:
                    continue
                layer_idx = int(key.split(".layers.")[1].split(".")[0])
                scalars[layer_idx] = sf.get_tensor(key).item()
        if not scalars:
            logger.warning("No layer_scalar weights found in checkpoint %s", hf_checkpoint)
            return None
        return scalars

    from safetensors import safe_open

    with open(index_path) as f:
        index = json.load(f)

    scalars: dict[int, float] = {}
    for key, filename in index["weight_map"].items():
        if "layer_scalar" not in key:
            continue
        layer_idx = int(key.split(".layers.")[1].split(".")[0])
        with safe_open(os.path.join(hf_checkpoint, filename), framework="pt", device="cpu") as sf:
            scalars[layer_idx] = sf.get_tensor(key).item()

    if not scalars:
        logger.warning("No layer_scalar weights found in checkpoint %s", hf_checkpoint)
        return None
    return scalars


def _broadcast_layer_scalars(scalars: dict[int, float] | None) -> dict[int, float] | None:
    """Broadcast the rank-0-read ``scalars`` dict to every rank.

    safetensors reads on every rank cause an O(world_size) fan-out of tiny
    reads on the shared filesystem; the dict itself is a few kilobytes. If
    ``torch.distributed`` isn't initialized (single-process run), we simply
    return the input dict.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return scalars
    obj = [scalars] if torch.distributed.get_rank() == 0 else [None]
    torch.distributed.broadcast_object_list(obj, src=0)
    return obj[0]


def _load_layer_scalars(inner, hf_checkpoint, config):
    # Wrong layer_scalars materially change activations vs HF (they're per-
    # layer multiplicative gains on the residual stream, not decorative), so
    # by default we fail hard if the load breaks. Set
    # GEMMA4_ALLOW_MISSING_LAYER_SCALARS=1 to downgrade to a warning and
    # train with the default value of 1.0 - only useful for debug runs
    # against a checkpoint that genuinely lacks these buffers.
    allow_missing = os.environ.get("GEMMA4_ALLOW_MISSING_LAYER_SCALARS") == "1"
    try:
        scalars = _read_layer_scalars_from_safetensors(hf_checkpoint) if _is_rank_zero() else None
        scalars = _broadcast_layer_scalars(scalars)
        if not scalars:
            if allow_missing:
                return
            raise RuntimeError(
                "No layer_scalar weights found in checkpoint; set "
                "GEMMA4_ALLOW_MISSING_LAYER_SCALARS=1 to proceed with "
                "default values (not numerically equivalent to HF)."
            )

        # Under pipeline-parallelism, inner.decoder.layers holds only this
        # rank's local subset. Translate the local index back to the global
        # (HF 0-indexed) layer index so we apply the right scalar per layer.
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        pp_offset = get_transformer_layer_offset(config)

        loaded = 0
        for i, layer in enumerate(inner.decoder.layers):
            if hasattr(layer, "layer_scalar"):
                global_idx = i + pp_offset
                if global_idx not in scalars:
                    if allow_missing:
                        logger.warning(
                            "layer_scalar for global layer %d missing; using default 1.0",
                            global_idx,
                        )
                    else:
                        raise KeyError(
                            f"layer_scalar for global layer {global_idx} "
                            f"missing in checkpoint (have: {sorted(scalars)[:10]}...); "
                            "checkpoint may be truncated."
                        )
                layer.layer_scalar.fill_(scalars.get(global_idx, 1.0))
                loaded += 1
        if _is_rank_zero():
            logger.info(
                "Applied %d/%d layer scalars (pp_offset=%d, range=%.4f..%.4f)",
                loaded,
                len(inner.decoder.layers),
                pp_offset,
                min(scalars.values()),
                max(scalars.values()),
            )
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if allow_missing:
            logger.warning("layer scalars unavailable (%s: %s); using default 1.0", type(e).__name__, e)
            return
        raise
