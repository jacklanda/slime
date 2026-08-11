"""Native Megatron Gemma4 transformer layer and config.

Extends the Gemma3 implementation from mbridge with Gemma4-specific features:
- Heterogeneous attention: global layers use head_dim=512, num_kv_heads=4;
  sliding layers use head_dim=256, num_kv_heads=16.
- attention_k_eq_v: global layers reuse K output as V (no v_proj).
- v_norm: RMSNorm without learnable scale applied to V states.
- layer_scalar: buffer multiplied after residual (not learned).
- final_logit_softcapping: applied to output logits in the model wrapper.
- MoE block (26B-A4B): Gemma4's custom router (with per-expert scale) plugged
  into Megatron's MoE infrastructure for proper expert-parallel sharding.
  The router is still custom (see Gemma4Router); dispatching + grouped-GEMM
  come from Megatron's MoELayer + TEGroupedMLP.
"""

import copy
import functools
import json
import logging
import os
from dataclasses import dataclass
from dataclasses import replace as dc_replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import make_viewless_tensor

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from mbridge.models.gemma3.transformer_config import Gemma3TransformerConfig
from slime_plugins.models.gemma4_activation import Gemma4GeluAndMul

# Gemma uses GeGLU, not SwiGLU.
_gelu_tanh = functools.partial(F.gelu, approximate="tanh")


@dataclass
class Gemma4TransformerConfig(Gemma3TransformerConfig):
    """Gemma4-specific config extending Gemma3."""

    global_kv_channels: int = 512
    global_num_query_groups: int = 4
    global_partial_rotary_factor: float = 0.25  # fraction of global head_dim that gets RoPE
    # Width of the global half of the concatenated DualRotaryEmbedding output.
    # Declared as a real field (not set via setattr on the shared config) so it
    # survives the `dc_replace` clone that global layers are built against -
    # otherwise global layers cannot find it and never slice the RoPE.
    dual_rope_global_dim: int = 0
    attention_k_eq_v: bool = True  # global layers: V = K (no v_proj)
    enable_moe_block: bool = False  # 26B-A4B MoE variant
    hidden_size_per_layer_input: int = 0  # E2B/E4B per-layer embeddings
    num_kv_shared_layers: int = 0  # E2B/E4B KV reuse in final layers
    use_double_wide_mlp: bool = False  # E2B/E4B wide MLP in KV-shared layers


class VNorm(nn.Module):
    """RMSNorm without learnable scale, matching Gemma4's v_norm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if os.environ.get("SLIME_GEMMA4_BATCH_INVARIANT") == "1" and x.is_cuda:
            from megatron.core.transformer.custom_layers.batch_invariant_kernels import BatchInvariantRMSNormFn

            return BatchInvariantRMSNormFn.apply(x, self.weight, self.eps, False)
        dtype = x.dtype
        x = x.float()
        return (x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)).to(dtype)


@dataclass
class Gemma4TransformerLayerSubmodules(TransformerLayerSubmodules):
    post_attention_layernorm: ModuleSpec | type = IdentityOp
    post_feedforward_layernorm: ModuleSpec | type = IdentityOp
    post_per_layer_input_norm: ModuleSpec | type = IdentityOp
    # For MoE-enabled variants (26B-A4B), the primary `mlp` submodule is swapped
    # to a Gemma4MoELayer and the original dense MLP moves to `dense_mlp`. This
    # keeps the `.mlp.experts.linear_fc...` naming that mbridge's EP auto-handling
    # expects while preserving Gemma4's dense+MoE-in-parallel structure.
    dense_mlp: ModuleSpec | type = IdentityOp


def _residual_add_rmsnorm_like_sglang(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match SGLang Gemma4's residual-add + pre-FFN RMSNorm ordering."""
    output_dtype = hidden_states.dtype
    residual_fp32 = hidden_states.float() + residual.float()
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
    if eps is None:
        raise RuntimeError(f"Gemma4 RMSNorm module {type(norm).__name__} has no epsilon attribute")
    weight = norm.weight.float()
    if getattr(norm, "zero_centered_gamma", False):
        weight = weight + 1.0
    variance = residual_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = residual_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight).to(output_dtype), residual_fp32.to(output_dtype)


class Gemma4Router(nn.Module):
    """Gemma4 MoE router.

    The router equation (mirroring HF ``Gemma4TextTopkRouter``) is:

        h_norm   = RMSNorm_no_scale(h)              # VNorm: no learnable scale
        h_scaled = h_norm * scale / sqrt(H)         # learnable per-hidden scale
        logits   = proj(h_scaled)                   # [T, E]
        probs    = softmax(logits, dim=-1)
        top_w, top_i = topk(probs, k=top_k)
        top_w    = top_w / top_w.sum(dim=-1, keepdim=True)   # renormalize
        top_w    = top_w * per_expert_scale[top_i]           # per-expert scale

    The renormalise-then-scale order is load-bearing and must match HF: it
    produces ``top_w.sum() == per_expert_scale.mean_over_selected`` rather
    than a renormalised-back-to-1 distribution. Reversing the order (scale
    first, then renormalise) would cancel ``per_expert_scale``.
    ``test_router_matches_hf_reference_equation`` guards this.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_moe_experts
        self.top_k = config.moe_router_topk
        self.scalar_root_size = self.hidden_size**-0.5
        self.norm = VNorm(self.hidden_size, eps=config.layernorm_epsilon)
        self.proj = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(self.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(self.num_experts))

    def forward(self, hidden_states):
        h = self.norm(hidden_states)
        h = h * self.scale * self.scalar_root_size
        logits = self.proj(h)
        probs = torch.softmax(logits, dim=-1)
        top_k_weights, top_k_index = torch.topk(probs, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return top_k_weights, top_k_index

    def set_layer_number(self, layer_number):
        pass


class Gemma4MoELayer(MoELayer):
    """Gemma4 MoE block: Megatron's MoELayer with Gemma4's custom router.

    Megatron's MoELayer hardcodes its own ``TopKRouter`` which uses a
    softmax-with-expert-bias scheme. Gemma4 has its own router semantics
    (no-scale RMSNorm -> learnable per-hidden scale -> proj -> softmax -> topk ->
    per-expert scale multiplier). We reuse all of Megatron's infrastructure
    for dispatching (alltoall), expert parallelism, and grouped-GEMM expert
    computation - but swap in our ``Gemma4Router`` and convert its compact
    (top_k_weights [T, K], top_k_index [T, K]) output into Megatron's
    expected (probs [T, E], routing_map [T, E]) format inside ``route()``.
    """

    def __init__(self, config, submodules=None, layer_number=None, pg_collection=None):
        # Fall back to Megatron's global parallel_state when pg_collection isn't
        # explicitly passed. TransformerLayer only forwards pg_collection when
        # submodules.mlp.module is *exactly* one of
        # (MoELayer, GroupedMLP, TEGroupedMLP, SequentialMLP) - an identity check
        # via `in`, so Gemma4MoELayer (a MoELayer subclass) slips through and
        # receives None. BaseMoELayer.__init__ then crashes on `pg_collection.ep`.
        # Same fallback MoELayer.__init__ uses when invoked directly.
        if pg_collection is None:
            from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

            pg_collection = get_default_pg_collection()
        BaseMoELayer.__init__(self, config=config, layer_number=layer_number, pg_collection=pg_collection)
        self.moe_layer_recompute = False
        self.shared_experts_recompute = False
        self.submodules = submodules

        self.router = Gemma4Router(config)

        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAllGatherTokenDispatcher,
            MoEAlltoAllTokenDispatcher,
            MoEFlexTokenDispatcher,
        )

        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")

        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        self.shared_experts = None

        from megatron.core.transformer.moe.moe_utils import MoECudaGraphTensorStore

        self.cudagraph_tensor_store = MoECudaGraphTensorStore()

        # pre_feedforward_layernorm_2: applied to experts' input ONLY (router
        # input stays un-normed). Matches HF Gemma4TextDecoderLayer:
        #   hidden_states_flat = residual            # router input (un-normed)
        #   hidden_states_2 = pre_feedforward_layernorm_2(hidden_states_flat)
        #   hidden_states_2 = experts(hidden_states_2, top_k_index, top_k_weights)
        self.pre_feedforward_layernorm_2 = TENorm(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon,
        )

    def route(self, hidden_states: torch.Tensor):
        """Call ``Gemma4Router`` and pack its output into Megatron's
        ``(probs, routing_map)`` format.

        ``Gemma4Router`` emits compact top-k tensors:
            top_k_weights: [T, K] - routing weights (already scaled by per_expert_scale)
            top_k_index:   [T, K] - which experts each token routes to
        Megatron's dispatcher wants:
            probs:       [T, E] - weight per (token, expert), 0 where not routed
            routing_map: [T, E] - boolean mask
        """
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        top_k_weights, top_k_index = self.router(flat)

        num_tokens = flat.shape[0]
        num_experts = self.config.num_moe_experts
        probs = torch.zeros(
            num_tokens,
            num_experts,
            dtype=top_k_weights.dtype,
            device=top_k_weights.device,
        )
        probs.scatter_(1, top_k_index, top_k_weights)
        routing_map = probs != 0
        return probs, routing_map

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_input: torch.Tensor | None = None,
    ):
        """Gemma4 MoE forward with split router / experts inputs.

        HF's ``Gemma4TextDecoderLayer`` routes based on the *un-normed* residual
        but feeds the experts the *pre-ff-norm-2'd* residual:

            hidden_states_flat = residual                       # un-normed
            _, tk_w, tk_i = self.router(hidden_states_flat)
            experts_input = self.pre_feedforward_layernorm_2(hidden_states_flat)
            output        = self.experts(experts_input, tk_i, tk_w)

        We take the un-normed residual in ``hidden_states`` and apply
        ``pre_feedforward_layernorm_2`` internally to obtain the experts
        input. The router path uses the un-normed residual directly. Callers
        may pass a different ``router_input`` for tests or ablations; when
        ``router_input is None`` (the normal case) the router sees the same
        un-normed residual the layer was called with.

        We inline the Megatron parent's ``forward`` body here - rather than
        calling ``super().forward`` with a side-channel stash - so the
        router input is passed explicitly end-to-end and the code is safe
        under activation checkpointing / recomputation.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError("During training, performance may degrade if MoE and tensor " "parallelism are enabled without also enabling sequence parallelism.")

        router_in = router_input if router_input is not None else hidden_states
        experts_in = self.pre_feedforward_layernorm_2(hidden_states)

        def custom_forward(experts_in, router_in):
            # Gemma4 has no shared experts; shared_experts_compute returns None.
            shared_expert_output = self.shared_experts_compute(experts_in)
            probs, routing_map = self.route(router_in)
            experts_in2, probs = self.preprocess(experts_in, probs, routing_map)
            dispatched_input, probs = self.dispatch(experts_in2, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
            output = self.combine(output)
            output = self.postprocess(output, shared_expert_output)
            return output, mlp_bias

        # moe_layer_recompute is forced to False in __init__; call directly.
        return custom_forward(experts_in, router_in)


class Gemma4TransformerLayer(TransformerLayer):
    """Gemma4 transformer layer with heterogeneous attention and layer_scalar."""

    def __init__(
        self,
        config: Gemma4TransformerConfig,
        submodules: Gemma4TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        global_layer_number = layer_number + get_transformer_layer_offset(config)
        # Megatron passes `layer_number` as 1-indexed (default 1), so in 0-indexed
        # HF space a global layer is `(i+1) % pattern == 0` -> `i % pattern == pattern-1`.
        # Equivalently: `is_sliding` when `global_layer_number % pattern != 0`.
        self.is_sliding = bool(global_layer_number % config.sliding_window_pattern)
        self._is_global = not self.is_sliding

        # Global layers have different head_dim (kv_channels) and num_kv_heads
        # (num_query_groups). Build the layer against a *cloned* config with
        # those overrides so we never mutate the shared transformer config.
        # Mutation would be reentrant-unsafe under concurrent layer
        # construction and leak global-layer shapes into sibling sliding
        # layers if an exception were raised during super().__init__.
        layer_config = (
            dc_replace(
                config,
                kv_channels=config.global_kv_channels,
                num_query_groups=config.global_num_query_groups,
            )
            if self._is_global
            else config
        )
        super().__init__(
            config=layer_config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )

        # Do not silently train a Gemma4 layer without its pre-FFN RMSNorm.
        # This guard is Gemma4-only and catches callers that provide a stale
        # custom ModuleSpec instead of the canonical spec above.
        if isinstance(self.pre_mlp_layernorm, IdentityOp):
            raise RuntimeError("Gemma4 requires an explicit pre_mlp_layernorm; received IdentityOp. " "Use slime_plugins.models.gemma4:get_gemma4_spec.")

        self.self_attention._is_global = self._is_global
        self.self_attention._global_layer_idx = global_layer_number - 1
        self.self_attention._kv_shared_layer_index = getattr(config, "kv_shared_layer_map", {}).get(global_layer_number - 1)
        self.self_attention._store_full_length_kv = (global_layer_number - 1) in getattr(config, "kv_store_layers", set())

        # Global layers require this because head_dim=512 exceeds flash attention's limit (256).
        # Local layers also use SDPA for consistency.
        self.self_attention.core_attention = SDPACoreAttention(
            config=config,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.causal,
            softmax_scale=config.softmax_scale,
        )
        self.self_attention.core_attention._is_sliding = self.is_sliding

        self.post_attention_layernorm = build_module(
            submodules.post_attention_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm = build_module(
            submodules.post_feedforward_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # The released Gemma4 checkpoints store layer_scalar in BF16, and
        # SGLang constructs this buffer under its BF16 model-init dtype.  A
        # float32 shape-[1] scalar would promote `hidden_states * scalar` to
        # float32, changing every later layer and doubling residual activations.
        self.register_buffer("layer_scalar", torch.ones(1, dtype=config.params_dtype))

        self.global_layer_idx = global_layer_number - 1
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = torch.nn.Linear(
                config.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection = torch.nn.Linear(
                self.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
            )
            self.post_per_layer_input_norm = build_module(
                submodules.post_per_layer_input_norm,
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )

        self.is_kv_shared_layer = self.global_layer_idx in getattr(config, "kv_shared_layer_map", {})
        if getattr(config, "use_double_wide_mlp", False) and self.is_kv_shared_layer and not getattr(config, "enable_moe_block", False):
            wide_config = dc_replace(config, ffn_hidden_size=config.ffn_hidden_size * 2)
            self.mlp = build_module(
                submodules.mlp,
                config=wide_config,
                tp_group=self.tp_group,
            )

        # MoE block (26B-A4B): super().__init__ already built self.mlp from the
        # layer spec, which when enable_moe_block=True is a Gemma4MoELayer (not
        # a dense MLP). We also build a parallel `dense_mlp` for Gemma4's
        # dense + MoE combined-FFN pattern. The two outputs are summed in
        # forward().
        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.dense_mlp = build_module(
                submodules.dense_mlp,
                config=config,
            )
            self.post_feedforward_layernorm_1 = TENorm(
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )
            # pre_feedforward_layernorm_2 now lives INSIDE Gemma4MoELayer
            # (matching HF Gemma4TextDecoderLayer semantics: router sees un-normed
            # residual, experts see pre_feedforward_layernorm_2(residual)). This
            # attribute is kept on the MoE block so mbridge/state-dict paths
            # don't change.
            self.post_feedforward_layernorm_2 = TENorm(
                config=config,
                hidden_size=config.hidden_size,
                eps=config.layernorm_epsilon,
            )

        # Keep the spec as Megatron's native MLP so TP-group plumbing and
        # checkpoint keys remain unchanged. Only the selected Gemma-4 dense
        # branch receives a module-local activation dispatch override; the
        # shared config and MoE experts retain their original behavior.
        dense_mlp = self.dense_mlp if self.enable_moe_block else self.mlp
        dense_mlp.config = copy.copy(dense_mlp.config)
        dense_mlp.config.use_te_activation_func = True
        dense_mlp.activation_func = Gemma4GeluAndMul()

    def _forward_dense_ffn(self, pre_mlp_ln):
        """Run the dense MLP. ``self.mlp`` is the dense MLP directly for the
        31B variant."""
        out, bias = self.mlp(pre_mlp_ln)
        return out + bias if bias is not None else out

    def _forward_moe_ffn(self, residual, pre_mlp_ln):
        """Run dense + MoE in parallel and sum (26B-A4B variant).

        Mirrors HF ``Gemma4TextDecoderLayer.forward`` (transformers
        modeling_gemma4.py:1376-1391): dense branch goes through
        ``post_feedforward_layernorm_1``, MoE branch through
        ``post_feedforward_layernorm_2``, the two are summed, and the outer
        ``Gemma4TransformerLayer.forward`` applies ``post_feedforward_layernorm``
        to the sum - 3 post-FFN LNs total for MoE layers is correct.

        HF routes on the un-normed residual but feeds experts the
        ``pre_feedforward_layernorm_2``'d residual; Gemma4MoELayer applies
        that norm internally, so we pass the un-normed residual directly.
        """
        dense_out, dense_bias = self.dense_mlp(pre_mlp_ln)
        if dense_bias is not None:
            dense_out = dense_out + dense_bias
        mlp_output = self.post_feedforward_layernorm_1(dense_out)

        moe_output, _ = self.mlp(residual)
        moe_output = self.post_feedforward_layernorm_2(moe_output)

        return mlp_output + moe_output

    def _forward_per_layer_input(self, hidden_states, per_layer_inputs):
        if per_layer_inputs is None:
            raise RuntimeError("Gemma4 per-layer inputs are missing. E2B/E4B PLE currently requires " "the Gemma4 provider hooks installed by --custom-model-provider-path.")

        residual = hidden_states
        per_layer_input = per_layer_inputs[..., self.global_layer_idx, :]
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.config.activation_func(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            global_dim = getattr(self.config, "dual_rope_global_dim", 0)
            if global_dim > 0 and rotary_pos_emb.shape[-1] > global_dim:
                if self.is_sliding:
                    rotary_pos_emb = rotary_pos_emb[..., global_dim:]
                else:
                    rotary_pos_emb = rotary_pos_emb[..., :global_dim]
        elif isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[1] if self.is_sliding else rotary_pos_emb[0]
        if isinstance(attention_mask, tuple):
            attention_mask = attention_mask[1] if self.is_sliding else attention_mask[0]

        # Global layers use partial RoPE: only `partial_rotary_factor` of the
        # 512-wide head is rotated. That is expressed by ZERO-PADDING inv_freq
        # (see the provider), NOT by narrowing the tensor - the width handed to
        # Megatron must stay the full `global_kv_channels`, because
        # `_apply_rotary_pos_emb_bshd` slices `t[..., :rot_dim]` and pairs dim i
        # with dim i + rot_dim/2. Narrowing to `global_head_dim * partial` (128)
        # would pair dim i with i+64 instead of the correct i+256, rotating the
        # wrong subspace on every full-attention layer.
        if not self.is_sliding and rotary_pos_emb is not None:
            expected = self.config.global_kv_channels
            if rotary_pos_emb.shape[-1] != expected:
                raise RuntimeError(f"Gemma4 global layer expected a {expected}-wide rotary embedding " f"(partial rotation is encoded as zeroed inv_freq tails), got " f"{rotary_pos_emb.shape[-1]}. Check that `dual_rope_global_dim` is set " "on the config used to build this layer.")

        residual = hidden_states

        extra_kwargs = {}
        if inference_context is not None:
            extra_kwargs["inference_context"] = inference_context
        elif inference_params is not None:
            extra_kwargs["inference_params"] = inference_params

        input_layernorm_output = self.input_layernorm(hidden_states)

        hidden_states, hidden_states_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **extra_kwargs,
        )

        if hidden_states_bias is not None:
            hidden_states = hidden_states + hidden_states_bias
        hidden_states = self.post_attention_layernorm(hidden_states)
        pre_mlp_layernorm_output, residual = _residual_add_rmsnorm_like_sglang(
            hidden_states,
            residual,
            self.pre_mlp_layernorm,
        )
        if self.enable_moe_block:
            hidden_states = self._forward_moe_ffn(residual, pre_mlp_layernorm_output)
        else:
            hidden_states = self._forward_dense_ffn(pre_mlp_layernorm_output)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            hidden_states = self._forward_per_layer_input(hidden_states, context)

        hidden_states = hidden_states * self.layer_scalar

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )

        if self.config.external_cuda_graph and self.training:
            return output
        return output, context


class SDPACoreAttention(nn.Module):
    """Gemma4 core attention.

    Replaces TE's DotProductAttention because:
    - Global layers have head_dim=512, which flash-attn 2.x doesn't support.
    - Sliding-window layers need an explicit left-window mask (HF behavior).
    - Context-parallelism on the global layers needs an all-gather+full-attn
      path with a differentiable K/V gather.

    Dispatch at call time (packed / thd shape):
      - CP > 1 (any layer) : all-gather K/V, apply causal + optional
        sliding-window mask computed from slime zig-zag global indices.
      - global  + CP == 1  : sub-sequence causal SDPA (no O(T^2) mask alloc).
      - sliding + CP == 1  : flash_attn_varlen_func with (sw-1, 0) window.
    """

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type="self",
        attention_dropout=None,
        softmax_scale=None,
        **kwargs,
    ):
        super().__init__()
        # Megatron's SelfAttention.__init__ passes a few kwargs (e.g. cp_comm_type,
        # model_comm_pgs) intended for TE's DotProductAttention. We accept-and-ignore
        # by name rather than asserting empty; a strict assert breaks whenever
        # Megatron/TE add a new kwarg. If a kwarg shows up here that we *should*
        # honor (e.g. a new softmax dtype), it will surface as a behavioral bug
        # in parity, which is what the test suite covers.
        del kwargs
        self.config = config
        self.softmax_scale = softmax_scale
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        self._is_sliding = False  # set by Gemma4TransformerLayer

    def _resolve_scale(self, hn: int) -> float:
        return self.softmax_scale if self.softmax_scale is not None else (hn**-0.5)

    @staticmethod
    def _zigzag_global_indices(local_len, cp_rank, cp_size, device):
        """Global positions of this rank's local Q tokens under slime's
        zig-zag CP layout (matches cp_utils.slice_with_cp).

        Local tokens on rank r occupy two global sub-ranges:
          [r*cs, (r+1)*cs) and [(2*cp-r-1)*cs, (2*cp-r)*cs)
        where cs = local_len / 2 = seq_len / (2*cp_size).
        """
        cs = local_len // 2
        first = torch.arange(cp_rank * cs, (cp_rank + 1) * cs, device=device)
        second = torch.arange(
            (2 * cp_size - cp_rank - 1) * cs,
            (2 * cp_size - cp_rank) * cs,
            device=device,
        )
        return torch.cat([first, second])

    @staticmethod
    def _cp_unzigzag_permutation(cu_seqlens_list, cp_size, device):
        """Map rank-major CP-gathered K/V tokens back to packed global order."""
        total_local_len = sum((cu_seqlens_list[i + 1] - cu_seqlens_list[i]) // cp_size for i in range(len(cu_seqlens_list) - 1))
        local_prefix = 0
        perm_parts = []
        for s_idx in range(len(cu_seqlens_list) - 1):
            seq_len_global = cu_seqlens_list[s_idx + 1] - cu_seqlens_list[s_idx]
            cs = seq_len_global // (2 * cp_size)
            g = torch.arange(seq_len_global, device=device)
            chunk = g // cs
            owner = torch.where(chunk < cp_size, chunk, 2 * cp_size - 1 - chunk)
            local_in_rank = torch.where(
                chunk < cp_size,
                g - owner * cs,
                cs + (g - (2 * cp_size - 1 - owner) * cs),
            )
            perm_parts.append(owner * total_local_len + local_prefix + local_in_rank)
            local_prefix += seq_len_global // cp_size
        return torch.cat(perm_parts)

    def _forward_cp_subseq_mask(self, query, key, value, packed_seq_params, sliding_window=None):
        """CP>1 path for any layer: all-gather K/V, then loop over sub-seqs
        and apply a per-sub-seq attention mask built from zig-zag global
        positions. Supports causal-only (global layers) and causal +
        sliding-window (sliding layers).

        Under slime's CP convention, ``packed_seq_params.cu_seqlens_q`` holds
        GLOBAL boundaries: each packed sub-sequence on this rank represents
        ``(cu[i+1] - cu[i])`` tokens globally but only ``(cu[i+1] - cu[i]) //
        cp_size`` tokens locally (the zig-zag slice of this rank's two
        chunks, concatenated as [first, second]).
        """
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        cp_group = parallel_state.get_context_parallel_group()
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        t_local = query.shape[0]
        np_q, hn = query.shape[1], query.shape[2]
        scale = self._resolve_scale(hn)

        # Differentiable all-gather along the token dim. forward: AG,
        # backward: RS - so K/V grads on non-owning ranks flow back to the
        # originating rank. The raw `dist.all_gather_into_tensor` has no
        # autograd rule and PyTorch prints a "silently incorrect behavior"
        # warning + drops those grads.
        k_full = gather_from_sequence_parallel_region(key.contiguous(), group=cp_group)
        v_full = gather_from_sequence_parallel_region(value.contiguous(), group=cp_group)
        # gather_from_sequence_parallel_region stacks each rank's chunk
        # consecutively in rank order. Under zig-zag, each rank's [2*cs]
        # local tokens are [chunk_r_first, chunk_r_second]. So the gathered
        # tensor layout is [r0_first, r0_second, r1_first, r1_second, ...].
        # We need to un-zig-zag into pure global order so mask indices line
        # up. Build a permutation that maps gathered index -> global index.
        device = query.device
        dtype = query.dtype
        cu_seqlens = packed_seq_params.cu_seqlens_q if packed_seq_params is not None else None

        # Sanity: for each packed sub-seq, the GLOBAL length must be
        # divisible by 2*cp_size so chunk_size is integer. With cp_size=1 this
        # reduces to even-length, which the CP=1 parity-test harness may
        # violate (no zig-zag pre-slicing). Skip the check there; permutation
        # is identity under cp_size=1 so odd length is harmless.
        if cu_seqlens is not None and cp_size > 1:
            expected_t_local = 0
            for s_idx in range(len(cu_seqlens) - 1):
                s_len = (cu_seqlens[s_idx + 1] - cu_seqlens[s_idx]).item()
                assert s_len % (2 * cp_size) == 0, f"sub-sequence {s_idx} global length ({s_len}) is not " f"divisible by 2*cp_size ({2 * cp_size}); `slice_with_cp` " "should pad before packing"
                expected_t_local += s_len // cp_size
            assert expected_t_local == t_local, f"packed-seq local length mismatch: sum(seq_len // cp_size) = " f"{expected_t_local}, but query.shape[0] = {t_local}"

        if cu_seqlens is None:
            t_full_total = k_full.shape[0]
            cu_seqlens_list = [0, t_full_total]
        else:
            cu_seqlens_list = cu_seqlens.tolist()

        # With cp_size=1 the zigzag degenerates to identity and all-gather is
        # a no-op; skip the permutation (and the floor-div that would drop the
        # trailing odd token for seq_len_global % 2 == 1).
        if cp_size > 1:
            perm = self._cp_unzigzag_permutation(cu_seqlens_list, cp_size, device)
            k_full = k_full.index_select(0, perm)
            v_full = v_full.index_select(0, perm)

        out = torch.empty(t_local, np_q * hn, dtype=dtype, device=device)

        local_offset = 0
        for s_idx in range(len(cu_seqlens_list) - 1):
            seq_start = cu_seqlens_list[s_idx]
            seq_len_global = cu_seqlens_list[s_idx + 1] - seq_start
            local_len = seq_len_global // cp_size  # this sub-seq's local Q count

            q_seq = query[local_offset : local_offset + local_len]
            k_seq = k_full[seq_start : seq_start + seq_len_global]
            v_seq = v_full[seq_start : seq_start + seq_len_global]

            # Each rank owns exactly two zig-zag chunks of this sub-sequence,
            # and both are *contiguous* in global position. A contiguous query
            # block whose last row is global position `g_end - 1` needs no
            # explicit mask at all: causality is "attend to K[:g_end], aligned
            # bottom-right", and the sliding window is a fixed left offset from
            # each row. Both are expressible as kernel-native arguments, so we
            # never build the [local_len, seq_len_global] score/mask tensors
            # that forced the math backend and made attention O(T^2) in memory
            # (12+ GiB per global layer at a 40K context under CP2, which is
            # what exhausted the 80 GiB ranks during the actor update).
            if cp_size > 1:
                chunk_size = local_len // 2
                # `_zigzag_global_indices` order: [chunk cp_rank, chunk 2*cp-cp_rank-1].
                chunk_starts = (cp_rank * chunk_size, (2 * cp_size - cp_rank - 1) * chunk_size)
                chunk_bounds = [(i * chunk_size, chunk_size, g) for i, g in enumerate(chunk_starts)]
            else:
                chunk_bounds = [(0, local_len, 0)]

            for local_start, chunk_len, global_start in chunk_bounds:
                if chunk_len == 0:
                    continue
                q_chunk = q_seq[local_start : local_start + chunk_len]
                # Causal: nothing past this chunk's last global position.
                k_end = global_start + chunk_len
                self._attend_contiguous_chunk(
                    q_chunk,
                    k_seq,
                    v_seq,
                    k_end=k_end,
                    global_start=global_start,
                    scale=scale,
                    sliding_window=sliding_window,
                    out=out[local_offset + local_start : local_offset + local_start + chunk_len],
                )
            local_offset += local_len

        return out

    def _attend_contiguous_chunk(
        self,
        q_chunk,
        k_seq,
        v_seq,
        *,
        k_end,
        global_start,
        scale,
        sliding_window,
        out,
    ):
        """Attend one contiguous query chunk without building an explicit mask.

        ``q_chunk`` holds global positions ``[global_start, k_end)`` of a packed
        sub-sequence whose gathered K/V is ``k_seq``/``v_seq`` in global order.
        Causality then means "keys ``[:k_end]``, aligned bottom-right", and a
        sliding window is a constant left offset, so both constraints become
        kernel arguments instead of an ``O(chunk_len * seq_len)`` bias tensor.
        Writes the ``[chunk_len, np_q * hn]`` result into ``out``.
        """
        chunk_len, np_q, hn = q_chunk.shape
        nk = k_seq.shape[1]
        has_window = sliding_window is not None and sliding_window > 0

        # Trim keys the window can never reach. Flash needs the trimmed start to
        # stay aligned with the bottom-right anchor, so keep whole rows only.
        k_start = max(0, global_start - (sliding_window - 1)) if has_window else 0
        k_chunk = k_seq[k_start:k_end]
        v_chunk = v_seq[k_start:k_end]

        if hn <= 256:
            # flash-attn supports GQA natively and takes the window directly.
            from flash_attn import flash_attn_varlen_func

            cu_q = torch.tensor([0, chunk_len], device=q_chunk.device, dtype=torch.int32)
            cu_k = torch.tensor([0, k_chunk.shape[0]], device=q_chunk.device, dtype=torch.int32)
            o = flash_attn_varlen_func(
                q_chunk.contiguous(),
                k_chunk.contiguous(),
                v_chunk.contiguous(),
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=chunk_len,
                max_seqlen_k=k_chunk.shape[0],
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=scale,
                causal=True,
                window_size=(sliding_window - 1, 0) if has_window else (-1, -1),
            )
            out.copy_(o.reshape(chunk_len, -1))
            return

        # Global layers use head_dim=512, which flash-attn 2.x rejects, so they
        # run on SDPA's memory-efficient kernel. That kernel has no GQA support,
        # so broadcast K/V heads explicitly - a [seq_len, np_q, hn] copy, which
        # is linear in sequence length rather than quadratic. Global layers are
        # never sliding, so bottom-right causal alone is exact.
        assert not has_window, "Gemma4 global (head_dim>256) layers must not use a sliding window"
        k4 = k_seq[:k_end].unsqueeze(0).transpose(1, 2)  # [1, nk, k_end, hn]
        v4 = v_seq[:k_end].unsqueeze(0).transpose(1, 2)
        if np_q != nk:
            repeat = np_q // nk
            k4 = k4.unsqueeze(2).expand(1, nk, repeat, k_end, hn).reshape(1, np_q, k_end, hn)
            v4 = v4.unsqueeze(2).expand(1, nk, repeat, k_end, hn).reshape(1, np_q, k_end, hn)
        o = F.scaled_dot_product_attention(
            q_chunk.unsqueeze(0).transpose(1, 2),
            k4,
            v4,
            attn_mask=causal_lower_right(chunk_len, k_end),
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=scale,
        )
        out.copy_(o.transpose(1, 2).reshape(chunk_len, -1))

    def _forward_thd_flash(self, query, key, value, cu_seqlens):
        """Sliding-window or head_dim<=256 path via flash_attn_varlen_func.

        CP==1 only. For CP>1, `_forward_cp_subseq_mask` handles zig-zag.

        Sliding-window layers must pass `window_size=(sliding_window-1, 0)` so
        only tokens within `sliding_window` positions back are attended to -
        this matches HF's `sliding_window_mask_function`. Global layers and
        dense-attention sliding layers use the default full-causal window.
        """
        from flash_attn import flash_attn_varlen_func

        window_size = (-1, -1)  # full causal when causal=True
        if self._is_sliding:
            sw = getattr(self.config, "sliding_window", None)
            if sw and sw > 0:
                window_size = (int(sw) - 1, 0)

        cu = cu_seqlens.to(torch.int32)
        max_seqlen = (cu[1:] - cu[:-1]).max().item()
        out = flash_attn_varlen_func(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self._resolve_scale(query.shape[2]),
            causal=True,
            window_size=window_size,
        )
        return out.reshape(query.shape[0], -1)

    def _forward_thd_sdpa_per_subseq(self, query, key, value, cu_seqlens):
        """Per-sub-sequence causal SDPA - used when flash-attn can't handle
        head_dim (global layer w/o CP). Avoids materializing a [T, T] mask.
        """
        np_q, hn = query.shape[1], query.shape[2]
        nk = key.shape[1]
        scale = self._resolve_scale(hn)
        out = torch.empty(query.shape[0], np_q * hn, dtype=query.dtype, device=query.device)
        for i in range(len(cu_seqlens) - 1):
            s = cu_seqlens[i].item()
            e = cu_seqlens[i + 1].item()
            q4 = query[s:e].unsqueeze(0).transpose(1, 2)  # [1, np, L, hn]
            k4 = key[s:e].unsqueeze(0).transpose(1, 2)
            v4 = value[s:e].unsqueeze(0).transpose(1, 2)
            o = F.scaled_dot_product_attention(
                q4,
                k4,
                v4,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=scale,
                is_causal=True,
                enable_gqa=(np_q != nk),
            )
            out[s:e] = o.transpose(1, 2).reshape(e - s, -1)
        return out

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, packed_seq_params=None, **kwargs):
        cp_size = getattr(self.config, "context_parallel_size", 1) or 1
        is_thd = query.dim() == 3

        force_cp_path = getattr(self.config, "force_cp_subseq_mask", False)

        if is_thd:
            if cp_size > 1 or force_cp_path:
                sw = None
                if self._is_sliding:
                    sw_cfg = getattr(self.config, "sliding_window", None)
                    if sw_cfg and sw_cfg > 0:
                        sw = int(sw_cfg)
                return self._forward_cp_subseq_mask(
                    query,
                    key,
                    value,
                    packed_seq_params,
                    sliding_window=sw,
                )

            cu_seqlens = None
            if packed_seq_params is not None:
                cu_seqlens = packed_seq_params.cu_seqlens_q

            hn = query.shape[2]
            if cu_seqlens is not None:
                if hn <= 256:
                    return self._forward_thd_flash(query, key, value, cu_seqlens)
                return self._forward_thd_sdpa_per_subseq(query, key, value, cu_seqlens)

            q = query.unsqueeze(0).transpose(1, 2)
            k = key.unsqueeze(0).transpose(1, 2)
            v = value.unsqueeze(0).transpose(1, 2)
            nq, nk = q.shape[1], k.shape[1]
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self._resolve_scale(hn),
                is_causal=True,
                enable_gqa=(nq != nk),
            )
            return out.transpose(1, 2).reshape(query.shape[0], -1)

        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        nq, nk = q.shape[1], k.shape[1]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self._resolve_scale(query.shape[3]),
            is_causal=True,
            enable_gqa=(nq != nk),
        )
        return out.permute(2, 0, 1, 3).reshape(out.size(2), out.size(0), -1)


class Gemma4SelfAttention(SelfAttention):
    """SelfAttention with Gemma4-specific modifications:
    - v_norm: RMSNorm without learnable scale applied to value states.
    - attention_k_eq_v: on global layers the linear_qkv projection emits
      ``[q, k]`` only (no v_proj) and V is derived from K - specifically
      ``V = v_norm(raw_k)`` while ``K = k_norm(raw_k)``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_global = False  # set by Gemma4TransformerLayer after construction
        self._global_layer_idx = None
        self._kv_shared_layer_index = None
        self._store_full_length_kv = False
        self.v_norm = VNorm(self.hidden_size_per_attention_head, eps=self.config.layernorm_epsilon)

    def _split_qkv_global_k_eq_v(self, hidden_states):
        """Split linear_qkv output for global K=V layers.

        The Mcore linear_qkv weight for a K=V global layer is built with
        ``v_proj_weight == k_proj_weight`` (see Gemma4Bridge + convert_gemma4_to_hf),
        so ``linear_qkv(h)`` emits Q/K/V with ``raw_k == raw_v``. Gemma4's
        per-head norms then apply as ``key = k_norm(raw_k)`` and
        ``value = v_norm(raw_k)`` - *not* ``v_norm(k_norm(raw_k))``. We
        reimplement the split here rather than calling the parent so we
        don't have to mutate ``self.k_layernorm`` mid-forward.

        Returns (query[sq,b,np,hn], key[sq,b,ng,hn], value[sq,b,ng,hn]).
        """
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        num_query_heads_per_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        new_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (num_query_heads_per_group + 2) * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_shape)

        q_width = num_query_heads_per_group * self.hidden_size_per_attention_head
        hn = self.hidden_size_per_attention_head
        query, raw_key, _raw_value = torch.split(mixed_qkv, [q_width, hn, hn], dim=3)
        query = query.reshape(query.size(0), query.size(1), -1, hn)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        value = self.v_norm(raw_key)
        key = self.k_layernorm(raw_key) if self.k_layernorm is not None else raw_key
        return query, key, value

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
        if self._is_global and self.config.attention_k_eq_v and split_qkv:
            if output_gate:
                raise NotImplementedError("output_gate is not supported together with attention_k_eq_v")
            query, key, value = self._split_qkv_global_k_eq_v(hidden_states)
            return self._apply_kv_sharing(query, key, value)

        result = super().get_query_key_value_tensors(hidden_states, key_value_states, output_gate=output_gate, split_qkv=split_qkv)
        if not split_qkv:
            return result

        if output_gate:
            query, key, value, gate = result
            value = self.v_norm(value)
            query, key, value = self._apply_kv_sharing(query, key, value)
            return query, key, value, gate

        query, key, value = result
        value = self.v_norm(value)
        return self._apply_kv_sharing(query, key, value)

    def _apply_kv_sharing(self, query, key, value):
        runtime_state = getattr(self, "_gemma4_runtime_state", None)
        if runtime_state is None:
            raise RuntimeError("Gemma4 KV-sharing runtime state is missing. E2B/E4B requires " "the Gemma4 provider hooks installed by --custom-model-provider-path.")
        shared_states = runtime_state.shared_kv_states

        if self._store_full_length_kv and self._global_layer_idx is not None:
            shared_states[self._global_layer_idx] = (key, value)

        if self._kv_shared_layer_index is None:
            return query, key, value

        if self._kv_shared_layer_index not in shared_states:
            raise RuntimeError(f"Gemma4 layer {self._global_layer_idx} needs shared KV from layer " f"{self._kv_shared_layer_index}, but it has not been produced. " "Use pipeline-model-parallel-size=1 for E2B/E4B KV-sharing models.")
        shared_key, shared_value = shared_states[self._kv_shared_layer_index]
        return query, shared_key.to(query.device), shared_value.to(query.device)


def _build_moe_submodule_spec(config):
    """Build the MoE submodule spec (Gemma4MoELayer + TE GroupedMLP experts)."""
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend

    base_spec = get_moe_module_spec_for_backend(
        backend=TESpecProvider(),
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=False,  # use plain F.gelu(approximate='tanh') from config.activation_func
    )
    return ModuleSpec(
        module=Gemma4MoELayer,
        submodules=base_spec.submodules,
        metainfo=base_spec.metainfo,
    )


def get_gemma4_layer_spec_te(config=None) -> ModuleSpec:
    """Layer spec for Gemma4 using native Megatron attention with TE.

    If ``config.enable_moe_block`` is set, the main ``mlp`` submodule is a
    :class:`Gemma4MoELayer` (so that the state-dict path
    ``.mlp.experts.linear_fc*.weight*`` matches mbridge's EP auto-handling),
    and the original dense MLP moves to a sibling ``dense_mlp`` submodule that
    the layer forward sums with the MoE output. For the 31B dense variant,
    ``enable_moe_block=False`` and ``mlp`` stays as the normal Megatron MLP.
    """
    # dense_mlp: use a plain (non-fused-layernorm) linear_fc1 so our explicit
    # `pre_mlp_layernorm` in the layer forward is the sole norm applied to the
    # MLP input. Using TELayerNormColumnParallelLinear here would apply a
    # SECOND layernorm inside fc1, resulting in double-normalization and
    # ~8x inflated MLP outputs.
    dense_mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
    if config is not None and getattr(config, "enable_moe_block", False):
        mlp_spec = _build_moe_submodule_spec(config)
        dense_spec = dense_mlp_spec
    else:
        mlp_spec = dense_mlp_spec
        dense_spec = IdentityOp

    submods = Gemma4TransformerLayerSubmodules(
        self_attention=ModuleSpec(
            module=Gemma4SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=TENorm,
                k_layernorm=TENorm,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        # Gemma4 has an explicit RMSNorm before the gated MLP.  Keep this in
        # the base spec as well as in get_gemma4_spec(); conversion and direct
        # layer construction use this function without the later override.
        pre_mlp_layernorm=TENorm,
        mlp=mlp_spec,
        mlp_bda=get_bias_dropout_add,
        post_attention_layernorm=TENorm,
        post_feedforward_layernorm=TENorm,
        post_per_layer_input_norm=TENorm,
        dense_mlp=dense_spec,
    )
    return ModuleSpec(module=Gemma4TransformerLayer, submodules=submods)


@functools.lru_cache(maxsize=4)
def _load_hf_text_config(hf_checkpoint):
    """Load HF config and unwrap `text_config` if it's a multimodal wrapper.

    Cached via lru_cache so repeated callers (model provider, mbridge, weight
    converter) all share the same parsed object.
    """
    from transformers import AutoConfig
    from slime.utils.hf_config import register_gemma4_config_aliases

    register_gemma4_config_aliases()
    try:
        cfg = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
    except ValueError as exc:
        # Keep Megatron-side config loading usable with older Transformers
        # releases that predate Gemma 4. SGLang itself still requires native
        # Gemma 4 Transformers support and uses the alias above.
        config_path = os.path.join(hf_checkpoint, "config.json")
        try:
            with open(config_path) as config_file:
                raw_config = json.load(config_file)
            text_config = raw_config.get("text_config")
            if not isinstance(text_config, dict):
                raise exc
            from transformers import PretrainedConfig

            cfg = PretrainedConfig.from_dict(text_config)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise exc
    text_config = getattr(cfg, "text_config", None)
    if isinstance(text_config, dict):
        from transformers import PretrainedConfig

        text_config = PretrainedConfig.from_dict(text_config)
    return text_config if text_config is not None else cfg


class _Gemma4MoELayerWarningFilter(logging.Filter):
    """Silence the once-per-layer Megatron warning:
        'Unknown MLP type: <class Gemma4MoELayer>. Using default kwargs.'
    Megatron's TransformerLayer.__init__ recognizes a hardcoded tuple of MLP
    classes via `==` (not issubclass), so Gemma4MoELayer (a MoELayer subclass)
    falls through to the default-kwargs branch. That branch is correct for us
    - Gemma4MoELayer.__init__ fetches its own pg_collection via
    get_default_pg_collection - but the warning spams 30 lines per layer at
    init and confuses log readers. See gemma4_provider.py install hook.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("Unknown MLP type" in msg and "Gemma4MoELayer" in msg)


def _install_moe_warning_filter():
    """Silence the per-layer "Unknown MLP type: Gemma4MoELayer" warning.

    Megatron's TransformerLayer compares MLP class identity via ``==``, so
    MoELayer subclasses hit the default-kwargs branch and log a warning.
    The default-kwargs branch is correct for us (Gemma4MoELayer fetches
    pg_collection itself); filter the noise.
    """
    tl_logger = logging.getLogger("megatron.core.transformer.transformer_layer")
    if getattr(tl_logger, "_gemma4_moe_filter_installed", False):
        return
    tl_logger.addFilter(_Gemma4MoELayerWarningFilter())
    tl_logger._gemma4_moe_filter_installed = True


def _assert_hf_features_supported(hf_text):
    """Fail loudly on Gemma4 HF features this plugin doesn't implement."""
    # Text-only training assumes causal attention; HF's "all" mode disables it.
    if getattr(hf_text, "use_bidirectional_attention", "vision") == "all":
        raise NotImplementedError("Gemma4 use_bidirectional_attention='all' disables causal masking; not supported.")


def _kv_sharing_maps(hf_text):
    first_shared = hf_text.num_hidden_layers - getattr(hf_text, "num_kv_shared_layers", 0)
    if first_shared <= 0 or first_shared >= hf_text.num_hidden_layers:
        return {}, set()

    prev_layers = list(hf_text.layer_types[:first_shared])
    store_layers = {len(prev_layers) - 1 - prev_layers[::-1].index(layer_type) for layer_type in set(prev_layers)}
    shared_map = {}
    for layer_idx in range(first_shared, hf_text.num_hidden_layers):
        layer_type = hf_text.layer_types[layer_idx]
        shared_map[layer_idx] = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
    return shared_map, store_layers


def _apply_core_config(config, hf_text):
    """Set Gemma4's non-MoE, non-RoPE config fields.

    Mutates ``config`` in place. Promotes its ``__class__`` to
    ``Gemma4TransformerConfig`` so the new dataclass fields are reachable
    from downstream Megatron code.
    """
    # Gemma uses GeGLU (gated gelu-tanh), not SwiGLU.
    config.gated_linear_unit = True
    config.activation_func = _gelu_tanh
    config.bias_activation_fusion = False

    # No MoE-vs-dense layer scheduling: every layer is our Gemma4TransformerLayer
    # and the MoE block lives inside its forward. An all-zero list keeps
    # transformer_block's non_homogeneous_layers=True branch active (correct for
    # 26B's differing global vs sliding head_dim / num_kv_heads).
    # Rationale for using moe_layer_freq as the flag: Megatron's
    # TransformerBlock.__init__ sets ``non_homogeneous_layers = True`` iff
    # ``config.moe_layer_freq is not None``. We only need that flag on -
    # the actual dense/MoE dispatch happens inside
    # Gemma4TransformerLayer.forward, so the list contents are never
    # consulted by TransformerBlock itself. If a future Megatron refactor
    # starts reading the list per-layer, we need a Gemma4-specific schedule
    # instead.
    config.moe_layer_freq = [0] * config.num_layers

    # Mirror Megatron's own misspelling (`hetereogenous_*`) - correcting it
    # would silently no-op on Megatron's read path.
    config.hetereogenous_dist_checkpoint = True

    config.__class__ = Gemma4TransformerConfig
    config.global_kv_channels = hf_text.global_head_dim
    config.global_num_query_groups = getattr(hf_text, "num_global_key_value_heads", None) or hf_text.num_key_value_heads
    config.attention_k_eq_v = getattr(hf_text, "attention_k_eq_v", True)
    config.final_logit_softcapping = getattr(hf_text, "final_logit_softcapping", 30.0)
    config.sliding_window = hf_text.sliding_window
    config.hidden_size_per_layer_input = getattr(hf_text, "hidden_size_per_layer_input", 0) or 0
    config.num_kv_shared_layers = getattr(hf_text, "num_kv_shared_layers", 0) or 0
    config.use_double_wide_mlp = bool(getattr(hf_text, "use_double_wide_mlp", False))
    config.kv_shared_layer_map, config.kv_store_layers = _kv_sharing_maps(hf_text)

    # `sliding_window_pattern` isn't in Gemma4 HF configs - infer from
    # layer_types (first full_attention layer's 1-indexed position).
    layer_types = list(getattr(hf_text, "layer_types", []))
    try:
        config.sliding_window_pattern = layer_types.index("full_attention") + 1
    except ValueError:
        config.sliding_window_pattern = 6

    # Q/K norms handle softmax scaling; Megatron's default of 1/sqrt(hn) is wrong.
    config.softmax_scale = 1.0
    # Fused RoPE ignores zeroed inv_freq tails; we need unfused for partial-rotary.
    config.apply_rope_fusion = False


def _apply_moe_config(config, hf_text):
    """Set MoE fields if this is a MoE variant (26B-A4B)."""
    config.enable_moe_block = getattr(hf_text, "enable_moe_block", False)
    if not config.enable_moe_block:
        return

    config.num_moe_experts = hf_text.num_experts
    config.moe_router_topk = hf_text.top_k_experts
    config.moe_ffn_hidden_size = hf_text.moe_intermediate_size
    # Megatron MoE infrastructure reads these even though our custom router
    # bypasses its scoring logic; defaults mirror a working Qwen3.5-A3B config.
    config.moe_token_dispatcher_type = getattr(config, "moe_token_dispatcher_type", None) or "alltoall"
    config.moe_grouped_gemm = getattr(config, "moe_grouped_gemm", None) or True
    config.moe_aux_loss_coeff = 0.0  # Gemma4 router has no aux loss
    config.moe_router_load_balancing_type = getattr(config, "moe_router_load_balancing_type", None) or "none"
    config.moe_router_score_function = getattr(config, "moe_router_score_function", None) or "softmax"
    config.moe_router_topk_scaling_factor = getattr(config, "moe_router_topk_scaling_factor", None) or 1.0
    config.moe_router_pre_softmax = False


def get_rope_local_base_freq(hf_text) -> float:
    """Extract sliding-attention RoPE theta from an HF Gemma4 text config.

    Single source of truth for both the model provider and the mbridge
    config builder - otherwise the 10000.0 default would drift between
    call sites.
    """
    return (getattr(hf_text, "rope_parameters", {}) or {}).get("sliding_attention", {}).get("rope_theta", 10000.0)


def _apply_rope_config(config, hf_text):
    rope_params = getattr(hf_text, "rope_parameters", {}) or {}
    config.rope_local_base_freq = get_rope_local_base_freq(hf_text)
    config.global_partial_rotary_factor = rope_params.get("full_attention", {}).get("partial_rotary_factor", 0.25)
    # Set here, before `get_gemma4_layer_spec_te` builds the layers, because
    # global layers are constructed against a `dc_replace` clone of this config.
    # The provider also assigns this when it installs DualRotaryEmbedding, but
    # that happens *after* construction and so only reaches the sliding layers,
    # which still hold the original config object. A global layer that cannot
    # see this width skips the RoPE slice entirely, receives the full
    # (global + local) concatenation, and Megatron then rotates only the
    # leading `rot_dim` columns - pairing dim i with i+64 instead of i+256 and
    # silently applying the wrong rotation on every full-attention layer.
    config.dual_rope_global_dim = int(hf_text.global_head_dim)


def _guard_cp_sliding_window(args, config):
    """Fail if per-rank CP token cap is smaller than the sliding window.

    Strong signal of a miscounted CP sizing - we'd train on truncated
    attention windows otherwise.
    """
    cp_size = getattr(args, "context_parallel_size", 1) or 1
    if cp_size <= 1:
        return
    max_tokens = getattr(args, "max_tokens_per_gpu", None)
    if max_tokens is not None and max_tokens < config.sliding_window:
        raise ValueError(f"context_parallel_size={cp_size} with max_tokens_per_gpu={max_tokens} " f"< sliding_window={config.sliding_window}: per-rank CP chunk cap is " "smaller than the sliding window. Reduce CP or raise max_tokens_per_gpu.")


def get_gemma4_spec(args, config, vp_stage):
    """Return the native Gemma4 layer spec with proper config overrides."""
    hf_text = _load_hf_text_config(args.hf_checkpoint)

    _install_moe_warning_filter()
    _assert_hf_features_supported(hf_text)
    _apply_core_config(config, hf_text)
    _apply_moe_config(config, hf_text)
    _apply_rope_config(config, hf_text)
    _guard_cp_sliding_window(args, config)

    spec = get_gemma4_layer_spec_te(config)
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    if not getattr(config, "enable_moe_block", False):
        spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
    spec.submodules.mlp.metainfo = {"fuse_pre_mlp_layernorm": False}
    spec.submodules.pre_mlp_layernorm = TESpecProvider().layer_norm()
    return spec
