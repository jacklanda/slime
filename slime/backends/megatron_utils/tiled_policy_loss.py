import os
from types import MethodType


def install_tiled_policy_loss(model) -> None:
    """Let the policy loss project response hidden states in bounded tiles."""
    if not hasattr(model, "_postprocess"):
        raise RuntimeError("Tiled policy loss requires a Megatron GPTModel with _postprocess")

    original_postprocess = model._postprocess

    def _postprocess_without_full_logits(self, *args, **kwargs):
        if os.environ.get("SLIME_TILED_POLICY_LOSS_ACTIVE") == "1":
            hidden_states = kwargs.get("hidden_states", args[0] if args else None)
            labels = kwargs.get("labels", args[3] if len(args) > 3 else None)
            if hidden_states is None or labels is not None:
                raise RuntimeError("Tiled policy loss requires hidden states and labels=None")
            # Match GPTModel's normal [s,b,v] -> [b,s,v] output layout. The
            # policy loss owns the output projection tile-by-tile.
            return hidden_states.transpose(0, 1).contiguous()
        return original_postprocess(*args, **kwargs)

    model._postprocess = MethodType(_postprocess_without_full_logits, model)
