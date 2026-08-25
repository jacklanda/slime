import io
import pickle

import torch



def _normalize_legacy_te_extra_state(state):
    """Mirror the checkpoint compatibility conversion without importing Megatron."""
    if not isinstance(state, io.BytesIO):
        return state
    state.seek(0)
    legacy = torch.load(state, map_location="cpu", weights_only=False)
    if legacy is None or (
        isinstance(legacy, (list, tuple))
        and all(isinstance(item, torch.Tensor) and item.numel() == 0 for item in legacy)
    ):
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(pickle.dumps(legacy)), dtype=torch.uint8).clone()


def test_normalize_empty_legacy_te_extra_state():
    payload = io.BytesIO()
    torch.save([torch.empty(0, dtype=torch.uint8)], payload)
    normalized = _normalize_legacy_te_extra_state(payload)
    assert isinstance(normalized, torch.Tensor)
    assert normalized.dtype == torch.uint8
    assert normalized.numel() == 0


def test_normalize_nonempty_legacy_te_extra_state():
    payload = io.BytesIO()
    torch.save({"forward": {"value": 3}}, payload)
    normalized = _normalize_legacy_te_extra_state(payload)
    assert isinstance(normalized, torch.Tensor)
    assert pickle.loads(normalized.numpy().tobytes()) == {"forward": {"value": 3}}
