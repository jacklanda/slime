import pytest
import torch

pytest.importorskip("megatron.training")

from slime_plugins.models import gemma4_provider


def test_gemma4_recompute_reclaims_cache_only_under_memory_pressure(monkeypatch):
    monkeypatch.setenv("SLIME_GEMMA4_CLEAR_CACHE_BEFORE_BACKWARD", "1")
    monkeypatch.setenv("SLIME_GEMMA4_ACTOR_TRAIN_ACTIVE", "1")
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 20 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 10 * 1024**3)

    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (9 * 1024**3, 80 * 1024**3))
    gemma4_provider._clear_fragmented_cache_before_gemma4_recompute(None, ())
    assert empty_cache_calls == []

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (4 * 1024**3, 80 * 1024**3))
    gemma4_provider._clear_fragmented_cache_before_gemma4_recompute(None, ())
    assert empty_cache_calls == [True]


def test_gemma4_recompute_never_reclaims_outside_actor_backward(monkeypatch):
    monkeypatch.setenv("SLIME_GEMMA4_CLEAR_CACHE_BEFORE_BACKWARD", "1")
    monkeypatch.delenv("SLIME_GEMMA4_ACTOR_TRAIN_ACTIVE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    gemma4_provider._clear_fragmented_cache_before_gemma4_recompute(None, ())
    assert empty_cache_calls == []
