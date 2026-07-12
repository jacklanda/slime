import torch

from slime.utils.tensor_backper import TensorBackuper


def test_tensor_backuper_can_disable_pinned_memory(monkeypatch):
    monkeypatch.setenv("SLIME_TENSOR_BACKUP_PIN_MEMORY", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    pin_memory_values = []
    real_empty_like = torch.empty_like

    def fake_empty_like(*args, **kwargs):
        pin_memory_values.append(kwargs.get("pin_memory"))
        return real_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", fake_empty_like)

    param = torch.ones(2, dtype=torch.float32)
    backuper = TensorBackuper.create(lambda: [("param", param)], single_tag=None)
    backuper.backup("actor")

    assert pin_memory_values == [False]
    assert torch.equal(backuper.get("actor")["param"], param)


def test_tensor_backuper_falls_back_when_pinned_allocation_fails(monkeypatch):
    monkeypatch.delenv("SLIME_TENSOR_BACKUP_PIN_MEMORY", raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    pin_memory_values = []
    real_empty_like = torch.empty_like

    def fake_empty_like(*args, **kwargs):
        pin_memory_values.append(kwargs.get("pin_memory"))
        if kwargs.get("pin_memory"):
            raise RuntimeError("pinned allocation failed")
        return real_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", fake_empty_like)

    param = torch.ones(2, dtype=torch.float32)
    backuper = TensorBackuper.create(lambda: [("param", param)], single_tag=None)
    backuper.backup("ref")

    assert pin_memory_values == [True, False]
    assert torch.equal(backuper.get("ref")["param"], param)
