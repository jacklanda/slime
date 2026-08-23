import json
import sys
import types
from pathlib import Path

import pytest


NUM_GPUS = 0


def _load_checkpoint_module(monkeypatch):
    checkpointing = types.ModuleType("megatron.training.checkpointing")
    checkpointing.load_checkpoint = lambda *args, **kwargs: None
    checkpointing.save_checkpoint = lambda *args, **kwargs: None
    global_vars = types.ModuleType("megatron.training.global_vars")
    global_vars.get_args = lambda: None
    training = types.ModuleType("megatron.training")
    megatron = types.ModuleType("megatron")
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    monkeypatch.setitem(sys.modules, "megatron.training.checkpointing", checkpointing)
    monkeypatch.setitem(sys.modules, "megatron.training.global_vars", global_vars)
    sys.modules.pop("slime.backends.megatron_utils.checkpoint", None)

    from slime.backends.megatron_utils import checkpoint

    return checkpoint


def _write_complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / ".metadata").write_bytes(b"torch metadata")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "sharded_backend": "torch_dist",
                "sharded_backend_version": 1,
                "common_backend": "torch",
                "common_backend_version": 1,
            }
        )
    )
    (path / "common.pt").write_bytes(b"common state")
    (path / "__0_0.distcp").write_bytes(b"shard")


def test_publish_checkpoint_atomically_advances_latest(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    staging = tmp_path / ".iter_0000022.incomplete"
    _write_complete_checkpoint(staging / "iter_0000022")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("21")

    checkpoint._publish_checkpoint(staging, tmp_path, 22)

    assert checkpoint._checkpoint_is_complete(tmp_path / "iter_0000022")
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "22"
    assert not staging.exists()


def test_incomplete_staging_does_not_advance_latest(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    staging = tmp_path / ".iter_0000022.incomplete"
    staged_checkpoint = staging / "iter_0000022"
    staged_checkpoint.mkdir(parents=True)
    (staged_checkpoint / "common.pt").write_bytes(b"common state")
    (staged_checkpoint / "__0_0.distcp").write_bytes(b"partial shard")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("21")

    with pytest.raises(RuntimeError, match="complete checkpoint"):
        checkpoint._publish_checkpoint(staging, tmp_path, 22)

    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "21"
    assert not (tmp_path / "iter_0000022").exists()


def test_cleanup_removes_only_failed_checkpoint_artifacts(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("21")
    _write_complete_checkpoint(tmp_path / "iter_0000021")
    incomplete = tmp_path / "iter_0000022"
    incomplete.mkdir()
    (incomplete / "common.pt").write_bytes(b"common state")
    stale_staging = tmp_path / ".iter_0000023.incomplete"
    stale_staging.mkdir()

    checkpoint._remove_failed_checkpoint_artifacts(tmp_path, 22)

    assert (tmp_path / "iter_0000021").is_dir()
    assert not incomplete.exists()
    assert not stale_staging.exists()


def test_recovery_reconciles_tracker_and_cleans_failed_newer_save(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("21")
    _write_complete_checkpoint(tmp_path / "iter_0000022")
    incomplete = tmp_path / "iter_0000023"
    incomplete.mkdir()
    (incomplete / "common.pt").write_bytes(b"common state")

    checkpoint._recover_checkpoint_state(tmp_path)

    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "22"
    assert (tmp_path / "iter_0000022").is_dir()
    assert not incomplete.exists()


def test_explicit_resume_rewinds_tracker_and_removes_newer_checkpoints(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("59")
    _write_complete_checkpoint(tmp_path / "iter_0000058")
    _write_complete_checkpoint(tmp_path / "iter_0000059")
    stale_staging = tmp_path / ".iter_0000060.incomplete"
    stale_staging.mkdir()

    checkpoint._rewind_checkpoint_state(tmp_path, 58)

    assert checkpoint._checkpoint_is_complete(tmp_path / "iter_0000058")
    assert not (tmp_path / "iter_0000059").exists()
    assert not stale_staging.exists()
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "58"


def test_load_recovers_incomplete_checkpoint_before_megatron_load(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("21")
    _write_complete_checkpoint(tmp_path / "iter_0000021")
    incomplete = tmp_path / "iter_0000022"
    incomplete.mkdir()
    (incomplete / "common.pt").write_bytes(b"common state")
    args = types.SimpleNamespace(
        load=str(tmp_path),
        use_dist_ckpt=True,
        ckpt_format="torch_dist",
    )
    loaded = []

    def fake_load(**kwargs):
        loaded.append(kwargs)
        return 21, 0

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", fake_load)

    result = checkpoint.load_checkpoint(None, None, None, {}, False)

    assert result == (21, 0)
    assert len(loaded) == 1
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "21"
    assert not incomplete.exists()


def test_failed_save_restores_save_dir_and_leaves_latest_unchanged(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("21")
    _write_complete_checkpoint(tmp_path / "iter_0000021")
    args = types.SimpleNamespace(
        save=str(tmp_path),
        use_dist_ckpt=True,
        ckpt_format="torch_dist",
        async_save=False,
    )

    def failed_save(iteration, *unused_args, **unused_kwargs):
        partial = Path(args.save) / f"iter_{iteration:07d}"
        partial.mkdir(parents=True)
        (partial / "__0_0.distcp").write_bytes(b"partial shard")
        raise OSError("owner node lost")

    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_save_checkpoint_megatron", failed_save)

    with pytest.raises(OSError, match="owner node lost"):
        checkpoint.save_checkpoint(22)

    assert args.save == str(tmp_path)
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "21"
    assert not (tmp_path / "iter_0000022").exists()


def test_transaction_waits_for_async_save_before_publish(monkeypatch, tmp_path):
    checkpoint = _load_checkpoint_module(monkeypatch)
    args = types.SimpleNamespace(
        save=str(tmp_path),
        use_dist_ckpt=True,
        ckpt_format="torch_dist",
        async_save=True,
    )
    saved_checkpoint = None

    def fake_save(iteration, *unused_args, **unused_kwargs):
        nonlocal saved_checkpoint
        saved_checkpoint = Path(args.save) / f"iter_{iteration:07d}"
        saved_checkpoint.mkdir(parents=True)
        (saved_checkpoint / "common.pt").write_bytes(b"common state")
        (saved_checkpoint / "__0_0.distcp").write_bytes(b"shard")
        return "saved"

    def finalize(*, blocking):
        assert blocking is True
        assert saved_checkpoint is not None
        (saved_checkpoint / ".metadata").write_bytes(b"torch metadata")
        (saved_checkpoint / "metadata.json").write_text(
            json.dumps(
                {
                    "sharded_backend": "torch_dist",
                    "sharded_backend_version": 1,
                    "common_backend": "torch",
                    "common_backend_version": 1,
                }
            )
        )

    async_utils = types.ModuleType("megatron.training.async_utils")
    async_utils.maybe_finalize_async_save = finalize
    monkeypatch.setitem(sys.modules, "megatron.training.async_utils", async_utils)
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_save_checkpoint_megatron", fake_save)

    assert checkpoint.save_checkpoint(22) == "saved"
    assert args.save == str(tmp_path)
    assert checkpoint._checkpoint_is_complete(tmp_path / "iter_0000022")
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "22"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
