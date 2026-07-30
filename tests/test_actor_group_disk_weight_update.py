from argparse import Namespace

import pytest

from slime.ray.actor_group import RayTrainGroup


class _RemoteCall:
    def __init__(self, fn):
        self.remote = fn


class _Engine:
    def __init__(self, reported_version=None):
        self.version = "default"
        self.reported_version = reported_version
        self.pull_weights = _RemoteCall(lambda _version: None)
        self.pause_generation = _RemoteCall(lambda: None)
        self.flush_cache = _RemoteCall(lambda: None)
        self.update_weights_from_disk = _RemoteCall(self._update)
        self.get_weight_version = _RemoteCall(
            lambda: self.reported_version if self.reported_version is not None else self.version
        )
        self.continue_generation = _RemoteCall(lambda: None)

    def _update(self, *, model_path, weight_version):
        self.version = weight_version


def _group(monkeypatch, engines):
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda value: value)
    rollout_manager = Namespace(
        onload_weights=_RemoteCall(lambda: None),
        get_updatable_engines_and_lock=_RemoteCall(lambda: (engines, None, 0, None, None)),
    )
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(
        offload_rollout=False,
        update_weight_local_checkpoint_dir=None,
        update_weight_disk_keep_files=True,
        verify_rollout_weight_versions=True,
        ci_test=False,
    )
    group._rollout_manager = rollout_manager
    return group


@pytest.mark.unit
def test_full_disk_reload_verifies_versions_after_engine_update(monkeypatch, tmp_path):
    engines = [_Engine() for _ in range(4)]

    _group(monkeypatch, engines)._reload_rollout_weights_from_disk(tmp_path, "1")

    assert [engine.version for engine in engines] == ["1"] * 4


@pytest.mark.unit
def test_full_disk_reload_rejects_version_mismatch(monkeypatch, tmp_path):
    engines = [_Engine(), _Engine(reported_version="default")]

    with pytest.raises(RuntimeError, match="expected '1'.*default"):
        _group(monkeypatch, engines)._reload_rollout_weights_from_disk(tmp_path, "1")


@pytest.mark.unit
def test_full_disk_update_requires_active_training_actors():
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.role = "actor"
    group.args = Namespace(update_weight_mode="full", update_weight_transport="disk")
    group._actor_handlers = []

    with pytest.raises(RuntimeError, match=r"call create\(\) first"):
        group.update_weights()
