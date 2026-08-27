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
        self.continue_generation_calls = 0
        self.pull_weights = _RemoteCall(lambda _version: None)
        self.pause_generation = _RemoteCall(lambda: None)
        self.flush_cache = _RemoteCall(lambda: None)
        self.update_weights_from_disk = _RemoteCall(self._update)
        self.get_weight_version = _RemoteCall(lambda: self.reported_version if self.reported_version is not None else self.version)
        self.continue_generation = _RemoteCall(self._continue_generation)

    def _update(self, *, model_path, weight_version):
        self.version = weight_version

    def _continue_generation(self):
        self.continue_generation_calls += 1


def _group(monkeypatch, engines):
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda value: value)
    rollout_manager = Namespace(
        onload_weights=_RemoteCall(lambda: None),
        get_updatable_engines_and_lock=_RemoteCall(lambda: (engines, None, 0, None, None)),
        set_latest_rollout_weight=_RemoteCall(lambda _path, _version: None),
    )
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = Namespace(
        offload_rollout=False,
        update_weight_local_checkpoint_dir=None,
        update_weight_disk_keep_files=True,
        use_fault_tolerance=False,
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
    assert [engine.continue_generation_calls for engine in engines] == [1] * 4


@pytest.mark.unit
def test_full_disk_reload_keeps_generation_paused_until_kv_onload(monkeypatch, tmp_path):
    engines = [_Engine() for _ in range(4)]
    group = _group(monkeypatch, engines)
    group.args.offload_rollout = True

    group._reload_rollout_weights_from_disk(tmp_path, "1")

    assert [engine.version for engine in engines] == ["1"] * 4
    assert [engine.continue_generation_calls for engine in engines] == [0] * 4


@pytest.mark.unit
def test_full_disk_reload_rejects_version_mismatch(monkeypatch, tmp_path):
    engines = [_Engine(), _Engine(reported_version="default")]

    with pytest.raises(RuntimeError, match="expected '1'.*default"):
        _group(monkeypatch, engines)._reload_rollout_weights_from_disk(tmp_path, "1")


@pytest.mark.unit
def test_fault_tolerant_disk_reload_retains_latest_weight_only(monkeypatch, tmp_path):
    previous = tmp_path / "weight_v000001"
    current = tmp_path / "weight_v000002"
    previous.mkdir()
    current.mkdir()

    group = _group(monkeypatch, [_Engine()])
    group.args.update_weight_disk_keep_files = False
    group.args.use_fault_tolerance = True
    group._reload_rollout_weights_from_disk(current, "2")

    assert not previous.exists()
    assert current.exists()


@pytest.mark.unit
def test_full_disk_update_requires_active_training_actors():
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.role = "actor"
    group.args = Namespace(update_weight_mode="full", update_weight_transport="disk")
    group._actor_handlers = []

    with pytest.raises(RuntimeError, match=r"call create\(\) first"):
        group.update_weights()
