from types import SimpleNamespace

import pytest

from slime.ray.actor_group import RayTrainGroup


NUM_GPUS = 0


class _Actor:
    def __init__(self, actor_id):
        self._ray_actor_id = SimpleNamespace(hex=lambda: actor_id)


def _patch_probe(monkeypatch, probe):
    """Stub the per-node ``_surviving_train_processes`` Ray task.

    ``probe(pids)`` stands in for one poll and returns the pids still holding
    GPU memory. ``.options(...).remote(pids)`` is made to return the result
    directly and ``ray.get`` to pass it through, so no Ray runtime is needed.
    """
    monkeypatch.setattr(
        "slime.ray.actor_group._surviving_train_processes",
        SimpleNamespace(options=lambda **_kwargs: SimpleNamespace(remote=probe)),
    )
    # The real strategy validates node_id as a hex Ray NodeID.
    monkeypatch.setattr(
        "slime.ray.actor_group.NodeAffinitySchedulingStrategy",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda refs: list(refs))


def test_release_train_removes_only_previous_non_periodic_checkpoint(tmp_path):
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(save=str(tmp_path), save_interval=20)

    temporary = tmp_path / "iter_0000018"
    temporary.mkdir()
    group._remove_previous_temporary_checkpoint(19)
    assert not temporary.exists()

    permanent = tmp_path / "iter_0000019"
    permanent.mkdir()
    group._remove_previous_temporary_checkpoint(20)
    assert permanent.is_dir()

    current = tmp_path / "iter_0000020"
    current.mkdir()
    group._remove_previous_temporary_checkpoint(20)
    assert current.is_dir()


def test_release_train_cleans_temporary_critic_checkpoint_without_releasing_critic(monkeypatch, tmp_path):
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.role = "critic"
    group.args = SimpleNamespace(
        save=str(tmp_path),
        save_interval=10,
        release_train=True,
    )
    group._actor_handlers = [SimpleNamespace(save_model=SimpleNamespace(remote=lambda *_args, **_kwargs: None))]
    temporary = tmp_path / "iter_0000000"
    temporary.mkdir()
    monkeypatch.setattr("slime.ray.actor_group.ray.get", lambda refs: refs)

    group.save_model(1)

    assert not temporary.exists()
    assert group._actor_handlers


@pytest.mark.unit
def test_release_train_waits_for_force_killed_actor_processes_to_exit(monkeypatch):
    actors = [_Actor("actor-0"), _Actor("actor-1")]
    group = RayTrainGroup.__new__(RayTrainGroup)
    group._actor_handlers = actors[:]
    states = {
        "actor-0": SimpleNamespace(node_id="node-0", pid=100),
        "actor-1": SimpleNamespace(node_id="node-0", pid=101),
    }
    # Probe results per poll: 100 lingers on the GPU one extra round.
    polls = [{100, 101}, {100}, set()]
    killed = []

    monkeypatch.setattr("slime.ray.actor_group.ray.kill", lambda actor, **kwargs: killed.append((actor, kwargs)))
    monkeypatch.setattr("slime.ray.actor_group.get_actor", lambda actor_id, **_kwargs: states[actor_id])
    _patch_probe(monkeypatch, lambda _pids: polls.pop(0))
    monkeypatch.setattr("slime.ray.actor_group.time.sleep", lambda _seconds: None)

    group.release()

    assert group._actor_handlers == []
    assert killed == [(actor, {"no_restart": True}) for actor in actors]
    assert polls == []


@pytest.mark.unit
def test_release_train_fails_before_rollout_onload_when_actor_process_survives(monkeypatch):
    actor = _Actor("actor")
    group = RayTrainGroup.__new__(RayTrainGroup)
    group._actor_handlers = [actor]

    monkeypatch.setattr("slime.ray.actor_group.ray.kill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "slime.ray.actor_group.get_actor",
        lambda *_args, **_kwargs: SimpleNamespace(node_id="node-0", pid=100),
    )
    _patch_probe(monkeypatch, lambda _pids: {100})
    monkeypatch.setattr("slime.ray.actor_group.time.sleep", lambda _seconds: None)
    monotonic_values = iter([0, 10_000])
    monkeypatch.setattr("slime.ray.actor_group.time.monotonic", lambda: next(monotonic_values))

    with pytest.raises(TimeoutError, match="actor processes"):
        group.release()


@pytest.mark.unit
def test_release_train_waits_for_process_still_holding_gpu_memory(monkeypatch):
    """A dead-but-not-reaped trainer must keep release() blocking.

    Resuming the colocated SGLang KV pool while the trainer still owns its
    CUDA context is what produced CUDA_ERROR_OUT_OF_MEMORY and killed the
    scheduler subprocess.
    """
    actor = _Actor("actor")
    group = RayTrainGroup.__new__(RayTrainGroup)
    group._actor_handlers = [actor]
    polls = [{100}, {100}, set()]
    sleeps = []

    monkeypatch.setattr("slime.ray.actor_group.ray.kill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "slime.ray.actor_group.get_actor",
        lambda *_args, **_kwargs: SimpleNamespace(node_id="node-0", pid=100),
    )
    _patch_probe(monkeypatch, lambda _pids: polls.pop(0))
    monkeypatch.setattr("slime.ray.actor_group.time.sleep", sleeps.append)

    group.release()

    assert polls == []
    assert len(sleeps) == 2


@pytest.mark.unit
def test_release_train_preserves_handles_when_process_identity_is_unavailable(monkeypatch):
    actor = _Actor("actor")
    group = RayTrainGroup.__new__(RayTrainGroup)
    group._actor_handlers = [actor]
    monkeypatch.setattr("slime.ray.actor_group.get_actor", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="Cannot identify training actor process"):
        group.release()

    assert group._actor_handlers == [actor]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
