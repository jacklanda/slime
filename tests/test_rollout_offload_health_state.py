from argparse import Namespace

import pytest

from slime.ray.rollout import RolloutServer, ServerGroup
from slime.utils.health_monitor import RolloutHealthMonitor

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, name, calls, result=None):
        self.name = name
        self.calls = calls
        self.result = result if result is not None else f"{name}-result"

    def remote(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))
        return self.result


class _FakeEngine:
    def __init__(self):
        self.calls = []
        self.pause_generation = _RemoteMethod("pause_generation", self.calls)
        self.continue_generation = _RemoteMethod("continue_generation", self.calls)
        self.release_memory_occupation = _RemoteMethod("release_memory_occupation", self.calls)
        self.resume_memory_occupation = _RemoteMethod("resume_memory_occupation", self.calls)
        self.health_generate = _RemoteMethod("health_generate", self.calls)


def _ray_get(value):
    if isinstance(value, list):
        return [_ray_get(item) for item in value]
    return value


def _make_server(engine):
    args = Namespace(num_gpus_per_node=1, debug_train_only=False)
    group = ServerGroup(
        args=args,
        pg=None,
        all_engines=[engine],
        num_gpus_per_engine=1,
        num_new_engines=0,
        needs_offload=True,
        worker_type="regular",
    )
    return RolloutServer(server_groups=[group]), group


def test_recover_skips_generation_health_check_until_kv_is_onloaded(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.ray.get", _ray_get)

    engine = _FakeEngine()
    server, group = _make_server(engine)

    server.offload()
    assert group.generation_health_check_enabled is False

    server.onload_weights()
    assert group.generation_health_check_enabled is False

    group.start_engines = lambda port_cursors: ([], port_cursors)
    server.recover(health_check_timeout=3.0)
    assert [name for name, _args, _kwargs in engine.calls].count("health_generate") == 0

    server.onload_kv()
    assert group.generation_health_check_enabled is True
    assert [name for name, _args, _kwargs in engine.calls].count("continue_generation") == 1

    server.recover(health_check_timeout=3.0)
    assert [name for name, _args, _kwargs in engine.calls].count("health_generate") == 1


def test_health_monitor_skips_generation_check_while_group_is_not_generation_ready(monkeypatch):
    def fail_if_called(_value):
        raise AssertionError("health_generate should not be awaited while generation is disabled")

    monkeypatch.setattr("slime.utils.health_monitor.ray.get", fail_if_called)

    engine = _FakeEngine()
    _server_obj, group = _make_server(engine)
    group.generation_health_check_enabled = False
    monitor = RolloutHealthMonitor(
        group,
        Namespace(
            rollout_health_check_interval=1.0,
            rollout_health_check_timeout=1.0,
            rollout_health_check_first_wait=0.0,
        ),
    )

    monitor._check_engine_health(0, engine)

    assert [name for name, _args, _kwargs in engine.calls].count("health_generate") == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
