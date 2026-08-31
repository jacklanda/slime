from argparse import Namespace
import threading
import time

import pytest
import requests

from slime.ray.rollout import RolloutManager, RolloutServer, ServerGroup
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
    def __init__(self, *, health_generate_result=True, health_actor_result=True, pause_result=None):
        self.calls = []
        self.pause_generation = _RemoteMethod("pause_generation", self.calls, pause_result)
        self.continue_generation = _RemoteMethod("continue_generation", self.calls)
        self.flush_cache = _RemoteMethod("flush_cache", self.calls)
        self.release_memory_occupation = _RemoteMethod("release_memory_occupation", self.calls)
        self.resume_memory_occupation = _RemoteMethod("resume_memory_occupation", self.calls)
        self.health_generate = _RemoteMethod("health_generate", self.calls, health_generate_result)
        self.health_actor = _RemoteMethod("health_actor", self.calls, health_actor_result)
        self.shutdown = _RemoteMethod("shutdown", self.calls)
        self.register_to_router = _RemoteMethod("register_to_router", self.calls)
        self.update_weights_from_disk = _RemoteMethod("update_weights_from_disk", self.calls)


def _ray_get(value):
    if isinstance(value, list):
        return [_ray_get(item) for item in value]
    if isinstance(value, Exception):
        raise value
    return value


def _make_server(engine):
    args = Namespace(num_gpus_per_node=1, debug_train_only=False, rollout_health_check_timeout=3.0)
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
    assert [name for name, _args, _kwargs in engine.calls[:3]] == [
        "health_actor",
        "pause_generation",
        "release_memory_occupation",
    ]
    assert engine.calls[1][2] == {}

    server.onload_weights()
    assert group.generation_health_check_enabled is False

    group.start_engines = lambda port_cursors, **_kwargs: ([], port_cursors)
    server.recover(health_check_timeout=3.0)
    assert [name for name, _args, _kwargs in engine.calls].count("health_generate") == 0

    server.onload_kv()
    assert group.generation_health_check_enabled is True
    assert [name for name, _args, _kwargs in engine.calls].count("continue_generation") == 1

    server.recover(health_check_timeout=3.0)
    assert [name for name, _args, _kwargs in engine.calls].count("health_generate") == 1


def test_offload_marks_dead_actor_and_keeps_other_engines_progressing(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.ray.get", _ray_get)
    monkeypatch.setattr("slime.ray.rollout.ray.kill", lambda *_args, **_kwargs: None)

    dead = _FakeEngine(health_actor_result=RuntimeError("actor died"))
    healthy = _FakeEngine()
    args = Namespace(num_gpus_per_node=1, debug_train_only=False, rollout_health_check_timeout=3.0)
    group = ServerGroup(
        args=args,
        pg=None,
        all_engines=[dead, healthy],
        num_gpus_per_engine=1,
        num_new_engines=0,
        needs_offload=True,
        worker_type="regular",
    )
    server = RolloutServer(server_groups=[group])

    server.offload()

    assert group.all_engines[0] is None
    assert [name for name, _args, _kwargs in healthy.calls] == [
        "health_actor",
        "pause_generation",
        "release_memory_occupation",
    ]


def test_offload_isolates_actor_that_dies_during_pause(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.ray.get", _ray_get)
    monkeypatch.setattr("slime.ray.rollout.ray.kill", lambda *_args, **_kwargs: None)

    engine = _FakeEngine(pause_result=RuntimeError("node heartbeat lost"))
    server, group = _make_server(engine)

    server.offload()

    assert group.all_engines == [None]
    assert [name for name, _args, _kwargs in engine.calls].count("release_memory_occupation") == 0


def test_dead_actor_found_during_offload_is_rebuilt_before_weight_update(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.ray.get", _ray_get)
    monkeypatch.setattr("slime.ray.rollout.ray.kill", lambda *_args, **_kwargs: None)

    dead = _FakeEngine(pause_result=RuntimeError("node heartbeat lost"))
    server, group = _make_server(dead)

    server.offload()
    assert group.all_engines == [None]

    replacement = _FakeEngine()

    def start_replacement(port_cursors, **_kwargs):
        group.all_engines[0] = replacement
        group.num_new_engines = 1
        group.pending_router_registration_indices.add(0)
        return ["replacement-init"], port_cursors

    group.start_engines = start_replacement
    server.recover(health_check_timeout=3.0)

    assert group.all_engines == [replacement]
    assert [name for name, _args, _kwargs in replacement.calls] == [
        "release_memory_occupation",
        "resume_memory_occupation",
    ]
    assert replacement.calls[1][2] == {"tags": ["weights"]}


def test_active_rollout_reboost_loads_current_weights_before_router_registration(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.ray.get", _ray_get)

    server, group = _make_server(None)
    replacement = _FakeEngine()

    def start_replacement(port_cursors, **kwargs):
        assert kwargs == {"register_to_router": False}
        group.all_engines[0] = replacement
        group.num_new_engines = 1
        group.pending_router_registration_indices.add(0)
        return ["replacement-init"], port_cursors

    group.start_engines = start_replacement
    server.recover(active_rollout_weight=("/weights/weight_v000002", "2"))

    assert [name for name, _args, _kwargs in replacement.calls] == [
        "pause_generation",
        "release_memory_occupation",
        "update_weights_from_disk",
        "resume_memory_occupation",
        "continue_generation",
        "register_to_router",
    ]
    assert replacement.calls[1][2] == {"tags": ["kv_cache", "cuda_graph"]}
    assert replacement.calls[2][2] == {
        "model_path": "/weights/weight_v000002",
        "weight_version": "2",
    }
    assert replacement.calls[3][2] == {"tags": ["kv_cache", "cuda_graph"]}
    assert group.generation_health_check_enabled is True
    assert group.pending_router_registration_indices == set()


def test_reboost_before_first_weight_publication_recovers_initial_checkpoint(monkeypatch):
    server, group = _make_server(None)
    events = []
    server.recover = lambda **kwargs: events.append(("recover", kwargs))
    server.onload_kv = lambda: events.append(("onload_kv", {}))

    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = Namespace(rollout_health_check_timeout=3.0)
    manager.servers = {"default": server}
    manager._latest_rollout_weight = None
    manager._reboost_lock = threading.Lock()
    manager.health_monitoring_pause = lambda: events.append(("pause", {}))
    manager.health_monitoring_resume = lambda: events.append(("resume", {}))

    manager._reboost_engine_group(group, 0)

    assert events == [
        ("pause", {}),
        ("recover", {"health_check_timeout": 3.0}),
        ("onload_kv", {}),
        ("resume", {}),
    ]


def test_manager_onload_weights_pauses_health_monitor_before_memory_restore(monkeypatch):
    server, _group = _make_server(_FakeEngine())
    events = []
    server.onload_weights = lambda: events.append("onload_weights")

    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.servers = {"default": server}
    manager.health_monitoring_pause = lambda: events.append("pause")

    manager.onload_weights()

    assert events[:2] == ["pause", "onload_weights"]


def test_onload_checks_actor_liveness_without_generation(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.ray.get", _ray_get)
    monkeypatch.setattr("slime.ray.rollout.ray.kill", lambda *_args, **_kwargs: None)

    engine = _FakeEngine(health_actor_result=False)
    server, group = _make_server(engine)
    group.generation_health_check_enabled = False

    server.onload_weights()

    assert group.all_engines == [None]
    assert [name for name, _args, _kwargs in engine.calls].count("resume_memory_occupation") == 0


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


def test_health_monitor_reboosts_after_marking_engine_dead(monkeypatch):
    monkeypatch.setattr("slime.utils.health_monitor.ray.get", _ray_get)
    monkeypatch.setattr("slime.utils.health_monitor.ray.kill", lambda *_args, **_kwargs: None)

    engine = _FakeEngine(health_generate_result=RuntimeError("sglang unavailable"))
    _server_obj, group = _make_server(engine)
    reboosts = []
    monitor = RolloutHealthMonitor(
        group,
        Namespace(
            rollout_health_check_interval=1.0,
            rollout_health_check_timeout=1.0,
            rollout_health_check_first_wait=0.0,
        ),
        on_engine_failure=lambda failed_group, engine_id: reboosts.append((failed_group, engine_id)),
    )

    monitor._check_engine_health(0, engine)

    assert group.all_engines == [None]
    assert reboosts == [(group, 0)]


def test_health_monitor_tolerates_transient_http_503(monkeypatch):
    monkeypatch.setattr("slime.utils.health_monitor.ray.get", _ray_get)
    monkeypatch.setattr("slime.utils.health_monitor.ray.kill", lambda *_args, **_kwargs: None)

    response = requests.Response()
    response.status_code = 503
    error = requests.HTTPError("service unavailable", response=response)
    engine = _FakeEngine(health_generate_result=error)
    _server_obj, group = _make_server(engine)
    monitor = RolloutHealthMonitor(
        group,
        Namespace(
            rollout_health_check_interval=1.0,
            rollout_health_check_timeout=1.0,
            rollout_health_check_first_wait=0.0,
            rollout_health_check_failure_threshold=3,
        ),
    )

    assert monitor._check_engine_health(0, engine) is True
    assert group.all_engines == [engine]
    assert monitor._check_engine_health(0, engine) is True
    assert group.all_engines == [engine]
    assert monitor._check_engine_health(0, engine) is False
    assert group.all_engines == [None]


def test_health_monitor_resets_transient_failures_after_success(monkeypatch):
    monkeypatch.setattr("slime.utils.health_monitor.ray.get", _ray_get)
    response = requests.Response()
    response.status_code = 503
    engine = _FakeEngine(health_generate_result=requests.HTTPError(response=response))
    _server_obj, group = _make_server(engine)
    monitor = RolloutHealthMonitor(
        group,
        Namespace(
            rollout_health_check_interval=1.0,
            rollout_health_check_timeout=1.0,
            rollout_health_check_first_wait=0.0,
            rollout_health_check_failure_threshold=2,
        ),
    )

    assert monitor._check_engine_health(0, engine) is True
    engine.health_generate.result = True
    assert monitor._check_engine_health(0, engine) is True
    engine.health_generate.result = requests.HTTPError(response=response)
    assert monitor._check_engine_health(0, engine) is True
    assert group.all_engines == [engine]


def test_health_monitor_pause_waits_for_inflight_generation_check(monkeypatch):
    check_started = threading.Event()
    release_check = threading.Event()

    def blocking_ray_get(_value):
        check_started.set()
        assert release_check.wait(timeout=2.0)
        return True

    monkeypatch.setattr("slime.utils.health_monitor.ray.get", blocking_ray_get)

    engine = _FakeEngine()
    _server_obj, group = _make_server(engine)
    monitor = RolloutHealthMonitor(
        group,
        Namespace(
            rollout_health_check_interval=1.0,
            rollout_health_check_timeout=1.0,
            rollout_health_check_first_wait=0.0,
        ),
    )
    monitor._pause_event = threading.Event()

    check_thread = threading.Thread(target=monitor._run_health_checks)
    check_thread.start()
    assert check_started.wait(timeout=1.0)

    pause_thread = threading.Thread(target=monitor.pause)
    pause_thread.start()
    time.sleep(0.05)
    assert pause_thread.is_alive()

    release_check.set()
    pause_thread.join(timeout=1.0)
    check_thread.join(timeout=1.0)
    assert not pause_thread.is_alive()
    assert not check_thread.is_alive()


def test_offload_restarts_local_engines_after_incomplete_abort(monkeypatch):
    events = []

    class FakeServer:
        def restart_with_overrides(self, overrides):
            events.append(("restart", overrides))

        def offload(self):
            events.append(("offload", None))

    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = Namespace(
        _rollout_abort_engine_restart_required=True,
        rollout_external=False,
        ci_test=False,
        use_fault_tolerance=False,
        rollout_only_inference_fast_path=False,
        debug_rollout_only=True,
    )
    manager.servers = {"default": FakeServer()}
    manager._health_monitors = []
    manager._get_rollout_data = lambda rollout_id: ([object()], {})
    manager._save_debug_rollout_data = lambda *_args, **_kwargs: None
    monkeypatch.setattr("slime.ray.rollout._log_rollout_data", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("slime.ray.rollout.init_http_client", lambda _args: events.append(("init", None)))

    with pytest.raises(RuntimeError, match=r"call offload\(\)"):
        manager.generate(59)
    assert events == []
    manager.offload()
    assert events == [("restart", {}), ("init", None)]
    assert manager.args._rollout_abort_engine_restart_required is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
