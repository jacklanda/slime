from argparse import Namespace

from slime.backends.sglang_utils.sglang_engine import SGLangEngine


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"status": "ok"}


class _VersionResponse(_Response):
    def json(self):
        return {"weight_version": "v9"}


def _engine(args):
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = args
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 15000
    engine._offloaded_memory_tags = set()
    return engine


def test_pause_and_continue_generation_use_control_timeout(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return _Response()

    monkeypatch.setattr("slime.backends.sglang_utils.sglang_engine.requests.post", fake_post)

    engine = _engine(Namespace(rollout_health_check_timeout=7.0, rollout_generation_control_timeout=123.0))

    assert engine.pause_generation() == {"status": "ok"}
    assert engine.continue_generation() == {"status": "ok"}

    assert calls == [
        ("http://127.0.0.1:15000/pause_generation", {}, 123.0),
        ("http://127.0.0.1:15000/continue_generation", {}, 123.0),
    ]


def test_generation_control_timeout_falls_back_to_health_timeout():
    engine = _engine(Namespace(rollout_health_check_timeout=7.0))

    assert engine._generation_control_timeout() == 7.0


def test_get_weight_version_uses_health_timeout(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return _VersionResponse()

    monkeypatch.setattr("slime.backends.sglang_utils.sglang_engine.requests.get", fake_get)
    engine = _engine(Namespace(rollout_health_check_timeout=7.0))

    assert engine.get_weight_version() == "v9"
    assert calls == [("http://127.0.0.1:15000/model_info", 7.0)]


def test_disk_weight_update_releases_temporary_cuda_allocator_cache():
    calls = []
    engine = _engine(Namespace())
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))

    engine.update_weights_from_disk("/tmp/weights", weight_version="7")

    assert calls == [
        (
            "update_weights_from_disk",
            {"model_path": "/tmp/weights", "torch_empty_cache": True, "weight_version": "7"},
        )
    ]


def test_memory_occupation_transitions_are_idempotent(monkeypatch):
    calls = []
    engine = _engine(Namespace())
    engine.flush_cache = lambda: calls.append(("flush_cache", None))
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload)) or {"status": "ok"}

    engine.release_memory_occupation(tags=["weights"])
    engine.release_memory_occupation(tags=["weights"])
    engine.resume_memory_occupation(tags=["weights"])
    engine.resume_memory_occupation(tags=["weights"])

    assert calls == [
        ("flush_cache", None),
        ("release_memory_occupation", {"tags": ["weights"]}),
        ("resume_memory_occupation", {"tags": ["weights"]}),
    ]
    assert engine._offloaded_memory_tags == set()


def test_memory_occupation_only_sends_requested_state_delta():
    calls = []
    engine = _engine(Namespace())
    engine.flush_cache = lambda: None
    engine._make_request = lambda endpoint, payload: calls.append((endpoint, payload))

    engine.release_memory_occupation()
    engine.resume_memory_occupation(tags=["weights"])
    engine.release_memory_occupation(tags=["weights", "kv_cache"])

    assert calls[-1] == ("release_memory_occupation", {"tags": ["weights"]})
