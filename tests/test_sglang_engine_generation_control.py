from argparse import Namespace

from slime.backends.sglang_utils.sglang_engine import SGLangEngine


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"status": "ok"}


def _engine(args):
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.args = args
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 15000
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
