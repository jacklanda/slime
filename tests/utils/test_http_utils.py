import asyncio
from argparse import Namespace

from slime.utils import http_utils


def test_reset_http_client_closes_old_transport_before_reinitializing(monkeypatch):
    events = []

    class OldClient:
        async def aclose(self):
            events.append("close")

    new_client = object()

    def init_http_client(_args):
        events.append("init")
        http_utils._http_client = new_client

    monkeypatch.setattr(http_utils, "_http_client", OldClient())
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)
    monkeypatch.setattr(http_utils, "init_http_client", init_http_client)

    asyncio.run(http_utils.reset_http_client(Namespace()))

    assert events == ["close", "init"]
    assert http_utils._http_client is new_client


def test_reset_http_client_kills_distributed_post_actors(monkeypatch):
    events = []
    actors = [object(), object()]

    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_post_actors", actors)
    monkeypatch.setattr(http_utils, "_post_actor_idx", 1)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(
        "ray.kill",
        lambda actor, no_restart: events.append(("kill", actor, no_restart)),
    )
    monkeypatch.setattr(http_utils, "init_http_client", lambda _args: events.append(("init",)))

    asyncio.run(http_utils.reset_http_client(Namespace()))

    assert events == [
        ("kill", actors[0], True),
        ("kill", actors[1], True),
        ("init",),
    ]
    assert http_utils._post_actors == []
    assert http_utils._post_actor_idx == 0
    assert http_utils._distributed_post_enabled is False
