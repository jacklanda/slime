import asyncio
from argparse import Namespace

import httpx
import pytest

from slime.utils import http_utils


class _FailingResponse:
    status_code = 503
    text = "engine unavailable"

    def raise_for_status(self):
        request = httpx.Request("POST", "http://engine/generate")
        response = httpx.Response(self.status_code, request=request, text=self.text)
        raise httpx.HTTPStatusError("upstream failure", request=request, response=response)

    async def aread(self):
        return b""

    async def aclose(self):
        pass


class _FailingClient:
    async def post(self, *_args, **_kwargs):
        return _FailingResponse()


def test_post_strips_httpx_transport_state_from_terminal_errors():
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(http_utils._post(_FailingClient(), "http://engine/generate", {}, max_retries=1))

    error = exc_info.value
    assert error.__dict__["_request"] is None
    assert error.__dict__["response"] is None
    assert str(error) == "HTTP 503 for http://engine/generate: engine unavailable"


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
