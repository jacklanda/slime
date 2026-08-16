import asyncio
import logging
import os
import time
from typing import Any

from slime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

ABORT_RETRY_INTERVAL_SECONDS = float(os.environ.get("SLIME_SGLANG_ABORT_RETRY_INTERVAL_SECONDS", "0.5"))
ABORT_TIMEOUT_SECONDS = float(os.environ.get("SLIME_SGLANG_ABORT_TIMEOUT_SECONDS", "30"))
ABORT_HTTP_TIMEOUT_SECONDS = float(os.environ.get("SLIME_SGLANG_ABORT_HTTP_TIMEOUT_SECONDS", "5"))


def num_requests_from_load(load: Any) -> int:
    if isinstance(load, list):
        return sum(num_requests_from_load(item) for item in load)

    if not isinstance(load, dict):
        return 0

    if "loads" in load:
        return num_requests_from_load(load["loads"])

    for key in ("num_reqs", "num_total_reqs", "total_reqs"):
        value = load.get(key)
        if isinstance(value, int):
            return value

    running = load.get("num_running_reqs", load.get("total_running_reqs"))
    waiting = load.get("num_waiting_reqs", load.get("total_waiting_reqs"))
    return (running if isinstance(running, int) else 0) + (waiting if isinstance(waiting, int) else 0)


async def _abort_server_once(url: str) -> None:
    try:
        # The normal rollout client retries transient failures for up to a
        # minute.  Abort is a control-plane operation and must not inherit
        # that unbounded wait while the training step is being torn down.
        await asyncio.wait_for(
            post(f"{url}/abort_request", {"abort_all": True}, max_retries=1),
            timeout=ABORT_HTTP_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning(f"Failed to abort SGLang server at {url}: {e}")


async def _get_server_num_requests(url: str) -> int:
    try:
        return num_requests_from_load(await asyncio.wait_for(get(f"{url}/v1/loads?include=core"), timeout=ABORT_HTTP_TIMEOUT_SECONDS))
    except Exception as e:
        logger.warning(f"Failed to get SGLang server load from {url} via /v1/loads: {e}; trying /get_load")
        try:
            return num_requests_from_load(await asyncio.wait_for(get(f"{url}/get_load"), timeout=ABORT_HTTP_TIMEOUT_SECONDS))
        except asyncio.TimeoutError as timeout_error:
            raise RuntimeError(f"Timed out getting SGLang server load from {url}") from timeout_error


async def abort_server_until_idle(
    url: str,
    retry_interval: float = ABORT_RETRY_INTERVAL_SECONDS,
    timeout: float = ABORT_TIMEOUT_SECONDS,
) -> bool:
    """Abort all requests, but never let a stuck SGLang server hang training."""
    attempt = 1
    deadline = time.monotonic() + max(0.0, timeout)
    while (remaining := deadline - time.monotonic()) > 0:
        logger.info(f"Abort request for SGLang server {url}")
        try:
            await asyncio.wait_for(_abort_server_once(url), timeout=remaining)
        except asyncio.TimeoutError:
            break

        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            num_requests = await asyncio.wait_for(_get_server_num_requests(url), timeout=remaining)
        except asyncio.TimeoutError:
            break
        except Exception as e:
            logger.warning(f"Failed to get SGLang server load from {url}: {e}")
            await asyncio.sleep(min(retry_interval, max(0.0, deadline - time.monotonic())))
            attempt += 1
            continue

        if num_requests <= 0:
            return True

        logger.info(f"SGLang server {url} still has {num_requests} requests after abort attempt {attempt}; " f"retrying in {retry_interval} seconds.")
        await asyncio.sleep(min(retry_interval, max(0.0, deadline - time.monotonic())))
        attempt += 1

    logger.error(
        "SGLang server %s still has requests after abort deadline (%.1fs); " "continuing so the rollout cannot hang indefinitely.",
        url,
        timeout,
    )
    return False


async def abort_servers_until_idle(urls: list[str]) -> bool:
    results = await asyncio.gather(*(abort_server_until_idle(url) for url in urls))
    return all(results)
