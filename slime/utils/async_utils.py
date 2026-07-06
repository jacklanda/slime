import asyncio
import logging
import os
import threading

logger = logging.getLogger(__name__)

__all__ = ["get_async_loop", "run"]


def _new_event_loop() -> asyncio.AbstractEventLoop:
    # The rollout event loop is a known CPU bottleneck under high trajectory
    # concurrency; uvloop's faster selector/protocol machinery buys headroom
    # for free. Opt out with SLIME_ASYNC_UVLOOP=0.
    if os.environ.get("SLIME_ASYNC_UVLOOP", "1").lower() in {"1", "true", "yes", "y", "on"}:
        try:
            import uvloop
        except ImportError:
            pass
        else:
            logger.info("async_utils: using uvloop for the background event loop")
            return uvloop.new_event_loop()
    return asyncio.new_event_loop()


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self.loop = _new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
