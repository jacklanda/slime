import logging
import threading
from collections.abc import Callable

import ray


logger = logging.getLogger(__name__)


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, server_group, args, on_engine_failure: Callable[[object, int], None] | None = None):
        self._server_group = server_group
        self._on_engine_failure = on_engine_failure

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active
        # pause() is a lifecycle barrier: memory must not be released while a
        # generation health request is still executing.
        self._check_lock = threading.RLock()

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._server_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking and drain any check already in flight."""
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False
        # RLock keeps this safe when an engine-failure callback pauses the
        # monitor from the monitor thread itself.
        with self._check_lock:
            pass

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        failed_engine_ids = []
        with self._check_lock:
            for rollout_engine_id, engine in enumerate(self._server_group.engines):
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                if self._pause_event is not None and self._pause_event.is_set():
                    break
                if not self._check_engine_health(rollout_engine_id, engine, notify_failure=False):
                    failed_engine_ids.append(rollout_engine_id)

        # Recovery can pause every monitor. Run callbacks after releasing this
        # monitor's lock so simultaneous failures cannot deadlock each other.
        for rollout_engine_id in failed_engine_ids:
            self._notify_engine_failure(rollout_engine_id)

    def _check_engine_health(self, rollout_engine_id, engine, *, notify_failure: bool = True) -> bool:
        if engine is None:
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return True

        if not self._server_group.generation_health_check_enabled:
            logger.info(
                "Skipping health check for engine %s because its server group is not generation-ready",
                rollout_engine_id,
            )
            return True

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            logger.error(
                f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing actor. Exception: {e}"
            )
            self._server_group.mark_engine_group_dead(rollout_engine_id)
            if notify_failure:
                self._notify_engine_failure(rollout_engine_id)
            return False
        else:
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")
            return True

    def _notify_engine_failure(self, rollout_engine_id: int) -> None:
        if self._on_engine_failure is not None:
            try:
                self._on_engine_failure(self._server_group, rollout_engine_id)
            except Exception:
                # Recovery is best effort; the next health check or the
                # normal weight-update recovery path can retry it.
                logger.exception(
                    "Automatic rollout engine reboost failed for engine %s",
                    rollout_engine_id,
                )
