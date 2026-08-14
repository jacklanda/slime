"""Bounded, crash-recovering process isolation for local MCP tool calls."""

from __future__ import annotations

import concurrent.futures
import atexit
import copy
import json
import multiprocessing
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from slime.rollout.failure_types import MCPLeaseTimeout


_worker_toolsets: dict[str, Any] = {}
_process_context: multiprocessing.context.BaseContext | None = None
_process_context_lock = threading.Lock()


def _worker_toolset(task: dict[str, Any]):
    os.environ["SLIME_LOCAL_MCP_PROCESS_WORKER"] = "1"
    from .env import LocalMCPToolset

    key = str(task.get("tools_py") or task.get("data_root"))
    toolset = _worker_toolsets.get(key)
    if toolset is None:
        toolset = LocalMCPToolset(task)
        _worker_toolsets[key] = toolset
    return toolset


def _worker_call(task: dict[str, Any], name: str, arguments: dict[str, Any], relax_empty: bool) -> dict[str, Any]:
    toolset = _worker_toolset(task)
    result = toolset.call_raw(name, arguments, relax_empty=relax_empty)
    # Keep the IPC contract stable even when generated tools return local
    # enums, dataclasses, or other objects that multiprocessing cannot pickle.
    result = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    return {"result": result, "call_info": dict(toolset.last_call_info)}


def _worker_describe(task: dict[str, Any]) -> dict[str, Any]:
    toolset = _worker_toolset(task)
    return {
        "schemas": toolset.schemas(),
        "load_error": toolset.load_error,
        "load_warning": toolset.load_warning,
        "tool_aliases": dict(toolset.tool_aliases),
        "tool_names": list(toolset.tools),
    }


def _worker_verify(task: dict[str, Any], verification_code: str, answer: Any) -> dict[str, Any]:
    toolset = _worker_toolset(task)
    workspace = Path(str(task.get("data_root") or task.get("tools_py"))).resolve()
    if workspace.is_file():
        workspace = workspace.parent
    previous_cwd = Path.cwd()
    previous_base_dir = os.environ.get("MCP_SERVER_BASE_DIR")
    try:
        os.chdir(workspace)
        os.environ["MCP_SERVER_BASE_DIR"] = str(workspace)
        namespace: dict[str, Any] = {}
        exec(verification_code, namespace)
        verify = namespace.get("verify")
        if not callable(verify):
            return {"has_verifier": False, "result": None}
        result = verify(toolset, answer)
    finally:
        if previous_base_dir is None:
            os.environ.pop("MCP_SERVER_BASE_DIR", None)
        else:
            os.environ["MCP_SERVER_BASE_DIR"] = previous_base_dir
        os.chdir(previous_cwd)
    result = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    return {"has_verifier": True, "result": result}


def _worker_ping() -> int:
    from .env import LocalMCPToolset  # noqa: F401

    return os.getpid()


def _default_workers() -> int:
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(32, max(1, available // 2)))


def _get_process_context() -> multiprocessing.context.BaseContext:
    global _process_context
    with _process_context_lock:
        if _process_context is None:
            method = os.environ.get(
                "SLIME_LOCAL_MCP_PROCESS_START_METHOD",
                "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn",
            )
            context = multiprocessing.get_context(method)
            if method == "forkserver":
                # The fork server is a clean, single-threaded template. Preload
                # the heavy MCP runtime once, then fork a fresh process for
                # every operation without inheriting the rollout/Ray process.
                context.set_forkserver_preload(["slime.rollout.fused_agent.env"])
            _process_context = context
        return _process_context


class _ProcessShard:
    def __init__(self, index: int):
        self.index = index
        self.lock = threading.Lock()
        self.executor: concurrent.futures.ProcessPoolExecutor | None = None

    def _start(self) -> concurrent.futures.ProcessPoolExecutor:
        if self.executor is None:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=_get_process_context(),
            )
        return self.executor

    def call(
        self,
        task: dict[str, Any],
        name: str,
        arguments: dict[str, Any],
        relax_empty: bool,
        timeout: float,
    ) -> dict[str, Any]:
        with self.lock:
            executor = self._start()
            future = executor.submit(_worker_call, task, name, arguments, relax_empty)
            try:
                return future.result(timeout=timeout)
            except BaseException:
                self._terminate_locked()
                raise

    def submit(self, fn, *args) -> concurrent.futures.Future:
        with self.lock:
            return self._start().submit(fn, *args)

    def _terminate_locked(self) -> None:
        executor, self.executor = self.executor, None
        if executor is None:
            return
        # Python 3.12 has no public terminate_workers API. Terminate only this
        # one failed shard before non-blocking shutdown; healthy shards remain.
        for process in list(getattr(executor, "_processes", {}).values()):
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)

    def terminate(self) -> None:
        with self.lock:
            self._terminate_locked()

    def close(self) -> None:
        with self.lock:
            self._terminate_locked()


class LocalMCPProcessPool:
    def __init__(self, workers: int | None = None):
        configured_value = os.environ.get("SLIME_LOCAL_MCP_PROCESS_WORKERS", "").strip()
        configured = workers if workers is not None else (int(configured_value) if configured_value else _default_workers())
        self.workers = max(1, configured)
        self.shards = [_ProcessShard(index) for index in range(self.workers)]
        self._condition = threading.Condition()
        self._free = deque(range(self.workers))
        self._closed = False
        self._describe_lock = threading.Lock()
        self._describe_cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._describe_inflight: dict[tuple[Any, ...], concurrent.futures.Future] = {}

    def _acquire(self, timeout: float | None = None) -> int:
        with self._condition:
            if self._closed:
                raise RuntimeError("Local MCP process pool is closed")
            deadline = None if timeout is None else time.monotonic() + timeout
            while not self._free and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise MCPLeaseTimeout(f"No local MCP process worker available after {timeout:.1f}s")
                self._condition.wait(timeout=remaining)
            if self._closed:
                raise RuntimeError("Local MCP process pool is closed")
            return self._free.popleft()

    def _release(self, index: int) -> None:
        try:
            # Generated modules and verifiers may mutate arbitrary globals.
            # End the operation process before making its shard available again.
            self.shards[index].terminate()
        finally:
            with self._condition:
                if not self._closed:
                    self._free.append(index)
                self._condition.notify_all()

    def warm(self) -> list[int]:
        futures = [shard.submit(_worker_ping) for shard in self.shards]
        timeout = max(1.0, float(os.environ.get("SLIME_LOCAL_MCP_PROCESS_WARM_TIMEOUT", "60")))
        pids = []
        for shard, future in zip(self.shards, futures, strict=True):
            try:
                pids.append(future.result(timeout=timeout))
            except BaseException:
                shard.terminate()
                raise
        return pids

    def call(
        self,
        task: dict[str, Any],
        name: str,
        arguments: dict[str, Any],
        *,
        relax_empty: bool,
    ) -> dict[str, Any]:
        timeout = max(1.0, float(os.environ.get("SLIME_LOCAL_MCP_PROCESS_TIMEOUT", "120")))
        lease_timeout = max(0.1, float(os.environ.get("SLIME_LOCAL_MCP_LEASE_TIMEOUT", str(timeout))))
        shard_index = self._acquire(timeout=lease_timeout)
        try:
            return self.shards[shard_index].call(task, name, arguments, relax_empty, timeout)
        finally:
            self._release(shard_index)

    @staticmethod
    def _describe_cache_key(task: dict[str, Any]) -> tuple[Any, ...]:
        tools_value = task.get("tools_py")
        data_root_value = task.get("data_root")
        tools_path = Path(str(tools_value)).resolve() if tools_value else None
        data_root = Path(str(data_root_value)).resolve() if data_root_value else None
        try:
            stat = tools_path.stat() if tools_path is not None else None
        except OSError:
            stat = None
        return (
            str(data_root) if data_root is not None else None,
            str(tools_path) if tools_path is not None else None,
            stat.st_mtime_ns if stat is not None else None,
            stat.st_size if stat is not None else None,
        )

    def _describe_uncached(self, task: dict[str, Any]) -> dict[str, Any]:
        timeout = max(1.0, float(os.environ.get("SLIME_LOCAL_MCP_PROCESS_TIMEOUT", "120")))
        lease_timeout = max(0.1, float(os.environ.get("SLIME_LOCAL_MCP_LEASE_TIMEOUT", str(timeout))))
        shard_index = self._acquire(timeout=lease_timeout)
        try:
            return self.shards[shard_index].submit(_worker_describe, task).result(timeout=timeout)
        finally:
            self._release(shard_index)

    def describe(self, task: dict[str, Any]) -> dict[str, Any]:
        key = self._describe_cache_key(task)
        with self._describe_lock:
            cached = self._describe_cache.get(key)
            if cached is not None:
                self._describe_cache.move_to_end(key)
                return copy.deepcopy(cached)
            future = self._describe_inflight.get(key)
            if future is None:
                future = concurrent.futures.Future()
                self._describe_inflight[key] = future
                leader = True
            else:
                leader = False

        if not leader:
            return copy.deepcopy(future.result())

        try:
            description = self._describe_uncached(task)
        except BaseException as exc:
            with self._describe_lock:
                self._describe_inflight.pop(key, None)
                future.set_exception(exc)
            raise

        with self._describe_lock:
            if not description.get("load_error"):
                self._describe_cache[key] = copy.deepcopy(description)
                self._describe_cache.move_to_end(key)
                max_entries = max(1, int(os.environ.get("SLIME_LOCAL_MCP_DESCRIBE_CACHE_SIZE", "4096")))
                while len(self._describe_cache) > max_entries:
                    self._describe_cache.popitem(last=False)
            self._describe_inflight.pop(key, None)
            future.set_result(copy.deepcopy(description))
        return copy.deepcopy(description)

    def verify(self, task: dict[str, Any], verification_code: str, answer: Any) -> dict[str, Any]:
        timeout = max(1.0, float(os.environ.get("SLIME_LOCAL_MCP_PROCESS_TIMEOUT", "120")))
        lease_timeout = max(0.1, float(os.environ.get("SLIME_LOCAL_MCP_LEASE_TIMEOUT", str(timeout))))
        shard_index = self._acquire(timeout=lease_timeout)
        try:
            return self.shards[shard_index].submit(
                _worker_verify,
                task,
                verification_code,
                answer,
            ).result(timeout=timeout)
        finally:
            self._release(shard_index)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._free.clear()
            self._condition.notify_all()
        for shard in self.shards:
            shard.close()
        with self._describe_lock:
            self._describe_cache.clear()


_pool: LocalMCPProcessPool | None = None
_pool_lock = threading.Lock()


def get_local_mcp_process_pool() -> LocalMCPProcessPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = LocalMCPProcessPool()
        return _pool


def close_local_mcp_process_pool() -> None:
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.close()


atexit.register(close_local_mcp_process_pool)
