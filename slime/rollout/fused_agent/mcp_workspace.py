"""Filesystem workspaces for local MCP tasks.

MCP task assets are immutable inputs, but many generated tools use SQLite,
JSON, or other files relative to the task sandbox. By default, a workspace is
copied for each trajectory. Task-scoped workspaces can be enabled explicitly
when trajectories should reuse the same mutable task state.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_ROOT = "/share/nlp/share/gem/runs"
_COPY_WORKERS_DEFAULT = 32
_copy_semaphore = threading.BoundedSemaphore(
    max(1, int(os.environ.get("SLIME_MCP_ENV_COPY_CONCURRENCY", str(_COPY_WORKERS_DEFAULT))))
)
_task_workspace_lock = threading.Lock()
_task_workspaces: dict[tuple[int, Path, Path], Future[tuple[Path, str]]] = {}


def _resolve_input_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        Path.cwd() / "experiments" / "fused" / "assets" / path.name,
        Path.cwd().parent / "rllm" / path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def is_local_mcp_task(task: dict[str, Any]) -> bool:
    transport = str(task.get("mcp_transport") or "").strip().lower()
    source = str(task.get("data_source") or task.get("benchmark") or "").strip().lower()
    normalized_source = source.replace("-", "_").replace(" ", "_")
    return bool(
        (task.get("tools_py") or task.get("data_root"))
        and transport != "atlas"
        and normalized_source != "mcp_atlas"
        and not task.get("mcp_atlas_eval")
    )


def _workspace_root(args_root: str | os.PathLike[str] | None) -> Path:
    configured = args_root or os.environ.get("SLIME_MCP_ENV_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("RUN_ROOT", _DEFAULT_ROOT)) / "cache" / "mcp_envs"


def _copy_tree(source: Path, destination: Path) -> None:
    with _copy_semaphore:
        destination.mkdir(parents=True, exist_ok=False)
        rsync = shutil.which("rsync")
        if rsync:
            subprocess.run(
                [
                    rsync,
                    "-a",
                    "--exclude=/logs/",
                    "--exclude=/runs/",
                    "--exclude=/__pycache__/",
                    "--exclude=**/__pycache__/**",
                    "--",
                    f"{source}/",
                    f"{destination}/",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return

        def ignore(_directory: str, names: list[str]) -> set[str]:
            return {name for name in names if name in {"logs", "runs", "__pycache__"}}

        shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True, ignore=ignore)


@dataclass
class MCPWorkspace:
    """A handle to a copied MCP sandbox."""

    path: Path
    source_root: Path
    task_id: str
    root: Path
    task_scoped: bool = False
    _closed: bool = False

    @property
    def tools_path(self) -> Path:
        return self.path / "tools.py"

    def apply(self, task: dict[str, Any]) -> dict[str, Any]:
        rewritten = dict(task)
        source_tools = _resolve_input_path(task.get("tools_py"))
        source_root = _resolve_input_path(task.get("data_root")) or (source_tools.parent if source_tools else None)
        if source_root is None or not source_root.is_dir():
            raise FileNotFoundError(f"MCP task has no readable data_root/tools_py directory: {task!r}")
        source_root = source_root.resolve()
        if source_root != self.source_root.resolve():
            raise ValueError(f"MCP workspace source mismatch: expected {self.source_root}, got {source_root}")
        rewritten["data_root"] = str(self.path)
        if source_tools is not None:
            source_tools = source_tools.resolve()
            try:
                relative_tools = source_tools.relative_to(source_root)
            except ValueError as exc:
                raise ValueError(f"tools_py={source_tools} is outside data_root={source_root}") from exc
            rewritten["tools_py"] = str(self.path / relative_tools)
        rewritten["mcp_env_id"] = self.task_id
        rewritten["mcp_env_source_root"] = str(source_root)
        return rewritten

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.task_scoped:
            _remove_workspace(self.path, self.root)


def _remove_workspace(path: Path, root: Path) -> None:
    if root not in path.parents:
        raise RuntimeError(f"Refusing to remove MCP workspace outside configured root: {path}")
    shutil.rmtree(path, ignore_errors=True)
    try:
        path.parent.rmdir()
    except OSError:
        pass


def _prepare_task_workspace(source_root: Path, configured_root: Path, task_root: Path) -> tuple[Path, str]:
    key = (os.getpid(), configured_root, source_root)
    with _task_workspace_lock:
        future = _task_workspaces.get(key)
        if future is None:
            future = Future()
            _task_workspaces[key] = future
            create = True
        else:
            create = False

    if not create:
        return future.result()

    task_id = f"shared-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    destination = task_root / task_id
    staging = task_root / f".tmp-{task_id}"
    try:
        _copy_tree(source_root, staging)
        staging.replace(destination)
        result = (destination, task_id)
        future.set_result(result)
        return result
    except BaseException as exc:
        shutil.rmtree(staging, ignore_errors=True)
        future.set_exception(exc)
        with _task_workspace_lock:
            if _task_workspaces.get(key) is future:
                del _task_workspaces[key]
        raise


def cleanup_task_workspaces() -> None:
    """Remove task-scoped workspaces owned by this process."""

    pid = os.getpid()
    with _task_workspace_lock:
        owned = [(key, future) for key, future in _task_workspaces.items() if key[0] == pid]
        for key, _future in owned:
            del _task_workspaces[key]

    for (_owner_pid, configured_root, _source_root), future in owned:
        if not future.done() or future.cancelled():
            continue
        try:
            path, _task_id = future.result()
        except BaseException:
            continue
        _remove_workspace(path, configured_root)


atexit.register(cleanup_task_workspaces)


def prepare_mcp_workspace(
    task: dict[str, Any],
    session_id: str,
    root: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], MCPWorkspace]:
    """Prepare a trajectory- or task-scoped local MCP workspace."""

    if not is_local_mcp_task(task):
        raise ValueError("prepare_mcp_workspace called for a non-local MCP task")
    source_tools = _resolve_input_path(task.get("tools_py"))
    source_root = _resolve_input_path(task.get("data_root")) or (source_tools.parent if source_tools else None)
    if source_root is None or not source_root.is_dir():
        raise FileNotFoundError(f"MCP task has no readable data_root/tools_py directory: {task!r}")
    source_root = source_root.resolve()
    configured_root = _workspace_root(root).resolve()
    task_key = hashlib.sha256(str(source_root).encode("utf-8")).hexdigest()[:20]
    task_root = configured_root / f"task-{task_key}"
    task_root.mkdir(parents=True, exist_ok=True)
    scope = os.environ.get("SLIME_MCP_WORKSPACE_SCOPE", "trajectory").strip().lower()
    if scope not in {"trajectory", "task"}:
        raise ValueError(f"SLIME_MCP_WORKSPACE_SCOPE must be 'trajectory' or 'task', got {scope!r}")
    if scope == "task":
        destination, task_id = _prepare_task_workspace(source_root, configured_root, task_root)
        workspace = MCPWorkspace(destination, source_root, task_id, configured_root, task_scoped=True)
        return workspace.apply(task), workspace

    task_id = f"{session_id}-{uuid.uuid4().hex[:12]}"
    destination = task_root / task_id
    staging = task_root / f".tmp-{task_id}"
    try:
        _copy_tree(source_root, staging)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    workspace = MCPWorkspace(destination, source_root, task_id, configured_root)
    return workspace.apply(task), workspace


def cleanup_mcp_workspace(workspace: MCPWorkspace | None) -> None:
    if workspace is not None:
        workspace.close()
