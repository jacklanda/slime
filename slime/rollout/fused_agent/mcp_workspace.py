"""Per-trajectory filesystem workspaces for local MCP tasks.

MCP task assets are immutable inputs, but many generated tools use SQLite,
JSON, or other files relative to the task sandbox.  A workspace is copied
before a trajectory starts and removed after it finishes, so concurrent
trajectories never share mutable task state.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_ROOT = "/share/nlp/share/gem/runs"
_COPY_WORKERS_DEFAULT = 32
_copy_semaphore = threading.BoundedSemaphore(
    max(1, int(os.environ.get("SLIME_MCP_ENV_COPY_CONCURRENCY", str(_COPY_WORKERS_DEFAULT))))
)


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
    """A single trajectory's copied MCP sandbox."""

    path: Path
    source_root: Path
    task_id: str
    root: Path
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
        if self.root not in self.path.parents:
            raise RuntimeError(f"Refusing to remove MCP workspace outside configured root: {self.path}")
        shutil.rmtree(self.path, ignore_errors=True)
        try:
            self.path.parent.rmdir()
        except OSError:
            pass


def prepare_mcp_workspace(
    task: dict[str, Any],
    session_id: str,
    root: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], MCPWorkspace]:
    """Copy one local MCP task into a unique trajectory directory."""

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
