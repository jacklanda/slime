from __future__ import annotations

import io
import os
import re
import shlex
import tarfile
import threading
import uuid
from pathlib import PurePosixPath
from typing import Any


DEFAULT_ET_CWD = "/home/user"
DEFAULT_SWE_CWD = "/testbed"
REWARD_FILE = "/logs/verifier/reward.txt"

_CLIENT_LOCK = threading.Lock()
_SHARED_DOCKER_CLIENT = None


def is_et_task(task: dict[str, Any]) -> bool:
    if task.get("data_source") == "endless_terminals":
        return True
    return bool(
        task.get("docker_image")
        and task.get("final_state_test")
        and task.get("instruction")
        and not task.get("repo_name")
        and not task.get("commit_hash")
    )


def get_docker_client():
    global _SHARED_DOCKER_CLIENT
    if _SHARED_DOCKER_CLIENT is not None:
        return _SHARED_DOCKER_CLIENT
    import docker

    with _CLIENT_LOCK:
        if _SHARED_DOCKER_CLIENT is None:
            base_url = os.environ.get("DOCKER_HOST") or None
            api_version = os.environ.get("DOCKER_API_VERSION", "auto")
            if base_url:
                _SHARED_DOCKER_CLIENT = docker.DockerClient(
                    base_url=base_url,
                    version=api_version,
                    timeout=120,
                    max_pool_size=int(os.environ.get("SLIME_FUSED_DOCKER_POOL_SIZE", "64")),
                )
            else:
                _SHARED_DOCKER_CLIENT = docker.from_env(timeout=120)
    return _SHARED_DOCKER_CLIENT


def tool_schemas(mode: str) -> list[dict]:
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Run a non-interactive shell command inside the task container.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string", "description": "Command to execute."},
                        "command": {"type": "string", "description": "Alias for cmd."},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "file_editor",
                "description": "View, create, replace, or insert text in a file inside the task container.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "One of view, create, str_replace, insert."},
                        "path": {"type": "string", "description": "Absolute or working-directory relative file path."},
                        "file_text": {"type": "string", "description": "Content for create."},
                        "old_str": {"type": "string", "description": "Exact text to replace."},
                        "new_str": {"type": "string", "description": "Replacement text."},
                        "insert_line": {"type": "integer", "description": "Line number before which text is inserted."},
                    },
                    "required": ["command", "path"],
                },
            },
        },
    ]
    if mode == "cli":
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the repository with grep.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text or regex to search for."},
                            "path": {"type": "string", "description": "Directory or file to search."},
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    schemas.append(
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the task and trigger grading.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Use submit."},
                        "result": {"type": "string", "description": "Optional final note."},
                    },
                    "required": ["command"],
                },
            },
        }
    )
    return schemas


class DockerTaskEnvironment:
    def __init__(self, task: dict[str, Any], *, mode: str):
        self.task = task
        self.mode = mode
        self.client = None
        self.container = None
        self.container_name = ""
        self.cwd = DEFAULT_ET_CWD if mode == "et" else DEFAULT_SWE_CWD
        self.step_timeout = int(task.get("step_timeout") or os.environ.get("SLIME_FUSED_DOCKER_STEP_TIMEOUT", "90"))
        self.reward_timeout = int(
            task.get("reward_timeout")
            or task.get("verifier_timeout_sec")
            or os.environ.get("SLIME_FUSED_DOCKER_REWARD_TIMEOUT", "300")
        )
        self.reward_debug: dict[str, Any] = {}
        self._closed = False

    def reset(self) -> tuple[str, dict[str, Any]]:
        image = self.task.get("docker_image")
        if not image:
            raise ValueError("Docker fused task is missing docker_image")
        self.client = get_docker_client()
        self.client.images.get(image)

        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", str(self.task.get("task_id") or self.task.get("instance_id") or "fused"))[:40]
        self.container_name = f"slime-fused-{self.mode}-{safe_id}-{uuid.uuid4().hex[:10]}"
        kwargs = {
            "image": image,
            "command": ["sleep", "infinity"],
            "name": self.container_name,
            "detach": True,
            "tty": False,
            "stdin_open": False,
            "network_mode": os.environ.get("SLIME_FUSED_DOCKER_NETWORK", "host"),
            "auto_remove": False,
            "labels": {"slime": "fused_agent", "slime_task_mode": self.mode},
        }
        memory_mb = int(self.task.get("memory_mb") or 0)
        cpus = float(self.task.get("cpus") or 0)
        if memory_mb > 0:
            kwargs["mem_limit"] = f"{memory_mb}m"
        if cpus > 0:
            kwargs["nano_cpus"] = int(cpus * 1e9)
        self.container = self.client.containers.run(**kwargs)
        try:
            self.container.reload()
        except Exception:
            pass

        if self.mode == "et":
            question = str(self.task.get("instruction") or self.task.get("question") or "")
            info = {"task_type": "et", "cwd": self.cwd, "container_name": self.container_name}
        else:
            question = str(
                self.task.get("problem_statement")
                or self.task.get("instruction")
                or self.task.get("question")
                or self.task.get("prompt")
                or ""
            )
            info = {"task_type": "cli", "cwd": self.cwd, "container_name": self.container_name}
        env_context = self._environment_context()
        if env_context:
            info["env_context"] = env_context
        return question, info

    def schemas(self) -> list[dict]:
        return tool_schemas(self.mode)

    def step(self, name: str, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        if self.container is None:
            return "Error: Docker container is not initialized.", 0.0, True, {"termination_reason": "ENV_INIT_ERROR"}
        if name in {"finish", "submit"}:
            reward = self.compute_final_reward()
            return "Submitted.", reward, True, {"reward_debug": self.reward_debug, "cwd": self.cwd}
        if name == "str_replace_editor":
            name = "file_editor"
        if name == "execute_bash":
            return self._execute_bash(args)
        if name == "file_editor":
            return self._file_editor(args)
        if name == "search" and self.mode == "cli":
            return self._search(args)
        return f"Error: tool {name} is not available for task mode {self.mode}", 0.0, False, {"cwd": self.cwd}

    def _execute_bash(self, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        command = str(args.get("cmd") or args.get("command") or "")
        if not command:
            return "Error: execute_bash requires cmd.", 0.0, False, {"cwd": self.cwd}
        rc, output = self._exec(command, timeout=self.step_timeout)
        self._update_cwd(command)
        if rc == 124:
            output = f"The command timed out after {self.step_timeout}s.\n{output}"
        return output or f"Command exited with code {rc}.", 0.0, False, {"cwd": self.cwd, "exit_code": rc}

    def _file_editor(self, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        command = str(args.get("command") or "")
        path = str(args.get("path") or "")
        if not command or not path:
            return "Error: file_editor requires command and path.", 0.0, False, {"cwd": self.cwd}
        abs_path = self._abs_path(path)
        try:
            if command == "view":
                rc, out = self._exec(f"sed -n '1,240p' {shlex.quote(abs_path)}", timeout=self.step_timeout)
                return out if rc == 0 else out or f"Error: view failed for {abs_path}", 0.0, False, {"cwd": self.cwd, "exit_code": rc}
            if command == "create":
                content = str(args.get("file_text") or args.get("content") or "")
                self._put_text(abs_path, content)
                return f"Created {abs_path}.", 0.0, False, {"cwd": self.cwd}
            if command == "str_replace":
                old = str(args.get("old_str") or "")
                new = str(args.get("new_str") or "")
                text = self._read_text(abs_path)
                count = text.count(old) if old else 0
                if count != 1:
                    return f"Error: old_str matched {count} occurrences in {abs_path}; expected exactly one.", 0.0, False, {"cwd": self.cwd}
                self._put_text(abs_path, text.replace(old, new, 1))
                return f"Replaced text in {abs_path}.", 0.0, False, {"cwd": self.cwd}
            if command == "insert":
                insert = str(args.get("new_str") or args.get("file_text") or "")
                line = int(args.get("insert_line") or 1)
                lines = self._read_text(abs_path).splitlines(keepends=True)
                idx = max(0, min(len(lines), line - 1))
                lines.insert(idx, insert if insert.endswith("\n") else insert + "\n")
                self._put_text(abs_path, "".join(lines))
                return f"Inserted text into {abs_path}.", 0.0, False, {"cwd": self.cwd}
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}", 0.0, False, {"cwd": self.cwd}
        return f"Error: unsupported file_editor command {command}.", 0.0, False, {"cwd": self.cwd}

    def _search(self, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        query = str(args.get("query") or args.get("pattern") or "")
        path = str(args.get("path") or ".")
        if not query:
            return "Error: search requires query.", 0.0, False, {"cwd": self.cwd}
        cmd = f"grep -RIn --exclude-dir=.git -- {shlex.quote(query)} {shlex.quote(path)} | head -200"
        rc, output = self._exec(cmd, timeout=self.step_timeout)
        if rc not in (0, 1):
            return output or f"Search failed with code {rc}.", 0.0, False, {"cwd": self.cwd, "exit_code": rc}
        return output or "No matches found.", 0.0, False, {"cwd": self.cwd, "exit_code": rc}

    def compute_final_reward(self) -> float:
        if self.container is None:
            self.reward_debug = {"type": self.mode, "reward": 0.0, "verifier_error": "container_missing"}
            return 0.0
        if self.mode == "et":
            return self._compute_et_reward()
        return self._compute_cli_reward()

    def _compute_et_reward(self) -> float:
        try:
            self._exec("mkdir -p /tests /logs/verifier", timeout=15)
            self._put_text("/tests/test_final_state.py", str(self.task.get("final_state_test") or ""))
            self._put_text("/tests/test.sh", str(self.task.get("test_script") or ""), mode=0o755)
            rc, output = self._exec("bash /tests/test.sh", timeout=self.reward_timeout)
            cat_rc, reward_text = self._exec(f"cat {REWARD_FILE}", timeout=15)
            reward_text = (reward_text or "").strip()
            reward = 1.0 if cat_rc == 0 and reward_text == "1" else 0.0
            self.reward_debug = {
                "type": "endless_terminals",
                "reward": reward,
                "resolved": reward >= 1.0,
                "reward_source": "verifier_reward_txt",
                "test_script_exit_code": rc,
                "reward_file": reward_text,
                "log_head": output[:1500],
                "log_tail": output[-1500:],
            }
            return reward
        except Exception as e:
            self.reward_debug = {"type": "endless_terminals", "reward": 0.0, "verifier_error": f"{type(e).__name__}: {e}"}
            return 0.0

    def _compute_cli_reward(self) -> float:
        eval_script = self.task.get("eval_script")
        verifier = self.task.get("verifier")
        try:
            if eval_script:
                self._put_text("/tmp/slime_fused_run_tests.sh", str(eval_script), mode=0o755)
                rc, output = self._exec("bash /tmp/slime_fused_run_tests.sh", timeout=self.reward_timeout)
                omnigril = _parse_omnigril_exit_code(output)
                reward = 1.0 if (omnigril == 0 or (omnigril is None and rc == 0)) else 0.0
                self.reward_debug = {
                    "type": "gemcli",
                    "reward": reward,
                    "resolved": reward >= 1.0,
                    "reward_source": "eval_script",
                    "exit_code": rc,
                    "omnigril_exit_code": omnigril,
                    "log_head": output[:1500],
                    "log_tail": output[-1500:],
                }
                return reward
            if isinstance(verifier, dict) and verifier.get("verification_code"):
                namespace: dict[str, Any] = {}
                exec(str(verifier["verification_code"]), namespace)
                verify = namespace.get("verify")
                if callable(verify):
                    result = verify(self, None)
                    reward = float(result.get("score", 1.0 if result.get("passed") else 0.0)) if isinstance(result, dict) else float(result)
                    self.reward_debug = {"type": "cli", "reward": reward, "reward_source": "verification_code", "verifier": result}
                    return reward
        except Exception as e:
            self.reward_debug = {"type": "cli", "reward": 0.0, "verifier_error": f"{type(e).__name__}: {e}"}
            return 0.0
        self.reward_debug = {"type": "cli", "reward": 0.0, "verifier_error": "eval_script_missing"}
        return 0.0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.container is None:
            return
        if os.environ.get("SLIME_FUSED_KEEP_CONTAINER", "0") == "1":
            return
        try:
            self.container.stop(timeout=2)
        except Exception:
            pass
        try:
            self.container.remove(force=True)
        except Exception:
            pass
        self.container = None

    def _environment_context(self) -> str:
        if self.container is None:
            return ""
        rc, out = self._exec("pwd; find . -maxdepth 2 -mindepth 1 | sed 's#^./##' | sort | head -200", timeout=20)
        return out[:12000] if rc == 0 else ""

    def _exec(self, command: str, *, timeout: int = 60) -> tuple[int, str]:
        if self.container is None:
            return -1, "container missing"
        wrapped = f"timeout {int(timeout)} sh -lc {shlex.quote(command)}"
        res = self.container.exec_run(
            cmd=["/bin/sh", "-lc", wrapped],
            workdir=self.cwd,
            stdout=True,
            stderr=True,
            demux=False,
        )
        out = res.output or b""
        if isinstance(out, tuple):
            out = (out[0] or b"") + (out[1] or b"")
        text = out.decode("utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;]*m|\r", "", text)
        return int(res.exit_code if res.exit_code is not None else -1), text

    def _put_text(self, container_path: str, content: str, mode: int = 0o644) -> None:
        if self.container is None:
            raise RuntimeError("container missing")
        parent = str(PurePosixPath(container_path).parent)
        self._exec(f"mkdir -p {shlex.quote(parent)}", timeout=15)
        data = content.encode("utf-8")
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            info = tarfile.TarInfo(name=PurePosixPath(container_path).name)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
        stream.seek(0)
        self.container.put_archive(parent, stream.read())

    def _read_text(self, container_path: str) -> str:
        if self.container is None:
            raise RuntimeError("container missing")
        bits, _stat = self.container.get_archive(container_path)
        data = b"".join(bits)
        stream = io.BytesIO(data)
        with tarfile.open(fileobj=stream, mode="r") as tar:
            member = next((m for m in tar.getmembers() if m.isfile()), None)
            if member is None:
                return ""
            f = tar.extractfile(member)
            return (f.read() if f else b"").decode("utf-8", errors="replace")

    def _abs_path(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return str(PurePosixPath(self.cwd) / path)

    def _update_cwd(self, command: str) -> None:
        match = re.search(r"(?:^|[;&]\s*)cd\s+([^;&|]+)\s*$", command.strip())
        if not match:
            return
        target = match.group(1).strip().strip("'\"")
        if target:
            self.cwd = self._abs_path(target)


def _parse_omnigril_exit_code(output: str) -> int | None:
    matches = re.findall(r"OMNIGRIL_EXIT_CODE\s*=\s*(-?\d+)", output or "")
    return int(matches[-1]) if matches else None
