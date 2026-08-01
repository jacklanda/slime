from __future__ import annotations

import asyncio
from collections import OrderedDict
import functools
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Callable

import aiohttp

from slime.rollout.rm_hub.f1 import normalize_answer

from .docker_env import DockerTaskEnvironment, is_et_task
from .parser import ToolCall, tool_schema
from .prompts import finish_schema, web_search_schema
from .search_gym import access_schema, search_schema

logger = logging.getLogger(__name__)

# One keepalive HTTP session per event loop for retrieval/summarize calls.
# The previous per-call ClientSession forced a fresh TCP handshake for every
# web_search; under a wave of hundreds of concurrent searches that both
# multiplied connection churn and, with a starved event loop, left hundreds
# of half-set-up connections whose request bodies were never sent.
_shared_http_session: aiohttp.ClientSession | None = None
_shared_http_session_loop: asyncio.AbstractEventLoop | None = None
_retrieval_runtime_loop: asyncio.AbstractEventLoop | None = None
_retrieval_semaphore: asyncio.Semaphore | None = None
_retrieval_cache: OrderedDict[str, Any] = OrderedDict()
_retrieval_inflight: dict[str, asyncio.Task] = {}
_atlas_http_session: aiohttp.ClientSession | None = None
_atlas_http_session_loop: asyncio.AbstractEventLoop | None = None
_atlas_tool_cache: dict[str, list[dict[str, Any]]] = {}
_atlas_tool_cache_lock = threading.Lock()
_ATLAS_MUTATING_TOOLS = {
    "airtable_create_field",
    "airtable_create_record",
    "airtable_create_table",
    "airtable_delete_record",
    "airtable_update_field",
    "airtable_update_record",
    "airtable_update_table",
    "anili_add_list_entry",
    "anili_delete_activity",
    "anili_delete_thread",
    "anili_favourite_anime",
    "anili_favourite_character",
    "anili_favourite_manga",
    "anili_favourite_staff",
    "anili_favourite_studio",
    "anili_follow_user",
    "anili_post_message_activity",
    "anili_post_text_activity",
    "anili_remove_list_entry",
    "anili_update_list_entry",
    "anili_update_user",
    "desktop-commander_create_directory",
    "desktop-commander_edit_block",
    "desktop-commander_force_terminate",
    "desktop-commander_give_feedback_to_desktop_commander",
    "desktop-commander_interact_with_process",
    "desktop-commander_kill_process",
    "desktop-commander_move_file",
    "desktop-commander_set_config_value",
    "desktop-commander_write_file",
    "filesystem_create_directory",
    "filesystem_edit_file",
    "filesystem_move_file",
    "filesystem_write_file",
    "git_git_add",
    "git_git_checkout",
    "git_git_commit",
    "git_git_create_branch",
    "git_git_reset",
    "github_add_issue_comment",
    "github_create_branch",
    "github_create_issue",
    "github_create_or_update_file",
    "github_create_pull_request",
    "github_create_pull_request_review_comment",
    "github_create_repository",
    "github_fork_repository",
    "github_merge_pull_request",
    "github_push_files",
    "github_update_issue",
    "github_update_pull_request",
    "github_update_pull_request_branch",
    "google-workspace_create_event",
    "google-workspace_delete_event",
    "google-workspace_modify_email",
    "google-workspace_send_email",
    "google-workspace_update_event",
    "lara-translate_add_translation",
    "lara-translate_create_memory",
    "lara-translate_delete_memory",
    "lara-translate_delete_translation",
    "lara-translate_import_tmx",
    "lara-translate_update_memory",
    "memory_add_observations",
    "memory_create_entities",
    "memory_create_relations",
    "memory_delete_entities",
    "memory_delete_observations",
    "memory_delete_relations",
    "mongodb_create-collection",
    "mongodb_create-index",
    "mongodb_delete-many",
    "mongodb_drop-collection",
    "mongodb_drop-database",
    "mongodb_insert-many",
    "mongodb_rename-collection",
    "mongodb_switch-connection",
    "mongodb_update-many",
    "notion_API-create-a-comment",
    "notion_API-create-a-database",
    "notion_API-delete-a-block",
    "notion_API-patch-block-children",
    "notion_API-patch-page",
    "notion_API-post-page",
    "notion_API-update-a-block",
    "notion_API-update-a-database",
    "slack_conversations_add_message",
}
_mcp_tool_cwd_lock = threading.RLock()
_warned_partial_mcp_modules: set[Path] = set()


def _get_shared_http_session() -> aiohttp.ClientSession:
    global _shared_http_session, _shared_http_session_loop
    loop = asyncio.get_running_loop()
    if _shared_http_session is None or getattr(_shared_http_session, "closed", False) or _shared_http_session_loop is not loop:
        _shared_http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120),
            connector=aiohttp.TCPConnector(
                limit=max(1, int(os.environ.get("RLLM_RETRIEVAL_CONCURRENCY", "160"))),
                limit_per_host=max(1, int(os.environ.get("RLLM_RETRIEVAL_CONCURRENCY", "160"))),
                ttl_dns_cache=300,
            ),
        )
        _shared_http_session_loop = loop
    return _shared_http_session


def _get_retrieval_runtime() -> tuple[asyncio.Semaphore, OrderedDict[str, Any], dict[str, asyncio.Task]]:
    global _retrieval_runtime_loop, _retrieval_semaphore, _retrieval_cache, _retrieval_inflight
    loop = asyncio.get_running_loop()
    if _retrieval_runtime_loop is not loop:
        _retrieval_runtime_loop = loop
        _retrieval_semaphore = asyncio.Semaphore(max(1, int(os.environ.get("RLLM_RETRIEVAL_CONCURRENCY", "160"))))
        _retrieval_cache = OrderedDict()
        _retrieval_inflight = {}
    assert _retrieval_semaphore is not None
    return _retrieval_semaphore, _retrieval_cache, _retrieval_inflight


def _get_atlas_http_session() -> aiohttp.ClientSession:
    global _atlas_http_session, _atlas_http_session_loop
    loop = asyncio.get_running_loop()
    if _atlas_http_session is None or _atlas_http_session.closed or _atlas_http_session_loop is not loop:
        concurrency = max(1, int(os.environ.get("MCP_ATLAS_CONCURRENCY", "5")))
        timeout = max(1.0, float(os.environ.get("MCP_ATLAS_TOOL_TIMEOUT", "120")))
        _atlas_http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=aiohttp.TCPConnector(
                limit=concurrency,
                limit_per_host=concurrency,
                ttl_dns_cache=300,
            ),
        )
        _atlas_http_session_loop = loop
    return _atlas_http_session


def _atlas_auth_headers() -> dict[str, str]:
    token = os.environ.get("MCP_ATLAS_AUTH_TOKEN", "")
    if not token:
        raise RuntimeError("MCP_ATLAS_AUTH_TOKEN is required for the MCP-Atlas tool API")
    return {"Authorization": f"Bearer {token}"}


def _atlas_read_only() -> bool:
    return os.environ.get("MCP_ATLAS_READ_ONLY", "true").strip().lower() not in {"0", "false", "no", "off"}


WEB_SEARCH_OBSERVATION_MAX_WORDS = 256
_RETRIEVAL_CHUNK_WORD_BUDGET = 256
_MIN_RETRIEVAL_DOC_WORDS = 25
_SUMMARY_REQUEST_MAX_WORDS = 2048
_MIN_SUMMARY_REQUEST_MAX_WORDS = 128
_SUMMARY_MAX_REDUCTION_ROUNDS = 4


def _now_monotonic() -> float:
    return time.monotonic()


def normalize_task(row: dict[str, Any]) -> dict[str, Any]:
    task = dict(row or {})
    extra = task.get("extra_info")
    if isinstance(extra, dict):
        merged = dict(extra)
        merged.update({k: v for k, v in task.items() if k != "extra_info"})
        task = merged
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        merged = dict(metadata)
        merged.update({k: v for k, v in task.items() if k != "metadata"})
        task = merged
    return task


def resolve_task_mode(task: dict[str, Any]) -> str:
    if is_et_task(task):
        return "et"
    if task.get("docker_image"):
        return "cli"
    if _is_atlas_mcp_task(task):
        return "mcp"
    if task.get("tools_py") or task.get("data_root") or task.get("environment"):
        return "mcp"
    return "web_search"


def _is_atlas_mcp_task(task: dict[str, Any]) -> bool:
    transport = str(task.get("mcp_transport") or "").strip().lower()
    source = str(task.get("data_source") or task.get("benchmark") or "").strip().lower()
    normalized_source = source.replace("-", "_").replace(" ", "_")
    return transport == "atlas" or normalized_source == "mcp_atlas" or bool(task.get("mcp_atlas_eval"))


def _atlas_enabled_tool_names(task: dict[str, Any]) -> list[str]:
    raw_tools = task.get("enabled_tools")
    if raw_tools is None:
        raw_tools = task.get("ENABLED_TOOLS")
    if raw_tools is None:
        raw_tools = []
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except json.JSONDecodeError:
            raw_tools = [raw_tools]
    if hasattr(raw_tools, "tolist"):
        raw_tools = raw_tools.tolist()
    if not isinstance(raw_tools, (list, tuple)):
        raw_tools = [raw_tools]
    names = []
    for item in raw_tools:
        name = item.get("name") if isinstance(item, dict) else item
        if name and str(name) not in names:
            names.append(str(name))
    return names


def _load_atlas_tool_schemas(base_url: str) -> list[dict[str, Any]]:
    base_url = base_url.rstrip("/")
    with _atlas_tool_cache_lock:
        cached = _atlas_tool_cache.get(base_url)
        if cached is not None:
            return cached
        request = urllib.request.Request(
            f"{base_url}/list-tools",
            data=b"{}",
            headers={"Content-Type": "application/json", **_atlas_auth_headers()},
            method="POST",
        )
        timeout = max(1.0, float(os.environ.get("MCP_ATLAS_LIST_TOOLS_TIMEOUT", "180")))
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, list):
            raise ValueError(f"MCP-Atlas /list-tools returned {type(payload).__name__}, expected list")
        _atlas_tool_cache[base_url] = payload
        return payload


class AtlasMCPToolset:
    def __init__(self, task: dict[str, Any]):
        self.task = task
        self.base_url = str(
            task.get("mcp_sandbox_url")
            or os.environ.get("MCP_SANDBOX_URL")
            or "http://127.0.0.1:1984"
        ).rstrip("/")
        self.enabled_tools = _atlas_enabled_tool_names(task)
        self._schemas: dict[str, dict[str, Any]] = {}
        self._dirty_servers: set[str] = set()
        self.load_error = ""
        self.load_warning = ""
        try:
            all_tools = _load_atlas_tool_schemas(self.base_url)
            by_name = {str(tool.get("name")): tool for tool in all_tools if tool.get("name")}
            missing = [name for name in self.enabled_tools if name not in by_name]
            if missing:
                self.load_warning = "MCP-Atlas tools not exposed by the sandbox: " + ", ".join(missing)
            for name in self.enabled_tools:
                if _atlas_read_only() and name in _ATLAS_MUTATING_TOOLS:
                    continue
                tool = by_name.get(name)
                if tool is None:
                    continue
                parameters = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {"type": "object"}
                self._schemas[name] = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description") or ""),
                        "parameters": parameters,
                    },
                }
            if not self.enabled_tools:
                self.load_error = "MCP-Atlas task does not define ENABLED_TOOLS"
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    def schemas(self) -> list[dict[str, Any]]:
        return [self._schemas[name] for name in self.enabled_tools if name in self._schemas] + [finish_schema()]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        if _atlas_read_only() and name in _ATLAS_MUTATING_TOOLS:
            return f"Error: MCP-Atlas read-only evaluation blocks mutating tool {name}"
        if name not in self._schemas:
            return f"Error: MCP-Atlas tool {name} is not enabled for this task"
        server_name = name.split("_", 1)[0]
        if name in _ATLAS_MUTATING_TOOLS:
            self._dirty_servers.add(server_name)
        try:
            async with _get_atlas_http_session().post(
                f"{self.base_url}/call-tool",
                headers=_atlas_auth_headers(),
                json={
                    "tool_name": name,
                    "tool_args": arguments,
                    "use_cache": server_name not in self._dirty_servers,
                },
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    return f"Error calling {name}: HTTP {response.status}: {body[:1000]}"
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    return _cap_atlas_tool_result(body)
                return _cap_atlas_tool_result(_format_atlas_tool_result(payload))
        except Exception as exc:
            return f"Error calling {name}: {type(exc).__name__}: {exc}"


def _format_atlas_tool_result(payload: Any) -> str:
    if not isinstance(payload, list):
        return json.dumps(payload, ensure_ascii=False, default=str)
    parts = []
    for block in payload:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") in {"image", "audio"}:
            parts.append(f"[{block.get('type')} content omitted from text observation]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False, default=str))
    return "\n".join(part for part in parts if part)


def _cap_atlas_tool_result(result: str) -> str:
    limit = max(0, int(os.environ.get("SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH", "4096")))
    if limit == 0 or len(result) <= limit:
        return result
    return result[:limit] + f"\n\n[MCP-Atlas tool output truncated to {limit} characters; original length: {len(result)}]"


class LocalMCPToolset:
    def __init__(self, task: dict[str, Any]):
        self.task = task
        self.tools_py = self._resolve_tools_py(task)
        self.tools: dict[str, Callable[..., Any]] = {}
        self.descriptions: dict[str, str] = {}
        self.load_error = ""
        self.load_warning = ""
        if self.tools_py:
            try:
                self._load_tools(self.tools_py)
            except Exception as e:
                self.load_error = f"{type(e).__name__}: {e}"

    def _resolve_tools_py(self, task: dict[str, Any]) -> Path | None:
        tools_py = task.get("tools_py")
        data_root = task.get("data_root")
        if not tools_py and data_root:
            tools_py = str(Path(data_root) / "tools.py")
        if not tools_py:
            return None
        path = Path(str(tools_py))
        if not path.is_absolute():
            candidates = [
                Path.cwd() / path,
                Path.cwd() / "experiments" / "fused" / "assets" / path.name,
                Path.cwd().parent / "rllm" / path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        if path.exists():
            return path
        return None

    def _load_tools(self, tools_py: Path) -> None:
        registry: dict[str, tuple[Callable[..., Any], str, dict[str, Any]]] = {}

        class FakeFastMCP:
            def __init__(self, *_args, **_kwargs):
                pass

            def tool(self, description: str | None = None, **_kwargs):
                def deco(fn):
                    # Generated MCP assets can contain concatenated source blocks
                    # that repeatedly assign globals such as BASE_DIR. Preserve
                    # the values visible at definition time so a later block
                    # cannot redirect an earlier tool to another asset directory.
                    registry[fn.__name__] = (
                        fn,
                        description or inspect.getdoc(fn) or "",
                        dict(fn.__globals__),
                    )
                    return fn

                return deco

        old_modules = {name: sys.modules.get(name) for name in ("mcp", "mcp.server", "mcp.server.fastmcp", "tools")}
        exec_error: Exception | None = None
        try:
            mcp_mod = types.ModuleType("mcp")
            server_mod = types.ModuleType("mcp.server")
            fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
            fake_mcp = FakeFastMCP("Tools")
            mcp_mod.tool = fake_mcp.tool
            server_mod.fastmcp = fastmcp_mod
            fastmcp_mod.FastMCP = FakeFastMCP
            sys.modules["mcp"] = mcp_mod
            sys.modules["mcp.server"] = server_mod
            sys.modules["mcp.server.fastmcp"] = fastmcp_mod

            module_name = f"_slime_fused_tools_{abs(hash(str(tools_py)))}"
            spec = importlib.util.spec_from_file_location(module_name, tools_py)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            module.mcp = FakeFastMCP("Tools")
            sys.modules[module_name] = module
            sys.modules["tools"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            exec_error = e
        finally:
            for name, mod in old_modules.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        module_globals = dict(module.__dict__)
        for name, (fn, description, definition_globals) in registry.items():
            function_globals = module_globals.copy()
            function_globals.update(definition_globals)
            bound_fn = types.FunctionType(fn.__code__, function_globals, fn.__name__, fn.__defaults__, fn.__closure__)
            bound_fn.__kwdefaults__ = fn.__kwdefaults__
            bound_fn.__annotations__ = fn.__annotations__
            bound_fn.__dict__.update(fn.__dict__)
            bound_fn.__module__ = fn.__module__
            bound_fn.__qualname__ = fn.__qualname__
            self.tools[name] = self._run_from_asset_dir(bound_fn, tools_py.parent)
            self.descriptions[name] = description

        if exec_error is not None:
            if not self.tools:
                raise exec_error
            self.load_warning = f"{type(exec_error).__name__}: {exec_error}"
            if tools_py not in _warned_partial_mcp_modules:
                _warned_partial_mcp_modules.add(tools_py)
                logger.warning(
                    "MCP tool module %s failed after registering %d tools; keeping the registered tools: %s",
                    tools_py,
                    len(self.tools),
                    self.load_warning,
                )

    @staticmethod
    def _run_from_asset_dir(fn: Callable[..., Any], asset_dir: Path) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with _mcp_tool_cwd_lock:
                previous_cwd = Path.cwd()
                try:
                    os.chdir(asset_dir)
                    return fn(*args, **kwargs)
                finally:
                    os.chdir(previous_cwd)

        return wrapped

    def schemas(self) -> list[dict]:
        schemas = []
        for name, fn in sorted(self.tools.items()):
            sig = inspect.signature(fn)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                default = param.default
                ann = param.annotation
                typ = "string"
                if ann in (int, "int"):
                    typ = "integer"
                elif ann in (float, "float"):
                    typ = "number"
                elif ann in (bool, "bool"):
                    typ = "boolean"
                elif getattr(ann, "__origin__", None) is list or ann in (list, "list"):
                    typ = "array"
                properties[param_name] = {"type": typ, "description": ""}
                if default is inspect._empty:
                    required.append(param_name)
            schemas.append(tool_schema(name, self.descriptions.get(name, ""), properties, required))
        schemas.append(finish_schema())
        return schemas

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self.tools[name]

    def __getattr__(self, name: str) -> Any:
        if name in self.tools:
            return self.tools[name]
        raise AttributeError(name)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = self.call_raw(name, arguments)
        if isinstance(result, str) and result.startswith("Error:"):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    def call_raw(self, name: str, arguments: dict[str, Any]) -> Any:
        fn = self.tools.get(name)
        if fn is None:
            return f"Error: unknown tool {name}"
        try:
            kwargs = _coerce_kwargs(fn, arguments)
            return fn(**kwargs)
        except Exception as e:
            return f"Error calling {name}: {type(e).__name__}: {e}"


def _coerce_kwargs(fn: Callable[..., Any], arguments: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name not in arguments:
            continue
        value = arguments[name]
        ann = param.annotation
        try:
            if ann in (int, "int"):
                value = int(value)
            elif ann in (float, "float"):
                value = float(value)
            elif ann in (bool, "bool") and isinstance(value, str):
                value = value.lower() in {"1", "true", "yes", "y", "on"}
        except Exception:
            pass
        kwargs[name] = value
    return kwargs


class FusedEnvironment:
    def __init__(
        self,
        task: dict[str, Any],
        *,
        retrieval_url: str | None = None,
        retrieval_max_results: int = 5,
        enable_tools: bool = True,
        search_gym: bool = False,
    ):
        self.task = normalize_task(task)
        self.mode = resolve_task_mode(self.task)
        self.retrieval_url = retrieval_url or os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:65432")
        self.retrieval_max_results = retrieval_max_results
        self.enable_tools = enable_tools
        self.search_gym = search_gym
        self.answer = ""
        self.tool_calls = 0
        self.web_search_queries: set[str] = set()
        self.web_search_cache: dict[str, Any] = {}
        self.reward_debug: dict[str, Any] = {}
        if enable_tools and self.mode == "mcp":
            self.mcp_tools = AtlasMCPToolset(self.task) if _is_atlas_mcp_task(self.task) else LocalMCPToolset(self.task)
        else:
            self.mcp_tools = None
        self.docker_env = DockerTaskEnvironment(self.task, mode=self.mode) if enable_tools and self.mode in {"cli", "et"} else None

    def reset(self) -> tuple[str, dict[str, Any]]:
        if not self.enable_tools:
            question = (
                self.task.get("question")
                or self.task.get("query")
                or self.task.get("input")
                or self.task.get("problem_statement")
                or self.task.get("prompt")
                or self._question_from_environment()
            )
            return str(question), {"task_type": self.mode}
        if self.mode == "mcp":
            question = self.task.get("question") or self.task.get("problem_statement") or self._question_from_environment()
            info = {"task_type": "mcp", "tools_json": self.tools(), "difficulty": self.task.get("difficulty", "")}
            if self.mcp_tools is not None and self.mcp_tools.load_error:
                info["env_error"] = self.mcp_tools.load_error
            if self.mcp_tools is not None and self.mcp_tools.load_warning:
                info["env_warning"] = self.mcp_tools.load_warning
            return str(question), info
        if self.mode in {"cli", "et"} and self.docker_env is not None:
            return self.docker_env.reset()
        question = self.task.get("question") or self.task.get("query") or self.task.get("input") or self.task.get("problem_statement") or self.task.get("prompt") or ""
        return str(question), {"task_type": "web search"}

    def _question_from_environment(self) -> str:
        env = self.task.get("environment")
        if isinstance(env, dict):
            question = env.get("task") or env.get("question") or env.get("instruction")
            if question:
                return str(question)
        prompt = self.task.get("prompt")
        if isinstance(prompt, list) and prompt:
            content = prompt[-1].get("content")
            if content and content != "placeholder":
                return str(content)
        return json.dumps(self.task.get("reward_model") or {}, ensure_ascii=False)

    def tools(self) -> list[dict]:
        if not self.enable_tools:
            return [finish_schema()]
        if self.mode == "mcp" and self.mcp_tools is not None:
            return self.mcp_tools.schemas()
        if self.mode in {"cli", "et"} and self.docker_env is not None:
            return self.docker_env.schemas()
        if self.mode == "web_search":
            if self.search_gym:
                return [search_schema(), access_schema(), finish_schema()]
            return [web_search_schema(), finish_schema()]
        return [finish_schema()]

    async def step(self, action: ToolCall | str) -> tuple[str, float, bool, dict[str, Any]]:
        if isinstance(action, str):
            return action, 0.0, False, {"parser/unknown_total": 1}
        name = action.name
        args = action.arguments or {}
        if name in {"finish", "submit"}:
            self.answer = str(args.get("result") or args.get("answer") or "")
            if self.mode == "mcp" and _contains_nested_tool_call(self.answer):
                self.reward_debug = {
                    "type": self.mode,
                    "reward": 0.0,
                    "invalid_finish_payload": "nested_tool_call",
                    "tool_calls": self.tool_calls,
                }
                return (
                    "Error: finish.result must be a pure JSON value, not a nested <tool_call> payload.",
                    0.0,
                    True,
                    {
                        "termination_reason": "ABNORMAL_NESTED_FINISH_PAYLOAD",
                        "reward_debug": self.reward_debug,
                    },
                )
            reward = self.compute_final_reward()
            return "Submitted.", reward, True, {"reward_debug": self.reward_debug}
        self.tool_calls += 1
        if self.mode == "web_search" and name in {"web_search", "search"}:
            return await self._step_web_search(args)
        if self.mode == "web_search" and self.search_gym and name == "access":
            return await self._step_web_access(args)
        if self.mode == "mcp" and self.mcp_tools is not None:
            if self.mcp_tools.load_error:
                return f"Error: MCP tools failed to load: {self.mcp_tools.load_error}", 0.0, False, {"tools/load_error": 1}
            started_at = _now_monotonic()
            if isinstance(self.mcp_tools, AtlasMCPToolset):
                result = await self.mcp_tools.call(name, args)
            else:
                result = self.mcp_tools.call(name, args)
            return (
                result,
                0.0,
                False,
                {
                    "tools/calls": self.tool_calls,
                    "tools/mcp_tool_elapsed_s": _now_monotonic() - started_at,
                },
            )
        if self.mode in {"cli", "et"} and self.docker_env is not None:
            return self.docker_env.step(name, args)
        return f"Error: tool {name} is not available for task mode {self.mode}", 0.0, False, {}

    async def _step_web_access(self, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        url = str(args.get("url") or "").strip()
        if not url:
            return "<information>\nNo More Information is Found for this URL.\n</information>", 0.0, False, {}
        try:
            async with _get_shared_http_session().post(
                _normalize_access_url(self.retrieval_url), json={"urls": [url]}
            ) as response:
                response.raise_for_status()
                payload = await response.json()
            result = (payload.get("result") or [{}])[0] or {}
            page = str(result.get("contents") or result.get("page") or "")[:250000]
            if not page:
                page = "No More Information is Found for this URL."
            return f"<information>\n>>>> Page 1 >>>>\n\n{page}\n</information>", 0.0, False, {"tools/access_calls": self.tool_calls}
        except Exception as exc:
            return f"<information>\nNo More Information is Found for this URL.\n</information>\nAccess failed: {type(exc).__name__}: {exc}", 0.0, False, {"tools/access_failed": 1}

    async def _step_web_search(self, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        query = str(args.get("query") or "")
        if not query:
            return "Error: web_search requires query.", 0.0, False, {}
        try:
            requested_max_results = int(args.get("max_results") or self.retrieval_max_results)
        except (TypeError, ValueError):
            requested_max_results = 1
        max_results = max(1, min(requested_max_results, self.retrieval_max_results))
        self.web_search_queries.add(_normalize_search_query(query))
        retrieval_mode = os.environ.get("RLLM_RETRIEVAL_MODE", "hybrid")
        retrieval_max_words = int(os.environ.get("RLLM_RETRIEVAL_MAX_WORDS", "4096"))
        retrieve_retry_budget = max(0, int(os.environ.get("RLLM_RETRIEVAL_RETRY_BUDGET", "0")))
        summary_retry_budget = max(0, int(os.environ.get("RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "0")))
        use_summary_requested = os.environ.get("RLLM_RETRIEVAL_SUMMARIZE", "0") == "1"
        payload = {
            "query": query,
            "top_k": max_results,
            "topk": max_results,
            "max_results": max_results,
            "max_words": retrieval_max_words,
            "mode": str(retrieval_mode or "lexical").strip() or "lexical",
            "return_scores": False,
        }
        metrics = {
            "tools/search_calls": self.tool_calls,
            "tools/search_summary_retries": 0,
            "tools/search_summary_failures": 0,
            "tools/search_summary_fallbacks": 0,
            "tools/search_summary_elapsed_s": 0.0,
            "tools/search_retrieve_elapsed_s": 0.0,
            "tools/search_retrieve_retries": 0,
            "tools/search_retrieve_failures": 0,
            "tools/search_lexrank_summary": 0,
            "tools/search_episode_cache_hits": 0,
            "tools/search_global_cache_hits": 0,
            "tools/search_singleflight_hits": 0,
        }
        retrieve_started_at = _now_monotonic()
        try:
            data, retrieve_retries, cache_source = await _retrieve_json_cached(
                _normalize_retrieve_url(self.retrieval_url),
                payload,
                retry_budget=retrieve_retry_budget,
                episode_cache=self.web_search_cache,
            )
            metrics["tools/search_retrieve_retries"] = retrieve_retries
            if cache_source is not None:
                metrics[f"tools/search_{cache_source}_hits"] = 1
        except Exception as e:
            metrics["tools/search_failed"] = 1
            metrics["tools/search_retrieve_failures"] = 1
            return f"Search failed: {type(e).__name__}: {e}", 0.0, False, metrics
        finally:
            metrics["tools/search_retrieve_elapsed_s"] = _now_monotonic() - retrieve_started_at

        documents, format_metadata = _format_retrieval_documents(data, max_results=max_results)
        content = "\n\n".join(documents)
        summary_used = False
        if use_summary_requested:
            summary_started_at = _now_monotonic()
            try:
                summary, summary_retries = await _summarize_with_retries(
                    _get_shared_http_session(),
                    self.retrieval_url,
                    documents,
                    retry_budget=summary_retry_budget,
                )
                metrics["tools/search_summary_retries"] = summary_retries
                if summary:
                    content = summary
                    summary_used = True
                else:
                    metrics["tools/search_summary_failures"] = 1
                    metrics["tools/search_summary_fallbacks"] = 1
                    logger.warning("Summarize endpoint returned empty content; falling back to chunked docs for this request")
            except Exception as e:
                metrics["tools/search_summary_retries"] = summary_retry_budget
                metrics["tools/search_summary_failures"] = 1
                metrics["tools/search_summary_fallbacks"] = 1
                logger.warning(
                    "Summarize service unavailable (%s: %s); falling back to chunked docs for this request",
                    type(e).__name__,
                    e,
                )
            finally:
                metrics["tools/search_summary_elapsed_s"] = _now_monotonic() - summary_started_at

        if use_summary_requested and not summary_used:
            metrics["tools/search_summary_fallbacks"] = 1

        metrics.update(
            {
                "search_summary_used": summary_used,
                "search_summary_requested": use_summary_requested,
                **format_metadata,
            }
        )
        word_budget = WEB_SEARCH_OBSERVATION_MAX_WORDS if summary_used else _RETRIEVAL_CHUNK_WORD_BUDGET
        return _limit_words(content, max_words=word_budget), 0.0, False, metrics

    def compute_final_reward(self, *, require_tool_evidence: bool = True) -> float:
        verifier_reward = self._compute_verifier_reward(require_tool_evidence=require_tool_evidence)
        if verifier_reward is not None:
            return verifier_reward
        if self.mode == "mcp" and self.mcp_tools is not None and self.mcp_tools.load_error:
            self.reward_debug = {"type": self.mode, "reward": 0.0, "tools_load_error": self.mcp_tools.load_error}
            return 0.0
        if self.mode in {"cli", "et"} and self.docker_env is not None:
            reward = self.docker_env.compute_final_reward()
            self.reward_debug = self.docker_env.reward_debug
            return reward
        label = self.task.get("ground_truth") or self.task.get("reward_model") or self.task.get("label") or self.task.get("answer")
        gt = _extract_ground_truth(label)
        pred = _strip_boxed(self.answer)
        if gt is None:
            self.reward_debug = {
                "type": self.mode,
                "reward": 0.0,
                "no_ground_truth": True,
                "prediction": pred,
                "tool_calls": self.tool_calls,
            }
            return 0.0
        min_unique_searches = int(os.environ.get("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "2"))
        if require_tool_evidence and self.mode == "web_search" and len(self.web_search_queries) < min_unique_searches:
            self.reward_debug = {
                "type": self.mode,
                "reward": 0.0,
                "insufficient_searches": True,
                "unique_search_calls": len(self.web_search_queries),
                "min_unique_search_calls": min_unique_searches,
                "prediction": pred,
                "tool_calls": self.tool_calls,
            }
            return 0.0
        reward = _exact_match_reward(pred, gt)
        self.reward_debug = {
            "type": self.mode,
            "reward": reward,
            "exact_match": bool(reward),
            "prediction": pred,
            "ground_truth": gt,
            "tool_calls": self.tool_calls,
        }
        if self.mode == "web_search":
            self.reward_debug["unique_search_calls"] = len(self.web_search_queries)
            self.reward_debug["min_unique_search_calls"] = min_unique_searches
        return float(reward)

    def _compute_verifier_reward(self, *, require_tool_evidence: bool = True) -> float | None:
        verifier = self.task.get("verifier")
        if not isinstance(verifier, dict) or not verifier.get("verification_code"):
            return None
        if require_tool_evidence and self.mode == "mcp" and self.tool_calls <= 0:
            self.reward_debug = {
                "type": self.mode,
                "reward": 0.0,
                "verifier_skipped": "no_tool_calls",
                "tool_calls": self.tool_calls,
            }
            return 0.0
        answer = _parse_answer_payload(self.answer)
        namespace: dict[str, Any] = {}
        try:
            exec(str(verifier["verification_code"]), namespace)
            verify = namespace.get("verify")
            if not callable(verify):
                return None
            result = verify(self.mcp_tools, answer)
            if isinstance(result, dict):
                passed = bool(result.get("passed") or result.get("success"))
                score = float(result.get("score", 1.0 if passed else 0.0))
                reward = score if passed else 0.0
                self.reward_debug = {
                    "type": self.mode,
                    "reward": reward,
                    "verifier_passed": passed,
                    "verifier": result,
                    "tool_calls": self.tool_calls,
                }
                return reward
            reward = float(result)
            self.reward_debug = {"type": self.mode, "reward": reward, "verifier": result, "tool_calls": self.tool_calls}
            return reward
        except Exception as e:
            self.reward_debug = {
                "type": self.mode,
                "reward": 0.0,
                "verifier_error": f"{type(e).__name__}: {e}",
                "tool_calls": self.tool_calls,
            }
            return 0.0

    def close(self) -> None:
        if self.docker_env is not None:
            self.docker_env.close()


def _extract_ground_truth(label: Any) -> Any:
    if isinstance(label, dict):
        for key in ("ground_truth", "target", "answer", "answers"):
            value = label.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, (list, tuple, set, dict)) and not value:
                continue
            return value
    return label


def _strip_boxed(text: str) -> str:
    text = text or ""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return text.strip()
    start = idx + len(marker)
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos].strip()
    return text.strip()


def _exact_match_reward(prediction: str, ground_truth: Any) -> float:
    targets = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    normalized_prediction = normalize_answer(str(prediction))
    return 1.0 if any(normalize_answer(str(target)) == normalized_prediction for target in targets) else 0.0


def _normalize_search_query(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def _contains_nested_tool_call(text: str) -> bool:
    return "<tool_call" in str(text or "").lower() or "</tool_call>" in str(text or "").lower()


def _parse_answer_payload(text: str) -> Any:
    stripped = _strip_boxed(text)
    try:
        return json.loads(stripped)
    except Exception:
        return stripped


def _format_retrieval(data: Any) -> str:
    rows = _retrieval_rows(data)
    if rows:
        return _limit_words("\n".join(_format_retrieval_row(idx, row) for idx, row in enumerate(rows, start=1)))
    if isinstance(data, str):
        return _limit_words(_compact_text(data))
    return _limit_words(_plain_retrieval_text(data))


def _retrieval_rows(data: Any) -> list[Any]:
    if isinstance(data, dict):
        result = data.get("result") or data.get("results") or data.get("data") or data.get("documents")
    else:
        result = data
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("result", "results", "data", "documents", "items"):
            nested = result.get(key)
            if isinstance(nested, list):
                return nested
        return [result]
    return []


def _format_retrieval_row(idx: int, row: Any) -> str:
    title, content = _retrieval_title_and_content(row)
    if title and content:
        return f"[{idx}] {title}: {content}"
    if title:
        return f"[{idx}] {title}"
    if content:
        return f"[{idx}] {content}"
    return f"[{idx}] {_plain_retrieval_text(row)}"


def _retrieval_title_and_content(row: Any) -> tuple[str, str]:
    if not isinstance(row, dict):
        return "", _compact_text(str(row))

    content_obj = row.get("content")
    source = content_obj if isinstance(content_obj, dict) else row
    title = source.get("title") or row.get("title") or source.get("name") or row.get("name")
    content = source.get("summary") or source.get("chunk_text") or source.get("snippet") or source.get("text") or source.get("original_text")
    if content is None and isinstance(content_obj, str):
        content = content_obj
    return _compact_text(str(title)) if title is not None else "", _compact_text(str(content)) if content is not None else ""


async def _post_json_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    *,
    retry_budget: int,
) -> tuple[Any, int]:
    last_error: Exception | None = None
    retries_used = 0
    for attempt in range(retry_budget + 1):
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise aiohttp.ClientResponseError(
                        resp.request_info,
                        resp.history,
                        status=resp.status,
                        message=body or resp.reason,
                        headers=resp.headers,
                    )
                return await resp.json(content_type=None), retries_used
        except Exception as e:
            last_error = e
            if attempt >= retry_budget:
                break
            retries_used += 1
    assert last_error is not None
    raise last_error


async def _retrieve_json_cached(
    url: str,
    payload: dict[str, Any],
    *,
    retry_budget: int,
    episode_cache: dict[str, Any],
) -> tuple[Any, int, str | None]:
    cache_key = json.dumps(
        {
            "url": url,
            **payload,
            "query": _normalize_search_query(str(payload.get("query") or "")),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    if cache_key in episode_cache:
        return episode_cache[cache_key], 0, "episode_cache"

    semaphore, cache, inflight = _get_retrieval_runtime()
    cache_size = max(0, int(os.environ.get("RLLM_RETRIEVAL_CACHE_SIZE", "4096")))
    if cache_key in cache:
        data = cache.pop(cache_key)
        cache[cache_key] = data
        episode_cache[cache_key] = data
        return data, 0, "global_cache"

    task = inflight.get(cache_key)
    cache_source = "singleflight" if task is not None else None
    if task is None:

        async def fetch():
            async with semaphore:
                return await _post_json_with_retries(
                    _get_shared_http_session(),
                    url,
                    payload,
                    retry_budget=retry_budget,
                )

        task = asyncio.create_task(fetch())
        inflight[cache_key] = task

    try:
        data, retries_used = await asyncio.shield(task)
    finally:
        if task.done() and inflight.get(cache_key) is task:
            inflight.pop(cache_key, None)

    episode_cache[cache_key] = data
    if cache_size > 0:
        cache[cache_key] = data
        cache.move_to_end(cache_key)
        while len(cache) > cache_size:
            cache.popitem(last=False)
    return data, 0 if cache_source is not None else retries_used, cache_source


async def _summarize_with_retries(
    session: aiohttp.ClientSession,
    retrieval_url: str,
    documents: list[str],
    *,
    retry_budget: int,
    request_max_words: int | None = None,
    reduction_round: int = 0,
) -> tuple[str | None, int]:
    request_budget = max(
        _MIN_SUMMARY_REQUEST_MAX_WORDS,
        request_max_words if request_max_words is not None else int(os.environ.get("RLLM_RETRIEVAL_SUMMARY_MAX_WORDS_PER_REQUEST", str(_SUMMARY_REQUEST_MAX_WORDS))),
    )
    summary_url = _normalize_summary_url(retrieval_url)
    batches = _build_summary_batches(documents, max_words=request_budget)
    partial_summaries: list[str] = []
    retries_used = 0
    try:
        for batch in batches:
            payload = {"documents": [{"content": document} for document in batch], "max_length": WEB_SEARCH_OBSERVATION_MAX_WORDS}
            summary_data, batch_retries = await _post_json_with_retries(
                session,
                summary_url,
                payload,
                retry_budget=retry_budget,
            )
            retries_used += batch_retries
            candidate = str((summary_data or {}).get("summary", "")).split("# Summary:", 1)[-1].strip()
            if not candidate:
                return None, retries_used
            partial_summaries.append(candidate)
    except Exception as e:
        if reduction_round < _SUMMARY_MAX_REDUCTION_ROUNDS and request_budget > _MIN_SUMMARY_REQUEST_MAX_WORDS and _is_summary_length_error(e):
            fallback_summary, fallback_retries = await _summarize_with_retries(
                session,
                retrieval_url,
                documents,
                retry_budget=retry_budget,
                request_max_words=max(_MIN_SUMMARY_REQUEST_MAX_WORDS, request_budget // 2),
                reduction_round=reduction_round + 1,
            )
            return fallback_summary, retries_used + fallback_retries
        raise

    if len(partial_summaries) == 1:
        return partial_summaries[0], retries_used
    if reduction_round >= _SUMMARY_MAX_REDUCTION_ROUNDS:
        return _limit_words("\n\n".join(partial_summaries)), retries_used

    merged_summary, merge_retries = await _summarize_with_retries(
        session,
        retrieval_url,
        [f"[Partial Summary {idx}] {summary}" for idx, summary in enumerate(partial_summaries, start=1)],
        retry_budget=retry_budget,
        request_max_words=request_budget,
        reduction_round=reduction_round + 1,
    )
    return merged_summary, retries_used + merge_retries


def _normalize_retrieve_url(retrieval_url: str) -> str:
    retrieval_url = retrieval_url.rstrip("/")
    return retrieval_url if retrieval_url.endswith("/retrieve") else f"{retrieval_url}/retrieve"


def _normalize_access_url(retrieval_url: str) -> str:
    retrieval_url = retrieval_url.rstrip("/")
    base = retrieval_url[: -len("/retrieve")] if retrieval_url.endswith("/retrieve") else retrieval_url
    return f"{base}/access"


def _normalize_summary_url(retrieval_url: str) -> str:
    retrieval_url = retrieval_url.rstrip("/")
    base = retrieval_url[: -len("/retrieve")] if retrieval_url.endswith("/retrieve") else retrieval_url
    return f"{base}/summarize"


def _build_summary_batches(documents: list[str], *, max_words: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_words = 0
    for document in documents:
        for chunk in _split_text_word_chunks(document, max_words=max_words):
            chunk_words = len(chunk.split())
            if current_batch and current_words + chunk_words > max_words:
                batches.append(current_batch)
                current_batch = []
                current_words = 0
            current_batch.append(chunk)
            current_words += chunk_words
    if current_batch:
        batches.append(current_batch)
    return batches or [["No relevant documents found."]]


def _split_text_word_chunks(text: str, *, max_words: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    if len(words) <= max_words:
        return [" ".join(words)]
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), max_words)]


def _is_summary_length_error(error: Exception) -> bool:
    return "longer than the maximum model length" in str(error).lower()


def _format_retrieval_documents(data: Any, *, max_results: int) -> tuple[list[str], dict[str, Any]]:
    rows = _retrieval_rows(data)
    if not rows:
        fallback = _format_retrieval(data)
        return [fallback or "No relevant documents found."], {
            "search_num_unique": 0,
            "search_num_duplicates": 0,
            "search_num_short_filtered": 0,
        }

    documents: list[str] = []
    seen_signatures: set[str] = set()
    duplicate_count = 0
    skipped_short = 0
    for row in rows:
        content = _extract_retrieval_document_text(row)
        if not content:
            continue
        signature = _normalize_retrieval_doc_signature(" ".join(str(value) for value in (row.get("title") if isinstance(row, dict) else None, _extract_retrieval_document_url(row), content) if value))
        if signature and signature in seen_signatures:
            duplicate_count += 1
            continue
        if signature:
            seen_signatures.add(signature)
        if len(content.split()) < _MIN_RETRIEVAL_DOC_WORDS and not (
            isinstance(row, dict) and row.get("search_snippet") is True
        ):
            skipped_short += 1
            continue
        title = _extract_retrieval_document_title(row, content)
        documents.append(f"[Result {len(documents) + 1}] Title: {title}\nContent: {content.strip()}")
        if len(documents) >= max_results:
            break

    if not documents:
        return ["No usable evidence was found for this query. The returned passages were duplicates, too short, or too generic. " "Do not submit an answer from this result. Rewrite the query with a specific title, quoted phrase, named entity, date, number, or one clue from the question, then search again."], {
            "search_num_unique": 0,
            "search_num_duplicates": duplicate_count,
            "search_num_short_filtered": skipped_short,
        }

    if skipped_short:
        documents.append(f"[{skipped_short} short fragments were filtered; narrow the query if you need more detail]")
    if duplicate_count:
        documents.append(f"[{duplicate_count} duplicate passages were removed before display]")
    return documents, {
        "search_num_unique": len(documents),
        "search_num_duplicates": duplicate_count,
        "search_num_short_filtered": skipped_short,
    }


def _normalize_retrieval_doc_signature(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    return "".join(ch for ch in normalized if ch.isalnum() or ch.isspace())[:500]


def _extract_retrieval_document_text(row: Any) -> str:
    if not isinstance(row, dict):
        return _compact_text(str(row))
    document = row.get("document")
    if isinstance(document, dict):
        for key in ("contents", "text", "passage"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(document, str) and document.strip():
        return document.strip()

    content = row.get("content")
    if isinstance(content, dict):
        candidates = [value for key in ("original_text", "chunk_text", "text") if isinstance((value := content.get(key)), str) and value.strip()]
        if candidates:
            return max(candidates, key=lambda text: len(text.split())).strip()
    elif isinstance(content, str) and content.strip():
        return content.strip()

    candidates = []
    for key in ("contents", "chunk_text", "snippet", "text", "passage", "summary", "original_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    if candidates:
        return max(candidates, key=lambda text: len(text.split())).strip()
    return ""


def _extract_retrieval_document_title(row: Any, content: str) -> str:
    if not isinstance(row, dict):
        return "Untitled"
    for key in ("title", "source", "url", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = row.get("content")
    if isinstance(nested, dict):
        for key in ("title", "source", "url", "name"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    first_line = content.strip().splitlines()[0].strip() if content.strip() else ""
    if first_line and len(first_line.split()) <= 16:
        return first_line
    return "Untitled"


def _extract_retrieval_document_url(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("url", "source"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = row.get("content")
    if isinstance(nested, dict):
        for key in ("url", "source"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _compact_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    return " ".join(text.split())


def _limit_words(text: str, max_words: int = WEB_SEARCH_OBSERVATION_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _plain_retrieval_text(value: Any) -> str:
    parts: list[str] = []

    def collect(item: Any, key: str = "") -> None:
        normalized_key = key.lower()
        if normalized_key in {
            "url",
            "link",
            "id",
            "lexical_rank",
            "lexical_score",
            "fusion_score",
            "title_overlap_boost",
            "fusion_profile",
            "query_class",
            "retrieval_channels",
        }:
            return
        if isinstance(item, dict):
            for child_key, child_value in item.items():
                collect(child_value, str(child_key))
            return
        if isinstance(item, list):
            for child in item:
                collect(child, key)
            return
        if item is not None:
            text = _compact_text(str(item))
            if text:
                parts.append(text)

    collect(value)
    return " ".join(parts)
