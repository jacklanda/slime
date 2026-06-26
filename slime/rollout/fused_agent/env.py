from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import aiohttp

from slime.rollout.rm_hub.f1 import normalize_answer

from .docker_env import DockerTaskEnvironment, is_et_task
from .parser import ToolCall, tool_schema
from .prompts import finish_schema, web_search_schema

logger = logging.getLogger(__name__)

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
    if task.get("tools_py") or task.get("data_root") or task.get("environment"):
        return "mcp"
    return "web_search"


class LocalMCPToolset:
    def __init__(self, task: dict[str, Any]):
        self.task = task
        self.tools_py = self._resolve_tools_py(task)
        self.tools: dict[str, Callable[..., Any]] = {}
        self.descriptions: dict[str, str] = {}
        self.load_error = ""
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
        registry: dict[str, tuple[Callable[..., Any], str]] = {}

        class FakeFastMCP:
            def __init__(self, *_args, **_kwargs):
                pass

            def tool(self, description: str | None = None, **_kwargs):
                def deco(fn):
                    registry[fn.__name__] = (fn, description or inspect.getdoc(fn) or "")
                    return fn

                return deco

        old_modules = {name: sys.modules.get(name) for name in ("mcp", "mcp.server", "mcp.server.fastmcp", "tools")}
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
        finally:
            for name, mod in old_modules.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.tools = {name: fn for name, (fn, _) in registry.items()}
        self.descriptions = {name: desc for name, (_, desc) in registry.items()}

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
    def __init__(self, task: dict[str, Any], *, retrieval_url: str | None = None, retrieval_max_results: int = 5):
        self.task = normalize_task(task)
        self.mode = resolve_task_mode(self.task)
        self.retrieval_url = retrieval_url or os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:65432")
        self.retrieval_max_results = retrieval_max_results
        self.answer = ""
        self.tool_calls = 0
        self.web_search_queries: set[str] = set()
        self.reward_debug: dict[str, Any] = {}
        self.mcp_tools = LocalMCPToolset(self.task) if self.mode == "mcp" else None
        self.docker_env = DockerTaskEnvironment(self.task, mode=self.mode) if self.mode in {"cli", "et"} else None

    def reset(self) -> tuple[str, dict[str, Any]]:
        if self.mode == "mcp":
            question = self.task.get("question") or self.task.get("problem_statement") or self._question_from_environment()
            info = {"task_type": "mcp", "tools_json": self.tools(), "difficulty": self.task.get("difficulty", "")}
            if self.mcp_tools is not None and self.mcp_tools.load_error:
                info["env_error"] = self.mcp_tools.load_error
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
        if self.mode == "mcp" and self.mcp_tools is not None:
            return self.mcp_tools.schemas()
        if self.mode in {"cli", "et"} and self.docker_env is not None:
            return self.docker_env.schemas()
        if self.mode == "web_search":
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
        if self.mode == "web_search" and name == "web_search":
            return await self._step_web_search(args)
        if self.mode == "mcp" and self.mcp_tools is not None:
            if self.mcp_tools.load_error:
                return f"Error: MCP tools failed to load: {self.mcp_tools.load_error}", 0.0, False, {"tools/load_error": 1}
            started_at = _now_monotonic()
            result = self.mcp_tools.call(name, args)
            return result, 0.0, False, {
                "tools/calls": self.tool_calls,
                "tools/mcp_tool_elapsed_s": _now_monotonic() - started_at,
            }
        if self.mode in {"cli", "et"} and self.docker_env is not None:
            return self.docker_env.step(name, args)
        return f"Error: tool {name} is not available for task mode {self.mode}", 0.0, False, {}

    async def _step_web_search(self, args: dict[str, Any]) -> tuple[str, float, bool, dict[str, Any]]:
        query = str(args.get("query") or "")
        max_results = int(args.get("max_results") or self.retrieval_max_results)
        if not query:
            return "Error: web_search requires query.", 0.0, False, {}
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
        }
        retrieve_started_at = _now_monotonic()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                data, retrieve_retries = await _post_json_with_retries(
                    session,
                    _normalize_retrieve_url(self.retrieval_url),
                    payload,
                    retry_budget=retrieve_retry_budget,
                )
                metrics["tools/search_retrieve_retries"] = retrieve_retries
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
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                    summary, summary_retries = await _summarize_with_retries(
                        session,
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
                    logger.warning(
                        "Summarize endpoint returned empty content; falling back to chunked docs for this request"
                    )
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

    def compute_final_reward(self) -> float:
        verifier_reward = self._compute_verifier_reward()
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
        if self.mode == "web_search" and len(self.web_search_queries) < min_unique_searches:
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

    def _compute_verifier_reward(self) -> float | None:
        verifier = self.task.get("verifier")
        if not isinstance(verifier, dict) or not verifier.get("verification_code"):
            return None
        if self.mode == "mcp" and self.tool_calls <= 0:
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
    content = (
        source.get("summary")
        or source.get("chunk_text")
        or source.get("snippet")
        or source.get("text")
        or source.get("original_text")
    )
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
        request_max_words
        if request_max_words is not None
        else int(os.environ.get("RLLM_RETRIEVAL_SUMMARY_MAX_WORDS_PER_REQUEST", str(_SUMMARY_REQUEST_MAX_WORDS))),
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
        if (
            reduction_round < _SUMMARY_MAX_REDUCTION_ROUNDS
            and request_budget > _MIN_SUMMARY_REQUEST_MAX_WORDS
            and _is_summary_length_error(e)
        ):
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
        signature = _normalize_retrieval_doc_signature(
            " ".join(str(value) for value in (row.get("title") if isinstance(row, dict) else None, _extract_retrieval_document_url(row), content) if value)
        )
        if signature and signature in seen_signatures:
            duplicate_count += 1
            continue
        if signature:
            seen_signatures.add(signature)
        if len(content.split()) < _MIN_RETRIEVAL_DOC_WORDS:
            skipped_short += 1
            continue
        title = _extract_retrieval_document_title(row, content)
        documents.append(f"[Result {len(documents) + 1}] Title: {title}\nSnippet: {content.strip()}")
        if len(documents) >= max_results:
            break

    if not documents:
        return [
            "No usable evidence was found for this query. The returned passages were duplicates, too short, or too generic. "
            "Do not submit an answer from this result. Rewrite the query with a specific title, quoted phrase, named entity, date, number, or one clue from the question, then search again."
        ], {
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
        candidates = [
            value
            for key in ("original_text", "chunk_text", "text")
            if isinstance((value := content.get(key)), str) and value.strip()
        ]
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
