from __future__ import annotations

import atexit
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from slime.rollout.fused_agent.history import strip_trailing_chat_template_stop
from slime.rollout.fused_agent.parser import ToolCall as SlimeToolCall
from slime.rollout.fused_agent.parser import make_tool_parser


PROTOCOL_VERSION = 2
AGENT_INSTRUCTION = """You are a tool agent helping a user under the supplied domain policy.
For mobile_data_issue and mms_issue troubleshooting, never transfer to a human while any listed repair step remains
unattempted; execute the supplied repair tools in policy order first.
For these connectivity tasks, do not investigate bills or payments unless the task explicitly includes an overdue-bill
or suspension condition; stay focused on the listed network, device, and MMS corrections.
Use the available tools step by step and communicate with the user whenever required information is missing.
Prefer read/query tools before consequential actions, provide complete and valid tool arguments, and adapt to tool results.
Only call functions listed in the supplied tool definitions. When the policy says that the user must perform an
action whose function is not supplied to you, ask the user to perform it in natural language; never emit a tool call
for that user action. If a matching device operation is supplied (for example, enable_roaming), call it directly. For
troubleshooting tasks, perform every applicable supplied corrective operation in policy order before transferring to a
human; do not transfer while a required check or correction remains unattempted.
Do not combine a user-facing reply and a tool call in the same turn. When the request is complete, clearly report the result."""

_THINK_RE = re.compile(r"\s*<think\b[^>]*>(.*?)</think\s*>\s*", re.DOTALL)
_HTTP_LOCAL = threading.local()
_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSIONS_LOCK = threading.Lock()
_NEXT_DP_RANK = 0


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    model_path: str
    context_length: int
    dp_size: int
    use_session: bool
    discard_historical_thinking: bool
    enable_thinking: bool
    remote: bool = False


_CONFIG: Config | None = None


def configure(**kwargs: Any) -> None:
    global _CONFIG
    _CONFIG = Config(**kwargs)


def _config() -> Config:
    if _CONFIG is None:
        raise RuntimeError("tau2 fused transport is not configured")
    return _CONFIG


def _post(url: str, payload: dict[str, Any], *, attempts: int = 4) -> Any:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _HTTP_LOCAL.session = session
    delay = 1.0
    for attempt in range(attempts):
        try:
            response = session.post(
                url,
                json=payload,
                headers={"Authorization": "Bearer EMPTY"},
                timeout=(10, 600),
            )
            if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt + 1 < attempts:
                time.sleep(delay)
                delay = min(delay * 2, 15)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 15)
    raise RuntimeError("unreachable")


def _schemas(tools: list[Any]) -> list[dict[str, Any]]:
    return [tool.openai_schema for tool in tools]


def _parser(tools: list[Any]):
    schemas = _schemas(tools)
    valid_tools = {schema["function"]["name"] for schema in schemas}
    return make_tool_parser(_config().model_path, valid_tools=valid_tools)


def normalize_external_user_message(message: Any) -> Any:
    if message.tool_calls and message.content:
        message.content = None
    return message


def _render_messages(messages: list[Any], tools: list[Any], parser: Any) -> list[dict[str, str]]:
    from tau2.data_model.message import AssistantMessage, SystemMessage, ToolMessage, UserMessage

    schemas = "\n".join(json.dumps(schema, ensure_ascii=False) for schema in _schemas(tools))
    rendered: list[dict[str, str]] = []
    tools_injected = False
    tool_names_by_id: dict[str, str] = {}
    for message in messages:
        if isinstance(message, SystemMessage):
            content = str(message.content or "").strip()
            if not tools_injected:
                content += "\n" + parser.get_tool_prompt(schemas)
                tools_injected = True
            rendered.append({"role": "system", "content": content})
        elif isinstance(message, UserMessage):
            rendered.append({"role": "user", "content": str(message.content or "")})
        elif isinstance(message, ToolMessage):
            tool_name = tool_names_by_id.get(str(message.id), "tool")
            rendered.append(
                {
                    "role": "user",
                    "content": parser.format_tool_observation(
                        tool_name, str(message.content or "")
                    ),
                }
            )
        elif isinstance(message, AssistantMessage):
            for call in message.tool_calls or []:
                tool_names_by_id[str(call.id)] = str(call.name)
            raw_content = message.raw_data.get("slime_fused_content") if message.raw_data else None
            if isinstance(raw_content, str) and not _config().discard_historical_thinking:
                # Keep tool markup and visible replies, but do not replay old chain-of-thought.
                content = _THINK_RE.sub("", raw_content).strip()
            else:
                content = str(message.content or "")
                if message.tool_calls:
                    actions = [SlimeToolCall(name=call.name, arguments=call.arguments) for call in message.tool_calls]
                    content += ("\n" if content else "") + "".join(parser.format_action(action) for action in actions)
            rendered.append({"role": "assistant", "content": content})
    return rendered


def _last_session_id(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        raw_data = getattr(message, "raw_data", None)
        if raw_data and raw_data.get("slime_fused_session_id"):
            return str(raw_data["slime_fused_session_id"])
    return None


def _close_session(session_id: str) -> None:
    with _SESSIONS_LOCK:
        state = _SESSIONS.pop(session_id, None)
    if state and state.get("opened"):
        try:
            _post(f'{state["root"]}/close_session', {"session_id": session_id}, attempts=1)
        except Exception:
            pass


def close_sessions(messages: list[Any] | None) -> None:
    for message in messages or []:
        raw_data = getattr(message, "raw_data", None)
        if raw_data and raw_data.get("slime_fused_session_id"):
            _close_session(str(raw_data["slime_fused_session_id"]))


def close_thread_sessions() -> None:
    owner_thread = threading.get_ident()
    with _SESSIONS_LOCK:
        session_ids = [session_id for session_id, state in _SESSIONS.items() if state["owner_thread"] == owner_thread]
    for session_id in session_ids:
        _close_session(session_id)


def cleanup_sessions() -> None:
    with _SESSIONS_LOCK:
        session_ids = list(_SESSIONS)
    for session_id in session_ids:
        _close_session(session_id)


def _common_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def _finish_reason(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("type")
    return value if isinstance(value, str) and value else None


def _normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize lossless formatting variants emitted by the model."""
    normalized = dict(arguments)
    phone = normalized.get("phone_number")
    if isinstance(phone, str):
        digits = re.sub(r"\D", "", phone)
        if len(digits) == 10:
            normalized["phone_number"] = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    return normalized


def _session_generate(messages: list[Any], prompt_ids: list[int], sampling: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    global _NEXT_DP_RANK
    config = _config()
    session_id = _last_session_id(messages)
    with _SESSIONS_LOCK:
        state = _SESSIONS.get(session_id) if session_id else None
        if state is None:
            session_id = uuid.uuid4().hex
            state = {
                "root": config.base_url.removesuffix("/v1"),
                "dp_rank": _NEXT_DP_RANK % max(config.dp_size, 1),
                "opened": False,
                "rid": None,
                "expected": None,
                "owner_thread": threading.get_ident(),
            }
            _NEXT_DP_RANK += 1
            _SESSIONS[session_id] = state
    try:
        session_open_seconds = 0.0
        if not state["opened"]:
            open_started = time.perf_counter()
            opened = _post(
                f'{state["root"]}/open_session',
                {"capacity_of_str_len": config.context_length * 8, "session_id": session_id, "timeout": 900},
            )
            if opened != session_id:
                raise RuntimeError(f"SGLang returned unexpected session id {opened!r}")
            state["opened"] = True
            session_open_seconds = time.perf_counter() - open_started
        request_ids = prompt_ids
        prefix = 0
        session_params: dict[str, Any] = {"id": session_id, "rid": state["rid"]}
        expected = state["expected"]
        if expected is not None:
            prefix = _common_prefix(prompt_ids, expected)
            if prefix == 0:
                raise RuntimeError("rendered prompt has no common prefix with the SGLang session")
            request_ids = prompt_ids[prefix:]
            if not request_ids:
                raise RuntimeError("session prompt delta is empty")
            if prefix != len(expected):
                session_params["offset"] = prefix
        output = _post(
            f'{state["root"]}/generate',
            {
                "rid": uuid.uuid4().hex,
                "input_ids": request_ids,
                "sampling_params": sampling,
                "session_params": session_params,
                "routed_dp_rank": state["dp_rank"],
            },
            attempts=1,
        )
        meta = output.get("meta_info") or {}
        output_ids = output.get("output_ids") or []
        rid = meta.get("id") or output.get("rid")
        if output.get("text") and (not output_ids or not rid):
            raise ValueError("SGLang session response omitted output token IDs or request ID")
        state["rid"] = rid
        state["expected"] = [*prompt_ids, *output_ids]
        return str(output.get("text") or ""), {
            "slime_fused_session_id": session_id,
            "slime_fused_dp_rank": state["dp_rank"],
            "slime_fused_completion_tokens": len(output_ids),
            "slime_fused_session_delta_tokens": len(request_ids),
            "slime_fused_session_tokens_avoided": prefix,
            "slime_fused_session_rollback_tokens": len(expected) - prefix if expected is not None else 0,
            "slime_fused_session_open_seconds": session_open_seconds,
            "slime_fused_finish_reason": _finish_reason(meta.get("finish_reason")),
        }
    except Exception:
        _close_session(session_id)
        raise


def _stateless_generate(rendered: list[dict[str, str]], sampling: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = _config()
    response = _post(
        f'{config.base_url.rstrip("/")}/chat/completions',
        {
            "model": config.model,
            "messages": rendered,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
            **sampling,
        },
    )
    usage = response.get("usage") or {}
    choice = response["choices"][0]
    return str(choice["message"].get("content") or ""), {
        "slime_fused_session_fallback": True,
        "slime_fused_completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "slime_fused_finish_reason": _finish_reason(choice.get("finish_reason")),
    }


def generate(
    model: str,
    messages: list[Any],
    tools: list[Any] | None = None,
    tool_choice: str | None = None,
    call_name: str | None = None,
    **kwargs: Any,
) -> Any:
    del model, tool_choice, call_name
    from tau2.data_model.message import AssistantMessage, ToolCall

    generate_started = time.perf_counter()
    tools = tools or []
    parser = _parser(tools)
    render_started = time.perf_counter()
    rendered = _render_messages(messages, tools, parser)
    render_seconds = time.perf_counter() - render_started
    config = _config()
    tokenize_started = time.perf_counter()
    tokenized = _post(
        f'{config.base_url.rstrip("/")}/tokenize',
        {
            "model": config.model,
            "messages": rendered,
            "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        },
    )
    tokenize_seconds = time.perf_counter() - tokenize_started
    prompt_ids = tokenized.get("tokens")
    if not isinstance(prompt_ids, list) or any(not isinstance(token, int) for token in prompt_ids):
        raise ValueError("SGLang /tokenize returned invalid token IDs")
    remaining = config.context_length - len(prompt_ids)
    if remaining <= 0:
        raise ValueError(f"fused GEM prompt exceeds the {config.context_length}-token context limit")
    max_tokens = min(int(kwargs.get("max_tokens") or remaining), remaining)
    sampling = {
        "temperature": kwargs.get("temperature", 0.0),
        "top_p": kwargs.get("top_p", 1.0),
        "top_k": kwargs.get("top_k", -1),
        "max_tokens": max_tokens,
    }
    if kwargs.get("seed") is not None:
        sampling["seed"] = kwargs["seed"]
    if config.use_session:
        session_sampling = dict(sampling)
        session_sampling["max_new_tokens"] = session_sampling.pop("max_tokens")
        if "seed" in session_sampling:
            session_sampling["sampling_seed"] = session_sampling.pop("seed")
        session_sampling.update(
            {"skip_special_tokens": False, "spaces_between_special_tokens": False, "no_stop_trim": True}
        )
        try:
            inference_started = time.perf_counter()
            raw_content, metadata = _session_generate(messages, prompt_ids, session_sampling)
            inference_seconds = time.perf_counter() - inference_started
        except Exception as exc:
            inference_started = time.perf_counter()
            raw_content, metadata = _stateless_generate(rendered, sampling)
            inference_seconds = time.perf_counter() - inference_started
            metadata["slime_fused_session_error"] = f"{type(exc).__name__}: {exc}"
    else:
        inference_started = time.perf_counter()
        raw_content, metadata = _stateless_generate(rendered, sampling)
        inference_seconds = time.perf_counter() - inference_started
    raw_content = strip_trailing_chat_template_stop(raw_content)
    parsed_actions = parser.parse(raw_content)
    think_match = _THINK_RE.match(raw_content)
    visible_content = raw_content[think_match.end() :] if think_match else raw_content
    for action in reversed(parsed_actions):
        if action.start is None or action.end is None:
            continue
        start = action.start - (think_match.end() if think_match else 0)
        end = action.end - (think_match.end() if think_match else 0)
        if 0 <= start <= end <= len(visible_content):
            visible_content = visible_content[:start] + visible_content[end:]
    tau_tool_calls = [
        ToolCall(
            id=uuid.uuid4().hex,
            name=action.name,
            arguments=_normalize_tool_arguments(action.arguments),
        )
        for action in parsed_actions
    ]
    usage = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": int(metadata.get("slime_fused_completion_tokens", 0)),
    }
    total_seconds = time.perf_counter() - generate_started
    return AssistantMessage(
        role="assistant",
        content=None if tau_tool_calls else (visible_content.strip() or None),
        tool_calls=tau_tool_calls or None,
        cost=0.0,
        usage=usage,
        generation_time_seconds=total_seconds,
        raw_data={
            "slime_fused_content": raw_content,
            "slime_fused_protocol_version": PROTOCOL_VERSION,
            "slime_fused_render_seconds": render_seconds,
            "slime_fused_tokenize_seconds": tokenize_seconds,
            "slime_fused_inference_seconds": inference_seconds,
            "slime_fused_total_seconds": total_seconds,
            "slime_fused_prompt_tokens": len(prompt_ids),
            **metadata,
        },
    )


atexit.register(cleanup_sessions)
