from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import pytest

from slime_plugins.evals import tau2_fused_transport as transport

NUM_GPUS = 0


@dataclass
class SystemMessage:
    content: str


@dataclass
class UserMessage:
    content: str


@dataclass
class ToolMessage:
    content: str
    id: str = "1"


@dataclass
class TauToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[TauToolCall] | None = None
    cost: float = 0
    usage: dict | None = None
    raw_data: dict = field(default_factory=dict)
    generation_time_seconds: float | None = None


class Tool:
    openai_schema = {
        "type": "function",
        "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}},
    }


class Parser:
    def get_tool_prompt(self, schemas):
        return "TOOLS:" + schemas

    def format_tool_observation(self, name, content):
        return f"OBS:{name}:{content}"

    def format_action(self, action):
        return f'<tool_call>{{"name":"{action.name}"}}</tool_call>'

    def parse(self, content):
        marker = '<tool_call>{"name":"lookup"}</tool_call>'
        if marker not in content:
            return []
        start = content.index(marker)
        return [transport.SlimeToolCall(name="lookup", arguments={}, start=start, end=start + len(marker))]


def _install_tau_message_module(monkeypatch):
    module = types.ModuleType("tau2.data_model.message")
    module.SystemMessage = SystemMessage
    module.UserMessage = UserMessage
    module.ToolMessage = ToolMessage
    module.AssistantMessage = AssistantMessage
    module.ToolCall = TauToolCall
    monkeypatch.setitem(sys.modules, "tau2", types.ModuleType("tau2"))
    monkeypatch.setitem(sys.modules, "tau2.data_model", types.ModuleType("tau2.data_model"))
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", module)


def _configure(**overrides):
    values = {
        "base_url": "http://127.0.0.1:18081/v1",
        "model": "checkpoint",
        "model_path": "/models/qwen3",
        "context_length": 40960,
        "dp_size": 2,
        "use_session": False,
        "discard_historical_thinking": False,
        "enable_thinking": True,
    }
    values.update(overrides)
    transport.configure(**values)


def test_render_uses_slime_tool_definition_action_and_observation(monkeypatch):
    _install_tau_message_module(monkeypatch)
    _configure()
    parser = Parser()
    prior = AssistantMessage(
        content=None,
        tool_calls=[TauToolCall(id="1", name="lookup", arguments={})],
        raw_data={"slime_fused_content": "<think>reason</think>RAW"},
    )
    rendered = transport._render_messages(
        [SystemMessage("policy"), prior, ToolMessage("result")], [Tool()], parser
    )
    assert rendered[0]["content"].startswith("policy\nTOOLS:")
    assert rendered[1]["content"] == "<think>reason</think>RAW"
    assert rendered[2] == {"role": "user", "content": "OBS:lookup:result"}

    _configure(discard_historical_thinking=True)
    rendered = transport._render_messages([SystemMessage("policy"), prior], [Tool()], parser)
    assert rendered[1]["content"] == '<tool_call>{"name":"lookup"}</tool_call>'


def test_generate_parses_raw_fused_action_and_keeps_wire_content(monkeypatch):
    _install_tau_message_module(monkeypatch)
    _configure()
    monkeypatch.setattr(transport, "_parser", lambda tools: Parser())
    responses = iter(
        [
            {"tokens": [1, 2, 3], "count": 3},
            {
                "choices": [
                    {
                        "message": {"content": '<think>reason</think>done<tool_call>{"name":"lookup"}</tool_call>'},
                        "finish_reason": "length",
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(transport, "_post", lambda *args, **kwargs: next(responses))
    result = transport.generate("unused", [SystemMessage("policy"), UserMessage("request")], [Tool()])
    assert result.content is None
    assert result.tool_calls[0].name == "lookup"
    assert result.raw_data["slime_fused_protocol_version"] == 2
    assert result.raw_data["slime_fused_content"].startswith("<think>reason</think>")
    assert result.raw_data["slime_fused_prompt_tokens"] == 3
    assert result.raw_data["slime_fused_render_seconds"] >= 0
    assert result.raw_data["slime_fused_tokenize_seconds"] >= 0
    assert result.raw_data["slime_fused_inference_seconds"] >= 0
    assert result.raw_data["slime_fused_total_seconds"] >= 0
    assert result.raw_data["slime_fused_finish_reason"] == "length"


def test_external_user_tool_call_drops_mixed_visible_content():
    message = AssistantMessage(
        content="I will run that now.",
        tool_calls=[TauToolCall(id="1", name="lookup", arguments={})],
    )

    assert transport.normalize_external_user_message(message) is message
    assert message.content is None
    assert message.tool_calls[0].name == "lookup"


def test_native_session_routes_sticky_dp_and_sends_only_prompt_delta(monkeypatch):
    _configure(use_session=True)
    transport.cleanup_sessions()
    calls = []

    def post(url, payload, **kwargs):
        calls.append((url, payload))
        if url.endswith("/open_session"):
            return payload["session_id"]
        return {
            "text": "ok",
            "output_ids": [9],
            "meta_info": {"id": "rid-1", "finish_reason": {"type": "length"}},
        }

    monkeypatch.setattr(transport, "_post", post)
    first_text, first_meta = transport._session_generate(
        [], [1, 2], {"max_new_tokens": 8, "sampling_seed": 42}
    )
    history = [AssistantMessage(raw_data={"slime_fused_session_id": first_meta["slime_fused_session_id"]})]
    second_text, second_meta = transport._session_generate(history, [1, 2, 9, 3], {"max_new_tokens": 8})

    generate_calls = [payload for url, payload in calls if url.endswith("/generate")]
    assert first_text == second_text == "ok"
    assert generate_calls[0]["input_ids"] == [1, 2]
    assert generate_calls[0]["sampling_params"]["sampling_seed"] == 42
    assert generate_calls[1]["input_ids"] == [3]
    assert generate_calls[0]["routed_dp_rank"] == generate_calls[1]["routed_dp_rank"]
    assert second_meta["slime_fused_session_id"] == first_meta["slime_fused_session_id"]
    assert first_meta["slime_fused_session_delta_tokens"] == 2
    assert first_meta["slime_fused_session_tokens_avoided"] == 0
    assert first_meta["slime_fused_finish_reason"] == "length"
    assert second_meta["slime_fused_session_delta_tokens"] == 1
    assert second_meta["slime_fused_session_tokens_avoided"] == 3
    transport.cleanup_sessions()


def test_native_session_rolls_back_changed_generation_suffix(monkeypatch):
    _configure(use_session=True)
    transport.cleanup_sessions()
    calls = []

    def post(url, payload, **kwargs):
        calls.append((url, payload))
        if url.endswith("/open_session"):
            return payload["session_id"]
        return {"text": "ok", "output_ids": [8, 9], "meta_info": {"id": "rid-1"}}

    monkeypatch.setattr(transport, "_post", post)
    _, first_meta = transport._session_generate([], [1, 2, 3], {"max_new_tokens": 8})
    history = [AssistantMessage(raw_data={"slime_fused_session_id": first_meta["slime_fused_session_id"]})]
    _, second_meta = transport._session_generate(history, [1, 2, 4, 5], {"max_new_tokens": 8})

    second_request = [payload for url, payload in calls if url.endswith("/generate")][1]
    assert second_request["input_ids"] == [4, 5]
    assert second_request["session_params"]["offset"] == 2
    assert second_meta["slime_fused_session_rollback_tokens"] == 3
    transport.cleanup_sessions()


def test_generate_translates_trial_seed_for_native_sglang(monkeypatch):
    _install_tau_message_module(monkeypatch)
    _configure(use_session=True)
    monkeypatch.setattr(transport, "_parser", lambda tools: Parser())
    monkeypatch.setattr(transport, "_post", lambda *args, **kwargs: {"tokens": [1], "count": 1})
    captured = {}

    def session_generate(messages, prompt_ids, sampling):
        captured.update(sampling)
        return "done", {"slime_fused_session_id": "session"}

    monkeypatch.setattr(transport, "_session_generate", session_generate)
    result = transport.generate(
        "unused", [SystemMessage("policy"), UserMessage("request")], [Tool()], seed=123
    )
    assert result.content == "done"
    assert captured["sampling_seed"] == 123
    assert "seed" not in captured


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
