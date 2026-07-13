import json
import os
import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from slime.rollout.fused_agent.env import FusedEnvironment, _exact_match_reward, _format_retrieval, normalize_task, resolve_task_mode
import slime.rollout.fused_agent.generate as fused_generate
from slime.rollout.fused_agent.generate import (
    _format_tool_observation,
    _initial_messages,
    _last_assistant_context_start_idx,
    _render_prompt_ids,
    _default_response_loss_mask,
    _valid_tool_names,
)
from slime.rollout.fused_agent.parser import (
    Gemma4ToolParser,
    Qwen3CoderToolParser,
    QwenToolParser,
    ToolCall,
    make_tool_parser,
    tool_schema,
)
from slime.rollout.fused_agent.prompts import (
    FUSED_SEARCH_SYSTEM_PROMPT,
    build_system_prompt,
    finish_schema,
    web_search_schema,
)
from slime.utils import visualization as rollout_visualization
from slime.utils.types import Sample


NUM_GPUS = 0


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = "\n".join(m["role"] + ":" + m["content"] for m in messages)
        if add_generation_prompt:
            text = f"{text}\nassistant:"
        return [ord(c) % 251 for c in text] if tokenize else text

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class FakeChatTemplateTokenizer:
    def __init__(self, *, drift_assistant_end: bool = False):
        self.drift_assistant_end = drift_assistant_end

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        chunks = []
        for m in messages:
            end = "<|im_end_drift|>" if self.drift_assistant_end and m["role"] == "assistant" else "<|im_end|>"
            content = str(m["content"])
            if self.drift_assistant_end and m["role"] == "assistant":
                content = content.replace("</tool_call>", "</tool_call_drift>", 1)
            chunks.append(f"<|im_start|>{m['role']}\n{content}{end}\n")
        text = "".join(chunks)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return [ord(c) for c in text] if tokenize else text

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class FakeGemma4Tokenizer:
    name_or_path = "/share/nlp/share/plm/gemma-4-E2B-it"
    unk_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **_kwargs):
        chunks = ["<bos>"]
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            content = str(message.get("content") or "")
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            body = ""
            if role == "model" and reasoning:
                body += f"<|channel>thought\n{reasoning}\n<channel|>"
            if role == "system" and tools:
                for schema in tools:
                    body += f"<|tool>{Gemma4ToolParser._format_function_declaration(schema)}<tool|>"
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", tool_call)
                body += (
                    f"<|tool_call>call:{function['name']}"
                    f"{Gemma4ToolParser._format_argument(function.get('arguments') or {}, escape_keys=False)}"
                    "<tool_call|>"
                )
            if message.get("tool_responses"):
                body += "<|tool_response>"
                for response in message["tool_responses"]:
                    body += (
                        f"response:{response['name']}"
                        f"{Gemma4ToolParser._format_argument(response.get('response') or {}, escape_keys=False)}"
                    )
                body += "<tool_response|>"
            body += content
            chunks.append(f"<|turn>{role}\n{body}<turn|>\n")
        if add_generation_prompt:
            chunks.append("<|turn>model\n")
        text = "".join(chunks)
        return [ord(ch) for ch in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id


def _run_generate_with_fake_sglang(
    sample: Sample,
    calls: list[dict],
    env: dict[str, str] | None = None,
    *,
    evaluation: bool = False,
    tokenizer=None,
):
    async def fake_call_sglang(
        args,
        prompt_ids,
        sampling_params,
        *,
        session_id,
        evaluation=False,
        request_semaphore=None,
        session_params=None,
        context_token_count=None,
        server_url=None,
    ):
        item = calls.pop(0)
        text = item["text"]
        return {
            "text": text,
            "output_ids": [ord(c) for c in text],
            "output_logprobs": [-0.1] * len(text),
            "finish_reason": item.get("finish_reason", "stop"),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(text),
        }

    old_generate_state = fused_generate.GenerateState
    old_call = fused_generate._call_sglang
    old_env = {key: os.environ.get(key) for key in (env or {})}
    if env:
        os.environ.update(env)
    fake_tokenizer = tokenizer or FakeTokenizer()
    fused_generate.GenerateState = lambda args: SimpleNamespace(
        tokenizer=fake_tokenizer,
        semaphore=asyncio.Semaphore(1000),
    )
    fused_generate._call_sglang = fake_call_sglang
    try:
        import asyncio

        return asyncio.run(
            fused_generate.generate(
                SimpleNamespace(rollout_max_context_len=100000, fused_harness="unified_gem"),
                sample,
                {"max_new_tokens": 1024, "temperature": 1.0, "top_p": 1.0},
                evaluation=evaluation,
            )
        )
    finally:
        fused_generate.GenerateState = old_generate_state
        fused_generate._call_sglang = old_call
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _response_text(sample: Sample) -> str:
    return "".join(chr(tok) for tok in sample.tokens[-sample.response_length :])


def _masked_text(sample: Sample) -> str:
    return "".join(chr(tok) for tok, mask in zip(sample.tokens[-sample.response_length :], sample.loss_mask, strict=False) if mask)


def _unmasked_text(sample: Sample) -> str:
    return "".join(chr(tok) for tok, mask in zip(sample.tokens[-sample.response_length :], sample.loss_mask, strict=False) if not mask)


def _policy_masked_text(sample: Sample) -> str:
    policy_mask = sample.policy_loss_mask if sample.policy_loss_mask is not None else sample.loss_mask
    return "".join(chr(tok) for tok, mask in zip(sample.tokens[-sample.response_length :], policy_mask, strict=False) if mask)


def _policy_unmasked_text(sample: Sample) -> str:
    policy_mask = sample.policy_loss_mask if sample.policy_loss_mask is not None else sample.loss_mask
    return "".join(chr(tok) for tok, mask in zip(sample.tokens[-sample.response_length :], policy_mask, strict=False) if not mask)


def _visualized_text_by_style(sample: Sample, tokenizer, style: str) -> str:
    rendered = rollout_visualization._token_mask_text(sample, tokenizer)
    assert rendered is not None
    plain = rendered.plain
    return "".join(plain[span.start : span.end] for span in rendered.spans if str(span.style) == style)


def _visualized_text_by_styles(sample: Sample, tokenizer, styles: set[str]) -> str:
    rendered = rollout_visualization._token_mask_text(sample, tokenizer)
    assert rendered is not None
    plain = rendered.plain
    return "".join(plain[span.start : span.end] for span in rendered.spans if str(span.style) in styles)


def _visualized_text_by_styles_from_rendered(rendered, styles: set[str]) -> str:
    plain = rendered.plain
    return "".join(plain[span.start : span.end] for span in rendered.spans if str(span.style) in styles)


def _local_mcp_sample(tmp_path: Path, *, question: str = "Use tools") -> Sample:
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Echo value")
def echo(value: str) -> dict:
    return {"echo": value}
""",
        encoding="utf-8",
    )
    return Sample(
        prompt="placeholder",
        label=None,
        metadata={
            "question": question,
            "data_root": str(asset),
            "tools_py": str(asset / "tools.py"),
            "verifier": {
                "verification_code": """
def verify(tools, answer):
    return {"passed": isinstance(answer, dict) and answer.get("done") is True}
"""
            },
        },
    )


def _echo_call(value: str) -> str:
    return f'<tool_call>{{"name":"echo","arguments":{{"value":"{value}"}}}}</tool_call>'


def _finish_call(payload: str = '{\\"done\\":true}') -> str:
    return f'<tool_call>{{"name":"finish","arguments":{{"command":"submit","result":"{payload}"}}}}</tool_call>'


def _gemma4_echo_call(value: str) -> str:
    return f'<|tool_call>call:echo{{value:<|"|>{value}<|"|>}}<tool_call|>'


def _gemma4_finish_call(payload: str = '{"done":true}') -> str:
    return f'<|tool_call>call:finish{{command:<|"|>submit<|"|>,result:<|"|>{payload}<|"|>}}<tool_call|>'


def _search_call(query: str) -> str:
    return f'<tool_call>{{"name":"web_search","arguments":{{"query":"{query}","max_results":3}}}}</tool_call>'


def test_normalize_rllm_extra_info_task():
    row = {
        "prompt": [{"role": "user", "content": "placeholder"}],
        "data_source": "mcp",
        "extra_info": {
            "question": "Find data",
            "tools_py": "tools.py",
            "data_root": "asset",
        },
    }

    task = normalize_task(row)

    assert task["question"] == "Find data"
    assert task["tools_py"] == "tools.py"
    assert resolve_task_mode(task) == "mcp"


def test_mcp_task_dump_gets_stable_data_source(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path)
    task = fused_generate._task_from_sample(sample)

    dumped = fused_generate._task_for_dump(task)

    assert dumped["data_source"] == "mcp"


def test_max_steps_for_mode_uses_mode_specific_env(monkeypatch):
    monkeypatch.setenv("FUSED_MAX_STEPS", "96")
    monkeypatch.setenv("FUSED_MCP_MAX_STEPS", "16")
    monkeypatch.setenv("FUSED_WEB_SEARCH_MAX_STEPS", "4")

    assert fused_generate._max_steps_for_mode("mcp", 96) == 16
    assert fused_generate._max_steps_for_mode("web_search", 96) == 4
    assert fused_generate._max_steps_for_mode("cli", 96) == 96


def test_qwen_tool_parser_finish_and_answer_fallback():
    parser = QwenToolParser(valid_tools={"web_search", "finish"})

    calls = parser.parse('<tool_call>{"name":"web_search","arguments":{"query":"abc"}}</tool_call>')
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "abc"}
    assert parser.format_action(calls[0]) == '<tool_call>{"name":"web_search","arguments":{"query":"abc"}}</tool_call>'

    calls = parser.parse("reasoning\n\\boxed{Final}")
    assert calls[0].name == "finish"
    assert calls[0].arguments["result"] == "Final"

    calls = parser.parse('<tool_call>{"name":"submit","arguments":{}}</tool_call>')
    assert calls[0].name == "finish"

    calls = parser.parse('<tool_call>{"name":"finish","arguments":{"command":"submit","result":"The answer is \\\\boxed{Fixed Answer}.}}</tool_call>')
    assert calls[0].name == "finish"
    assert calls[0].arguments["result"] == "The answer is \\\\boxed{Fixed Answer}."


def test_qwen_tool_parser_accepts_legacy_function_actions():
    parser = QwenToolParser(valid_tools={"web_search", "finish"})

    calls = parser.parse(
        """
<think>
Need evidence.
</think>
<function=web_search>
  <parameter=query>report PRO-0824 title</parameter>
  <parameter=max_results>5</parameter>
</function>
"""
    )

    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "report PRO-0824 title", "max_results": "5"}


def test_qwen_tool_parser_records_tool_call_span():
    raw = '<think>reason</think>\n<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
    parser = QwenToolParser(valid_tools={"web_search"})

    action = parser.parse(raw)[0]

    assert raw[action.start : action.end] == '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'


def test_make_tool_parser_selects_by_model_name():
    assert type(make_tool_parser("/share/nlp/share/plm/gemma-4-E2B-it")) is Gemma4ToolParser
    assert type(make_tool_parser("/models/gemma4-12B-it")) is Gemma4ToolParser
    assert type(make_tool_parser("/share/nlp/share/plm/Qwen3.5-4B")) is Qwen3CoderToolParser
    assert type(make_tool_parser("/models/Qwen3-Coder-30B")) is Qwen3CoderToolParser
    # Qwen3 (and anything else) keeps the JSON parser unchanged.
    assert type(make_tool_parser("/share/nlp/share/plm/Qwen3-4B")) is QwenToolParser
    assert type(make_tool_parser(None)) is QwenToolParser


def test_qwen3_coder_parser_parses_xml_calls_with_schema_coercion():
    tools = [web_search_schema(), finish_schema()]
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search", "finish", "submit"})
    # get_tool_prompt loads the parameter config used for type coercion.
    parser.get_tool_prompt("\n".join(json.dumps(t, indent=0, ensure_ascii=False) for t in tools))

    raw = "<tool_call>\n<function=web_search>\n" "<parameter=query>\nsenior comic artist\n</parameter>\n" "<parameter=max_results>\n4\n</parameter>\n" "</function>\n</tool_call>"
    calls = parser.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    # max_results is coerced to int per the schema; query stays a string.
    assert calls[0].arguments == {"query": "senior comic artist", "max_results": 4}
    # start/end bound the full <tool_call> span so verifier finish checks work.
    assert raw[calls[0].start : calls[0].end].startswith("<tool_call>")

    action_text = parser.format_action(calls[0])
    assert action_text == (
        "<tool_call>\n<function=web_search>\n"
        "<parameter=query>\nsenior comic artist\n</parameter>\n"
        "<parameter=max_results>\n4\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert parser.parse(action_text)[0].arguments == {"query": "senior comic artist", "max_results": 4}


def test_initial_messages_configure_runtime_qwen35_parser_schema():
    tools = [web_search_schema(), finish_schema()]
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search", "finish", "submit"})

    _initial_messages("gem", "webqa", "question", tools, "Qwen3.5-4B", tool_parser=parser)
    call = parser.parse(
        "<tool_call>\n<function=web_search>\n"
        "<parameter=query>\nquery\n</parameter>\n"
        "<parameter=max_results>\n10\n</parameter>\n"
        "</function>\n</tool_call>"
    )[0]

    assert call.arguments == {"query": "query", "max_results": 10}


def test_qwen3_coder_parser_normalizes_submit_and_keeps_boxed_fallback():
    parser = make_tool_parser("qwen3-coder", valid_tools={"web_search", "finish", "submit"})

    submit = parser.parse("<tool_call>\n<function=submit>\n" "<parameter=command>submit</parameter>\n" "<parameter=result>Milton Caniff</parameter>\n" "</function>\n</tool_call>")
    assert submit[0].name == "finish"
    assert submit[0].arguments["result"] == "Milton Caniff"

    boxed = parser.parse("reasoning without any tool call\n\\boxed{42}")
    assert boxed[0].name == "finish"
    assert boxed[0].arguments["result"] == "42"


def test_gemma4_tool_prompt_uses_native_declarations():
    tools = [web_search_schema(), finish_schema()]
    parser = make_tool_parser("gemma-4-E2B-it", valid_tools={"web_search", "finish"})

    prompt = parser.get_tool_prompt("\n".join(json.dumps(t, indent=0, ensure_ascii=False) for t in tools))

    assert "<|tool>declaration:web_search{" in prompt
    assert "query:{description:<|\"|>Search query.<|\"|>,type:<|\"|>STRING<|\"|>}" in prompt
    assert "<|tool_call>call:TOOL_NAME{" in prompt
    assert "<tool_call|><|tool_response>" in prompt


def test_gemma4_tool_prompt_formats_complex_json_schema_without_python_repr():
    schema = {
        "type": "function",
        "function": {
            "name": "complex",
            "description": "Complex schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "display name": {"type": "string", "description": "Human readable name"},
                    "maybe": {"type": ["string", "null"], "description": "Optional text"},
                    "mode": {"anyOf": [{"type": "string", "enum": ["fast", "safe"]}, {"type": "null"}]},
                    "nums": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "description": "Numbers",
                        "default": [1, 2],
                        "minItems": 1,
                        "maxItems": 4,
                    },
                    "pair": {"type": "array", "items": [{"type": "string"}, {"type": "integer"}]},
                    "payload": {"type": "object", "additionalProperties": {"type": "string"}},
                    "profile": {"$ref": "#/$defs/Profile"},
                    "sealed": {"type": "object", "additionalProperties": False},
                    "token": {
                        "type": "string",
                        "const": "ok",
                        "pattern": "^[a-z]+$",
                        "minLength": 2,
                        "maxLength": 8,
                        "$defs": {"Alias": {"type": "string", "description": "Short name"}},
                    },
                },
                "required": ["nums"],
                "$defs": {
                    "Profile": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                        "required": ["name"],
                    }
                },
                "definitions": {"legacy type": {"type": "string", "enum": ["old"]}},
            },
        },
    }

    declaration = Gemma4ToolParser._format_function_declaration(schema)

    assert '<|"|>display name<|"|>:{description:<|"|>Human readable name<|"|>,type:<|"|>STRING<|"|>}' in declaration
    assert "maybe:{description:<|\"|>Optional text<|\"|>,type:[<|\"|>STRING<|\"|>,<|\"|>NULL<|\"|>]}" in declaration
    assert (
        "nums:{description:<|\"|>Numbers<|\"|>,default:[1,2],items:{minimum:1,type:<|\"|>INTEGER<|\"|>},"
        "minItems:1,maxItems:4,type:<|\"|>ARRAY<|\"|>}"
    ) in declaration
    assert "pair:{items:[{type:<|\"|>STRING<|\"|>},{type:<|\"|>INTEGER<|\"|>}],type:<|\"|>ARRAY<|\"|>}" in declaration
    assert "mode:{anyOf:[{enum:[<|\"|>fast<|\"|>,<|\"|>safe<|\"|>],type:<|\"|>STRING<|\"|>},{type:<|\"|>NULL<|\"|>}]}" in declaration
    assert "payload:{additionalProperties:{type:<|\"|>STRING<|\"|>},type:<|\"|>OBJECT<|\"|>}" in declaration
    assert "profile:{$ref:<|\"|>#/$defs/Profile<|\"|>}" in declaration
    assert "sealed:{additionalProperties:false,type:<|\"|>OBJECT<|\"|>}" in declaration
    assert (
        "token:{const:<|\"|>ok<|\"|>,$defs:{Alias:{description:<|\"|>Short name<|\"|>,type:<|\"|>STRING<|\"|>}},"
        "pattern:<|\"|>^[a-z]+$<|\"|>,minLength:2,maxLength:8,type:<|\"|>STRING<|\"|>}"
    ) in declaration
    assert (
        "$defs:{Profile:{properties:{age:{type:<|\"|>INTEGER<|\"|>},name:{type:<|\"|>STRING<|\"|>}},"
        "required:[<|\"|>name<|\"|>],type:<|\"|>OBJECT<|\"|>}}"
    ) in declaration
    assert 'definitions:{<|"|>legacy type<|"|>:{enum:[<|"|>old<|"|>],type:<|"|>STRING<|"|>}}' in declaration
    assert "['STRING', 'NULL']" not in declaration


def test_gemma4_tool_parser_parses_native_calls_and_formats_observation():
    parser = make_tool_parser("gemma4", valid_tools={"web_search", "finish", "submit"})

    calls = parser.parse(
        '<|tool_call>call:web_search{query:<|"|>Tokyo weather<|"|>,max_results:3}<tool_call|><|tool_response>'
    )

    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "Tokyo weather", "max_results": 3}

    action_text = parser.format_action(calls[0])
    assert action_text == '<|tool_call>call:web_search{max_results:3,query:<|"|>Tokyo weather<|"|>}<tool_call|>'

    observation = parser.format_tool_observation("web_search", "Sunny")
    assert observation == '<|tool_response>response:web_search{value:<|"|>Sunny<|"|>}<tool_response|>'


def test_gemma4_formatter_uses_json_string_when_value_contains_native_quote_marker():
    parser = make_tool_parser("gemma4", valid_tools={"echo"})
    value = 'prefix <|"|> sentinel suffix'

    action_text = parser.format_action(ToolCall("echo", {"value": value}))
    calls = parser.parse(action_text)

    assert '<|"|>prefix <|"|> sentinel suffix<|"|>' not in action_text
    assert '"prefix <|\\"|> sentinel suffix"' in action_text
    assert len(calls) == 1
    assert calls[0].arguments == {"value": value}

    observation = parser.format_tool_observation("echo", value)
    assert '<|"|>prefix <|"|> sentinel suffix<|"|>' not in observation
    assert 'response:echo{value:"prefix <|\\"|> sentinel suffix"}' in observation


def test_gemma4_formatter_quotes_unsafe_argument_keys_for_round_trip():
    parser = make_tool_parser("gemma4", valid_tools={"echo"})
    action = ToolCall("echo", {"display name": "Gemma 4", "safe_key": 7})

    action_text = parser.format_action(action)
    calls = parser.parse(action_text)

    assert '<|"|>display name<|"|>:<|"|>Gemma 4<|"|>' in action_text
    assert "safe_key:7" in action_text
    assert len(calls) == 1
    assert calls[0].arguments == {"display name": "Gemma 4", "safe_key": 7}


def test_gemma4_tool_response_template_quotes_unsafe_response_keys():
    parser = make_tool_parser("gemma4", valid_tools={"echo"})
    message = parser.assistant_tool_results_message(
        [ToolCall("echo", {"value": "x"})],
        [{"display name": "Gemma 4", "safe_key": 7}],
    )

    rendered = FakeGemma4Tokenizer().apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert 'response:echo{<|"|>display name<|"|>:<|"|>Gemma 4<|"|>,safe_key:7}' in rendered


def test_gemma4_tool_parser_formats_multiple_tool_responses():
    parser = make_tool_parser("gemma4", valid_tools={"first", "second"})
    actions = [
        ToolCall("first", {"value": "a"}),
        ToolCall("second", {"value": "b"}),
    ]

    message = parser.assistant_tool_results_message(actions, [{"ok": 1}, "plain"])

    assert message == {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "first", "arguments": {"value": "a"}}},
            {"function": {"name": "second", "arguments": {"value": "b"}}},
        ],
        "tool_responses": [
            {"name": "first", "response": {"ok": 1}},
            {"name": "second", "response": {"value": "plain"}},
        ],
    }


def test_gemma4_tool_parser_handles_nested_arguments_and_finish_alias():
    parser = make_tool_parser("gemma_4", valid_tools={"finish", "submit"})

    calls = parser.parse(
        '<|tool_call>call:submit{command:<|"|>submit<|"|>,result:{answer:<|"|>42<|"|>,sources:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>'
    )

    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "submit", "result": {"answer": "42", "sources": ["a", "b"]}}


def test_gemma4_tool_parser_accepts_python_style_bare_literals():
    parser = make_tool_parser("gemma4", valid_tools={"set_flags"})

    calls = parser.parse(
        '<|tool_call>call:set_flags{enabled:True,disabled:False,missing:None,quoted:"True"}<tool_call|>'
    )

    assert len(calls) == 1
    assert calls[0].name == "set_flags"
    assert calls[0].arguments == {
        "enabled": True,
        "disabled": False,
        "missing": None,
        "quoted": "True",
    }


def test_gemma4_tool_parser_does_not_truncate_braces_inside_strings():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    calls = parser.parse(
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>The answer is \\boxed{Chacruna}. {"ok": true}<|"|>}<tool_call|><|tool_response>'
    )

    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {
        "command": "submit",
        "result": 'The answer is \\boxed{Chacruna}. {"ok": true}',
    }
    assert calls[0].end == calls[0].start + len(
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>The answer is \\boxed{Chacruna}. {"ok": true}<|"|>}<tool_call|>'
    )


def test_gemma4_tool_parser_parses_logged_finish_call():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    calls = parser.parse('<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>Chacruna<|"|>}<tool_call|><|tool_response>')

    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "submit", "result": "Chacruna"}


def test_gemma4_tool_parser_repairs_finish_command_missing_native_quote_before_result(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>submit,result:<|"|>answer<|"|>}<tool_call|><|tool_response>'
        )

    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "submit", "result": "answer"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_finish_command_value_before_result(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>finish,result:<|"|>answer<|"|>}<tool_call|><|tool_response>'
        )

    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "finish", "result": "answer"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_finish_result_with_extra_native_marker(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>submit<|"|>,'
            'result:<|"|>The literal marker <|"|> appeared in prose.<|"|>}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {
        "command": "submit",
        "result": 'The literal marker <|"|> appeared in prose.',
    }
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_missing_command_quote_with_json_quoted_result(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>submit,result:"Could not identify \\"Creek E\\"."}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "submit", "result": 'Could not identify \\"Creek E\\".'}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_finish_function_like_command_payload(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>finish(result="I am unable to identify it.")"}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "finish", "result": "I am unable to identify it."}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_finish_newline_command_payload(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>finish\nThe question asks for a gene.}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "finish", "result": "The question asks for a gene."}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_finish_braced_result_payload(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>finish{result:<|"|>Wordian formation<|"|>}}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "finish", "result": "Wordian formation"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_finish_json_answer_payload(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>submit,{"answer":"High Performance Fortran"}}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "submit", "result": "High Performance Fortran"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_incomplete_finish_json_answer_payload(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>submit,{"answer":"High Performance Fortran}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "submit", "result": "High Performance Fortran"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_instruction_like_finish_command(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>Finish the task and submit the final result with the synthesized answer.<|"|>,'
            'result:<|"|>Journal of Hand Surgery<|"|>}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "submit", "result": "Journal of Hand Surgery"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_function_like_command_argument(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>finish(command="Which theory was used with Classical Laminate Theory?")<|"|>,'
            'result="The provided search results do not name it}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {
        "command": "finish",
        "result": "The provided search results do not name it",
    }
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_unclosed_finish_result(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:finish{command:<|"|>Please synthesize the search results to answer the question.<|"|>,'
            'result:<|"|>Based on the search results, the relevant journal is Journal of Hand Surgery'
            '<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {
        "command": "submit",
        "result": "Based on the search results, the relevant journal is Journal of Hand Surgery",
    }
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_web_search_query_equals(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:web_search{query="Greek phrase primary manuscript authority"}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"query": "Greek phrase primary manuscript authority"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_repairs_web_search_query_with_extra_quotes(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse(
            '<|tool_call>call:web_search{query:<|"|>EEG system adaptive gain control neuronal populations daily imaging"""}<tool_call|>'
        )

    assert len(calls) == 1
    assert calls[0].arguments == {"query": "EEG system adaptive gain control neuronal populations daily imaging"}
    assert "Failed to parse Gemma4 tool-call arguments" not in caplog.text


def test_gemma4_tool_parser_accepts_json_like_quoted_arguments():
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})

    calls = parser.parse('<|tool_call>call:web_search{query:"entity with spaces",max_results:3}<tool_call|>')

    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "entity with spaces", "max_results": 3}

    calls = parser.parse('<|tool_call>call:web_search{"query":"who founded Mimikama?","max_results":3}<tool_call|>')

    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "who founded Mimikama?", "max_results": 3}


def test_gemma4_tool_parser_decodes_json_like_unicode_escapes():
    parser = make_tool_parser("gemma4", valid_tools={"echo"})

    calls = parser.parse(
        '<|tool_call>call:echo{text:"\\u003cscript\\u003e",emoji:"\\ud83d\\ude00"}<tool_call|>'
    )

    assert len(calls) == 1
    assert calls[0].arguments == {"text": "<script>", "emoji": "😀"}


def test_gemma4_tool_parser_tolerates_unescaped_quotes_inside_json_like_strings():
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})

    calls = parser.parse('<|tool_call>call:web_search{query:"who wrote "In Context Music" with Angus Tarnawsky?",max_results:3}<tool_call|>')

    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {
        "query": 'who wrote "In Context Music" with Angus Tarnawsky?',
        "max_results": 3,
    }


def test_gemma4_tool_parser_accepts_braces_inside_json_like_quoted_strings():
    parser = make_tool_parser("gemma4", valid_tools={"web_search", "finish"})

    calls = parser.parse('<|tool_call>call:web_search{query:"literal } brace",max_results:3}<tool_call|>')
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "literal } brace", "max_results": 3}

    calls = parser.parse('<|tool_call>call:web_search{query:"literal { brace",max_results:3}<tool_call|>')
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "literal { brace", "max_results": 3}

    calls = parser.parse('<|tool_call>call:finish{command:"submit",result:"{\\"done\\":true}"}<tool_call|>')
    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "submit", "result": '{"done":true}'}


def test_gemma4_tool_parser_logs_malformed_arguments_without_traceback(caplog):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse('<|tool_call>call:web_search{query:"unterminated}<tool_call|>')

    assert calls == []
    assert "Failed to parse Gemma4 tool-call arguments" in caplog.text
    assert "Traceback" not in caplog.text


def test_gemma4_parser_error_span_finds_unclosed_native_tool_call():
    response = '<|tool_call>call:web_search{query:<|"|>bad<|"|>'

    assert fused_generate._parser_error_action_span(response) == (0, len(response))


def test_gemma4_parser_error_span_skips_leading_thought_channel():
    thought = "<|channel>thought\nNeed evidence.\n<channel|>\n"
    tail = "I forgot to call a tool."

    assert fused_generate._parser_error_action_span(thought + tail) == (len(thought), len(thought + tail))


def test_gemma4_parser_error_credit_assignment_masks_only_bad_native_action():
    bad = '<|tool_call>call:web_search{query:<|"|>bad<|"|>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Use web_search before submit"}),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert sample.metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"
    assert _policy_masked_text(sample) == bad


def test_gemma4_malformed_closed_tool_call_terminates_as_parser_error():
    bad = '<|tool_call>call:web_search{query:"unterminated}<tool_call|>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Use web_search before submit"}),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert sample.metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"
    assert _policy_masked_text(sample) == bad


def test_gemma4_valid_call_then_unclosed_native_marker_terminates_as_parser_error(tmp_path: Path):
    valid = _gemma4_echo_call("before-error")
    malformed = '<|tool_call>call:echo{value:<|"|>unterminated'
    response = valid + malformed

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger mixed Gemma4 parser error"),
        [{"text": response}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert sample.metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"
    assert _policy_masked_text(sample) == malformed
    assert "before-error" not in sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]["observation"]


def test_gemma4_valid_call_then_malformed_closed_native_marker_terminates_as_parser_error(tmp_path: Path):
    valid = _gemma4_echo_call("before-error")
    malformed = '<|tool_call>call:echo{value:"unterminated}<tool_call|>'
    response = valid + malformed

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger mixed Gemma4 parser error"),
        [{"text": response}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert sample.metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"
    assert _policy_masked_text(sample) == malformed
    assert "before-error" not in sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]["observation"]


def test_tito_model_type_detects_gemma4_without_changing_qwen_detection():
    assert fused_generate._tito_model_type("/share/nlp/share/plm/gemma-4-E2B-it") == "gemma4"
    assert fused_generate._tito_model_type("/models/Qwen3.5-4B") == "qwen3_5"
    assert fused_generate._tito_model_type("/models/Qwen3-4B") == "qwen3"


def test_build_system_prompt_switches_tool_format_by_model():
    tools = [web_search_schema(), finish_schema()]

    coder = build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, "/share/nlp/share/plm/Qwen3.5-4B")
    assert "<function=FUNCTION_NAME>" in coder
    assert "<parameter=PARAMETER_NAME>" in coder

    legacy = build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, "/share/nlp/share/plm/Qwen3-4B")
    assert "<function=FUNCTION_NAME>" not in legacy
    assert '{"name": <function-name>, "arguments": <args-json-object>}' in legacy

    gemma4 = build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, "/share/nlp/share/plm/gemma-4-E2B-it")
    assert gemma4 == FUSED_SEARCH_SYSTEM_PROMPT.strip()
    assert "<|tool>declaration:web_search{" not in gemma4


def test_explicit_model_series_overrides_noncanonical_checkpoint_path(monkeypatch):
    checkpoint = "checkpoints/FusedRL/webqa-dapo-q3.5-4b/iter_0000019_hf"
    tools = [web_search_schema(), finish_schema()]

    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen3.5")
    assert isinstance(make_tool_parser(checkpoint), Qwen3CoderToolParser)
    assert "<function=FUNCTION_NAME>" in build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, checkpoint)

    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen3")
    assert type(make_tool_parser(checkpoint)) is QwenToolParser
    assert '{"name": <function-name>, "arguments": <args-json-object>}' in build_system_prompt(
        FUSED_SEARCH_SYSTEM_PROMPT, tools, checkpoint
    )


def test_invalid_explicit_model_series_fails_closed(monkeypatch):
    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen-next")
    with pytest.raises(ValueError, match="Unsupported FUSED_MODEL_SERIES"):
        make_tool_parser("/models/Qwen3-4B")


def test_resolve_cli_and_et_modes_without_docker_reset():
    et_task = {
        "data_source": "endless_terminals",
        "task_id": "t1",
        "docker_image": "example/image",
        "instruction": "Create a file",
        "test_script": "echo 1 >/logs/verifier/reward.txt",
        "final_state_test": "def test_ok(): pass",
    }
    cli_task = {"docker_image": "example/image", "problem_statement": "Fix bug", "eval_script": "exit 0"}

    assert resolve_task_mode(et_task) == "et"
    assert resolve_task_mode(cli_task) == "cli"

    et_env = FusedEnvironment(et_task)
    cli_env = FusedEnvironment(cli_task)
    assert "execute_bash" in _valid_tool_names(et_env.tools())
    assert "file_editor" in _valid_tool_names(et_env.tools())
    assert "search" not in _valid_tool_names(et_env.tools())
    assert "search" in _valid_tool_names(cli_env.tools())

    et_messages = _initial_messages("gem", "et", "Create a file", et_env.tools())
    cli_messages = _initial_messages("gem", "cli", "Fix bug", cli_env.tools())
    assert "Docker container" in et_messages[0]["content"]
    assert "repository issue" in cli_messages[0]["content"]


def test_local_mcp_toolset_and_verifier(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Return value")
def get_value() -> dict:
    return {"answer": 3}
""",
        encoding="utf-8",
    )
    task = {
        "question": "Return answer",
        "data_root": str(asset),
        "tools_py": str(asset / "tools.py"),
        "verifier": {
            "verification_code": """
def verify(tools, answer):
    return {"passed": isinstance(answer, dict) and answer.get("answer") == 3}
"""
        },
    }
    env = FusedEnvironment(task)

    observation, info = env.reset()
    assert observation == "Return answer"
    assert info["task_type"] == "mcp"
    assert "get_value" in _valid_tool_names(env.tools())
    assert env.mcp_tools.call("get_value", {}) == json.dumps({"answer": 3})

    env.answer = json.dumps({"answer": 3})
    env.tool_calls = 1
    assert env.compute_final_reward() == 1.0


def test_local_mcp_tools_keep_definition_time_asset_directory(tmp_path: Path):
    asset = tmp_path / "asset"
    data = asset / "data"
    data.mkdir(parents=True)
    (data / "first.json").write_text('{"value": "first"}', encoding="utf-8")
    (asset / "tools.py").write_text(
        """
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Tools")
BASE_DIR = Path(__file__).parent

@mcp.tool(description="Read the first asset")
def read_first() -> dict:
    with open(BASE_DIR / "data" / "first.json", encoding="utf-8") as file:
        return json.load(file)

# Concatenated generated tool blocks can rebind this global after read_first
# is defined. The first tool must retain its own block's asset directory.
BASE_DIR = Path(__file__).parent.parent
""",
        encoding="utf-8",
    )

    env = FusedEnvironment({"question": "Read data", "tools_py": str(asset / "tools.py")})

    assert env.mcp_tools.call("read_first", {}) == json.dumps({"value": "first"})


def test_mcp_verifier_reward_requires_a_tool_call(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Submit without retrieving")
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps({"done": True})

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_skipped"] == "no_tool_calls"
    assert env.reward_debug["tool_calls"] == 0

    env.tool_calls = 1
    assert env.compute_final_reward() == 1.0


def test_mcp_tool_load_error_is_nonfatal(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text("def broken(:\n", encoding="utf-8")
    env = FusedEnvironment({"question": "Return answer", "tools_py": str(asset / "tools.py")})

    _observation, info = env.reset()

    assert info["task_type"] == "mcp"
    assert "env_error" in info
    assert env.compute_final_reward() == 0.0


def test_mcp_tool_elapsed_time_is_reported(tmp_path: Path, monkeypatch):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Return value")
def get_value() -> dict:
    return {"answer": 3}
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Return answer", "tools_py": str(asset / "tools.py")})
    monotonic_values = iter([50.0, 50.15])
    monkeypatch.setattr("slime.rollout.fused_agent.env._now_monotonic", lambda: next(monotonic_values))

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("get_value", {})))

    assert json.loads(observation) == {"answer": 3}
    assert reward == 0.0
    assert done is False
    assert info["tools/calls"] == 1
    assert info["tools/mcp_tool_elapsed_s"] == pytest.approx(0.15)


def test_render_prompt_ids_accepts_batch_encoding_like_object():
    class FakeBatch:
        data = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return FakeBatch()

    expected_prefix = [ord(ch) for ch in "<think>\n\n</think>\n\n"]
    assert _render_prompt_ids(FakeTokenizer(), [{"role": "user", "content": "x"}]) == [1, 2, 3, *expected_prefix]


def test_render_prompt_ids_controls_thinking_by_default():
    class FakeTokenizer:
        def __init__(self):
            self.enable_thinking_values = []

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True):
            self.enable_thinking_values.append(enable_thinking)
            return [1, 2, 3]

    tokenizer = FakeTokenizer()

    assert _render_prompt_ids(tokenizer, [{"role": "user", "content": "x"}]) == [1, 2, 3]
    assert tokenizer.enable_thinking_values == [False]

    _render_prompt_ids(tokenizer, [{"role": "user", "content": "x"}], disable_thinking=False)
    assert tokenizer.enable_thinking_values == [False, True]


def test_context_span_rendering_uses_same_thinking_control():
    class FakeTokenizer:
        def __init__(self):
            self.enable_thinking_values = []

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True):
            self.enable_thinking_values.append(enable_thinking)
            return [1] * len(messages)

    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "y"},
    ]

    assert _last_assistant_context_start_idx(tokenizer, messages) == 2
    assert tokenizer.enable_thinking_values == [False]


def test_render_prompt_ids_prepares_historical_assistant_empty_thinking_shell():
    class FakeTokenizer:
        def __init__(self):
            self.messages = []

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True):
            self.messages.append(messages)
            return [1, 2, 3]

    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'},
    ]

    _render_prompt_ids(tokenizer, messages)

    rendered_messages = tokenizer.messages[0]
    assert rendered_messages[1]["content"] == messages[1]["content"]
    assert rendered_messages[1]["reasoning_content"] == "\n"


def test_render_prompt_ids_fallback_appends_empty_thinking_prefix():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            assert tokenize is True
            assert add_generation_prompt is True
            return [1, 2, 3]

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [ord(ch) for ch in text]}

    prompt_ids = _render_prompt_ids(FakeTokenizer(), [{"role": "user", "content": "x"}])

    assert prompt_ids[:3] == [1, 2, 3]
    assert prompt_ids[3:] == [ord(ch) for ch in "<think>\n\n</think>\n\n"]


def test_gemma4_render_prompt_ids_falls_back_to_inline_native_tools_when_tools_kwarg_is_unsupported():
    class NoToolsGemma4Tokenizer:
        name_or_path = "/models/gemma-4-E2B-it"

        def __init__(self):
            self.rendered_messages = None

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, **kwargs):
            if "tools" in kwargs:
                raise TypeError("apply_chat_template() got an unexpected keyword argument 'tools'")
            self.rendered_messages = messages
            text = "\n".join(str(message.get("content") or "") for message in messages)
            return [ord(ch) for ch in text]

    tokenizer = NoToolsGemma4Tokenizer()
    tools = [web_search_schema(), finish_schema()]

    prompt_ids = _render_prompt_ids(
        tokenizer,
        [{"role": "system", "content": FUSED_SEARCH_SYSTEM_PROMPT.strip()}, {"role": "user", "content": "Who?"}],
        tools=tools,
    )
    rendered = "".join(chr(ch) for ch in prompt_ids)

    assert tokenizer.rendered_messages is not None
    assert "<|tool>declaration:web_search{" in rendered
    assert "<|tool>declaration:finish{" in rendered
    assert "When you need to call a tool" not in rendered
    assert "Do not wrap Gemma4 tool calls" not in rendered
    assert "<tools>" not in rendered
    assert "<function=FUNCTION_NAME>" not in rendered


def test_gemma4_render_prompt_ids_does_not_add_qwen_think_shell_when_enable_thinking_is_unsupported():
    class NoThinkingGemma4Tokenizer:
        name_or_path = "/models/gemma-4-E2B-it"

        def __init__(self):
            self.rendered_messages = None

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
            self.rendered_messages = messages
            text = "\n".join(f"{message['role']}:{message.get('content') or ''}" for message in messages)
            return [ord(ch) for ch in text]

    tokenizer = NoThinkingGemma4Tokenizer()

    prompt_ids = _render_prompt_ids(
        tokenizer,
        [
            {"role": "user", "content": "Who?"},
            {"role": "assistant", "content": '<|tool_call>call:echo{value:<|"|>x<|"|>}<tool_call|>'},
        ],
        tools=[web_search_schema()],
    )
    rendered = "".join(chr(ch) for ch in prompt_ids)

    assert tokenizer.rendered_messages is not None
    assert "<think>" not in rendered
    assert "</think>" not in rendered
    assert '<|tool_call>call:echo{value:<|"|>x<|"|>}<tool_call|>' in rendered


def test_disable_thinking_prompt_prefix_stays_masked_as_prompt_context():
    class PromptPrefixTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True):
            text = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
            if add_generation_prompt:
                text += "assistant:\n"
            if add_generation_prompt and not enable_thinking:
                text += "<think>\n\n</think>\n\n"
            return [ord(c) for c in text] if tokenize else text

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    action = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Use tools"}),
        [{"text": action}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=PromptPrefixTokenizer(),
    )

    sample = result[0]
    response_text = _response_text(sample)
    assert response_text == action
    assert "<think>\n\n</think>\n\n" not in response_text
    assert _masked_text(sample) == action


def test_disable_thinking_token_mask_view_includes_empty_thinking_prefix_as_masked_prompt():
    tokenizer = FakeTokenizer()
    prefix = "<think>\n\n</think>\n\n"
    action = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'
    prompt = "prompt:"
    sample = Sample(
        prompt="placeholder",
        tokens=[ord(ch) for ch in (prompt + prefix + action)],
        response_length=len(action),
        loss_mask=[1] * len(action),
        reward=0.0,
        status=Sample.Status.COMPLETED,
    )

    visual_masked = _visualized_text_by_style(sample, tokenizer, rollout_visualization._MASKED_TOKEN_STYLE)
    visual_unmasked = _visualized_text_by_styles(
        sample,
        tokenizer,
        {
            rollout_visualization._UNMASKED_TOKEN_STYLE,
            rollout_visualization._REWARD_NEG_STYLE,
            rollout_visualization._REWARD_POS_STYLE,
        },
    )
    visual_shell = _visualized_text_by_style(sample, tokenizer, rollout_visualization._EMPTY_THINK_SHELL_STYLE)

    assert prefix.replace("\n", "\\n") in visual_shell
    assert prefix.replace("\n", "\\n") not in visual_masked
    assert action in visual_unmasked


def test_disable_thinking_default_loss_mask_masks_leading_think_block():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(ch) for ch in text]

    response = '<think>\nplan\n</think>\n<tool_call>{"name":"finish","arguments":{"command":"submit","result":"x"}}</tool_call>'
    mask = _default_response_loss_mask(FakeTokenizer(), response, output_len=len(response), disable_thinking=True)

    split = response.index("<tool_call>")
    assert mask == [0] * split + [1] * (len(response) - split)


class _ThinkTokenizer:
    """Minimal tokenizer exposing ``</think>`` as a single token id, like the
    Qwen3 / Qwen3.5 tokenizers. ``close_id`` differs between the two families
    (Qwen3=151668, Qwen3.5=248069); the exact value is irrelevant to the masking
    logic, only that ``convert_tokens_to_ids('</think>')`` resolves to it."""

    unk_token_id = 0

    def __init__(self, close_id: int) -> None:
        self._close_id = close_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._close_id if token == "</think>" else self.unk_token_id


@pytest.mark.parametrize("close_id", [151668, 248069])
def test_disable_thinking_loss_mask_token_domain_masks_through_close_think(close_id):
    # Qwen3-shaped output_ids: <think>(open) reasoning... </think> \n\n answer...
    # We only need the </think> id present; use distinct filler ids for the rest.
    tokenizer = _ThinkTokenizer(close_id)
    output_ids = [900, 901, 902, close_id, 271, 800, 801]  # </think> at index 3
    mask = _default_response_loss_mask(tokenizer, "irrelevant", output_len=len(output_ids), disable_thinking=True, output_ids=output_ids)
    # masked through </think> (index 3), answer (incl. trailing \n\n) trainable
    assert mask == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("close_id", [151668, 248069])
def test_disable_thinking_loss_mask_token_domain_qwen35_shape(close_id):
    # Qwen3.5-shaped output_ids: reasoning starts immediately (<think> was
    # injected into the prompt, not generated), </think> still inside output.
    tokenizer = _ThinkTokenizer(close_id)
    output_ids = [700, 701, close_id, 271, 800]  # </think> at index 2
    mask = _default_response_loss_mask(tokenizer, "irrelevant", output_len=len(output_ids), disable_thinking=True, output_ids=output_ids)
    assert mask == [0, 0, 0, 1, 1]


@pytest.mark.parametrize("close_id", [151668, 248069])
def test_disable_thinking_loss_mask_no_misfired_think_trains_full(close_id):
    # Well-behaved disable-thinking response: empty shell lives in the prompt, so
    # output_ids carry no </think> -> nothing to mask, full response trains.
    tokenizer = _ThinkTokenizer(close_id)
    output_ids = [800, 801, 802, 803]
    mask = _default_response_loss_mask(tokenizer, "answer only", output_len=len(output_ids), disable_thinking=True, output_ids=output_ids)
    assert mask == [1, 1, 1, 1]


@pytest.mark.parametrize("close_id", [151668, 248069])
def test_enable_thinking_loss_mask_trains_reasoning_and_answer(close_id):
    # enable-thinking: reasoning IS the learning signal, so the whole response is
    # trained regardless of model family or where </think> lands. None => all-ones
    # in the builder.
    tokenizer = _ThinkTokenizer(close_id)
    output_ids = [900, 901, close_id, 271, 800]
    mask = _default_response_loss_mask(tokenizer, "reason </think> answer", output_len=len(output_ids), disable_thinking=False, output_ids=output_ids)
    assert mask is None


def test_disable_thinking_loss_mask_falls_back_to_char_level_without_close_id():
    # A tokenizer that cannot resolve </think> (returns unk) must fall back to the
    # pre-existing character-level path rather than mis-masking.
    class NoThinkTokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return self.unk_token_id

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

    response = "<think>\nplan\n</think>\n<tool_call>x</tool_call>"
    output_ids = [ord(ch) for ch in response]
    mask = _default_response_loss_mask(NoThinkTokenizer(), response, output_len=len(output_ids), disable_thinking=True, output_ids=output_ids)
    split = response.index("<tool_call>")
    assert mask == [0] * split + [1] * (len(response) - split)


def test_gemma4_disable_thinking_loss_mask_masks_leading_thought_channel():
    thought = "<|channel>thought\nNeed evidence.\n<channel|>\n"
    action = '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>answer<|"|>}<tool_call|>'

    mask = _default_response_loss_mask(
        FakeGemma4Tokenizer(),
        thought + action,
        output_len=len(thought + action),
        disable_thinking=True,
        output_ids=[ord(ch) for ch in thought + action],
    )

    assert mask == [0] * len(thought) + [1] * len(action)


def test_gemma4_extract_thought_channel_excludes_native_action():
    thought = "<|channel>thought\nNeed evidence.\n<channel|>"
    action = '\n<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>answer<|"|>}<tool_call|>'

    assert fused_generate._extract_thought(thought + action) == thought


def test_disable_thinking_generate_masks_model_generated_think_tokens():
    think = "<think>\nNeed evidence.\n</think>\n"
    action = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Use tools"}),
        [{"text": think + action}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
    )

    sample = result[0]
    assert think not in _masked_text(sample)
    assert action in _masked_text(sample)
    assert think in _unmasked_text(sample)


def test_gemma4_non_action_masking_masks_generated_thought_channel():
    thought = "<|channel>thought\nNeed evidence.\n<channel|>\n"
    action = _gemma4_finish_call('{"done":true}')

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Use tools"}),
        [{"text": thought + action}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert thought not in _masked_text(sample)
    assert action in _masked_text(sample)
    assert thought in _unmasked_text(sample)


def test_initial_messages_include_tool_prompt():
    schemas = [
        tool_schema("web_search", "Search", {"query": {"type": "string", "description": ""}}, ["query"]),
        tool_schema("finish", "Finish", {"result": {"type": "string", "description": ""}}, []),
    ]

    messages = _initial_messages("unified_gem", "web search", "Who?", schemas)

    assert messages[0]["role"] == "system"
    assert "<tools>" in messages[0]["content"]
    assert "web_search" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_gemma4_initial_messages_leave_tools_to_chat_template():
    schemas = [web_search_schema(), finish_schema()]

    messages = _initial_messages(
        "gem",
        "web search",
        "Who?",
        schemas,
        model_name="/share/nlp/share/plm/gemma-4-E2B-it",
    )

    assert messages[0]["content"] == FUSED_SEARCH_SYSTEM_PROMPT.strip()
    assert "<|tool>declaration:" not in messages[0]["content"]

    rendered = FakeGemma4Tokenizer().apply_chat_template(messages, tools=schemas, tokenize=False)
    assert "<|tool>declaration:web_search{" in rendered
    assert "<|tool>declaration:finish{" in rendered


def test_cot_initial_messages_do_not_include_fused_tool_prompt():
    schemas = [web_search_schema(), finish_schema()]

    messages = _initial_messages("cot", "web search", "Who?", schemas)

    assert messages[0]["role"] == "system"
    assert "<tools>" not in messages[0]["content"]
    assert "web_search" not in messages[0]["content"]
    assert "general agent" not in messages[0]["content"]
    assert "research assistant" not in messages[0]["content"]


def test_cot_generate_ignores_tool_calls_and_scores_as_reasoning_only(monkeypatch):
    async def fail_step(self, action):
        raise AssertionError(f"cot harness must not execute tools: {action}")

    monkeypatch.setattr(FusedEnvironment, "step", fail_step)
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence"}),
        [{"text": "\\boxed{answer}\n<tool_call>{\"name\":\"web_search\",\"arguments\":{\"query\":\"x\"}}</tool_call>"}],
        {
            "FUSED_HARNESS": "cot",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
    )

    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "reasoning_only"
    assert sample.metadata["fused_tool_call_turns"] == 0
    assert sample.metadata["fused_reward_debug"]["tool_calls"] == 0


def test_tool_observation_has_execution_output_header():
    observation = _format_tool_observation("web_search", "AHPL is a hardware description language.")

    assert observation == ("<tool_response>\n" "Execution output of [web_search]:\n" "AHPL is a hardware description language.\n" "</tool_response>")


def test_web_search_retrieval_formats_plain_text_without_json_or_urls():
    payload = {
        "results": [
            {
                "id": 62835,
                "content": {
                    "url": "https://example.com/nfpa",
                    "title": "National Fluid Power Association",
                    "chunk_text": "The National\nFluid Power Association\tis a trade association.",
                },
                "lexical_rank": 3,
            }
        ]
    }

    text = _format_retrieval(payload)

    assert text == "[1] National Fluid Power Association: The National Fluid Power Association is a trade association."
    assert "https://" not in text
    assert '"content"' not in text
    assert "chunk_text" not in text


def test_web_search_tool_observation_normalizes_structured_payload():
    observation = _format_tool_observation(
        "web_search",
        [
            {
                "content": {
                    "url": "https://example.com/xmlspy",
                    "title": "XMLSpy",
                    "chunk_text": "XMLSpy is an XML editor.\n\nDevelopment started in 1999.",
                }
            }
        ],
    )

    assert observation == ("<tool_response>\n" "Execution output of [web_search]:\n" "[1] XMLSpy: XMLSpy is an XML editor. Development started in 1999.\n" "</tool_response>")


def test_web_search_tool_observation_normalizes_json_string_payload():
    raw_json = json.dumps(
        [
            {
                "content": {
                    "url": "https://example.com/iron-man",
                    "title": "Iron Man",
                    "summary": "Iron Man is a superhero.\nCreated by Marvel.",
                }
            }
        ]
    )

    observation = _format_tool_observation("web_search", raw_json)

    assert "[1] Iron Man: Iron Man is a superhero. Created by Marvel." in observation
    assert "https://" not in observation
    assert '"summary"' not in observation


def test_web_search_retrieval_limits_observation_to_256_words():
    first_doc = " ".join(f"alpha{i}" for i in range(180))
    second_doc = " ".join(f"beta{i}" for i in range(180))
    payload = {
        "results": [
            {"content": {"title": "First", "chunk_text": first_doc}},
            {"content": {"title": "Second", "summary": second_doc}},
        ]
    }

    text = _format_retrieval(payload)

    assert len(text.split()) == 256
    assert "beta179" not in text


def test_web_search_tool_observation_limits_json_string_to_256_words():
    raw_json = json.dumps(
        [
            {
                "content": {
                    "title": "Long",
                    "chunk_text": " ".join(f"token{i}" for i in range(300)),
                }
            }
        ]
    )

    observation = _format_tool_observation("web_search", raw_json)
    body = observation.split("Execution output of [web_search]:\n", 1)[1].rsplit("\n</tool_response>", 1)[0]

    assert len(body.split()) == 256
    assert "token299" not in body


def test_web_search_falls_back_to_one_for_malformed_max_results(monkeypatch):
    class FakeResponse:
        status = 200
        reason = "OK"
        request_info = SimpleNamespace(real_url="http://127.0.0.1:65432/retrieve")
        history = ()
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            text = " ".join(f"fallback{i}" for i in range(30))
            return {"results": [{"content": {"title": "Doc", "chunk_text": text}}]}

        async def text(self):
            return json.dumps(await self.json())

    class FakeSession:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        def post(self, url, json=None):
            FakeSession.calls.append((url, json))
            return FakeResponse()

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    env = FusedEnvironment({"question": "Find evidence"})

    observation, reward, done, info = asyncio.run(
        env.step(fused_generate.ToolCall("web_search", {"query": "q", "max_results": "10\n</<|im_start|>user>"}))
    )

    assert observation.startswith("[Result 1] Title: Doc\nSnippet: fallback0 fallback1")
    assert reward == 0.0
    assert done is False
    assert info["tools/search_calls"] == 1
    assert FakeSession.calls[0][1]["top_k"] == 1
    assert FakeSession.calls[0][1]["topk"] == 1
    assert FakeSession.calls[0][1]["max_results"] == 1


def test_no_ground_truth_reward_is_zero_even_after_tool_call():
    env = FusedEnvironment({"question": "Find evidence"})
    env.answer = "confident answer"

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["no_ground_truth"] is True
    assert env.reward_debug["tool_calls"] == 0

    env.tool_calls = 1
    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["tool_calls"] == 1


def test_webqa_reward_requires_two_unique_searches(monkeypatch):
    env = FusedEnvironment({"question": "Find evidence", "reward_model": {"answer": "correct answer"}})
    env.answer = "\\boxed{correct answer}"
    env.tool_calls = 1
    env.web_search_queries = {"same query"}

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["insufficient_searches"] is True
    assert env.reward_debug["unique_search_calls"] == 1

    env.tool_calls = 2
    env.web_search_queries = {"first query", "second query"}
    assert env.compute_final_reward() == 1.0


def test_webqa_min_unique_searches_can_be_overridden(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    env = FusedEnvironment({"question": "Find evidence", "reward_model": {"answer": "correct answer"}})
    env.answer = "\\boxed{correct answer}"
    env.tool_calls = 1
    env.web_search_queries = {"single query"}

    assert env.compute_final_reward() == 1.0


def test_webqa_reward_is_binary_exact_match_after_normalization(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "4500 South and 4700 South"})
    env.tool_calls = 1
    env.web_search_queries = {"single query"}

    env.answer = "\\boxed{wrong south wrong wrong wrong}"
    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["exact_match"] is False

    env.answer = "\\boxed{4500 South and 4700 South}"
    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["exact_match"] is True
    assert env.reward_debug["min_unique_search_calls"] == 1


def test_webqa_reward_extracts_nested_boxed_answer(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "answer {with braces}"})
    env.tool_calls = 1
    env.web_search_queries = {"single query"}
    env.answer = "final: \\boxed{answer {with braces}}"

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["prediction"] == "answer {with braces}"


def test_webqa_reward_prefers_nested_ground_truth_over_empty_reward_model(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    env = FusedEnvironment(
        {
            "question": "Find evidence",
            "ground_truth": "correct answer",
            "reward_model": {"ground_truth": None, "style": "rule"},
        }
    )
    env.tool_calls = 1
    env.web_search_queries = {"single query"}
    env.answer = "\\boxed{correct answer}"

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["ground_truth"] == "correct answer"


def test_web_search_uses_summary_when_enabled(monkeypatch):
    class FakeResponse:
        def __init__(self, status: int, payload):
            self.status = status
            self._payload = payload
            self.reason = "OK"
            self.request_info = SimpleNamespace(real_url="http://127.0.0.1:65432/summarize")
            self.history = ()
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            FakeSession.calls.append((url, json))
            if url.endswith("/retrieve"):
                payload = {
                    "results": [
                        {
                            "title": "Doc",
                            "content": {
                                "original_text": " ".join(f"word{i}" for i in range(80)),
                            },
                        }
                    ]
                }
                return FakeResponse(200, payload)
            if url.endswith("/summarize"):
                return FakeResponse(200, {"summary": "# Summary: concise answer"})
            raise AssertionError(url)

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    monotonic_values = iter([100.0, 100.01, 100.25, 100.26])
    monkeypatch.setattr("slime.rollout.fused_agent.env._now_monotonic", lambda: next(monotonic_values))
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARIZE", "1")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "2")
    env = FusedEnvironment({"question": "Find evidence"})

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q"})))

    assert observation == "concise answer"
    assert reward == 0.0
    assert done is False
    assert info["search_summary_used"] is True
    assert info["tools/search_summary_retries"] == 0
    assert info["tools/search_summary_elapsed_s"] == pytest.approx(0.01)
    assert info["tools/search_retrieve_elapsed_s"] == pytest.approx(0.01)
    assert [url for url, _ in FakeSession.calls] == [
        "http://127.0.0.1:65432/retrieve",
        "http://127.0.0.1:65432/summarize",
    ]


def test_web_search_summary_failure_falls_back_to_documents(monkeypatch):
    class FakeResponse:
        def __init__(self, status: int, payload):
            self.status = status
            self._payload = payload
            self.reason = "OK"
            self.request_info = SimpleNamespace(real_url="http://127.0.0.1:65432/summarize")
            self.history = ()
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            FakeSession.calls.append((url, json))
            if url.endswith("/retrieve"):
                payload = {
                    "results": [
                        {
                            "title": "Doc",
                            "content": {
                                "original_text": " ".join(f"word{i}" for i in range(80)),
                            },
                        }
                    ]
                }
                return FakeResponse(200, payload)
            if url.endswith("/summarize"):
                raise asyncio.TimeoutError()
            raise AssertionError(url)

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    monotonic_values = iter([10.0, 10.02, 10.5, 10.51])
    monkeypatch.setattr("slime.rollout.fused_agent.env._now_monotonic", lambda: next(monotonic_values))
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARIZE", "1")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "2")
    env = FusedEnvironment({"question": "Find evidence"})

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q"})))

    assert "[Result 1] Title: Doc" in observation
    assert info["search_summary_used"] is False
    assert info["tools/search_summary_failures"] == 1
    assert info["tools/search_summary_fallbacks"] == 1
    assert info["tools/search_summary_retries"] == 2
    assert info["tools/search_summary_elapsed_s"] == pytest.approx(0.01)
    assert info["tools/search_retrieve_elapsed_s"] == pytest.approx(0.02)
    assert reward == 0.0
    assert done is False
    assert len([url for url, _ in FakeSession.calls if url.endswith("/summarize")]) == 3


def test_web_search_summary_failure_does_not_disable_future_summary_attempts(monkeypatch):
    class FakeResponse:
        def __init__(self, status: int, payload):
            self.status = status
            self._payload = payload
            self.reason = "OK"
            self.request_info = SimpleNamespace(real_url="http://127.0.0.1:65432/summarize")
            self.history = ()
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            FakeSession.calls.append((url, json))
            if url.endswith("/retrieve"):
                payload = {
                    "results": [
                        {
                            "title": "Doc",
                            "content": {
                                "original_text": " ".join(f"word{i}" for i in range(80)),
                            },
                        }
                    ]
                }
                return FakeResponse(200, payload)
            if url.endswith("/summarize"):
                raise asyncio.TimeoutError()
            raise AssertionError(url)

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    monotonic_values = iter([20.0, 20.02, 20.4, 20.41, 21.0, 21.02, 21.9, 21.91])
    monkeypatch.setattr("slime.rollout.fused_agent.env._now_monotonic", lambda: next(monotonic_values))
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARIZE", "1")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "2")
    env = FusedEnvironment({"question": "Find evidence"})

    first = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q1"})))
    second = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q2"})))

    first_info = first[3]
    second_info = second[3]
    assert first_info["search_summary_requested"] is True
    assert second_info["search_summary_requested"] is True
    assert first_info["tools/search_summary_retries"] == 2
    assert second_info["tools/search_summary_retries"] == 2
    assert first_info["tools/search_summary_elapsed_s"] == pytest.approx(0.01)
    assert second_info["tools/search_summary_elapsed_s"] == pytest.approx(0.01)
    assert first_info["tools/search_retrieve_elapsed_s"] == pytest.approx(0.02)
    assert second_info["tools/search_retrieve_elapsed_s"] == pytest.approx(0.02)
    assert len([url for url, _ in FakeSession.calls if url.endswith("/summarize")]) == 6


def test_web_search_summary_splits_oversized_requests(monkeypatch):
    class FakeResponse:
        def __init__(self, status: int, payload):
            self.status = status
            self._payload = payload
            self.reason = "OK"
            self.request_info = SimpleNamespace(real_url="http://127.0.0.1:65432/summarize")
            self.history = ()
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            FakeSession.calls.append((url, json))
            if url.endswith("/retrieve"):
                long_doc = " ".join(f"word{i}" for i in range(1500))
                payload = {
                    "results": [
                        {
                            "title": "Doc",
                            "content": {
                                "original_text": long_doc,
                            },
                        }
                    ]
                }
                return FakeResponse(200, payload)
            if url.endswith("/summarize"):
                documents = json["documents"]
                total_words = sum(len(str(item.get("content", "")).split()) for item in documents)
                if total_words > 700:
                    return FakeResponse(
                        500,
                        {"detail": "Summarization Error: ValueError('The decoder prompt (length 9000) is longer than the maximum model length of 8192. Make sure that `max_model_len` is no smaller than the number of text tokens.')"},
                    )
                return FakeResponse(200, {"summary": f"# Summary: summarized {total_words}"})
            raise AssertionError(url)

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    monotonic_values = iter([30.0, 30.01, 30.5, 30.6])
    monkeypatch.setattr("slime.rollout.fused_agent.env._now_monotonic", lambda: next(monotonic_values))
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARIZE", "1")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "0")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_MAX_WORDS_PER_REQUEST", "1200")
    env = FusedEnvironment({"question": "Find evidence"})

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q"})))

    assert observation.startswith("summarized ")
    assert reward == 0.0
    assert done is False
    assert info["search_summary_used"] is True
    summarize_calls = [payload for url, payload in FakeSession.calls if url.endswith("/summarize")]
    assert len(summarize_calls) == 5
    assert sum(len(str(item.get("content", "")).split()) for item in summarize_calls[0]["documents"]) > 700
    assert max(sum(len(str(item.get("content", "")).split()) for item in payload["documents"]) for payload in summarize_calls[1:]) <= 700


def test_web_search_skips_summary_when_disabled(monkeypatch):
    class FakeResponse:
        def __init__(self, status: int, payload):
            self.status = status
            self._payload = payload
            self.reason = "OK"
            self.request_info = None
            self.history = ()
            self.headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        calls = []

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None):
            FakeSession.calls.append((url, json))
            payload = {
                "results": [
                    {
                        "title": "Doc",
                        "content": {
                            "original_text": " ".join(f"word{i}" for i in range(80)),
                        },
                    }
                ]
            }
            return FakeResponse(200, payload)

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARIZE", "0")
    env = FusedEnvironment({"question": "Find evidence"})

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q"})))

    assert "[Result 1] Title: Doc" in observation
    assert info["search_summary_requested"] is False
    assert [url for url, _ in FakeSession.calls] == ["http://127.0.0.1:65432/retrieve"]
    assert reward == 0.0
    assert done is False


def test_web_search_cache_and_singleflight_share_retrieval_work(monkeypatch):
    calls = 0
    active = 0
    max_active = 0

    async def fake_post(_session, _url, payload, *, retry_budget):
        nonlocal calls, active, max_active
        calls += 1
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"results": [{"title": payload["query"], "content": "answer"}]}, 0

    monkeypatch.setattr("slime.rollout.fused_agent.env._post_json_with_retries", fake_post)
    monkeypatch.setattr("slime.rollout.fused_agent.env._get_shared_http_session", lambda: object())
    monkeypatch.setenv("RLLM_RETRIEVAL_CONCURRENCY", "1")
    monkeypatch.setenv("RLLM_RETRIEVAL_CACHE_SIZE", "8")

    async def scenario():
        first = FusedEnvironment({"question": "q"})
        first_result = await first.step(fused_generate.ToolCall("web_search", {"query": "same query"}))
        repeated_result = await first.step(fused_generate.ToolCall("web_search", {"query": "  SAME   query "}))

        second = FusedEnvironment({"question": "q"})
        global_result = await second.step(fused_generate.ToolCall("web_search", {"query": "same query"}))

        third = FusedEnvironment({"question": "q"})
        fourth = FusedEnvironment({"question": "q"})
        singleflight_results = await asyncio.gather(
            third.step(fused_generate.ToolCall("web_search", {"query": "new query"})),
            fourth.step(fused_generate.ToolCall("web_search", {"query": "new query"})),
        )
        await asyncio.gather(
            FusedEnvironment({"question": "q"}).step(fused_generate.ToolCall("web_search", {"query": "unique one"})),
            FusedEnvironment({"question": "q"}).step(fused_generate.ToolCall("web_search", {"query": "unique two"})),
        )
        return first_result, repeated_result, global_result, singleflight_results

    first_result, repeated_result, global_result, singleflight_results = asyncio.run(scenario())

    assert first_result[3]["tools/search_episode_cache_hits"] == 0
    assert repeated_result[3]["tools/search_episode_cache_hits"] == 1
    assert global_result[3]["tools/search_global_cache_hits"] == 1
    assert sum(result[3]["tools/search_singleflight_hits"] for result in singleflight_results) == 1
    assert calls == 4
    assert max_active == 1


def test_task_from_sample_promotes_extra_info_ground_truth():
    sample = Sample(
        prompt="Question",
        label={"ground_truth": None, "style": "rule"},
        metadata={
            "extra_info": {"ground_truth": "nested answer", "question": "Question"},
            "data_source": "web_search",
        },
    )

    task = fused_generate._task_from_sample(sample)

    assert task["ground_truth"] == "nested answer"
    assert task["reward_model"]["ground_truth"] == "nested answer"


def test_exact_match_reward_accepts_any_list_target():
    assert _exact_match_reward("the correct answer", ["wrong answer", "correct answer"]) == 1.0
    assert _exact_match_reward("partial answer", ["partial answer plus extra"]) == 0.0


def test_call_sglang_aborts_request_on_timeout(monkeypatch):
    calls = []

    async def fake_post(url, payload, headers=None):
        calls.append(("generate", url, payload, headers))
        raise httpx.ReadTimeout("timeout")

    class FakeClient:
        async def post(self, url, json=None, timeout=None):
            calls.append(("abort", url, json, timeout))

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    monkeypatch.setattr(fused_generate.http_utils, "_http_client", FakeClient())

    args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000, router_policy="consistent_hashing")

    try:
        asyncio.run(fused_generate._call_sglang(args, [1, 2], {"max_new_tokens": 4}, session_id="sid"))
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError("expected ReadTimeout")

    assert calls[0][0] == "generate"
    assert calls[0][2]["rid"]
    assert calls[0][3] == {"X-SMG-Routing-Key": "sid"}
    assert calls[1] == (
        "abort",
        "http://127.0.0.1:30000/abort_request",
        {"rid": calls[0][2]["rid"]},
        5.0,
    )


def test_call_sglang_aborts_native_session_request_on_direct_engine(monkeypatch):
    calls = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        calls.append(("generate", url, payload, headers))
        raise httpx.ReadTimeout("timeout")

    class FakeClient:
        async def post(self, url, json=None, timeout=None):
            calls.append(("abort", url, json, timeout))

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    monkeypatch.setattr(fused_generate.http_utils, "_http_client", FakeClient())
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="manual",
    )

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(
            fused_generate._call_sglang(
                args,
                [1, 2],
                {"max_new_tokens": 4},
                session_id="sid",
                session_params={"id": "sid", "rid": "rid-0"},
                server_url="http://engine-0",
            )
        )

    assert calls[1] == (
        "abort",
        "http://engine-0/abort_request",
        {"rid": calls[0][2]["rid"]},
        5.0,
    )


def test_call_sglang_eval_uses_text_without_logprobs(monkeypatch):
    requests = []

    async def fake_post(url, payload, headers=None):
        requests.append((url, payload, headers))
        return {
            "text": "final answer",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "completion_tokens": 2,
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
        sglang_context_length=100,
        rollout_max_context_len=100,
    )

    result = asyncio.run(
        fused_generate._call_sglang(
            args,
            [1, 2],
            {"max_new_tokens": 4},
            session_id="sid",
            evaluation=True,
        )
    )

    assert requests[0][1]["return_logprob"] is False
    assert requests[0][2] == {"X-SMG-Routing-Key": "sid"}
    assert result == {
        "text": "final answer",
        "finish_reason": "stop",
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "output_ids": [],
        "rid": requests[0][1]["rid"],
    }


def test_manual_router_policy_sends_sticky_routing_header():
    args = SimpleNamespace(router_policy="manual")

    assert fused_generate._routing_headers(args, "session-1") == {"X-SMG-Routing-Key": "session-1"}


def test_eval_session_engine_pool_assigns_new_sessions_to_min_load():
    pool = fused_generate.EvalSessionEnginePool(["http://engine-0/", "http://engine-1"])

    first = pool.acquire()
    second = pool.acquire()
    third = pool.acquire()
    pool.release(first)
    fourth = pool.acquire()

    assert (first, second, third, fourth) == (
        "http://engine-0",
        "http://engine-1",
        "http://engine-0",
        "http://engine-1",
    )


def test_eval_session_engine_pool_caps_sessions_and_cools_down_failed_engine():
    pool = fused_generate.EvalSessionEnginePool(
        ["http://engine-0"],
        max_sessions_per_engine=1,
    )

    engine = pool.acquire()
    assert engine == "http://engine-0"
    assert pool.acquire() is None

    assert pool.record_transport_failure(engine) is True
    assert pool.record_transport_failure(engine) is False
    pool.release(engine)
    assert pool.acquire() is None


def test_eval_sglang_session_releases_engine_slot_when_open_is_cancelled(monkeypatch):
    async def cancelled_post(_self, url, payload, max_retries=60):
        raise asyncio.CancelledError

    monkeypatch.setattr(fused_generate.EvalSessionEnginePool, "post_control", cancelled_post)
    pool = fused_generate.EvalSessionEnginePool(
        ["http://engine-0"],
        max_sessions_per_engine=1,
    )
    session = fused_generate.EvalSGLangSession(
        args=SimpleNamespace(),
        session_id="sid",
        capacity=100,
        enabled=True,
        engine_pool=pool,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(session.prepare_request([1, 2]))

    assert session.server_url is None
    assert pool.acquire() == "http://engine-0"


def test_eval_sglang_session_sends_verified_prompt_deltas_and_closes(monkeypatch):
    requests = []

    async def fake_post(_self, url, payload, max_retries=60):
        requests.append((url, payload, max_retries))
        if url.endswith("/open_session"):
            return payload["session_id"]
        return ""

    monkeypatch.setenv("SLIME_FUSED_EVAL_USE_SGLANG_SESSION", "true")
    monkeypatch.setattr(fused_generate.EvalSessionEnginePool, "post_control", fake_post)
    monkeypatch.setattr(fused_generate, "_EVAL_ENGINE_POOL_LOOP", None)
    monkeypatch.setattr(fused_generate, "_EVAL_ENGINE_POOL", None)
    args = SimpleNamespace(
        sglang_engine_urls=["http://engine-0", "http://engine-1"],
        router_policy="manual",
    )

    async def scenario():
        session = fused_generate.EvalSGLangSession.for_eval(args, "sid", 100)
        first_ids, first_params = await session.prepare_request([1, 2])
        await session.record_response(full_prompt_ids=[1, 2], output_ids=[3], output_text="x", rid="rid-1")
        delta_ids, delta_params = await session.prepare_request([1, 2, 3, 4, 5])
        await session.close()
        return session, first_ids, first_params, delta_ids, delta_params

    session, first_ids, first_params, delta_ids, delta_params = asyncio.run(scenario())

    assert first_ids == [1, 2]
    assert first_params == {"id": "sid", "rid": None}
    assert delta_ids == [4, 5]
    assert delta_params == {"id": "sid", "rid": "rid-1"}
    assert session.session_turns == 1
    assert session.delta_tokens == 2
    assert session.full_tokens_avoided == 3
    assert [request[0] for request in requests] == [
        "http://engine-0/open_session",
        "http://engine-0/close_session",
    ]


def test_eval_sglang_session_replaces_only_normalized_assistant_suffix(monkeypatch):
    async def fake_post(_self, url, payload, max_retries=60):
        return payload["session_id"] if url.endswith("/open_session") else ""

    monkeypatch.setenv("SLIME_FUSED_EVAL_USE_SGLANG_SESSION", "true")
    monkeypatch.setattr(fused_generate.EvalSessionEnginePool, "post_control", fake_post)
    monkeypatch.setattr(fused_generate, "_EVAL_ENGINE_POOL_LOOP", None)
    monkeypatch.setattr(fused_generate, "_EVAL_ENGINE_POOL", None)
    args = SimpleNamespace(
        sglang_engine_urls=["http://engine-0"],
        router_policy="manual",
        sglang_server_concurrency=8,
        sglang_max_running_requests=8,
    )

    async def scenario():
        session = fused_generate.EvalSGLangSession.for_eval(args, "sid", 100)
        await session.prepare_request([1, 2])
        await session.record_response(
            full_prompt_ids=[1, 2],
            output_ids=[3, 99],
            output_text="assistant",
            rid="rid-1",
        )
        request_ids, params = await session.prepare_request([1, 2, 3, 4, 5])
        await session.close()
        return session, request_ids, params

    session, request_ids, params = asyncio.run(scenario())

    assert request_ids == [4, 5]
    assert params == {"id": "sid", "rid": "rid-1", "offset": 3}
    assert session.enabled is True
    assert session.fallback_count == 0
    assert session.delta_tokens == 2
    assert session.full_tokens_avoided == 3


def test_eval_sglang_session_falls_back_when_prompt_is_not_an_extension(monkeypatch):
    requests = []

    async def fake_post(_self, url, payload, max_retries=60):
        requests.append(url)
        return payload["session_id"] if url.endswith("/open_session") else ""

    monkeypatch.setenv("SLIME_FUSED_EVAL_USE_SGLANG_SESSION", "true")
    monkeypatch.setattr(fused_generate.EvalSessionEnginePool, "post_control", fake_post)
    monkeypatch.setattr(fused_generate, "_EVAL_ENGINE_POOL_LOOP", None)
    monkeypatch.setattr(fused_generate, "_EVAL_ENGINE_POOL", None)
    args = SimpleNamespace(sglang_engine_urls=["http://engine-0"], router_policy="manual")

    async def scenario():
        session = fused_generate.EvalSGLangSession.for_eval(args, "sid", 100)
        await session.prepare_request([1, 2])
        await session.record_response(full_prompt_ids=[1, 2], output_ids=[3], output_text="x", rid="rid-1")
        request_ids, params = await session.prepare_request([9, 10])
        return session, request_ids, params

    session, request_ids, params = asyncio.run(scenario())

    assert request_ids == [9, 10]
    assert params is None
    assert session.enabled is False
    assert session.fallback_count == 1
    assert requests == ["http://engine-0/open_session", "http://engine-0/close_session"]


def test_call_sglang_session_uses_direct_engine_and_delta_context_limit(monkeypatch):
    requests = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        requests.append((url, payload, max_retries, headers))
        return {
            "text": "x",
            "output_ids": [7],
            "meta_info": {
                "id": "rid-1",
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 5,
                "completion_tokens": 1,
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="manual",
        sglang_context_length=10,
        rollout_max_context_len=10,
    )

    result = asyncio.run(
        fused_generate._call_sglang(
            args,
            [4, 5],
            {"max_new_tokens": 2},
            session_id="sid",
            evaluation=True,
            session_params={"id": "sid", "rid": "rid-0"},
            context_token_count=5,
            server_url="http://engine-0",
        )
    )

    assert requests[0][0] == "http://engine-0/generate"
    assert requests[0][1]["input_ids"] == [4, 5]
    assert requests[0][1]["session_params"] == {"id": "sid", "rid": "rid-0"}
    assert requests[0][2] == 1
    assert requests[0][3] == {"X-SMG-Routing-Key": "sid", "Connection": "close"}
    assert result["output_ids"] == [7]
    assert result["rid"] == "rid-1"


def test_effective_sglang_context_limit_uses_margin(monkeypatch):
    monkeypatch.setenv("SGLANG_CONTEXT_LENGTH_MARGIN", "8")
    args = SimpleNamespace(sglang_context_length=100, rollout_max_context_len=120)

    assert fused_generate._effective_sglang_context_limit(args) == 92


def test_call_sglang_rejects_over_context_before_http(monkeypatch):
    calls = []

    async def fake_post(url, payload, headers=None):
        calls.append((url, payload, headers))
        raise AssertionError("http post should not be called")

    monkeypatch.setenv("SGLANG_CONTEXT_LENGTH_MARGIN", "2")
    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy=None,
        sglang_context_length=10,
        rollout_max_context_len=10,
    )

    with pytest.raises(fused_generate.SGLangContextLengthExceededError):
        asyncio.run(fused_generate._call_sglang(args, [1, 2, 3, 4, 5, 6, 7], {"max_new_tokens": 2}, session_id="sid"))

    assert calls == []


def test_custom_generate_with_mocked_sglang(tmp_path: Path, monkeypatch=None):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Return value")
def get_value() -> dict:
    return {"answer": 3}
""",
        encoding="utf-8",
    )
    sample = Sample(
        prompt="placeholder",
        label=None,
        metadata={
            "question": "Return answer",
            "data_root": str(asset),
            "tools_py": str(asset / "tools.py"),
            "verifier": {
                "verification_code": """
def verify(tools, answer):
    return {"passed": isinstance(answer, dict) and answer.get("answer") == 3}
"""
            },
        },
    )

    calls = [
        {"text": '<tool_call>{"name":"get_value","arguments":{}}</tool_call>'},
        {"text": '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"{\\"answer\\":3}"}}</tool_call>'},
    ]

    result = _run_generate_with_fake_sglang(sample, calls)

    assert isinstance(result, list)
    assert result
    assert result[0].reward == 1.0
    assert result[0].response_length > 0
    assert any(result[0].loss_mask)


def test_eval_generate_defers_reward_to_benchmark_verifier():
    sample = Sample(
        prompt="placeholder",
        label="B",
        metadata={
            "question": "Which option is correct?",
            "data_source": "gpqa_diamond",
            "rm_type": "benchmark_verifier",
            "choices": ["wrong", "right"],
            "correct_letter": "B",
        },
    )

    response = "After checking the choices, the final answer is B."
    result = _run_generate_with_fake_sglang(
        sample,
        [{"text": response}],
        {"FUSED_HARNESS": "cot"},
        evaluation=True,
    )

    assert isinstance(result, list)
    assert result[0].response == response
    assert result[0].reward == 0.0
    assert result[0].tokens == []
    assert result[0].response_length == len(response)
    assert result[0].metadata["fused_prompt_length_tokens"] > 0
    assert result[0].metadata["fused_completion_length_tokens"] == len(response)
    assert result[0].rollout_log_probs is None
    assert result[0].loss_mask is None
    assert "rllm_episode" not in result[0].metadata

    from slime.rollout import sglang_rollout
    from slime.rollout.rm_hub.benchmark_verifier import reward_func

    assert sglang_rollout._should_rescore_eval_sample(SimpleNamespace(rm_type=""), result[0], evaluation=True)

    assert asyncio.run(reward_func(SimpleNamespace(hf_checkpoint=None), result[0], evaluation=True)) == 1.0


def test_repeated_search_credit_assignment_masks_only_repeated_turn():
    sample = Sample(prompt="placeholder", label={"answer": "x"}, metadata={"question": "Search twice"})
    first = '<tool_call>{"name":"web_search","arguments":{"query":"same"}}</tool_call>'
    second = '<tool_call>{"name":"web_search","arguments":{"query":"same"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        sample,
        [{"text": first}, {"text": second}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "1",
        },
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].reward == 0.0
    assert result[0].metadata["credit_assignment_event"] == "repeated_search_query"
    assert sum(result[0].loss_mask) == len(first) + len(second)
    assert sum(result[0].policy_loss_mask) == len(second)
    decoded_trained = _policy_masked_text(result[0])
    assert decoded_trained == second
    rendered = rollout_visualization._token_mask_text(result[0], FakeTokenizer())
    visual_unmasked = _visualized_text_by_styles_from_rendered(
        rendered,
        {rollout_visualization._UNMASKED_TOKEN_STYLE, rollout_visualization._REWARD_NEG_STYLE},
    )
    assert visual_unmasked.count("<tool_call>") == 2


def test_repeated_search_warns_once_before_credit_assignment():
    sample = Sample(prompt="placeholder", label={"answer": "x"}, metadata={"question": "Search twice"})
    first = '<tool_call>{"name":"web_search","arguments":{"query":"same query"}}</tool_call>'
    repeated_once = '<tool_call>{"name":"web_search","arguments":{"query":"  SAME   query "}}</tool_call>'
    repeated_twice = '<tool_call>{"name":"web_search","arguments":{"query":"same query"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        sample,
        [{"text": first}, {"text": repeated_once}, {"text": repeated_twice}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "2",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["credit_assignment_event"] == "repeated_search_query"
    assert sample.metadata["duplicate_query_count"] == 2
    assert sample.metadata["fused_traj_steps"] == 3
    assert "Repeated search query detected" in _policy_unmasked_text(sample)
    assert _policy_masked_text(sample) == repeated_twice


def test_multiturn_agentic_mask_keeps_prompts_and_observations_masked(tmp_path: Path):
    turns = [_echo_call("alpha"), _echo_call("beta"), _finish_call()]

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use two tools then finish"),
        [{"text": text} for text in turns],
        {"CREDIT_ASSIGNMENT_ENABLE": "True"},
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert _masked_text(sample) == "".join(turns)
    assert "system:" not in _masked_text(sample)
    assert "user:" not in _masked_text(sample)
    assert "<tool_response>" not in _masked_text(sample)
    assert "Execution output of [echo]" not in _masked_text(sample)
    assert "Execution output of [echo]" in _unmasked_text(sample)


def test_high_turn_agentic_mask_stays_strict_for_observations(tmp_path: Path):
    tool_turns = [_echo_call(f"v{i:02d}") for i in range(32)]
    finish = _finish_call()

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Stress many turns"),
        [{"text": text} for text in [*tool_turns, finish]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "FUSED_MAX_STEPS": "40",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert _masked_text(sample) == "".join([*tool_turns, finish])
    assert len(sample.loss_mask) == sample.response_length
    assert sum(sample.loss_mask) == sum(len(text) for text in [*tool_turns, finish])
    unmasked = _unmasked_text(sample)
    assert unmasked.count("<tool_response>") == len(tool_turns)
    assert all(f"v{i:02d}" in unmasked for i in range(32))


def test_long_horizon_web_search_mask_and_visualization_unmask_all_assistant_generations():
    tool_turns = [_search_call(f"evidence query {i:02d}") for i in range(24)]
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'
    tokenizer = FakeChatTemplateTokenizer()

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find long horizon evidence"}),
        [{"text": text} for text in [*tool_turns, finish]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "FUSED_MAX_STEPS": "30",
        },
        tokenizer=tokenizer,
    )

    assert len(result) == 1
    sample = result[0]
    expected_unmasked = "".join([*tool_turns, finish])
    assert _masked_text(sample) == expected_unmasked
    assert sum(sample.loss_mask) == len(expected_unmasked)

    response = _response_text(sample)
    assert response.count("<|im_start|>assistant\n") == len(tool_turns)
    assert response.count("<tool_response>") == 24
    masked_context = _policy_unmasked_text(sample)
    assert "<|im_start|>system\\n" not in _masked_text(sample)
    assert "<|im_start|>user\\n" not in _masked_text(sample)
    assert "<tool_response>" not in _masked_text(sample)
    assert "<tool_response>" in masked_context

    visual_unmasked = _visualized_text_by_styles(
        sample,
        tokenizer,
        {rollout_visualization._UNMASKED_TOKEN_STYLE, rollout_visualization._REWARD_POS_STYLE},
    )
    visual_masked = _visualized_text_by_style(sample, tokenizer, rollout_visualization._MASKED_TOKEN_STYLE)
    assert visual_unmasked == expected_unmasked
    assert "<|im_start|>system\\n" in visual_masked
    assert "<|im_start|>user\\n" in visual_masked
    assert visual_masked.count("<tool_response>") == 24


def test_gemma4_tito_masks_tool_response_context_and_trains_only_actions(tmp_path: Path):
    first = _gemma4_echo_call("before-finish")
    finish = _gemma4_finish_call('{"done":true}')

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use echo before finish"),
        [{"text": first}, {"text": finish}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert _masked_text(sample) == finish
    full_text = "".join(chr(tok) for tok in sample.tokens)
    assert "response:echo{echo:<|\"|>before-finish<|\"|>}" in full_text
    assert "<tool_response>\nExecution output" not in full_text
    assert first not in full_text
    assert "<|turn>model" not in _masked_text(sample)
    assert len(sample.loss_mask) == sample.response_length
    assert sum(sample.loss_mask) == len(finish)


def test_gemma4_batches_multiple_independent_tool_calls_in_one_turn(tmp_path: Path):
    first_turn = _gemma4_echo_call("alpha") + _gemma4_echo_call("beta")
    finish = _gemma4_finish_call('{"done":true}')

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Call echo twice, then finish"),
        [{"text": first_turn}, {"text": finish}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["fused_tool_call_turns"] == 1
    first_step = sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]
    assert first_step["action"] == first_turn
    assert '{"echo": "alpha"}' in first_step["observation"]
    assert '{"echo": "beta"}' in first_step["observation"]
    assert _masked_text(sample) == finish
    full_text = "".join(chr(tok) for tok in sample.tokens)
    assert "response:echo{echo:<|\"|>alpha<|\"|>}" in full_text
    assert "response:echo{echo:<|\"|>beta<|\"|>}" in full_text
    assert "<tool_response>\nExecution output" not in full_text


def test_gemma4_batched_tool_results_record_only_executed_actions(monkeypatch, tmp_path: Path):
    first = _gemma4_echo_call("alpha")
    second = _gemma4_echo_call("beta")
    original_step = FusedEnvironment.step

    async def done_after_first_step(self, action):
        if isinstance(action, ToolCall) and action.name == "echo":
            return {"echo": action.arguments["value"]}, 1.0, True, {"reward_debug": {"forced_done": True}}
        return await original_step(self, action)

    monkeypatch.setattr(FusedEnvironment, "step", done_after_first_step)

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Call echo twice"),
        [{"text": first + second}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    first_step = sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]
    assert first_step["action"] == first
    assert first_step["done"] is True
    assert "alpha" in first_step["observation"]
    assert "beta" not in first_step["observation"]


def test_long_horizon_visualization_keeps_actions_unmasked_under_assistant_replay_drift():
    tool_turns = [_search_call(f"drift evidence query {i:02d}") for i in range(20)]
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'
    tokenizer = FakeChatTemplateTokenizer(drift_assistant_end=True)

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find long horizon evidence"}),
        [{"text": text} for text in [*tool_turns, finish]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "FUSED_MAX_STEPS": "30",
        },
        tokenizer=tokenizer,
    )

    assert len(result) == 1
    sample = result[0]
    unmasked_text = _masked_text(sample)
    assert "<tool_response>" not in _masked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)
    for idx in range(len(tool_turns)):
        assert f"drift evidence query {idx:02d}" in unmasked_text
    assert '"name":"finish"' in unmasked_text
    assert "\\\\boxed{answer}" in unmasked_text

    visual_unmasked = _visualized_text_by_styles(
        sample,
        tokenizer,
        {rollout_visualization._UNMASKED_TOKEN_STYLE, rollout_visualization._REWARD_POS_STYLE},
    )
    visual_masked = _visualized_text_by_style(sample, tokenizer, rollout_visualization._MASKED_TOKEN_STYLE)
    for idx in range(len(tool_turns)):
        query = f"drift evidence query {idx:02d}"
        assert query in visual_unmasked
        assert query not in visual_masked
    assert '"name":"finish"' in visual_unmasked
    assert "\\\\boxed{answer}" in visual_unmasked


def test_group_visualization_combines_sibling_samples_into_full_episode_mask():
    tokenizer = FakeChatTemplateTokenizer()
    first_action = _search_call("first evidence")
    second_action = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Find evidence"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    first_tokens = prompt + [ord(c) for c in first_action]
    first = Sample(
        rollout_id=7,
        tokens=first_tokens,
        response_length=len(first_action),
        loss_mask=[1] * len(first_action),
        reward=1.0,
        metadata={"rllm_episode": {"trajectories": [{"steps": [{}, {}]}]}},
    )

    replay_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Find evidence"},
            {"role": "assistant", "content": first_action},
            {"role": "user", "content": "<tool_response>ok</tool_response>"},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    second_tokens = replay_prompt + [ord(c) for c in second_action]
    second = Sample(
        rollout_id=7,
        tokens=second_tokens,
        response_length=len(second_action),
        loss_mask=[1] * len(second_action),
        reward=1.0,
        metadata=first.metadata,
    )

    single_rendered = rollout_visualization._token_mask_text(second, tokenizer)
    assert single_rendered is not None
    single_unmasked = _visualized_text_by_styles_from_rendered(
        single_rendered,
        {rollout_visualization._UNMASKED_TOKEN_STYLE, rollout_visualization._REWARD_POS_STYLE},
    )
    single_masked = _visualized_text_by_style(second, tokenizer, rollout_visualization._MASKED_TOKEN_STYLE)
    assert "first evidence" in single_unmasked
    assert "first evidence" not in single_masked

    combined_rendered = rollout_visualization._token_mask_text(first, tokenizer, related_samples=[first, second])
    assert combined_rendered is not None
    combined_unmasked = _visualized_text_by_styles_from_rendered(
        combined_rendered,
        {rollout_visualization._UNMASKED_TOKEN_STYLE, rollout_visualization._REWARD_POS_STYLE},
    )
    combined_masked = _visualized_text_by_styles_from_rendered(
        combined_rendered,
        {rollout_visualization._MASKED_TOKEN_STYLE},
    )
    assert "first evidence" in combined_unmasked
    assert '"name":"finish"' in combined_unmasked
    assert "first evidence" not in combined_masked


def test_trailing_im_end_is_not_replayed_twice_in_next_prompt():
    first = _search_call("single stop") + "<|im_end|>"
    second = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call><|im_end|>'
    tokenizer = FakeChatTemplateTokenizer()

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Check stop replay"}),
        [{"text": first}, {"text": second}],
        {"CREDIT_ASSIGNMENT_ENABLE": "True"},
        tokenizer=tokenizer,
    )

    assert len(result) == 1
    response = _response_text(result[0])
    assert "<|im_end|><|im_end|>" not in response
    assert "</tool_call><|im_end|>\n<|im_start|>user" in response
    assert response.count("<|im_end|>") == 3
    assert "single stop" in _masked_text(result[0])
    assert "\\\\boxed{answer}" in _masked_text(result[0])


def test_long_horizon_credit_assignment_masks_prior_assistant_generations_intentionally():
    prior_turns = [_search_call(f"unique query {i:02d}") for i in range(18)]
    repeated = _search_call("unique query 07")
    tokenizer = FakeChatTemplateTokenizer()

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find long horizon evidence"}),
        [{"text": text} for text in [*prior_turns, repeated]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_MAX_STEPS": "25",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "1",
        },
        tokenizer=tokenizer,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "repeated_search_query"
    assert repeated in _masked_text(sample)
    assert _policy_masked_text(sample) == repeated
    masked_context = _policy_unmasked_text(sample)
    for prior in prior_turns:
        assert prior in masked_context
    assert masked_context.count("<tool_response>") == len(prior_turns)

    visual_unmasked = _visualized_text_by_styles(
        sample,
        tokenizer,
        {rollout_visualization._UNMASKED_TOKEN_STYLE, rollout_visualization._REWARD_NEG_STYLE},
    )
    visual_masked = _visualized_text_by_style(sample, tokenizer, rollout_visualization._MASKED_TOKEN_STYLE)
    assert repeated in visual_unmasked
    for prior in prior_turns:
        assert prior in visual_unmasked
    assert repeated not in visual_masked
    assert visual_masked.count("<tool_response>") == len(prior_turns)


def test_direct_boxed_answer_without_tool_is_penalized_and_masked():
    direct = "<think>I know it.</think>\n\\boxed{answer}"

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence first"}),
        [{"text": direct}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["fused_termination"] == "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL"
    assert sample.metadata["fused_tool_call_turns"] == 0
    assert "\\boxed{answer}" in _masked_text(sample)
    assert "<think>I know it.</think>" in _unmasked_text(sample)
    assert _policy_masked_text(sample) == ""
    assert sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]["action"].startswith("<tool_call>")


def test_direct_finish_without_tool_is_penalized_and_masked():
    direct = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Use web_search before submit"}),
        [{"text": direct}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["fused_termination"] == "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL"
    assert direct in _masked_text(sample)
    assert _policy_masked_text(sample) == ""
    assert sample.metadata["rllm_episode"]["metadata"]["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["rllm_episode"]["metadata"]["credit_assignment_error_step_index"] == 0


def test_mcp_nested_finish_payload_is_rejected_and_masked(tmp_path: Path):
    nested = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"' '\\u003ctool_call\\u003e{\\"name\\": \\"finish\\", \\"arguments\\": {}}\\u003c/tool_call\\u003e' '"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use tools before finish"),
        [{"text": _echo_call("evidence")}, {"text": nested}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert sample.metadata["fused_termination"] == "ABNORMAL_NESTED_FINISH_PAYLOAD"
    assert sample.metadata["fused_reward_debug"]["invalid_finish_payload"] == "nested_tool_call"
    assert nested in _masked_text(sample)
    assert "Execution output of [echo]" in _unmasked_text(sample)


def test_disable_thinking_visualization_keeps_thinking_empty_for_action_only_response():
    action = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    step = fused_generate._episode_step(
        observation="",
        response=action,
        action=action,
        reward=0.0,
        done=True,
        messages=[{"role": "assistant", "content": action}],
        llm_time=0.0,
        env_time=0.0,
        disable_thinking=True,
    )

    assert step["thought"] == ""
    assert step["info"]["disable_thinking"] is True
    assert rollout_visualization._step_thinking_and_response(step) == ("", action)


def test_legacy_function_tool_action_executes_and_trains(tmp_path: Path):
    legacy_tool = """
<think>
Need the tool.
</think>
<function=echo>
  <parameter=value>legacy-value</parameter>
</function>
"""
    finish = _finish_call()

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use legacy tool syntax"),
        [{"text": legacy_tool}, {"text": finish}],
        {"CREDIT_ASSIGNMENT_ENABLE": "True"},
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_tool_call_turns"] == 1
    assert "legacy-value" in _masked_text(sample)
    assert "Execution output of [echo]" in _unmasked_text(sample)


def test_no_ground_truth_direct_finish_without_tool_gets_zero_reward_even_when_credit_disabled():
    direct = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"confident answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label=None, metadata={"question": "Use web_search before submit"}),
        [{"text": direct}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "False",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["fused_termination"] == "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL"
    assert direct in _masked_text(sample)
    assert _policy_masked_text(sample) == ""
    assert sample.metadata["rllm_episode"]["metadata"]["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["rllm_episode"]["trajectories"][0]["reward"] == 0.0
    assert sample.metadata["rllm_episode"]["metrics"]["turn/tool_call_turn"] == 0.0


def test_finish_after_non_finish_tool_is_not_direct_submit_credit_event():
    search = _search_call("evidence before answer")
    search2 = _search_call("second evidence before answer")
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence first"}),
        [{"text": search}, {"text": search2}, {"text": finish}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_tool_call_turns"] == 2
    assert _masked_text(sample) == search + search2 + finish
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_mixed_tool_and_boxed_answer_breaks_loop_and_masks_only_error_turn():
    first = _search_call("first evidence")
    mixed = _search_call("second evidence") + "\n\\boxed{answer}"
    unused_finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search before answer"}),
        [{"text": first}, {"text": mixed}, {"text": unused_finish}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "mixed_tool_and_answer"
    assert sample.metadata["fused_termination"] == "ABNORMAL_MIXED_TOOL_AND_ANSWER"
    assert sample.metadata["mixed_tool_and_answer"] is True
    assert sample.metadata["fused_traj_steps"] == 2
    assert mixed in _masked_text(sample)
    assert _policy_masked_text(sample) == mixed
    assert first in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_mixed_tool_and_submit_call_breaks_loop_and_masks_only_error_turn():
    first = _search_call("first evidence")
    tool = _search_call("second evidence")
    submit = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'
    mixed = tool + submit

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search before answer"}),
        [{"text": first}, {"text": mixed}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "mixed_tool_and_answer"
    assert sample.metadata["fused_termination"] == "ABNORMAL_MIXED_TOOL_AND_ANSWER"
    assert sample.metadata["fused_traj_steps"] == 2
    assert mixed in _masked_text(sample)
    assert _policy_masked_text(sample) == mixed
    assert first in _policy_unmasked_text(sample)


def test_eval_allows_mixed_tool_and_submit_call():
    first = _search_call("first evidence")
    tool = _search_call("second evidence")
    submit = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'
    mixed = tool + submit
    final = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search before answer"}),
        [{"text": first}, {"text": mixed}, {"text": final}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER": "True",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_traj_steps"] == 3
    assert sample.metadata.get("mixed_tool_and_answer") is None
    assert sample.response == final
    assert sample.tokens == []
    assert sample.loss_mask is None


def test_eval_disables_direct_submit_abnormal_detection():
    direct = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence first"}),
        [{"text": direct}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "True",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["reward_debug"]["insufficient_searches"] is True


def test_eval_disables_repeated_search_abnormal_detection():
    first = _search_call("same query")
    repeated = _search_call("same query")
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search twice"}),
        [{"text": first}, {"text": repeated}, {"text": finish}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "1",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert "duplicate_search_detected" not in sample.metadata
    assert sample.tokens == []


def test_eval_disables_parser_error_abnormal_detection(tmp_path: Path):
    bad = "I cannot produce a tool call here."

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger parser error"),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert "tool_parser_error_count" not in sample.metadata
    assert sample.metadata["reward_debug"]["reward"] == 0.0


def test_eval_disables_tool_burst_abnormal_detection(tmp_path: Path):
    burst = "".join(_echo_call(f"burst{i}") for i in range(5))
    finish = _finish_call()

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger tool burst"),
        [{"text": burst}, {"text": finish}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS": "True",
            "FUSED_MAX_TOOL_CALLS_PER_TURN": "4",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_tool_call_turns"] == 1


def test_eval_disables_ngram_repetition_abnormal_detection(tmp_path: Path):
    reasoning = "<think>" + ("loop phrase " * 60) + "</think>\n"
    action = _echo_call("after-repeat")
    finish = _finish_call()

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger ngram repetition"),
        [{"text": reasoning + action}, {"text": finish}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION": "True",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_N": "2",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "0.20",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS": "128",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert "ngram_repetition_detected" not in sample.metadata


def test_eval_disables_length_abnormal_detection(tmp_path: Path):
    truncated = "plain truncated answer"

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Answer directly"),
        [{"text": truncated, "finish_reason": "length"}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_MAX_RESPONSE_LEN": "True",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["reward_debug"]["reward"] == 0.0


def test_boxed_text_inside_tool_arguments_is_not_mixed_answer():
    search = '<tool_call>{"name":"web_search","arguments":{"query":"literal \\\\boxed{not-answer}"}}</tool_call>'
    search2 = _search_call("second evidence after literal boxed")
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search before answer"}),
        [{"text": search}, {"text": search2}, {"text": finish}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_MIXED_TOOL_AND_ANSWER": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert _masked_text(sample) == search + search2 + finish


def test_first_step_direct_submit_is_penalized_even_when_credit_assignment_is_disabled():
    direct = "\\boxed{answer}"

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence first"}),
        [{"text": direct}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "False",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["fused_termination"] == "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL"
    assert direct in _masked_text(sample)
    assert _policy_masked_text(sample) == ""


def test_first_step_answer_tag_submit_is_penalized_and_masked():
    direct = "<think>Guessing.</think>\n<answer>answer</answer>"

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence first"}),
        [{"text": direct}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_DIRECT_SUBMIT_WITHOUT_TOOL": "False",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "direct_submit_without_tool"
    assert sample.metadata["fused_termination"] == "ABNORMAL_DIRECT_SUBMIT_WITHOUT_TOOL"
    assert "<answer>answer</answer>" in _masked_text(sample)
    assert "<think>Guessing.</think>" in _unmasked_text(sample)
    assert _policy_masked_text(sample) == ""


def test_direct_submit_credit_assignment_masks_all_tokens():
    assert (
        fused_generate._credit_assignment_loss_mask(
            output_len=7,
            turn_index=0,
            credit_event="direct_submit_without_tool",
            credit_step_index=0,
        )
        == [0] * 7
    )


def test_mixed_tool_and_answer_credit_assignment_masks_only_error_turn():
    assert (
        fused_generate._credit_assignment_loss_mask(
            output_len=5,
            turn_index=0,
            credit_event="mixed_tool_and_answer",
            credit_step_index=1,
        )
        == [0] * 5
    )
    assert (
        fused_generate._credit_assignment_loss_mask(
            output_len=5,
            turn_index=1,
            credit_event="mixed_tool_and_answer",
            credit_step_index=1,
        )
        == [1] * 5
    )


def test_credit_assignment_policy_mask_never_unmasks_base_loss_mask_tokens():
    base_loss_mask = [0, 0, 1, 1, 0, 1, 0]

    assert fused_generate._credit_assignment_loss_mask(
        output_len=len(base_loss_mask),
        turn_index=0,
        credit_event="mixed_tool_and_answer",
        credit_step_index=0,
        base_loss_mask=base_loss_mask,
    ) == base_loss_mask

    assert fused_generate._credit_assignment_loss_mask(
        output_len=len(base_loss_mask),
        turn_index=0,
        credit_event="tool_parser_error",
        credit_step_index=0,
        parser_error_token_window=5,
        base_loss_mask=base_loss_mask,
    ) == [0, 0, 1, 1, 0, 1, 0]

    assert fused_generate._credit_assignment_loss_mask(
        output_len=len(base_loss_mask),
        turn_index=0,
        credit_event="too_many_tool_calls",
        credit_step_index=0,
        action_span=(1, 6),
        base_loss_mask=base_loss_mask,
    ) == [0, 0, 1, 1, 0, 1, 0]


def test_record_pending_turns_credit_assignment_intersects_response_loss_mask():
    manager = fused_generate.TrajectoryManager()
    sid = "masked-thinking-credit"
    output_ids = [101, 102, 103, 104, 105]
    pending_turns = [
        {
            "turn": fused_generate.TurnRecord(
                prompt_ids=[1, 2],
                context_delta_ids=[1, 2],
                output_ids=output_ids,
                output_log_probs=[-0.1] * len(output_ids),
                loss_mask=[0, 1, 0, 1, 1],
                finish_reason="stop",
            ),
            "prompt_messages": [{"role": "user", "content": "x"}],
            "response_message": {"role": "assistant", "content": "y"},
            "metadata": {"sid": sid, "step": 0},
            "credit_assignment_action_span": (1, 4),
        }
    ]

    fused_generate._record_pending_turns(
        manager,
        session_id=sid,
        pending_turns=pending_turns,
        credit_event="tool_parser_error",
        credit_step_index=0,
        parser_error_token_window=256,
    )
    samples = manager.get_trajectory(sid, base_sample=Sample(index=0, prompt=""), reward=0.0)

    assert len(samples) == 1
    assert samples[0].loss_mask == [0, 1, 0, 1, 1]
    assert samples[0].policy_loss_mask == [0, 1, 0, 1, 0]
    assert samples[0].rollout_log_probs == [0.0, -0.1, 0.0, -0.1, -0.1]


def test_record_pending_turns_credit_assignment_with_tito_context_masks_only_error_action():
    manager = fused_generate.TrajectoryManager()
    sid = "multi-turn-tito-action-span-credit"
    first_output = [101, 102, 103]
    context_delta = [201, 202, 203, 204]
    second_output = [301, 302, 303, 304, 305, 306]
    pending_turns = [
        {
            "turn": fused_generate.TurnRecord(
                prompt_ids=[1, 2],
                context_delta_ids=[1, 2],
                output_ids=first_output,
                output_log_probs=[-0.1] * len(first_output),
                loss_mask=[1, 1, 1],
                finish_reason="tool_calls",
            ),
            "prompt_messages": [{"role": "user", "content": "u"}],
            "response_message": {"role": "assistant", "content": "a1"},
            "metadata": {"sid": sid, "step": 0},
        },
        {
            "turn": fused_generate.TurnRecord(
                prompt_ids=[1, 2, *first_output, *context_delta],
                context_delta_ids=context_delta,
                output_ids=second_output,
                output_log_probs=[-0.2] * len(second_output),
                loss_mask=[1, 0, 1, 1, 0, 1],
                finish_reason="stop",
            ),
            "prompt_messages": [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "content": "tool-observation"},
            ],
            "response_message": {"role": "assistant", "content": "a2"},
            "metadata": {"sid": sid, "step": 1},
            "credit_assignment_action_span": (1, 5),
        },
    ]

    fused_generate._record_pending_turns(
        manager,
        session_id=sid,
        pending_turns=pending_turns,
        credit_event="repeated_search_query",
        credit_step_index=1,
        parser_error_token_window=256,
    )
    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=0, prompt="", rollout_id=99),
        reward=0.0,
        extra_metadata={"credit_assignment_event": "repeated_search_query"},
    )

    assert len(samples) == 1
    assert samples[0].loss_mask == [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1]
    assert samples[0].policy_loss_mask == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert samples[0].metadata["credit_assignment_event"] == "repeated_search_query"
    assert samples[0].rollout_log_probs == [
        -0.1,
        -0.1,
        -0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.2,
        0.0,
        -0.2,
        -0.2,
        0.0,
        -0.2,
    ]


def test_record_pending_turns_long_horizon_tito_credit_assignment_is_iterative_and_aligned():
    manager = fused_generate.TrajectoryManager()
    sid = "long-horizon-record-pending-turns"
    pending_turns = []
    expected_loss_mask = []
    expected_logprobs = []
    prompt_messages = [{"role": "user", "content": "u"}]
    total_turns = 1005
    for idx in range(total_turns):
        context_delta = [10_000 + idx]
        output_ids = [20_000 + idx * 3, 20_001 + idx * 3, 20_002 + idx * 3]
        loss_mask = [1, 0, 1] if idx % 17 == 0 else [1, 1, 1]
        if idx > 0:
            expected_loss_mask.extend([0])
            expected_logprobs.extend([0.0])
        expected_loss_mask.extend(loss_mask)
        expected_logprobs.extend([-0.5 if mask else 0.0 for mask in loss_mask])
        response_message = {"role": "assistant", "content": f"a{idx}"}
        pending_turns.append(
            {
                "turn": fused_generate.TurnRecord(
                    prompt_ids=[1, 2, *context_delta],
                    context_delta_ids=context_delta,
                    output_ids=output_ids,
                    output_log_probs=[-0.5] * len(output_ids),
                    loss_mask=loss_mask,
                    finish_reason="tool_calls" if idx + 1 < total_turns else "stop",
                ),
                "prompt_messages": list(prompt_messages),
                "response_message": response_message,
                "metadata": {"sid": sid, "step": idx},
                "credit_assignment_action_span": (1, 3) if idx + 1 == total_turns else None,
            }
        )
        prompt_messages.extend(
            [
                response_message,
                {"role": "tool", "content": f"obs{idx}"},
            ]
        )

    fused_generate._record_pending_turns(
        manager,
        session_id=sid,
        pending_turns=pending_turns,
        credit_event="max_turns_exceeded",
        credit_step_index=total_turns - 1,
        parser_error_token_window=256,
    )
    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=0, prompt="", rollout_id=1005),
        reward=0.0,
        extra_metadata={"credit_assignment_event": "max_turns_exceeded"},
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.response_length == len(expected_loss_mask)
    assert sample.loss_mask == expected_loss_mask
    assert sample.rollout_log_probs == expected_logprobs
    assert sum(sample.policy_loss_mask) == 2
    expected_policy_mask = [0] * sample.response_length
    final_response_start = sample.response_length - len(pending_turns[-1]["turn"].output_ids)
    expected_policy_mask[final_response_start + 1] = 1
    expected_policy_mask[final_response_start + 2] = 1
    assert sample.policy_loss_mask == expected_policy_mask
    assert sample.metadata["credit_assignment_event"] == "max_turns_exceeded"


def test_parser_error_credit_assignment_masks_only_error_turn_after_history(tmp_path: Path):
    good = _echo_call("before-error")
    bad = "I cannot produce a tool call here."

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger parser error"),
        [{"text": good}, {"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert bad in _masked_text(sample)
    assert _policy_masked_text(sample) == bad
    assert good in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_parser_error_credit_assignment_masks_only_error_tail(tmp_path: Path):
    good = _echo_call("before-error-tail")
    bad = "<think>" + ("reasoning " * 20) + '<tool_call>{"name":"echo"'
    malformed_action = '<tool_call>{"name":"echo"'

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger long parser error"),
        [{"text": good}, {"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR_TOKEN_WINDOW": "16",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert malformed_action in _masked_text(sample)
    assert _policy_masked_text(sample) == malformed_action
    assert good in _policy_unmasked_text(sample)
    assert bad[: -len(malformed_action)] in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_tool_burst_credit_assignment_masks_only_burst_turn_after_history(tmp_path: Path):
    good = _echo_call("before-burst")
    reasoning = "<think>Need many calls.</think>\n"
    burst = "".join(_echo_call(f"burst{i}") for i in range(5))

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger tool burst"),
        [{"text": good}, {"text": reasoning + burst}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS": "True",
            "FUSED_MAX_TOOL_CALLS_PER_TURN": "4",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "too_many_tool_calls"
    assert burst in _masked_text(sample)
    assert _policy_masked_text(sample) == burst
    assert reasoning in _policy_unmasked_text(sample)
    assert good in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_repeated_search_credit_assignment_after_many_prior_turns_masks_only_repeated_turn():
    prior_queries = [f"q{i:02d}" for i in range(12)]
    prior_turns = [f'<tool_call>{{"name":"web_search","arguments":{{"query":"{query}"}}}}</tool_call>' for query in prior_queries]
    reasoning = "<think>Try the same query again.</think>\n"
    repeated = '<tool_call>{"name":"web_search","arguments":{"query":"q07"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "x"}, metadata={"question": "Search many times"}),
        [{"text": text} for text in [*prior_turns, reasoning + repeated]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_MAX_STEPS": "20",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "1",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "repeated_search_query"
    assert repeated in _masked_text(sample)
    assert _policy_masked_text(sample) == repeated
    assert reasoning in _policy_unmasked_text(sample)
    for prior in prior_turns:
        assert prior in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_ngram_repetition_credit_assignment_masks_only_repeated_turn_action(tmp_path: Path):
    good = _echo_call("before-ngram-repeat")
    reasoning = "<think>" + ("loop phrase " * 60) + "</think>\n"
    action = _echo_call("after-repeat")

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger ngram repetition"),
        [{"text": good}, {"text": reasoning + action}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION": "True",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_N": "2",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "0.20",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_MIN_TOKENS": "128",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "ngram_repetition"
    assert sample.metadata["fused_termination"] == "ABNORMAL_NGRAM_REPETITION"
    assert sample.metadata["ngram_repetition_detected"] is True
    assert action in _masked_text(sample)
    assert _policy_masked_text(sample) == action
    assert reasoning in _policy_unmasked_text(sample)
    assert good in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_max_turns_credit_assignment_masks_only_final_turn_action(tmp_path: Path):
    first = _echo_call("before-max-turns")
    final_reasoning = "<think>One more try.</think>\n"
    final_action = _echo_call("at-limit")

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger max turns"),
        [{"text": first}, {"text": final_reasoning + final_action}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_MAX_TURNS": "True",
            "FUSED_MCP_MAX_STEPS": "2",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "max_turns_exceeded"
    assert sample.metadata["fused_termination"] == "max_turns_exceeded"
    assert final_action in _masked_text(sample)
    assert _policy_masked_text(sample) == final_action
    assert final_reasoning in _policy_unmasked_text(sample)
    assert first in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_max_response_len_credit_assignment_masks_only_error_tail():
    assert fused_generate._credit_assignment_loss_mask(
        output_len=10,
        turn_index=0,
        credit_event="max_response_len_exceeded",
        credit_step_index=0,
        parser_error_token_window=4,
    ) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_search_bypass_credit_assignment_keeps_full_action_mask():
    first = '<tool_call>{"name":"web_search","arguments":{"query":"use search"}}</tool_call>'
    second = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    assert (
        fused_generate._credit_assignment_loss_mask(
            output_len=len(first),
            turn_index=0,
            credit_event="search_bypass",
            credit_step_index=1,
        )
        is None
    )
    assert (
        fused_generate._credit_assignment_loss_mask(
            output_len=len(second),
            turn_index=1,
            credit_event="search_bypass",
            credit_step_index=1,
        )
        is None
    )


def test_tail_guard_credit_assignment_masks_all_actions():
    loss_mask = fused_generate._credit_assignment_loss_mask(
        output_len=3,
        turn_index=0,
        credit_event="tail_guard_early_stop",
        credit_step_index=None,
    )
    manager = fused_generate.TrajectoryManager()
    sid = "tail-guard"
    manager.record_turn(
        sid,
        turn=fused_generate.TurnRecord(
            prompt_ids=[1, 2],
            output_ids=[3, 4, 5],
            finish_reason="stop",
            output_log_probs=[-0.1, -0.1, -0.1],
            policy_loss_mask=loss_mask,
        ),
        prompt_messages=[{"role": "user", "content": "x"}],
        response_message={"role": "assistant", "content": "y"},
    )

    samples = manager.get_trajectory(
        sid,
        base_sample=Sample(index=0, prompt="x"),
        reward=1.0,
        allow_fully_masked=True,
    )

    assert len(samples) == 1
    assert samples[0].reward == 1.0
    assert samples[0].loss_mask == [1, 1, 1]
    assert samples[0].policy_loss_mask == [0, 0, 0]


if __name__ == "__main__":
    test_normalize_rllm_extra_info_task()
    test_qwen_tool_parser_finish_and_answer_fallback()
    test_resolve_cli_and_et_modes_without_docker_reset()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_local_mcp_toolset_and_verifier(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_mcp_tool_load_error_is_nonfatal(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_custom_generate_with_mocked_sglang(Path(tmp))
    test_render_prompt_ids_accepts_batch_encoding_like_object()
    test_initial_messages_include_tool_prompt()
