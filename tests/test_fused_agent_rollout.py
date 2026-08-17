import json
import os
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import slime.rollout.fused_agent.env as fused_env
from slime.rollout.fused_agent.env import (
    FusedEnvironment,
    _exact_match_reward,
    _format_retrieval,
    _limit_summary_input,
    _summary_units,
    normalize_task,
    resolve_task_mode,
)
from slime.rollout.fused_agent.mcp_workspace import cleanup_task_workspaces, prepare_mcp_workspace
import slime.rollout.fused_agent.generate as fused_generate
from slime.rollout.fused_agent.generate import (
    _format_tool_observation,
    _initial_messages,
    _last_assistant_context_start_idx,
    _render_gemma4_tito_delta_ids,
    _render_qwen3_tito_delta_ids,
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
    FUSED_MCP_SYSTEM_PROMPT,
    FUSED_SEARCH_SYSTEM_PROMPT,
    build_system_prompt,
    finish_schema,
    normalize_harness,
    web_search_schema,
)
from slime.rollout.fused_agent.rllm_deepresearch import (
    SEARCH_SYSTEM_PROMPT as RLLM_DR_SEARCH_SYSTEM_PROMPT,
    local_search_schema,
    parse_refine_response,
)
from slime.rollout.fused_agent import deepsearch_world as dsw
from slime.rollout.fused_agent.search_gym import SEARCH_GYM_SYSTEM_PROMPT
from slime.rollout.fused_agent.agentcpm_explore import (
    AGENTCPM_EXPLORE_SYSTEM_PROMPT,
    build_messages as build_agentcpm_explore_messages,
    fetch_url_schema as agentcpm_fetch_url_schema,
    search_schema as agentcpm_search_schema,
)
from slime.utils import visualization as rollout_visualization
from slime.utils.types import Sample


NUM_GPUS = 0


def test_parser_error_log_can_be_disabled_without_changing_parser_state(monkeypatch):
    monkeypatch.setenv("SLIME_TOOL_PARSER_ERROR_LOG_ENABLED", "false")
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("parser log should stay disabled"))
    parser = SimpleNamespace(last_schema_errors=["invalid tool call"])

    fused_generate._record_tool_parser_errors(
        args=SimpleNamespace(dump_details=None),
        base_sample=Sample(index=1, group_index=2, rollout_id=3),
        session_id="session",
        rollout_step=0,
        model_name="qwen3",
        response="bad response",
        parser=parser,
        prompt_ids=[1, 2],
        evaluation=False,
    )

    assert parser.last_schema_errors == ["invalid tool call"]


def test_parser_error_log_remains_available_outside_rejection_sampling(tmp_path, monkeypatch):
    log_path = tmp_path / "parser-errors.jsonl"
    monkeypatch.setenv("SLIME_TOOL_PARSER_ERROR_LOG_ENABLED", "true")
    monkeypatch.setenv("SLIME_TOOL_PARSER_ERROR_LOG", str(log_path))
    parser = SimpleNamespace(
        last_schema_errors=["invalid tool call"],
        last_schema_error_kinds=["invalid_json"],
        last_schema_error_spans=[[1, 4]],
    )

    fused_generate._record_tool_parser_errors(
        args=SimpleNamespace(dump_details=None),
        base_sample=Sample(index=1, group_index=2, rollout_id=3),
        session_id="session",
        rollout_step=4,
        model_name="qwen3",
        response="bad response",
        parser=parser,
        prompt_ids=[1, 2],
        evaluation=False,
    )

    record = json.loads(log_path.read_text())
    assert record["rollout_id"] == 3
    assert record["errors"] == ["invalid tool call"]
    assert record["error_kinds"] == ["invalid_json"]
    assert record["response"] == "bad response"


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


class FakeQwen3ChatTemplateTokenizer(FakeChatTemplateTokenizer):
    name_or_path = "/models/Qwen3-8B"
    unk_token_id = -1

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            # The character tokenizer represents the special token literally;
            # use a dedicated id so boundary completion is easy to assert.
            return 0x10FFFF
        return self.unk_token_id


class FakeQwen35ChatTemplateTokenizer(FakeQwen3ChatTemplateTokenizer):
    name_or_path = "/models/Qwen3.5-4B"

    def __init__(self, *, drift_assistant_end: bool = False):
        super().__init__(drift_assistant_end=drift_assistant_end)
        self.tools_seen = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        self.tools_seen.append(kwargs.pop("tools", None))
        kwargs.pop("enable_thinking", None)
        if any(message.get("role") == "tool" for message in messages) and not any(message.get("role") == "user" for message in messages):
            raise ValueError("No user query found in messages")
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


class RecordingToolsTokenizer(FakeTokenizer):
    name_or_path = "/models/Qwen3-8B"

    def __init__(self):
        self.tools_seen = []
        self.chat_template_kwargs_seen = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, tools=None, **_kwargs):
        self.tools_seen.append(tools)
        self.chat_template_kwargs_seen.append(_kwargs)
        return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


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
                body += f"<|tool_call>call:{function['name']}" f"{Gemma4ToolParser._format_argument(function.get('arguments') or {}, escape_keys=False)}" "<tool_call|>"
            if message.get("tool_responses"):
                body += "<|tool_response>"
                for response in message["tool_responses"]:
                    body += f"response:{response['name']}" f"{Gemma4ToolParser._format_argument(response.get('response') or {}, escape_keys=False)}"
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
    prompt_ids_seen: list[list[int]] | None = None,
    sampling_params_seen: list[dict] | None = None,
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
        expected_weight_version=None,
        require_weight_version=False,
    ):
        if prompt_ids_seen is not None:
            prompt_ids_seen.append(list(prompt_ids))
        if sampling_params_seen is not None:
            sampling_params_seen.append(dict(sampling_params))
        item = calls.pop(0)
        text = item["text"]
        return {
            "text": text,
            "output_ids": [ord(c) for c in text],
            "output_logprobs": [-0.1] * len(text),
            "finish_reason": item.get("finish_reason", "stop"),
            "prompt_tokens": item.get("prompt_tokens", len(prompt_ids)),
            "cached_tokens": item.get("cached_tokens", 0),
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
    evidence = tools["echo"](value="verification")
    return {
        "passed": (
            isinstance(answer, dict)
            and answer.get("done") is True
            and evidence.get("echo") == "verification"
        )
    }
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


def test_fused_training_accumulates_real_prefix_cache_metrics_once(tmp_path: Path):
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path),
        [
            {"text": _echo_call("first"), "prompt_tokens": 100, "cached_tokens": 0},
            {"text": _finish_call(), "prompt_tokens": 180, "cached_tokens": 96},
        ],
        {"SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false"},
    )
    samples = result if isinstance(result, list) else [result]

    assert sum(sample.prefix_cache_info.cached_tokens for sample in samples) == 96
    assert sum(sample.prefix_cache_info.total_prompt_tokens for sample in samples) == 280
    assert samples[0].metadata["fused_profile"]["prefix_cache_hit_rate"] == pytest.approx(96 / 280)


def _search_call(query: str) -> str:
    return f'<tool_call>{{"name":"web_search","arguments":{{"query":"{query}","max_results":3}}}}</tool_call>'


def test_search_gym_uses_checkpoint_tool_call_protocol(monkeypatch):
    tokenizer = RecordingToolsTokenizer()
    sample = Sample(
        prompt="placeholder",
        label="answer",
        metadata={"question": "What is the answer?", "data_source": "searchR1_nq"},
    )

    async def fake_retrieve(_url, payload, *, retry_budget, episode_cache):
        del payload, retry_budget, episode_cache
        return {"results": [{"content": {"title": "Doc", "chunk_text": "evidence"}}]}, 0, None

    monkeypatch.setattr(fused_env, "_retrieve_json_cached", fake_retrieve)

    result = _run_generate_with_fake_sglang(
        sample,
        [
            {"text": '<think>reasoning\n<tool_call>{"name":"search","arguments":{"query":"facts"}}</tool_call>'},
            {"text": "more reasoning</think>\n<answer>answer</answer>"},
        ],
        {"FUSED_HARNESS": "search_gym", "FUSED_DISABLE_THINKING": "False"},
        evaluation=True,
        tokenizer=tokenizer,
    )

    assert result[0].metadata["fused_termination"] == "env_done"
    assert result[0].metadata["fused_tool_call_turns"] == 1
    assert tokenizer.tools_seen
    assert all(tools is None for tools in tokenizer.tools_seen)
    assert "<tool_call>" in SEARCH_GYM_SYSTEM_PROMPT
    assert "<search>" not in SEARCH_GYM_SYSTEM_PROMPT
    parser = make_tool_parser(tokenizer.name_or_path, valid_tools={"search", "finish"})
    assert parser.parse("reasoning</think><search>legacy</search>") == []


def test_agentcpm_explore_uses_upstream_inline_tool_protocol(monkeypatch):
    tokenizer = RecordingToolsTokenizer()
    sample = Sample(
        prompt="placeholder",
        label="answer",
        metadata={"question": "What is the answer?", "data_source": "asearcher"},
    )

    async def fake_retrieve(_url, payload, *, retry_budget, episode_cache):
        del payload, retry_budget, episode_cache
        return {"results": [{"content": {"title": "Doc", "chunk_text": "evidence"}}]}, 0, None

    monkeypatch.setattr(fused_env, "_retrieve_json_cached", fake_retrieve)
    result = _run_generate_with_fake_sglang(
        sample,
        [
            {"text": '<think>research</think><tool_call>{"name":"search","arguments":{"query":["facts"]}}</tool_call>'},
            {"text": "<think>done</think><answer>answer</answer>"},
        ],
        {"FUSED_HARNESS": "agentcpm_explore", "FUSED_DISABLE_THINKING": "False"},
        evaluation=True,
        tokenizer=tokenizer,
    )

    assert normalize_harness("agentcpm-explore") == "agentcpm_explore"
    assert result[0].metadata["fused_termination"] == "env_done"
    assert result[0].metadata["fused_tool_call_turns"] == 1
    assert tokenizer.tools_seen and all(tools is None for tools in tokenizer.tools_seen)

    messages = build_agentcpm_explore_messages("question", [agentcpm_search_schema(), agentcpm_fetch_url_schema()], current_date="2026-08-03")
    assert AGENTCPM_EXPLORE_SYSTEM_PROMPT.startswith("You are a deep research assistant.")
    assert '"name": "search"' in messages[0]["content"]
    assert '"name": "fetch_url"' in messages[0]["content"]
    assert "Current date: 2026-08-03" in messages[0]["content"]
    assert messages[1]["content"] == "Your task is to answer the user's question: question"


def test_search_gym_rejects_unclosed_think_with_text_after_action():
    response = '<think>reasoning<tool_call>{"name":"search","arguments":{"query":"facts"}}</tool_call>tail'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="answer", metadata={"question": "What is the answer?"}),
        [{"text": response}],
        {"FUSED_HARNESS": "search_gym", "FUSED_DISABLE_THINKING": "False"},
        evaluation=True,
        tokenizer=RecordingToolsTokenizer(),
    )

    assert result[0].metadata["fused_termination"] == "ABNORMAL_EVAL_RESPONSE"
    assert result[0].metadata["eval_response_anomalies"] == ["unbalanced_tags"]


def test_deepsearch_world_plan_then_answer_uses_independent_prompts():
    tokenizer = RecordingToolsTokenizer()
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Ulm", metadata={"question": "Where was Einstein born?"}),
        [
            {"text": '<think>Plan the entity.</think>\n{"completed_list":[],"todo_list":["Einstein"],"experience":[],"information":[]}'},
            {"text": "<think>The visited evidence confirms the answer.</think><answer>Ulm</answer>"},
        ],
        {
            "FUSED_HARNESS": "deepsearch_world",
            "FUSED_DISABLE_THINKING": "False",
            "SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE": "1",
        },
        evaluation=True,
        tokenizer=tokenizer,
    )

    sample = result[0]
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_traj_steps"] == 2
    assert sample.response.endswith("<answer>Ulm</answer>")
    assert tokenizer.tools_seen == [None, None]
    assert all(kwargs["enable_thinking"] is False for kwargs in tokenizer.chat_template_kwargs_seen)


def test_deepsearch_world_parser_accepts_json_tools_for_qwen35_checkpoint():
    parser = dsw.DeepSearchWorldParser(valid_tools={"web_search_wiki", "visit_wiki", "finish"})
    response = '<think>Search next.</think><tool_call>{"name":"web_search_wiki","arguments":{"query":"Ulm"}}</tool_call>'

    assert parser.parse(response) == [ToolCall("web_search_wiki", {"query": "Ulm"}, response.index("<tool_call>"), len(response))]


def test_deepsearch_world_parser_uses_last_action_after_replayed_example():
    parser = dsw.DeepSearchWorldParser(valid_tools={"web_search_wiki", "visit_wiki", "finish"})
    response = "replayed example <answer>South Melbourne</answer>\n" '<think>Now handle the real task.</think><tool_call>{"name":"web_search_wiki",' '"arguments":{"query":"Khanzada Begum"}}</tool_call>'

    action = parser.parse(response)[0]

    assert action.name == "web_search_wiki"
    assert action.arguments == {"query": "Khanzada Begum"}


def test_deepsearch_world_caps_generation_at_historical_1024_tokens():
    sampling_params_seen = []
    _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Ulm", metadata={"question": "Where was Einstein born?"}),
        [
            {"text": '<think>Plan.</think>{"completed_list":[],"todo_list":[],"experience":[],"information":[]}'},
            {"text": "<think>Known.</think><answer>Ulm</answer>"},
        ],
        {"FUSED_HARNESS": "deepsearch_world", "PER_STEP_MAX_TOKENS": "8192"},
        evaluation=True,
        sampling_params_seen=sampling_params_seen,
    )

    assert [params["max_new_tokens"] for params in sampling_params_seen] == [1024, 1024]


def test_deepsearch_world_runs_end_phase_after_action_limit():
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Ulm", metadata={"question": "Where was Einstein born?"}),
        [
            {"text": '<think>Plan.</think>{"completed_list":[],"todo_list":[],"experience":[],"information":[]}'},
            {"text": "<think>I need more evidence.</think>"},
            {"text": "<think>Best answer from the available evidence.</think><answer>Ulm</answer>"},
        ],
        {
            "FUSED_HARNESS": "deepsearch_world",
            "DEEPSEARCH_WORLD_MAX_STEPS": "1",
            "FUSED_DISABLE_THINKING": "False",
        },
        evaluation=True,
    )

    assert result[0].metadata["fused_termination"] == "env_done"
    assert result[0].metadata["fused_traj_steps"] == 3


def test_deepsearch_world_recent_steps_keeps_only_last_two():
    steps = [dsw.step_record(f"<think>thought {index}</think>") for index in range(3)]
    recent = dsw.format_recent_steps(steps)

    assert "thought 0" not in recent
    assert "thought 1" in recent
    assert "thought 2" in recent


def test_deepsearch_world_local_search_result_can_be_visited(monkeypatch):
    async def fake_retrieve(_url, _payload, *, retry_budget, episode_cache):
        del retry_budget, episode_cache
        return (
            {
                "results": [
                    {
                        "title": "Albert Einstein",
                        "document": {"text": ("Albert Einstein was born in Ulm, Germany, on 14 March 1879. " "He later became a theoretical physicist known for developing the theory of relativity. " "The biographical page identifies Ulm as his place of birth.")},
                    }
                ]
            },
            0,
            None,
        )

    monkeypatch.setattr(fused_env, "_retrieve_json_cached", fake_retrieve)
    env = FusedEnvironment(
        {"question": "Where was Einstein born?", "data_source": "asearcher"},
        retrieval_url="http://retriever",
        deepsearch_world=True,
    )

    search_result, _, _, _ = asyncio.run(env.step(ToolCall("web_search_wiki", {"query": "Albert Einstein"})))
    url = next(line.removeprefix("URL: ") for line in search_result.splitlines() if line.startswith("URL: "))
    page, _, _, info = asyncio.run(env.step(ToolCall("visit_wiki", {"url": url})))

    assert "born in Ulm" in page
    assert info["tools/visit_cache_hit"] == 1


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


def test_rllm_deepresearch_harness_aliases_and_prompt():
    assert normalize_harness("rllm_dr") == "rllm_deepresearch"
    assert normalize_harness("deepresearch") == "rllm_deepresearch"
    assert normalize_harness("cutbill") == "cut_bill"
    assert normalize_harness("cut-bill") == "cut_bill"
    tools = [local_search_schema(), finish_schema()]
    question = "Who?When ready, output the final answer enclosed in <answer> and </answer> tags. " "Do not generate any content after the </answer> tag."
    messages = _initial_messages("rllm_deepresearch", "web_search", question, tools, "Qwen/Qwen3-8B")

    assert messages[0]["content"].startswith(RLLM_DR_SEARCH_SYSTEM_PROMPT)
    assert '"name": "local_search"' in messages[0]["content"]
    assert '"name": "finish"' in messages[0]["content"]
    assert "only by calling finish" in messages[0]["content"]
    assert "\\boxed{} format" not in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Who?"}


def test_rllm_deepresearch_refine_response_requires_both_blocks():
    assert parse_refine_response("<think>reason</think><information>evidence</information>") == "evidence"
    with pytest.raises(ValueError):
        parse_refine_response("<information>evidence</information>")
    with pytest.raises(ValueError):
        parse_refine_response("<think>reason</think>evidence")


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


def test_qwen_tool_parser_ignores_tool_call_markup_inside_thinking():
    parser = QwenToolParser(valid_tools={"web_search", "finish"})
    hidden = '<tool_call>{"name":"web_search","arguments":{"query":"draft"}}</tool_call>'
    action = '<tool_call>{"name":"web_search","arguments":{"query":"actual"}}</tool_call>'
    response = f"<think>Next I will emit {hidden}</think>\n{action}"

    calls = parser.parse(response)

    assert [(call.name, call.arguments) for call in calls] == [("web_search", {"query": "actual"})]
    assert calls[0].start == response.index(action)
    assert calls[0].end == len(response)


def test_qwen_tool_parser_keeps_multiple_actions_after_thinking():
    parser = QwenToolParser(valid_tools={"web_search", "finish"})
    first = '<tool_call>{"name":"web_search","arguments":{"query":"one"}}</tool_call>'
    second = '<tool_call>{"name":"web_search","arguments":{"query":"two"}}</tool_call>'

    calls = parser.parse(f"<think>plan</think>\n{first}{second}")

    assert [call.arguments["query"] for call in calls] == ["one", "two"]


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


def test_qwen_parser_ignores_tool_tags_inside_thinking_and_requires_closed_calls():
    parser = make_tool_parser("Qwen3-8B", valid_tools={"web_search"})
    response = '<think>do not execute <tool_call>{"name":"web_search","arguments":{}}</tool_call></think>' '<tool_call>{"name":"web_search","arguments":{"query":"q"}}</tool_call>'
    calls = parser.parse(response)
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "q"}

    assert parser.parse('<tool_call>{"name":"web_search","arguments":{"query":"q"}}') == []


def test_qwen_parser_does_not_execute_malformed_arguments_and_repairs_only_trailing_comma():
    parser = make_tool_parser("Qwen3-8B", valid_tools={"web_search"})
    malformed = '<tool_call>{"name":"web_search","arguments":["q"]}</tool_call>'
    assert parser.parse(malformed) == []

    repaired = '<tool_call>{"name":"web_search","arguments":{"query":"q",}}</tool_call>'
    calls = parser.parse(repaired)
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "q"}


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
    assert action_text == ("<tool_call>\n<function=web_search>\n" "<parameter=query>\nsenior comic artist\n</parameter>\n" "<parameter=max_results>\n4\n</parameter>\n" "</function>\n</tool_call>")
    assert parser.parse(action_text)[0].arguments == {"query": "senior comic artist", "max_results": 4}


def test_qwen35_bare_function_block_is_a_protocol_error():
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})

    calls = parser.parse(
        "<function=web_search>\n<parameter=query>facts</parameter>\n</function>"
    )

    assert calls == []
    assert parser.last_schema_errors == [
        "tool function block is missing the required <tool_call> wrapper"
    ]


def test_qwen35_bare_function_terminates_rollout_as_tool_parser_error():
    bad = "<function=web_search>\n<parameter=query>facts</parameter>\n</function>"

    result = _run_generate_with_fake_sglang(
        Sample(
            prompt="placeholder",
            label={"answer": "answer"},
            metadata={"question": "Use web_search before submit"},
        ),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeQwen35ChatTemplateTokenizer(),
    )

    assert len(result) == 1
    assert result[0].reward == 0.0
    assert result[0].metadata["credit_assignment_event"] == "tool_parser_error"
    assert result[0].metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"


def test_initial_messages_configure_runtime_qwen35_parser_schema():
    tools = [web_search_schema(), finish_schema()]
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search", "finish", "submit"})

    _initial_messages("gem", "webqa", "question", tools, "Qwen3.5-4B", tool_parser=parser)
    call = parser.parse("<tool_call>\n<function=web_search>\n" "<parameter=query>\nquery\n</parameter>\n" "<parameter=max_results>\n10\n</parameter>\n" "</function>\n</tool_call>")[0]

    assert call.arguments == {"query": "query", "max_results": 10}


def test_qwen35_native_tool_template_is_the_only_tool_prompt():
    tools = [web_search_schema(), finish_schema()]
    tokenizer = FakeQwen35ChatTemplateTokenizer()
    parser = make_tool_parser(tokenizer.name_or_path, valid_tools={"web_search", "finish"})
    messages = _initial_messages(
        "gem",
        "web_search",
        "question",
        tools,
        tokenizer.name_or_path,
        tool_parser=parser,
        inline_tool_prompt=False,
    )

    assert "<tools>" not in messages[0]["content"]
    _render_prompt_ids(tokenizer, messages, tools=tools, disable_thinking=False)
    assert tokenizer.tools_seen == [tools]
    assert parser.parse("<tool_call>\n<function=web_search>\n<parameter=query>q</parameter>\n</function>\n</tool_call>")[0].arguments == {"query": "q"}


def test_qwen3_coder_parser_accepts_colon_before_integer_value():
    tools = [web_search_schema()]
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})
    parser.get_tool_prompt("\n".join(json.dumps(t, indent=0, ensure_ascii=False) for t in tools))

    call = parser.parse("<tool_call>\n<function=web_search>\n" "<parameter=query>\n: 10\n</parameter>\n" "<parameter=max_results>\n: 10\n</parameter>\n" "</function>\n</tool_call>")[0]

    assert call.arguments == {"query": ": 10", "max_results": 10}


def test_qwen3_coder_parser_normalizes_quoted_parameter_names_and_drops_unknown_fields(caplog):
    tools = [web_search_schema()]
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})
    parser.get_tool_prompt("\n".join(json.dumps(t, indent=0, ensure_ascii=False) for t in tools))

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse("<tool_call>\n<function=web_search>\n" '<parameter="query>\nseed cone\n</parameter>\n' "<parameter=include_full_text>\ntrue\n</parameter>\n" "</function>\n</tool_call>")

    assert len(calls) == 1
    assert calls[0].arguments == {"query": "seed cone"}
    assert parser.last_schema_errors == [
        "malformed parameter name '\"query'",
        "unknown parameter 'include_full_text' for tool 'web_search'",
    ]
    assert "not defined in the tool parameters" not in caplog.text


def test_qwen35_schema_violation_terminates_current_turn_as_tool_parser_error():
    bad = "<tool_call>\n<function=web_search>\n" '<parameter="query>seed cone</parameter>\n' "<parameter=include_full_text>true</parameter>\n" "</function>\n</tool_call>"

    result = _run_generate_with_fake_sglang(
        Sample(
            prompt="placeholder",
            label={"answer": "answer"},
            metadata={"question": "Use web_search before submit"},
        ),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeQwen35ChatTemplateTokenizer(),
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_event"] == "tool_parser_error"
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert malformed_action in _masked_text(sample)
    assert _policy_masked_text(sample) == malformed_action
    assert sample.metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"
    assert sample.metadata["tool_parser_errors"] == [
        "malformed parameter name '\"query'",
        "unknown parameter 'include_full_text' for tool 'web_search'",
    ]
    assert _policy_masked_text(sample) == bad


def test_qwen35_invalid_typed_parameter_is_a_schema_error():
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse("<tool_call>\n<function=web_search>\n" "<parameter=query>facts</parameter>\n" "<parameter=max_results>5junk</parameter>\n" "</function>\n</tool_call>")

    assert calls[0].arguments == {"query": "facts"}
    assert parser.last_schema_errors == ["invalid value for parameter 'max_results' of tool 'web_search'"]


@pytest.mark.parametrize(
    "empty_block",
    [
        "<parameter=max_results></parameter>",
        "<parameter=max_results>\n</parameter>",
        "<parameter=max_results>   </parameter>",
    ],
)
def test_qwen35_empty_typed_parameter_is_omitted_without_a_schema_error(empty_block):
    # An empty typed block carries no value to coerce.  Dropping the argument lets
    # the tool surface a recoverable missing-argument error, whereas recording a
    # schema error would terminate the trajectory via credit assignment.
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse("<tool_call>\n<function=web_search>\n" "<parameter=query>facts</parameter>\n" f"{empty_block}\n" "</function>\n</tool_call>")

    assert calls[0].arguments == {"query": "facts"}
    assert parser.last_schema_errors == []


def test_qwen35_empty_string_parameter_is_preserved():
    # Only typed parameters are omitted when empty; an empty string parameter is a
    # legitimate value and must still reach the tool.
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse("<tool_call>\n<function=web_search>\n<parameter=query></parameter>\n</function>\n</tool_call>")

    assert calls[0].arguments == {"query": ""}
    assert parser.last_schema_errors == []


@pytest.mark.parametrize(
    ("parameter", "value", "expected"),
    [
        ("max_results", "10</result>", 10),
        ("max_results", "10", 10),
        ("query", "literal < tag", "literal < tag"),
    ],
)
def test_qwen35_parser_repairs_only_leaked_scalar_closing_fragment(parameter, value, expected):
    parser = make_tool_parser("Qwen3.5-4B", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))
    calls = parser.parse(f"<tool_call>\n<function=web_search>\n<parameter={parameter}>{value}</parameter>\n" "</function>\n</tool_call>")
    assert calls[0].arguments[parameter] == expected
    assert parser.last_schema_errors == []


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
    assert 'query:{description:<|"|>Search query.<|"|>,type:<|"|>STRING<|"|>}' in prompt
    assert "<|tool_call>call:TOOL_NAME{" in prompt
    assert "<tool_call|><|tool_response>" not in prompt
    assert "TOOL_NAME must exactly match one of: finish, web_search" in prompt
    assert "Never emit <|channel>call:, finish.result:" in prompt
    assert 'finish{command:<|"|>submit<|"|>,result:<|"|>CONCISE_FINAL_ANSWER<|"|>}' in prompt


def test_gemma4_tool_prompt_uses_structured_finish_contract_for_union_schema():
    schema = finish_schema(structured_result=True)
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    prompt = parser.get_tool_prompt(json.dumps(schema, ensure_ascii=False))

    assert 'finish{command:<|"|>submit<|"|>,result:{...}}' in prompt
    assert "finish.result" in prompt
    assert 'result:<|"|>CONCISE_FINAL_ANSWER<|"|>' not in prompt


def test_gemma4_tool_parser_strictly_validates_declared_tool_names_and_argument_types():
    schema = tool_schema(
        "submit_result_difficulty_3",
        "Submit a structured result.",
        {"result": {"type": "object"}},
        ["result"],
    )
    parser = make_tool_parser("gemma4", valid_tools={"submit_result_difficulty_3"})
    prompt = parser.get_tool_prompt(json.dumps(schema))

    assert "submit_result_difficulty_3.result" in prompt
    legacy_string = '<|tool_call>call:submit_result_difficulty_3{result:<|"|>{"framework_analysis": []}<|"|>}<tool_call|>'
    calls = parser.parse(legacy_string)
    assert calls[0].arguments == {"result": {"framework_analysis": []}}
    assert parser.last_schema_errors == []

    invalid = '<|tool_call>call:submit_result_difficulty_3{result:<|"|>not-json<|"|>}<tool_call|>'
    assert parser.parse(invalid) == []
    assert "invalid type for parameter 'result'" in parser.last_schema_errors[0]
    invalid_value = '<|"|>not-json<|"|>'
    value_start = invalid.index(invalid_value)
    assert parser.last_schema_error_spans == [(value_start, value_start + len(invalid_value))]
    assert parser.last_schema_error_kinds == ["invalid_parameter_type"]

    calls = parser.parse(
        '<|tool_call>call:submit_result_difficulty_3{result:{framework_analysis:[]}}<tool_call|>'
    )
    assert calls[0].arguments == {"result": {"framework_analysis": []}}
    assert parser.last_schema_errors == []
    assert parser.last_schema_error_spans == []

    assert parser.parse('<|tool_call>call:submit_result_difficulty3{result:{}}<tool_call|>') == []
    assert parser.last_schema_errors == ["unknown tool name 'submit_result_difficulty3'"]
    assert parser.last_schema_error_kinds == ["unknown_tool"]


def test_gemma4_tool_parser_safely_coerces_schema_typed_arguments():
    schema = tool_schema(
        "typed_tool",
        "Typed arguments.",
        {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "enabled": {"type": "boolean"},
            "items": {"type": "array"},
            "payload": {"type": "object"},
            "optional_count": {"type": "integer"},
        },
        ["count", "ratio", "enabled", "items", "payload"],
    )
    parser = make_tool_parser("gemma4", valid_tools={"typed_tool"})
    parser.get_tool_prompt(json.dumps(schema))

    calls = parser.parse(
        '<|tool_call>call:typed_tool{count:<|"|>3<|"|>,ratio:<|"|>1.25e2<|"|>,'
        'enabled:<|"|>true<|"|>,items:<|"|>["a","b"]<|"|>,'
        'payload:<|"|>{"answer":42}<|"|>,optional_count:null}<tool_call|>'
    )

    assert calls[0].arguments == {
        "count": 3,
        "ratio": 125.0,
        "enabled": True,
        "items": ["a", "b"],
        "payload": {"answer": 42},
    }
    assert len(parser.last_schema_coercions) == 6
    assert parser.last_schema_errors == []


@pytest.mark.parametrize(
    "arguments",
    [
        '{count:<|"|>3.5<|"|>,label:<|"|>ok<|"|>}',
        '{count:3,label:[<|"|>not<|"|>,<|"|>scalar<|"|>]}',
        '{count:null,label:<|"|>ok<|"|>}',
    ],
)
def test_gemma4_tool_parser_keeps_ambiguous_type_mismatches_fatal(arguments):
    schema = tool_schema(
        "typed_tool",
        "Typed arguments.",
        {"count": {"type": "integer"}, "label": {"type": "string"}},
        ["count", "label"],
    )
    parser = make_tool_parser("gemma4", valid_tools={"typed_tool"})
    parser.get_tool_prompt(json.dumps(schema))

    assert parser.parse(f"<|tool_call>call:typed_tool{arguments}<tool_call|>") == []
    assert parser.last_schema_error_kinds == ["invalid_parameter_type"]


def test_gemma4_tool_parser_localizes_emitted_schema_errors_but_not_omissions():
    schema = tool_schema(
        "finish",
        "Finish.",
        {"command": {"type": "string"}, "result": {"type": "object"}},
        ["command", "result"],
    )
    schema["function"]["parameters"]["additionalProperties"] = False
    parser = make_tool_parser("gemma4", valid_tools={"finish"})
    parser.get_tool_prompt(json.dumps(schema))

    unknown_parameter = (
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,'
        'extra:1,result:{answer:<|"|>42<|"|>}}<tool_call|>'
    )
    assert parser.parse(unknown_parameter) == []
    extra_start = unknown_parameter.index("extra")
    assert parser.last_schema_error_spans == [(extra_start, extra_start + len("extra"))]
    assert parser.last_schema_error_kinds == ["unknown_parameter"]

    missing_parameter = '<|tool_call>call:finish{command:<|"|>submit<|"|>}<tool_call|>'
    assert parser.parse(missing_parameter) == []
    assert parser.last_schema_error_spans == [None]
    assert parser.last_schema_error_kinds == ["missing_parameter"]


def test_gemma4_tool_parser_preserves_official_grep_extension_arguments():
    schema = tool_schema(
        "Grep",
        "Search file contents using regex patterns.",
        {
            "glob": {"type": "string"},
            "path": {"type": "string"},
            "pattern": {"type": "string"},
        },
        ["pattern"],
    )
    parser = make_tool_parser("gemma4", valid_tools={"Grep"})
    parser.get_tool_prompt(json.dumps(schema))
    response = (
        '<|tool_call>call:Grep{-i:true,-n:true,output_mode:<|"|>content<|"|>,'
        'pattern:<|"|>TODO<|"|>}<tool_call|>'
    )

    calls = parser.parse(response)

    assert calls[0].arguments == {
        "-i": True,
        "-n": True,
        "output_mode": "content",
        "pattern": "TODO",
    }
    assert parser.last_schema_errors == []


def test_gemma4_tool_parser_validates_schema_valued_additional_properties():
    schema = tool_schema("labels", "Attach labels.", {"target": {"type": "string"}}, ["target"])
    schema["function"]["parameters"]["additionalProperties"] = {"type": "integer"}
    parser = make_tool_parser("gemma4", valid_tools={"labels"})
    prompt = parser.get_tool_prompt(json.dumps(schema))

    assert "parameters:{additionalProperties:{type:" in prompt or ",additionalProperties:{type:" in prompt

    calls = parser.parse(
        '<|tool_call>call:labels{target:<|"|>pod<|"|>,priority:2}<tool_call|>'
    )
    assert calls[0].arguments == {"target": "pod", "priority": 2}
    assert parser.last_schema_errors == []

    assert parser.parse(
        '<|tool_call>call:labels{target:<|"|>pod<|"|>,priority:<|"|>high<|"|>}<tool_call|>'
    ) == []
    assert parser.last_schema_error_kinds == ["invalid_parameter_type"]


def test_gemma4_tool_parser_localizes_structured_result_syntax_error():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})
    parser.get_tool_prompt(json.dumps(finish_schema(structured_result=True)))
    malformed = (
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,'
        'result:{items:[{value:1}{value:2}]}}<tool_call|>'
    )

    assert parser.parse(malformed) == []
    assert parser.last_schema_error_kinds == ["invalid_syntax"]
    start, end = parser.last_schema_error_spans[0]
    assert malformed[start:end] == "{"


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
    assert 'maybe:{description:<|"|>Optional text<|"|>,type:[<|"|>STRING<|"|>,<|"|>NULL<|"|>]}' in declaration
    assert ('nums:{description:<|"|>Numbers<|"|>,default:[1,2],items:{minimum:1,type:<|"|>INTEGER<|"|>},' 'minItems:1,maxItems:4,type:<|"|>ARRAY<|"|>}') in declaration
    assert 'pair:{items:[{type:<|"|>STRING<|"|>},{type:<|"|>INTEGER<|"|>}],type:<|"|>ARRAY<|"|>}' in declaration
    assert 'mode:{anyOf:[{enum:[<|"|>fast<|"|>,<|"|>safe<|"|>],type:<|"|>STRING<|"|>},{type:<|"|>NULL<|"|>}]}' in declaration
    assert 'payload:{additionalProperties:{type:<|"|>STRING<|"|>},type:<|"|>OBJECT<|"|>}' in declaration
    assert 'profile:{$ref:<|"|>#/$defs/Profile<|"|>}' in declaration
    assert 'sealed:{additionalProperties:false,type:<|"|>OBJECT<|"|>}' in declaration
    assert ('token:{const:<|"|>ok<|"|>,$defs:{Alias:{description:<|"|>Short name<|"|>,type:<|"|>STRING<|"|>}},' 'pattern:<|"|>^[a-z]+$<|"|>,minLength:2,maxLength:8,type:<|"|>STRING<|"|>}') in declaration
    assert ('$defs:{Profile:{properties:{age:{type:<|"|>INTEGER<|"|>},name:{type:<|"|>STRING<|"|>}},' 'required:[<|"|>name<|"|>],type:<|"|>OBJECT<|"|>}}') in declaration
    assert 'definitions:{<|"|>legacy type<|"|>:{enum:[<|"|>old<|"|>],type:<|"|>STRING<|"|>}}' in declaration
    assert "['STRING', 'NULL']" not in declaration


def test_gemma4_function_declaration_preserves_response_schema():
    schema = tool_schema("lookup", "Look up one item.", {}, [])
    schema["function"]["response"] = {
        "type": "object",
        "description": "Lookup result.",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    declaration = Gemma4ToolParser._format_function_declaration(schema)

    assert 'response:{description:<|"|>Lookup result.<|"|>' in declaration
    assert 'properties:{value:{type:<|"|>STRING<|"|>}}' in declaration
    assert 'required:[<|"|>value<|"|>]' in declaration


def test_gemma4_tool_parser_parses_native_calls_and_formats_observation():
    parser = make_tool_parser("gemma4", valid_tools={"web_search", "finish", "submit"})

    calls = parser.parse('<|tool_call>call:web_search{query:<|"|>Tokyo weather<|"|>,max_results:3}<tool_call|><|tool_response>')

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


def test_gemma4_tool_parser_handles_nested_arguments_with_exact_finish_name():
    parser = make_tool_parser("gemma_4", valid_tools={"finish"})

    calls = parser.parse('<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{answer:<|"|>42<|"|>,sources:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>')

    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "submit", "result": {"answer": "42", "sources": ["a", "b"]}}


def test_gemma4_tool_parser_does_not_alias_tool_names():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    assert parser.parse('<|tool_call>call:submit{command:<|"|>submit<|"|>,result:<|"|>42<|"|>}<tool_call|>') == []
    assert parser.last_schema_errors == ["unknown tool name 'submit'"]


def test_gemma4_tool_parser_accepts_python_style_bare_literals():
    parser = make_tool_parser("gemma4", valid_tools={"set_flags"})

    calls = parser.parse('<|tool_call>call:set_flags{enabled:True,disabled:False,missing:None,quoted:"True"}<tool_call|>')

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

    calls = parser.parse('<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>The answer is \\boxed{Chacruna}. {"ok": true}<|"|>}<tool_call|><|tool_response>')

    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {
        "command": "submit",
        "result": 'The answer is \\boxed{Chacruna}. {"ok": true}',
    }
    assert calls[0].end == calls[0].start + len('<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>The answer is \\boxed{Chacruna}. {"ok": true}<|"|>}<tool_call|>')


def test_gemma4_tool_parser_parses_logged_finish_call():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    calls = parser.parse('<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>Chacruna<|"|>}<tool_call|><|tool_response>')

    assert len(calls) == 1
    assert calls[0].name == "finish"
    assert calls[0].arguments == {"command": "submit", "result": "Chacruna"}


@pytest.mark.parametrize(
    "response",
    [
        '<|tool_call>call:finish{command:<|"|>submit,result:<|"|>answer<|"|>}<tool_call|><|tool_response>',
        '<|tool_call>call:finish{command:<|"|>finish,result:<|"|>answer<|"|>}<tool_call|><|tool_response>',
        '<|tool_call>call:finish{command:<|"|>finish(result="unable")"}<tool_call|>',
        '<|tool_call>call:finish{command:<|"|>finish\\nanswer}<tool_call|>',
        '<|tool_call>call:finish{command:<|"|>finish{result:<|"|>answer<|"|>}}<tool_call|>',
        '<|tool_call>call:finish{command:<|"|>submit,{"answer":"answer"}}<tool_call|>',
        '<|tool_call>call:finish{command:<|"|>submit,{"answer":"answer}<tool_call|>',
        '<|tool_call>call:web_search{query:<|"|>query"""}<tool_call|>',
    ],
)
def test_gemma4_tool_parser_rejects_malformed_calls_without_repair(response):
    parser = make_tool_parser("gemma4", valid_tools={"finish", "web_search"})
    parser.get_tool_prompt(
        "\n".join(json.dumps(tool, ensure_ascii=False) for tool in [web_search_schema(), finish_schema()])
    )

    assert parser.parse(response) == []
    assert parser.last_schema_errors


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


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            '{query<|"|>CFD simulation hypersonic flow extrusion algorithm normal spacing<|"|>}',
            {"query": "CFD simulation hypersonic flow extrusion algorithm normal spacing"},
        ),
        (
            '{query:<|"|>"The Database of Religious History" "Open Scholarship in Practice"<|"|>]}',
            {"query": '"The Database of Religious History" "Open Scholarship in Practice"'},
        ),
        (
            '{query:"Religious Research Association" "The Database of Religious History" '
            '"Open Scholarship in Practice" 2019}',
            {
                "query": 'Religious Research Association" "The Database of Religious History" '
                '"Open Scholarship in Practice" 2019'
            },
        ),
        (
            '{query="patent US20170027168A1 optical pitch tuning range"}',
            {"query": "patent US20170027168A1 optical pitch tuning range"},
        ),
        (
            '{query:"layer 7 cortical interface stroke" max_results:3}',
            {"query": "layer 7 cortical interface stroke", "max_results": 3},
        ),
        (
            '{query: "taro cultivars clinical trial NCT03196219 JFT Ganança" '
            'specific characteristic<|"|>}',
            {"query": 'taro cultivars clinical trial NCT03196219 JFT Ganança" specific characteristic'},
        ),
    ],
)
def test_gemma4_tool_parser_accepts_logged_string_syntax_variants(arguments, expected):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse(f"<|tool_call>call:web_search{arguments}<tool_call|><|tool_response>")

    assert len(calls) == 1
    assert calls[0].arguments == expected
    assert parser.last_schema_errors == []


def test_gemma4_web_search_accepts_terminal_native_query_without_close_marker():
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))
    response = (
        '<|tool_call>call:web_search{query:<|"|>Virú Valley Gallinazo style stylized fish adobe motif '
        'PIC 2016 5590 1982 pages 337-338**}<tool_call|>'
    )

    calls = parser.parse(response)

    assert len(calls) == 1
    assert calls[0].arguments == {
        "query": "Virú Valley Gallinazo style stylized fish adobe motif PIC 2016 5590 1982 pages 337-338**"
    }
    assert parser.last_schema_errors == []


@pytest.mark.parametrize(
    ("response", "repairs"),
    [
        (
            '<|tool_call>call:web_search{query:<|"|>deterministic query<tool_call|>',
            ["missing_native_string_end", "missing_argument_object_end"],
        ),
        (
            '<|tool_call>call:web_search{query:<|"|>deterministic query<|"|><tool_call|>',
            ["missing_argument_object_end"],
        ),
        (
            '<|tool_call>call:web_search{query:"deterministic query<tool_call|>',
            ["missing_quoted_string_end", "missing_argument_object_end"],
        ),
        (
            '<|tool_call>call:web_search{query:<|"|>deterministic query<|"|>}<|tool_response>',
            ["missing_tool_call_end"],
        ),
    ],
)
def test_gemma4_tool_parser_repairs_only_deterministic_call_closures(response, repairs):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse(response)

    assert calls[0].arguments == {"query": "deterministic query"}
    assert parser.last_syntax_repairs == repairs
    assert parser.last_schema_errors == []


@pytest.mark.parametrize(
    ("response", "expected_result", "expected_repairs"),
    [
        (
            '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{items:[1,2]',
            {"items": [1, 2]},
            ["missing_argument_closers", "missing_tool_call_end"],
        ),
        (
            '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{done:true,},}<tool_call|>',
            {"done": True},
            ["trailing_comma"],
        ),
        (
            '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>{"done":true}<|"|>}<tool_call|>',
            {"done": True},
            ["json_string_result"],
        ),
        (
            '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{done:true}}])<tool_call|>',
            {"done": True},
            ["extra_wrapper_closer"],
        ),
    ],
)
def test_gemma4_finish_shadow_repairs_are_deterministic(response, expected_result, expected_repairs):
    parser = make_tool_parser("gemma4", valid_tools={"finish"})
    parser.get_tool_prompt(json.dumps(finish_schema(result_schema={"type": "object"}), ensure_ascii=False))

    calls = parser.parse(response)
    shadow = parser.repair_finish_shadow(response, calls)

    assert shadow is not None
    assert shadow.arguments["result"] == expected_result
    assert parser.last_shadow_finish_repairs == expected_repairs


@pytest.mark.parametrize(
    ("response", "expected_result", "repair"),
    [
        ('{"done":true}', {"done": True}, "bare_json"),
        ('<answer>{"done":true}</answer>', {"done": True}, "answer_json"),
        ('{"done":true,}', {"done": True}, "bare_json"),
        ('{"done":true}}]', {"done": True}, "bare_json"),
    ],
)
def test_finish_shadow_accepts_bounded_json_wrappers(response, expected_result, repair):
    parser = QwenToolParser(valid_tools={"finish"})

    calls = parser.parse(response)
    shadow = parser.repair_finish_shadow(response, calls)

    assert shadow is not None
    assert shadow.arguments["result"] == expected_result
    assert parser.last_shadow_finish_repairs[0] == repair


@pytest.mark.parametrize("response", ['{"items":[1}}', '{"done": tru}', '<answer>{"done":true} trailing'])
def test_finish_shadow_rejects_ambiguous_json(response):
    parser = QwenToolParser(valid_tools={"finish"})

    assert parser.repair_finish_shadow(response, parser.parse(response)) is None


@pytest.mark.parametrize(
    "response",
    [
        '<|tool_call>call:web_search{query:<|"|>truncated without a boundary',
        '<|tool_call>call:web_search{query:<|"|>ambiguous """ query}<tool_call|>',
        '<|tool_call>call:web_search{query:<|"|>query<|"|>,payload:{items:[1,2]<tool_call|>',
        '<|tool_call>call:web_search{query:<|"|>query<|"|>} trailing prose',
    ],
)
def test_gemma4_tool_parser_does_not_repair_ambiguous_truncation(response):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    assert parser.parse(response) == []
    assert parser.last_syntax_repairs == []


@pytest.mark.parametrize(
    ("arguments", "expected_query", "expected_max_results"),
    [
        (
            '{max_results:3,query:"bacteria ethylbenzene growth inhibition continuous bioreactor" '
            '"merouani" phd}',
            'bacteria ethylbenzene growth inhibition continuous bioreactor" "merouani" phd',
            3,
        ),
        (
            '{query: "Emerg. Infect. Dis. 17:2099" genomic island recA ctxAB IncC}',
            'Emerg. Infect. Dis. 17:2099" genomic island recA ctxAB IncC',
            None,
        ),
        (
            '{query:"""Nigunim" "New Music Buff" "The WholeNote" "Volume 27 Issue 6" 2017 violin<|"|>}',
            '""Nigunim" "New Music Buff" "The WholeNote" "Volume 27 Issue 6" 2017 violin',
            None,
        ),
        (
            '{query:<|"|>New Music Buff "Nigunim" violin 2017 WholeNote review<|"|>]}',
            'New Music Buff "Nigunim" violin 2017 WholeNote review',
            None,
        ),
        (
            '{query:\\"bioprinting\\" "Bingham plastic" gelatin "direct extrusion" '
            '"version 2.0" 2025 throughput}',
            '\\"bioprinting\\" "Bingham plastic" gelatin "direct extrusion" "version 2.0" 2025 throughput',
            None,
        ),
        (
            '{query:"The Database of Religious History" 2023 lecture series '
            '"Open Scholarship in Practice" October 2019}',
            'The Database of Religious History" 2023 lecture series "Open Scholarship in Practice" October 2019',
            None,
        ),
        (
            '{query:<|"|>"Tom Butcher" "optical joint transform correlation" '
            "Austronesian root 'deflate'</strong>}",
            '"Tom Butcher" "optical joint transform correlation" Austronesian root \'deflate\'</strong>',
            None,
        ),
    ],
)
def test_gemma4_web_search_logged_queries_do_not_require_cached_schema(
    arguments, expected_query, expected_max_results
):
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})

    calls = parser.parse(f"<|tool_call>call:web_search{arguments}<tool_call|>")

    assert len(calls) == 1
    assert calls[0].arguments["query"] == expected_query
    if expected_max_results is not None:
        assert calls[0].arguments["max_results"] == expected_max_results
    assert parser.last_schema_errors == []


def test_gemma4_schema_string_preserves_bare_brackets():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})
    parser.get_tool_prompt(json.dumps(finish_schema(), ensure_ascii=False))

    calls = parser.parse(
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:JSONArray[]}<tool_call|>'
    )

    assert len(calls) == 1
    assert calls[0].arguments == {"command": "submit", "result": "JSONArray[]"}
    assert parser.last_schema_errors == []


def test_gemma4_tool_parser_still_rejects_unclosed_native_finish_result():
    parser = make_tool_parser("gemma4", valid_tools={"finish"})
    parser.get_tool_prompt(json.dumps(finish_schema(), ensure_ascii=False))
    response = (
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:<|"|>'
        '{"error":"Analysis incomplete due to missing data."}}<tool_call|>'
    )

    assert parser.parse(response) == []
    assert parser.last_schema_errors


def test_gemma4_tool_parser_decodes_json_like_unicode_escapes():
    parser = make_tool_parser("gemma4", valid_tools={"echo"})

    calls = parser.parse('<|tool_call>call:echo{text:"\\u003cscript\\u003e",emoji:"\\ud83d\\ude00"}<tool_call|>')

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
    parser = make_tool_parser("gemma4", valid_tools={"finish"})

    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.parser"):
        calls = parser.parse('<|tool_call>call:finish{command:"submit",result:"unterminated}<tool_call|>')

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
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert _policy_masked_text(sample) == bad


def test_gemma4_malformed_closed_tool_call_terminates_as_parser_error():
    bad = '<|tool_call>call:finish{command:"submit",result:"unterminated}<tool_call|>'

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
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert _policy_masked_text(sample) == bad


@pytest.mark.parametrize(
    ("bad", "penalized_text", "error_kind"),
    [
        (
            '<|tool_call>call:unknown_tool{value:<|"|>bad<|"|>}<tool_call|>',
            "unknown_tool",
            "unknown_tool",
        ),
        (
            '<|tool_call>call:echo{value:{bad:true}}<tool_call|>',
            "{bad:true}",
            "invalid_parameter_type",
        ),
    ],
)
def test_gemma4_parser_error_credit_assignment_penalizes_only_localized_schema_token(
    tmp_path: Path,
    bad: str,
    penalized_text: str,
    error_kind: str,
):
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger a localized schema error"),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert sample.metadata["tool_parser_error_kinds"] == [error_kind]
    assert _policy_masked_text(sample) == penalized_text


def test_gemma4_missing_required_parameter_penalizes_terminal_action(tmp_path: Path):
    bad = '<|tool_call>call:echo{}<tool_call|>'

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger a missing parameter"),
        [{"text": bad}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert sample.metadata["tool_parser_error_kinds"] == ["missing_parameter"]
    assert _policy_masked_text(sample) == bad


def test_gemma4_extension_parameter_is_preserved_by_parser_and_safely_dispatched(tmp_path: Path):
    unknown = '<|tool_call>call:echo{value:<|"|>good<|"|>,unknown_parameter:<|"|>bad<|"|>}<tool_call|>'
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Recover from an optional parameter"),
        [{"text": unknown}, {"text": _gemma4_finish_call()}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "env_done"
    assert "tool_parser_error_recoverable" not in sample.metadata
    first_step = sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]
    assert "unknown_parameter" in first_step["action"]


def test_gemma4_deterministic_syntax_repair_continues_rollout(tmp_path: Path):
    missing_call_end = '<|tool_call>call:echo{value:<|"|>good<|"|>}<|tool_response>'
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Recover a deterministic call boundary"),
        [{"text": missing_call_end}, {"text": _gemma4_finish_call()}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["tool_parser_syntax_repairs"] == ["missing_tool_call_end"]
    assert sample.metadata["credit_assignment_event"] is None


def test_mcp_finish_shadow_repair_preserves_policy_output_and_masks_turn(tmp_path: Path):
    evidence_call = _gemma4_echo_call("evidence")
    raw_response = '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{done:true}'
    calls = [{"text": evidence_call}, {"text": raw_response}]
    sample_spec = _local_mcp_sample(tmp_path, question="Repair the final answer only in the executor")
    sample_spec.metadata["answer_schema"] = {
        "type": "object",
        "properties": {"done": {"const": True}},
        "required": ["done"],
        "additionalProperties": False,
    }

    result = _run_generate_with_fake_sglang(
        sample_spec,
        calls,
        {
            "SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false",
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert calls == []
    sample = result[0] if isinstance(result, list) else result
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["tool_parser_shadow_finish_repairs"] == [
        "missing_argument_object_end",
        "missing_tool_call_end",
    ]
    assert sample.metadata["tool_parser_shadow_finish_policy_masked"] is True
    assert _response_text(sample).endswith(raw_response)
    assert sample.tokens[-len(raw_response) :] == [ord(char) for char in raw_response]
    assert _policy_masked_text(sample) == evidence_call
    assert _policy_unmasked_text(sample).endswith(raw_response)
    policy_mask = sample.policy_loss_mask if sample.policy_loss_mask is not None else sample.loss_mask
    assert policy_mask[-len(raw_response) :] == [0] * len(raw_response)
    episode_step = sample.metadata["rllm_episode"]["trajectories"][0]["steps"][1]
    assert episode_step["model_response"] == raw_response
    assert episode_step["chat_completions"][-1]["content"] == raw_response


@pytest.mark.parametrize(
    ("raw_response", "repair"),
    [
        ('{"done":true}', "bare_json"),
        ('<answer>{"done":true}</answer>', "answer_json"),
    ],
)
def test_mcp_json_wrapper_shadow_executes_without_rewriting_policy_output(
    tmp_path: Path,
    raw_response: str,
    repair: str,
):
    evidence_call = _echo_call("evidence")
    calls = [{"text": evidence_call}, {"text": raw_response}]
    sample_spec = _local_mcp_sample(tmp_path, question="Accept a bounded JSON final wrapper")
    sample_spec.metadata["answer_schema"] = {
        "type": "object",
        "properties": {"done": {"const": True}},
        "required": ["done"],
    }

    result = _run_generate_with_fake_sglang(
        sample_spec,
        calls,
        {
            "SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
            "FUSED_DISABLE_THINKING": "True",
        },
    )

    assert calls == []
    sample = result[0] if isinstance(result, list) else result
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["tool_parser_shadow_finish_repairs"][0] == repair
    assert sample.tokens[-len(raw_response) :] == [ord(char) for char in raw_response]
    policy_mask = sample.policy_loss_mask if sample.policy_loss_mask is not None else sample.loss_mask
    assert policy_mask[-len(raw_response) :] == [0] * len(raw_response)
    final_step = sample.metadata["rllm_episode"]["trajectories"][0]["steps"][1]
    assert final_step["model_response"] == raw_response
    assert final_step["chat_completions"][-1]["content"] == raw_response


def test_mcp_finish_shadow_repair_requires_original_answer_schema(tmp_path: Path):
    raw_response = '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{done:<|"|>yes<|"|>}'
    calls = [{"text": raw_response}]
    sample_spec = _local_mcp_sample(tmp_path, question="Reject a schema-invalid shadow answer")
    sample_spec.metadata["answer_schema"] = {
        "type": "object",
        "properties": {"done": {"type": "boolean"}},
        "required": ["done"],
    }

    result = _run_generate_with_fake_sglang(
        sample_spec,
        calls,
        {
            "SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    assert calls == []
    sample = result[0] if isinstance(result, list) else result
    assert sample.reward == 0.0
    assert sample.metadata["fused_termination"] == "ABNORMAL_PARSE_ERROR"
    assert sample.metadata["tool_parser_error_kinds"] == ["finish_answer_schema"]
    assert "tool_parser_shadow_finish_repairs" not in sample.metadata


def test_mcp_valid_finish_remains_policy_trainable(tmp_path: Path):
    response = '<|tool_call>call:finish{command:<|"|>submit<|"|>,result:{done:true}}<tool_call|>'
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Submit a valid final answer"),
        [{"text": _gemma4_echo_call("evidence")}, {"text": response}],
        {
            "SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
            "FUSED_DISABLE_THINKING": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0] if isinstance(result, list) else result
    assert sample.reward == 1.0
    assert _policy_masked_text(sample).endswith(response)
    assert "tool_parser_shadow_finish_repairs" not in sample.metadata


def test_gemma4_web_search_recovers_terminal_missing_ordinary_quote():
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse(
        '<|tool_call>call:web_search{query:"research paper optical correlation}<tool_call|><|tool_response>'
    )

    assert len(calls) == 1
    assert calls[0].arguments["query"] == "research paper optical correlation"
    assert parser.last_schema_errors == []


def test_gemma4_tool_parser_accepts_logged_channel_call_wrapper():
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse(
        '<|channel>call:web_search{query:<|"|>Cheirolepidiaceae cone phyllotaxy<|"|>}'
        '<tool_call|><|tool_response>'
    )

    assert len(calls) == 1
    assert calls[0].arguments == {"query": "Cheirolepidiaceae cone phyllotaxy"}
    assert parser.last_schema_errors == []
    assert fused_generate._response_has_malformed_tool_call(
        '<|channel>call:web_search{query:<|"|>Cheirolepidiaceae cone phyllotaxy<|"|>}'
        '<tool_call|><|tool_response>'
    ) is False


def test_gemma4_tool_parser_ignores_extra_wrapper_closer_after_argument_object():
    parser = make_tool_parser("gemma4", valid_tools={"web_search"})
    parser.get_tool_prompt(json.dumps(web_search_schema(), ensure_ascii=False))

    calls = parser.parse(
        '<|tool_call>call:web_search{query:<|"|>ethylbenzene degradation bacteria<|"|>}]'
        '<tool_call|><|tool_response>'
    )

    assert len(calls) == 1
    assert calls[0].arguments == {"query": "ethylbenzene degradation bacteria"}
    assert parser.last_schema_errors == []
    assert fused_generate._response_has_malformed_tool_call(
        '<|tool_call>call:web_search{query:<|"|>ethylbenzene degradation bacteria<|"|>}]'
        '<tool_call|><|tool_response>'
    ) is False


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
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert _policy_masked_text(sample) == malformed
    assert "before-error" not in sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]["observation"]


def test_gemma4_valid_call_then_ambiguous_nested_truncation_terminates_as_parser_error(tmp_path: Path):
    valid = _gemma4_echo_call("before-error")
    malformed = '<|tool_call>call:echo{value:{nested:[1,2}<tool_call|>'
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
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert _policy_masked_text(sample)
    assert _policy_masked_text(sample) in malformed
    assert "before-error" not in sample.metadata["rllm_episode"]["trajectories"][0]["steps"][0]["observation"]


def test_tito_model_type_detects_gemma4_without_changing_qwen_detection():
    assert fused_generate._tito_model_type("/share/nlp/share/plm/gemma-4-E2B-it") == "gemma4"
    assert fused_generate._tito_model_type("/models/Qwen3.5-4B") == "qwen3_5"
    assert fused_generate._tito_model_type("/models/Qwen3-4B") == "qwen3"


def test_gemma4_tito_delta_does_not_duplicate_generated_tool_response_handoff():
    tokenizer = FakeGemma4Tokenizer()
    tools = [web_search_schema(), finish_schema()]
    action = '<|tool_call>call:web_search{query:<|"|>evidence<|"|>}<tool_call|><|tool_response>'
    old_messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": action},
    ]
    new_messages = [
        old_messages[0],
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "evidence"}}}],
            "tool_responses": [{"name": "web_search", "response": {"value": "result"}}],
        },
    ]
    prefix_ids = tokenizer.encode(action, add_special_tokens=False)

    delta_ids = fused_generate._render_gemma4_tito_delta_ids(
        tokenizer,
        old_messages,
        new_messages,
        prefix_ids,
        tools=tools,
        disable_thinking=True,
    )
    continuation = tokenizer.decode(delta_ids, skip_special_tokens=False)

    assert continuation == "<turn|>\n<|turn>tool\nresult<turn|>\n<|turn>model\n"


def test_gemma4_tito_delta_keeps_tool_response_handoff_when_generation_omits_it():
    tokenizer = FakeGemma4Tokenizer()
    tools = [web_search_schema(), finish_schema()]
    action = '<|tool_call>call:web_search{query:<|"|>evidence<|"|>}<tool_call|>'
    old_messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": action},
    ]
    new_messages = [
        old_messages[0],
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "evidence"}}}],
            "tool_responses": [{"name": "web_search", "response": {"value": "result"}}],
        },
    ]
    prefix_ids = tokenizer.encode(action, add_special_tokens=False)

    delta_ids = fused_generate._render_gemma4_tito_delta_ids(
        tokenizer,
        old_messages,
        new_messages,
        prefix_ids,
        tools=tools,
        disable_thinking=True,
    )
    continuation = tokenizer.decode(delta_ids, skip_special_tokens=False)

    assert continuation == "<turn|>\n<|turn>tool\nresult<turn|>\n<|turn>model\n"


def test_gemma4_tito_delta_does_not_duplicate_generated_turn_end():
    tokenizer = FakeGemma4Tokenizer()
    action = '<|tool_call>call:web_search{query:<|"|>evidence<|"|>}<tool_call|><turn|>'
    old_messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": action.removesuffix("<turn|>")},
    ]
    new_messages = [
        old_messages[0],
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "web_search", "arguments": {"query": "evidence"}}}],
            "tool_responses": [{"name": "web_search", "response": {"value": "result"}}],
        },
    ]

    delta_ids = _render_gemma4_tito_delta_ids(
        tokenizer,
        old_messages,
        new_messages,
        tokenizer.encode(action, add_special_tokens=False),
        tools=None,
        disable_thinking=True,
    )

    assert tokenizer.decode(delta_ids) == "\n<|turn>tool\nresult<turn|>\n<|turn>model\n"


def test_build_system_prompt_switches_tool_format_by_model():
    tools = [web_search_schema(), finish_schema()]

    coder = build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, "/share/nlp/share/plm/Qwen3.5-4B")
    assert "<function=FUNCTION_NAME>" in coder
    assert "<parameter=PARAMETER_NAME>" in coder

    legacy = build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, "/share/nlp/share/plm/Qwen3-4B")
    assert "<function=FUNCTION_NAME>" not in legacy
    assert '{"name": <function-name>, "arguments": <args-json-object>}' in legacy

    gemma4 = build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, "/share/nlp/share/plm/gemma-4-E2B-it")
    assert gemma4.startswith(FUSED_SEARCH_SYSTEM_PROMPT.strip())
    assert "Gemma4 native tool-call contract:" in gemma4
    assert "TOOL_NAME must exactly match one of: finish, web_search" in gemma4
    assert "<|tool>declaration:web_search{" not in gemma4


def test_explicit_model_series_overrides_noncanonical_checkpoint_path(monkeypatch):
    checkpoint = "checkpoints/FusedRL/webqa-dapo-q3.5-4b/iter_0000019_hf"
    tools = [web_search_schema(), finish_schema()]

    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen3.5")
    assert isinstance(make_tool_parser(checkpoint), Qwen3CoderToolParser)
    assert "<function=FUNCTION_NAME>" in build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, checkpoint)

    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen3")
    assert type(make_tool_parser(checkpoint)) is QwenToolParser
    assert '{"name": <function-name>, "arguments": <args-json-object>}' in build_system_prompt(FUSED_SEARCH_SYSTEM_PROMPT, tools, checkpoint)


def test_invalid_explicit_model_series_fails_closed(monkeypatch):
    monkeypatch.setenv("FUSED_MODEL_SERIES", "qwen-next")
    with pytest.raises(ValueError, match="Unsupported FUSED_MODEL_SERIES"):
        make_tool_parser("/models/Qwen3-4B")


def test_qwen35_fused_sampling_uses_model_specific_parameters():
    sampling_params_seen = []

    _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="answer", metadata={"question": "Answer directly"}),
        [{"text": "answer"}],
        {"FUSED_HARNESS": "cot", "FUSED_MODEL_SERIES": "qwen3.5"},
        evaluation=True,
        sampling_params_seen=sampling_params_seen,
    )

    assert sampling_params_seen == [
        {
            "max_new_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
    ]


def test_qwen3_fused_sampling_keeps_existing_parameters():
    sampling_params_seen = []

    _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="answer", metadata={"question": "Answer directly"}),
        [{"text": "answer"}],
        {"FUSED_HARNESS": "cot", "FUSED_MODEL_SERIES": "qwen3"},
        evaluation=True,
        sampling_params_seen=sampling_params_seen,
    )

    assert sampling_params_seen == [{"max_new_tokens": 1024, "temperature": 1.0, "top_p": 1.0}]


def test_qwen35_training_sampling_keeps_training_parameters():
    sampling_params_seen = []

    _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="answer", metadata={"question": "Answer directly"}),
        [{"text": "answer"}],
        {"FUSED_HARNESS": "cot", "FUSED_MODEL_SERIES": "qwen3.5"},
        sampling_params_seen=sampling_params_seen,
    )

    assert sampling_params_seen == [{"max_new_tokens": 1024, "temperature": 1.0, "top_p": 1.0}]


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
    evidence = tools["get_value"]()
    return {"passed": answer == evidence}
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


def test_local_mcp_finish_uses_explicit_dataset_answer_schema(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Submit result")
def submit_result_difficulty_2(result: dict) -> dict:
    return result
""",
        encoding="utf-8",
    )
    answer_schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    env = FusedEnvironment(
        {
            "question": "Return the answer",
            "difficulty": 2,
            "tools_py": str(asset / "tools.py"),
            "answer_schema_json": json.dumps(answer_schema),
        }
    )

    finish = next(schema for schema in env.tools() if schema["function"]["name"] == "finish")
    result_schema = finish["function"]["parameters"]["properties"]["result"]
    assert result_schema == {**answer_schema, "description": "Final answer or JSON value."}
    env.close()


def test_local_mcp_enabled_tools_limits_schema_and_model_execution(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool()
def public_lookup() -> dict:
    return {"value": "public"}

@mcp.tool()
def verifier_only_lookup() -> dict:
    return {"value": "private"}
""",
        encoding="utf-8",
    )
    env = FusedEnvironment(
        {
            "question": "Use the public lookup",
            "tools_py": str(asset / "tools.py"),
            "enabled_tools": ["public_lookup"],
        }
    )

    assert _valid_tool_names(env.tools()) == {"public_lookup", "finish"}
    assert json.loads(env.mcp_tools.call("public_lookup", {})) == {"value": "public"}
    assert env.mcp_tools.call("verifier_only_lookup", {}) == (
        "Error: local MCP tool verifier_only_lookup is not enabled for this task"
    )
    # Verifiers retain access to hidden evidence helpers without exposing them to the model.
    assert env.mcp_tools["verifier_only_lookup"]() == {"value": "private"}


def test_local_mcp_invalid_enabled_tool_fails_fast(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def lookup(): return {'ok': True}\n",
        encoding="utf-8",
    )
    env = FusedEnvironment(
        {"question": "Lookup", "tools_py": str(asset / "tools.py"), "enabled_tools": ["missing"]}
    )

    _observation, info = env.reset()

    assert "env_error" in info
    assert "missing" in info["env_error"]


def test_local_mcp_compact_finish_keeps_full_validation_and_allows_repair(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def lookup(): return {'ok': True}\n",
        encoding="utf-8",
    )
    answer_schema = {
        "type": "object",
        "description": "Verbose top-level annotation",
        "properties": {
            "grade": {
                "type": "string",
                "enum": ["C-10", "C-9"],
                "description": "Verbose field annotation",
            }
        },
        "required": ["grade"],
    }
    env = FusedEnvironment(
        {
            "question": "Grade the item",
            "tools_py": str(asset / "tools.py"),
            "answer_schema": answer_schema,
            "mcp_compact_finish_schema": True,
        }
    )
    finish = next(schema for schema in env.tools() if schema["function"]["name"] == "finish")
    parameters = finish["function"]["parameters"]

    assert parameters["required"] == ["result"]
    assert "command" not in parameters["properties"]
    assert parameters["properties"]["result"] == {
        "type": "object",
        "properties": {"grade": {"type": "string", "enum": ["C-10", "C-9"]}},
        "required": ["grade"],
        "description": "Final answer or JSON value.",
    }

    error, reward, done, info = asyncio.run(env.step(ToolCall("finish", {"result": {"grade": "C-6"}})))
    assert done is False
    assert reward == 0.0
    assert "result.grade" in error
    assert info["tools/finish_validation_error"] == 1

    _observation, _reward, done, _info = asyncio.run(
        env.step(ToolCall("finish", {"result": {"grade": "C-10"}}))
    )
    assert done is True


def test_local_mcp_tool_groups_load_monotonically_and_refresh_parser(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool()
def catalog() -> list:
    return ["item"]

@mcp.tool()
def inspect_item() -> dict:
    return {"condition": "good"}
""",
        encoding="utf-8",
    )
    env = FusedEnvironment(
        {
            "question": "Inspect the item",
            "tools_py": str(asset / "tools.py"),
            "enabled_tools": ["catalog", "inspect_item"],
            "initial_tools": ["catalog"],
            "tool_groups": {
                "inspection": {
                    "description": "Inspect item condition",
                    "tools": ["inspect_item"],
                }
            },
        }
    )
    tools = env.tools()
    parser = QwenToolParser(valid_tools=_valid_tool_names(tools))

    assert _valid_tool_names(tools) == {"catalog", "load_tool_group", "finish"}
    assert env.mcp_tools.call("inspect_item", {}) == (
        "Error: local MCP tool inspect_item is not enabled for this task"
    )

    observation, _reward, done, info = asyncio.run(
        env.step(ToolCall("load_tool_group", {"group": "inspection"}))
    )
    declarations = fused_generate._refresh_lazy_mcp_tools(env, tools, parser, info)

    assert done is False
    assert info["tools/schema_changed"] == 1
    assert "inspect_item" in observation
    assert "inspect_item" in declarations
    assert "inspect_item" in _valid_tool_names(tools)
    assert "inspect_item" in parser.valid_tools
    assert json.loads(env.mcp_tools.call("inspect_item", {})) == {"condition": "good"}


def test_local_mcp_lazy_group_executes_new_tool_in_next_rollout_turn(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Load and call echo")
    sample.metadata.update(
        {
            "enabled_tools": ["echo"],
            "tool_groups": {"echo_tools": {"description": "Echo a value", "tools": ["echo"]}},
        }
    )
    load = '<tool_call>{"name":"load_tool_group","arguments":{"group":"echo_tools"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        sample,
        [{"text": load}, {"text": _echo_call("loaded")}, {"text": _finish_call()}],
        {
            "SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false",
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
    )
    samples = result if isinstance(result, list) else [result]

    assert samples[-1].reward == 1.0
    assert samples[-1].metadata["fused_termination"] == "env_done"
    assert samples[-1].metadata["fused_tool_call_turns"] == 2


def test_gemma4_lazy_group_adds_native_declaration_in_system_turn(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Load and call echo")
    sample.metadata.update(
        {
            "enabled_tools": ["echo"],
            "tool_groups": {"echo_tools": {"description": "Echo a value", "tools": ["echo"]}},
        }
    )
    load = '<|tool_call>call:load_tool_group{group:<|"|>echo_tools<|"|>}<tool_call|>'
    prompts = []

    result = _run_generate_with_fake_sglang(
        sample,
        [{"text": load}, {"text": _gemma4_echo_call("loaded")}, {"text": _gemma4_finish_call()}],
        {
            "SLIME_LOCAL_MCP_PROCESS_ISOLATION": "false",
            "FUSED_MODEL_SERIES": "gemma4",
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
        prompt_ids_seen=prompts,
    )
    rendered_prompts = ["".join(chr(token) for token in prompt) for prompt in prompts]

    assert result[-1].reward == 1.0
    assert "declaration:echo{" not in rendered_prompts[0]
    assert "<|turn>system\n<|tool>declaration:echo{" in rendered_prompts[1]
    tool_response = rendered_prompts[1].split("<|turn>tool\n", 1)[1].split("<turn|>", 1)[0]
    assert "declaration:echo{" not in tool_response


def test_local_mcp_loading_group_does_not_satisfy_evidence_requirement(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    sample = _local_mcp_sample(tmp_path, question="Load without retrieving")
    sample.metadata.update(
        {
            "enabled_tools": ["echo"],
            "tool_groups": {"echo_tools": ["echo"]},
        }
    )
    env = FusedEnvironment(sample.metadata)

    asyncio.run(env.step(ToolCall("load_tool_group", {"group": "echo_tools"})))
    _observation, reward, done, info = asyncio.run(
        env.step(ToolCall("finish", {"result": {"done": True}}))
    )

    assert done is True
    assert reward == 0.0
    assert info["reward_debug"]["verifier_skipped"] == "no_tool_calls"
    assert info["reward_debug"]["tool_calls"] == 1
    assert info["reward_debug"]["evidence_tool_calls"] == 0


def test_local_mcp_process_cache_key_includes_tool_policy(tmp_path: Path):
    from slime.rollout.fused_agent.mcp_process_pool import _toolset_cache_key

    tools_py = tmp_path / "tools.py"
    tools_py.write_text("# tool asset\n", encoding="utf-8")
    first = _toolset_cache_key({"tools_py": str(tools_py), "enabled_tools": ["first"]})
    second = _toolset_cache_key({"tools_py": str(tools_py), "enabled_tools": ["second"]})

    assert first != second


def test_local_mcp_process_describe_does_not_reuse_another_allowlist(tmp_path: Path):
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    tools_py = tmp_path / "tools.py"
    tools_py.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def first(): return 1\n"
        "@mcp.tool()\n"
        "def second(): return 2\n",
        encoding="utf-8",
    )
    pool = LocalMCPProcessPool(workers=1)
    try:
        first = pool.describe({"tools_py": str(tools_py), "enabled_tools": ["first"]})
        second = pool.describe({"tools_py": str(tools_py), "enabled_tools": ["second"]})
    finally:
        pool.close()

    schema_names = lambda description: {
        schema["function"]["name"] for schema in description["schemas"]
    }
    assert schema_names(first) == {"first", "finish"}
    assert schema_names(second) == {"second", "finish"}


def test_local_mcp_verifier_results_support_raw_and_legacy_wrapped_access(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Return records")
def get_records() -> list[dict]:
    return [{"title": "Evidence"}]

@mcp.tool(description="Return summary")
def get_summary() -> dict:
    return {"title": "Evidence"}
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Find evidence", "tools_py": str(asset / "tools.py")})

    records = env.mcp_tools["get_records"]()
    summary = env.mcp_tools["get_summary"]()
    assert isinstance(records, list) and records[0]["title"] == "Evidence"
    assert records["result"] is records
    assert records.get("result") is records
    assert isinstance(summary, dict) and summary["title"] == "Evidence"
    assert summary["result"] is summary
    assert summary.get("result") is summary


def test_local_mcp_verifier_toolset_supports_mapping_get(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def lookup(): return {'value': 3}\n",
        encoding="utf-8",
    )
    toolset = FusedEnvironment({"question": "Lookup", "tools_py": str(asset / "tools.py")}).mcp_tools

    assert callable(toolset.get("lookup"))
    assert toolset.get("lookup")() == {"value": 3}
    assert toolset.get("missing") is None
    assert toolset.get("missing", "fallback") == "fallback"
    assert "lookup" in list(toolset)
    toolset.close()


def test_local_mcp_verifier_can_validate_quotes_without_exposing_internal_tool(tmp_path: Path):
    asset = tmp_path / "asset"
    (asset / "data").mkdir(parents=True)
    (asset / "data" / "source.json").write_text(
        json.dumps(
            {
                "title": "Source title",
                "summary": "Document summary.",
                "content": "",
                "sections": [
                    {
                        "heading": "Nested evidence",
                        "text": "The documented requirement is supported by this exact evidence sentence.",
                        "bullets": ["A second nested evidence statement is also available."],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (asset / "data" / "same-title.json").write_text(
        json.dumps(
            {
                "title": "Source title",
                "summary": "A different same-title record must not shadow the matching record.",
                "content": "",
            }
        ),
        encoding="utf-8",
    )
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="List sources")
def list_sources() -> list[str]:
    return ["Source title"]
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Audit evidence", "tools_py": str(asset / "tools.py")})

    exposed_names = {schema["function"]["name"] for schema in env.tools()}
    assert "_slime_verify_evidence_quote" not in exposed_names
    result = env.mcp_tools["_slime_verify_evidence_quote"](
        source_document="Source title",
        evidence_quote="supported by this exact evidence sentence",
    )
    assert result == {"source_found": True, "matched": True}
    invented = env.mcp_tools["_slime_verify_evidence_quote"](
        source_document="Source title",
        evidence_quote="This purported evidence sentence is completely invented.",
    )
    unknown_source = env.mcp_tools["_slime_verify_evidence_quote"](
        source_document="Unknown source",
        evidence_quote="supported by this exact evidence sentence",
    )
    assert invented == {"source_found": True, "matched": False}
    assert unknown_source == {"source_found": False, "matched": False}
    assert env.mcp_tools.answer_has_local_evidence(
        {"claim": "The documented requirement is supported by this exact evidence sentence."}
    )
    assert not env.mcp_tools.answer_has_local_evidence(
        {"claim": "This fabricated unsupported claim does not occur in the local snapshot."}
    )
    env.close()


def test_local_mcp_workspace_isolated_per_trajectory_and_cleaned(tmp_path: Path):
    source = tmp_path / "source" / "_sandbox"
    (source / "data").mkdir(parents=True)
    (source / "tools.py").write_text("from mcp.server.fastmcp import FastMCP\nmcp = FastMCP('Tools')\n", encoding="utf-8")
    (source / "data" / "state.json").write_text('{"value": 1}', encoding="utf-8")
    workspace_root = tmp_path / "run" / "cache" / "mcp_envs"
    task = {"data_root": str(source), "tools_py": str(source / "tools.py"), "data_source": "mcp"}

    first_task, first = prepare_mcp_workspace(task, "trajectory-1", workspace_root)
    second_task, second = prepare_mcp_workspace(task, "trajectory-2", workspace_root)
    assert first.path != second.path
    assert first_task["data_root"] == str(first.path)
    assert second_task["tools_py"] == str(second.path / "tools.py")
    (first.path / "data" / "state.json").write_text('{"value": 99}', encoding="utf-8")
    assert (second.path / "data" / "state.json").read_text(encoding="utf-8") == '{"value": 1}'
    assert (source / "data" / "state.json").read_text(encoding="utf-8") == '{"value": 1}'

    first.close()
    second.close()
    assert not first.path.exists()
    assert not second.path.exists()


def test_local_mcp_workspace_reused_per_task(tmp_path: Path, monkeypatch):
    import concurrent.futures

    source = tmp_path / "source" / "_sandbox"
    (source / "data").mkdir(parents=True)
    (source / "tools.py").write_text("from mcp.server.fastmcp import FastMCP\nmcp = FastMCP('Tools')\n", encoding="utf-8")
    (source / "data" / "state.json").write_text('{"value": 1}', encoding="utf-8")
    workspace_root = tmp_path / "run" / "cache" / "mcp_envs"
    task = {"data_root": str(source), "tools_py": str(source / "tools.py"), "data_source": "mcp"}
    monkeypatch.setenv("SLIME_MCP_WORKSPACE_SCOPE", "task")

    def prepare(index):
        return prepare_mcp_workspace(task, f"trajectory-{index}", workspace_root)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(prepare, range(8)))

    rewritten_tasks = [rewritten for rewritten, _workspace in prepared]
    workspaces = [workspace for _rewritten, workspace in prepared]
    first = workspaces[0]
    assert {workspace.path for workspace in workspaces} == {first.path}
    assert {workspace.task_id for workspace in workspaces} == {first.task_id}
    assert {rewritten["data_root"] for rewritten in rewritten_tasks} == {str(first.path)}
    (first.path / "data" / "state.json").write_text('{"value": 99}', encoding="utf-8")
    assert all(
        (workspace.path / "data" / "state.json").read_text(encoding="utf-8") == '{"value": 99}'
        for workspace in workspaces
    )

    for workspace in workspaces:
        workspace.close()
    assert first.path.exists()
    cleanup_task_workspaces()
    assert not first.path.exists()


def test_local_mcp_tool_module_loading_is_serialized_and_does_not_leak_sys_modules(tmp_path: Path):
    import concurrent.futures
    import sys

    from slime.rollout.fused_agent.env import FusedEnvironment

    roots = []
    for index in range(8):
        root = tmp_path / f"task-{index}" / "_sandbox"
        root.mkdir(parents=True)
        (root / "tools.py").write_text(
            "from mcp.server.fastmcp import FastMCP\n"
            "mcp = FastMCP('Tools')\n"
            "@mcp.tool()\n"
            "def value():\n"
            f"    return {{'task': {index}}}\n",
            encoding="utf-8",
        )
        roots.append(root)

    def load(root):
        env = FusedEnvironment({"data_root": str(root), "tools_py": str(root / "tools.py")})
        return json.loads(env.mcp_tools.call("value", {}))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(roots)) as pool:
        values = list(pool.map(load, roots))

    assert sorted(item["task"] for item in values) == list(range(len(roots)))
    assert not any(name.startswith("_slime_fused_tools_") for name in sys.modules)


def test_local_mcp_tools_redirect_generated_absolute_and_env_paths(tmp_path: Path):
    asset = tmp_path / "asset"
    data = asset / "data"
    data.mkdir(parents=True)
    (data / "value.json").write_text('{"value": "isolated"}', encoding="utf-8")
    (asset / "tools.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def from_tmp():\n"
        "    return json.loads((Path('/tmp/mcp/data') / 'value.json').read_text())\n"
        "@mcp.tool()\n"
        "def from_env():\n"
        "    return json.loads((Path(os.getenv('MCP_SERVER_BASE_DIR', '/tmp')) / 'data/value.json').read_text())\n",
        encoding="utf-8",
    )
    env = FusedEnvironment({"data_root": str(asset), "tools_py": str(asset / "tools.py")})

    assert json.loads(env.mcp_tools.call("from_tmp", {})) == {"value": "isolated"}
    assert json.loads(env.mcp_tools.call("from_env", {})) == {"value": "isolated"}


def test_local_mcp_process_pool_runs_workspaces_in_parallel(tmp_path: Path, monkeypatch):
    import concurrent.futures
    import time

    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "true")
    root = tmp_path / "shared-process-task"
    root.mkdir()
    (root / "tools.py").write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def work(delay):\n"
        "    time.sleep(delay)\n"
        "    return {'pid': os.getpid(), 'cwd': str(Path.cwd())}\n",
        encoding="utf-8",
    )
    task = {"data_root": str(root), "tools_py": str(root / "tools.py")}

    pool = LocalMCPProcessPool(workers=4)
    try:
        assert len(set(pool.warm())) == 4
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as threads:
            outputs = list(
                threads.map(
                    lambda _index: pool.call(
                        task,
                        "work",
                        {"delay": 0.4},
                        relax_empty=False,
                    ),
                    range(4),
                )
            )
    finally:
        pool.close()

        elapsed = time.perf_counter() - started
    assert elapsed < 1.2
    assert len({output["result"]["pid"] for output in outputs}) == 4
    assert {output["result"]["cwd"] for output in outputs} == {str(root)}


def test_local_mcp_process_pool_matches_odyssey_shard_and_task_load(tmp_path: Path, monkeypatch):
    import concurrent.futures

    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    tasks = []
    for root_index in range(4):
        root = tmp_path / f"task-{root_index}"
        root.mkdir()
        (root / "tools.py").write_text(
            "import os, uuid\n"
            "from mcp.server.fastmcp import FastMCP\n"
            f"ROOT_INDEX = {root_index}\n"
            "INSTANCE_ID = uuid.uuid4().hex\n"
            "mcp = FastMCP('Tools')\n"
            "@mcp.tool()\n"
            "def inspect(trajectory):\n"
            "    return {\n"
            "        'root': ROOT_INDEX,\n"
            "        'trajectory': trajectory,\n"
            "        'pid': os.getpid(),\n"
            "        'instance_id': INSTANCE_ID,\n"
            "    }\n",
            encoding="utf-8",
        )
        tasks.append({"data_root": str(root), "tools_py": str(root / "tools.py")})

    monkeypatch.setenv("SLIME_LOCAL_MCP_TOOLSET_CACHE_SIZE", "2")
    pool = LocalMCPProcessPool(workers=32)

    def run_trajectory(trajectory: int) -> tuple[dict, dict]:
        root_index = trajectory % len(tasks)
        task = tasks[root_index]
        description = pool.describe(task)
        assert description["load_error"] == ""
        assert "inspect" in description["tool_names"]
        called = pool.call(task, "inspect", {"trajectory": trajectory}, relax_empty=False)["result"]
        verified = pool.verify(
            task,
            f"def verify(tools, answer):\n    return tools.inspect({trajectory})\n",
            None,
        )["result"]
        return called, verified

    try:
        initial_pids = pool.warm()
        assert len(set(initial_pids)) == 32
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as threads:
            outputs = list(threads.map(run_trajectory, range(128)))
        final_pids = pool.warm()
        cached_key_counts = [len(shard.cached_keys) for shard in pool.shards]
    finally:
        pool.close()

    assert final_pids == initial_pids
    assert all(count <= 2 for count in cached_key_counts)
    for trajectory, (called, verified) in enumerate(outputs):
        expected_root = trajectory % len(tasks)
        assert called["root"] == expected_root
        assert called["trajectory"] == trajectory
        assert called["pid"] in initial_pids
        assert verified["root"] == expected_root
        assert verified["trajectory"] == trajectory
        assert verified["pid"] in initial_pids


def test_local_mcp_process_pool_restarts_only_polluted_shard(tmp_path: Path):
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    root = tmp_path / "pollution-task"
    root.mkdir()
    (root / "tools.py").write_text(
        "import os\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def status():\n"
        "    return {'pid': os.getpid(), 'polluted': 'MCP_TEST_POLLUTION' in os.environ}\n"
        "@mcp.tool()\n"
        "def pollute():\n"
        "    os.environ['MCP_TEST_POLLUTION'] = '1'\n"
        "    return {'pid': os.getpid()}\n",
        encoding="utf-8",
    )
    task = {"data_root": str(root), "tools_py": str(root / "tools.py")}
    pool = LocalMCPProcessPool(workers=2)
    try:
        initial_pids = pool.warm()
        healthy = pool.call(task, "status", {}, relax_empty=False)["result"]
        repeated = pool.call(task, "status", {}, relax_empty=False)["result"]
        polluted = pool.call(task, "pollute", {}, relax_empty=False)["result"]
        replacement_pids = pool.warm()
        recovered = pool.call(task, "status", {}, relax_empty=False)["result"]
    finally:
        pool.close()

    assert healthy == repeated
    assert polluted["pid"] in initial_pids
    assert len(set(initial_pids) - set(replacement_pids)) == 1
    assert len(set(replacement_pids) - set(initial_pids)) == 1
    assert polluted["pid"] not in replacement_pids
    assert recovered["pid"] in replacement_pids
    assert recovered["polluted"] is False


def test_local_mcp_process_pool_restarts_only_timed_out_shard(tmp_path: Path, monkeypatch):
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    root = tmp_path / "timeout-task"
    root.mkdir()
    (root / "tools.py").write_text(
        "import time\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "mcp = FastMCP('Tools')\n"
        "@mcp.tool()\n"
        "def stall(seconds):\n"
        "    time.sleep(seconds)\n"
        "    return {'completed': True}\n",
        encoding="utf-8",
    )
    task = {"data_root": str(root), "tools_py": str(root / "tools.py")}
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_TIMEOUT", "1")
    pool = LocalMCPProcessPool(workers=2)
    try:
        initial_pids = pool.warm()
        with pytest.raises(TimeoutError):
            pool.call(task, "stall", {"seconds": 2}, relax_empty=False)
        replacement_pids = pool.warm()
    finally:
        pool.close()

    assert len(set(initial_pids) & set(replacement_pids)) == 1
    assert len(set(initial_pids) - set(replacement_pids)) == 1
    assert len(set(replacement_pids) - set(initial_pids)) == 1


def test_local_mcp_describe_cache_is_singleflight_versioned_and_copied(tmp_path: Path, monkeypatch):
    import concurrent.futures
    import threading
    import time

    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    tools_py = tmp_path / "tools.py"
    tools_py.write_text("# v1\n", encoding="utf-8")
    task = {"data_root": str(tmp_path), "tools_py": str(tools_py)}
    pool = LocalMCPProcessPool(workers=1)
    calls = 0
    calls_lock = threading.Lock()

    def describe_uncached(_task):
        nonlocal calls
        with calls_lock:
            calls += 1
            version = calls
        time.sleep(0.02)
        return {
            "schemas": [{"name": "tool", "version": version}],
            "load_error": "",
            "load_warning": "",
            "tool_aliases": {},
            "tool_names": ["tool"],
        }

    monkeypatch.setattr(pool, "_describe_uncached", describe_uncached)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as threads:
            descriptions = list(threads.map(lambda _index: pool.describe(task), range(32)))
        assert calls == 1
        descriptions[0]["schemas"][0]["name"] = "mutated"
        assert descriptions[1]["schemas"][0]["name"] == "tool"

        tools_py.write_text("# version two\n", encoding="utf-8")
        refreshed = pool.describe(task)
        assert calls == 2
        assert refreshed["schemas"][0]["version"] == 2
    finally:
        pool.close()


def test_local_mcp_describe_cache_does_not_cache_failures(tmp_path: Path, monkeypatch):
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    tools_py = tmp_path / "tools.py"
    tools_py.write_text("# tools\n", encoding="utf-8")
    task = {"data_root": str(tmp_path), "tools_py": str(tools_py)}
    pool = LocalMCPProcessPool(workers=1)
    calls = 0

    def describe_uncached(_task):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("describe stalled")
        return {"schemas": [], "load_error": "", "load_warning": "", "tool_aliases": {}, "tool_names": []}

    monkeypatch.setattr(pool, "_describe_uncached", describe_uncached)
    try:
        with pytest.raises(TimeoutError, match="describe stalled"):
            pool.describe(task)
        assert pool.describe(task)["load_error"] == ""
        assert calls == 2
    finally:
        pool.close()


def test_local_mcp_process_pool_release_reuses_worker(monkeypatch):
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    pool = LocalMCPProcessPool(workers=1)
    shard = pool.shards[0]
    index = pool._acquire()
    terminate_calls = []
    monkeypatch.setattr(shard, "terminate", lambda: terminate_calls.append(True))

    pool._release(index)

    assert terminate_calls == []
    assert pool._acquire() == 0
    pool._release(index)
    pool.close()


def test_local_mcp_process_pool_prefers_free_cached_shard_without_waiting():
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    pool = LocalMCPProcessPool(workers=2)
    cache_key = ("task", "tools", None, None)
    pool.shards[0].cached_keys.add(cache_key)
    first = pool._acquire()
    pool._release(first)

    assert list(pool._free) == [1, 0]
    assert pool._acquire(cache_key=cache_key) == 0
    assert pool._acquire(cache_key=cache_key) == 1

    pool._release(0)
    pool._release(1)
    pool.close()


def test_local_mcp_process_shard_terminates_only_on_failure_or_pollution(monkeypatch):
    import concurrent.futures

    from slime.rollout.fused_agent.mcp_process_pool import _ProcessShard, _WORKER_STATE_POLLUTED

    shard = _ProcessShard(0)
    terminate_calls = []
    monkeypatch.setattr(shard, "terminate", lambda: terminate_calls.append(True))

    successful = concurrent.futures.Future()
    successful.set_result({"result": "ok", _WORKER_STATE_POLLUTED: False})
    assert shard.result(successful, 1.0) == {"result": "ok"}
    assert terminate_calls == []

    polluted = concurrent.futures.Future()
    polluted.set_result({"result": "ok", _WORKER_STATE_POLLUTED: True})
    assert shard.result(polluted, 1.0) == {"result": "ok"}
    assert terminate_calls == [True]

    failed = concurrent.futures.Future()
    failed.set_exception(TimeoutError("worker stalled"))
    with pytest.raises(TimeoutError, match="worker stalled"):
        shard.result(failed, 1.0)
    assert terminate_calls == [True, True]


def test_local_mcp_process_pool_reuses_worker_and_bounds_toolset_cache(tmp_path: Path, monkeypatch):
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    def make_task(root: Path, value: str) -> dict:
        root.mkdir()
        (root / "tools.py").write_text(
            "import os, uuid\n"
            "from mcp.server.fastmcp import FastMCP\n"
            f"VALUE = {value!r}\n"
            "INSTANCE_ID = uuid.uuid4().hex\n"
            "mcp = FastMCP('Tools')\n"
            "@mcp.tool()\n"
            "def identity():\n"
            "    return {'pid': os.getpid(), 'instance_id': INSTANCE_ID, 'value': VALUE}\n",
            encoding="utf-8",
        )
        return {"data_root": str(root), "tools_py": str(root / "tools.py")}

    first_task = make_task(tmp_path / "first", "first")
    second_task = make_task(tmp_path / "second", "second")
    monkeypatch.setenv("SLIME_LOCAL_MCP_TOOLSET_CACHE_SIZE", "1")
    pool = LocalMCPProcessPool(workers=1)
    try:
        worker_pid = pool.warm()[0]
        assert pool.describe(first_task)["load_error"] == ""
        first = pool.call(first_task, "identity", {}, relax_empty=False)["result"]
        repeated = pool.call(first_task, "identity", {}, relax_empty=False)["result"]
        verified = pool.verify(
            first_task,
            "def verify(tools, answer):\n    return tools.identity()\n",
            None,
        )["result"]
        after_verify = pool.call(first_task, "identity", {}, relax_empty=False)["result"]
        second = pool.call(second_task, "identity", {}, relax_empty=False)["result"]
        reloaded_first = pool.call(first_task, "identity", {}, relax_empty=False)["result"]
    finally:
        pool.close()

    assert first["pid"] == worker_pid
    assert repeated == first
    assert verified == first
    assert after_verify == first
    assert second["pid"] == first["pid"]
    assert second["instance_id"] != first["instance_id"]
    assert second["value"] == "second"
    assert reloaded_first["pid"] == first["pid"]
    assert reloaded_first["instance_id"] != first["instance_id"]


def test_local_mcp_process_pool_lease_acquire_times_out(monkeypatch):
    from slime.rollout.failure_types import MCPLeaseTimeout
    from slime.rollout.fused_agent.mcp_process_pool import LocalMCPProcessPool

    pool = LocalMCPProcessPool(workers=1)
    index = pool._acquire()
    try:
        with pytest.raises(MCPLeaseTimeout, match="No local MCP process worker"):
            pool._acquire(timeout=0.01)
    finally:
        pool._release(index)
        pool.close()


def test_local_mcp_process_failure_propagates_retryable_infra(monkeypatch):
    from slime.rollout.fused_agent import mcp_process_pool
    from slime.rollout.fused_agent.env import LocalMCPToolset

    class BrokenPool:
        def call(self, *_args, **_kwargs):
            raise TimeoutError("worker stalled")

    toolset = LocalMCPToolset.__new__(LocalMCPToolset)
    toolset.task = {"tools_py": "/tmp/tools.py"}
    toolset.tools = {"lookup": lambda: None}
    toolset.tool_aliases = {}
    toolset.last_call_info = {}
    toolset.last_infra_failure = ""
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "true")
    monkeypatch.delenv("SLIME_LOCAL_MCP_PROCESS_WORKER", raising=False)
    monkeypatch.setattr(mcp_process_pool, "get_local_mcp_process_pool", lambda: BrokenPool())

    result = toolset.call_raw("lookup", {})

    assert result.startswith("Error calling lookup: isolated worker TimeoutError")
    assert toolset.last_infra_failure == "local_mcp_TimeoutError"
    assert toolset.last_call_info["failure_class"] == "retryable_infra"


def test_local_mcp_process_isolates_module_load_verifier_and_sequential_trajectories(tmp_path: Path, monkeypatch):
    import os

    from slime.rollout.fused_agent.env import FusedEnvironment
    from slime.rollout.fused_agent.mcp_process_pool import close_local_mcp_process_pool

    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "true")
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_WORKERS", "1")
    marker = "SLIME_MCP_TEST_MODULE_SIDE_EFFECT"
    monkeypatch.delenv(marker, raising=False)

    def task(root: Path) -> dict:
        root.mkdir()
        (root / "tools.py").write_text(
            "import os, uuid\n"
            "from pathlib import Path\n"
            "from mcp.server.fastmcp import FastMCP\n"
            f"os.environ['{marker}'] = 'worker-only'\n"
            "INSTANCE_ID = uuid.uuid4().hex\n"
            "mcp = FastMCP('Tools')\n"
            "@mcp.tool()\n"
            "def identity():\n"
            "    return {'pid': os.getpid(), 'cwd': str(Path.cwd()), 'instance_id': INSTANCE_ID}\n",
            encoding="utf-8",
        )
        return {
            "data_root": str(root),
            "tools_py": str(root / "tools.py"),
            "verifier": {
                "verification_code": (
                    "import os\n"
                    "from pathlib import Path\n"
                    "def verify(tools, answer):\n"
                    "    observed = tools.identity()\n"
                    "    return {'passed': str(Path.cwd()) == answer['cwd'], "
                    "'worker_instance_id': observed['instance_id']}\n"
                )
            },
        }

    first = FusedEnvironment(task(tmp_path / "first"))
    first_result = json.loads(first.mcp_tools.call("identity", {}))
    first.answer = json.dumps(first_result)
    first.tool_calls = 1
    assert first.compute_final_reward() == 1.0
    assert first.reward_debug["verifier"]["worker_instance_id"] != first_result["instance_id"]
    assert marker not in os.environ
    first.close()

    second = FusedEnvironment(task(tmp_path / "second"))
    second_result = json.loads(second.mcp_tools.call("identity", {}))
    second.close()
    close_local_mcp_process_pool()

    assert first_result["pid"] != second_result["pid"]
    assert first_result["cwd"] == str(tmp_path / "first")
    assert second_result["cwd"] == str(tmp_path / "second")


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


def test_local_mcp_tools_resolve_relative_data_paths_from_asset_directory(tmp_path: Path):
    asset = tmp_path / "asset"
    data = asset / "data"
    data.mkdir(parents=True)
    (data / "value.json").write_text('{"value": "asset"}', encoding="utf-8")
    (asset / "tools.py").write_text(
        """
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Tools")

@mcp.tool(description="Read a generated relative data path")
def read_relative() -> dict:
    with open(Path("data/value.json"), encoding="utf-8") as file:
        return json.load(file)
""",
        encoding="utf-8",
    )
    original_cwd = Path.cwd()
    env = FusedEnvironment({"question": "Read data", "tools_py": str(asset / "tools.py")})

    assert env.mcp_tools.call("read_relative", {}) == json.dumps({"value": "asset"})
    assert Path.cwd() == original_cwd


def test_local_mcp_tools_repair_referenced_missing_all_enum_member(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from enum import Enum
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

class CallStatus(Enum):
    CLOSED = "closed"
    UPCOMING = "upcoming"

@mcp.tool(description="Return the selected status")
def get_status(status: CallStatus = CallStatus.ALL) -> dict:
    return {"status": status.value}
""",
        encoding="utf-8",
    )

    env = FusedEnvironment({"question": "Return status", "tools_py": str(asset / "tools.py")})

    assert env.mcp_tools.load_error == ""
    assert env.mcp_tools.load_warning == ""
    status_schema = next(
        schema for schema in env.tools() if schema["function"]["name"] == "get_status"
    )
    assert status_schema["function"]["parameters"]["properties"]["status"] == {
        "type": "string",
        "description": "",
        "enum": ["closed", "upcoming", "all"],
    }
    assert json.loads(env.mcp_tools.call("get_status", {})) == {"status": "all"}
    assert json.loads(env.mcp_tools.call("get_status", {"status": "closed"})) == {
        "status": "closed"
    }


def test_local_mcp_tools_coerce_normalized_enum_values_and_enum_lists(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from __future__ import annotations
from enum import Enum
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

class MaintenanceModel(str, Enum):
    PREDICTIVE = "Predictive Maintenance Model"
    PREVENTIVE = "Preventive Maintenance Model"

@mcp.tool(description="Return normalized enum values")
def get_models(primary: MaintenanceModel, models: list[MaintenanceModel]) -> dict:
    return {"primary": primary.value, "models": [model.value for model in models]}
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Return models", "tools_py": str(asset / "tools.py")})
    schema = next(schema for schema in env.tools() if schema["function"]["name"] == "get_models")

    assert schema["function"]["parameters"]["properties"]["models"]["items"]["enum"] == [
        "Predictive Maintenance Model",
        "Preventive Maintenance Model",
    ]
    assert json.loads(
        env.mcp_tools.call(
            "get_models",
            {"primary": "predictive", "models": ["PREDICTIVE", "preventive"]},
        )
    ) == {
        "primary": "Predictive Maintenance Model",
        "models": ["Predictive Maintenance Model", "Preventive Maintenance Model"],
    }


def test_local_mcp_tools_alias_names_longer_than_protocol_limit(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    long_name = "retrieve_sections_with_heading_keyword_and_minimum_bullet_count_for_analysis"
    (asset / "tools.py").write_text(
        f"""
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Return a value")
def {long_name}() -> dict:
    return {{"ok": True}}
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Return value", "tools_py": str(asset / "tools.py")})
    exposed_names = [schema["function"]["name"] for schema in env.tools() if schema["function"]["name"] != "finish"]

    assert len(exposed_names) == 1
    assert len(exposed_names[0]) <= 64
    assert exposed_names[0] != long_name
    assert json.loads(env.mcp_tools.call(exposed_names[0], {})) == {"ok": True}
    assert env.mcp_tools[long_name]() == {"ok": True}


def test_local_mcp_tools_repair_definition_time_parent_asset_directory(tmp_path: Path):
    asset = tmp_path / "asset"
    data = asset / "data"
    data.mkdir(parents=True)
    (data / "value.json").write_text('{"value": "asset"}', encoding="utf-8")
    (asset / "tools.py").write_text(
        """
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")
BASE_DIR = Path(__file__).parent.parent

@mcp.tool(description="Read the local asset despite a generated parent path")
def read_value() -> dict:
    with open(BASE_DIR / "data" / "value.json", encoding="utf-8") as file:
        return json.load(file)
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Read value", "tools_py": str(asset / "tools.py")})

    assert json.loads(env.mcp_tools.call("read_value", {})) == {"value": "asset"}


def test_local_mcp_tools_keep_registered_tools_after_late_module_error(tmp_path: Path, caplog):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Return value")
def get_value() -> dict:
    return {"answer": 3}

@tool(description="Malformed generated suffix")
def submit_result(result: dict) -> dict:
    return result
""",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING", logger="slime.rollout.fused_agent.env"):
        env = FusedEnvironment({"question": "Return answer", "tools_py": str(asset / "tools.py")})
        second_env = FusedEnvironment({"question": "Return answer", "tools_py": str(asset / "tools.py")})

    _observation, info = env.reset()

    assert "get_value" in _valid_tool_names(env.tools())
    assert env.mcp_tools.call("get_value", {}) == json.dumps({"answer": 3})
    assert second_env.mcp_tools.call("get_value", {}) == json.dumps({"answer": 3})
    assert "env_error" not in info
    assert "NameError" in info["env_warning"]
    assert caplog.text.count("keeping the registered tools") == 1


def test_local_mcp_uses_single_finish_tool_with_submission_result_schema(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Submit structured result")
def submit_result_difficulty_3(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError(f"result must be a dict, got {type(result).__name__}")
    return result

@mcp.tool(description="Submit list result")
def submit_result_difficulty_1(result: list) -> list:
    return result
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Submit answer", "difficulty": 3, "tools_py": str(asset / "tools.py")})
    tool_names = {schema["function"]["name"] for schema in env.tools()}
    finish = next(schema for schema in env.tools() if schema["function"]["name"] == "finish")

    assert tool_names == {"finish"}
    assert finish["function"]["parameters"]["properties"]["result"]["type"] == "object"

    _observation, _reward, done, _info = asyncio.run(
        env.step(ToolCall("finish", {"command": "submit", "result": {"done": True}}))
    )
    assert done is True
    assert env.answer == '{"done":true}'
    assert json.loads(env.answer) == {"done": True}


def test_mcp_verifier_reward_requires_a_tool_call(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Submit without retrieving")
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps({"done": True})

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_skipped"] == "no_tool_calls"
    assert env.reward_debug["tool_calls"] == 0

    env.tool_calls = 1
    assert env.compute_final_reward() == 1.0


def test_mcp_empty_retrieval_retries_with_default_filters(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Retrieve matching records")
def get_records(query: str = "", min_items: int = 0) -> list[dict]:
    if query or min_items:
        return []
    return [{"title": "Broad evidence"}]
""",
        encoding="utf-8",
    )
    env = FusedEnvironment({"question": "Find evidence", "tools_py": str(asset / "tools.py")})

    observation, reward, done, info = asyncio.run(
        env.step(ToolCall("get_records", {"query": "overly specific", "min_items": 3}))
    )

    assert json.loads(observation) == {
        "_slime_query_fallback": "The original filters returned no results; this result uses the tool defaults.",
        "result": [{"title": "Broad evidence"}],
    }
    assert reward == 0.0
    assert done is False
    assert info["tools/mcp_empty_result"] == 0
    assert info["tools/mcp_query_fallback_attempted"] == 1
    assert info["tools/mcp_query_fallback_used"] == 1
    assert env.mcp_empty_tool_results == 0
    assert env.mcp_nonempty_tool_results == 1


def test_mcp_verifier_tool_calls_ignore_unknown_generated_filters_without_relaxing_evidence_query(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Tools")

@mcp.tool(description="Retrieve matching records")
def get_records(query: str = "") -> list[dict]:
    return [] if query else [{"title": "Evidence"}]
""",
        encoding="utf-8",
    )
    task = {
        "question": "Find evidence",
        "tools_py": str(asset / "tools.py"),
        "verifier": {
            "verification_code": """
def verify(tools, answer):
    records = tools['get_records'](query='', generated_unknown_filter=True)
    return {'passed': bool(records), 'message': 'verified from records'}
"""
        },
    }
    env = FusedEnvironment(task)
    env.answer = json.dumps({"title": "Evidence"})
    env.tool_calls = 1

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["verifier_passed"] is True

    env.task["verifier"]["verification_code"] = """
def verify(tools, answer):
    records = tools['get_records'](query='too narrow', generated_unknown_filter=True)
    return {'passed': bool(records), 'message': 'verified from records'}
"""
    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_passed"] is False


def test_mcp_verifier_rejects_empty_answer_even_when_verifier_passes(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Extract evidence")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    return {'passed': True, 'message': 'No requirements found in documents'}
"""
    env = FusedEnvironment(sample.metadata)
    env.answer = "[]"
    env.tool_calls = 1

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_skipped"] == "empty_answer"


def test_mcp_verifier_rejects_degenerate_success_message(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Extract evidence")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    tools['echo'](value='checked')
    return {'passed': True, 'message': 'No requirements found in documents'}
"""
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps([{"requirement": "nonempty but unsupported"}])
    env.tool_calls = 1

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_passed"] is False
    assert env.reward_debug["verifier"]["rejected_degenerate_success"] is True


def test_mcp_verifier_rejects_swallowed_tool_error_by_default(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Reject verifier fallback")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    return {
        'passed': True,
        'message': 'Basic structure valid (tool verification failed)',
        'details': "Tool error: 'list' object has no attribute 'get'",
    }
"""
    monkeypatch.delenv("SLIME_MCP_STRICT_VERIFIER", raising=False)
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps({"done": True})
    env.tool_calls = 1

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_passed"] is False
    assert env.reward_debug["verifier"]["rejected_verifier_error"] is True


def test_mcp_verifier_rejects_success_without_tool_evidence_by_default(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Reject structural-only verifier")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    return {'passed': True, 'message': 'All structural checks passed'}
"""
    monkeypatch.delenv("SLIME_MCP_STRICT_VERIFIER", raising=False)
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps({"done": True})
    env.tool_calls = 1

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_passed"] is False
    assert env.reward_debug["verifier"]["rejected_missing_tool_evidence"] is True
    assert env.reward_debug["verifier_calls"] == []


def test_mcp_verifier_applies_strict_checks_to_boolean_success(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Reject boolean verifier bypass")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    return True
"""
    monkeypatch.delenv("SLIME_MCP_STRICT_VERIFIER", raising=False)
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps({"done": True})
    env.tool_calls = 1

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier"] is True
    assert env.reward_debug["strict_rejection"] == "rejected_missing_tool_evidence"


def test_mcp_verifier_rejects_duplicate_top_level_collection_items(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Return distinct evidence records")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    evidence = tools['echo'](value='checked')
    return {'passed': evidence.get('echo') == 'checked'}
"""
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    env = FusedEnvironment(sample.metadata)
    env.tool_calls = 1
    evidence = {"claim": "A supported evidence record"}
    env.answer = json.dumps([evidence, evidence])

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier"]["rejected_duplicate_collection_item"] is True

    env.answer = json.dumps([evidence, {"claim": "A different supported evidence record"}])
    assert env.compute_final_reward() == 1.0


def test_mcp_verifier_requires_local_evidence_anchor_when_snapshot_exists(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Ground the answer")
    data = Path(sample.metadata["data_root"]) / "data"
    data.mkdir()
    (data / "source.json").write_text(
        json.dumps(
            {
                "title": "Grounding source",
                "sections": [
                    {
                        "heading": "Documented operational requirement",
                        "text": "The source explicitly documents this supported operational requirement.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    evidence = tools['echo'](value='checked')
    return {'passed': evidence.get('echo') == 'checked'}
"""
    sample.metadata["answer_schema_json"] = json.dumps(
        {
            "type": "object",
            "properties": {
                "category": {"enum": ["Documented operational requirement"]},
                "claim": {"type": "string"},
            },
        }
    )
    monkeypatch.setenv("SLIME_LOCAL_MCP_PROCESS_ISOLATION", "false")
    env = FusedEnvironment(sample.metadata)
    env.tool_calls = 1
    env.answer = json.dumps({"claim": "A fabricated unsupported claim with no local evidence."})

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier"]["rejected_missing_local_evidence_anchor"] is True

    env.answer = json.dumps({"category": "Documented operational requirement"})
    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier"]["rejected_missing_local_evidence_anchor"] is True

    env.answer = json.dumps(
        {"claim": "The source explicitly documents this supported operational requirement."}
    )
    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["local_evidence_overlap"] is True


@pytest.mark.parametrize(
    "result",
    [
        {
            "passed": True,
            "message": "Verified 0 of 2 technologies in data sources",
            "details": {"total_entries": 2, "verified_count": 0},
        },
        {
            "passed": True,
            "message": "Verification completed",
            "details": {
                "verification_details": [
                    {"technology": "unsupported", "verified_in_source": False},
                ]
            },
        },
    ],
)
def test_mcp_strict_verifier_rejects_success_without_verified_evidence(
    tmp_path: Path, monkeypatch, result: dict
):
    sample = _local_mcp_sample(tmp_path, question="Reject unverified success")
    sample.metadata["verifier"]["verification_code"] = f"""
def verify(tools, answer):
    return {result!r}
"""
    monkeypatch.setenv("SLIME_MCP_STRICT_VERIFIER", "true")
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps([{"technology": "unsupported"}])
    env.tool_calls = 1

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["verifier_passed"] is False
    assert env.reward_debug["verifier"]["rejected_unverified_success"] is True


def test_mcp_unverified_success_can_be_explicitly_allowed(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Keep legacy unverified success")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    return {
        'passed': True,
        'message': 'Verified 0 of 2 technologies in data sources',
        'details': {'total_entries': 2, 'verified_count': 0},
    }
"""
    monkeypatch.setenv("SLIME_MCP_STRICT_VERIFIER", "false")
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps([{"technology": "unsupported"}])
    env.tool_calls = 1

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["verifier_passed"] is True


def test_mcp_verifier_error_fallback_can_be_explicitly_allowed(tmp_path: Path, monkeypatch):
    sample = _local_mcp_sample(tmp_path, question="Keep legacy verifier behavior")
    sample.metadata["verifier"]["verification_code"] = """
def verify(tools, answer):
    return {'passed': True, 'message': 'Basic structure valid (tool verification failed)'}
"""
    monkeypatch.setenv("SLIME_MCP_STRICT_VERIFIER", "false")
    env = FusedEnvironment(sample.metadata)
    env.answer = json.dumps({"done": True})
    env.tool_calls = 1

    assert env.compute_final_reward() == 1.0


def test_mcp_verifier_allows_explicitly_valid_empty_answer(tmp_path: Path):
    sample = _local_mcp_sample(tmp_path, question="Confirm that there are no matching records")
    sample.metadata["verifier"] = {
        "allow_empty_answer": True,
        "verification_code": """
def verify(tools, answer):
    tools['echo'](value='checked')
    return {'passed': answer == [], 'message': 'No requirements found in documents'}
""",
    }
    env = FusedEnvironment(sample.metadata)
    env.answer = "[]"
    env.tool_calls = 1

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["verifier_passed"] is True


def test_mcp_tool_load_error_is_nonfatal(tmp_path: Path):
    asset = tmp_path / "asset"
    asset.mkdir()
    (asset / "tools.py").write_text("def broken(:\n", encoding="utf-8")
    env = FusedEnvironment({"question": "Return answer", "tools_py": str(asset / "tools.py")})

    _observation, info = env.reset()

    assert info["task_type"] == "mcp"
    assert "env_error" in info
    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["failure_class"] == "permanent_task_failure"
    assert "infra_failure" not in env.reward_debug


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


def test_mcp_atlas_task_filters_schemas_and_calls_http_tool(monkeypatch):
    monkeypatch.setenv("MCP_ATLAS_AUTH_TOKEN", "test-token")
    monkeypatch.setattr(
        fused_env,
        "_load_atlas_tool_schemas",
        lambda _url: [
            {
                "name": "calculator_add",
                "description": "Add numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
            {
                "name": "filesystem_read_text_file",
                "description": "Read a file",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {
                "name": "submit_result_difficulty_2",
                "description": "Submit result",
                "inputSchema": {
                    "type": "object",
                    "properties": {"result": {"type": "object"}},
                    "required": ["result"],
                },
            },
        ],
    )

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def text(self):
            return json.dumps([{"type": "text", "text": '{"result":5}'}])

    class Session:
        def __init__(self):
            self.requests = []

        def post(self, url, headers, json):
            self.requests.append((url, json))
            assert headers == {"Authorization": "Bearer test-token"}
            return Response()

    session = Session()
    monkeypatch.setattr(fused_env, "_get_atlas_http_session", lambda: session)
    task = {
        "question": "Add two numbers",
        "data_source": "mcp_atlas",
        "mcp_transport": "atlas",
        "mcp_sandbox_url": "http://atlas.test:1984",
        "enabled_tools": ["calculator_add", "submit_result_difficulty_2"],
    }
    env = FusedEnvironment(task)

    observation, info = env.reset()
    result, reward, done, metrics = asyncio.run(env.step(ToolCall("calculator_add", {"a": 2, "b": 3})))

    assert resolve_task_mode(task) == "mcp"
    assert observation == "Add two numbers"
    assert info["task_type"] == "mcp"
    assert _valid_tool_names(env.tools()) == {"calculator_add", "finish"}
    finish = next(schema for schema in env.tools() if schema["function"]["name"] == "finish")
    assert finish["function"]["parameters"]["properties"]["result"]["type"] == "object"
    assert result == '{"result":5}'
    assert reward == 0.0
    assert done is False
    assert metrics["tools/calls"] == 1
    assert session.requests == [
        (
            "http://atlas.test:1984/call-tool",
            {"tool_name": "calculator_add", "tool_args": {"a": 2, "b": 3}, "use_cache": True},
        )
    ]


def test_mcp_atlas_bypasses_sandbox_cache_after_mutating_a_server(monkeypatch):
    monkeypatch.setenv("MCP_ATLAS_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("MCP_ATLAS_READ_ONLY", "false")
    monkeypatch.setattr(
        fused_env,
        "_load_atlas_tool_schemas",
        lambda _url: [
            {
                "name": "slack_conversations_add_message",
                "description": "Post a message",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "slack_channels_list",
                "description": "List channels",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
    )

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def text(self):
            return json.dumps([{"type": "text", "text": "channels"}])

    class Session:
        def __init__(self):
            self.requests = []

        def post(self, url, headers, json):
            self.requests.append((url, json))
            assert headers == {"Authorization": "Bearer test-token"}
            return Response()

    session = Session()
    monkeypatch.setattr(fused_env, "_get_atlas_http_session", lambda: session)
    env = FusedEnvironment(
        {
            "mcp_atlas_eval": True,
            "enabled_tools": ["slack_conversations_add_message", "slack_channels_list"],
        }
    )

    asyncio.run(env.step(ToolCall("slack_conversations_add_message", {"channel_id": "C1", "text": "test"})))
    asyncio.run(env.step(ToolCall("slack_channels_list", {})))

    assert [request[1]["use_cache"] for request in session.requests] == [False, False]


def test_mcp_atlas_read_only_hides_and_blocks_mutating_tools(monkeypatch):
    monkeypatch.setenv("MCP_ATLAS_READ_ONLY", "true")
    monkeypatch.setattr(
        fused_env,
        "_load_atlas_tool_schemas",
        lambda _url: [
            {
                "name": "slack_conversations_add_message",
                "description": "Post a message",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "slack_channels_list",
                "description": "List channels",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
    )
    monkeypatch.setattr(
        fused_env,
        "_get_atlas_http_session",
        lambda: pytest.fail("read-only rejection must happen before an HTTP request"),
    )
    env = FusedEnvironment(
        {
            "mcp_atlas_eval": True,
            "enabled_tools": ["slack_conversations_add_message", "slack_channels_list"],
        }
    )

    assert _valid_tool_names(env.tools()) == {"slack_channels_list", "finish"}
    observation, reward, done, _metrics = asyncio.run(env.step(ToolCall("slack_conversations_add_message", {"channel_id": "C1", "text": "test"})))

    assert observation == "Error: MCP-Atlas read-only evaluation blocks mutating tool slack_conversations_add_message"
    assert reward == 0.0
    assert done is False


def test_mcp_atlas_reports_tools_missing_from_sandbox(monkeypatch):
    monkeypatch.setattr(fused_env, "_load_atlas_tool_schemas", lambda _url: [])
    env = FusedEnvironment(
        {
            "question": "Use an unavailable tool",
            "mcp_atlas_eval": True,
            "enabled_tools": ["anili_get_anime"],
        }
    )

    _observation, info = env.reset()

    assert "anili_get_anime" in info["env_warning"]
    assert _valid_tool_names(env.tools()) == {"finish"}


def test_mcp_atlas_accepts_parquet_ndarray_enabled_tools(monkeypatch):
    monkeypatch.setattr(
        fused_env,
        "_load_atlas_tool_schemas",
        lambda _url: [
            {
                "name": "wikipedia_search_wikipedia",
                "description": "Search Wikipedia",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
    )
    env = FusedEnvironment(
        {
            "question": "Find an article",
            "mcp_atlas_eval": True,
            "enabled_tools": np.array(["wikipedia_search_wikipedia"], dtype=object),
        }
    )

    _observation, info = env.reset()

    assert "env_warning" not in info
    assert _valid_tool_names(env.tools()) == {"wikipedia_search_wikipedia", "finish"}


def test_mcp_atlas_caps_large_tool_output(monkeypatch):
    monkeypatch.setenv("MCP_ATLAS_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("SLIME_FUSED_MAX_TOOL_OUTPUT_LENGTH", "8")
    monkeypatch.setattr(
        fused_env,
        "_load_atlas_tool_schemas",
        lambda _url: [
            {
                "name": "anili_get_anime",
                "description": "Get anime",
                "inputSchema": {"type": "object", "properties": {"id": {"type": "number"}}},
            }
        ],
    )

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def text(self):
            return json.dumps([{"type": "text", "text": "abcdefghijklmnop"}])

    class Session:
        def post(self, _url, headers, json):
            assert headers == {"Authorization": "Bearer test-token"}
            return Response()

    monkeypatch.setattr(fused_env, "_get_atlas_http_session", lambda: Session())
    env = FusedEnvironment({"mcp_atlas_eval": True, "enabled_tools": ["anili_get_anime"]})

    output, *_ = asyncio.run(env.step(ToolCall("anili_get_anime", {"id": 1})))

    assert output.startswith("abcdefgh\n\n[MCP-Atlas tool output truncated to 8 characters")
    assert "original length: 16" in output


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


def test_qwen3_tito_delta_preserves_raw_assistant_token_prefix():
    tokenizer = FakeQwen3ChatTemplateTokenizer(drift_assistant_end=True)
    old_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "<think> raw spacing </think>\n<tool_call>{}</tool_call>"},
    ]
    new_messages = [
        *old_messages,
        {"role": "user", "content": "<tool_response>result</tool_response>"},
    ]
    raw_prefix = [11, 12, 13]

    delta = _render_qwen3_tito_delta_ids(
        tokenizer,
        old_messages,
        new_messages,
        raw_prefix,
        disable_thinking=False,
    )
    merged = raw_prefix + delta

    assert merged[: len(raw_prefix)] == raw_prefix
    assert delta[0] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert "<tool_response>result</tool_response>" in tokenizer.decode(delta[2:])
    assert tokenizer.decode(delta[2:]).endswith("<|im_start|>assistant\n")


def test_qwen35_tito_delta_uses_original_query_after_tool_response_users():
    tokenizer = FakeQwen35ChatTemplateTokenizer()
    old_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "<think>x</think><tool_call>{}</tool_call>"},
        {"role": "user", "content": "<tool_response>evidence</tool_response>"},
        {"role": "assistant", "content": "<think>next</think><tool_call>{}</tool_call>"},
    ]
    new_messages = [*old_messages, {"role": "user", "content": "<tool_response>more</tool_response>"}]

    delta = _render_qwen3_tito_delta_ids(tokenizer, old_messages, new_messages, [11, 12, 13], disable_thinking=False)

    assert tokenizer.decode(delta[2:]).endswith("<|im_start|>assistant\n")


def test_qwen3_tito_delta_rejects_history_rewrite():
    tokenizer = FakeQwen3ChatTemplateTokenizer()
    old_messages = [{"role": "user", "content": "original"}]
    rewritten_messages = [
        {"role": "user", "content": "rewritten"},
        {"role": "user", "content": "next"},
    ]

    with pytest.raises(ValueError, match="append-only"):
        _render_qwen3_tito_delta_ids(
            tokenizer,
            old_messages,
            rewritten_messages,
            [1, 2, 3],
            disable_thinking=False,
        )


def test_gemma4_tito_delta_appends_only_tool_response_after_raw_action():
    tokenizer = FakeGemma4Tokenizer()
    history = [{"role": "user", "content": "find evidence"}]
    old_messages = [*history, {"role": "assistant", "content": "raw non-canonical action"}]
    new_messages = [
        *history,
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "echo", "arguments": {"value": "alpha"}}}],
            "tool_responses": [{"name": "echo", "response": {"echo": "alpha"}}],
        },
    ]

    delta = _render_gemma4_tito_delta_ids(
        tokenizer,
        old_messages,
        new_messages,
        tools=None,
        disable_thinking=True,
    )
    rendered_delta = tokenizer.decode(delta)

    assert rendered_delta.startswith("<turn|>\n<|turn>tool\n")
    assert '{"echo": "alpha"}' in rendered_delta
    assert "<|tool_call>" not in rendered_delta
    assert rendered_delta.endswith("<turn|>\n<|turn>model\n")


def test_gemma4_strict_tito_avoids_structured_assistant_replay_segments(tmp_path: Path):
    first = _gemma4_echo_call("alpha")
    finish = _gemma4_finish_call('{"done":true}')
    prompt_ids_seen: list[list[int]] = []

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use echo before finish"),
        [{"text": first}, {"text": finish}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
            "SLIME_FUSED_STRICT_TITO": "True",
        },
        tokenizer=FakeGemma4Tokenizer(),
        prompt_ids_seen=prompt_ids_seen,
    )

    assert len(result) == 1
    assert result[0].metadata["segment_count"] == 1
    assert result[0].metadata["fused_tito_boundary_count"] == 0
    assert result[0].metadata["fused_tito_incremental_turns"] == 1
    assert result[0].metadata["fused_tito_prompt_prefix_mismatch_turns"] == 0
    assert prompt_ids_seen[1][: len(prompt_ids_seen[0]) + len(first)] == [
        *prompt_ids_seen[0],
        *[ord(character) for character in first],
    ]


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


def test_historical_thinking_compaction_removes_complete_assistant_blocks_without_mutation():
    tool_call = '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
    messages = [
        {"role": "system", "content": "keep <think>system</think>"},
        {"role": "assistant", "content": f"<think>first\nplan</think>\n\n{tool_call}"},
        {"role": "user", "content": "keep <think>user</think>"},
        {"role": "assistant", "content": "incomplete <think>must be removed"},
        {"role": "assistant", "content": "prefix<think>second</think>\n\nsuffix"},
    ]

    compacted = fused_generate._messages_without_historical_thinking(messages)

    assert compacted[0] is messages[0]
    assert compacted[1]["content"] == tool_call
    assert compacted[2] is messages[2]
    assert compacted[3]["content"] == "incomplete"
    assert compacted[4]["content"] == "prefix\n\nsuffix"
    assert messages[1]["content"] == f"<think>first\nplan</think>\n\n{tool_call}"

    rendered = FakeChatTemplateTokenizer().apply_chat_template(compacted, tokenize=False)
    assert f"<|im_start|>assistant\n{tool_call}" in rendered
    assert "<|im_start|>assistant\n\n\n<tool_call>" not in rendered


def test_historical_thinking_compaction_removes_gemma_thought_channel_blocks():
    messages = [
        {"role": "assistant", "content": "<|channel>thought\nfirst\nplan\n<channel|>\naction"},
        {"role": "user", "content": "keep <|channel>thought\nuser text\n<channel|>"},
        {"role": "assistant", "content": "incomplete <|channel>thought\nmust be removed"},
        {"role": "assistant", "content": "<think>a</think><|channel>thought\nb\n<channel|>mixed"},
    ]

    compacted = fused_generate._messages_without_historical_thinking(messages)

    assert compacted[0]["content"] == "action"
    assert compacted[1] is messages[1]
    assert compacted[2]["content"] == "incomplete"
    assert compacted[3]["content"] == "mixed"
    assert messages[0]["content"] == "<|channel>thought\nfirst\nplan\n<channel|>\naction"


def test_historical_thinking_compaction_strips_bare_closer_reasoning():
    # Qwen3.5-style templates pre-open ``<think>\n`` in the generation prompt,
    # so recorded assistant content carries the reasoning with NO opening tag:
    # ``thought</think>\n\nanswer``. The template later re-extracts reasoning by
    # splitting on the bare ``</think>``; the compaction must strip it the same
    # way or historical thinking silently survives in every served prompt.
    messages = [
        {"role": "assistant", "content": "private plan\nstep two</think>\n\nfinal answer"},
        {"role": "user", "content": "bare </think> in user text stays"},
        {"role": "assistant", "content": "a</think>mid<think>closed</think>\n\ntail"},
        {"role": "assistant", "content": "answer</think>\n\nreal<think>unclosed is removed"},
    ]

    compacted = fused_generate._messages_without_historical_thinking(messages)

    assert compacted[0]["content"] == "final answer"
    assert compacted[1] is messages[1]
    assert compacted[2]["content"] == "mid\n\ntail"
    assert compacted[3]["content"] == "real"
    assert messages[0]["content"] == "private plan\nstep two</think>\n\nfinal answer"


def test_unclosed_thinking_rebuilds_only_parsed_actions_without_raw_fallback():
    raw_action = '<tool_call>\n{"name": "echo", "arguments": {"value": "alpha"}}\n</tool_call>'
    structured_calls = [{"function": {"name": "echo", "arguments": {"value": "beta"}}}]
    messages = [
        {"role": "assistant", "content": "<think>private\n" + raw_action},
        {"role": "assistant", "content": "public prefix<think>private with no action"},
        {
            "role": "assistant",
            "content": "visible prefix<think>private structured content",
            "tool_calls": structured_calls,
        },
    ]

    compacted = fused_generate._messages_without_historical_thinking(
        messages,
        parser=QwenToolParser(valid_tools={"echo"}),
    )

    assert compacted[0]["content"] == _echo_call("alpha")
    assert compacted[1]["content"] == ""
    assert compacted[2]["content"] == "visible prefix"
    assert compacted[2]["tool_calls"] is structured_calls
    assert messages[0]["content"] == "<think>private\n" + raw_action


def test_enabled_thinking_discards_prior_thoughts_and_emits_prompt_equal_segments(tmp_path: Path):
    first_thought = "<think>private first-round plan</think>\n"
    second_thought = "<think>fresh second-round plan</think>\n"
    first_action = _echo_call("alpha")
    final_action = _finish_call()

    samples = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Think independently on each turn"),
        [{"text": first_thought + first_action}, {"text": second_thought + final_action}],
        {
            "FUSED_DISABLE_THINKING": "False",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeChatTemplateTokenizer(),
    )

    assert len(samples) == 2
    second_segment = "".join(chr(token) for token in samples[1].tokens)
    assert first_thought not in second_segment
    assert first_action in second_segment
    assert second_thought in second_segment
    assert [sample.reward for sample in samples] == [0.0, 1.0]
    assert len({sample.metadata["parent_traj_id"] for sample in samples}) == 1
    assert samples[0].metadata["parent_traj_id"]
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1]
    assert all(sample.metadata["prompt_equal_loss"] for sample in samples)

    # The dumped episode (shared by both segments) must carry the segment
    # breakdown and the thinking-dropout marking for offline readers.
    episode = samples[0].metadata["rllm_episode"]
    assert episode["metadata"]["prompt_equal_loss"] is True
    assert episode["metadata"]["segment_count"] == 2
    assert ("segment_num", "2") in rollout_visualization._episode_summary_rows(samples[0], episode, None)
    assert episode["metadata"]["parent_traj_id"] == samples[0].metadata["parent_traj_id"]
    assert episode["metadata"]["historical_thinking_discard_steps"] == 1
    steps = episode["trajectories"][0]["steps"]
    assert steps[0]["info"]["historical_thinking_discarded"] is False
    assert steps[1]["info"]["historical_thinking_discarded"] is True
    assert steps[1]["info"]["tito_context_reason"] == "historical_thinking_discard"
    # chat_completions reflect the SERVED prompt (thinking stripped), not the
    # raw history.
    second_step_history = json.dumps(steps[1]["chat_completions"])
    assert "private first-round plan" not in second_step_history


def test_unclosed_thinking_is_discarded_without_losing_qwen_tool_call(tmp_path: Path):
    private_thought = "<think>private unclosed plan\n"
    raw_first_action = '<tool_call>\n{"name": "echo", "arguments": {"value": "alpha"}}\n</tool_call>'
    canonical_first_action = _echo_call("alpha")
    prompt_ids_seen = []

    samples = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Recover the action after malformed thinking"),
        [{"text": private_thought + raw_first_action}, {"text": _finish_call()}],
        {
            "FUSED_DISABLE_THINKING": "False",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeChatTemplateTokenizer(),
        prompt_ids_seen=prompt_ids_seen,
    )

    second_prompt = "".join(chr(token) for token in prompt_ids_seen[1])
    assert "private unclosed plan" not in second_prompt
    assert canonical_first_action in second_prompt
    assert raw_first_action not in second_prompt
    assert len(samples) == 2
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1]
    assert all(sample.metadata["prompt_equal_loss"] for sample in samples)
    steps = samples[0].metadata["rllm_episode"]["trajectories"][0]["steps"]
    assert steps[1]["info"]["historical_thinking_discarded"] is True
    assert steps[1]["info"]["tito_context_reason"] == "historical_thinking_discard"


def test_enabled_thinking_discards_gemma_thought_channel_and_emits_prompt_equal_segments(tmp_path: Path):
    first_thought = "<|channel>thought\nprivate first-round plan\n<channel|>\n"
    second_thought = "<|channel>thought\nfresh second-round plan\n<channel|>\n"
    first_action = _gemma4_echo_call("alpha")
    final_action = _gemma4_finish_call()
    prompt_ids_seen = []

    samples = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Think independently on each turn"),
        [{"text": first_thought + first_action}, {"text": second_thought + final_action}],
        {
            "FUSED_DISABLE_THINKING": "False",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
        prompt_ids_seen=prompt_ids_seen,
    )

    # Gemma4's tool flow rewrites the assistant turn into a structured
    # tool_calls/tool_responses message; the strict TITO delta keeps the raw
    # first-turn tokens training in place. Discarding the historical thought
    # rewrites the served history, so the second turn forks into a fresh
    # prompt-equal segment -- same shape as the Qwen3 discard flow.
    second_prompt = "".join(chr(token) for token in prompt_ids_seen[1])
    assert "private first-round plan" not in second_prompt
    assert "call:echo" in second_prompt
    assert len(samples) == 2
    first_segment = "".join(chr(token) for token in samples[0].tokens)
    second_segment = "".join(chr(token) for token in samples[1].tokens)
    assert "private first-round plan" in first_segment  # raw turn keeps its tokens
    assert "private first-round plan" not in second_segment
    assert "fresh second-round plan" in second_segment
    assert [sample.reward for sample in samples] == [0.0, 1.0]
    assert len({sample.metadata["parent_traj_id"] for sample in samples}) == 1
    assert samples[0].metadata["parent_traj_id"]
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1]
    assert all(sample.metadata["segment_count"] == 2 for sample in samples)
    assert all(sample.metadata["prompt_equal_loss"] for sample in samples)


def test_unclosed_gemma_thinking_is_discarded_without_losing_tool_call(tmp_path: Path):
    private_thought = "<|channel>thought\nprivate unclosed plan\n"
    prompt_ids_seen = []

    samples = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Recover the Gemma action after malformed thinking"),
        [{"text": private_thought + _gemma4_echo_call("alpha")}, {"text": _gemma4_finish_call()}],
        {
            "FUSED_DISABLE_THINKING": "False",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeGemma4Tokenizer(),
        prompt_ids_seen=prompt_ids_seen,
    )

    second_prompt = "".join(chr(token) for token in prompt_ids_seen[1])
    assert "private unclosed plan" not in second_prompt
    assert "call:echo" in second_prompt
    assert len(samples) == 2
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1]
    assert all(sample.metadata["prompt_equal_loss"] for sample in samples)


def test_discard_flag_without_emitted_thinking_keeps_single_tito_segment(tmp_path: Path):
    samples = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="No thoughts, straight answers"),
        [{"text": _echo_call("alpha")}, {"text": _finish_call()}],
        {
            "FUSED_DISABLE_THINKING": "False",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeChatTemplateTokenizer(),
    )

    assert len(samples) == 1
    assert samples[0].metadata["prompt_equal_loss"] is True
    assert samples[0].metadata["segment_index"] == 0
    assert samples[0].metadata["segment_count"] == 1


def test_prompt_equal_is_independent_of_historical_thinking_discard(tmp_path: Path):
    thought = "<think>model emitted this despite disable-thinking</think>\n"
    samples = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Keep existing disable-thinking behavior"),
        [{"text": thought + _echo_call("alpha")}, {"text": _finish_call()}],
        {
            "FUSED_DISABLE_THINKING": "True",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "CREDIT_ASSIGNMENT_ENABLE": "False",
        },
        tokenizer=FakeChatTemplateTokenizer(),
    )

    assert len(samples) == 1
    assert thought in "".join(chr(token) for token in samples[0].tokens)
    assert samples[0].metadata["prompt_equal_loss"] is True
    assert samples[0].metadata["segment_count"] == 1


def test_eval_rollout_discards_prior_thinking_before_next_assistant_step(tmp_path: Path):
    first_thought = "<think>private eval plan</think>\n"
    first_action = _echo_call("eval-alpha")
    prompt_ids_seen = []

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Compact eval history"),
        [{"text": first_thought + first_action}, {"text": "<think>fresh eval plan</think>\n" + _finish_call()}],
        {
            "FUSED_DISABLE_THINKING": "False",
            "FUSED_DISCARD_HISTORICAL_THINKING": "True",
            "SLIME_FUSED_EVAL_TRAJECTORY_SAMPLE_RATE": "1",
        },
        evaluation=True,
        tokenizer=FakeChatTemplateTokenizer(),
        prompt_ids_seen=prompt_ids_seen,
    )

    assert len(result) == 1
    assert len(prompt_ids_seen) == 2
    second_prompt = "".join(chr(token) for token in prompt_ids_seen[1])
    assert first_thought not in second_prompt
    assert first_action in second_prompt
    episode = result[0].metadata["rllm_episode"]
    assert result[0].metadata["segment_count"] == 1
    assert episode["metadata"]["segment_count"] == 1
    assert ("segment_num", "1") in rollout_visualization._episode_summary_rows(result[0], episode, None)
    assert episode["metadata"]["discard_historical_thinking_enabled"] is True
    assert episode["metadata"]["historical_thinking_discard_steps"] == 1
    steps = episode["trajectories"][0]["steps"]
    assert steps[0]["info"]["tito_context_reason"] == "evaluation"
    assert steps[0]["info"]["historical_thinking_discarded"] is False
    assert steps[1]["info"]["tito_context_reason"] == "evaluation"
    assert steps[1]["info"]["historical_thinking_discarded"] is True


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


def test_gemma4_render_prompt_ids_inlines_native_tools_before_chat_template():
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

    assert tokenizer.rendered_messages is None
    assert "<|tool>declaration:web_search{" in rendered
    assert "<|tool>declaration:finish{" in rendered
    assert "When you need to call a tool" not in rendered
    assert "Do not wrap Gemma4 tool calls" not in rendered
    assert "<tools>" not in rendered
    assert "<function=FUNCTION_NAME>" not in rendered


def test_gemma4_render_prompt_ids_preserves_union_finish_schema():
    prompt_ids = _render_prompt_ids(
        FakeGemma4Tokenizer(),
        [
            {"role": "system", "content": FUSED_MCP_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": "Extract."},
        ],
        tools=[finish_schema(structured_result=True)],
    )
    rendered = "".join(chr(token) for token in prompt_ids)

    assert 'result:{description:<|"|>Final answer or JSON value.<|"|>,anyOf:[' in rendered
    assert '{type:<|"|>OBJECT<|"|>}' in rendered
    assert '{type:<|"|>ARRAY<|"|>}' in rendered
    assert 'type:<|"|><|"|>' not in rendered


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

    assert tokenizer.rendered_messages is None
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
def test_disable_thinking_loss_mask_stray_closer_after_tool_call_trains_full(close_id):
    # A stray bare </think> AFTER a legitimate tool call is an artifact, not a
    # mis-fired reasoning block; masking through it would zero-mask the action.
    tokenizer = _ThinkTokenizer(close_id)
    response = '<tool_call>{"name":"echo","arguments":{"value":"x"}}</tool_call> stray </think> tail'
    output_ids = [800, 801, 802, close_id, 803]
    mask = _default_response_loss_mask(tokenizer, response, output_len=len(output_ids), disable_thinking=True, output_ids=output_ids)
    assert mask == [1, 1, 1, 1, 1]


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


def test_web_search_user_prompt_defaults_to_short_and_supports_long(monkeypatch):
    schemas = [web_search_schema(), finish_schema()]

    monkeypatch.delenv("FUSED_WEB_SEARCH_USER_PROMPT", raising=False)
    short_messages = _initial_messages("gem", "web_search", "Who?", schemas)
    assert "Use web_search to gather evidence." in short_messages[1]["content"]
    assert "Instructions:" not in short_messages[1]["content"]

    monkeypatch.setenv("FUSED_WEB_SEARCH_USER_PROMPT", "long")
    long_messages = _initial_messages("gem", "web_search", "Who?", schemas)
    assert "<question>\nWho?\n</question>" in long_messages[1]["content"]
    assert "Search as many times as needed" in long_messages[1]["content"]
    assert "call finish exactly once" in long_messages[1]["content"]
    assert "final answer should also be clearly stated" not in long_messages[1]["content"]
    assert "only use web_search and finish" in long_messages[1]["content"]


def test_web_search_prompt_removes_conflicting_answer_tag_instruction():
    schemas = [web_search_schema(), finish_schema()]
    question = "Who?When ready, output the final answer enclosed in <answer> and </answer> tags. " "Do not generate any content after the </answer> tag."

    messages = _initial_messages("gem", "web_search", question, schemas)

    assert "<question>\nWho?\n</question>" in messages[1]["content"]
    assert "enclosed in <answer>" not in messages[1]["content"]
    assert "do not use <answer> tags or \\boxed{}" in messages[0]["content"]


def test_gemma4_initial_messages_leave_declaration_rendering_to_prompt_builder():
    schemas = [web_search_schema(), finish_schema()]

    messages = _initial_messages(
        "gem",
        "web search",
        "Who?",
        schemas,
        model_name="/share/nlp/share/plm/gemma-4-E2B-it",
    )

    assert messages[0]["content"].startswith(FUSED_SEARCH_SYSTEM_PROMPT.strip())
    assert "Gemma4 native tool-call contract:" in messages[0]["content"]
    assert "<|tool>declaration:" not in messages[0]["content"]

    rendered = fused_generate._apply_chat_template(
        FakeGemma4Tokenizer(),
        messages,
        tools=schemas,
        tokenize=False,
        add_generation_prompt=True,
        disable_thinking=False,
    )
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


def test_rag_initial_messages_include_retrieved_context_without_tools():
    messages = _initial_messages(
        "rag",
        "web search",
        "Who?",
        [web_search_schema(), finish_schema()],
        retrieved_context="[1] Source: Evidence.",
    )

    assert messages[0]["content"] == fused_generate.COT_SYSTEM_PROMPT
    assert "<question>\nWho?\n</question>" in messages[1]["content"]
    assert "<context>\n[1] Source: Evidence.\n</context>" in messages[1]["content"]
    assert "web_search" not in messages[0]["content"]


def test_rag_generate_retrieves_once_then_scores_as_reasoning_only(monkeypatch):
    calls = []

    async def fake_step(self, action):
        calls.append(action)
        self.tool_calls += 1
        return "[1] Source: answer evidence.", 0.0, False, {"tools/search_calls": 1}

    monkeypatch.setattr(FusedEnvironment, "step", fake_step)
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence"}),
        [{"text": "\\boxed{answer}"}],
        {"FUSED_HARNESS": "rag", "CREDIT_ASSIGNMENT_ENABLE": "False"},
    )

    sample = result[0]
    assert [(call.name, call.arguments) for call in calls] == [("web_search", {"query": "Find evidence", "max_results": 5})]
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "reasoning_only"
    assert sample.metadata["fused_reward_debug"]["tool_calls"] == 1


def test_cot_generate_ignores_tool_calls_and_scores_as_reasoning_only(monkeypatch):
    async def fail_step(self, action):
        raise AssertionError(f"cot harness must not execute tools: {action}")

    monkeypatch.setattr(FusedEnvironment, "step", fail_step)
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence"}),
        [{"text": '\\boxed{answer}\n<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'}],
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


def test_rag_search_retrieval_uses_configured_context_word_limit(monkeypatch):
    async def fake_retrieve(_url, _payload, *, retry_budget, episode_cache):
        return (
            {
                "results": [
                    {
                        "content": {
                            "title": "Long",
                            "chunk_text": " ".join(f"token{i}" for i in range(1200)),
                        }
                    }
                ]
            },
            0,
            None,
        )

    monkeypatch.setattr(fused_env, "_retrieve_json_cached", fake_retrieve)
    monkeypatch.setenv("RAG_CONTEXT_MAX_WORDS", "1024")
    env = FusedEnvironment({"question": "q"}, rag=True)

    observation, _, _, _ = asyncio.run(env.step(ToolCall("web_search", {"query": "q"})))

    assert len(observation.split()) == 1024
    assert "token1199" not in observation


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

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q", "max_results": "10\n</<|im_start|>user>"})))

    assert observation.startswith("[Result 1] Title: Doc\nContent: fallback0 fallback1")
    assert reward == 0.0
    assert done is False
    assert info["tools/search_calls"] == 1
    assert FakeSession.calls[0][1]["top_k"] == 1
    assert FakeSession.calls[0][1]["topk"] == 1
    assert FakeSession.calls[0][1]["max_results"] == 1


def test_web_search_clamps_model_requested_results_to_environment_limit(monkeypatch):
    payloads = []

    async def fake_retrieve(_url, payload, *, retry_budget, episode_cache):
        del retry_budget, episode_cache
        payloads.append(payload)
        content = "The retrieved document contains concrete historical evidence with named entities, dates, locations, and detailed context. " "It identifies the relevant person, explains the relationship in the question, and cites the year in which the documented event occurred."
        return {"results": [{"content": {"title": "Doc", "chunk_text": content}}]}, 0, None

    monkeypatch.setattr("slime.rollout.fused_agent.env._retrieve_json_cached", fake_retrieve)
    env = FusedEnvironment({"question": "Find evidence"}, retrieval_max_results=2)

    observation, reward, done, _ = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q", "max_results": 5})))

    assert observation.startswith("[Result 1] Title: Doc")
    assert reward == 0.0
    assert done is False
    assert payloads[0]["top_k"] == 2
    assert payloads[0]["topk"] == 2
    assert payloads[0]["max_results"] == 2


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


def test_webqa_normalized_target_span_mode_scores_verbose_gemma_answer(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "Indios Bárbaros"})
    env.tool_calls = 3
    env.web_search_queries = {"first query", "second query", "third query"}
    env.answer = 'The term is "Indios Bárbaros" (barbarian Indians).'

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["match_mode"] == "normalized_target_span"
    assert env.reward_debug["exact_match"] is False
    assert env.reward_debug["normalized_target_span_match"] is True


def test_webqa_normalized_target_span_mode_scores_concise_qualified_answer(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "DAMA/NaI"})
    env.tool_calls = 1
    env.web_search_queries = {"query"}
    env.answer = "DAMA/NaI experiment"

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["exact_match"] is False
    assert env.reward_debug["normalized_target_span_match"] is True


def test_webqa_normalized_target_span_mode_rejects_short_generic_target(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "MUSIC"})
    env.tool_calls = 1
    env.web_search_queries = {"query"}
    env.answer = "The answer is MUSIC"

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["normalized_target_span_match"] is False

    env.answer = "MUSIC"
    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["exact_match"] is True
    assert env.reward_debug["normalized_target_span_match"] is False


def test_webqa_normalized_target_span_mode_rejects_long_explanation(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "Frost Laws"})
    env.tool_calls = 1
    env.web_search_queries = {"query"}
    env.answer = (
        "The legal framework implicitly referenced is the general traffic regulation system, "
        "which may include Frost Laws among several broader seasonal restrictions."
    )

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["normalized_target_span_match"] is False


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        ("Macworld Conference & Expo", "Macworld Conference and Expo"),
        ("The Huzita–Hatori Axioms", "Huzita-Hatori axioms"),
        ("The Music Ontology (MO)", "Music Ontology"),
        ("Variational Quantum Eigensolver (VQE)", "VQE (Variational Quantum Eigensolver)"),
    ],
)
def test_webqa_safe_structured_equivalence(monkeypatch, ground_truth, prediction):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    env = FusedEnvironment({"question": "structured equivalence", "ground_truth": ground_truth})
    env.web_search_queries = {"query"}
    env.answer = prediction

    assert env.compute_final_reward() == 1.0
    assert env.reward_debug["structured_alias_match"] is True
    assert env.reward_debug["reward_match_reason"] == "structured_alias"


@pytest.mark.parametrize(
    ("ground_truth", "prediction"),
    [
        ("The Music Ontology (MO)", "Music Ontology (MOO)"),
        ("Field Music", "Music Field"),
    ],
)
def test_webqa_structured_equivalence_rejects_unsafe_fuzzy_matches(monkeypatch, ground_truth, prediction):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    env = FusedEnvironment({"question": "unsafe equivalence", "ground_truth": ground_truth})
    env.web_search_queries = {"query"}
    env.answer = prediction

    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["structured_alias_match"] is False


def test_webqa_approved_alias_is_question_scoped(monkeypatch, tmp_path):
    question = "Which specific mutation is requested?"
    question_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
    registry = tmp_path / "aliases.json"
    registry.write_text(
        json.dumps(
            {
                "approved_aliases": {
                    question_hash: {
                        "ground_truth": ["DNMT3A R882H mutation"],
                        "aliases": ["R882H"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "normalized_target_span")
    monkeypatch.setenv("FUSED_WEBQA_ALIAS_REGISTRY_PATH", str(registry))

    matched = FusedEnvironment({"question": question, "ground_truth": "DNMT3A R882H mutation"})
    matched.web_search_queries = {"query"}
    matched.answer = "R882H"
    assert matched.compute_final_reward() == 1.0
    assert matched.reward_debug["approved_alias_match"] is True

    other = FusedEnvironment({"question": "A different question", "ground_truth": "DNMT3A R882H mutation"})
    other.web_search_queries = {"query"}
    other.answer = "R882H"
    assert other.compute_final_reward() == 0.0
    assert other.reward_debug["approved_alias_match"] is False

    changed_target = FusedEnvironment({"question": question, "ground_truth": "DNMT3A R882C mutation"})
    changed_target.web_search_queries = {"query"}
    changed_target.answer = "R882H"
    assert changed_target.compute_final_reward() == 0.0
    assert changed_target.reward_debug["approved_alias_match"] is False


def test_webqa_alias_registry_accounts_for_all_153_audited_candidates():
    registry_path = Path(fused_env.__file__).with_name("webqa_alias_registry.json")
    audit = json.loads(registry_path.read_text(encoding="utf-8"))["audit"]

    def expand_ranks(value):
        ranks = []
        for item in value.split(","):
            if "-" in item:
                start, end = (int(part) for part in item.split("-"))
                ranks.extend(range(start, end + 1))
            else:
                ranks.append(int(item))
        return ranks

    approved = expand_ranks(audit["approved_ranks"])
    rejected = expand_ranks(audit["rejected_ranks"])
    assert len(approved) == audit["approved_candidate_pairs"] == 22
    assert len(rejected) == audit["rejected_candidate_pairs"] == 131
    assert sorted(approved + rejected) == list(range(1, 154))


def test_webqa_rejects_unknown_reward_match_mode(monkeypatch):
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    monkeypatch.setenv("FUSED_WEBQA_REWARD_MATCH_MODE", "fallback")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "answer"})
    env.tool_calls = 1
    env.web_search_queries = {"query"}
    env.answer = "answer"

    with pytest.raises(ValueError, match="Unsupported FUSED_WEBQA_REWARD_MATCH_MODE"):
        env.compute_final_reward()


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


def test_openrouter_summary_backend_returns_raw_message_content(monkeypatch):
    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return {"choices": [{"finish_reason": "stop", "message": {"content": '{"summary": "Raw model answer"}'}}]}

    class FakeSession:
        calls = []

        def post(self, url, json=None, headers=None):
            self.calls.append((url, json, headers))
            return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_MODEL", "test/model")
    session = FakeSession()

    summary, retries = asyncio.run(fused_env._summarize_openrouter_batch(session, "First document\n\nSecond document", retry_budget=0))

    assert summary == "Raw model answer"
    assert retries == 0
    url, payload, headers = session.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["model"] == "test/model"
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0
    assert payload["reasoning"] == {"enabled": False}
    assert payload["response_format"]["type"] == "json_schema"
    assert "First document\n\nSecond document" in payload["messages"][0]["content"]
    assert headers["Authorization"] == "Bearer test-key"


def test_summary_units_handle_latin_and_cjk_without_extra_dependencies():
    assert _summary_units("Microsoft was founded in 1975.") == 5
    assert _summary_units("微软成立于1975年。") == 7


def test_summary_input_limit_applies_units_and_character_cap():
    mixed = "微软成立于1975年。 Microsoft was founded in 1975. "
    limited = _limit_summary_input(mixed * 1000)
    assert _summary_units(limited) <= fused_env.SUMMARY_MAX_INPUT_WORDS
    assert len(limited) <= fused_env.SUMMARY_MAX_INPUT_CHARS

    cjk_limited = _limit_summary_input("中" * 9000)
    assert len(cjk_limited) == fused_env.SUMMARY_MAX_INPUT_WORDS
    assert _summary_units(cjk_limited) == fused_env.SUMMARY_MAX_INPUT_WORDS


def test_openrouter_summary_length_finish_reason_falls_back_without_parsing(monkeypatch):
    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}

    class FakeSession:
        def post(self, url, json=None, headers=None):
            return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    summary, retries = asyncio.run(fused_env._summarize_openrouter_batch(FakeSession(), "document", retry_budget=0))
    assert summary is None
    assert retries == 0


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


def test_web_search_retrieval_failure_is_not_an_ordinary_zero_reward(monkeypatch):
    async def fail_retrieve(*_args, **_kwargs):
        raise asyncio.TimeoutError("retrieval unavailable")

    monkeypatch.setattr(fused_env, "_retrieve_json_cached", fail_retrieve)
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "1")
    env = FusedEnvironment({"question": "Find evidence", "ground_truth": "answer"})

    observation, reward, done, info = asyncio.run(env.step(ToolCall("web_search", {"query": "q"})))
    env.answer = "wrong"

    assert observation.startswith("Search failed: TimeoutError")
    assert reward == 0.0
    assert done is False
    assert info["infra_failure"] is True
    assert env.compute_final_reward() == 0.0
    assert env.reward_debug["infra_failure"] is True
    assert env.reward_debug["infra_failure_reasons"] == ["search_retrieval_TimeoutError"]


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


def test_web_search_summary_uses_one_request_with_8192_word_input(monkeypatch):
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
                long_doc = " ".join(f"word{i}" for i in range(9000))
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
                return FakeResponse(200, {"summary": f"# Summary: summarized {total_words}"})
            raise AssertionError(url)

    monkeypatch.setattr("slime.rollout.fused_agent.env.aiohttp.ClientSession", FakeSession)
    monotonic_values = iter([30.0, 30.01, 30.5, 30.6])
    monkeypatch.setattr("slime.rollout.fused_agent.env._now_monotonic", lambda: next(monotonic_values))
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARIZE", "1")
    monkeypatch.setenv("RLLM_RETRIEVAL_SUMMARY_RETRY_BUDGET", "0")
    env = FusedEnvironment({"question": "Find evidence"})

    observation, reward, done, info = asyncio.run(env.step(fused_generate.ToolCall("web_search", {"query": "q"})))

    assert observation.startswith("summarized ")
    assert reward == 0.0
    assert done is False
    assert info["search_summary_used"] is True
    summarize_calls = [payload for url, payload in FakeSession.calls if url.endswith("/summarize")]
    assert len(summarize_calls) == 1
    assert len(summarize_calls[0]["documents"]) == 1
    summary_input = summarize_calls[0]["documents"][0]["content"]
    assert len(summary_input) == fused_env.SUMMARY_MAX_INPUT_CHARS
    assert _summary_units(summary_input) <= fused_env.SUMMARY_MAX_INPUT_WORDS
    assert summarize_calls[0]["max_length"] == 512


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

    async def fake_post(url, payload, max_retries=60, headers=None):
        calls.append(("generate", url, payload, max_retries, headers))
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
    assert calls[0][3] == 1
    assert calls[0][4] == {"X-SMG-Routing-Key": "sid"}
    assert calls[1] == (
        "abort",
        "http://127.0.0.1:30000/abort_request",
        {"rid": calls[0][2]["rid"]},
        5.0,
    )


def test_call_sglang_retries_read_error_after_aborting_old_request(monkeypatch):
    generate_calls = []
    abort_calls = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        generate_calls.append((url, dict(payload), max_retries, headers))
        if len(generate_calls) == 1:
            raise httpx.ReadError("connection closed")
        return {
            "text": "x",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 7, "x"]],
            },
        }

    class FakeClient:
        async def post(self, url, json=None, timeout=None):
            abort_calls.append((url, json, timeout))

    monkeypatch.setenv("SLIME_SGLANG_TRANSPORT_RETRY_TIMES", "1")
    monkeypatch.setenv("SLIME_SGLANG_TRANSPORT_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    monkeypatch.setattr(fused_generate.http_utils, "_http_client", FakeClient())
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
        rollout_top_p=1.0,
    )

    result = asyncio.run(fused_generate._call_sglang(args, [1, 2], {"max_new_tokens": 4}, session_id="sid"))

    assert result["output_ids"] == [7]
    assert len(generate_calls) == 2
    assert generate_calls[0][1]["rid"] != generate_calls[1][1]["rid"]
    assert all(call[2] == 1 for call in generate_calls)
    assert abort_calls == [
        (
            "http://127.0.0.1:30000/abort_request",
            {"rid": generate_calls[0][1]["rid"]},
            5.0,
        )
    ]


def test_call_sglang_aborts_native_session_request_on_direct_engine(monkeypatch):
    calls = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        calls.append(("generate", url, payload, headers))
        raise httpx.ReadError("connection closed")

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

    with pytest.raises(httpx.ReadError):
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

    assert len(calls) == 2
    assert calls[1] == (
        "abort",
        "http://engine-0/abort_request",
        {"rid": calls[0][2]["rid"]},
        5.0,
    )


def test_call_sglang_eval_uses_text_without_logprobs(monkeypatch):
    requests = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        requests.append((url, payload, headers))
        return {
            "text": "final answer",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 2,
                "cached_tokens": 1,
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
        "cached_tokens": 1,
        "completion_tokens": 2,
        "output_ids": [],
        "rid": requests[0][1]["rid"],
    }


def test_call_sglang_reuses_growing_prefixes_under_concurrent_multiturn_load(monkeypatch):
    previous_prompts = {}
    requests = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        del url, max_retries
        await asyncio.sleep(0)
        routing_key = headers["X-SMG-Routing-Key"]
        prompt = list(payload["input_ids"])
        previous = previous_prompts.get(routing_key, [])
        assert prompt[: len(previous)] == previous
        previous_prompts[routing_key] = prompt
        requests.append((routing_key, prompt))
        return {
            "text": "turn",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": len(prompt),
                "cached_tokens": len(previous),
                "completion_tokens": 1,
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
        sglang_context_length=4096,
        rollout_max_context_len=4096,
    )

    async def run_trajectory(trajectory: int):
        prompt = [1000 + trajectory, *range(7)]
        outputs = []
        for turn in range(4):
            if turn:
                prompt.extend([trajectory, turn, 2000 + turn, 3000 + turn])
            outputs.append(
                await fused_generate._call_sglang(
                    args,
                    list(prompt),
                    {"max_new_tokens": 8},
                    session_id=f"trajectory-{trajectory}",
                    evaluation=True,
                )
            )
        return outputs

    async def run_load():
        return await asyncio.gather(*(run_trajectory(index) for index in range(32)))

    outputs = asyncio.run(run_load())

    assert len(requests) == 128
    assert set(previous_prompts) == {f"trajectory-{index}" for index in range(32)}
    for trajectory_outputs in outputs:
        assert [output["prompt_tokens"] for output in trajectory_outputs] == [8, 12, 16, 20]
        assert [output["cached_tokens"] for output in trajectory_outputs] == [0, 8, 12, 16]
        assert sum(output["cached_tokens"] for output in trajectory_outputs) == 36


def test_call_sglang_rollout_only_skips_training_replay_metadata(monkeypatch):
    requests = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        requests.append(payload)
        return {
            "text": "answer",
            "output_ids": [7, 8],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 3,
                "cached_tokens": 2,
                "completion_tokens": 2,
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        debug_rollout_only=True,
        rollout_only_inference_fast_path=True,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="manual",
        rollout_top_p=0.95,
    )

    result = asyncio.run(
        fused_generate._call_sglang(
            args,
            [1, 2, 3],
            {"max_new_tokens": 4, "top_p": 0.95},
            session_id="sid",
        )
    )

    assert requests[0]["return_logprob"] is False
    assert "custom_params" not in requests[0]["sampling_params"]
    assert result == {
        "text": "answer",
        "output_ids": [7, 8],
        "output_logprobs": [],
        "finish_reason": "stop",
        "weight_version": None,
        "prompt_tokens": 3,
        "cached_tokens": 2,
        "completion_tokens": 2,
    }


def test_call_sglang_debug_rollout_keeps_training_evidence_without_fast_path(monkeypatch):
    requests = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        requests.append(payload)
        return {
            "text": "x",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 7, "x"]],
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        debug_rollout_only=True,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="manual",
        rollout_top_p=1.0,
    )

    result = asyncio.run(
        fused_generate._call_sglang(args, [1, 2], {"max_new_tokens": 1}, session_id="sid")
    )

    assert requests[0]["return_logprob"] is True
    assert result["output_ids"] == [7]
    assert result["output_logprobs"] == [-0.25]


def test_call_sglang_strict_weight_version_is_returned_and_pinned(monkeypatch):
    async def fake_post(url, payload, max_retries=60, headers=None):
        return {
            "text": "x",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 7, "x"]],
                "weight_version": "v7",
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="consistent_hashing",
    )

    result = asyncio.run(
        fused_generate._call_sglang(
            args,
            [1, 2],
            {"max_new_tokens": 1},
            session_id="sid",
            expected_weight_version="v7",
            require_weight_version=True,
        )
    )
    assert result["weight_version"] == "v7"

    with pytest.raises(fused_generate.SGLangWeightVersionError, match="changed within"):
        asyncio.run(
            fused_generate._call_sglang(
                args,
                [1, 2],
                {"max_new_tokens": 1},
                session_id="sid",
                expected_weight_version="v8",
                require_weight_version=True,
            )
        )


def test_call_sglang_training_requests_and_returns_top_p_replay(monkeypatch):
    requests = []

    async def fake_post(url, payload, max_retries=60, headers=None):
        requests.append(payload)
        return {
            "text": "x",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 7, "x"]],
                "top_p_token_ids": [7, 8],
                "top_p_token_offsets": [0, 2],
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="manual",
        rollout_top_p=0.95,
    )

    result = asyncio.run(
        fused_generate._call_sglang(
            args,
            [1, 2],
            {"max_new_tokens": 1, "top_p": 0.95, "custom_params": {"existing": True}},
            session_id="sid",
        )
    )

    assert requests[0]["sampling_params"]["custom_params"] == {
        "existing": True,
        "return_top_p_token_ids": True,
    }
    assert result["rollout_top_p_token_ids"] == [7, 8]
    assert result["rollout_top_p_token_offsets"] == [0, 2]


def test_call_sglang_training_rejects_missing_top_p_replay(monkeypatch):
    async def fake_post(url, payload, max_retries=60, headers=None):
        return {
            "text": "x",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 7, "x"]],
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        router_policy="manual",
        rollout_top_p=0.95,
    )

    with pytest.raises(ValueError, match="omitted top-p token replay metadata"):
        asyncio.run(
            fused_generate._call_sglang(
                args,
                [1, 2],
                {"max_new_tokens": 1, "top_p": 0.95},
                session_id="sid",
            )
        )


def test_call_sglang_strict_weight_version_rejects_missing(monkeypatch):
    async def fake_post(url, payload, max_retries=60, headers=None):
        return {
            "text": "x",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.25, 7, "x"]],
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000, router_policy="manual")

    with pytest.raises(fused_generate.SGLangWeightVersionError, match="omitted weight_version"):
        asyncio.run(
            fused_generate._call_sglang(
                args,
                [1, 2],
                {"max_new_tokens": 1},
                session_id="sid",
                require_weight_version=True,
            )
        )


def test_call_sglang_training_rejects_text_without_token_logprobs(monkeypatch):
    async def fake_post(url, payload, max_retries=60, headers=None):
        return {
            "text": "generated without evidence",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "weight_version": "v1",
            },
        }

    monkeypatch.setattr(fused_generate.http_utils, "post", fake_post)
    args = SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000, router_policy="manual")

    with pytest.raises(ValueError, match="without output token/logprob evidence"):
        asyncio.run(
            fused_generate._call_sglang(
                args,
                [1, 2],
                {"max_new_tokens": 4},
                session_id="sid",
                require_weight_version=True,
            )
        )


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
    evidence = tools["get_value"]()
    return {"passed": answer == evidence}
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


def test_rllm_deepresearch_eval_runs_searches_and_finish(monkeypatch):
    searches = []

    async def fake_search(action, *, retrieval_url, max_results):
        searches.append((action.arguments["query"], retrieval_url, max_results))
        return f"summary for {action.arguments['query']}", {"tool_return_error": 0, "refine_error": 0}

    monkeypatch.setattr(fused_generate, "run_rllm_deepresearch_search", fake_search)
    first = '<tool_call>{"name":"local_search","arguments":{"query":"alpha"}}</tool_call>'
    second = '<tool_call>{"name":"local_search","arguments":{"query":"beta"}}</tool_call>'
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"Paris"}}</tool_call>'
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Paris", metadata={"question": "Where?"}),
        [{"text": first}, {"text": second}, {"text": finish}],
        {
            "FUSED_HARNESS": "rllm_deepresearch",
            "RETRIEVAL_SERVER_URL": "http://retriever",
        },
        evaluation=True,
    )

    assert [item[0] for item in searches] == ["alpha", "beta"]
    assert all(item[2] == 10 for item in searches)
    assert result[0].response == finish
    assert result[0].reward == 1.0
    assert result[0].metadata["fused_termination"] == "env_done"
    assert result[0].metadata["fused_tool_call_turns"] == 2


def test_cut_bill_eval_runs_search_and_finishes_with_boxed_answer(monkeypatch):
    searches = []
    sampling_params_seen = []

    async def fake_search(action, *, retrieval_url, max_results):
        searches.append((action.arguments["query"], retrieval_url, max_results))
        return "Your query is: capital. The search results are summarized as following: Paris.", {
            "tool_return_error": 0,
            "refine_error": 0,
        }

    monkeypatch.setattr(fused_generate, "run_cut_bill_search", fake_search)
    search = "<think>\nI should search.\n</think>\n\n" '<tool_call>\n{"name": "local_search", "arguments": {"query": "capital"}}\n</tool_call>'
    answer = "<think>\nThe result identifies Paris.\n</think>\n\n\\boxed{Paris}"
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Paris", metadata={"question": "Where?"}),
        [{"text": search}, {"text": answer}],
        {
            "FUSED_HARNESS": "cut_bill",
            "RETRIEVAL_SERVER_URL": "http://retriever",
            "PER_STEP_MAX_TOKENS": "8192",
        },
        evaluation=True,
        sampling_params_seen=sampling_params_seen,
    )

    assert searches == [("capital", "http://retriever", 10)]
    assert result[0].response == answer
    assert result[0].reward == 1.0
    assert result[0].metadata["fused_termination"] == "env_done"
    assert result[0].metadata["fused_tool_call_turns"] == 1
    assert [params["max_new_tokens"] for params in sampling_params_seen] == [8192, 8192]


def test_cut_bill_eval_uses_cut_bill_duplicate_search_termination(monkeypatch):
    searches = []

    async def fake_search(action, **_kwargs):
        searches.append(action.arguments["query"])
        return "summary", {"tool_return_error": 0, "refine_error": 0}

    monkeypatch.setattr(fused_generate, "run_cut_bill_search", fake_search)
    repeated = '<tool_call>{"name":"local_search","arguments":{"query":"same"}}</tool_call>'
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="x", metadata={"question": "Search"}),
        [{"text": repeated}, {"text": repeated}],
        {"FUSED_HARNESS": "cut_bill", "RETRIEVAL_SERVER_URL": "http://retriever"},
        evaluation=True,
    )

    assert searches == ["same"]
    assert result[0].metadata["fused_termination"] == "cut_bill_duplicate_search"
    assert result[0].metadata["duplicate_search_detected"] is True


def test_rllm_deepresearch_eval_keeps_no_tool_call_termination():
    response = "I cannot find enough information."
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Paris", metadata={"question": "Where?"}),
        [{"text": response}],
        {
            "FUSED_HARNESS": "rllm_deepresearch",
            "RETRIEVAL_SERVER_URL": "http://retriever",
        },
        evaluation=True,
    )

    assert result[0].response == response
    assert result[0].metadata["fused_termination"] == "rllm_dr_no_tool_call"


def test_rllm_deepresearch_eval_accepts_multiple_tool_calls(monkeypatch):
    searches = []

    async def fake_search(action, **_kwargs):
        searches.append(action.arguments["query"])
        return f"result for {action.arguments['query']}", {"tool_return_error": 0, "refine_error": 0}

    monkeypatch.setattr(fused_generate, "run_rllm_deepresearch_search", fake_search)
    response = '<tool_call>{"name":"local_search","arguments":{"query":"alpha"}}</tool_call>' '<tool_call>{"name":"local_search","arguments":{"query":"beta"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="Paris", metadata={"question": "Where?"}),
        [
            {"text": response},
            {"text": '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"Paris"}}</tool_call>'},
        ],
        {
            "FUSED_HARNESS": "rllm_deepresearch",
            "RETRIEVAL_SERVER_URL": "http://retriever",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "1.0",
        },
        evaluation=True,
    )

    assert searches == ["alpha", "beta"]
    assert result[0].metadata["fused_termination"] == "env_done"


def test_rllm_deepresearch_eval_stops_on_duplicate_search(monkeypatch):
    searches = []

    async def fake_search(action, **_kwargs):
        searches.append(action.arguments["query"])
        return "summary", {"tool_return_error": 0, "refine_error": 0}

    monkeypatch.setattr(fused_generate, "run_rllm_deepresearch_search", fake_search)
    repeated = '<tool_call>{"name":"local_search","arguments":{"query":"same"}}</tool_call>'
    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label="x", metadata={"question": "Search"}),
        [{"text": repeated}, {"text": repeated}],
        {"FUSED_HARNESS": "rllm_dr", "RETRIEVAL_SERVER_URL": "http://retriever"},
        evaluation=True,
    )

    assert searches == ["same"]
    assert result[0].reward == 0.0
    assert result[0].metadata["fused_termination"] == "rllm_dr_duplicate_search"
    assert result[0].metadata["duplicate_search_detected"] is True


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


def test_new_search_query_resets_repeated_search_strikes():
    same = _search_call("same query")
    different = _search_call("different query")

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search before answer"}),
        [
            {"text": same},
            {"text": same},
            {"text": different},
            {"text": same},
            {"text": same},
            {"text": same},
        ],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "2",
        },
        evaluation=True,
    )

    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["fused_termination"] == "repeated_query_early_stop"
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["duplicate_query_count"] == 2


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
        {
            rollout_visualization._UNMASKED_TOKEN_STYLE,
            rollout_visualization._UNMASKED_TOOL_CALL_STYLE,
            rollout_visualization._REWARD_POS_STYLE,
        },
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
    # The exact generated action stays policy-trainable. TiTO appends only the
    # official model-turn boundary and role:tool result as masked context.
    assert _masked_text(sample) == first + finish
    full_text = "".join(chr(tok) for tok in sample.tokens)
    assert first in full_text
    assert '<|turn>tool\n{"echo": "before-finish"}<turn|>' in full_text
    assert "<tool_response>\nExecution output" not in full_text
    # The full served context is preserved: system/user prompt precedes the turns.
    assert "system" in full_text
    assert "<|turn>model" not in _masked_text(sample)
    assert len(sample.loss_mask) == sample.response_length
    assert sum(sample.loss_mask) == len(first) + len(finish)


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
    # Raw first-turn tool calls keep training; official role:tool turns are context.
    assert _masked_text(sample) == first_turn + finish
    full_text = "".join(chr(tok) for tok in sample.tokens)
    assert '<|turn>tool\n{"echo": "alpha"}<turn|>' in full_text
    assert '<|turn>tool\n{"echo": "beta"}<turn|>' in full_text
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

    assert len(result) == 21
    assert all(sample.metadata["segment_count"] == 21 for sample in result)
    assert all(sample.metadata["fused_tito_boundary_count"] == 20 for sample in result)
    assert all(sample.metadata["fused_tito_prompt_prefix_mismatch_turns"] == 20 for sample in result)
    assert [sample.metadata["turn_spans"][0]["turn_index"] for sample in result] == list(range(1, 22))
    assert all(len(sample.metadata["turn_spans"]) == 1 for sample in result)
    episode = result[0].metadata["rllm_episode"]
    assert episode["metadata"]["segment_count"] == 21
    assert ("segment_num", "21") in rollout_visualization._episode_summary_rows(result[0], episode, None)
    unmasked_text = "".join(_masked_text(sample) for sample in result)
    assert "<tool_response>" not in unmasked_text
    # A forced boundary strips the new segment's leading prompt. Historical
    # tool responses therefore remain context-only and are absent from the
    # response-aligned policy masks.
    assert "<tool_response>" not in "".join(_policy_unmasked_text(sample) for sample in result)
    for idx in range(len(tool_turns)):
        assert f"drift evidence query {idx:02d}" in unmasked_text
    assert '"name":"finish"' in unmasked_text
    assert "\\\\boxed{answer}" in unmasked_text

    visual_unmasked = "".join(
        _visualized_text_by_styles(
            sample,
            tokenizer,
            {
                rollout_visualization._UNMASKED_TOKEN_STYLE,
                rollout_visualization._UNMASKED_TOOL_CALL_STYLE,
                rollout_visualization._REWARD_POS_STYLE,
            },
        )
        for sample in result
    )
    for idx in range(len(tool_turns)):
        query = f"drift evidence query {idx:02d}"
        assert query in visual_unmasked
    assert '"name":"finish"' in visual_unmasked
    assert "\\\\boxed{answer}" in visual_unmasked

    segment_view = rollout_visualization._episode_token_mask_view(
        result[0],
        tokenizer,
        related_samples=list(reversed(result)),
    )
    assert len(segment_view.renderables) == 21
    assert segment_view.renderables[0].title.plain == "Segment 1/21 | Step 1"
    assert segment_view.renderables[-1].title.plain.startswith("Segment 21/21 | Step 21")
    for index, panel in enumerate(segment_view.renderables[:-1]):
        assert f"drift evidence query {index:02d}" in panel.renderable.plain
    assert '"name":"finish"' in segment_view.renderables[-1].renderable.plain


@pytest.mark.parametrize("tokenizer", [FakeQwen3ChatTemplateTokenizer(), FakeQwen35ChatTemplateTokenizer()])
def test_qwen3_family_incremental_tito_avoids_assistant_replay_drift_segments(tokenizer):
    if isinstance(tokenizer, FakeQwen35ChatTemplateTokenizer):
        tool_turns = ["<tool_call>\n<function=web_search>\n" f"<parameter=query>evidence query {index}</parameter>\n" "<parameter=max_results>3</parameter>\n</function>\n</tool_call>" for index in range(3)]
        finish = "<tool_call>\n<function=finish>\n<parameter=command>submit</parameter>\n" "<parameter=result>\\boxed{answer}</parameter>\n</function>\n</tool_call>"
    else:
        tool_turns = [_search_call(f"evidence query {index}") for index in range(3)]
        finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'
    tokenizer.drift_assistant_end = True
    prompt_ids_seen: list[list[int]] = []

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Find evidence"}),
        [{"text": text} for text in [*tool_turns, finish]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "False",
            "FUSED_MAX_STEPS": "8",
            "SLIME_FUSED_STRICT_TITO": "True",
        },
        tokenizer=tokenizer,
        prompt_ids_seen=prompt_ids_seen,
    )

    assert len(result) == 1
    assert result[0].metadata["segment_count"] == 1
    assert result[0].metadata["fused_tito_boundary_count"] == 0
    assert result[0].metadata["fused_tito_incremental_turns"] == 3
    assert result[0].metadata["fused_tito_prompt_prefix_mismatch_turns"] == 0
    assert result[0].metadata["fused_tito_incremental_tokenization_failed_turns"] == 0
    for previous_prompt, generated_text, next_prompt in zip(prompt_ids_seen, tool_turns, prompt_ids_seen[1:]):
        exact_previous_prefix = previous_prompt + [ord(character) for character in generated_text]
        assert next_prompt[: len(exact_previous_prefix)] == exact_previous_prefix


def test_strict_tito_rejects_historical_thinking_discard():
    with pytest.raises(ValueError, match="Strict TiTO requires append-only history"):
        _run_generate_with_fake_sglang(
            Sample(prompt="placeholder", label="answer", metadata={"question": "Find evidence"}),
            [{"text": _search_call("evidence")}],
            {
                "FUSED_DISABLE_THINKING": "False",
                "FUSED_DISCARD_HISTORICAL_THINKING": "True",
                "SLIME_FUSED_STRICT_TITO": "True",
            },
            tokenizer=FakeQwen35ChatTemplateTokenizer(),
        )


def test_group_visualization_orders_and_renders_every_segment_without_fuzzy_matching():
    tokenizer = FakeChatTemplateTokenizer()
    first_action = _search_call("first evidence")
    second_action = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"\\\\boxed{answer}"}}</tool_call>'

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Find evidence"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    first_tokens = prompt + [ord(c) for c in first_action]
    shared_episode = {"trajectories": [{"steps": [{}, {}]}]}
    first = Sample(
        rollout_id=7,
        tokens=first_tokens,
        response_length=len(first_action),
        loss_mask=[1] * len(first_action),
        reward=1.0,
        metadata={
            "rllm_episode": shared_episode,
            "parent_traj_id": "episode-7",
            "segment_index": 0,
            "segment_count": 2,
            "turn_spans": [
                {
                    "turn_index": 1,
                    "response_token_start": len(prompt),
                    "response_token_end": len(first_tokens),
                    "trained": True,
                    "truncated": False,
                }
            ],
        },
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
        metadata={
            "rllm_episode": shared_episode,
            "parent_traj_id": "episode-7",
            "segment_index": 1,
            "segment_count": 2,
            "turn_spans": [
                {
                    "turn_index": 2,
                    "response_token_start": len(replay_prompt),
                    "response_token_end": len(second_tokens),
                    "trained": True,
                    "truncated": False,
                }
            ],
        },
    )

    single_rendered = rollout_visualization._token_mask_text(second, tokenizer)
    assert single_rendered is not None
    single_unmasked = _visualized_text_by_styles_from_rendered(
        single_rendered,
        {
            rollout_visualization._UNMASKED_TOKEN_STYLE,
            rollout_visualization._UNMASKED_TOOL_CALL_STYLE,
            rollout_visualization._REWARD_POS_STYLE,
        },
    )
    single_masked_tool = _visualized_text_by_style(second, tokenizer, rollout_visualization._MASKED_TOOL_CALL_STYLE)
    assert "first evidence" not in single_unmasked
    assert "first evidence" in single_masked_tool

    ordered = rollout_visualization._ordered_segment_samples([second, first])
    assert ordered == [first, second]
    segment_view = rollout_visualization._episode_token_mask_view(
        first,
        tokenizer,
        related_samples=[second, first],
    )
    from rich.panel import Panel

    assert len(segment_view.renderables) == 2
    assert all(isinstance(renderable, Panel) for renderable in segment_view.renderables)
    assert [renderable.border_style for renderable in segment_view.renderables] == [
        rollout_visualization._TOKEN_MASK_BORDER_STYLE,
        rollout_visualization._TOKEN_MASK_BORDER_STYLE,
    ]
    assert [renderable.title.plain for renderable in segment_view.renderables] == [
        "Segment 1/2 | Step 1",
        "Segment 2/2 | Step 2",
    ]
    combined_unmasked = "".join(
        _visualized_text_by_styles_from_rendered(
            rollout_visualization._token_mask_text(segment, tokenizer),
            {
                rollout_visualization._UNMASKED_TOKEN_STYLE,
                rollout_visualization._UNMASKED_TOOL_CALL_STYLE,
                rollout_visualization._REWARD_POS_STYLE,
            },
        )
        for segment in ordered
    )
    assert "first evidence" in combined_unmasked
    assert '"name":"finish"' in combined_unmasked

    with pytest.raises(ValueError, match="expected 2 trajectory segments, got 1"):
        rollout_visualization._ordered_segment_samples([first])
    with pytest.raises(ValueError, match="expected 2 trajectory segments, got 1"):
        rollout_visualization._episode_token_mask_view(first, tokenizer, related_samples=[first])

    second.metadata.pop("turn_spans")
    with pytest.raises(ValueError, match="segment 1 is missing turn_spans provenance"):
        rollout_visualization._episode_token_mask_view(
            first,
            tokenizer,
            related_samples=[first, second],
        )


def test_token_mask_view_keeps_whitespace_between_historical_tool_calls_masked():
    tokenizer = FakeChatTemplateTokenizer()
    first = _search_call("first")
    second = _search_call("second")
    finish = _finish_call()
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Find evidence"},
            {"role": "assistant", "content": first + "\n" + second},
            {"role": "user", "content": "<tool_response>ok</tool_response>"},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    sample = Sample(
        tokens=prompt + [ord(char) for char in finish],
        response_length=len(finish),
        loss_mask=[1] * len(finish),
    )

    rendered = rollout_visualization._token_mask_text(sample, tokenizer)
    assert rendered is not None
    masked_tool_calls = _visualized_text_by_styles_from_rendered(rendered, {rollout_visualization._MASKED_TOOL_CALL_STYLE})
    unmasked_tool_calls = _visualized_text_by_styles_from_rendered(rendered, {rollout_visualization._UNMASKED_TOOL_CALL_STYLE})
    assert "first" in masked_tool_calls
    assert "second" in masked_tool_calls
    assert '"name":"finish"' in unmasked_tool_calls

    separator_start = rendered.plain.index("</tool_call>\\n<tool_call>") + len("</tool_call>")
    separator_style = next(str(span.style) for span in rendered.spans if span.start <= separator_start < span.end)
    assert separator_style == rollout_visualization._MASKED_TOKEN_STYLE


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
    repeated = _search_call("unique query 17")
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
        {
            rollout_visualization._UNMASKED_TOKEN_STYLE,
            rollout_visualization._UNMASKED_TOOL_CALL_STYLE,
            rollout_visualization._REWARD_NEG_STYLE,
        },
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


def test_gemma4_visualization_separates_native_thought_channel_from_action():
    action = (
        '<|tool_call>call:finish{command:<|"|>submit<|"|>,'
        'result:<|"|>Austrocallerya megasperma<|"|>}<tool_call|><eos>'
    )
    response = (
        "<|channel>thought\n"
        "The evidence supports the modern accepted name.\n"
        "I will submit the scientific name.\n"
        f"<channel|>{action}"
    )
    step = {
        "model_response": response,
        "thought": "<|channel>thought\nstale fallback<channel|>",
    }

    assert rollout_visualization._step_thinking_and_response(step) == (
        "The evidence supports the modern accepted name.\nI will submit the scientific name.",
        action,
    )
    assert rollout_visualization._step_thinking_and_response(
        {
            "model_response": f"<|channel>thought\n<channel|>{action}",
            "thought": "stale fallback",
        }
    ) == ("", action)


def test_step_visualization_separates_thinking_panel_with_blank_line():
    from rich.console import Console

    console = Console(width=80, force_terminal=False, color_system=None, record=True)
    rollout_visualization._print_step(
        console,
        {
            "thought": "internal plan",
            "response": "tool call",
            "observation": "tool result",
            "reward": 0.0,
            "done": False,
        },
        0,
        1,
        max_chars=200,
    )

    lines = console.export_text().splitlines()
    thinking_line = next(index for index, line in enumerate(lines) if "Thinking" in line)
    assert thinking_line > 0
    assert lines[thinking_line - 1].strip("│ ") == ""


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


def test_eval_accepts_mixed_tool_and_submit_call():
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
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "1.0",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["credit_assignment_event"] is None
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_traj_steps"] == 3
    assert "eval_response_anomalies" not in sample.metadata
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


def test_eval_stops_repeated_search_with_distinct_termination_reason():
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
    assert sample.metadata["fused_termination"] == "repeated_query_early_stop"
    assert sample.metadata["duplicate_search_detected"] is True
    assert sample.metadata["duplicate_query_count"] == 1
    assert sample.metadata["fused_traj_steps"] == 2
    assert sample.tokens == []


def test_eval_accepts_finish_after_repeated_search_warning():
    search = _search_call("same query")
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"answer"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "answer"}, metadata={"question": "Search then answer"}),
        [{"text": search}, {"text": search}, {"text": finish}],
        {
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "2",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "1.0",
        },
        evaluation=True,
    )

    sample = result[0]
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_traj_steps"] == 3
    assert sample.response == finish


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


def test_eval_accepts_multiple_tool_calls_in_one_response(tmp_path: Path):
    burst = "".join(_echo_call(f"burst{i}") for i in range(2))

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Trigger tool burst"),
        [{"text": burst}, {"text": _finish_call()}],
        {
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "1.0",
        },
        evaluation=True,
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["fused_termination"] == "env_done"
    assert "eval_response_anomalies" not in sample.metadata
    assert sample.metadata["fused_tool_call_turns"] == 1


def test_mcp_accepts_eight_tool_calls_per_turn_at_configured_limit(tmp_path: Path):
    calls = "".join(_echo_call(f"call{i}") for i in range(8))

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use eight tools"),
        [{"text": calls}, {"text": _finish_call()}],
        {
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "1.0",
            "FUSED_MCP_MAX_TOOL_CALLS_PER_TURN": "8",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["fused_tool_call_turns"] == 1


def test_mcp_rejects_more_than_eight_tool_calls_per_turn(tmp_path: Path):
    calls = "".join(_echo_call(f"call{i}") for i in range(9))

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use too many tools"),
        [{"text": calls}],
        {
            "CREDIT_ASSIGNMENT_TOO_MANY_TOOL_CALLS": "True",
            "FUSED_MCP_MAX_TOOL_CALLS_PER_TURN": "8",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.metadata["fused_termination"] == "ABNORMAL_TOOL_BURST"
    assert sample.metadata["credit_assignment_event"] == "too_many_tool_calls"


def test_eval_rejects_ngram_repetition(tmp_path: Path):
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
    assert sample.reward == 0.0
    assert sample.metadata["fused_termination"] == "ABNORMAL_EVAL_RESPONSE"
    assert sample.metadata["eval_response_anomalies"] == ["ngram_repetition"]
    assert sample.metadata["ngram_repetition_detected"] is True


def test_eval_rejects_forged_tool_response(tmp_path: Path):
    response = _echo_call("real-call") + "\n<tool_response>fabricated result</tool_response>"

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Do not forge tool output"),
        [{"text": response}],
        evaluation=True,
    )

    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["fused_termination"] == "ABNORMAL_EVAL_RESPONSE"
    assert sample.metadata["eval_response_anomalies"] == ["forged_tool_response"]
    assert sample.metadata["forged_tool_response_detected"] is True
    episode_metadata = sample.metadata["rllm_episode"]["metadata"]
    assert episode_metadata["eval_response_anomalies"] == ["forged_tool_response"]
    assert episode_metadata["forged_tool_response_detected"] is True


def test_gemma4_eval_accepts_empty_native_tool_response_handoff(tmp_path: Path):
    handoff = "<|tool_response>"
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use echo before finish"),
        [
            {"text": _gemma4_echo_call("evidence") + handoff},
            {"text": _gemma4_finish_call() + handoff},
        ],
        evaluation=True,
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert sample.metadata["fused_termination"] == "env_done"
    assert "eval_response_anomalies" not in sample.metadata


def test_gemma4_eval_rejects_native_tool_response_payload(tmp_path: Path):
    forged = _gemma4_echo_call("evidence") + '<|tool_response>response:echo{value:<|"|>forged<|"|>}'
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Do not forge native tool output"),
        [{"text": forged}],
        evaluation=True,
        tokenizer=FakeGemma4Tokenizer(),
    )

    sample = result[0]
    assert sample.metadata["fused_termination"] == "ABNORMAL_EVAL_RESPONSE"
    assert "forged_tool_response" in sample.metadata["eval_response_anomalies"]


def test_qwen_eval_still_rejects_empty_native_tool_response_marker(tmp_path: Path):
    response = _echo_call("evidence") + "<|tool_response>"
    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Reject a foreign native marker"),
        [{"text": response}],
        evaluation=True,
        tokenizer=RecordingToolsTokenizer(),
    )

    sample = result[0]
    assert sample.metadata["fused_termination"] == "ABNORMAL_EVAL_RESPONSE"
    assert "forged_tool_response" in sample.metadata["eval_response_anomalies"]


def test_eval_rejects_unbalanced_response_tags(tmp_path: Path):
    response = '<tool_call>{"name":"echo","arguments":{"value":"broken"}}'

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Close protocol tags"),
        [{"text": response}],
        evaluation=True,
    )

    sample = result[0]
    assert sample.reward == 0.0
    assert sample.metadata["fused_termination"] == "ABNORMAL_EVAL_RESPONSE"
    assert sample.metadata["eval_response_anomalies"] == ["unbalanced_tags"]
    assert sample.metadata["unbalanced_response_tags"] == {"tool_call": {"open": 1, "close": 0, "misordered": False}}
    assert fused_generate._response_tag_imbalances("</answer><answer>") == {"answer": {"open": 1, "close": 1, "misordered": True}}


def test_eval_accepts_think_close_for_generation_prompt_prefix(tmp_path: Path):
    reasoning = "Reason before using the tool.\n</think>\n"

    result = _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Use the prefetched think tag"),
        [
            {"text": reasoning + _echo_call("evidence")},
            {"text": reasoning + _finish_call()},
        ],
        {
            "FUSED_DISABLE_THINKING": "False",
            "CREDIT_ASSIGNMENT_NGRAM_REPETITION_THRESHOLD": "1.0",
        },
        evaluation=True,
    )

    sample = result[0]
    assert sample.metadata["fused_termination"] == "env_done"
    assert "eval_response_anomalies" not in sample.metadata
    assert (
        fused_generate._response_tag_imbalances(
            "reasoning</think>",
            allow_leading_think_close=True,
        )
        == {}
    )
    assert (
        fused_generate._response_tag_imbalances(
            "<think>reasoning</think>",
            allow_leading_think_close=True,
        )
        == {}
    )
    assert (
        fused_generate._response_tag_imbalances(
            _echo_call("no reasoning tags"),
            allow_leading_think_close=True,
        )
        == {}
    )
    assert fused_generate._response_tag_imbalances(
        "reasoning</think></think>",
        allow_leading_think_close=True,
    ) == {"think": {"open": 0, "close": 2, "misordered": True}}


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

    assert (
        fused_generate._credit_assignment_loss_mask(
            output_len=len(base_loss_mask),
            turn_index=0,
            credit_event="mixed_tool_and_answer",
            credit_step_index=0,
            base_loss_mask=base_loss_mask,
        )
        == base_loss_mask
    )

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
        credit_event="tool_parser_error",
        credit_step_index=0,
        error_attribution="unattributable",
        base_loss_mask=base_loss_mask,
    ) == base_loss_mask

    assert fused_generate._credit_assignment_loss_mask(
        output_len=len(base_loss_mask),
        turn_index=0,
        credit_event="too_many_tool_calls",
        credit_step_index=0,
        action_span=(1, 6),
        base_loss_mask=base_loss_mask,
    ) == [0, 0, 1, 1, 0, 1, 0]


def test_parser_error_action_span_is_capped_to_configured_tail_window():
    assert fused_generate._credit_assignment_loss_mask(
        output_len=12,
        turn_index=0,
        credit_event="tool_parser_error",
        credit_step_index=0,
        parser_error_token_window=4,
        action_span=(1, 10),
        base_loss_mask=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    ) == [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]


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
    assert sample.metadata["credit_assignment_error_attribution"] == "localized"
    assert bad in _masked_text(sample)
    assert good not in _policy_masked_text(sample)
    assert _policy_masked_text(sample) == bad


@pytest.mark.parametrize(
    ("tokenizer", "bad_response", "expected_model"),
    [
        (FakeChatTemplateTokenizer(), "I cannot produce a tool call here.", None),
        (
            FakeGemma4Tokenizer(),
            '<|tool_call>call:echo{value:{nested:[1,2}<tool_call|>',
            "gemma4",
        ),
    ],
)
def test_tool_parser_errors_are_appended_to_run_log(
    monkeypatch, tmp_path: Path, tokenizer, bad_response: str, expected_model: str | None
):
    episode_dir = tmp_path / "run" / "logs" / "episodes"
    monkeypatch.setenv("SLIME_EPISODE_LOG_DIR", str(episode_dir))

    _run_generate_with_fake_sglang(
        _local_mcp_sample(tmp_path, question="Persist the parser failure"),
        [{"text": bad_response}],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_TOOL_PARSER_ERROR": "True",
        },
        tokenizer=tokenizer,
    )

    log_path = episode_dir.parent / "tool_parser_errors.log"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["session_id"]
    assert record["step"] == 0
    assert record["response"] == bad_response
    assert record["response_length"] == len(bad_response)
    assert record["errors"]
    assert "error_kinds" in record
    assert "error_spans" in record
    assert record["parser"]
    assert record["model"] == expected_model


def test_structurally_incomplete_tool_call_penalizes_only_malformed_action(tmp_path: Path):
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


def test_profile_tool_times_accumulate_across_steps():
    retrieval, mcp = fused_generate._accumulate_profile_tool_times(
        [
            {"tools/search_retrieve_elapsed_s": 1.25, "tools/search_summary_elapsed_s": 0.5},
            {"tools/mcp_tool_elapsed_s": 2.0},
        ],
        0.25,
        0.75,
    )

    assert retrieval == pytest.approx(2.0)
    assert mcp == pytest.approx(2.75)


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
    assert good in _policy_unmasked_text(sample)
    assert reasoning in _policy_unmasked_text(sample)
    assert "<tool_response>" in _policy_unmasked_text(sample)


def test_nonconsecutive_repeated_search_does_not_trigger_credit_assignment():
    prior_queries = [f"q{i:02d}" for i in range(12)]
    prior_turns = [f'<tool_call>{{"name":"web_search","arguments":{{"query":"{query}"}}}}</tool_call>' for query in prior_queries]
    reasoning = "<think>Try the same query again.</think>\n"
    repeated = '<tool_call>{"name":"web_search","arguments":{"query":"q07"}}</tool_call>'
    finish = '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"x"}}</tool_call>'

    result = _run_generate_with_fake_sglang(
        Sample(prompt="placeholder", label={"answer": "x"}, metadata={"question": "Search many times"}),
        [{"text": text} for text in [*prior_turns, reasoning + repeated, finish]],
        {
            "CREDIT_ASSIGNMENT_ENABLE": "True",
            "CREDIT_ASSIGNMENT_REPEATED_SEARCH_QUERY": "True",
            "FUSED_MAX_STEPS": "20",
            "FUSED_REPEATED_SEARCH_MAX_STRIKES": "1",
        },
    )

    assert len(result) == 1
    sample = result[0]
    assert sample.reward == 1.0
    assert sample.metadata["fused_termination"] == "env_done"
    assert sample.metadata["credit_assignment_event"] is None


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
    ) == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


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
    test_historical_thinking_compaction_removes_complete_assistant_blocks_without_mutation()
    with tempfile.TemporaryDirectory() as tmp:
        test_enabled_thinking_discards_prior_thoughts_and_emits_prompt_equal_segments(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_prompt_equal_is_independent_of_historical_thinking_discard(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_eval_rollout_discards_prior_thinking_before_next_assistant_step(Path(tmp))
    test_render_prompt_ids_accepts_batch_encoding_like_object()
    test_initial_messages_include_tool_prompt()
