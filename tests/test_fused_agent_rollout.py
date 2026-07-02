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
from slime.rollout.fused_agent.parser import QwenToolParser, tool_schema
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


def _run_generate_with_fake_sglang(
    sample: Sample,
    calls: list[dict],
    env: dict[str, str] | None = None,
    *,
    evaluation: bool = False,
    tokenizer=None,
):
    async def fake_call_sglang(args, prompt_ids, sampling_params, *, session_id):
        item = calls.pop(0)
        text = item["text"]
        return {
            "output_ids": [ord(c) for c in text],
            "output_logprobs": [-0.1] * len(text),
            "finish_reason": item.get("finish_reason", "stop"),
        }

    old_generate_state = fused_generate.GenerateState
    old_call = fused_generate._call_sglang
    old_env = {key: os.environ.get(key) for key in (env or {})}
    if env:
        os.environ.update(env)
    fake_tokenizer = tokenizer or FakeTokenizer()
    fused_generate.GenerateState = lambda args: SimpleNamespace(tokenizer=fake_tokenizer)
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

    assert step["thought"] == action
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
    assert mixed in _masked_text(sample)
    assert _policy_masked_text(sample) == first + mixed + final


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
    assert "Repeated search query detected" not in _policy_unmasked_text(sample)


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
