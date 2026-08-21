import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import openrouter_evals

NUM_GPUS = 0


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    def __init__(self):
        self.calls = []
        self.completions = iter(
            [
                '<tool_call>{"name":"web_search","arguments":{"query":"Ada Lovelace"}}</tool_call>',
                '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"1815"}}</tool_call>',
            ]
        )

    async def post(self, url, json):
        self.calls.append((url, json))
        if url == "/chat/completions":
            return _Response({"choices": [{"message": {"content": next(self.completions)}}]})
        return _Response(
            {
                "result": [
                    {
                        "title": "Ada Lovelace",
                        "search_snippet": True,
                        "document": {"contents": "Ada Lovelace was born in 1815."},
                    }
                ]
            }
        )


class _PrematureFinishClient(_Client):
    def __init__(self):
        super().__init__()
        self.completions = iter(
            [
                '<tool_call>{"name":"finish","arguments":{"result":"guess"}}</tool_call>',
                '<tool_call>{"name":"web_search","arguments":{"query":"Ada Lovelace birth year"}}</tool_call>',
                '<tool_call>{"name":"finish","arguments":{"result":"1815"}}</tool_call>',
            ]
        )


def test_gem_messages_declare_search_tools_and_parser_format():
    messages, parser = openrouter_evals.gem_messages("When was Ada Lovelace born?", "deepseek/deepseek-v4-flash-0731")

    system = messages[0]["content"]
    assert "You have access to a web_search tool" in system
    assert '"name": "web_search"' in system
    assert '"name": "finish"' in system
    assert '<tool_call>' in system
    assert parser.parse('<tool_call>{"name":"finish","arguments":{"result":"1815"}}</tool_call>')[0].name == "finish"


def test_gem_messages_strip_conflicting_benchmark_answer_instruction():
    prompt = (
        "What country lies north of Chad?"
        "When ready, output the final answer enclosed in <answer> and </answer> tags. "
        "Do not generate any content after the </answer> tag."
    )

    messages, _ = openrouter_evals.gem_messages(prompt, "deepseek/deepseek-v4-flash-0731")

    assert "<question>\nWhat country lies north of Chad?\n</question>" in messages[1]["content"]
    assert "output the final answer enclosed in <answer>" not in messages[1]["content"]


def test_openrouter_launcher_defaults_to_gem_harness():
    launcher = Path(openrouter_evals.__file__).with_name("evals.sh").read_text(encoding="utf-8")

    assert '[ "${MODEL_SERIES}" = "openrouter" ] && [ "${harness_explicit}" = "false" ]' in launcher
    assert "FUSED_HARNESS=gem" in launcher


def test_run_gem_executes_search_and_returns_finish_result():
    client = _Client()
    args = SimpleNamespace(
        model="deepseek/deepseek-v4-flash-0731",
        max_steps=4,
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        max_retries=1,
        retrieval_max_results=10,
        retrieval_url="http://serper.test",
    )

    answer, trace, status = asyncio.run(openrouter_evals.run_gem(client, asyncio.Semaphore(1), args, "question"))

    assert (answer, status) == ("1815", "completed")
    assert [step["actions"][0]["name"] for step in trace] == ["web_search", "finish"]
    assert client.calls[1][0] == "http://serper.test/retrieve"
    second_completion_messages = client.calls[2][1]["messages"]
    observation = second_completion_messages[-2]["content"]
    assert observation.startswith(
        "<tool_response>\nExecution output of [web_search]:\n"
        "[Result 1] Title: Ada Lovelace Snippet: Ada Lovelace was born in 1815."
    )
    assert "Ada Lovelace was born in 1815" in observation
    assert observation.endswith("\n</tool_response>")


def test_run_gem_requires_search_before_finish():
    client = _PrematureFinishClient()
    args = SimpleNamespace(
        model="deepseek/deepseek-v4-flash-0731",
        max_steps=4,
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
        max_retries=1,
        retrieval_max_results=10,
        retrieval_url="http://serper.test",
    )

    answer, trace, status = asyncio.run(openrouter_evals.run_gem(client, asyncio.Semaphore(1), args, "question"))

    assert (answer, status) == ("1815", "completed")
    assert [step["actions"][0]["name"] for step in trace] == ["finish", "web_search", "finish"]
    assert "call web_search at least once" in trace[0]["observation"]


def test_write_trajectories_uses_pretty_utf8_json(tmp_path):
    output = tmp_path / "trajectories.json"

    openrouter_evals.write_trajectories(output, [(("b", 1, 0), {"response": "北京"}), (("a", 0, 0), {"response": "上海"})])

    text = output.read_text(encoding="utf-8")
    assert text.startswith('{\n    "training_step": 0,')
    payload = __import__("json").loads(text)
    assert payload["num_episodes"] == 2
    assert payload["trajectories"] == [{"response": "上海"}, {"response": "北京"}]
    assert "北京" in text
    assert "\\u5317" not in text


def test_judge_trajectory_uses_final_finish_call_and_records_binary_reward(monkeypatch):
    captured = {}

    async def fake_reward(args, sample, evaluation=False):
        captured.update(args=args, sample=sample, evaluation=evaluation)
        sample.metadata["grm"] = {"model": args.grm_model, "score": 1.0}
        return 1.0

    monkeypatch.setattr(openrouter_evals.openrouter_grm, "reward_func", fake_reward)
    trace = [
        {"response": '<tool_call>{"name":"web_search","arguments":{"query":"Ada"}}</tool_call>', "actions": [{"name": "web_search"}]},
        {
            "response": '<tool_call>{"name":"finish","arguments":{"result":"1815"}}</tool_call>',
            "actions": [{"name": "finish"}],
            "done": True,
        },
    ]
    args = SimpleNamespace(grm_model="google/gemini-3-flash-preview")

    sample = asyncio.run(
        openrouter_evals.judge_trajectory(
            args,
            index=0,
            prompt="When?",
            label="1815",
            response="1815",
            trace=trace,
            source="bamboogle",
            metadata={"strict_exact_match": True},
        )
    )

    assert sample.reward == 1.0
    assert captured["evaluation"] is True
    assert '"name":"finish"' in captured["sample"].response
    assert sample.metadata["benchmark_eval"] is True
    assert sample.metadata["strict_exact_match"] is True
    assert sample.metadata["fused_reward_debug"]["tool_calls"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
