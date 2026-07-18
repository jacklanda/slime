import asyncio
import types

import pytest

from slime.rollout import sglang_rollout
from slime.rollout.rm_hub import openrouter_grm
from slime.utils.types import Sample

NUM_GPUS = 0


class Args:
    enable_use_grm_evals = True
    grm_model = "deepseek/deepseek-v4-flash"
    grm_concurrency = 128
    grm_max_retries = 1
    grm_failure_reward = 0.0
    grm_temperature = 0.0
    grm_max_new_tokens = 128
    grm_max_input_tokens = 2000
    rm_type = "math"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    is_closed = False

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.requests = []

    async def post(self, path, json):
        self.calls += 1
        self.requests.append(json)
        assert path == "/chat/completions"
        assert json["model"] == Args.grm_model
        return FakeResponse(self.payload)


def _sample():
    return Sample(
        index=0,
        prompt="question",
        response="final answer is \\boxed{42}",
        label="42",
        status=Sample.Status.COMPLETED,
        metadata={},
    )


def test_openrouter_grm_scores_batch(monkeypatch):
    client = FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]})
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    rewards = asyncio.run(openrouter_grm.reward_func(Args(), [_sample(), _sample()], evaluation=True))

    assert rewards == [1.0, 1.0]
    assert client.calls == 2
    response_format = client.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["properties"]["score"]["enum"] == [0, 1]
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert client.requests[0]["reasoning"] == {"effort": "none"}
    assert client.requests[0]["provider"] == {"require_parameters": True}
    assert client.requests[0]["max_tokens"] == Args.grm_max_new_tokens


def test_openrouter_grm_does_not_send_deepseek_reasoning_parameter_to_gemini():
    args = Args()
    args.grm_model = "google/gemini-3-flash-preview"

    payload = openrouter_grm._build_payload(args, _sample())

    assert "reasoning" not in payload
    assert payload["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('```json\n{"score": 1}\n```', 1.0),
        ('The answer is too specific.\n{"score": 0}', 0.0),
    ],
)
def test_openrouter_grm_accepts_json_wrapped_in_model_explanation(content, expected):
    payload = {"choices": [{"message": {"content": content}}]}

    assert openrouter_grm._parse_reward(payload) == expected


def test_openrouter_grm_uses_final_answer_step_only(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]})
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, _sample(), evaluation=True))

    assert reward == 1.0
    assert client.calls == 1
    prompt = client.requests[0]["messages"][1]["content"]
    assert "Final submitted answer:" in prompt
    assert "\\boxed{42}" in prompt
    assert "Question:\nquestion" not in prompt


def test_openrouter_grm_limits_complete_input_by_tokens():
    args = Args()
    args.grm_max_input_tokens = 160

    payload = openrouter_grm._build_payload(args, _sample(), final_answer_step="discard " * 500 + "keep this answer")

    messages = payload["messages"]
    input_tokens = sum(
        len(openrouter_grm._GRM_TOKENIZER.encode_ordinary(message["content"])) for message in messages
    )
    assert input_tokens <= args.grm_max_input_tokens
    assert messages[1]["content"].count("discard") < 500
    assert "keep this answer" in messages[1]["content"]


def test_benchmark_eval_uses_rule_verifier_before_grm_for_multiple_choice(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 0}'}}]})
    sample = Sample(
        response='<tool_call>{"name":"finish","arguments":{"result":"D"}}</tool_call>',
        label="Peroxisome proliferator-activated receptor gamma",
        metadata={
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "data_source": "medqa",
            "question": "Which molecule increases insulin sensitivity?",
            "options": ["Catecholamines", "Glucagon", "Glucocorticoids", "Peroxisome proliferator-activated receptor gamma"],
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 1.0
    assert client.calls == 0
    assert sample.metadata["grm"]["judge"] == "benchmark_verifier"
    assert sample.metadata["grm"]["hybrid_rule_match"] is True
    assert sample.metadata["grm"]["model"] == "benchmark_verifier"


def test_benchmark_eval_semantic_fallback_receives_question(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]})
    sample = Sample(
        response='<tool_call>{"name":"finish","arguments":{"result":"Wilhelm Röntgen"}}</tool_call>',
        label=["Wilhelm Conrad Röntgen"],
        metadata={
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "data_source": "search_r1",
            "question": "Who received the first Nobel Prize in Physics?",
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 1.0
    assert client.calls == 1
    prompt = client.requests[0]["messages"][1]["content"]
    assert "Question:\nWho received the first Nobel Prize in Physics?" in prompt
    assert "Accepted ground truth:\n[" in prompt
    assert "Final submitted answer:\nWilhelm Röntgen" in prompt


def test_benchmark_eval_semantic_fallback_appends_multiple_choice_options(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 0}'}}]})
    sample = Sample(
        response='<tool_call>{"name":"finish","arguments":{"result":"A"}}</tool_call>',
        label="Peroxisome proliferator-activated receptor gamma",
        metadata={
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "data_source": "medqa",
            "question": "Which molecule increases insulin sensitivity?",
            "options": ["Catecholamines", "Glucagon", "Glucocorticoids", "Peroxisome proliferator-activated receptor gamma"],
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    prompt = client.requests[0]["messages"][1]["content"]
    assert "Options:\nA. Catecholamines" in prompt
    assert "D. Peroxisome proliferator-activated receptor gamma" in prompt


def test_benchmark_eval_does_not_rule_short_circuit_overbroad_phrase_match(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 0}'}}]})
    sample = Sample(
        response=(
            '<tool_call>{"name":"finish","arguments":{"result":'
            '"All Sailor Guardians have talismans, including Sailor Pluto."}}</tool_call>'
        ),
        label=["Haruka", "Michiru", "Sailor Pluto"],
        metadata={
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "data_source": "search_r1",
            "question": "Who has the talismans in Sailor Moon S?",
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    assert client.calls == 1


def test_benchmark_eval_rejects_unsubmitted_search_query_without_calling_grm(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]})
    sample = Sample(
        response='<tool_call>{"name":"web_search","arguments":{"query":"Ada Lovelace"}}</tool_call>',
        label=["Ada Lovelace"],
        metadata={
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "data_source": "search_r1",
            "question": "Who wrote the first computer program?",
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    assert client.calls == 0


def test_openrouter_grm_does_not_use_metadata_answer_as_submission(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 0}'}}]})
    sample = Sample(
        index=0,
        prompt="question",
        response="",
        label="Lady Mary Henrietta Powlett",
        status=Sample.Status.COMPLETED,
        metadata={
            "answer": "Lady Mary Henrietta Powlett",
            "rllm_episode": {
                "trajectories": [
                    {
                        "steps": [
                            {
                                "action": '<tool_call>{"name":"finish","arguments":{"command":"submit","result":"Mary Montagu, Duchess of Montagu"}}</tool_call>'
                            }
                        ]
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    assert client.calls == 1
    prompt = client.requests[0]["messages"][1]["content"]
    assert "Mary Montagu, Duchess of Montagu" in prompt
    assert "Final submitted answer step:\nLady Mary Henrietta Powlett" not in prompt


def test_openrouter_grm_without_response_or_episode_submission_returns_failure_reward(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]})
    sample = Sample(
        index=0,
        prompt="question",
        response="",
        label="42",
        status=Sample.Status.COMPLETED,
        metadata={"answer": "42"},
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    assert client.calls == 0


def test_openrouter_grm_failure_falls_back_to_rule_based(monkeypatch):
    class FailingClient:
        is_closed = False

        async def post(self, path, json):
            raise RuntimeError("boom")

    args = Args()
    args.grm_failure_reward = 0.25
    monkeypatch.setattr(openrouter_grm, "_CLIENT", FailingClient())
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, _sample(), evaluation=True))

    assert reward == 1.0


def test_openrouter_grm_parse_failure_falls_back_to_rule_based(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": "not-json-no-score"}}]})
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, _sample(), evaluation=True))

    assert reward == 1.0
    assert client.calls == 1


def test_openrouter_grm_failure_for_benchmark_semantic_match_uses_failure_reward(monkeypatch):
    class FailingClient:
        is_closed = False

        async def post(self, path, json):
            raise RuntimeError("boom")

    args = Args()
    sample = Sample(
        response='<tool_call>{"name":"finish","arguments":{"result":"Wilhelm Röntgen"}}</tool_call>',
        label="Wilhelm Conrad Röntgen",
        metadata={"rm_type": "benchmark_verifier", "data_source": "simpleqa_verified", "benchmark_eval": True},
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", FailingClient())
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    assert sample.metadata["grm"]["fallback"] is True
    assert sample.metadata["grm"]["fallback_rm_type"] == "grm_failure_reward"


def test_benchmark_semantic_failure_does_not_restore_non_decisive_rule_match(monkeypatch):
    class FailingClient:
        is_closed = False

        async def post(self, path, json):
            raise RuntimeError("boom")

    args = Args()
    sample = Sample(
        response=(
            '<tool_call>{"name":"finish","arguments":{"result":'
            '"All Sailor Guardians have talismans, including Sailor Pluto."}}</tool_call>'
        ),
        label=["Haruka", "Michiru", "Sailor Pluto"],
        metadata={
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "data_source": "search_r1",
            "question": "Who has the talismans in Sailor Moon S?",
        },
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", FailingClient())
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 0.0
    assert sample.metadata["grm"]["fallback_rm_type"] == "grm_failure_reward"


def test_eval_generate_and_rm_rescores_generated_samples_with_grm(monkeypatch):
    async def fake_generate(args, sample, sampling_params, evaluation=False):
        generated = _sample()
        generated.reward = 0.0
        generated.status = Sample.Status.COMPLETED
        return [generated]

    class ArgsWithGenerate(Args):
        partial_rollout = False
        mask_offpolicy_in_partial_rollout = False
        group_rm = False
        custom_rm_path = None
        custom_generate_function_path = None
        grm_custom_rm_path = "slime.rollout.rm_hub.openrouter_grm.reward_func"

    args = ArgsWithGenerate()
    prompt_sample = Sample(generate_function_path="tests.fake_generate")
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: fake_generate)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: types.SimpleNamespace(semaphore=_NoopAsyncContext(), aborted=False, dp_rank_context=_noop_context))
    monkeypatch.setattr(openrouter_grm, "_CLIENT", FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]}))
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    samples = asyncio.run(sglang_rollout.generate_and_rm(args, prompt_sample, {}, evaluation=True))

    assert samples[0].reward == 1.0
    assert samples[0].custom_rm_path == args.grm_custom_rm_path


class _NoopAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _NoopContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _noop_context():
    return _NoopContext()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
