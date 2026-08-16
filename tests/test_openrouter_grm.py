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
    grm_mode = "score"


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


class SequencedFakeClient(FakeClient):
    def __init__(self, payloads):
        super().__init__(None)
        self.payloads = list(payloads)

    async def post(self, path, json):
        self.calls += 1
        self.requests.append(json)
        assert path == "/chat/completions"
        return FakeResponse(self.payloads.pop(0))


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


def test_openrouter_grm_selects_train_and_eval_models():
    args = Args()
    args.train_grm_model = "train/judge"
    args.eval_grm_model = "eval/judge"

    train_payload = openrouter_grm._build_payload(args, _sample())
    eval_payload = openrouter_grm._build_payload(args, _sample(), evaluation=True)

    assert train_payload["model"] == "train/judge"
    assert eval_payload["model"] == "eval/judge"


def test_mcp_atlas_grm_scores_each_claim_and_averages_partial_credit(monkeypatch):
    client = SequencedFakeClient(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"claim_text":"claim one","coverage_outcome":"fulfilled","justification":"Covered.","confidence_level":0.9}'
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"claim_text":"claim two","coverage_outcome":"partially_fulfilled","justification":"Incomplete.","confidence_level":0.8}'
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"claim_text":"claim three","coverage_outcome":"not_fulfilled","justification":"Missing.","confidence_level":0.95}'
                        }
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)
    sample = Sample(
        prompt="Use the available MCP tools.",
        response='<tool_call>{"name":"finish","arguments":{"result":"final report"}}</tool_call>',
        label='["claim one", "claim two", "claim three"]',
        metadata={"benchmark_eval": True, "rm_type": "benchmark_verifier", "data_source": "mcp_atlas"},
    )

    reward = asyncio.run(openrouter_grm.reward_func(Args(), sample, evaluation=True))

    assert reward == 0.5
    assert client.calls == 3
    assert sample.metadata["grm"]["judge"] == "mcp_atlas_claims"
    assert sample.metadata["grm"]["total_claims"] == 3
    assert sample.metadata["grm"]["fully_covered_claims"] == 1
    assert sample.metadata["grm"]["partially_covered_claims"] == 1
    assert sample.metadata["verification"]["protocol"] == "mcp_atlas_claim_coverage"
    assert "CLAIM TO EVALUATE:\nclaim one" in client.requests[0]["messages"][1]["content"]
    assert "MODEL RESPONSE TO ANALYZE:\nfinal report" in client.requests[0]["messages"][1]["content"]


def test_mcp_atlas_grm_missing_submission_scores_zero_without_judge_call(monkeypatch):
    client = SequencedFakeClient([])
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)
    sample = Sample(
        prompt="Use the available MCP tools.",
        response="",
        label=["claim one", "claim two"],
        metadata={"data_source": "mcp-atlas", "mcp_atlas_eval": True},
    )

    reward = asyncio.run(openrouter_grm.reward_func(Args(), sample, evaluation=True))

    assert reward == 0.0
    assert client.calls == 0
    assert sample.metadata["grm"]["failure"] == "missing_submission"
    assert sample.metadata["grm"]["total_claims"] == 2


def test_mcp_atlas_grm_requires_claims():
    sample = Sample(
        prompt="Use the available MCP tools.",
        response='<tool_call>{"name":"finish","arguments":{"result":"final report"}}</tool_call>',
        label=None,
        metadata={"data_source": "mcp_atlas"},
    )

    with pytest.raises(ValueError, match="requires GTFA_CLAIMS"):
        asyncio.run(openrouter_grm.reward_func(Args(), sample, evaluation=True))


def test_openrouter_grm_equivalence_mode_uses_judgement_schema_and_persists_json(monkeypatch):
    args = Args()
    args.grm_mode = "equivalence"
    client = FakeClient(
        {"choices": [{"message": {"content": '{"rationale":"Same answer.","judgement":"Correct"}'}}]}
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    sample = _sample()
    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=False))

    assert reward == 1.0
    request = client.requests[0]
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["judgement"]["enum"] == ["Correct", "Incorrect"]
    assert "Question: question" in request["messages"][1]["content"]
    assert "Labeled Answer: 42" in request["messages"][1]["content"]
    assert "Predicted Answer: \\boxed{42}" in request["messages"][1]["content"]
    assert sample.metadata["grm"]["judge_json"] == {"rationale": "Same answer.", "judgement": "Correct"}


def test_openrouter_grm_equivalence_parser_normalizes_judgement_case():
    payload = {
        "choices": [{"message": {"content": 'prefix {"rationale":"Equivalent.","judgement":" correct! "}'}}]
    }
    assert openrouter_grm._parse_judge_json(payload) == {
        "rationale": "Equivalent.",
        "judgement": "Correct",
    }


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


def test_openrouter_grm_uses_openai_credentials_for_custom_endpoint(monkeypatch):
    args = Args()
    args.grm_base_url = None
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "custom-endpoint-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://judge.example/v1/")
    monkeypatch.setattr(openrouter_grm, "_CLIENT", None)

    client = openrouter_grm._get_client(args)

    assert str(client.base_url) == "https://judge.example/v1/"
    assert client.headers["Authorization"] == "Bearer custom-endpoint-key"
    asyncio.run(client.aclose())
    monkeypatch.setattr(openrouter_grm, "_CLIENT", None)


def test_openrouter_grm_does_not_send_openai_key_to_openrouter(monkeypatch):
    args = Args()
    args.grm_base_url = None
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only-key")
    monkeypatch.setattr(openrouter_grm, "_CLIENT", None)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        openrouter_grm._get_client(args)


def test_openrouter_credentials_ignore_unrelated_openai_endpoint(monkeypatch):
    args = Args()
    args.grm_base_url = None
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://judge.example/v1")
    monkeypatch.setattr(openrouter_grm, "_CLIENT", None)

    client = openrouter_grm._get_client(args)

    assert str(client.base_url) == "https://openrouter.ai/api/v1/"
    assert client.headers["Authorization"] == "Bearer openrouter-key"
    asyncio.run(client.aclose())
    monkeypatch.setattr(openrouter_grm, "_CLIENT", None)


def test_openrouter_grm_parse_failure_falls_back_to_rule_based(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": "not-json-no-score"}}]})
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, _sample(), evaluation=True))

    assert reward == 1.0
    assert client.calls == 1


def test_train_grm_timeout_falls_back_to_final_answer_exact_match(monkeypatch):
    class TimingOutClient:
        is_closed = False

        async def post(self, path, json):
            raise TimeoutError("judge timed out")

    args = Args()
    sample = Sample(
        response=(
            '<tool_call>{"name":"finish","arguments":'
            '{"result":"The Eiffel Tower!"}}</tool_call>'
        ),
        label={"ground_truth": "eiffel tower"},
        metadata={},
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", TimingOutClient())
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=False))

    assert reward == 1.0
    assert sample.metadata["grm"]["fallback"] is True
    assert sample.metadata["grm"]["fallback_rm_type"] == "exact_match"
    assert "judge timed out" in sample.metadata["grm"]["error"]


def test_train_grm_failure_exact_match_uses_last_finish_result(monkeypatch):
    class FailingClient:
        is_closed = False

        async def post(self, path, json):
            raise RuntimeError("boom")

    args = Args()
    sample = Sample(
        response=(
            '<tool_call>{"name":"finish","arguments":{"result":"correct"}}</tool_call>\n'
            '<tool_call>{"name":"finish","arguments":{"result":"wrong"}}</tool_call>'
        ),
        label="correct",
        metadata={},
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", FailingClient())
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=False))

    assert reward == 0.0
    assert sample.metadata["grm"]["fallback_rm_type"] == "exact_match"


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
        generated.metadata["rm_type"] = "benchmark_verifier"
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


def test_train_generate_and_rm_rescores_generated_samples_with_grm(monkeypatch):
    async def fake_generate(args, sample, sampling_params, evaluation=False):
        generated = _sample()
        generated.reward = 0.0
        generated.status = Sample.Status.COMPLETED
        return [generated]

    class ArgsWithGenerate(Args):
        enable_use_grm_train = True
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

    samples = asyncio.run(sglang_rollout.generate_and_rm(args, prompt_sample, {}, evaluation=False))

    assert samples[0].reward == 1.0
    assert samples[0].custom_rm_path == args.grm_custom_rm_path
    assert samples[0].metadata["grm"]["score"] == 1.0
    assert samples[0].metadata["grm"]["evaluation"] is False


def test_train_generate_and_rm_preserves_mcp_environment_reward(monkeypatch):
    async def fake_generate(args, sample, sampling_params, evaluation=False):
        return [
            Sample(
                index=0,
                prompt="Use the available MCP tools.",
                response='<tool_call>{"name":"finish","arguments":{"result":"final report"}}</tool_call>',
                label={"ground_truth": None, "style": "rule"},
                reward=1.0,
                status=Sample.Status.COMPLETED,
                metadata={"fused_task_type": "mcp"},
            )
        ]

    class ArgsWithGenerate(Args):
        enable_use_grm_train = True
        partial_rollout = False
        mask_offpolicy_in_partial_rollout = False
        group_rm = False
        custom_rm_path = None
        custom_generate_function_path = None
        grm_custom_rm_path = "slime.rollout.rm_hub.openrouter_grm.reward_func"

    args = ArgsWithGenerate()
    prompt_sample = Sample(generate_function_path="tests.fake_generate")
    client = FakeClient({"choices": [{"message": {"content": '{"score": 0}'}}]})
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: fake_generate)
    monkeypatch.setattr(
        sglang_rollout,
        "GenerateState",
        lambda args: types.SimpleNamespace(
            semaphore=_NoopAsyncContext(), aborted=False, dp_rank_context=_noop_context
        ),
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    samples = asyncio.run(sglang_rollout.generate_and_rm(args, prompt_sample, {}, evaluation=False))

    assert samples[0].reward == 1.0
    assert samples[0].custom_rm_path is None
    assert "grm" not in samples[0].metadata
    assert client.calls == 0


def test_eval_generate_and_rm_preserves_mcp_dataset_verifier_reward(monkeypatch):
    async def fake_generate(args, sample, sampling_params, evaluation=False):
        return [
            Sample(
                index=0,
                prompt="Use the available MCP tools.",
                response='<tool_call>{"name":"finish","arguments":{"result":"final report"}}</tool_call>',
                label=["claim"],
                reward=0.5,
                status=Sample.Status.COMPLETED,
                metadata={"fused_task_type": "mcp"},
            )
        ]

    class ArgsWithGenerate(Args):
        partial_rollout = False
        mask_offpolicy_in_partial_rollout = False
        group_rm = False
        custom_rm_path = None
        custom_generate_function_path = None
        grm_custom_rm_path = "slime.rollout.rm_hub.openrouter_grm.reward_func"

    args = ArgsWithGenerate()
    prompt_sample = Sample(generate_function_path="tests.fake_generate")
    client = FakeClient({"choices": [{"message": {"content": '{"score": 0}'}}]})
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: fake_generate)
    monkeypatch.setattr(
        sglang_rollout,
        "GenerateState",
        lambda args: types.SimpleNamespace(
            semaphore=_NoopAsyncContext(), aborted=False, dp_rank_context=_noop_context
        ),
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    samples = asyncio.run(sglang_rollout.generate_and_rm(args, prompt_sample, {}, evaluation=True))

    assert samples[0].reward == 0.5
    assert samples[0].custom_rm_path is None
    assert "grm" not in samples[0].metadata
    assert client.calls == 0


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
