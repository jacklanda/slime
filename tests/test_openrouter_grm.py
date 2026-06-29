import asyncio
import types

from slime.rollout import sglang_rollout
from slime.rollout.rm_hub import openrouter_grm
from slime.utils.types import Sample


class Args:
    enable_use_grm_evals = True
    grm_model = "deepseek/deepseek-v4-flash"
    grm_concurrency = 128
    grm_max_retries = 1
    grm_failure_reward = 0.0
    grm_temperature = 0.0
    grm_max_tokens = 128
    grm_max_trajectory_chars = 2000
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


def test_openrouter_grm_uses_final_answer_step_only(monkeypatch):
    args = Args()
    client = FakeClient({"choices": [{"message": {"content": '{"score": 1}'}}]})
    monkeypatch.setattr(openrouter_grm, "_CLIENT", client)
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, _sample(), evaluation=True))

    assert reward == 1.0
    assert client.calls == 1
    prompt = client.requests[0]["messages"][1]["content"]
    assert "Final submitted answer step:" in prompt
    assert "\\boxed{42}" in prompt
    assert "question" not in prompt


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


def test_openrouter_grm_failure_falls_back_to_benchmark_verifier(monkeypatch):
    class FailingClient:
        is_closed = False

        async def post(self, path, json):
            raise RuntimeError("boom")

    args = Args()
    sample = Sample(
        response='<tool_call>{"name":"finish","arguments":{"result":"Ada Lovelace"}}</tool_call>',
        label="Ada Lovelace",
        metadata={"rm_type": "benchmark_verifier", "data_source": "simpleqa_verified", "benchmark_eval": True},
    )
    monkeypatch.setattr(openrouter_grm, "_CLIENT", FailingClient())
    monkeypatch.setattr(openrouter_grm, "_SEMAPHORE", None)

    reward = asyncio.run(openrouter_grm.reward_func(args, sample, evaluation=True))

    assert reward == 1.0
    assert sample.metadata["grm"]["fallback"] is True
    assert sample.metadata["grm"]["fallback_rm_type"] == "benchmark_verifier"


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
