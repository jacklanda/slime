import asyncio
import io
from argparse import Namespace
from contextlib import nullcontext

import pytest

from slime.rollout import sglang_rollout
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.types import Sample

NUM_GPUS = 0


def test_eval_progress_log_stream_emits_complete_lines():
    sink = io.StringIO()
    stream = sglang_rollout._EvalProgressLogStream(sink)

    stream.write("\rEval mixed (3 datasets): 2%| | 85/3622")

    assert sink.getvalue() == "Eval mixed (3 datasets): 2%| | 85/3622\n"


async def _fake_generate_and_rm(_args, sample, _sampling_params, evaluation=False):
    if sample.index == 1:
        await sglang_rollout.asyncio.sleep(10)
    sample.status = Sample.Status.COMPLETED
    sample.reward = 1.0
    return sample


async def _fake_generate_and_rm_with_one_error(_args, sample, _sampling_params, evaluation=False):
    if sample.index == 1:
        raise ValueError("bad Qwen3.5 trajectory")
    sample.status = Sample.Status.COMPLETED
    sample.reward = 1.0
    return sample


class _FakeGenerateState:
    aborted = False

    def __init__(self, _args):
        pass


def test_generate_group_timeout_returns_failed_tail_sample(monkeypatch):
    monkeypatch.setenv("SLIME_ROLLOUT_GROUP_TIMEOUT", "1")
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _fake_generate_and_rm)

    group = [
        Sample(index=0, prompt="a", reward=None),
        Sample(index=1, prompt="b", reward=None),
    ]

    out = sglang_rollout.asyncio.run(
        sglang_rollout.generate_and_rm_group(
            Namespace(sglang_enable_deterministic_inference=False, group_rm=False),
            group,
            sampling_params={},
        )
    )

    assert len(out) == 2
    assert out[0].status == Sample.Status.COMPLETED
    assert out[0].reward == 1.0
    assert out[1].status == Sample.Status.FAILED
    assert out[1].reward == 0.0
    assert out[1].metadata["fused_error"] == "rollout_group_timeout"


def test_generate_group_isolates_sample_exception(monkeypatch, caplog):
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "generate_and_rm", _fake_generate_and_rm_with_one_error)
    group = [Sample(index=0, prompt="a", reward=None), Sample(index=1, prompt="b", reward=None)]

    with caplog.at_level("ERROR", logger="slime.rollout.sglang_rollout"):
        out = asyncio.run(
            sglang_rollout.generate_and_rm_group(
                Namespace(sglang_enable_deterministic_inference=False, group_rm=False),
                group,
                sampling_params={},
            )
        )

    assert out[0].status == Sample.Status.COMPLETED
    assert out[0].reward == 1.0
    assert out[1].status == Sample.Status.FAILED
    assert out[1].reward == 0.0
    assert out[1].metadata["fused_termination"] == "rollout_task_exception"
    assert out[1].metadata["rollout_exception_type"] == "ValueError"
    assert out[1].metadata["rollout_exception"] == "bad Qwen3.5 trajectory"
    assert "bad Qwen3.5 trajectory" in caplog.text


class _FakeRolloutGenerateState:
    def __init__(self, args):
        self.args = args
        self.remaining_batch_size = 0
        self.pendings = set()
        self.pending_groups = {}
        self.aborted = False

    def submit_generate_tasks(self, samples):
        for group in samples:
            task = sglang_rollout.asyncio.create_task(sglang_rollout.asyncio.sleep(10))
            self.pendings.add(task)
            self.pending_groups[task] = group
        self.remaining_batch_size += len(samples)

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.pending_groups = {}


class _FakeCompletedRolloutGenerateState(_FakeRolloutGenerateState):
    def submit_generate_tasks(self, samples):
        for group in samples:
            for sample in group:
                sample.status = Sample.Status.COMPLETED
                sample.reward = 0.0
                sample.response = "ok"
                sample.response_length = 1
            task = sglang_rollout.asyncio.create_task(sglang_rollout.asyncio.sleep(0, result=group))
            self.pendings.add(task)
            self.pending_groups[task] = group
        self.remaining_batch_size += len(samples)


def test_rollout_collection_timeout_returns_failed_group(monkeypatch):
    monkeypatch.setenv("SLIME_ROLLOUT_GROUP_TIMEOUT", "1")
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeRolloutGenerateState)

    source_group = [
        [Sample(index=0, prompt="a", reward=None), Sample(index=1, prompt="b", reward=None)],
    ]

    out, aborted = sglang_rollout.asyncio.run(
        sglang_rollout.generate_rollout_async(
            Namespace(
                rollout_global_dataset=True,
                dynamic_sampling_filter_path=None,
                rollout_batch_size=1,
                n_samples_per_prompt=2,
                over_sampling_batch_size=1,
                rollout_sample_filter_path=None,
                rollout_all_samples_process_path=None,
                partial_rollout=False,
            ),
            rollout_id=0,
            data_source=lambda _batch_size: source_group,
        )
    )

    assert aborted == []
    group = out.samples[0]
    assert len(group) == 2
    assert group[0].status == Sample.Status.FAILED
    assert group[0].reward == 0.0
    assert group[0].metadata["fused_error"] == "rollout_group_timeout"


def test_sync_rollout_relaxes_dynamic_filter_after_configured_groups(monkeypatch):
    monkeypatch.setenv("SLIME_ROLLOUT_GROUP_TIMEOUT", "1")
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeCompletedRolloutGenerateState)

    def drop_all(_args, _samples, **_kwargs):
        return DynamicFilterOutput(keep=False, reason="zero_std_0")

    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: drop_all)

    source_group = [
        [Sample(index=0, prompt="a", reward=None), Sample(index=1, prompt="b", reward=None)],
    ]

    out, aborted = sglang_rollout.asyncio.run(
        sglang_rollout.generate_rollout_async(
            Namespace(
                rollout_global_dataset=True,
                dynamic_sampling_filter_path="drop_all",
                rollout_batch_size=1,
                n_samples_per_prompt=2,
                over_sampling_batch_size=1,
                rollout_sample_filter_path=None,
                rollout_all_samples_process_path=None,
                partial_rollout=False,
                fully_async_filter_relax_after_groups=1,
            ),
            rollout_id=0,
            data_source=lambda _batch_size: source_group,
        )
    )

    assert aborted == []
    assert len(out.samples) == 1
    assert out.metrics["rollout/dynamic_filter/completed_groups"] == 1
    assert out.metrics["rollout/dynamic_filter/kept_groups"] == 1
    assert out.metrics["rollout/dynamic_filter/drop_relaxed_zero_std_0"] == 1


def test_sync_rollout_does_not_relax_timeout_only_groups(monkeypatch):
    monkeypatch.setenv("SLIME_ROLLOUT_GROUP_TIMEOUT", "1")
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeRolloutGenerateState)

    def drop_all(_args, _samples, **_kwargs):
        return DynamicFilterOutput(keep=False, reason="zero_std_0")

    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: drop_all)

    source_group = [
        [Sample(index=0, prompt="a", reward=None), Sample(index=1, prompt="b", reward=None)],
    ]

    with pytest.raises(asyncio.TimeoutError):
        sglang_rollout.asyncio.run(
            sglang_rollout.asyncio.wait_for(
                sglang_rollout.generate_rollout_async(
                    Namespace(
                        rollout_global_dataset=True,
                        dynamic_sampling_filter_path="drop_all",
                        rollout_batch_size=1,
                        n_samples_per_prompt=2,
                        over_sampling_batch_size=1,
                        rollout_sample_filter_path=None,
                        rollout_all_samples_process_path=None,
                        partial_rollout=False,
                        fully_async_filter_relax_after_groups=1,
                    ),
                    rollout_id=0,
                    data_source=lambda _batch_size: source_group,
                ),
                timeout=2,
            )
        )


def test_eval_generation_limits_inflight_tasks_and_preserves_order(monkeypatch):
    active = 0
    max_active = 0

    async def fake_generate(_args, sample, sampling_params, evaluation=False):
        nonlocal active, max_active
        assert evaluation is True
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01 * (4 - sample.index))
        active -= 1
        sample.reward = 1.0
        sample.response = str(sample.index)
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
    dataset = type("DatasetStub", (), {"samples": [Sample(prompt=str(i)) for i in range(4)]})()
    dataset_cfg = EvalDatasetConfig(name="eval", path="unused", n_samples_per_eval_prompt=1)
    args = Namespace(
        eval_max_inflight_tasks=2,
        enable_use_grm_evals=False,
        sglang_enable_deterministic_inference=False,
    )

    samples = asyncio.run(
        sglang_rollout._generate_eval_samples_bounded(
            args,
            dataset,
            dataset_cfg,
            {"max_new_tokens": 4},
        )
    )

    assert max_active == 2
    assert [sample.index for sample in samples] == [0, 1, 2, 3]


def test_parse_eval_engine_metrics_aggregates_worker_load():
    metrics = """
# HELP sglang:num_running_reqs The number of running requests.
sglang_num_running_reqs{worker_addr="http://engine-0"} 20
sglang_num_running_reqs{worker_addr="http://engine-1"} 18
sglang_num_queue_reqs{worker_addr="http://engine-0"} 2
sglang_num_queue_reqs{worker_addr="http://engine-1"} 1
sglang_token_usage{worker_addr="http://engine-0"} 0.72
sglang_token_usage{worker_addr="http://engine-1"} 0.81
sglang_cache_hit_rate{worker_addr="http://engine-0"} 0.80
sglang_cache_hit_rate{worker_addr="http://engine-1"} 0.60
"""

    load = sglang_rollout._parse_eval_engine_metrics(metrics)

    assert load == sglang_rollout.EvalEngineLoad(
        engine_count=2,
        running_requests=38.0,
        waiting_requests=3.0,
        max_token_usage=0.81,
        mean_cache_hit_rate=0.7,
    )


def test_eval_concurrency_controller_uses_hysteresis_and_pressure_backoff():
    args = Namespace(
        eval_max_inflight_tasks=512,
        eval_initial_inflight_tasks=384,
        eval_concurrency_step=32,
        eval_concurrency_poll_interval=5.0,
        sglang_server_concurrency=56,
    )
    controller = sglang_rollout.EvalConcurrencyController(args, total=1000)
    underfed = sglang_rollout.EvalEngineLoad(8, 200, 0, 0.70, 0.75)
    pressured = sglang_rollout.EvalEngineLoad(8, 400, 120, 0.93, 0.40)

    assert controller.update(underfed) == 384
    assert controller.update(underfed) == 416
    assert controller.update(pressured) == 384


def test_adaptive_eval_scheduler_refills_to_updated_target(monkeypatch):
    active = 0
    max_active = 0

    async def fake_generate(_args, sample, sampling_params, evaluation=False):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        sample.reward = 1.0
        sample.status = Sample.Status.COMPLETED
        return sample

    class FakeController:
        def __init__(self, _args, _total):
            self.target = 2

        def poll(self, _args):
            return 4

        async def close(self):
            return None

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
    monkeypatch.setattr(sglang_rollout, "EvalConcurrencyController", FakeController)
    dataset = type("DatasetStub", (), {"samples": [Sample(prompt=str(i)) for i in range(8)]})()
    dataset_cfg = EvalDatasetConfig(name="eval", path="unused", n_samples_per_eval_prompt=1)
    args = Namespace(
        eval_max_inflight_tasks=8,
        eval_adaptive_concurrency=True,
        enable_use_grm_evals=False,
        sglang_enable_deterministic_inference=False,
    )

    samples = asyncio.run(sglang_rollout._generate_eval_samples_bounded(args, dataset, dataset_cfg, {"max_new_tokens": 4}))

    assert max_active == 4
    assert len(samples) == 8


def test_eval_datasets_share_one_global_inflight_budget(monkeypatch):
    active_datasets = 0
    max_active_datasets = 0

    async def fake_eval_dataset(_args, _rollout_id, dataset_cfg):
        nonlocal active_datasets, max_active_datasets
        active_datasets += 1
        max_active_datasets = max(max_active_datasets, active_datasets)
        await asyncio.sleep(0.01)
        active_datasets -= 1
        return {dataset_cfg.name: {"rewards": [], "truncated": [], "samples": []}}

    monkeypatch.setattr(sglang_rollout, "eval_rollout_single_dataset", fake_eval_dataset)
    args = Namespace(
        group_rm=False,
        eval_datasets=[
            EvalDatasetConfig(name="first", path="unused"),
            EvalDatasetConfig(name="second", path="unused"),
        ],
    )

    output, _ = asyncio.run(sglang_rollout.eval_rollout(args, rollout_id=0))

    assert max_active_datasets == 1
    assert list(output.data) == ["first", "second"]


def test_mixed_eval_round_robins_datasets_with_one_global_budget(monkeypatch):
    active = 0
    max_active = 0
    started = []

    async def fake_generate(_args, sample, sampling_params, evaluation=False):
        nonlocal active, max_active
        assert evaluation is True
        active += 1
        max_active = max(max_active, active)
        started.append(sample.prompt)
        await asyncio.sleep(0.01)
        active -= 1
        sample.reward = 1.0
        sample.status = Sample.Status.COMPLETED
        sample.response = sampling_params["dataset"]
        return sample

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate)
    first = type("DatasetStub", (), {"samples": [Sample(prompt=f"first-{i}") for i in range(4)]})()
    second = type("DatasetStub", (), {"samples": [Sample(prompt=f"second-{i}") for i in range(4)]})()
    first_cfg = EvalDatasetConfig(name="first", path="unused", n_samples_per_eval_prompt=1)
    second_cfg = EvalDatasetConfig(name="second", path="unused", n_samples_per_eval_prompt=1)
    args = Namespace(
        eval_max_inflight_tasks=2,
        eval_adaptive_concurrency=False,
        enable_use_grm_evals=False,
        sglang_enable_deterministic_inference=False,
    )

    data = asyncio.run(
        sglang_rollout._generate_mixed_eval_samples_bounded(
            args,
            [(first, {"dataset": "first"}, first_cfg), (second, {"dataset": "second"}, second_cfg)],
        )
    )

    assert max_active == 2
    assert started[:2] == ["first-0", "second-0"]
    assert [sample.index for sample in data["first"]] == [0, 1, 2, 3]
    assert [sample.index for sample in data["second"]] == [0, 1, 2, 3]
    assert {sample.response for sample in data["first"]} == {"first"}
    assert {sample.response for sample in data["second"]} == {"second"}


def test_eval_custom_generator_can_manage_request_concurrency(monkeypatch):
    async def custom_generate(_args, sample, _sampling_params, evaluation=False):
        assert evaluation is True
        sample.reward = 1.0
        sample.response = "done"
        sample.status = Sample.Status.COMPLETED
        return sample

    custom_generate.manages_eval_request_concurrency = True

    class BlockedOuterState:
        aborted = False
        semaphore = asyncio.Semaphore(0)

        def __init__(self, _args):
            pass

        def dp_rank_context(self):
            return nullcontext()

    monkeypatch.setattr(sglang_rollout, "GenerateState", BlockedOuterState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: custom_generate)
    args = Namespace(
        custom_generate_function_path="custom",
        group_rm=False,
        rm_type="",
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )

    sample = asyncio.run(
        asyncio.wait_for(
            sglang_rollout.generate_and_rm(args, Sample(prompt="q"), {}, evaluation=True),
            timeout=0.2,
        )
    )

    assert sample.response == "done"
    assert sample.metadata["eval_retry_count"] == 0
    assert sample.metadata["eval_retry_termination_reasons"] == []


def test_eval_retries_non_env_done_termination_before_reward(monkeypatch):
    attempts = 0
    scored_responses = []

    async def custom_generate(_args, sample, sampling_params, evaluation=False):
        nonlocal attempts
        assert evaluation is True
        attempts += 1
        sample.response = f"attempt-{attempts}"
        sample.status = Sample.Status.COMPLETED
        sample.metadata["fused_termination"] = "max_context_len_exceeded" if attempts < 3 else "env_done"
        return sample

    async def fake_rm(_args, sample, evaluation=False):
        assert evaluation is True
        scored_responses.append(sample.response)
        return 1.0

    class UnblockedState:
        aborted = False
        semaphore = asyncio.Semaphore(1)

        def __init__(self, _args):
            pass

        def dp_rank_context(self):
            return nullcontext()

    monkeypatch.setattr(sglang_rollout, "GenerateState", UnblockedState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: custom_generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", fake_rm)
    args = Namespace(
        custom_generate_function_path="custom",
        eval_termination_retry_times=4,
        group_rm=False,
        rm_type="",
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )

    sample = asyncio.run(sglang_rollout.generate_and_rm(args, Sample(index=7, prompt="q"), {}, evaluation=True))

    assert attempts == 3
    assert scored_responses == ["attempt-3"]
    assert sample.metadata["eval_retry_count"] == 2
    assert sample.metadata["eval_retry_termination_reasons"] == [
        "max_context_len_exceeded",
        "max_context_len_exceeded",
    ]


def test_eval_does_not_retry_reasoning_only_termination(monkeypatch):
    attempts = 0
    scored_responses = []

    async def custom_generate(_args, sample, _sampling_params, evaluation=False):
        nonlocal attempts
        assert evaluation is True
        attempts += 1
        sample.response = "reasoning"
        sample.status = Sample.Status.COMPLETED
        sample.metadata["fused_termination"] = "reasoning_only"
        return sample

    async def fake_rm(_args, sample, evaluation=False):
        assert evaluation is True
        scored_responses.append(sample.response)
        return 1.0

    class UnblockedState:
        aborted = False
        semaphore = asyncio.Semaphore(1)

        def __init__(self, _args):
            pass

        def dp_rank_context(self):
            return nullcontext()

    monkeypatch.setattr(sglang_rollout, "GenerateState", UnblockedState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: custom_generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", fake_rm)
    args = Namespace(
        custom_generate_function_path="custom",
        eval_termination_retry_times=4,
        group_rm=False,
        rm_type="",
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )

    sample = asyncio.run(sglang_rollout.generate_and_rm(args, Sample(index=7, prompt="q"), {}, evaluation=True))

    assert attempts == 1
    assert scored_responses == ["reasoning"]
    assert sample.metadata["eval_retry_count"] == 0
    assert sample.metadata["eval_retry_termination_reasons"] == []


def test_eval_does_not_retry_duplicate_search_termination(monkeypatch):
    attempts = 0

    async def custom_generate(_args, sample, _sampling_params, evaluation=False):
        nonlocal attempts
        assert evaluation is True
        attempts += 1
        sample.reward = 0.0
        sample.status = Sample.Status.COMPLETED
        sample.metadata["fused_termination"] = "cut_bill_duplicate_search"
        return sample

    class UnblockedState:
        aborted = False
        semaphore = asyncio.Semaphore(1)

        def __init__(self, _args):
            pass

        def dp_rank_context(self):
            return nullcontext()

    monkeypatch.setattr(sglang_rollout, "GenerateState", UnblockedState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: custom_generate)
    args = Namespace(
        custom_generate_function_path="custom",
        eval_termination_retry_times=4,
        group_rm=False,
        rm_type="",
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )

    sample = asyncio.run(sglang_rollout.generate_and_rm(args, Sample(prompt="q"), {}, evaluation=True))

    assert attempts == 1
    assert sample.metadata["eval_retry_count"] == 0
    assert sample.metadata["eval_retry_termination_reasons"] == []


def test_eval_stops_after_configured_termination_retries(monkeypatch):
    attempts = 0

    async def custom_generate(_args, sample, _sampling_params, evaluation=False):
        nonlocal attempts
        attempts += 1
        sample.reward = 0.0
        sample.status = Sample.Status.COMPLETED
        sample.metadata["termination_reason"] = "ABNORMAL_EVAL_RESPONSE"
        return sample

    class UnblockedState:
        aborted = False
        semaphore = asyncio.Semaphore(1)

        def __init__(self, _args):
            pass

        def dp_rank_context(self):
            return nullcontext()

    monkeypatch.setattr(sglang_rollout, "GenerateState", UnblockedState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: custom_generate)
    args = Namespace(
        custom_generate_function_path="custom",
        eval_termination_retry_times=4,
        group_rm=False,
        rm_type="",
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
    )

    sample = asyncio.run(sglang_rollout.generate_and_rm(args, Sample(prompt="q"), {}, evaluation=True))

    assert attempts == 5
    assert sample.metadata["eval_retry_count"] == 4
    assert sample.metadata["eval_retry_termination_reasons"] == ["ABNORMAL_EVAL_RESPONSE"] * 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
