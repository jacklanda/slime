import asyncio
import io
from argparse import Namespace
from collections import Counter
from contextlib import nullcontext

import pytest

from slime.rollout import sglang_rollout
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.rollout.task_family import select_task_family_quota_groups
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.types import Sample

NUM_GPUS = 0


def test_eval_progress_log_stream_emits_complete_lines():
    sink = io.StringIO()
    stream = sglang_rollout._EvalProgressLogStream(sink)

    stream.write("\rEval mixed (3 datasets): 2%| | 85/3622")

    assert sink.getvalue() == "Eval mixed (3 datasets): 2%| | 85/3622\n"


def test_same_weight_sglang_prefill_diagnostic_aligns_response_tokens(monkeypatch):
    monkeypatch.setenv("SLIME_DIAGNOSE_SGLANG_PREFILL_LOGPROBS", "1")
    monkeypatch.setenv("SLIME_DIAGNOSE_SGLANG_HIDDEN_STATES", "1")
    requests = []

    async def fake_post(url, payload, max_retries):
        requests.append((url, payload, max_retries))
        return {
            "meta_info": {
                "weight_version": "7",
                "input_token_logprobs": [[None, 11], [-0.41, 12], [-0.52, 13]],
                "hidden_states": [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]],
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    sample = Sample(
        index=3,
        tokens=[10, 11, 12, 13],
        response_length=2,
        rollout_log_probs=[-0.4, -0.5],
        loss_mask=[1, 1],
        policy_loss_mask=[0, 1],
        metadata={"rollout_weight_version": "7", "fused_task_type": "mcp"},
    )

    metrics = asyncio.run(
        sglang_rollout._diagnose_sglang_prefill_logprobs(
            Namespace(sglang_router_ip="router", sglang_router_port=30000),
            [[sample]],
            rollout_id=4,
        )
    )

    assert requests[0][0] == "http://router:30000/generate"
    assert requests[0][1]["input_ids"] == sample.tokens
    assert requests[0][1]["logprob_start_len"] == 1
    assert requests[0][1]["return_hidden_states"] is True
    assert requests[0][2] == 1
    assert sample.metadata["sglang_prefill_diagnostic"] == {
        "status": "ok",
        "rollout_id": 4,
        "task_family": "mcp",
        "response_length": 2,
        "expected_weight_version": "7",
        "observed_weight_version": "7",
        "hidden_probe_columns": [0, 1],
        "sglang_final_hidden_probe": [[2.0, 20.0], [3.0, 30.0]],
        "compared_tokens": 1,
        "decode_prefill_abs_diff": pytest.approx(0.02),
        "prefill_log_probs": [-0.41, -0.52],
    }
    assert metrics["rollout/diagnostic/sglang_prefill_successful_samples"] == 1
    assert metrics["rollout/diagnostic/sglang_decode_prefill_compared_tokens"] == 1
    assert metrics["rollout/diagnostic/sglang_decode_prefill_abs_diff"] == pytest.approx(0.02)


def test_sglang_prefill_diagnostic_rejects_different_weight_version(monkeypatch):
    monkeypatch.setenv("SLIME_DIAGNOSE_SGLANG_PREFILL_LOGPROBS", "1")

    async def fake_post(_url, _payload, max_retries):
        assert max_retries == 1
        return {
            "meta_info": {
                "weight_version": "8",
                "input_token_logprobs": [[None, 10], [-0.1, 11]],
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    sample = Sample(
        index=5,
        tokens=[10, 11],
        response_length=1,
        rollout_log_probs=[-0.1],
        metadata={"rollout_weight_version": "7"},
    )

    metrics = asyncio.run(
        sglang_rollout._diagnose_sglang_prefill_logprobs(
            Namespace(sglang_router_ip="router", sglang_router_port=30000),
            [[sample]],
            rollout_id=4,
        )
    )

    diagnostic = sample.metadata["sglang_prefill_diagnostic"]
    assert diagnostic["status"] == "weight_version_mismatch"
    assert diagnostic["expected_weight_version"] == "7"
    assert diagnostic["observed_weight_version"] == "8"
    assert "prefill_log_probs" not in diagnostic
    assert metrics["rollout/diagnostic/sglang_prefill_successful_samples"] == 0
    assert metrics["rollout/diagnostic/sglang_decode_prefill_compared_tokens"] == 0


def test_sglang_prefill_recomputes_all_rollout_logprobs(monkeypatch):
    monkeypatch.setenv("SLIME_DIAGNOSE_SGLANG_PREFILL_LOGPROBS", "0")
    monkeypatch.setenv("SLIME_SGLANG_RECOMPUTE_ROLLOUT_LOGPROBS", "1")
    requests = []

    async def fake_post(_url, payload, max_retries):
        assert max_retries == 1
        requests.append(payload)
        token = payload["input_ids"][-1]
        return {
            "meta_info": {
                "weight_version": "9",
                "input_token_logprobs": [[None, 1], [-float(token), token]],
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    samples = [
        Sample(
            index=index,
            tokens=[1, token],
            response_length=1,
            rollout_log_probs=[-0.1],
            metadata={"rollout_weight_version": "9"},
        )
        for index, token in enumerate((2, 3))
    ]

    metrics = asyncio.run(
        sglang_rollout._diagnose_sglang_prefill_logprobs(
            Namespace(sglang_router_ip="router", sglang_router_port=30000),
            [[sample] for sample in samples],
            rollout_id=6,
        )
    )

    assert [sample.rollout_log_probs for sample in samples] == [[-2.0], [-3.0]]
    assert all("return_hidden_states" not in payload for payload in requests)
    assert {payload["extra_key"] for payload in requests} == {
        "slime-first-divergence-6-0",
        "slime-first-divergence-6-1",
    }
    assert metrics["rollout/diagnostic/sglang_recomputed_samples"] == 2


def test_sglang_prefill_recompute_fails_closed_on_weight_mismatch(monkeypatch):
    monkeypatch.setenv("SLIME_DIAGNOSE_SGLANG_PREFILL_LOGPROBS", "0")
    monkeypatch.setenv("SLIME_SGLANG_RECOMPUTE_ROLLOUT_LOGPROBS", "1")

    async def fake_post(_url, _payload, max_retries):
        assert max_retries == 1
        return {
            "meta_info": {
                "weight_version": "10",
                "input_token_logprobs": [[None, 1], [-0.2, 2]],
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    sample = Sample(
        index=0,
        tokens=[1, 2],
        response_length=1,
        rollout_log_probs=[-0.1],
        metadata={"rollout_weight_version": "9"},
    )

    with pytest.raises(RuntimeError, match="successful=0, requested=1"):
        asyncio.run(
            sglang_rollout._diagnose_sglang_prefill_logprobs(
                Namespace(sglang_router_ip="router", sglang_router_port=30000),
                [[sample]],
                rollout_id=6,
            )
        )
    assert sample.rollout_log_probs == [-0.1]


def test_sglang_prefill_recompute_is_atomic(monkeypatch):
    monkeypatch.setenv("SLIME_DIAGNOSE_SGLANG_PREFILL_LOGPROBS", "0")
    monkeypatch.setenv("SLIME_SGLANG_RECOMPUTE_ROLLOUT_LOGPROBS", "1")

    async def fake_post(_url, payload, max_retries):
        assert max_retries == 1
        token = payload["input_ids"][-1]
        return {
            "meta_info": {
                "weight_version": "wrong" if token == 3 else "9",
                "input_token_logprobs": [[None, 1], [-float(token), token]],
            }
        }

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    samples = [
        Sample(
            index=index,
            tokens=[1, token],
            response_length=1,
            rollout_log_probs=[-0.1],
            metadata={"rollout_weight_version": "9"},
        )
        for index, token in enumerate((2, 3))
    ]

    with pytest.raises(RuntimeError, match="successful=1, requested=2"):
        asyncio.run(
            sglang_rollout._diagnose_sglang_prefill_logprobs(
                Namespace(sglang_router_ip="router", sglang_router_port=30000),
                [[sample] for sample in samples],
                rollout_id=6,
            )
        )
    assert [sample.rollout_log_probs for sample in samples] == [[-0.1], [-0.1]]


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
    assert out[1].metadata["rollout_timeout_stage"] == "unknown"
    assert out[1].metadata["failure_class"] == "policy_failure"


def test_generate_group_cancellation_reaps_sample_tasks(monkeypatch):
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def wait_until_cancelled(_args, sample, _sampling_params, evaluation=False):
        child_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            child_cancelled.set()

    async def run_test():
        monkeypatch.setattr(sglang_rollout, "generate_and_rm", wait_until_cancelled)
        group_task = asyncio.create_task(
            sglang_rollout.generate_and_rm_group(
                Namespace(
                    sglang_enable_deterministic_inference=False,
                    group_rm=False,
                    rollout_infra_retry_times=0,
                ),
                [Sample(index=0, prompt="a")],
                sampling_params={},
            )
        )
        await child_started.wait()
        group_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await group_task
        assert child_cancelled.is_set()

    asyncio.run(run_test())


def test_sync_rollout_enforces_task_family_quota(monkeypatch):
    groups = []
    for index, family in enumerate(
        ["mcp", "mcp", "mcp", "web_search", "mcp", "web_search", "web_search", "web_search"]
    ):
        sample = Sample(
            index=index,
            prompt=f"q{index}",
            response="answer",
            response_length=1,
            reward=1.0,
            status=Sample.Status.COMPLETED,
            metadata={"fused_task_type": family, "fused_traj_steps": index},
        )
        groups.append([sample])

    class QuotaGenerateState:
        def __init__(self, _args):
            self.reset()

        def reset(self):
            self.remaining_batch_size = 0
            self.pendings = set()
            self.pending_groups = {}
            self.aborted = False

        def submit_generate_tasks(self, submitted_groups):
            async def complete(group):
                return group

            for group in submitted_groups:
                task = asyncio.create_task(complete(group))
                self.pendings.add(task)
                self.pending_groups[task] = group
            self.remaining_batch_size += len(submitted_groups)

    source_batches = [groups[:4], groups[4:]]

    def data_source(_num_samples):
        return source_batches.pop(0)

    monkeypatch.setattr(sglang_rollout, "GenerateState", QuotaGenerateState)
    monkeypatch.setattr(sglang_rollout, "maybe_print_rollout_group", lambda *_args, **_kwargs: None)
    args = Namespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=4,
        rollout_task_family_quotas="mcp=0.5,webqa=0.5",
        rollout_task_family_top_mean_steps=True,
        rollout_all_samples_process_path=None,
        fully_async_filter_relax_after_groups=0,
        n_samples_per_prompt=1,
        over_sampling_batch_size=4,
        rollout_sample_filter_path=None,
    )

    output, aborted = asyncio.run(sglang_rollout.generate_rollout_async(args, 0, data_source))

    families = [group[0].metadata["fused_task_type"] for group in output.samples]
    assert families.count("mcp") == 2
    assert families.count("web_search") == 2
    assert [group[0].index for group in output.samples] == [2, 4, 6, 7]
    assert output.metrics["rollout/task_family_selected/mcp"] == 2
    assert output.metrics["rollout/task_family_selected/webqa"] == 2
    assert output.metrics["rollout/task_family_selected_mean_steps/mcp"] == 3
    assert output.metrics["rollout/task_family_selected_mean_steps/webqa"] == 6.5
    assert output.metrics["rollout/config/task_family_top_mean_steps"] == 1
    assert aborted == []
    assert source_batches == []


def test_task_family_quota_selects_top_mean_steps_and_dedupes_trajectory_segments():
    mcp_high_max = [
        Sample(index=0, metadata={"fused_task_type": "mcp", "parent_traj_id": "a", "fused_traj_steps": 10}),
        Sample(index=1, metadata={"fused_task_type": "mcp", "parent_traj_id": "a", "fused_traj_steps": 10}),
        Sample(index=2, metadata={"fused_task_type": "mcp", "parent_traj_id": "b", "fused_traj_steps": 0}),
    ]
    mcp_high_mean = [
        Sample(index=3, metadata={"fused_task_type": "mcp", "parent_traj_id": "c", "fused_traj_steps": 6}),
        Sample(index=4, metadata={"fused_task_type": "mcp", "parent_traj_id": "d", "fused_traj_steps": 6}),
    ]
    webqa_low = [Sample(index=5, metadata={"fused_task_type": "webqa", "fused_traj_steps": 3})]
    webqa_high = [Sample(index=6, metadata={"fused_task_type": "webqa", "fused_traj_steps": 9})]

    selected = select_task_family_quota_groups(
        [mcp_high_max, webqa_low, mcp_high_mean, webqa_high],
        target=2,
        quota_spec="webqa=0.5,mcp=0.5",
        prefer_higher_mean_steps=True,
    )

    assert selected == [webqa_high, mcp_high_mean]


def test_sync_rollout_progress_log_breaks_down_task_families(monkeypatch, caplog):
    groups = []
    family_and_keep = [
        *(("webqa", True) for _ in range(5)),
        *(("mcp", True) for _ in range(2)),
        *(("webqa", False) for _ in range(2)),
        ("mcp", False),
    ]
    for index, (family, keep) in enumerate(family_and_keep):
        groups.append(
            [
                Sample(
                    index=index,
                    prompt=f"q{index}",
                    response="answer",
                    response_length=1,
                    reward=1.0,
                    status=Sample.Status.COMPLETED,
                    metadata={"fused_task_type": family, "keep": keep},
                )
            ]
        )
    final_group = [
        Sample(
            index=10,
            prompt="q10",
            response="answer",
            response_length=1,
            reward=1.0,
            status=Sample.Status.COMPLETED,
            metadata={"fused_task_type": "mcp", "keep": True},
        )
    ]

    class ProgressGenerateState:
        def __init__(self, _args):
            self.reset()

        def reset(self):
            self.remaining_batch_size = 0
            self.pendings = set()
            self.pending_groups = {}
            self.aborted = False

        def submit_generate_tasks(self, submitted_groups):
            async def complete(group):
                return group

            for group in submitted_groups:
                task = asyncio.create_task(complete(group))
                self.pendings.add(task)
                self.pending_groups[task] = group
            self.remaining_batch_size += len(submitted_groups)

    source_batches = [groups, [final_group]]

    def data_source(_num_samples):
        return source_batches.pop(0)

    def dynamic_filter(_args, samples, **_kwargs):
        return DynamicFilterOutput(keep=samples[0].metadata["keep"], reason="test_drop")

    class ProgressClock:
        values = iter([1000.0, 1337.8, 1338.0])

        @classmethod
        def time(cls):
            return next(cls.values)

    monkeypatch.setattr(sglang_rollout, "GenerateState", ProgressGenerateState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda _path: dynamic_filter)
    monkeypatch.setattr(sglang_rollout, "maybe_print_rollout_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sglang_rollout, "time", ProgressClock)
    args = Namespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path="test.dynamic_filter",
        rollout_batch_size=8,
        rollout_task_family_quotas=None,
        rollout_all_samples_process_path=None,
        fully_async_filter_relax_after_groups=0,
        n_samples_per_prompt=1,
        over_sampling_batch_size=10,
        rollout_sample_filter_path=None,
    )

    with caplog.at_level("INFO", logger="slime.rollout.sglang_rollout"):
        output, aborted = asyncio.run(sglang_rollout.generate_rollout_async(args, 1, data_source))

    assert len(output.samples) == 8
    assert aborted == []
    assert (
        "rollout 1: valid=7/8 (webqa:5, mcp:2), "
        "dropped=3/10 (webqa:2, mcp:1), pending=0, elapsed=337s"
    ) in caplog.messages


def test_sync_rollout_admission_only_quota_never_collects_past_target(monkeypatch):
    groups = []
    for index, family in enumerate(["webqa", "mcp", "webqa", "mcp", "webqa", "mcp"]):
        groups.append(
            [
                Sample(
                    index=index,
                    prompt=f"q{index}",
                    response="answer",
                    response_length=1,
                    reward=1.0,
                    status=Sample.Status.COMPLETED,
                    metadata={"fused_task_type": family},
                )
            ]
        )

    class AdmissionGenerateState:
        def __init__(self, _args):
            self.reset()

        def reset(self):
            self.remaining_batch_size = 0
            self.pendings = set()
            self.pending_groups = {}
            self.aborted = False

        def submit_generate_tasks(self, submitted_groups):
            async def complete(group):
                return group

            for group in submitted_groups:
                task = asyncio.create_task(complete(group))
                self.pendings.add(task)
                self.pending_groups[task] = group
            self.remaining_batch_size += len(submitted_groups)

    class AdmissionSource:
        plans = []

        def get_samples_by_family(self, plan):
            self.plans.append(plan)
            return groups

    source = AdmissionSource()
    monkeypatch.setenv("SLIME_SYNC_MIN_PENDING_GROUPS", "6")
    monkeypatch.setattr(sglang_rollout, "GenerateState", AdmissionGenerateState)
    monkeypatch.setattr(sglang_rollout, "maybe_print_rollout_group", lambda *_args, **_kwargs: None)
    args = Namespace(
        rollout_global_dataset=True,
        dynamic_sampling_filter_path=None,
        rollout_batch_size=4,
        rollout_task_family_quotas="webqa=0.5,mcp=0.5",
        rollout_task_family_admission_only=True,
        rollout_all_samples_process_path=None,
        fully_async_filter_relax_after_groups=0,
        n_samples_per_prompt=1,
        over_sampling_batch_size=6,
        rollout_sample_filter_path=None,
    )

    output, aborted = asyncio.run(sglang_rollout.generate_rollout_async(args, 0, source))

    assert len(output.samples) == 4
    assert source.plans == [{"webqa": 3, "mcp": 3}]
    assert output.metrics["rollout/dynamic_filter/completed_groups"] == 6
    assert output.metrics["rollout/task_family_submitted/webqa"] == 3
    assert output.metrics["rollout/task_family_submitted/mcp"] == 3
    assert output.metrics["rollout/config/task_family_admission_only"] == 1
    assert aborted == []


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


def test_generate_group_retries_only_failed_infra_slot(monkeypatch):
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    calls = []

    async def fail_once(_args, sample, sampling_params, evaluation=False):
        calls.append((sample.index, sampling_params.get("sampling_seed"), sample.session_id))
        if sample.index == 1 and sum(index == 1 for index, _seed, _session in calls) == 1:
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
            sample.metadata["failure_class"] = "retryable_infra"
            return sample
        sample.status = Sample.Status.COMPLETED
        sample.reward = 1.0
        return sample

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fail_once)
    group = [Sample(index=0, prompt="a"), Sample(index=1, prompt="b")]
    out = asyncio.run(
        sglang_rollout.generate_and_rm_group(
            Namespace(
                sglang_enable_deterministic_inference=False,
                group_rm=False,
                rollout_infra_retry_times=1,
            ),
            group,
            sampling_params={"sampling_seed": 71},
        )
    )

    assert [index for index, _seed, _session in calls] == [0, 1, 1]
    assert [seed for _index, seed, _session in calls] == [71, 71, 71]
    assert calls[1][2] != calls[2][2]
    assert all(sample.status == Sample.Status.COMPLETED for sample in out)
    assert out[0].metadata["infra_retry_count"] == 0
    assert out[1].metadata["infra_retry_count"] == 1


def test_generate_group_does_not_retry_permanent_task_failure(monkeypatch):
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    calls = 0

    async def fail_permanently(_args, sample, _sampling_params, evaluation=False):
        nonlocal calls
        calls += 1
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        sample.metadata["failure_class"] = "permanent_task_failure"
        return sample

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fail_permanently)
    out = asyncio.run(
        sglang_rollout.generate_and_rm_group(
            Namespace(
                sglang_enable_deterministic_inference=False,
                group_rm=False,
                rollout_infra_retry_times=3,
            ),
            [Sample(index=0, prompt="a")],
            sampling_params={},
        )
    )

    assert calls == 1
    assert out[0].metadata["failure_class"] == "permanent_task_failure"
    assert out[0].metadata["infra_retry_count"] == 0


@pytest.mark.parametrize(
    ("stage", "failure_class"),
    [
        ("tool_operation", "policy_failure"),
        ("llm_decode", "policy_failure"),
        ("verifier", "permanent_task_failure"),
    ],
)
def test_group_timeout_classifies_current_trajectory_stage(stage, failure_class):
    sample = Sample(index=0, metadata={"rollout_stage": stage})

    timed_out = sglang_rollout._timeout_sample(sample, evaluation=False)

    assert timed_out.metadata["rollout_timeout_stage"] == stage
    assert timed_out.metadata["failure_class"] == failure_class


@pytest.mark.parametrize(
    ("status_code", "failure_class"),
    [
        (429, "retryable_infra"),
        (503, "retryable_infra"),
        (400, "permanent_task_failure"),
    ],
)
def test_http_failure_classification_distinguishes_transient_statuses(status_code, failure_class):
    response = type("Response", (), {"status_code": status_code})()
    exc = type("HTTPFailure", (RuntimeError,), {})(f"HTTP {status_code}")
    exc.response = response

    assert sglang_rollout._exception_failure_class(exc).value == failure_class


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


class _FakeTimeoutThenCompletedRolloutGenerateState(_FakeRolloutGenerateState):
    def submit_generate_tasks(self, samples):
        for group in samples:
            if group[0].index < 2:
                task = sglang_rollout.asyncio.create_task(sglang_rollout.asyncio.sleep(10))
            else:
                for sample in group:
                    sample.status = Sample.Status.COMPLETED
                    sample.reward = 1.0
                    sample.response = "ok"
                    sample.response_length = 1
                task = sglang_rollout.asyncio.create_task(sglang_rollout.asyncio.sleep(0, result=group))
            self.pendings.add(task)
            self.pending_groups[task] = group
        self.remaining_batch_size += len(samples)


def test_rollout_collection_excludes_timeout_group_from_training(monkeypatch):
    monkeypatch.setenv("SLIME_ROLLOUT_GROUP_TIMEOUT", "1")
    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeTimeoutThenCompletedRolloutGenerateState)

    source_groups = [
        [Sample(index=0, prompt="a", reward=None), Sample(index=1, prompt="b", reward=None)],
        [Sample(index=2, prompt="c", reward=None), Sample(index=3, prompt="d", reward=None)],
    ]

    def data_source(_batch_size):
        return [source_groups.pop(0)]

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
                rollout_infra_retry_times=0,
            ),
            rollout_id=0,
            data_source=data_source,
        )
    )

    assert aborted == []
    group = out.samples[0]
    assert len(group) == 2
    assert [sample.index for sample in group] == [2, 3]
    assert all(sample.status == Sample.Status.COMPLETED for sample in group)
    assert out.metrics["rollout/dynamic_filter/drop_policy_failure"] == 1


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


def test_abort_cancels_pending_before_using_known_engine_urls(monkeypatch):
    pending_group = [Sample(index=0, prompt="a", response="partial")]
    events = []

    class AbortGenerateState:
        def __init__(self, _args):
            self.aborted = False
            self.pending_groups = {}
            self.pendings = set()
            task = asyncio.create_task(asyncio.sleep(10))
            task.add_done_callback(lambda _task: events.append("cancelled"))
            self.pendings.add(task)
            self.pending_groups[task] = pending_group

    async def unexpected_router_request(_url):
        raise AssertionError("known engine URLs should avoid router discovery")

    aborted_urls = []

    async def abort_known_engines(urls):
        events.append("abort")
        aborted_urls.extend(urls)
        return False

    monkeypatch.setattr(sglang_rollout, "GenerateState", AbortGenerateState)
    monkeypatch.setattr(sglang_rollout, "get", unexpected_router_request)
    monkeypatch.setattr(sglang_rollout, "abort_servers_until_idle", abort_known_engines)

    aborted = asyncio.run(
        sglang_rollout.abort(
            Namespace(
                partial_rollout=True,
                sglang_router_ip="router",
                sglang_router_port=30000,
                sglang_engine_urls=["http://engine-0", None, "http://engine-1"],
            ),
            rollout_id=7,
        )
    )

    assert aborted_urls == ["http://engine-0", "http://engine-1"]
    assert events == ["cancelled", "abort"]
    assert aborted == [pending_group]
    assert pending_group[0].metadata["start_rollout_id"] == 7


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


def test_training_custom_generator_can_manage_decode_request_concurrency(monkeypatch):
    async def custom_generate(_args, sample, _sampling_params, evaluation=False):
        assert evaluation is False
        sample.reward = 1.0
        sample.response = "done"
        sample.status = Sample.Status.COMPLETED
        return sample

    custom_generate.manages_request_concurrency = True

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
            sglang_rollout.generate_and_rm(args, Sample(prompt="q"), {}, evaluation=False),
            timeout=0.2,
        )
    )

    assert sample.response == "done"


def test_task_family_submission_plan_compensates_for_lower_roi():
    plan = sglang_rollout._task_family_submission_plan(
        submit_limit=10,
        target=20,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter({"mcp": 1, "webqa": 4}),
        completed=Counter({"mcp": 8, "webqa": 5}),
    )

    assert sum(plan.values()) == 10
    assert plan["mcp"] > plan["webqa"]


def test_task_family_submission_plan_sizes_cold_start_by_quota_over_roi():
    plan = sglang_rollout._task_family_submission_plan(
        submit_limit=96,
        target=16,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter(),
        completed=Counter(),
    )

    assert plan == {"mcp": 16, "webqa": 16}


def test_task_family_submission_plan_refills_when_pending_yield_cannot_cover_tail():
    plan = sglang_rollout._task_family_submission_plan(
        submit_limit=10,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter({"mcp": 3, "webqa": 4}),
        completed=Counter({"mcp": 20, "webqa": 8}),
        pending=Counter({"mcp": 1}),
    )

    # Raw group counting sees 3 accepted + 1 pending MCP group and stalls.
    # At the observed ROI, that pending group cannot cover the expected deficit.
    assert plan == {"mcp": 4}


def test_task_family_submission_plan_does_not_overfill_healthy_pending_wave():
    plan = sglang_rollout._task_family_submission_plan(
        submit_limit=96,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter(),
        completed=Counter(),
        pending=Counter({"mcp": 8, "webqa": 8}),
    )

    assert plan == {}


def test_task_family_submission_plan_maintains_aggressive_pending_reservoir():
    plan = sglang_rollout._task_family_submission_plan(
        submit_limit=128,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter(),
        completed=Counter(),
        pending=Counter(),
        min_pending_groups=64,
    )

    assert plan == {"mcp": 32, "webqa": 32}


def test_task_family_submission_plan_stops_refilling_satisfied_family():
    plan = sglang_rollout._task_family_submission_plan(
        submit_limit=128,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter({"mcp": 3, "webqa": 4}),
        completed=Counter({"mcp": 20, "webqa": 8}),
        pending=Counter({"mcp": 1, "webqa": 20}),
        min_pending_groups=64,
    )

    assert plan == {"mcp": 63}


def test_task_family_submission_plan_reduces_only_mcp_only_reservoir():
    mcp_only_plan = sglang_rollout._task_family_submission_plan(
        submit_limit=128,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter({"mcp": 3, "webqa": 4}),
        completed=Counter({"mcp": 20, "webqa": 8}),
        pending=Counter({"mcp": 1, "webqa": 20}),
        min_pending_groups=24,
        mcp_only_min_pending_groups=8,
    )
    mixed_plan = sglang_rollout._task_family_submission_plan(
        submit_limit=128,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter(),
        completed=Counter(),
        pending=Counter(),
        min_pending_groups=24,
        mcp_only_min_pending_groups=8,
    )
    webqa_only_plan = sglang_rollout._task_family_submission_plan(
        submit_limit=128,
        target=8,
        quotas={"mcp": 0.5, "webqa": 0.5},
        accepted=Counter({"mcp": 4, "webqa": 3}),
        completed=Counter({"mcp": 8, "webqa": 8}),
        pending=Counter({"mcp": 20, "webqa": 1}),
        min_pending_groups=24,
        mcp_only_min_pending_groups=8,
    )

    assert mcp_only_plan == {"mcp": 7}
    assert mixed_plan == {"mcp": 12, "webqa": 12}
    assert webqa_only_plan == {"webqa": 23}


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
