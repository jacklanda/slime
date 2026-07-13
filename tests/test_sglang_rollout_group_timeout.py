import asyncio
from argparse import Namespace
from contextlib import nullcontext

import pytest

from slime.rollout import sglang_rollout
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.types import Sample

NUM_GPUS = 0


async def _fake_generate_and_rm(_args, sample, _sampling_params, evaluation=False):
    if sample.index == 1:
        await sglang_rollout.asyncio.sleep(10)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
