import asyncio
import gc
from types import SimpleNamespace
import weakref

import pytest
import slime.rollout.sglang_rollout as sglang_rollout
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

NUM_GPUS = 0


class _DoneTask:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result

    def done(self):
        return True


class _FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.remaining_batch_size = 0
        self.pendings = set()
        self.pending_groups = {}
        self._submitted = False

    def submit_generate_tasks(self, samples):
        if self._submitted:
            return
        self._submitted = True
        self.remaining_batch_size += len(samples)
        self.pendings.add(object())

    def reset(self):
        pass


def _sample(index: int, reward: float):
    return Sample(
        index=index,
        rollout_id=index,
        prompt="prompt",
        response="response",
        reward=reward,
        response_length=1,
        loss_mask=[1],
    )


def test_sync_dynamic_filter_receives_flattened_fanout_group(monkeypatch):
    group = [[_sample(0, 0.0), _sample(1, 1.0)]]
    seen_samples = []

    def dynamic_filter(args, samples, **kwargs):
        seen_samples.append(samples)
        assert all(isinstance(sample, Sample) for sample in samples)
        return True

    async def fake_wait(pendings, return_when, timeout=None):
        return {_DoneTask(group)}, set()

    monkeypatch.setattr(sglang_rollout, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(sglang_rollout, "load_function", lambda path: dynamic_filter)
    monkeypatch.setattr(asyncio, "wait", fake_wait)
    monkeypatch.setattr(sglang_rollout, "abort", lambda args, rollout_id: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(sglang_rollout, "maybe_print_rollout_group", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        over_sampling_batch_size=1,
        n_samples_per_prompt=1,
        dynamic_sampling_filter_path="test.dynamic_filter",
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
    )

    output, aborted_samples = asyncio.run(sglang_rollout.generate_rollout_async(args, rollout_id=0, data_source=lambda num_samples: [[_sample(0, 0.0)]]))

    assert aborted_samples == []
    assert output.samples == [group]
    assert seen_samples == [[group[0][0], group[0][1]]]


@pytest.mark.parametrize("process_all_samples", [False, True])
def test_sync_dynamic_filter_only_retains_dropped_groups_for_processor(monkeypatch, process_all_samples):
    dropped_group = [[_sample(0, 0.0)]]
    kept_group = [[_sample(1, 1.0)]]
    dropped_sample_ref = weakref.ref(dropped_group[0][0])
    processed_samples = []
    filter_calls = 0
    wait_calls = 0

    class GenerateState:
        def __init__(self, args):
            self.remaining_batch_size = 0
            self.pendings = set()
            self.pending_groups = {}

        def submit_generate_tasks(self, samples):
            self.remaining_batch_size += len(samples)
            self.pendings.add(object())

        def reset(self):
            pass

    def dynamic_filter(args, samples, **kwargs):
        nonlocal filter_calls, dropped_group
        filter_calls += 1
        if filter_calls == 1:
            dropped_group = None
            return DynamicFilterOutput(keep=False, reason="drop")

        gc.collect()
        assert (dropped_sample_ref() is not None) is process_all_samples
        return DynamicFilterOutput(keep=True)

    def process_func(args, samples, data_source):
        processed_samples.extend(samples)

    def load_function(path):
        return process_func if path == "test.process_all_samples" else dynamic_filter

    async def fake_wait(pendings, return_when, timeout=None):
        nonlocal wait_calls
        group = dropped_group if wait_calls == 0 else kept_group
        wait_calls += 1
        return {_DoneTask(group)}, set()

    monkeypatch.setattr(sglang_rollout, "GenerateState", GenerateState)
    monkeypatch.setattr(sglang_rollout, "load_function", load_function)
    monkeypatch.setattr(asyncio, "wait", fake_wait)
    monkeypatch.setattr(sglang_rollout, "abort", lambda args, rollout_id: asyncio.sleep(0, result=[]))
    monkeypatch.setattr(sglang_rollout, "maybe_print_rollout_group", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        over_sampling_batch_size=1,
        n_samples_per_prompt=1,
        dynamic_sampling_filter_path="test.dynamic_filter",
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path="test.process_all_samples" if process_all_samples else None,
    )

    output, aborted_samples = asyncio.run(
        sglang_rollout.generate_rollout_async(
            args,
            rollout_id=0,
            data_source=lambda num_samples: [[_sample(filter_calls, 0.0)]],
        )
    )

    assert aborted_samples == []
    assert output.samples == [kept_group]
    assert [group[0][0].index for group in processed_samples] == ([0, 1] if process_all_samples else [])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
