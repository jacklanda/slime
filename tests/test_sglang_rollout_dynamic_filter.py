import asyncio
from types import SimpleNamespace

import slime.rollout.sglang_rollout as sglang_rollout
from slime.utils.types import Sample


class _DoneTask:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.remaining_batch_size = 0
        self.pendings = set()
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

    async def fake_wait(pendings, return_when):
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

    output, aborted_samples = asyncio.run(
        sglang_rollout.generate_rollout_async(args, rollout_id=0, data_source=lambda num_samples: [[_sample(0, 0.0)]])
    )

    assert aborted_samples == []
    assert output.samples == [group]
    assert seen_samples == [[group[0][0], group[0][1]]]
