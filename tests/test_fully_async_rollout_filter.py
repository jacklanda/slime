import asyncio
from types import SimpleNamespace

import slime.rollout.fully_async_rollout as fully_async
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample


class FakeWorker:
    def __init__(self, groups):
        self.groups = list(enumerate(groups))
        self.resumed = False
        self.paused = False

    def get_completed_groups(self):
        groups = self.groups
        self.groups = []
        return groups

    def queue_size(self):
        return 0

    def resume(self):
        self.resumed = True

    def pause(self):
        self.paused = True


def _sample(
    index: int,
    reward: float,
    family: str | None = None,
    *,
    steps: int | None = None,
    termination: str | None = None,
    credit_event: str | None = None,
):
    metadata = {"fused_task_type": family} if family else {}
    if steps is not None:
        metadata["fused_traj_steps"] = steps
    if termination is not None:
        metadata["fused_termination"] = termination
    if credit_event is not None:
        metadata["credit_assignment_event"] = credit_event
    return Sample(index=index, rollout_id=index, reward=reward, response_length=1, loss_mask=[1], metadata=metadata)


def test_fully_async_dynamic_filter_keeps_nonzero_variance_groups(monkeypatch):
    groups = [
        [_sample(0, 0.0), _sample(1, 0.0)],
        [_sample(2, 0.0), _sample(3, 1.0)],
    ]
    worker = FakeWorker(groups)

    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        reward_key=None,
    )

    output = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert isinstance(output, RolloutFnTrainOutput)
    assert output.samples == [groups[1]]
    assert output.metrics["rollout/dynamic_filter/completed_groups"] == 2
    assert output.metrics["rollout/dynamic_filter/dropped_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/kept_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1
    assert worker.resumed is True
    assert worker.paused is True


def test_fully_async_omits_candidate_and_selected_fused_distributions(monkeypatch):
    groups = [
        [
            _sample(0, 0.0, "web_search", steps=2, termination="ABNORMAL_REPEATED_QUERY", credit_event="repeated_search_query"),
            _sample(1, 0.0, "web_search", steps=2, termination="ABNORMAL_REPEATED_QUERY", credit_event="repeated_search_query"),
        ],
        [
            _sample(2, 0.0, "web_search", steps=4, termination="env_done"),
            _sample(3, 1.0, "web_search", steps=4, termination="env_done"),
        ],
    ]
    worker = FakeWorker(groups)

    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("FUSED_WEBQA_MIN_UNIQUE_SEARCHES", "3")
    monkeypatch.setenv("FUSED_REPEATED_SEARCH_MAX_STRIKES", "4")

    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        fully_async_filter_relax_after_groups=0,
        reward_key=None,
    )

    output = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert output.samples == [groups[1]]
    assert not any(key.startswith("rollout/candidate/") for key in output.metrics)
    assert not any(key.startswith("rollout/selected/") for key in output.metrics)
    assert output.metrics["rollout/config/fused_webqa_min_unique_searches"] == 3
    assert output.metrics["rollout/config/fused_repeated_search_max_strikes"] == 4


def test_fully_async_dynamic_filter_relaxes_after_completed_group_budget(monkeypatch):
    groups = [
        [_sample(0, 0.0), _sample(1, 0.0)],
        [_sample(2, 0.0), _sample(3, 0.0)],
    ]
    worker = FakeWorker(groups)

    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)

    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path="slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std",
        fully_async_filter_relax_after_groups=2,
        reward_key=None,
    )

    output = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert output.samples == [groups[1]]
    assert output.metrics["rollout/dynamic_filter/completed_groups"] == 2
    assert output.metrics["rollout/dynamic_filter/dropped_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/kept_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1
    assert output.metrics["rollout/dynamic_filter/drop_relaxed_zero_std_0.0"] == 1


def test_task_family_quota_selection_prefers_requested_mix():
    groups = [
        [_sample(0, 1.0, "mcp")],
        [_sample(1, 1.0, "mcp")],
        [_sample(2, 1.0, "web_search")],
        [_sample(3, 1.0, "cli")],
    ]
    args = SimpleNamespace(rollout_task_family_quotas="webqa=0.5,mcp=0.25,cli=0.25")

    selected = fully_async._select_task_family_quota_groups(groups, target=4, args=args)

    families = [fully_async._sample_group_task_family(group) for group in selected]
    assert families.count("webqa") == 1
    assert families.count("mcp") == 2
    assert families.count("cli") == 1


def test_task_family_quota_selection_fills_missing_family_from_remaining():
    groups = [
        [_sample(0, 1.0, "mcp")],
        [_sample(1, 1.0, "mcp")],
        [_sample(2, 1.0, "web_search")],
    ]
    args = SimpleNamespace(rollout_task_family_quotas="webqa=0.5,cli=0.5")

    selected = fully_async._select_task_family_quota_groups(groups, target=3, args=args)

    assert selected == [groups[2], groups[0], groups[1]]
