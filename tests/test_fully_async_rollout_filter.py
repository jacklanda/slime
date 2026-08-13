import asyncio
import json
import threading
import time
from types import SimpleNamespace

import torch

import slime.rollout.fully_async_rollout as fully_async
from slime.rollout.sglang_rollout import generate_and_rm_group
from slime.ray.rollout import RolloutManager
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.dynamic_sampling_filters import is_infra_failure
from slime.utils.types import Sample


NUM_GPUS = 0


def test_fully_async_stop_cancels_inflight_tasks_and_joins_worker(monkeypatch):
    started = threading.Event()
    cleaned = threading.Event()

    async def generate_until_cancelled(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate_until_cancelled)
    monkeypatch.setattr(
        fully_async,
        "GenerateState",
        lambda _args: SimpleNamespace(sampling_params={}),
    )
    worker = fully_async.AsyncRolloutWorker(
        SimpleNamespace(rollout_only_inference_fast_path=True),
        FakeDataBuffer([[object()]]),
        concurrency=1,
    )
    worker.start()
    assert started.wait(timeout=2)

    worker.stop(cancel_inflight=True)

    assert cleaned.wait(timeout=2)
    assert not worker.worker_thread.is_alive()
    assert worker.queue_size() == 0


def test_fully_async_inference_work_limit_prevents_speculative_tail_groups(monkeypatch):
    generated = []

    async def generate_immediately(_args, group, **_kwargs):
        generated.append(group)
        return group

    monkeypatch.setattr(fully_async, "generate_and_rm_group", generate_immediately)
    monkeypatch.setattr(
        fully_async,
        "GenerateState",
        lambda _args: SimpleNamespace(sampling_params={}),
    )
    groups = [[_sample(index, 0.0)] for index in range(6)]
    data_buffer = FakeDataBuffer(groups)
    worker = fully_async.AsyncRolloutWorker(
        SimpleNamespace(rollout_only_inference_fast_path=True),
        data_buffer,
        concurrency=4,
    )
    worker.start()
    worker.resume(work_limit=2)
    deadline = time.monotonic() + 2
    while worker.queue_size() < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop(cancel_inflight=True)

    assert len(generated) == 2
    assert len(data_buffer.groups) == 4


def test_rollout_manager_dispose_stops_worker_only_for_inference_fast_path(monkeypatch):
    calls = []
    monkeypatch.setattr(fully_async, "shutdown_fully_async_rollout_worker", lambda **kwargs: calls.append(kwargs))
    rollout_manager_class = RolloutManager.__ray_metadata__.modified_class

    def dispose(fast_path):
        manager = rollout_manager_class.__new__(rollout_manager_class)
        manager.args = SimpleNamespace(
            rollout_only_inference_fast_path=fast_path,
            rollout_function_path="slime.rollout.fully_async_rollout.generate_rollout_fully_async",
        )
        manager._debug_dump_future = None
        manager._debug_dump_executor = None
        manager._health_monitors = []
        monkeypatch.setattr("slime.ray.rollout.logging_utils.finish_tracking", lambda _args: None)
        manager.dispose()

    dispose(False)
    assert calls == []

    dispose(True)
    assert calls == [{"cancel_inflight": True}]


class FakeDataBuffer:
    def __init__(self, groups):
        self.groups = list(groups)

    def get_samples(self, count):
        groups = self.groups[:count]
        self.groups = self.groups[count:]
        return groups

    def add_samples(self, groups):
        self.groups.extend(groups)

    def __len__(self):
        return len(self.groups)


def test_fully_async_adaptive_concurrency_uses_configured_moderate_window(monkeypatch):
    monkeypatch.setenv("SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY", "true")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_INITIAL_CONCURRENCY", "8")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_CONCURRENCY_STEP", "4")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_CONCURRENCY_POLL_INTERVAL", "10")
    args = SimpleNamespace(sglang_server_concurrency=64)

    controller = fully_async._create_adaptive_concurrency_controller(args, capacity=512)

    assert controller is not None
    assert controller.target == 8
    assert controller.minimum == 4
    assert controller.maximum == 16
    assert controller.step == 4
    assert controller.poll_interval == 10


def test_fully_async_adaptive_concurrency_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SLIME_FULLY_ASYNC_ADAPTIVE_CONCURRENCY", "false")

    controller = fully_async._create_adaptive_concurrency_controller(
        SimpleNamespace(sglang_server_concurrency=64),
        capacity=512,
    )

    assert controller is None


def test_fully_async_keep_all_preserves_failed_groups_across_shard_boundaries(monkeypatch):
    first = _sample(0, 0.0)
    first.status = Sample.Status.FAILED
    first.metadata["fused_error"] = "rollout_group_timeout"
    groups = [[first], [_sample(1, 1.0)]]
    worker = FakeWorker(groups)
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    first_shard = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))
    second_shard = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=1, data_buffer=None))

    assert first_shard.samples == [groups[0]]
    assert second_shard.samples == [groups[1]]
    assert worker.groups == []


def test_fully_async_keep_all_ignores_task_family_quota_candidate_sampling(monkeypatch):
    groups = [[_sample(0, 0.0, "mcp")], [_sample(1, 1.0, "web_search")]]
    worker = FakeWorker(groups)
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=1,
        rollout_task_family_quotas="mcp=0.5,webqa=0.5",
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    first_shard = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))
    second_shard = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=1, data_buffer=None))

    assert first_shard.samples == [groups[0]]
    assert second_shard.samples == [groups[1]]
    assert worker.groups == []


def test_debug_rollout_shard_preserves_all_trajectories_and_metadata(tmp_path):
    samples = []
    for group_index in range(2):
        for trajectory_index in range(32):
            accepted = trajectory_index % 2 == 0
            sample = Sample(
                group_index=group_index,
                index=group_index * 32 + trajectory_index,
                rollout_id=7,
                prompt=f"task-{group_index}",
                response=f"trajectory-{trajectory_index}",
                reward=1.0 if accepted else 0.0,
                response_length=1,
                loss_mask=[1],
                status=Sample.Status.COMPLETED if trajectory_index != 31 else Sample.Status.FAILED,
                session_id=f"session-{group_index}-{trajectory_index}",
                metadata={
                    "accepted": accepted,
                    "rejection_reason": None if accepted else "incorrect_answer",
                    "correct": accepted,
                    "nested": {"trajectory_index": trajectory_index, "tool_calls": ["search", "submit"]},
                    "rllm_episode": {"id": f"trajectory-{group_index}-{trajectory_index}"},
                },
            )
            samples.append(sample)

    rollout_manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = rollout_manager_class.__new__(rollout_manager_class)
    manager.args = SimpleNamespace(
        save_debug_rollout_data=str(tmp_path / "rollout_data" / "{rollout_id}.pt"),
        n_samples_per_prompt=32,
    )
    manager._pending_rllm_episode_samples = {}

    manager._save_debug_rollout_data(samples, rollout_id=7, evaluation=False)

    path = tmp_path / "rollout_data" / "7.pt"
    dumped = torch.load(path, weights_only=False)
    assert dumped["rollout_id"] == 7
    assert dumped["num_task_groups"] == 2
    assert dumped["samples_per_task_group"] == 32
    assert dumped["num_samples"] == 64
    assert [group["sample_positions"] for group in dumped["task_groups"]] == [list(range(32)), list(range(32, 64))]
    assert dumped["samples"] == [sample.to_dict() for sample in samples]
    assert not path.with_suffix(".pt.tmp").exists()


class FakeWorker:
    def __init__(self, groups):
        self.groups = list(enumerate(groups))
        self.resumed = False
        self.paused = False
        self.resume_calls = []

    def get_completed_groups(self, limit=None):
        split = len(self.groups) if limit is None else limit
        groups = self.groups[:split]
        self.groups = self.groups[split:]
        return groups

    def get_completed_groups_for_range(self, start, end):
        selected = [(gid, group) for gid, group in self.groups if start <= gid < end]
        selected_ids = {gid for gid, _ in selected}
        self.groups = [(gid, group) for gid, group in self.groups if gid not in selected_ids]
        return selected

    def requeue_completed_groups(self, groups):
        self.groups = list(groups) + self.groups

    def queue_size(self):
        return 0

    def exhausted_and_idle(self):
        return not self.groups

    def resume(self, work_limit=None, continuous=False):
        self.resumed = True
        self.work_limit = work_limit
        self.resume_calls.append((work_limit, continuous))

    def pause(self):
        self.paused = True


def test_fully_async_collects_valid_groups_and_reports_candidate_shortfall(monkeypatch):
    invalid = [[_sample(i * 2, 0.0), _sample(i * 2 + 1, 0.0)] for i in range(3)]
    valid = [[_sample(10, 0.0), _sample(11, 1.0)]]
    worker = FakeWorker([*invalid, *valid])
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_VALID_GROUPS_PER_SHARD", "2")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_MAX_CANDIDATE_GROUPS_PER_SHARD", "4")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_only_inference_fast_path=True,
        rollout_batch_size=64,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    output = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))

    assert output.samples == valid
    assert output.metrics["rollout/dynamic_filter/valid_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/valid_group_shortfall"] == 1
    assert output.metrics["rollout/dynamic_filter/drop_zero_reward_variance"] == 3


def test_valid_group_collection_requeues_batch_overflow(monkeypatch):
    first = [_sample(0, 0.0), _sample(1, 1.0)]
    second = [_sample(2, 0.0), _sample(3, 1.0)]
    worker = FakeWorker([first, second])
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_VALID_GROUPS_PER_SHARD", "1")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_MAX_CANDIDATE_GROUPS_PER_SHARD", "2")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_only_inference_fast_path=True,
        rollout_batch_size=64,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    first_output = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=0, data_buffer=None))
    second_output = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=1, data_buffer=None))

    assert first_output.samples == [first]
    assert second_output.samples == [second]


def test_generate_group_records_group_profile(monkeypatch):
    sample = _sample(0, 1.0)

    async def generate_one(_args, item, _sampling_params, evaluation=False):
        return item

    monkeypatch.setattr("slime.rollout.sglang_rollout.generate_and_rm", generate_one)
    args = SimpleNamespace(
        sglang_server_concurrency=1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_min_p=0.0,
        rollout_presence_penalty=0.0,
        rollout_repetition_penalty=1.0,
        rollout_max_response_len=8,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        group_rm=False,
    )
    monkeypatch.setattr("slime.rollout.sglang_rollout.GenerateState", lambda _args: SimpleNamespace(aborted=False))

    result = asyncio.run(generate_and_rm_group(args, [sample], {}, evaluation=False))

    profile = result[0].metadata["fused_profile"]
    assert profile["group_end_time_s"] >= profile["group_start_time_s"]
    assert profile["group_total_time_s"] >= 0


def test_trajectory_profile_json_keeps_rejected_candidates(tmp_path, monkeypatch):
    profile_dir = tmp_path / "profiles"
    monkeypatch.setenv("SLIME_FUSED_PROFILE_DIR", str(profile_dir))
    sample = _sample(7, 0.0, termination="ABNORMAL_PARSE_ERROR", credit_event="tool_parser_error")
    sample.session_id = "trajectory-7"
    sample.metadata["fused_profile"] = {"llm_decode_time_s": 1.5}

    fully_async._write_trajectory_profile_shard(
        SimpleNamespace(reward_key=None), 3, 11, [sample], selected_for_shard=False
    )

    path = profile_dir / "rollout_000003_group_000000011.json"
    payload = json.loads(path.read_text())
    record = payload["trajectories"][0]
    assert payload["selected_for_shard"] is False
    assert record["parser_error"] is True
    assert record["termination_reason"] == "ABNORMAL_PARSE_ERROR"
    assert record["profile"]["llm_decode_time_s"] == 1.5
    assert "\n    \"trajectories\": [\n" in path.read_text()


class SizedDataBuffer:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_cross_shard_prefetch_keeps_next_shard_results_separate(monkeypatch):
    groups = [[_sample(index, 0.0)] for index in range(6)]
    worker = FakeWorker(groups)
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_CROSS_SHARD_PREFETCH", "true")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_only_inference_fast_path=True,
        rollout_batch_size=2,
        start_rollout_id=5,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    first = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=5, data_buffer=SizedDataBuffer(16)))
    second = asyncio.run(fully_async._generate_rollout_async(args, rollout_id=6, data_buffer=SizedDataBuffer(16)))

    assert [group[0].index for group in first.samples] == [0, 1]
    assert [group[0].index for group in second.samples] == [2, 3]
    assert worker.resume_calls == [(4, True), (6, True)]
    assert worker.paused is False


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


def test_fully_async_inference_fast_path_does_not_wrap_final_dataset_batch(monkeypatch):
    groups = [[_sample(index, 0.0)] for index in range(64)]
    worker = FakeWorker(groups)
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_NO_DATASET_WRAP", "true")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_only_inference_fast_path=True,
        rollout_batch_size=64,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    output = asyncio.run(
        fully_async._generate_rollout_async(args, rollout_id=341, data_buffer=SizedDataBuffer(21874))
    )

    assert len(output.samples) == 50
    assert worker.work_limit == 50
    assert len(worker.groups) == 14


def test_fully_async_training_path_keeps_fixed_final_dataset_batch(monkeypatch):
    groups = [[_sample(index, 0.0)] for index in range(64)]
    worker = FakeWorker(groups)
    monkeypatch.setattr(fully_async, "_get_global_worker", lambda args, data_buffer: worker)
    monkeypatch.setenv("SLIME_FULLY_ASYNC_KEEP_ALL_GROUPS", "true")
    monkeypatch.setenv("SLIME_FULLY_ASYNC_NO_DATASET_WRAP", "true")
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_only_inference_fast_path=False,
        rollout_batch_size=64,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )

    output = asyncio.run(
        fully_async._generate_rollout_async(args, rollout_id=341, data_buffer=SizedDataBuffer(21874))
    )

    assert len(output.samples) == 64
    assert worker.work_limit is None


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
    assert output.metrics["rollout/dynamic_filter/valid_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/roi"] == 0.5
    assert output.metrics["rollout/dynamic_filter/dropped_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/kept_groups"] == 1
    assert output.metrics["rollout/dynamic_filter/drop_zero_std_0.0"] == 1
    assert worker.resumed is True
    assert worker.paused is True


def test_fully_async_dynamic_filter_excludes_infra_failure_from_zero_reward(monkeypatch):
    failed = _sample(0, 0.0)
    failed.status = Sample.Status.FAILED
    failed.metadata.update({"fused_error": "rollout_group_timeout"})
    groups = [
        [failed, _sample(1, 0.0)],
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

    assert output.samples == [groups[1]]
    assert output.metrics["rollout/dynamic_filter/drop_infra_failure"] == 1


def test_verifier_exception_is_classified_as_infra_failure():
    sample = _sample(0, 0.0)
    sample.status = Sample.Status.COMPLETED
    sample.metadata["fused_reward_debug"] = {"reward": 0.0, "verifier_error": "RuntimeError: unavailable"}

    assert is_infra_failure(sample) is True


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


def test_task_family_quota_selection_splits_eight_groups_evenly():
    groups = [
        *[[_sample(index, 1.0, "mcp")] for index in range(6)],
        *[[_sample(index, 1.0, "web_search")] for index in range(6, 12)],
    ]
    args = SimpleNamespace(rollout_task_family_quotas="mcp=0.5,webqa=0.5")

    selected = fully_async._select_task_family_quota_groups(groups, target=8, args=args)

    families = [fully_async._sample_group_task_family(group) for group in selected]
    assert families.count("mcp") == 4
    assert families.count("webqa") == 4


def test_task_family_quota_selection_fills_missing_family_from_remaining():
    groups = [
        [_sample(0, 1.0, "mcp")],
        [_sample(1, 1.0, "mcp")],
        [_sample(2, 1.0, "web_search")],
    ]
    args = SimpleNamespace(rollout_task_family_quotas="webqa=0.5,cli=0.5")

    selected = fully_async._select_task_family_quota_groups(groups, target=3, args=args)

    assert selected == [groups[2], groups[0], groups[1]]
