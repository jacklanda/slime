import importlib
from types import SimpleNamespace

import pytest
import torch

from slime.utils.train_infer_consistency import mismatch_bucket_contributions, validate_rollout_weight_versions


NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, callback):
        self._callback = callback

    def remote(self, *args, **kwargs):
        return self._callback(*args, **kwargs)


class _RolloutManager:
    def __init__(self, events):
        self.check_weights = _RemoteMethod(lambda action: events.append(action))
        self.generate = _RemoteMethod(lambda rollout_id: events.append(f"generate:{rollout_id}"))
        self.load = _RemoteMethod(lambda rollout_id: events.append(f"load:{rollout_id}"))
        self.save = _RemoteMethod(lambda rollout_id: events.append(f"save:{rollout_id}"))
        self.save_rllm_episodes = _RemoteMethod(lambda rollout_id: events.append(f"episodes:{rollout_id}"))
        self.eval = _RemoteMethod(lambda rollout_id: events.append(f"eval:{rollout_id}"))
        self.dispose = _RemoteMethod(lambda: events.append("dispose"))


class _ActorModel:
    def __init__(self, events):
        self._events = events

    def update_weights(self):
        self._events.append("update_weights")

    def create(self):
        self._events.append("create")


@pytest.mark.parametrize("module_name", ["train", "train_async"])
def test_weight_equality_check_snapshots_loaded_actor_weights(monkeypatch, module_name):
    train_module = importlib.import_module(module_name)
    events = []
    rollout_manager = _RolloutManager(events)
    actor_model = _ActorModel(events)
    args = SimpleNamespace(
        check_weight_update_equal=True,
        colocate=False,
        eval_interval=None,
        num_rollout=0,
        offload_rollout=False,
        release_train=False,
        start_rollout_id=0,
    )

    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda args, pg: (rollout_manager, None),
    )
    monkeypatch.setattr(
        train_module,
        "create_training_models",
        lambda args, pgs, manager: (actor_model, None),
    )
    monkeypatch.setattr(train_module, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)

    train_module.train(args)

    assert events[:5] == ["update_weights", "snapshot", "reset_tensors", "update_weights", "compare"]


def test_release_train_recreates_actor_before_equality_recheck(monkeypatch):
    train_module = importlib.import_module("train")
    events = []
    rollout_manager = _RolloutManager(events)
    actor_model = _ActorModel(events)
    args = SimpleNamespace(
        check_weight_update_equal=True,
        eval_interval=None,
        num_rollout=0,
        offload_rollout=False,
        release_train=True,
        start_rollout_id=0,
    )

    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda args, pg: (rollout_manager, None),
    )
    monkeypatch.setattr(
        train_module,
        "create_training_models",
        lambda args, pgs, manager: (actor_model, None),
    )
    monkeypatch.setattr(train_module, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)

    train_module.train(args)

    assert events[:6] == ["update_weights", "snapshot", "reset_tensors", "create", "update_weights", "compare"]


def test_debug_rollout_only_never_creates_or_calls_training_models(monkeypatch):
    train_module = importlib.import_module("train")
    events = []
    rollout_manager = _RolloutManager(events)
    args = SimpleNamespace(
        debug_rollout_only=True,
        eval_interval=None,
        num_rollout=2,
        release_train=False,
        rollout_global_dataset=True,
        rollout_only_skip_episode_dump=True,
        start_rollout_id=0,
    )

    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda args, pg: (rollout_manager, None))
    monkeypatch.setattr(
        train_module,
        "create_training_models",
        lambda *args, **kwargs: pytest.fail("rollout-only mode must not create trainer actors"),
    )
    monkeypatch.setattr(train_module, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: events.append("finish_tracking"))
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)

    train_module.train(args)

    assert events == [
        "load:-1",
        "generate:0",
        "save:0",
        "generate:1",
        "save:1",
        "dispose",
        "finish_tracking",
    ]


def test_mismatch_bucket_contributions_are_additive_per_dimension():
    differences = torch.tensor([1.0, 3.0, 2.0])
    templates = [torch.empty(2), torch.empty(1)]
    bucket_ids = [(0, 0, 0), (1, 1, 2)]
    reducer = torch.sum

    metrics = mismatch_bucket_contributions(differences, templates, bucket_ids, reducer)

    assert metrics["train_rollout_logprob_abs_diff/webqa_contribution"].item() == 4.0
    assert metrics["train_rollout_logprob_abs_diff/mcp_contribution"].item() == 2.0
    assert metrics["train_rollout_logprob_abs_diff/single_segment_contribution"].item() == 4.0
    assert metrics["train_rollout_logprob_abs_diff/multi_segment_contribution"].item() == 2.0
    assert metrics["train_rollout_logprob_abs_diff/short_contribution"].item() == 4.0
    assert metrics["train_rollout_logprob_abs_diff/long_contribution"].item() == 2.0
    assert sum(metrics[f"train_rollout_logprob_abs_diff/{name}_contribution"] for name in ("webqa", "mcp", "cli", "other")).item() == pytest.approx(differences.sum().item())


def test_mismatch_bucket_contributions_reject_misaligned_sequences():
    with pytest.raises(ValueError, match="bucket count"):
        mismatch_bucket_contributions(torch.ones(1), [torch.empty(1)], [], torch.sum)


def test_validate_rollout_weight_versions_checks_every_reporting_engine():
    validate_rollout_weight_versions(9, ["9", None, 9])

    with pytest.raises(RuntimeError, match="engines=.*v8"):
        validate_rollout_weight_versions("v9", ["v9", None, "v8"])

    with pytest.raises(RuntimeError, match="No SGLang engine"):
        validate_rollout_weight_versions("v9", [None, None])
