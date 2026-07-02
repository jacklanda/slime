import sys
import types

import pytest


NUM_GPUS = 0


def _install_megatron_stubs(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    distributed_mod = types.ModuleType("megatron.core.distributed")
    enums_mod = types.ModuleType("megatron.core.enums")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    optimizer_mod = types.ModuleType("megatron.core.optimizer")
    optimizer_inner_mod = types.ModuleType("megatron.core.optimizer.optimizer")
    scheduler_mod = types.ModuleType("megatron.core.optimizer_param_scheduler")
    pp_mod = types.ModuleType("megatron.core.pipeline_parallel")
    utils_mod = types.ModuleType("megatron.core.utils")
    global_vars_mod = types.ModuleType("megatron.training.global_vars")
    training_mod = types.ModuleType("megatron.training.training")
    training_pkg = types.ModuleType("megatron.training")
    checkpointing_mod = types.ModuleType("megatron.training.checkpointing")

    core_mod.mpu = types.SimpleNamespace()
    distributed_mod.DistributedDataParallel = object
    distributed_mod.finalize_model_grads = lambda *args, **kwargs: None
    enums_mod.ModelType = types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")
    gpt_mod.GPTModel = object
    optimizer_mod.OptimizerConfig = object
    optimizer_mod.get_megatron_optimizer = lambda *args, **kwargs: None
    optimizer_inner_mod.MegatronOptimizer = object
    scheduler_mod.OptimizerParamScheduler = object
    pp_mod.get_forward_backward_func = lambda *args, **kwargs: None
    utils_mod.get_model_config = lambda *args, **kwargs: None
    utils_mod.unwrap_model = lambda model: model
    global_vars_mod.get_args = lambda: types.SimpleNamespace(rollout_max_response_len=None)
    training_mod.get_model = lambda *args, **kwargs: None
    checkpointing_mod.load_checkpoint = lambda *args, **kwargs: None
    checkpointing_mod.save_checkpoint = lambda *args, **kwargs: None

    modules = {
        "megatron": megatron_mod,
        "megatron.core": core_mod,
        "megatron.core.distributed": distributed_mod,
        "megatron.core.enums": enums_mod,
        "megatron.core.models.gpt": gpt_mod,
        "megatron.core.optimizer": optimizer_mod,
        "megatron.core.optimizer.optimizer": optimizer_inner_mod,
        "megatron.core.optimizer_param_scheduler": scheduler_mod,
        "megatron.core.pipeline_parallel": pp_mod,
        "megatron.core.utils": utils_mod,
        "megatron.training": training_pkg,
        "megatron.training.checkpointing": checkpointing_mod,
        "megatron.training.global_vars": global_vars_mod,
        "megatron.training.training": training_mod,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    checkpoint_stub = types.ModuleType("slime.backends.megatron_utils.checkpoint")
    checkpoint_stub.load_checkpoint = lambda *args, **kwargs: None
    checkpoint_stub.save_checkpoint = lambda *args, **kwargs: None
    cp_utils_stub = types.ModuleType("slime.backends.megatron_utils.cp_utils")
    cp_utils_stub.reduce_train_step_metrics = lambda loss_dict: loss_dict
    data_stub = types.ModuleType("slime.backends.megatron_utils.data")
    data_stub.DataIterator = object
    data_stub.get_batch = lambda *args, **kwargs: None
    loss_stub = types.ModuleType("slime.backends.megatron_utils.loss")
    loss_stub.ROLLOUT_TOP_P_TOKEN_KEYS = ()
    loss_stub.get_rollout_top_p_logprob_kwargs = lambda *args, **kwargs: {}
    loss_stub.loss_function = lambda *args, **kwargs: None
    provider_stub = types.ModuleType("slime.backends.megatron_utils.model_provider")
    provider_stub.get_model_provider_func = lambda *args, **kwargs: None
    for name, module in {
        "slime.backends.megatron_utils.checkpoint": checkpoint_stub,
        "slime.backends.megatron_utils.cp_utils": cp_utils_stub,
        "slime.backends.megatron_utils.data": data_stub,
        "slime.backends.megatron_utils.loss": loss_stub,
        "slime.backends.megatron_utils.model_provider": provider_stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def _episode_metrics_for_actor_update(monkeypatch):
    sys.modules.pop("slime.backends.megatron_utils.model", None)
    _install_megatron_stubs(monkeypatch)
    from slime.backends.megatron_utils.model import _episode_metrics_for_actor_update

    return _episode_metrics_for_actor_update


def test_actor_update_episode_metrics_logs_zero_when_no_valid_episode(monkeypatch):
    metrics_fn = _episode_metrics_for_actor_update(monkeypatch)

    metrics = metrics_fn(
        {
            "episode_metrics_data": {
                "raw_rewards": [1.0],
                "metadata": [{"fused_task_type": "webqa"}],
                "remove_sample": [True],
                "loss_mask_sums": [3],
                "response_lengths": [4],
                "statuses": ["completed"],
            }
        }
    )

    assert metrics["episode/num"] == 0.0
    assert metrics["episode/training_reward/mean"] == 0.0
    assert "episode/reward/mean" not in metrics
    assert "episode/pass@1" not in metrics
    assert "episode/correct" not in metrics


def test_actor_update_episode_metrics_only_uses_actor_update_valid_episodes(monkeypatch):
    metrics_fn = _episode_metrics_for_actor_update(monkeypatch)

    metrics = metrics_fn(
        {
            "episode_metrics_data": {
                "raw_rewards": [1.0, 0.0, 1.0],
                "metadata": [
                    {"fused_task_type": "webqa", "fused_traj_steps": 2},
                    {"fused_task_type": "webqa", "fused_traj_steps": 9},
                    {"fused_task_type": "cli", "fused_traj_steps": 4},
                ],
                "group_indices": [10, 20, 30],
                "remove_sample": [False, False, True],
                "loss_mask_sums": [3, 0, 3],
                "prompt_lengths": [2, 2, 2],
                "response_lengths": [4, 5, 6],
                "statuses": ["completed", "completed", "completed"],
            }
        }
    )

    assert metrics["episode/num"] == 1.0
    assert metrics["episode/training_reward/mean"] == 1.0
    assert metrics["episode/reward/webqa/mean"] == 1.0
    assert "episode/pass@1" not in metrics
    assert metrics["episode/traj/steps"] == 2.0


def test_actor_update_episode_metrics_logs_fused_abnormal_termination_details(monkeypatch):
    metrics_fn = _episode_metrics_for_actor_update(monkeypatch)

    metrics = metrics_fn(
        {
            "episode_metrics_data": {
                "raw_rewards": [0.0, 0.0, 0.0],
                "metadata": [
                    {"fused_task_type": "webqa", "fused_termination": "ABNORMAL_PARSE_ERROR"},
                    {"fused_task_type": "webqa", "fused_termination": "ABNORMAL_TOOL_BURST"},
                    {"fused_task_type": "webqa", "fused_termination": "ABNORMAL_REPEATED_QUERY"},
                ],
                "group_indices": [1, 2, 3],
                "remove_sample": [False, False, False],
                "loss_mask_sums": [3, 3, 3],
                "prompt_lengths": [2, 2, 2],
                "response_lengths": [4, 4, 4],
                "statuses": ["completed", "completed", "completed"],
            }
        }
    )

    assert metrics["episode/termination_reason/abnormal_parse_error"] == pytest.approx(1 / 3)
    assert metrics["episode/termination_reason/abnormal_tool_burst"] == pytest.approx(1 / 3)
    assert metrics["episode/termination_reason/abnormal_repeated_query"] == pytest.approx(1 / 3)
    assert metrics["episode/termination_warning/abnormal_or_limit"] == 1.0
    assert metrics["episode/termination_reason/error"] == 0.0
    assert metrics["episode/termination_reason/unknown"] == 0.0


def test_actor_update_episode_metrics_logs_context_and_response_limit_terminations(monkeypatch):
    metrics_fn = _episode_metrics_for_actor_update(monkeypatch)

    metrics = metrics_fn(
        {
            "episode_metrics_data": {
                "raw_rewards": [0.0, 0.0],
                "metadata": [
                    {"fused_task_type": "webqa", "fused_termination": "max_context_len_exceeded"},
                    {"fused_task_type": "webqa", "fused_termination": "max_response_len_exceeded"},
                ],
                "group_indices": [1, 2],
                "remove_sample": [False, False],
                "loss_mask_sums": [3, 3],
                "prompt_lengths": [2, 2],
                "response_lengths": [4, 4],
                "statuses": ["completed", "completed"],
            }
        }
    )

    assert metrics["episode/termination_reason/max_context_len_exceeded"] == 0.5
    assert metrics["episode/termination_reason/max_response_len_exceeded"] == 0.5
    assert metrics["episode/termination_warning/abnormal_or_limit"] == 1.0


def test_actor_update_episode_metrics_uses_best_sample_termination_per_group(monkeypatch):
    metrics_fn = _episode_metrics_for_actor_update(monkeypatch)

    metrics = metrics_fn(
        {
            "episode_metrics_data": {
                "raw_rewards": [0.0, 1.0],
                "metadata": [
                    {"fused_task_type": "webqa", "fused_termination": "ABNORMAL_PARSE_ERROR"},
                    {"fused_task_type": "webqa", "fused_termination": "env_done"},
                ],
                "group_indices": [7, 7],
                "remove_sample": [False, False],
                "loss_mask_sums": [3, 3],
                "prompt_lengths": [2, 2],
                "response_lengths": [4, 4],
                "statuses": ["completed", "completed"],
            }
        }
    )

    assert metrics["episode/reward/webqa/mean"] == 1.0
    assert "episode/pass@1" not in metrics
    assert metrics["episode/termination_reason/env_done"] == 1.0
    assert metrics["episode/termination_reason/error"] == 0.0


def test_actor_update_episode_metrics_aggregates_search_summary_metrics(monkeypatch):
    metrics_fn = _episode_metrics_for_actor_update(monkeypatch)

    metrics = metrics_fn(
        {
            "episode_metrics_data": {
                "raw_rewards": [1.0, 0.0],
                "metadata": [
                    {
                        "fused_task_type": "webqa",
                        "rllm_episode": {
                            "metrics": {
                                "tools/search_summary_failures": 0,
                                "tools/search_summary_retries": 1,
                                "tools/search_summary_fallbacks": 0,
                                "tools/search_summary_elapsed_s": 0.4,
                            }
                        },
                    },
                    {
                        "fused_task_type": "webqa",
                        "rllm_episode": {
                            "metrics": {
                                "tools/search_summary_failures": 1,
                                "tools/search_summary_retries": 2,
                                "tools/search_summary_fallbacks": 1,
                                "tools/search_summary_elapsed_s": 0.8,
                            }
                        },
                    },
                ],
                "group_indices": [1, 2],
                "remove_sample": [False, False],
                "loss_mask_sums": [3, 3],
                "prompt_lengths": [2, 2],
                "response_lengths": [4, 4],
                "statuses": ["completed", "completed"],
            }
        }
    )

    assert metrics["episode/tools/search_summary_failures"] == 0.5
    assert metrics["episode/tools/search_summary_retries"] == 1.5
    assert metrics["episode/tools/search_summary_fallbacks"] == 0.5
    assert metrics["episode/tools/search_summary_elapsed_s"] == pytest.approx(0.6)
