import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model
from tqdm import tqdm

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model
from slime.utils import logging_utils
from slime.utils.memory_utils import clear_memory
from slime.utils.metric_utils import format_metrics_for_display
from slime.utils.visualization import print_metrics_table

from .checkpoint import load_checkpoint, save_checkpoint
from .cp_utils import reduce_train_step_metrics
from .data import DataIterator, get_batch
from .loss import (
    ROLLOUT_TOP_P_TOKEN_KEYS,
    get_log_probs_and_entropy,
    get_rollout_top_p_logprob_kwargs,
    loss_function,
)
from .model_provider import get_model_provider_func
from .stateless_adam import StatelessAdam

logger = logging.getLogger(__name__)


def _disable_tqdm_for_non_main_rank() -> bool:
    return not (mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1)


def _should_update_microbatch_pbar(model) -> bool:
    if _disable_tqdm_for_non_main_rank():
        return False

    while hasattr(model, "module"):
        model = model.module
    vp_stage = getattr(model, "vp_stage", None)
    if mpu.get_virtual_pipeline_model_parallel_world_size() is not None and vp_stage is not None:
        return mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    return mpu.is_pipeline_last_stage(ignore_virtual=True)


def _wrap_forward_step_with_microbatch_pbar(forward_step_func, pbar):
    if pbar is None:
        return forward_step_func

    def wrapped_forward_step(*args, **kwargs):
        result = forward_step_func(*args, **kwargs)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if model is not None and _should_update_microbatch_pbar(model):
            pbar.update(1)
        return result

    return wrapped_forward_step


def _with_rollout_top_p_token_keys(args: Namespace, keys: Sequence[str]) -> list[str]:
    if args.rollout_top_p == 1.0:
        return list(keys)
    return [*keys, *ROLLOUT_TOP_P_TOKEN_KEYS]


def _episode_metrics_for_actor_update(rollout_data: dict | None) -> dict:
    if not rollout_data:
        return {}
    metrics_data = rollout_data.get("episode_metrics_data")
    if not metrics_data:
        return {}

    samples = []
    segmented_samples: dict[str, tuple[int, dict]] = {}
    raw_rewards = rollout_data.get("raw_reward", metrics_data.get("raw_rewards", []))
    metadata_list = metrics_data.get("metadata", [])
    group_indices = metrics_data.get("group_indices", [])
    rollout_ids = metrics_data.get("rollout_ids", [])
    sample_indices = metrics_data.get("sample_indices", [])
    remove_sample = metrics_data.get("remove_sample", [])
    loss_mask_sums = metrics_data.get("loss_mask_sums", [])
    prompt_lengths = metrics_data.get("prompt_lengths", [])
    response_lengths = metrics_data.get("response_lengths", [])
    num_samples = len(raw_rewards)

    valid_indices = []
    for i in range(num_samples):
        if i < len(remove_sample) and remove_sample[i]:
            continue
        if i < len(loss_mask_sums) and loss_mask_sums[i] == 0:
            continue
        valid_indices.append(i)
        metadata = metadata_list[i] if i < len(metadata_list) and isinstance(metadata_list[i], dict) else {}
        if "fused_task_type" not in metadata:
            continue
        parent_traj_id = metadata.get("parent_traj_id")
        if parent_traj_id is not None:
            group_id = str(parent_traj_id)
        else:
            group_id = group_indices[i] if i < len(group_indices) else None
            if group_id is None:
                group_id = rollout_ids[i] if i < len(rollout_ids) else None
            if group_id is None:
                group_id = sample_indices[i] if i < len(sample_indices) else i
        sample = {
            "group_id": group_id,
            "reward": float(raw_rewards[i]),
            "metadata": metadata,
        }
        if parent_traj_id is None:
            samples.append(sample)
            continue
        segment_index = int(metadata.get("segment_index", 0) or 0)
        previous = segmented_samples.get(group_id)
        if previous is None or segment_index >= previous[0]:
            segmented_samples[group_id] = (segment_index, sample)

    samples.extend(sample for _, sample in segmented_samples.values())

    # response_length/* and prompt_length/* are logged
    # on the rollout side (rollout.py::compute_metrics_from_samples) to avoid
    # duplicating the same distribution under two prefixes.
    metrics: dict[str, float] = {}
    metrics.update(
        {
            "episode/num": 0.0,
            "episode/reward": 0.0,
            "episode/training_reward/mean": 0.0,
        }
    )
    if not valid_indices:
        return metrics

    group_rewards: dict[object, list[float]] = {}
    group_task_types: dict[object, str] = {}
    group_terminations: dict[object, str] = {}
    group_termination_rewards: dict[object, float] = {}
    workflow_values: dict[str, list[float]] = {}
    group_steps: dict[object, float] = {}
    group_tool_call_turns: dict[object, float] = {}
    group_prompt_tokens: dict[object, int] = {}
    group_response_tokens: dict[object, int] = {}

    for sample in samples:
        metadata = sample["metadata"]
        group_id = sample["group_id"]
        reward = sample["reward"]
        group_rewards.setdefault(group_id, []).append(reward)
        task_type = str(metadata.get("fused_task_type") or metadata.get("task_type") or metadata.get("data_source") or "unknown")
        group_task_types.setdefault(group_id, task_type)

        if group_id not in group_steps:
            steps = _coerce_finite_float(metadata.get("fused_traj_steps") or metadata.get("traj_steps"))
            if steps is not None:
                group_steps[group_id] = steps
        if group_id not in group_tool_call_turns:
            tool_call_turns = _coerce_finite_float(metadata.get("fused_tool_call_turns") or metadata.get("tool_call_turns") or metadata.get("tool_call_turn"))
            if tool_call_turns is not None:
                group_tool_call_turns[group_id] = tool_call_turns

        termination = metadata.get("fused_termination") or metadata.get("termination_reason")
        if termination:
            termination = str(termination)
            previous_reward = group_termination_rewards.get(group_id)
            if previous_reward is None or reward > previous_reward or (reward == previous_reward and _termination_priority(termination) > _termination_priority(group_terminations[group_id])):
                group_terminations[group_id] = termination
                group_termination_rewards[group_id] = reward

        rllm_episode = metadata.get("rllm_episode")
        rllm_metrics = rllm_episode.get("metrics") if isinstance(rllm_episode, dict) else None
        if isinstance(rllm_metrics, dict):
            for key, value in rllm_metrics.items():
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    workflow_values.setdefault(str(key), []).append(float(value))

        reward_debug = metadata.get("fused_reward_debug")
        if isinstance(reward_debug, dict):
            for key, value in reward_debug.items():
                if isinstance(value, bool):
                    value = int(value)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    workflow_values.setdefault(key, []).append(float(value))

    direct_tool_metric_groups = set()
    for i in valid_indices:
        metadata = metadata_list[i] if i < len(metadata_list) and isinstance(metadata_list[i], dict) else {}
        parent_traj_id = metadata.get("parent_traj_id")
        if parent_traj_id is not None:
            group_id = str(parent_traj_id)
        else:
            group_id = group_indices[i] if i < len(group_indices) else None
            if group_id is None:
                group_id = rollout_ids[i] if i < len(rollout_ids) else None
            if group_id is None:
                group_id = sample_indices[i] if i < len(sample_indices) else i
        group_prompt_tokens[group_id] = group_prompt_tokens.get(group_id, 0) + int(prompt_lengths[i] if i < len(prompt_lengths) else 0)
        group_response_tokens[group_id] = group_response_tokens.get(group_id, 0) + int(response_lengths[i] if i < len(response_lengths) else 0)
        if group_id in direct_tool_metric_groups:
            continue
        direct_tool_metric_groups.add(group_id)
        for key in _RLLM_EPISODE_TOOL_KEYS:
            value = metadata.get(key)
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)) and math.isfinite(value):
                workflow_values.setdefault(key, []).append(float(value))

    if not samples:
        return metrics

    episode_rewards_by_source: dict[str, list[float]] = {}
    episode_turn_values: dict[str, list[float]] = {}
    prompt_token_sum = 0
    response_token_sum = 0
    sample_reward_sum = 0.0
    sample_count = 0
    for group_id, rewards in group_rewards.items():
        task_type = group_task_types[group_id]
        suffix = _metric_task_suffix(task_type)
        reward = max(rewards)
        episode_rewards_by_source.setdefault(task_type, []).append(reward)
        prompt_token_sum += group_prompt_tokens.get(group_id, 0)
        response_token_sum += group_response_tokens.get(group_id, 0)
        sample_reward_sum += sum(rewards)
        sample_count += len(rewards)
        if group_id in group_steps:
            steps = group_steps[group_id]
            episode_turn_values.setdefault("traj/steps", []).append(steps)
            episode_turn_values.setdefault(f"traj/steps/{suffix}", []).append(steps)
        if group_id in group_tool_call_turns:
            tool_call_turns = group_tool_call_turns[group_id]
            episode_turn_values.setdefault("turn/tool_call_turn", []).append(tool_call_turns)
            episode_turn_values.setdefault(f"turn/tool_call_turn/{suffix}", []).append(tool_call_turns)

    num_groups = len(group_rewards)
    training_reward_mean = float(sample_reward_sum / sample_count)
    metrics.update(
        {
            "episode/num": float(num_groups),
            # Keep the public episode reward on the same all-trajectory
            # denominator as the reward used by the actor update. Workflow
            # metrics are absent on abnormal terminations and therefore cannot
            # define this aggregate without silently dropping failed episodes.
            "episode/reward": training_reward_mean,
            # Sample-equal-weight mean: average over every individual sample's
            # reward (flattened across groups), NOT the per-group max. The
            # per-group-max variant saturates to ~1.0 under large
            # n_samples_per_prompt because it effectively measures pass@n.
            "episode/training_reward/mean": training_reward_mean,
            "episode/prompt_tokens": float(prompt_token_sum / num_groups),
            "episode/response_tokens": float(response_token_sum / num_groups),
        }
    )
    for source, rewards in episode_rewards_by_source.items():
        metrics[f"episode/reward/{source}/mean"] = float(sum(rewards) / len(rewards))
    termination_counts = {reason: 0 for reason in _RLLM_TERMINATION_REASONS}
    for group_id in group_rewards:
        termination_counts[_normalize_termination_reason(group_terminations.get(group_id, "unknown"))] += 1
    total_terminations = sum(termination_counts.values())
    for termination, count in termination_counts.items():
        metrics[f"episode/termination_reason/{termination}"] = count / total_terminations
    termination_warning_count = sum(count for reason, count in termination_counts.items() if _is_warning_termination(reason))
    metrics["episode/termination_warning/abnormal_or_limit"] = termination_warning_count / total_terminations
    completed_rewards = workflow_values.pop("reward", [])
    if completed_rewards:
        metrics["episode/reward/completed_mean"] = float(sum(completed_rewards) / len(completed_rewards))
    for key, values in workflow_values.items():
        mean_value = float(sum(values) / len(values))
        metrics[f"episode/{key}"] = mean_value
    for key, values in episode_turn_values.items():
        metrics[f"episode/{key}"] = float(sum(values) / len(values))
    return metrics


_RLLM_EPISODE_TOOL_KEYS = (
    "tools/search_summary_failures",
    "tools/search_summary_retries",
    "tools/search_summary_fallbacks",
    "tools/search_summary_elapsed_s",
    "tools/search_retrieve_elapsed_s",
    "tools/search_retrieve_retries",
    "tools/search_retrieve_failures",
    "tools/search_lexrank_summary",
    "tools/mcp_tool_elapsed_s",
    "tools/mcp_failures_per_step",
    "tools/mcp_failures",
)

_RLLM_TERMINATION_REASONS = (
    "unknown",
    "timeout",
    "max_turns_exceeded",
    "repeated_query_early_stop",
    "max_response_len_exceeded",
    "max_context_len_exceeded",
    "abnormal_parse_error",
    "invalid_react_structure",
    "invalid_final_step",
    "abnormal_tool_burst",
    "abnormal_direct_submit_without_tool",
    "abnormal_repeated_query",
    "abnormal_ngram_repetition",
    "abnormal_search_bypass",
    "abnormal_mixed_tool_and_answer",
    "tail_guard_early_stop",
    "env_init_error",
    "error",
    "env_done",
)


def _normalize_termination_reason(reason: str) -> str:
    normalized = reason.lower().replace("-", "_")
    aliases = {
        "max_response_length_exceeded": "max_response_len_exceeded",
        "step_token_budget_exhausted": "max_response_len_exceeded",
        "truncation": "max_response_len_exceeded",
        "max_prompt_length_exceeded": "max_context_len_exceeded",
        "prompt_truncation": "max_context_len_exceeded",
        "env_timeout": "timeout",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _RLLM_TERMINATION_REASONS:
        return "unknown"
    return normalized


def _termination_priority(reason: str) -> int:
    order = {
        "env_done": 5,
        "max_response_len_exceeded": 4,
        "max_context_len_exceeded": 3,
        "max_turns_exceeded": 2,
        "timeout": 1,
        "error": 0,
        "unknown": -1,
    }
    normalized = _normalize_termination_reason(reason)
    if normalized in order:
        return order[normalized]
    if _is_warning_termination(normalized):
        return 0
    return order["unknown"]


def _is_warning_termination(reason: str) -> bool:
    normalized = _normalize_termination_reason(reason)
    return normalized.startswith("abnormal_") or normalized in {
        "invalid_react_structure",
        "invalid_final_step",
        "tail_guard_early_stop",
        "env_init_error",
        "error",
        "max_context_len_exceeded",
        "max_response_len_exceeded",
    }


def _coerce_finite_float(value) -> float | None:
    if isinstance(value, bool):
        value = int(value)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _metric_task_suffix(task_type: str) -> str:
    normalized = task_type.lower().replace(" ", "_").replace("-", "_")
    if normalized in {"web_search", "search", "webqa"}:
        return "webqa"
    if normalized in {"mcp", "cli"}:
        return normalized
    if normalized in {"et", "endless_terminal", "endless_terminals", "swe"}:
        return "cli"
    return normalized or "unknown"


def _iter_critic_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


try:
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args
except ImportError:

    def get_load_checkpoint_path_by_args(args, load_arg="load"):
        from megatron.training.checkpointing import (
            get_checkpoint_name,
            get_checkpoint_tracker_filename,
            isfile,
            read_metadata,
        )

        """Get the checkpoint path based on the arguments."""
        load_dir = getattr(args, load_arg)
        iteration, release = -1, False
        tracker_filename = "because load directory is not defined"
        if load_dir is not None:
            tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if isfile(tracker_filename):
                iteration, release = read_metadata(tracker_filename)
            else:
                load_dir, checkpoint_step = os.path.split(load_dir)
                if checkpoint_step == "release" or checkpoint_step.startswith("iter_"):
                    release = checkpoint_step == "release"
                    if not release:
                        iteration = int(checkpoint_step.split("_")[1])

        # Allow user to specify the loaded iteration.
        if getattr(args, "ckpt_step", None):
            iteration = args.ckpt_step

        return get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (tensor_metadata for key, tensor_metadata in checkpoint_metadata.items() if key == param_name or key.endswith(f".{param_name}")),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = "missing from checkpoint metadata" if checkpoint_shape is None else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(args: Namespace, model: Sequence[DDP]) -> None:
    init_method_std = getattr(args, "init_method_std", None)
    if init_method_std is None:
        init_method_std = 0.02
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        output_layer.weight.data.normal_(mean=0.0, std=init_method_std)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training. ``train_iters`` is an estimate of the total
    # number of training steps — it's only used to size Megatron's LR decay
    # schedule (and ``lr_decay_iters`` defaults to it). With variable per-rollout
    # sample counts (dynamic sampling / filtering / custom step splitter) the
    # *actual* total can drift; the schedule still tracks the true progress via
    # ``opt_param_scheduler.num_steps`` (samples consumed, also persisted across
    # resume), so the worst case is the cosine/linear schedule reaches its
    # plateau slightly early or late. Pass ``--lr-decay-iters`` explicitly if you
    # need exact decay control.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def _noop_init_state_fn(*args, **kwargs) -> None:
    return None


def _disable_distributed_optimizer_state_initialization(optimizer: MegatronOptimizer) -> None:
    for megatron_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        if megatron_optimizer.__class__.__name__ == "DistributedOptimizer":
            megatron_optimizer.init_state_fn = _noop_init_state_fn


@contextmanager
def _patch_megatron_adam(adam_cls):
    import megatron.core.optimizer as megatron_optimizer
    import megatron.core.optimizer.distrib_optimizer as megatron_distrib_optimizer

    missing = object()
    old_adam = megatron_optimizer.Adam
    old_cpu_adam = getattr(megatron_optimizer, "CPUAdam", missing)
    old_distrib_adam = megatron_distrib_optimizer.Adam
    try:
        megatron_optimizer.Adam = adam_cls
        if old_cpu_adam is not missing:
            megatron_optimizer.CPUAdam = adam_cls
        megatron_distrib_optimizer.Adam = adam_cls
        yield
    finally:
        megatron_optimizer.Adam = old_adam
        if old_cpu_adam is not missing:
            megatron_optimizer.CPUAdam = old_cpu_adam
        megatron_distrib_optimizer.Adam = old_distrib_adam


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    if args.use_stateless_adam:
        assert config.optimizer == "adam", "Stateless Adam only supports --optimizer adam."
        assert args.no_save_optim, "Stateless Adam does not save Adam moment states. Please set --no-save-optim."

    optimizer_context = _patch_megatron_adam(StatelessAdam) if args.use_stateless_adam else nullcontext()
    with optimizer_context:
        optimizer = get_megatron_optimizer(
            config=config,
            model_chunks=model,
            use_gloo_process_groups=args.enable_gloo_process_groups,
        )
    if args.use_stateless_adam:
        _disable_distributed_optimizer_state_initialization(optimizer)
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
    use_rollout_top_p_replay: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.
        use_rollout_top_p_replay (bool): Whether to pass rollout top-p token sets
            to the post-forward log-prob callback when top-p rollout is enabled.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])
    batch_keys = [
        "tokens",
        "loss_masks",
        "multimodal_train_inputs",
        "total_lengths",
        "response_lengths",
    ]
    if use_rollout_top_p_replay:
        batch_keys = _with_rollout_top_p_token_keys(args, batch_keys)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            batch_keys,
            args.data_pad_size_multiplier,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
            # Log-prob kernels upcast one bounded chunk at a time. Upcasting the
            # full [tokens, vocab/TP] output here can require tens of GiB.
            "fp32_output": not ((args.fp16 or args.bf16) and f is get_log_probs_and_entropy),
        }
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])
        output_tensor = model(**forward_kwargs)

        output_kwargs = {
            "args": args,
            "unconcat_tokens": unconcat_tokens,
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "with_entropy": args.use_rollout_entropy,
        }
        if use_rollout_top_p_replay:
            output_kwargs.update(get_rollout_top_p_logprob_kwargs(args, batch))

        return output_tensor, partial(f, **output_kwargs)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{(store_prefix or getattr(model[0], 'role', 'actor')).rstrip('_')} forward",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )
    forward_step_with_progress = _wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step_with_progress,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
        )
    microbatch_pbar.close()

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
    microbatch_pbar=None,
) -> tuple[dict[str, float], float]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Rollout count for this training step
            (total across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts, which may emit >1 training
            sample under compact / subagent). Used both as the loss
            normalizer inside the closure and as the LR scheduler
            ``increment``. In the common case (1 rollout = 1 sample) this
            equals the per-step sample count, so behavior is unchanged.

    Returns:
        tuple[dict[str, float], float]: Reduced loss dictionary (last stage only)
        and gradient norm for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            _with_rollout_top_p_token_keys(
                args,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    "policy_loss_masks",
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "teacher_log_probs",
                    "rollout_mask_sums",
                    "policy_rollout_mask_sums",
                    "mismatch_bucket_ids",
                ],
            ),
            args.data_pad_size_multiplier,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            position_ids = None
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
                "fp32_output": not ((args.fp16 or args.bf16) and args.loss_type == "policy_loss"),
            }

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            output_tensor = model(**forward_kwargs)

        tiled_output_layer = None
        tiled_output_weight = None
        tiled_policy_loss_active = (
            os.environ.get("SLIME_TILED_POLICY_LOSS_ACTIVE") == "1"
            or os.environ.get("SLIME_GEMMA4_ACTOR_TRAIN_ACTIVE") == "1"
        )
        if tiled_policy_loss_active and mpu.is_pipeline_last_stage():
            inner_model = model
            while hasattr(inner_model, "module"):
                inner_model = inner_model.module
            tiled_output_layer = inner_model.output_layer
            if inner_model.share_embeddings_and_output_weights:
                tiled_output_weight = inner_model.shared_embedding_or_output_weight()

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(
            loss_function,
            args,
            batch,
            num_microbatches,
            step_global_batch_size,
            tiled_output_layer=tiled_output_layer,
            tiled_output_weight=tiled_output_weight,
        )

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )

    valid_step = True
    grad_norm = float("nan")
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate. Use the per-step global_batch_size when dynamic
        # batching is on so the scheduler's samples-seen counter tracks reality.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=mpu.get_context_parallel_world_size(),
            dp_with_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    global_batch_sizes: Sequence[int],
    rollout_data: dict | None = None,
    train_metrics_table_extra: dict | None = None,
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        global_batch_sizes (Sequence[int]): Rollout count per step (total
            across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts of a prompt). Same length as
            ``num_microbatches``; consumed by ``train_one_step`` for loss
            scaling and LR scheduler increments. Equals per-step sample count
            in the common case (1 rollout = 1 sample).
        rollout_data (dict | None): Full rollout batch, used only for
            actor-update-scoped metrics that must be computed after train-side
            filters and postprocess hooks have run.
        train_metrics_table_extra (dict | None): Additional metrics to include
            in the printed train metrics table. These are display-only; normal
            train logging still uses the per-step train metrics.
    """
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), f"num_microbatches and global_batch_sizes must have the same length, " f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, "When overlap_grad_reduce is True, config.no_sync_func must be None; " "a custom no_sync_func is not supported when overlapping grad-reduce"
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            logger.info("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{getattr(model[0], 'role', 'actor')} train",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):
        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # here we assume only one mtp layer
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1:
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            log_dict = {f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val for key, val in loss_dict.items()}
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses

            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            # Per-step gbs — uneven step sizes are easy to miss without this.
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
            log_dict["train/step"] = accumulated_step_id
            if role == "actor":
                if "train/loss" not in log_dict and "train/pg_loss" in log_dict:
                    log_dict["train/loss"] = log_dict["train/pg_loss"]
                if "train/lr" not in log_dict and "train/lr-pg_0" in log_dict:
                    log_dict["train/lr"] = log_dict["train/lr-pg_0"]
                log_dict.update(_episode_metrics_for_actor_update(rollout_data))
            logging_utils.log(args, log_dict, step_key="train/step", rollout_id=rollout_id)

            if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
                assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                # R3 replays rollout routing for the actor path, while ref
                # log-probs are computed with normal routing. The initial
                # actor/ref KL is therefore not expected to be exactly zero.
                if accumulated_step_id == 0 and not getattr(args, "use_rollout_routing_replay", False) and "train/kl_loss" in log_dict:
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {format_metrics_for_display(log_dict)}")
            if role == "actor" and getattr(args, "print_train_metrics_table", False):
                table_log_dict = {**(train_metrics_table_extra or {}), **log_dict}
                print_metrics_table(table_log_dict, accumulated_step_id, title=f"Actor Update {accumulated_step_id}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    microbatch_pbar.close()
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    args = get_args()
    pre_hook_disabled = should_disable_forward_pre_hook(args)
    if pre_hook_disabled:
        disable_forward_pre_hook(model)
    try:
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
        )
    finally:
        if pre_hook_disabled:
            enable_forward_pre_hook(model)


def initialize_model_and_optimizer(args: Namespace, role: str = "actor") -> tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    if reinit_critic_output_layer:
        _reinitialize_critic_output_layer(args, model)
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
    clear_memory()

    return model, optimizer, opt_param_scheduler, iteration
