import logging
from time import perf_counter

import ray

from slime.utils import logging_utils
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import (
    configure_logger,
    finish_tracking,
    init_tracking,
    suppress_known_training_warnings,
)
from slime.utils.misc import should_run_periodic_action
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.train_step_metrics import build_step_timing_metrics


logger = logging.getLogger(__name__)


def _run_rollout_only(args, rollout_manager, num_rollout_per_epoch):
    """Run inference rollouts without allocating or calling trainer actors."""
    if args.start_rollout_id is None:
        args.start_rollout_id = 0
    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        step_start = perf_counter()
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        generated_samples = ray.get(rollout_manager.generate.remote(rollout_id))
        if generated_samples == 0:
            logger.info("rollout-only dataset exhausted at step %s", rollout_id)
            break

        if not getattr(args, "rollout_only_skip_episode_dump", False):
            ray.get(rollout_manager.save_rllm_episodes.remote(rollout_id))

        # The inference fast path commits the cursor before its asynchronous
        # shard write. Other rollout-only workflows persist it here.
        if args.rollout_global_dataset and not getattr(args, "rollout_only_inference_fast_path", False):
            ray.get(rollout_manager.save.remote(rollout_id))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        logger.info("rollout-only step %s completed in %.2fs", rollout_id, perf_counter() - step_start)


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    if getattr(args, "debug_rollout_only", False):
        try:
            _run_rollout_only(args, rollout_manager, num_rollout_per_epoch)
        finally:
            ray.get(rollout_manager.dispose.remote())
            finish_tracking(args)
        return

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))
        if release_train:
            actor_model.create()
        actor_model.update_weights()
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        step_start = perf_counter()
        phase_times = {}

        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            phase_start = perf_counter()
            ray.get(rollout_manager.eval.remote(rollout_id))
            phase_times["pre_train_eval"] = perf_counter() - phase_start

        phase_start = perf_counter()
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        phase_times["rollout_phase"] = perf_counter() - phase_start

        if args.offload_rollout:
            phase_start = perf_counter()
            ray.get(rollout_manager.offload.remote())
            phase_times["rollout_offload"] = perf_counter() - phase_start

        if release_train:
            phase_start = perf_counter()
            actor_model.create()
            phase_times["trainer_create"] = perf_counter() - phase_start

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        actor_train_results = None
        phase_start = perf_counter()
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                actor_train_results = ray.get(
                    actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs)
                )
            else:
                ray.get(value_refs)
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        train_time = perf_counter() - phase_start

        useful_training_tokens = None
        if actor_train_results is not None:
            for result in actor_train_results:
                if isinstance(result, dict) and "useful_training_tokens" in result:
                    useful_training_tokens = int(result["useful_training_tokens"])
                    break

        if actor_trains:
            phase_start = perf_counter()
            ray.get(rollout_manager.save_rllm_episodes.remote(rollout_id))
            phase_times["episode_save"] = perf_counter() - phase_start

        should_save = release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        )
        if should_save:
            phase_start = perf_counter()
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            phase_times["checkpoint"] = perf_counter() - phase_start

        phase_start = perf_counter()
        offload_train(actor_trains)
        phase_times["train_cleanup"] = perf_counter() - phase_start
        if args.offload_rollout and not release_train:
            phase_start = perf_counter()
            ray.get(rollout_manager.onload_weights.remote())
            phase_times["rollout_weights_onload"] = perf_counter() - phase_start
        should_update_weights = release_train or (rollout_id + 1) % args.update_weights_interval == 0
        if should_update_weights:
            phase_start = perf_counter()
            actor_model.update_weights()
            phase_times["update_weights"] = perf_counter() - phase_start

        if args.offload_rollout:
            phase_start = perf_counter()
            ray.get(rollout_manager.onload_kv.remote())
            phase_times["rollout_kv_onload"] = perf_counter() - phase_start

        should_eval = should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch)
        if should_eval:
            phase_start = perf_counter()
            ray.get(rollout_manager.eval.remote(rollout_id))
            phase_times["eval"] = perf_counter() - phase_start

        step_time = perf_counter() - step_start
        timing_metrics = build_step_timing_metrics(
            step_time=step_time,
            train_time=train_time,
            phase_times=phase_times,
            useful_training_tokens=useful_training_tokens,
        )
        logger.info("step perf %s: %s", rollout_id, timing_metrics)
        timing_metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, timing_metrics, step_key="rollout/step", rollout_id=rollout_id)

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    suppress_known_training_warnings()
    args = parse_args()
    train(args)
