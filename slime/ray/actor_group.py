import logging
import os
import shutil
import time
from pathlib import Path

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from ray.util.state import get_actor

from slime.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST, add_default_ray_env_vars
from slime.utils.train_infer_consistency import validate_rollout_weight_versions

logger = logging.getLogger(__name__)

# Tearing down eight trainer processes that each hold tens of GB of device
# memory is dominated by CUDA context teardown, not by the kill itself.
_TRAIN_ACTOR_SHUTDOWN_TIMEOUT = float(os.environ.get("SLIME_TRAIN_ACTOR_SHUTDOWN_TIMEOUT", "180"))
_TRAIN_ACTOR_SHUTDOWN_POLL_INTERVAL = 0.25


@ray.remote(num_cpus=0)
def _surviving_train_processes(pids):
    """Return which of ``pids`` still exist or still hold device memory on this node.

    ``ray.kill`` only guarantees the actor is unreachable; the worker process
    keeps its CUDA context -- and therefore its device memory -- until the
    driver finishes reclaiming it. NVML's compute-process list is the ground
    truth for "the memory is actually back", so it is what gates the resume of
    the colocated SGLang KV pool.
    """
    import psutil

    pids = set(pids)
    surviving = set()
    for pid in pids:
        try:
            if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                surviving.add(pid)
        except psutil.NoSuchProcess:
            continue
        except Exception:
            # Unknown liveness must not be read as "exited"; retry next poll.
            surviving.add(pid)

    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if proc.pid in pids:
                        surviving.add(proc.pid)
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.warning("Cannot verify trainer GPU memory release via NVML, falling back to process liveness: %s", e)

    return surviving


class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """

    def __init__(
        self,
        args,
        num_nodes,
        num_gpus_per_node,
        pg: tuple[PlacementGroup, list[int], list[int]],
        num_gpus_per_actor: float = 1,
        role: str = "actor",
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        actor_cls=None,
    ) -> None:
        self.args = args
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self._pg = pg
        self._num_gpus_per_actor = num_gpus_per_actor
        self.role = role
        self._actor_cls = actor_cls
        self._with_ref = with_ref
        self._with_opd_teacher = with_opd_teacher
        self._rollout_manager = None
        self._disk_weight_version = getattr(args, "update_weight_start_version", 0)
        self._actor_handlers = []

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices, _reordered_gpu_ids = pg

        env_vars = {
            # because sglang will always set NCCL_CUMEM_ENABLE to 0
            # we need also set it to 0 to prevent nccl error.
            "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
            **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
            **self.args.train_env_vars,
        }
        if "SLIME_TENSOR_BACKUP_PIN_MEMORY" in os.environ:
            env_vars["SLIME_TENSOR_BACKUP_PIN_MEMORY"] = os.environ["SLIME_TENSOR_BACKUP_PIN_MEMORY"]

        if self.args.offload_train and self.args.train_backend == "megatron":
            import torch_memory_saver

            for path in [
                "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
                "torch_memory_saver_hook_mode_preload.abi3.so",
            ]:
                dynlib_path = os.path.join(
                    os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                    path,
                )
                if os.path.exists(dynlib_path):
                    break
            else:
                raise FileNotFoundError("Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed.")

            env_vars["LD_PRELOAD"] = dynlib_path
            env_vars["TMS_INIT_ENABLE"] = "1"
            env_vars.setdefault("TMS_INIT_ENABLE_CPU_BACKUP", "1")

        # We cannot do routing replay for critic.
        if self.args.use_routing_replay and self.role == "actor":
            env_vars["ENABLE_ROUTING_REPLAY"] = "1"

        if self._actor_cls is None:
            from slime.backends.megatron_utils.actor import MegatronTrainRayActor

            actor_impl = MegatronTrainRayActor
        else:
            actor_impl = self._actor_cls

        actor_options = {
            "num_gpus": 1,
            "runtime_env": {"env_vars": add_default_ray_env_vars(env_vars)},
        }
        if getattr(self.args, "rollout_data_transport", "object-store") == "nixl":
            actor_options["enable_tensor_transport"] = True
        TrainRayActor = ray.remote(**actor_options)(actor_impl)

        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        """Do one rollout training. Returns a list of Ray refs (one per worker).

        For critics, each ref resolves to ``{"values": [cpu tensors...]}`` (or ``{}``
        for non-last-PP-stage workers). Actor refs resolve to ``None``.

        ``external_data`` may be a list (one item per worker) or a single dict
        broadcast to all workers.
        """
        if isinstance(external_data, list):
            assert len(external_data) == len(self._actor_handlers)
            return [actor.train.remote(rollout_id, rollout_data_ref, external_data=ed) for actor, ed in zip(self._actor_handlers, external_data, strict=False)]
        return [actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data) for actor in self._actor_handlers]

    def save_model(self, rollout_id, force_sync=False):
        """Save actor model"""
        ret = ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])
        if self._release_train_enabled():
            self.args.load = self.args.save
            self.args.ckpt_step = None
            self.args.finetune = False
            self.args.no_load_optim = self.args.no_save_optim
            self.args.no_load_rng = False
            self._remove_previous_temporary_checkpoint(rollout_id)
        return ret

    def _remove_previous_temporary_checkpoint(self, rollout_id):
        previous_rollout_id = rollout_id - 1
        if previous_rollout_id < 0 or (previous_rollout_id + 1) % self.args.save_interval == 0:
            return

        checkpoint_dir = Path(self.args.save) / f"iter_{previous_rollout_id:07d}"
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)
            logger.info("Removed temporary release-train checkpoint %s", checkpoint_dir)

    def update_weights(self):
        """Broadcast weights from rank 0 to all other ranks."""
        if not self._full_disk_weight_update_enabled():
            return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

        if not self._actor_handlers:
            raise RuntimeError("Cannot publish full disk weights without active training actors; call create() first")

        weight_version = self._disk_weight_version + 1
        disk_weight_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{weight_version:06d}"
        ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
        self._disk_weight_version = weight_version
        if self._release_train_enabled():
            self.release()
        self._reload_rollout_weights_from_disk(disk_weight_dir, str(weight_version))

    def onload(self):
        return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

    def offload(self):
        return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

    def release(self):
        actors = self._actor_handlers
        if not actors:
            return

        actor_ids = [actor._ray_actor_id.hex() for actor in actors]
        node_processes: dict[str, set[int]] = {}
        for actor_id in actor_ids:
            state = get_actor(actor_id, timeout=1)
            if state is None or state.node_id is None or state.pid is None:
                raise RuntimeError(f"Cannot identify training actor process before release: actor_id={actor_id}")
            node_processes.setdefault(state.node_id, set()).add(int(state.pid))

        self._actor_handlers = []
        for actor in actors:
            ray.kill(actor, no_restart=True)

        # Wait until every trainer process is actually gone from its node's
        # GPUs. The colocated SGLang engines resume their KV pool right after
        # this returns, and that allocation dies with CUDA_ERROR_OUT_OF_MEMORY
        # -- taking the scheduler subprocess with it -- if a trainer still
        # holds the memory. ray.kill only makes the actor unreachable; the
        # worker keeps its CUDA context for as long as teardown takes.
        deadline = time.monotonic() + _TRAIN_ACTOR_SHUTDOWN_TIMEOUT
        pending = dict(node_processes)
        while pending:
            node_ids = list(pending)
            surviving_per_node = ray.get([_surviving_train_processes.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)).remote(sorted(pending[node_id])) for node_id in node_ids])
            pending = {node_id: surviving for node_id, surviving in zip(node_ids, surviving_per_node, strict=True) if surviving}
            if not pending:
                break
            if time.monotonic() >= deadline:
                remaining = {node_id: sorted(pids) for node_id, pids in pending.items()}
                raise TimeoutError(f"Timed out waiting for {sum(len(p) for p in pending.values())} training actor processes to " f"release their GPU memory after {_TRAIN_ACTOR_SHUTDOWN_TIMEOUT}s: {remaining}")
            time.sleep(_TRAIN_ACTOR_SHUTDOWN_POLL_INTERVAL)

    def create(self, rollout_manager=None):
        if self._actor_handlers:
            return None
        if rollout_manager is not None:
            self._rollout_manager = rollout_manager
        self.args.update_weight_start_version = self._disk_weight_version
        self._allocate_gpus_for_actor(self._pg, self._num_gpus_per_actor)
        start_rollout_ids = ray.get(
            [
                actor.init.remote(
                    self.args,
                    self.role,
                    with_ref=self._with_ref,
                    with_opd_teacher=self._with_opd_teacher,
                )
                for actor in self._actor_handlers
            ]
        )
        if self._rollout_manager is not None:
            self.set_rollout_manager(self._rollout_manager)
        return start_rollout_ids

    def clear_memory(self):
        return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

    def set_rollout_manager(self, rollout_manager):
        self._rollout_manager = rollout_manager
        return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])

    def _release_train_enabled(self):
        return self.role == "actor" and getattr(self.args, "release_train", False)

    def _full_disk_weight_update_enabled(self):
        return self.role == "actor" and self.args.update_weight_mode == "full" and self.args.update_weight_transport == "disk"

    def _reload_rollout_weights_from_disk(self, disk_weight_dir, weight_version):
        assert self._rollout_manager is not None, "disk weight update requires a rollout manager."
        if self.args.offload_rollout:
            ray.get(self._rollout_manager.onload_weights.remote())
        engines, *_ = ray.get(self._rollout_manager.get_updatable_engines_and_lock.remote())
        if not engines:
            if not self.args.update_weight_disk_keep_files:
                shutil.rmtree(disk_weight_dir, ignore_errors=True)
            return
        if self.args.update_weight_local_checkpoint_dir:
            # each host pulls the published checkpoint onto local disk (e.g. NVMe) and
            # the engines reload from there; the pull is disk-only, so it runs before
            # pause and overlaps generation
            ray.get([engine.pull_weights.remote(int(weight_version)) for engine in engines])
            model_path = self.args.update_weight_local_checkpoint_dir
        else:
            model_path = str(disk_weight_dir)
        ray.get([engine.pause_generation.remote() for engine in engines])
        ray.get([engine.flush_cache.remote() for engine in engines])
        ray.get(
            [
                engine.update_weights_from_disk.remote(
                    model_path=model_path,
                    weight_version=weight_version,
                )
                for engine in engines
            ]
        )
        verify_versions = self.args.ci_test or getattr(self.args, "verify_rollout_weight_versions", False)
        if verify_versions:
            engine_versions = ray.get([engine.get_weight_version.remote() for engine in engines])
            validate_rollout_weight_versions(weight_version, engine_versions)
        if not self.args.update_weight_disk_keep_files:
            shutil.rmtree(disk_weight_dir, ignore_errors=True)
        ray.get([engine.continue_generation.remote() for engine in engines])
