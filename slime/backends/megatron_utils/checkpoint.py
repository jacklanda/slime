import json
import logging
import os
import re
import shutil
from pathlib import Path

import torch

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint_megatron
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

try:
    # Checkpoint staging copies every shard GPU->CPU with `non_blocking=True`, which makes
    # torch allocate a *pinned* host buffer per tensor. On this CUDA/driver stack that
    # cudaHostAlloc can fail with cudaErrorInvalidValue once the process has already
    # registered a lot of pinned memory (the weight backuper pins a full model copy, and
    # release-train recreates the actor every step so registrations churn). The failure is
    # per-rank and non-deterministic: some ranks write their .distcp shards fine while
    # others die, leaving a torn checkpoint dir with no latest_checkpointed_iteration.txt.
    #
    # Retry the bucket with blocking (pageable) copies instead of taking the job down. This
    # is slower for the affected bucket but produces an identical checkpoint.
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    _ACCELERATOR_ERROR = getattr(torch, "AcceleratorError", RuntimeError)

    @staticmethod
    def _preload_tensors_with_pinned_fallback(write_buckets, non_blocking=True):
        result = []
        for bucket in write_buckets:
            file_name, storage_key, (bytes_data, tensor_data) = bucket
            try:
                staged = [(item, tensor.to("cpu", non_blocking=non_blocking)) for item, tensor in tensor_data]
                if non_blocking:
                    torch.cuda.synchronize()
            except (_ACCELERATOR_ERROR, RuntimeError):
                if not non_blocking:
                    raise
                logger.warning(
                    "Pinned checkpoint staging failed for %s; retrying with blocking copies.",
                    file_name,
                    exc_info=True,
                )
                torch.cuda.synchronize()
                staged = [(item, tensor.to("cpu", non_blocking=False)) for item, tensor in tensor_data]
            result.append((file_name, storage_key, (bytes_data, staged)))
        return result

    FileSystemWriterAsync.preload_tensors = _preload_tensors_with_pinned_fallback

except ImportError:
    pass

__all__ = ["save_checkpoint"]


_ITERATION_DIR_RE = re.compile(r"iter_(\d{7})")
_STAGING_DIR_RE = re.compile(r"\.iter_(\d{7})\.incomplete")


def _checkpoint_is_complete(path: Path) -> bool:
    try:
        required_files = (path / ".metadata", path / "metadata.json", path / "common.pt")
        if not all(checkpoint_file.is_file() and checkpoint_file.stat().st_size > 0 for checkpoint_file in required_files):
            return False
        if not any(path.glob("*.distcp")):
            return False
        metadata = json.loads((path / "metadata.json").read_text())
        return isinstance(metadata, dict) and metadata.get("sharded_backend") == "torch_dist" and metadata.get("sharded_backend_version") == 1 and metadata.get("common_backend") == "torch" and metadata.get("common_backend_version") == 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _read_latest_iteration(save_dir: Path) -> int | None:
    tracker = save_dir / "latest_checkpointed_iteration.txt"
    try:
        return int(tracker.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _remove_failed_checkpoint_artifacts(save_dir: Path, iteration: int) -> None:
    """Remove staging directories and final directories proven to be incomplete."""
    latest_iteration = _read_latest_iteration(save_dir)
    for path in save_dir.iterdir():
        staging_match = _STAGING_DIR_RE.fullmatch(path.name)
        if staging_match and path.is_dir():
            logger.warning("Removing stale checkpoint staging directory %s", path)
            shutil.rmtree(path)
            continue

        iteration_match = _ITERATION_DIR_RE.fullmatch(path.name)
        if not iteration_match or not path.is_dir() or _checkpoint_is_complete(path):
            continue
        checkpoint_iteration = int(iteration_match.group(1))
        if checkpoint_iteration == iteration or (latest_iteration is not None and checkpoint_iteration > latest_iteration):
            logger.warning("Removing incomplete checkpoint directory %s", path)
            shutil.rmtree(path)


def _recover_checkpoint_state(save_dir: Path) -> None:
    """Reconcile the tracker with complete checkpoints and remove failed saves."""
    latest_iteration = _read_latest_iteration(save_dir)
    complete_iterations = sorted(int(match.group(1)) for path in save_dir.iterdir() if (match := _ITERATION_DIR_RE.fullmatch(path.name)) and path.is_dir() and _checkpoint_is_complete(path))
    if latest_iteration is not None and not complete_iterations:
        raise RuntimeError(f"Checkpoint tracker points to iteration {latest_iteration}, but {save_dir} contains no complete torch_dist checkpoint")
    if complete_iterations:
        recovered_iteration = complete_iterations[-1]
        latest_is_complete = latest_iteration in complete_iterations
        if not latest_is_complete or recovered_iteration > latest_iteration:
            logger.warning(
                "Recovering checkpoint tracker from %s to complete iteration %s",
                latest_iteration,
                recovered_iteration,
            )
            _atomic_write_tracker(save_dir, recovered_iteration)
            latest_iteration = recovered_iteration
    _remove_failed_checkpoint_artifacts(save_dir, iteration=-1)


def _atomic_write_tracker(save_dir: Path, iteration: int) -> None:
    tracker = save_dir / "latest_checkpointed_iteration.txt"
    temporary_tracker = save_dir / ".latest_checkpointed_iteration.txt.tmp"
    with temporary_tracker.open("w") as output:
        output.write(str(iteration))
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_tracker, tracker)
    _fsync_directory(save_dir)


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _publish_checkpoint(staging_dir: Path, save_dir: Path, iteration: int) -> None:
    staged_checkpoint = staging_dir / f"iter_{iteration:07d}"
    final_checkpoint = save_dir / f"iter_{iteration:07d}"
    if not _checkpoint_is_complete(staged_checkpoint):
        raise RuntimeError(f"Checkpoint save did not produce a complete checkpoint: {staged_checkpoint}")
    if final_checkpoint.exists():
        if _checkpoint_is_complete(final_checkpoint):
            raise FileExistsError(f"Refusing to replace complete checkpoint: {final_checkpoint}")
        shutil.rmtree(final_checkpoint)
    os.replace(staged_checkpoint, final_checkpoint)
    _fsync_directory(save_dir)
    _atomic_write_tracker(save_dir, iteration)
    shutil.rmtree(staging_dir, ignore_errors=True)


def _distributed_rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _raise_rank_zero_error(error: BaseException | None) -> None:
    if not torch.distributed.is_initialized():
        if error is not None:
            raise error
        return

    error_message = [None if error is None else f"{type(error).__name__}: {error}"]
    torch.distributed.broadcast_object_list(error_message, src=0)
    if error is not None:
        raise error
    if error_message[0] is not None:
        raise RuntimeError(f"Rank 0 checkpoint operation failed: {error_message[0]}")


def save_checkpoint(iteration, *args, **kwargs):
    """Save a torch_dist checkpoint and publish it as one filesystem transaction."""
    megatron_args = get_args()
    if not (getattr(megatron_args, "use_dist_ckpt", False) and getattr(megatron_args, "ckpt_format", None) == "torch_dist" and getattr(megatron_args, "save", None)):
        return _save_checkpoint_megatron(iteration, *args, **kwargs)

    save_dir = Path(megatron_args.save)
    staging_dir = save_dir / f".iter_{iteration:07d}.incomplete"
    rank = _distributed_rank()

    preparation_error = None
    if rank == 0:
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            _recover_checkpoint_state(save_dir)
            _remove_failed_checkpoint_artifacts(save_dir, iteration)
            final_checkpoint = save_dir / f"iter_{iteration:07d}"
            if not final_checkpoint.exists():
                staging_dir.mkdir()
        except BaseException as exc:
            preparation_error = exc
    _raise_rank_zero_error(preparation_error)
    final_checkpoint = save_dir / f"iter_{iteration:07d}"
    if _checkpoint_is_complete(final_checkpoint):
        raise FileExistsError(f"Refusing to replace complete checkpoint: {final_checkpoint}")

    original_save_dir = megatron_args.save
    megatron_args.save = str(staging_dir)
    try:
        result = _save_checkpoint_megatron(iteration, *args, **kwargs)
        if getattr(megatron_args, "async_save", False):
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)
        _barrier()

        publish_error = None
        if rank == 0:
            try:
                _publish_checkpoint(staging_dir, save_dir, iteration)
            except BaseException as exc:
                publish_error = exc
        _raise_rank_zero_error(publish_error)
        if rank != 0 and (not _checkpoint_is_complete(save_dir / f"iter_{iteration:07d}") or _read_latest_iteration(save_dir) != iteration):
            raise RuntimeError(f"Rank 0 failed to publish checkpoint iteration {iteration}")
        return result
    finally:
        megatron_args.save = original_save_dir


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(load_path), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        load_dir = Path(load_path)
        if getattr(args, "use_dist_ckpt", False) and getattr(args, "ckpt_format", None) == "torch_dist" and (load_dir / "latest_checkpointed_iteration.txt").is_file():
            recovery_error = None
            if _distributed_rank() == 0:
                try:
                    _recover_checkpoint_state(load_dir)
                except BaseException as exc:
                    recovery_error = exc
            _raise_rank_zero_error(recovery_error)
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(re.fullmatch(r"iter_\d{7}", Path(path).name))


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True))
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
