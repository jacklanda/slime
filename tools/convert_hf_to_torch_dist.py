import gc
import os
import shutil
import sys
from pathlib import Path


# Megatron-LM only packages megatron-core; megatron.training is loaded from
# the source checkout. Discover the standard sibling checkout used by slime.
megatron_lm_path = os.environ.get("MEGATRON_LM_PATH")
if megatron_lm_path is None:
    sibling_checkout = Path(__file__).resolve().parents[2] / "Megatron-LM"
    if (sibling_checkout / "megatron" / "training").is_dir():
        megatron_lm_path = str(sibling_checkout)
if megatron_lm_path is not None and megatron_lm_path not in sys.path:
    sys.path.insert(0, megatron_lm_path)

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import slime_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import print_memory


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("hf_checkpoint_positional", nargs="?", help="HuggingFace model path")
    parser.add_argument("--hf-checkpoint", type=str, default=None, help="HuggingFace model path")
    parser.add_argument(
        "--custom-model-provider-path",
        type=str,
        default=None,
        help="Path to a custom model provider function.",
    )
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_convertion_args)
    if args.hf_checkpoint is not None and args.hf_checkpoint_positional is not None:
        raise ValueError("Specify the HuggingFace checkpoint either positionally or with --hf-checkpoint, not both.")
    args.hf_checkpoint = args.hf_checkpoint or args.hf_checkpoint_positional
    if args.hf_checkpoint is None:
        raise ValueError("A HuggingFace checkpoint path is required.")
    del args.hf_checkpoint_positional
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def main():
    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

    # Conversion is a one-shot offline path; prefer a slower blocking D2H copy
    # over Megatron's default non-blocking checkpoint preload, which can fail
    # with cudaErrorInvalidValue on some driver / shared-filesystem setups.
    _orig_preload_tensors = filesystem_async_module.FileSystemWriterAsync.preload_tensors

    def _preload_tensors_blocking(write_buckets, non_blocking=True):
        return _orig_preload_tensors(write_buckets, non_blocking=False)

    filesystem_async_module.FileSystemWriterAsync.preload_tensors = staticmethod(_preload_tensors_blocking)

    if torch.version.hip:
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    args = get_args()
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        # change to release ckpt
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
