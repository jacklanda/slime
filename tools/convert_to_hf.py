import os

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training.training import get_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from slime.backends.megatron_utils.checkpoint import load_checkpoint
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding
from slime.backends.megatron_utils.model import get_model_provider_func
from slime.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers
from slime.backends.megatron_utils.update_weight.hf_weight_iterator_direct import _get_megatron_local_param_infos
from slime.utils.arguments import parse_args
from slime.utils.distributed_utils import init_gloo_group


def add_checkpoint_args(parser):
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the converted HF model.",
    )
    parser.add_argument(
        "--check-same",
        action="store_true",
        default=False,
        help="Check if the converted model is the same as the original model.",
    )
    return parser


def main(args):
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.distributed_backend)
        args.rank = dist.get_rank()
        args.local_rank = local_rank
    init_gloo_group()

    init(args)

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    is_save_rank = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )

    # Conversion only needs model weights; avoid allocating DDP grad buffers and optimizer state.
    args.no_load_optim = True
    args.no_load_rng = True
    model = get_model(get_model_provider_func(args, "actor"), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    model_name = type(hf_config).__name__.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    vocab_size = tokenizer.vocab_size if args.vocab_size is None else args.vocab_size

    param_infos = _get_megatron_local_param_infos(args, model)

    state_dict = {}
    rank = dist.get_rank()
    for info in param_infos:
        if dist.get_rank() == info.src_rank:
            for name_, param_ in named_params_and_buffers(args, model):
                if name_ == info.name:
                    param = param_
                    break
        else:
            param = torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device())

        if pp_size > 1:
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                torch.distributed.broadcast(param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group())

        # broadcast params across ep ranks
        if ep_size > 1:
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                torch.distributed.broadcast(param, src=src_rank, group=mpu.get_expert_model_parallel_group())

        for key, value in info.attrs.items():
            setattr(param, key, value)

        param = all_gather_param(info.name, param)
        param = remove_padding(info.name, param, vocab_size)
        # use torch.distributed
        if is_save_rank:
            converted_named_tensors = convert_to_hf(args, model_name, info.name, param)
            for name, param in converted_named_tensors:
                state_dict[name] = param.cpu()
        del param

    if is_save_rank:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint, torch_dtype="auto", device_map="cpu", trust_remote_code=True
        )

        if args.check_same:
            for name, param in hf_model.named_parameters():
                if name in state_dict:
                    assert (
                        param.shape == state_dict[name].shape
                    ), f"Shape mismatch for {name}: {param.shape} vs {state_dict[name].shape}"
                    assert torch.all(param == state_dict[name]), f"Value mismatch for {name}"
                else:
                    print(f"Warning: {name} not found in state_dict")

        if args.output_dir:
            tokenizer.save_pretrained(args.output_dir)
            print(hf_model.load_state_dict(state_dict, strict=False))
            hf_model.save_pretrained(args.output_dir)

    dist.barrier()


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_checkpoint_args)
    main(args)
