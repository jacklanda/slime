import argparse
import copy
import json
import logging
import os
from typing import Any

import yaml

from slime.backends.sglang_utils.arguments import sglang_parse_args
from slime.backends.sglang_utils.arguments import validate_args as sglang_validate_args
from slime.backends.sglang_utils.external import apply_external_engine_info_to_args
from slime.utils.data import read_file
from slime.utils.eval_config import EvalDatasetConfig, build_eval_dataset_configs, ensure_dataset_list
from slime.utils.hf_config import register_gemma4_config_aliases
from slime.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


def reset_arg(parser, name, **kwargs):
    """
    Reset the default value of a Megatron argument.
    :param parser: The argument parser.
    :param name: The name of the argument to reset.
    :param default: The new default value.
    """
    for action in parser._actions:
        if name in action.option_strings:
            if "default" in kwargs:
                action.default = kwargs["default"]
            break
    else:
        parser.add_argument(name, **kwargs)


def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # Ray
        def add_cluster_arguments(parser):
            parser.add_argument("--actor-num-nodes", type=int, default=1, help="Number of nodes for training actor")
            parser.add_argument(
                "--actor-num-gpus-per-node", type=int, default=8, help="Number of gpus per node for training actor"
            )

            parser.add_argument(
                "--rollout-num-gpus",
                type=int,
                default=None,
                help=(
                    "Number of GPUs for inference. Note that when using --colocate, "
                    "i.e. the training and the inference engines are on the same gpus, this param will be set as "
                    "actor_num_gpus_per_node * actor_num_nodes unless it is explicitly set. "
                    "Set it to 0 to launch routers without local SGLang engines."
                ),
            )
            parser.add_argument(
                "--rollout-num-gpus-per-engine",
                type=int,
                default=1,
                help="Number of GPUs per inference engine, just like the tp_size in sglang.",
            )
            parser.add_argument(
                "--num-gpus-per-node",
                type=int,
                default=8,
                help=(
                    "Number of gpus per node for rollout."
                    "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
                ),
            )
            parser.add_argument(
                "--colocate",
                action="store_true",
                default=False,
                help=(
                    "Whether to colocate the inference engines and the actor. "
                    "Turning this on will also set --offload to true."
                ),
            )
            parser.add_argument(
                "--offload",
                action="store_true",
                default=False,
                help=("Equivalent to --offload-train + --offload-rollout. "),
            )
            parser.add_argument(
                "--offload-train",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the training actor to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )
            parser.add_argument(
                "--offload-rollout",
                action=argparse.BooleanOptionalAction,
                help=(
                    "Whether to offload the rollout generator to CPU during training. "
                    "This will always be true when --colocate is set."
                ),
            )

            reset_arg(parser, "--distributed-backend", type=str, default="nccl")
            reset_arg(parser, "--distributed-timeout-minutes", type=int, default=10)

            return parser

        def add_train_arguments(parser):
            parser.add_argument(
                "--true-on-policy-mode",
                action="store_true",
                default=False,
                help="Enable the internal deterministic train/rollout parity path.",
            )
            parser.add_argument(
                "--recompute-logprobs-via-prefill",
                action="store_true",
                default=False,
                help="Replace decode logprobs with deterministic clean-prefill scores.",
            )
            parser.add_argument(
                "--use-yarn-rope",
                action="store_true",
                help="Use static YaRN RoPE in the Megatron actor model.",
            )
            parser.add_argument(
                "--yarn-rope-scaling-factor",
                type=float,
                default=1.0,
                help="Static YaRN scaling factor used by the Megatron actor model.",
            )
            parser.add_argument(
                "--yarn-original-max-position-embeddings",
                type=int,
                default=32768,
                help="Native context length used to train the model before YaRN scaling.",
            )
            # --train-backend is parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--qwen-gdn-backend",
                type=str,
                choices=["fla", "flashqla"],
                default="fla",
                help="GDN implementation backend for Qwen linear-attention layers.",
            )
            parser.add_argument(
                "--train-env-vars",
                type=json.loads,
                default="{}",
                help="Extra environment variables for training process, e.g. PyTorch memory management ones.",
            )
            parser.add_argument(
                "--train-memory-margin-bytes",
                type=int,
                default=1024**3,
                help="Add margin for train memory allocation. By default we will reserve 1GB as margin.",
            )
            parser.add_argument(
                "--megatron-to-hf-mode",
                choices=["raw", "bridge"],
                default="raw",
                help="The method to convert megatron weights to hugging face weights for SGLang.",
            )
            # Delta weight sync.
            parser.add_argument(
                "--update-weight-mode",
                choices=["full", "delta"],
                default="full",
                help=(
                    "Weight sync strategy. 'full' (default) broadcasts every parameter "
                    "every sync. 'delta' diffs each sync against a pinned-CPU snapshot of the "
                    "previous one and ships only the changed bytes (disk transport only)."
                ),
            )
            parser.add_argument(
                "--update-weight-transport",
                choices=["nccl", "disk"],
                default="nccl",
                help=(
                    "Carrier for weight sync. In full mode, 'nccl' broadcasts chunks and "
                    "'disk' writes a complete HF checkpoint under --update-weight-disk-dir "
                    "before engines reload it. Delta mode is 'disk' only: each host applies the "
                    "published deltas into its local checkpoint and reloads via update_weights_from_disk."
                ),
            )
            parser.add_argument(
                "--release-train",
                action="store_true",
                default=False,
                help=(
                    "Release Megatron training actors during rollout and recreate them before each train step. "
                    "Requires disk weight sync and --save for Megatron reload."
                ),
            )
            parser.add_argument(
                "--update-weight-disk-dir",
                type=str,
                default=None,
                help=(
                    "Filesystem directory for disk-backed weight sync. In --update-weight-mode=full, "
                    "one complete HF checkpoint directory is written per sync. In delta mode, "
                    "one delta directory (changed tensors only) is written per sync."
                ),
            )
            parser.add_argument(
                "--update-weight-disk-keep-files",
                action="store_true",
                default=False,
                help=(
                    "Skip cleanup of full-checkpoint directories written by "
                    "--update-weight-mode=full --update-weight-transport=disk."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-encoding",
                choices=["xor", "overwrite"],
                default="xor",
                help=(
                    "On-disk delta encoding for --update-weight-mode=delta --update-weight-transport=disk. "
                    "'xor' (default): new ^ old — smallest wire and fastest, but an involution that must be "
                    "applied exactly once against the correct base (applying it twice reverts). 'overwrite': "
                    "changed positions + new absolute values — larger, but idempotent (re-applicable any "
                    "number of times). Both are byte-level and dtype-blind; the engine reads the choice from "
                    "each version's index metadata."
                ),
            )
            parser.add_argument(
                "--update-weight-delta-checksum",
                choices=["xxh3-128", "blake3", "adler32"],
                default="xxh3-128",
                help=(
                    "Per-tensor integrity checksum for disk delta apply. The checksum is not the "
                    "apply bottleneck (the apply is decompress + XOR bound), so this is a digest-"
                    "property choice, not a speed one. 'xxh3-128' (default): widest fast non-"
                    "cryptographic digest, negligible accidental-corruption collisions. 'blake3': "
                    "cryptographic digest, for untrusted storage. 'adler32': 32-bit, for interop "
                    "with systems that expect it. The engine reads the choice from each version's "
                    "index metadata."
                ),
            )
            parser.add_argument(
                "--custom-update-weight-post-write-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function called on each trainer rank after a disk weight "
                    "sync's files are written (full or delta), before the engines read them — to "
                    "publish the writes on a non-POSIX filesystem (no cross-host visibility "
                    "without an explicit sync). "
                    "Signature: ``def hook(args, version_dir: str, rollout_engines) -> None``; the hook gates itself."
                ),
            )
            parser.add_argument(
                "--update-weight-local-checkpoint-dir",
                type=str,
                default=None,
                help=(
                    "Rollout-host-local directory (NVMe) holding a full HF checkpoint kept in "
                    "sync by each engine's /pull_weights: every host copies a published full "
                    "checkpoint as-is or patches published deltas in place, and the engines "
                    "reload from it. Required for --update-weight-mode=delta "
                    "--update-weight-transport=disk; optional for full disk sync (engines then "
                    "pull to local disk instead of reading the shared dir directly). The "
                    "read-side counterpart of --custom-update-weight-post-write-path is the engine's "
                    "--sglang-custom-pull-weights-pre-read-hook."
                ),
            )
            parser.add_argument(
                "--custom-model-provider-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom model provider function. "
                    "If set, we will use this function instead of the default model provider. "
                    "The function should have the signature "
                    "`def custom_model_provider(pre_process: bool, post_process: bool, vp_stage: int | None = None) -> GPTModel`. "
                    "Example: 'my_module.my_model_provider'."
                ),
            )
            parser.add_argument(
                "--recompute-loss-function",
                action="store_true",
                help="Whether to disable recompute loss function to save memory during training.",
            )
            parser.add_argument(
                "--log-probs-chunk-size", type=int, default=-1, help="Chunk size to compute log probs to save memory"
            )
            parser.add_argument(
                "--only-train-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to TRAIN. All other parameters will be FROZEN. 
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Train ONLY MoE experts:
                            --only-train-params-name-list experts

                        2. Train ONLY Indexer parameters:
                            --only-train-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Train ONLY Layer 20 to 23:
                            --only-train-params-name-list layers\.2[0-3]\.
                        """,
            )

            parser.add_argument(
                "--freeze-params-name-list",
                type=str,
                nargs="*",
                default=None,
                help="""List of regex patterns of parameter names to FREEZE. Other parameters will remain trainable.
                        Supports Python regex syntax (re.search).

                        Examples:
                        1. Freeze Embeddings and Output Layer (common for fine-tuning):
                            --freeze-params-name-list embedding output_layer

                        2. Freeze Indexer parameters:
                            --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj

                        3. Freeze specific projection layers (e.g., all Gate/Up projections):
                            --freeze-params-name-list linear_fc1
                        """,
            )
            parser.add_argument(
                "--allgather-cp",
                action="store_true",
                default=False,
            )

            return parser

        # rollout
        def add_rollout_arguments(parser):
            parser.add_argument(
                "--hf-checkpoint",
                type=str,
                default=None,
                help=(
                    "The huggingface checkpoint of the trained model. "
                    "This is used to initialize sglang and also provide the tokenizer. "
                    "Note that, we will always update the parameters in sglang with that of megatron before training, "
                    "so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. "
                    "It doesn't necessary need to contain the most up-to-date parameters."
                ),
            )
            parser.add_argument(
                "--model-name",
                type=str,
                default=None,
                help=(
                    "The name of the model, this is used to convert the megatron weights into huggingface format. "
                    "If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. "
                    "Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
                ),
            )
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
            parser.add_argument(
                "--rollout-temperature",
                type=float,
                default=1.0,
                help="the temperature for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-top-p", type=float, default=1.0, help="the top-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-top-k", type=int, default=-1, help="the top-k for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-min-p", type=float, default=0.0, help="the min-p for the inference engine during rollout."
            )
            parser.add_argument(
                "--rollout-presence-penalty",
                type=float,
                default=0.0,
                help="the presence penalty for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-repetition-penalty",
                type=float,
                default=1.0,
                help="the repetition penalty for the inference engine during rollout.",
            )
            parser.add_argument(
                "--rollout-max-context-len",
                type=int,
                default=None,
                help=(
                    "The maximum context size for the inference engine during rollout."
                    "It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
                ),
            )
            parser.add_argument(
                "--rollout-max-prompt-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the prompt for the inference engine during rollout. "
                    "If set, we will filter out the long prompts during initialization of the global dataset. "
                    "This is not recommended if the dataset is large."
                ),
            )
            parser.add_argument(
                "--rollout-max-response-len",
                type=int,
                default=None,
                help=(
                    "The maximum length of the response for the inference engine during rollout. "
                    "It is basically `max_tokens` in sglang."
                ),
            )
            parser.add_argument(
                "--rollout-skip-special-tokens",
                action="store_true",
                default=False,
                help=(
                    "Whether to skip special tokens in the response during rollout. "
                    "This is useful when you want to use the response as a prompt for the next rollout."
                ),
            )
            parser.add_argument(
                "--rollout-stop",
                type=str,
                nargs="+",
                default=None,
                help=(
                    "The stop words for the inference engine during rollout. "
                    "It can be a list of strings or a single string. "
                    "It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
                ),
            )
            parser.add_argument(
                "--rollout-stop-token-ids",
                type=int,
                nargs="+",
                default=None,
                help=(
                    "The stop token ids for the inference engine during rollout. "
                    "It can be a list of integers or a single integer."
                ),
            )
            parser.add_argument(
                "--rollout-shuffle",
                action="store_true",
                default=False,
                help=("Whether to shuffle the prompts during rollout."),
            )
            parser.add_argument(
                "--rollout-seed",
                type=int,
                default=42,
                help=(
                    "The seed for the random number generator during rollout. "
                    "This is used to shuffle the prompts and also for the random sampling of the prompts."
                ),
            )

            # sampling
            parser.add_argument(
                "--over-sampling-batch-size",
                type=int,
                default=None,
                help=(
                    "This defines the granularity of the sampling batch in the rollout function. "
                    "When the number of available samples falls below the target, a sampling "
                    "operation of size over_sampling_batch_size will be triggered."
                    "Regardless of whether partial rollout is used or filters are applied, "
                    "the sampling granularity is always determined by this value. "
                    "If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
                ),
            )
            parser.add_argument(
                "--dynamic-sampling-filter-path",
                type=str,
                default=None,
                help=(
                    "This is the filter function for dynamic sampling. "
                    "It should be able to judge whether the result of a prompt should be selected or not."
                    "We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples."
                    "You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
                ),
            )
            parser.add_argument(
                "--fully-async-filter-relax-after-groups",
                type=int,
                default=0,
                help=(
                    "For fully-async rollout, accept dynamically filtered groups after this many "
                    "completed groups. 0 disables relaxation."
                ),
            )
            parser.add_argument(
                "--rollout-infra-retry-times",
                type=int,
                default=2,
                help="Number of replacements for each logical rollout slot after retryable infrastructure failures.",
            )
            parser.add_argument(
                "--rollout-task-family-quotas",
                type=str,
                default=None,
                help=(
                    "Optional comma-separated task family quotas for rollout batches, "
                    "for example `webqa=0.4,mcp=0.4,cli=0.2`. Applied after dynamic "
                    "filtering by both synchronous and fully-async rollout collectors."
                ),
            )
            parser.add_argument(
                "--rollout-task-family-admission-only",
                action="store_true",
                default=False,
                help=(
                    "For synchronous rollout, apply --rollout-task-family-quotas only when admitting prompts. "
                    "Stop after rollout_batch_size accepted groups without enforcing family quotas after filtering."
                ),
            )
            parser.add_argument(
                "--rollout-task-family-top-mean-steps",
                action="store_true",
                default=False,
                help=(
                    "Within each post-filter task-family quota, select groups with the highest "
                    "mean fused trajectory step count."
                ),
            )

            # partial rollout
            parser.add_argument(
                "--partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to use partial rollout. "
                    "If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. "
                    "This is useful for long responses."
                ),
            )
            parser.add_argument(
                "--mask-offpolicy-in-partial-rollout",
                action="store_true",
                default=False,
                help=(
                    "Whether to mask previous generation in partial rollout. "
                    "If set, only on-policy generated tokens will be used in training"
                ),
            )
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
            parser.add_argument(
                "--custom-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging rollout data. The signature of the functions is: "
                    "def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )
            parser.add_argument(
                "--custom-eval-rollout-log-function-path",
                type=str,
                default=None,
                help=(
                    "The custom function for logging eval rollout data. "
                    "def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool. "
                    "The return value indicates whether to skip the default logging. "
                ),
            )

            parser.add_argument(
                "--buffer-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the buffer filter function. "
                    "It should be able to select the samples in the buffer. "
                    "The function should take list[list[Sample]] and return list[list[Sample]]."
                ),
            )
            parser.add_argument(
                "--enable-quota-bucket-sampling",
                action="store_true",
                default=False,
                help=(
                    "Enable quota-based bucket sampling in the rollout buffer. "
                    "The stock fused-agent buffer filter reserves fixed quotas for short/mid/long "
                    "trajectories by fused trajectory steps."
                ),
            )
            # update weight
            parser.add_argument(
                "--update-weight-buffer-size",
                type=int,
                default=512 * 1024**2,
                help=(
                    "buffer size for update weight, in bytes. "
                    "This is used for updating weights by chunk and should be useful for MoE models."
                ),
            )
            parser.add_argument(
                "--update-weights-interval",
                type=int,
                default=1,
                help="Push actor weights to rollout every N completed training rounds",
            )
            parser.add_argument(
                "--verify-rollout-weight-versions",
                action="store_true",
                default=False,
                help="After each weight update, require every SGLang engine to report the new version",
            )
            parser.add_argument(
                "--keep-old-actor",
                action="store_true",
                help="Whether to keep the rollout model on training process",
            )

            parser.add_argument(
                "--rollout-data-postprocess-path",
                type=str,
                default=None,
                help=(
                    "The called after we have all the rollout data including log_probs. "
                    "It may be helpful for updating loss mask."
                ),
            )
            parser.add_argument(
                "--rollout-data-transport",
                type=str,
                choices=["object-store", "nixl"],
                default="object-store",
                help=(
                    "Transport for rollout data refs sent from rollout manager to trainer. Large rollout "
                    "fields are tensorized on CPU before the refs are stored. Set to nixl to transfer "
                    "those torch tensors via Ray NIXL."
                ),
            )
            parser.add_argument(
                "--rollout-external-engine-addrs",
                type=str,
                default=None,
                nargs="+",
                help="Address and ports of the external engines.",
            )
            return parser

        def add_fault_tolerance_arguments(parser):
            parser.add_argument(
                "--use-fault-tolerance",
                action="store_true",
                default=False,
                help="Whether to enable the fault tolerance function during rollout.",
            )
            parser.add_argument(
                "--rollout-health-check-interval",
                type=float,
                default=30.0,
                help="Interval in seconds between rollout engine /health_generate checks during generate/eval.",
            )
            parser.add_argument(
                "--rollout-health-check-timeout",
                type=float,
                default=30.0,
                help="Timeout in seconds to wait for a rollout engine /health_generate response before killing it.",
            )
            parser.add_argument(
                "--rollout-health-check-failure-threshold",
                type=int,
                default=3,
                help="Consecutive HTTP 503 health checks tolerated before marking a rollout engine dead.",
            )
            parser.add_argument(
                "--rollout-generation-control-timeout",
                type=float,
                default=300.0,
                help=(
                    "Timeout in seconds for rollout engine lifecycle control requests such as "
                    "/pause_generation and /continue_generation during weight updates."
                ),
            )
            parser.add_argument(
                "--rollout-health-check-first-wait",
                type=float,
                default=0,
                help="Initial grace period (in seconds) before starting health checks. This allows time for model compilation and initialization. Increase this value significantly when using deepgemm.",
            )
            return parser

        # data
        def add_data_arguments(parser):
            # dataset
            # TODO: maybe add an num_epoch and calculate the num_rollout from buffer
            parser.add_argument(
                "--num-rollout",
                type=int,
                default=None,
                help="Number of rollout steps. If not set, we will calculate the number of rollout steps from the dataset size.",
            )
            parser.add_argument(
                "--num-epoch",
                type=int,
                default=None,
                help=(
                    "Number of epochs for the training. "
                    "This is used to calculate the number of rollout steps from the dataset size. "
                    "If set, we will calculate the number of rollout steps as `num_rollout = num_epoch * dataset_size // rollout_batch_size`."
                    "If both `--num-epoch` and `--num-rollout` are set, `--num-epoch` will be ignored."
                ),
            )

            parser.add_argument(
                "--disable-rollout-global-dataset",
                action="store_false",
                dest="rollout_global_dataset",
                help=(
                    "Whether to use a global dataset for rollout. "
                    "If set, the rollout will use the `--prompt-data` as the prompt dataset, "
                    "and the prompts for rollout will be sampled from the dataset. "
                    "If not set, you need to manage the data by your self."
                ),
            )

            parser.add_argument(
                "--data-source-path",
                type=str,
                default="slime.rollout.data_source.RolloutDataSourceWithBuffer",
                help="The data source class for rollout data.",
            )
            parser.add_argument(
                "--prompt-data",
                type=str,
                default=None,
                help=(
                    "The path to the prompt data. "
                    "Currently we only support jsonl format, and each line should contains --input-key and --label-key, "
                    "which will be used as the prompt and the label respectively. "
                    "If you want to use a custom template, you can set --apply-chat-template to true, in that case, "
                    "the input should be the same structure as an openai message, e.g. [{'role': 'user', 'content': 'blabla'}]. "
                ),
            )
            parser.add_argument("--apply-chat-template", action="store_true", default=False)
            # Temporarily be JSON-serialized str, will be a real dict after using Omegaconf
            parser.add_argument("--apply-chat-template-kwargs", type=json.loads, default="{}")
            parser.add_argument("--input-key", type=str, default="input", help="JSON dataset key")
            parser.add_argument("--label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--multimodal-keys",
                type=json.loads,
                default=None,
                help=(
                    'JSON string for multimodal data mapping media types to data keys. Example: \'{"image": "image_file"}\''
                ),
            )
            parser.add_argument("--metadata-key", type=str, default="metadata", help="JSON dataset key")
            parser.add_argument(
                "--tool-key",
                type=str,
                default="tools",
                help=(
                    "When need to add tools during apply_chat_template, you should provide the key for the tools in the prompt dataset."
                ),
            )

            parser.add_argument(
                "--start-rollout-id",
                type=int,
                default=None,
                help=(
                    "The starting rollout step, if not set, will try to load the step from --load when doing continue training, "
                    "otherwise will be set to 0, meaning training from start."
                ),
            )

            # batch sizes
            parser.add_argument(
                "--rollout-batch-size",
                type=int,
                required=True,
                help=(
                    "The number of prompts in each rollout step. "
                    "The total data returned should be rollout_batch_size * n_samples_per_prompt. "
                ),
            )
            parser.add_argument(
                "--n-samples-per-prompt", type=int, default=1, help="Number of responses for each prompt in generation"
            )

            # gbs of the training, note that the gbs is of sample, not of prompts,
            # so if you hope to train 1 step for each rollout, the global_bach_size should be set as
            # `rollout_batch_size * n_samples_per_prompt`.
            reset_arg(parser, "--global-batch-size", type=int, default=None)
            parser.add_argument(
                "--num-steps-per-rollout",
                type=int,
                default=None,
                help=(
                    "Number of steps per rollout, e.g. It is equivalent to setting gbs as "
                    "`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout`."
                ),
            )
            # mbs for the training, will be ignored if `use_dynamic_batch_size` is set.
            reset_arg(parser, "--micro-batch-size", type=int, default=1)
            parser.add_argument(
                "--balance-data",
                action="store_true",
                default=False,
                help=(
                    "Balance estimated training FLOPs between data parallel ranks with `karmarkar_karp`. "
                    "Micro-batch packing still follows the configured static/dynamic batching unless "
                    "`--balance-by-flops` is also set. "
                    "Note that this may allocate the different response of the same prompt into different training steps."
                ),
            )

            parser.add_argument(
                "--balance-by-flops",
                action="store_true",
                default=False,
                help=(
                    "Use FLOPs-based workload estimation (coeff*L + L²) for micro-batch "
                    "partitioning via Karmarkar-Karp instead of first-fit token packing. "
                    "The linear coefficient is auto-computed from model config (hidden_size, "
                    "ffn_hidden_size, swiglu, MoE experts). Captures the quadratic cost of "
                    "attention, producing more balanced micro-batches when sequence lengths "
                    "vary widely. This may create micro-batches whose total tokens exceed "
                    "--max-tokens-per-gpu and cause OOM. Also enables --balance-data. "
                    "Requires --use-dynamic-batch-size."
                ),
            )

            parser.add_argument(
                "--use-dynamic-batch-size",
                action="store_true",
                default=False,
                help=(
                    "Because the sample length varies, to maximize the GPU utilization, "
                    "we will use the dynamic batch size to adjust the micro batch size according to the maximum number of tokens each gpu can run. "
                    "For example, if we have 3 samples, with the length of 100, 200, and 300, and the max_tokens_per_gpu is 300, when enabling "
                    "dynamic batch size, slime will make 2 micro batches, i.e. [100, 200], [300]."
                ),
            )
            parser.add_argument(
                "--max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for dynamic batch size. "
                    "Note that when enabling context parallel (CP), the max tokens per gpu should be around "
                    "`max_response_len // cp_size` instead of `max_response_len`."
                ),
            )
            parser.add_argument(
                "--log-probs-max-tokens-per-gpu",
                type=int,
                default=None,
                help=(
                    "The maximum number of tokens per GPU for calculating log probs. "
                    "This is used to calculate the log probs of the responses during rollout, "
                    "and should be set to a larger value than `max_tokens_per_gpu` if you want better performance. "
                ),
            )
            return parser

        def add_eval_arguments(parser):
            parser.add_argument(
                "--eval-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the eval generation function."
                    "If not set, we will use rollout_function_path as the default. "
                ),
            )

            # change the default value of eval_interval from Megatron to None
            reset_arg(parser, "--eval-interval", type=int, default=None)

            parser.add_argument(
                "--eval-prompt-data",
                type=str,
                default=None,
                nargs="+",
                help=(
                    "Path to the evaluation prompt data, "
                    "should first input the name of the eval dataset and then the path, e.g. "
                    "aime /path/to/aime.jsonl"
                ),
            )
            parser.add_argument(
                "--eval-config",
                type=str,
                default=None,
                help=(
                    "Path to an OmegaConf YAML/JSON file describing evaluation datasets. "
                    "When provided, this overrides --eval-prompt-data."
                ),
            )
            parser.add_argument(
                "--skip-eval-before-train",
                action="store_true",
                default=False,
                help="Whether to skip evaluation before training.",
            )

            # The following keys are used to override the rollout version during eval.
            parser.add_argument("--eval-input-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-label-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument("--eval-tool-key", type=str, default=None, help="JSON dataset key")
            parser.add_argument(
                "--n-samples-per-eval-prompt",
                type=int,
                default=1,
                help="number of responses for each prompt in generation",
            )
            parser.add_argument("--eval-temperature", type=float, default=None)
            parser.add_argument("--eval-top-p", type=float, default=None)
            parser.add_argument("--eval-top-k", type=int, default=None)
            parser.add_argument("--eval-max-response-len", type=int, default=None)
            parser.add_argument("--eval-max-prompt-len", type=int, default=None)
            parser.add_argument("--eval-min-new-tokens", type=int, default=None)
            parser.add_argument("--eval-max-context-len", type=int, default=None)
            parser.add_argument(
                "--eval-max-inflight-tasks",
                type=int,
                default=384,
                help="Hard limit for the number of evaluation trajectories scheduled at once.",
            )
            parser.add_argument(
                "--eval-initial-inflight-tasks",
                type=int,
                default=None,
                help="Initial evaluation trajectory concurrency. Defaults to --eval-max-inflight-tasks.",
            )
            parser.add_argument(
                "--eval-adaptive-concurrency",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Adjust evaluation trajectory concurrency from SGLang engine load metrics.",
            )
            parser.add_argument(
                "--eval-mix-datasets",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Interleave samples from all eval datasets under one global inflight budget.",
            )
            parser.add_argument(
                "--eval-termination-retry-times",
                type=int,
                default=0,
                help=(
                    "Number of retries for an eval trajectory whose termination_reason is not a successful "
                    "termination (env_done or reasoning_only). The initial attempt is not included."
                ),
            )
            parser.add_argument(
                "--eval-concurrency-step",
                type=int,
                default=32,
                help="Number of evaluation trajectories added or removed by each adaptive adjustment.",
            )
            parser.add_argument(
                "--eval-concurrency-poll-interval",
                type=float,
                default=5.0,
                help="Seconds between SGLang engine metric samples for adaptive evaluation concurrency.",
            )
            parser.add_argument(
                "--eval-restart-sglang-server",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Restart local SGLang rollout servers with eval memory/concurrency overrides around interval eval.",
            )
            parser.add_argument("--eval-sglang-mem-fraction-static", type=float, default=None)
            parser.add_argument("--eval-sglang-server-concurrency", type=int, default=None)
            parser.add_argument("--eval-sglang-max-running-requests", type=int, default=None)

            return parser

        def add_algo_arguments(parser):
            parser.add_argument(
                "--ref-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for reference model. "
                    "When --load is not set, this will be used as the initial checkpoint for training. "
                ),
            )
            parser.add_argument(
                "--ref-ckpt-step", type=int, default=None, help="The checkpoint step for reference model. "
            )
            reset_arg(parser, "--load", type=str, default=None)
            reset_arg(parser, "--save", type=str, default=None)
            reset_arg(parser, "--save-interval", type=int, default=None)
            reset_arg(parser, "--async-save", action="store_true")
            reset_arg(
                parser,
                "--no-save-optim",
                action="store_true",
                default=False,
                help=(
                    "If set, do not save the optimizer state when saving checkpoints. "
                    "This reduces checkpoint size but disables training resumption from the saved checkpoint."
                ),
            )
            parser.add_argument(
                "--save-hf",
                type=str,
                default=None,
                help=(
                    "Path to save the model in HuggingFace format when using Megatron backend. "
                    "The model will be saved to `save_hf.format(rollout_id)`. "
                    "In raw Megatron-to-HF mode, weights are saved with the same quantization config "
                    "as `--hf-checkpoint`. "
                ),
            )
            reset_arg(parser, "--seed", type=int, default=1234)
            reset_arg(parser, "--clip-grad", type=float, default=1.0)
            reset_arg(parser, "--calculate-per-token-loss", action="store_true")
            reset_arg(parser, "--lr", type=float, default=1e-6)

            parser.add_argument(
                "--num-critic-only-steps",
                type=int,
                default=0,
                help="Number of initial rollout steps that train critic only; set >= num_rollout for critic-only runs",
            )
            parser.add_argument(
                "--megatron-config-path",
                type=str,
                default=None,
                help=(
                    "Path to a structured YAML config for Megatron roles. The file should use "
                    "a top-level 'megatron' key with role-tagged entries; the critic runtime will "
                    "select exactly one entry with role=critic. Legacy 'critic' configs are still accepted."
                ),
            )

            parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip range")
            parser.add_argument("--eps-clip-high", type=float, default=None, help="PPO clip upper range")
            parser.add_argument(
                "--eps-clip-c",
                type=float,
                default=None,
                help="lower bound of the value for Dual-clip PPO from https://arxiv.org/pdf/1912.09729",
            )
            parser.add_argument("--value-clip", type=float, default=0.2, help="the clip for value loss")
            parser.add_argument(
                "--kl-coef",
                type=float,
                default=0.00,
                help="KL penalty coefficient for reward shaping. This is applied to the reward signal before advantage calculation.",
            )
            parser.add_argument(
                "--loss-type",
                type=str,
                choices=["policy_loss", "sft_loss", "custom_loss"],
                default="policy_loss",
                help=(
                    "Choose loss type, currently support ppo policy_loss or sft_loss, "
                    "if custom_loss is set, we will use the function path from `--custom-loss-function-path`."
                ),
            )
            parser.add_argument(
                "--custom-loss-function-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom loss function, if the loss_type is `custom_loss`, "
                    "we will use this function to calculate the loss. "
                ),
            )
            parser.add_argument(
                "--kl-loss-type",
                type=str,
                choices=["k1", "k2", "k3", "low_var_kl"],
                default="k1",
                help="Choose KL loss type: kl, k2, k3, low_var_kl",
            )
            parser.add_argument(
                "--advantage-estimator",
                type=str,
                choices=[
                    "grpo",
                    "gspo",
                    "cispo",
                    "reinforce_plus_plus",
                    "reinforce_plus_plus_baseline",
                    "ppo",
                ],
                default="grpo",
                help=(
                    "Advantage estimator to use. Note: on-policy distillation (OPD) is now orthogonal "
                    "to the advantage estimator. Use --opd-kl-coef > 0 to enable OPD on top of any estimator."
                ),
            )
            parser.add_argument(
                "--disable-compute-advantages-and-returns",
                action="store_false",
                dest="compute_advantages_and_returns",
                help=(
                    "Whether to disable computing advantages and returns. "
                    "If set, we will not compute the advantages and returns, "
                    "This is useful for sft or custom loss function."
                ),
            )
            parser.add_argument(
                "--custom-advantage-function-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom advantage/returns computation function. "
                    "When set, this function replaces the built-in compute_advantages_and_returns. "
                    "Signature: def custom_fn(args, rollout_data) -> None. "
                    "The function should set rollout_data['advantages'] and rollout_data['returns'] in-place. "
                    "Critic values are available in rollout_data['values']. "
                    "(e.g., my_module.py:my_advantage_fn)."
                ),
            )
            parser.add_argument(
                "--srppo-mode",
                type=str,
                default="shrinkage-critic-1-gradient-1",
                choices=[
                    "shrinkage-critic-k-gradient-1",
                    "shrinkage-critic-1-gradient-1",
                    "shrinkage-critic-k-gradient-k",
                    "shrinkage-critic-1-gradient-k",
                    "raw",
                ],
                help="SR-PPO sequential-credit mode used by the custom advantage hook.",
            )
            parser.add_argument("--srppo-pass-at-k", type=float, default=4.0)
            parser.add_argument("--srppo-normalize-gradient-k", action="store_true")
            parser.add_argument(
                "--use-kl-loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
            )
            parser.add_argument(
                "--kl-loss-coef",
                type=float,
                default=0.0,
                help="KL penalty coefficient for the loss function. This is added to the final PPO loss.",
            )
            parser.add_argument(
                "--use-unbiased-kl",
                action="store_true",
                default=False,
                help="Whether to enable unbiased KL estimation.",
            )
            parser.add_argument(
                "--ref-update-interval",
                type=int,
                default=None,
                help="Interval (in rollout steps) to update ref model from actor. If None, ref model is not updated.",
            )
            parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy loss coef")
            parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
            parser.add_argument("--lambd", type=float, default=1.0, help="PPO GAE lambd")
            parser.add_argument("--normalize-advantages", action="store_true", default=False)
            parser.add_argument(
                "--disable-grpo-std-normalization",
                action="store_false",
                dest="grpo_std_normalization",
                help="from Dr.GRPO https://arxiv.org/pdf/2503.20783",
            )
            parser.add_argument(
                "--disable-rewards-normalization",
                action="store_false",
                dest="rewards_normalization",
                help="Disable rewards normalization",
            )
            parser.add_argument(
                "--use-rollout-entropy",
                action="store_true",
                default=False,
                help=(
                    "Whether to calculate the entropy when calculating the logprobs from actor and reference model. "
                    "This is useful for doing special loss mask."
                ),
            )
            parser.add_argument(
                "--get-mismatch-metrics",
                action="store_true",
                default=False,
                help="Whether to calculate the mismatch metrics.",
            )
            parser.add_argument(
                "--reset-optimizer-states",
                action="store_true",
                default=False,
                help=(
                    "Whether to reset optimizer states after each rollout. "
                    "If enabled, the optimizer's history will be cleared at the end of each rollout, which can sometimes help with training stability or fulfill specific experiment requirements."
                ),
            )
            parser.add_argument(
                "--use-stateless-adam",
                action="store_true",
                default=False,
                help=(
                    "Whether to use a stateless Adam optimizer that does not persist the first/second moment "
                    "estimates across steps. Requires --optimizer adam and --no-save-optim."
                ),
            )
            parser.add_argument(
                "--use-rollout-logprobs",
                action="store_true",
                default=False,
                help=(
                    "Whether to use the rollout logprobs when calculating the importance sampling ratios. "
                    "If not set, we will use the logprobs from the actor model."
                ),
            )
            # Off-Policy Correction using Importance Sampling: https://fengyao.notion.site/off-policy-rl
            parser.add_argument(
                "--use-tis",
                action="store_true",
                default=False,
                help="Enable TIS from https://fengyao.notion.site/off-policy-rl for off-policy importance sampling.",
            )
            parser.add_argument(
                "--tis-clip",
                type=float,
                default=2.0,
                help="Clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--tis-clip-low",
                type=float,
                default=0,
                help="Lower bound clipping threshold C for importance sampling ratios to control variance.",
            )
            parser.add_argument(
                "--custom-tis-function-path",
                type=str,
                default=None,
                help="Path to the custom TIS/RS function (e.g., examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp).",
            )
            parser.add_argument(
                "--custom-pg-loss-reducer-function-path",
                type=str,
                default=None,
                help="Path to a custom reducer function for pg_loss only. When set, pg_loss will use this custom reducer while other metrics (pg_clipfrac, ppo_kl, entropy_loss, etc.) still use the default sum_of_sample_mean.",
            )
            parser.add_argument(
                "--credit-assignment-enable",
                action="store_true",
                default=False,
                help="Enable metadata-driven partial credit assignment policy masks.",
            )
            parser.add_argument(
                "--credit-assignment-tool-parser-error",
                action="store_true",
                default=False,
                help="Mask policy-gradient tokens to the abnormal action span for tool parser errors.",
            )
            parser.add_argument(
                "--credit-assignment-repeated-search-query",
                action="store_true",
                default=False,
                help="Mask policy-gradient tokens to the abnormal action span for repeated search queries.",
            )
            parser.add_argument(
                "--credit-assignment-too-many-tool-calls",
                action="store_true",
                default=False,
                help="Mask policy-gradient tokens to the abnormal action span for excessive tool calls.",
            )
            parser.add_argument(
                "--credit-assignment-ngram-repetition",
                action="store_true",
                default=False,
                help="Mask policy-gradient tokens to the abnormal action span for turn-level n-gram repetition.",
            )
            parser.add_argument(
                "--credit-assignment-search-bypass",
                action="store_true",
                default=False,
                help="Keep full action-token policy mask for search-bypass penalties.",
            )
            parser.add_argument(
                "--credit-assignment-mixed-tool-and-answer",
                action="store_true",
                default=False,
                help="Mask policy-gradient tokens to the abnormal turn when a tool call and answer appear together.",
            )

            parser.add_argument(
                "--use-routing-replay",
                action="store_true",
                default=False,
                help="The routing replay technique from https://arxiv.org/abs/2507.18071",
            )
            parser.add_argument(
                "--use-rollout-routing-replay",
                action="store_true",
                default=False,
                help="The rollout routing replay technique from https://arxiv.org/abs/2510.11370",
            )
            parser.add_argument(
                "--use-opsm",
                action="store_true",
                default=False,
                help="Whether to enable Off-Policy Sequence Masking (OPSM).",
            )
            parser.add_argument(
                "--opsm-delta",
                type=float,
                default=1e-4,
                help="The threshold for Off-Policy Sequence Masking (OPSM).",
            )
            return parser

        def add_on_policy_distillation_arguments(parser):
            """Add on-policy distillation (OPD) related arguments.

            OPD is orthogonal to advantage estimators and can be applied on top of
            any estimator (GRPO, PPO, etc.) by adding a KL penalty to advantages.
            """
            parser.add_argument(
                "--use-opd",
                action="store_true",
                default=False,
                help="Enable on-policy distillation (OPD). Must specify --opd-type when enabled.",
            )
            parser.add_argument(
                "--opd-type",
                type=str,
                choices=["sglang", "megatron"],
                default=None,
                help=(
                    "Type of on-policy distillation. "
                    "'sglang': Teacher log-probs are obtained from external SGLang server during rollout. "
                    "'megatron': Teacher model is loaded via --opd-teacher-load and forwarded during training."
                ),
            )
            parser.add_argument(
                "--opd-kl-coef",
                type=float,
                default=1.0,
                help="On-policy distillation KL penalty coefficient. Default is 1.0.",
            )
            parser.add_argument(
                "--opd-teacher-load",
                type=str,
                default=None,
                help=(
                    "The checkpoint for OPD teacher model. Required when --opd-type=megatron. "
                    "The teacher model should have the same architecture as policy/ref model."
                ),
            )
            parser.add_argument(
                "--opd-teacher-ckpt-step", type=int, default=None, help="The checkpoint step for OPD teacher model."
            )
            return parser

        # wandb
        def add_wandb_arguments(parser):
            # wandb parameters
            parser.add_argument("--use-wandb", action="store_true", default=False)
            parser.add_argument(
                "--wandb-mode",
                type=str,
                default=None,
                choices=["online", "offline", "disabled"],
                help="W&B mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var.",
            )
            parser.add_argument(
                "--wandb-dir",
                type=str,
                default=None,
                help="Directory to store wandb logs. Default is ./wandb in current directory.",
            )
            parser.add_argument("--wandb-key", type=str, default=None)
            parser.add_argument("--wandb-host", type=str, default=None)
            parser.add_argument("--wandb-team", type=str, default=None)
            parser.add_argument("--wandb-group", type=str, default=None)
            reset_arg(parser, "--wandb-project", type=str, default=None)
            parser.add_argument(
                "--disable-wandb-random-suffix",
                action="store_false",
                dest="wandb_random_suffix",
                default=True,
                help=(
                    "Whether to add a random suffix to the wandb run name. "
                    "By default, we will add a random 6 length string with characters to the run name."
                ),
            )
            parser.add_argument(
                "--wandb-always-use-train-step",
                action="store_true",
                default=False,
                help=(
                    "Whether the legacy rollout/step and eval/step fields should mirror train/step. "
                    "W&B charts use train/step as their step metric regardless of this option."
                ),
            )
            parser.add_argument(
                "--log-multi-turn",
                action="store_true",
                default=False,
                help="Whether to log information for multi-turn rollout.",
            )
            parser.add_argument(
                "--log-passrate",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument(
                "--show-sglang-server-logs",
                action="store_true",
                default=False,
                help=(
                    "Show SGLang engine/server INFO logs in the console, including Prefill/Decode batch lines "
                    'and HTTP "POST /generate" access logs. Disabled by default to keep training logs readable.'
                ),
            )
            parser.add_argument(
                "--print-rollout-trajectory",
                action="store_true",
                default=False,
                help=(
                    "Print one trajectory from each accepted rollout group. "
                    "The response is split into masked and unmasked action text using loss_mask."
                ),
            )
            parser.add_argument(
                "--print-rollout-trajectory-width",
                type=int,
                default=88,
                help=(
                    "Maximum terminal columns used by rollout trajectory visualization. "
                    "The effective width is capped by the current terminal width after reserving prefix columns."
                ),
            )
            parser.add_argument(
                "--print-rollout-trajectory-prefix-width",
                type=int,
                default=32,
                help=(
                    "Terminal columns reserved for log prefixes such as Ray actor pid labels before drawing rollout "
                    "trajectory boxes. Increase this when large fonts or long log prefixes wrap box borders."
                ),
            )
            parser.add_argument(
                "--no-print-rollout-trajectory-color",
                dest="print_rollout_trajectory_color",
                action="store_false",
                default=True,
                help="Disable ANSI color output for rollout trajectory visualization.",
            )
            parser.add_argument(
                "--print-train-metrics-table",
                action="store_true",
                default=False,
                help="Print actor train metrics as a table after each actor update.",
            )
            parser.add_argument(
                "--log-reward-category",
                type=str,
                default=None,
                help=(
                    "Log statistics of the category of reward, such as why the reward function considers it as failed. "
                    "Specify the key in the reward dict using this argument."
                ),
            )
            parser.add_argument(
                "--log-correct-samples",
                action="store_true",
                default=False,
                help="Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout.",
            )
            parser.add_argument("--wandb-run-id", type=str, default=None)
            parser.add_argument(
                "--wandb-skip-resume-first-step",
                action="store_true",
                default=False,
                help=(
                    "Skip all W&B metrics associated with the first rollout after resuming an existing run. "
                    "Requires --wandb-run-id. TensorBoard logging is unaffected."
                ),
            )
            return parser

        # tensorboard
        def add_tensorboard_arguments(parser):
            # tb_project_name, tb_experiment_name
            parser.add_argument("--use-tensorboard", action="store_true", default=False)
            parser.add_argument(
                "--tb-project-name",
                type=str,
                default=None,
                help="Directory to store tensorboard logs. Default is  os.environ.get('TENSORBOARD_DIR') directory.",
            )
            parser.add_argument("--tb-experiment-name", type=str, default=None)

            return parser

        # debug
        def add_debug_arguments(parser):
            parser.add_argument(
                "--save-debug-rollout-data",
                type=str,
                default=None,
                help=(
                    "Save the rollout data to this path for debugging. "
                    "The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
                ),
            )
            # --load-debug-rollout-data, --debug-rollout-only, --debug-train-only
            # are parsed early in _pre_parse_mode() and merged later.
            parser.add_argument(
                "--load-forge-rollout-data",
                type=str,
                default=None,
                help=(
                    "Path (or {rollout_id} template) to a dumped rollout .pt file replayed by "
                    "slime.rollout.forge_load.generate_rollout. Mirrors --load-debug-rollout-data's "
                    "format(rollout_id=...) convention: a path without the placeholder is treated as "
                    "a literal file and reused across every rollout_id; a path containing {rollout_id} "
                    "loads a per-rollout file (with eval_<id>.pt for the eval pipeline). Unlike "
                    "--load-debug-rollout-data, this does NOT force debug_train_only / skip_sglang -- "
                    "sglang servers, router, weight_update and the colocate offload/onload dance all "
                    "stay live, which is the point (memory measurement at long context)."
                ),
            )
            parser.add_argument(
                "--replay-and-generate",
                action="store_true",
                default=False,
                help=(
                    "Use the forge rollout dump for the first replay step, then run a real SGLang rollout "
                    "after the actor weight update. Requires --load-forge-rollout-data and an explicit start id."
                ),
            )
            parser.add_argument(
                "--load-debug-rollout-data-subsample",
                type=float,
                default=None,
                help="Subsample a portion of the debug rollout data for faster debugging.",
            )
            parser.add_argument(
                "--rollout-only-inference-fast-path",
                action="store_true",
                default=False,
                help=(
                    "In --debug-rollout-only mode, emit lightweight inference samples without requesting "
                    "rollout logprobs or building trainable trajectory segments."
                ),
            )
            parser.add_argument(
                "--rollout-only-skip-episode-dump",
                action="store_true",
                default=False,
                help=(
                    "In --debug-rollout-only mode, skip the duplicate rLLM episode shard write. "
                    "Use this when a custom offline pipeline consumes --save-debug-rollout-data instead."
                ),
            )
            parser.add_argument(
                "--async-save-debug-rollout-data",
                action="store_true",
                default=False,
                help="Write debug rollout .pt files in a background thread so the next rollout can start immediately.",
            )
            parser.add_argument(
                "--save-debug-train-data",
                type=str,
                default=None,
                help=(
                    "Save the train data to this path for debugging. "
                    "The file will be saved to `save_debug_train_data.format(rollout_id)`."
                ),
            )
            parser.add_argument(
                "--dump-details",
                type=str,
                default=None,
                help=("Dump all details of training for post-hoc analysis and visualization."),
            )
            # use together with --record-memory-history and --memory-snapshot-path (defined in Megatron)
            parser.add_argument(
                "--memory-snapshot-dir",
                type=str,
                default=".",
            )
            parser.add_argument(
                "--memory-snapshot-num-steps",
                type=int,
                default=None,
            )
            parser.add_argument(
                "--profile-target",
                type=str,
                choices=["train_overall", "train_actor", "train_log_probs"],
                default=["train_overall"],
                nargs="+",
            )
            parser.add_argument(
                "--memory-recorder",
                type=str,
                choices=["torch", "memray"],
                default="torch",
            )
            reset_arg(parser, "--record-memory-history", action="store_true", default=False)
            parser.add_argument("--check-weight-update-equal", action="store_true")
            return parser

        def add_network_arguments(parser):
            parser.add_argument("--http-proxy", type=str, default=None)
            parser.add_argument("--use-distributed-post", action="store_true", default=False)
            return parser

        def add_reward_model_arguments(parser):
            parser.add_argument(
                "--rm-type",
                type=str,
                default=None,
                help="Type of the reward model",
            )
            parser.add_argument(
                "--reward-key",
                type=str,
                default=None,
                help=(
                    "Some reward model may return a dict instead of a value, "
                    "this is the key to extract the reward value from the dict. "
                ),
            )
            parser.add_argument(
                "--eval-reward-key",
                type=str,
                default=None,
                help="The eval variant for --reward-key",
            )
            parser.add_argument(
                "--group-rm", action="store_true", default=False, help="Whether to do rm on a whole group."
            )
            parser.add_argument(
                "--rm-url",
                type=str,
                default=None,
                help="URL for the reward model service for --rm-type remote_rm, e.g. http://localhost:8000",
            )
            parser.add_argument(
                "--custom-rm-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom reward model function. "
                    "If set, we will use this function to calculate the reward instead of the default one. "
                    "The function should have the signature `def custom_rm(args, sample) -> float`."
                ),
            )
            parser.add_argument(
                "--enable-use-grm-train",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Whether WebQA training rollouts should use OpenRouter GRM with rule-based fallback. "
                    "Other task families retain their existing rewards."
                ),
            )
            parser.add_argument(
                "--enable-use-grm-evals",
                action=argparse.BooleanOptionalAction,
                default=False,
                help=(
                    "Whether WebQA interval eval rollouts should use OpenRouter GRM before rule-based fallback. "
                    "Other task families retain their dataset verifiers."
                ),
            )
            parser.add_argument(
                "--grm-custom-rm-path",
                type=str,
                default="slime.rollout.rm_hub.openrouter_grm.reward_func",
                help="Custom RM path used when GRM train or eval scoring is enabled.",
            )
            parser.add_argument("--grm-base-url", type=str, default="https://openrouter.ai/api/v1")
            parser.add_argument("--grm-model", type=str, default="deepseek/deepseek-v4-flash")
            parser.add_argument(
                "--train-grm-model",
                type=str,
                default=None,
                help="Training GRM model. Defaults to --grm-model for compatibility.",
            )
            parser.add_argument(
                "--eval-grm-model",
                type=str,
                default=None,
                help="Evaluation GRM model. Defaults to --grm-model for compatibility.",
            )
            parser.add_argument(
                "--grm-mode",
                type=str,
                choices=("score", "equivalence"),
                default="score",
                help="GRM protocol: legacy binary score JSON or answer-equivalence judgement JSON.",
            )
            parser.add_argument("--grm-openrouter-api-key", type=str, default=None)
            parser.add_argument("--grm-openrouter-site-url", type=str, default=None)
            parser.add_argument("--grm-openrouter-app-name", type=str, default="GRM")
            parser.add_argument("--grm-concurrency", type=int, default=128)
            parser.add_argument("--grm-max-connections", type=int, default=128)
            parser.add_argument("--grm-timeout", type=float, default=60.0)
            parser.add_argument("--grm-max-retries", type=int, default=3)
            parser.add_argument("--grm-retry-base-delay", type=float, default=0.5)
            parser.add_argument("--grm-retry-max-delay", type=float, default=8.0)
            parser.add_argument("--grm-max-input-tokens", type=int, default=24000)
            parser.add_argument("--grm-max-new-tokens", type=int, default=128)
            parser.add_argument("--grm-temperature", type=float, default=0.0)
            parser.add_argument("--grm-failure-reward", type=float, default=0.0)
            parser.add_argument("--grm-system-prompt", type=str, default=None)
            parser.add_argument(
                "--custom-reward-post-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the custom function that will post process reward, by default it will be the normalization for grpo. "
                ),
            )
            parser.add_argument(
                "--custom-convert-samples-to-train-data-path",
                type=str,
                default=None,
                help=(
                    "Path to a custom function that converts samples to training data. "
                    "If set, this function will replace the default _convert_samples_to_train_data. "
                    "The function should have the signature `def convert_samples_to_train_data(args, samples) -> dict`."
                ),
            )
            return parser

        def add_rollout_buffer_arguments(parser):
            parser.add_argument(
                "--rollout-buffer-url",
                type=str,
                default=None,
                help="URL for the rollout buffer",
            )

            parser.add_argument(
                "--fetch-trajectory-retry-times",
                type=int,
                default=-1,
                help="Number of times to retry fetching trajectory, -1 means unlimited retry",
            )
            parser.add_argument(
                "--min-batch-collection-ratio",
                type=float,
                default=1,
                help="Minimum batch collection ratio",
            )
            parser.add_argument(
                "--rollout-task-type",
                type=str,
                default="math",
            )
            parser.add_argument(
                "--loss-mask-type",
                type=str,
                default="qwen",
                choices=["qwen", "qwen3", "qwen3_5", "gemma4", "distill_qwen"],
                help="Loss mask type",
            )
            parser.add_argument(
                "--data-pad-size-multiplier",
                type=int,
                default=128,
                help="Multiplier for data padding size in data processing.",
            )
            parser.add_argument(
                "--rollout-sample-filter-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout sample filter function. "
                    "This function determines whether a sample will participate in loss calculation. "
                    "The function should take args and samples (list[Sample]) as input, and return None. "
                    "Please directly modify the remove_sample attribute of Sample. "
                    "Note: This attribute does not determine whether the sample participates in advantage normalization."
                ),
            )
            parser.add_argument(
                "--rollout-all-samples-process-path",
                type=str,
                default=None,
                help=(
                    "Path to the rollout all samples process function that "
                    "can process all samples including filtered ones."
                ),
            )
            return parser

        def add_custom_megatron_plugins_arguments(parser):
            """
            Add custom Megatron plugins arguments.
            This is a placeholder for any additional arguments that might be needed.
            """
            # Custom arguments can be added here
            parser.add_argument(
                "--custom-megatron-init-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-log-prob-hook-path",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--custom-megatron-before-train-step-hook-path",
                type=str,
                default=None,
            )
            return parser

        def add_mtp_training_arguments(parser):
            """Add MTP training specific arguments."""
            reset_arg(parser, "--mtp-num-layers", type=int, default=None)
            reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
            parser.add_argument(
                "--enable-mtp-training",
                action="store_true",
                default=False,
                help="Enable MTP layer parameter updates during training",
            )

            return parser

        def add_ci_arguments(parser):
            parser.add_argument(
                "--ci-test",
                action="store_true",
            )
            parser.add_argument(
                "--ci-disable-kl-checker",
                action="store_true",
            )
            parser.add_argument(
                "--ci-save-grad-norm",
                type=str,
                default=None,
            )
            parser.add_argument(
                "--ci-load-grad-norm",
                type=str,
                default=None,
            )
            return parser

        # Add custom arguments in front to prevent overwritten some slime arguments.
        if add_custom_arguments is not None:
            parser = add_custom_arguments(parser)

        parser = add_cluster_arguments(parser)
        parser = add_train_arguments(parser)
        parser = add_rollout_arguments(parser)
        parser = add_fault_tolerance_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_eval_arguments(parser)
        parser = add_algo_arguments(parser)
        parser = add_on_policy_distillation_arguments(parser)
        parser = add_wandb_arguments(parser)
        parser = add_tensorboard_arguments(parser)
        parser = add_debug_arguments(parser)
        parser = add_network_arguments(parser)
        parser = add_reward_model_arguments(parser)
        parser = add_rollout_buffer_arguments(parser)
        parser = add_mtp_training_arguments(parser)
        parser = add_ci_arguments(parser)
        parser = add_custom_megatron_plugins_arguments(parser)
        reset_arg(
            parser,
            "--custom-config-path",
            type=str,
            default=None,
            help="Path to the YAML config for custom function arguments.",
        )
        reset_arg(parser, "--padded-vocab-size", type=int, default=None)

        return parser

    return add_slime_arguments


def _pre_parse_mode():
    """Pre-parse CLI to extract arguments that control parsing flow.

    These arguments are removed from add_slime_arguments to avoid
    registering them twice.  The returned namespace is merged into
    the final ``args`` after Phase 2 parsing.
    """
    temp_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    temp_parser.add_argument("--train-backend", type=str, choices=["megatron"], default="megatron")
    temp_parser.add_argument("--debug-rollout-only", action="store_true", default=False)
    temp_parser.add_argument("--debug-train-only", action="store_true", default=False)
    temp_parser.add_argument("--load-debug-rollout-data", type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    return temp_args


def _apply_yarn_sglang_override(args):
    if not getattr(args, "use_yarn_rope", False):
        return

    try:
        model_overrides = json.loads(args.sglang_json_model_override_args)
    except json.JSONDecodeError as exc:
        raise ValueError("--sglang-json-model-override-args must be valid JSON") from exc
    if not isinstance(model_overrides, dict):
        raise ValueError("--sglang-json-model-override-args must contain a JSON object")
    context_length = int(args.sglang_context_length)
    if context_length <= 0:
        raise ValueError("--sglang-context-length must be positive when YaRN is enabled")
    model_overrides["max_position_embeddings"] = context_length
    model_overrides["rope_scaling"] = {
        "rope_type": "yarn",
        "factor": args.yarn_rope_scaling_factor,
        "original_max_position_embeddings": args.yarn_original_max_position_embeddings,
    }
    args.sglang_json_model_override_args = json.dumps(model_overrides, separators=(",", ":"))


def parse_args(add_custom_arguments=None):
    # Users may call `parse_args` very early, thus we ensure logger is configured here
    configure_logger()
    register_gemma4_config_aliases()

    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    # Phase 1: Parse sglang args independently (separate parser, parse_known_args).
    # Skipped when sglang servers are not needed.
    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    # Phase 2: Parse megatron + slime args.
    # Uses ignore_unknown_args=True so that --sglang-* and pre-parsed CLI flags
    # are silently ignored by the megatron parser.
    from slime.backends.megatron_utils.arguments import megatron_parse_args
    from slime.backends.megatron_utils.arguments import validate_args as megatron_validate_args

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        skip_hf_validate=pre.debug_rollout_only,
    )

    # Merge pre-parsed args into the main namespace
    for key, value in vars(pre).items():
        setattr(args, key, value)

    # Merge sglang args into the main namespace
    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)
        _apply_yarn_sglang_override(args)

    slime_validate_args(args)

    if pre.train_backend == "megatron" and not args.debug_rollout_only:
        megatron_validate_args(args)

    if not args.debug_train_only:
        sglang_validate_args(args)

    return args


def _apply_megatron_role_overrides(base_args, overrides, role):
    role_args = copy.deepcopy(base_args)
    ignored_keys = {"num_nodes", "num_gpus_per_node"}

    # Apply overrides from the YAML config.
    # Unspecified keys inherit from base_args via deepcopy.
    for key, value in overrides.items():
        if key in ignored_keys:
            logger.info(f"Ignoring {role} config key '{key}'; GPU allocation always follows CLI args.")
            continue
        if not hasattr(role_args, key):
            logger.warning(f"{role.capitalize()} config key '{key}' is not a known argument, setting it anyway.")
        else:
            # YAML safe_load doesn't parse scientific notation (e.g. 1e-5) as float.
            # Coerce the value to match the type of the existing attribute.
            original = getattr(role_args, key)
            if original is not None and isinstance(value, str) and isinstance(original, (int, float)):
                try:
                    value = type(original)(value)
                except (ValueError, TypeError):
                    pass
        setattr(role_args, key, value)

    if role == "critic":
        # Critic-specific: disable features that only apply to actors.
        role_args.kl_coef = 0
        role_args.use_opd = False
        role_args.custom_advantage_function_path = None
        role_args.untie_embeddings_and_output_weights = True
        if "disable_param_buffers_cpu_backup" not in overrides:
            role_args.disable_param_buffers_cpu_backup = False

    return role_args


def parse_megatron_role_args(base_args, megatron_config_path, role):
    """Parse role-specific arguments from a unified Megatron YAML config.

    The config must contain a top-level ``megatron`` list with per-role entries.
    Missing roles inherit the base args unchanged.
    """
    assert role in {"actor", "critic"}, f"Unsupported Megatron config role: {role}"

    with open(megatron_config_path) as f:
        raw_config = yaml.safe_load(f) or {}

    assert "megatron" in raw_config, (
        "megatron config must contain a top-level 'megatron' list, e.g. "
        "megatron: [{name: default, role: actor, overrides: {...}}]"
    )

    overrides = {}
    megatron_entries = raw_config["megatron"]
    assert isinstance(megatron_entries, list), (
        "megatron config 'megatron' field must be a list, e.g. "
        "megatron: [{name: default, role: actor, overrides: {...}}]"
    )
    role_entries = [entry for entry in megatron_entries if entry.get("role") == role]
    assert len(role_entries) <= 1, (
        f"megatron config must contain at most one entry with role={role}, e.g. "
        f"megatron: [{{name: default, role: {role}, overrides: {{...}}}}]"
    )
    if role_entries:
        role_entry = role_entries[0]
        overrides = role_entry.get("overrides") or role_entry.get("args") or {}
    else:
        logger.info(
            f"No megatron config entry with role={role} found in {megatron_config_path}; using inherited args."
        )

    role_args = _apply_megatron_role_overrides(base_args, overrides, role)
    logger.info(
        f"Parsed megatron config for role={role} from {megatron_config_path}: overrides = {list(overrides.keys())}"
    )

    return role_args


def _resolve_eval_datasets(args) -> list[EvalDatasetConfig]:
    """
    Build evaluation dataset configurations from either --eval-config or --eval-prompt-data.
    """
    datasets_config = []
    defaults: dict[str, Any] = {}

    if args.eval_config:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.eval_config)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise ValueError("--eval-config must contain a mapping at the root.")

        eval_cfg = cfg_dict.get("eval", cfg_dict)
        if not isinstance(eval_cfg, dict):
            raise ValueError("--eval-config must define an `eval` mapping or be a mapping itself.")

        defaults = dict(eval_cfg.get("defaults") or {})
        datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
        if not datasets_config:
            raise ValueError("--eval-config does not define any datasets under `eval.datasets`.")
    elif args.eval_prompt_data:
        values = list(args.eval_prompt_data)
        if len(values) == 1:
            logger.info("[legacy] only one eval_prompt_data detected, will assume it is data for aime")
            values = ["aime", values[0]]
        if len(values) % 2 != 0:
            raise ValueError("eval prompt data must be provided as name/path pairs.")
        datasets_config = [{"name": values[i], "path": values[i + 1]} for i in range(0, len(values), 2)]
    else:
        datasets_config = []

    eval_datasets = build_eval_dataset_configs(args, datasets_config, defaults)
    if eval_datasets:
        args.eval_prompt_data = [item for dataset in eval_datasets for item in (dataset.name, dataset.path)]
    else:
        args.eval_prompt_data = None

    return eval_datasets


def slime_validate_args(args):
    args.eval_datasets = _resolve_eval_datasets(args)

    if args.recompute_logprobs_via_prefill and not args.true_on_policy_mode:
        raise ValueError("--recompute-logprobs-via-prefill requires --true-on-policy-mode")
    if args.true_on_policy_mode:
        if getattr(args, "fully_async", False):
            raise ValueError("--true-on-policy-mode is only supported by synchronous rollout")
        if getattr(args, "partial_rollout", False):
            raise ValueError("--true-on-policy-mode does not support partial rollout")
        if not getattr(args, "bf16", False):
            raise ValueError("Qwen3 true-on-policy currently requires --bf16")
        if args.tensor_model_parallel_size != args.rollout_num_gpus_per_engine:
            raise ValueError(
                "--true-on-policy-mode requires matching training and rollout tensor parallel sizes; "
                f"got {args.tensor_model_parallel_size} and {args.rollout_num_gpus_per_engine}"
            )
        if getattr(args, "num_steps_per_rollout", None) not in (None, 1):
            raise ValueError("--true-on-policy-mode requires --num-steps-per-rollout 1")
        if getattr(args, "update_weights_interval", 1) != 1:
            raise ValueError("--true-on-policy-mode requires --update-weights-interval 1")
        incompatible_sampling = []
        if args.rollout_top_p != 1.0:
            incompatible_sampling.append("--rollout-top-p must be 1.0")
        if args.rollout_top_k != -1:
            incompatible_sampling.append("--rollout-top-k must be -1")
        if args.rollout_min_p != 0.0:
            incompatible_sampling.append("--rollout-min-p must be 0.0")
        if args.rollout_presence_penalty != 0.0:
            incompatible_sampling.append("--rollout-presence-penalty must be 0.0")
        if args.rollout_repetition_penalty != 1.0:
            incompatible_sampling.append("--rollout-repetition-penalty must be 1.0")
        if incompatible_sampling:
            raise ValueError(
                "--true-on-policy-mode requires unmodified sampling logits: "
                + ", ".join(incompatible_sampling)
            )
        # Keep the internal mode self-contained. Launchers expose only their
        # public --true-on-policy switch; these are the cross-engine contract.
        # A100 TP4, CUDA graphs off: raw decode matches clean prefill, including ~15K contexts.
        # args.recompute_logprobs_via_prefill = True  # Keep recomputation opt-in, not forced by TOP.
        # args.batch_invariant_mode = True  # The SGLang backend enables the required kernels itself.
        # args.deterministic_mode = True  # Disabling this AND global torch determinism preserved exact parity.
        args.sequence_parallel = False
        args.true_on_policy_contract = "qwen3_dense_true_on_policy_v1"
        args.transformer_impl = "local"
        # args.fp32_residual_connection = False  # The retained SGLang residual-pair contract owns this math.
        args.bias_swiglu_fusion = False
        args.apply_rope_fusion = False
        # args.use_cpu_initialization = True  # Only RoPE frequencies need CPU math; GPTModel retains that below.

    if getattr(args, "update_weights_interval", 1) < 1:
        raise ValueError("--update-weights-interval must be at least 1")

    if getattr(args, "eval_termination_retry_times", 0) < 0:
        raise ValueError("--eval-termination-retry-times must be a non-negative integer.")

    if getattr(args, "wandb_skip_resume_first_step", False) and (not args.use_wandb or args.wandb_run_id is None):
        raise ValueError("--wandb-skip-resume-first-step requires --use-wandb and --wandb-run-id.")

    if args.kl_coef != 0 or args.use_kl_loss:
        if not os.path.exists(args.ref_load):
            raise FileNotFoundError(f"ref_load {args.ref_load} does not exist, please check the path.")

        if not os.path.exists(os.path.join(args.ref_load, "latest_checkpointed_iteration.txt")):
            logger.info(
                f"ref_load {args.ref_load} does not have latest_checkpointed_iteration.txt, "
                "please make sure it is a valid megatron checkpoint directory."
            )

    # Validate on-policy distillation (OPD) arguments
    if args.use_opd:
        if args.opd_type is None:
            raise ValueError("--opd-type must be specified when --use-opd is enabled. Choose 'sglang' or 'megatron'.")

        if args.opd_type == "megatron":
            if args.opd_teacher_load is None:
                raise ValueError(
                    "--opd-teacher-load is required when --opd-type=megatron. "
                    "Please provide the path to the teacher model checkpoint."
                )
            if not os.path.exists(args.opd_teacher_load):
                raise FileNotFoundError(
                    f"opd_teacher_load {args.opd_teacher_load} does not exist, please check the path."
                )
            if not os.path.exists(os.path.join(args.opd_teacher_load, "latest_checkpointed_iteration.txt")):
                logger.info(
                    f"opd_teacher_load {args.opd_teacher_load} does not have latest_checkpointed_iteration.txt, "
                    "please make sure it is a valid megatron checkpoint directory."
                )

        elif args.opd_type == "sglang":
            if args.opd_teacher_load is not None:
                raise ValueError(
                    "--opd-teacher-load should not be set when --opd-type=sglang. "
                    "In sglang mode, teacher log-probs are obtained from external server during rollout."
                )
    else:
        # If OPD is not enabled, opd_teacher_load should not be set
        if args.opd_teacher_load is not None:
            raise ValueError("--opd-teacher-load is set but --use-opd is not enabled. Please add --use-opd flag.")

    explicit_debug_rollout_load = args.debug_rollout_only and args.load is not None
    explicit_debug_rollout_start = args.debug_rollout_only and args.start_rollout_id is not None

    if args.megatron_to_hf_mode == "bridge":
        if (
            args.load is not None
            and os.path.exists(args.load)
            and os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            # If is a Megatron checkpoint, won't use bridge to load hf weight.
            pass
        else:
            if args.load is None:
                args.load = args.ref_load or args.hf_checkpoint
            # If is a HF checkpoint, set start_rollout_id to 0 here.
            if not explicit_debug_rollout_start:
                args.start_rollout_id = 0
    else:
        if (
            args.load is None
            or not os.path.exists(args.load)
            or not os.path.exists(os.path.join(args.load, "latest_checkpointed_iteration.txt"))
        ):
            args.no_load_optim = True
            args.no_load_rng = True
            args.finetune = True
            if not explicit_debug_rollout_load:
                args.load = args.ref_load
            if args.ref_ckpt_step is not None:
                args.ckpt_step = args.ref_ckpt_step
            if not explicit_debug_rollout_start:
                args.start_rollout_id = 0

    if args.eval_interval is not None:
        assert args.eval_datasets, "Evaluation datasets must be configured when eval_interval is set."
        assert args.eval_max_inflight_tasks > 0, "eval_max_inflight_tasks must be positive."
        if args.eval_initial_inflight_tasks is not None:
            assert (
                0 < args.eval_initial_inflight_tasks <= args.eval_max_inflight_tasks
            ), "eval_initial_inflight_tasks must be positive and no larger than eval_max_inflight_tasks."
        assert args.eval_concurrency_step > 0, "eval_concurrency_step must be positive."
        assert args.eval_concurrency_poll_interval > 0, "eval_concurrency_poll_interval must be positive."

    if args.save_interval is not None:
        assert args.save is not None, "'--save' is required when save_interval is set."

    assert not (args.kl_coef != 0 and args.kl_loss_coef != 0), "Only one of kl_coef and kl_loss_coef can be set"

    if args.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
        assert args.normalize_advantages, (
            "The 'reinforce_plus_plus' and 'reinforce_plus_plus_baseline' advantage estimators "
            "require advantage normalization. Please add `--normalize-advantages` to your command."
        )

    if (
        args.advantage_estimator in ["grpo", "gspo", "cispo"]
        and args.normalize_advantages
        and args.rewards_normalization
        and args.grpo_std_normalization
    ):
        # Group std-normalization already makes advantages mean-0/std-1 per prompt
        # group, weighting every sequence equally. The global whitening pass in
        # compute_advantages_and_returns then re-centers on a TOKEN-weighted mean.
        # The two disagree whenever response length correlates with reward -- which
        # it does, since wrong/truncated trajectories run longer -- so whitening
        # adds a uniform positive constant to every token. That constant is an
        # unconditional likelihood push whose token mass sits mostly on wrong
        # trajectories, and it destroys GRPO's per-group zero-mean baseline. It is
        # also non-stationary: it is re-estimated from whatever groups survive
        # dynamic sampling. See tests/test_group_reward_normalization.py.
        logger.warning(
            "--normalize-advantages is redundant with GRPO group std-normalization and injects a "
            "length-correlated bias: group norm centers per group (sequence-weighted) while whitening "
            "re-centers globally (token-weighted). Prefer dropping --normalize-advantages, or add "
            "--disable-grpo-std-normalization if global whitening is what you want."
        )

    if args.use_rollout_logprobs:
        assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

    _validate_webqa_ground_truths(args)

    if args.get_mismatch_metrics:
        assert (
            args.custom_tis_function_path is not None
        ), "custom_tis_function_path must be set when get_mismatch_metrics is set"

        if args.use_rollout_logprobs:
            logger.info(
                "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
            )

    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None, "max_tokens_per_gpu must be set when use_dynamic_batch_size is set"
        if args.log_probs_max_tokens_per_gpu is None:
            args.log_probs_max_tokens_per_gpu = args.max_tokens_per_gpu

    if getattr(args, "balance_by_flops", False):
        assert args.use_dynamic_batch_size, "--balance-by-flops requires --use-dynamic-batch-size"
        args.balance_data = True

    if args.eps_clip_high is None:
        args.eps_clip_high = args.eps_clip

    if args.advantage_estimator == "cispo" and args.eps_clip < 1.0:
        logger.warning(
            "CISPO is canonically single-sided, but --eps-clip=%s keeps the lower clip bound %s active. "
            "Set --eps-clip 1.0 (and tune --eps-clip-high, e.g. 4.0) for the canonical wide setting.",
            args.eps_clip,
            1.0 - args.eps_clip,
        )

    if args.eval_reward_key is None:
        args.eval_reward_key = args.reward_key

    if args.dump_details is not None:
        args.save_debug_rollout_data = f"{args.dump_details}/rollout_data/{{rollout_id}}.pt"
        args.save_debug_train_data = f"{args.dump_details}/train_data/{{rollout_id}}_{{rank}}.pt"

    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True

    args.rollout_external = args.rollout_external_engine_addrs is not None

    if args.rollout_external and not args.debug_train_only:
        apply_external_engine_info_to_args(args, logger=logger)

    args.use_critic = args.advantage_estimator == "ppo"
    # Critic always uses the same GPU count as actor.
    args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
    args.critic_num_nodes = args.actor_num_nodes

    if args.offload:
        args.offload_train = True
        args.offload_rollout = True
    del args.offload

    if args.debug_rollout_only:
        if args.colocate and args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        elif args.rollout_num_gpus == 0:
            args.actor_num_gpus_per_node = 0
            args.actor_num_nodes = 0
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0

    if (
        getattr(args, "rollout_only_inference_fast_path", False)
        or getattr(args, "rollout_only_skip_episode_dump", False)
        or getattr(args, "async_save_debug_rollout_data", False)
    ) and not args.debug_rollout_only:
        raise ValueError("rollout-only fast-path options require --debug-rollout-only")

    assert not (args.debug_rollout_only and args.debug_train_only), (
        "debug_rollout_only and debug_train_only cannot be set at the same time, " "please set only one of them."
    )

    # Colocate normally offloads Megatron between rollout and train.  Release-train mode
    # releases Megatron actors instead, so only rollout needs memory-saver offload.
    if args.colocate:
        if args.release_train:
            if args.offload_train:
                logger.info("Ignoring --offload-train because --release-train releases train actors instead.")
            args.offload_train = False
            if args.offload_rollout is False:
                logger.info("Ignoring --no-offload-rollout because colocated --release-train needs rollout offload.")
            args.offload_rollout = True
        elif args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        elif args.rollout_num_gpus == 0:
            logger.info("rollout_num_gpus is 0 under colocate; no local SGLang engines will be launched.")

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False

    if args.use_critic and not args.release_train:
        args.offload_train = True

    if args.offload_train:
        args.disable_grad_buffers_cpu_backup = True
        args.disable_param_buffers_cpu_backup = True

    if args.eval_function_path is None:
        args.eval_function_path = args.rollout_function_path

    if args.num_steps_per_rollout is not None:
        global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
        if args.global_batch_size is not None:
            assert args.global_batch_size == global_batch_size, (
                f"global_batch_size {args.global_batch_size} is not equal to "
                f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
                f"// num_steps_per_rollout {args.num_steps_per_rollout}"
            )
        args.global_batch_size = global_batch_size

    if args.n_samples_per_prompt == 1:
        args.grpo_std_normalization = False
        logger.info("n_samples_per_prompt is set to 1, grpo_std_normalization will be set to False.")

    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size

    assert args.over_sampling_batch_size >= args.rollout_batch_size, (
        f"over_sampling_batch_size {args.over_sampling_batch_size} should be greater than or equal to "
        f"rollout_batch_size {args.rollout_batch_size}"
    )

    if args.num_epoch is not None:
        if args.num_rollout is not None:
            logger.info("Both num_epoch and num_rollout are set, num_epoch will be ignored.")
        else:
            assert args.rollout_global_dataset, (
                "num_epoch is set, but rollout_global_dataset is not set, "
                "please remove --disable-rollout-global-dataset to use num_epoch"
            )
    else:
        # if num_epoch is not set, we should set num_rollout
        assert args.num_rollout is not None, (
            "num_epoch is not set, but num_rollout is not set, " "please set --num-rollout or --num-epoch"
        )

    if args.enable_mtp_training:
        assert args.mtp_num_layers, "mtp_num_layers must be set when enable_mtp_training is set"

    if args.use_rollout_routing_replay:
        args.use_routing_replay = True

    if args.custom_config_path:
        with open(args.custom_config_path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(args, k):
                logger.info(f"Warning: Argument {k} is already set to {getattr(args, k)}, will override with {v}.")
            setattr(args, k, v)

    if args.eval_max_context_len is None:
        logger.info(
            f"args.eval_max_context_len is not set. Use args.rollout_max_context_len {args.rollout_max_context_len} as default value."
        )
        args.eval_max_context_len = args.rollout_max_context_len

    if args.rollout_max_context_len is not None:
        if args.rollout_max_prompt_len is None:
            args.rollout_max_prompt_len = args.rollout_max_context_len - 1
            logger.info(
                f"args.rollout_max_prompt_len is not set. Use args.rollout_max_context_len - 1 ({args.rollout_max_context_len} - 1) as default value so that there is at least one generated token to compute loss."
            )
        assert (
            args.rollout_max_prompt_len <= args.rollout_max_context_len - 1
        ), f"args.rollout_max_prompt_len ({args.rollout_max_prompt_len}) must be smaller than args.rollout_max_context_len ({args.rollout_max_context_len}) so that there is at least one generated token to compute loss."

    if args.only_train_params_name_list and args.freeze_params_name_list:
        raise ValueError("You can only specify ONE of: --only-train-params-name-list, or --freeze-params-name-list.")

    # disk-backed sync (full or delta) writes on the trainer and reads on the engines: needs a shared dir
    if args.update_weight_transport == "disk" and not args.update_weight_disk_dir:
        raise ValueError(
            "--update-weight-transport=disk requires --update-weight-disk-dir to point at "
            "a filesystem shared between the trainer and the rollout engines."
        )
    if args.release_train:
        if args.train_backend != "megatron":
            raise ValueError("--release-train is only supported with the Megatron train backend.")
        if args.keep_old_actor:
            raise ValueError("--release-train does not support --keep-old-actor.")
        if args.save is None:
            raise ValueError("--release-train requires --save so the next Megatron actor can reload.")
        if args.save_interval is None:
            args.save_interval = 1
        if args.update_weight_mode != "full" or args.update_weight_transport != "disk":
            raise ValueError("--release-train requires --update-weight-mode=full and --update-weight-transport=disk.")
    if args.update_weight_mode == "delta":
        if args.update_weight_transport != "disk":
            raise ValueError(
                "--update-weight-mode=delta requires --update-weight-transport=disk, "
                f"got {args.update_weight_transport!r}."
            )
        if args.colocate:
            raise ValueError(
                "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
                "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
                "(snapshot + diff + encode) is pure overhead."
            )
        if not args.update_weight_local_checkpoint_dir:
            raise ValueError(
                "--update-weight-mode=delta requires --update-weight-local-checkpoint-dir "
                "(a rollout-host-local NVMe directory)."
            )


def _validate_webqa_ground_truths(args):
    prompt_data = getattr(args, "prompt_data", None)
    if not prompt_data:
        return
    rollout_path = str(getattr(args, "rollout_function_path", "") or "")
    custom_generate = str(getattr(args, "custom_generate_function_path", "") or "")
    explicit_webqa = "webqa" in prompt_data.lower() or "webqa" in rollout_path.lower()
    if not explicit_webqa and "fused_agent" not in custom_generate.lower():
        return

    label_key = getattr(args, "label_key", None)
    metadata_key = getattr(args, "metadata_key", "metadata")

    inspected = 0
    sampled = 0
    missing = 0
    sample_limit = int(os.environ.get("FUSED_WEBQA_GROUND_TRUTH_SAMPLE_LIMIT", "64"))
    for row in read_file(prompt_data):
        inspected += 1
        metadata = row.get(metadata_key) or {}
        if not explicit_webqa:
            sources = [row.get("data_source"), row.get("task_type"), row.get("benchmark"), row.get("dataset")]
            if isinstance(metadata, dict):
                sources.extend(
                    metadata.get(key) for key in ("data_source", "task_type", "benchmark", "dataset")
                )
            normalized_sources = {
                str(source).strip().lower().replace("-", "_").replace(" ", "_")
                for source in sources
                if source
            }
            if not normalized_sources.intersection({"webqa", "web_search", "search"}):
                if inspected >= sample_limit:
                    break
                continue

        sampled += 1
        label = row.get(label_key) if label_key is not None else row.get("reward_model")
        if isinstance(label, dict):
            gt = label.get("ground_truth") or label.get("target") or label.get("answer") or label.get("answers")
        else:
            gt = label
        if gt is None or (isinstance(gt, str) and not gt.strip()):
            if isinstance(metadata, dict):
                gt = (
                    metadata.get("ground_truth")
                    or metadata.get("target")
                    or metadata.get("answer")
                    or metadata.get("answers")
                )
        if gt is None or (isinstance(gt, str) and not gt.strip()):
            missing += 1
        if inspected >= sample_limit:
            break

    if sampled > 0 and missing == sampled:
        raise ValueError(
            f"webqa prompt-data sample check failed: {sampled}/{sampled} rows missing ground_truth. "
            "Expected a usable ground_truth in reward_model or extra_info."
        )
    if sampled > 0 and missing > 0:
        logger.warning(
            "webqa prompt-data sample check: %s/%s sampled rows are missing ground_truth; "
            "reward will be zero for those rows.",
            missing,
            sampled,
        )
