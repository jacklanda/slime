import logging
import os
import re
import sys
import warnings

import wandb

from . import wandb_utils
from .metric_utils import compute_train_step
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False

_SUPPRESSED_LOG_PATTERNS = [
    re.compile(r"^Failed to load .*/torchao/_C_.*"),
    re.compile(r"^Unable to import `torchao` Tensor objects\."),
    re.compile(r"^`cache_position` is part of Qwen3ASR.*forward's signature, but not documented\."),
    re.compile(r"^NUMA affinity is already constrained for process, skipping NUMA node configuration for GPU\."),
]

# Line patterns suppressed at the raw stdout/stderr layer. These cover third-party
# noise that bypasses the logging filter above: import-time ``logger.warning`` calls
# emitted via ``logging.lastResort`` before any root handler exists (torchao optional
# CUDA kernels), and bare ``print()`` statements that never go through logging at all
# (transformers ``auto_docstring`` lint, the gem/lang_rl import banner). None of these
# affect training; they only clutter the Ray worker logs.
_SUPPRESSED_OUTPUT_PATTERNS = [
    re.compile(r"Failed to load .*/torchao/_C_.*\.so"),
    re.compile(r"Unable to import `torchao` Tensor objects\."),
    re.compile(r"transformers>=5\.0 support is experimental\. Unified Hugging Face checkpoint export"),
    re.compile(r"^\s*(?:\(pid=\d+\)\s+)?_warnings\.warn\(\s*(?:\[repeated \d+x across cluster\])?\s*$"),
    re.compile(r"^\(pid=\d+\)\s*(?:\[repeated \d+x across cluster\])?\s*$"),
    re.compile(r"\[ERROR\] `\w+` is part of .* but not documented\. Make sure to add it to the docstring"),
    re.compile(r"^LANG_RL Log directory:"),
]


class _TrainingNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in _SUPPRESSED_LOG_PATTERNS)


class _FilteredStream:
    """Drop known-noise lines while passing everything else through verbatim.

    Filtering is line-wise so a single ``write`` that batches a noise line with
    legitimate output only loses the noise line. ``splitlines(keepends=True)``
    plus ``"".join`` is lossless for any text we keep, so non-matching writes
    (including ``\\r``-based progress bars) are byte-for-byte unchanged.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._suppress_next_linebreak = False

    def write(self, text):
        if not text:
            return self._wrapped.write(text)
        kept = []
        dropped = False
        for line in text.splitlines(keepends=True):
            if self._suppress_next_linebreak:
                self._suppress_next_linebreak = False
                if not line.strip():
                    dropped = True
                    continue

            if any(pattern.search(line) for pattern in _SUPPRESSED_OUTPUT_PATTERNS):
                dropped = True
                self._suppress_next_linebreak = not line.endswith(("\n", "\r"))
                continue
            kept.append(line)

        if dropped:
            if kept:
                self._wrapped.write("".join(kept))
            # Report a full write so callers that check the return value don't retry.
            return len(text)
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# Backwards-compatible alias; the wrapper now covers stdout and stderr alike.
_FilteredStderr = _FilteredStream


def suppress_fastapi_deprecation():
    """Silence FastAPI's ORJSONResponse deprecation warning.

    Emitted from inside the spawned SGLang HTTP server process, so it has to be
    registered there rather than relying on the trainer's filters. No-op when
    FastAPI is unavailable.
    """
    try:
        from fastapi.exceptions import FastAPIDeprecationWarning
    except Exception:
        return
    warnings.filterwarnings("ignore", category=FastAPIDeprecationWarning)


def suppress_known_training_warnings():
    if not isinstance(sys.stdout, _FilteredStream):
        sys.stdout = _FilteredStream(sys.stdout)
    if not isinstance(sys.stderr, _FilteredStream):
        sys.stderr = _FilteredStream(sys.stderr)

    if "fastapi.exceptions" in sys.modules:
        suppress_fastapi_deprecation()

    warnings.filterwarnings(
        "ignore",
        message=r"transformers>=5\.0 support is experimental\..*",
        category=UserWarning,
        module=r"modelopt\.torch",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var.*",
        category=FutureWarning,
        module=r"ray\._private\.worker",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`load_state_dict` is deprecated and will be removed in future versions\..*",
        category=FutureWarning,
        module=r"megatron\.core\.dist_checkpointing\.strategies\.torch",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Please use DTensor instead and we are deprecating ShardedTensor\.",
        category=FutureWarning,
        module=r"torch\.distributed\.checkpoint\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"barrier\(\): using the device under current context\..*",
        category=UserWarning,
        module=r"torch\.distributed\.c10d_logger",
    )


def suppress_gemma4_expected_training_warnings():
    """Suppress warnings caused by Gemma4's intentional trainer configuration."""
    if os.environ.get("SLIME_GEMMA4_SUPPRESS_EXPECTED_TRAINING_WARNINGS") != "1":
        return

    # The Gemma4 launcher deliberately disables compiler caches. Dynamo reports
    # that this also disables PGO, which is expected and does not affect results.
    warnings.filterwarnings(
        "ignore",
        message=r"dynamo_pgo force disabled by torch\.compiler\.config\.force_disable_caches",
        category=UserWarning,
        module=r"torch\._dynamo\.pgo",
    )

    # Megatron DDP keeps AccumulateGrad nodes alive to host its per-parameter
    # hooks. Gemma4 does not use trainer CUDA graphs or overlapped grad-reduce,
    # so a node surviving from the initialization stream is intentional here.
    import torch

    setter = getattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch", None)
    if setter is not None:
        setter(False)


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True
    suppress_known_training_warnings()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    noise_filter = _TrainingNoiseFilter()
    logging.getLogger().addFilter(noise_filter)
    for handler in logging.getLogger().handlers:
        handler.addFilter(noise_filter)


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


_DROPPED_TRACKING_PREFIXES = (
    "rollout/candidate/",
    "rollout/selected/",
    "rollout/batch/",
)
_DROPPED_TRACKING_KEYS = {
    "episode/correct",
    "episode/pass@1",
    "episode/reward/mean",
}


def _drop_untracked_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in _DROPPED_TRACKING_KEYS and not any(key.startswith(prefix) for prefix in _DROPPED_TRACKING_PREFIXES)
    }


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str, rollout_id: int | None = None):
    metrics = _drop_untracked_metrics(metrics)
    if args.use_wandb:
        skip_first_rollout = getattr(args, "wandb_skip_resume_first_step", False)
        if skip_first_rollout and rollout_id is None:
            raise ValueError("rollout_id is required when --wandb-skip-resume-first-step is enabled.")
        skip_rollout_id = getattr(args, "_wandb_skip_resume_rollout_id", None)
        if skip_first_rollout and skip_rollout_id is None:
            skip_rollout_id = rollout_id
            args._wandb_skip_resume_rollout_id = rollout_id
            logging.getLogger(__name__).info(
                "Skipping W&B metrics for the first rollout after resume (rollout_id=%s).", rollout_id
            )
        if not skip_first_rollout or rollout_id != skip_rollout_id:
            wandb_metrics = dict(metrics)
            if "train/step" not in wandb_metrics:
                if rollout_id is None:
                    raise ValueError("rollout_id is required to derive train/step for W&B logging.")
                wandb_metrics["train/step"] = compute_train_step(args, rollout_id)
            wandb.log(wandb_metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
