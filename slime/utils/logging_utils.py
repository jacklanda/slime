import logging
import re
import sys
import warnings

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False

_SUPPRESSED_LOG_PATTERNS = [
    re.compile(r"^Failed to load .*/torchao/_C_.*"),
    re.compile(r"^Unable to import `torchao` Tensor objects\."),
    re.compile(r"^`cache_position` is part of Qwen3ASR.*forward's signature, but not documented\."),
    re.compile(r"^NUMA affinity is already constrained for process, skipping NUMA node configuration for GPU\."),
]


class _TrainingNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in _SUPPRESSED_LOG_PATTERNS)


class _FilteredStderr:
    _SUPPRESSED_SUBSTRINGS = (
        "[ERROR] `cache_position` is part of Qwen3ASR",
        "but not documented. Make sure to add it to the docstring",
    )

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        if all(substr in text for substr in self._SUPPRESSED_SUBSTRINGS):
            return len(text)
        return self._wrapped.write(text)

    def flush(self):
        return self._wrapped.flush()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def suppress_known_training_warnings():
    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)

    warnings.filterwarnings(
        "ignore",
        message=r"transformers>=5\.0 support is experimental\..*",
        category=UserWarning,
        module=r"modelopt\.torch",
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


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
