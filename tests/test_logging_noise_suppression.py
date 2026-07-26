import importlib.util
import io
import sys
import types
import warnings
from pathlib import Path

import pytest

NUM_GPUS = 0


def load_logging_utils(monkeypatch):
    # Stub the optional tracking deps so importing logging_utils doesn't pull in
    # wandb/tensorboard, which aren't needed for the stream-filter behavior here.
    wandb_mod = types.ModuleType("wandb")
    wandb_mod.run = None
    wandb_mod.logged = []
    wandb_mod.log = lambda metrics: wandb_mod.logged.append(metrics)
    wandb_utils_mod = types.ModuleType("slime.utils.wandb_utils")
    tb_mod = types.ModuleType("slime.utils.tensorboard_utils")
    tb_mod._TensorboardAdapter = object

    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.wandb_utils", wandb_utils_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.tensorboard_utils", tb_mod)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "utils" / "logging_utils.py"
    module_name = "test_logging_noise_suppression_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # logging_utils does `from . import wandb_utils`, so it must resolve as a
    # submodule of a real `slime.utils` package.
    module.__package__ = "slime.utils"
    spec.loader.exec_module(module)
    return module


SUPPRESSED_LINES = [
    "WARNING:torchao:Failed to load /env/site-packages/torchao/_C_cutlass_90a.abi3.so: oops\n",
    "Unable to import `torchao` Tensor objects. This may affect loading checkpoints serialized with `torchao`\n",
    "(pid=123) UserWarning: transformers>=5.0 support is experimental. Unified Hugging Face checkpoint export for quantized checkpoints may not work for some models yet.\n",
    "  _warnings.warn(\n",
    "(pid=123)   _warnings.warn(\n",
    "(pid=123)   _warnings.warn( [repeated 6x across cluster]\n",
    "(pid=123)\n",
    "[ERROR] `cache_position` is part of Qwen3ASRThinkerTextModel.forward's signature, " "but not documented. Make sure to add it to the docstring of the function in /x/y.py.\n",
    "LANG_RL Log directory: None\n",
]

KEPT_LINES = [
    "Training step 5 loss=0.12\n",
    "WARNING:torchao:some other torchao message we should keep\n",
    "worker called _warnings.warn( while handling a real error\n",
    "(RolloutManager pid=123) healthy\n",
    "[ERROR] real training error: CUDA out of memory\n",
]


def test_filtered_stream_drops_known_noise(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    sink = io.StringIO()
    stream = mod._FilteredStream(sink)

    for line in SUPPRESSED_LINES:
        ret = stream.write(line)
        assert ret == len(line)  # callers must see a full write
    assert sink.getvalue() == ""


def test_filtered_stream_passes_legitimate_output(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    sink = io.StringIO()
    stream = mod._FilteredStream(sink)

    for line in KEPT_LINES:
        stream.write(line)
    assert sink.getvalue() == "".join(KEPT_LINES)


def test_filtered_stream_mixed_batch_keeps_only_clean_lines(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    sink = io.StringIO()
    stream = mod._FilteredStream(sink)

    batched = KEPT_LINES[0] + SUPPRESSED_LINES[0] + KEPT_LINES[1]
    stream.write(batched)
    assert sink.getvalue() == KEPT_LINES[0] + KEPT_LINES[1]


def test_filtered_stream_preserves_separately_written_linebreaks(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    sink = io.StringIO()
    stream = mod._FilteredStream(sink)

    stream.write("first log")
    stream.write("\n")
    stream.write("second log")
    stream.write("\n")

    assert sink.getvalue() == "first log\nsecond log\n"


def test_filtered_stream_drops_only_linebreak_after_suppressed_fragment(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    sink = io.StringIO()
    stream = mod._FilteredStream(sink)

    stream.write("before\n")
    stream.write("(pid=123)   _warnings.warn(")
    stream.write("\n")
    stream.write("(SGLangEngine pid=456) after\n")

    assert sink.getvalue() == "before\n(SGLangEngine pid=456) after\n"


def test_log_drops_untracked_metric_namespaces(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    args = types.SimpleNamespace(
        use_wandb=True,
        use_tensorboard=False,
        wandb_skip_resume_first_step=False,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        global_batch_size=32,
    )

    mod.log(
        args,
        {
            "rollout/step": 3,
            "rollout/candidate/steps/mean": 2.0,
            "rollout/selected/termination/env_done": 1.0,
            "rollout/batch/num_tasks": 8,
            "episode/correct": 1.0,
            "episode/pass@1": 1.0,
            "episode/reward/mean": 1.0,
            "episode/training_reward/mean": 0.5,
        },
        step_key="rollout/step",
        rollout_id=3,
    )

    assert sys.modules["wandb"].logged == [
        {
            "rollout/step": 3,
            "episode/training_reward/mean": 0.5,
            "train/step": 6,
        }
    ]


def test_log_skips_every_wandb_update_from_first_resumed_rollout(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    args = types.SimpleNamespace(
        use_wandb=True,
        use_tensorboard=False,
        wandb_skip_resume_first_step=True,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        global_batch_size=32,
    )

    mod.log(args, {"rollout/step": 7, "rollout/reward": 0.1}, step_key="rollout/step", rollout_id=7)
    mod.log(args, {"train/step": 21, "train/loss": 1.0}, step_key="train/step", rollout_id=7)
    mod.log(args, {"rollout/step": 7, "perf/step_time": 2.0}, step_key="rollout/step", rollout_id=7)
    mod.log(args, {"rollout/step": 8, "rollout/reward": 0.9}, step_key="rollout/step", rollout_id=8)

    assert sys.modules["wandb"].logged == [{"rollout/step": 8, "rollout/reward": 0.9, "train/step": 16}]


def test_log_does_not_skip_tensorboard_for_first_resumed_rollout(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    tensorboard_logs = []

    class TensorboardAdapter:
        def __init__(self, args):
            pass

        def log(self, data, step):
            tensorboard_logs.append((data, step))

    monkeypatch.setattr(mod, "_TensorboardAdapter", TensorboardAdapter)
    args = types.SimpleNamespace(
        use_wandb=True,
        use_tensorboard=True,
        wandb_skip_resume_first_step=True,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        global_batch_size=32,
    )

    mod.log(args, {"rollout/step": 7, "rollout/reward": 0.1}, step_key="rollout/step", rollout_id=7)

    assert sys.modules["wandb"].logged == []
    assert tensorboard_logs == [({"rollout/reward": 0.1}, 7)]


def test_log_preserves_explicit_train_step(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    args = types.SimpleNamespace(
        use_wandb=True,
        use_tensorboard=False,
        wandb_skip_resume_first_step=False,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        global_batch_size=32,
    )

    mod.log(args, {"train/step": 7, "train/loss": 0.25}, step_key="train/step", rollout_id=3)

    assert sys.modules["wandb"].logged == [{"train/step": 7, "train/loss": 0.25}]


def test_log_adds_train_step_only_to_wandb_payload(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    tensorboard_logs = []

    class TensorboardAdapter:
        def __init__(self, args):
            pass

        def log(self, data, step):
            tensorboard_logs.append((data, step))

    monkeypatch.setattr(mod, "_TensorboardAdapter", TensorboardAdapter)
    args = types.SimpleNamespace(
        use_wandb=True,
        use_tensorboard=True,
        wandb_skip_resume_first_step=False,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        global_batch_size=32,
    )

    mod.log(args, {"rollout/step": 3, "rollout/reward": 0.5}, step_key="rollout/step", rollout_id=3)

    assert sys.modules["wandb"].logged == [{"rollout/step": 3, "rollout/reward": 0.5, "train/step": 6}]
    assert tensorboard_logs == [({"rollout/reward": 0.5}, 3)]


def test_suppress_is_idempotent_and_wraps_both_streams(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    real_out, real_err = sys.stdout, sys.stderr
    try:
        mod.suppress_known_training_warnings()
        wrapped_out, wrapped_err = sys.stdout, sys.stderr
        assert isinstance(wrapped_out, mod._FilteredStream)
        assert isinstance(wrapped_err, mod._FilteredStream)

        # A second call must not double-wrap.
        mod.suppress_known_training_warnings()
        assert sys.stdout is wrapped_out
        assert sys.stderr is wrapped_err
    finally:
        sys.stdout, sys.stderr = real_out, real_err


def test_suppress_ignores_ray_accelerator_override_future_warning(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    mod.suppress_known_training_warnings()

    with warnings.catch_warnings(record=True) as caught:
        warnings.warn_explicit(
            "Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default).",
            FutureWarning,
            filename="ray/_private/worker.py",
            lineno=2051,
            module="ray._private.worker",
        )

    assert caught == []


def test_filtered_stderr_alias(monkeypatch):
    mod = load_logging_utils(monkeypatch)
    assert mod._FilteredStderr is mod._FilteredStream


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
