import importlib
import sys
import types

import pytest

NUM_GPUS = 0


def test_wandb_common_uses_train_step_for_every_metric_namespace(monkeypatch):
    wandb_mod = types.ModuleType("wandb")
    calls = []

    def define_metric(name, **kwargs):
        calls.append((name, kwargs))

    wandb_mod.define_metric = define_metric
    wandb_mod.Settings = lambda **kwargs: kwargs
    wandb_mod.run = types.SimpleNamespace(id="run-1")

    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
    sys.modules.pop("slime.utils.wandb_utils", None)
    mod = importlib.import_module("slime.utils.wandb_utils")

    mod._init_wandb_common()

    custom_step_definitions = {
        name: kwargs["step_metric"]
        for name, kwargs in calls
        if name.endswith("/*") and "step_metric" in kwargs
    }
    assert custom_step_definitions
    assert set(custom_step_definitions.values()) == {"train/step"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
