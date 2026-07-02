import importlib
import sys
import types


def test_wandb_common_defines_rollout_length_metrics(monkeypatch):
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

    assert ("response_length/*", {"step_metric": "rollout/step"}) in calls
    assert ("prompt_length/*", {"step_metric": "rollout/step"}) in calls
    assert ("response/*", {"step_metric": "rollout/step"}) in calls
