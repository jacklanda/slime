from types import SimpleNamespace

import pytest

from slime.ray.actor_group import RayTrainGroup


@pytest.mark.unit
def test_train_env_can_disable_torch_memory_saver_cpu_backup(monkeypatch):
    captured = {}
    args = SimpleNamespace(
        offload_train=True,
        train_backend="megatron",
        train_env_vars={"TMS_INIT_ENABLE_CPU_BACKUP": "0"},
        update_weight_start_version=0,
        use_routing_replay=False,
        rollout_data_transport="object-store",
    )
    group = RayTrainGroup(args, 1, 1, (object(), [0], [0]), actor_cls=object)

    def stop_after_capturing_env(**options):
        captured.update(options["runtime_env"]["env_vars"])
        raise RuntimeError("captured actor environment")

    monkeypatch.setattr("slime.ray.actor_group.ray.remote", stop_after_capturing_env)

    with pytest.raises(RuntimeError, match="captured actor environment"):
        group._allocate_gpus_for_actor(group._pg, 1)

    assert captured["TMS_INIT_ENABLE"] == "1"
    assert captured["TMS_INIT_ENABLE_CPU_BACKUP"] == "0"
