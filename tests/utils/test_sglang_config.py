"""Unit tests for SglangConfig multi-model parsing with update_weights."""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_yaml(data: dict) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.flush()
    return f.name


class TestSglangConfigUpdateWeights:
    def test_update_weights_default_true(self):
        """Models without explicit update_weights should default to True."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    }
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        config.models[0].resolve(Namespace(hf_checkpoint="/tmp/hf", rollout_num_gpus_per_engine=1))
        assert len(config.models) == 1
        assert config.models[0].update_weights is True

    def test_update_weights_explicit_false(self):
        """Models with update_weights: false should be parsed correctly."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "update_weights": True,
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "model_path": "/path/to/ref",
                        "engine_groups": [{"worker_type": "regular", "num_gpus": 2}],
                    },
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        assert len(config.models) == 2
        assert config.models[0].name == "actor"
        assert config.models[0].update_weights is True
        assert config.models[1].name == "ref"
        assert config.models[1].update_weights is False
        assert config.models[1].model_path == "/path/to/ref"

    def test_multi_model_total_gpus(self):
        """total_num_gpus should sum across all models."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml(
            {
                "sglang": [
                    {
                        "name": "actor",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 8}],
                    },
                    {
                        "name": "ref",
                        "update_weights": False,
                        "server_groups": [{"worker_type": "regular", "num_gpus": 4}],
                    },
                ]
            }
        )
        config = SglangConfig.from_yaml(path)
        assert config.total_num_gpus == 12

    def test_config_allows_model_with_no_server_groups(self):
        """A model with no server groups can expose a router without local engines."""
        from slime.backends.sglang_utils.sglang_config import SglangConfig

        path = _write_yaml({"sglang": [{"name": "default", "server_groups": []}]})

        config = SglangConfig.from_yaml(path)

        assert len(config.models) == 1
        assert config.models[0].name == "default"
        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0


class TestZeroGpuRolloutConfig:
    def test_resolve_default_zero_gpu_config_has_no_server_groups(self):
        from slime.ray.rollout import _resolve_sglang_config

        args = Namespace(sglang_config=None, prefill_num_servers=None, rollout_num_gpus=0)

        config = _resolve_sglang_config(args)

        assert len(config.models) == 1
        assert config.models[0].name == "default"
        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0

    def test_zero_gpu_config_takes_precedence_over_prefill_num_servers(self):
        from slime.ray.rollout import _resolve_sglang_config

        args = Namespace(sglang_config=None, prefill_num_servers=1, rollout_num_gpus=0)

        config = _resolve_sglang_config(args)

        assert config.models[0].server_groups == []
        assert config.total_num_gpus == 0

    def test_start_rollout_servers_zero_gpu_starts_router_without_engines(self, monkeypatch):
        from slime.ray import rollout as rollout_module

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456

        monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)
        args = Namespace(
            rollout_external=False,
            sglang_config=None,
            prefill_num_servers=None,
            rollout_num_gpus=0,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=False,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = rollout_module.start_rollout_servers(args, pg=(None, [], []))

        assert list(servers) == ["default"]
        assert init_handles == []
        server = servers["default"]
        assert server.router_ip == "127.0.0.1"
        assert server.router_port == 3456
        assert server.server_groups == []
        assert server.engines == []
        assert args.sglang_router_ip == "127.0.0.1"
        assert args.sglang_router_port == 3456
        assert args.sglang_model_routers == {"default": ("127.0.0.1", 3456)}

    def test_start_rollout_servers_defers_engine_wait(self, monkeypatch):
        from slime.ray import rollout as rollout_module

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456

        def fake_start_engines(self, port_cursors=None):
            self.all_engines = [object() for _ in self.all_engines]
            return [f"init-{self.rank_offset}"], port_cursors or {}

        ray_get_calls = []

        def fake_ray_get(refs):
            ray_get_calls.append(refs)

        monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)
        monkeypatch.setattr(rollout_module.ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(rollout_module.ray, "get", fake_ray_get)

        args = Namespace(
            rollout_external=False,
            sglang_config=None,
            prefill_num_servers=None,
            rollout_num_gpus=2,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=False,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = rollout_module.start_rollout_servers(args, pg=(None, [], []))

        assert list(servers) == ["default"]
        assert init_handles == ["init-0"]
        assert ray_get_calls == []

    def test_start_rollout_servers_waits_for_epd_encoder_before_non_encoder(self, monkeypatch):
        from slime.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
        from slime.ray import rollout as rollout_module

        class FakeRemoteMethod:
            def __init__(self, value):
                self.value = value

            def remote(self):
                return self.value

        class FakeEngine:
            def __init__(self, url_ref):
                self.get_url = FakeRemoteMethod(url_ref)

        def fake_start_router(args, *, has_pd_disaggregation=False, force_new=False):
            assert has_pd_disaggregation is False
            assert force_new is False
            return "127.0.0.1", 3456

        def fake_resolve_sglang_config(args):
            return SglangConfig(
                models=[
                    ModelConfig(
                        name="default",
                        server_groups=[
                            ServerGroupConfig(worker_type="encoder", num_gpus=1),
                            ServerGroupConfig(worker_type="regular", num_gpus=1),
                        ],
                    )
                ]
            )

        def fake_start_engines(self, port_cursors=None):
            if self.worker_type == "encoder":
                self.all_engines = [FakeEngine("encoder-url-ref") for _ in self.all_engines]
            else:
                self.all_engines = [object() for _ in self.all_engines]
            return [f"{self.worker_type}-init-{self.rank_offset}"], port_cursors or {}

        ray_get_calls = []

        def fake_ray_get(refs):
            ray_get_calls.append(refs)
            if refs == ["encoder-url-ref"]:
                return ["http://encoder"]
            return None

        monkeypatch.setattr(rollout_module, "_start_router", fake_start_router)
        monkeypatch.setattr(rollout_module, "_resolve_sglang_config", fake_resolve_sglang_config)
        monkeypatch.setattr(rollout_module.ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(rollout_module.ray, "get", fake_ray_get)

        args = Namespace(
            rollout_external=False,
            rollout_num_gpus_per_engine=1,
            num_gpus_per_node=8,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            offload_rollout=False,
            hf_checkpoint="/tmp/hf",
        )

        servers, init_handles = rollout_module.start_rollout_servers(args, pg=(None, [], []))

        groups = servers["default"].server_groups
        assert [group.worker_type for group in groups] == ["encoder", "regular"]
        assert groups[1].sglang_overrides["language_only"] is True
        assert groups[1].sglang_overrides["encoder_urls"] == ["http://encoder"]
        assert init_handles == ["regular-init-1"]
        assert ray_get_calls == [["encoder-init-0"], ["encoder-url-ref"]]


class TestRolloutServerRecovery:
    def test_recover_updatable_engines_probes_without_fault_tolerance(self, monkeypatch):
        from slime.ray import rollout as rollout_module

        recover_calls = []
        server = rollout_module.RolloutServer(server_groups=[], update_weights=True)

        def fake_recover(health_check_timeout=None):
            recover_calls.append(health_check_timeout)

        monkeypatch.setattr(server, "recover", fake_recover)
        manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
        manager = object.__new__(manager_cls)
        manager.args = Namespace(rollout_health_check_timeout=7.0)
        manager.servers = {"default": server}
        manager.rollout_id = 3
        manager.rollout_engine_lock = "lock"
        manager._health_monitors = []

        engines, lock, num_new, gpu_counts, gpu_offsets = manager.recover_updatable_engines()

        assert recover_calls == [7.0]
        assert engines == []
        assert lock == "lock"
        assert num_new == 0
        assert gpu_counts == []
        assert gpu_offsets == []

    def test_recover_updatable_engines_skips_probe_before_first_rollout(self):
        from slime.ray import rollout as rollout_module

        class FakeServer:
            update_weights = True
            engines = ["engine"]
            engine_gpu_counts = [1]
            engine_gpu_offsets = [0]
            num_new_engines = 0

            def recover(self, health_check_timeout=None):
                raise AssertionError("initial weight sync should not probe or restart rollout engines")

        manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
        manager = object.__new__(manager_cls)
        manager.args = Namespace(rollout_health_check_timeout=7.0)
        manager.servers = {"default": FakeServer()}
        manager.rollout_id = -1
        manager.rollout_engine_lock = "lock"
        manager._health_monitors = []

        engines, lock, num_new, gpu_counts, gpu_offsets = manager.recover_updatable_engines()

        assert engines == ["engine"]
        assert lock == "lock"
        assert num_new == 0
        assert gpu_counts == [1]
        assert gpu_offsets == [0]

    def test_recover_marks_unhealthy_actor_for_restart(self, monkeypatch):
        from slime.ray import rollout as rollout_module

        events = []

        class FakeRemoteMethod:
            def __init__(self, value=None, exc=None):
                self.value = value
                self.exc = exc

            def remote(self, **kwargs):
                if self.exc is not None:
                    return self.exc
                return self.value

        class FakeEngine:
            def __init__(self):
                self.health_generate = FakeRemoteMethod(exc=RuntimeError("connection refused"))
                self.shutdown = FakeRemoteMethod("shutdown-ref")

        def fake_ray_get(refs):
            if isinstance(refs, RuntimeError):
                raise refs
            events.append(("ray_get", refs))

        def fake_ray_kill(engine):
            events.append(("ray_kill", engine))

        def fake_remove_worker_from_router(worker_url, router_ip, router_port):
            events.append(("remove", worker_url, router_ip, router_port))

        def fake_start_engines(self, port_cursors=None):
            events.append(("start",))
            self.all_engines[0] = object()
            self.num_new_engines = 1
            self.engine_urls[0] = "http://10.0.0.2:15002"
            return ["init-ref"], port_cursors or {}

        old_engine = FakeEngine()

        monkeypatch.setattr(rollout_module.ray, "get", fake_ray_get)
        monkeypatch.setattr(rollout_module.ray, "kill", fake_ray_kill)
        monkeypatch.setattr(rollout_module, "remove_worker_from_router", fake_remove_worker_from_router)
        monkeypatch.setattr(rollout_module.ServerGroup, "start_engines", fake_start_engines)

        group = rollout_module.ServerGroup(
            args=Namespace(num_gpus_per_node=1, rollout_num_gpus_per_engine=1),
            pg=(None, [], []),
            all_engines=[old_engine],
            num_gpus_per_engine=1,
            num_new_engines=0,
            worker_type="regular",
            router_ip="127.0.0.1",
            router_port=3000,
            engine_urls=["http://10.0.0.1:15002"],
        )
        server = rollout_module.RolloutServer(server_groups=[group])

        server.recover(health_check_timeout=0.1)

        assert events == [
            ("ray_get", "shutdown-ref"),
            ("ray_kill", old_engine),
            ("remove", "http://10.0.0.1:15002", "127.0.0.1", 3000),
            ("start",),
            ("ray_get", ["init-ref"]),
        ]

    def test_recover_removes_stale_router_worker_before_restart(self, monkeypatch):
        from slime.ray import rollout as rollout_module

        events = []

        def fake_remove_worker_from_router(worker_url, router_ip, router_port):
            events.append(("remove", worker_url, router_ip, router_port))

        def fake_start_engines(self, port_cursors=None):
            events.append(("start",))
            self.all_engines[0] = object()
            self.num_new_engines = 1
            self.engine_urls[0] = "http://10.0.0.2:15002"
            return ["init-ref"], port_cursors or {}

        ray_get_calls = []

        def fake_ray_get(refs):
            ray_get_calls.append(refs)

        monkeypatch.setattr(rollout_module, "remove_worker_from_router", fake_remove_worker_from_router)
        monkeypatch.setattr(rollout_module.ServerGroup, "start_engines", fake_start_engines)
        monkeypatch.setattr(rollout_module.ray, "get", fake_ray_get)

        group = rollout_module.ServerGroup(
            args=Namespace(num_gpus_per_node=1, rollout_num_gpus_per_engine=1),
            pg=(None, [], []),
            all_engines=[None],
            num_gpus_per_engine=1,
            num_new_engines=0,
            worker_type="regular",
            router_ip="127.0.0.1",
            router_port=3000,
            engine_urls=["http://10.0.0.1:15002"],
        )
        server = rollout_module.RolloutServer(server_groups=[group])

        server.recover()

        assert events == [
            ("remove", "http://10.0.0.1:15002", "127.0.0.1", 3000),
            ("start",),
        ]
        assert ray_get_calls == [["init-ref"]]


class TestGetModelUrl:
    def test_get_model_url_basic(self):
        """get_model_url should return the correct URL for a named model."""
        from argparse import Namespace

        from slime.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={
                "actor": ("10.0.0.1", 3000),
                "ref": ("10.0.0.1", 3001),
            },
        )
        assert get_model_url(args, "actor") == "http://10.0.0.1:3000/generate"
        assert get_model_url(args, "ref") == "http://10.0.0.1:3001/generate"
        assert get_model_url(args, "ref", "/v1/chat/completions") == "http://10.0.0.1:3001/v1/chat/completions"

    def test_get_model_url_fallback(self):
        """get_model_url should fall back to default router if model not found."""
        from argparse import Namespace

        from slime.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
            sglang_model_routers={"actor": ("10.0.0.1", 3000)},
        )
        assert get_model_url(args, "unknown") == "http://10.0.0.1:3000/generate"

    def test_get_model_url_no_routers(self):
        """get_model_url should work when sglang_model_routers is not set."""
        from argparse import Namespace

        from slime.rollout.sglang_rollout import get_model_url

        args = Namespace(
            sglang_router_ip="10.0.0.1",
            sglang_router_port=3000,
        )
        assert get_model_url(args, "anything") == "http://10.0.0.1:3000/generate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
