from __future__ import annotations

import json
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import slime.rollout.fused_agent.parser as fused_parser
from slime_plugins.evals import vitabench_launcher


def _args(tmp_path: Path):
    root = tmp_path / "vitabench"
    (root / "src" / "vita").mkdir(parents=True)
    (root / "src" / "vita" / "cli.py").touch()
    (root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["requests", "vllm>=0.8.5", "PyYAML"]\n',
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    return vitabench_launcher.parse_args(
        [
            "--vitabench-root",
            str(root),
            "--model",
            str(model),
            "--served-model-name",
            "checkpoint-9",
            "--output-dir",
            str(tmp_path / "output"),
            "--venv-dir",
            str(tmp_path / "venv"),
            "--sglang-python-bin",
            sys.executable,
            "--cuda-visible-devices",
            "0,1",
            "--tp-size",
            "1",
            "--dp-size",
            "2",
            "--context-length",
            "40960",
            "--mem-fraction-static",
            "0.9",
            "--user-model",
            "user-model",
            "--user-base-url",
            "https://judge.example/v1",
            "--user-api-key",
            "user-key",
            "--evaluator-model",
            "judge-model",
            "--evaluator-base-url",
            "https://judge.example/v1",
            "--evaluator-api-key",
            "judge-key",
        ]
    )


def test_vitabench_dependencies_exclude_vllm(tmp_path: Path):
    args = _args(tmp_path)

    assert vitabench_launcher._project_dependencies(args.vitabench_root / "pyproject.toml") == [
        "requests",
        "PyYAML",
    ]


def test_vitabench_finds_compatible_sglang_libstdcxx(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    prefix = tmp_path / "sglang-env"
    library = prefix / "lib" / "libstdc++.so.6"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"binary\0GLIBCXX_3.4.32\0")
    monkeypatch.setattr(vitabench_launcher.subprocess, "check_output", lambda *args, **kwargs: str(prefix) + "\n")

    assert vitabench_launcher._resolve_sglang_libstdcxx(args) == library.resolve()


def test_vitabench_model_config_disables_sessions_for_data_parallel_server(tmp_path: Path, capsys):
    args = _args(tmp_path)
    args.output_dir.mkdir()

    config_path = vitabench_launcher.write_model_config(args)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = {item["name"]: item for item in config["models"]}

    assert models["slime-agent"]["base_url"] == "http://127.0.0.1:18080/v1"
    assert models["slime-agent"]["model"] == "checkpoint-9"
    assert models["slime-agent"]["backend"] == "slime_fused_gem"
    assert models["slime-agent"]["tool_parser_model_name"] == str(args.model.resolve())
    assert models["slime-agent"]["max_context_tokens"] == 40960
    assert models["slime-agent"]["use_sglang_session"] is False
    assert models["slime-agent"]["dp_size"] == 2
    assert models["slime-agent"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert models["slime-agent"]["top_k"] == -1
    assert models["vitabench-user"]["base_url"] == "https://judge.example/v1"
    assert models["vitabench-user"]["max_tokens"] == 4096
    assert models["vitabench-evaluator"]["model"] == "judge-model"
    assert models["vitabench-evaluator"]["max_tokens"] == 8192
    assert "session lifecycle requests cannot be pinned" in capsys.readouterr().out


def test_vitabench_model_config_keeps_sessions_for_single_dp_worker(tmp_path: Path):
    args = _args(tmp_path)
    args.dp_size = 1
    args.tp_size = 2
    args.output_dir.mkdir()

    config = json.loads(vitabench_launcher.write_model_config(args).read_text(encoding="utf-8"))
    agent = next(item for item in config["models"] if item["name"] == "slime-agent")

    assert agent["use_sglang_session"] is True


def test_vitabench_preflight_has_no_runtime_side_effects(tmp_path: Path, capsys):
    args = _args(tmp_path)
    args.preflight_only = True

    assert vitabench_launcher.main(
        [
            *sys.argv[:0],
            "--vitabench-root",
            str(args.vitabench_root),
            "--model",
            str(args.model),
            "--served-model-name",
            args.served_model_name,
            "--output-dir",
            str(args.output_dir),
            "--venv-dir",
            str(args.venv_dir),
            "--sglang-python-bin",
            args.sglang_python_bin,
            "--cuda-visible-devices",
            args.cuda_visible_devices,
            "--tp-size",
            str(args.tp_size),
            "--dp-size",
            str(args.dp_size),
            "--context-length",
            str(args.context_length),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--user-model",
            args.user_model,
            "--user-base-url",
            args.user_base_url,
            "--user-api-key",
            args.user_api_key,
            "--evaluator-model",
            args.evaluator_model,
            "--evaluator-base-url",
            args.evaluator_base_url,
            "--evaluator-api-key",
            args.evaluator_api_key,
            "--preflight-only",
        ]
    ) == 0
    assert "VitaBench preflight complete" in capsys.readouterr().out
    assert not args.venv_dir.exists()
    assert not args.output_dir.exists()


def test_vitabench_sglang_command_enables_reasoning_and_determinism(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    args.enable_thinking = "true"
    args.deterministic_inference = "true"
    (args.model / "config.json").write_text('{"max_position_embeddings": 32768}', encoding="utf-8")
    captured = {}

    class Process:
        pid = 123

        def poll(self):
            return None

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(vitabench_launcher.subprocess, "Popen", popen)
    runtime = tmp_path / "libstdc++.so.6"
    monkeypatch.setattr(vitabench_launcher, "_resolve_sglang_libstdcxx", lambda *args: runtime)
    monkeypatch.setattr(vitabench_launcher, "_endpoint_ready", lambda *args: True)
    monkeypatch.setattr(vitabench_launcher, "_port_is_available", lambda *args: True)

    process, _ = vitabench_launcher.start_sglang(args)

    assert process.pid == 123
    assert captured["command"][-3:] == ["--reasoning-parser", "qwen3", "--enable-deterministic-inference"]
    assert "--reasoning-parser" in captured["command"]
    assert captured["kwargs"]["env"]["LD_PRELOAD"].split(":", 1)[0] == str(runtime)
    assert captured["kwargs"]["env"]["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] == "1"


def test_vitabench_clamps_unsupported_context_extension_and_discards_reasoning(tmp_path: Path, capsys):
    args = _args(tmp_path)
    (args.model / "config.json").write_text('{"max_position_embeddings": 32768}', encoding="utf-8")

    vitabench_launcher.normalize_context_settings(args)

    assert args.context_length == 32768
    assert args.discard_historical_thinking == "true"
    assert "model-native 32768" in capsys.readouterr().out


def test_vitabench_keeps_extended_context_with_rope_scaling(tmp_path: Path):
    args = _args(tmp_path)
    (args.model / "config.json").write_text(
        '{"max_position_embeddings": 32768, "rope_scaling": {"type": "yarn", "factor": 4}}',
        encoding="utf-8",
    )

    vitabench_launcher.normalize_context_settings(args)

    assert args.context_length == 40960
    assert args.discard_historical_thinking == "false"


def test_vitabench_overwrite_removes_all_old_outputs_and_bootstrap_compiles(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    args.overwrite = "true"
    args.discard_historical_thinking = "true"
    result_path = args.output_dir / "simulations.json"
    csv_path = args.output_dir / "metrics.csv"
    result_path.write_text("old", encoding="utf-8")
    csv_path.write_text("old", encoding="utf-8")
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        compile(command[2], "<vitabench-bootstrap>", "exec")
        result_path.write_text(
            '{"tasks": [{"id": 1}], "simulations": '
            '[{"task_id": 1, "trial": 0, "reward_info": {"reward": 1.0}}]}',
            encoding="utf-8",
        )
        csv_path.write_text("avg_reward\n1.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vitabench_launcher.subprocess, "run", run)

    assert vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml") == 0
    assert csv_path.read_text(encoding="utf-8") == "avg_reward\n1.0\n"
    assert "strip_historical_thinking" in captured["command"][2]
    assert 'sampling_params["sampling_seed"] = sampling_params.pop("seed")' in captured["command"][2]
    assert 'endswith("/close_session")' in captured["command"][2]
    assert 'state["stable_prefix_length"] = 1' in captured["command"][2]
    assert "tolerant_evaluator_extracter" in captured["command"][2]
    assert "generate_with_reasoning_only_fallback" in captured["command"][2]
    assert "import json" in captured["command"][2]
    assert "environment_module.traceback = types.SimpleNamespace" in captured["command"][2]
    python_path = captured["kwargs"]["env"]["PYTHONPATH"].split(":")
    assert str(args.vitabench_root / "src") in python_path
    assert str(Path(vitabench_launcher.__file__).resolve().parents[2]) in python_path


def _run_bootstrap_against_stub_vita(bootstrap: str, monkeypatch) -> None:
    """Execute the launcher bootstrap with the ``vita`` artifact replaced by stubs."""
    llm_utils = types.ModuleType("vita.utils.llm_utils")
    llm_utils.SLIME_FUSED_GEM_PROTOCOL_VERSION = 2
    llm_utils._post_json = lambda url, data, headers, **kwargs: None
    llm_utils.requests = types.SimpleNamespace(Session=lambda: None)
    llm_utils._HTTP_SESSION_LOCAL = threading.local()
    llm_utils._SGLANG_SESSIONS = {}
    llm_utils._common_prefix_length = lambda left, right: 0
    llm_utils.format_messages = lambda messages: messages
    llm_utils.generate = lambda model, *args, **kwargs: None
    llm_utils.models = {}
    modules = {
        "vita": types.ModuleType("vita"),
        "vita.utils": types.ModuleType("vita.utils"),
        "vita.utils.llm_utils": llm_utils,
        "vita.environment": types.ModuleType("vita.environment"),
        "vita.environment.environment": types.ModuleType("vita.environment.environment"),
        "vita.evaluator": types.ModuleType("vita.evaluator"),
        "vita.evaluator.evaluator_traj": types.ModuleType("vita.evaluator.evaluator_traj"),
        "vita.cli": types.ModuleType("vita.cli"),
        "vllm": types.ModuleType("vllm"),
    }
    modules["vita.evaluator.evaluator_traj"].evaluator_extracter = lambda content: []
    modules["vita.cli"].main = lambda: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attribute = name.rpartition(".")
        if parent:
            setattr(modules[parent], attribute, module)
    monkeypatch.setattr(fused_parser, "make_tool_parser", fused_parser.make_tool_parser)

    exec(compile(bootstrap, "<vitabench-bootstrap>", "exec"), {"__name__": "__vitabench_bootstrap__"})


def test_vitabench_bootstrap_sends_unknown_tool_names_to_the_environment(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    captured = {}

    def run(command, **kwargs):
        captured["bootstrap"] = command[2]
        (args.output_dir / "simulations.json").write_text(
            '{"tasks": [{"id": 1}], "simulations": ' '[{"task_id": 1, "trial": 0, "reward_info": {"reward": 1.0}}]}',
            encoding="utf-8",
        )
        (args.output_dir / "metrics.csv").write_text("avg_reward\n1.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vitabench_launcher.subprocess, "run", run)
    assert vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml") == 0

    _run_bootstrap_against_stub_vita(captured["bootstrap"], monkeypatch)
    parser = fused_parser.make_tool_parser("Qwen3-8B", valid_tools={"instore_product_search_recommend"})

    # A hallucinated tool name reaches the environment, which answers with an
    # error ToolMessage instead of leaking raw XML into the assistant's turn.
    hallucinated = parser.parse('<tool_call>{"name": "get_instore_product_info", "arguments": {"shop_id": "S1"}}</tool_call>')
    assert [(call.name, call.arguments) for call in hallucinated] == [("get_instore_product_info", {"shop_id": "S1"})]
    declared = parser.parse('<tool_call>{"name": "instore_product_search_recommend", "arguments": {"keywords": ["饭团"]}}</tool_call>')
    assert [call.name for call in declared] == ["instore_product_search_recommend"]
    # Prose that merely contains a JSON object is not a tool call.
    assert parser.parse('推荐 {"name": "鸡柳紫米饭团", "price": 12}') == []


def test_vitabench_retries_an_incomplete_result_without_overwrite(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    result_path = args.output_dir / "simulations.json"
    result_path.write_text('{"tasks": [{"id": 1}], "simulations": []}', encoding="utf-8")

    def run(command, **kwargs):
        result_path.write_text(
            '{"tasks": [{"id": 1}], "simulations": '
            '[{"task_id": 1, "trial": 0, "reward_info": {"reward": 1.0}}]}',
            encoding="utf-8",
        )
        (args.output_dir / "metrics.csv").write_text("avg_reward\n1.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vitabench_launcher.subprocess, "run", run)

    assert vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml") == 0


def test_vitabench_preserves_a_complete_result_without_overwrite(tmp_path: Path):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    (args.output_dir / "simulations.json").write_text(
        '{"tasks": [{"id": 1}], "simulations": [{"task_id": 1, "trial": 0}]}',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="result already exists"):
        vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml")


def test_vitabench_refuses_to_reuse_an_occupied_port(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    monkeypatch.setattr(vitabench_launcher, "_port_is_available", lambda *args: False)

    with pytest.raises(RuntimeError, match="already in use"):
        vitabench_launcher.start_sglang(args)


def test_vitabench_stops_sglang_when_startup_times_out(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    args.wait_timeout = 1
    killed = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(vitabench_launcher, "_port_is_available", lambda *args: True)
    monkeypatch.setattr(vitabench_launcher, "_endpoint_ready", lambda *args: False)
    monkeypatch.setattr(vitabench_launcher, "_resolve_sglang_libstdcxx", lambda *args: None)
    monkeypatch.setattr(vitabench_launcher.subprocess, "Popen", lambda *args, **kwargs: Process())
    monotonic = iter((0.0, 2.0))
    monkeypatch.setattr(vitabench_launcher.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(vitabench_launcher.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(TimeoutError):
        vitabench_launcher.start_sglang(args)

    assert killed == [(123, vitabench_launcher.signal.SIGTERM)]


def test_vitabench_rejects_duplicate_or_missing_runs(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    result_path = args.output_dir / "simulations.json"

    def run(command, **kwargs):
        result_path.write_text(
            '{"tasks": [{"id": 1}, {"id": 2}], "simulations": '
            '[{"task_id": 1, "trial": 0}, {"task_id": 1, "trial": 0}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vitabench_launcher.subprocess, "run", run)

    assert vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml") == 1


@pytest.mark.parametrize("reward", [True, float("nan"), float("inf")])
def test_vitabench_rejects_non_finite_or_boolean_rewards(tmp_path: Path, monkeypatch, reward):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    result_path = args.output_dir / "simulations.json"
    csv_path = args.output_dir / "metrics.csv"

    def run(command, **kwargs):
        result_path.write_text(
            json.dumps(
                {
                    "tasks": [{"id": 1}],
                    "simulations": [{"task_id": 1, "trial": 0, "reward_info": {"reward": reward}}],
                }
            ),
            encoding="utf-8",
        )
        csv_path.write_text("avg_reward\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vitabench_launcher.subprocess, "run", run)

    assert vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml") == 1


def test_vitabench_rejects_malformed_trial_without_crashing(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    result_path = args.output_dir / "simulations.json"

    def run(command, **kwargs):
        result_path.write_text(
            '{"tasks": [{"id": 1}], "simulations": '
            '[{"task_id": 1, "trial": "not-an-int", "reward_info": {"reward": 1.0}}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(vitabench_launcher.subprocess, "run", run)

    assert vitabench_launcher.run_vitabench(args, Path(sys.executable), tmp_path / "models.yaml") == 1
