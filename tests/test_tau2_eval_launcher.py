from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from slime_plugins.evals import tau2_launcher

NUM_GPUS = 0


def _args(tmp_path: Path):
    root = tmp_path / "tau2-bench"
    (root / "src" / "tau2").mkdir(parents=True)
    (root / "src" / "tau2" / "cli.py").touch()
    (root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["requests", "torch>=2", "transformers", "rank-bm25", "PyYAML"]\n',
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    return tau2_launcher.parse_args(
        [
            "--tau2-root", str(root), "--model", str(model), "--model-series", "qwen3",
            "--served-model-name", "checkpoint",
            "--output-dir", str(tmp_path / "output"), "--venv-dir", str(tmp_path / "venv"),
            "--python-bin", sys.executable, "--sglang-python-bin", sys.executable,
            "--cuda-visible-devices", "0,1", "--tp-size", "1", "--dp-size", "2",
            "--context-length", "40960", "--mem-fraction-static", "0.9",
            "--domain", "airline",
            "--user-model", "openai/user", "--user-base-url", "https://example.test/v1",
            "--user-api-key", "secret",
        ]
    )


def test_tau2_dependencies_exclude_local_inference_stack(tmp_path: Path):
    args = _args(tmp_path)
    assert tau2_launcher._project_dependencies(args.tau2_root / "pyproject.toml") == ["requests", "PyYAML"]


def test_tau2_accepts_openrouter_model_id_without_local_checkpoint(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.model = Path("deepseek/deepseek-v3.2")
    args.model_series = "openrouter"
    monkeypatch.setattr(tau2_launcher, "_python_version", lambda python: (3, 12))

    tau2_launcher.validate(args)


def test_tau2_finds_compatible_sglang_libstdcxx(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    prefix = tmp_path / "sglang-env"
    library = prefix / "lib" / "libstdc++.so.6"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"binary\0GLIBCXX_3.4.32\0")
    monkeypatch.setattr(tau2_launcher.subprocess, "check_output", lambda *args, **kwargs: str(prefix) + "\n")

    assert tau2_launcher._resolve_sglang_libstdcxx(args) == library.resolve()


def test_tau2_preflight_has_no_runtime_side_effects(tmp_path: Path, capsys):
    args = _args(tmp_path)
    argv = [
        "--tau2-root", str(args.tau2_root), "--model", str(args.model), "--model-series", args.model_series,
        "--served-model-name", args.served_model_name, "--output-dir", str(args.output_dir),
        "--venv-dir", str(args.venv_dir), "--python-bin", args.python_bin,
        "--sglang-python-bin", args.sglang_python_bin, "--cuda-visible-devices", args.cuda_visible_devices,
        "--tp-size", str(args.tp_size), "--dp-size", str(args.dp_size),
        "--context-length", str(args.context_length), "--mem-fraction-static", str(args.mem_fraction_static),
        "--user-model", args.user_model, "--user-base-url", args.user_base_url,
        "--user-api-key", args.user_api_key, "--preflight-only",
    ]
    assert tau2_launcher.main(argv) == 0
    assert "tau2 preflight complete" in capsys.readouterr().out
    assert not args.output_dir.exists()
    assert not args.venv_dir.exists()


def test_tau2_run_uses_public_api_and_validates_results(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        compile(command[2], "<tau2-bootstrap>", "exec")
        payload = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        captured["serialized_config"] = Path(command[3]).read_text(encoding="utf-8")
        settings = payload["settings"]
        captured["settings"] = settings
        captured["transport"] = payload["transport"]
        captured["user_llm"] = payload["user_llm"]
        captured["evaluator_llm"] = payload["evaluator_llm"]
        result_dir = Path(settings["save_to"])
        result_dir.mkdir(parents=True)
        (result_dir / "results.json").write_text(
            '{"tasks":[{"id":"1"}],"simulations":['
            '{"task_id":"1","trial":0,"termination_reason":"user_stop","reward_info":{"reward":1.0},'
            '"messages":[{"role":"assistant","raw_data":{"slime_fused_protocol_version":2,'
            '"slime_fused_session_id":"session-1"}}]}]}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tau2_launcher.subprocess, "run", run)
    assert tau2_launcher.run_tau2(args, Path(sys.executable)) == 0
    assert not list(args.output_dir.glob(".tau2-run-config-*.json"))
    settings = captured["settings"]
    assert settings["llm_agent"] == "openai/checkpoint"
    assert settings["llm_args_agent"]["max_tokens"] == 8192
    assert settings["max_retries"] == 3
    assert captured["transport"]["base_url"] == "http://127.0.0.1:18081/v1"
    assert captured["transport"]["model_path"] == str(args.model.resolve())
    assert captured["transport"]["use_session"] is True
    assert captured["user_llm"] == {"model": "openai/user", "base_url": "https://example.test/v1"}
    assert captured["evaluator_llm"] == {
        "model": "openai/user",
        "base_url": "https://example.test/v1",
        "json_mode": True,
        "max_tokens": 4096,
    }
    assert "llm_agent.AGENT_INSTRUCTION = tau2_fused_transport.AGENT_INSTRUCTION" in captured["command"][2]
    assert "original_del=batch._original_del" in captured["command"][2]
    assert "asyncio.base_events.BaseEventLoop.__del__ = safe_event_loop_del" in captured["command"][2]
    assert "tau2_fused_transport.normalize_external_user_message(message)" in captured["command"][2]
    assert 'simulation.termination_reason.value == "user_error"' in captured["command"][2]
    assert 'simulation.termination_reason.value == "context_window_exceeded"' in captured["command"][2]
    assert 'simulation.termination_reason.value in {"max_steps", "context_window_exceeded"}' not in captured["command"][2]
    assert 'get("slime_fused_finish_reason") == "length"' not in captured["command"][2]
    assert 'raise RuntimeError("tau2 trajectory contains a max-length agent response")' not in captured["command"][2]
    assert "evaluator_nl_assertions.generate = generate_evaluator" in captured["command"][2]
    assert "interface_agent.generate = generate_evaluator" in captured["command"][2]
    assert 'config["base_url"].rstrip("/") + "/chat/completions"' in captured["command"][2]
    assert "official_generate" not in captured["command"][2]
    assert '"Authorization": f"Bearer {os.environ[api_key_env]}"' in captured["command"][2]
    assert 'tool_calls = getattr(message, "tool_calls", None)' in captured["command"][2]
    assert 'item["tool_call_id"] = message.id' in captured["command"][2]
    assert 'payload["response_format"] = {"type": "json_object"}' in captured["command"][2]
    assert 'payload.setdefault("max_tokens", config["max_tokens"])' in captured["command"][2]
    assert '"json_mode": true' in captured["serialized_config"]
    assert 'json.JSONDecoder().raw_decode(candidate[start:])' in captured["command"][2]
    assert 'raise ValueError("OpenRouter evaluator response contained no JSON object")' in captured["command"][2]
    assert "api_key" not in settings["llm_args_user"]
    assert "secret" not in captured["serialized_config"]
    assert captured["env"]["TAU2_USER_API_KEY"] == "secret"
    assert captured["env"]["TAU2_EVALUATOR_API_KEY"] == "secret"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "OPENAI_BASE_URL" not in captured["env"]
    assert captured["env"]["TAU2_DATA_DIR"] == str((args.tau2_root / "data").resolve())
    assert captured["env"]["FUSED_MODEL_SERIES"] == "qwen3"


@pytest.mark.parametrize(
    ("termination_reason", "raw_data"),
    [
        ("infrastructure_error", {"slime_fused_protocol_version": 2, "slime_fused_session_id": "session-1"}),
        ("user_error", {"slime_fused_protocol_version": 2, "slime_fused_session_id": "session-1"}),
        ("user_stop", {"slime_fused_session_id": "session-1"}),
        ("user_stop", {"slime_fused_protocol_version": 2}),
    ],
)
def test_tau2_result_validation_rejects_invalid_runs(
    tmp_path: Path, monkeypatch, capsys, termination_reason: str, raw_data: dict
):
    args = _args(tmp_path)
    args.output_dir.mkdir()

    def run(command, **kwargs):
        payload = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        result_dir = Path(payload["settings"]["save_to"])
        result_dir.mkdir(parents=True)
        result = {
            "tasks": [{"id": "1"}],
            "simulations": [
                {
                    "task_id": "1",
                    "trial": 0,
                    "termination_reason": termination_reason,
                    "reward_info": None if termination_reason == "infrastructure_error" else {"reward": 1.0},
                    "messages": [{"role": "assistant", "raw_data": raw_data}],
                }
            ],
        }
        (result_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tau2_launcher.subprocess, "run", run)
    assert tau2_launcher.run_tau2(args, Path(sys.executable)) == 1
    if termination_reason == "infrastructure_error":
        stderr = capsys.readouterr().err
        assert "tau2 result validation failed" in stderr
        assert "infrastructure_errors=1" in stderr
        assert "missing_rewards=1" in stderr


def test_tau2_result_validation_rejects_empty_benchmark(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.output_dir.mkdir()

    def run(command, **kwargs):
        payload = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        result_dir = Path(payload["settings"]["save_to"])
        result_dir.mkdir(parents=True)
        (result_dir / "results.json").write_text('{"tasks":[],"simulations":[]}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tau2_launcher.subprocess, "run", run)
    assert tau2_launcher.run_tau2(args, Path(sys.executable)) == 1


def test_evals_script_routes_tau2_to_official_pipeline():
    launcher = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text(encoding="utf-8")
    assert 'TAU2_ROOT="${BENCHMARKS_ROOT}/tau^2-bench"' in launcher
    assert 'TAU2_USER_MODEL="${TAU2_USER_MODEL:-openai/gpt-4.1}"' in launcher
    assert 'TAU2_EVALUATOR_MODEL="${TAU2_EVALUATOR_MODEL:-anthropic/claude-opus-4.5}"' in launcher
    assert 'TAU2_TEMPERATURE="${TAU2_TEMPERATURE:-0.2}"' in launcher
    assert 'TAU2_MAX_TOKENS="${TAU2_MAX_TOKENS:-16384}"' in launcher
    assert 'TAU2_MAX_STEPS="${TAU2_MAX_STEPS:-200}"' in launcher
    assert 'TAU2_MAX_RETRIES="${TAU2_MAX_RETRIES:-4}"' in launcher
    assert "slime_plugins.evals.tau2_launcher" in launcher
    assert '--num-tasks "${TAU2_NUM_TASKS}"' in launcher
    assert '--max-retries "${TAU2_MAX_RETRIES}"' in launcher
    assert '--temperature "${TAU2_TEMPERATURE}"' in launcher
    assert 'temperature=${TAU2_TEMPERATURE}, context=${EVAL_MAX_CONTEXT_LEN}, max_tokens=${TAU2_MAX_TOKENS}' in launcher
    assert 'is_truthy "${CLEANUP}" && ! is_truthy "${PREFLIGHT_ONLY}"' in launcher
    assert '--discard-historical-thinking "${DISCARD_HISTORICAL_THINKING}"' in launcher
    assert '--use-sglang-session "${TAU2_USE_SGLANG_SESSION}"' in launcher
    assert '--user-api-key "${TAU2_USER_API_KEY}"' not in launcher
    assert "export TAU2_USER_API_KEY" in launcher
    assert 'exec "${TAU2_CMD[@]}"' in launcher


def test_tau2_max_retries_is_configurable(tmp_path: Path):
    args = _args(tmp_path)
    assert args.max_retries == 3
    assert tau2_launcher.parse_args(
        [
            "--tau2-root", str(args.tau2_root), "--model", str(args.model),
            "--model-series", args.model_series, "--served-model-name", args.served_model_name,
            "--output-dir", str(args.output_dir), "--venv-dir", str(args.venv_dir),
            "--python-bin", args.python_bin, "--sglang-python-bin", args.sglang_python_bin,
            "--cuda-visible-devices", args.cuda_visible_devices, "--tp-size", str(args.tp_size),
            "--dp-size", str(args.dp_size), "--context-length", str(args.context_length),
            "--mem-fraction-static", str(args.mem_fraction_static), "--domain", args.domain,
            "--user-model", args.user_model, "--user-base-url", args.user_base_url,
            "--user-api-key", args.user_api_key, "--max-retries", "7",
        ]
    ).max_retries == 7


def test_tau2_domain_selection_defaults_to_complete_text_benchmark():
    assert tau2_launcher._domains("all") == ["airline", "retail", "telecom"]
    assert tau2_launcher._domains("telecom,airline,telecom") == ["telecom", "airline"]


def test_tau2_global_concurrency_is_distributed_across_domains():
    assert tau2_launcher._domain_concurrency(["airline", "retail", "telecom"], 64, 0) == {
        "airline": 12,
        "retail": 26,
        "telecom": 26,
    }
    assert sum(tau2_launcher._domain_concurrency(["airline", "retail", "telecom"], 128, 24).values()) == 128


def test_tau2_performance_summary_reports_completion_p99():
    simulations = [
        {
            "messages": [
                {
                    "raw_data": {
                        "slime_fused_protocol_version": 2,
                        "slime_fused_total_seconds": seconds,
                        "slime_fused_completion_tokens": tokens,
                    }
                }
            ]
        }
        for seconds, tokens in ((1.0, 10), (2.0, 20), (3.0, 30))
    ]
    summary = tau2_launcher._performance_summary(simulations, 6.0)
    assert summary["agent_turns"] == 3
    assert summary["simulations_per_minute"] == 30
    assert summary["metrics"]["slime_fused_completion_tokens"]["p99"] == 30


def test_tau2_concurrency_sweep_requires_enough_simulations(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(tau2_launcher, "_python_version", lambda python: (3, 12))
    args.concurrency_sweep = "64,96,128,192"
    args.num_tasks = 24
    args.num_trials = 8
    tau2_launcher.validate(args)
    assert args.concurrency_sweep_values == [64, 96, 128, 192]

    args.num_trials = 1
    with pytest.raises(SystemExit, match="planned_simulations=24"):
        tau2_launcher.validate(args)


def test_tau2_domains_run_in_parallel_with_shared_budget(tmp_path: Path, monkeypatch):
    args = _args(tmp_path)
    args.domain = "all"
    args.domains = ["airline", "retail", "telecom"]
    args.max_concurrency = 64
    args.output_dir.mkdir()
    barrier = threading.Barrier(3)
    observed = {}

    def run(command, **kwargs):
        payload = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        settings = payload["settings"]
        observed[settings["domain"]] = settings["max_concurrency"]
        barrier.wait(timeout=2)
        result_dir = Path(settings["save_to"])
        result_dir.mkdir(parents=True)
        result = {
            "tasks": [{"id": "1"}],
            "simulations": [
                {
                    "task_id": "1",
                    "trial": 0,
                    "termination_reason": "user_stop",
                    "reward_info": {"reward": 1.0},
                    "messages": [
                        {
                            "role": "assistant",
                            "raw_data": {
                                "slime_fused_protocol_version": 2,
                                "slime_fused_session_id": "session-1",
                                "slime_fused_total_seconds": 1.0,
                            },
                        }
                    ],
                }
            ],
        }
        (result_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tau2_launcher.subprocess, "run", run)
    assert tau2_launcher.run_tau2(args, Path(sys.executable)) == 0
    assert observed == {"airline": 12, "retail": 26, "telecom": 26}
    summary = json.loads((args.output_dir / "performance_summary.json").read_text(encoding="utf-8"))
    assert set(summary["domains"]) == {"airline", "retail", "telecom"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
