from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from pathlib import Path


def _boolean(value: str) -> str:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return "true"
    if normalized in {"0", "false", "no", "off"}:
        return "false"
    raise argparse.ArgumentTypeError("must be a boolean")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tau2-bench against a local SGLang endpoint or OpenAI-compatible API")
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-series", choices=("openrouter", "qwen3", "qwen3.5", "gemma4"), required=True)
    parser.add_argument("--model-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--venv-dir", type=Path, required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--sglang-python-bin", required=True)
    parser.add_argument("--sglang-libstdcxx", type=Path)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--tp-size", type=_positive_int, required=True)
    parser.add_argument("--dp-size", type=_positive_int, required=True)
    parser.add_argument("--context-length", type=_positive_int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--port", type=_positive_int, default=18081)
    parser.add_argument("--wait-timeout", type=_positive_int, default=1800)
    parser.add_argument("--domain", default="all")
    parser.add_argument("--num-trials", type=_positive_int, default=1)
    parser.add_argument("--num-tasks", type=_non_negative_int, default=0)
    parser.add_argument("--max-steps", type=_positive_int, default=200)
    parser.add_argument("--max-retries", type=_non_negative_int, default=3)
    parser.add_argument("--max-concurrency", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=_positive_int, default=8192)
    parser.add_argument("--concurrency-sweep", default="")
    parser.add_argument("--enable-thinking", type=_boolean, default="false")
    parser.add_argument("--discard-historical-thinking", type=_boolean, default="false")
    parser.add_argument("--use-sglang-session", type=_boolean, default="true")
    parser.add_argument("--deterministic-inference", type=_boolean, default="false")
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--user-api-key")
    parser.add_argument("--user-max-tokens", type=_positive_int, default=4096)
    parser.add_argument("--evaluator-model")
    parser.add_argument("--evaluator-base-url")
    parser.add_argument("--evaluator-api-key")
    parser.add_argument("--overwrite", type=_boolean, default="false")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _truthy(value: str) -> bool:
    return value == "true"


def _domains(value: str) -> list[str]:
    supported = ("airline", "retail", "telecom")
    requested = supported if value.strip().lower() == "all" else tuple(
        item.strip().lower() for item in value.split(",") if item.strip()
    )
    invalid = sorted(set(requested) - set(supported))
    if not requested or invalid:
        raise SystemExit(f"--domain must be all or a comma-separated subset of {','.join(supported)}; invalid={invalid}")
    return list(dict.fromkeys(requested))


def _concurrency_sweep(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--concurrency-sweep must be a comma-separated list of positive integers") from exc
    if not values or any(value < 1 for value in values):
        raise SystemExit("--concurrency-sweep must be a comma-separated list of positive integers")
    return list(dict.fromkeys(values))


def _domain_concurrency(domains: list[str], total: int, num_tasks: int) -> dict[str, int]:
    if total < len(domains):
        raise SystemExit("--max-concurrency must be at least the number of selected tau2 domains")
    task_counts = {"airline": 50, "retail": 114, "telecom": 114}
    weights = {domain: min(task_counts[domain], num_tasks) if num_tasks else task_counts[domain] for domain in domains}
    remaining = total - len(domains)
    weight_sum = sum(weights.values())
    exact = {domain: remaining * weights[domain] / weight_sum for domain in domains}
    allocated = {domain: 1 + int(exact[domain]) for domain in domains}
    for domain in sorted(domains, key=lambda item: exact[item] - int(exact[item]), reverse=True):
        if sum(allocated.values()) == total:
            break
        allocated[domain] += 1
    return allocated


def _planned_simulations(domains: list[str], num_tasks: int, num_trials: int) -> int:
    task_counts = {"airline": 50, "retail": 114, "telecom": 114}
    tasks = sum(min(task_counts[domain], num_tasks) if num_tasks else task_counts[domain] for domain in domains)
    return tasks * num_trials


def _python_version(python: str | Path) -> tuple[int, int]:
    result = subprocess.run(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = result.stdout.strip().split(".")
    return int(major), int(minor)


def validate(args: argparse.Namespace) -> None:
    args.evaluator_model = args.evaluator_model or args.user_model
    args.evaluator_base_url = args.evaluator_base_url or args.user_base_url
    args.evaluator_api_key = args.evaluator_api_key or args.user_api_key
    if not (args.tau2_root / "pyproject.toml").is_file():
        raise SystemExit(f"tau2 pyproject.toml is missing under {args.tau2_root}")
    if not (args.tau2_root / "src" / "tau2" / "cli.py").is_file():
        raise SystemExit(f"tau2 source is incomplete under {args.tau2_root}")
    if args.model_series != "openrouter" and not Path(args.model).exists():
        raise SystemExit(f"Model does not exist: {args.model}")
    for name in ("python_bin", "sglang_python_bin"):
        path = Path(getattr(args, name))
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"Python is not executable: {path}")
    if args.sglang_libstdcxx is not None and not args.sglang_libstdcxx.is_file():
        raise SystemExit(f"SGLang libstdc++ does not exist: {args.sglang_libstdcxx}")
    if args.model_series != "openrouter" and args.tp_size * args.dp_size != len([x for x in args.cuda_visible_devices.split(",") if x]):
        raise SystemExit("tau2 requires tp-size * dp-size to equal the number of visible GPUs")
    if not 0 < args.mem_fraction_static <= 1:
        raise SystemExit("--mem-fraction-static must be in (0, 1]")
    if args.port > 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < -1 or args.top_k == 0:
        raise SystemExit("Invalid sampling parameters")
    if not args.user_api_key or not args.evaluator_api_key:
        raise SystemExit("tau2 user simulator and evaluator API keys are required")
    args.domains = _domains(args.domain)
    args.concurrency_sweep_values = _concurrency_sweep(args.concurrency_sweep)
    planned_simulations = _planned_simulations(args.domains, args.num_tasks, args.num_trials)
    if args.concurrency_sweep_values and planned_simulations < max(args.concurrency_sweep_values):
        raise SystemExit(
            "tau2 concurrency sweep cannot exercise its largest budget: "
            f"planned_simulations={planned_simulations}, largest_concurrency={max(args.concurrency_sweep_values)}; "
            "select more domains/tasks or increase --num-trials"
        )
    for concurrency in args.concurrency_sweep_values or [args.max_concurrency]:
        _domain_concurrency(args.domains, concurrency, args.num_tasks)
    try:
        version = _python_version(args.python_bin)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise SystemExit(f"Unable to query tau2 Python version: {exc}") from exc
    if not (version >= (3, 12) and version < (3, 14)):
        raise SystemExit(f"tau2 requires Python >=3.12,<3.14; got {version[0]}.{version[1]}")


def _project_dependencies(project_file: Path) -> list[str]:
    import tomllib

    excluded = ("torch", "transformers", "vllm", "rank-bm25")
    with project_file.open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]
    return [item for item in dependencies if not item.lower().replace("_", "-").startswith(excluded)]


def ensure_venv(args: argparse.Namespace) -> Path:
    python = args.venv_dir / "bin" / "python"
    dependencies = _project_dependencies(args.tau2_root / "pyproject.toml")
    fingerprint = hashlib.sha256((str(_python_version(args.python_bin)) + "\n" + "\n".join(dependencies)).encode()).hexdigest()
    marker = args.venv_dir / ".slime-tau2-ready"
    if python.is_file() and marker.is_file() and marker.read_text().strip() == fingerprint:
        import_check = subprocess.run(
            [str(python), "-c", "import deepdiff, litellm, loguru, pandas, pydantic, requests, toml, yaml"],
            capture_output=True,
        )
        if import_check.returncode == 0:
            return python
    print(f"Preparing isolated tau2 environment under {args.venv_dir}", flush=True)
    if args.venv_dir.exists():
        shutil.rmtree(args.venv_dir)
    subprocess.run([args.python_bin, "-m", "venv", str(args.venv_dir)], check=True)
    subprocess.run([str(python), "-m", "pip", "install", *dependencies], check=True)
    marker.write_text(fingerprint + "\n", encoding="utf-8")
    return python


def _endpoint_ready(url: str, expected_model: str) -> bool:
    request = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    return expected_model in {str(item.get("id")) for item in payload.get("data", [])}


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _resolve_sglang_libstdcxx(args: argparse.Namespace) -> Path | None:
    if args.sglang_libstdcxx is not None:
        candidate = args.sglang_libstdcxx.resolve()
    else:
        prefix = subprocess.check_output(
            [args.sglang_python_bin, "-c", "import sys; print(sys.prefix)"], text=True
        ).strip()
        candidate = (Path(prefix) / "lib" / "libstdc++.so.6").resolve()
    try:
        if b"GLIBCXX_3.4.32" in candidate.read_bytes():
            return candidate
    except OSError:
        pass
    if args.sglang_libstdcxx is not None:
        raise RuntimeError(f"SGLang libstdc++ lacks GLIBCXX_3.4.32: {candidate}")
    return None


def start_sglang(args: argparse.Namespace) -> tuple[subprocess.Popen[str], Path]:
    if not _port_is_available(args.port):
        raise RuntimeError(f"Port {args.port} is already in use; pass --tau2-port with a free port")
    log_path = args.output_dir / "sglang.log"
    command = [
        args.sglang_python_bin, "-m", "sglang.launch_server", "--model-path", str(args.model.resolve()),
        "--served-model-name", args.served_model_name, "--host", "127.0.0.1", "--port", str(args.port),
        "--tp-size", str(args.tp_size), "--dp-size", str(args.dp_size), "--context-length", str(args.context_length),
        "--mem-fraction-static", str(args.mem_fraction_static), "--api-key", "EMPTY", "--trust-remote-code",
        "--tool-call-parser", "qwen",
    ]
    if _truthy(args.enable_thinking):
        command.extend(["--reasoning-parser", "qwen3"])
    if _truthy(args.deterministic_inference):
        command.append("--enable-deterministic-inference")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    libstdcxx = _resolve_sglang_libstdcxx(args)
    if libstdcxx is not None:
        env["LD_PRELOAD"] = str(libstdcxx) + (f':{env["LD_PRELOAD"]}' if env.get("LD_PRELOAD") else "")
        print(f"SGLang C++ runtime: {libstdcxx}", flush=True)
    print("SGLang command:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True, text=True)
    deadline = time.monotonic() + args.wait_timeout
    try:
        while time.monotonic() < deadline:
            if _endpoint_ready(f"http://127.0.0.1:{args.port}/v1/models", args.served_model_name):
                return process, log_path
            if process.poll() is not None:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
                raise RuntimeError(f"SGLang exited before becoming ready. Log: {log_path}\n{tail}")
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for SGLang. Log: {log_path}")
    except BaseException:
        stop_process(process)
        raise


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _performance_summary(simulations: list[dict], elapsed_seconds: float) -> dict:
    metric_names = (
        "slime_fused_render_seconds",
        "slime_fused_tokenize_seconds",
        "slime_fused_inference_seconds",
        "slime_fused_total_seconds",
        "slime_fused_prompt_tokens",
        "slime_fused_completion_tokens",
        "slime_fused_session_delta_tokens",
        "slime_fused_session_tokens_avoided",
        "slime_fused_session_rollback_tokens",
        "slime_fused_session_open_seconds",
    )
    values = {name: [] for name in metric_names}
    for simulation in simulations:
        for message in simulation.get("messages") or []:
            raw_data = message.get("raw_data") or {}
            if raw_data.get("slime_fused_protocol_version") != 2:
                continue
            for name in metric_names:
                value = raw_data.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    values[name].append(float(value))
    metrics = {}
    for name, samples in values.items():
        if samples:
            metrics[name] = {
                "count": len(samples),
                "sum": sum(samples),
                "mean": sum(samples) / len(samples),
                "p50": _percentile(samples, 0.50),
                "p95": _percentile(samples, 0.95),
                "p99": _percentile(samples, 0.99),
                "max": max(samples),
            }
    return {
        "elapsed_seconds": elapsed_seconds,
        "simulations": len(simulations),
        "simulations_per_minute": len(simulations) * 60 / elapsed_seconds if elapsed_seconds > 0 else None,
        "agent_turns": len(values["slime_fused_total_seconds"]),
        "metrics": metrics,
    }


def run_tau2(args: argparse.Namespace, venv_python: Path) -> int:
    domains = getattr(args, "domains", _domains(args.domain))
    result_dirs = {domain: args.output_dir / domain for domain in domains}
    if _truthy(args.overwrite):
        for result_dir in result_dirs.values():
            if result_dir.exists():
                shutil.rmtree(result_dir)
    agent_args = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    user_args = {
        "base_url": args.user_base_url.rstrip("/"),
        "temperature": 0.0,
        "max_tokens": args.user_max_tokens,
    }
    bootstrap = """
import asyncio.base_events
import json
import os
import sys
import tau2.agent.llm_agent as llm_agent
import tau2.environment.utils.interface_agent as interface_agent
import tau2.evaluator.evaluator_nl_assertions as evaluator_nl_assertions
import tau2.runner.batch as batch
import tau2.user.user_simulator as user_simulator
import httpx
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import TextRunConfig
from tau2.run import run_domain
from slime_plugins.evals import tau2_fused_transport

def safe_event_loop_del(self, original_del=batch._original_del):
    try:
        original_del(self)
    except (AttributeError, TypeError):
        pass

asyncio.base_events.BaseEventLoop.__del__ = safe_event_loop_del

with open(sys.argv[1], encoding="utf-8") as file:
    payload = json.load(file)
tau2_fused_transport.configure(**payload["transport"])
llm_agent.AGENT_INSTRUCTION = tau2_fused_transport.AGENT_INSTRUCTION

def generate_external(config, api_key_env, *args, **kwargs):
    messages = kwargs.pop("messages", args[1] if len(args) > 1 else None)
    if messages is None:
        raise TypeError("tau2 external generation requires messages")
    tools = kwargs.pop("tools", None)
    serialized_messages = []
    for message in messages:
        item = {"role": message.role, "content": message.content or ""}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in tool_calls
            ]
        if message.role == "tool" and getattr(message, "id", None):
            item["tool_call_id"] = message.id
        serialized_messages.append(item)
    payload = {"model": config["model"], "messages": serialized_messages}
    if tools:
        payload["tools"] = [tool.openai_schema for tool in tools]
        payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
    for name in ("temperature", "top_p", "max_tokens", "response_format"):
        if name in kwargs and kwargs[name] is not None:
            payload[name] = kwargs[name]
    if config.get("max_tokens") is not None:
        payload.setdefault("max_tokens", config["max_tokens"])
    if config.get("json_mode"):
        payload["response_format"] = {"type": "json_object"}
    timeout = kwargs.get("timeout", 60)
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    response = httpx.post(
        endpoint,
        headers={"Authorization": f"Bearer {os.environ[api_key_env]}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]
    raw_tool_calls = message.get("tool_calls", [])
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if config.get("json_mode"):
        parsed_content = None
        for candidate in (content, message.get("reasoning"), message.get("reasoning_content")):
            if not isinstance(candidate, str):
                continue
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                parsed_content, _ = json.JSONDecoder().raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_content, dict):
                break
            parsed_content = None
        if parsed_content is None:
            raise ValueError("OpenRouter evaluator response contained no JSON object")
        content = json.dumps(parsed_content)
    elif (not isinstance(content, str) or not content.strip()) and not raw_tool_calls:
        raise ValueError("OpenRouter response contained no assistant text")
    tool_calls = []
    for call in raw_tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(parsed_arguments, dict):
            raise ValueError("OpenRouter returned non-object tool arguments")
        tool_calls.append(
            ToolCall(id=call.get("id", ""), name=function.get("name", ""), arguments=parsed_arguments)
        )
    tool_calls = tool_calls or None
    usage = data.get("usage")
    return AssistantMessage.text(
        content,
        tool_calls=tool_calls,
        usage=usage,
        raw_data=data,
    )

def generate_user(*args, **kwargs):
    message = generate_external(payload["user_llm"], "TAU2_USER_API_KEY", *args, **kwargs)
    return tau2_fused_transport.normalize_external_user_message(message)

def generate_evaluator(*args, **kwargs):
    return generate_external(payload["evaluator_llm"], "TAU2_EVALUATOR_API_KEY", *args, **kwargs)

def generate_agent(*args, **kwargs):
    return generate_external(payload["agent_llm"], "TAU2_AGENT_API_KEY", *args, **kwargs)

llm_agent.generate = generate_agent if payload["transport"].get("remote") else tau2_fused_transport.generate

user_simulator.generate = generate_user
evaluator_nl_assertions.generate = generate_evaluator
interface_agent.generate = generate_evaluator
original_run_single_task = batch.run_single_task

def run_single_task_with_session_cleanup(*args, **kwargs):
    simulation = None
    try:
        simulation = original_run_single_task(*args, **kwargs)
        if simulation.termination_reason.value == "user_error":
            raise RuntimeError("user simulator violated the tau2 communication protocol")
        # max_steps is a normal tau2 termination.  The official evaluator
        # records it with zero reward; converting it into an exception makes
        # the batch runner retry and finally misclassify it as an infrastructure
        # error with an empty trajectory.
        if simulation.termination_reason.value == "context_window_exceeded":
            raise RuntimeError("tau2 trajectory ended abnormally: context_window_exceeded")
        return simulation
    finally:
        if simulation:
            tau2_fused_transport.close_sessions(simulation.messages)
        tau2_fused_transport.close_thread_sessions()

batch.run_single_task = run_single_task_with_session_cleanup
run_domain(TextRunConfig(**payload["settings"]))
"""
    concurrency_by_domain = _domain_concurrency(domains, args.max_concurrency, args.num_tasks)
    settings = [
        {
            "domain": domain,
            "task_split_name": "base",
            "num_tasks": args.num_tasks or None,
            "agent": "llm_agent",
            "llm_agent": f"openai/{args.served_model_name}",
            "llm_args_agent": dict(agent_args),
            "user": "user_simulator",
            "llm_user": args.user_model,
            "llm_args_user": dict(user_args),
            "num_trials": args.num_trials,
            "max_steps": args.max_steps,
            "max_retries": args.max_retries,
            "max_concurrency": concurrency_by_domain[domain],
            "seed": args.seed,
            "save_to": str(result_dirs[domain].resolve()),
            "auto_resume": True,
            "enforce_communication_protocol": True,
        }
        for domain in domains
    ]
    transport = {
        "base_url": args.model_base_url.rstrip("/") if args.model_series == "openrouter" else f"http://127.0.0.1:{args.port}/v1",
        "model": args.served_model_name,
        "model_path": str(args.model.resolve()) if args.model_series != "openrouter" else str(args.model),
        "remote": args.model_series == "openrouter",
        "context_length": args.context_length,
        "dp_size": args.dp_size,
        "use_session": _truthy(args.use_sglang_session),
        "discard_historical_thinking": _truthy(args.discard_historical_thinking),
        "enable_thinking": _truthy(args.enable_thinking),
    }
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_BASE_URL", None)
    env["TAU2_USER_API_KEY"] = args.user_api_key
    env["TAU2_AGENT_API_KEY"] = args.user_api_key
    env["TAU2_EVALUATOR_API_KEY"] = args.evaluator_api_key or args.user_api_key
    env["TAU2_DATA_DIR"] = str((args.tau2_root / "data").resolve())
    env["FUSED_MODEL_SERIES"] = args.model_series
    slime_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((str(args.tau2_root / "src"), str(slime_root)))
    print(
        f"tau2 run: domains={','.join(domains)}, tasks={args.num_tasks or 'all'}, trials={args.num_trials}, "
        f"retries={args.max_retries}, "
        f"global_concurrency={args.max_concurrency}, domain_concurrency={concurrency_by_domain}",
        flush=True,
    )
    config_paths = {}
    commands = {}
    try:
        for setting in settings:
            domain = setting["domain"]
            config_path = args.output_dir / f".tau2-run-config-{os.getpid()}-{domain}-{uuid.uuid4().hex}.json"
            config_path.write_text(
                json.dumps(
                    {
                        "settings": setting,
                        "transport": transport,
                        "agent_llm": {"model": str(args.model), "base_url": args.model_base_url.rstrip("/"), "max_tokens": args.max_tokens},
                        "user_llm": {"model": args.user_model, "base_url": args.user_base_url.rstrip("/")},
                        "evaluator_llm": {
                            "model": args.evaluator_model or args.user_model,
                            "base_url": (args.evaluator_base_url or args.user_base_url).rstrip("/"),
                            "json_mode": True,
                            "max_tokens": args.user_max_tokens,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path.chmod(0o600)
            config_paths[domain] = config_path
            commands[domain] = [str(venv_python), "-c", bootstrap, str(config_path)]

        domain_elapsed = {}

        def run_domain_process(domain: str) -> int:
            started = time.perf_counter()
            try:
                return subprocess.run(commands[domain], cwd=args.tau2_root, env=env).returncode
            finally:
                domain_elapsed[domain] = time.perf_counter() - started

        with ThreadPoolExecutor(max_workers=len(domains)) as executor:
            futures = {executor.submit(run_domain_process, domain): domain for domain in domains}
            return_codes = {futures[future]: future.result() for future in as_completed(futures)}
    finally:
        for config_path in config_paths.values():
            config_path.unlink(missing_ok=True)
    failed_domains = {domain: code for domain, code in return_codes.items() if code}
    if failed_domains:
        print(f"tau2 domain subprocesses failed: {failed_domains}", file=sys.stderr)
        return next(iter(failed_domains.values()))
    performance = {
        "global_concurrency": args.max_concurrency,
        "domain_concurrency": concurrency_by_domain,
        "max_tokens": args.max_tokens,
        "domains": {},
    }
    for domain, result_dir in result_dirs.items():
        result_path = result_dir / "results.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            simulations = payload["simulations"]
            tasks = payload["tasks"]
            task_ids = {str(task["id"]) for task in tasks}
            expected_runs = {(task_id, trial) for task_id in task_ids for trial in range(args.num_trials)}
            actual_runs = [(str(item["task_id"]), int(item["trial"])) for item in simulations]
            infra_errors = [item for item in simulations if item.get("termination_reason") == "infrastructure_error"]
            user_errors = [item for item in simulations if item.get("termination_reason") == "user_error"]
            rewards = [
                reward_info.get("reward") if isinstance(reward_info := item.get("reward_info"), dict) else None
                for item in simulations
            ]
            missing_rewards = sum(value is None for value in rewards)
            protocol_missing = 0
            session_missing = 0
            session_fallbacks = 0
            for simulation in simulations:
                fused_messages = [
                    message
                    for message in simulation.get("messages") or []
                    if (message.get("raw_data") or {}).get("slime_fused_protocol_version") == 2
                ]
                if args.model_series != "openrouter" and not fused_messages:
                    protocol_missing += 1
                if args.model_series != "openrouter" and _truthy(args.use_sglang_session) and not any(
                    (message.get("raw_data") or {}).get("slime_fused_session_id") for message in fused_messages
                ):
                    session_missing += 1
                session_fallbacks += sum(
                    1
                    for message in fused_messages
                    if (message.get("raw_data") or {}).get("slime_fused_session_error")
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"Unable to validate tau2 results at {result_path}: {exc}", file=sys.stderr)
            return 1
        actual_set = set(actual_runs)
        if (
            not tasks
            or len(task_ids) != len(tasks)
            or len(actual_runs) != len(actual_set)
            or actual_set != expected_runs
            or infra_errors
            or user_errors
            or protocol_missing
            or session_missing
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in rewards
            )
        ):
            print(
                f"tau2 result validation failed for {domain}: completed={len(actual_runs)}/{len(expected_runs)}, "
                f"tasks={len(tasks)}, "
                f"duplicates={len(actual_runs) - len(actual_set)}, missing={len(expected_runs - actual_set)}, "
                f"unexpected={len(actual_set - expected_runs)}, infrastructure_errors={len(infra_errors)}, "
                f"user_errors={len(user_errors)}, "
                f"missing_rewards={missing_rewards}, "
                f"protocol_missing={protocol_missing}, session_missing={session_missing}",
                file=sys.stderr,
            )
            return 1
        print(
            f"tau2 complete: domain={domain}, reward={sum(rewards) / len(rewards):.4f}, "
            f"runs={len(rewards)}, session_fallbacks={session_fallbacks}"
        )
        performance["domains"][domain] = _performance_summary(simulations, domain_elapsed[domain])
        domain_performance_path = result_dir / "performance.json"
        domain_performance_path.write_text(
            json.dumps(performance["domains"][domain], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (args.output_dir / "performance_summary.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.user_api_key = args.user_api_key or os.environ.get("TAU2_USER_API_KEY")
    args.evaluator_model = args.evaluator_model or args.user_model
    args.evaluator_base_url = args.evaluator_base_url or args.user_base_url
    args.evaluator_api_key = (
        args.evaluator_api_key or os.environ.get("TAU2_EVALUATOR_API_KEY") or args.user_api_key
    )
    args.python_bin = str(Path(args.python_bin).resolve())
    args.sglang_python_bin = str(Path(args.sglang_python_bin).resolve())
    validate(args)
    if args.preflight_only:
        sweep = args.concurrency_sweep_values or [args.max_concurrency]
        allocations = {value: _domain_concurrency(args.domains, value, args.num_tasks) for value in sweep}
        print(f"tau2 preflight complete: transport=slime_fused_gem/v2, session={args.use_sglang_session}, domains={','.join(args.domains)}, tasks={'all' if not args.num_tasks else args.num_tasks}, trials={args.num_trials}, retries={args.max_retries}, concurrency={sweep}, allocations={allocations}, max_tokens={args.max_tokens}, dp={args.dp_size}, tp={args.tp_size}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    venv_python = ensure_venv(args)
    process = None
    try:
        if args.model_series != "openrouter":
            process, _ = start_sglang(args)
        sweep = args.concurrency_sweep_values or [args.max_concurrency]
        sweep_summaries = {}
        for concurrency in sweep:
            run_args = copy(args)
            run_args.max_concurrency = concurrency
            if args.concurrency_sweep_values:
                run_args.output_dir = args.output_dir / "concurrency-sweep" / f"c{concurrency}"
                run_args.output_dir.mkdir(parents=True, exist_ok=True)
            return_code = run_tau2(run_args, venv_python)
            if return_code:
                return return_code
            if args.concurrency_sweep_values:
                sweep_summaries[str(concurrency)] = json.loads(
                    (run_args.output_dir / "performance_summary.json").read_text(encoding="utf-8")
                )
        if args.concurrency_sweep_values:
            (args.output_dir / "concurrency_sweep_summary.json").write_text(
                json.dumps(sweep_summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return 0
    finally:
        if process is not None:
            stop_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
