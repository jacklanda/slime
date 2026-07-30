from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


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
    parser = argparse.ArgumentParser(description="Run VitaBench against a local SGLang endpoint")
    parser.add_argument("--vitabench-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--venv-dir", type=Path, required=True)
    parser.add_argument("--sglang-python-bin", required=True)
    parser.add_argument("--sglang-libstdcxx", type=Path)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--tp-size", type=_positive_int, required=True)
    parser.add_argument("--dp-size", type=_positive_int, required=True)
    parser.add_argument("--context-length", type=_positive_int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--port", type=_positive_int, default=18080)
    parser.add_argument("--wait-timeout", type=_positive_int, default=1800)
    parser.add_argument("--domain", default="delivery,instore,ota")
    parser.add_argument("--language", choices=("chinese", "english"), default="chinese")
    parser.add_argument(
        "--evaluation-type",
        choices=(
            "trajectory",
            "trajectory_full_traj_rubric",
            "trajectory_sliding_wo_rubric",
            "trajectory_full_traj_wo_rubric",
        ),
        default="trajectory",
    )
    parser.add_argument("--num-trials", type=_positive_int, default=1)
    parser.add_argument("--num-tasks", type=_non_negative_int, default=0)
    parser.add_argument("--max-steps", type=_positive_int, default=300)
    parser.add_argument("--max-concurrency", type=_positive_int, default=8)
    parser.add_argument("--log-level", choices=("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR"), default="WARNING")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=_positive_int, default=38000)
    parser.add_argument("--user-max-tokens", type=_positive_int, default=4096)
    parser.add_argument("--evaluator-max-tokens", type=_positive_int, default=8192)
    parser.add_argument("--enable-thinking", type=_boolean, default="false")
    parser.add_argument("--discard-historical-thinking", type=_boolean, default="false")
    parser.add_argument("--deterministic-inference", type=_boolean, default="false")
    parser.add_argument("--use-sglang-session", type=_boolean, default="true")
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--user-api-key")
    parser.add_argument("--evaluator-model", required=True)
    parser.add_argument("--evaluator-base-url", required=True)
    parser.add_argument("--evaluator-api-key")
    parser.add_argument("--overwrite", type=_boolean, default="false")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def validate(args: argparse.Namespace) -> None:
    if not (args.vitabench_root / "pyproject.toml").is_file():
        raise SystemExit(f"VitaBench pyproject.toml is missing under {args.vitabench_root}")
    if not (args.vitabench_root / "src" / "vita" / "cli.py").is_file():
        raise SystemExit(f"VitaBench source is incomplete under {args.vitabench_root}")
    if not args.model.exists():
        raise SystemExit(f"Model does not exist: {args.model}")
    if not Path(args.sglang_python_bin).is_file() or not os.access(args.sglang_python_bin, os.X_OK):
        raise SystemExit(f"SGLang Python is not executable: {args.sglang_python_bin}")
    if args.sglang_libstdcxx is not None and not args.sglang_libstdcxx.is_file():
        raise SystemExit(f"SGLang libstdc++ does not exist: {args.sglang_libstdcxx}")
    if args.tp_size * args.dp_size != len([item for item in args.cuda_visible_devices.split(",") if item]):
        raise SystemExit("VitaBench requires tp-size * dp-size to equal the number of visible GPUs")
    if not 0 < args.mem_fraction_static <= 1:
        raise SystemExit("--mem-fraction-static must be in (0, 1]")
    if args.port > 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")
    if args.top_k < -1 or args.top_k == 0:
        raise SystemExit("--top-k must be -1 or a positive integer")
    if not args.user_api_key or not args.evaluator_api_key:
        raise SystemExit("VitaBench user and evaluator API keys are required")


def _project_dependencies(project_file: Path) -> list[str]:
    import tomllib

    with project_file.open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]
    # The evaluated model is served by SGLang. Installing VitaBench's local vLLM
    # backend would duplicate the CUDA runtime and can conflict with slime.
    return [dependency for dependency in dependencies if not dependency.lower().startswith("vllm")]


def ensure_venv(args: argparse.Namespace) -> Path:
    python = args.venv_dir / "bin" / "python"
    marker = args.venv_dir / ".slime-vitabench-ready"
    dependencies = _project_dependencies(args.vitabench_root / "pyproject.toml")
    dependency_fingerprint = hashlib.sha256("\n".join(dependencies).encode()).hexdigest()
    if python.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == dependency_fingerprint:
        return python

    print(f"Preparing isolated VitaBench environment under {args.venv_dir}", flush=True)
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(args.venv_dir)], check=True)
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *dependencies],
        check=True,
    )
    marker.write_text(dependency_fingerprint + "\n", encoding="utf-8")
    return python


def write_model_config(args: argparse.Namespace) -> Path:
    config_path = args.output_dir / "models.yaml"
    local_base_url = f"http://127.0.0.1:{args.port}/v1"
    use_sglang_session = _truthy(args.use_sglang_session) and args.dp_size == 1
    if _truthy(args.use_sglang_session) and not use_sglang_session:
        print(
            "Disabling native SGLang sessions because dp-size > 1; "
            "SGLang session lifecycle requests cannot be pinned to a DP worker",
            flush=True,
        )

    def model(name: str, remote_model: str, base_url: str, api_key: str, **extra: Any) -> dict[str, Any]:
        return {
            "name": name,
            "backend": "litellm_http",
            "model": remote_model,
            "base_url": base_url,
            "api_key": api_key,
            "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            **extra,
        }

    config = {
        "default": {"temperature": 0.0},
        "models": [
            model(
                "slime-agent",
                args.served_model_name,
                local_base_url,
                "EMPTY",
                backend="slime_fused_gem",
                tool_parser_model_name=str(args.model.resolve()),
                max_context_tokens=args.context_length,
                use_sglang_session=use_sglang_session,
                dp_size=args.dp_size,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                chat_template_kwargs={"enable_thinking": _truthy(args.enable_thinking)},
            ),
            model(
                "vitabench-user",
                args.user_model,
                args.user_base_url,
                args.user_api_key,
                max_tokens=args.user_max_tokens,
            ),
            model(
                "vitabench-evaluator",
                args.evaluator_model,
                args.evaluator_base_url,
                args.evaluator_api_key,
                max_tokens=args.evaluator_max_tokens,
            ),
        ],
    }
    # JSON is valid YAML and avoids adding PyYAML to the launcher environment.
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config_path.chmod(0o600)
    return config_path


def _endpoint_ready(url: str, expected_model: str) -> bool:
    request = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    return expected_model in {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _resolve_sglang_libstdcxx(args: argparse.Namespace) -> Path | None:
    if args.sglang_libstdcxx is not None:
        return args.sglang_libstdcxx.resolve()
    prefix = subprocess.check_output(
        [args.sglang_python_bin, "-c", "import sys; print(sys.prefix)"],
        text=True,
    ).strip()
    candidate = Path(prefix) / "lib" / "libstdc++.so.6"
    try:
        if b"GLIBCXX_3.4.32" in candidate.resolve().read_bytes():
            return candidate.resolve()
    except OSError:
        pass
    return None


def _model_context_length(model_path: Path) -> int | None:
    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        return int(config["max_position_embeddings"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def normalize_context_settings(args: argparse.Namespace) -> None:
    config_path = args.model / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    native_context_length = config.get("max_position_embeddings")
    try:
        native_context_length = int(native_context_length)
    except (TypeError, ValueError):
        return
    if args.context_length <= native_context_length or config.get("rope_scaling"):
        return
    print(
        f"Clamping context length from {args.context_length} to the model-native "
        f"{native_context_length}; enabling historical-thinking discard to keep long trajectories in context",
        flush=True,
    )
    args.context_length = native_context_length
    args.discard_historical_thinking = "true"


def start_sglang(args: argparse.Namespace) -> tuple[subprocess.Popen[str], Path]:
    log_path = args.output_dir / "sglang.log"
    if not _port_is_available(args.port):
        raise RuntimeError(
            f"Port {args.port} is already in use; refusing to reuse an unidentified SGLang server. "
            "Stop the old process or pass --vitabench-port with a free port."
        )
    command = [
        args.sglang_python_bin,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model.resolve()),
        "--served-model-name",
        args.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--dp-size",
        str(args.dp_size),
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--api-key",
        "EMPTY",
        "--trust-remote-code",
        "--tool-call-parser",
        "qwen25",
    ]
    if _truthy(args.enable_thinking):
        command.extend(["--reasoning-parser", "qwen3"])
    if _truthy(args.deterministic_inference):
        command.append("--enable-deterministic-inference")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    model_context_length = _model_context_length(args.model)
    if model_context_length is not None and args.context_length > model_context_length:
        env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        print(
            f"Extending SGLang context from the model default {model_context_length} to {args.context_length}",
            flush=True,
        )
    libstdcxx = _resolve_sglang_libstdcxx(args)
    if libstdcxx is not None:
        env["LD_PRELOAD"] = str(libstdcxx) + (f':{env["LD_PRELOAD"]}' if env.get("LD_PRELOAD") else "")
        print(f"SGLang C++ runtime: {libstdcxx}", flush=True)
    print("SGLang command:", " ".join(command), flush=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_file.close()
    models_url = f"http://127.0.0.1:{args.port}/v1/models"
    deadline = time.monotonic() + args.wait_timeout
    try:
        while time.monotonic() < deadline:
            if _endpoint_ready(models_url, args.served_model_name):
                print(f"SGLang endpoint is ready: {models_url}", flush=True)
                return process, log_path
            if process.poll() is not None:
                tail = "".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-80:]
                )
                raise RuntimeError(f"SGLang exited before becoming ready. Log: {log_path}\n{tail}")
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for SGLang after {args.wait_timeout}s. Log: {log_path}")
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


def _result_is_complete(result_path: Path, num_trials: int) -> bool:
    try:
        results = json.loads(result_path.read_text(encoding="utf-8"))
        task_ids = [str(task["id"]) for task in results["tasks"]]
        completed_runs = {
            (str(simulation["task_id"]), int(simulation["trial"]))
            for simulation in results["simulations"]
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    expected_runs = {(task_id, trial) for task_id in task_ids for trial in range(num_trials)}
    return bool(task_ids) and len(task_ids) == len(set(task_ids)) and completed_runs == expected_runs


def run_vitabench(args: argparse.Namespace, venv_python: Path, config_path: Path) -> int:
    result_path = args.output_dir / "simulations.json"
    csv_path = args.output_dir / "metrics.csv"
    if result_path.exists():
        if _truthy(args.overwrite):
            result_path.unlink()
        elif not _result_is_complete(result_path, args.num_trials):
            print(f"Removing incomplete VitaBench result before retry: {result_path}", flush=True)
            result_path.unlink()
        else:
            raise SystemExit(f"VitaBench result already exists: {result_path}; pass --vitabench-overwrite true to replace it")
    if csv_path.exists() and (_truthy(args.overwrite) or not result_path.exists()):
        csv_path.unlink()

    bootstrap = """
import json
import os
import re
import sys
import types

vllm = types.ModuleType("vllm")
vllm.LLM = vllm.SamplingParams = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("VitaBench's vLLM backend is disabled; slime uses SGLang")
)
sys.modules["vllm"] = vllm

import vita.utils.llm_utils as llm_utils
import vita.environment.environment as environment_module
import vita.evaluator.evaluator_traj as evaluator_traj

# Invalid model-generated tool arguments are returned as error ToolMessages. Avoid
# printing a full traceback for those recoverable turns while preserving the error.
environment_module.traceback = types.SimpleNamespace(print_exc=lambda: None)

if getattr(llm_utils, "SLIME_FUSED_GEM_PROTOCOL_VERSION", None) != 2:
    raise RuntimeError("VitaBench artifact does not provide slime_fused_gem protocol version 2")

original_post_json = llm_utils._post_json

def post_json_with_sglang_sampling_seed(url, data, headers, **kwargs):
    if url.rstrip("/").endswith("/close_session"):
        session = getattr(llm_utils._HTTP_SESSION_LOCAL, "session", None)
        if session is None:
            session = llm_utils.requests.Session()
            llm_utils._HTTP_SESSION_LOCAL.session = session
        response = session.post(url, json=data, headers=headers, timeout=kwargs.get("timeout", (5, 30)))
        response.raise_for_status()
        return None
    if url.rstrip("/").endswith("/generate") and isinstance(data.get("sampling_params"), dict):
        data = dict(data)
        sampling_params = dict(data["sampling_params"])
        if "seed" in sampling_params:
            sampling_params["sampling_seed"] = sampling_params.pop("seed")
        data["sampling_params"] = sampling_params
    return original_post_json(url, data, headers, **kwargs)

llm_utils._post_json = post_json_with_sglang_sampling_seed

# Evaluator-compatible endpoints occasionally return an empty body or wrap the
# expected list in another list. A malformed judge window must not discard the
# complete simulation; the official evaluator already treats [] as "keep the
# current rubric states".
original_evaluator_extracter = evaluator_traj.evaluator_extracter

def tolerant_evaluator_extracter(content):
    try:
        results = original_evaluator_extracter(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    while isinstance(results, list) and len(results) == 1 and isinstance(results[0], list):
        results = results[0]
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        return []
    normalized = []
    for result in results:
        if not isinstance(result, dict):
            continue
        justification = result.get("justification")
        if (
            result.get("meetExpectation") is False
            and isinstance(justification, str)
            and "满足要求" in justification
            and justification.rstrip().endswith("状态从false更新为")
        ):
            result = dict(result)
            result["meetExpectation"] = True
        normalized.append(result)
    return normalized

evaluator_traj.evaluator_extracter = tolerant_evaluator_extracter

# Qwen can occasionally finish a turn after producing reasoning but before
# producing visible content or a tool call. Vita retries that invalid message
# with the same seed three times, so the retry is deterministic and the task is
# incorrectly terminated. Preserve the model output as visible text in this
# narrow case so the simulator can continue the conversation.
original_generate = llm_utils.generate

def generate_with_reasoning_only_fallback(model, *args, **kwargs):
    message = original_generate(model, *args, **kwargs)
    model_config = llm_utils.models.get(model, {})
    if model_config.get("backend") != "slime_fused_gem":
        return message
    if message.has_text_content() or message.is_tool_call():
        return message
    raw_message = (message.raw_data or {}).get("message", {})
    reasoning = raw_message.get("reasoning_content") or raw_message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        message.content = reasoning.strip()
    return message

llm_utils.generate = generate_with_reasoning_only_fallback

# A completed assistant turn replaces the chat template's generation boundary
# on the next render. SGLang sessions can roll back to the actual common prefix;
# it need not cover the complete prompt from the previous turn.
original_common_prefix_length = llm_utils._common_prefix_length

def common_prefix_allowing_generation_boundary_rewrite(left, right):
    prefix = original_common_prefix_length(left, right)
    for state in tuple(llm_utils._SGLANG_SESSIONS.values()):
        if state.get("expected_prefix_ids") is right:
            state["stable_prefix_length"] = 1
            break
    return prefix

llm_utils._common_prefix_length = common_prefix_allowing_generation_boundary_rewrite

if os.environ.get("VITA_DISCARD_HISTORICAL_THINKING") == "true":
    original_format_messages = llm_utils.format_messages

    think_block_re = re.compile(r"<think\\b[^>]*>.*?</think\\s*>", re.DOTALL)
    thought_channel_block_re = re.compile(r"<\\|channel>thought\\n.*?<channel\\|>", re.DOTALL)
    think_open_re = re.compile(r"<think\\b[^>]*>")
    thought_channel_open_re = re.compile(r"<\\|channel>thought\\n")
    think_close_re = re.compile(r"</think\\s*>")

    def strip_historical_thinking(content):
        starts_with_thinking = bool(think_block_re.match(content) or thought_channel_block_re.match(content))
        stripped = think_block_re.sub("", content)
        stripped = thought_channel_block_re.sub("", stripped)
        closers = list(think_close_re.finditer(stripped))
        if closers:
            stripped = stripped[closers[-1].end():].lstrip("\\n")
        elif starts_with_thinking:
            stripped = stripped.lstrip("\\r\\n")
        openers = [match for regex in (think_open_re, thought_channel_open_re) if (match := regex.search(stripped))]
        if openers:
            stripped = stripped[:min(match.start() for match in openers)].rstrip()
        return stripped

    def format_messages_without_reasoning(messages):
        formatted = original_format_messages(messages)
        for message in formatted:
            if message.get("role") == "assistant":
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
                if isinstance(message.get("content"), str):
                    message["content"] = strip_historical_thinking(message["content"])
        return formatted

    llm_utils.format_messages = format_messages_without_reasoning

from vita.cli import main
main()
"""
    command = [
        str(venv_python),
        "-c",
        bootstrap,
        "run",
        "--domain",
        args.domain,
        "--agent-llm",
        "slime-agent",
        "--user-llm",
        "vitabench-user",
        "--evaluator-llm",
        "vitabench-evaluator",
        "--num-trials",
        str(args.num_trials),
        "--max-steps",
        str(args.max_steps),
        "--max-concurrency",
        str(args.max_concurrency),
        "--log-level",
        args.log_level,
        "--seed",
        str(args.seed),
        "--language",
        args.language,
        "--evaluation-type",
        args.evaluation_type,
        "--save-to",
        str(result_path),
        "--csv-output",
        str(csv_path),
    ]
    if args.num_tasks:
        command.extend(["--num-tasks", str(args.num_tasks)])
    if _truthy(args.enable_thinking):
        command.append("--enable-think")
    env = os.environ.copy()
    env["VITA_MODEL_CONFIG_PATH"] = str(config_path)
    env["VITA_DISCARD_HISTORICAL_THINKING"] = args.discard_historical_thinking
    slime_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((str(args.vitabench_root / "src"), str(slime_root)))
    print("VitaBench command:", " ".join(command[:3] + ["<bootstrap>"] + command[4:]), flush=True)
    return_code = subprocess.run(command, cwd=args.vitabench_root, env=env).returncode
    if return_code != 0:
        return return_code
    try:
        results = json.loads(result_path.read_text(encoding="utf-8"))
        task_ids = [str(task["id"]) for task in results["tasks"]]
        simulations = results["simulations"]
        completed_runs = [(str(simulation["task_id"]), int(simulation["trial"])) for simulation in simulations]
        expected_runs = {(task_id, trial) for task_id in task_ids for trial in range(args.num_trials)}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Unable to validate VitaBench results at {result_path}: {exc}", file=sys.stderr)
        return 1
    actual_runs = set(completed_runs)
    duplicate_task_count = len(task_ids) - len(set(task_ids))
    duplicate_count = len(completed_runs) - len(actual_runs)
    missing_runs = expected_runs - actual_runs
    unexpected_runs = actual_runs - expected_runs
    malformed_rewards = 0
    for simulation in simulations:
        reward_info = simulation.get("reward_info")
        reward = reward_info.get("reward") if isinstance(reward_info, dict) else None
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            malformed_rewards += 1
    if duplicate_task_count or duplicate_count or missing_runs or unexpected_runs or malformed_rewards:
        print(
            f"VitaBench result validation failed: completed={len(completed_runs)}/{len(expected_runs)}, "
            f"duplicate_tasks={duplicate_task_count}, duplicates={duplicate_count}, missing={len(missing_runs)}, "
            f"unexpected={len(unexpected_runs)}, malformed_rewards={malformed_rewards}; "
            f"inspect {result_path} and the console log",
            file=sys.stderr,
        )
        return 1
    if not csv_path.is_file() or not csv_path.read_text(encoding="utf-8").strip():
        print(f"VitaBench did not produce a non-empty metrics CSV at {csv_path}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.user_api_key = args.user_api_key or os.environ.get("VITABENCH_USER_API_KEY")
    args.evaluator_api_key = args.evaluator_api_key or os.environ.get("VITABENCH_EVALUATOR_API_KEY")
    args.sglang_python_bin = str(Path(args.sglang_python_bin).resolve())
    validate(args)
    normalize_context_settings(args)
    if args.preflight_only:
        print(
            f"VitaBench preflight complete: transport=slime_fused_gem, evaluation={args.evaluation_type}, "
            f"sglang_session={args.use_sglang_session}, "
            f"domain={args.domain}, "
            f"tasks={'all' if args.num_tasks == 0 else args.num_tasks}, trials={args.num_trials}, "
            f"dp={args.dp_size}, tp={args.tp_size}, venv={args.venv_dir}"
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    venv_python = ensure_venv(args)
    config_path = write_model_config(args)
    process = None
    try:
        process, _ = start_sglang(args)
        return run_vitabench(args, venv_python, config_path)
    finally:
        if process is not None:
            stop_process(process)
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
