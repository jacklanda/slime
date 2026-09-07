"""Run the official WorkBench inference/scoring loop against slime's SGLang server.

The benchmark is intentionally executed in the existing slime Python environment.  WorkBench's
pure-Python dependencies (openai/pandas) are already part of the supported runtime; this launcher
never creates an environment or installs packages.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any


def _bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("must be a boolean")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official WorkBench with slime/SGLang")
    parser.add_argument("--workbench-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--domains", default="all")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--enable-thinking", type=_bool, default=False)
    parser.add_argument("--overwrite", type=_bool, default=False)
    parser.add_argument("--structured-outputs", type=_bool, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _ready(url: str, model: str) -> bool:
    try:
        request = urllib.request.Request(
            url,
            headers={"Authorization": "Bearer EMPTY"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.load(response)
        return model in {str(item.get("id")) for item in payload.get("data", [])}
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _resolve_sglang_libstdcxx() -> Path | None:
    """Find the conda/runtime libstdc++ required by FlashInfer extensions.

    Some cluster environments prepend an older GCC module library to
    ``LD_LIBRARY_PATH``.  FlashInfer wheels compiled with newer toolchains
    then fail during CUDA-graph capture with a misleading scheduler crash.
    """
    try:
        prefix = subprocess.check_output(
            [sys.executable, "-c", "import sys; print(sys.prefix)"],
            text=True,
        ).strip()
        candidate = (Path(prefix) / "lib" / "libstdc++.so.6").resolve()
        if candidate.is_file() and b"GLIBCXX_3.4.32" in candidate.read_bytes():
            return candidate
    except (OSError, subprocess.CalledProcessError):
        pass
    return None


def _start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[str]:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"port {args.port} is already in use")
    command = [
        sys.executable,
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
    if args.max_running_requests is not None:
        command += ["--max-running-requests", str(args.max_running_requests)]
    if args.enable_thinking:
        command += ["--reasoning-parser", "qwen3"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    libstdcxx = _resolve_sglang_libstdcxx()
    if libstdcxx is not None:
        env["LD_PRELOAD"] = str(libstdcxx) + (f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else "")
        print(f"SGLang C++ runtime: {libstdcxx}", flush=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True, text=True)
    log_file.close()
    deadline = time.monotonic() + 1800
    try:
        while time.monotonic() < deadline:
            if _ready(f"http://127.0.0.1:{args.port}/v1/models", args.served_model_name):
                return process
            if process.poll() is not None:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
                raise RuntimeError(f"SGLang exited before readiness; see {log_path}\n{tail}")
            time.sleep(3)
        raise TimeoutError(f"timed out waiting for SGLang; see {log_path}")
    except BaseException:
        _stop(process)
        raise


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _task_files(root: Path, domains: str) -> list[Path]:
    files = sorted((root / "data" / "processed" / "tasks_and_outcomes").glob("*_tasks_and_outcomes.csv"))
    if domains == "all":
        return files
    selected = {item.strip().lower() for item in domains.split(",") if item.strip()}
    return [path for path in files if path.stem.removesuffix("_tasks_and_outcomes") in selected]


def _print_results_table(scored_domains: list[tuple[str, Any]]) -> None:
    columns = ("Domain", "Tasks", "Correct", "Accuracy", "Exact match", "Side effects", "Run errors")
    rows: list[tuple[str, ...]] = []
    totals = {"tasks": 0, "correct": 0, "exact": 0, "side_effects": 0, "errors": 0}

    for domain, scored in scored_domains:
        tasks = len(scored)
        correct = int(scored["correct"].sum())
        exact = int(scored["exact_match"].sum())
        side_effects = int(scored["unwanted_side_effects"].sum())
        errors = int(scored["error"].astype(bool).sum())
        rows.append(
            (
                domain,
                str(tasks),
                str(correct),
                f"{correct / tasks:.2%}" if tasks else "n/a",
                str(exact),
                str(side_effects),
                str(errors),
            )
        )
        totals["tasks"] += tasks
        totals["correct"] += correct
        totals["exact"] += exact
        totals["side_effects"] += side_effects
        totals["errors"] += errors

    tasks = totals["tasks"]
    rows.append(
        (
            "Overall",
            str(tasks),
            str(totals["correct"]),
            f"{totals['correct'] / tasks:.2%}" if tasks else "n/a",
            str(totals["exact"]),
            str(totals["side_effects"]),
            str(totals["errors"]),
        )
    )
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]

    def format_row(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nWorkBench evaluation results", flush=True)
    print(separator)
    print(format_row(columns))
    print(separator)
    for index, row in enumerate(rows):
        if index == len(rows) - 1:
            print(separator)
        print(format_row(row))
    print(separator, flush=True)


def run(args: argparse.Namespace) -> int:
    root = args.workbench_root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "evals" / "inference.py").is_file():
        raise SystemExit(f"WorkBench source is incomplete under {root}")
    if not args.model.is_dir():
        raise SystemExit(f"Model does not exist: {args.model}")
    if args.tp_size < 1 or args.dp_size < 1 or args.workers < 1 or args.limit < 0:
        raise SystemExit("tp-size, dp-size, and workers must be positive; limit must be non-negative")
    if args.max_running_requests is not None and args.max_running_requests < 1:
        raise SystemExit("max-running-requests must be positive")
    if args.tp_size * args.dp_size != len([x for x in args.cuda_visible_devices.split(",") if x]):
        raise SystemExit("tp-size * dp-size must equal the number of visible GPUs")
    task_files = _task_files(root, args.domains)
    if not task_files:
        raise SystemExit(f"No WorkBench task files selected by --domains {args.domains!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        print(f"WorkBench preflight complete: {len(task_files)} task files; output={args.output_dir}")
        return 0

    process = _start_server(args, args.output_dir / "sglang.log")
    try:
        sys.path.insert(0, str(root))
        from src.evals import agent, inference, metrics
        from src.evals.agent import ModelConfig, Route
        import pandas as pd

        agent._REQUEST_TIMEOUT_SECONDS = max(
            float(agent._REQUEST_TIMEOUT_SECONDS),
            float(args.max_tokens) / 50.0 + 60.0,
        )
        agent._HARD_DEADLINE_SECONDS = max(
            float(agent._HARD_DEADLINE_SECONDS),
            float(args.max_tokens) / 50.0 + 60.0,
        )

        route = Route(args.served_model_name, f"http://127.0.0.1:{args.port}/v1", "EMPTY", "slime_sglang", True)
        agent.MODEL_REGISTRY["slime-agent"] = ModelConfig(args.served_model_name, True, "openrouter")
        original_resolve_route = agent.resolve_route
        agent.resolve_route = lambda name: route if name == "slime-agent" else original_resolve_route(name)  # type: ignore[method-assign]
        inference.resolve_route = agent.resolve_route
        original_create = agent._create_with_deadline

        def create_with_budget(client, **kwargs):
            kwargs.setdefault("max_tokens", args.max_tokens)
            return original_create(client, **kwargs)

        agent._create_with_deadline = create_with_budget
        workdir = Path(tempfile.mkdtemp(prefix="slime-workbench-"))
        data_dir = workdir / "data"
        data_dir.mkdir()
        (data_dir / "processed").symlink_to(root / "data" / "processed", target_is_directory=True)
        (data_dir / "raw").symlink_to(root / "data" / "raw", target_is_directory=True)
        (data_dir / "results").mkdir()
        os.chdir(workdir)
        scored_domains = []
        for task_path in task_files:
            selected_path = task_path
            if args.limit and args.limit < 99999:
                frame = pd.read_csv(task_path).head(args.limit)
                selected_path = workdir / task_path.name
                frame.to_csv(selected_path, index=False)
            result = inference.generate_results(
                str(selected_path), "slime-agent", "all", workers=args.workers,
                log_traces=True, act_without_confirmation=True,
                structured_outputs=args.structured_outputs, resume=not args.overwrite,
            )
            ground_truth = pd.read_csv(selected_path)
            # pandas returns the CSV action column as a string; metrics expects
            # the original list of action expressions.
            ground_truth["outcome"] = ground_truth["outcome"].apply(ast.literal_eval)
            scored = metrics.calculate_metrics(ground_truth, result)
            domain = task_path.stem.removesuffix("_tasks_and_outcomes")
            scored_domains.append((domain, scored))
        import shutil
        shutil.copytree(data_dir / "results", args.output_dir / "results", dirs_exist_ok=True)
        _print_results_table(scored_domains)
        print(f"WorkBench complete: output={args.output_dir}")
        return 0
    finally:
        _stop(process)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
