#!/usr/bin/env python3
"""Independently re-audit a dumped slime evaluation with three judge models."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import httpx
from rich.console import Console
from rich.table import Table


AUDIT_PROMPT = """You are an exacting independent auditor for open-domain QA evaluation.
For every item, decide whether PREDICTION factually and sufficiently answers QUESTION as of 2026.
REFERENCE is evidence, not infallible: it may be stale, overly narrow, ambiguous, or wrong.
Accept genuine aliases, translations, reasonable rounding, a precise location contained by the reference,
and updated answers to time-sensitive questions. Reject related but different entities, wrong relations or hops,
contradictions, and answers missing essential requested information. Classify reference_status as valid,
outdated_or_wrong, or ambiguous. Return every input id exactly once as JSON:
{"results":[{"id":0,"correct":true,"reference_status":"valid","confidence":0.95,"reason":"brief reason"}]}.
Keep each reason under 25 words."""

ARBITRATION_PROMPT = """You are the final arbiter for disputed open-domain QA labels.
Decide factual correctness as of 2026, not mechanical string matching. REFERENCE may be stale or wrong.
Critically inspect both prior auditor opinions. Accept true aliases, translations, reasonable rounding,
precise contained locations, and current facts. Reject wrong entities, relations, hops, contradictions,
or answers missing essential information. Return every input id exactly once as JSON:
{"results":[{"id":0,"correct":true,"reference_status":"valid","confidence":0.95,"reason":"brief reason"}]}.
reference_status must be valid, outdated_or_wrong, or ambiguous. Keep each reason under 25 words."""

REFERENCE_STATUSES = {"valid", "outdated_or_wrong", "ambiguous"}
CONFIDENCE_WORDS = {"high": 0.95, "medium": 0.7, "low": 0.4}
BATCH_SIZE = 64
CONCURRENCY = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="Evaluated global_steps_*.json episode dump")
    parser.add_argument("--output-dir", type=Path, help="Cache and report directory (default: INPUT.revals)")
    parser.add_argument("--judge-models", nargs=2, default=["openai/gpt-5.6-luna", "deepseek/deepseek-v4-flash-0731"])
    parser.add_argument("--arbiter-model", default="openai/gpt-5.6-terra")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--no-protocol-check", action="store_true", help="Do not require finish + env_done")
    parser.add_argument("--report-only", action="store_true", help="Only aggregate existing complete caches")
    args = parser.parse_args()
    args.output_dir = args.output_dir or Path(str(args.trajectory) + ".revals")
    return args


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    episodes = data.get("trajectories") if isinstance(data, dict) else data
    if not isinstance(episodes, list):
        raise ValueError(f"{path} must be an episode list or contain a trajectories list")
    return episodes


def get_question(episode: dict[str, Any]) -> Any:
    task = episode.get("task") if isinstance(episode.get("task"), dict) else {}
    for key in ("question", "query", "input", "problem", "prompt"):
        if task.get(key) is not None:
            return task[key]
    return task


def get_benchmark(episode: dict[str, Any]) -> str:
    task = episode.get("task") if isinstance(episode.get("task"), dict) else {}
    candidates = [task.get("benchmark"), task.get("source")]
    tools_kwargs = task.get("tools_kwargs") if isinstance(task.get("tools_kwargs"), dict) else {}
    search = tools_kwargs.get("search") if isinstance(tools_kwargs.get("search"), dict) else {}
    create_kwargs = search.get("create_kwargs") if isinstance(search.get("create_kwargs"), dict) else {}
    candidates.insert(0, create_kwargs.get("data_source"))
    candidates.append(task.get("data_source"))
    for value in candidates:
        if value and str(value).lower() not in {"asearcher", "unknown"}:
            return str(value)
    return str(next((value for value in candidates if value), "unknown"))


def get_last_action_result(episode: dict[str, Any]) -> Any:
    trajectories = episode.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        return None
    trajectory = trajectories[-1]
    if not isinstance(trajectory, dict):
        return None
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
        return None
    action = steps[-1].get("action")
    if isinstance(action, dict):
        payloads = [action]
    elif isinstance(action, str):
        try:
            root = ET.fromstring(f"<root>{action}</root>")
        except ET.ParseError:
            return None
        payloads = []
        for element in root.iter("tool_call"):
            try:
                payloads.append(json.loads("".join(element.itertext())))
            except json.JSONDecodeError:
                continue
    else:
        return None
    for payload in reversed(payloads):
        if not isinstance(payload, dict) or payload.get("name") not in {"finish", "submit"}:
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, dict) and "result" in arguments:
            return arguments["result"]
    return None


def audit_item(index: int, episode: dict[str, Any]) -> dict[str, Any]:
    verification = episode.get("verification") if isinstance(episode.get("verification"), dict) else {}
    return {
        "id": index,
        "question": get_question(episode),
        "reference": verification.get("ground_truth", episode.get("task", {}).get("ground_truth")),
        "prediction": get_last_action_result(episode),
    }


def confidence(result: dict[str, Any]) -> float:
    value = result.get("confidence", 1.0)
    if isinstance(value, (int, float)):
        return float(value)
    return CONFIDENCE_WORDS.get(str(value).lower(), 1.0)


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("reference_status", result.get("ref", "ambiguous"))
    if status not in REFERENCE_STATUSES:
        status = "ambiguous"
    correct = result.get("correct", result.get("prediction_correct"))
    if not isinstance(correct, bool):
        raise ValueError(f"judge result {result.get('id')} has non-boolean correct field: {correct!r}")
    return {
        "id": int(result["id"]),
        "correct": correct,
        "reference_status": status,
        "confidence": confidence(result),
        "reason": str(result.get("reason", "")),
    }


def cache_dir(output_dir: Path, stage: str, model: str) -> Path:
    safe_model = model.replace("/", "__").replace(":", "_")
    path = output_dir / "cache" / stage / safe_model
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_manifest(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    manifest_path = args.output_dir / "manifest.json"
    item_digest = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = {
        "trajectory": str(args.trajectory.resolve()),
        "item_digest": item_digest,
        "num_samples": len(items),
        "judge_models": args.judge_models,
        "arbiter_model": args.arbiter_model,
        "batch_size": BATCH_SIZE,
        "concurrency_per_model": CONCURRENCY,
        "low_confidence_threshold": args.low_confidence_threshold,
        "protocol_check": not args.no_protocol_check,
        "audit_prompt_digest": hashlib.sha256(AUDIT_PROMPT.encode()).hexdigest(),
        "arbitration_prompt_digest": hashlib.sha256(ARBITRATION_PROMPT.encode()).hexdigest(),
    }
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached != manifest:
            raise ValueError(
                f"{args.output_dir} contains caches for a different input or configuration; "
                "choose another --output-dir or remove the stale cache"
            )
        return
    if args.report_only:
        raise ValueError(f"{manifest_path} is required with --report-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


async def request_batch(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, args: argparse.Namespace,
                        model: str, prompt: str, batch: list[dict[str, Any]], destination: Path) -> bool:
    if destination.exists():
        return True
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(batch, ensure_ascii=False)}],
        "response_format": {"type": "json_object"},
    }
    error = ""
    for attempt in range(args.max_retries):
        try:
            async with semaphore:
                response = await client.post("/chat/completions", json=payload, timeout=args.timeout)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            results = [normalize_result(item) for item in json.loads(content)["results"]]
            if {item["id"] for item in results} != {item["id"] for item in batch}:
                raise ValueError("judge response ids do not match request ids")
            destination.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            destination.with_suffix(".err").unlink(missing_ok=True)
            return True
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if isinstance(exc, json.JSONDecodeError):
                break
            if attempt + 1 < args.max_retries:
                await asyncio.sleep(min(20, 2**attempt))
    destination.with_suffix(".err").write_text(error)
    if len(batch) > 1:
        midpoint = len(batch) // 2
        parts_dir = destination.parent / ".parts"
        parts_dir.mkdir(exist_ok=True)
        part_paths = [
            parts_dir / f"{destination.stem}-0.json",
            parts_dir / f"{destination.stem}-1.json",
        ]
        completed = await asyncio.gather(*(
            request_batch(client, semaphore, args, model, prompt, part, path)
            for part, path in zip((batch[:midpoint], batch[midpoint:]), part_paths, strict=True)
        ))
        if all(completed):
            results = []
            for path in part_paths:
                results.extend(json.loads(path.read_text()))
                path.unlink()
            destination.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            destination.with_suffix(".err").unlink(missing_ok=True)
            return True
    return False


async def run_stage(args: argparse.Namespace, api_key: str, model: str, stage: str, prompt: str,
                    items: list[dict[str, Any]], batch_size: int) -> dict[int, dict[str, Any]]:
    directory = cache_dir(args.output_dir, stage, model)
    batches = [items[offset:offset + batch_size] for offset in range(0, len(items), batch_size)]
    if not args.report_only:
        headers = {"Authorization": f"Bearer {api_key}", "X-Title": "slime revals"}
        semaphore = asyncio.Semaphore(CONCURRENCY)
        limits = httpx.Limits(max_connections=CONCURRENCY + 2, max_keepalive_connections=CONCURRENCY)
        async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, limits=limits) as client:
            for offset in range(0, len(batches), CONCURRENCY * 3):
                group = batches[offset:offset + CONCURRENCY * 3]
                completed = await asyncio.gather(*(
                    request_batch(client, semaphore, args, model, prompt, batch, directory / f"{index:05d}.json")
                    for index, batch in enumerate(group, start=offset)
                ))
                print(f"[{stage}/{model}] {min(offset + len(group), len(batches))}/{len(batches)} batches; "
                      f"{sum(completed)}/{len(completed)} succeeded", flush=True)
    results: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        for result in json.loads(path.read_text()):
            normalized = normalize_result(result)
            results[normalized["id"]] = normalized
    expected = {item["id"] for item in items}
    if set(results) != expected:
        missing = len(expected - set(results))
        raise RuntimeError(f"{stage}/{model} cache is incomplete: {missing} of {len(expected)} items missing")
    return results


def protocol_valid(episode: dict[str, Any]) -> bool:
    return get_last_action_result(episode) is not None and episode.get("termination_reason") == "env_done"


def old_label(episode: dict[str, Any]) -> bool:
    return bool(float(episode.get("eval_reward") or 0))


def build_arbitration_items(items: list[dict[str, Any]], first: dict[int, dict[str, Any]],
                            second: dict[int, dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    disputed = []
    for item in items:
        item_id = item["id"]
        if first[item_id]["correct"] == second[item_id]["correct"] and min(
            first[item_id]["confidence"], second[item_id]["confidence"]
        ) >= threshold:
            continue
        disputed.append({**item, "auditor_A": first[item_id], "auditor_B": second[item_id]})
    return disputed


def write_reports(args: argparse.Namespace, episodes: list[dict[str, Any]], first: dict[int, dict[str, Any]],
                  second: dict[int, dict[str, Any]], arbitration: dict[int, dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reference_counts: Counter[str] = Counter()
    benchmark_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        a, b = first[index], second[index]
        if index in arbitration:
            decision = arbitration[index]
            decision_source = "arbiter"
        else:
            if a["correct"] != b["correct"]:
                raise RuntimeError(f"sample {index} disagrees but has no arbitration")
            decision = a
            decision_source = "auditor_agreement"
        valid = args.no_protocol_check or protocol_valid(episode)
        corrected = bool(decision["correct"] and valid)
        if index in arbitration:
            reference_status = decision["reference_status"]
        elif a["reference_status"] == b["reference_status"]:
            reference_status = a["reference_status"]
        else:
            reference_status = "auditor_disagreement"
        reference_counts[reference_status] += 1
        row = {
            "id": index,
            "episode_id": episode.get("episode_id"),
            "benchmark": get_benchmark(episode),
            "original_correct": old_label(episode),
            "corrected_correct": corrected,
            "protocol_valid": valid,
            "decision_source": decision_source,
            "reference_status": reference_status,
            "auditor_A": a,
            "auditor_B": b,
            "arbitration": arbitration.get(index),
        }
        rows.append(row)
        benchmark_rows[row["benchmark"]].append(row)

    old_correct = sum(row["original_correct"] for row in rows)
    corrected_correct = sum(row["corrected_correct"] for row in rows)
    false_positives = sum(row["original_correct"] and not row["corrected_correct"] for row in rows)
    false_negatives = sum(not row["original_correct"] and row["corrected_correct"] for row in rows)
    summary = {
        "num_samples": len(rows),
        "original_correct": old_correct,
        "original_accuracy": old_correct / len(rows),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "net_change": corrected_correct - old_correct,
        "corrected_correct": corrected_correct,
        "corrected_accuracy": corrected_correct / len(rows),
        "reference_audit": dict(reference_counts),
        "benchmarks": {},
    }
    for benchmark, samples in sorted(benchmark_rows.items()):
        original = sum(row["original_correct"] for row in samples)
        corrected = sum(row["corrected_correct"] for row in samples)
        summary["benchmarks"][benchmark] = {
            "num_samples": len(samples),
            "original_correct": original,
            "original_accuracy": original / len(samples),
            "false_positives": sum(row["original_correct"] and not row["corrected_correct"] for row in samples),
            "false_negatives": sum(not row["original_correct"] and row["corrected_correct"] for row in samples),
            "corrected_correct": corrected,
            "corrected_accuracy": corrected / len(samples),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (args.output_dir / "audit.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_summary(summary)


def print_summary(summary: dict[str, Any]) -> None:
    console = Console()
    total = summary["num_samples"]
    console.print("\n[bold]关键结果：[/bold]")
    console.print(f"- 原 Judge Acc：{summary['original_accuracy']:.2%}（{summary['original_correct']} / {total}）")
    console.print(f"- 假阳：{summary['false_positives']}")
    console.print(f"- 假阴：{summary['false_negatives']}")
    console.print(f"- 净增加正确样本：{summary['net_change']}")
    console.print(f"- 修正后 Judge Acc：{summary['corrected_accuracy']:.2%}（{summary['corrected_correct']} / {total}）")

    table = Table(title="分数据集修正后", show_lines=True)
    table.add_column("数据集")
    table.add_column("样本数", justify="right")
    table.add_column("原始", justify="right")
    table.add_column("修正后", justify="right")
    table.add_column("假阳", justify="right")
    table.add_column("假阴", justify="right")
    for benchmark, values in summary["benchmarks"].items():
        table.add_row(benchmark, str(values["num_samples"]), f"{values['original_accuracy']:.1%}",
                      f"{values['corrected_accuracy']:.1%}", str(values["false_positives"]), str(values["false_negatives"]))
    console.print(table)

    references = summary["reference_audit"]
    console.print("\n[bold]Reference 审计：[/bold]")
    console.print(f"- 有效：{references.get('valid', 0)}")
    console.print(f"- 错误或过时：{references.get('outdated_or_wrong', 0)}")
    console.print(f"- 歧义：{references.get('ambiguous', 0)}")
    console.print(f"- 两位审计模型意见不一：{references.get('auditor_disagreement', 0)}")


async def async_main(args: argparse.Namespace) -> None:
    episodes = load_episodes(args.trajectory)
    items = [audit_item(index, episode) for index, episode in enumerate(episodes)]
    validate_manifest(args, items)
    api_key = os.environ.get(args.api_key_env, "")
    if not args.report_only and not api_key:
        raise RuntimeError(f"environment variable {args.api_key_env} is not set")
    judge_tasks = [
        asyncio.create_task(
            run_stage(args, api_key, model, "audit", AUDIT_PROMPT, items, BATCH_SIZE),
            name=f"audit-{model}",
        )
        for model in args.judge_models
    ]
    print(
        f"Running primary judges concurrently: {args.judge_models[0]} and {args.judge_models[1]}; "
        f"up to {CONCURRENCY} requests per judge",
        flush=True,
    )
    first, second = await asyncio.gather(*judge_tasks)
    arbitration_items = build_arbitration_items(items, first, second, args.low_confidence_threshold)
    print(f"Arbitrating {len(arbitration_items)} disagreements or low-confidence samples", flush=True)
    arbitration = await run_stage(args, api_key, args.arbiter_model, "arbitration", ARBITRATION_PROMPT,
                                  arbitration_items, BATCH_SIZE) if arbitration_items else {}
    write_reports(args, episodes, first, second, arbitration)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
