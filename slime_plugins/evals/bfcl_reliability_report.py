from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _score_category(path: Path, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row.get("test_category"):
            return str(row["test_category"])
    name = path.stem.removesuffix("_score")
    return name.split("_", 2)[-1]


def build_report(result_dir: Path, score_dir: Path, artifact_name: str) -> dict[str, Any]:
    artifact = artifact_name.replace("/", "_")
    result_root = result_dir / artifact
    score_root = score_dir / artifact
    manifest = json.loads((result_root / "run_manifest.json").read_text(encoding="utf-8"))

    categories = []
    parallel_total = 0
    parallel_protocol_mismatches = 0
    force_terminated = 0
    for path in sorted(score_root.rglob("*_score.json")):
        rows = _read_jsonl(path)
        summary = next((row for row in rows if "accuracy" in row and "total_count" in row), None)
        if summary is None:
            continue
        category = _score_category(path, rows)
        correct = int(summary["correct_count"])
        total = int(summary["total_count"])
        low, high = wilson_interval(correct, total)
        categories.append(
            {
                "category": category,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )

        failures = [row for row in rows if row.get("valid") is False]
        if "parallel" in category:
            parallel_total += total
            parallel_protocol_mismatches += sum(1 for row in failures if str(row.get("error_type", "")).endswith("wrong_count") and len(row.get("model_result_decoded") or []) == 1 and len(row.get("possible_answer") or []) > 1)
        if category.startswith("multi_turn"):
            force_terminated += sum(1 for row in failures if isinstance(row.get("error"), dict) and row["error"].get("error_type") == "multi_turn:force_terminated")

    multi_turn_total = 0
    context_exhausted = 0
    step_limit_reached = 0
    max_steps = str(manifest.get("max_agent_steps", ""))
    for path in sorted((result_root / "multi_turn").glob("*_result.json")):
        rows = _read_jsonl(path)
        multi_turn_total += len(rows)
        for row in rows:
            result = row.get("result")
            if isinstance(result, str) and "No generation budget remains within the model context window" in result:
                context_exhausted += 1
            messages = _strings(row)
            if any(f"forced to quit after {max_steps} total steps" in message or f"Finish action detected at the {max_steps}-step limit" in message for message in messages):
                step_limit_reached += 1

    def diagnostic(count: int, total: int) -> dict[str, Any]:
        return {"count": count, "total": total, "rate": count / total if total else 0.0}

    warnings = ["BFCL Overall Acc uses official benchmark weighting and is not an end-to-end fused-agent success rate."]
    if not str(manifest.get("seed", "")):
        warnings.append("Generation seed is missing; repeated runs are not directly comparable.")
    if str(manifest.get("discard_historical_thinking", "")).lower() not in {"1", "true", "yes", "on"}:
        warnings.append("Historical reasoning is retained and can consume long-context generation budget.")
    if parallel_protocol_mismatches:
        warnings.append("Parallel categories require batched calls, while the fused runtime executes parsed actions sequentially; interpret official parallel accuracy separately.")
    if context_exhausted:
        warnings.append("Some multi-turn samples exhausted the configured model context window.")
    if step_limit_reached:
        warnings.append("Some multi-turn samples reached the configured agent step limit.")

    return {
        "artifact": artifact_name,
        "configuration": {
            "seed": manifest.get("seed", ""),
            "temperature": manifest.get("temperature", ""),
            "context_length": manifest.get("context_length", ""),
            "max_agent_steps": manifest.get("max_agent_steps", ""),
            "discard_historical_thinking": manifest.get("discard_historical_thinking", ""),
        },
        "diagnostics": {
            "parallel_protocol_mismatch": diagnostic(parallel_protocol_mismatches, parallel_total),
            "multi_turn_context_exhaustion": diagnostic(context_exhausted, multi_turn_total),
            "multi_turn_step_limit_reached": diagnostic(step_limit_reached, multi_turn_total),
            "multi_turn_force_terminated": diagnostic(force_terminated, multi_turn_total),
        },
        "category_uncertainty": categories,
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], score_dir: Path) -> None:
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "reliability_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (score_dir / "data_category_uncertainty.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "correct", "total", "accuracy", "wilson_95_low", "wilson_95_high"],
        )
        writer.writeheader()
        writer.writerows(report["category_uncertainty"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BFCL fused-agent reliability diagnostics.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    args = parser.parse_args()
    report = build_report(args.result_dir, args.score_dir, args.artifact_name)
    write_report(report, args.score_dir)
    print(f"BFCL reliability report: {args.score_dir / 'reliability_report.json'}")


if __name__ == "__main__":
    main()
