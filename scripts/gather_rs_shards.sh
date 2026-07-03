#!/usr/bin/env bash

set -euo pipefail

DEFAULT_INPUT="experiments/rejection_sampling/offline-rs-slime-fused-20260703103257/train"

INPUT_PATH="${1:-${DEFAULT_INPUT}}"
OUTPUT_DIR="${2:-}"

# Allow paths copied from Codex references such as @experiments/...
INPUT_PATH="${INPUT_PATH#@}"
if [[ -n "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR#@}"
fi

python - "${INPUT_PATH}" "${OUTPUT_DIR}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


input_path = Path(sys.argv[1])
output_arg = sys.argv[2]

if input_path.name == "train":
    train_dir = input_path
    output_dir = Path(output_arg) if output_arg else input_path.parent
else:
    train_dir = input_path / "train"
    output_dir = Path(output_arg) if output_arg else input_path

if not train_dir.is_dir():
    raise SystemExit(f"train directory not found: {train_dir}")

json_files = sorted(
    train_dir.glob("*.json"),
    key=lambda p: (
        0,
        int(p.stem.rsplit("_", 1)[-1]),
    )
    if p.stem.rsplit("_", 1)[-1].isdigit()
    else (1, p.name),
)
if not json_files:
    raise SystemExit(f"no json shard files found under: {train_dir}")


def canonical_task(task):
    return json.dumps(task, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def is_correct(traj):
    if "is_correct" in traj:
        return bool(traj["is_correct"])
    metrics = traj.get("metrics") or {}
    if "reward" in metrics:
        return float(metrics["reward"]) == 1.0
    if "exact_match" in metrics:
        return float(metrics["exact_match"]) == 1.0
    return False


tasks = {}
stats = defaultdict(lambda: {"total_count": 0, "correct_count": 0, "source_files": set(), "training_steps": set()})

for shard in json_files:
    with shard.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    training_step = payload.get("training_step")
    trajectories = payload.get("trajectories", [])
    for traj in trajectories:
        task = traj.get("task")
        if not isinstance(task, dict):
            continue

        task_hash = traj.get("task_hash", "")
        key = (task_hash, canonical_task(task))
        tasks.setdefault(key, task)

        entry = stats[key]
        entry["total_count"] += 1
        entry["correct_count"] += int(is_correct(traj))
        entry["source_files"].add(shard.name)
        if training_step is not None:
            entry["training_steps"].add(training_step)

accepted = []
for (task_hash, _), task in tasks.items():
    entry = stats[(task_hash, canonical_task(task))]
    total_count = entry["total_count"]
    correct_count = entry["correct_count"]
    if correct_count == 0 or correct_count == total_count:
        continue

    solve_rate = correct_count / total_count
    row = {
        **task,
        "task_hash": task_hash,
        "solve_rate": solve_rate,
        "pass_rate": solve_rate,
        "correct_count": correct_count,
        "total_count": total_count,
        "source_files": sorted(entry["source_files"]),
        "training_steps": sorted(entry["training_steps"]),
    }
    accepted.append(row)

accepted.sort(
    key=lambda row: (
        -row["solve_rate"],
        row.get("data_source", ""),
        row.get("question", ""),
        row["task_hash"],
    )
)

output_dir.mkdir(parents=True, exist_ok=True)
json_path = output_dir / "accepted_tasks.json"
parquet_path = output_dir / "accepted_tasks.parquet"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(accepted, f, indent=4, ensure_ascii=False)
    f.write("\n")

pd.DataFrame(accepted).to_parquet(parquet_path, index=False)

print(f"read shards: {len(json_files)}")
print(f"accepted tasks: {len(accepted)}")
print(f"wrote: {json_path}")
print(f"wrote: {parquet_path}")
PY
