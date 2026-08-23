from __future__ import annotations

import argparse
import json
import string
import uuid
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - runtime dependency in slime env
    pd = None


EXCLUDED_DIRS = {
    ".claude",
    "agentcpm_evals",
    "dr_tulu_evals",
}


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("data", "examples", "records", "test", "validation", "train"):
                value = data.get(key)
                if isinstance(value, list):
                    data = value
                    break
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a record list")
        return [x for x in data if isinstance(x, dict)]
    if suffix == ".parquet":
        if pd is None:
            raise ImportError("pandas is required to normalize parquet benchmarks")
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported benchmark source: {path}")


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value
    return None


def _format_options(options: Any) -> tuple[str, list[str]]:
    if hasattr(options, "tolist"):
        options = options.tolist()
    if not isinstance(options, (list, tuple)) or not options:
        return "", []
    option_list = [str(x) for x in options]
    lines = [f"{letter}. {text}" for letter, text in zip(string.ascii_uppercase, option_list, strict=False)]
    return "\nOptions:\n" + "\n".join(lines), option_list


def _normalize_answer(answer: Any) -> Any:
    if hasattr(answer, "tolist"):
        answer = answer.tolist()
    if isinstance(answer, list) and answer:
        return answer
    return answer


def _ground_truth_from_reward_model(reward_model: Any) -> Any:
    if not isinstance(reward_model, dict):
        return None
    ground_truth = reward_model.get("ground_truth")
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target")
        if target is not None:
            return target
    return ground_truth


def _normalize_record(row: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    row = dict(row)
    task_id = _first(
        row,
        ("task_id", "id", "uuid", "pid", "idx", "question_id", "problem_idx", "task_group_id"),
    ) or str(uuid.uuid4())

    options_suffix, options = _format_options(_first(row, ("options", "choices")))
    prompt = _first(row, ("input", "query", "question", "problem", "puzzle", "prompt"))
    if dataset_name == "zebra_logic":
        prompt = "\n".join(str(x) for x in (row.get("puzzle"), row.get("question")) if x)
    if prompt is None:
        prompt = json.dumps(row, ensure_ascii=False, default=_json_default)
    elif isinstance(prompt, str) and options_suffix and "Options:" not in prompt:
        prompt = f"{prompt}{options_suffix}"

    answer = _normalize_answer(
        _first(
            {
                **row,
                "_reward_model_ground_truth": _ground_truth_from_reward_model(row.get("reward_model")),
            },
            (
                "ground_truth_answer",
                "gt_answer",
                "_reward_model_ground_truth",
                "answer",
                "answer_letter",
                "solution",
                "final_answer",
                "target",
                "Final answer",
            ),
        )
    )

    extra_info = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else dict(row)
    if options:
        extra_info.setdefault("options", options)
        extra_info.setdefault("choices", options)
    if "answer_letter" in row:
        extra_info.setdefault("correct_letter", row["answer_letter"])
    if dataset_name in {"gpqa", "gpqa_diamond"} and isinstance(answer, str) and len(answer.strip()) == 1:
        extra_info.setdefault("correct_letter", answer.strip().upper())

    return {
        "task_id": str(task_id),
        "input": prompt,
        "gt_answer": answer,
        "ground_truth_answer": answer,
        "target": answer,
        "data_source": dataset_name,
        "prompt": prompt,
        "ability": row.get("ability", "search"),
        "reward_model": row.get("reward_model", {"ground_truth": answer, "style": "rule"}),
        "extra_info": extra_info,
    }


def _benchmark_entries(root: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if bench_dir.name == "frontierscience":
            for split_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
                if split_dir.name in {"olympiad", "research"}:
                    entries.append((f"frontierscience_{split_dir.name}", split_dir))
            continue
        entries.append((bench_dir.name, bench_dir))
    return entries


def _expand_benchmark_names(names: set[str]) -> set[str]:
    expanded = set(names)
    if "frontierscience" in expanded:
        expanded.update({"frontierscience_olympiad", "frontierscience_research"})
    return expanded


def _source_files(bench_dir: Path) -> list[Path]:
    jsonl_files = sorted(p for p in bench_dir.glob("*.jsonl") if p.is_file())
    if jsonl_files:
        return jsonl_files
    preferred = bench_dir / "data.json"
    if preferred.is_file():
        return [preferred]
    data_dir = bench_dir / "data"
    if data_dir.is_dir():
        files = sorted(
            p
            for p in data_dir.iterdir()
            if p.suffix.lower() in {".parquet", ".json", ".jsonl"} and p.is_file()
        )
        if files:
            return files
    return sorted(
        p for p in bench_dir.iterdir() if p.suffix.lower() in {".parquet", ".json", ".jsonl"} and p.is_file()
    )


def discover_benchmarks(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.benchmarks_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    include = _expand_benchmark_names({x for x in args.include.split(",") if x and x != "all"})
    exclude = EXCLUDED_DIRS | _expand_benchmark_names({x for x in args.exclude.split(",") if x})
    datasets: list[dict[str, Any]] = []

    for name, bench_dir in _benchmark_entries(root):
        if name in exclude or (include and name not in include):
            continue

        if name == "officeqa":
            from slime_plugins.evals.officeqa import prepare_dataset

            path, _, _ = prepare_dataset(
                bench_dir,
                cache_dir / "officeqa.jsonl",
                limit=args.limit_per_benchmark,
            )
        else:
            path = None

        verl_path = bench_dir / "data_verl.parquet"
        if path is not None:
            pass
        elif verl_path.is_file():
            path = verl_path
        else:
            sources = _source_files(bench_dir)
            if not sources:
                continue
            records: list[dict[str, Any]] = []
            for source in sources:
                try:
                    records.extend(_normalize_record(row, name) for row in _load_records(source))
                except Exception as exc:
                    print(f"skip {source}: {exc}", flush=True)
            if args.limit_per_benchmark > 0:
                records = records[: args.limit_per_benchmark]
            if not records:
                continue
            path = cache_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

        entry = {
            "name": name,
            "path": str(path),
            "input_key": "input",
            "label_key": "ground_truth_answer",
            "metadata_key": "extra_info",
            "custom_generate_function_path": args.custom_generate_function_path,
            "metadata_overrides": {
                "benchmark_eval": True,
                "rm_type": "benchmark_verifier",
                "data_source": name,
            },
        }
        if name == "officeqa":
            entry["rm_type"] = "benchmark_verifier"
            entry["metadata_overrides"].update({"rm_type": "benchmark_verifier", "oracle_mode": "gem-text-oracle"})
            entry["max_response_len"] = args.long_response_len
        if name == "bamboogle":
            entry["metadata_overrides"]["strict_exact_match"] = True
        if name in {"mcp-atlas", "mcp_atlas"}:
            entry["metadata_overrides"].update(
                {
                    "data_source": "mcp_atlas",
                    "mcp_transport": "atlas",
                    "mcp_atlas_eval": True,
                }
            )
            entry["max_response_len"] = args.long_response_len
        if name in {"browsecomp_plus", "browse_comp", "frontierscience_research"}:
            entry["max_response_len"] = args.long_response_len
        datasets.append(entry)

    if not datasets:
        raise SystemExit(f"No benchmarks discovered under {root}")
    return datasets


def write_config(args: argparse.Namespace) -> None:
    datasets = discover_benchmarks(args)
    output = Path(args.output_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "eval": {
            "defaults": {
                "n_samples_per_eval_prompt": args.n_samples_per_prompt,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "input_key": "input",
                "label_key": "ground_truth_answer",
                "metadata_key": "extra_info",
                "custom_generate_function_path": args.custom_generate_function_path,
                "metadata_overrides": {
                    "benchmark_eval": True,
                    "rm_type": "benchmark_verifier",
                },
            },
            "datasets": datasets,
        }
    }
    import yaml

    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote eval config: {output}")
    print(f"Benchmarks: {', '.join(item['name'] for item in datasets)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-root", default="experiments/artifacts/benchmarks")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--include", default="all")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--limit-per-benchmark", type=int, default=0)
    parser.add_argument("--n-samples-per-prompt", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--long-response-len", type=int, default=35000)
    parser.add_argument("--custom-generate-function-path", default="slime.rollout.fused_agent.generate.generate")
    return parser.parse_args()


if __name__ == "__main__":
    write_config(parse_args())
