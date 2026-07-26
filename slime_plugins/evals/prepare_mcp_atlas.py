from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"TASK", "ENABLED_TOOLS", "PROMPT", "GTFA_CLAIMS"}
DEFAULT_DATASET_URL = "https://huggingface.co/datasets/ScaleAI/MCP-Atlas/resolve/main/MCP-Atlas.parquet"


def normalize_enabled_tools(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"ENABLED_TOOLS must be a list or JSON list, got {type(value).__name__}")
    names = []
    for item in value:
        name = item.get("name") if isinstance(item, dict) else item
        if name and str(name) not in names:
            names.append(str(name))
    return names


def uses_brave(tools: list[str]) -> bool:
    return any("brave" in name.lower() for name in tools)


def normalize_gtfa_claims(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()

    parsed = value
    if isinstance(value, str):
        raw = value.strip()
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(raw)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(candidate, list):
                parsed = candidate
                break

        # One public row contains a literal newline inside a Python-quoted
        # claim. Escape only after the normal parsers reject the source so the
        # claim remains one item instead of falling back to line splitting.
        if parsed is None and ("\n" in raw or "\r" in raw):
            escaped_newlines = raw.replace("\r", "\\r").replace("\n", "\\n")
            try:
                candidate = ast.literal_eval(escaped_newlines)
            except (ValueError, SyntaxError):
                candidate = None
            if isinstance(candidate, list):
                parsed = candidate

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("GTFA_CLAIMS must encode a list of claims")
    claims = [str(claim) for claim in parsed]
    if not claims:
        raise ValueError("GTFA_CLAIMS must contain at least one claim")
    return json.dumps(claims, ensure_ascii=False)


def normalize_mcp_atlas_dataframe(
    frame: pd.DataFrame,
    *,
    include_brave: bool = False,
    limit: int = 0,
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"MCP-Atlas dataset is missing columns: {sorted(missing_columns)}")

    records = []
    for row in frame.to_dict(orient="records"):
        tools = normalize_enabled_tools(row["ENABLED_TOOLS"])
        if not include_brave and uses_brave(tools):
            continue
        task_id = str(row["TASK"])
        prompt = str(row["PROMPT"])
        claims = normalize_gtfa_claims(row["GTFA_CLAIMS"])
        extra_info = {
            "task_id": task_id,
            "question": prompt,
            "GTFA_CLAIMS": claims,
            "enabled_tools": tools,
            "data_source": "mcp_atlas",
            "benchmark": "mcp_atlas",
            "mcp_transport": "atlas",
            "mcp_atlas_eval": True,
        }
        records.append(
            {
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt,
                "gt_answer": claims,
                "ground_truth_answer": claims,
                "target": claims,
                "data_source": "mcp_atlas",
                "ability": "mcp",
                "reward_model": {"ground_truth": claims, "style": "mcp_atlas_claim_coverage"},
                "extra_info": extra_info,
            }
        )
        if limit > 0 and len(records) >= limit:
            break
    return pd.DataFrame.from_records(records)


def atlas_missing_tools(frame: pd.DataFrame, sandbox_url: str) -> dict[str, list[str]]:
    request = urllib.request.Request(
        f"{sandbox_url.rstrip('/')}/list-tools",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['MCP_ATLAS_AUTH_TOKEN']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        available = {str(tool["name"]) for tool in json.load(response)}
    missing: dict[str, list[str]] = {}
    for row in frame.to_dict(orient="records"):
        task_id = str(row["task_id"])
        enabled = row["extra_info"]["enabled_tools"]
        task_missing = [name for name in enabled if name not in available]
        if task_missing:
            missing[task_id] = task_missing
    return missing


def filter_supported_tasks(
    frame: pd.DataFrame,
    missing_tools_by_task: dict[str, list[str]],
) -> pd.DataFrame:
    """Keep only tasks whose complete enabled-tool set exists in the sandbox."""
    unsupported_task_ids = set(missing_tools_by_task)
    if not unsupported_task_ids:
        return frame.reset_index(drop=True)
    return frame.loc[~frame["task_id"].astype(str).isin(unsupported_task_ids)].reset_index(drop=True)


def read_mcp_atlas_source(source: str) -> pd.DataFrame:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=120) as response, tempfile.NamedTemporaryFile(
            suffix=".parquet"
        ) as temp_file:
            shutil.copyfileobj(response, temp_file)
            temp_file.flush()
            return pd.read_parquet(temp_file.name)
    path = Path(source)
    if not path.is_file():
        raise SystemExit(f"MCP-Atlas source not found: {path}")
    return pd.read_parquet(path)


def prepare(args: argparse.Namespace) -> None:
    frame = read_mcp_atlas_source(args.input)
    normalized = normalize_mcp_atlas_dataframe(
        frame,
        include_brave=args.include_brave,
        limit=args.limit,
    )
    expected_rows = 500 if args.include_brave else 437
    if args.limit <= 0 and len(normalized) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} MCP-Atlas rows, got {len(normalized)}")

    missing: dict[str, list[str]] | None = None
    if args.sandbox_url:
        missing = atlas_missing_tools(normalized, args.sandbox_url)
    elif args.supported_only:
        raise SystemExit("--supported-only requires --sandbox-url (or MCP_SANDBOX_URL)")

    source_rows = len(normalized)
    if args.supported_only:
        normalized = filter_supported_tasks(normalized, missing or {})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(output, index=False)
    print(f"Wrote {len(normalized)} MCP-Atlas tasks to {output}")

    if missing is not None:
        missing_tools = sorted({name for names in missing.values() for name in names})
        supported_rows = source_rows - len(missing)
        if args.supported_only:
            print(
                f"Sandbox compatibility filter: kept {supported_rows}/{source_rows} fully supported tasks; "
                f"removed {len(missing)} tasks referencing {len(missing_tools)} unavailable tools"
            )
        else:
            print(
                f"Sandbox compatibility: {supported_rows}/{source_rows} tasks fully supported; "
                f"{len(missing)} tasks reference {len(missing_tools)} unavailable tools"
            )
        if missing_tools:
            print("Unavailable tools: " + ", ".join(missing_tools))
        if args.require_all_tools and missing and not args.supported_only:
            raise SystemExit("MCP-Atlas sandbox does not expose every tool required by the selected tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MCP-Atlas for slime evaluation")
    parser.add_argument(
        "--input",
        default=os.environ.get("MCP_ATLAS_SOURCE", DEFAULT_DATASET_URL),
        help="Public MCP-Atlas parquet containing TASK, ENABLED_TOOLS, PROMPT, and GTFA_CLAIMS.",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/benchmarks/mcp-atlas/data_verl.parquet",
    )
    parser.add_argument("--include-brave", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sandbox-url", default=os.environ.get("MCP_SANDBOX_URL", ""))
    parser.add_argument(
        "--supported-only",
        action="store_true",
        help="Drop every task that references a tool unavailable from --sandbox-url.",
    )
    parser.add_argument("--require-all-tools", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
