from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from slime_plugins.evals.fused_benchmark_config import discover_benchmarks
from slime_plugins.evals.prepare_mcp_atlas import (
    filter_supported_tasks,
    normalize_gtfa_claims,
    normalize_mcp_atlas_dataframe,
)


@pytest.mark.unit
def test_frontierscience_splits_are_discovered_as_separate_benchmarks(tmp_path: Path):
    root = tmp_path / "benchmarks"
    olympiad = root / "frontierscience" / "olympiad"
    research = root / "frontierscience" / "research"
    olympiad.mkdir(parents=True)
    research.mkdir(parents=True)
    (olympiad / "test.jsonl").write_text(
        json.dumps(
            {
                "problem": "Compute the value.\nFINAL ANSWER:",
                "answer": "42",
                "subject": "physics",
                "task_group_id": "olympiad-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (research / "test.jsonl").write_text(
        json.dumps(
            {
                "problem": "Context: ...\nQuestion: Explain the method.",
                "answer": "Points: describe the method.",
                "subject": "biology",
                "task_group_id": "research-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        benchmarks_root=str(root),
        cache_dir=str(tmp_path / "cache"),
        include="frontierscience",
        exclude="",
        limit_per_benchmark=0,
        custom_generate_function_path="slime.rollout.fused_agent.generate.generate",
        long_response_len=38000,
    )

    datasets = discover_benchmarks(args)

    assert [dataset["name"] for dataset in datasets] == [
        "frontierscience_olympiad",
        "frontierscience_research",
    ]
    assert datasets[1]["max_response_len"] == 38000
    normalized = Path(datasets[0]["path"]).read_text(encoding="utf-8")
    assert '"data_source": "frontierscience_olympiad"' in normalized


@pytest.mark.unit
def test_bamboogle_uses_strict_exact_match(tmp_path: Path):
    benchmark = tmp_path / "benchmarks" / "bamboogle"
    benchmark.mkdir(parents=True)
    (benchmark / "data.json").write_text(
        json.dumps([{"question": "Who?", "answer": "Ada Lovelace"}]),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        benchmarks_root=str(tmp_path / "benchmarks"),
        cache_dir=str(tmp_path / "cache"),
        include="bamboogle",
        exclude="",
        limit_per_benchmark=0,
        custom_generate_function_path="slime.rollout.fused_agent.generate.generate",
        long_response_len=38000,
    )

    datasets = discover_benchmarks(args)

    assert datasets[0]["metadata_overrides"]["strict_exact_match"] is True


@pytest.mark.unit
def test_existing_verl_dataset_is_always_preferred_over_raw_source(tmp_path: Path):
    benchmark = tmp_path / "benchmarks" / "example"
    benchmark.mkdir(parents=True)
    (benchmark / "data.json").write_text(
        json.dumps([{"question": "raw", "answer": "raw-answer"}]),
        encoding="utf-8",
    )
    verl_path = benchmark / "data_verl.parquet"
    pd.DataFrame([{"input": "normalized", "ground_truth_answer": "normalized-answer"}]).to_parquet(
        verl_path,
        index=False,
    )
    args = argparse.Namespace(
        benchmarks_root=str(tmp_path / "benchmarks"),
        cache_dir=str(tmp_path / "cache"),
        include="example",
        exclude="",
        limit_per_benchmark=0,
        custom_generate_function_path="slime.rollout.fused_agent.generate.generate",
        long_response_len=38000,
    )

    datasets = discover_benchmarks(args)

    assert datasets[0]["path"] == str(verl_path)


@pytest.mark.unit
def test_mcp_atlas_preparation_filters_brave_and_preserves_claim_metadata():
    raw = pd.DataFrame(
        [
            {
                "TASK": "supported-task",
                "ENABLED_TOOLS": json.dumps(["wikipedia_search_wikipedia"]),
                "PROMPT": "Find an article",
                "GTFA_CLAIMS": json.dumps(["The response names the article"]),
            },
            {
                "TASK": "brave-task",
                "ENABLED_TOOLS": json.dumps(["brave-search_brave_web_search"]),
                "PROMPT": "Search the web",
                "GTFA_CLAIMS": json.dumps(["The response reports the result"]),
            },
        ]
    )

    normalized = normalize_mcp_atlas_dataframe(raw)

    assert normalized["task_id"].tolist() == ["supported-task"]
    assert normalized.iloc[0]["ground_truth_answer"] == json.dumps(["The response names the article"])
    assert normalized.iloc[0]["extra_info"] == {
        "task_id": "supported-task",
        "question": "Find an article",
        "GTFA_CLAIMS": json.dumps(["The response names the article"]),
        "enabled_tools": ["wikipedia_search_wikipedia"],
        "data_source": "mcp_atlas",
        "benchmark": "mcp_atlas",
        "mcp_transport": "atlas",
        "mcp_atlas_eval": True,
    }


@pytest.mark.unit
def test_mcp_atlas_preparation_can_keep_only_sandbox_supported_tasks():
    frame = pd.DataFrame(
        [
            {"task_id": "supported", "input": "ok"},
            {"task_id": "missing-one-tool", "input": "drop"},
            {"task_id": "missing-many-tools", "input": "drop"},
        ]
    )

    filtered = filter_supported_tasks(
        frame,
        {
            "missing-one-tool": ["anili_search_anime"],
            "missing-many-tools": ["f1-mcp-server_get_event_info", "f1-mcp-server_get_event_schedule"],
        },
    )

    assert filtered["task_id"].tolist() == ["supported"]


@pytest.mark.unit
def test_mcp_atlas_claim_normalization_repairs_literal_newline_in_quoted_claim():
    malformed = "['README translated into Croatian:\\n```\\ntext\\n```\n']"

    normalized = normalize_gtfa_claims(malformed)

    assert json.loads(normalized) == ["README translated into Croatian:\n```\ntext\n```\n"]


@pytest.mark.unit
def test_mcp_atlas_eval_config_selects_remote_transport(tmp_path: Path):
    benchmark = tmp_path / "benchmarks" / "mcp-atlas"
    benchmark.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "input": "Find an article",
                "ground_truth_answer": '["claim"]',
                "extra_info": {"enabled_tools": ["wikipedia_search_wikipedia"]},
            }
        ]
    ).to_parquet(benchmark / "data_verl.parquet", index=False)
    args = argparse.Namespace(
        benchmarks_root=str(tmp_path / "benchmarks"),
        cache_dir=str(tmp_path / "cache"),
        include="mcp-atlas",
        exclude="",
        limit_per_benchmark=0,
        custom_generate_function_path="slime.rollout.fused_agent.generate.generate",
        long_response_len=38000,
    )

    datasets = discover_benchmarks(args)

    assert datasets[0]["metadata_overrides"]["data_source"] == "mcp_atlas"
    assert datasets[0]["metadata_overrides"]["mcp_transport"] == "atlas"
    assert datasets[0]["metadata_overrides"]["mcp_atlas_eval"] is True
    assert datasets[0]["max_response_len"] == 38000
