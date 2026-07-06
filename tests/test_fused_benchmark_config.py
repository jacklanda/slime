from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from slime_plugins.evals.fused_benchmark_config import discover_benchmarks


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
        prefer_verl=True,
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
