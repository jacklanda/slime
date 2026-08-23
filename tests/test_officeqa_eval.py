from __future__ import annotations

import csv
import asyncio
import json
from pathlib import Path

import pytest

from slime.rollout.fused_agent.generate import _initial_messages
from slime.rollout.rm_hub.officeqa import reward_func
from slime_plugins.evals.officeqa import SYSTEM_PROMPT, prepare_dataset


def _fixture(root: Path) -> None:
    parsed = root / "treasury_bulletins_parsed" / "jsons"
    parsed.mkdir(parents=True)
    (parsed / "treasury_bulletin_1941_01.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"content": "wrong page", "bbox": [{"page_id": 14}]},
                        {"content": "Total expenditures 2,602", "bbox": [{"page_id": 15}]},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    with (root / "officeqa_pro.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["uid", "question", "answer", "source_docs", "source_files", "difficulty"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "uid": "UID0001",
                "question": "What was the total?",
                "answer": "2,602",
                "source_docs": "https://fraser.stlouisfed.org/title/treasury-bulletin-407/january-1941-1?page=15",
                "source_files": "treasury_bulletin_1941_01.txt",
                "difficulty": "hard",
            }
        )
    (root / "reward.py").write_text("def score_answer(*args, **kwargs): return 1\n", encoding="utf-8")


@pytest.mark.unit
def test_prepare_officeqa_uses_only_oracle_page_and_official_reward(tmp_path: Path):
    _fixture(tmp_path)
    output, reward, count = prepare_dataset(tmp_path, tmp_path / "normalized.jsonl")
    record = json.loads(output.read_text(encoding="utf-8"))

    assert count == 1
    assert reward == tmp_path / "reward.py"
    assert "Total expenditures 2,602" in record["input"]
    assert "wrong page" not in record["input"]
    assert record["extra_info"]["oracle_mode"] == "gem-text-oracle"
    assert record["extra_info"]["system_prompt"] == SYSTEM_PROMPT
    assert record["extra_info"]["rm_type"] == "officeqa"


@pytest.mark.unit
def test_fused_agent_accepts_dataset_system_prompt_override():
    messages = _initial_messages("gem", "web_search", "oracle\nquestion", [], system_prompt=SYSTEM_PROMPT)

    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "oracle\nquestion"},
    ]


@pytest.mark.unit
def test_evals_launcher_wires_officeqa_options():
    launcher = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text(encoding="utf-8")

    assert '--officeqa-use-sglang-session) OFFICEQA_USE_SGLANG_SESSION=' in launcher
    assert '--officeqa-overwrite) OFFICEQA_OVERWRITE=true' in launcher
    assert 'NATIVE_SGLANG_SESSION="${OFFICEQA_USE_SGLANG_SESSION}"' in launcher
    assert 'EVAL_MAX_RESPONSE_LEN=50000' in launcher
    assert 'WEB_SEARCH_MAX_STEPS=20' in launcher
    assert 'FUSED_WEBQA_MIN_UNIQUE_SEARCHES=0' in launcher


@pytest.mark.unit
def test_officeqa_reward_requires_xml_and_records_tolerances(tmp_path: Path):
    reward = tmp_path / "reward.py"
    reward.write_text(
        "def score_answer(ground_truth, predicted, tolerance): return float(tolerance >= 0.01)\n",
        encoding="utf-8",
    )

    class Sample:
        label = "10"
        response = "10"
        metadata = {"official_reward_path": str(reward)}

    sample = Sample()
    assert asyncio.run(reward_func(None, sample)) == 0.0
    assert sample.metadata["officeqa_incomplete"] is True

    sample.response = "<FINAL_ANSWER>10</FINAL_ANSWER>"
    assert asyncio.run(reward_func(None, sample)) == 0.0
    assert sample.metadata["officeqa_scores"] == {"0.0%": 0.0, "0.1%": 0.0, "1.0%": 1.0, "5.0%": 1.0}
