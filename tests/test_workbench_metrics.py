from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


WORKBENCH_ROOT = Path(__file__).parents[1] / "experiments" / "artifacts" / "benchmarks" / "workbench"
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from src.evals.metrics import compute_metrics  # noqa: E402
from src.tools.email import search_emails  # noqa: E402


def test_compute_metrics_parses_csv_serialized_action_lists(monkeypatch):
    monkeypatch.chdir(WORKBENCH_ROOT)
    action = 'analytics.create_plot.func(time_min="2023-11-21", time_max="2023-11-29", value_to_plot="total_visits", plot_type="bar")'
    ground_truth = pd.DataFrame({"task": ["plot"], "outcome": [repr([action])]})
    predictions = pd.DataFrame({"task": ["plot"], "function_calls": [[action]], "error": [""]})

    result = compute_metrics(ground_truth, predictions)

    assert result.loc[0, "ground_truth"] == [action]
    assert bool(result.loc[0, "correct"])


def test_search_emails_accepts_sender_filter(monkeypatch):
    monkeypatch.chdir(WORKBENCH_ROOT)
    result = search_emails(sender="anaya")
    assert isinstance(result, str)
    assert "sender/recipient" in result
