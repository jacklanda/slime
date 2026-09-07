from __future__ import annotations

import pandas as pd

from slime_plugins.evals import workbench_launcher


def test_workbench_readiness_probe_authenticates_to_sglang(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": [{"id": "checkpoint"}]}'

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(workbench_launcher.urllib.request, "urlopen", urlopen)

    assert workbench_launcher._ready("http://127.0.0.1:18082/v1/models", "checkpoint")
    assert captured["request"].full_url == "http://127.0.0.1:18082/v1/models"
    assert captured["request"].get_header("Authorization") == "Bearer EMPTY"
    assert captured["timeout"] == 3


def test_print_results_table_includes_domains_and_weighted_overall(capsys):
    calendar = pd.DataFrame(
        {
            "correct": [True, False],
            "exact_match": [True, False],
            "unwanted_side_effects": [False, True],
            "error": ["", "tool failure"],
        }
    )
    email = pd.DataFrame(
        {
            "correct": [True],
            "exact_match": [False],
            "unwanted_side_effects": [False],
            "error": [""],
        }
    )

    workbench_launcher._print_results_table([("calendar", calendar), ("email", email)])

    output = capsys.readouterr().out
    assert "WorkBench evaluation results" in output
    assert "| calendar | 2     | 1       | 50.00%" in output
    assert "| email    | 1     | 1       | 100.00%" in output
    assert "| Overall  | 3     | 2       | 66.67%" in output
