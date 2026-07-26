import logging
from types import SimpleNamespace

from slime.utils.types import Sample
from slime_plugins.evals.results_table import format_eval_results_table, log_eval_results_table


def _sample(
    index: int,
    source: str,
    *,
    steps: int,
    tool_calls: int,
    termination_reason: str = "env_done",
) -> Sample:
    return Sample(
        index=index,
        tokens=[1, 2, 3],
        response_length=1,
        reward=0.0,
        response="ok",
        metadata={
            "fused_traj_steps": steps,
            "fused_termination": termination_reason,
            "fused_reward_debug": {"tool_calls": tool_calls},
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "data_source": source,
                    }
                }
            }
        },
    )


def test_format_eval_results_table_includes_overall_and_sources():
    args = SimpleNamespace(
        eval_datasets=[SimpleNamespace(name="search_r1", n_samples_per_eval_prompt=1)],
        n_samples_per_eval_prompt=1,
    )
    data = {
        "search_r1": {
            "rewards": [1.0, 0.0, 1.0, 0.0],
            "samples": [
                _sample(0, "searchR1_nq", steps=2, tool_calls=1),
                _sample(1, "searchR1_nq", steps=4, tool_calls=3),
                _sample(2, "searchR1_triviaqa", steps=6, tool_calls=5),
                _sample(
                    3,
                    "searchR1_triviaqa",
                    steps=8,
                    tool_calls=7,
                    termination_reason="max_context_len_exceeded",
                ),
            ],
        }
    }

    assert format_eval_results_table(args, data) == (
        "Benchmark              pass@1 mean (%)   pass@1 std (%)  pass^1 mean (%)   pass^1 std (%)     # steps  # tool calls  # abnormal / all\n"
        "━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━\n"
        "overall / search_r1               50.0              0.0             50.0              0.0         5.0           4.0             1 / 4\n"
        "─────────────────────  ───────────────  ───────────────  ───────────────  ───────────────  ──────────  ────────────  ────────────────\n"
        "nq                                50.0              0.0             50.0              0.0         3.0           2.0             0 / 2\n"
        "─────────────────────  ───────────────  ───────────────  ───────────────  ───────────────  ──────────  ────────────  ────────────────\n"
        "triviaqa                          50.0              0.0             50.0              0.0         7.0           6.0             1 / 2"
    )


def test_format_eval_results_table_includes_power_of_two_pass_at_k_columns():
    args = SimpleNamespace(
        eval_datasets=[SimpleNamespace(name="search_r1", n_samples_per_eval_prompt=4)],
        n_samples_per_eval_prompt=1,
    )
    data = {
        "search_r1": {
            "rewards": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        }
    }

    table = format_eval_results_table(args, data)

    assert "pass@1 mean (%)" in table
    assert "pass@2 mean (%)" in table
    assert "pass@4 mean (%)" in table
    assert "search_r1" in table


def test_abnormal_column_counts_prompts_instead_of_sampled_trajectories():
    args = SimpleNamespace(
        eval_datasets=[SimpleNamespace(name="search_r1", n_samples_per_eval_prompt=2)],
        n_samples_per_eval_prompt=1,
    )
    data = {
        "search_r1": {
            "rewards": [0.0, 0.0, 1.0, 0.0],
            "samples": [
                _sample(0, "searchR1_nq", steps=2, tool_calls=1),
                _sample(1, "searchR1_nq", steps=2, tool_calls=1, termination_reason="timeout"),
                _sample(2, "searchR1_nq", steps=2, tool_calls=1),
                _sample(3, "searchR1_nq", steps=2, tool_calls=1),
            ],
        }
    }

    table = format_eval_results_table(args, data)

    assert table.count("1 / 2") == 2


def test_log_eval_results_table_keeps_default_logging_enabled(caplog):
    args = SimpleNamespace(eval_datasets=[], n_samples_per_eval_prompt=1)
    data = {"nq": {"rewards": [1.0]}}

    with caplog.at_level(logging.INFO, logger="slime_plugins.evals.results_table"):
        assert log_eval_results_table(0, args, data, None) is False

    assert "eval 0 results:" in caplog.text
    assert "100.0" in caplog.text
