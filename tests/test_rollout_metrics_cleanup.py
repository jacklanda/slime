from types import SimpleNamespace

import pytest

from slime.ray.rollout import (
    _compute_mcp_atlas_coverage_metrics,
    _format_eval_log_dict_for_display,
    _log_eval_rollout_data,
    compute_metrics_from_samples,
)
from slime.utils.metric_utils import compute_pass_at_k_and_pass_all
from slime.utils.types import Sample

NUM_GPUS = 0


def test_compute_metrics_from_samples_omits_response_aborted_ratio():
    args = SimpleNamespace(
        advantage_estimator="ppo",
        log_reward_category=None,
        rollout_max_response_len=8,
        rollout_max_prompt_len=16,
    )
    samples = [
        Sample(
            index=0,
            tokens=[1, 2, 3, 4],
            response_length=2,
            reward=0.0,
            response="ok",
            metadata={},
        ),
        Sample(
            index=1,
            tokens=[5, 6, 7, 8, 9],
            response_length=3,
            reward=0.0,
            response="done",
            metadata={},
            status=Sample.Status.ABORTED,
        ),
    ]

    metrics = compute_metrics_from_samples(args, samples)

    assert "response/aborted_ratio" not in metrics
    assert "response_length/mean" in metrics
    assert "prompt_length/mean" in metrics


def test_compute_metrics_from_samples_uses_compact_eval_token_lengths():
    args = SimpleNamespace(
        advantage_estimator="ppo",
        log_reward_category=None,
        rollout_max_response_len=8,
        rollout_max_prompt_len=16,
    )
    sample = Sample(
        tokens=[],
        response="compact",
        response_length=7,
        reward=1.0,
        metadata={
            "fused_prompt_length_tokens": 11,
            "fused_completion_length_tokens": 7,
        },
    )

    metrics = compute_metrics_from_samples(args, [sample])

    assert metrics["prompt_length/mean"] == 11.0
    assert metrics["response_length/mean"] == 7.0
    assert metrics["prompt_length/clip_ratio"] == 0.0
    assert metrics["response_length/clip_ratio"] == 0.0


def test_compute_pass_at_k_and_pass_all_uses_full_eval_group_size():
    metrics = compute_pass_at_k_and_pass_all([0.0, 1.0, 0.0, 1.0, 1.0, 1.0], group_size=3)

    assert metrics == {
        "pass@1/mean": 0.667,
        "pass@1/std": 0.236,
        "pass@2/mean": 0.833,
        "pass@2/std": 0.236,
        "pass@3/mean": 1.0,
        "pass@3/std": 0.236,
        "pass^3/mean": 0.5,
        "pass^3/std": 0.236,
    }


def test_compute_pass_at_k_and_pass_all_rounds_to_three_decimals():
    metrics = compute_pass_at_k_and_pass_all([0.0, 0.0, 1.0], group_size=1)

    assert metrics == {
        "pass@1/mean": 0.333,
        "pass@1/std": 0.0,
        "pass^1/mean": 0.333,
        "pass^1/std": 0.0,
    }


def test_mcp_atlas_coverage_metrics_report_official_thresholds_and_judge_health():
    samples = [
        Sample(metadata={"verification": {"per_claim": [{"score": 1.0}]}, "grm": {}}),
        Sample(
            metadata={
                "verification": {"per_claim": [{"score": 0.0, "error": "judge timeout"}]},
                "grm": {},
            }
        ),
        Sample(metadata={"verification": {"per_claim": []}, "grm": {"failure": "missing_submission"}}),
        Sample(metadata={"verification": {"per_claim": [{"score": 0.5}]}, "grm": {}}),
    ]

    metrics = _compute_mcp_atlas_coverage_metrics([1.0, 0.75, 0.5, 0.0], samples)

    assert metrics == {
        "coverage/mean": 0.5625,
        "coverage/pass_rate_0.50": 0.75,
        "coverage/pass_rate_0.75": 0.5,
        "coverage/judge_failure_ratio": 0.25,
        "coverage/missing_submission_ratio": 0.25,
    }


def test_eval_rollout_log_adds_pass_at_k_and_pass_all(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.logging_utils.log", lambda *args, **kwargs: None)
    monkeypatch.setattr("slime.ray.rollout.compute_rollout_step", lambda args, rollout_id: 0)
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        eval_datasets=[SimpleNamespace(name="frontierscience_olympiad", n_samples_per_eval_prompt=3)],
        n_samples_per_eval_prompt=1,
        log_passrate=False,
        num_rollout_per_epoch=1,
    )
    data = {
        "frontierscience_olympiad": {
            "rewards": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "truncated": [False, False, False, False, False, False],
        }
    }

    log_dict = _log_eval_rollout_data(rollout_id=0, args=args, data=data)

    assert "eval/frontierscience_olympiad/pass@3" not in log_dict
    assert "eval/frontierscience_olympiad/pass^3" not in log_dict
    assert log_dict["eval/frontierscience_olympiad/pass@1/mean"] == 0.667
    assert log_dict["eval/frontierscience_olympiad/pass@1/std"] == 0.236
    assert log_dict["eval/frontierscience_olympiad/pass@2/mean"] == 0.833
    assert log_dict["eval/frontierscience_olympiad/pass@2/std"] == 0.236
    assert log_dict["eval/frontierscience_olympiad/pass@3/mean"] == 1.0
    assert log_dict["eval/frontierscience_olympiad/pass@3/std"] == 0.236
    assert log_dict["eval/frontierscience_olympiad/pass^3/mean"] == 0.5
    assert log_dict["eval/frontierscience_olympiad/pass^3/std"] == 0.236


def test_eval_rollout_log_display_formats_percentage_metrics_only():
    display = _format_eval_log_dict_for_display(
        {
            "eval/search_r1": 0.20664,
            "eval/search_r1/pass@1/mean": 0.207,
            "eval/search_r1/rewards/web_search": 0.20664,
            "eval/search_r1/response_length/mean": 258.93184,
            "eval/search_r1/nq/num_problems": 500,
        }
    )

    assert display["eval/search_r1"] == 20.7
    assert display["eval/search_r1/pass@1/mean"] == 20.7
    assert display["eval/search_r1/rewards/web_search"] == 20.7
    assert display["eval/search_r1/response_length/mean"] == 258.93184
    assert display["eval/search_r1/nq/num_problems"] == 500


def test_eval_rollout_log_adds_source_pass_metrics(monkeypatch):
    monkeypatch.setattr("slime.ray.rollout.logging_utils.log", lambda *args, **kwargs: None)
    monkeypatch.setattr("slime.ray.rollout.compute_rollout_step", lambda args, rollout_id: 0)
    args = SimpleNamespace(
        custom_eval_rollout_log_function_path=None,
        eval_datasets=[SimpleNamespace(name="search_r1", n_samples_per_eval_prompt=2)],
        n_samples_per_eval_prompt=1,
        log_passrate=False,
        num_rollout_per_epoch=1,
        advantage_estimator="ppo",
        log_reward_category=None,
        rollout_max_response_len=8,
        rollout_max_prompt_len=16,
    )

    def sample(index: int, source: str, *, steps: int | None = None, tool_calls: int | None = None) -> Sample:
        metadata = {
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "data_source": source,
                    }
                }
            }
        }
        if steps is not None:
            metadata["fused_traj_steps"] = steps
        if tool_calls is not None:
            metadata["fused_reward_debug"] = {"tool_calls": tool_calls}
        return Sample(
            index=index,
            tokens=[1, 2, 3],
            response_length=1,
            reward=0.0,
            response="ok",
            metadata=metadata,
        )

    data = {
        "search_r1": {
            "rewards": [0.0, 1.0, 0.0, 0.0],
            "truncated": [False, False, False, False],
            "samples": [
                sample(0, "searchR1_nq", steps=2, tool_calls=1),
                sample(1, "searchR1_nq", steps=4, tool_calls=3),
                sample(2, "searchR1_triviaqa", steps=5, tool_calls=4),
                sample(3, "searchR1_triviaqa"),
            ],
        }
    }

    log_dict = _log_eval_rollout_data(rollout_id=0, args=args, data=data)

    assert "eval/search_r1/nq" not in log_dict
    assert "eval/search_r1/nq/num_samples" not in log_dict
    assert log_dict["eval/search_r1/nq/num_episdoes"] == 2
    assert log_dict["eval/search_r1/nq/num_problems"] == 1
    assert log_dict["eval/search_r1/nq/steps"] == 3.0
    assert log_dict["eval/search_r1/nq/tool_calls"] == 2.0
    assert "eval/search_r1/nq/pass@2" not in log_dict
    assert "eval/search_r1/nq/pass^2" not in log_dict
    assert log_dict["eval/search_r1/nq/pass@2/mean"] == 1.0
    assert log_dict["eval/search_r1/nq/pass@2/std"] == 0.5
    assert log_dict["eval/search_r1/nq/pass^2/mean"] == 0.0
    assert log_dict["eval/search_r1/nq/pass^2/std"] == 0.5
    assert "eval/search_r1/triviaqa" not in log_dict
    assert "eval/search_r1/triviaqa/num_samples" not in log_dict
    assert log_dict["eval/search_r1/triviaqa/num_episdoes"] == 2
    assert log_dict["eval/search_r1/triviaqa/num_problems"] == 1
    assert log_dict["eval/search_r1/triviaqa/steps"] == 5.0
    assert log_dict["eval/search_r1/triviaqa/tool_calls"] == 4.0
    assert "eval/search_r1/triviaqa/pass@2" not in log_dict
    assert "eval/search_r1/triviaqa/pass^2" not in log_dict
    assert log_dict["eval/search_r1/triviaqa/pass@2/mean"] == 0.0
    assert log_dict["eval/search_r1/triviaqa/pass@2/std"] == 0.0
    assert log_dict["eval/search_r1/triviaqa/pass^2/mean"] == 0.0
    assert log_dict["eval/search_r1/triviaqa/pass^2/std"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
