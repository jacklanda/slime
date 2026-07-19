import json
import types

from slime.utils.episode_dump import save_rllm_episode_batch
from slime.utils.types import Sample


NUM_GPUS = 0


def _sample_with_episode() -> Sample:
    return Sample(
        metadata={
            "rllm_episode": {
                "id": "episode-1",
                "task": "answer",
                "is_correct": True,
                "trajectories": [
                    {
                        "name": "main",
                        "uid": "traj-1",
                        "reward": 1.0,
                        "steps": [
                            {
                                "observation": "q",
                                "thought": "",
                                "action": "<tool_call>...</tool_call>",
                                "model_response": "plain response",
                                "reward": 1.0,
                                "done": True,
                                "info": {"disable_thinking": True},
                            }
                        ],
                    }
                ],
            }
        }
    )


def test_eval_episode_dump_uses_evals_global_steps_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(
        wandb_project="FusedRL",
        wandb_group="run",
        grm_model="judge/model",
        grm_temperature=0.6,
        grm_max_input_tokens=131072,
        grm_max_new_tokens=1024,
    )
    eval_sample = _sample_with_episode()
    eval_sample.reward = 0.0
    eval_sample.metadata["grm"] = {
        "judge": "grm",
        "model": "judge/model",
        "mode": "equivalence",
        "judge_json": {"rationale": "Different answer.", "judgement": "Incorrect"},
        "score": 0.0,
    }

    save_rllm_episode_batch(args, rollout_id=0, samples=[_sample_with_episode()], mode="train")
    save_rllm_episode_batch(args, rollout_id=0, samples=[eval_sample], mode="eval")

    train_path = tmp_path / "experiments" / "logs" / "FusedRL" / "run" / "train" / "global_steps_0.json"
    eval_path = tmp_path / "experiments" / "logs" / "FusedRL" / "run" / "evals" / "global_steps_0.json"

    assert train_path.exists()
    assert eval_path.exists()
    dumped = json.loads(eval_path.read_text())
    assert dumped["mode"] == "eval"
    episode = dumped["trajectories"][0]
    assert episode["workflow_reward"] == 1.0
    assert episode["eval_reward"] == 0.0
    assert episode["is_correct"] is False
    assert episode["judge"] == "grm"
    assert episode["grm"] == {
        "model": "judge/model",
        "temperature": 0.6,
        "max_input_tokens": 131072,
        "max_new_tokens": 1024,
    }
    assert episode["mode"] == "equivalence"
    assert episode["judge_json"] == {"rationale": "Different answer.", "judgement": "Incorrect"}
    step = episode["trajectories"][0]["steps"][0]
    assert step["thought"] == ""
    assert step["model_response"] == "plain response"
    assert step["disable_thinking"] is True


def test_eval_episode_dump_records_rule_judge_without_grm_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(wandb_project="FusedRL", wandb_group="run")
    sample = _sample_with_episode()
    sample.metadata["grm"] = {"judge": "benchmark_verifier", "model": "benchmark_verifier", "score": 1.0}

    save_rllm_episode_batch(args, rollout_id=0, samples=[sample], mode="eval")

    dumped = json.loads(
        (tmp_path / "experiments" / "logs" / "FusedRL" / "run" / "evals" / "global_steps_0.json").read_text()
    )
    episode = dumped["trajectories"][0]
    assert episode["judge"] == "rule"
    assert "grm" not in episode


def test_eval_episode_dump_respects_episode_log_dir_sibling_evals(tmp_path, monkeypatch):
    episode_dir = tmp_path / "custom" / "episodes"
    monkeypatch.setenv("SLIME_EPISODE_LOG_DIR", str(episode_dir))
    args = types.SimpleNamespace(wandb_project="ignored", wandb_group="ignored")

    save_rllm_episode_batch(args, rollout_id=3, samples=[_sample_with_episode()], mode="eval")

    assert (tmp_path / "custom" / "evals" / "global_steps_3.json").exists()


def test_episode_dump_custom_root_separates_train_and_eval(tmp_path, monkeypatch):
    dump_root = tmp_path / "custom_dump"
    monkeypatch.setenv("SLIME_EPISODE_LOG_DIR", str(dump_root))
    args = types.SimpleNamespace(wandb_project="ignored", wandb_group="ignored")

    save_rllm_episode_batch(args, rollout_id=5, samples=[_sample_with_episode()], mode="train")
    save_rllm_episode_batch(args, rollout_id=5, samples=[_sample_with_episode()], mode="eval")

    assert (dump_root / "train" / "global_steps_5.json").exists()
    assert (dump_root / "evals" / "global_steps_5.json").exists()
