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
                        "steps": [{"observation": "q", "reward": 1.0, "done": True}],
                    }
                ],
            }
        }
    )


def test_eval_episode_dump_uses_evals_global_steps_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = types.SimpleNamespace(wandb_project="FusedRL", wandb_group="run")

    save_rllm_episode_batch(args, rollout_id=0, samples=[_sample_with_episode()], mode="train")
    save_rllm_episode_batch(args, rollout_id=0, samples=[_sample_with_episode()], mode="eval")

    train_path = tmp_path / "experiments" / "logs" / "FusedRL" / "run" / "train" / "global_steps_0.json"
    eval_path = tmp_path / "experiments" / "logs" / "FusedRL" / "run" / "evals" / "global_steps_0.json"

    assert train_path.exists()
    assert eval_path.exists()
    assert json.loads(eval_path.read_text())["mode"] == "eval"


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
