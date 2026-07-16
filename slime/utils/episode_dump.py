import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def save_rllm_episode_batch(args, *, rollout_id: int, samples: list[Sample], mode: str = "train", epoch: int = 0) -> None:
    episodes = _collect_rllm_episodes(samples)
    if not episodes:
        return

    log_dir = _episode_log_dir(args, mode=mode)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / _batch_filename(rollout_id, mode, epoch)
    batch_data = {
        "training_step": rollout_id,
        "epoch": epoch,
        "mode": mode,
        "num_episodes": len(episodes),
        "trajectories": [_episode_to_batch_dict(episode, rollout_id, mode, epoch) for episode in episodes],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(batch_data, f, indent=4, ensure_ascii=False, default=str)
        f.write("\n")
    logger.info("Saved %s episode trajectories to %s", len(episodes), file_path)


def _collect_rllm_episodes(samples: list[Sample]) -> list[dict[str, Any]]:
    episodes = []
    seen_episode_ids = set()
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        episode = metadata.get("rllm_episode")
        if not isinstance(episode, dict):
            continue
        episode_id = str(episode.get("id") or "")
        if episode_id and episode_id in seen_episode_ids:
            continue
        if episode_id:
            seen_episode_ids.add(episode_id)
        episodes.append(episode)
    return episodes


def _episode_log_dir(args, mode: str = "train") -> Path:
    override = os.environ.get("SLIME_EPISODE_LOG_DIR")
    if override:
        root = Path(override)
        if root.name in {"episodes", "train", "evals"}:
            root = root.parent
        return root / ("evals" if mode == "eval" else "train")

    project = _safe_path_part(getattr(args, "wandb_project", None) or "slime")
    group = _safe_path_part(getattr(args, "wandb_group", None) or getattr(args, "wandb_run_id", None) or "default")
    subdir = "evals" if mode == "eval" else "train"
    return Path("experiments") / "logs" / project / group / subdir


def _batch_filename(step: int, mode: str, epoch: int) -> str:
    if mode in {"train", "eval"}:
        return f"global_steps_{step}.json"
    return f"{mode}_global_steps_{step}_epoch_{epoch}.json"


def _episode_to_batch_dict(episode: dict[str, Any], step: int, mode: str, epoch: int) -> dict[str, Any]:
    task = _sanitize_task(episode.get("task"))
    metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
    info = episode.get("info") if isinstance(episode.get("info"), dict) else {}
    timing = info.get("timing") if isinstance(info.get("timing"), dict) else {}
    return {
        "training_step": step,
        "epoch": epoch,
        "mode": mode,
        "episode_id": episode.get("id"),
        "session_id": episode.get("session_id"),
        "task": task,
        "task_hash": _compute_task_hash(task),
        "is_correct": bool(episode.get("is_correct")),
        "termination_reason": episode.get("termination_reason"),
        "metrics": episode.get("metrics") or {},
        "metadata": metadata,
        "timing": timing,
        "trajectories": [_trajectory_to_batch_dict(traj) for traj in episode.get("trajectories", []) if isinstance(traj, dict)],
    }


def _trajectory_to_batch_dict(trajectory: dict[str, Any]) -> dict[str, Any]:
    info = trajectory.get("info") if isinstance(trajectory.get("info"), dict) else {}
    timing = info.get("timing") if isinstance(info.get("timing"), dict) else {}
    steps = [step for step in trajectory.get("steps", []) if isinstance(step, dict)]
    return {
        "name": trajectory.get("name"),
        "uid": trajectory.get("uid"),
        "reward": trajectory.get("reward"),
        "num_steps": len(steps),
        "timing": timing,
        "steps": [_step_to_batch_dict(step) for step in steps],
    }


def _step_to_batch_dict(step: dict[str, Any]) -> dict[str, Any]:
    info = step.get("info") if isinstance(step.get("info"), dict) else {}
    timing = info.get("timing") if isinstance(info.get("timing"), dict) else {}
    result = {
        "observation": step.get("observation"),
        "thought": step.get("thought") or "",
        "action": step.get("action"),
        "reward": float(step.get("reward") or 0.0),
        "done": bool(step.get("done")),
        "model_response": step.get("model_response") or "",
        "chat_completions": step.get("chat_completions") or [],
        "disable_thinking": bool(info.get("disable_thinking")),
        "timing": timing,
    }
    if "tito_context_reason" in info:
        # Marks how the served prompt related to the TiTO accumulator; in
        # particular whether historical thinking was dropped before this step.
        result["tito_context_reason"] = info["tito_context_reason"]
        result["historical_thinking_discarded"] = bool(info.get("historical_thinking_discarded"))
    return result


def _compute_task_hash(task: Any, length: int = 8) -> str:
    task_str = json.dumps(task, sort_keys=True, default=str)
    return hashlib.sha256(task_str.encode("utf-8")).hexdigest()[:length]


def _sanitize_task(task: Any) -> Any:
    if isinstance(task, dict):
        return {k: v for k, v in task.items() if k not in ("image", "images")}
    return task


def _safe_path_part(value: Any) -> str:
    text = str(value or "default").strip() or "default"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
