from __future__ import annotations

import re

from slime_plugins.evals.officeqa import load_official_reward


_FINAL_ANSWER_RE = re.compile(r"<FINAL_ANSWER>.*?</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE)


async def reward_func(args, sample, **kwargs) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    extra = metadata.get("extra_info") if isinstance(metadata.get("extra_info"), dict) else metadata
    reward_path = extra.get("official_reward_path")
    if not reward_path:
        raise ValueError("OfficeQA sample is missing official_reward_path")
    ground_truth = sample.label
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth")
    prediction = sample.response if isinstance(sample.response, str) else str(sample.response or "")
    if _FINAL_ANSWER_RE.search(prediction) is None:
        metadata["officeqa_scores"] = {"0.0%": 0.0, "0.1%": 0.0, "1.0%": 0.0, "5.0%": 0.0}
        metadata["officeqa_incomplete"] = True
        return 0.0
    scorer = load_official_reward(str(reward_path))
    try:
        scores = {
            label: float(scorer.score_answer(str(ground_truth), prediction, tolerance=tolerance))
            for label, tolerance in (("0.0%", 0.0), ("0.1%", 0.001), ("1.0%", 0.01), ("5.0%", 0.05))
        }
    except (TypeError, ValueError) as exc:
        scores = {"0.0%": 0.0, "0.1%": 0.0, "1.0%": 0.0, "5.0%": 0.0}
        metadata["officeqa_score_error"] = f"{type(exc).__name__}: {exc}"
    metadata["officeqa_scores"] = scores
    metadata["officeqa_incomplete"] = False
    return scores["0.0%"]
