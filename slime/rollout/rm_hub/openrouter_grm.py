"""OpenRouter LM-as-a-Judge reward function.

Use with:
    --custom-rm-path slime.rollout.rm_hub.openrouter_grm.reward_func

The function is fully async and supports both single-sample and batched
``--group-rm`` calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from typing import Any

import httpx

from slime.rollout.rm_hub import benchmark_verifier
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.f1 import f1_score
from slime.rollout.rm_hub.gpqa import compute_gpqa_reward
from slime.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from slime.rollout.rm_hub.math_utils import grade_answer_verl
from slime.rollout.fused_agent.parser import QwenToolParser, _extract_boxed
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_CLIENT: httpx.AsyncClient | None = None
_SEMAPHORE: asyncio.Semaphore | None = None
_FINISH_PARSER = QwenToolParser(valid_tools={"finish", "submit"})

_DEFAULT_SYSTEM_PROMPT = (
    "You are a strict binary answer judge. Return only JSON: {\"score\": 0} or {\"score\": 1}. "
    "Score 1 if the trajectory contains, implies, or finally answers the ground truth correctly. "
    "Score 0 otherwise. Ignore style, verbosity, and irrelevant intermediate mistakes if the final answer is correct."
)


async def reward_func(args, sample_or_samples: Sample | list[Sample], **kwargs):
    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    evaluation = bool(kwargs.get("evaluation", False))

    rewards = await asyncio.gather(*[_score_one(args, sample, evaluation=evaluation) for sample in samples])

    return rewards if isinstance(sample_or_samples, list) else rewards[0]


async def _score_one(args, sample: Sample, *, evaluation: bool) -> float:
    if not _has_answer_material(sample):
        return float(getattr(args, "grm_failure_reward", 0.0))

    async with _get_semaphore(args):
        payload = _build_payload(args, sample)
        max_retries = max(1, int(getattr(args, "grm_max_retries", 3)))
        for attempt in range(max_retries):
            try:
                response = await _get_client(args).post("/chat/completions", json=payload)
                response.raise_for_status()
                reward = _parse_reward(response.json())
                sample.metadata.setdefault("grm", {})
                sample.metadata["grm"].update(
                    {
                        "model": getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
                        "score": reward,
                        "evaluation": evaluation,
                        "fallback": False,
                    }
                )
                return reward
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= max_retries:
                    logger.warning(
                        "OpenRouter GRM failed after %d attempts for sample index=%s: %r",
                        max_retries,
                        getattr(sample, "index", None),
                        exc,
                    )
                    return await _fallback_rule_based_reward(args, sample, evaluation=evaluation, error=exc)
                await asyncio.sleep(_retry_sleep(args, attempt))

    return await _fallback_rule_based_reward(args, sample, evaluation=evaluation)


def _get_client(args) -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        api_key = getattr(args, "grm_openrouter_api_key", None) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OpenRouter GRM requires OPENROUTER_API_KEY or --grm-openrouter-api-key.")

        timeout = float(getattr(args, "grm_timeout", 60.0))
        max_connections = max(1, int(getattr(args, "grm_max_connections", 128)))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = getattr(args, "grm_openrouter_site_url", None) or os.environ.get("OPENROUTER_SITE_URL")
        title = getattr(args, "grm_openrouter_app_name", None) or os.environ.get("OPENROUTER_APP_NAME")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        _CLIENT = httpx.AsyncClient(
            base_url=(getattr(args, "grm_base_url", None) or "https://openrouter.ai/api/v1").rstrip("/"),
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            headers=headers,
            trust_env=False,
        )
    return _CLIENT


def _get_semaphore(args) -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(max(1, int(getattr(args, "grm_concurrency", 32))))
    return _SEMAPHORE


def _build_payload(args, sample: Sample) -> dict[str, Any]:
    return {
        "model": getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
        "messages": [
            {"role": "system", "content": getattr(args, "grm_system_prompt", None) or _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": _build_judge_prompt(args, sample)},
        ],
        "temperature": float(getattr(args, "grm_temperature", 0.0)),
        "max_tokens": int(getattr(args, "grm_max_tokens", 16)),
        "response_format": {"type": "json_object"},
    }


def _build_judge_prompt(args, sample: Sample) -> str:
    max_chars = int(getattr(args, "grm_max_trajectory_chars", 24000))
    final_answer_step = _final_answer_step(sample)
    if len(final_answer_step) > max_chars:
        final_answer_step = final_answer_step[-max_chars:]
    return (
        "Ground truth:\n"
        f"{_stringify(sample.label)}\n\n"
        "Final submitted answer step:\n"
        f"{final_answer_step}\n\n"
        "Does this final submitted answer correctly answer the ground truth? Return only JSON."
    )


def _sample_trajectory(sample: Sample) -> str:
    parts = []
    if sample.prompt:
        parts.append(_stringify(sample.prompt))
    if sample.response:
        parts.append(_stringify(sample.response))
    trajectory = "\n".join(parts).strip()
    if not trajectory and isinstance(sample.metadata, dict):
        for key in ("trajectory", "episode", "messages"):
            if key in sample.metadata:
                return _stringify(sample.metadata[key])
    return trajectory


def _final_answer_step(sample: Sample) -> str:
    response = _stringify(sample.response)
    finish_call = _extract_last_finish_call(response)
    if finish_call:
        return finish_call

    boxed_span = _extract_last_boxed_span(response)
    if boxed_span:
        return boxed_span

    if isinstance(sample.metadata, dict):
        for key in ("final_answer", "answer", "submitted_answer"):
            if key in sample.metadata:
                return _stringify(sample.metadata[key])

    return _last_nonempty_response_chunk(response)


def _extract_last_finish_call(text: str) -> str | None:
    calls = _FINISH_PARSER.parse(text)
    for call in reversed(calls):
        if call.name not in {"finish", "submit"}:
            continue
        if call.start is not None and call.end is not None:
            snippet = text[call.start : call.end].strip()
            if snippet:
                return snippet
        result = call.arguments.get("result")
        if result is not None:
            return _stringify(result).strip()
    return None


def _extract_last_boxed_span(text: str) -> str | None:
    search_end = len(text)
    while search_end > 0:
        prefix = text[:search_end]
        start = prefix.rfind("\\boxed{")
        if start < 0:
            return None
        boxed, _, _ = _extract_boxed(prefix[start:])
        if boxed:
            return prefix[start:].strip()
        search_end = start
    return None


def _last_nonempty_response_chunk(response: str, max_lines: int = 12) -> str:
    lines = [line for line in str(response or "").splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:]).strip()


def _parse_reward(payload: dict[str, Any]) -> float:
    message = payload["choices"][0]["message"]
    content = message.get("content")
    if content is None:
        content = message.get("reasoning") or _reasoning_details_text(message.get("reasoning_details"))
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    text = str(content).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\b([01])\b", text)
        if not match:
            raise ValueError(f"GRM response did not contain a binary score: {text!r}")
        return float(match.group(1))
    score = parsed.get("score")
    if score in (0, 1, 0.0, 1.0, "0", "1"):
        return float(score)
    raise ValueError(f"GRM JSON score must be 0 or 1, got {score!r}")


def _reasoning_details_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict):
            parts.append(str(item.get("text") or item.get("content") or ""))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _retry_sleep(args, attempt: int) -> float:
    base = float(getattr(args, "grm_retry_base_delay", 0.5))
    max_delay = float(getattr(args, "grm_retry_max_delay", 8.0))
    return min(max_delay, base * (2**attempt)) + random.random() * base


def _has_answer_material(sample: Sample) -> bool:
    return bool(sample.label is not None and _final_answer_step(sample))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


async def _fallback_rule_based_reward(args, sample: Sample, *, evaluation: bool = False, error: Exception | None = None) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or getattr(args, "rm_type", None) or "").strip()
    if rm_type == "benchmark_verifier" or metadata.get("benchmark_eval"):
        reward = float(await benchmark_verifier.reward_func(args, sample))
        sample.metadata.setdefault("grm", {})
        sample.metadata["grm"].update(
            {
                "score": reward,
                "evaluation": evaluation,
                "fallback": True,
                "fallback_rm_type": "benchmark_verifier",
                **({"error": repr(error)} if error is not None else {}),
            }
        )
        return reward

    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "deepscaler":
        return float(get_deepscaler_rule_based_reward(response, label))
    if rm_type == "dapo":
        result = compute_score_dapo(response, label)
        return float(result["score"] if isinstance(result, dict) else result)
    if rm_type == "math":
        return float(1 if grade_answer_verl(response, label) else 0)
    if rm_type == "f1":
        return float(f1_score(response, label)[0])
    if rm_type == "gpqa":
        return float(compute_gpqa_reward(response, label, metadata=metadata))
    if rm_type == "ifbench":
        from slime.rollout.rm_hub.ifbench import compute_ifbench_reward

        return float(compute_ifbench_reward(response, label, metadata=metadata))
    if rm_type == "random":
        return float(random.randint(0, 1))
    reward = float(getattr(args, "grm_failure_reward", 0.0))
    sample.metadata.setdefault("grm", {})
    sample.metadata["grm"].update(
        {
            "score": reward,
            "evaluation": evaluation,
            "fallback": True,
            "fallback_rm_type": rm_type or "grm_failure_reward",
            **({"error": repr(error)} if error is not None else {}),
        }
    )
    return reward
