"""OpenRouter LM-as-a-Judge reward function.

Use with:
    --custom-rm-path slime.rollout.rm_hub.openrouter_grm.reward_func

The function is fully async and supports both single-sample and batched
``--group-rm`` calls.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import random
from typing import Any

import httpx
import tiktoken

from slime.rollout.rm_hub import benchmark_verifier
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.f1 import f1_score
from slime.rollout.rm_hub.gpqa import compute_gpqa_reward
from slime.rollout.rm_hub.math_dapo_utils import compute_score as compute_score_dapo
from slime.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from slime.rollout.rm_hub.math_utils import grade_answer_verl
from slime.rollout.fused_agent.parser import ActiveFinishParser, _extract_boxed
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_CLIENT: httpx.AsyncClient | None = None
_SEMAPHORE: asyncio.Semaphore | None = None
_FINISH_PARSER = ActiveFinishParser(valid_tools={"finish", "submit"})
_GRM_TOKENIZER = tiktoken.get_encoding("o200k_base")


_DEFAULT_SYSTEM_PROMPT = (
    'You are a strict binary answer judge. Return only JSON: {"score": 0} or {"score": 1}. '
    "Score 1 if the final submitted answer correctly answers the question given the accepted ground truth. "
    "Accept semantic aliases, abbreviations, equivalent dates or numbers, and multiple-choice letters that map to the "
    "correct option. Score 0 for contradictions, wrong alternatives, evasions, or no submitted answer. "
    "Ignore style, verbosity, and irrelevant intermediate mistakes if the final answer is correct."
)
_EQUIVALENCE_SYSTEM_PROMPT = "You are an evaluation assistant."
_MCP_ATLAS_SYSTEM_PROMPT = (
    "You are evaluating whether a model response covers one expert-defined claim. "
    "Return a rigorous structured judgement based only on the claim and response."
)

_SCORE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "binary_reward",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "integer", "enum": [0, 1]}},
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}

_EQUIVALENCE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_equivalence_judgement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rationale": {"type": "string", "description": "Brief rationale for the judgement."},
                "judgement": {
                    "type": "string",
                    "enum": ["Correct", "Incorrect"],
                    "description": "Whether the predicted answer is equivalent to the labeled answer.",
                },
            },
            "required": ["rationale", "judgement"],
            "additionalProperties": False,
        },
    },
}

_MCP_ATLAS_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "mcp_atlas_claim_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string"},
                "coverage_outcome": {
                    "type": "string",
                    "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"],
                },
                "justification": {"type": "string"},
                "confidence_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["claim_text", "coverage_outcome", "justification", "confidence_level"],
            "additionalProperties": False,
        },
    },
}

_MCP_ATLAS_COVERAGE_SCORES = {
    "fulfilled": 1.0,
    "partially_fulfilled": 0.5,
    "not_fulfilled": 0.0,
}


async def reward_func(args, sample_or_samples: Sample | list[Sample], **kwargs):
    _FINISH_PARSER.set(getattr(args, "hf_checkpoint", None))
    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    evaluation = bool(kwargs.get("evaluation", False))

    rewards = await asyncio.gather(*[_score_one(args, sample, evaluation=evaluation) for sample in samples])

    return rewards if isinstance(sample_or_samples, list) else rewards[0]


async def _score_one(args, sample: Sample, *, evaluation: bool) -> float:
    if evaluation and _is_mcp_atlas_sample(sample):
        return await _score_mcp_atlas_sample(args, sample)

    benchmark_eval = evaluation and _uses_benchmark_verifier(args, sample)
    if benchmark_eval:
        rule_reward = float(await benchmark_verifier.reward_func(args, sample))
        if rule_reward > 0 and _benchmark_rule_match_is_decisive(sample):
            sample.metadata.setdefault("grm", {})
            sample.metadata["grm"].update(
                {
                    "model": "benchmark_verifier",
                    "score": rule_reward,
                    "evaluation": True,
                    "fallback": False,
                    "judge": "benchmark_verifier",
                    "hybrid_rule_match": True,
                }
            )
            sample.metadata["verification"] = benchmark_verifier.get_verification_details(
                sample, score=rule_reward, verifier="rule", model="benchmark_verifier"
            )
            return rule_reward

    final_answer_step = _benchmark_final_answer_step(sample) if benchmark_eval else _final_answer_step(sample)
    if sample.label is None or not final_answer_step:
        reward = float(getattr(args, "grm_failure_reward", 0.0))
        if evaluation:
            sample.metadata["verification"] = benchmark_verifier.get_verification_details(
                sample,
                score=reward,
                verifier="model",
                model=getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
            )
        return reward

    async with _get_semaphore(args):
        payload = _build_payload(args, sample, final_answer_step=final_answer_step, include_question=benchmark_eval)
        max_retries = max(1, int(getattr(args, "grm_max_retries", 3)))
        for attempt in range(max_retries):
            try:
                response = await _get_client(args).post("/chat/completions", json=payload)
                response.raise_for_status()
                response_json = response.json()
                if _grm_mode(args) == "equivalence":
                    judge_json = _parse_judge_json(response_json)
                    reward = _judge_reward(judge_json)
                else:
                    judge_json = _parse_score_json(response_json)
                    reward = float(judge_json["score"])
                sample.metadata.setdefault("grm", {})
                sample.metadata["grm"].update(
                    {
                        "model": getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
                        "judge": "grm",
                        "mode": _grm_mode(args),
                        "judge_json": judge_json,
                        "score": reward,
                        "evaluation": evaluation,
                        "fallback": False,
                    }
                )
                sample.metadata["verification"] = benchmark_verifier.get_verification_details(
                    sample,
                    score=reward,
                    verifier="model",
                    model=getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
                    judge_json=judge_json,
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
                    if benchmark_eval:
                        reward = float(getattr(args, "grm_failure_reward", 0.0))
                        sample.metadata.setdefault("grm", {})
                        sample.metadata["grm"].update(
                            {
                                "model": getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
                                "judge": "grm",
                                "mode": _grm_mode(args),
                                "score": reward,
                                "evaluation": True,
                                "fallback": True,
                                "fallback_rm_type": "grm_failure_reward",
                                "error": repr(exc),
                            }
                        )
                        sample.metadata["verification"] = benchmark_verifier.get_verification_details(
                            sample,
                            score=reward,
                            verifier="model",
                            model=getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
                        )
                        return reward
                    return await _fallback_rule_based_reward(args, sample, evaluation=evaluation, error=exc)
                await asyncio.sleep(_retry_sleep(args, attempt))

    return await _fallback_rule_based_reward(args, sample, evaluation=evaluation)


async def _score_mcp_atlas_sample(args, sample: Sample) -> float:
    claims = _mcp_atlas_claims(sample)
    if not claims:
        raise ValueError("MCP-Atlas evaluation requires GTFA_CLAIMS in the sample label or metadata.")

    final_answer = _benchmark_final_answer_step(sample)
    if not final_answer:
        return _record_mcp_atlas_result(args, sample, claims, [], score=0.0, failure="missing_submission")

    results = await asyncio.gather(
        *[_evaluate_mcp_atlas_claim(args, claim, final_answer) for claim in claims]
    )
    score = round(sum(result["score"] for result in results) / len(claims), 3)
    return _record_mcp_atlas_result(args, sample, claims, results, score=score)


async def _evaluate_mcp_atlas_claim(args, claim: str, response: str) -> dict[str, Any]:
    payload = _build_mcp_atlas_payload(args, claim, response)
    max_retries = max(1, int(getattr(args, "grm_max_retries", 3)))
    async with _get_semaphore(args):
        for attempt in range(max_retries):
            try:
                api_response = await _get_client(args).post("/chat/completions", json=payload)
                api_response.raise_for_status()
                judgement = _parse_mcp_atlas_judgement(api_response.json())
                return {
                    "claim": claim,
                    "score": _MCP_ATLAS_COVERAGE_SCORES[judgement["coverage_outcome"]],
                    **judgement,
                }
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= max_retries:
                    logger.warning("MCP-Atlas claim judge failed after %d attempts: %r", max_retries, exc)
                    return {
                        "claim": claim,
                        "score": 0.0,
                        "coverage_outcome": "not_fulfilled",
                        "justification": f"Evaluation failed: {exc!r}",
                        "confidence_level": 0.1,
                        "error": repr(exc),
                    }
                await asyncio.sleep(_retry_sleep(args, attempt))

    raise AssertionError("unreachable")


def _build_mcp_atlas_payload(args, claim: str, response: str) -> dict[str, Any]:
    model = getattr(args, "grm_model", "deepseek/deepseek-v4-flash")
    prompt = _fit_mcp_atlas_prompt(args, claim, response)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _MCP_ATLAS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(getattr(args, "grm_temperature", 0.0)),
        "max_tokens": int(getattr(args, "grm_max_new_tokens", 128)),
        "response_format": (
            {"type": "json_object"} if str(model).startswith("google/") else _MCP_ATLAS_RESPONSE_FORMAT
        ),
        "provider": {"require_parameters": True},
    }
    if str(model).startswith("deepseek/"):
        payload["reasoning"] = {"effort": "none"}
    return payload


def _fit_mcp_atlas_prompt(args, claim: str, response: str) -> str:
    prefix = f"""SCORING CRITERIA:
- fulfilled: The response completely and accurately covers all key details in the claim.
- partially_fulfilled: The response covers some, but not all, key details in the claim.
- not_fulfilled: The response does not substantively cover the claim.

NUMERICAL GUIDELINES:
- Treat values within 5% as matching unless exact precision is essential.
- For percentages, allow a difference of 1 percentage point.
- Treat mathematically equivalent expressions as matching.

CLAIM TO EVALUATE:
{claim}

MODEL RESPONSE TO ANALYZE:
"""
    suffix = """

Return JSON with claim_text, coverage_outcome, a concise justification, and confidence_level from 0 to 1."""
    max_input_tokens = int(getattr(args, "grm_max_input_tokens", 24000))
    fixed_tokens = len(_GRM_TOKENIZER.encode_ordinary(_MCP_ATLAS_SYSTEM_PROMPT + prefix + suffix))
    if fixed_tokens > max_input_tokens:
        raise ValueError(
            f"--grm-max-input-tokens={max_input_tokens} is smaller than the MCP-Atlas fixed judge input "
            f"({fixed_tokens} tokens)"
        )
    response_tokens = _GRM_TOKENIZER.encode_ordinary(response)
    response_tokens = response_tokens[: max_input_tokens - fixed_tokens]
    return prefix + _GRM_TOKENIZER.decode(response_tokens) + suffix


def _parse_mcp_atlas_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_json_object(_response_content(payload), "coverage_outcome")
    outcome = str(parsed.get("coverage_outcome", "")).strip().lower()
    if outcome not in _MCP_ATLAS_COVERAGE_SCORES:
        raise ValueError(f"Unsupported MCP-Atlas coverage_outcome: {outcome!r}")
    try:
        confidence = float(parsed.get("confidence_level", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid MCP-Atlas confidence_level: {parsed.get('confidence_level')!r}") from exc
    return {
        "coverage_outcome": outcome,
        "justification": str(parsed.get("justification", "")).strip(),
        "confidence_level": min(1.0, max(0.0, confidence)),
    }


def _record_mcp_atlas_result(
    args,
    sample: Sample,
    claims: list[str],
    results: list[dict[str, Any]],
    *,
    score: float,
    failure: str | None = None,
) -> float:
    fully_covered = sum(result.get("score") == 1.0 for result in results)
    partially_covered = sum(result.get("score") == 0.5 for result in results)
    details = {
        "model": getattr(args, "grm_model", "deepseek/deepseek-v4-flash"),
        "judge": "mcp_atlas_claims",
        "mode": "claim_coverage",
        "score": score,
        "evaluation": True,
        "fallback": failure is not None or any("error" in result for result in results),
        "total_claims": len(claims),
        "fully_covered_claims": fully_covered,
        "partially_covered_claims": partially_covered,
        "per_claim": results,
    }
    if failure:
        details["failure"] = failure
    sample.metadata.setdefault("grm", {}).update(details)
    sample.metadata["verification"] = {
        "verifier": "model",
        "model": details["model"],
        "protocol": "mcp_atlas_claim_coverage",
        "score": score,
        "total_claims": len(claims),
        "fully_covered_claims": fully_covered,
        "partially_covered_claims": partially_covered,
        "per_claim": results,
        **({"failure": failure} if failure else {}),
    }
    return score


def _is_mcp_atlas_sample(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    source = str(metadata.get("data_source") or metadata.get("benchmark") or "").lower()
    normalized_source = source.replace("-", "_").replace(" ", "_")
    return normalized_source == "mcp_atlas" or bool(metadata.get("mcp_atlas_eval"))


def _mcp_atlas_claims(sample: Sample) -> list[str]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    raw_claims = metadata.get("GTFA_CLAIMS") or metadata.get("gtfa_claims") or metadata.get("claims")
    if raw_claims is None:
        raw_claims = sample.label
    if raw_claims is None:
        return []
    if isinstance(raw_claims, (list, tuple)):
        return [str(claim).strip() for claim in raw_claims if str(claim).strip()]
    if not isinstance(raw_claims, str):
        raw_claims = str(raw_claims)
    raw_claims = raw_claims.strip()
    if not raw_claims:
        return []
    if raw_claims.startswith("[") and raw_claims.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw_claims)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, list):
                return [str(claim).strip() for claim in parsed if str(claim).strip()]
    claims = []
    for line in raw_claims.replace("||", "\n").splitlines():
        claim = line.strip().lstrip("-*• ").strip()
        if claim:
            claims.append(claim)
    return claims or [raw_claims]


def _get_client(args) -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        base_url = getattr(args, "grm_base_url", None)
        api_key = getattr(args, "grm_openrouter_api_key", None) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            if base_url:
                api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenRouter GRM requires OPENROUTER_API_KEY or --grm-openrouter-api-key; "
                "a custom GRM endpoint may instead use OPENAI_API_KEY with OPENAI_BASE_URL."
            )

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
            base_url=(base_url or "https://openrouter.ai/api/v1").rstrip("/"),
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


def _build_payload(
    args,
    sample: Sample,
    *,
    final_answer_step: str | None = None,
    include_question: bool = False,
) -> dict[str, Any]:
    model = getattr(args, "grm_model", "deepseek/deepseek-v4-flash")
    system_prompt = getattr(args, "grm_system_prompt", None) or (
        _EQUIVALENCE_SYSTEM_PROMPT if _grm_mode(args) == "equivalence" else _DEFAULT_SYSTEM_PROMPT
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_judge_prompt(
                    args,
                    sample,
                    final_answer_step=final_answer_step,
                    include_question=include_question,
                    system_prompt=system_prompt,
                ),
            },
        ],
        "temperature": float(getattr(args, "grm_temperature", 0.0)),
        "max_tokens": int(getattr(args, "grm_max_new_tokens", 16)),
        "response_format": (
            {"type": "json_object"}
            if str(model).startswith("google/")
            else (_EQUIVALENCE_RESPONSE_FORMAT if _grm_mode(args) == "equivalence" else _SCORE_RESPONSE_FORMAT)
        ),
        "provider": {"require_parameters": True},
    }
    if str(model).startswith("deepseek/"):
        payload["reasoning"] = {"effort": "none"}
    return payload


def _build_judge_prompt(
    args,
    sample: Sample,
    *,
    final_answer_step: str | None = None,
    include_question: bool = False,
    system_prompt: str | None = None,
) -> str:
    max_input_tokens = int(getattr(args, "grm_max_input_tokens", 24000))
    if max_input_tokens < 1:
        raise ValueError("--grm-max-input-tokens must be >= 1")
    system_prompt = (
        system_prompt
        if system_prompt is not None
        else getattr(args, "grm_system_prompt", None)
        or (_EQUIVALENCE_SYSTEM_PROMPT if _grm_mode(args) == "equivalence" else _DEFAULT_SYSTEM_PROMPT)
    )
    final_answer_step = final_answer_step if final_answer_step is not None else _final_answer_step(sample)
    if _grm_mode(args) != "equivalence":
        question = _judge_question(sample) if include_question else ""
        question_section = f"Question:\n{question}\n\n" if question else ""
        prefix = question_section + "Accepted ground truth:\n" + f"{_stringify(sample.label)}\n\n" + "Final submitted answer:\n"
        suffix = "\n\nDoes the final submitted answer correctly answer the question? Return only JSON."
    else:
        question = _judge_question(sample)
        prefix = (
            "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.\n\n"
            f"Question: {question}\n\nLabeled Answer: {_stringify(sample.label)}\n\nPredicted Answer: "
        )
        suffix = (
            "\n\nDid the model give an answer **equivalent** to the labeled answer? Please respond with \"Correct\" if they are equivalent, or \"Incorrect\" if they are not equivalent.\n\n"
            "The output should be in the following json format:\n"
            "{\n    \"rationale\": your rationale for the judgement, as a text,\n"
            "    \"judgement\": your judgement, can only be \"Correct\" or \"Incorrect\",\n}"
        )
    fixed_tokens = len(_GRM_TOKENIZER.encode_ordinary(system_prompt)) + len(
        _GRM_TOKENIZER.encode_ordinary(prefix + suffix)
    )
    if fixed_tokens > max_input_tokens:
        raise ValueError(
            f"--grm-max-input-tokens={max_input_tokens} is smaller than the fixed judge input ({fixed_tokens} tokens)"
        )

    answer_tokens = _GRM_TOKENIZER.encode_ordinary(final_answer_step)
    answer_budget = max_input_tokens - fixed_tokens
    answer_tokens = answer_tokens[-answer_budget:] if answer_budget else []
    while True:
        prompt = prefix + _GRM_TOKENIZER.decode(answer_tokens) + suffix
        input_tokens = len(_GRM_TOKENIZER.encode_ordinary(system_prompt)) + len(
            _GRM_TOKENIZER.encode_ordinary(prompt)
        )
        if input_tokens <= max_input_tokens:
            return prompt
        if not answer_tokens:
            raise ValueError("Unable to fit the fixed judge input within --grm-max-input-tokens")
        answer_tokens = answer_tokens[input_tokens - max_input_tokens :]


def _uses_benchmark_verifier(args, sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or getattr(args, "rm_type", None) or "").strip()
    return rm_type == "benchmark_verifier" or bool(metadata.get("benchmark_eval"))


def _judge_question(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = ""
    for key in ("question", "query", "input", "problem"):
        value = metadata.get(key)
        if value:
            question = _stringify(value)
            break
    if not question:
        question = _stringify(sample.prompt)

    options = metadata.get("options") or metadata.get("choices")
    if isinstance(options, dict):
        options = list(options.values())
    if isinstance(options, (list, tuple)) and options and not all(str(option) in question for option in options):
        option_lines = [f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)]
        question = f"{question}\nOptions:\n" + "\n".join(option_lines)
    return question


def _benchmark_final_answer_step(sample: Sample) -> str | None:
    answer = benchmark_verifier._extract_final_answer(_stringify(sample.response))
    if answer:
        return answer
    episode_answer = _rllm_episode_final_answer_step(sample)
    if not episode_answer:
        return None
    return benchmark_verifier._extract_final_answer(episode_answer) or episode_answer


def _benchmark_rule_match_is_decisive(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    data_source = str(metadata.get("data_source") or metadata.get("benchmark") or "").lower()
    if data_source in {"medqa", "scienceqa", "gpqa", "gpqa_diamond"}:
        return True
    if data_source in {"browsecomp_plus", "frontierscience_research"}:
        return False

    prediction = _benchmark_final_answer_step(sample)
    if not prediction:
        return False
    for answer in benchmark_verifier._candidate_answers(sample.label, metadata):
        if benchmark_verifier._normalize_answer_for_benchmark(
            prediction
        ) == benchmark_verifier._normalize_answer_for_benchmark(answer):
            return True
        prediction_dates = benchmark_verifier._date_variants(prediction)
        answer_dates = benchmark_verifier._date_variants(answer)
        if prediction_dates and answer_dates and prediction_dates & answer_dates:
            return True
    return False


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

    rllm_episode_answer = _rllm_episode_final_answer_step(sample)
    if rllm_episode_answer:
        return rllm_episode_answer

    return _last_nonempty_response_chunk(response)


def _rllm_episode_final_answer_step(sample: Sample) -> str | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    episode = metadata.get("rllm_episode")
    if not isinstance(episode, dict):
        return None

    for action in reversed(_rllm_episode_actions(episode)):
        finish_call = _extract_last_finish_call(action)
        if finish_call:
            return finish_call
        boxed_span = _extract_last_boxed_span(action)
        if boxed_span:
            return boxed_span
    return None


def _rllm_episode_actions(episode: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    trajectories = episode.get("trajectories")
    if not isinstance(trajectories, list):
        return actions
    for trajectory in trajectories:
        if not isinstance(trajectory, dict):
            continue
        steps = trajectory.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if action:
                actions.append(_stringify(action))
    return actions


def _extract_last_finish_call(text: str) -> str | None:
    calls = _FINISH_PARSER.get().parse(text)
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


def _grm_mode(args) -> str:
    mode = str(getattr(args, "grm_mode", "score") or "score").strip().lower()
    if mode not in {"score", "equivalence"}:
        raise ValueError(f"Unsupported GRM mode {mode!r}; expected 'score' or 'equivalence'.")
    return mode


def _response_content(payload: dict[str, Any]) -> str:
    message = payload["choices"][0]["message"]
    content = message.get("content")
    if content is None:
        content = message.get("reasoning") or _reasoning_details_text(message.get("reasoning_details"))
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content).strip()


def _parse_json_object(text: str, required_key: str) -> dict[str, Any]:
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and required_key in candidate:
                parsed = candidate
        if parsed is None:
            raise ValueError(f"GRM response did not contain valid JSON: {text!r}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"GRM response JSON must be an object, got {type(parsed).__name__}")
    if required_key not in parsed:
        raise ValueError(f"GRM response JSON is missing required key {required_key!r}")
    return parsed


def _parse_score_json(payload: dict[str, Any]) -> dict[str, int]:
    parsed = _parse_json_object(_response_content(payload), "score")
    score = parsed.get("score")
    if score in (0, 1, 0.0, 1.0, "0", "1"):
        return {"score": int(float(score))}
    raise ValueError(f"GRM JSON score must be 0 or 1, got {score!r}")


def _parse_judge_json(payload: dict[str, Any]) -> dict[str, str]:
    parsed = _parse_json_object(_response_content(payload), "judgement")
    judgement = str(parsed.get("judgement", "")).strip().strip("`*_.,:;!").lower()
    if judgement not in {"correct", "incorrect"}:
        raise ValueError(f'GRM JSON judgement must be "Correct" or "Incorrect", got {parsed.get("judgement")!r}')
    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = _stringify(rationale)
    return {"rationale": rationale, "judgement": judgement.capitalize()}


def _judge_reward(judge_json: dict[str, str]) -> float:
    return 1.0 if judge_json["judgement"] == "Correct" else 0.0


def _parse_reward(payload: dict[str, Any]) -> float:
    """Compatibility helper for callers expecting a score-mode reward."""
    return float(_parse_score_json(payload)["score"])


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
