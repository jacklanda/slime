"""Rule-based verifier for common search/QA benchmark evals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import string
from typing import Any

from slime.rollout.fused_agent.parser import ActiveFinishParser, _extract_boxed
from slime.rollout.rm_hub.f1 import normalize_answer
from slime.utils.types import Sample

_FINISH_PARSER = ActiveFinishParser(valid_tools={"finish", "submit"})
_OPTION_LETTERS = set(string.ascii_uppercase[:10])
_UNIT_SUFFIX_PATTERN = re.compile(
    r"^\s*(-?\d[\d,\.\s]*)\s*"
    r"(?:%|"
    r"usd|euros?|eur|gbp|dollars?|cents?|pounds?|yen|jpy|rmb|cny|"
    r"kg|kgs|g|mg|lb|lbs|oz|tons?|tonnes?|"
    r"km|m|cm|mm|mi|ft|in|inch|inches|yd|yards?|"
    r"k|m|b|bn|mn|millions?|billions?|thousands?|"
    r"j|kj|mj|cal|kcal|w|kw|mw|hp|"
    r"v|mv|kv|a|ma|hz|khz|mhz|ghz|"
    r"s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?|"
    r"people|persons|students|votes"
    r")\s*$",
    re.IGNORECASE,
)
_DATE_DMY_PATTERN = re.compile(r"^\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\s*$")
_DATE_ISO_PATTERN = re.compile(r"^\s*(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\s*$")
_LATEX_TEXT_WRAPPERS = (
    r"\operatorname",
    r"\underline",
    r"\textrm",
    r"\textbf",
    r"\textit",
    r"\textsf",
    r"\texttt",
    r"\mathbf",
    r"\mathrm",
    r"\mathit",
    r"\mathsf",
    r"\mathtt",
    r"\emph",
    r"\text",
)
_MCQ_COMMIT_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"(?:the\s+)?(?:correct\s+|final\s+|right\s+)?answer\s+(?:is|would\s+be|should\s+be|must\s+be|=|:)\s*\(?\*{0,2}([A-J])\*{0,2}\)?(?![A-Za-z0-9])",
        r"final\s+answer\s*(?:is|:|=)\s*\(?\*{0,2}([A-J])\*{0,2}\)?(?![A-Za-z0-9])",
        r"(?:option|choice)\s+\(?\*{0,2}([A-J])\*{0,2}\)?\s+is\s+(?:the\s+)?(?:correct|right|final|best|answer)\b",
        r"\(?\*{0,2}([A-J])\*{0,2}\)?\s+is\s+(?:the\s+)?(?:correct|right|final|best)\s+(?:answer|choice|option)\b",
        r"\bI(?:'?ll| will| am going to| have decided to| shall)?\s+(?:go\s+with|choose|select|pick|finalize|finalise|settle\s+on|decide\s+on|conclude\s+with)\s+(?:option\s+|choice\s+|with\s+)?\(?\*{0,2}([A-J])\*{0,2}\)?(?![A-Za-z0-9])",
        r"\bmy\s+(?:final\s+)?(?:answer|choice)\s+is\s*\(?\*{0,2}([A-J])\*{0,2}\)?(?![A-Za-z0-9])",
        r"\bgoing\s+with\s+(?:option\s+|choice\s+)?\(?([A-J])\)?(?![A-Za-z0-9])",
        r"\b(?:most\s+likely|likely|intended|probable|best|safe)\s+(?:answer|choice|option)\s*(?:is|would\s+be|:|=)?\s*\(?\*{0,2}([A-J])\*{0,2}\)?(?![A-Za-z0-9])",
        r"\b(?:this|that|it)\s+(?:matches|fits|corresponds\s+to|points\s+to)\s+(?:option|choice|answer)\s+\(?\*{0,2}([A-J])\*{0,2}\)?(?![A-Za-z0-9])",
    )
]
_MCQ_TENTATIVE_PREFIX = re.compile(
    r"(?:\bif\b|\bassum|\bsuppose|\bunless|\bmaybe\b|\bmight\b|\bperhaps\b|\bpossibl|\bwhat\s+if\b|\bcould\s+be\b|\bguess\b|\bnot\s+sure\b|\bunsure\b|\beither\b)",
    flags=re.IGNORECASE,
)
_MCQ_ENUM_FORWARD = re.compile(r"\s*(?:or|/|,|and|nor)\s+[A-J]\b", flags=re.IGNORECASE)
_MCQ_NEGATIVE_FOLLOW = re.compile(r"^\s*(?:is\s+)?(?:likely\s+)?(?:not|incorrect|wrong|unlikely|impossible|inconsistent|ruled\s+out|fails?|does\s+not)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class AnswerExtraction:
    text: str
    source: str


async def reward_func(args, sample_or_samples: Sample | list[Sample], **kwargs):
    _FINISH_PARSER.set(getattr(args, "hf_checkpoint", None))
    samples = sample_or_samples if isinstance(sample_or_samples, list) else [sample_or_samples]
    rewards = []
    for sample in samples:
        reward = _score_sample(sample)
        sample.metadata.setdefault("verification", get_verification_details(sample, score=reward))
        rewards.append(reward)
    return rewards if isinstance(sample_or_samples, list) else rewards[0]


def get_verification_details(
    sample: Sample,
    *,
    score: float,
    verifier: str = "rule",
    model: str | None = None,
    judge_json: Any = None,
) -> dict[str, Any]:
    """Return the answer evidence used by an evaluation verifier."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    data_source = str(metadata.get("data_source") or metadata.get("benchmark") or "").lower()
    allow_plain_text = data_source not in {"browsecomp_plus", "frontierscience_research"}
    extraction = _extract_final_answer_with_source(str(sample.response or ""), allow_plain_text=allow_plain_text)
    prediction = extraction.text if extraction is not None else None
    source = extraction.source if extraction is not None else None
    if extraction is None:
        episode = metadata.get("rllm_episode")
        if isinstance(episode, dict):
            episode_text = "\n".join(
                str(step.get("model_response") or step.get("action") or "")
                for trajectory in episode.get("trajectories", []) if isinstance(trajectory, dict)
                for step in trajectory.get("steps", []) if isinstance(step, dict)
            )
            extraction = _extract_final_answer_with_source(episode_text, allow_plain_text=allow_plain_text)
            if extraction is not None:
                prediction, source = extraction.text, extraction.source
    details: dict[str, Any] = {
        "verifier": verifier,
        "ground_truth": _reward_ground_truth(sample.label, metadata),
        "prediction": prediction,
        "prediction_source": source,
        "score": float(score),
    }
    if model is not None:
        details["model"] = model
    if judge_json is not None:
        details["judge_json"] = judge_json
    return details


def _score_sample(sample: Sample) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if metadata.get("eval_response_anomalies"):
        return 0.0
    data_source = str(metadata.get("data_source") or metadata.get("benchmark") or "").lower()
    label = sample.label

    if data_source == "browsecomp_plus":
        ground_truth = _reward_ground_truth(label, metadata)
        if not isinstance(ground_truth, dict):
            ground_truth = {"metric": "token_f1", "target": ground_truth}
        return _score_search_r1_like(sample.response, ground_truth)

    if data_source == "frontierscience_research":
        ground_truth = _reward_ground_truth(label, metadata)
        if not isinstance(ground_truth, dict):
            ground_truth = {"metric": "rouge_l", "target": ground_truth}
        return _score_search_r1_like(sample.response, ground_truth)

    if data_source in {"gpqa_diamond", "gpqa"}:
        return _score_choice_letter(sample.response, label, metadata=_gpqa_metadata(metadata, label))

    if data_source in {"medqa", "scienceqa"}:
        return _score_multiple_choice(sample.response, label, metadata.get("options"))

    answers = _candidate_answers(label, metadata)
    return _score_short_answer(
        sample.response,
        answers,
        strict_exact_match=bool(metadata.get("strict_exact_match")),
    )


def _gpqa_metadata(metadata: dict[str, Any], label: Any) -> dict[str, Any]:
    out = dict(metadata)
    if label is not None:
        out.setdefault("correct_letter", str(label).strip().upper())
    return out


def _score_multiple_choice(response: str, label: Any, options: Any) -> float:
    label_text = "" if label is None else str(label).strip()
    option_list = [str(option) for option in options] if isinstance(options, list) else []
    correct_letter = _letter_for_label(label_text, option_list)
    prediction = _extract_final_answer_with_source(response, allow_plain_text=True)
    if correct_letter:
        predicted_letter = _extract_answer_letter(response, len(option_list) or 10)
        if predicted_letter:
            return 1.0 if predicted_letter == correct_letter else 0.0

    if prediction is not None and option_list:
        mapped_option = _map_letter_to_option_text(prediction.text, option_list)
        if mapped_option != prediction.text:
            return _score_short_answer_text(mapped_option, [label_text])

    candidates = [label_text]
    if correct_letter and option_list:
        idx = string.ascii_uppercase.index(correct_letter)
        if idx < len(option_list):
            candidates.append(_strip_option_prefix(option_list[idx]))
    return _score_short_answer(response, candidates)


def _score_choice_letter(response: str, label: Any, metadata: dict[str, Any]) -> float:
    valid_letters = _valid_choice_letters(metadata)
    correct_letter = _correct_choice_letter(label, metadata, valid_letters)
    predicted_letter = _extract_answer_letter(response, len(valid_letters))
    if predicted_letter and correct_letter:
        return 1.0 if predicted_letter == correct_letter else 0.0
    if predicted_letter and isinstance(label, str):
        return 1.0 if predicted_letter == label.strip().upper() else 0.0
    return 0.0


def _valid_choice_letters(metadata: dict[str, Any]) -> list[str]:
    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)
    if choices:
        return list(string.ascii_uppercase[: len(choices)])
    valid_letters = metadata.get("valid_letters")
    if valid_letters:
        return [str(letter).strip().upper() for letter in valid_letters if str(letter).strip()]
    return list(string.ascii_uppercase[:10])


def _correct_choice_letter(label: Any, metadata: dict[str, Any], valid_letters: list[str]) -> str | None:
    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
        if correct_letter in valid_letters:
            return correct_letter
    if isinstance(label, str):
        label_text = label.strip().upper()
        if len(label_text) == 1 and label_text in valid_letters:
            return label_text
    if isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            return valid_letters[idx]
    return None


def _letter_for_label(label: str, options: list[str]) -> str | None:
    if len(label) == 1 and label.upper() in string.ascii_uppercase:
        return label.upper()
    normalized_label = normalize_answer(label)
    for idx, option in enumerate(options):
        body = _strip_option_prefix(option)
        if normalize_answer(body) == normalized_label or normalize_answer(option) == normalized_label:
            return string.ascii_uppercase[idx]
    return None


def _extract_answer_letter(response: str, num_options: int) -> str | None:
    extraction = _extract_final_answer_with_source(response, allow_plain_text=True)
    if extraction is None:
        return None
    text = _strip_latex_wrappers(extraction.text)
    valid = set(string.ascii_uppercase[:num_options])
    committed = _infer_committed_mcq_letter(text, valid)
    if committed:
        return committed
    direct = _parse_option_label(text)
    if direct and direct in valid:
        return direct
    patterns = [
        r"(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*([A-Z])\b",
        r"\b([A-Z])\s*(?:is\s*(?:the)?\s*correct)\b",
        r"<answer>\s*([A-Z])\s*</answer>",
        r"^\s*(?:the\s+)?(?:answer\s+is\s+)?([A-Z])\s*[\.\)]\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    candidates = re.findall(r"\b([A-Z])\b", text)
    for letter in reversed(candidates):
        letter = letter.upper()
        if letter in valid:
            return letter
    return None


def _candidate_answers(label: Any, metadata: dict[str, Any]) -> list[str]:
    answers = []
    for value in (label, metadata.get("answer"), metadata.get("ground_truth"), metadata.get("target")):
        if isinstance(value, list):
            answers.extend(str(item) for item in value if item is not None)
        elif value is not None:
            answers.append(str(value))
    seen = set()
    out = []
    for answer in answers:
        key = normalize_answer(answer)
        if key and key not in seen:
            seen.add(key)
            out.append(answer)
    return out


def _reward_ground_truth(label: Any, metadata: dict[str, Any]) -> Any:
    if isinstance(label, dict) and "ground_truth" in label:
        return label["ground_truth"]
    if isinstance(label, dict):
        return label
    value = metadata.get("ground_truth")
    if value is not None:
        return value
    value = metadata.get("target")
    if value is not None:
        return value
    value = metadata.get("answer")
    if value is not None:
        return value
    return label


def _score_search_r1_like(response: str, ground_truth: Any) -> float:
    """Score Search-R1-like QA answers using the same metrics.

    BrowseComp+ uses token-F1 by default in search-agent-rl's converter. Slime
    fused-agent evals extract the prediction from finish or ``\boxed{}``, with
    ``<answer>`` kept only as a compatibility fallback.
    """

    extraction = _extract_final_answer_with_source(response, allow_plain_text=False)
    if extraction is None:
        return 0.0
    answer = extraction.text

    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target")
        metric = str(ground_truth.get("metric") or "em").lower()
    else:
        targets = ground_truth
        metric = None

    refs = _as_refs(targets)
    if metric in {"token_f1", "f1", "bag_f1"}:
        best = max((_token_f1(answer, ref) for ref in refs), default=0.0)
        return _apply_anti_gibberish_penalty(best, answer)
    if metric in {"rouge_l", "rouge-l", "rouge_l_f1", "rouge"}:
        best = max((_rouge_l_f1(answer, ref) for ref in refs), default=0.0)
        return _apply_anti_gibberish_penalty(best, answer)
    return float(any(_short_answer_match(answer, ref) for ref in refs))


def _extract_answer_tag(response: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", str(response or ""), flags=re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _extract_final_answer(response: str) -> str | None:
    extraction = _extract_final_answer_with_source(response, allow_plain_text=True)
    return extraction.text if extraction is not None else None


def _extract_final_answer_with_source(response: str, *, allow_plain_text: bool) -> AnswerExtraction | None:
    text = _strip_think_prefix(response)
    finish_result = _extract_finish_result(text)
    if finish_result is not None:
        return AnswerExtraction(_clean_submitted_answer(_boxed_or_text(finish_result)), "finish")

    boxed, _, _ = _extract_boxed(text)
    if boxed:
        return AnswerExtraction(_clean_submitted_answer(boxed), "boxed")

    answer_tag = _extract_answer_tag(text)
    if answer_tag:
        return AnswerExtraction(_clean_submitted_answer(answer_tag), "answer_tag")

    if allow_plain_text:
        plain = _extract_plain_final_answer(text)
        if plain:
            return AnswerExtraction(_strip_latex_wrappers(plain), "plain_text")

    return None


def _extract_finish_result(text: str) -> str | None:
    calls = _FINISH_PARSER.get().parse(text)
    for call in reversed(calls):
        if call.name not in {"finish", "submit"}:
            continue
        if not _looks_like_real_finish_call(text, call.start, call.end):
            continue
        result = call.arguments.get("result")
        if result is not None:
            return str(result).strip()
    return None


def _looks_like_real_finish_call(text: str, start: int | None, end: int | None) -> bool:
    if start is None or end is None:
        return False
    snippet = text[start:end].lstrip()
    return snippet.startswith("<tool_call>") or snippet.startswith("{") or snippet.startswith("<function=")


def _boxed_or_text(text: str) -> str:
    boxed, _, _ = _extract_boxed(text)
    return boxed if boxed else str(text or "").strip()


def _clean_submitted_answer(text: str) -> str:
    text = _strip_latex_wrappers(str(text or "").strip())
    text = re.sub(r"\\+(?=\s*[A-Za-z0-9])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_latex_wrappers(text: str) -> str:
    out = str(text or "").strip()
    for _ in range(8):
        changed = False
        for wrapper in _LATEX_TEXT_WRAPPERS:
            if not out.startswith(wrapper):
                continue
            tail = out[len(wrapper) :].lstrip()
            if not tail.startswith("{"):
                continue
            end = _matching_brace_end(tail, 0)
            if end is None:
                continue
            trailing = tail[end + 1 :].strip()
            if trailing:
                continue
            out = tail[1:end].strip()
            changed = True
            break
        if not changed:
            break
    if out.startswith("$") and out.endswith("$") and len(out) >= 2:
        out = out[1:-1].strip()
    return out.replace("\\$", "$").replace("\\%", "%").replace("\\,", " ").replace("\\;", " ").replace("\\:", " ").strip()


def _matching_brace_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 1
    for pos in range(start + 1, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos
    return None


def _extract_plain_final_answer(text: str) -> str | None:
    text = _strip_tool_responses(str(text or "")).strip()
    if not text:
        return None
    tail = _tail_after_last_tool_call(text).strip()
    if not tail:
        return None
    if tail.startswith("<tool_call>"):
        return None
    return _cleanup_plain_answer(tail)


def _tail_after_last_tool_call(text: str) -> str:
    matches = list(re.finditer(r"<tool_call>.*?</tool_call>", text, flags=re.DOTALL))
    if not matches:
        return text
    return text[matches[-1].end() :]


def _strip_tool_responses(text: str) -> str:
    return re.sub(r"<tool_response>.*?</tool_response>", "\n", text, flags=re.DOTALL)


def _cleanup_plain_answer(text: str) -> str | None:
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^assistant\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    text = "\n".join(lines[-4:]).strip()
    for pattern in (
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*(.+)$",
        r"(?:therefore|thus|so),?\s+(?:the\s+)?(?:answer\s+)?(?:is|:)\s*(.+)$",
        r"(?:best\s+(?:described|represented)\s+by)\s+(.+)$",
    ):
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if matches:
            text = matches[-1].group(1).strip()
            break
    text = text.strip(" \t\r\n`*_")
    return text or None


def _parse_option_label(text: str) -> str | None:
    match = re.match(
        r"(?is)^\s*(?:final\s+)?(?:answer\s*(?:is)?\s*:?\s*)?(?:option|choice)?\s*([A-J])\s*(?:[.)\]:、-]|\b)",
        str(text or ""),
    )
    return match.group(1).upper() if match else None


def _infer_committed_mcq_letter(text: str, valid_letters: set[str]) -> str | None:
    best_pos = -1
    best_letter = None
    scan = re.sub(r"<think>|</think>", " ", str(text or ""))
    for pattern in _MCQ_COMMIT_PATTERNS:
        for match in pattern.finditer(scan):
            letter = match.group(1).upper()
            if letter not in valid_letters:
                continue
            prefix = scan[max(0, match.start() - 90) : match.start()]
            prefix = re.split(r"[.!?\n]", prefix)[-1]
            if _MCQ_TENTATIVE_PREFIX.search(prefix):
                continue
            if _MCQ_ENUM_FORWARD.match(scan[match.end() : match.end() + 12]):
                continue
            if _MCQ_NEGATIVE_FOLLOW.search(scan[match.end() : match.end() + 48]):
                continue
            if match.start() > best_pos:
                best_pos = match.start()
                best_letter = letter
    return best_letter


def _map_letter_to_option_text(prediction: str, options: list[str]) -> str:
    letter = _parse_option_label(prediction)
    if not letter:
        return prediction
    idx = string.ascii_uppercase.find(letter)
    if idx < 0 or idx >= len(options):
        return prediction
    return _strip_option_prefix(options[idx])


def _as_refs(targets: Any) -> list[str]:
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, list):
        return [str(item) for item in targets]
    return [str(targets)]


def _normalize_for_long_answer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _simple_tokenize(text: str, max_tokens: int = 4096) -> list[str]:
    text = _normalize_for_long_answer(_strip_latex_wrappers(text))
    if not text:
        return []
    if " " in text:
        tokens = [token for token in text.split(" ") if token]
    else:
        tokens = [char for char in text if not char.isspace()]
    return tokens[:max_tokens]


def _token_f1(pred: str, ref: str, max_tokens: int = 4096) -> float:
    pred_tokens = _simple_tokenize(pred, max_tokens=max_tokens)
    ref_tokens = _simple_tokenize(ref, max_tokens=max_tokens)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    tp = sum(common.values())
    if tp == 0:
        return 0.0
    precision = tp / max(len(pred_tokens), 1)
    recall = tp / max(len(ref_tokens), 1)
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(pred: str, ref: str, max_tokens: int = 2048) -> float:
    pred_tokens = _simple_tokenize(pred, max_tokens=max_tokens)
    ref_tokens = _simple_tokenize(ref, max_tokens=max_tokens)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / max(len(pred_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    return 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)


def _ngram_repetition_ratio(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    return 0.0 if not ngrams else 1.0 - (len(set(ngrams)) / len(ngrams))


def _apply_anti_gibberish_penalty(score: float, pred: str) -> float:
    tokens = _simple_tokenize(pred, max_tokens=8192)
    if not tokens:
        return 0.0
    rep4 = _ngram_repetition_ratio(tokens, n=4)
    score = score * (1.0 - min(0.7, max(0.0, rep4 - 0.2)))
    if len(tokens) > 3000:
        score *= 0.85
    if len(tokens) > 6000:
        score *= 0.7
    return float(max(0.0, min(1.0, score)))


def _score_short_answer(response: str, answers: list[str], *, strict_exact_match: bool = False) -> float:
    if not response or not answers:
        return 0.0
    final_text = _final_answer_text(response)
    if not final_text:
        return 0.0
    for answer in answers:
        if strict_exact_match:
            if _normalize_answer_for_benchmark(final_text) == _normalize_answer_for_benchmark(answer):
                return 1.0
            continue
        if _short_answer_match(final_text, answer):
            return 1.0
    return 0.0


def _short_answer_match(prediction: str, answer: str) -> bool:
    normalized_prediction = _normalize_answer_for_benchmark(prediction)
    normalized_answer = _normalize_answer_for_benchmark(answer)
    if not normalized_prediction or not normalized_answer:
        return False
    if normalized_prediction == normalized_answer:
        return True
    pred_date_variants = _date_variants(prediction)
    answer_date_variants = _date_variants(answer)
    if pred_date_variants and answer_date_variants and pred_date_variants & answer_date_variants:
        return True
    return _contains_answer_phrase(normalized_prediction, normalized_answer)


def _normalize_answer_for_benchmark(text: str) -> str:
    text = _strip_latex_wrappers(str(text or ""))
    numeric = _normalize_number_token(text)
    if numeric is not None:
        return numeric
    return normalize_answer(text)


def _normalize_number_token(text: str) -> str | None:
    value = str(text or "").strip()
    if not value:
        return None
    match = _UNIT_SUFFIX_PATTERN.match(value)
    if match:
        return _canonical_number(match.group(1))
    if re.match(r"^-?\d[\d,\.\s]*$", value):
        return _canonical_number(value)
    return None


def _canonical_number(value: str) -> str | None:
    number = value.replace(",", "").replace(" ", "")
    try:
        parsed = float(number)
    except ValueError:
        return None
    return str(int(parsed)) if parsed.is_integer() else number


def _date_variants(text: str) -> set[str]:
    value = str(text or "").strip()
    iso = _DATE_ISO_PATTERN.match(value)
    if iso:
        year, month, day = iso.group(1), int(iso.group(2)), int(iso.group(3))
        return {f"{year}-{month:02d}-{day:02d}"}
    dmy = _DATE_DMY_PATTERN.match(value)
    if not dmy:
        return set()
    first, second, year = int(dmy.group(1)), int(dmy.group(2)), dmy.group(3)
    if len(year) == 2:
        year = ("20" + year) if int(year) < 50 else ("19" + year)
    variants = set()
    if 1 <= first <= 12 and 1 <= second <= 31:
        variants.add(f"{year}-{first:02d}-{second:02d}")
    if 1 <= second <= 12 and 1 <= first <= 31:
        variants.add(f"{year}-{second:02d}-{first:02d}")
    return variants


def _contains_answer_phrase(response: str, answer: str) -> bool:
    return bool(re.search(rf"(^|\s){re.escape(answer)}($|\s)", response))


def _final_answer_text(response: str) -> str:
    text = _strip_think_prefix(response)
    return _extract_final_answer(text) or ""


def _strip_think_prefix(response: str) -> str:
    text = str(response or "")
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    return text


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^\s*[A-Z]\s*[\.\)]\s*", "", option.strip(), flags=re.IGNORECASE)
