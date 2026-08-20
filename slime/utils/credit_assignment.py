from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Any


_TOOL_PARSER_ERROR_REASONS = {
    "ABNORMAL_PARSE_ERROR",
    "ABNORMAL_PARSE_ERROR_LOOP",
    "INVALID_REACT_STRUCTURE",
    "INVALID_FINAL_STEP",
    "tool_parse_error",
    "tool_call_parse_exception",
}
_REPEATED_SEARCH_QUERY_REASONS = {
    "ABNORMAL_REPEATED_QUERY",
    "repeated_query",
}
_TOO_MANY_TOOL_CALL_REASONS = {
    "ABNORMAL_TOOL_BURST",
}
_SEARCH_BYPASS_REASONS = {
    "ABNORMAL_SEARCH_BYPASS",
}
_NGRAM_REPETITION_REASONS = {
    "ABNORMAL_NGRAM_REPETITION",
    "ngram_repetition",
}
_MIXED_TOOL_AND_ANSWER_REASONS = {
    "ABNORMAL_MIXED_TOOL_AND_ANSWER",
    "mixed_tool_and_answer",
}
_METADATA_CREDIT_EVENTS = {
    "tool_parser_error",
    "think_parser_error",
    "repeated_search_query",
    "too_many_tool_calls",
    "search_bypass",
    "direct_submit_without_tool",
    "mixed_tool_and_answer",
    "tail_guard_early_stop",
    "ngram_repetition",
    "max_turns_exceeded",
    "max_response_len_exceeded",
}


@dataclass(frozen=True)
class CreditAssignmentConfig:
    enable: bool = False
    tool_parser_error: bool = False
    repeated_search_query: bool = False
    too_many_tool_calls: bool = False
    search_bypass: bool = False
    mixed_tool_and_answer: bool = False
    ngram_repetition: bool = False

    @classmethod
    def from_args(cls, args: Namespace) -> "CreditAssignmentConfig":
        return cls(
            enable=getattr(args, "credit_assignment_enable", False),
            tool_parser_error=getattr(args, "credit_assignment_tool_parser_error", False),
            repeated_search_query=getattr(args, "credit_assignment_repeated_search_query", False),
            too_many_tool_calls=getattr(args, "credit_assignment_too_many_tool_calls", False),
            search_bypass=getattr(args, "credit_assignment_search_bypass", False),
            mixed_tool_and_answer=getattr(args, "credit_assignment_mixed_tool_and_answer", False),
            ngram_repetition=getattr(args, "credit_assignment_ngram_repetition", False),
        )


def _metadata_flag(metadata: dict[str, Any] | None, *keys: str) -> bool:
    if not metadata:
        return False
    return any(bool(metadata.get(key)) for key in keys)


def _metadata_int(metadata: dict[str, Any] | None, *keys: str) -> int | None:
    if not metadata:
        return None
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _reason_matches(metadata: dict[str, Any] | None, reasons: set[str]) -> bool:
    if not metadata:
        return False
    candidates = (
        metadata.get("termination_reason"),
        metadata.get("filter_reason"),
        metadata.get("abnormal_reason"),
        metadata.get("credit_assignment_reason"),
    )
    return any(str(candidate) in reasons for candidate in candidates if candidate is not None)


def enabled_credit_assignment_event(
    metadata: dict[str, Any] | None,
    config: CreditAssignmentConfig,
) -> str | None:
    if not config.enable:
        return None

    metadata_event = str((metadata or {}).get("credit_assignment_event") or "")
    if metadata_event in _METADATA_CREDIT_EVENTS:
        return metadata_event

    if config.tool_parser_error and (
        # Some terminal paths only persist the abnormal termination reason;
        # retain parser credit even when the explicit event was lost while
        # serializing the trajectory metadata.
        _reason_matches(metadata, _TOOL_PARSER_ERROR_REASONS)
        or (
        _metadata_flag(metadata, "tool_call_parse_error", "parse_error", "tool_parser_error", "parse_tool_args_error")
        or _metadata_int(metadata, "tool_parser_error_count", "total_parse_tool_args_error", "parse_tool_args_error")
        )
    ):
        return "tool_parser_error"

    if config.too_many_tool_calls and (
        _metadata_flag(metadata, "too_many_tool_call", "excessive_parallel_calls")
        or _metadata_int(metadata, "too_many_tool_call_count", "excessive_parallel_calls")
        or _reason_matches(metadata, _TOO_MANY_TOOL_CALL_REASONS)
    ):
        return "too_many_tool_calls"

    if config.repeated_search_query and (
        _metadata_flag(metadata, "duplicate_search_detected", "repeated_query", "duplicate_query")
        or _metadata_int(metadata, "searched_query_count", "duplicate_query_count")
        or _reason_matches(metadata, _REPEATED_SEARCH_QUERY_REASONS)
    ):
        return "repeated_search_query"

    if config.ngram_repetition and (
        _metadata_flag(metadata, "ngram_repetition_detected", "repetition_detected")
        or _metadata_int(metadata, "ngram_repetition_total")
        or _reason_matches(metadata, _NGRAM_REPETITION_REASONS)
    ):
        return "ngram_repetition"

    if config.mixed_tool_and_answer and (
        _metadata_flag(metadata, "mixed_tool_and_answer")
        or _reason_matches(metadata, _MIXED_TOOL_AND_ANSWER_REASONS)
    ):
        return "mixed_tool_and_answer"

    if config.search_bypass and (
        _metadata_flag(metadata, "search_bypass", "bypass_termination")
        or _metadata_int(metadata, "reward/bypass_termination")
        or _reason_matches(metadata, _SEARCH_BYPASS_REASONS)
    ):
        return "search_bypass"

    return None


def mask_only_action_span(mask: list[int], start: int, end: int) -> list[int]:
    if start < 0 or end <= start or end > len(mask):
        return [0] * len(mask)
    return [0] * start + list(mask[start:end]) + [0] * (len(mask) - end)


def excluded_from_reward_baseline(
    metadata: dict[str, Any] | None,
    config: CreditAssignmentConfig,
) -> bool:
    """True when a sample's reward is a credit-assignment penalty, not a score.

    These samples are terminated by a format/behaviour guard and assigned a
    synthetic reward of 0.0, indistinguishable from a trajectory that ran to
    completion and answered wrong. Leaving them in the group mean makes the
    baseline track the model's *formatting* failure rate rather than its task
    success rate, which biases every sibling's advantage in both directions at
    once: it lowers the baseline, so correct trajectories are over-credited, and
    it shrinks the gap to 0.0, so the penalty itself is weakened.

    Measured on odyssey-gemma4-e4b-think-dev38 (19.9% of samples penalized, all
    with reward exactly 0.0), excluding them from the baseline moves the mean
    winner advantage 0.407 -> 0.282 (-31%) and the mean penalty -0.323 -> -0.565
    (75% stronger).

    Attributable failures are still trained on: they keep a localized policy
    loss and are scored *against* the clean baseline, so the penalty survives.
    Omission/truncation failures may intentionally carry an all-zero policy mask
    because no emitted token can be blamed reliably. Both lose their vote in
    computing the baseline.
    """
    return enabled_credit_assignment_event(metadata, config) is not None


def build_policy_loss_mask(
    *,
    metadata: dict[str, Any] | None,
    loss_mask: list[int],
    config: CreditAssignmentConfig,
    parser_error_token_window: int = 256,
) -> tuple[list[int] | None, str | None]:
    event = enabled_credit_assignment_event(metadata, config)
    if event is None:
        return None, None
    if event == "search_bypass":
        return list(loss_mask), event
    if event in {"direct_submit_without_tool", "tail_guard_early_stop"}:
        return [0] * len(loss_mask), event

    start = _metadata_int(
        metadata,
        "credit_assignment_action_start",
        "action_start",
        "error_action_start",
        "abnormal_action_start",
    )
    end = _metadata_int(
        metadata,
        "credit_assignment_action_end",
        "action_end",
        "error_action_end",
        "abnormal_action_end",
    )
    if event == "tool_parser_error":
        if start is not None and end is not None and 0 <= start < end <= len(loss_mask):
            penalized_len = max(0, min(end - start, parser_error_token_window))
            if penalized_len == 0:
                return [0] * len(loss_mask), event
            return mask_only_action_span(loss_mask, end - penalized_len, end), event
        penalized_len = max(0, min(len(loss_mask), parser_error_token_window))
        if penalized_len == 0:
            return [0] * len(loss_mask), event
        return [0] * (len(loss_mask) - penalized_len) + list(loss_mask[-penalized_len:]), event
    if start is None or end is None:
        return list(loss_mask), event
    return mask_only_action_span(loss_mask, start, end), event
