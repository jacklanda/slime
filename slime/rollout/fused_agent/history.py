from __future__ import annotations

import re
from typing import Any


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL)
_THOUGHT_CHANNEL_BLOCK_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>")
_THOUGHT_CHANNEL_OPEN_RE = re.compile(r"<\|channel>thought\n")
_THINK_CLOSE_RE = re.compile(r"</think\s*>")


def strip_historical_thinking(content: str) -> tuple[str, bool]:
    starts_with_thinking = (
        _THINK_BLOCK_RE.match(content) is not None
        or _THOUGHT_CHANNEL_BLOCK_RE.match(content) is not None
    )
    stripped = _THINK_BLOCK_RE.sub("", content)
    stripped = _THOUGHT_CHANNEL_BLOCK_RE.sub("", stripped)

    closers = list(_THINK_CLOSE_RE.finditer(stripped))
    if closers:
        stripped = stripped[closers[-1].end() :].lstrip("\n")
    elif starts_with_thinking:
        stripped = stripped.lstrip("\r\n")

    openers = [
        match
        for regex in (_THINK_OPEN_RE, _THOUGHT_CHANNEL_OPEN_RE)
        if (match := regex.search(stripped))
    ]
    if not openers:
        return stripped, False
    return stripped[: min(match.start() for match in openers)].rstrip(), True


def messages_without_historical_thinking(
    messages: list[dict[str, Any]],
    *,
    parser=None,
) -> list[dict[str, Any]]:
    prepared = []
    for message in messages:
        if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
            prepared.append(message)
            continue
        original_content = message["content"]
        content, had_unclosed_thinking = strip_historical_thinking(original_content)
        if had_unclosed_thinking and parser is not None and not message.get("tool_calls"):
            content = "".join(parser.format_action(action) for action in parser.parse(original_content))
        prepared.append(message if content == original_content else {**message, "content": content})
    return prepared


def strip_trailing_chat_template_stop(text: str) -> str:
    stripped = text
    while True:
        without_ws = stripped.rstrip()
        if not without_ws.endswith("<|im_end|>"):
            return stripped
        stripped = without_ws[: -len("<|im_end|>")]
