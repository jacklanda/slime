"""Model-output parsing helpers for agent harnesses."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of one decoded model output."""

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]
    ill_formed: bool = False


def parse_model_output(
    raw_output: str,
    *,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
) -> ParsedModelOutput:
    """Parse raw model text into reasoning, visible text, and tool uses.

    The heavy format-specific work is delegated to SGLang's reasoning and
    function-call parsers. The XML fallback covers Anthropic-style tool-call
    text that some coding-agent models still emit occasionally.
    """
    reasoning, body_text = "", raw_output
    if _is_gemma4_tool_parser(tool_parser_name or "") or _is_gemma4_tool_parser(reasoning_parser_name or ""):
        reasoning, body_text = _split_gemma4_thought_channel(body_text)

    if reasoning_parser_name and not _is_gemma4_tool_parser(reasoning_parser_name):
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        r, b = ReasoningParser(
            model_type=reasoning_parser_name,
            stream_reasoning=False,
        ).parse_non_stream(raw_output)
        reasoning, body_text = r or "", b or ""
        if not reasoning and "</think>" in body_text:
            reasoning, body_text = body_text.split("</think>", 1)

    body_text, tool_uses, ill_formed = parse_tool_uses(body_text, tools_schema, tool_parser_name)
    return ParsedModelOutput(
        reasoning=reasoning,
        text=(body_text or "").strip(),
        tool_uses=tool_uses,
        ill_formed=ill_formed,
    )


def parse_tool_uses(
    body_text: str,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Parse tool calls from body text and return visible text plus tool uses."""
    tool_uses: list[dict[str, Any]] = []
    ill_formed = False
    if tool_parser_name and tools_schema and _is_gemma4_tool_parser(tool_parser_name):
        body_text, tool_uses = parse_gemma4_tool_uses(body_text, tools_schema, tool_parser_name)
        if tool_uses:
            return body_text, tool_uses, _has_gemma4_native_tool_call_marker(body_text)
        if _has_gemma4_native_tool_call_marker(body_text):
            return body_text, tool_uses, True
        return body_text, tool_uses, ill_formed

    if tool_parser_name and tools_schema:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        parser = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name)
        calls = []
        if parser.has_tool_call(body_text):
            try:
                body_text, calls = parser.parse_non_stream(body_text)
            except Exception:
                logger.exception("[agent.parsing] sglang tool-call parsing failed; falling back")
        for c in calls:
            try:
                args = json.loads(c.parameters or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": c.parameters}
                ill_formed = True
            tool_uses.append({"name": c.name or "tool", "input": args})

    if not tool_uses and tools_schema:
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses, ill_formed


def _is_gemma4_tool_parser(tool_parser_name: str) -> bool:
    normalized = str(tool_parser_name or "").strip().lower().replace("-", "_")
    return "gemma4" in normalized or "gemma_4" in normalized


def _has_gemma4_native_tool_call_marker(text: str) -> bool:
    text = str(text or "")
    return any(
        marker in text
        for marker in (
            "<|tool_call>",
            "<tool_call|>",
            "<|tool_response>",
            "<tool_response|>",
        )
    )


def _split_gemma4_thought_channel(text: str) -> tuple[str, str]:
    text = str(text or "")
    stripped = text.lstrip()
    prefix_len = len(text) - len(stripped)
    marker = "<|channel>thought"
    if not stripped.startswith(marker):
        return "", text

    content_start = prefix_len + len(marker)
    if content_start < len(text) and text[content_start] == "\n":
        content_start += 1
    end = text.find("<channel|>", content_start)
    if end < 0:
        return "", text

    body_start = end + len("<channel|>")
    while body_start < len(text) and text[body_start] in {"\n", "\r"}:
        body_start += 1
    return text[content_start:end].strip("\r\n"), text[:prefix_len] + text[body_start:]


def parse_gemma4_tool_uses(
    body_text: str,
    tools_schema: list[dict],
    tool_parser_name: str = "gemma4",
) -> tuple[str, list[dict[str, Any]]]:
    """Parse Gemma4 native tool calls without requiring SGLang parser support."""
    from slime.rollout.fused_agent.parser import make_tool_parser

    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    parser = make_tool_parser(tool_parser_name, valid_tools={str(name) for name in valid_tools if name})
    calls = parser.parse(body_text)
    if not calls:
        return body_text, []

    tool_uses = [{"name": call.name, "input": call.arguments or {}} for call in calls]
    cleaned = body_text
    spans = sorted(
        (
            (start, _gemma4_tool_call_cleanup_end(body_text, end))
            for start, end in ((call.start, call.end) for call in calls)
            if start is not None and end is not None
        ),
        reverse=True,
    )
    for start, end in spans:
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned, tool_uses


def _gemma4_tool_call_cleanup_end(text: str, end: int) -> int:
    marker = "<|tool_response>"
    pos = end
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if not text.startswith(marker, pos):
            return pos
        pos += len(marker)


def parse_xml_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Anthropic-style XML tool calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last = 0
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        body_text,
        flags=re.DOTALL,
    ):
        name, inner = m.group(1), m.group(2)
        if name in valid_tools:
            args = {
                p.group(1): p.group(2).strip()
                for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
            }
            tool_uses.append({"name": name, "input": args})
            cleaned_parts.append(body_text[last : m.start()])
            last = m.end()
    cleaned_parts.append(body_text[last:])
    return "".join(cleaned_parts), tool_uses
