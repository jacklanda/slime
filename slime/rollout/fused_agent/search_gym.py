"""SearchGym prompt and tool schema matching the released model protocol."""
from __future__ import annotations

from typing import Any

from .parser import tool_schema


SEARCH_GYM_SYSTEM_PROMPT = """You are an autonomous research agent that answers questions with a web browser.
Reason inside <think>...</think>. To search, emit exactly one JSON call to the search
tool inside <tool_call>...</tool_call>. Search as many times as needed and keep
queries unique. Search results are returned inside <tool_response>...</tool_response>.
When no more external information is needed, emit the concise final answer inside
<answer>...</answer>. Use the language of the question."""

SEARCH_GYM_USER_PROMPT = """Answer the following question. Search for external information when needed.

{question}"""


def search_schema() -> dict[str, Any]:
    return tool_schema("search", "Search the web.", {"query": {"type": "string"}}, ["query"])
