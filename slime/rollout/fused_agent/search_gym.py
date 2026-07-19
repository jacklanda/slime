"""Compatibility adapter for SearchGym's native XML agent protocol.

SearchGym models emit ``<thought>`` followed by exactly one of ``<search>``,
``<access>`` or ``<answer>``.  The adapter deliberately keeps that wire format
instead of asking the model to translate it into slime's JSON tool-call format.
"""
from __future__ import annotations

import re
from typing import Any

from .parser import ToolCall, tool_schema


SEARCH_GYM_SYSTEM_PROMPT = """You are an autonomous research agent that answers questions with a web browser.
Reason inside <thought>...</thought>. After each thought emit exactly one action:
<search>query</search> to search the web, <access>url</access> to read a result URL,
or <answer>final answer</answer> when you are done. Search and verify evidence from
different perspectives before answering. Use the language of the question. Never
emit more than one action in a response."""

SEARCH_GYM_USER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant answers it. The Assistant analyzes the given question and information in the mind, retains important relevant information, calls a search engine to find necessary information, accesses web pages with certain urls, and provides the user with the answer. The Assistant conducts search by <search> query </search>, access certain url by <access> url </access>, and the top search results and url page will be returned between <information> and </information>. Note that abstracts or snippets in search results may not fully represent the actual content of the document. If the Assistant suspects potential relevance despite abstract mismatch, it should access the URL to verify the content. The reasoning processes are enclosed within <think> </think>. Finally, the Assistant provides answer inside <answer> and </answer>. The language of your answer should align with the question.\n\nUser:\n\n{question}\n\nAssistant:\n<think>"""


def search_schema() -> dict[str, Any]:
    return tool_schema("search", "Search the web.", {"query": {"type": "string"}}, ["query"])


def access_schema() -> dict[str, Any]:
    return tool_schema("access", "Read a web page by URL.", {"url": {"type": "string"}}, ["url"])


def finish_schema() -> dict[str, Any]:
    return tool_schema("finish", "Submit the final answer.", {"result": {"type": "string"}}, ["result"])


class SearchGymToolParser:
    """Parser/formatter for SearchGym's ``search/access/answer`` tags."""

    def __init__(self, valid_tools: set[str] | None = None):
        self.valid_tools = valid_tools

    def get_tool_prompt(self, tools_schema: str) -> str:
        return (
            "\n# SearchGym actions\n"
            "Use exactly one XML action after your thought: "
            "<search>query</search>, <access>url</access>, or <answer>answer</answer>.\n"
            f"Available action schemas:\n{tools_schema}"
        )

    def parse(self, text: str) -> list[ToolCall]:
        text = text or ""
        matches: list[ToolCall] = []
        patterns = (
            ("search", r"<search\s*>(.*?)</search\s*>"),
            ("access", r"<access\s*>(.*?)</access\s*>"),
            ("finish", r"<answer\s*>(.*?)</answer\s*>"),
        )
        for name, pattern in patterns:
            found = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
            if not found:
                continue
            match = found[-1]
            value = match.group(1).strip()
            key = "result" if name == "finish" else ("query" if name == "search" else "url")
            return [ToolCall(name, {key: value}, match.start(), match.end())]
        return matches

    def format_action(self, action: ToolCall) -> str:
        if action.name == "search":
            return f"<search>{action.arguments.get('query', '')}</search>"
        if action.name == "access":
            return f"<access>{action.arguments.get('url', '')}</access>"
        return f"<answer>{action.arguments.get('result', action.arguments.get('answer', ''))}</answer>"

    def format_tool_observation(self, tool_name: str, output_text: str) -> str:
        # SearchGym appends information to the same user/assistant text stream.
        output_text = str(output_text)
        if "<information>" not in output_text:
            output_text = f"<information>\n{output_text}\n</information>"
        return f"{output_text}\n<think>"

    def assistant_tool_result_message(self, action: ToolCall, response: Any) -> dict[str, Any] | None:
        return None

    def assistant_tool_results_message(self, actions: list[ToolCall], responses: list[Any]) -> dict[str, Any] | None:
        return None
