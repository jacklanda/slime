"""AgentCPM-Explore evaluation prompt and Serper-backed tool schemas."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .parser import tool_schema


AGENTCPM_EXPLORE_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective resKEEponse. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query. You are provided with functions:

{tools_description}

IMPORTANT: ALWAYS adhere to this exact format for tool use:
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

Current date: {current_date}
"""

AGENTCPM_EXPLORE_USER_PROMPT = "Your task is to answer the user's question: {question}"


def search_schema() -> dict[str, Any]:
    return tool_schema(
        "search",
        "Google search supports parallel processing of multiple (at most 3) queries. "
        "The tool retrieves the top 10 results for each query in parallel.",
        {
            "query": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
        },
        ["query"],
    )


def fetch_url_schema() -> dict[str, Any]:
    return tool_schema(
        "fetch_url",
        "Fetch webpage(s) and online pdf(s) and return their contents. "
        "Supports parallel processing of multiple (at most 3) URLs.",
        {
            "url": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            },
            "purpose": {"type": "string"},
        },
        ["url", "purpose"],
    )


def build_messages(
    question: str,
    tools: list[dict[str, Any]],
    *,
    current_date: str | None = None,
) -> list[dict[str, str]]:
    tools_description = (
        "<tools>\n"
        + "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
        + "\n</tools>"
    )
    system = AGENTCPM_EXPLORE_SYSTEM_PROMPT.format(
        tools_description=tools_description,
        current_date=current_date or datetime.now().strftime("%Y-%m-%d"),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": AGENTCPM_EXPLORE_USER_PROMPT.format(question=question)},
    ]
