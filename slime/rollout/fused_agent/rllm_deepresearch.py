from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from .env import _get_shared_http_session, _post_json_with_retries, _retrieval_rows
from .parser import ToolCall, tool_schema


SEARCH_SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \\boxed{} format

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \\boxed{} format."""

REFINE_PROMPT = """**TASK:**
Synthesize the key information from the **[Retrieved Documents]** that is relevant to the **[Current Query]**.

**INSTRUCTIONS:**
1.  **Extract & Merge:** Identify all relevant facts and combine them. Eliminate redundancy. You should provide information for deep research, not answer to current query.
2.  **Provide Information, Not an Answer:** Your output should be a self-contained block of information, NOT a direct, short answer to the current query.
3.  **Handle Insufficient Information:** If the documents do not contain relevant information for the query, state that the provided sources are insufficient and suggest that further investigation may be needed. You can also provide some further investigation direction and query rewrite suggestions.
4.  **Format:** Enclose the entire synthesized output within `<information>` and `</information>` tags. Add no other text.

**CONTEXT:**
- **[Current Query]:** {query}
- **[Retrieved Documents]:** {documents}

**SYNTHESIZED INFORMATION:**
"""


def local_search_schema() -> dict:
    return tool_schema(
        "local_search",
        "Search for information using a dense retrieval server with Wikipedia corpus",
        {"query": {"type": "string", "description": "Search query to retrieve relevant documents"}},
        ["query"],
    )


def parse_refine_response(content: str) -> str:
    if "<think>" not in content or "</think>" not in content:
        raise ValueError("Refine response must contain a complete <think> block")
    answer = content.split("</think>", 1)[1]
    start = answer.find("<information>")
    end = answer.find("</information>", start + len("<information>"))
    if start < 0 or end < 0:
        raise ValueError("Refine response must contain a complete <information> block")
    return answer[start + len("<information>") : end].strip()


async def run_search(action: ToolCall, *, retrieval_url: str, max_results: int) -> tuple[str, dict[str, Any]]:
    query = str((action.arguments or {}).get("query") or "")
    if not query:
        return "Error: local_search requires query.", {"tool_return_error": 1, "refine_error": 0}

    urls = [url.strip().rstrip("/") for url in retrieval_url.split(",") if url.strip()]
    if not urls:
        return "Error: No retrieval servers available.", {"tool_return_error": 1, "refine_error": 0}
    random.shuffle(urls)
    retry_budget = min(max(1, int(os.environ.get("RLLM_DR_RETRIEVAL_MAX_RETRIES", "2"))), len(urls))
    last_error: Exception | None = None
    data: Any = None
    for url in urls[:retry_budget]:
        try:
            data, _ = await _post_json_with_retries(
                _get_shared_http_session(),
                url if url.endswith("/retrieve") else f"{url}/retrieve",
                {"query": query, "top_k": max_results},
                retry_budget=0,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        return f"Error: All retrieval servers failed. Last error: {last_error}", {"tool_return_error": 1, "refine_error": 0}

    raw_result = _format_original_results(data, max_results=max_results)
    if os.environ.get("RLLM_DR_USE_REFINE", "0").lower() in {"0", "false", "no", "off"}:
        return raw_result, {"tool_return_error": 0, "refine_error": 0}

    refined = await _refine(query, raw_result)
    if refined is None:
        return raw_result, {"tool_return_error": 0, "refine_error": 1}
    return refined, {"tool_return_error": 0, "refine_error": 0}


def _format_original_results(data: Any, *, max_results: int) -> str:
    rows = _retrieval_rows(data)
    if not rows:
        return "No relevant documents found for the query."
    max_chars = max(1, int(os.environ.get("RLLM_DR_MAX_CONTENT_LENGTH", "4000")))
    documents = []
    for idx, row in enumerate(rows[:max_results], start=1):
        content: Any = row
        if isinstance(row, dict):
            content = row.get("content", row.get("document", row))
        if isinstance(content, dict):
            # Match rllm_dr's LocalRetrievalTool._extract_raw_content exactly:
            # nested corpus rows without a ``contents``/``content`` key are
            # rendered with Python's dict representation.
            content = content.get("contents", content.get("content", str(content)))
        if not isinstance(content, str):
            content = str(content)
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        documents.append(f"[Document {idx}]\n{content}\n")
    return "\n".join(documents)


async def _refine(query: str, documents: str) -> str | None:
    configured = os.environ.get("RLLM_DR_REFINE_SERVER_URL") or os.environ.get("REFINE_SERVER_URL")
    if not configured:
        raise RuntimeError(
            "rllm_deepresearch requires RLLM_DR_REFINE_SERVER_URL (comma-separated OpenAI-compatible base URLs), "
            "or explicitly set RLLM_DR_USE_REFINE=0"
        )
    urls = [url.strip().rstrip("/") for url in configured.split(",") if url.strip()]
    attempts = max(1, int(os.environ.get("RLLM_DR_REFINE_MAX_RETRIES", "3")))
    payload = {
        "model": os.environ.get("RLLM_DR_REFINE_MODEL", "Qwen/Qwen3-8B"),
        "messages": [{"role": "user", "content": REFINE_PROMPT.format(query=query, documents=documents)}],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    for _ in range(attempts):
        url = random.choice(urls)
        try:
            data, _ = await _post_json_with_retries(
                _get_shared_http_session(),
                url if url.endswith("/chat/completions") else f"{url}/chat/completions",
                payload,
                retry_budget=0,
            )
            content = data["choices"][0]["message"]["content"]
            summary = parse_refine_response(content)
            return f"Your query is: {query}. The search results are summarized as following: {summary}"
        except Exception:
            await asyncio.sleep(0)
    return None
