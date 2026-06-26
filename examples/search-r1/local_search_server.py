"""Local retrieval client for Search-R1 rollouts."""

import os
from typing import Any

import aiohttp


_MIN_DOC_WORDS = 25
_WORD_BUDGET = 256


def _normalize_retrieve_url(search_url: str) -> str:
    search_url = search_url.rstrip("/")
    return search_url if search_url.endswith("/retrieve") else f"{search_url}/retrieve"


def _normalize_server_url(search_url: str) -> str:
    search_url = search_url.rstrip("/")
    return search_url[: -len("/retrieve")] if search_url.endswith("/retrieve") else search_url


def _truncate_words(text: str, max_words: int = _WORD_BUDGET) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _normalize_doc_signature(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    return "".join(ch for ch in normalized if ch.isalnum() or ch.isspace())[:500]


def _extract_document_text(item: dict[str, Any]) -> str:
    document = item.get("document")
    if isinstance(document, dict):
        for key in ("contents", "text", "passage"):
            value = document.get(key)
            if isinstance(value, str) and value.strip():
                return value
    elif isinstance(document, str) and document.strip():
        return document

    content = item.get("content")
    if isinstance(content, dict):
        candidates = [
            value
            for key in ("original_text", "chunk_text", "text")
            if isinstance((value := content.get(key)), str) and value.strip()
        ]
        if candidates:
            return max(candidates, key=lambda text: len(text.split()))
    elif isinstance(content, str) and content.strip():
        return content

    for key in ("contents", "chunk_text", "text", "passage"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_document_title(item: dict[str, Any], text: str) -> str:
    for key in ("title", "source", "url"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    content = item.get("content")
    if isinstance(content, dict):
        for key in ("title", "source", "url"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first_line and len(first_line.split()) <= 16:
        return first_line
    return "Untitled"


def _get_retrieval_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    results = response.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]

    batch_results = response.get("result")
    if isinstance(batch_results, list) and batch_results:
        first_batch = batch_results[0]
        if isinstance(first_batch, list):
            return [item for item in first_batch if isinstance(item, dict)]
    return []


def _format_search_results(results: list[dict[str, Any]], max_results: int) -> list[str]:
    if not results:
        return ["No relevant documents found."]

    documents = []
    seen_signatures = set()
    duplicate_count = 0
    skipped_short = 0
    for result in results:
        content = _extract_document_text(result)
        if not content:
            continue
        signature = _normalize_doc_signature(" ".join(str(value) for value in (result.get("title"), result.get("url"), content) if value))
        if signature and signature in seen_signatures:
            duplicate_count += 1
            continue
        if signature:
            seen_signatures.add(signature)
        if len(content.split()) < _MIN_DOC_WORDS:
            skipped_short += 1
            continue

        title = _extract_document_title(result, content)
        documents.append(f"[Result {len(documents) + 1}] Title: {title}\nSnippet: {content.strip()}")
        if len(documents) >= max_results:
            break

    if not documents:
        return [
            "No usable evidence was found for this query. The returned passages were duplicates, too short, or too generic. "
            "Do not submit an answer from this result. Rewrite the query with a specific title, quoted phrase, named entity, date, number, or one clue from the question, then search again."
        ]
    if skipped_short:
        documents.append(f"[{skipped_short} short fragments were filtered; narrow the query if you need more detail]")
    if duplicate_count:
        documents.append(f"[{duplicate_count} duplicate passages were removed before display]")
    return documents


async def _summarize_documents(
    session: aiohttp.ClientSession,
    search_url: str,
    documents: list[str],
    timeout_obj: aiohttp.ClientTimeout,
) -> str | None:
    payload = {
        "documents": [{"content": document} for document in documents],
        "max_length": _WORD_BUDGET,
    }
    try:
        async with session.post(f"{_normalize_server_url(search_url)}/summarize", json=payload, timeout=timeout_obj) as resp:
            if resp.status != 200:
                return None
            summary_data = await resp.json()
    except Exception:
        return None

    summary = str(summary_data.get("summary", "")).split("# Summary:", 1)[-1].strip()
    return summary or None


async def local_search(
    search_url: str,
    query: str,
    top_k: int = 5,
    timeout: int = 60,
    proxy: str | None = None,
    retrieval_mode: str | None = None,
    retrieval_max_words: int | None = None,
) -> str:
    """
    Call local retrieval server and format results like google_search_server.py.

    Args:
        search_url: Retrieval server base URL or /retrieve URL
        query: Search query string
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds (default: 60)
        proxy: Proxy URL if needed
        retrieval_mode: Retrieval mode requested from the server
        retrieval_max_words: Maximum words per returned passage requested from the server

    Returns:
        Formatted retrieval text for direct insertion into the agent observation.
    """
    if retrieval_mode is None:
        retrieval_mode = os.environ.get("RLLM_RETRIEVAL_MODE", "hybrid")
    if retrieval_max_words is None:
        retrieval_max_words = int(os.environ.get("RLLM_RETRIEVAL_MAX_WORDS", "4096"))

    # Align with rllm's LocalRetrievalTool payload. Older Search-R1 servers
    # ignore unknown fields and still accept top_k/topk aliases.
    payload = {
        "query": query,
        "top_k": top_k,
        "topk": top_k,
        "max_words": int(retrieval_max_words),
        "mode": str(retrieval_mode or "lexical").strip() or "lexical",
        "return_scores": False,
    }

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    session_kwargs = {}
    if proxy:
        session_kwargs["proxy"] = proxy

    try:
        async with aiohttp.ClientSession(**session_kwargs) as session:
            async with session.post(_normalize_retrieve_url(search_url), json=payload, timeout=timeout_obj) as resp:
                resp.raise_for_status()
                result = await resp.json()

            documents = _format_search_results(_get_retrieval_results(result), top_k)
            content = "\n\n".join(documents)
            if os.environ.get("RLLM_RETRIEVAL_SUMMARIZE", "0") == "1":
                summary = await _summarize_documents(session, search_url, documents, timeout_obj)
                if summary:
                    content = summary
    except Exception as e:
        print(f"Error calling local search engine at {search_url}: {e}")
        return "No relevant documents found."

    return _truncate_words(content)
