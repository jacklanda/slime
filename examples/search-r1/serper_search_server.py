"""Serper-backed HTTP service compatible with slime retrieval clients."""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Upstream Serper proxy; the local /retrieve adapter listens on its own host/port.
DEFAULT_SEARCH_URL = "http://10.2.152.50:9999/search"
DEFAULT_MAX_WORDS = 4096
MAX_RESULTS = 10


class RetrievalRequest(BaseModel):
    """Accept both the slime request and legacy Search-R1 aliases."""

    model_config = ConfigDict(extra="ignore")

    query: str | None = None
    q: str | None = None
    queries: list[str] | None = None
    top_k: int | None = None
    topk: int | None = None
    max_results: int | None = None
    num: int | None = None
    max_words: int = Field(default=DEFAULT_MAX_WORDS, ge=1)
    page: int = Field(default=1, ge=1)
    return_scores: bool = False

    @model_validator(mode="after")
    def validate_queries(self) -> "RetrievalRequest":
        if not any(value and value.strip() for value in (self.query, self.q)) and not any(query.strip() for query in self.queries or []):
            raise ValueError("query or queries must contain at least one non-empty search query")
        return self

    def query_list(self) -> list[str]:
        if self.query and self.query.strip():
            return [self.query.strip()]
        if self.q and self.q.strip():
            return [self.q.strip()]
        return [query.strip() for query in self.queries or [] if query.strip()]

    def result_limit(self) -> int:
        requested = self.top_k or self.topk or self.max_results or self.num or MAX_RESULTS
        return max(1, min(requested, MAX_RESULTS))


class SerperClient:
    def __init__(self, proxy_token: str, *, timeout: float = 60):
        if not proxy_token:
            raise RuntimeError("SERPER_PROXY_TOKEN is required")
        self._headers = {"X-API-KEY": proxy_token, "Content-Type": "application/json"}
        self._search_url = os.environ.get("SERPER_SEARCH_URL", DEFAULT_SEARCH_URL)
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=self._timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post(self, url: str, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._session is None:
            raise RuntimeError("Serper client has not been started")
        async with self._session.post(url, json=payload) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Serper returned HTTP {response.status}: {body[:500]}")
            try:
                result = await response.json(content_type=None)
            except ValueError as exc:
                raise RuntimeError("Serper returned a non-JSON response") from exc
        if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
            raise RuntimeError("Serper returned an unexpected response shape")
        return result

    async def search(self, query: str, *, limit: int, page: int, max_words: int) -> list[dict[str, Any]]:
        return (await self.search_many([query], limit=limit, page=page, max_words=max_words))[0]

    async def search_many(
        self, queries: list[str], *, limit: int, page: int, max_words: int
    ) -> list[list[dict[str, Any]]]:
        payload = [{"q": query, "gl": "us", "hl": "en", "page": page, "num": limit} for query in queries]
        results = await self._post(self._search_url, payload)
        if len(results) != len(queries):
            raise RuntimeError(f"Serper returned {len(results)} result batches for {len(queries)} queries")
        return [
            [_retrieval_row(item, max_words=max_words) for item in result.get("organic", []) if isinstance(item, dict)][
                :limit
            ]
            for result in results
        ]


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _retrieval_row(item: dict[str, Any], *, max_words: int) -> dict[str, Any]:
    title = str(item.get("title") or "Untitled").strip()
    link = str(item.get("link") or "").strip()
    snippet = str(item.get("snippet") or "").strip()
    return {
        "title": title,
        "url": link,
        "link": link,
        "search_snippet": True,
        "document": {"contents": _truncate_words(snippet or "No search snippet available.", max_words)},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = SerperClient(
        os.environ.get("SERPER_PROXY_TOKEN", ""),
        timeout=float(os.environ.get("SERPER_TIMEOUT", "60")),
    )
    await client.start()
    app.state.serper_client = client
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="slime Serper retrieval service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve")
async def retrieve(request: RetrievalRequest) -> dict[str, Any]:
    client: SerperClient = app.state.serper_client
    try:
        batches = await client.search_many(
            request.query_list(), limit=request.result_limit(), page=request.page, max_words=request.max_words
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"result": batches[0] if request.query is not None or request.q is not None else batches}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Serper-backed slime retrieval service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65433)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
