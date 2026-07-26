"""Serper-backed HTTP service compatible with slime retrieval clients."""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


DEFAULT_SEARCH_URL = "https://google.serper.dev/search"
DEFAULT_SCRAPE_URL = "https://scrape.serper.dev"
DEFAULT_MAX_WORDS = 4096
MAX_RESULTS = 100


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
        requested = self.top_k or self.topk or self.max_results or self.num or 5
        return max(1, min(requested, MAX_RESULTS))


class AccessRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    urls: list[str] | None = None
    include_markdown: bool = Field(default=True, validation_alias=AliasChoices("include_markdown", "includeMarkdown"))

    @model_validator(mode="after")
    def validate_urls(self) -> "AccessRequest":
        if not (self.url and self.url.strip()) and not any(url.strip() for url in self.urls or []):
            raise ValueError("url or urls must contain at least one non-empty URL")
        return self

    def url_list(self) -> list[str]:
        if self.url and self.url.strip():
            return [self.url.strip()]
        return [url.strip() for url in self.urls or [] if url.strip()]


class SerperClient:
    def __init__(self, api_key: str, *, timeout: float = 60, scrape_concurrency: int = 8):
        if not api_key:
            raise RuntimeError("SERPER_API_KEY is required")
        self._headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        self._search_url = os.environ.get("SERPER_SEARCH_URL", DEFAULT_SEARCH_URL)
        self._scrape_url = os.environ.get("SERPER_SCRAPE_URL", DEFAULT_SCRAPE_URL)
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._scrape_semaphore = asyncio.Semaphore(scrape_concurrency)
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=self._timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        if not isinstance(result, dict):
            raise RuntimeError("Serper returned an unexpected response shape")
        return result

    async def scrape(self, url: str, *, include_markdown: bool = True) -> dict[str, Any]:
        async with self._scrape_semaphore:
            return await self._post(self._scrape_url, {"url": url, "includeMarkdown": include_markdown})

    async def search(self, query: str, *, limit: int, page: int, max_words: int) -> list[dict[str, Any]]:
        payload = await self._post(self._search_url, {"q": query, "num": limit, "page": page})
        organic = [item for item in payload.get("organic", []) if isinstance(item, dict)][:limit]
        scraped = await asyncio.gather(
            *(self._scrape_result(item) for item in organic),
            return_exceptions=True,
        )
        return [
            _retrieval_row(item, page_data if isinstance(page_data, dict) else {}, max_words=max_words)
            for item, page_data in zip(organic, scraped, strict=True)
        ]

    async def _scrape_result(self, item: dict[str, Any]) -> dict[str, Any]:
        url = str(item.get("link") or "").strip()
        if not url:
            return {}
        try:
            return await self.scrape(url)
        except Exception:
            return {}


def _page_text(payload: dict[str, Any]) -> str:
    for key in ("markdown", "text", "content", "contents"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _truncate_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def _retrieval_row(item: dict[str, Any], page: dict[str, Any], *, max_words: int) -> dict[str, Any]:
    snippet = str(item.get("snippet") or "").strip()
    text = _truncate_words(_page_text(page) or snippet or "No page content available.", max_words)
    return {"document": {"contents": text}}


def _access_result(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"url": url, "contents": _page_text(payload)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = SerperClient(
        os.environ.get("SERPER_API_KEY", ""),
        timeout=float(os.environ.get("SERPER_TIMEOUT", "60")),
        scrape_concurrency=max(1, int(os.environ.get("SERPER_SCRAPE_CONCURRENCY", "8"))),
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
        batches = await asyncio.gather(
            *(
                client.search(query, limit=request.result_limit(), page=request.page, max_words=request.max_words)
                for query in request.query_list()
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"result": batches[0] if request.query is not None or request.q is not None else batches}


@app.post("/access")
async def access(request: AccessRequest) -> dict[str, Any]:
    client: SerperClient = app.state.serper_client
    urls = request.url_list()
    payloads = await asyncio.gather(
        *(client.scrape(url, include_markdown=request.include_markdown) for url in urls),
        return_exceptions=True,
    )
    return {
        "result": [
            _access_result(url, payload if isinstance(payload, dict) else {})
            for url, payload in zip(urls, payloads, strict=True)
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Serper-backed slime retrieval service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65433)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
