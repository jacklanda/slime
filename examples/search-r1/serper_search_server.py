"""Serper-backed HTTP service compatible with slime retrieval clients."""

from __future__ import annotations

import argparse
import asyncio
from html.parser import HTMLParser
import ipaddress
import os
import socket
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urljoin, urlsplit

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
    def validate_queries(self) -> RetrievalRequest:
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


class AccessRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=10)


class SerperClient:
    def __init__(self, proxy_token: str, *, timeout: float = 60):
        if not proxy_token:
            raise RuntimeError("SERPER_PROXY_TOKEN is required")
        self._headers = {"X-API-KEY": proxy_token, "Content-Type": "application/json"}
        self._search_url = os.environ.get("SERPER_SEARCH_URL", DEFAULT_SEARCH_URL)
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._access_session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=self._timeout)
        self._access_session = aiohttp.ClientSession(timeout=self._timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._access_session is not None:
            await self._access_session.close()
            self._access_session = None

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

    async def access(self, url: str, *, max_redirects: int = 5) -> str:
        if self._access_session is None:
            raise RuntimeError("Serper client has not been started")
        current_url = url
        for _ in range(max_redirects + 1):
            await _validate_public_url(current_url)
            async with self._access_session.get(
                current_url,
                allow_redirects=False,
                headers={"User-Agent": "slime-deepsearch-world/1.0"},
            ) as response:
                if 300 <= response.status < 400 and response.headers.get("Location"):
                    current_url = urljoin(current_url, response.headers["Location"])
                    continue
                response.raise_for_status()
                raw = await response.content.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise RuntimeError("page exceeds the 2 MB access limit")
                charset = response.charset or "utf-8"
                text = raw.decode(charset, errors="replace")
                if "html" in response.headers.get("Content-Type", "").lower() or "<html" in text[:1000].lower():
                    text = _html_to_text(text)
                return _truncate_words(text, DEFAULT_MAX_WORDS)
        raise RuntimeError(f"page exceeded {max_redirects} redirects")


class _PageTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


def _html_to_text(value: str) -> str:
    parser = _PageTextExtractor()
    parser.feed(value)
    return " ".join(parser.parts)


async def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("access URL must be an unauthenticated HTTP(S) URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    loop = asyncio.get_running_loop()
    addresses = await loop.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("access URL hostname did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("access URL resolved to a non-public address")


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


@app.post("/access")
async def access(request: AccessRequest) -> dict[str, Any]:
    client: SerperClient = app.state.serper_client
    results = []
    for url in request.urls:
        try:
            results.append({"url": url, "contents": await client.access(url)})
        except Exception as exc:
            results.append({"url": url, "error": f"{type(exc).__name__}: {exc}", "contents": ""})
    return {"result": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Serper-backed slime retrieval service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65433)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
