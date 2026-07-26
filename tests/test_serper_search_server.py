import asyncio
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError


def _load_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "search-r1" / "serper_search_server.py"
    spec = importlib.util.spec_from_file_location("serper_search_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


serper = _load_module()


def test_retrieval_request_supports_existing_aliases():
    request = serper.RetrievalRequest(query=" apple inc ", topk=3, max_words=20)

    assert request.query_list() == ["apple inc"]
    assert request.result_limit() == 3

    serper_request = serper.RetrievalRequest(q="apple inc", num=4)
    assert serper_request.query_list() == ["apple inc"]
    assert serper_request.result_limit() == 4


def test_retrieval_request_supports_batches_and_rejects_empty_queries():
    request = serper.RetrievalRequest(queries=[" first ", "", "second"], top_k=2)

    assert request.query_list() == ["first", "second"]
    assert request.result_limit() == 2
    with pytest.raises(ValidationError):
        serper.RetrievalRequest(query="  ")


def test_retrieval_row_prefers_scraped_markdown_and_truncates():
    row = serper._retrieval_row(
        {"title": "Example", "link": "https://example.com", "snippet": "search snippet"},
        {"markdown": "one two three four"},
        max_words=3,
    )

    assert row == {"document": {"contents": "one two three ..."}}


def test_search_falls_back_to_snippet_when_scrape_fails(monkeypatch):
    client = serper.SerperClient("test-key")

    async def fake_post(url, payload):
        assert payload == {"q": "apple", "num": 1, "page": 3}
        return {
            "organic": [
                {"title": "Apple", "link": "https://example.com/apple", "snippet": "fallback search result snippet"}
            ]
        }

    async def failed_scrape(_url):
        raise RuntimeError("scrape unavailable")

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(client, "_scrape_result", failed_scrape)

    rows = asyncio.run(client.search("apple", limit=1, page=3, max_words=20))

    assert rows[0] == {"document": {"contents": "fallback search result snippet"}}


def test_retrieve_returns_flat_legacy_document_list_after_search_and_scrape():
    calls = []

    class FakeClient:
        async def search(self, query, *, limit, page, max_words):
            calls.append((query, limit, page, max_words))
            return [{"document": {"contents": "Scraped page content"}}]

    serper.app.state.serper_client = FakeClient()
    response = asyncio.run(serper.retrieve(serper.RetrievalRequest(query="apple", max_results=2, page=3)))

    assert calls == [("apple", 2, 3, serper.DEFAULT_MAX_WORDS)]
    assert response == {"result": [{"document": {"contents": "Scraped page content"}}]}


def test_access_result_matches_search_gym_contract():
    assert serper._access_result("https://example.com", {"markdown": "# Page"}) == {
        "url": "https://example.com",
        "contents": "# Page",
    }
    assert not serper.AccessRequest(url="https://example.com", includeMarkdown=False).include_markdown


def test_evals_exposes_serper_as_retrieval_backend():
    script = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text(encoding="utf-8")

    assert "--retrieval-backend local|serper" in script
    assert 'python3 "${REPO_ROOT}/examples/search-r1/serper_search_server.py"' in script
    assert 'export RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}"' in script
