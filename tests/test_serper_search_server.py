import asyncio
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError


NUM_GPUS = 0


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


def test_retrieval_request_defaults_to_ten_results_and_caps_larger_requests():
    assert serper.RetrievalRequest(query="apple inc").result_limit() == 10
    assert serper.RetrievalRequest(query="apple inc", max_results=20).result_limit() == 10


def test_retrieval_request_supports_batches_and_rejects_empty_queries():
    request = serper.RetrievalRequest(queries=[" first ", "", "second"], top_k=2)

    assert request.query_list() == ["first", "second"]
    assert request.result_limit() == 2
    with pytest.raises(ValidationError):
        serper.RetrievalRequest(query="  ")


def test_retrieval_row_preserves_search_metadata_and_truncates_snippet():
    row = serper._retrieval_row(
        {"title": "Example", "link": "https://example.com", "snippet": "search snippet"},
        max_words=1,
    )

    assert row == {
        "title": "Example",
        "url": "https://example.com",
        "link": "https://example.com",
        "search_snippet": True,
        "document": {"contents": "search ..."},
    }


def test_search_uses_only_search_response_snippets(monkeypatch):
    client = serper.SerperClient("test-key")
    calls = []

    async def fake_post(url, payload):
        calls.append((url, payload))
        assert payload == [{"q": "apple", "gl": "us", "hl": "en", "num": 1, "page": 3}]
        return [{
            "organic": [
                {"title": "Apple", "link": "https://example.com/apple", "snippet": "fallback search result snippet"}
            ]
        }]

    monkeypatch.setattr(client, "_post", fake_post)

    rows = asyncio.run(client.search("apple", limit=1, page=3, max_words=20))

    assert calls == [
        (serper.DEFAULT_SEARCH_URL, [{"q": "apple", "gl": "us", "hl": "en", "num": 1, "page": 3}])
    ]
    assert rows[0] == {
        "title": "Apple",
        "url": "https://example.com/apple",
        "link": "https://example.com/apple",
        "search_snippet": True,
        "document": {"contents": "fallback search result snippet"},
    }


def test_retrieve_returns_flat_legacy_document_list_after_search():
    calls = []

    class FakeClient:
        async def search_many(self, queries, *, limit, page, max_words):
            calls.append((queries, limit, page, max_words))
            return [[{"document": {"contents": "Scraped page content"}}]]

    serper.app.state.serper_client = FakeClient()
    response = asyncio.run(serper.retrieve(serper.RetrievalRequest(query="apple", max_results=2, page=3)))

    assert calls == [(["apple"], 2, 3, serper.DEFAULT_MAX_WORDS)]
    assert response == {"result": [{"document": {"contents": "Scraped page content"}}]}


def test_html_to_text_removes_script_and_style_content():
    text = serper._html_to_text(
        "<html><style>hidden css</style><body><h1>Title</h1><script>hidden js</script><p>Page fact</p></body></html>"
    )

    assert text == "Title Page fact"


def test_access_endpoint_returns_page_contents_and_per_url_errors():
    class FakeClient:
        async def access(self, url):
            if url.endswith("bad"):
                raise ValueError("blocked")
            return "Full page content"

    serper.app.state.serper_client = FakeClient()
    response = asyncio.run(
        serper.access(serper.AccessRequest(urls=["https://example.com/good", "https://example.com/bad"]))
    )

    assert response["result"][0] == {"url": "https://example.com/good", "contents": "Full page content"}
    assert response["result"][1]["contents"] == ""
    assert "ValueError: blocked" in response["result"][1]["error"]

def test_evals_exposes_serper_as_retrieval_backend():
    script = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text(encoding="utf-8")

    assert "--retrieval-backend local|serper" in script
    assert 'python3 "${REPO_ROOT}/examples/search-r1/serper_search_server.py"' in script
    assert 'export RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://10.2.152.50:65432}"' in script
    assert 'serper:*) export RETRIEVAL_MAX_RESULTS="${RETRIEVAL_MAX_RESULTS:-10}"' in script


def test_evals_exposes_rag_harness_with_cot_runtime_defaults():
    script = (Path(__file__).resolve().parents[1] / "experiments" / "evals.sh").read_text()

    assert "Harness: bare, cot, rag, react" in script
    assert 'cot|rag|bare) UNIFIED_SYSTEM_PROMPT=False' in script
    assert '[ "${FUSED_HARNESS}" = "cot" ] || [ "${FUSED_HARNESS}" = "rag" ]' in script
    assert "--rag-context-max-words N" in script
    assert 'RAG_CONTEXT_MAX_WORDS="${RAG_CONTEXT_MAX_WORDS:-1024}"' in script
    assert "export RAG_CONTEXT_MAX_WORDS" in script
    assert "--summary-backend local|openrouter" in script
    assert 'SUMMARY_BACKEND="${SUMMARY_BACKEND:-openrouter}"' in script
    assert 'SUMMARY_MODEL="${SUMMARY_MODEL:-qwen/qwen3-30b-a3b-instruct-2507}"' in script


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
