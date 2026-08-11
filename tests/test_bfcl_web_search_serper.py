import importlib.util
from pathlib import Path


NUM_GPUS = 0


def _load_web_search_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/artifacts/benchmarks/gorilla/berkeley-function-call-leaderboard"
        / "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/web_search.py"
    )
    spec = importlib.util.spec_from_file_location("bfcl_web_search_serper_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bfcl_web_search_uses_slime_retrieval_service(monkeypatch):
    module = _load_web_search_module()
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "result": [
                    {
                        "title": "Apple",
                        "url": "https://example.com/apple",
                        "document": {"contents": "Search result snippet"},
                    }
                ]
            }

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return Response()

    monkeypatch.setenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:65433/")
    monkeypatch.setattr(module.requests, "post", fake_post)

    result = module.WebSearchAPI().search_engine_query("apple", max_results=3)

    assert calls == [
        ("http://127.0.0.1:65433/retrieve", {"query": "apple", "max_results": 3}, 65)
    ]
    assert result == [
        {
            "title": "Apple",
            "href": "https://example.com/apple",
            "body": "Search result snippet",
        }
    ]


def test_bfcl_web_search_requires_retrieval_url(monkeypatch):
    module = _load_web_search_module()
    monkeypatch.delenv("RETRIEVAL_SERVER_URL", raising=False)

    assert module.WebSearchAPI().search_engine_query("apple") == {
        "error": "RETRIEVAL_SERVER_URL is not configured."
    }
