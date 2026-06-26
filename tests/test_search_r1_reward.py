import importlib.util
from pathlib import Path


def _load_qa_em_format():
    path = Path(__file__).resolve().parents[1] / "examples" / "search-r1" / "qa_em_format.py"
    spec = importlib.util.spec_from_file_location("search_r1_qa_em_format", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _solution(answer: str) -> str:
    return (
        "<|im_start|>assistant\n"
        "<think>searching</think>"
        "<search>query</search>"
        "<information>4500 South and 4700 South</information>"
        "<think>answering</think>"
        "<answer>draft</answer>"
        f"<answer>{answer}</answer>"
    )


def test_search_r1_reward_is_binary_exact_match():
    qa_em_format = _load_qa_em_format()
    ground_truth = {"target": ["4500 South and 4700 South"]}

    assert qa_em_format.compute_score_em(_solution("wrong south wrong wrong wrong"), ground_truth, format_score=0.2) == 0
    assert qa_em_format.compute_score_em(_solution("4500 South and 4700 South"), ground_truth, format_score=0.2) == 1.0
