from __future__ import annotations

import csv
import importlib.util
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are an agent that is an expert in answering questions related to the U.S treasury & economy.

You must always provide an answer that is the result of any computation or reasoning to determine the answer to the question.
Use maximum precision in your calculations unless otherwise specified by the question.

You also have access to web search in the case that you need to look something up.

REQUIRED FORMAT for completion:
When you have the final answer, keep any final reasoning you used before getting to the answer and then only return the value in the XML tags.

<REASONING>
[final reasoning - including steps & sources used]
</REASONING>
<FINAL_ANSWER>
[value]
</FINAL_ANSWER>

Never respond with follow-up questions.

FAILURE CONDITION: If you do not produce a <FINAL_ANSWER> tag, your response will be considered incomplete and you will fail the task."""

REWARD_URL = "https://raw.githubusercontent.com/databricks/officeqa/main/reward.py"
_PAGE_RE = re.compile(r"[?&]page=(\d+)")
_MONTHS = {
    month: index
    for index, month in enumerate(
        ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"),
        start=1,
    )
}
_URL_DATE_RE = re.compile(r"/(%s)-(\d{4})-" % "|".join(_MONTHS), re.IGNORECASE)


def _page_text(json_path: Path, page: int) -> str:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    elements = payload.get("document", {}).get("elements", [])
    chunks: list[str] = []
    for element in elements:
        boxes = element.get("bbox") or []
        if not any(int(box.get("page_id", -1)) == page for box in boxes):
            continue
        content = str(element.get("content") or "").strip()
        if content:
            chunks.append(content)
    if not chunks:
        raise ValueError(f"no parsed content for page {page} in {json_path}")
    return "\n".join(chunks)


def _oracle_pages(row: dict[str, str], root: Path) -> list[dict[str, Any]]:
    urls = [item.strip() for item in re.split(r"[;\n]+", row["source_docs"]) if item.strip()]
    files = [item.strip() for item in re.split(r"[;\n]+", row["source_files"]) if item.strip()]
    listed_files = set(files)
    pages = []
    for url in urls:
        match = _PAGE_RE.search(url)
        if match is None:
            raise ValueError(f"{row['uid']}: source URL has no page number: {url}")
        date_match = _URL_DATE_RE.search(url)
        if date_match is None:
            raise ValueError(f"{row['uid']}: cannot map source URL to a Treasury Bulletin: {url}")
        month, year = date_match.groups()
        filename = f"treasury_bulletin_{year}_{_MONTHS[month.lower()]:02d}.txt"
        if filename not in listed_files:
            raise ValueError(f"{row['uid']}: URL maps to {filename}, which is absent from source_files")
        page = int(match.group(1))
        json_path = root / "treasury_bulletins_parsed" / "jsons" / f"{Path(filename).stem}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"{row['uid']}: parsed document is missing: {json_path}")
        pages.append({"source_url": url, "source_file": filename, "page": page, "text": _page_text(json_path, page)})
    return pages


def ensure_official_reward(root: Path) -> Path:
    reward_path = root / "reward.py"
    if reward_path.is_file():
        return reward_path
    try:
        with urllib.request.urlopen(REWARD_URL, timeout=30) as response:
            source = response.read()
    except OSError as exc:
        raise RuntimeError(
            f"OfficeQA's official reward.py is missing at {reward_path} and could not be downloaded from {REWARD_URL}"
        ) from exc
    compile(source, str(reward_path), "exec")
    reward_path.write_bytes(source)
    return reward_path


def prepare_dataset(root: Path, output: Path, limit: int = 0) -> tuple[Path, Path, int]:
    csv_path = root / "officeqa_pro.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"OfficeQA Pro CSV is missing: {csv_path}")
    reward_path = ensure_official_reward(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with csv_path.open(encoding="utf-8", newline="") as source, output.open("w", encoding="utf-8") as target:
        for row in csv.DictReader(source):
            if limit > 0 and count >= limit:
                break
            pages = _oracle_pages(row, root)
            oracle = "\n\n".join(
                f"[Oracle PDF page: {page['source_file']}, page {page['page']}]\n{page['text']}" for page in pages
            )
            prompt = f"{oracle}\n\nBased on the PDF page(s) shown above, please answer the following question: {row['question']}"
            record = {
                "task_id": row["uid"],
                "input": prompt,
                "ground_truth_answer": row["answer"],
                "data_source": "officeqa",
                "ability": "search",
                "reward_model": {"ground_truth": row["answer"], "style": "rule"},
                "extra_info": {
                    "uid": row["uid"],
                    "question": row["question"],
                    "oracle_pages": pages,
                    "oracle_mode": "gem-text-oracle",
                    "system_prompt": SYSTEM_PROMPT,
                    "official_reward_path": str(reward_path.resolve()),
                    "rm_type": "officeqa",
                    "data_source": "officeqa",
                    "benchmark_eval": True,
                    "oracle_prompt": prompt,
                },
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"OfficeQA dataset is empty: {csv_path}")
    return output, reward_path, count


_REWARD_MODULES: dict[str, Any] = {}


def load_official_reward(path: str):
    if path not in _REWARD_MODULES:
        spec = importlib.util.spec_from_file_location("officeqa_official_reward", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load OfficeQA reward module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REWARD_MODULES[path] = module
    return _REWARD_MODULES[path]


def score_prediction(ground_truth: Any, predicted: str, reward_path: str) -> tuple[float, dict[str, float]]:
    scorer = load_official_reward(reward_path)
    scores = {
        label: float(scorer.score_answer(str(ground_truth), predicted, tolerance=tolerance))
        for label, tolerance in (("0.0%", 0.0), ("0.1%", 0.001), ("1.0%", 0.01), ("5.0%", 0.05))
    }
    return scores["0.0%"], scores
