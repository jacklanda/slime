#!/usr/bin/env python3
"""Evaluate benchmark datasets through the OpenRouter chat completions API."""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import yaml
from tqdm import tqdm

from slime.rollout.fused_agent.parser import ToolCall, make_tool_parser
from slime.rollout.fused_agent.env import _format_retrieval_documents
from slime.rollout.rm_hub import openrouter_grm
from slime.rollout.fused_agent.prompts import (
    COT_SYSTEM_PROMPT,
    build_web_search_messages,
    finish_schema,
    web_search_schema,
)
from slime.utils.types import Sample
from slime_plugins.evals.results_table import format_eval_results_table


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_trajectories(path: Path, results: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    records = [item[1] for item in sorted(results, key=lambda item: item[0])]
    payload = {
        "training_step": 0,
        "epoch": 0,
        "mode": "eval",
        "num_episodes": len(records),
        "trajectories": records,
    }
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4, ensure_ascii=False, default=_json_default)
        stream.write("\n")
    temporary.replace(path)


async def request_completion(client, semaphore, payload, max_retries):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"].get("content", "")
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                if attempt + 1 == max_retries:
                    raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {exc}") from exc
                await asyncio.sleep(min(2**attempt, 30) + random.random())


def _question_from_prompt(prompt) -> str:
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(prompt)


def gem_messages(prompt, model: str):
    schemas = [web_search_schema(), finish_schema()]
    parser = make_tool_parser(model, valid_tools={"web_search", "finish"})
    messages = build_web_search_messages(
        _question_from_prompt(prompt),
        schemas,
        model,
        tool_parser=parser,
        user_prompt=os.environ.get("FUSED_WEB_SEARCH_USER_PROMPT", "short"),
    )
    return messages, parser


async def run_gem(client, semaphore, args, prompt):
    messages, tool_parser = gem_messages(prompt, args.model)
    trace = []
    search_calls = 0
    for _ in range(args.max_steps):
        step_started_at = datetime.now(timezone.utc)
        llm_started_at = time.monotonic()
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        response = await request_completion(client, semaphore, payload, args.max_retries)
        llm_time = time.monotonic() - llm_started_at
        messages.append({"role": "assistant", "content": response})
        step_messages = copy.deepcopy(messages)
        actions = tool_parser.parse(response)
        trace_step = {
            "response": response,
            "actions": [{"name": action.name, "arguments": action.arguments} for action in actions],
        }
        trace.append(trace_step)
        if len(actions) != 1:
            observation = "Error: emit exactly one valid tool call using the format declared in the system prompt."
        elif actions[0].name == "finish":
            if search_calls:
                trace_step["observation"] = tool_parser.format_tool_observation("finish", "Submitted.")
                trace_step["done"] = True
                trace_step["chat_completions"] = step_messages
                trace_step["timing"] = {
                    "start_timestamp": step_started_at.isoformat(),
                    "end_timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_time": llm_time,
                    "env_time": 0.0,
                }
                return str(actions[0].arguments.get("result", "")), trace, "completed"
            observation = "Error: call web_search at least once before finish so the answer is grounded in retrieved evidence."
        else:
            search_calls += 1
            query = str(actions[0].arguments.get("query", "")).strip()
            if not query:
                observation = "Error: web_search requires a non-empty query."
            else:
                requested = actions[0].arguments.get("max_results", args.retrieval_max_results)
                try:
                    max_results = max(1, min(int(requested), args.retrieval_max_results))
                except (TypeError, ValueError):
                    max_results = args.retrieval_max_results
                try:
                    search_response = await client.post(
                        args.retrieval_url.rstrip("/") + "/retrieve",
                        json={"query": query, "top_k": max_results, "topk": max_results, "max_results": max_results},
                    )
                    search_response.raise_for_status()
                    documents, _ = _format_retrieval_documents(
                        search_response.json(),
                        max_results=max_results,
                    )
                    observation = "\n\n".join(documents) or "No relevant search results were returned."
                except (httpx.HTTPError, ValueError) as exc:
                    observation = f"Search failed: {type(exc).__name__}: {exc}"
        rendered_observation = tool_parser.format_tool_observation(actions[0].name if len(actions) == 1 else "parser", observation)
        trace_step.update(
            {
                "observation": rendered_observation,
                "done": False,
                "chat_completions": step_messages,
                "timing": {
                    "start_timestamp": step_started_at.isoformat(),
                    "end_timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_time": llm_time,
                    "env_time": max(0.0, time.monotonic() - llm_started_at - llm_time),
                },
            }
        )
        messages.append({"role": "user", "content": rendered_observation})
    return trace[-1]["response"] if trace else "", trace, "max_steps"


async def judge_trajectory(
    args,
    *,
    index: int,
    prompt,
    label,
    response: str,
    trace: list[dict],
    source: str,
    metadata: dict | None = None,
) -> Sample:
    termination = "env_done" if trace and trace[-1].get("done") else "max_turns_exceeded"
    final_model_response = trace[-1]["response"] if trace else response
    sample_metadata = dict(metadata or {})
    sample_metadata.update(
        {
            "benchmark_eval": True,
            "rm_type": "benchmark_verifier",
            "benchmark": source,
            "data_source": sample_metadata.get("data_source") or source,
            "fused_traj_steps": len(trace),
            "fused_termination": termination,
            "fused_reward_debug": {
                "tool_calls": sum(
                    action.get("name") == "web_search"
                    for step in trace
                    for action in step.get("actions", [])
                )
            },
        }
    )
    sample = Sample(
        index=index,
        prompt=prompt,
        response=final_model_response,
        label=label,
        status=Sample.Status.COMPLETED if termination == "env_done" else Sample.Status.TRUNCATED,
        metadata=sample_metadata,
    )
    sample.reward = float(await openrouter_grm.reward_func(args, sample, evaluation=True))
    return sample


async def main(args):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    os.environ["FUSED_WEB_SEARCH_USER_PROMPT"] = args.user_prompt
    config = yaml.safe_load(Path(args.eval_config).read_text(encoding="utf-8"))
    jobs = []
    for dataset in config.get("eval", {}).get("datasets", []):
        path = Path(dataset["path"])
        rows = load_rows(path)
        if args.limit:
            rows = rows[: args.limit]
        name = dataset.get("name", path.stem)
        input_key = dataset.get("input_key", "input")
        label_key = dataset.get("label_key", "ground_truth_answer")
        metadata_key = dataset.get("metadata_key", "extra_info")
        metadata_overrides = dataset.get("metadata_overrides") or {}
        for index, row in enumerate(rows):
            for sample_index in range(args.n_samples):
                prompt = row.get(input_key, row.get("input", ""))
                metadata = row.get(metadata_key) if isinstance(row.get(metadata_key), dict) else {}
                metadata = {
                    **metadata,
                    **{key: row[key] for key in ("data_source", "reward_model", "options") if key in row},
                    **metadata_overrides,
                }
                jobs.append((name, index, sample_index, prompt, row.get(label_key), row, metadata))

    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    judge_args = SimpleNamespace(**vars(args), rm_type="benchmark_verifier", hf_checkpoint=args.model)
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=args.timeout, limits=limits) as client:
        async def run(job):
            name, index, sample_index, prompt, label, row, metadata = job
            started_at = datetime.now(timezone.utc)
            started_monotonic = time.monotonic()
            if args.harness == "gem":
                response, trace, status = await run_gem(client, semaphore, args, prompt)
            else:
                messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
                if args.harness == "cot":
                    messages = [{"role": "system", "content": COT_SYSTEM_PROMPT}, *messages]
                payload = {"model": args.model, "messages": messages, "temperature": args.temperature,
                           "top_p": args.top_p, "max_tokens": args.max_tokens}
                response = await request_completion(client, semaphore, payload, args.max_retries)
                trace, status = [], "completed"
            if args.enable_use_grm_evals:
                judge_started_at = time.monotonic()
                sample = await judge_trajectory(
                    judge_args,
                    index=index,
                    prompt=prompt,
                    label=label,
                    response=response,
                    trace=trace,
                    source=name,
                    metadata=metadata,
                )
                reward_time = time.monotonic() - judge_started_at
            else:
                sample = Sample(
                    index=index,
                    prompt=prompt,
                    response=response,
                    label=label,
                    reward=0.0,
                    status=Sample.Status.COMPLETED,
                    metadata={"benchmark": name, "fused_traj_steps": len(trace), "fused_termination": status},
                )
                reward_time = 0.0
            search_calls = sum(
                    action.get("name") == "web_search"
                    for step in trace
                    for action in step.get("actions", [])
                )
            ended_at = datetime.now(timezone.utc)
            timing = {
                "start_timestamp": started_at.isoformat(),
                "end_timestamp": ended_at.isoformat(),
                "llm_time": sum(step.get("timing", {}).get("llm_time", 0.0) for step in trace),
                "env_time": sum(step.get("timing", {}).get("env_time", 0.0) for step in trace),
                "reward_time": reward_time,
                "total_time": time.monotonic() - started_monotonic,
            }
            episode_key = f"{name}:{index}:{sample_index}:{args.model}"
            episode_id = hashlib.md5(episode_key.encode()).hexdigest() + f":{sample_index}"
            steps = []
            output_parser = make_tool_parser(args.model)
            for step in trace:
                actions = step.get("actions", [])
                action = ""
                if actions:
                    action = output_parser.format_action(ToolCall(actions[0]["name"], actions[0].get("arguments") or {}))
                steps.append(
                    {
                        "observation": step.get("observation", ""),
                        "thought": step.get("response", ""),
                        "action": action,
                        "reward": 0.0,
                        "done": bool(step.get("done")),
                        "model_response": step.get("response", ""),
                        "chat_completions": step.get("chat_completions", []),
                        "disable_thinking": False,
                        "timing": step.get("timing", {}),
                        "tito_context_reason": "evaluation",
                        "historical_thinking_discarded": False,
                    }
                )
            task = dict(row)
            task.update({"index": index, "question": _question_from_prompt(prompt), "data_source": name, "ground_truth": label})
            return (name, index, sample_index), {
                "training_step": 0,
                "epoch": 0,
                "mode": "eval",
                "episode_id": episode_id,
                "session_id": uuid.uuid4().hex,
                "task": task,
                "task_hash": hashlib.sha1(episode_key.encode()).hexdigest()[:8],
                "is_correct": bool(sample.reward >= 1.0),
                "workflow_reward": 0.0,
                "eval_reward": sample.reward,
                "termination_reason": "env_done" if status == "completed" else status,
                "metrics": {"traj/steps": float(len(trace)), "tool_calls": float(search_calls)},
                "metadata": {
                    "model": args.model,
                    "prediction": response,
                    "grm": sample.metadata.get("grm"),
                    "verification": sample.metadata.get("verification"),
                    "timing": timing,
                },
                "timing": timing,
                "trajectories": [{
                    "name": f"{name}_{index}",
                    "uid": str(uuid.uuid4()),
                    "reward": sample.reward,
                    "num_steps": len(steps),
                    "timing": timing,
                    "steps": steps,
                }],
            }, sample

        tasks = [asyncio.create_task(run(job)) for job in jobs]
        results = []
        with tqdm(total=len(tasks), desc="OpenRouter eval", unit="sample", dynamic_ncols=True) as progress:
            for task in asyncio.as_completed(tasks):
                results.append(await task)
                progress.update()

    output = Path(args.output)
    write_trajectories(output, results)
    print(f"Wrote {len(results)} trajectories: {output}")
    if args.harness == "gem":
        step_count = sum(record["trajectories"][0]["num_steps"] for _, record, _ in results)
        search_count = sum(int(record["metrics"]["tool_calls"]) for _, record, _ in results)
        print(
            f"ReAct summary: samples={len(results)}; model_turns={step_count}; "
            f"web_search_calls={search_count}; avg_turns={step_count / max(len(results), 1):.2f}"
        )
    if args.enable_use_grm_evals:
        samples_by_dataset = {}
        for _, record, sample in sorted(results, key=lambda item: item[0]):
            dataset = record["task"]["data_source"]
            samples_by_dataset.setdefault(dataset, []).append(sample)
        table_args = SimpleNamespace(
            eval_datasets=[
                SimpleNamespace(name=name, n_samples_per_eval_prompt=args.n_samples)
                for name in samples_by_dataset
            ],
            n_samples_per_eval_prompt=args.n_samples,
        )
        table_data = {
            name: {"rewards": [float(sample.reward) for sample in samples], "samples": samples}
            for name, samples in samples_by_dataset.items()
        }
        print(
            format_eval_results_table(
                table_args,
                table_data,
                split_metric_columns=True,
                extended_termination_columns=False,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--harness", choices=("gem", "cot", "bare"), default="gem")
    parser.add_argument("--user-prompt", choices=("long", "short"), default="short")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:65433")
    parser.add_argument("--retrieval-max-results", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--enable-use-grm-evals", type=lambda value: value.lower() in {"1", "true", "yes", "on"}, default=True)
    parser.add_argument("--grm-model", default="google/gemini-3-flash-preview")
    parser.add_argument("--grm-base-url", default="")
    parser.add_argument("--grm-mode", choices=("score", "equivalence"), default="score")
    parser.add_argument("--grm-concurrency", type=int, default=128)
    parser.add_argument("--grm-max-connections", type=int, default=128)
    parser.add_argument("--grm-timeout", type=float, default=60)
    parser.add_argument("--grm-max-retries", type=int, default=32)
    parser.add_argument("--grm-max-input-tokens", type=int, default=131072)
    parser.add_argument("--grm-max-new-tokens", type=int, default=2048)
    parser.add_argument("--grm-temperature", type=float, default=0.6)
    parser.add_argument("--grm-failure-reward", type=float, default=0.0)
    asyncio.run(main(parser.parse_args()))
