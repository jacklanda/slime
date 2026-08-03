from __future__ import annotations

import json
import re
from typing import Any

from .parser import QwenToolParser, ToolCall, tool_schema


UNIFIED_SYSTEM_PROMPT = """You are a multi-step QA agent that plans research, calls tools, and produces concise final answers.

## Rules
1. Call one tool at a time; wait for its result before deciding the next step.
2. If a tool call fails or returns irrelevant results, try a different query or tool; do not repeat the same call.
3. Each tool call is stateless; pass all necessary context explicitly in the arguments.
4. Use `<answer>` only when you already have the confirmed answer from tool results.
5. Every response must start with `<think>...</think>`.

## Tool Call Guidelines
- Mandatory pattern: `web_search_wiki` (captions only) -> `visit_wiki` (full page) -> extract fact -> answer.
- Only facts from a visited page may be used to answer.

## Final Answer Format
- Number: digits only, no commas, and no units unless specified.
- String: no articles or abbreviations unless specified.
- List: comma-separated, applying the rules above to each element.
- `<answer>` must contain the direct fact, number, name, or list, not a process description."""

PLAN_SYSTEM_PROMPT = """You are the planning module of the Agent, responsible for high-level strategy and progress-state updates for question answering tasks.

## Available Information
- `Target Task`: The specific task to be completed.
- `Recent Steps`: The most recent actions taken by the agent.
- `Previous Progress State`: A JSON representation of the task's progress.
- `Tool Definitions`

## Progress State Schema
The progress state MUST remain concise and contain all four writable fields:
- completed_list (List[str]): Finished steps and confirmed findings.
- todo_list (List[str]): A living checklist. Remove finished items, add follow-ups, and reorder by urgency.
- experience (List[str]): Lessons learned, failed strategies, or useful tips.
- information (List[str]): Confirmed facts extracted from page content.

Output a valid JSON dict with all four fields. On the first call, identify every searchable entity from the task text."""

ACTION_SYSTEM_PROMPT = """You are the action module of the Agent. Select and call the appropriate tool for the next step.

## Rules
1. Call one tool at a time and wait for its result before deciding the next step.
2. If a call fails or is irrelevant, change the query or tool; do not repeat the same call.
3. Tool calls are stateless. Pass all necessary context in their arguments.
4. Use `<answer>` only for a confirmed, specific answer obtained from tool results.
5. Never generate `[REFLECTION]` or `[STATE_UPDATE]`.
6. `<think>...</think>` is mandatory before every `<tool_call>` or `<answer>`.
7. Use `web_search_wiki` to find candidate pages, then `visit_wiki` to verify the full page before answering.

## Tool Definitions
- web_search_wiki: search the local corpus. Arguments: {"query": string}.
- visit_wiki: read a page returned by search. Arguments: {"url": string, "goal": string}."""

END_SYSTEM_PROMPT = """Generate the best concise answer from the research available. Return `<think>...</think>` followed by `<answer>DIRECT ANSWER</answer>`. The answer must be a specific fact, number, name, or comma-separated list."""

EMPTY_STATE = {"completed_list": [], "todo_list": [], "experience": [], "information": []}


class DeepSearchWorldParser(QwenToolParser):
    """Parse the checkpoint's JSON/XML protocol independently of its model family.

    Generated text can replay prompt examples before emitting the actual action.
    Selecting the last complete protocol element preserves normal single-action
    behavior while preventing stale example answers from terminating the episode.
    """

    _ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
    _TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)

    def parse(self, model_response: str) -> list[ToolCall]:
        text = model_response or ""
        candidates: list[tuple[int, ToolCall]] = []
        for match in self._ANSWER_RE.finditer(text):
            answer = match.group(1).strip()
            if answer:
                candidates.append(
                    (match.start(), ToolCall("finish", {"command": "submit", "result": answer}, match.start(), match.end()))
                )
        for match in self._TOOL_RE.finditer(text):
            payload = _fix_json_quotes(match.group(1).strip())
            parsed = self._parse_payload(payload)
            if not isinstance(parsed, dict):
                continue
            name = self._normalize_name(str(parsed.get("name", "")))
            arguments = parsed.get("arguments", {})
            if name and isinstance(arguments, dict):
                candidates.append((match.start(), ToolCall(name, arguments, match.start(), match.end())))
        if not candidates:
            return []
        return [max(candidates, key=lambda item: item[0])[1]]


def _fix_json_quotes(text: str) -> str:
    text = re.sub(r'"\s*queries\s*"\s*:\s*\[\s*""(.*?)""\s*\]', r'"queries": ["\1"]', text)
    return text.replace('""', '"')


def tools() -> list[dict[str, Any]]:
    return [
        tool_schema(
            "web_search_wiki",
            "Search the local knowledge corpus and return candidate pages with captions.",
            {"query": {"type": "string", "description": "Entity or fact to search for."}},
            ["query"],
        ),
        tool_schema(
            "visit_wiki",
            "Visit a page returned by web_search_wiki and read its contents.",
            {
                "url": {"type": "string", "description": "URL returned by web_search_wiki."},
                "goal": {"type": "string", "description": "Fact to extract from the page."},
            },
            ["url"],
        ),
    ]


def plan_messages(task: str, state: dict[str, Any] | None = None) -> list[dict[str, str]]:
    state = state or EMPTY_STATE
    user = f"""[STATE_UPDATE]
## Target Task
{task}

## Recent Steps
(no steps yet)

## Current Progress State
{json.dumps(state, ensure_ascii=False)}

## Output
Return `<think>` describing the entities and research plan, followed by a JSON object with completed_list, todo_list, experience, and information. Do not output reflection messages."""
    return [{"role": "system", "content": PLAN_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def action_messages(task: str, state: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = f"""## Target Task
{task}

## Recent Steps
{format_recent_steps(steps)}

## Progress State
{json.dumps(state, ensure_ascii=False)}

## Output
Use exactly one format:
<think>I know: ... I still need: ... Next: ...</think>
<tool_call>{{"name":"web_search_wiki","arguments":{{"query":"..."}}}}</tool_call>

or, only after visiting evidence:
<think>I know the confirmed fact. No more searches required.</think>
<answer>DIRECT ANSWER</answer>"""
    return [
        {"role": "system", "content": UNIFIED_SYSTEM_PROMPT + "\n\n" + ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def end_messages(task: str, state: dict[str, Any], steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    recent = format_recent_steps(steps)
    user = f"""## Target Task
{task}

## Recent Steps
{recent}

## Progress State
{json.dumps(state, ensure_ascii=False)}

## Final Step
{recent.split('### Step')[-1].strip() if steps else '(no steps)'}

## Stop Reason
The agent reached the maximum number of steps. Use all information gathered so far.

## Output
Return `<think>` followed by `<answer>DIRECT ANSWER</answer>`."""
    return [{"role": "system", "content": END_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_plan(text: str) -> dict[str, list[Any]]:
    candidates = []
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        candidates.append(fenced.group(1))
    outer = re.search(r"\{[\s\S]*\}", text)
    if outer:
        candidates.append(outer.group(0))
    for candidate in reversed(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return {
                key: list(value.get(key, [])) if isinstance(value.get(key), list) else []
                for key in EMPTY_STATE
            }
    return {key: [] for key in EMPTY_STATE}


def parse_answer(text: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE))
    return matches[-1].group(1).strip() if matches else None


def ensure_think_tags(text: str) -> str:
    if re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE):
        return text
    boundary = re.search(r"<(?:tool_call|answer)>", text, re.IGNORECASE)
    if boundary:
        return f"<think>{text[:boundary.start()].strip()}</think>\n{text[boundary.start():].lstrip()}"
    return f"<think>{text.strip()}</think>"


def step_record(response: str, action: ToolCall | None = None, observation: str = "") -> dict[str, Any]:
    thought_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    return {
        "thought": thought_match.group(1).strip() if thought_match else "",
        "tool_name": action.name if action else "",
        "tool_args": action.arguments if action else {},
        "observation": observation,
    }


def format_recent_steps(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return "(no steps yet)"
    blocks = []
    for index, step in enumerate(steps[-2:], start=max(1, len(steps) - 1)):
        blocks.append(
            f"### Step {index}\n"
            f"Thought: {step.get('thought', '')}\n"
            f"Action: {step.get('tool_name', '')}({json.dumps(step.get('tool_args', {}), ensure_ascii=False)})\n"
            f"Observation: {step.get('observation', '')}"
        )
    return "\n\n".join(blocks)
