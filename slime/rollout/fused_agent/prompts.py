from __future__ import annotations

import json

from .parser import Gemma4ToolParser, make_tool_parser, tool_schema


FUSED_SEARCH_SYSTEM_PROMPT = """You are a research assistant that answers questions by searching for relevant information. You have access to a web_search tool for looking up facts, and a finish tool to submit your final answer.

RULES:
1. Call web_search as many times as needed until you have concrete evidence for every part of the question.
2. If a search result is short, vague, or only echoes the query, issue a new query with different keywords.
3. For multi-hop questions, decompose into sub-questions and search each sub-question separately.
4. Synthesize the search results to form an accurate, concise answer.
5. Emit exactly one tool call per assistant response.
6. Submit the final answer only by calling finish. Put only the concise answer in finish.result; do not use <answer> tags or \\boxed{}.
"""

FUSED_SEARCH_USER_PROMPT = """Answer the following question by searching for relevant information.

<question>
{problem_statement}
</question>

Use web_search to gather evidence. When ready, call finish with result set to the concise final answer."""

FUSED_SEARCH_LONG_USER_PROMPT = """Answer the following question by searching for relevant information.

<question>
{problem_statement}
</question>

Instructions:
1. Use the web_search tool to find relevant information. Search as many times as needed and do not stop after a fixed number of searches - keep querying until you have concrete supporting evidence.
2. If a search result is short, vague, or just echoes the question, issue a new query with different keywords or add named entities, dates, or numbers.
3. For multi-hop questions, decompose into sub-questions and search each separately.
4. Write queries in the same language as the question (e.g., Chinese question -> Chinese query).
5. Synthesize the search results to form an accurate answer grounded in retrieved text.
6. When you have found the answer, call finish exactly once with the concise answer in the result parameter. Do not also emit <answer> tags, \\boxed{{}}, or plain-text final-answer prose.

IMPORTANT: Do NOT use file editing tools (file_editor, execute_bash, search) for this task — only use web_search and finish.
"""

FUSED_MCP_SYSTEM_PROMPT = """You are a tool agent. You are given a task to complete using the provided tools.

CRITICAL RULES:
1. Use available non-finish tools when they are relevant to the task. If none apply, do not call an unrelated tool.
2. Plan your approach, then call tools step by step to collect evidence.
3. Start retrieval broadly. If a tool returns an empty result, retry with fewer filters or use a listing tool before narrowing the query.
4. Be precise in tool arguments and respect parameter types.
5. Emit tool calls exactly in the format shown in the Tools section below.
6. Submit your final answer as a JSON value via the finish tool: a JSON array directly when the task asks for multiple items, not wrapped in another object.
7. Never submit an empty answer when the task asks you to extract or analyze evidence. The finish result must be pure JSON and must not contain another tool call.
"""

FUSED_MCP_USER_PROMPT = """Solve the following task using the available tools.

<task>
{problem_statement}
</task>

Use relevant tools to retrieve any data you need. Submit using finish when complete."""

FUSED_CLI_SYSTEM_PROMPT = """You are a CLI agent tasked with resolving a repository issue in a Linux environment. You have access to tools that operate inside the task container.

RULES:
1. Explore the repository before editing. Verify paths exist before modifying files.
2. Make the smallest source change that satisfies the task.
3. After edits, run syntax checks and targeted tests when available.
4. Do not submit until you have evidence the fix works.
5. Use only tools shown in the current schema."""

FUSED_CLI_USER_PROMPT = """Consider the following issue:

<issue>
{problem_statement}
</issue>

Explore, edit, verify, and submit using finish when the issue is resolved."""

FUSED_ET_SYSTEM_PROMPT = """You are a Linux CLI agent operating in a self-contained Docker container. The task is a free-standing system or scripting task, and an automated verifier grades the final filesystem state.

RULES:
1. Start by inspecting files and directories in the container.
2. Prefer small shell commands or file edits that directly satisfy the instruction.
3. Verify requested invariants before submitting.
4. Do not call finish until the final state is present.
5. Use only tools shown in the current schema."""

FUSED_ET_USER_PROMPT = """Complete the following task in the Linux container.

<task>
{problem_statement}
</task>

Inspect the current state, make the required changes, verify them, and submit using finish."""

FUSED_UNIFIED_SYSTEM_PROMPT = """You are a general agent that can solve MCP/general tool-use tasks, CLI/SWE tasks, Endless-Terminal tasks, and Web Search QA tasks.

GENERAL RULES:
1. Infer the task family from the user message and the available tool schemas.
2. Use only tools shown in the current schema.
3. Plan before acting, call tools step by step, and adapt when a tool fails.
4. Submit only when the available evidence supports the final answer or final filesystem state.

TASK-SPECIFIC RULES:
- MCP: start with broad/listing queries; after an empty result, remove filters and retrieve evidence before submitting a JSON value.
- MCP: the finish result must be pure JSON, never an embedded tool call.
- MCP: do not submit an empty result for an extraction or evidence-analysis task.
- CLI/SWE: explore the repository, make minimal edits, verify syntax, and run relevant tests before submitting.
- Endless Terminal: inspect the filesystem, make the requested final-state changes, verify them, then submit.
- Web Search QA: search until every claim is grounded, then submit a concise final answer.
- Emit tool calls exactly in the format shown in the Tools section below."""

REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions by reasoning and searching for relevant information. Use Thought/Action/Observation until ready, then call finish."""
REACT_USER_PROMPT = """Task:
{problem_statement}

Use the ReAct loop and call available tools in the format shown in the Tools section."""
COT_SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""
COT_USER_PROMPT = "{problem_statement}"
RAG_USER_PROMPT = """Answer the question using the retrieved context below. Reason step by step, and put your final answer within \\boxed{{}}.

<question>
{problem_statement}
</question>

<context>
{retrieved_context}
</context>"""


def finish_schema(
    *,
    structured_result: bool = False,
    result_schema: dict | None = None,
) -> dict:
    if result_schema is not None:
        result_schema = dict(result_schema)
        result_schema.setdefault("description", "Final answer or JSON value.")
    elif structured_result:
        result_schema = {
            "type": ["string", "object", "array", "number", "boolean", "null"],
            "description": "Final answer or JSON value.",
        }
    else:
        result_schema = {
            "type": "string",
            "description": "Final answer or JSON value.",
        }
    return tool_schema(
        "finish",
        "Finish the task and submit the final result.",
        {
            "command": {"type": "string", "description": "Use submit."},
            "result": result_schema,
        },
        ["command", "result"],
    )


def web_search_schema() -> dict:
    return tool_schema(
        "web_search",
        "Search for information relevant to the question.",
        {
            "query": {"type": "string", "description": "Search query."},
            "max_results": {"type": "integer", "description": "Maximum number of results."},
        },
        ["query"],
    )


def build_system_prompt(
    base_prompt: str,
    schemas: list[dict],
    model_name: str | None = None,
    *,
    tool_parser=None,
    inline_tool_prompt: bool = True,
) -> str:
    parser = tool_parser or make_tool_parser(model_name)
    schemas_str = "\n".join(json.dumps(schema, indent=0, ensure_ascii=False) for schema in schemas)
    tool_prompt = parser.get_tool_prompt(schemas_str)
    if isinstance(parser, Gemma4ToolParser):
        contract = parser.get_tool_contract()
        return base_prompt.strip() + ("\n\n" + contract if contract else "")
    if not inline_tool_prompt:
        return base_prompt.strip()
    return base_prompt.strip() + "\n" + tool_prompt


def normalize_harness(harness: str | None) -> str:
    value = (harness or "gem").strip().lower().replace("-", "_")
    aliases = {
        "unified": "unified_gem",
        "fused": "gem",
        "chain_of_thought": "cot",
        "no_system": "bare",
        "rllm_dr": "rllm_deepresearch",
        "deepresearch": "rllm_deepresearch",
        "cutbill": "cut_bill",
        "deepsearch_world": "deepsearch_world",
    }
    value = aliases.get(value, value)
    if value not in {"gem", "unified_gem", "react", "cot", "rag", "bare", "rllm_deepresearch", "cut_bill", "search_gym", "deepsearch_world", "agentcpm_explore"}:
        raise ValueError(f"Invalid fused harness: {harness!r}")
    return value
