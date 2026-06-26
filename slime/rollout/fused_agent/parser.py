from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    start: int | None = None
    end: int | None = None


class QwenToolParser:
    tool_call_begin = "<tool_call>"
    tool_call_end = "</tool_call>"
    tool_output_begin = "<tool_response>"
    tool_output_end = "</tool_response>"

    _ALIASES = {
        "submit": "finish",
        "function_call": "finish",
        "search_result": "web_search",
        "str_replace_editor": "file_editor",
    }

    def __init__(self, valid_tools: set[str] | None = None):
        self.valid_tools = valid_tools

    def get_tool_prompt(self, tools_schema: str) -> str:
        return (
            "\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_schema}\n</tools>\n\n"
            "For each function call, return a valid json object with function name and arguments "
            "within a pairwise <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "Make sure all curly braces and XML tags are correctly balanced and closed strictly."
        )

    def parse(self, model_response: str) -> list[ToolCall]:
        calls = []
        for payload, start, end in self._extract_payloads(model_response or ""):
            parsed = self._parse_payload(payload)
            if parsed is None:
                continue
            name = self._normalize_name(str(parsed.get("name", "")))
            if not name:
                continue
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"result": args} if name == "finish" else {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=name, arguments=args, start=start, end=end))
        if not calls:
            calls.extend(self._extract_answer_fallback(model_response or ""))
        return calls

    def _extract_payloads(self, text: str) -> list[tuple[str | tuple[str, str], int, int]]:
        payloads = [
            (match.group(1), match.start(), match.end())
            for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
        ]
        if payloads:
            return payloads
        # Accept a bare JSON tool call after reasoning.
        idx = text.rfind("</think>")
        search_text = text[idx + len("</think>") :] if idx >= 0 else text
        m = re.search(r'\{\s*"name"\s*:\s*"[^"]+".*?\}', search_text, flags=re.DOTALL)
        if m:
            offset = idx + len("</think>") if idx >= 0 else 0
            return [(m.group(0), offset + m.start(), offset + m.end())]
        offset = idx + len("</think>") if idx >= 0 else 0
        return [
            ((match.group(1), match.group(2)), offset + match.start(), offset + match.end())
            for match in re.finditer(
                r"<function=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</function>",
                search_text,
                flags=re.DOTALL,
            )
        ]

    def _parse_payload(self, payload: str) -> dict[str, Any] | None:
        if isinstance(payload, tuple):
            return self._parse_legacy_function_payload(payload)
        payload = payload.strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            obj = self._repair_finish_payload(payload) or self._extract_first_json(payload)
        if not isinstance(obj, dict):
            return None
        if "name" in obj:
            if "arguments" not in obj:
                obj = {"name": obj.get("name"), "arguments": {k: v for k, v in obj.items() if k != "name"}}
            return obj
        return None

    @staticmethod
    def _parse_legacy_function_payload(payload: tuple[str, str]) -> dict[str, Any]:
        name, body = payload
        args: dict[str, Any] = {}
        for key, value in re.findall(
            r"<parameter=([A-Za-z0-9_.-]+)>\s*(.*?)\s*</parameter>",
            body,
            flags=re.DOTALL,
        ):
            args[key] = value.strip()
        return {"name": name, "arguments": args}

    @staticmethod
    def _repair_finish_payload(payload: str) -> dict[str, Any] | None:
        if '"finish"' not in payload and "'finish'" not in payload and '"submit"' not in payload and "'submit'" not in payload:
            return None
        result_match = re.search(r'"result"\s*:\s*"', payload, flags=re.DOTALL)
        if result_match is None:
            return None
        start = result_match.end()
        end = payload.rfind("</tool_call>")
        search_end = end if end >= start else len(payload)
        result_end = search_end
        for marker in ('"}}', '"}}', '"}', '}'):
            idx = payload.rfind(marker, start, search_end)
            if idx >= start:
                result_end = idx
                break
        result = payload[start:result_end].strip()
        while result.endswith("}") and result.count("}") > result.count("{"):
            result = result[:-1].rstrip()
        if not result:
            return None
        return {"name": "finish", "arguments": {"command": "submit", "result": result}}

    @staticmethod
    def _extract_first_json(text: str) -> dict[str, Any] | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _normalize_name(self, name: str) -> str:
        name = name.strip()
        name = self._ALIASES.get(name, name)
        if self.valid_tools is None or name in self.valid_tools:
            return name
        underscored = name.replace("-", "_")
        if underscored in self.valid_tools:
            return underscored
        return ""

    def _extract_answer_fallback(self, text: str) -> list[ToolCall]:
        matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        if matches:
            match = matches[-1]
            return [ToolCall("finish", {"command": "submit", "result": match.group(1).strip()}, match.start(), match.end())]
        boxed, start, end = _extract_boxed(text)
        if boxed:
            return [ToolCall("finish", {"command": "submit", "result": boxed}, start, end)]
        return []


def _extract_boxed(text: str) -> tuple[str, int | None, int | None]:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return "", None, None
    start = idx + len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip(), idx, i + 1
    return "", None, None


def tool_schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }
