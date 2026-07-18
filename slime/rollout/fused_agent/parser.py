from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


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

    def format_action(self, action: ToolCall) -> str:
        payload = {"name": action.name, "arguments": action.arguments or {}}
        return "<tool_call>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</tool_call>"

    def format_tool_observation(self, tool_name: str, output_text: str) -> str:
        return "<tool_response>\n" f"Execution output of [{tool_name}]:\n" f"{output_text}\n" "</tool_response>"

    def assistant_tool_result_message(self, action: ToolCall, response: Any) -> dict[str, Any] | None:
        return None

    def assistant_tool_results_message(self, actions: list[ToolCall], responses: list[Any]) -> dict[str, Any] | None:
        return None

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
        idx = text.rfind("</think>")
        offset = idx + len("</think>") if idx >= 0 else 0
        search_text = text[offset:]
        payloads = [
            (match.group(1), offset + match.start(), offset + match.end())
            for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", search_text, flags=re.DOTALL)
        ]
        if payloads:
            return payloads
        # Accept a bare JSON tool call after reasoning.
        m = re.search(r'\{\s*"name"\s*:\s*"[^"]+".*?\}', search_text, flags=re.DOTALL)
        if m:
            return [(m.group(0), offset + m.start(), offset + m.end())]
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
        for marker in ('"}}', '"}}', '"}', "}"):
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


class Qwen3CoderToolParser(QwenToolParser):
    """Parser for the Qwen3.5 / Qwen3-Coder XML tool-call format.

    Qwen3 emits JSON inside ``<tool_call>`` tags; Qwen3.5/Qwen3-Coder emits
    function and parameter blocks instead::

        <tool_call>
        <function=web_search>
        <parameter=query>value</parameter>
        </function>
        </tool_call>

    Ported from ``rllm.parser.tool_parser.Qwen3CoderToolParser`` and adapted to
    slime's ``ToolCall`` (start/end offsets) and fallback chain: when no
    ``<function=>`` call is present we defer to the shared ``<answer>`` /
    ``\\boxed{}`` -> ``finish`` fallback so disable-thinking rollouts still submit.
    """

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL)
    _FUNCTION_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.DOTALL)
    _PARAMETER_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)

    def __init__(self, valid_tools: set[str] | None = None):
        super().__init__(valid_tools=valid_tools)
        self.tool_call_prefix = "<function="
        self._tool_parameter_config: dict[str, dict[str, dict[str, Any]]] = {}

    def parse(self, model_response: str) -> list[ToolCall]:
        text = model_response or ""
        calls: list[ToolCall] = []
        for inner, start, end in self._iter_tool_call_regions(text):
            for function_name, parameters in self._FUNCTION_RE.findall(inner):
                function_name = function_name.strip()
                if not function_name:
                    continue
                parsed = self._parse_xml_function_call(function_name, parameters)
                name = self._normalize_name(parsed["name"])
                if not name:
                    continue
                calls.append(ToolCall(name=name, arguments=parsed["arguments"], start=start, end=end))
        if not calls:
            calls.extend(self._extract_answer_fallback(text))
        return calls

    def format_action(self, action: ToolCall) -> str:
        lines = ["<tool_call>", f"<function={action.name}>"]
        for name, value in (action.arguments or {}).items():
            if value is None:
                rendered = "null"
            elif isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                rendered = str(value)
            lines.extend([f"<parameter={name}>", rendered, "</parameter>"])
        lines.extend(["</function>", "</tool_call>"])
        return "\n".join(lines)

    def _iter_tool_call_regions(self, text: str) -> list[tuple[str, int, int]]:
        regions = [(match.group(1) if match.group(1) is not None else (match.group(2) or ""), match.start(), match.end()) for match in self._TOOL_CALL_RE.finditer(text)]
        if regions:
            return regions
        # No <tool_call> wrapper: accept bare <function=...></function> blocks, each
        # its own region so start/end bound the function span for finish-call checks.
        if self.tool_call_prefix in text:
            return [(match.group(0), match.start(), match.end()) for match in self._FUNCTION_RE.finditer(text)]
        return regions

    @staticmethod
    def _strip_outer_newline(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        return value

    @staticmethod
    def _iter_tool_schemas(tools_schema: str) -> list[dict[str, Any]]:
        tools_schema = tools_schema.strip()
        if not tools_schema:
            return []
        try:
            loaded = json.loads(tools_schema)
            if isinstance(loaded, list):
                return [item for item in loaded if isinstance(item, dict)]
            if isinstance(loaded, dict):
                return [loaded]
        except (json.JSONDecodeError, ValueError):
            pass
        # The tool prompt concatenates several JSON objects separated by newlines;
        # decode them one at a time.
        decoder = json.JSONDecoder()
        schemas: list[dict[str, Any]] = []
        idx = 0
        while idx < len(tools_schema):
            while idx < len(tools_schema) and tools_schema[idx].isspace():
                idx += 1
            if idx >= len(tools_schema):
                break
            try:
                obj, next_idx = decoder.raw_decode(tools_schema, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(obj, dict):
                schemas.append(obj)
            elif isinstance(obj, list):
                schemas.extend(item for item in obj if isinstance(item, dict))
            idx = next_idx
        return schemas

    @classmethod
    def _extract_parameter_config(cls, tools_schema: str) -> dict[str, dict[str, dict[str, Any]]]:
        config: dict[str, dict[str, dict[str, Any]]] = {}
        for schema in cls._iter_tool_schemas(tools_schema):
            function_schema = schema.get("function", schema)
            if not isinstance(function_schema, dict):
                continue
            name = function_schema.get("name")
            parameters = function_schema.get("parameters", {})
            if not isinstance(name, str) or not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties", {})
            if isinstance(properties, dict):
                config[name] = {str(k): v for k, v in properties.items() if isinstance(v, dict)}
        return config

    @staticmethod
    def _convert_param_value(param_value: str, param_name: str, param_config: dict[str, dict[str, Any]], func_name: str) -> Any:
        if param_value.lower() == "null":
            return None
        if param_name not in param_config:
            if param_config:
                logger.warning(
                    "Parsed parameter '%s' is not defined in the tool parameters for tool '%s'; returning string value.",
                    param_name,
                    func_name,
                )
            return param_value

        param_schema = param_config[param_name]
        param_type = str(param_schema.get("type", "string")).strip().lower()

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        if param_type.startswith("int") or param_type.startswith("uint") or param_type.startswith("long") or param_type.startswith("short") or param_type.startswith("unsigned"):
            try:
                # Qwen3.5 occasionally emits ``<parameter=name>: 10``. Treat the
                # extra colon as a formatting delimiter for integer parameters.
                return int(re.sub(r"^:\s*", "", param_value))
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not an integer in tool '%s'; returning string value.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_value = float(param_value)
                return float_value if float_value - int(float_value) != 0 else int(float_value)
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a float in tool '%s'; returning string value.",
                    param_value,
                    param_name,
                    func_name,
                )
                return param_value
        if param_type in ["boolean", "bool", "binary"]:
            bool_value = param_value.lower()
            if bool_value not in ["true", "false"]:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a boolean in tool '%s'; degenerating to false.",
                    param_value,
                    param_name,
                    func_name,
                )
            return bool_value == "true"
        if param_type in ["object", "array"] or param_type.startswith("dict") or param_type.startswith("list"):
            try:
                return json.loads(param_value)
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not valid JSON in tool '%s'; trying ast.literal_eval().",
                    param_value,
                    param_name,
                    func_name,
                )
        try:
            return ast.literal_eval(param_value)
        except Exception:
            logger.warning(
                "Parsed value '%s' of parameter '%s' cannot be converted via ast.literal_eval() in tool '%s'; " "returning string value.",
                param_value,
                param_name,
                func_name,
            )
            return param_value

    def _parse_xml_function_call(self, function_name: str, parameters: str) -> dict[str, Any]:
        param_config = self._tool_parameter_config.get(function_name, {})
        arguments: dict[str, Any] = {}
        for param_name, param_value in self._PARAMETER_RE.findall(parameters):
            param_name = param_name.strip()
            param_value = self._strip_outer_newline(str(param_value))
            arguments[param_name] = self._convert_param_value(param_value, param_name, param_config, function_name)
        return {"name": function_name.strip(), "arguments": arguments}

    def get_tool_prompt(self, tools_schema: str) -> str:
        # Cache the parameter types so parse() can coerce <parameter> strings.
        self._tool_parameter_config = self._extract_parameter_config(tools_schema)
        return (
            "\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tools_schema}\n</tools>\n\n"
            "For each function call, return the function name and parameters within a pairwise "
            "<tool_call></tool_call> XML block:\n"
            "<tool_call>\n"
            "<function=FUNCTION_NAME>\n"
            "<parameter=PARAMETER_NAME>\n"
            "PARAMETER_VALUE\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "For multiple parameters, emit one <parameter=...></parameter> block per parameter "
            "inside the same function block.\n"
            "For multiple tool calls, emit multiple <tool_call></tool_call> blocks.\n"
            "Do not put JSON tool-call objects inside <tool_call> for this model family."
        )


class Gemma4ToolParser(QwenToolParser):
    """Parser and prompt formatter for Gemma4's native tool-call template.

    Gemma4 renders tool definitions as ``<|tool>declaration:name{...}<tool|>``
    in the system turn and emits calls as ``<|tool_call>call:name{...}<tool_call|>``.
    Tool arguments are a compact object syntax rather than JSON: keys are
    bare/quoted, strings are wrapped in ``<|"|>...<|"|>``, and nested arrays /
    objects use JSON-like delimiters.
    """

    tool_call_begin = "<|tool_call>"
    tool_call_end = "<tool_call|>"
    tool_output_begin = "<|tool_response>"
    tool_output_end = "<tool_response|>"
    _CALL_PREFIX_RE = re.compile(r"\s*call:([A-Za-z0-9_.-]+)\s*", re.DOTALL)

    def get_tool_prompt(self, tools_schema: str) -> str:
        declarations = []
        for schema in Qwen3CoderToolParser._iter_tool_schemas(tools_schema):
            declaration = self._format_function_declaration(schema)
            if declaration:
                declarations.append(f"<|tool>{declaration}<tool|>")
        if not declarations:
            return ""
        return (
            "\n"
            + "".join(declarations)
            + "\n"
            "When you need to call a tool, emit exactly:\n"
            "<|tool_call>call:TOOL_NAME{ARGUMENT_NAME:<|\"|>ARGUMENT_VALUE<|\"|>}<tool_call|><|tool_response>\n"
            "Wait for the tool response before continuing. Do not wrap Gemma4 tool calls in XML or JSON."
        )

    def format_action(self, action: ToolCall) -> str:
        return f"<|tool_call>call:{action.name}{self._format_argument(action.arguments or {}, escape_keys=False)}<tool_call|>"

    def format_tool_observation(self, tool_name: str, output_text: str) -> str:
        return "<|tool_response>" f"response:{tool_name}{{value:{self._format_argument(output_text, escape_keys=False)}}}" "<tool_response|>"

    def assistant_tool_result_message(self, action: ToolCall, response: Any) -> dict[str, Any]:
        return self.assistant_tool_results_message([action], [response])

    def assistant_tool_results_message(self, actions: list[ToolCall], responses: list[Any]) -> dict[str, Any]:
        tool_responses = []
        for action, response in zip(actions, responses, strict=True):
            if not isinstance(response, (dict, list)):
                response = {"value": response}
            tool_responses.append({"name": action.name, "response": response})
        return {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": action.name, "arguments": action.arguments or {}}}
                for action in actions
            ],
            "tool_responses": tool_responses,
        }

    def parse(self, model_response: str) -> list[ToolCall]:
        text = model_response or ""
        calls: list[ToolCall] = []
        for raw_name, raw_args, start, end in self._iter_native_tool_calls(text):
            name = self._normalize_name(raw_name)
            if not name:
                continue
            try:
                args = self._parse_object(raw_args)
            except ValueError as exc:
                args = self._repair_arguments(name, raw_args)
                if args is not None:
                    calls.append(ToolCall(name=name, arguments=args, start=start, end=end))
                    continue
                logger.warning(
                    "Failed to parse Gemma4 tool-call arguments for tool '%s': %s. raw_args=%r",
                    raw_name,
                    exc,
                    raw_args[:512],
                )
                logger.debug("Gemma4 tool-call argument parse failure.", exc_info=True)
                continue
            if name == "finish" and isinstance(args, dict):
                cls_command = self._normalize_finish_command(str(args.get("command", "")))
                if cls_command is not None:
                    args["command"] = cls_command
            calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}, start=start, end=end))
        if not calls:
            calls.extend(self._extract_answer_fallback(text))
        return calls

    @classmethod
    def _repair_arguments(cls, name: str, raw_args: str) -> dict[str, Any] | None:
        if name == "finish":
            return cls._repair_finish_arguments(raw_args)
        if name == "web_search":
            return cls._repair_web_search_arguments(raw_args)
        return None

    @classmethod
    def _repair_web_search_arguments(cls, raw_args: str) -> dict[str, Any] | None:
        text = raw_args.strip()
        if not text.startswith("{"):
            return None
        inner = text[1:-1] if text.endswith("}") else text[1:]
        for separator in (":", "="):
            prefix = f"query{separator}"
            if inner.strip().startswith(prefix):
                raw_value = inner.strip()[len(prefix) :]
                query = cls._repair_text_value(
                    raw_value,
                    allow_unclosed_native=raw_value.strip().startswith('<|"|>'),
                )
                if query:
                    return {"query": query}
        return None

    @classmethod
    def _repair_finish_arguments(cls, raw_args: str) -> dict[str, Any] | None:
        text = raw_args.strip()
        if not text.startswith("{"):
            return None

        command_marker = 'command:<|"|>'
        command_start = text.find(command_marker)
        if command_start < 0:
            return None
        command_value_start = command_start + len(command_marker)
        result_key = ""
        result_start = -1
        for candidate in (",result:", ",result=", "{result:"):
            result_start = text.find(candidate, command_value_start)
            if result_start >= 0:
                result_key = candidate
                break
        if result_start < 0:
            payload_end = -1 if text.endswith("}") else len(text)
            return cls._repair_finish_command_payload(text[command_value_start:payload_end])

        command_close = text.find('<|"|>', command_value_start, result_start)
        if command_close >= 0:
            command = text[command_value_start:command_close].strip()
            if text[command_close + len('<|"|>') : result_start].strip():
                return None
        else:
            command = text[command_value_start:result_start].strip()
        command = cls._normalize_finish_command(command)
        if command is None:
            return None

        if result_key.startswith("{") and text.endswith("}}"):
            result_end = -2
        elif text.endswith("}"):
            result_end = -1
        else:
            result_end = len(text)
        result = cls._repair_finish_result_value(text[result_start + len(result_key) : result_end])
        if result is None:
            return None
        return {"command": command, "result": result}

    @staticmethod
    def _normalize_finish_command(command: str) -> str | None:
        command = command.strip()
        if command in {"submit", "finish"}:
            return command
        normalized = command.lower()
        if normalized.startswith(("finish(", "submit(")):
            return normalized.split("(", 1)[0]
        if "submit" in normalized or "final result" in normalized:
            return "submit"
        if "synthesize" in normalized and "answer" in normalized:
            return "submit"
        return None

    @classmethod
    def _repair_finish_result_value(cls, raw_value: str) -> Any:
        return cls._repair_text_value(raw_value, allow_unclosed_native=True)

    @classmethod
    def _repair_text_value(cls, raw_value: str, *, allow_unclosed_native: bool) -> Any:
        text = raw_value.strip()
        if not text:
            return None
        marker = '<|"|>'
        if text.startswith(marker):
            end = text.rfind(marker, len(marker))
            if end < len(marker):
                if not allow_unclosed_native:
                    return None
                return text[len(marker) :].rstrip('"}').rstrip('"').strip()
            if text[end + len(marker) :].strip():
                return None
            return text[len(marker) : end]
        if text[0] in {'"', "'"}:
            quote = text[0]
            end = text.rfind(quote, 1)
            if end <= 0:
                if allow_unclosed_native:
                    return text[1:].rstrip('"}').rstrip('"').strip()
                return None
            if text[end + 1 :].strip().strip(")"):
                return None
            return text[1:end]
        try:
            return cls._parse_object("{result:" + text + "}")["result"]
        except ValueError:
            return None

    @classmethod
    def _repair_finish_command_payload(cls, raw_payload: str) -> dict[str, Any] | None:
        text = raw_payload.strip()
        for command in ("submit", "finish"):
            prefix = f'{command}(result='
            if text.startswith(prefix):
                result_text = text[len(prefix) :].strip()
                if result_text.endswith(')"'):
                    result_text = result_text[:-1].rstrip()
                if result_text.endswith(")"):
                    result_text = result_text[:-1].rstrip()
                result = cls._repair_finish_result_value(result_text)
                if result is not None:
                    return {"command": command, "result": result}
            prefix = f'{command}(command='
            if text.startswith(prefix):
                result_text = text[len(prefix) :].strip()
                if result_text.endswith(')"'):
                    result_text = result_text[:-1].rstrip()
                if result_text.endswith(")"):
                    result_text = result_text[:-1].rstrip()
                result = cls._repair_finish_result_value(result_text)
                if result is not None:
                    return {"command": command, "result": result}

        for command in ("submit", "finish"):
            if text == command:
                return None
            if text.startswith(command + ","):
                result = cls._repair_finish_json_payload(text[len(command) + 1 :].strip())
                if result is not None:
                    return {"command": command, "result": result}
            if text.startswith(command) and text[len(command) : len(command) + 1].isspace():
                result = text[len(command) :].strip()
                if result:
                    return {"command": command, "result": result}
        return None

    @classmethod
    def _repair_finish_json_payload(cls, raw_payload: str) -> Any:
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return cls._repair_incomplete_finish_json_payload(raw_payload)
        if isinstance(payload, dict):
            if "result" in payload:
                return payload["result"]
            if "answer" in payload:
                return payload["answer"]
        return payload

    @classmethod
    def _repair_incomplete_finish_json_payload(cls, raw_payload: str) -> Any:
        text = raw_payload.strip()
        for key in ("answer", "result"):
            prefix = f'{{"{key}":'
            if text.startswith(prefix):
                return cls._repair_text_value(text[len(prefix) :], allow_unclosed_native=True)
        return None

    @classmethod
    def _iter_native_tool_calls(cls, text: str) -> list[tuple[str, str, int, int]]:
        calls = []
        search_pos = 0
        while True:
            start = text.find(cls.tool_call_begin, search_pos)
            if start < 0:
                return calls

            pos = start + len(cls.tool_call_begin)
            prefix = cls._CALL_PREFIX_RE.match(text, pos)
            if prefix is None:
                search_pos = pos
                continue

            raw_name = prefix.group(1)
            obj_start = prefix.end()
            try:
                obj_end = cls._find_argument_object_end(text, obj_start)
            except ValueError:
                end_marker_start = text.find(cls.tool_call_end, obj_start)
                if end_marker_start >= 0:
                    end = end_marker_start + len(cls.tool_call_end)
                    calls.append((raw_name, text[obj_start:end_marker_start], start, end))
                    search_pos = end
                    continue
                search_pos = pos
                continue

            end_marker_start = cls._skip_ws(text, obj_end)
            if not text.startswith(cls.tool_call_end, end_marker_start):
                search_pos = obj_end
                continue

            end = end_marker_start + len(cls.tool_call_end)
            calls.append((raw_name, text[obj_start:obj_end], start, end))
            search_pos = end

    @staticmethod
    def _skip_ws(text: str, pos: int) -> int:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        return pos

    @staticmethod
    def _find_argument_object_end(text: str, start: int) -> int:
        start = Gemma4ToolParser._skip_ws(text, start)
        if start >= len(text) or text[start] != "{":
            raise ValueError(f"Expected Gemma4 argument object at offset {start}.")

        marker = '<|"|>'
        depth = 0
        pos = start
        while pos < len(text):
            if text.startswith(marker, pos):
                end = text.find(marker, pos + len(marker))
                if end < 0:
                    raise ValueError("Unterminated Gemma4 string.")
                pos = end + len(marker)
                continue

            ch = text[pos]
            if ch in {'"', "'"}:
                pos = Gemma4ToolParser._find_quoted_string_end(text, pos, close_follow={":", ",", "}", "]"})
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return pos + 1
                if depth < 0:
                    raise ValueError(f"Unbalanced Gemma4 argument object at offset {pos}.")
            pos += 1

        raise ValueError("Unterminated Gemma4 argument object.")

    @staticmethod
    def _find_quoted_string_end(text: str, start: int, close_follow: set[str]) -> int:
        quote = text[start]
        pos = start + 1
        escaped = False
        while pos < len(text):
            ch = text[pos]
            pos += 1
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch != quote:
                continue
            next_pos = Gemma4ToolParser._skip_ws(text, pos)
            if next_pos >= len(text) or text[next_pos] in close_follow:
                return pos
        raise ValueError("Unterminated quoted Gemma4 string.")

    @classmethod
    def _format_function_declaration(cls, tool_data: dict[str, Any]) -> str:
        function = tool_data.get("function", tool_data)
        if not isinstance(function, dict) or not function.get("name"):
            return ""
        parts = [f"declaration:{function['name']}{{"]
        if function.get("description"):
            parts.append(f"description:{cls._format_argument(str(function['description']))}")
        params = function.get("parameters")
        if isinstance(params, dict):
            if len(parts) > 1:
                parts.append(",")
            parts.append("parameters:{")
            properties = params.get("properties")
            if isinstance(properties, dict):
                parts.append("properties:{")
                parts.append(cls._format_parameters(properties))
                parts.append("}")
            required = params.get("required") or []
            if required:
                if isinstance(properties, dict):
                    parts.append(",")
                parts.append("required:")
                parts.append(cls._format_argument(list(required)))
            if params.get("type"):
                if isinstance(properties, dict) or required:
                    parts.append(",")
                parts.append(f"type:{cls._format_schema_type(params['type'])}")
            for key in ("$defs", "definitions"):
                rendered = cls._format_schema_definitions(params.get(key))
                if rendered:
                    if isinstance(properties, dict) or required or params.get("type"):
                        parts.append(",")
                    parts.append(f"{key}:{rendered}")
            parts.append("}")
        parts.append("}")
        return "".join(parts)

    @classmethod
    def _format_parameters(cls, properties: dict[str, Any]) -> str:
        rendered = []
        for key, value in sorted(properties.items()):
            if not isinstance(value, dict):
                continue
            fields = cls._format_schema_fields(value)
            rendered.append(f"{cls._format_schema_key(key)}:{{{','.join(fields)}}}")
        return ",".join(rendered)

    @classmethod
    def _format_schema_fields(cls, schema: dict[str, Any]) -> list[str]:
        fields = []
        if schema.get("description"):
            fields.append(f"description:{cls._format_argument(str(schema['description']))}")
        if schema.get("enum"):
            fields.append(f"enum:{cls._format_argument(schema['enum'])}")
        if "const" in schema:
            fields.append(f"const:{cls._format_argument(schema['const'])}")
        if "default" in schema:
            fields.append(f"default:{cls._format_argument(schema['default'])}")
        if schema.get("nullable"):
            fields.append("nullable:true")
        if isinstance(schema.get("properties"), dict):
            fields.append(f"properties:{{{cls._format_parameters(schema['properties'])}}}")
            if schema.get("required"):
                fields.append(f"required:{cls._format_argument(list(schema['required']))}")
        if isinstance(schema.get("items"), dict):
            fields.append(f"items:{{{','.join(cls._format_schema_fields(schema['items']))}}}")
        elif isinstance(schema.get("items"), list):
            fields.append(f"items:{cls._format_schema_variants(schema['items'])}")
        if isinstance(schema.get("additionalProperties"), dict):
            fields.append(f"additionalProperties:{{{','.join(cls._format_schema_fields(schema['additionalProperties']))}}}")
        elif isinstance(schema.get("additionalProperties"), bool):
            fields.append(f"additionalProperties:{cls._format_argument(schema['additionalProperties'])}")
        for key in ("$defs", "definitions"):
            rendered = cls._format_schema_definitions(schema.get(key))
            if rendered:
                fields.append(f"{key}:{rendered}")
        for key in (
            "$ref",
            "format",
            "pattern",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "minItems",
            "maxItems",
            "uniqueItems",
            "minProperties",
            "maxProperties",
        ):
            if key in schema:
                fields.append(f"{key}:{cls._format_argument(schema[key])}")
        for key in ("anyOf", "oneOf", "allOf"):
            variants = schema.get(key)
            if isinstance(variants, list):
                fields.append(f"{key}:{cls._format_schema_variants(variants)}")
        if schema.get("type") is not None:
            fields.append(f"type:{cls._format_schema_type(schema['type'])}")
        return fields

    @classmethod
    def _format_schema_definitions(cls, definitions: Any) -> str:
        if not isinstance(definitions, dict):
            return ""
        rendered = []
        for key, value in sorted(definitions.items()):
            if isinstance(value, dict):
                rendered.append(f"{cls._format_schema_key(key)}:{{{','.join(cls._format_schema_fields(value))}}}")
            else:
                rendered.append(f"{cls._format_schema_key(key)}:{cls._format_argument(value)}")
        return "{" + ",".join(rendered) + "}"

    @classmethod
    def _format_schema_key(cls, key: Any) -> str:
        key = str(key)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
            return key
        return cls._format_argument(key)

    @classmethod
    def _format_schema_variants(cls, variants: list[Any]) -> str:
        rendered = []
        for variant in variants:
            if isinstance(variant, dict):
                rendered.append("{" + ",".join(cls._format_schema_fields(variant)) + "}")
            else:
                rendered.append(cls._format_argument(variant))
        return "[" + ",".join(rendered) + "]"

    @classmethod
    def _format_schema_type(cls, schema_type: Any) -> str:
        if isinstance(schema_type, str):
            return cls._format_argument(schema_type.upper())
        if isinstance(schema_type, (list, tuple)):
            return cls._format_argument([str(item).upper() for item in schema_type])
        return cls._format_argument(schema_type)

    @classmethod
    def _format_argument(cls, argument: Any, escape_keys: bool = True) -> str:
        if isinstance(argument, str):
            if '<|"|>' in argument:
                return json.dumps(argument, ensure_ascii=False)
            return '<|"|>' + argument + '<|"|>'
        if isinstance(argument, bool):
            return "true" if argument else "false"
        if argument is None:
            return "null"
        if isinstance(argument, dict):
            items = []
            for key, value in sorted(argument.items()):
                rendered_key = cls._format_argument(str(key)) if escape_keys else cls._format_schema_key(key)
                items.append(f"{rendered_key}:{cls._format_argument(value, escape_keys=escape_keys)}")
            return "{" + ",".join(items) + "}"
        if isinstance(argument, (list, tuple)):
            return "[" + ",".join(cls._format_argument(item, escape_keys=escape_keys) for item in argument) + "]"
        return str(argument)

    @classmethod
    def _parse_object(cls, text: str) -> dict[str, Any]:
        parser = _Gemma4ArgumentParser(text)
        value = parser.parse_value()
        parser.skip_ws()
        if parser.pos != len(parser.text):
            raise ValueError(f"Unexpected trailing Gemma4 argument text at offset {parser.pos}.")
        if not isinstance(value, dict):
            raise ValueError("Gemma4 tool-call arguments must be an object.")
        return value


class _Gemma4ArgumentParser:
    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0

    def parse_value(self) -> Any:
        self.skip_ws()
        if self.text.startswith('<|"|>', self.pos):
            return self.parse_gemma_string()
        ch = self.peek()
        if ch in {'"', "'"}:
            return self.parse_quoted_string(close_follow={",", "}", "]"})
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            return self.parse_array()
        token = self.parse_token()
        normalized_token = token.lower()
        if normalized_token == "true":
            return True
        if normalized_token == "false":
            return False
        if normalized_token in {"null", "none"}:
            return None
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    def parse_object(self) -> dict[str, Any]:
        self.expect("{")
        result: dict[str, Any] = {}
        self.skip_ws()
        if self.peek() == "}":
            self.pos += 1
            return result
        while True:
            key = self.parse_key()
            self.skip_ws()
            self.expect(":")
            result[key] = self.parse_value()
            self.skip_ws()
            ch = self.peek()
            if ch == ",":
                self.pos += 1
                continue
            if ch == "}":
                self.pos += 1
                return result
            raise ValueError(f"Expected ',' or '}}' at offset {self.pos}.")

    def parse_array(self) -> list[Any]:
        self.expect("[")
        result = []
        self.skip_ws()
        if self.peek() == "]":
            self.pos += 1
            return result
        while True:
            result.append(self.parse_value())
            self.skip_ws()
            ch = self.peek()
            if ch == ",":
                self.pos += 1
                continue
            if ch == "]":
                self.pos += 1
                return result
            raise ValueError(f"Expected ',' or ']' at offset {self.pos}.")

    def parse_key(self) -> str:
        self.skip_ws()
        if self.text.startswith('<|"|>', self.pos):
            return self.parse_gemma_string()
        if self.peek() in {'"', "'"}:
            return self.parse_quoted_string(close_follow={":"})
        return self.parse_token(stop_chars={":", " ", "\n", "\t", "\r"})

    def parse_gemma_string(self) -> str:
        marker = '<|"|>'
        self.pos += len(marker)
        end = self.text.find(marker, self.pos)
        if end < 0:
            raise ValueError("Unterminated Gemma4 string.")
        value = self.text[self.pos : end]
        self.pos = end + len(marker)
        return value

    def parse_quoted_string(self, close_follow: set[str]) -> str:
        quote = self.peek()
        if quote not in {'"', "'"}:
            raise ValueError(f"Expected quoted string at offset {self.pos}.")
        self.pos += 1
        chars = []
        escaped = False
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            self.pos += 1
            if escaped:
                chars.append(self._decode_escape(ch))
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                next_pos = self._skip_ws_pos(self.pos)
                if next_pos >= len(self.text) or self.text[next_pos] in close_follow:
                    return "".join(chars)
                chars.append(ch)
                continue
            chars.append(ch)
        raise ValueError("Unterminated quoted string.")

    def _skip_ws_pos(self, pos: int) -> int:
        while pos < len(self.text) and self.text[pos].isspace():
            pos += 1
        return pos

    def _decode_escape(self, ch: str) -> str:
        if ch == "u":
            return self._decode_unicode_escape()
        return {
            '"': '"',
            "'": "'",
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }.get(ch, ch)

    def _decode_unicode_escape(self) -> str:
        if self.pos + 4 > len(self.text):
            return "u"
        digits = self.text[self.pos : self.pos + 4]
        if not all(ch in "0123456789abcdefABCDEF" for ch in digits):
            return "u"
        self.pos += 4
        codepoint = int(digits, 16)
        if 0xD800 <= codepoint <= 0xDBFF and self.text.startswith("\\u", self.pos) and self.pos + 6 <= len(self.text):
            low_digits = self.text[self.pos + 2 : self.pos + 6]
            if all(ch in "0123456789abcdefABCDEF" for ch in low_digits):
                low = int(low_digits, 16)
                if 0xDC00 <= low <= 0xDFFF:
                    self.pos += 6
                    codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
        try:
            return chr(codepoint)
        except ValueError:
            return "\\u" + digits

    def parse_token(self, stop_chars: set[str] | None = None) -> str:
        stop_chars = stop_chars or {",", "}", "]", " ", "\n", "\t", "\r"}
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in stop_chars:
            self.pos += 1
        if self.pos == start:
            raise ValueError(f"Expected token at offset {self.pos}.")
        return self.text[start : self.pos]

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def expect(self, char: str) -> None:
        self.skip_ws()
        if self.peek() != char:
            raise ValueError(f"Expected {char!r} at offset {self.pos}.")
        self.pos += 1


def resolve_tool_model_name(model_name: str | None) -> str | None:
    """Prefer the explicitly configured fused model series over path inference."""
    model_series = os.environ.get("FUSED_MODEL_SERIES")
    if model_series is None:
        return model_name

    normalized = model_series.strip().lower().replace("-", "_")
    aliases = {
        "qwen3": "qwen3",
        "qwen3.5": "qwen3.5",
        "qwen3_5": "qwen3.5",
        "gemma4": "gemma4",
        "gemma_4": "gemma4",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported FUSED_MODEL_SERIES: {model_series!r}")
    return aliases[normalized]


def make_tool_parser(model_name: str | None, valid_tools: set[str] | None = None) -> QwenToolParser:
    """Select and build the tool parser for a model path/name.

    Qwen3.5/Qwen3-Coder use the XML function/parameter format, Gemma4 uses its
    native ``<|tool_call>call:...`` template, and everything else keeps the JSON
    ``<tool_call>`` format.
    """
    name = (resolve_tool_model_name(model_name) or "").lower()
    if "gemma4" in name or "gemma-4" in name or "gemma_4" in name:
        parser_class = Gemma4ToolParser
    elif any(x in name for x in ("qwen3.5", "qwen3-coder", "qwen3coder")):
        parser_class = Qwen3CoderToolParser
    else:
        parser_class = QwenToolParser
    return parser_class(valid_tools=valid_tools)


@lru_cache(maxsize=None)
def _cached_finish_parser(model_name: str | None, valid_tools: frozenset[str]) -> QwenToolParser:
    return make_tool_parser(model_name, valid_tools=set(valid_tools))


class ActiveFinishParser:
    """Model-aware ``finish`` parser for the reward path.

    ``reward_func`` runs several call-frames above the extraction helpers and a
    training run serves a single policy model, so rather than thread the model
    name through every signature, callers ``set()`` the active model once per
    reward batch and the deep helpers read the cached parser via ``get()``.
    """

    def __init__(self, valid_tools: set[str]):
        self._valid_tools = frozenset(valid_tools)
        self._model_name: str | None = None

    def set(self, model_name: str | None) -> None:
        self._model_name = resolve_tool_model_name(model_name)

    def get(self) -> QwenToolParser:
        return _cached_finish_parser(self._model_name, self._valid_tools)
