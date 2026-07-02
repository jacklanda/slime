from __future__ import annotations

import ast
import json
import logging
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
        payloads = [(match.group(1), match.start(), match.end()) for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)]
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
                return int(param_value)
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


def make_tool_parser(model_name: str | None, valid_tools: set[str] | None = None) -> QwenToolParser:
    """Select and build the tool parser for a model path/name.

    Mirrors rllm's ``ToolParser.get_parser`` detection, narrowed to the two
    families slime serves: Qwen3.5/Qwen3-Coder use the XML function/parameter
    format; everything else (Qwen3) keeps the JSON ``<tool_call>`` format.
    """
    name = (model_name or "").lower()
    parser_class = Qwen3CoderToolParser if any(x in name for x in ("qwen3.5", "qwen3-coder", "qwen3coder")) else QwenToolParser
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
        self._model_name = model_name

    def get(self) -> QwenToolParser:
        return _cached_finish_parser(self._model_name, self._valid_tools)
