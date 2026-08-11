from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)
_MISSING_PARAMETER = object()
_INVALID_PARAMETER = object()
_NO_SCHEMA_COERCION = object()
# A typed parameter the model opened but left empty (``<parameter=n></parameter>``).
# Distinct from _INVALID_PARAMETER: the value is absent rather than malformed, so
# the argument is dropped without recording a schema error.
_OMITTED_PARAMETER = object()


def _normalize_xml_parameter_name(name: str) -> str:
    """Normalize Qwen's occasionally quoted XML attribute names."""
    # XML syntax is ``parameter=name``; generation sometimes copies JSON and
    # emits ``parameter="name"`` (or leaves only the opening quote).  Strip
    # quote/punctuation noise before schema lookup.
    return name.strip().strip("\"'`").strip().rstrip(":").strip()


def _repair_xml_scalar_value(value: str) -> str:
    """Remove a leaked XML-like closing fragment from scalar values.

    Qwen3.5 occasionally emits ``10</result>`` or ``True</ Personnel>`` when
    closing a parameter block.  The fragment is outside the scalar token and
    must not change the typed argument.
    """
    value = value.strip()
    marker = value.find("</")
    if marker > 0:
        value = value[:marker].rstrip()
    return value.strip().strip("\"'`")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class _Gemma4NativeCall:
    raw_name: str
    raw_args: str
    start: int
    end: int
    name_span: tuple[int, int]
    args_start: int
    repairs: tuple[str, ...] = ()


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
                    # Do not silently turn malformed arguments into an empty
                    # call.  That hides protocol failures and can execute a
                    # different tool request than the model emitted.
                    continue
            if not isinstance(args, dict):
                continue
            calls.append(ToolCall(name=name, arguments=args, start=start, end=end))
        if not calls:
            calls.extend(self._extract_answer_fallback(model_response or ""))
        return calls

    def _extract_payloads(self, text: str) -> list[tuple[str | tuple[str, str], int, int]]:
        idx = text.rfind("</think>")
        offset = idx + len("</think>") if idx >= 0 else 0
        search_text = text[offset:]
        payloads = []
        cursor = 0
        begin = "<tool_call>"
        end_marker = "</tool_call>"
        while True:
            start = search_text.find(begin, cursor)
            if start < 0:
                break
            end = search_text.find(end_marker, start + len(begin))
            if end < 0:
                # An incomplete call is not a valid call. Do not parse JSON
                # from it or accidentally consume a later response fragment.
                break
            payloads.append(
                (
                    search_text[start + len(begin) : end].strip(),
                    offset + start,
                    offset + end + len(end_marker),
                )
            )
            cursor = end + len(end_marker)
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
                        candidate = re.sub(r",\s*([}\]])", r"\1", text[start : i + 1])
                        try:
                            return json.loads(candidate)
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
        self.last_schema_errors: list[str] = []

    def parse(self, model_response: str) -> list[ToolCall]:
        text = model_response or ""
        self.last_schema_errors = []
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
        if not calls and self.tool_call_prefix in text:
            self.last_schema_errors.append("tool function block is missing the required <tool_call> wrapper")
            return []
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
        return [
            (
                match.group(1) if match.group(1) is not None else (match.group(2) or ""),
                match.start(),
                match.end(),
            )
            for match in self._TOOL_CALL_RE.finditer(text)
        ]

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
                # Models occasionally emit optional-looking fields that are not
                # part of the advertised schema.  Do not turn these into
                # executable arguments (or flood rollout logs); the caller can
                # still execute the well-formed subset of the tool call.
                logger.debug(
                    "Ignoring parameter '%s' not defined in the tool parameters for tool '%s'.",
                    param_name,
                    func_name,
                )
                return _MISSING_PARAMETER
            # Without a schema (for example a standalone parser used by a
            # verifier), preserve the historical permissive behavior.
            return param_value

        param_schema = param_config[param_name]
        param_type = str(param_schema.get("type", "string")).strip().lower()

        if param_type not in {"object", "array"} and not param_type.startswith(("dict", "list")):
            param_value = _repair_xml_scalar_value(param_value)

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value

        # An empty typed block (``<parameter=step_number></parameter>``) carries no
        # value to coerce.  Treat it as an omitted argument rather than a malformed
        # one: the tool then raises a missing-argument error the model can read and
        # retry, instead of the schema error terminating the whole trajectory.
        if not param_value:
            logger.debug(
                "Parameter '%s' of tool '%s' was left empty; omitting the argument.",
                param_name,
                func_name,
            )
            return _OMITTED_PARAMETER
        if param_type.startswith("int") or param_type.startswith("uint") or param_type.startswith("long") or param_type.startswith("short") or param_type.startswith("unsigned"):
            try:
                # Qwen3.5 occasionally emits ``<parameter=name>: 10``. Treat the
                # extra colon as a formatting delimiter for integer parameters.
                return int(re.sub(r"^:\s*", "", param_value))
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not an integer in tool '%s'; rejecting tool call.",
                    param_value,
                    param_name,
                    func_name,
                )
                return _INVALID_PARAMETER
        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_value = float(param_value)
                return float_value if float_value - int(float_value) != 0 else int(float_value)
            except Exception:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a float in tool '%s'; rejecting tool call.",
                    param_value,
                    param_name,
                    func_name,
                )
                return _INVALID_PARAMETER
        if param_type in ["boolean", "bool", "binary"]:
            bool_value = param_value.lower()
            if bool_value not in ["true", "false"]:
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a boolean in tool '%s'; rejecting tool call.",
                    param_value,
                    param_name,
                    func_name,
                )
                return _INVALID_PARAMETER
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
                "Parsed value '%s' of parameter '%s' cannot be converted via ast.literal_eval() in tool '%s'; rejecting tool call.",
                param_value,
                param_name,
                func_name,
            )
            return _INVALID_PARAMETER

    def _parse_xml_function_call(self, function_name: str, parameters: str) -> dict[str, Any]:
        param_config = self._tool_parameter_config.get(function_name, {})
        arguments: dict[str, Any] = {}
        for raw_param_name, param_value in self._PARAMETER_RE.findall(parameters):
            param_name = _normalize_xml_parameter_name(raw_param_name)
            if not param_name:
                continue
            if param_name != raw_param_name.strip():
                self.last_schema_errors.append(f"malformed parameter name {raw_param_name.strip()!r}")
            param_value = self._strip_outer_newline(str(param_value))
            value = self._convert_param_value(param_value, param_name, param_config, function_name)
            if value is _OMITTED_PARAMETER:
                # Empty typed block: drop the argument and let the tool report the
                # missing parameter as a recoverable observation.
                continue
            if value is not _MISSING_PARAMETER:
                if value is not _INVALID_PARAMETER:
                    arguments[param_name] = value
                else:
                    self.last_schema_errors.append(f"invalid value for parameter {param_name!r} of tool {function_name!r}")
            else:
                self.last_schema_errors.append(f"unknown parameter {param_name!r} for tool {function_name!r}")
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
    _CALL_BEGIN_MARKERS = (tool_call_begin, "<|channel>")

    def __init__(self, valid_tools: set[str] | None = None):
        super().__init__(valid_tools=valid_tools)
        self.last_schema_errors: list[str] = []
        self.last_schema_error_spans: list[tuple[int, int] | None] = []
        self.last_schema_error_kinds: list[str] = []
        self.last_syntax_repairs: list[str] = []
        self.last_schema_coercions: list[str] = []
        self._tool_parameter_schemas: dict[str, dict[str, Any]] = {}

    def _normalize_name(self, name: str) -> str:
        name = name.strip()
        if self.valid_tools is None or name in self.valid_tools:
            return name
        return ""

    def get_tool_prompt(self, tools_schema: str) -> str:
        declarations = []
        self._tool_parameter_schemas = {}
        for schema in Qwen3CoderToolParser._iter_tool_schemas(tools_schema):
            function = schema.get("function", schema)
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                parameters = function.get("parameters")
                if isinstance(parameters, dict):
                    self._tool_parameter_schemas[function["name"]] = parameters
            declaration = self._format_function_declaration(schema)
            if declaration:
                declarations.append(f"<|tool>{declaration}<tool|>")
        if not declarations:
            return ""
        return "\n" + "".join(declarations) + "\n" + self.get_tool_contract()

    def get_tool_contract(self) -> str:
        tool_names = sorted(self._tool_parameter_schemas)
        if not tool_names:
            return ""
        structured_parameters = sorted(
            f"{tool_name}.{parameter_name}"
            for tool_name, parameters in self._tool_parameter_schemas.items()
            for parameter_name, schema in (parameters.get("properties") or {}).items()
            if isinstance(schema, dict)
            and self._schema_value_types(schema) & {"object", "array"}
        )
        contract = [
            "Gemma4 native tool-call contract:",
            "- Emit exactly one native call per assistant response:",
            '  <|tool_call>call:TOOL_NAME{ARGUMENT_NAME:<|"|>STRING_VALUE<|"|>}<tool_call|><|tool_response>',
            "- TOOL_NAME must exactly match one of: " + ", ".join(tool_names),
            "- Copy tool and parameter names verbatim. Never add prefixes, remove words, translate names, or escape underscores.",
            '- Use <|"|> only around STRING values. Emit OBJECT values as {...} and ARRAY values as [...] without quoting or escaping the structured value.',
            "- Balance every brace, bracket, string delimiter, and tool-call marker.",
            "- Never emit <|channel>call:, finish.result:, XML tool syntax, bare JSON, or plain final-answer prose.",
        ]
        if "finish" in tool_names:
            finish_parameters = self._tool_parameter_schemas["finish"]
            finish_result_schema = (finish_parameters.get("properties") or {}).get("result") or {}
            finish_result_types = self._schema_value_types(finish_result_schema)
            if "object" in finish_result_types:
                result_example = "{...}"
            elif "array" in finish_result_types:
                result_example = "[...]"
            else:
                result_example = '<|"|>CONCISE_FINAL_ANSWER<|"|>'
            contract.append(
                "- To finish, emit exactly: "
                f'<|tool_call>call:finish{{command:<|"|>submit<|"|>,result:{result_example}}}<tool_call|>'
            )
        if structured_parameters:
            contract.append("- These parameters require unquoted structured values: " + ", ".join(structured_parameters))
        return "\n".join(contract)

    @classmethod
    def _schema_value_types(cls, schema: dict[str, Any]) -> set[str]:
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            types = {schema_type.lower()}
        elif isinstance(schema_type, (list, tuple)):
            types = {str(item).lower() for item in schema_type}
        else:
            types = set()
        for key in ("anyOf", "oneOf"):
            variants = schema.get(key)
            if isinstance(variants, list):
                for variant in variants:
                    if isinstance(variant, dict):
                        types.update(cls._schema_value_types(variant))
        return types

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
            "tool_calls": [{"function": {"name": action.name, "arguments": action.arguments or {}}} for action in actions],
            "tool_responses": tool_responses,
        }

    def parse(self, model_response: str) -> list[ToolCall]:
        text = model_response or ""
        self.last_schema_errors = []
        self.last_schema_error_spans = []
        self.last_schema_error_kinds = []
        self.last_syntax_repairs = []
        self.last_schema_coercions = []
        calls: list[ToolCall] = []
        for native_call in self._iter_native_tool_calls(text):
            name = self._normalize_name(native_call.raw_name)
            if not name:
                self.last_schema_errors.append(f"unknown tool name {native_call.raw_name!r}")
                self.last_schema_error_spans.append(native_call.name_span)
                self.last_schema_error_kinds.append("unknown_tool")
                continue
            try:
                # A complete web_search object makes a terminal query boundary
                # unambiguous even when generation omits only its closing native
                # quote marker. Keep mutating finish calls strict.
                args, field_spans = self._parse_object(
                    native_call.raw_args,
                    parameter_schema=self._tool_parameter_schemas.get(name),
                    intrinsic_string_keys={"query"} if name == "web_search" else None,
                    terminal_native_string_keys={"query"} if name == "web_search" else None,
                )
            except ValueError as exc:
                self.last_schema_errors.append(f"invalid arguments for tool {name!r}: {exc}")
                self.last_schema_error_spans.append(self._syntax_error_span(native_call, exc))
                self.last_schema_error_kinds.append("invalid_syntax")
                logger.warning(
                    "Failed to parse Gemma4 tool-call arguments for tool '%s': %s. raw_args=%r",
                    native_call.raw_name,
                    exc,
                    native_call.raw_args[:512],
                )
                logger.debug("Gemma4 tool-call argument parse failure.", exc_info=True)
                continue
            self.last_syntax_repairs.extend(native_call.repairs)
            # Some generated MCP schemas historically declared structured
            # submit results as strings. Accept a JSON-encoded object/array so
            # old assets and newly generated native calls share one contract.
            if name.startswith("submit_result_") and isinstance(args.get("result"), str):
                try:
                    decoded_result = json.loads(args["result"])
                except (TypeError, json.JSONDecodeError):
                    pass
                else:
                    if isinstance(decoded_result, (dict, list)):
                        args["result"] = decoded_result
            args = self._coerce_arguments(name, args)
            schema_errors = self._validate_arguments(
                name,
                args,
                field_spans=field_spans,
                args_start=native_call.args_start,
            )
            if schema_errors:
                for error, span, kind in schema_errors:
                    self.last_schema_errors.append(error)
                    self.last_schema_error_spans.append(span)
                    self.last_schema_error_kinds.append(kind)
                if any(kind != "unknown_parameter" for _, _, kind in schema_errors):
                    continue
                # Unknown optional fields should not discard an otherwise
                # executable call. Keep the diagnostics/spans for credit
                # assignment, but pass only declared arguments to the tool.
                properties = (self._tool_parameter_schemas.get(name) or {}).get("properties") or {}
                args = {key: value for key, value in args.items() if key in properties}
            calls.append(
                ToolCall(
                    name=name,
                    arguments=args if isinstance(args, dict) else {},
                    start=native_call.start,
                    end=native_call.end,
                )
            )
        return calls

    @staticmethod
    def _syntax_error_span(native_call: _Gemma4NativeCall, exc: ValueError) -> tuple[int, int] | None:
        match = re.search(r"\boffset (\d+)\b", str(exc))
        if match is None or not native_call.raw_args:
            return None
        offset = min(int(match.group(1)), len(native_call.raw_args) - 1)
        position = native_call.args_start + offset
        return position, position + 1

    def _coerce_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        parameter_schema = self._tool_parameter_schemas.get(tool_name)
        if not isinstance(parameter_schema, dict):
            return arguments
        properties = parameter_schema.get("properties") or {}
        if not isinstance(properties, dict):
            return arguments
        required = set(parameter_schema.get("required") or ())
        coerced_arguments = dict(arguments)
        for name, value in arguments.items():
            schema = properties.get(name)
            if not isinstance(schema, dict) or self._value_matches_schema(value, schema):
                continue
            if value is None and name not in required:
                coerced_arguments.pop(name, None)
                self.last_schema_coercions.append(f"{tool_name}.{name}: dropped optional null")
                continue
            coerced = self._coerce_value_to_schema(value, schema)
            if coerced is _NO_SCHEMA_COERCION:
                continue
            coerced_arguments[name] = coerced
            self.last_schema_coercions.append(
                f"{tool_name}.{name}: {type(value).__name__} -> {type(coerced).__name__}"
            )
        return coerced_arguments

    @classmethod
    def _coerce_value_to_schema(cls, value: Any, schema: dict[str, Any]) -> Any:
        variants = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(variants, list):
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                coerced = cls._coerce_value_to_schema(value, variant)
                if coerced is not _NO_SCHEMA_COERCION and cls._value_matches_schema(coerced, variant):
                    return coerced
            return _NO_SCHEMA_COERCION
        expected = schema.get("type")
        if isinstance(expected, list):
            for item in expected:
                variant = {**schema, "type": item}
                coerced = cls._coerce_value_to_schema(value, variant)
                if coerced is not _NO_SCHEMA_COERCION and cls._value_matches_schema(coerced, variant):
                    return coerced
            return _NO_SCHEMA_COERCION
        if not isinstance(value, str) or expected is None:
            return _NO_SCHEMA_COERCION
        expected = str(expected).lower()
        stripped = value.strip()
        if expected in {"array", "object"}:
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                return _NO_SCHEMA_COERCION
            return decoded if cls._value_matches_schema(decoded, schema) else _NO_SCHEMA_COERCION
        if expected == "integer" and re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
        if expected == "number" and re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            stripped,
        ):
            number = float(stripped) if any(char in stripped.lower() for char in (".", "e")) else int(stripped)
            return number if math.isfinite(float(number)) else _NO_SCHEMA_COERCION
        if expected == "boolean" and stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        if expected == "null" and stripped.lower() in {"null", "none"}:
            return None
        return _NO_SCHEMA_COERCION

    def _validate_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        field_spans: dict[str, tuple[int, int, int, int]],
        args_start: int,
    ) -> list[tuple[str, tuple[int, int] | None, str]]:
        parameter_schema = self._tool_parameter_schemas.get(tool_name)
        if parameter_schema is None:
            return []
        properties = parameter_schema.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        errors: list[tuple[str, tuple[int, int] | None, str]] = [
            (f"missing required parameter {name!r} for tool {tool_name!r}", None, "missing_parameter")
            for name in parameter_schema.get("required") or []
            if name not in arguments
        ]
        for name, value in arguments.items():
            schema = properties.get(name)
            if not isinstance(schema, dict):
                field_span = field_spans.get(name)
                key_span = None if field_span is None else (args_start + field_span[0], args_start + field_span[1])
                errors.append((f"unknown parameter {name!r} for tool {tool_name!r}", key_span, "unknown_parameter"))
                continue
            if tool_name.startswith("submit_result_") and name == "result" and isinstance(value, (dict, list)):
                # Older MCP assets exposed ``result`` as string even though
                # the callable requires a structured value.
                continue
            if not self._value_matches_schema(value, schema):
                field_span = field_spans.get(name)
                value_span = None if field_span is None else (args_start + field_span[2], args_start + field_span[3])
                errors.append(
                    (
                        f"invalid type for parameter {name!r} of tool {tool_name!r}: "
                        f"expected {schema.get('type')!r}, got {type(value).__name__}",
                        value_span,
                        "invalid_parameter_type",
                    )
                )
        return errors

    @classmethod
    def _value_matches_schema(cls, value: Any, schema: dict[str, Any]) -> bool:
        variants = schema.get("anyOf") or schema.get("oneOf")
        if isinstance(variants, list):
            return any(isinstance(variant, dict) and cls._value_matches_schema(value, variant) for variant in variants)
        expected = schema.get("type")
        if isinstance(expected, list):
            return any(cls._value_matches_schema(value, {**schema, "type": item}) for item in expected)
        if expected is None:
            return True
        expected = str(expected).lower()
        checks = {
            "array": lambda: isinstance(value, list),
            "boolean": lambda: isinstance(value, bool),
            "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "null": lambda: value is None,
            "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
            "object": lambda: isinstance(value, dict),
            "string": lambda: isinstance(value, str),
        }
        return checks.get(expected, lambda: True)()

    @classmethod
    def _iter_native_tool_calls(cls, text: str) -> list[_Gemma4NativeCall]:
        calls: list[_Gemma4NativeCall] = []
        search_pos = 0
        while True:
            marker_matches = [
                (start, marker)
                for marker in cls._CALL_BEGIN_MARKERS
                if (start := text.find(marker, search_pos)) >= 0
            ]
            if not marker_matches:
                return calls
            start, begin_marker = min(marker_matches)

            pos = start + len(begin_marker)
            prefix = cls._CALL_PREFIX_RE.match(text, pos)
            if prefix is None:
                search_pos = pos
                continue

            raw_name = prefix.group(1)
            obj_start = prefix.end()
            name_span = (prefix.start(1), prefix.end(1))
            try:
                obj_end = cls._find_argument_object_end(text, obj_start)
            except ValueError:
                end_marker_start = text.find(cls.tool_call_end, obj_start)
                if end_marker_start >= 0:
                    end = end_marker_start + len(cls.tool_call_end)
                    raw_args = text[obj_start:end_marker_start]
                    repairs: tuple[str, ...] = ()
                    if cls._syntax_repair_allowed(raw_name):
                        repaired = cls._repair_incomplete_native_arguments(raw_args)
                        if repaired is not None:
                            raw_args, repairs = repaired
                    calls.append(
                        _Gemma4NativeCall(
                            raw_name=raw_name,
                            raw_args=raw_args,
                            start=start,
                            end=end,
                            name_span=name_span,
                            args_start=obj_start,
                            repairs=repairs,
                        )
                    )
                    search_pos = end
                    continue
                search_pos = pos
                continue

            end_marker_start = cls._skip_ws(text, obj_end)
            # Gemma4 occasionally closes the argument object correctly and then
            # leaks one or more array/parenthesis closers before the call marker.
            # The object boundary is already unambiguous, so ignore only this
            # narrow class of wrapper noise and keep argument parsing/schema
            # validation strict.
            while end_marker_start < len(text) and text[end_marker_start] in "])":
                end_marker_start = cls._skip_ws(text, end_marker_start + 1)
            if not text.startswith(cls.tool_call_end, end_marker_start):
                if cls._syntax_repair_allowed(raw_name) and cls._is_missing_call_end_boundary(
                    text, end_marker_start
                ):
                    calls.append(
                        _Gemma4NativeCall(
                            raw_name=raw_name,
                            raw_args=text[obj_start:obj_end],
                            start=start,
                            end=obj_end,
                            name_span=name_span,
                            args_start=obj_start,
                            repairs=("missing_tool_call_end",),
                        )
                    )
                    search_pos = max(obj_end, end_marker_start)
                    continue
                search_pos = obj_end
                continue

            end = end_marker_start + len(cls.tool_call_end)
            calls.append(
                _Gemma4NativeCall(
                    raw_name=raw_name,
                    raw_args=text[obj_start:obj_end],
                    start=start,
                    end=end,
                    name_span=name_span,
                    args_start=obj_start,
                )
            )
            search_pos = end

    @staticmethod
    def _syntax_repair_allowed(tool_name: str) -> bool:
        return tool_name not in {"finish", "submit"} and not tool_name.startswith("submit_result_")

    @classmethod
    def _repair_incomplete_native_arguments(cls, raw_args: str) -> tuple[str, tuple[str, ...]] | None:
        if not raw_args.lstrip().startswith("{"):
            return None
        repaired = raw_args.rstrip()
        repairs: list[str] = []
        marker = '<|"|>'
        if repaired.count(marker) % 2:
            marker_start = repaired.rfind(marker)
            value_start = marker_start + len(marker)
            value = repaired[value_start:]
            if '"""' in value or "'''" in value:
                return None
            if repaired.endswith("}") and not any(char in value[:-1] for char in "{}[]"):
                repaired = repaired[:-1] + marker + "}"
            elif not any(char in value for char in "{}[]"):
                repaired += marker
            else:
                return None
            repairs.append("missing_native_string_end")
        elif match := re.search(
            r'(?:^|[,{])\s*[A-Za-z_][A-Za-z0-9_.-]*\s*[:=]\s*"([^"{}\[\]]*)(})?$',
            repaired,
        ):
            value_end = match.end(1)
            repaired = repaired[:value_end] + '"' + repaired[value_end:]
            repairs.append("missing_quoted_string_end")

        missing_closers = cls._missing_argument_closers(repaired)
        if missing_closers is None or len(missing_closers) > 1:
            return None
        if missing_closers:
            if missing_closers != "}":
                return None
            repaired += missing_closers
            repairs.append("missing_argument_object_end")
        return (repaired, tuple(repairs)) if repairs else None

    @classmethod
    def _missing_argument_closers(cls, text: str) -> str | None:
        marker = '<|"|>'
        stack: list[str] = []
        pos = 0
        while pos < len(text):
            if text.startswith(marker, pos):
                end = text.find(marker, pos + len(marker))
                if end < 0:
                    return None
                pos = end + len(marker)
                continue
            ch = text[pos]
            if ch in {'"', "'"}:
                try:
                    pos = cls._find_quoted_string_end(text, pos, close_follow={":", ",", "}", "]"})
                except ValueError:
                    return None
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if not stack or stack.pop() != ch:
                    return None
            pos += 1
        return "".join(reversed(stack))

    @classmethod
    def _is_missing_call_end_boundary(cls, text: str, pos: int) -> bool:
        suffix = text[pos:]
        return not suffix or suffix.startswith(cls.tool_output_begin) or suffix.startswith("<eos>")

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
    def _parse_object(
        cls,
        text: str,
        *,
        parameter_schema: dict[str, Any] | None = None,
        intrinsic_string_keys: set[str] | None = None,
        terminal_native_string_keys: set[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, tuple[int, int, int, int]]]:
        properties = (parameter_schema or {}).get("properties") or {}
        string_keys = {
            str(name)
            for name, schema in properties.items()
            if isinstance(schema, dict) and cls._value_matches_schema("", schema)
        }
        string_keys.update(intrinsic_string_keys or ())
        parser = _Gemma4ArgumentParser(
            text,
            top_level_string_keys=string_keys,
            top_level_keys=set(map(str, properties)) | string_keys,
            terminal_native_string_keys=terminal_native_string_keys,
        )
        value = parser.parse_value()
        parser.skip_ws()
        if parser.pos != len(parser.text):
            raise ValueError(f"Unexpected trailing Gemma4 argument text at offset {parser.pos}.")
        if not isinstance(value, dict):
            raise ValueError("Gemma4 tool-call arguments must be an object.")
        return value, parser.top_level_field_spans


class _Gemma4ArgumentParser:
    def __init__(
        self,
        text: str,
        *,
        top_level_string_keys: set[str] | None = None,
        top_level_keys: set[str] | None = None,
        terminal_native_string_keys: set[str] | None = None,
    ):
        self.text = text
        self.pos = 0
        self.top_level_string_keys = top_level_string_keys or set()
        self.top_level_keys = top_level_keys or set(self.top_level_string_keys)
        self.terminal_native_string_keys = terminal_native_string_keys or set()
        self.object_depth = 0
        self.top_level_field_spans: dict[str, tuple[int, int, int, int]] = {}

    def parse_value(self, *, schema_string: bool = False, terminal_native_string: bool = False) -> Any:
        self.skip_ws()
        if self.text.startswith('<|"|>', self.pos):
            return self.parse_gemma_string(allow_terminal_object_close=terminal_native_string)
        ch = self.peek()
        if schema_string and ch in {'"', "'"}:
            return self.parse_schema_string(allow_terminal_object_close=terminal_native_string)
        if ch in {'"', "'"}:
            return self.parse_quoted_string(close_follow={",", "}", "]"})
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            return self.parse_array()
        if schema_string:
            return self.parse_schema_bare_string()
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
        self.object_depth += 1
        result: dict[str, Any] = {}
        self.skip_ws()
        if self.peek() == "}":
            self.pos += 1
            self.object_depth -= 1
            return result
        while True:
            self.skip_ws()
            key_start = self.pos
            key = self.parse_key()
            key_end = self.pos
            self.skip_ws()
            schema_string = self.object_depth == 1 and key in self.top_level_string_keys
            terminal_native_string = self.object_depth == 1 and key in self.terminal_native_string_keys
            if self.peek() in {":", "="}:
                self.pos += 1
            elif not (
                schema_string
                and (self.text.startswith('<|"|>', self.pos) or self.peek() in {'"', "'"})
            ):
                raise ValueError(f"Expected ':' at offset {self.pos}.")
            value_start = self._skip_ws_pos(self.pos)
            result[key] = self.parse_value(
                schema_string=schema_string,
                terminal_native_string=terminal_native_string,
            )
            if self.object_depth == 1:
                self.top_level_field_spans[key] = (key_start, key_end, value_start, self.pos)
            self.skip_ws()
            ch = self.peek()
            if ch == ",":
                self.pos += 1
                continue
            if ch == "}":
                self.pos += 1
                self.object_depth -= 1
                return result
            if schema_string and ch == "]":
                next_pos = self._skip_ws_pos(self.pos + 1)
                if next_pos < len(self.text) and self.text[next_pos] == "}" and not self.text[next_pos + 1 :].strip():
                    self.pos = next_pos + 1
                    self.object_depth -= 1
                    return result
            if self._declared_key_at(self.pos) is not None:
                continue
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
        return self.parse_token(stop_chars={":", "=", "<", '"', "'", " ", "\n", "\t", "\r"})

    def parse_schema_string(self, *, allow_terminal_object_close: bool = False) -> str:
        quote = self.peek()
        self.pos += 1
        start = self.pos
        marker = '<|"|>'
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "," and self._argument_key_at(self.pos + 1) is not None:
                break
            if ch.isspace() and self._declared_key_at(self.pos) is not None:
                break
            if ch == "}" and not self.text[self.pos + 1 :].strip():
                break
            self.pos += 1

        value = self.text[start : self.pos].rstrip()
        if value.endswith(quote):
            return value[: -len(quote)]
        if value.endswith(marker):
            return value[: -len(marker)]
        if allow_terminal_object_close and self.object_depth == 1 and value:
            # web_search.query is a read-only scalar and the enclosing object
            # close gives us an unambiguous terminal boundary.  Recover a
            # missing ordinary quote there just as we recover a missing native
            # <|"|> closer; mutating and structured tools never enable this.
            return value
        # Multiple quote characters inside the value prove that the model used
        # ordinary quotes as query punctuation and omitted only the outer close.
        # A plain one-sided quote remains an ambiguous truncation and is rejected.
        if quote in value:
            return value
        raise ValueError("Unterminated quoted Gemma4 string.")

    def parse_schema_bare_string(self) -> str:
        start = self.pos
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "," and self._declared_key_at(self.pos + 1) is not None:
                break
            if ch.isspace() and self._declared_key_at(self.pos) is not None:
                break
            if ch == "}" and not self.text[self.pos + 1 :].strip():
                break
            self.pos += 1
        value = self.text[start : self.pos].rstrip()
        if not value:
            raise ValueError(f"Expected string at offset {start}.")
        return value

    def _declared_key_at(self, pos: int) -> str | None:
        pos = self._skip_ws_pos(pos)
        for key in sorted(self.top_level_keys):
            if not self.text.startswith(key, pos):
                continue
            end = self._skip_ws_pos(pos + len(key))
            if end < len(self.text) and (
                self.text[end] in {":", "="} or self.text.startswith('<|"|>', end)
            ):
                return key
        return None

    def _argument_key_at(self, pos: int) -> str | None:
        """Return any syntactic argument key after an explicit comma."""
        pos = self._skip_ws_pos(pos)
        quoted_match = re.match(r"([\"'])([A-Za-z_][A-Za-z0-9_.-]*)\1\s*[:=]", self.text[pos:])
        if quoted_match:
            return quoted_match.group(2)
        match = re.match(r"[A-Za-z_][A-Za-z0-9_.-]*\s*[:=]", self.text[pos:])
        return match.group(0).rstrip(":=").strip() if match else None

    def parse_gemma_string(self, *, allow_terminal_object_close: bool = False) -> str:
        marker = '<|"|>'
        self.pos += len(marker)
        end = self.text.find(marker, self.pos)
        if end < 0:
            if allow_terminal_object_close and self.object_depth == 1:
                end = len(self.text) - 1
                value = self.text[self.pos : end]
                if (
                    self.text[end:] == "}"
                    and value
                    and not any(char in value for char in "{}[]")
                    and '"""' not in value
                    and "'''" not in value
                ):
                    self.pos = end
                    return value
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
