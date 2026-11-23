"""Model-output parsing helpers for agent harnesses."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any


logger = logging.getLogger(__name__)

_TOOL_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<tool_call>", "</tool_call>"),
    ("<|tool_call>", "<tool_call|>"),
)
_GEMMA4_RESPONSE_TERMINATORS: tuple[str, ...] = (
    "<turn|>",
    "<|tool_response>",
    "<eos>",
)


@dataclasses.dataclass(frozen=True)
class ParsedModelOutput:
    """Structured view of one decoded model output."""

    reasoning: str
    text: str
    tool_uses: list[dict[str, Any]]
    ill_formed: bool = False


def parse_model_output(
    raw_output: str,
    *,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
    reasoning_parser_name: str | None,
    reasoning_started_in_prompt: bool = False,
) -> ParsedModelOutput:
    """Parse raw model text into reasoning, visible text, and tool uses.

    The heavy format-specific work is delegated to SGLang's reasoning and
    function-call parsers. The XML fallback covers Anthropic-style tool-call
    text that some coding-agent models still emit occasionally.
    """
    reasoning, body_text = "", raw_output
    if reasoning_parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parser_kwargs: dict[str, Any] = {
            "model_type": reasoning_parser_name,
            "stream_reasoning": False,
        }
        # Gemma 4's canonical template opens ``<|channel>thought\n`` in the
        # prompt after a tool response.  The sampled ids then start with the
        # reasoning body and contain only the closing ``<channel|>``.  Tell
        # SGLang's parser about that prompt state; forcing every Gemma turn
        # would incorrectly classify a legitimate non-thinking first turn.
        if reasoning_parser_name.lower() == "gemma4" and reasoning_started_in_prompt:
            parser_kwargs["force_reasoning"] = True
        r, b = ReasoningParser(**parser_kwargs).parse_non_stream(raw_output)
        reasoning, body_text = r or "", b or ""
        if not reasoning and "</think>" in body_text:
            reasoning, body_text = body_text.split("</think>", 1)

    if reasoning_parser_name and reasoning_parser_name.lower() == "gemma4":
        body_text = _strip_gemma4_response_terminator(body_text)

    body_text, tool_uses, ill_formed = parse_tool_uses(body_text, tools_schema, tool_parser_name)
    return ParsedModelOutput(
        reasoning=reasoning,
        text=(body_text or "").strip(),
        tool_uses=tool_uses,
        ill_formed=ill_formed,
    )


def _strip_gemma4_response_terminator(text: str) -> str:
    """Drop only terminal Gemma response framing after channel parsing.

    The channel closer is deliberately not removed here: it is a semantic
    reasoning boundary and must be consumed by the reasoning parser first.
    """

    cleaned = text.rstrip()
    changed = True
    while changed:
        changed = False
        for marker in _GEMMA4_RESPONSE_TERMINATORS:
            if cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].rstrip()
                changed = True
                break
    return cleaned


def parse_tool_uses(
    body_text: str,
    tools_schema: list[dict] | None,
    tool_parser_name: str | None,
) -> tuple[str, list[dict[str, Any]], bool]:
    """Parse tool calls from body text and return visible text plus tool uses."""
    tool_uses: list[dict[str, Any]] = []
    ill_formed = False
    if tool_parser_name and tools_schema:
        from sglang.srt.entrypoints.openai.protocol import Function, Tool
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sg_tools = [Tool(type="function", function=Function(**d["function"])) for d in tools_schema]
        parser = FunctionCallParser(tools=sg_tools, tool_call_parser=tool_parser_name)
        calls = []
        if parser.has_tool_call(body_text):
            try:
                body_text, calls = parser.parse_non_stream(body_text)
            except Exception:
                logger.exception("[agent.parsing] sglang tool-call parsing failed; falling back")
        for c in calls:
            try:
                args, repaired = _loads_json_with_single_quote_escape_repair(c.parameters or "{}")
            except json.JSONDecodeError:
                args = {"_raw_arguments": c.parameters}
                ill_formed = True
            else:
                ill_formed = ill_formed or repaired
            tool_uses.append({"name": c.name or "tool", "input": args})

    if not tool_uses and tools_schema:
        json_tool_uses = parse_tagged_json_tool_uses(body_text, tools_schema)
        if json_tool_uses:
            return "", json_tool_uses, ill_formed
        body_text, tool_uses = parse_xml_tool_uses(body_text, tools_schema)

    return body_text, tool_uses, ill_formed


def parse_tagged_json_tool_uses(body_text: str, tools_schema: list[dict]) -> list[dict[str, Any]]:
    """Recover Qwen-style JSON tool calls wrapped in <tool_call> tags.

    ClawRL's legacy gateway recovered these when SGLang returned them as plain
    content instead of structured tool_calls. Keeping that behavior here makes
    the local /generate adapter match the old /v1/chat/completions path.
    """

    payloads = _iter_tagged_tool_payloads(body_text)
    if not payloads:
        return []
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    for payload in payloads:
        parsed = _parse_tagged_json_tool_payload(payload)
        if parsed is None:
            continue
        name, arguments = parsed
        if valid_tools and name not in valid_tools:
            continue
        tool_uses.append({"name": name, "input": arguments})
    return tool_uses


def _iter_tagged_tool_payloads(text: str) -> list[str]:
    stripped = str(text or "").strip()
    payloads: list[str] = []
    pos = 0
    while pos < len(stripped):
        matched_pair: tuple[str, str] | None = None
        matched_index: int | None = None
        for opener, closer in _TOOL_TAG_PAIRS:
            index = stripped.find(opener, pos)
            if index != -1 and (matched_index is None or index < matched_index):
                matched_pair = (opener, closer)
                matched_index = index
        if matched_pair is None or matched_index is None:
            break

        opener, closer = matched_pair
        payload_start = matched_index + len(opener)
        close_index = stripped.find(closer, payload_start)
        if close_index == -1:
            next_openers = [
                next_index
                for next_opener, _ in _TOOL_TAG_PAIRS
                if (next_index := stripped.find(next_opener, payload_start)) != -1
            ]
            payload_end = min(next_openers) if next_openers else len(stripped)
            payloads.append(stripped[payload_start:payload_end].strip())
            break

        payloads.append(stripped[payload_start:close_index].strip())
        pos = close_index + len(closer)
    return payloads


def _parse_tagged_json_tool_payload(payload: str) -> tuple[str, dict[str, Any]] | None:
    try:
        data, _repaired = _loads_json_with_single_quote_escape_repair(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    name = data.get("name") or data.get("action")
    arguments = data.get("arguments")
    function = data.get("function")
    if not isinstance(name, str) and isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", arguments)
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        try:
            arguments, _repaired_arguments = _loads_json_with_single_quote_escape_repair(arguments)
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(arguments, dict):
        return None
    return name.strip(), arguments


def _loads_json_with_single_quote_escape_repair(raw: str) -> tuple[Any, bool]:
    """Load JSON, repairing only invalid backslash-apostrophe escapes.

    Models sometimes embed Python source in a tool argument and emit a
    backslash-apostrophe sequence to preserve an apostrophe inside a Python
    string. JSON does not define that escape. On precisely that decoder error,
    make the backslash literal by doubling it and retry once. Other malformed
    JSON is left untouched so the caller can use its normal retry/recovery
    behavior.
    """

    try:
        return json.loads(raw), False
    except json.JSONDecodeError as original:
        if original.msg != "Invalid \\escape" or raw[original.pos : original.pos + 2] != "\\'":
            raise
        repaired = _escape_invalid_single_quote_escapes(raw)
        if repaired == raw:
            raise
        try:
            return json.loads(repaired), True
        except json.JSONDecodeError:
            raise original


def _escape_invalid_single_quote_escapes(raw: str) -> str:
    """Make odd backslash runs before apostrophes valid JSON escapes."""

    parts: list[str] = []
    index = 0
    while index < len(raw):
        if raw[index] != "\\":
            parts.append(raw[index])
            index += 1
            continue

        run_end = index
        while run_end < len(raw) and raw[run_end] == "\\":
            run_end += 1
        run_length = run_end - index
        if run_end < len(raw) and raw[run_end] == "'" and run_length % 2:
            run_length += 1
        parts.append("\\" * run_length)
        index = run_end
    return "".join(parts)


def parse_xml_tool_uses(body_text: str, tools_schema: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """Fallback parser for Anthropic-style XML tool calls."""
    valid_tools = {t.get("function", {}).get("name") for t in tools_schema}
    tool_uses: list[dict[str, Any]] = []
    cleaned_parts: list[str] = []
    last = 0
    for m in re.finditer(
        r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
        body_text,
        flags=re.DOTALL,
    ):
        name, inner = m.group(1), m.group(2)
        if name in valid_tools:
            args = {
                p.group(1): p.group(2).strip()
                for p in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", inner, flags=re.DOTALL)
            }
            tool_uses.append({"name": name, "input": args})
            cleaned_parts.append(body_text[last : m.start()])
            last = m.end()
    cleaned_parts.append(body_text[last:])
    return "".join(cleaned_parts), tool_uses
