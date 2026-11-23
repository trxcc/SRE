"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from polar.trajectory.models import CompletionRecord, Trace


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    extracted: list[int] = []
    for item in value:
        try:
            extracted.append(int(item))
        except (TypeError, ValueError):
            return None
    return extracted


def _paired_tokens_from_logprobs_content(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            return None
        token_id = item.get("token_id")
        logprob = item.get("logprob")
        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _paired_tokens_from_sglang_meta(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    meta_info = choice.get("meta_info")
    if not isinstance(meta_info, dict):
        return None
    output_logprobs = meta_info.get("output_token_logprobs")
    if not isinstance(output_logprobs, list) or not output_logprobs:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in output_logprobs:
        token_id = None
        logprob = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            logprob = item[0]
            token_id = item[1]
        elif isinstance(item, dict):
            token_id = item.get("token_id")
            logprob = item.get("logprob", item.get("token_logprob"))

        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _extract_response_tokens(
    response: dict[str, Any],
    choice: dict[str, Any],
) -> tuple[list[int], list[float] | None]:
    """Extract response ids and logprobs without mixing unaligned sources."""
    token_ids = choice.get("token_ids", response.get("token_ids"))
    response_ids = _coerce_int_list(token_ids)
    paired_candidates = (
        _paired_tokens_from_logprobs_content(choice),
        _paired_tokens_from_sglang_meta(choice),
    )

    if response_ids is not None:
        for paired in paired_candidates:
            if paired is None:
                continue
            paired_ids, paired_logprobs = paired
            if paired_ids == response_ids:
                return response_ids, paired_logprobs
        return response_ids, None

    for paired in paired_candidates:
        if paired is not None:
            return paired

    return [], None


def _extract_prompt_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(tool) for tool in tools if isinstance(tool, dict)]
    return extracted or None


def build_trace_from_completion(completion: CompletionRecord) -> Trace:
    """Normalize one stored completion record into a trajectory trace."""

    request = completion.request if isinstance(completion.request, dict) else {}
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    prompt_ids = (
        first_choice.get("input_token_ids")
        or first_choice.get("prompt_token_ids")
        or response.get("prompt_token_ids")
    )
    response_message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    response_ids, response_logprobs = _extract_response_tokens(response, first_choice)

    return Trace(
        prompt_ids=list(prompt_ids) if isinstance(prompt_ids, list) else [],
        response_ids=response_ids,
        loss_mask=[1] * len(response_ids),
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=response_logprobs,
        metadata=deepcopy(completion.metadata),
    )


def summarize_tool_usage(traces: list[Trace]) -> dict[str, Any]:
    """Return structured tool-call statistics for one agent trajectory.

    Count only tool calls emitted in completion response messages.  Prompt
    messages replay all earlier calls on every subsequent request, so counting
    those would inflate a long agent session quadratically.
    """
    tool_names: Counter[str] = Counter()
    tool_call_count = 0
    named_tool_call_count = 0
    tool_call_turn_count = 0
    assistant_message_count = 0
    parallel_tool_call_turn_count = 0
    finish_reason_tool_call_count = 0
    max_parallel_tool_calls = 0
    available_tool_counts: list[int] = []

    for trace in traces:
        if trace.tools is not None:
            available_tool_counts.append(len(trace.tools))
        if str(trace.finish_reason or "").strip().lower() == "tool_calls":
            finish_reason_tool_call_count += 1
        for message in trace.response_messages:
            if str(message.get("role") or "assistant").lower() == "assistant":
                assistant_message_count += 1
            raw_calls = message.get("tool_calls")
            calls = (
                [call for call in raw_calls if isinstance(call, dict)]
                if isinstance(raw_calls, list)
                else []
            )
            if not calls:
                continue
            tool_call_turn_count += 1
            tool_call_count += len(calls)
            max_parallel_tool_calls = max(max_parallel_tool_calls, len(calls))
            if len(calls) > 1:
                parallel_tool_call_turn_count += 1
            for call in calls:
                function = call.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                if not isinstance(name, str) or not name.strip():
                    name = call.get("name")
                if isinstance(name, str) and name.strip():
                    tool_names[name.strip()] += 1
                    named_tool_call_count += 1

    summary: dict[str, Any] = {
        "tool_calls": tool_call_count,
        "named_tool_calls": named_tool_call_count,
        "unnamed_tool_calls": tool_call_count - named_tool_call_count,
        "tool_call_turns": tool_call_turn_count,
        "assistant_messages": assistant_message_count,
        "parallel_tool_call_turns": parallel_tool_call_turn_count,
        "finish_reason_tool_calls": finish_reason_tool_call_count,
        "max_parallel_tool_calls": max_parallel_tool_calls,
        "used_tools": tool_call_count > 0,
        "unique_tools": len(tool_names),
        "calls_by_name": dict(sorted(tool_names.items())),
    }
    if available_tool_counts:
        summary.update(
            {
                "available_tools_min": min(available_tool_counts),
                "available_tools_max": max(available_tool_counts),
                "available_tools_mean": sum(available_tool_counts)
                / len(available_tool_counts),
            }
        )
    return summary
