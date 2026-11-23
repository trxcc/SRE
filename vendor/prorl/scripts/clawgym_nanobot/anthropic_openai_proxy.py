#!/usr/bin/env python3
"""Small Anthropic Messages API bridge for OpenAI-compatible local models.

The bridge is intentionally narrow: it implements the endpoints Claude Code
uses and preserves tool calls in both directions.  The upstream request is
non-streaming; Anthropic SSE events are emitted after the local model returns.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import os
import re
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


LOG = logging.getLogger("anthropic_openai_proxy")
app = FastAPI()
_UPSTREAM_COUNTER = itertools.count()
_UPSTREAM_LOCK = threading.RLock()
_UPSTREAM_IN_FLIGHT: dict[str, int] = {}
_ROUTE_UPSTREAM: dict[str, str] = {}
_ROUTE_REQUEST_COUNTS: dict[str, int] = {}
_MAX_ROUTE_ASSIGNMENTS = 16384
_METRICS_LOCK = threading.Lock()
_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length of\s+(?P<context>\d+)\s+tokens.*?"
    r"total of\s+(?P<total>\d+)\s+tokens:\s+"
    r"(?P<input>\d+)\s+tokens from the input messages and\s+"
    r"(?P<completion>\d+)\s+tokens for the completion",
    re.IGNORECASE | re.DOTALL,
)


def _text_blocks(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        str(block.get("text") or "")
        for block in value
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _thinking_blocks(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "\n".join(
        str(block.get("thinking") or "")
        for block in value
        if isinstance(block, dict) and block.get("type") == "thinking"
    ).strip()


def _bounded_history_text(value: str, env_name: str) -> str:
    """Bound retained Claude-Code history before forwarding to a smaller context model."""
    try:
        limit = int(os.environ.get(env_name, "0"))
    except ValueError:
        limit = 0
    if limit <= 0 or len(value) <= limit:
        return value
    marker = "\n...[earlier content truncated by WSClaw context guard]...\n"
    head = max(1, (limit - len(marker)) * 3 // 4)
    tail = max(1, limit - len(marker) - head)
    return value[:head] + marker + value[-tail:]


def _compact_forwarded_history(messages: list[dict[str, Any]]) -> None:
    """Keep recent tool evidence while bounding accumulated assistant/tool text."""
    try:
        budget = int(os.environ.get("WSCLAW_PROXY_HISTORY_MAX_CHARS", "0"))
    except ValueError:
        budget = 0
    if budget <= 0:
        return
    retained = 0
    marker = "[older history omitted by WSClaw context guard]"
    for message in reversed(messages):
        if message.get("role") not in {"assistant", "tool"}:
            continue
        for key in ("content", "reasoning_content"):
            value = message.get(key)
            if not isinstance(value, str) or not value:
                continue
            remaining = budget - retained
            if remaining <= len(marker):
                message[key] = marker
                continue
            if len(value) > remaining:
                message[key] = _bounded_history_text(
                    value,
                    "WSCLAW_PROXY_HISTORY_REMAINDER_MAX_CHARS",
                ) if os.environ.get("WSCLAW_PROXY_HISTORY_REMAINDER_MAX_CHARS") else (
                    value[: max(1, remaining - len(marker))] + marker
                )
                retained = budget
            else:
                retained += len(value)


def anthropic_messages_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Anthropic messages without flattening tool-use history."""
    messages: list[dict[str, Any]] = []
    system_parts = []
    system = _text_blocks(payload.get("system"))
    if system:
        system_parts.append(system)

    for raw in payload.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        content = raw.get("content")
        # Some Claude Code turns carry an additional system message in the
        # history. Qwen3.5/3.6 chat templates require the sole system turn to
        # be first, so fold every such instruction into one leading turn.
        if role == "system":
            text = _text_blocks(content)
            if text:
                system_parts.append(text)
            continue
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue

        text = _text_blocks(content)
        if role == "assistant":
            reasoning = _bounded_history_text(
                _thinking_blocks(content), "WSCLAW_PROXY_ASSISTANT_HISTORY_MAX_CHARS"
            )
            tool_calls = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": str(block.get("id") or f"tool_{uuid.uuid4().hex}"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                )
            text = _bounded_history_text(
                text, "WSCLAW_PROXY_ASSISTANT_HISTORY_MAX_CHARS"
            )
            message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if reasoning:
                message["reasoning_content"] = reasoning
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue

        if text:
            messages.append({"role": "user", "content": text})
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            result = block.get("content")
            if not isinstance(result, str):
                result = _text_blocks(result) or json.dumps(result, ensure_ascii=False)
            result = _bounded_history_text(
                result, "WSCLAW_PROXY_TOOL_RESULT_MAX_CHARS"
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": result,
                }
            )
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    _compact_forwarded_history(messages)
    return messages


def anthropic_tools_to_openai(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tools = []
    for raw in payload.get("tools") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(raw["name"]),
                    "description": str(raw.get("description") or ""),
                    "parameters": raw.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def openai_response_to_anthropic(
    response: dict[str, Any], *, requested_model: str
) -> dict[str, Any]:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        content.append(
            {
                "type": "thinking",
                "thinking": reasoning.strip(),
                "signature": "local-sglang-gemma4",
            }
        )
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": str(text)})
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": str(raw.get("id") or f"tool_{uuid.uuid4().hex}"),
                "name": str(function.get("name") or ""),
                "input": arguments,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})

    usage = response.get("usage") or {}
    has_tool = any(block["type"] == "tool_use" for block in content)
    finish_reason = choice.get("finish_reason")
    stop_reason = "tool_use" if has_tool else "max_tokens" if finish_reason == "length" else "end_turn"
    return {
        "id": str(response.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def anthropic_sse(message: dict[str, Any]):
    start = dict(message)
    start["content"] = []
    start["stop_reason"] = None
    start["usage"] = {
        "input_tokens": message["usage"]["input_tokens"],
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    yield _sse_event("message_start", {"type": "message_start", "message": start})
    for index, block in enumerate(message["content"]):
        if block["type"] == "thinking":
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            )
            if block.get("thinking"):
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": block["thinking"],
                        },
                    },
                )
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": block.get("signature") or "local-sglang-gemma4",
                    },
                },
            )
        elif block["type"] == "text":
            yield _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            )
            if block.get("text"):
                yield _sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}},
                )
        else:
            yield _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            yield _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                },
            )
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


def _normalize_upstream_bases(raw: str | list[str]) -> list[str]:
    values = raw.split(",") if isinstance(raw, str) else raw
    bases = []
    for item in values:
        base = str(item).strip().rstrip("/")
        if not base:
            continue
        parsed = urlsplit(base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RuntimeError(f"invalid WSClaw OpenAI API base: {base}")
        if not base.endswith("/v1"):
            base += "/v1"
        if base not in bases:
            bases.append(base)
    if not bases:
        raise RuntimeError("at least one WSClaw OpenAI API base is required")
    return bases


def _read_private_text(path: Path, *, label: str) -> str:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must be absolute: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            mode = stat.S_IMODE(os.fstat(handle.fileno()).st_mode)
            if mode != 0o600:
                raise RuntimeError(f"{label} file must have mode 0600: {path}")
            value = handle.read().strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read {label} file {path}: {exc}") from exc
    if not value:
        raise RuntimeError(f"{label} file is empty: {path}")
    return value


def _router_config_settings(path: Path) -> tuple[list[str], str]:
    raw_config = _read_private_text(path, label="router config")
    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid router config JSON {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"router config must contain a JSON object: {path}")

    raw_bases = config.get("api_bases")
    if raw_bases is None:
        raw_bases = config.get("api_base")
    if not isinstance(raw_bases, (str, list)):
        raise RuntimeError(f"router config api_base/api_bases is invalid: {path}")
    if isinstance(raw_bases, list) and not all(
        isinstance(item, str) for item in raw_bases
    ):
        raise RuntimeError(f"router config api_bases must contain strings: {path}")

    key_file = config.get("api_key_file")
    if not isinstance(key_file, str) or not key_file.strip():
        raise RuntimeError(f"router config api_key_file is missing: {path}")
    key = _read_private_text(Path(key_file), label="router API key")
    return _normalize_upstream_bases(raw_bases), key


def _upstream_bases() -> list[str]:
    raw = os.environ.get("WSCLAW_OPENAI_API_BASES") or os.environ.get(
        "WSCLAW_OPENAI_API_BASE", ""
    )
    return _normalize_upstream_bases(raw)


def _settings() -> tuple[list[str], str, str, float, int]:
    config_file = os.environ.get("WSCLAW_OPENAI_ROUTER_CONFIG_FILE", "").strip()
    if config_file:
        bases, key = _router_config_settings(Path(config_file))
    else:
        bases = _upstream_bases()
        key = os.environ.get("WSCLAW_OPENAI_API_KEY", "apiKey")
    return (
        bases,
        key,
        os.environ.get("WSCLAW_OPENAI_MODEL", "gemma-user-agent"),
        float(os.environ.get("WSCLAW_PROXY_TIMEOUT_SEC", "900")),
        int(os.environ.get("WSCLAW_PROXY_MAX_TOKENS", "8192")),
    )


def _request_routing_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if token.startswith("wsclaw-route-"):
            return token.removeprefix("wsclaw-route-")
    api_key = request.headers.get("x-api-key", "").strip()
    if api_key.startswith("wsclaw-route-"):
        return api_key.removeprefix("wsclaw-route-")
    return None


def _routing_key_hash(routing_key: str | None) -> str | None:
    if not routing_key:
        return None
    return hashlib.sha256(routing_key.encode()).hexdigest()[:16]


def _next_route_request_index(routing_key: str | None) -> int:
    if not routing_key:
        return 1
    with _UPSTREAM_LOCK:
        value = _ROUTE_REQUEST_COUNTS.get(routing_key, 0) + 1
        if routing_key not in _ROUTE_REQUEST_COUNTS and len(_ROUTE_REQUEST_COUNTS) >= _MAX_ROUTE_ASSIGNMENTS:
            _ROUTE_REQUEST_COUNTS.pop(next(iter(_ROUTE_REQUEST_COUNTS)))
        _ROUTE_REQUEST_COUNTS[routing_key] = value
        return value


def _server_timing_ms(value: str | None, metric: str) -> float | None:
    if not value:
        return None
    match = re.search(
        rf"(?:^|,)\s*{re.escape(metric)}\s*;[^,]*?dur=([0-9]+(?:\.[0-9]+)?)",
        value,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def _append_request_metrics(record: dict[str, Any]) -> None:
    raw_path = os.environ.get("WSCLAW_PROXY_METRICS_JSONL", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_absolute():
        LOG.error("WSCLAW_PROXY_METRICS_JSONL must be absolute: %s", path)
        return
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _METRICS_LOCK:
            descriptor = os.open(
                path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, payload.encode())
            finally:
                os.close(descriptor)
    except OSError as exc:
        LOG.error("failed to append proxy metrics to %s: %s", path, exc)


def _ordered_upstream_bases(
    bases: list[str],
    *,
    routing_key: str | None,
) -> list[str]:
    with _UPSTREAM_LOCK:
        return _ordered_upstream_bases_locked(bases, routing_key=routing_key)


def _ordered_upstream_bases_locked(
    bases: list[str],
    *,
    routing_key: str | None,
) -> list[str]:
    if routing_key:
        assigned = _ROUTE_UPSTREAM.get(routing_key)
        if assigned in bases:
            start_index = bases.index(assigned)
        else:
            digest = hashlib.sha256(f"judge-engine:{routing_key}".encode()).digest()
            preferred = int.from_bytes(digest[:8], "big") % len(bases)
            preferred_order = [
                (preferred + offset) % len(bases) for offset in range(len(bases))
            ]
            minimum = min(_UPSTREAM_IN_FLIGHT.get(base, 0) for base in bases)
            start_index = next(
                index
                for index in preferred_order
                if _UPSTREAM_IN_FLIGHT.get(bases[index], 0) == minimum
            )
            if len(_ROUTE_UPSTREAM) >= _MAX_ROUTE_ASSIGNMENTS:
                _ROUTE_UPSTREAM.pop(next(iter(_ROUTE_UPSTREAM)))
            _ROUTE_UPSTREAM[routing_key] = bases[start_index]
    else:
        start_index = next(_UPSTREAM_COUNTER) % len(bases)
    return bases[start_index:] + bases[:start_index]


def _reserve_upstream_bases(
    bases: list[str],
    *,
    routing_key: str | None,
) -> list[str]:
    with _UPSTREAM_LOCK:
        ordered = _ordered_upstream_bases_locked(bases, routing_key=routing_key)
        selected = ordered[0]
        _UPSTREAM_IN_FLIGHT[selected] = _UPSTREAM_IN_FLIGHT.get(selected, 0) + 1
        return ordered


def _reserve_upstream(base: str) -> None:
    with _UPSTREAM_LOCK:
        _UPSTREAM_IN_FLIGHT[base] = _UPSTREAM_IN_FLIGHT.get(base, 0) + 1


def _release_upstream(base: str) -> None:
    with _UPSTREAM_LOCK:
        remaining = _UPSTREAM_IN_FLIGHT.get(base, 0) - 1
        if remaining > 0:
            _UPSTREAM_IN_FLIGHT[base] = remaining
        else:
            _UPSTREAM_IN_FLIGHT.pop(base, None)


def _remember_upstream(routing_key: str | None, base: str) -> None:
    if not routing_key:
        return
    with _UPSTREAM_LOCK:
        _ROUTE_UPSTREAM[routing_key] = base


def reasoning_request_options() -> dict[str, Any]:
    enabled = os.environ.get("WSCLAW_PROXY_ENABLE_THINKING", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return {
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "separate_reasoning": True,
        }
    effort = os.environ.get("WSCLAW_PROXY_REASONING_EFFORT", "medium").strip().lower()
    if effort not in {"low", "medium", "high"}:
        effort = "medium"
    return {
        "reasoning_effort": effort,
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True,
        "stream_reasoning": True,
    }


def cap_completion_for_context_error(
    body: dict[str, Any],
    error_text: str,
) -> dict[str, Any] | None:
    enabled = os.environ.get("WSCLAW_PROXY_CONTEXT_OVERFLOW_RETRY", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    match = _CONTEXT_LIMIT_RE.search(error_text or "")
    if match is None:
        return None
    context_len = int(match.group("context"))
    input_tokens = int(match.group("input"))
    requested_completion = int(match.group("completion"))
    guard_tokens = max(
        0,
        int(os.environ.get("WSCLAW_PROXY_CONTEXT_GUARD_TOKENS", "64")),
    )
    allowed_completion = context_len - input_tokens - guard_tokens
    if allowed_completion <= 0:
        return None
    current_max = int(body.get("max_tokens") or requested_completion)
    capped_max = max(1, min(current_max, requested_completion, allowed_completion))
    if capped_max >= current_max:
        return None
    capped = dict(body)
    capped["max_tokens"] = capped_max
    return capped


def non_retryable_upstream_status(status_code: int | None) -> int | None:
    """Return request-global 4xx errors that must not be disguised as 502s."""
    if status_code in {400, 401, 403, 413, 422}:
        return status_code
    return None


@app.get("/")
@app.get("/health")
async def health() -> dict[str, Any]:
    bases, _key, model, _timeout, _cap = _settings()
    return {
        "status": "ok",
        "pid": os.getpid(),
        "upstreams": bases,
        "model": model,
        "reasoning": reasoning_request_options(),
    }


@app.head("/")
async def health_head() -> JSONResponse:
    return JSONResponse({})


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> dict[str, int]:
    payload = await request.json()
    chars = len(json.dumps(payload.get("messages") or [], ensure_ascii=False))
    chars += len(json.dumps(payload.get("tools") or [], ensure_ascii=False))
    return {"input_tokens": max(1, chars // 4)}


@app.post("/v1/messages")
async def messages(request: Request):
    payload = await request.json()
    bases, key, model, timeout, max_tokens_cap = _settings()
    routing_key = _request_routing_key(request)
    routing_hash = _routing_key_hash(routing_key)
    request_index = _next_route_request_index(routing_key)
    request_id = uuid.uuid4().hex
    tenant = os.environ.get("WSCLAW_ROUTER_TENANT", "").strip() or os.environ.get(
        "RUN_ID", "unknown"
    ).strip()
    workload_class = os.environ.get(
        "WSCLAW_ROUTER_WORKLOAD_CLASS", "judge"
    ).strip()
    ordered_bases = _reserve_upstream_bases(bases, routing_key=routing_key)
    body: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages_to_openai(payload),
        "max_tokens": min(int(payload.get("max_tokens") or max_tokens_cap), max_tokens_cap),
        "temperature": float(payload.get("temperature") or 0.0),
        "stream": False,
    }
    body.update(reasoning_request_options())
    tools = anthropic_tools_to_openai(payload)
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    started = time.perf_counter()
    base = ordered_bases[0]
    failures: list[str] = []
    terminal_status: int | None = None
    raw_response: dict[str, Any] | None = None
    raw_response_headers: dict[str, str] = {}
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 5.0))
    async with httpx.AsyncClient(timeout=client_timeout, trust_env=False) as client:
        for candidate_index, candidate in enumerate(ordered_bases):
            if candidate_index:
                _reserve_upstream(candidate)
            base = candidate
            try:
                context_retry_used = False
                while True:
                    upstream = await client.post(
                        f"{candidate}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            **({"X-SMG-Routing-Key": routing_key} if routing_key else {}),
                            "X-WSClaw-Tenant": tenant[:128],
                            "X-WSClaw-Workload-Class": workload_class[:64],
                            "X-WSClaw-Request-ID": request_id,
                        },
                        json=body,
                    )
                    if upstream.status_code == 400 and not context_retry_used:
                        capped = cap_completion_for_context_error(body, upstream.text)
                        if capped is not None:
                            LOG.warning(
                                "context overflow upstream=%s routing=%s; cap max_tokens %d -> %d",
                                candidate,
                                routing_key or "round-robin",
                                body["max_tokens"],
                                capped["max_tokens"],
                            )
                            body = capped
                            context_retry_used = True
                            continue
                    upstream.raise_for_status()
                    raw_response = upstream.json()
                    raw_response_headers = dict(upstream.headers)
                    _remember_upstream(routing_key, candidate)
                    break
                if raw_response is not None:
                    break
            except (httpx.HTTPError, ValueError) as exc:
                detail = getattr(getattr(exc, "response", None), "text", None) or str(exc)
                failures.append(f"{candidate}: {detail[:500]}")
                status_code = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                terminal_status = non_retryable_upstream_status(status_code)
                if status_code == 400:
                    LOG.warning(
                        "upstream rejected message roles: %s",
                        [message.get("role") for message in body.get("messages", [])],
                    )
                if terminal_status is not None:
                    LOG.warning(
                        "non-retryable upstream HTTP %d; returning it to the client: %s",
                        terminal_status,
                        failures[-1],
                    )
                else:
                    LOG.warning(
                        "upstream request failed, trying next endpoint: %s", failures[-1]
                    )
            finally:
                _release_upstream(candidate)
            if terminal_status is not None:
                break
    if raw_response is None:
        detail = "; ".join(failures)[:2000]
        _append_request_metrics(
            {
                "recorded_at_unix": time.time(),
                "request_id": request_id,
                "routing_key_hash": routing_hash,
                "session_request_index": request_index,
                "tenant": tenant[:128],
                "workload_class": workload_class[:64],
                "status": "non_retryable_error" if terminal_status else "error",
                "upstream_status": terminal_status,
                "failure_count": len(failures),
                "duration_seconds": time.perf_counter() - started,
            }
        )
        LOG.error("all upstream requests failed: %s", detail)
        raise HTTPException(status_code=terminal_status or 502, detail=detail)

    result = openai_response_to_anthropic(
        raw_response, requested_model=str(payload.get("model") or model)
    )
    usage = raw_response.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    server_timing = raw_response_headers.get("server-timing", "")
    router_duration_ms = _server_timing_ms(server_timing, "gemma_router")
    router_queue_ms = _server_timing_ms(server_timing, "gemma_router_queue")
    router_worker_id = raw_response_headers.get("x-gemma-router-worker-id")
    router_route = raw_response_headers.get("x-gemma-router-route")
    router_retry_count = int(
        raw_response_headers.get("x-gemma-router-retry-count") or 0
    )
    duration_seconds = time.perf_counter() - started
    metric_record = {
        "recorded_at_unix": time.time(),
        "request_id": request_id,
        "router_request_id": raw_response_headers.get("x-gemma-router-request-id"),
        "routing_key_hash": routing_hash,
        "session_request_index": request_index,
        "tenant": tenant[:128],
        "workload_class": workload_class[:64],
        "status": "ok",
        "message_count": len(body["messages"]),
        "tool_count": len(tools),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "cached_prompt_tokens": int(prompt_details.get("cached_tokens") or 0),
        "router_worker_id": router_worker_id,
        "router_route": router_route,
        "router_retry_count": router_retry_count,
        "router_duration_ms": router_duration_ms,
        "router_queue_ms": router_queue_ms,
        "duration_seconds": duration_seconds,
    }
    _append_request_metrics(metric_record)
    LOG.info(
        "request id=%s routing=%s round=%d worker=%s route=%s retries=%d "
        "model=%s messages=%d tools=%d prompt_tokens=%d completion_tokens=%d "
        "reasoning_tokens=%d router=%.3fs queue=%.3fs stop=%s duration=%.2fs",
        request_id,
        routing_hash or "round-robin",
        request_index,
        router_worker_id or "unknown",
        router_route or "unknown",
        router_retry_count,
        model,
        len(body["messages"]),
        len(tools),
        metric_record["prompt_tokens"],
        metric_record["completion_tokens"],
        metric_record["reasoning_tokens"],
        (router_duration_ms or 0.0) / 1000.0,
        (router_queue_ms or 0.0) / 1000.0,
        result["stop_reason"],
        duration_seconds,
    )
    response_headers = {
        name: value
        for name, value in (
            ("Server-Timing", server_timing),
            ("X-Gemma-Router-Request-ID", raw_response_headers.get("x-gemma-router-request-id")),
            ("X-Gemma-Router-Worker-ID", router_worker_id),
            ("X-Gemma-Router-Route", router_route),
            ("X-Gemma-Router-Retry-Count", str(router_retry_count)),
        )
        if value
    }
    if payload.get("stream"):
        return StreamingResponse(
            anthropic_sse(result),
            media_type="text/event-stream",
            headers=response_headers,
        )
    return JSONResponse(result, headers=response_headers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
