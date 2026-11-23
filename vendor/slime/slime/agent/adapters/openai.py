"""OpenAI Chat-Completions adapter for agent rollouts.

Mirrors slime.agent.adapters.anthropic but speaks the OpenAI
/v1/chat/completions protocol, so an OpenAI-compatible client (e.g. the Codex
CLI) can drive the slime sglang server. Each request is rendered with the served
model's chat template, sent to sglang /generate as input_ids, parsed, and folded
into a shared TrajectoryManager keyed by session id. finish_session(sid) drains
a session's trajectory into a list of Sample.

Only /v1/chat/completions is implemented; the Responses API (/v1/responses) is
out of scope. The section layout (adapter class -> translation -> reply building
-> request framing) mirrors slime.agent.adapters.anthropic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.adapters.common import (
    BaseAdapter,
    Reply,
    Session,
    TurnRecord,
    _chat_template_kwargs,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    sid_from_body,
)
from slime.agent.parsing import ParsedModelOutput
from slime.agent.parsing import parse_tagged_json_tool_uses

logger = logging.getLogger(__name__)

_REASONING_CONTEXT_POLICIES = {"legacy", "strip", "preserve"}
_TRUTHY = {"1", "true", "yes", "on", "enable", "enabled"}
_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length of\s+(?P<context>\d+)\s+tokens.*?"
    r"total of\s+(?P<total>\d+)\s+tokens:\s+"
    r"(?P<input>\d+)\s+tokens from the input messages and\s+"
    r"(?P<completion>\d+)\s+tokens for the completion",
    re.IGNORECASE | re.DOTALL,
)
_TAGGED_TOOL_CALL_RE = re.compile(
    r"(?:<tool_call>.*?</tool_call>|<\|tool_call>.*?<tool_call\|>)",
    re.DOTALL,
)


class OpenAIAdapter(BaseAdapter):
    """OpenAI Chat-Completions-compatible HTTP adapter: wire translation and
    reply framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "openai_adapter"
    max_token_keys = ("max_completion_tokens", "max_tokens", "max_output_tokens")
    stop_keys = ("stop",)

    def __init__(
        self,
        *,
        reasoning_context_policy: str | None = None,
        upstream_chat_completions: bool = False,
        upstream_chat_model: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reasoning_context_policy = _resolve_reasoning_context_policy(reasoning_context_policy)
        self.upstream_chat_completions = upstream_chat_completions
        self.upstream_chat_model = upstream_chat_model.strip() if isinstance(upstream_chat_model, str) else upstream_chat_model
        self.sglang_worker_urls = _worker_urls_from_env()
        self._sglang_worker_counts = {url: 0 for url in self.sglang_worker_urls}
        self._sglang_worker_lock = asyncio.Lock()

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/chat/completions", self._run_turn)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request, body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        messages = body.get("messages") or []
        if not isinstance(messages, list):
            raise web.HTTPBadRequest(text="messages must be a list")
        translated = _translate_messages(messages, reasoning_context_policy=self.reasoning_context_policy)
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        wire_message, manager_message, wire_finish = _build_reply_parts(
            parsed,
            raw_finish,
            reasoning_context_policy=self.reasoning_context_policy,
        )
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(wire_message, wire_finish),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        wire_message, wire_finish = reply.wire
        if stream:
            return await _render_stream(request, body, wire_message, wire_finish, in_tok, out_tok)
        return web.json_response(_render_response(body, wire_message, wire_finish, in_tok, out_tok))

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        if not self.upstream_chat_completions:
            return await super()._run_turn(request)
        return await self._run_turn_via_upstream_chat(request)

    async def _run_turn_via_upstream_chat(self, request: web.Request) -> web.StreamResponse:
        """ClawRL-compatible path: let SGLang's chat-completions endpoint render
        the prompt and parse Qwen reasoning/tool calls, then retokenize locally
        for TrajectoryManager.
        """

        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        session = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            translated, tools_schema = self._translate(body)
            forward_body = _upstream_chat_body(
                body,
                session=session,
                max_token_keys=self.max_token_keys,
                stop_keys=self.stop_keys,
                upstream_model=self.upstream_chat_model,
            )
            sglang_url, worker_url = await self._claim_sglang_url()
            try:
                output = await _call_sglang_chat_completions(
                    sglang_url,
                    forward_body,
                    session_id=sid,
                    routing_key=session.routing_key,
                    logger=self.logger,
                    log_prefix=self.log_prefix,
                )
            finally:
                await self._release_sglang_url(worker_url)
            choice = _first_choice(output)
            wire_message = _require_dict(choice.get("message"), "choice.message")
            raw_finish = str(choice.get("finish_reason") or "stop")
            wire_message, raw_finish, raw_tool_count, tool_calls_truncated = _normalize_upstream_wire_message(
                wire_message,
                finish_reason=raw_finish,
                tools_schema=tools_schema,
                session_id=sid,
            )
            choice["message"] = wire_message
            choice["finish_reason"] = raw_finish

            turn = _turn_record_from_chat_response(
                tokenizer=self.tokenizer,
                body=body,
                wire_message=wire_message,
                raw_finish=raw_finish,
                choice=choice,
                reasoning_context_policy=self.reasoning_context_policy,
            )
            manager_message = _manager_message_from_wire(
                wire_message,
                reasoning_context_policy=self.reasoning_context_policy,
            )
            in_tok, out_tok = len(turn.prompt_ids), len(turn.output_ids)
            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

            try:
                if stream:
                    response = await _render_stream(request, body, wire_message, raw_finish, in_tok, out_tok)
                else:
                    response = web.json_response(output)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            self._run_debug_callback(sid, translated, tools_schema, manager_message, turn)
            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=manager_message,
                request_messages=forward_body.get("messages"),
                request_tools=forward_body.get("tools"),
                chat_template_kwargs=_chat_template_kwargs(forward_body),
                metadata={
                    "sid": sid,
                    "upstream_chat_completions": True,
                    "raw_tool_call_count": raw_tool_count,
                    "tool_calls_truncated": tool_calls_truncated,
                },
            )
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)

    async def _claim_sglang_url(self) -> tuple[str, str | None]:
        if not self.sglang_worker_urls:
            return self.sglang_url, None
        async with self._sglang_worker_lock:
            worker_url = min(
                self.sglang_worker_urls,
                key=lambda url: self._sglang_worker_counts.get(url, 0),
            )
            self._sglang_worker_counts[worker_url] = self._sglang_worker_counts.get(worker_url, 0) + 1
        return worker_url, worker_url

    async def _release_sglang_url(self, worker_url: str | None) -> None:
        if not worker_url:
            return
        async with self._sglang_worker_lock:
            self._sglang_worker_counts[worker_url] = max(
                0,
                self._sglang_worker_counts.get(worker_url, 0) - 1,
            )


# --- Translation (OpenAI wire -> chat-template messages) ---


def _arguments_as_dict(arguments: Any) -> dict[str, Any]:
    """Coerce wire-shape tool_calls[].function.arguments into a dict.

    OpenAI sends arguments as a JSON-encoded string; the chat template and the
    trajectory manager's history matching both expect a mapping. Malformed
    payloads fall back to {"_raw_arguments": s}.
    """
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return {"_raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"_raw_arguments": arguments}
    return {"_raw_arguments": str(arguments)}


def _resolve_reasoning_context_policy(raw: str | None) -> str:
    """Resolve how OpenAI adapter history should handle reasoning_content."""

    value = (raw or "").strip().lower()
    if not value:
        for name in ("SLIME_AGENT_OPENAI_REASONING_CONTEXT_POLICY", "SLIME_AGENT_REASONING_CONTEXT_POLICY"):
            value = os.environ.get(name, "").strip().lower()
            if value:
                break
    if not value and os.environ.get("SLIME_AGENT_STRIP_REASONING_CONTEXT", "").strip().lower() in _TRUTHY:
        value = "strip"
    if not value:
        value = "legacy"
    if value not in _REASONING_CONTEXT_POLICIES:
        allowed = ", ".join(sorted(_REASONING_CONTEXT_POLICIES))
        raise ValueError(f"reasoning_context_policy must be one of {allowed}; got {value!r}")
    return value


def _translate_messages(messages: list[dict], *, reasoning_context_policy: str = "legacy") -> list[dict]:
    """OpenAI chat messages -> tokenizer chat-template messages.

    Mirrors anthropic._translate_messages so a replayed assistant turn compares
    equal (dict equality) to the leaf the manager appended on the previous
    request. Two invariants must hold:

      * tool_calls[i].function.arguments is a dict (not a JSON string): the chat
        template needs a mapping, and the manager matches history by dict
        equality regardless of key order.
      * Wire-only correlation ids are dropped (tool_call_id on tool messages,
        tool_calls[i].id on echoed assistant messages). Fresh ids are minted on
        each response, so keeping the wire ids would diverge the replay match.
    """
    translated: list[dict] = []
    tool_names_by_call_id: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "developer":  # OpenAI Responses API alias
            role = "system"

        if role in {"system", "user"}:
            translated.append({"role": role, "content": flatten_content(content)})
        elif role == "tool":
            # Drop tool_call_id -- it is a wire-only correlation field -- but
            # retain the stable function name.  Gemma's template uses it to
            # render ``response:<name>{...}``; dropping it silently produces
            # ``response:unknown`` on every later tool turn.
            tool_message: dict[str, Any] = {"role": "tool", "content": flatten_content(content)}
            name = msg.get("name")
            if not name and msg.get("tool_call_id") is not None:
                name = tool_names_by_call_id.get(str(msg["tool_call_id"]))
            if name:
                tool_message["name"] = str(name)
            translated.append(tool_message)
        elif role == "assistant":
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": flatten_content(content),
            }
            reasoning = msg.get("reasoning_content")
            if reasoning:
                if reasoning_context_policy != "strip":
                    assistant["reasoning_content"] = reasoning
            tool_calls = msg.get("tool_calls") or []
            normalized: list[dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                name = function.get("name") or call.get("name") or "tool"
                if call.get("id") is not None:
                    tool_names_by_call_id[str(call["id"])] = str(name)
                arguments = function.get("arguments")
                if arguments is None:
                    arguments = call.get("arguments", {})
                # NB: arguments stays a dict (not a JSON string), and the
                # wire-only id is dropped. See docstring above.
                normalized.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _arguments_as_dict(arguments),
                        },
                    }
                )
            if normalized:
                assistant["tool_calls"] = normalized
            translated.append(assistant)
        # unknown roles are silently dropped
    return translated


def _tools_to_chat_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI tools list to tokenizer chat-template tool schema."""
    if not tools:
        return None
    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") and tool.get("type") != "function":
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is not None:
            name = function.get("name")
            if not name:
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        else:
            name = tool.get("name")
            if not name:
                continue
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
    return normalized or None


# --- ClawRL-compatible upstream chat-completions path ---


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise web.HTTPBadRequest(text=f"{field_name} must be an object")
    return value


def _require_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise web.HTTPBadRequest(text=f"{field_name} must be a list")
    return value


def _upstream_chat_body(
    body: dict[str, Any],
    *,
    session: Session,
    max_token_keys: tuple[str, ...],
    stop_keys: tuple[str, ...],
    upstream_model: str | None,
) -> dict[str, Any]:
    forward = {key: value for key, value in body.items() if key not in {"session_id", "turn_type"}}
    forward["stream"] = False
    forward["logprobs"] = True
    forward["top_logprobs"] = int(forward.get("top_logprobs") or 1)
    # These SGLang extensions expose the exact rendered prompt IDs and sampled
    # output IDs.  Local re-tokenization is only a compatibility fallback.
    forward["return_prompt_token_ids"] = True
    forward["return_meta_info"] = True
    forward.pop("stream_options", None)
    if upstream_model:
        forward["model"] = upstream_model

    # Match the standalone WSClaw/fallback OpenAI request exactly.  Nanobot
    # normally supplies the function schemas but may omit an explicit choice;
    # SGLang's Gemma parser should receive the same unambiguous `auto` mode as
    # the known-good fallback router.
    if forward.get("tools") and not forward.get("tool_choice"):
        forward["tool_choice"] = "auto"

    defaults = dict(session.sampling_defaults or {})
    for src_key, dst_key in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("sampling_seed", "sampling_seed"),
    ):
        if src_key in defaults and forward.get(dst_key) is None:
            forward[dst_key] = defaults[src_key]

    if not any(forward.get(key) is not None for key in max_token_keys):
        max_new_tokens = defaults.get("max_new_tokens")
        if max_new_tokens is not None:
            forward["max_completion_tokens"] = int(max_new_tokens)

    if not any(forward.get(key) for key in stop_keys) and defaults.get("stop"):
        forward["stop"] = defaults["stop"]
    return forward


async def _call_sglang_chat_completions(
    sglang_url: str,
    body: dict[str, Any],
    *,
    session_id: str,
    routing_key: str | None,
    logger: logging.Logger,
    log_prefix: str,
) -> dict[str, Any]:
    normalized_routing_key = str(routing_key or "").strip()
    headers = {"X-SMG-Routing-Key": normalized_routing_key} if normalized_routing_key else None
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    last_error = ""
    context_retry_used = False
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        for attempt in range(6):
            async with sess.post(f"{sglang_url}/v1/chat/completions", json=body, headers=headers) as resp:
                if resp.status == 503 and attempt < 5:
                    last_error = await resp.text()
                    logger.info(
                        "[%s] sid=%s SGLang chat 503, retry %d/5: %.160s",
                        log_prefix,
                        session_id,
                        attempt + 1,
                        last_error,
                    )
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                if resp.status >= 400:
                    text = await resp.text()
                    capped = _cap_chat_body_for_context_error(body, text)
                    if resp.status == 400 and capped is not None and not context_retry_used:
                        old_max = _first_present_int(body, OpenAIAdapter.max_token_keys)
                        new_max = _first_present_int(capped, OpenAIAdapter.max_token_keys)
                        logger.warning(
                            "[%s] sid=%s SGLang chat context overflow; cap max_tokens %s -> %s and retry once: %.220s",
                            log_prefix,
                            session_id,
                            old_max,
                            new_max,
                            text,
                        )
                        body = capped
                        context_retry_used = True
                        continue
                    logger.warning(
                        "[%s] sid=%s SGLang chat upstream %d: %.300s",
                        log_prefix,
                        session_id,
                        resp.status,
                        text,
                    )
                    raise RuntimeError(f"sglang chat upstream {resp.status}: {text[:400]}")
                output = await resp.json(content_type=None)
                return _normalize_sglang_chat_response(output)
    raise RuntimeError(f"sglang chat upstream failed: {last_error[:400]}")


def _cap_chat_body_for_context_error(body: dict[str, Any], error_text: str) -> dict[str, Any] | None:
    if os.environ.get("CLAWGYM_SGLANG_CONTEXT_OVERFLOW_RETRY", "1").strip().lower() not in _TRUTHY:
        return None
    match = _CONTEXT_LIMIT_RE.search(error_text or "")
    if match is None:
        return None
    context_len = int(match.group("context"))
    input_tokens = int(match.group("input"))
    requested_completion = int(match.group("completion"))
    guard_tokens = _nonnegative_int_env("CLAWGYM_SGLANG_CONTEXT_GUARD_TOKENS", 0)
    allowed_completion = context_len - input_tokens - guard_tokens
    if allowed_completion <= 0:
        return None
    current_max = _first_present_int(body, OpenAIAdapter.max_token_keys)
    if current_max is None:
        current_max = requested_completion
    capped_max = max(1, min(current_max, requested_completion, allowed_completion))
    if capped_max >= current_max:
        return None

    capped = dict(body)
    changed = False
    for key in OpenAIAdapter.max_token_keys:
        if key in capped and capped[key] is not None:
            capped[key] = capped_max
            changed = True
    if not changed:
        capped["max_completion_tokens"] = capped_max
    return capped


def _first_present_int(body: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = body.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _allow_multi_tool_calls() -> bool:
    return os.environ.get("CLAWGYM_ALLOW_MULTI_TOOL_CALLS", "0").strip().lower() in _TRUTHY


def _researchclaw_compat() -> bool:
    return os.environ.get("CLAWGYM_NANOBOT_RESEARCHCLAW_COMPAT", "0").strip().lower() in _TRUTHY


def _worker_urls_from_env() -> list[str]:
    raw = os.environ.get("CLAWGYM_SGLANG_WORKER_URLS", "").strip()
    if not raw:
        return []
    urls: list[str] = []
    for part in raw.replace("\n", ",").split(","):
        url = part.strip().rstrip("/")
        if url:
            urls.append(url)
    return urls


def _first_choice(output: dict[str, Any]) -> dict[str, Any]:
    choices = _require_list(output.get("choices"), "choices")
    if not choices:
        raise web.HTTPBadGateway(text="SGLang chat response has no choices")
    return _require_dict(choices[0], "choices[0]")


def _normalize_sglang_chat_response(output: Any) -> dict[str, Any]:
    """Convert source SGLang token metadata into the adapter's exact shape."""
    if not isinstance(output, dict):
        raise ValueError("SGLang chat response must be an object")
    choices = output.get("choices")
    if not isinstance(choices, list):
        return output
    response_prompt_ids = output.get("prompt_token_ids")

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        prompt_ids = choice.get("input_token_ids", choice.get("prompt_token_ids"))
        if prompt_ids is None:
            prompt_ids = response_prompt_ids
        if prompt_ids is not None:
            normalized_prompt_ids = _exact_int_list(prompt_ids)
            if not normalized_prompt_ids:
                raise ValueError("SGLang chat response contains malformed prompt_token_ids")
            choice["input_token_ids"] = normalized_prompt_ids

        meta_info = choice.get("meta_info")
        if meta_info is None:
            _stamp_choice_token_ids_onto_logprobs(choice)
            continue
        if not isinstance(meta_info, dict):
            raise ValueError("SGLang chat response meta_info must be an object")
        raw_output_logprobs = meta_info.get("output_token_logprobs")
        if not isinstance(raw_output_logprobs, list) or not raw_output_logprobs:
            raise ValueError("SGLang chat response is missing meta_info.output_token_logprobs")

        token_ids: list[int] = []
        logprobs: list[float] = []
        token_texts: list[str | None] = []
        for item in raw_output_logprobs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                logprob, token_id = item[0], item[1]
                token_text = item[2] if len(item) >= 3 else None
            elif isinstance(item, dict):
                logprob = item.get("logprob", item.get("log_prob"))
                token_id = item.get("token_id")
                token_text = item.get("token")
            else:
                raise ValueError("SGLang output_token_logprobs entry is malformed")
            try:
                logprobs.append(float(logprob))
                token_ids.append(int(token_id))
            except (TypeError, ValueError) as exc:
                raise ValueError("SGLang output_token_logprobs entry is malformed") from exc
            token_texts.append(str(token_text) if token_text is not None else None)

        existing_token_ids = choice.get("token_ids")
        if existing_token_ids is not None and _exact_int_list(existing_token_ids) != token_ids:
            raise ValueError("SGLang token_ids do not match meta_info.output_token_logprobs")
        choice["token_ids"] = token_ids
        logprobs_obj = choice.get("logprobs")
        if not isinstance(logprobs_obj, dict):
            logprobs_obj = {}
            choice["logprobs"] = logprobs_obj
        content = logprobs_obj.get("content")
        if not isinstance(content, list) or len(content) != len(token_ids):
            content = [{} for _ in token_ids]
            logprobs_obj["content"] = content
        for index, (entry, token_id, logprob, token_text) in enumerate(
            zip(content, token_ids, logprobs, token_texts, strict=True)
        ):
            if not isinstance(entry, dict):
                entry = {}
                content[index] = entry
            returned_token_id = entry.get("token_id")
            if returned_token_id is not None and int(returned_token_id) != token_id:
                raise ValueError("SGLang logprob token_id does not match output_token_logprobs")
            entry["token_id"] = token_id
            entry["logprob"] = logprob
            if token_text is not None:
                entry.setdefault("token", token_text)
        choice.pop("meta_info", None)
    return output


def _exact_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _stamp_choice_token_ids_onto_logprobs(choice: dict[str, Any]) -> None:
    token_ids = _exact_int_list(choice.get("token_ids"))
    logprobs_obj = choice.get("logprobs")
    content = logprobs_obj.get("content") if isinstance(logprobs_obj, dict) else None
    if not token_ids or not isinstance(content, list) or len(content) != len(token_ids):
        return
    for entry, token_id in zip(content, token_ids, strict=True):
        if isinstance(entry, dict):
            entry.setdefault("token_id", token_id)


def _normalize_upstream_wire_message(
    message: dict[str, Any],
    *,
    finish_reason: str,
    tools_schema: list[dict] | None,
    session_id: str,
) -> tuple[dict[str, Any], str, int, bool]:
    wire = dict(message)
    tool_calls = wire.get("tool_calls")
    raw_tool_count = 0
    truncated = False
    if tool_calls is None:
        tool_calls = []
    elif isinstance(tool_calls, list):
        raw_tool_count = len(tool_calls)
        if not _researchclaw_compat() and len(tool_calls) > 1 and not _allow_multi_tool_calls():
            truncated = True
            tool_calls = tool_calls[:1]
    else:
        tool_calls = []

    content = flatten_content(wire.get("content"))
    reasoning = flatten_content(wire.get("reasoning_content"))
    pseudo_tool_call_count = 0
    recovered_from_reasoning = False
    if not tool_calls and tools_schema:
        recovered = parse_tagged_json_tool_uses(content, tools_schema)
        if _researchclaw_compat() and not recovered and not content.strip():
            recovered = parse_tagged_json_tool_uses(reasoning, tools_schema)
            recovered_from_reasoning = bool(recovered)
        if recovered:
            tool_calls = [
                {
                    "id": f"call_{secrets.token_hex(12)}",
                    "type": "function",
                    "function": {
                        "name": str(call.get("name") or "tool"),
                        "arguments": json.dumps(call.get("input") or {}, ensure_ascii=False),
                    },
                }
                for call in recovered
            ]
            raw_tool_count = len(tool_calls)
            pseudo_tool_call_count = len(tool_calls)
            content = ""
            wire["content"] = None
            finish_reason = "tool_calls"
            if recovered_from_reasoning:
                remaining_reasoning = _TAGGED_TOOL_CALL_RE.sub("", reasoning).strip()
                if remaining_reasoning:
                    wire["reasoning_content"] = remaining_reasoning
                else:
                    wire.pop("reasoning_content", None)

    # Apply one policy after native and tagged tool calls are normalized so the
    # wire response and the trajectory-manager echo always retain the same set.
    if _researchclaw_compat():
        raw_tool_count = len(tool_calls)
        if len(tool_calls) > 1 and not _allow_multi_tool_calls():
            truncated = True
            tool_calls = tool_calls[:1]

    wire["tool_calls"] = tool_calls
    if tool_calls and _researchclaw_compat():
        finish_reason = "tool_calls"
    if pseudo_tool_call_count:
        wire["content"] = None
    else:
        wire["content"] = content
    return wire, finish_reason, raw_tool_count, truncated


def _manager_message_from_wire(
    message: dict[str, Any],
    *,
    reasoning_context_policy: str = "legacy",
) -> dict[str, Any]:
    tool_calls = _normalized_manager_tool_calls(message.get("tool_calls"))
    manager_message: dict[str, Any] = {
        "role": "assistant",
        # Native mixed text+tool-call turns are echoed by Nanobot. Clearing the
        # content here would manufacture a message-history branch next turn.
        "content": (
            flatten_content(message.get("content"))
            if _researchclaw_compat()
            else ("" if tool_calls else flatten_content(message.get("content")))
        ),
    }
    reasoning = message.get("reasoning_content")
    if reasoning_context_policy == "preserve" and reasoning:
        manager_message["reasoning_content"] = str(reasoning)
    if tool_calls:
        manager_message["tool_calls"] = tool_calls if _researchclaw_compat() else tool_calls[:1]
    return manager_message


def _normalized_manager_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments")
        if arguments is None:
            arguments = call.get("arguments", {})
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _arguments_as_dict(arguments),
                },
            }
        )
    return normalized


def _normalize_messages_for_template(
    messages: list[dict[str, Any]],
    *,
    reasoning_context_policy: str = "legacy",
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        if item.get("role") == "developer":
            item["role"] = "system"
        if "content" in item and not isinstance(item["content"], str):
            item["content"] = flatten_content(item["content"])
        if reasoning_context_policy == "strip":
            item.pop("reasoning_content", None)
        if isinstance(item.get("tool_calls"), list):
            item["tool_calls"] = _normalized_manager_tool_calls(item["tool_calls"])
        normalized.append(item)
    return normalized


def _turn_record_from_chat_response(
    *,
    tokenizer,
    body: dict[str, Any],
    wire_message: dict[str, Any],
    raw_finish: str,
    choice: dict[str, Any],
    reasoning_context_policy: str = "legacy",
) -> TurnRecord:
    exact_prompt_ids = _choice_token_ids(choice, "input_token_ids", "prompt_token_ids")
    exact_response_ids = _choice_token_ids(choice, "token_ids")
    if exact_prompt_ids is not None and exact_response_ids is not None:
        response_logprobs = _extract_logprobs_from_chat_response(choice)
        if len(response_logprobs) != len(exact_response_ids):
            raise ValueError(
                "exact chat response token ids/logprobs length mismatch: "
                f"{len(exact_response_ids)} != {len(response_logprobs)}"
            )
        _validate_choice_logprob_token_ids(choice, exact_response_ids)
        return TurnRecord(
            prompt_ids=exact_prompt_ids,
            output_ids=exact_response_ids,
            finish_reason=raw_finish or "stop",
            output_log_probs=response_logprobs,
        )

    if os.environ.get("CLAWGYM_NANOBOT_REQUIRE_EXACT_TOKENS", "").strip().lower() in _TRUTHY:
        raise ValueError(
            "upstream chat response omitted exact input_token_ids/token_ids while "
            "CLAWGYM_NANOBOT_REQUIRE_EXACT_TOKENS is enabled"
        )

    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise web.HTTPBadRequest(text="messages must be a list")
    tools = body.get("tools")
    normalized_messages = _normalize_messages_for_template(
        messages,
        reasoning_context_policy=reasoning_context_policy,
    )
    normalized_response = _normalize_messages_for_template([wire_message])[0]
    template_kwargs = _chat_template_kwargs(body)

    prompt_text = tokenizer.apply_chat_template(
        normalized_messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_text = tokenizer.apply_chat_template(
        normalized_messages + [normalized_response],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template output must start with prompt_text")
    response_text = full_text[len(prompt_text) :]
    response_text, _ = _strip_synthetic_empty_think_prefix(
        response_text,
        str(wire_message.get("reasoning_content") or ""),
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    response_logprobs = _extract_logprobs_from_chat_response(choice)
    if len(response_logprobs) > len(response_ids):
        response_logprobs = response_logprobs[: len(response_ids)]
    elif len(response_logprobs) < len(response_ids):
        response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(response_ids),
        finish_reason=raw_finish or "stop",
        output_log_probs=response_logprobs,
    )


def _choice_token_ids(choice: dict[str, Any], *keys: str) -> list[int] | None:
    for key in keys:
        values = choice.get(key)
        if not isinstance(values, list):
            continue
        try:
            return [int(value) for value in values]
        except (TypeError, ValueError):
            return None
    return None


def _validate_choice_logprob_token_ids(choice: dict[str, Any], response_ids: list[int]) -> None:
    logprobs_obj = choice.get("logprobs")
    content = logprobs_obj.get("content") if isinstance(logprobs_obj, dict) else None
    if not isinstance(content, list):
        return
    for index, (entry, expected_id) in enumerate(zip(content, response_ids, strict=False)):
        if not isinstance(entry, dict) or entry.get("token_id") is None:
            continue
        if int(entry["token_id"]) != int(expected_id):
            raise ValueError(
                "chat response logprob token_id mismatch at index "
                f"{index}: {entry['token_id']} != {expected_id}"
            )


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    values: list[float] = []
    for item in content:
        if isinstance(item, dict) and item.get("logprob") is not None:
            values.append(float(item["logprob"]))
    return values


def _strip_synthetic_empty_think_prefix(response_text: str, reasoning_content: str) -> tuple[str, bool]:
    if reasoning_content.strip():
        return response_text, False
    for prefix in ("<think>\n\n</think>\n\n", "<think>\n</think>\n\n"):
        if response_text.startswith(prefix):
            return response_text[len(prefix) :], True
    return response_text, False


# --- Reply building: parsed output -> OpenAI wire message + manager_message ---


def _build_reply_parts(
    parsed: ParsedModelOutput,
    finish: str,
    *,
    reasoning_context_policy: str = "legacy",
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Return (wire_message, manager_message, wire_finish).

    wire_message follows the OpenAI Chat-Completions spec: tool_calls[].id is a
    unique correlation id and tool_calls[].function.arguments is a JSON-encoded
    string (clients depend on this).

    manager_message is the shape fed to record_turn: arguments is a dict so
    chat-template replay succeeds and the manager's history match (dict equality)
    holds against the echo on the next turn, and the wire-only id is omitted.
    """
    wire_tool_calls: list[dict[str, Any]] = []
    manager_tool_calls: list[dict[str, Any]] = []
    for tu in parsed.tool_uses:
        name = tu.get("name", "tool")
        args_dict = tu.get("input") or {}
        if not isinstance(args_dict, dict):
            args_dict = {"_raw_arguments": str(args_dict)}
        call_id = f"call_{secrets.token_hex(12)}"
        wire_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False, sort_keys=True),
                },
            }
        )
        manager_tool_calls.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args_dict,
                },
            }
        )

    wire_message: dict[str, Any] = {
        "role": "assistant",
        # send content=null when there are tool_calls: some OpenAI clients split
        # a mixed text+tool_calls turn into two echoed messages otherwise, which
        # diverges the history match against our leaf
        "content": None if wire_tool_calls else (parsed.text or None),
    }
    # manager_message must match what the client echoes on the next request, or
    # the manager's history match (dict equality) diverges and every turn forks.
    # Differences from wire_message, each needed to match the echo by default:
    #   * no reasoning_content in legacy/strip modes -- some clients strip it on
    #     echo. preserve mode keeps it for harnesses that replay reasoning.
    #   * tool-call count follows CLAWGYM_ALLOW_MULTI_TOOL_CALLS
    #   * empty content when tool_calls are present -- mirrors content=null above
    manager_message: dict[str, Any] = {
        "role": "assistant",
        "content": "" if wire_tool_calls else (parsed.text or ""),
    }
    if parsed.reasoning:
        wire_message["reasoning_content"] = parsed.reasoning
        if reasoning_context_policy == "preserve":
            manager_message["reasoning_content"] = parsed.reasoning
    if wire_tool_calls:
        selected_wire_calls = (
            wire_tool_calls
            if _researchclaw_compat() and _allow_multi_tool_calls()
            else wire_tool_calls[:1]
        )
        selected_manager_calls = (
            manager_tool_calls
            if _researchclaw_compat() and _allow_multi_tool_calls()
            else manager_tool_calls[:1]
        )
        wire_message["tool_calls"] = selected_wire_calls
        manager_message["tool_calls"] = selected_manager_calls

    if parsed.tool_uses:
        wire_finish = "tool_calls"
    elif finish == "length":
        wire_finish = "length"
    else:
        wire_finish = "stop"

    return wire_message, manager_message, wire_finish


# --- Request framing: session id + wire response/stream rendering ---


def _request_session_id(request: web.Request, body: dict) -> str:
    """Resolve sid from ClawRL-style model ids or OpenAI body hints."""
    return _sid_from_model(body) or sid_from_bearer(request) or sid_from_body(body) or "default"


def _sid_from_model(body: dict | None) -> str | None:
    if not isinstance(body, dict):
        return None
    model = body.get("model")
    if not isinstance(model, str):
        return None
    _, sep, sid = model.partition("__")
    if sep and sid:
        return sid
    return None


def _usage(in_tok: int, out_tok: int) -> dict[str, int]:
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }


def _render_response(
    body: dict,
    wire_message: dict[str, Any],
    wire_finish: str,
    in_tok: int,
    out_tok: int,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl_{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "slime-actor"),
        "choices": [
            {
                "index": 0,
                "message": wire_message,
                "finish_reason": wire_finish,
            }
        ],
        "usage": _usage(in_tok, out_tok),
    }


async def _render_stream(
    request: web.Request,
    body: dict,
    wire_message: dict[str, Any],
    wire_finish: str,
    in_tok: int,
    out_tok: int,
) -> web.StreamResponse:
    """Emit the OpenAI Chat-Completions SSE stream.

    Each chunk has the shape `data: {chatcmpl ...}\n\n`, ending with
    `data: [DONE]`. The whole turn is realised on the server before streaming
    (we have no token-level deltas from sglang here), so we emit one role chunk,
    then content / reasoning / tool_calls in single delta chunks each.
    """
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)
    completion_id = f"chatcmpl_{secrets.token_hex(12)}"
    created = int(time.time())

    async def emit(delta: dict[str, Any], finish_reason: str | None = None, usage: dict | None = None) -> None:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.get("model", "slime-actor"),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        await out.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())

    await emit({"role": "assistant"})
    reasoning = wire_message.get("reasoning_content")
    if reasoning:
        await emit({"reasoning_content": reasoning})
    content = wire_message.get("content")
    if content:
        await emit({"content": content})
    # emit all tool_calls in a single chunk: some clients accumulate per-index
    # arguments fragments across chunks, collapsing N parallel tool_calls into
    # one call with a concatenated (and unparseable) arguments string
    tool_calls = wire_message.get("tool_calls") or []
    if tool_calls:
        await emit({"tool_calls": [{**call, "index": idx} for idx, call in enumerate(tool_calls)]})
    await emit({}, finish_reason=wire_finish, usage=_usage(in_tok, out_tok))
    await out.write(b"data: [DONE]\n\n")
    return out
