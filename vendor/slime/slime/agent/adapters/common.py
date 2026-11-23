"""Shared adapter primitives for token-capturing agent rollouts.

A protocol adapter (Anthropic / OpenAI) subclasses BaseAdapter and fills in the
wire-specific hooks (_register_routes, _session_id, _translate, _build_reply,
_respond, and optionally _preprocess_body) plus a few class attributes (logger,
log_prefix, max_token_keys, stop_keys). The session lifecycle, per-sid turn cap,
inflight-task bookkeeping and the one-turn _run_turn pipeline are inherited.

flatten_content, tool_call_dict and manager_finish_reason cover the parts both
protocols handle identically.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TrajectoryManager, TurnRecord


__all__ = ["TurnRecord"]

_GEMMA4_THOUGHT_PROMPT_PREFIX = "<|channel>thought\n"


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults, context budget and token lineage.

    Trajectory state lives in the shared TrajectoryManager (BaseAdapter.manager),
    not here.  The lineage fields deliberately keep the semantic message history
    separate from the exact ids sampled by SGLang: messages are still used for
    tool execution/routing, while subsequent model prompts reuse the original
    ``prompt_ids + output_ids`` byte-for-byte whenever history only appends.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    routing_key: str | None = None
    lineage_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock, repr=False)
    last_prompt_messages: list[dict] | None = None
    last_response_message: dict | None = None
    last_prompt_ids: list[int] = dataclasses.field(default_factory=list)
    last_output_ids: list[int] = dataclasses.field(default_factory=list)
    last_finish_reason: str = ""
    last_tools: list[dict] | None = None
    last_chat_template_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    prompt_build_modes: dict[str, int] = dataclasses.field(default_factory=dict)
    prompt_build_reasons: dict[str, int] = dataclasses.field(default_factory=dict)
    raw_prefix_violations: int = 0

    def record_prompt_build(self, mode: str, reason: str) -> None:
        self.prompt_build_modes[mode] = self.prompt_build_modes.get(mode, 0) + 1
        self.prompt_build_reasons[reason] = self.prompt_build_reasons.get(reason, 0) + 1

    def lineage_metrics(self) -> dict[str, Any]:
        return {
            "prompt_build_modes": dict(self.prompt_build_modes),
            "prompt_build_reasons": dict(self.prompt_build_reasons),
            "raw_prefix_violations": int(self.raw_prefix_violations),
        }


@dataclasses.dataclass
class Reply:
    """Output of an adapter's _build_reply, consumed by _run_turn.

    manager_message and finish_reason feed record_turn and the debug callback;
    wire is opaque to the pipeline and only the adapter's own _respond reads it.
    """

    manager_message: dict
    finish_reason: str
    wire: Any


@dataclasses.dataclass(frozen=True)
class PromptBuild:
    """Exact ids for one request plus how they were constructed."""

    prompt_ids: list[int]
    mode: str
    reason: str
    segment_boundary_reason: str | None = None


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **(chat_template_kwargs or {}),
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def _starts_with(values: list[Any], prefix: list[Any]) -> bool:
    return len(values) >= len(prefix) and values[: len(prefix)] == prefix


def _last_subsequence_end(values: list[int], subsequence: list[int]) -> int | None:
    """Return the end offset of the final exact subsequence occurrence."""
    if not subsequence or len(subsequence) > len(values):
        return None
    for start in range(len(values) - len(subsequence), -1, -1):
        if values[start : start + len(subsequence)] == subsequence:
            return start + len(subsequence)
    return None


def _reasoning_started_in_prompt(prompt_ids: list[int], tokenizer, reasoning_parser_name: str | None) -> bool:
    """Whether the exact prompt token suffix already opened reasoning.

    Gemma 4 opens its thought channel in the generation prompt following a
    tool response.  This check intentionally compares token ids instead of
    decoded/re-rendered text so parsing follows the exact prompt sent to the
    model.
    """

    if not reasoning_parser_name or reasoning_parser_name.lower() != "gemma4":
        return False
    try:
        suffix_ids = list(tokenizer.encode(_GEMMA4_THOUGHT_PROMPT_PREFIX, add_special_tokens=False))
    except Exception:
        return False
    return bool(suffix_ids) and len(prompt_ids) >= len(suffix_ids) and prompt_ids[-len(suffix_ids) :] == suffix_ids


def _history_rewrite_reason(messages: list[dict], expected: list[dict]) -> str:
    if len(messages) < len(expected):
        return "history_removed"
    for index, (current, previous) in enumerate(zip(messages, expected, strict=False)):
        if current != previous:
            return "history_modified" if index < len(expected) - 1 else "assistant_echo_modified"
    return "history_not_appended"


def _full_render_build(
    prompt_ids: list[int],
    session: Session,
    *,
    reason: str,
    initial: bool = False,
) -> PromptBuild:
    mode = "initial" if initial else "history_rewrite"
    session.record_prompt_build(mode, reason)
    return PromptBuild(
        prompt_ids=prompt_ids,
        mode=mode,
        reason=reason,
        segment_boundary_reason=None if initial else reason,
    )


def _build_prompt_ids(
    messages: list[dict],
    tokenizer,
    session: Session,
    *,
    tools: list[dict] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> PromptBuild:
    """Build one prompt while preserving prior sampled ids on append-only turns.

    The semantic message history is still rendered in full as a contract check
    and to obtain only the newly appended tool/environment suffix.  Previously
    sampled assistant content is never decoded and re-tokenized into the model
    prompt.  If messages, tools, template controls, or the terminal framing no
    longer form a safe append, the method deliberately falls back to a full
    render and marks a real segment boundary for TrajectoryManager.
    """

    kwargs = dict(chat_template_kwargs or {})
    canonical_current = _render_token_ids(
        messages,
        tokenizer,
        tools=tools,
        add_generation_prompt=True,
        chat_template_kwargs=kwargs,
    )

    if session.last_prompt_messages is None or session.last_response_message is None:
        return _full_render_build(canonical_current, session, reason="first_request", initial=True)

    if tools != session.last_tools:
        return _full_render_build(canonical_current, session, reason="tools_changed")
    if kwargs != session.last_chat_template_kwargs:
        return _full_render_build(canonical_current, session, reason="chat_template_kwargs_changed")

    expected_history = [*session.last_prompt_messages, session.last_response_message]
    if len(messages) <= len(expected_history) or not _starts_with(messages, expected_history):
        return _full_render_build(
            canonical_current,
            session,
            reason=_history_rewrite_reason(messages, expected_history),
        )

    if not session.last_output_ids:
        return _full_render_build(canonical_current, session, reason="previous_output_empty")
    if session.last_finish_reason == "length":
        return _full_render_build(canonical_current, session, reason="previous_output_truncated")

    canonical_previous_prompt = _render_token_ids(
        session.last_prompt_messages,
        tokenizer,
        tools=session.last_tools,
        add_generation_prompt=True,
        chat_template_kwargs=session.last_chat_template_kwargs,
    )
    canonical_boundary = _render_token_ids(
        expected_history,
        tokenizer,
        tools=session.last_tools,
        add_generation_prompt=False,
        chat_template_kwargs=session.last_chat_template_kwargs,
    )
    if not _starts_with(canonical_boundary, canonical_previous_prompt):
        return _full_render_build(canonical_current, session, reason="canonical_response_not_appendable")
    canonical_response = canonical_boundary[len(canonical_previous_prompt) :]
    response_end = _last_subsequence_end(canonical_current, canonical_response)
    if response_end is None:
        return _full_render_build(canonical_current, session, reason="canonical_response_not_found")
    terminal_id = int(session.last_output_ids[-1])
    configured_stops = session.sampling_defaults.get("stop_token_ids")
    if isinstance(configured_stops, (list, tuple, set)) and configured_stops:
        try:
            stop_ids = {int(token_id) for token_id in configured_stops}
        except (TypeError, ValueError):
            stop_ids = set()
        if stop_ids and terminal_id not in stop_ids:
            return _full_render_build(canonical_current, session, reason="previous_terminal_not_configured_stop")

    try:
        terminal_index = len(canonical_response) - 1 - canonical_response[::-1].index(terminal_id)
    except ValueError:
        return _full_render_build(canonical_current, session, reason="terminal_not_in_canonical_response")

    # A template may place structural whitespace after its stop token (Gemma's
    # <turn|> is followed by a newline).  Keep that bridge masked, then append
    # only the newly rendered tool/environment/generation-prefix suffix.
    boundary_bridge = canonical_response[terminal_index + 1 :]
    canonical_tail = canonical_current[response_end:]
    exact_history = [*session.last_prompt_ids, *session.last_output_ids]
    prompt_ids = [*exact_history, *boundary_bridge, *canonical_tail]
    if not _starts_with(prompt_ids, exact_history):
        session.raw_prefix_violations += 1
        return _full_render_build(canonical_current, session, reason="raw_prefix_violation")

    session.record_prompt_build("raw_append", "append_only_history")
    return PromptBuild(prompt_ids=prompt_ids, mode="raw_append", reason="append_only_history")


def _remember_turn_lineage(
    session: Session,
    *,
    prompt_messages: list[dict],
    response_message: dict,
    turn: TurnRecord,
    tools: list[dict] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> None:
    session.last_prompt_messages = copy.deepcopy(prompt_messages)
    session.last_response_message = copy.deepcopy(response_message)
    session.last_prompt_ids = list(turn.prompt_ids)
    session.last_output_ids = list(turn.output_ids)
    session.last_finish_reason = str(turn.finish_reason or "")
    session.last_tools = copy.deepcopy(tools)
    session.last_chat_template_kwargs = copy.deepcopy(chat_template_kwargs or {})


def flatten_content(c: Any) -> str:
    """Flatten a wire content value into a chat-template string.

    Handles both Anthropic and OpenAI block shapes. A non-list value (str /
    dict / other) is returned via str() unchanged.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    return "\n".join(p for p in parts if p)


def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


def manager_finish_reason(tool_uses: list[dict], raw_finish: str) -> str:
    """Finish reason stored on the manager turn: tool_calls if the turn called a
    tool, else the raw sglang finish."""
    return "tool_calls" if tool_uses else (raw_finish or "stop")


class BaseAdapter:
    """Base HTTP adapter: session lifecycle plus the shared one-turn pipeline.

    See the module docstring for the class attributes and hooks a subclass must
    supply; everything else is inherited.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    log_prefix: str = "adapter"
    # body keys that cap max_new_tokens and carry stop sequences, in priority order
    max_token_keys: tuple[str, ...] = ()
    stop_keys: tuple[str, ...] = ()
    manager: Any

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        enable_realign: bool = True,
        prune_unacknowledged_empty_retries: bool = False,
        debug_callback: Callable[..., None] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)

        # one manager shared across all sids; per-sid trees live inside it.
        # fork_threshold_tokens left None means the manager uses its own default.
        mgr_kwargs: dict[str, Any] = {
            "enable_realign": enable_realign,
            "prune_unacknowledged_empty_retries": prune_unacknowledged_empty_retries,
        }
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)

        self.debug_callback: Callable[..., None] | None = debug_callback
        # per-sid turn cap: return 429 to kill the run once exceeded
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}

        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- wire hooks (subclass overrides) -------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register the protocol's POST route(s) and bind self._run_turn."""
        raise NotImplementedError

    def _session_id(self, request: web.Request, body: dict) -> str:
        raise NotImplementedError

    def _preprocess_body(self, body: dict) -> None:
        """Mutate the parsed body in place before sid resolution (default no-op)."""

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Return (chat_messages, tools_schema) from the wire body."""
        raise NotImplementedError

    def _build_reply(self, parsed, raw_finish: str, translated: list[dict], tools_schema: list[dict] | None) -> Reply:
        """Pack parsed model output into a Reply."""
        raise NotImplementedError

    async def _respond(
        self,
        request: web.Request,
        body: dict,
        reply: Reply,
        in_tok: int,
        out_tok: int,
        stream: bool,
    ) -> web.StreamResponse:
        raise NotImplementedError

    # -- session lifecycle ---------------------------------------------------

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
        routing_key: str | None = None,
    ) -> None:
        """Register a fresh per-sid Session; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
            routing_key=(str(routing_key).strip() if routing_key else sid),
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        """Mark a sid closed and drain its in-flight turn tasks."""
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return

        async def _drain() -> None:
            done, pending = await asyncio.wait(tasks, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                cancelled, stubborn = await asyncio.wait(pending, timeout=wait_timeout)
                done.update(cancelled)
                if stubborn:
                    self.logger.warning(
                        "[%s] sid=%s leaving %d cancellation-resistant request task(s) behind after %.1fs grace",
                        self.log_prefix,
                        sid,
                        len(stubborn),
                        wait_timeout,
                    )
            for task in done:
                if task.cancelled():
                    continue
                try:
                    task.exception()
                except (asyncio.CancelledError, Exception):
                    pass

        loop = tasks[0].get_loop()
        drain_future = asyncio.run_coroutine_threadsafe(_drain(), loop)
        try:
            await asyncio.wait_for(
                asyncio.wrap_future(drain_future),
                timeout=max(1.0, 2.0 * wait_timeout + 1.0),
            )
        except TimeoutError:
            drain_future.cancel()
            self.logger.warning(
                "[%s] sid=%s shutdown drain exceeded outer timeout; continuing cleanup",
                self.log_prefix,
                sid,
            )
        except Exception:
            self.logger.exception("[%s] sid=%s shutdown drain failed", self.log_prefix, sid)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list:
        """Drain a session's trajectory into fully-formed Sample objects.

        Waits out in-flight requests for the sid, linearises the per-sid tree,
        then decodes each sample's trained tail into .response (the manager is
        tokenizer-free, so the adapter that owns the tokenizer fills this in).
        Idempotent: a second call for an already-popped sid returns [].
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        session = self.store.pop(sid, None)
        max_sample_tokens = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0
        trajectory_metadata = dict(extra_metadata or {})
        if session is not None:
            trajectory_metadata["adapter_token_lineage"] = session.lineage_metrics()
        samples = self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=trajectory_metadata,
            max_sample_tokens=max_sample_tokens,
        )
        for s in samples:
            rlen = int(s.response_length or 0)
            s.response = (
                self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
            )
        return samples

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        self.store.pop(sid, None)
        self.manager.drop_session(sid)

    def session_lineage_metrics(self, sid: str) -> dict[str, Any]:
        session = self.store.get(sid)
        return session.lineage_metrics() if isinstance(session, Session) else {}

    # -- shared request pipeline ---------------------------------------------

    def _check_turn_cap(self, sid: str) -> web.Response | None:
        """Enforce max_turns_per_sid, returning a 429 response once exceeded.

        Increments the per-sid counter as a side effect when under the cap.
        """
        cap = self.max_turns_per_sid
        if cap is None:
            return None
        prior = self._sid_turn_count.get(sid, 0)
        if prior >= cap:
            self.logger.warning("[%s] sid=%s exceeded max_turns_per_sid=%d; killing run", self.log_prefix, sid, cap)
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (f"adapter: sid {sid!r} exceeded max_turns_per_sid={cap}; killing run"),
                    }
                },
                status=429,
            )
        self._sid_turn_count[sid] = prior + 1
        return None

    def _run_debug_callback(self, sid, translated, tools_schema, manager_message, turn) -> None:
        """Run the optional debug-only data dump callback; unset in production."""
        callback = self.debug_callback
        if callback is None:
            return
        try:
            callback(sid, translated, tools_schema, manager_message, turn)
        except Exception:
            self.logger.exception("debug_callback failed (sid=%s)", sid)

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One full agent turn: translate -> sglang -> parse -> append -> respond.

        The wire-specific steps are delegated to the subclass hooks; the rest
        (sid resolution, closed/cap guards, inflight tracking, record_turn) is
        shared across protocols.
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:  # session drained; refuse stragglers
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            async with s.lineage_lock:
                translated, tools_schema = self._translate(body)
                template_kwargs = _chat_template_kwargs(body)
                prompt_build = _build_prompt_ids(
                    translated,
                    tok,
                    s,
                    tools=tools_schema,
                    chat_template_kwargs=template_kwargs,
                )
                prompt_ids = prompt_build.prompt_ids

                turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)

                raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
                parsed = parse_model_output(
                    raw_output,
                    tools_schema=tools_schema,
                    tool_parser_name=self.tool_parser,
                    reasoning_parser_name=self.reasoning_parser,
                    reasoning_started_in_prompt=_reasoning_started_in_prompt(
                        prompt_ids,
                        tok,
                        self.reasoning_parser,
                    ),
                )
                reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
                turn = dataclasses.replace(
                    turn,
                    ill_formed=parsed.ill_formed,
                    segment_boundary_reason=prompt_build.segment_boundary_reason,
                )

                in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
                stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

                # Flush the response before recording the trajectory: a client that
                # disconnected during generation makes _respond raise here, and we
                # must not record a turn the client never received.
                try:
                    response = await self._respond(request, body, reply, in_tok, out_tok, stream)
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

                self._run_debug_callback(
                    sid,
                    translated,
                    tools_schema,
                    reply.manager_message,
                    turn,
                )

                self.manager.record_turn(
                    sid,
                    turn=turn,
                    prompt_messages=translated,
                    response_message=reply.manager_message,
                    metadata={
                        "sid": sid,
                        "prompt_build_mode": prompt_build.mode,
                        "prompt_build_reason": prompt_build.reason,
                    },
                )
                _remember_turn_lineage(
                    s,
                    prompt_messages=translated,
                    response_message=reply.manager_message,
                    turn=turn,
                    tools=tools_schema,
                    chat_template_kwargs=template_kwargs,
                )
                return response
        finally:
            self.inflight.get(sid, set()).discard(task)


def sid_from_bearer(request: web.Request) -> str | None:
    """sid from the Authorization: Bearer <sid> header, or None if absent."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def sid_from_body(body: dict | None) -> str | None:
    """sid from the OpenAI-shape body (metadata.session_id / user), or None."""
    if not body:
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata["session_id"])
    if body.get("user"):
        return str(body["user"])
    return None


def _chat_template_kwargs(body: dict | None) -> dict[str, Any]:
    """Return chat-template kwargs forwarded by OpenAI-compatible clients.

    Nanobot puts SGLang's Qwen thinking controls in extraBody, which arrives as
    top-level OpenAI request fields. The old ClawRL gateway forwarded those
    fields to SGLang /v1/chat/completions; the local /generate adapter must pass
    them into apply_chat_template to render the same prompt.
    """

    if not body:
        return {}
    merged: dict[str, Any] = {}
    for container in (body, body.get("custom_params")):
        if not isinstance(container, dict):
            continue
        for key in ("chat_template_kwargs", "apply_chat_template_kwargs"):
            value = container.get(key)
            if isinstance(value, dict):
                merged.update(value)
    return merged


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp


async def call_sglang_generate(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    *,
    adapter: BaseAdapter,
    session_id: str | None = None,
) -> TurnRecord:
    """POST one turn to sglang /generate and pack the reply into a TurnRecord.

    Module-level (not a method) so tests can monkeypatch it.
    """
    logger = adapter.logger
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] sid=%s prompt exceeds max_context_tokens (%d >= %d)",
                adapter.log_prefix,
                session_id,
                len(prompt_ids),
                session.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    sglang_url = adapter.sglang_url
    rid = uuid.uuid4().hex
    routing_key = str(getattr(session, "routing_key", "") or "").strip()
    headers = {"X-SMG-Routing-Key": routing_key} if routing_key else None
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": prompt_ids,
                "sampling_params": sp,
                "return_logprob": True,
            },
            headers=headers,
        ) as r:
            if r.status >= 400:
                text = await r.text()
                logger.warning(
                    "[%s] sid=%s rid=%s sglang upstream %d: %.200s",
                    adapter.log_prefix,
                    session_id,
                    rid,
                    r.status,
                    text,
                )
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            data = await r.json(content_type=None)
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        # free the sglang slot eagerly on client cancel/timeout, else the
        # orphaned generation keeps occupying KV until its own length cap
        logger.debug("[%s] sid=%s rid=%s turn aborted: %s", adapter.log_prefix, session_id, rid, type(e).__name__)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    )


async def _health(request: web.Request) -> web.Response:
    """Handler for /healthz and /v1/models readiness probes."""
    return web.json_response({"ok": True})
