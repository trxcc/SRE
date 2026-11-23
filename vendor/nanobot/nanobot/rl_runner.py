"""RL harness runner for nanobot.

This module wraps the regular nanobot CLI with the small amount of runtime
patching that the ClawGym RL harness needs:

* match ClawArena's nanobot runtime performance patches;
* attach session metadata to OpenAI-compatible requests;
* capture per-turn model/tool traces;
* write ClawArena-compatible clean transcript files.
"""

from __future__ import annotations

import asyncio
import ast
import contextvars
import csv
import difflib
import hashlib
import io
import inspect
import json
import os
import re
import secrets
import shlex
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_RUNNER_DIR = Path(__file__).resolve().parent
_SESSION_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nanobot_rl_session_key",
    default=None,
)
_TURN_TYPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "nanobot_rl_turn_type",
    default="side",
)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
_GEMMA_THOUGHT_CHANNEL_RE = re.compile(r"<\|channel>thought\s*")
_GEMMA_CHANNEL_END_RE = re.compile(r"<channel\|>|<\|channel\|>")
_GEMMA_TURN_END_RE = re.compile(r"(?:<turn\|>|<\|turn>)\s*$")
_GEMMA_STRING_DELIM = '<|"|>'
_SKILLS_RE = re.compile(r"\n---\n\n# (?:Active )?Skills\n.*?(?:</skills>\s*|\Z)", re.DOTALL)
_FALSE_VALUES = {"0", "false", "no", "off"}
_PREMATURE_STOP_RECOVERY_MARKER = "[nanobot-rl-premature-stop-recovery]"
_PREMATURE_STOP_RECOVERY_PROMPT = (
    f"{_PREMATURE_STOP_RECOVERY_MARKER}\n"
    "Your previous response described the next action but did not call a tool. "
    "Continue the task now by calling the appropriate tool. Do not stop with a plan; "
    "only provide a final answer after the requested files/actions are completed."
)
_PREMATURE_ACTION_RE = re.compile(
    r"(?is)\b(?:"
    r"let\s+me|now\s+let\s+me|i\s+need\s+to|i\s+should|"
    r"i\s+must|i(?:'|’)?ll|i\s+will|let(?:'|’)?s|next,?\s+i(?:'|’)?ll"
    r")\b.{0,260}\b(?:"
    r"read|inspect|check|list|open|create|write|generate|construct|build|"
    r"implement|run|execute|rewrite|finish|save|produce|make|parse|load|process|"
    r"fetch|retrieve"
    r")\b"
)
_PREMATURE_MISSING_INPUT_RE = re.compile(
    r"(?is)\b(?:cannot|can't|unable\s+to)\s+(?:complete|proceed|do|perform)\b"
    r".{0,500}\b(?:input/|workspace|local\s+files?|provided\s+(?:files?|data)|not\s+available|missing)\b"
)
_COMPLETION_HINT_RE = re.compile(
    r"(?is)\b(?:"
    r"all\s+(?:requested\s+)?(?:files|outputs|deliverables)\s+(?:have\s+been\s+)?(?:created|written|saved|completed)|"
    r"(?:i(?:'|’)?ve|i\s+have)\s+(?:completed|created|written|saved|updated|generated)|"
    r"task\s+completed|done[.!]?\s*$"
    r")"
)
_GLOB_MAGIC_RE = re.compile(r"[*?\[]")
_READ_FILE_LINE_RE = re.compile(r"^\s*\d+\|\s?")
_READ_FILE_LINE_CAPTURE_RE = re.compile(r"^\s*(\d+)\|\s?(.*)$")
_CONFIG_ENSURE_LINE_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*[:=].+$")
_CONFIG_KEY_VALUE_LINE_RE = re.compile(r"^\s*([A-Za-z_][\w.-]*)\s*([:=])\s*(.*?)\s*$")
_PLACEHOLDER_CONFIG_VALUES = {
    "",
    "...",
    "changeme",
    "none",
    "null",
    "replace-me",
    "replace_me",
    "tbd",
    "todo",
}
_INI_SECTION_RE = re.compile(r"^\s*\[([^\]\n]+)\]\s*$")
_NANOBOT_TOOL_NAMES = {
    "apply_patch",
    "complete_goal",
    "read_file",
    "write_file",
    "edit_file",
    "find_files",
    "list_dir",
    "glob",
    "grep",
    "long_task",
    "notebook_edit",
    "exec",
    "message",
    "spawn",
    "cron",
    "web_search",
    "web_fetch",
    "inspect_workspace_file",
}
_WORKSPACE_INSPECT_MAX_ATTEMPTS = 3
_TRANSIENT_FILESYSTEM_MARKERS = (
    "input/output error",
    "cannot read file data",
    "no such file or directory",
    "stale file handle",
    "transport endpoint is not connected",
    "temporarily unavailable",
    "workspace file not found",
)


def _remove_runner_dir_from_sys_path() -> None:
    """Avoid importing sibling nanobot.py as the top-level nanobot package."""
    runner_dir = str(_RUNNER_DIR)
    cleaned: list[str] = []
    for entry in sys.path:
        try:
            entry_path = Path(entry or os.getcwd()).resolve()
        except Exception:
            cleaned.append(entry)
            continue
        if str(entry_path) == runner_dir:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def _env_flag(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw_value = os.environ.get(name)
        if raw_value is None or not raw_value.strip():
            continue
        return raw_value.strip().lower() not in _FALSE_VALUES
    return default


def _env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw_value = os.environ.get(name)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default
    return default


def _enforce_researchclaw_agentic_judge_budget() -> None:
    """Keep production ResearchClaw judge subprocesses on the 32K/8K budget.

    The long-lived Polar rollout manager writes each judge config before it
    starts this short-lived runner.  Enforcing the production budget here also
    makes a boundary-time configuration change effective for subsequently
    launched judges without restarting the model or mixing an in-flight judge
    invocation with a new budget.
    """

    if os.environ.get("RESEARCHCLAW_AGENTIC_JUDGE") != "1":
        return
    config_raw = os.environ.get("NANOBOT_CONFIG", "").strip()
    if not config_raw:
        return
    config_path = Path(config_raw)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        defaults = payload["agents"]["defaults"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return
    if not isinstance(defaults, dict):
        return

    context_window = 32_768
    max_tokens = 8_192
    if (
        defaults.get("contextWindowTokens") == context_window
        and defaults.get("maxTokens") == max_tokens
    ):
        return
    defaults["contextWindowTokens"] = context_window
    defaults["maxTokens"] = max_tokens
    temporary = config_path.with_name(f".{config_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, config_path)
    print(
        "[nanobot_rl_runner] ResearchClaw agentic judge budget: "
        f"context_window={context_window} max_tokens={max_tokens}",
        file=sys.stderr,
    )


def _log_root() -> Path | None:
    value = (
        os.environ.get("NANOBOT_LOG_DIR")
        or os.environ.get("CLAWARENA_NANOBOT_LOG_DIR")
        or ""
    ).strip()
    return Path(value) if value else None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _filename_ts() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond // 1000:03d}Z"


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return str(value)


def _split_gemma_thought_channel(text: str | None) -> tuple[str | None, str | None]:
    if not isinstance(text, str) or not text:
        return None, text
    reasoning_parts: list[str] = []
    visible_parts: list[str] = []
    cursor = 0
    while match := _GEMMA_THOUGHT_CHANNEL_RE.search(text, cursor):
        visible_parts.append(text[cursor : match.start()])
        end = _GEMMA_CHANNEL_END_RE.search(text, match.end())
        if end is None:
            reasoning = text[match.end() :].strip()
            if reasoning:
                reasoning_parts.append(reasoning)
            cursor = len(text)
            break
        reasoning = text[match.end() : end.start()].strip()
        if reasoning:
            reasoning_parts.append(reasoning)
        cursor = end.end()
    if not reasoning_parts and not visible_parts:
        visible = _GEMMA_TURN_END_RE.sub("", text).strip() or None
        return None, visible
    visible_parts.append(text[cursor:])
    reasoning = "\n\n".join(reasoning_parts) or None
    visible = _GEMMA_TURN_END_RE.sub("", "".join(visible_parts)).strip() or None
    return reasoning, visible


def _clean_gemma_visible_control_markers(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    cleaned = _GEMMA_CHANNEL_END_RE.sub("", text)
    cleaned = cleaned.replace("<|channel>", "")
    cleaned = _GEMMA_TURN_END_RE.sub("", cleaned).strip()
    return cleaned or None


def _strip_think(text: str | None) -> str | None:
    if not text:
        return None
    _, text = _split_gemma_thought_channel(text)
    if not text:
        return None
    stripped = _THINK_RE.sub("", text).strip()
    return stripped or None


def _trim_skills_block(text: str | None) -> str | None:
    if not text:
        return text
    trimmed = _SKILLS_RE.sub("\n", text)
    return trimmed.strip() + "\n" if trimmed.strip() else None


def _sanitize_messages(messages: Any) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") not in {"system", "developer"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            trimmed = _trim_skills_block(content)
            if trimmed is not None:
                message["content"] = trimmed


def _patch_sanitize_model_messages() -> bool:
    """Trim RL-only prompt sections from the copy actually sent to the model.

    ``AgentRunner.run`` builds ``messages_for_model`` before invoking iteration
    hooks.  Sanitizing ``AgentHookContext.messages`` alone therefore updates the
    persisted history too late for the first request.  ``_snip_history`` is the
    last common governance step that returns the request copy, so patching its
    result keeps tracing and the served prompt consistent without mutating the
    generic nanobot runtime outside this RL wrapper.
    """
    try:
        from nanobot.agent import runner as runner_mod
    except Exception:
        return False

    runner_cls = runner_mod.AgentRunner
    if getattr(runner_cls, "_nanobot_rl_model_messages_sanitized", False):
        return False
    original_snip_history = runner_cls._snip_history

    def _patched_snip_history(self, *args: Any, **kwargs: Any):
        messages_for_model = original_snip_history(self, *args, **kwargs)
        _sanitize_messages(messages_for_model)
        return messages_for_model

    runner_cls._snip_history = _patched_snip_history
    runner_cls._nanobot_rl_model_messages_sanitized = True
    return True


def _patch_strip_max_tokens() -> bool:
    try:
        from nanobot.providers import openai_compat_provider
    except Exception:
        return False

    provider_cls = openai_compat_provider.OpenAICompatProvider
    if getattr(provider_cls, "_nanobot_rl_strip_max_tokens_patched", False):
        return False

    original_build_kwargs = provider_cls._build_kwargs

    def _patched_build_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_build_kwargs(self, *args, **kwargs)
        if isinstance(payload, dict):
            payload.pop("max_tokens", None)
        return payload

    provider_cls._build_kwargs = _patched_build_kwargs
    provider_cls._nanobot_rl_strip_max_tokens_patched = True
    return True


def _call_arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
) -> Any:
    if name in kwargs:
        return kwargs.get(name)
    if len(args) > position:
        return args[position]
    return None


def _patch_sglang_gemma_thinking_body() -> bool:
    try:
        from nanobot.providers import openai_compat_provider
    except Exception:
        return False

    provider_cls = openai_compat_provider.OpenAICompatProvider
    if getattr(provider_cls, "_nanobot_rl_sglang_gemma_thinking_patched", False):
        return False

    original_build_kwargs = provider_cls._build_kwargs

    def _patched_build_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_build_kwargs(self, *args, **kwargs)
        if not isinstance(payload, dict):
            return payload
        if not _env_flag(("NANOBOT_ENABLE_THINKING", "USER_AGENT_ENABLE_THINKING"), False):
            return payload

        model = _call_arg(args, kwargs, "model", 2) or getattr(self, "default_model", None)
        model_name = str(model or "").lower()
        if "gemma" not in model_name:
            return payload

        reasoning_effort = _call_arg(args, kwargs, "reasoning_effort", 5)
        if isinstance(reasoning_effort, str) and reasoning_effort.strip().lower() in _FALSE_VALUES | {"none", "minimal"}:
            return payload

        extra_body = payload.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        else:
            extra_body = dict(extra_body)
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        else:
            chat_template_kwargs = dict(chat_template_kwargs)
        chat_template_kwargs["enable_thinking"] = True
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        payload["extra_body"] = extra_body

        # SGLang's Gemma chat template consumes chat_template_kwargs; a flat
        # OpenAI-style reasoning_effort field is not needed for this endpoint.
        payload.pop("reasoning_effort", None)
        return payload

    provider_cls._build_kwargs = _patched_build_kwargs
    provider_cls._nanobot_rl_sglang_gemma_thinking_patched = True
    return True


def _iter_message_parts(message: Any) -> list[str]:
    parts: list[str] = []
    if not isinstance(message, dict):
        return parts

    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if text:
                    parts.append(str(text))
            elif part is not None:
                parts.append(json.dumps(part, ensure_ascii=False, default=str))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False, default=str))

    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)

    tool_calls = message.get("tool_calls")
    if tool_calls:
        parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))

    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)

    return parts


def _snapshot_readonly_input_symlinks(actual_path: Path) -> dict[str, Path]:
    """Capture the task-provided top-level input links before the agent runs."""
    input_root = Path(os.path.abspath(str(actual_path / "input")))
    try:
        children = tuple(input_root.iterdir())
    except OSError:
        return {}

    exposed: dict[str, Path] = {}
    for child in children:
        try:
            if child.is_symlink():
                exposed[child.name] = child.resolve(strict=True)
        except OSError:
            continue
    return exposed


def _resolve_snapshotted_readonly_input_path(
    *,
    actual_path: Path,
    raw_path: str,
    exposed_symlinks: dict[str, Path],
) -> Path | None:
    """Resolve a read through a task-provided input link without trusting new links."""
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = actual_path / candidate
    candidate = Path(os.path.abspath(str(candidate)))
    input_root = Path(os.path.abspath(str(actual_path / "input")))
    try:
        relative = candidate.relative_to(input_root)
    except ValueError:
        return None
    if not relative.parts:
        return None

    exposed_root = exposed_symlinks.get(relative.parts[0])
    if exposed_root is None:
        return None
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(exposed_root)
    except (OSError, ValueError):
        return None
    return resolved


class _HFChatTemplateTokenEstimator:
    """Count the exact prompt rendered by a Hugging Face chat template."""

    _MAX_CACHE_ENTRIES = 64

    def __init__(self, tokenizer: Any, source: str):
        self.tokenizer = tokenizer
        self.source = source
        self._cache: dict[tuple[int, bytes], int] = {}

    def _count_rendered(self, rendered: str) -> int:
        encoded = rendered.encode("utf-8")
        key = (len(encoded), hashlib.blake2b(encoded, digest_size=16).digest())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        count = len(token_ids)
        if len(self._cache) >= self._MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = count
        return count

    def prompt_tokens(self, messages: Any, tools: Any = None) -> int:
        message_list = list(messages or [])
        try:
            rendered = self.tokenizer.apply_chat_template(
                message_list,
                tools=list(tools) if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            # OpenAI-compatible traces store function.arguments as a JSON
            # string. Some HF templates (including Qwen3.5) expect a mapping
            # and raise while rendering a history that contains such calls.
            # Normalize only the estimator's private copy; never mutate the
            # actual conversation sent to the provider.
            normalized = self._normalize_tool_call_arguments(message_list)
            if normalized is message_list:
                raise
            rendered = self.tokenizer.apply_chat_template(
                normalized,
                tools=list(tools) if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(rendered, str):
            raise TypeError(
                "tokenizer.apply_chat_template(..., tokenize=False) did not return text"
            )
        return self._count_rendered(rendered)

    @staticmethod
    def _normalize_tool_call_arguments(messages: list[Any]) -> list[Any]:
        normalized_messages = messages
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            normalized_calls = tool_calls
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                call_payload = function if isinstance(function, dict) else tool_call
                arguments = call_payload.get("arguments")
                if not isinstance(arguments, str):
                    continue
                try:
                    parsed = json.loads(arguments)
                except (TypeError, ValueError):
                    parsed = None
                if not isinstance(parsed, dict):
                    # Conservatively preserve the raw text in the rendered
                    # estimate when a tool call was truncated mid-JSON.
                    parsed = {"_raw_arguments": arguments}
                if normalized_messages is messages:
                    normalized_messages = list(messages)
                if normalized_calls is tool_calls:
                    normalized_calls = list(tool_calls)
                    message_copy = dict(message)
                    message_copy["tool_calls"] = normalized_calls
                    normalized_messages[message_index] = message_copy
                call_copy = dict(tool_call)
                if isinstance(function, dict):
                    function_copy = dict(function)
                    function_copy["arguments"] = parsed
                    call_copy["function"] = function_copy
                else:
                    call_copy["arguments"] = parsed
                normalized_calls[call_index] = call_copy
        return normalized_messages

    def message_tokens(self, message: Any) -> int:
        # Per-message sizes are used only to choose an initial suffix. The
        # complete candidate is checked again with prompt_tokens(), which
        # includes exact Qwen role/tool framing and the generation prompt.
        payload = "\n".join(_iter_message_parts(message))
        if not payload:
            return 4
        return max(4, len(self.tokenizer.encode(payload, add_special_tokens=False)) + 4)


def _patch_hf_tokenizer_token_estimates(tokenizer_path: str) -> bool:
    try:
        from nanobot.utils import helpers
    except Exception:
        return False

    if getattr(helpers, "_nanobot_rl_token_estimates_patched", False):
        return False

    try:
        # AutoTokenizer imports the generation package, whose assisted-decoding
        # module conditionally imports sklearn. Some long-lived rollout
        # containers can expose a partial namespace-only sklearn installation
        # (for example sklearn/metrics without sklearn/__init__.py). Transformers
        # then treats the optional dependency as installed and fails before any
        # tokenizer code runs. Tokenization does not use sklearn, so normalize a
        # missing or broken optional install to "unavailable".
        try:
            from sklearn.metrics import roc_curve as _roc_curve  # noqa: F401
        except Exception:
            from transformers import utils as transformers_utils
            from transformers.utils import import_utils as transformers_import_utils

            transformers_import_utils.is_sklearn_available = lambda: False
            transformers_utils.is_sklearn_available = lambda: False

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the exact nanobot prompt tokenizer from "
            f"{tokenizer_path!r}"
        ) from exc
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            f"Exact nanobot prompt tokenizer has no chat_template: {tokenizer_path}"
        )

    estimator = _HFChatTemplateTokenEstimator(
        tokenizer,
        f"hf_chat_template:{tokenizer_path}",
    )

    def _patched_estimate_prompt_tokens(messages: Any, tools: Any = None) -> int:
        return estimator.prompt_tokens(messages, tools)

    def _patched_estimate_message_tokens(message: Any) -> int:
        return estimator.message_tokens(message)

    def _patched_estimate_prompt_tokens_chain(
        provider: Any,
        model: str,
        messages: Any,
        tools: Any = None,
    ):
        return estimator.prompt_tokens(messages, tools), estimator.source

    helpers.estimate_prompt_tokens = _patched_estimate_prompt_tokens
    helpers.estimate_message_tokens = _patched_estimate_message_tokens
    helpers.estimate_prompt_tokens_chain = _patched_estimate_prompt_tokens_chain
    helpers._nanobot_rl_token_estimates_patched = True
    helpers._nanobot_rl_hf_tokenizer_estimates_patched = True
    return True


def _patch_disable_tiktoken_token_estimates() -> bool:
    try:
        from nanobot.utils import helpers
    except Exception:
        return False

    if getattr(helpers, "_nanobot_rl_token_estimates_patched", False):
        return False

    payload_cache: dict[int, int] = {}

    def _rough_token_count(text: str) -> int:
        key = hash(text)
        cached = payload_cache.get(key)
        if cached is not None:
            return cached
        result = max(1, len(text.encode("utf-8")) // 4)
        payload_cache[key] = result
        return result

    def _patched_estimate_prompt_tokens(messages: Any, tools: Any = None) -> int:
        parts: list[str] = []
        for message in messages or []:
            parts.extend(_iter_message_parts(message))
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False, default=str))
        return _rough_token_count("\n".join(parts)) + len(messages or []) * 4

    def _patched_estimate_message_tokens(message: Any) -> int:
        payload = "\n".join(_iter_message_parts(message))
        return max(4, _rough_token_count(payload) + 4) if payload else 4

    def _patched_estimate_prompt_tokens_chain(provider: Any, model: str, messages: Any, tools: Any = None):
        provider_counter = getattr(provider, "estimate_prompt_tokens", None)
        if callable(provider_counter):
            try:
                tokens, source = provider_counter(messages, tools, model)
                if isinstance(tokens, (int, float)) and tokens > 0:
                    return int(tokens), str(source or "provider_counter")
            except Exception:
                pass
        return _patched_estimate_prompt_tokens(messages, tools), "char_heuristic"

    helpers.estimate_prompt_tokens = _patched_estimate_prompt_tokens
    helpers.estimate_message_tokens = _patched_estimate_message_tokens
    helpers.estimate_prompt_tokens_chain = _patched_estimate_prompt_tokens_chain
    helpers._nanobot_rl_token_estimates_patched = True
    helpers._nanobot_rl_disable_tiktoken_estimates_patched = True
    return True


def _patch_fast_consolidation_archive() -> bool:
    try:
        from nanobot.agent import memory as memory_mod
    except Exception:
        return False

    consolidator_cls = memory_mod.Consolidator
    if getattr(consolidator_cls, "_nanobot_rl_fast_archive_patched", False):
        return False

    original_archive = consolidator_cls.archive

    async def _patched_archive(self, messages: Any) -> str | None:
        if not messages:
            return None
        try:
            self.store.raw_archive(messages)
            return "[raw archive]"
        except Exception:
            return await original_archive(self, messages)

    consolidator_cls.archive = _patched_archive
    consolidator_cls._nanobot_rl_fast_archive_patched = True
    return True


def _patch_disable_consolidation() -> bool:
    try:
        from nanobot.agent import memory as memory_mod
    except Exception:
        return False

    consolidator_cls = memory_mod.Consolidator
    if getattr(consolidator_cls, "_nanobot_rl_disable_consolidation_patched", False):
        return False

    async def _patched_maybe_consolidate_by_tokens(self, session: Any, *args: Any, **kwargs: Any) -> None:
        return None

    consolidator_cls.maybe_consolidate_by_tokens = _patched_maybe_consolidate_by_tokens
    consolidator_cls._nanobot_rl_disable_consolidation_patched = True
    return True


def _patch_disabled_tools() -> bool:
    raw = (
        os.environ.get("NANOBOT_RL_DISABLED_TOOLS")
        or os.environ.get("CLAWARENA_NANOBOT_DISABLED_TOOLS")
        or ""
    )
    disabled = {name.strip() for name in raw.split(",") if name.strip()}
    if not disabled:
        return False
    try:
        from nanobot.agent import loop as loop_mod
    except Exception:
        return False

    loop_cls = loop_mod.AgentLoop
    if getattr(loop_cls, "_nanobot_rl_disabled_tools_patched", False):
        return False
    original_register = loop_cls._register_default_tools

    def _patched_register(self) -> None:
        original_register(self)
        for name in disabled:
            self.tools.unregister(name)

    loop_cls._register_default_tools = _patched_register
    loop_cls._nanobot_rl_disabled_tools_patched = True
    return True


def _patch_exec_path_prepend() -> bool:
    path = (
        os.environ.get("NANOBOT_RL_EXEC_PATH_PREPEND")
        or os.environ.get("CLAWARENA_NANOBOT_EXEC_PATH_PREPEND")
        or ""
    ).strip()
    if not path:
        return False
    try:
        from nanobot.agent.tools import shell as shell_mod
    except Exception:
        return False

    exec_cls = shell_mod.ExecTool
    if getattr(exec_cls, "_nanobot_rl_path_prepend_patched", False):
        return False
    original_spawn = exec_cls._spawn
    quoted_path = shlex.quote(path)

    async def _patched_spawn(
        command: str,
        cwd: str,
        env: dict[str, str],
        shell_program: str | None = None,
        login: bool = True,
    ):
        command = f'export PATH={quoted_path}:"$PATH"; {command}'
        return await original_spawn(command, cwd, env, shell_program, login)

    exec_cls._spawn = staticmethod(_patched_spawn)
    exec_cls._nanobot_rl_path_prepend_patched = True
    return True


def _workspace_inspector_tool_class() -> type[Any]:
    from nanobot.agent.tools.base import Tool, tool_parameters
    from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema

    @tool_parameters(
        tool_parameters_schema(
            path=StringSchema(
                "Workspace-relative path to inspect, usually under input/ or an output path."
            ),
            max_chars=IntegerSchema(
                12000,
                description="Maximum extracted preview characters (default 12000).",
                minimum=1000,
                maximum=100000,
            ),
            required=["path"],
        )
    )
    class WorkspaceInspectTool(Tool):
        """Read-only structured inspection for WSClaw workspace files."""

        _scopes = {"core"}
        _plugin_discoverable = False

        def __init__(self, workspace: Path):
            self.workspace = Path(workspace)

        @property
        def name(self) -> str:
            return "inspect_workspace_file"

        @property
        def description(self) -> str:
            return (
                "Inspect a local workspace file without modifying it. Use this for Office, PDF, "
                "spreadsheet, structured, email, markup, image, archive, and unfamiliar files. "
                "Returns JSON with parser status, extracted content, structure, counts, warnings, "
                "and a truncation flag. Inspect required source files before creating outputs and "
                "inspect generated binary artifacts before finishing."
            )

        @property
        def read_only(self) -> bool:
            return True

        async def execute(
            self,
            path: str | None = None,
            max_chars: int = 12000,
            **kwargs: Any,
        ) -> str:
            del kwargs
            if not path or not path.strip():
                return "Error: path is required."

            root = Path(os.path.abspath(self.workspace))
            raw_path = Path(path.strip()).expanduser()
            candidate = Path(
                os.path.abspath(raw_path if raw_path.is_absolute() else root / raw_path)
            )
            if (os.environ.get("NANOBOT_NATIVE_JUDGE_WORKSPACE") == "1"
                    and not raw_path.is_absolute() and not candidate.exists()):
                candidate_path = root / "workspace" / raw_path
                if candidate_path.is_file():
                    candidate = Path(os.path.abspath(candidate_path))
            try:
                inside_workspace = os.path.commonpath((str(root), str(candidate))) == str(root)
            except ValueError:
                inside_workspace = False
            if not inside_workspace:
                return f"Error: path must stay inside the workspace: {path}"
            for attempt in range(_WORKSPACE_INSPECT_MAX_ATTEMPTS):
                try:
                    if candidate.is_file():
                        break
                except OSError:
                    pass
                if attempt + 1 < _WORKSPACE_INSPECT_MAX_ATTEMPTS:
                    await asyncio.sleep(0.25 * (attempt + 1))
            else:
                return f"Error: workspace file not found: {path}"

            inspector = root / "tools" / "inspect_workspace_file.py"
            try:
                inspector_installed = inspector.is_file()
            except OSError:
                inspector_installed = False
            if not inspector_installed:
                return "Error: workspace inspector is not installed."

            preview_limit = max(1000, min(int(max_chars), 100000))
            for attempt in range(_WORKSPACE_INSPECT_MAX_ATTEMPTS):
                try:
                    process = await asyncio.create_subprocess_exec(
                        "python",
                        str(inspector),
                        (str(candidate.relative_to(root))
                         if (os.environ.get("NANOBOT_NATIVE_JUDGE_WORKSPACE") == "1")
                         else str(candidate)),
                        "--max-chars",
                        str(preview_limit),
                        cwd=str(root),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return "Error: workspace inspection timed out after 120 seconds."
                except Exception as exc:
                    detail = str(exc)
                    transient = any(
                        marker in detail.lower() for marker in _TRANSIENT_FILESYSTEM_MARKERS
                    )
                    if transient and attempt + 1 < _WORKSPACE_INSPECT_MAX_ATTEMPTS:
                        await asyncio.sleep(0.25 * (attempt + 1))
                        continue
                    return f"Error: workspace inspection failed: {exc}"

                stdout_text = stdout.decode("utf-8", errors="replace").strip()
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                detail = stderr_text or stdout_text or "no diagnostic output"
                transient = any(
                    marker in detail.lower() for marker in _TRANSIENT_FILESYSTEM_MARKERS
                )
                structured_error = False
                if stdout_text:
                    try:
                        structured = json.loads(stdout_text)
                    except json.JSONDecodeError:
                        structured = None
                    structured_error = isinstance(structured, dict) and structured.get(
                        "parse_status"
                    ) not in {None, "ok", "partial"}
                if transient and structured_error and attempt + 1 < _WORKSPACE_INSPECT_MAX_ATTEMPTS:
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue
                if process.returncode != 0 or not stdout_text:
                    if transient and attempt + 1 < _WORKSPACE_INSPECT_MAX_ATTEMPTS:
                        await asyncio.sleep(0.25 * (attempt + 1))
                        continue
                    if process.returncode != 0:
                        return (
                            f"Error: workspace inspector exited with {process.returncode}:\n{detail}"
                        )
                    return "Error: workspace inspector returned no JSON output."
                if stderr_text:
                    return f"{stdout_text}\n\nInspector warnings:\n{stderr_text}"
                return stdout_text
            return "Error: workspace inspection failed after transient filesystem retries."

    return WorkspaceInspectTool


def _patch_workspace_inspector_tool() -> bool:
    try:
        from nanobot.agent import loop as loop_mod
    except Exception:
        return False

    loop_cls = loop_mod.AgentLoop
    if getattr(loop_cls, "_nanobot_rl_workspace_inspector_patched", False):
        return False
    original_register = loop_cls._register_default_tools
    tool_cls = _workspace_inspector_tool_class()

    def _patched_register(self) -> None:
        original_register(self)
        inspector = Path(self.workspace) / "tools" / "inspect_workspace_file.py"
        if inspector.is_file():
            self.tools.register(tool_cls(Path(self.workspace)))

    loop_cls._register_default_tools = _patched_register
    loop_cls._nanobot_rl_workspace_inspector_patched = True
    return True


def _patch_workspace_display_and_translation() -> bool:
    actual = (
        os.environ.get("NANOBOT_WORKSPACE_ACTUAL")
        or os.environ.get("CLAWARENA_NANOBOT_WORKSPACE_ACTUAL")
        or ""
    ).strip()
    display = (
        os.environ.get("NANOBOT_WORKSPACE_DISPLAY")
        or os.environ.get("CLAWARENA_NANOBOT_WORKSPACE_DISPLAY")
        or ""
    ).strip()
    if not actual or not display:
        return False

    actual_path = Path(actual).expanduser()
    display_path = Path(display).expanduser()
    readonly_input_symlinks = _snapshot_readonly_input_symlinks(actual_path)
    patched = False

    def _normalize_workspace_alias_path(raw_path: str | None) -> str | None:
        if not raw_path:
            return raw_path
        if os.environ.get("NANOBOT_NATIVE_JUDGE_WORKSPACE") == "1":
            # Native judges expose a real workspace/ child. Preserve it, and
            # accept candidate-relative rubric paths when absent at judge root.
            relative = Path(raw_path)
            if (not relative.is_absolute()
                    and not (actual_path / relative).exists()
                    and (actual_path / "workspace" / relative).exists()):
                return str(Path("workspace") / relative)
            return raw_path
        normalized = raw_path.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized == "workspace":
            return "."
        if normalized.startswith("workspace/"):
            return normalized[len("workspace/") :] or "."
        return raw_path

    def _display_to_actual_path(raw_path: str | None) -> str | None:
        if not raw_path:
            return raw_path
        alias_path = _normalize_workspace_alias_path(raw_path)
        if alias_path != raw_path:
            return alias_path
        try:
            candidate = Path(raw_path).expanduser()
        except Exception:
            return raw_path
        if not candidate.is_absolute():
            return raw_path
        if candidate == display_path:
            return str(actual_path)
        try:
            rel = candidate.relative_to(display_path)
        except ValueError:
            return raw_path
        return str(actual_path / rel)

    def _replace_display_prefix(text: str | None) -> str | None:
        if not text:
            return text
        display_str = str(display_path)
        actual_str = str(actual_path)
        if text == display_str:
            return actual_str
        # Shell commands may use both child paths (``/app/file.py``) and the
        # workspace root itself (``script.py /app /app``).  Translating only
        # the slash-suffixed form leaves the latter behind, so the restricted
        # exec guard correctly—but misleadingly—classifies it as outside the
        # real ephemeral workspace.  Replace complete path tokens while
        # leaving unrelated strings such as ``/application`` untouched.
        token_pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-]){re.escape(display_str)}"
            rf"(?=$|[/\s\"'`;,:|&<>()])"
        )
        return token_pattern.sub(actual_str, text)

    def _resolve_readonly_input_symlink(tool: Any, raw_path: str | None) -> Path | None:
        """Allow read-only tools to follow task-controlled input links.

        Polar's local runtime exposes the complete persona workspace under
        ``workspace/input`` with top-level symlinks.  The normal workspace
        guard correctly rejects the resolved target because it lives on the
        shared AFS outside the ephemeral session directory.  Keep writes
        restricted to the session, but let read-only tools traverse only
        symlink targets that were installed directly below ``input`` by the
        workspace preparer.
        """
        if not raw_path or not bool(getattr(tool, "read_only", False)):
            return None
        translated = _display_to_actual_path(raw_path)
        if not isinstance(translated, str):
            return None
        return _resolve_snapshotted_readonly_input_path(
            actual_path=actual_path,
            raw_path=translated,
            exposed_symlinks=readonly_input_symlinks,
        )

    try:
        from nanobot.agent import context as context_mod
        from nanobot.utils.prompt_templates import render_template
        import platform

        context_cls = context_mod.ContextBuilder
        if not getattr(context_cls, "_nanobot_rl_workspace_display_patched", False):
            def _patched_get_identity(
                self,
                channel: str | None = None,
                workspace: Path | None = None,
                **_: Any,
            ) -> str:
                system = platform.system()
                runtime = (
                    f"{'macOS' if system == 'Darwin' else system} "
                    f"{platform.machine()}, Python {platform.python_version()}"
                )
                return render_template(
                    "agent/identity.md",
                    workspace_path=str(display_path),
                    runtime=runtime,
                    platform_policy=render_template("agent/platform_policy.md", system=system),
                    channel=channel or "",
                )

            context_cls._get_identity = _patched_get_identity
            context_cls._nanobot_rl_workspace_display_patched = True
            patched = True
    except Exception:
        pass

    try:
        from nanobot.agent.tools import filesystem as fs_mod

        if not getattr(fs_mod, "_nanobot_rl_workspace_translation_patched", False):
            original_resolve_path = getattr(fs_mod, "_resolve_path", None)

            if callable(original_resolve_path):
                def _patched_resolve_path(path, workspace=None, allowed_dir=None, extra_allowed_dirs=None):
                    translated = _display_to_actual_path(path) if isinstance(path, str) else path
                    return original_resolve_path(
                        translated,
                        workspace=workspace,
                        allowed_dir=allowed_dir,
                        extra_allowed_dirs=extra_allowed_dirs,
                    )

                fs_mod._resolve_path = _patched_resolve_path

            fs_tool_cls = getattr(fs_mod, "_FsTool", None)
            original_fs_resolve = getattr(fs_tool_cls, "_resolve", None)
            if callable(original_fs_resolve) and not getattr(fs_tool_cls, "_nanobot_rl_workspace_alias_patched", False):
                def _patched_fs_resolve(self, path):
                    translated = _display_to_actual_path(path) if isinstance(path, str) else path
                    try:
                        return original_fs_resolve(self, translated)
                    except PermissionError:
                        resolved = _resolve_readonly_input_symlink(self, translated)
                        if resolved is None:
                            raise
                        return resolved

                fs_tool_cls._resolve = _patched_fs_resolve
                fs_tool_cls._nanobot_rl_workspace_alias_patched = True
            fs_mod._nanobot_rl_workspace_translation_patched = True
            patched = True
    except Exception:
        pass

    try:
        from nanobot.agent.tools import shell as shell_mod

        exec_cls = shell_mod.ExecTool
        if not getattr(exec_cls, "_nanobot_rl_workspace_translation_patched", False):
            original_execute = exec_cls.execute

            async def _patched_execute(
                self,
                command: str | None = None,
                cmd: str | None = None,
                working_dir: str | None = None,
                workdir: str | None = None,
                timeout: int | None = None,
                **kwargs: Any,
            ):
                translated_command = command if command is not None else cmd
                translated_command = (
                    _replace_display_prefix(translated_command)
                    if isinstance(translated_command, str)
                    else translated_command
                )
                raw_working_dir = working_dir if working_dir is not None else workdir
                translated_working_dir = (
                    _display_to_actual_path(raw_working_dir)
                    if isinstance(raw_working_dir, str)
                    else raw_working_dir
                )
                return await original_execute(
                    self,
                    command=translated_command,
                    working_dir=translated_working_dir,
                    timeout=timeout,
                    **kwargs,
                )

            exec_cls.execute = _patched_execute
            exec_cls._nanobot_rl_workspace_translation_patched = True
            patched = True
    except Exception:
        pass

    return patched


def _strip_workspace_prefix_for_pattern(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not (os.environ.get("NANOBOT_NATIVE_JUDGE_WORKSPACE") == "1"):
        if normalized == "workspace":
            return "."
        if normalized.startswith("workspace/"):
            normalized = normalized[len("workspace/") :] or "."
    normalized = normalized.rstrip("/") or "."

    for env_name in (
        "NANOBOT_WORKSPACE_ACTUAL",
        "CLAWARENA_NANOBOT_WORKSPACE_ACTUAL",
        "NANOBOT_WORKSPACE_DISPLAY",
        "CLAWARENA_NANOBOT_WORKSPACE_DISPLAY",
    ):
        root = os.environ.get(env_name)
        if not root:
            continue
        root_normalized = root.strip().replace("\\", "/").rstrip("/")
        if not root_normalized:
            continue
        if normalized == root_normalized:
            return "."
        if normalized.startswith(f"{root_normalized}/"):
            return normalized[len(root_normalized) + 1 :] or "."
    return normalized


def _search_pattern_for_path(pattern: str | None, path: str | None) -> str | None:
    if not isinstance(pattern, str) or not isinstance(path, str):
        return pattern
    pattern_for_compare = _strip_workspace_prefix_for_pattern(pattern)
    path_for_compare = _strip_workspace_prefix_for_pattern(path)
    if not pattern_for_compare or path_for_compare in {"", "."}:
        return pattern

    prefixes = [path_for_compare]
    basename = path_for_compare.rsplit("/", 1)[-1]
    if basename and basename not in prefixes:
        prefixes.append(basename)

    for prefix in sorted(prefixes, key=len, reverse=True):
        if not prefix or prefix == ".":
            continue
        if pattern_for_compare == prefix:
            return "*"
        if pattern_for_compare.startswith(f"{prefix}/"):
            return pattern_for_compare[len(prefix) + 1 :] or "*"
    pattern_parts = [part for part in pattern_for_compare.split("/") if part and part != "."]
    path_parts = [part for part in path_for_compare.split("/") if part and part != "."]
    common_len = 0
    for pattern_part, path_part in zip(pattern_parts, path_parts):
        if pattern_part != path_part:
            break
        common_len += 1
    if common_len > 0 and common_len == len(pattern_parts) - 1:
        return pattern_parts[-1] or pattern
    return pattern


def _zero_depth_glob_patterns(pattern: str) -> list[str]:
    normalized = pattern.strip().replace("\\", "/")
    if "**/" not in normalized:
        return []
    collapsed = normalized.replace("**/", "")
    return [collapsed] if collapsed and collapsed != normalized else []


def _directory_stem_glob_patterns(pattern: str) -> list[str]:
    normalized = pattern.strip().replace("\\", "/")
    if normalized.count("/") != 1:
        return []
    directory, leaf = normalized.split("/", 1)
    if not directory or directory in {".", "**"}:
        return []
    if not leaf.startswith("*."):
        return []
    collapsed = f"{directory}{leaf[1:]}"
    return [collapsed] if collapsed != normalized else []


def _glob_compat_patterns(pattern: str) -> list[str]:
    variants: list[str] = []
    for candidate in (*_zero_depth_glob_patterns(pattern), *_directory_stem_glob_patterns(pattern)):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _split_glob_path(path: str | None) -> tuple[str, str] | None:
    if not isinstance(path, str):
        return None
    normalized = _strip_workspace_prefix_for_pattern(path)
    if not _GLOB_MAGIC_RE.search(normalized):
        return None
    parts = normalized.split("/")
    wildcard_index = None
    for index, part in enumerate(parts):
        if _GLOB_MAGIC_RE.search(part):
            wildcard_index = index
            break
    if wildcard_index is None:
        return None
    base_parts = parts[:wildcard_index]
    suffix_parts = parts[wildcard_index:]
    base = "/".join(base_parts).rstrip("/") or "."
    if normalized.startswith("/") and not base.startswith("/"):
        base = f"/{base}" if base != "." else "/"
    suffix = "/".join(part for part in suffix_parts if part).strip("/") or "*"
    return base, suffix


def _merge_glob_filter(path_suffix: str | None, glob_filter: str | None) -> str | None:
    if not path_suffix:
        return glob_filter
    suffix = path_suffix.strip().replace("\\", "/").strip("/")
    if not suffix:
        return glob_filter
    if not glob_filter:
        return suffix
    glob_normalized = glob_filter.strip().replace("\\", "/").lstrip("/")
    if not glob_normalized:
        return suffix
    return f"{suffix.rstrip('/')}/{glob_normalized}"


def _search_root_missing(tool: Any, path: str | None) -> bool:
    if not isinstance(path, str):
        return False
    resolver = getattr(tool, "_resolve", None)
    if not callable(resolver):
        return False
    try:
        target = resolver(path or ".")
    except Exception:
        return False
    try:
        return not target.exists()
    except Exception:
        return False


def _strip_read_file_line_numbers(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    nonblank = [line for line in lines if line.strip()]
    if not nonblank or not any(_READ_FILE_LINE_RE.match(line) for line in nonblank):
        return text
    return "".join(_READ_FILE_LINE_RE.sub("", line, count=1) for line in lines)


def _parse_read_file_numbered_lines(text: str) -> list[tuple[int, str]] | None:
    parsed: list[tuple[int, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _READ_FILE_LINE_CAPTURE_RE.match(line)
        if match is None:
            return None
        parsed.append((int(match.group(1)), match.group(2)))
    return parsed or None


def _resolve_edit_target(tool: Any, path: str | None) -> Path | None:
    if not isinstance(path, str) or not path.strip():
        return None
    resolver = getattr(tool, "_resolve", None)
    try:
        target = resolver(path) if callable(resolver) else Path(path).expanduser()
    except Exception:
        return None
    return target if isinstance(target, Path) else Path(target)


def _read_edit_target_text(tool: Any, path: str | None) -> str | None:
    target = _resolve_edit_target(tool, path)
    if target is None:
        return None
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return target.read_text(errors="replace")
        except Exception:
            return None
    except Exception:
        return None


def _replace_call_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    next_args = list(args)
    next_kwargs = dict(kwargs)
    if name in next_kwargs:
        next_kwargs[name] = value
    elif len(next_args) > position:
        next_args[position] = value
    else:
        next_kwargs[name] = value
    return tuple(next_args), next_kwargs


def _edit_not_found_result(result: Any) -> bool:
    return isinstance(result, str) and (
        "old_text not found" in result or "No similar text found" in result
    )


def _edit_new_text_already_present(tool: Any, path: str | None, new_text: str) -> bool:
    if not isinstance(path, str) or not path.strip() or not new_text.strip():
        return False
    file_text = _read_edit_target_text(tool, path)
    if file_text is None:
        return False
    normalized_file = file_text.replace("\r\n", "\n")
    normalized_new = new_text.replace("\r\n", "\n")
    if normalized_file.rstrip("\n") == normalized_new.rstrip("\n"):
        return True
    return len(normalized_new.strip()) >= 200 and normalized_new in normalized_file


_JSON_PARSE_MISSING = object()


def _loads_json_text(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except Exception:
        return _JSON_PARSE_MISSING


def _full_file_edit_variant_from_file(
    tool: Any,
    path: str | None,
    variants: list[tuple[str, str]],
) -> tuple[str, str] | None:
    file_text = _read_edit_target_text(tool, path)
    if file_text is None:
        return None
    normalized_file = file_text.replace("\r\n", "\n")
    file_json = _loads_json_text(normalized_file)
    for old_text, new_text in variants:
        normalized_old = old_text.replace("\r\n", "\n")
        if not normalized_old.strip():
            continue
        normalized_new = new_text.replace("\r\n", "\n")
        matches_full_file = normalized_file.rstrip("\n") == normalized_old.rstrip("\n")
        if not matches_full_file and file_json is not _JSON_PARSE_MISSING:
            old_json = _loads_json_text(normalized_old)
            matches_full_file = old_json is not _JSON_PARSE_MISSING and old_json == file_json
        if not matches_full_file:
            continue
        if normalized_file.endswith("\n") and normalized_new and not normalized_new.endswith("\n"):
            normalized_new = f"{normalized_new}\n"
        return normalized_file, normalized_new
    return None


def _text_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def _line_numbered_edit_variant_from_file(
    tool: Any,
    path: str | None,
    old_text: str,
    new_text: str,
) -> tuple[str, str] | None:
    numbered_old = _parse_read_file_numbered_lines(old_text)
    if numbered_old is None:
        return None
    line_numbers = [line_no for line_no, _ in numbered_old]
    if line_numbers != list(range(line_numbers[0], line_numbers[-1] + 1)):
        return None
    file_text = _read_edit_target_text(tool, path)
    if file_text is None:
        return None
    file_lines = file_text.replace("\r\n", "\n").splitlines(keepends=True)
    start = line_numbers[0]
    end = line_numbers[-1]
    if start < 1 or end > len(file_lines):
        return None
    if len(numbered_old) == 1 and r"\n" in numbered_old[0][1]:
        _, expected = numbered_old[0]
        expected_variants = [
            expected,
            _decode_edit_text_if_needed(expected),
            _decode_source_linebreaks(expected),
            _decode_source_linebreaks(_decode_edit_text_if_needed(expected)),
        ]
        for expected_variant in dict.fromkeys(expected_variants):
            expected_lines = expected_variant.splitlines()
            candidate_end = start + len(expected_lines) - 1
            if candidate_end > len(file_lines):
                continue
            actual_lines = [line.rstrip("\n") for line in file_lines[start - 1 : candidate_end]]
            if actual_lines == expected_lines:
                end = candidate_end
                break
        else:
            return None
    else:
        expected_lines: list[str] = []
        actual_lines: list[str] = []
        for line_no, expected in numbered_old:
            actual = file_lines[line_no - 1].rstrip("\n")
            expected_lines.append(expected)
            actual_lines.append(actual)
            if actual != expected:
                expected_block = "\n".join(expected_lines + [item for _, item in numbered_old[len(expected_lines) :]])
                actual_block = "\n".join(
                    [line.rstrip("\n") for line in file_lines[start - 1 : end]]
                )
                if _text_similarity(expected_block, actual_block) < 0.72:
                    return None
                break

    numbered_new = _parse_read_file_numbered_lines(new_text)
    if numbered_new is None:
        replacement = _strip_read_file_line_numbers(new_text.replace("\r\n", "\n"))
    else:
        replacement = "\n".join(line for _, line in numbered_new)
    original = "".join(file_lines[start - 1 : end])
    if original.endswith("\n") and replacement and not replacement.endswith("\n"):
        replacement = f"{replacement}\n"
    return original, replacement


def _parse_csv_rows(text: str) -> list[list[str]] | None:
    if "," not in text:
        return None
    try:
        rows = list(csv.reader(io.StringIO(text.replace("\r\n", "\n"))))
    except Exception:
        return None
    return rows or None


def _csv_equivalent_edit_variant_from_file(
    tool: Any,
    path: str | None,
    variants: list[tuple[str, str]],
) -> tuple[str, str] | None:
    file_text = _read_edit_target_text(tool, path)
    if file_text is None:
        return None
    file_lines = file_text.replace("\r\n", "\n").splitlines(keepends=True)
    if not file_lines:
        return None
    parsed_file_lines: list[list[str] | None] = []
    for line in file_lines:
        parsed = _parse_csv_rows(line.rstrip("\n"))
        parsed_file_lines.append(parsed[0] if parsed and len(parsed) == 1 else None)

    for old_text, new_text in variants:
        old_rows = _parse_csv_rows(old_text)
        if not old_rows or len(old_rows) > len(file_lines):
            continue
        row_count = len(old_rows)
        for start_index in range(0, len(file_lines) - row_count + 1):
            actual_rows = parsed_file_lines[start_index : start_index + row_count]
            if any(row is None for row in actual_rows):
                continue
            if list(actual_rows) != old_rows:
                continue
            original = "".join(file_lines[start_index : start_index + row_count])
            replacement = new_text.replace("\r\n", "\n")
            if original.endswith("\n") and replacement and not replacement.endswith("\n"):
                replacement = f"{replacement}\n"
            return original, replacement
    return None


def _path_looks_config_like(path: str | None) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    return suffix in {".cfg", ".conf", ".ini", ".service", ".yaml", ".yml", ".env"} or name in {
        ".env",
        "env",
    }


def _single_config_ensure_line(old_text: str, new_text: str) -> str | None:
    decoded_old = _decode_edit_text_if_needed(old_text).replace("\r\n", "\n").strip()
    decoded_new = _decode_edit_text_if_needed(new_text).replace("\r\n", "\n").strip()
    if decoded_old != decoded_new or not decoded_new:
        return None
    if "\n" in decoded_new or len(decoded_new) > 240:
        return None
    return decoded_new if _CONFIG_ENSURE_LINE_RE.match(decoded_new) else None


def _insert_line_in_ini_section(file_text: str, line_to_add: str, section_name: str) -> str | None:
    lines = file_text.splitlines(keepends=True)
    section_start: int | None = None
    section_end: int | None = None
    for index, line in enumerate(lines):
        match = _INI_SECTION_RE.match(line.strip())
        if match is None:
            continue
        if section_start is not None:
            section_end = index
            break
        if match.group(1).strip().lower() == section_name.lower():
            section_start = index + 1
    if section_start is None:
        return None
    if section_end is None:
        section_end = len(lines)

    insert_index = section_end
    while insert_index > section_start and not lines[insert_index - 1].strip():
        insert_index -= 1

    key = line_to_add.split("=", 1)[0].strip().lower()
    if key.startswith("restart"):
        for index in range(section_start, section_end):
            existing_key = lines[index].split("=", 1)[0].strip().lower()
            if existing_key == "restart":
                insert_index = index + 1
    elif key in {"cpuquota", "memorymax"}:
        for index in range(section_start, section_end):
            existing_key = lines[index].lstrip("# ").split("=", 1)[0].strip().lower()
            if existing_key in {"cpuquota", "memorymax"}:
                insert_index = max(insert_index, index + 1)

    lines.insert(insert_index, f"{line_to_add}\n")
    return "".join(lines)


def _append_config_line(file_text: str, line_to_add: str) -> str:
    separator = "" if not file_text or file_text.endswith("\n") else "\n"
    return f"{file_text}{separator}{line_to_add}\n"


def _config_ensure_line_edit_variant_from_file(
    tool: Any,
    path: str | None,
    old_text: str,
    new_text: str,
) -> tuple[str, str] | None:
    line_to_add = _single_config_ensure_line(old_text, new_text)
    if line_to_add is None:
        return None
    if not _path_looks_config_like(path):
        return None
    file_text = _read_edit_target_text(tool, path)
    if file_text is None:
        return None
    normalized_file = file_text.replace("\r\n", "\n")
    if any(line.strip() == line_to_add for line in normalized_file.splitlines()):
        return normalized_file, normalized_file

    if Path(path or "").suffix.lower() == ".service":
        inserted = _insert_line_in_ini_section(normalized_file, line_to_add, "Service")
        if inserted is not None:
            return normalized_file, inserted
    if not any(_INI_SECTION_RE.match(line.strip()) for line in normalized_file.splitlines()):
        return normalized_file, _append_config_line(normalized_file, line_to_add)
    return None


def _parse_config_key_value_block(text: str) -> list[tuple[str, str, str]] | None:
    decoded = _decode_edit_text_if_needed(text).replace("\r\n", "\n")
    entries: list[tuple[str, str, str]] = []
    for line in decoded.splitlines():
        if not line.strip():
            continue
        match = _CONFIG_KEY_VALUE_LINE_RE.match(line)
        if match is None:
            return None
        entries.append((match.group(1), match.group(2), match.group(3).strip()))
    if not entries:
        return None
    keys = [key for key, _, _ in entries]
    if len(keys) != len(set(keys)):
        return None
    return entries


def _config_value_is_placeholder(value: str) -> bool:
    stripped = value.strip().strip("\"'").strip()
    lowered = stripped.lower()
    return lowered in _PLACEHOLDER_CONFIG_VALUES or bool(re.fullmatch(r"<[^>\n]+>", stripped))


def _config_placeholder_block_edit_variant_from_file(
    tool: Any,
    path: str | None,
    old_text: str,
    new_text: str,
) -> tuple[str, str] | None:
    if not _path_looks_config_like(path):
        return None
    old_entries = _parse_config_key_value_block(old_text)
    new_entries = _parse_config_key_value_block(new_text)
    if old_entries is None or new_entries is None:
        return None
    if len(old_entries) != len(new_entries) or len(old_entries) < 2:
        return None
    old_keys = [key for key, _, _ in old_entries]
    new_keys = [key for key, _, _ in new_entries]
    if old_keys != new_keys:
        return None
    if not all(_config_value_is_placeholder(value) for _, _, value in old_entries):
        return None

    file_text = _read_edit_target_text(tool, path)
    if file_text is None:
        return None
    normalized_file = file_text.replace("\r\n", "\n")
    lines = normalized_file.splitlines(keepends=True)
    replacements = {key: value for key, _, value in new_entries}
    matched_indexes: dict[str, int] = {}

    for index, line in enumerate(lines):
        body = line.rstrip("\n")
        newline = line[len(body) :]
        match = re.match(
            r"^(\s*)(?:#\s*)?([A-Za-z_][\w.-]*)(\s*[:=]\s*)(.*?)(\s*)$",
            body,
        )
        if match is None:
            continue
        key = match.group(2)
        if key not in replacements:
            continue
        if key in matched_indexes:
            return None
        matched_indexes[key] = index
        lines[index] = f"{match.group(1)}{key}{match.group(3)}{replacements[key]}{match.group(5)}{newline}"

    if set(matched_indexes) != set(old_keys):
        return None
    updated = "".join(lines)
    if updated == normalized_file:
        return None
    return normalized_file, updated


def _call_argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
) -> Any:
    if name in kwargs:
        return kwargs.get(name)
    if len(args) > position:
        return args[position]
    return None


def _decode_edit_text_if_needed(text: str) -> str:
    if not any(marker in text for marker in (r"\n", r"\r", r"\t", r"\"")):
        return text
    decoded = _decode_loose_json_string(text)
    return decoded if isinstance(decoded, str) and decoded else text


def _escape_source_linebreaks(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", r"\n")


def _line_indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _escape_colon_block_linebreaks(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.split("\n")
    if len(lines) <= 1:
        return text
    parts: list[str] = []
    changed = False
    for index, line in enumerate(lines[:-1]):
        next_line = lines[index + 1]
        parts.append(line)
        if (
            line.rstrip().endswith(":")
            and next_line.strip()
            and _line_indent_width(next_line) > _line_indent_width(line)
        ):
            parts.append(r"\n")
            changed = True
        else:
            parts.append("\n")
    parts.append(lines[-1])
    return "".join(parts) if changed else text


def _trim_unbalanced_line_closing_parens(text: str) -> str:
    lines = text.replace("\r\n", "\n").splitlines(keepends=True)
    if not lines:
        return text
    changed = False
    trimmed_lines: list[str] = []
    for line in lines:
        body = line.rstrip("\n")
        eol = line[len(body) :]
        trailing_ws_len = len(body) - len(body.rstrip(" \t"))
        trailing_ws = body[len(body) - trailing_ws_len :] if trailing_ws_len else ""
        core = body[: len(body) - trailing_ws_len] if trailing_ws_len else body
        excess = core.count(")") - core.count("(")
        while excess > 0 and core.endswith(")"):
            core = core[:-1]
            excess -= 1
            changed = True
        trimmed_lines.append(f"{core}{trailing_ws}{eol}")
    return "".join(trimmed_lines) if changed else text


def _escape_accidental_closing_call_quote(text: str) -> str:
    for marker in ('")', "')"):
        if marker not in text:
            continue
        prefix, suffix = text.rsplit(marker, 1)
        quote = marker[0]
        if prefix.endswith("\\"):
            continue
        return f"{prefix}\\{quote}){suffix}"
    return text


def _strip_traceback_caret_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").splitlines(keepends=True)
    if not lines:
        return text
    filtered: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if stripped and set(stripped) == {"^"}:
            changed = True
            continue
        filtered.append(line)
    if not changed:
        return text
    return "".join(filtered)


def _decode_source_linebreaks(text: str) -> str:
    if r"\n" not in text:
        return text
    parts: list[str] = []
    index = 0
    while index < len(text):
        if not text.startswith(r"\n", index):
            parts.append(text[index])
            index += 1
            continue
        after = text[index + 2 :]
        stripped_after = after.lstrip(" \t")
        if stripped_after and not stripped_after.startswith(
            (r"\n", '"', "'", ")", "]", "}", ",", ".", "+")
        ):
            parts.append("\n")
        else:
            parts.append(r"\n")
        index += 2
    return "".join(parts)


def _strip_embedded_edit_text_closer(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed and stripped:
        changed = False
        for suffix in (r"\"", "\\", '"'):
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)]
                changed = True
                break
    return stripped


def _split_embedded_edit_new_text(old_text: str) -> tuple[str, str] | None:
    for marker in ('\\", "new_text": "', '", "new_text": "'):
        index = old_text.find(marker)
        if index == -1:
            continue
        old_raw = _strip_embedded_edit_text_closer(old_text[:index])
        new_raw = _strip_embedded_edit_text_closer(old_text[index + len(marker) :])
        old_decoded = _decode_edit_text_if_needed(old_raw)
        new_decoded = _decode_edit_text_if_needed(new_raw)
        if old_decoded and new_decoded:
            return old_decoded, new_decoded
    return None


def _repair_edit_file_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return arguments
    if "new_text" in arguments:
        return arguments
    old_text = arguments.get("old_text")
    if not isinstance(old_text, str):
        return arguments
    split = _split_embedded_edit_new_text(old_text)
    if split is None:
        return arguments
    repaired = dict(arguments)
    repaired["old_text"], repaired["new_text"] = split
    return repaired


def _edit_retry_text_variants(old_text: str, new_text: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []

    def add(candidate_old: str, candidate_new: str) -> None:
        pair = (candidate_old, candidate_new)
        if pair not in variants:
            variants.append(pair)

    def add_outer_indent(text: str, indent: str) -> str:
        if not text or text[:1].isspace():
            return text
        lines = text.splitlines(keepends=True)
        if not lines:
            return f"{indent}{text}"
        return "".join(f"{indent}{line}" if line.strip() else line for line in lines)

    def add_first_line_indent(text: str, indent: str) -> str:
        if not text or text[:1].isspace():
            return text
        lines = text.splitlines(keepends=True)
        if not lines:
            return f"{indent}{text}"
        for index, line in enumerate(lines):
            if line.strip():
                lines[index] = f"{indent}{line}"
                break
        return "".join(lines)

    def apply_reference_indents(reference: str, text: str) -> str:
        reference_lines = reference.splitlines(keepends=True)
        text_lines = text.splitlines(keepends=True)
        if len(reference_lines) != len(text_lines):
            return text
        adjusted: list[str] = []
        for reference_line, text_line in zip(reference_lines, text_lines):
            if not text_line.strip():
                adjusted.append(text_line)
                continue
            indent = reference_line[: len(reference_line) - len(reference_line.lstrip(" \t"))]
            stripped_text_line = text_line.lstrip(" \t")
            adjusted.append(f"{indent}{stripped_text_line}")
        return "".join(adjusted)

    def add_indented_variants() -> None:
        for base_old, base_new in list(variants):
            if not base_old or base_old[:1].isspace():
                continue
            for indent in ("    ", "        "):
                add(add_outer_indent(base_old, indent), add_outer_indent(base_new, indent))
                first_line_old = add_first_line_indent(base_old, indent)
                add(first_line_old, add_first_line_indent(base_new, indent))
                add(first_line_old, apply_reference_indents(first_line_old, _decode_source_linebreaks(base_new)))

    add(old_text, new_text)
    decoded_old = _decode_edit_text_if_needed(old_text)
    decoded_new = _decode_edit_text_if_needed(new_text)
    add(decoded_old, decoded_new)
    stripped_old = _strip_read_file_line_numbers(decoded_old.replace("\r\n", "\n"))
    stripped_new = _strip_read_file_line_numbers(decoded_new.replace("\r\n", "\n"))
    add(stripped_old, stripped_new)
    stripped_old = _strip_read_file_line_numbers(old_text.replace("\r\n", "\n"))
    stripped_new = _strip_read_file_line_numbers(new_text.replace("\r\n", "\n"))
    add(stripped_old, stripped_new)
    add(_escape_source_linebreaks(old_text), _decode_source_linebreaks(new_text))
    add(_escape_source_linebreaks(decoded_old), _decode_source_linebreaks(new_text))
    add(_escape_colon_block_linebreaks(old_text), _decode_source_linebreaks(new_text))
    add(_escape_colon_block_linebreaks(decoded_old), _decode_source_linebreaks(new_text))
    add(_trim_unbalanced_line_closing_parens(_decode_source_linebreaks(old_text)), _decode_source_linebreaks(new_text))
    add(_trim_unbalanced_line_closing_parens(_decode_source_linebreaks(decoded_old)), _decode_source_linebreaks(new_text))
    add(_escape_accidental_closing_call_quote(old_text), new_text)
    add(_escape_accidental_closing_call_quote(decoded_old), decoded_new)
    add(_strip_traceback_caret_lines(old_text), new_text)
    add(_strip_traceback_caret_lines(decoded_old), decoded_new)
    add_indented_variants()
    return variants


def _patch_edit_file_line_number_fallback() -> bool:
    try:
        from nanobot.agent.tools import filesystem as fs_mod
    except Exception:
        return False

    edit_cls = getattr(fs_mod, "EditFileTool", None)
    if edit_cls is None or getattr(edit_cls, "_nanobot_rl_line_number_fallback_patched", False):
        return False

    original_cast_params = getattr(edit_cls, "cast_params", None)
    original_execute = edit_cls.execute

    if callable(original_cast_params):
        def _patched_cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
            return _repair_edit_file_arguments(original_cast_params(self, params))

        edit_cls.cast_params = _patched_cast_params

    async def _patched_execute(self, *args: Any, _original_execute=original_execute, **kwargs: Any) -> Any:
        result = await _original_execute(self, *args, **kwargs)
        if not _edit_not_found_result(result):
            return result

        old_text = _call_argument(args, kwargs, "old_text", 1)
        new_text = _call_argument(args, kwargs, "new_text", 2)
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            return result
        path = _call_argument(args, kwargs, "path", 0)
        if _edit_new_text_already_present(self, path, new_text):
            return f"Successfully edited {path}"

        retry_variants = _edit_retry_text_variants(old_text, new_text)
        ensure_line_variant = _config_ensure_line_edit_variant_from_file(self, path, old_text, new_text)
        if ensure_line_variant is not None and ensure_line_variant not in retry_variants:
            retry_variants.insert(1, ensure_line_variant)
        placeholder_block_variant = _config_placeholder_block_edit_variant_from_file(
            self, path, old_text, new_text
        )
        if placeholder_block_variant is not None and placeholder_block_variant not in retry_variants:
            retry_variants.insert(1, placeholder_block_variant)
        numbered_variant = _line_numbered_edit_variant_from_file(self, path, old_text, new_text)
        if numbered_variant is not None and numbered_variant not in retry_variants:
            retry_variants.insert(1, numbered_variant)
        full_file_variant = _full_file_edit_variant_from_file(self, path, retry_variants)
        if full_file_variant is not None and full_file_variant not in retry_variants:
            retry_variants.insert(1, full_file_variant)
        csv_variant = _csv_equivalent_edit_variant_from_file(self, path, retry_variants)
        if csv_variant is not None and csv_variant not in retry_variants:
            retry_variants.insert(1, csv_variant)

        for retry_old, retry_new in retry_variants[1:]:
            retry_args, retry_kwargs = _replace_call_argument(args, kwargs, "old_text", 1, retry_old)
            retry_args, retry_kwargs = _replace_call_argument(retry_args, retry_kwargs, "new_text", 2, retry_new)
            retry_result = await _original_execute(self, *retry_args, **retry_kwargs)
            if not _edit_not_found_result(retry_result):
                return retry_result
            retry_path = _call_argument(retry_args, retry_kwargs, "path", 0)
            if _edit_new_text_already_present(self, retry_path, retry_new):
                return f"Successfully edited {retry_path}"
        return result

    edit_cls.execute = _patched_execute
    edit_cls._nanobot_rl_line_number_fallback_patched = True
    return True


def _patch_read_file_offset_zero_compat() -> bool:
    try:
        from nanobot.agent.tools import filesystem as fs_mod
    except Exception:
        return False

    read_cls = getattr(fs_mod, "ReadFileTool", None)
    if read_cls is None or getattr(read_cls, "_nanobot_rl_offset_zero_patched", False):
        return False

    original_cast_params = read_cls.cast_params
    original_execute = read_cls.execute

    def _normalize_offset(params: Any) -> Any:
        if isinstance(params, dict) and params.get("offset") == 0:
            params = dict(params)
            params["offset"] = 1
        return params

    def _patched_cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return _normalize_offset(original_cast_params(self, params))

    async def _patched_execute(self, *args: Any, _original_execute=original_execute, **kwargs: Any) -> Any:
        if kwargs.get("offset") == 0:
            kwargs = dict(kwargs)
            kwargs["offset"] = 1
        return await _original_execute(self, *args, **kwargs)

    read_cls.cast_params = _patched_cast_params
    read_cls.execute = _patched_execute
    read_cls._nanobot_rl_offset_zero_patched = True
    return True


def _patch_read_file_dedup_compat() -> bool:
    try:
        from nanobot.agent.tools import filesystem as fs_mod
    except Exception:
        return False

    patched = False
    read_cls = getattr(fs_mod, "ReadFileTool", None)
    if read_cls is not None and not getattr(read_cls, "_nanobot_rl_force_content_patched", False):
        original_execute = read_cls.execute

        async def _patched_execute(self, *args: Any, _original_execute=original_execute, **kwargs: Any) -> Any:
            if "force" not in kwargs:
                kwargs = dict(kwargs)
                kwargs["force"] = True
            return await _original_execute(self, *args, **kwargs)

        read_cls.execute = _patched_execute
        read_cls._nanobot_rl_force_content_patched = True
        patched = True

    file_state_mod = getattr(fs_mod, "file_state", None)
    is_unchanged = getattr(file_state_mod, "is_unchanged", None)
    if callable(is_unchanged) and not getattr(file_state_mod, "_nanobot_rl_disable_read_dedup_patched", False):
        def _patched_is_unchanged(*args: Any, **kwargs: Any) -> bool:
            return False

        file_state_mod.is_unchanged = _patched_is_unchanged
        file_state_mod._nanobot_rl_disable_read_dedup_patched = True
        patched = True

    return patched


def _duckduckgo_result_url(href: str) -> str:
    import html as html_lib
    from urllib.parse import parse_qs, unquote, urljoin, urlparse

    href = html_lib.unescape(href or "").strip()
    if not href:
        return ""
    parsed = urlparse(href)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    if parsed.scheme:
        return href
    return urljoin("https://duckduckgo.com", href)


def _parse_duckduckgo_html(query: str, body: str, n: int) -> str:
    import html as html_lib

    def _strip_tags(text: str) -> str:
        text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        return html_lib.unescape(text).strip()

    items: list[dict[str, str]] = []
    for block in re.split(r'<div[^>]+class="[^"]*\bresult\b[^"]*"[^>]*>', body):
        link_match = re.search(
            r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.I | re.S,
        )
        if not link_match:
            continue
        snippet_match = re.search(
            r'<a[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</a>',
            block,
            flags=re.I | re.S,
        ) or re.search(
            r'<div[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</div>',
            block,
            flags=re.I | re.S,
        )
        items.append(
            {
                "title": _strip_tags(link_match.group(2)),
                "url": _duckduckgo_result_url(link_match.group(1)),
                "content": _strip_tags(snippet_match.group(1)) if snippet_match else "",
            }
        )
        if len(items) >= n:
            break

    if not items:
        return f"No results for: {query}"
    lines = [f"Results for: {query}\n"]
    for index, item in enumerate(items[:n], 1):
        lines.append(f"{index}. {item['title']}\n   {item['url']}")
        if item["content"]:
            lines.append(f"   {item['content']}")
    return "\n".join(lines)


async def _duckduckgo_html_search(tool: Any, query: str, n: int) -> str:
    import httpx

    headers = {
        "User-Agent": getattr(tool, "user_agent", None)
        or "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    proxy = getattr(tool, "proxy", None)
    client_kwargs = {"timeout": 15.0, "follow_redirects": True}
    if proxy:
        client_kwargs["proxy"] = proxy
    try:
        client = httpx.AsyncClient(**client_kwargs)
    except TypeError:
        client_kwargs.pop("proxy", None)
        client = httpx.AsyncClient(**client_kwargs)
    async with client:
        response = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=headers,
        )
        response.raise_for_status()
    return _parse_duckduckgo_html(query, response.text, n)


def _patch_web_search_duckduckgo_fallback() -> bool:
    try:
        from nanobot.agent.tools import web as web_mod
    except Exception:
        return False

    search_cls = getattr(web_mod, "WebSearchTool", None)
    if search_cls is None or getattr(search_cls, "_nanobot_rl_duckduckgo_fallback_patched", False):
        return False

    original_search = search_cls._search_duckduckgo

    async def _patched_search_duckduckgo(self, query: str, n: int, _original_search=original_search) -> str:
        try:
            result = await _original_search(self, query, n)
        except Exception:
            result = "Error: DuckDuckGo search failed"
        if isinstance(result, str) and "DuckDuckGo search failed" not in result and "DDGS search failed" not in result:
            return result
        try:
            return await _duckduckgo_html_search(self, query, n)
        except Exception:
            return f"No results for: {query}"

    search_cls._search_duckduckgo = _patched_search_duckduckgo
    search_cls._nanobot_rl_duckduckgo_fallback_patched = True
    return True


def _patch_search_path_pattern_compat() -> bool:
    try:
        from nanobot.agent.tools import search as search_mod
    except Exception:
        return False

    patched = False

    original_match_glob = getattr(search_mod, "_match_glob", None)
    if callable(original_match_glob) and not getattr(search_mod, "_nanobot_rl_match_glob_compat_patched", False):
        def _patched_match_glob(rel_path: str, name: str, pattern: str):
            if original_match_glob(rel_path, name, pattern):
                return True
            for alternate in _glob_compat_patterns(pattern):
                if original_match_glob(rel_path, name, alternate):
                    return True
            return False

        search_mod._match_glob = _patched_match_glob
        search_mod._nanobot_rl_match_glob_compat_patched = True
        patched = True

    glob_cls = getattr(search_mod, "GlobTool", None)
    if glob_cls is not None and not getattr(glob_cls, "_nanobot_rl_path_pattern_compat_patched", False):
        original_execute = glob_cls.execute

        async def _patched_glob_execute(
            self,
            pattern: str,
            path: str = ".",
            *args: Any,
            _original_execute=original_execute,
            **kwargs: Any,
        ):
            split_path = _split_glob_path(path)
            if split_path is not None:
                path, path_suffix = split_path
                pattern = _merge_glob_filter(path_suffix, pattern) or pattern
            if _search_root_missing(self, path):
                return "No files found"
            return await _original_execute(
                self,
                _search_pattern_for_path(pattern, path),
                path,
                *args,
                **kwargs,
            )

        glob_cls.execute = _patched_glob_execute
        glob_cls._nanobot_rl_path_pattern_compat_patched = True
        patched = True

    grep_cls = getattr(search_mod, "GrepTool", None)
    if grep_cls is not None and not getattr(grep_cls, "_nanobot_rl_path_pattern_compat_patched", False):
        original_cast_params = getattr(grep_cls, "cast_params", None)
        original_execute = grep_cls.execute

        def _clamp_grep_context(params: Any) -> Any:
            if not isinstance(params, dict):
                return params
            repaired = params
            for key in ("context_before", "context_after"):
                value = repaired.get(key)
                if isinstance(value, bool):
                    continue
                if isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                if isinstance(value, int) and value > 20:
                    if repaired is params:
                        repaired = dict(params)
                    repaired[key] = 20
            return repaired

        if callable(original_cast_params):
            def _patched_grep_cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
                return _clamp_grep_context(original_cast_params(self, params))

            grep_cls.cast_params = _patched_grep_cast_params

        async def _patched_grep_execute(
            self,
            pattern: str,
            path: str = ".",
            *args: Any,
            glob: str | None = None,
            _original_execute=original_execute,
            **kwargs: Any,
        ):
            kwargs = _clamp_grep_context(kwargs)
            split_path = _split_glob_path(path)
            if split_path is not None:
                path, path_suffix = split_path
                glob = _merge_glob_filter(path_suffix, glob)
            if _search_root_missing(self, path):
                return f"No matches found for pattern '{pattern}' in {path}"
            if glob is None:
                return await _original_execute(self, pattern, path, *args, **kwargs)
            return await _original_execute(
                self,
                pattern,
                path,
                *args,
                glob=_search_pattern_for_path(glob, path),
                **kwargs,
            )

        grep_cls.execute = _patched_grep_execute
        grep_cls._nanobot_rl_path_pattern_compat_patched = True
        patched = True

    find_cls = getattr(search_mod, "FindFilesTool", None)
    if find_cls is not None and not getattr(find_cls, "_nanobot_rl_path_pattern_compat_patched", False):
        original_execute = find_cls.execute

        async def _patched_find_files_execute(
            self,
            path: str = ".",
            *args: Any,
            glob: str | None = None,
            _original_execute=original_execute,
            **kwargs: Any,
        ):
            split_path = _split_glob_path(path)
            if split_path is not None:
                path, path_suffix = split_path
                glob = _merge_glob_filter(path_suffix, glob)
            if _search_root_missing(self, path):
                return "No files found"
            if glob is None:
                return await _original_execute(self, path, *args, **kwargs)
            return await _original_execute(
                self,
                path,
                *args,
                glob=_search_pattern_for_path(glob, path),
                **kwargs,
            )

        find_cls.execute = _patched_find_files_execute
        find_cls._nanobot_rl_path_pattern_compat_patched = True
        patched = True

    return patched


_PSEUDO_PREFIXES = (
    '"arguments":',
    '{"arguments":',
    '"command":',
    '{"command":',
    '"name":',
    '{"name":',
    '"action":',
    '{"action":',
    "<tool_call>",
    "<|tool_call>",
)
_PSEUDO_TOOL_NAME_PATTERN = "|".join(
    re.escape(name) for name in sorted(_NANOBOT_TOOL_NAMES, key=len, reverse=True)
)
_PYTHON_STYLE_TOOL_CALL_RE = re.compile(
    rf"(?:^|[\r\n]|<channel\|>|<\|channel\|>|/)[ \t]*"
    rf"(?:print[ \t]*\([ \t]*)?"
    rf"(?P<name>{_PSEUDO_TOOL_NAME_PATTERN})[ \t]*(?P<open>\()",
    re.MULTILINE,
)
_CHANNEL_STYLE_TOOL_CALL_RE = re.compile(
    rf"(?:^|[\r\n]|<channel\|>|<\|channel\|>)[ \t]*"
    rf"(?:request:|call:|(?i:tool[ \t]+call:))?[ \t]*"
    rf"(?P<name>{_PSEUDO_TOOL_NAME_PATTERN})[ \t]*(?P<open>\{{)",
    re.MULTILINE,
)
_PLAIN_SHELL_COMMAND_RE = re.compile(
    r"^(?:bash|sh|python|python3|mkdir|ls|find|grep|rg|sed|awk|cp|mv|git|pytest)\b"
)
_FINCLAW_TOOL_SUBCOMMANDS = {
    "resolve_symbol",
    "get_financials",
    "get_stock_price",
    "get_news",
    "get_company_profile",
    "code_interpreter",
}


def _looks_like_pseudo_tool_call_text(text: str | None) -> bool:
    stripped = str(text or "").strip()
    if not stripped or "\\boxed{" in stripped:
        return False
    if any(stripped.startswith(prefix) for prefix in _PSEUDO_PREFIXES):
        return True
    if "<tool_call>" in stripped or "<|tool_call>" in stripped:
        return True
    if _PYTHON_STYLE_TOOL_CALL_RE.search(stripped):
        return True
    if _CHANNEL_STYLE_TOOL_CALL_RE.search(stripped):
        return True
    if "calling_tool:" in stripped and "{command:" in stripped:
        return True
    if "```json" in stripped and '"command"' in stripped:
        return True
    if "\n" not in stripped and _PLAIN_SHELL_COMMAND_RE.match(stripped):
        return True
    return '"arguments": {' in stripped and ('"command"' in stripped or '"code"' in stripped)


def _find_balanced_call_end(text: str, opener_index: int, opener: str, closer: str) -> int | None:
    """Return the end offset for a call while ignoring delimiters in strings/comments."""
    depth = 0
    quote: str | None = None
    triple = False
    gemma_string = False
    idx = opener_index
    while idx < len(text):
        char = text[idx]
        if opener == "{" and text.startswith(_GEMMA_STRING_DELIM, idx):
            gemma_string = not gemma_string
            idx += len(_GEMMA_STRING_DELIM)
            continue
        if gemma_string:
            idx += 1
            continue
        if quote is not None:
            if char == "\\":
                idx += 2
                continue
            if triple:
                marker = quote * 3
                if text.startswith(marker, idx):
                    quote = None
                    triple = False
                    idx += 3
                    continue
            elif char == quote:
                quote = None
                idx += 1
                continue
            idx += 1
            continue

        if char in {"'", '"'}:
            quote = char
            triple = text.startswith(char * 3, idx)
            idx += 3 if triple else 1
            continue
        if char == "#":
            newline = text.find("\n", idx + 1)
            if newline == -1:
                return None
            idx = newline + 1
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    return None


def _literal_pseudo_python_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        pass
    if isinstance(node, ast.Name):
        names = {"true": True, "false": False, "null": None}
        if node.id in names:
            return names[node.id]
    if isinstance(node, ast.List):
        return [_literal_pseudo_python_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_pseudo_python_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_pseudo_python_value(key): _literal_pseudo_python_value(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    raise ValueError("pseudo-tool argument is not a literal")


def _python_style_tool_call_payloads(text: str | None) -> list[dict[str, Any]]:
    source = str(text or "")
    payloads: list[dict[str, Any]] = []
    hybrid_header = re.search(
        r"(?m)^[ \t]*write_file\([ \t]*path=(?P<quote>[\"'])(?P<path>[^\"']+)(?P=quote),"
        r"[ \t]*content=(?:\"\"\"|''')",
        source,
    )
    hybrid_separator = '<|"|>,path:<|"|>'
    hybrid_closer = '<|"|>}<tool_call|>'
    if hybrid_header is not None and source.rstrip().endswith(hybrid_closer):
        separator_index = source.rfind(hybrid_separator, hybrid_header.end())
        closer_index = source.rfind(hybrid_closer, separator_index + len(hybrid_separator))
        if separator_index != -1 and closer_index != -1:
            tail_path = source[
                separator_index + len(hybrid_separator) : closer_index
            ].strip()
            if tail_path and tail_path == hybrid_header.group("path"):
                payloads.append(
                    {
                        "name": "write_file",
                        "arguments": {
                            "path": tail_path,
                            "content": source[hybrid_header.end() : separator_index],
                        },
                    }
                )
                return payloads
    cursor = 0
    while cursor < len(source):
        match = _PYTHON_STYLE_TOOL_CALL_RE.search(source, cursor)
        if match is None:
            break
        end = _find_balanced_call_end(source, match.start("open"), "(", ")")
        if end is None:
            cursor = match.end("open")
            continue
        expression = source[match.start("name") : end]
        cursor = end
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
            continue
        if node.func.id != match.group("name"):
            continue
        arguments: dict[str, Any] = {}
        try:
            for keyword in node.keywords:
                if keyword.arg is None:
                    raise ValueError("expanded keyword arguments are not supported")
                arguments[keyword.arg] = _literal_pseudo_python_value(keyword.value)
        except (TypeError, ValueError):
            continue
        if node.func.id == "exec" and "command" not in arguments and isinstance(arguments.get("cmd"), str):
            arguments["command"] = arguments.pop("cmd")
        payloads.append({"name": node.func.id, "arguments": arguments})
        if len(payloads) >= 32:
            break
    return payloads


def _channel_style_tool_call_payloads(text: str | None) -> list[dict[str, Any]]:
    source = str(text or "")
    payloads: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(source):
        match = _CHANNEL_STYLE_TOOL_CALL_RE.search(source, cursor)
        if match is None:
            break
        end = _find_balanced_call_end(source, match.start("open"), "{", "}")
        if end is None:
            cursor = match.end("open")
            continue
        payload = _parse_channel_call_payload(source[match.start("name") : end])
        cursor = end
        if payload is not None:
            payloads.append(payload)
        if len(payloads) >= 32:
            break
    return payloads


def _try_load_json(candidate: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(candidate)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _extract_json_string_field(text: str, field: str) -> str | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"])*)"'
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return json.loads(f'"{match.group(1)}"')
    except Exception:
        return match.group(1)


def _is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    pos = idx - 1
    while pos >= 0 and text[pos] == "\\":
        backslashes += 1
        pos -= 1
    return bool(backslashes % 2)


def _find_loose_json_string_span(
    text: str,
    field: str,
    next_field: str | None = None,
) -> tuple[int, int, int, int] | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*(?P<opener>(?:\\?"){{1,3}})'
    match = re.search(pattern, text)
    if not match:
        return None
    opener = match.group("opener")
    quote_count = opener.count('"')
    value_start = match.end()
    replace_start = match.start("opener")
    closer_options = []
    escaped_closer = r"\"" * quote_count
    plain_closer = '"' * quote_count
    if escaped_closer:
        closer_options.append(escaped_closer)
    if plain_closer not in closer_options:
        closer_options.append(plain_closer)

    if next_field:
        boundary = re.search(rf',\s*"{re.escape(next_field)}"\s*:', text[value_start:])
        if boundary:
            comma_index = value_start + boundary.start()
            value_end = comma_index - 1
            while value_end >= value_start and text[value_end].isspace():
                value_end -= 1
            if value_end >= value_start and text[value_end] == '"' and not _is_escaped(text, value_end):
                return replace_start, value_start, value_end, value_end + 1

    candidates: list[tuple[int, int, str]] = []
    for idx in range(value_start, len(text)):
        for closer in closer_options:
            if not text.startswith(closer, idx):
                continue
            if closer.startswith('"') and _is_escaped(text, idx):
                continue
            replace_end = idx + len(closer)
            suffix = text[replace_end:].lstrip()
            if suffix.startswith('"'):
                after_extra_quote = suffix[1:].lstrip()
                if after_extra_quote and after_extra_quote[0] in ",}]":
                    replace_end = len(text) - len(suffix) + 1
                    suffix = after_extra_quote
            if not suffix:
                continue
            if suffix[0] in ",}]":
                candidates.append((idx, replace_end, suffix[0]))
    if not candidates:
        return None
    for value_end, replace_end, delimiter in reversed(candidates):
        if delimiter in "}]":
            return replace_start, value_start, value_end, replace_end
    value_end, replace_end, _delimiter = candidates[-1]
    return replace_start, value_start, value_end, replace_end


def _decode_loose_json_string(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except Exception:
        return (
            raw.replace(r"\\", "\\")
            .replace(r"\"", '"')
            .replace(r"\n", "\n")
            .replace(r"\r", "\r")
            .replace(r"\t", "\t")
        )


def _repair_loose_json_string_fields(text: str) -> str:
    repaired = text
    for field in ("content", "command", "code", "old_text", "new_text"):
        span = _find_loose_json_string_span(
            repaired,
            field,
            next_field="new_text" if field == "old_text" else None,
        )
        if not span:
            continue
        replace_start, value_start, value_end, replace_end = span
        decoded = _decode_loose_json_string(repaired[value_start:value_end])
        repaired = (
            repaired[:replace_start]
            + json.dumps(decoded, ensure_ascii=False)
            + repaired[replace_end:]
        )
    return repaired


def _try_load_loose_json_object(candidate: str | None) -> dict[str, Any] | None:
    if not candidate:
        return None
    loaded = _try_load_json(candidate)
    if loaded is not None:
        return loaded
    stripped = candidate.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    repaired = _repair_loose_json_string_fields(candidate)
    if repaired == candidate:
        return None
    return _try_load_json(repaired)


def _iter_tagged_tool_payloads(text: str | None) -> list[str]:
    stripped = str(text or "").strip()
    if not stripped:
        return []

    tag_pairs = [
        ("<tool_call>", "</tool_call>"),
        ("<|tool_call>", "<tool_call|>"),
    ]
    payloads: list[str] = []
    pos = 0
    while pos < len(stripped):
        matched_pair: tuple[str, str] | None = None
        matched_index: int | None = None
        for opener, closer in tag_pairs:
            next_index = stripped.find(opener, pos)
            if next_index == -1:
                continue
            if matched_index is None or next_index < matched_index:
                matched_index = next_index
                matched_pair = (opener, closer)
        if matched_pair is None:
            break

        opener, closer = matched_pair
        pos = int(matched_index)
        payload_start = pos + len(opener)
        close_index = stripped.find(closer, payload_start)
        if close_index == -1:
            next_openers = [
                next_index
                for next_opener, _ in tag_pairs
                if (next_index := stripped.find(next_opener, payload_start)) != -1
            ]
            payload_end = min(next_openers) if next_openers else len(stripped)
            payloads.append(stripped[payload_start:payload_end].strip())
            break

        payloads.append(stripped[payload_start:close_index].strip())
        pos = close_index + len(closer)

    return payloads


def _parse_channel_call_payload(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("call:"):
        cleaned = cleaned[len("call:") :].strip()
    match = re.match(r"(?P<name>[A-Za-z_][\w.-]*)\s*\{(?P<body>.*)\}\s*$", cleaned, re.DOTALL)
    if not match:
        return None
    name = match.group("name").strip()
    body = match.group("body").strip()
    if not name:
        return None

    args: dict[str, Any] = {}
    pos = 0
    key_re = re.compile(r"\s*(?P<key>[A-Za-z_][\w.-]*)\s*:\s*", re.DOTALL)
    quoted_open = '<|"|>'
    quoted_close = '<|"|>'
    while pos < len(body):
        key_match = key_re.match(body, pos)
        if not key_match:
            break
        key = key_match.group("key")
        value_start = key_match.end()
        if body.startswith(quoted_open, value_start):
            value_start += len(quoted_open)
            value_end = body.find(quoted_close, value_start)
            if value_end == -1:
                value_end = len(body)
            value: Any = body[value_start:value_end]
            pos = value_end + len(quoted_close)
        elif value_start < len(body) and body[value_start] == '"':
            decoder = json.JSONDecoder()
            try:
                value, end = decoder.raw_decode(body[value_start:])
                pos = value_start + end
            except Exception:
                value_end = body.find(",", value_start)
                if value_end == -1:
                    value_end = len(body)
                value = body[value_start:value_end].strip().strip('"')
                pos = value_end
        else:
            value_end = body.find(",", value_start)
            if value_end == -1:
                value_end = len(body)
            raw_value = body[value_start:value_end].strip()
            if raw_value.lower() == "true":
                value = True
            elif raw_value.lower() == "false":
                value = False
            else:
                try:
                    value = int(raw_value)
                except ValueError:
                    try:
                        value = float(raw_value)
                    except ValueError:
                        value = raw_value
            pos = value_end
        args[key.replace("-", "_")] = value
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos < len(body) and body[pos] == ",":
            pos += 1
    if not args and body:
        return None
    return {"name": name, "arguments": args}


def _extract_json_object_field(text: str, field: str) -> str | None:
    marker = f'"{field}"'
    start = text.find(marker)
    if start == -1:
        return None
    brace_start = text.find("{", start)
    if brace_start == -1:
        return None
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : idx + 1]
    return None


def _parse_pseudo_payload(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    tagged_payloads = _iter_tagged_tool_payloads(cleaned)
    if tagged_payloads:
        cleaned = tagged_payloads[0]

    channel_call = _parse_channel_call_payload(cleaned)
    if channel_call is not None:
        return channel_call

    candidates = [cleaned]
    if not cleaned.startswith("{"):
        candidates.append("{" + cleaned + "}")
        candidates.append("{" + cleaned)
    if cleaned.endswith("}}"):
        candidates.append(cleaned[:-1])
    if not cleaned.startswith("{") and cleaned.endswith("}}"):
        candidates.append("{" + cleaned[:-1])

    for candidate in candidates:
        loaded = _try_load_loose_json_object(candidate)
        if loaded is not None:
            return loaded

    name = _extract_json_string_field(cleaned, "name")
    if name is None:
        name = _extract_json_string_field(cleaned, "action")
    arguments_text = _extract_json_object_field(cleaned, "arguments")
    arguments = _try_load_loose_json_object(arguments_text)
    if name is not None and arguments is not None:
        return {"name": name, "arguments": arguments}

    command = _extract_json_string_field(cleaned, "command")
    code = _extract_json_string_field(cleaned, "code")
    params_text = _extract_json_object_field(cleaned, "params")
    params = _try_load_loose_json_object(params_text)
    payload: dict[str, Any] = {}
    if command is not None:
        payload["command"] = command
    if code is not None:
        payload["code"] = code
    if params is not None:
        payload["params"] = params
    return payload or None


def _coerce_tool_arguments(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        loaded = _try_load_loose_json_object(arguments)
        if loaded is not None:
            return loaded
    return None


def _recover_pseudo_tool_call(text: str | None) -> tuple[str, dict[str, Any]] | None:
    if not _looks_like_pseudo_tool_call_text(text):
        return None
    return _recover_pseudo_tool_call_payload(str(text or ""))


def _recover_pseudo_tool_call_payload(text: str) -> tuple[str, dict[str, Any]] | None:
    payload = _parse_pseudo_payload(str(text or ""))
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not name.strip():
        action = payload.get("action")
        if isinstance(action, str) and action.strip():
            name = action
    if not isinstance(name, str) or not name.strip():
        function = payload.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments")
        elif isinstance(function, str) and function.strip():
            name = function
    coerced = _coerce_tool_arguments(arguments)
    if not isinstance(name, str) or not name.strip() or coerced is None:
        return None
    clean_name = name.strip()
    shell_command = _shell_command_from_pseudo_tool(clean_name, coerced)
    if shell_command:
        return "exec", {"command": shell_command}
    if clean_name not in _NANOBOT_TOOL_NAMES:
        return None
    if clean_name == "edit_file":
        coerced = _repair_edit_file_arguments(coerced)
        if not all(isinstance(coerced.get(key), str) for key in ("path", "old_text", "new_text")):
            return None
    return clean_name, coerced


def _recover_pseudo_tool_calls(text: str | None) -> list[tuple[str, dict[str, Any]]]:
    if not _looks_like_pseudo_tool_call_text(text):
        return []

    python_payloads = _python_style_tool_call_payloads(text)
    if python_payloads:
        recovered_python: list[tuple[str, dict[str, Any]]] = []
        for payload in python_payloads:
            tool_call = _recover_pseudo_tool_call_payload(json.dumps(payload, ensure_ascii=False))
            if tool_call is not None:
                recovered_python.append(tool_call)
        if recovered_python:
            return recovered_python

    channel_payloads = _channel_style_tool_call_payloads(text)
    if channel_payloads:
        recovered_channel: list[tuple[str, dict[str, Any]]] = []
        for payload in channel_payloads:
            tool_call = _recover_pseudo_tool_call_payload(json.dumps(payload, ensure_ascii=False))
            if tool_call is not None:
                recovered_channel.append(tool_call)
        if recovered_channel:
            return recovered_channel

    tagged_payloads = _iter_tagged_tool_payloads(text)
    if not tagged_payloads:
        tool_call = _recover_pseudo_tool_call_payload(str(text or ""))
        return [tool_call] if tool_call is not None else []

    recovered: list[tuple[str, dict[str, Any]]] = []
    for payload in tagged_payloads:
        tool_call = _recover_pseudo_tool_call_payload(payload)
        if tool_call is not None:
            recovered.append(tool_call)
    return recovered


def _shell_command_from_pseudo_tool(name: str, arguments: dict[str, Any]) -> str | None:
    command_arg = arguments.get("command")
    if isinstance(command_arg, str) and command_arg.strip():
        head = command_arg.strip().split(None, 1)[0]
        if name == command_arg.strip() or name == head or name.startswith(f"{head} "):
            return command_arg.strip()

    if name == "mkdir":
        path = arguments.get("path") or arguments.get("dir") or arguments.get("directory")
        if isinstance(path, str) and path.strip():
            return f"mkdir -p {shlex.quote(path.strip())}"
        return None

    if name == "ls" or name.startswith("ls "):
        path = arguments.get("path") or arguments.get("dir") or arguments.get("directory") or "."
        if not isinstance(path, str) or not path.strip():
            path = "."
        options = name[2:].strip()
        command = "ls"
        if options:
            command += f" {options}"
        command += f" {shlex.quote(path.strip())}"
        return command

    try:
        command_parts = shlex.split(name)
    except ValueError:
        command_parts = []
    if len(command_parts) == 2 and command_parts[0] in {"python", "python3"}:
        script = Path(command_parts[1])
        if (
            not script.is_absolute()
            and script.parts[:1] == ("tools",)
            and ".." not in script.parts
            and script.suffix == ".py"
        ):
            remaining = dict(arguments)
            path = remaining.pop("path", None)
            command = f"{command_parts[0]} {shlex.quote(str(script))}"
            if isinstance(path, str) and path.strip():
                command += f" {shlex.quote(path.strip())}"
            cli_args = _params_to_cli_args(remaining)
            if cli_args:
                command += f" {cli_args}"
            return command

    return None


def _params_to_cli_args(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in params.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                parts.extend([flag, shlex.quote(str(item))])
            continue
        if isinstance(value, dict):
            parts.extend([flag, shlex.quote(json.dumps(value, ensure_ascii=False))])
            continue
        parts.extend([flag, shlex.quote(str(value))])
    return " ".join(parts)


def _recover_pseudo_exec_command(text: str | None) -> str | None:
    if not _looks_like_pseudo_tool_call_text(text):
        return None
    stripped = str(text or "").strip()
    if "\n" not in stripped and _PLAIN_SHELL_COMMAND_RE.match(stripped):
        try:
            if shlex.split(stripped):
                return stripped
        except ValueError:
            return None

    loose_exec = re.fullmatch(
        r'exec\s*\(\s*(?:command|cmd)\s*=\s*"(?P<command>[\s\S]*)"\s*\)',
        stripped,
    )
    if loose_exec is not None:
        command = _decode_loose_json_string(loose_exec.group("command")).strip()
        try:
            if shlex.split(command):
                return command
        except ValueError:
            pass

    fenced_command = re.search(
        r'```json\s*\{\s*"command"\s*:\s*"(?P<command>[\s\S]*)"\s*\}\s*```\s*$',
        stripped,
    )
    if fenced_command is not None:
        command = _decode_loose_json_string(fenced_command.group("command")).strip()
        try:
            if shlex.split(command):
                return command
        except ValueError:
            pass

    calling_tool = re.search(
        r'calling_tool:exec\{command:"(?P<command>[\s\S]*)\}\s*<tool_call\|>\s*$',
        stripped,
    )
    if calling_tool is not None:
        raw_command = calling_tool.group("command")
        # Gemma sometimes appends a second serialized wrapper (`"}"`) after
        # the actual quoted command. Keep the command's escaped closing quote.
        if raw_command.endswith(r'\""}"'):
            raw_command = raw_command[:-3]
        command = _decode_loose_json_string(raw_command).strip()
        try:
            if shlex.split(command):
                return command
        except ValueError:
            pass
    payload = _parse_pseudo_payload(str(text or ""))
    if not isinstance(payload, dict):
        return None
    arguments = payload.get("arguments")
    if isinstance(arguments, dict):
        payload = arguments

    code = payload.get("code")
    if isinstance(code, str) and code.strip():
        return f"./finclaw_code {shlex.quote(code)}"

    command = payload.get("command")
    params = payload.get("params")
    if not isinstance(command, str) or not command.strip():
        return None
    command = command.strip()
    if command.startswith("./"):
        return command
    if command.startswith("finclaw_tool ") or command.startswith("finclaw_code "):
        return f"./{command}"
    head = command.split(None, 1)[0]
    if isinstance(params, dict) and params:
        command = f"{command} {_params_to_cli_args(params)}".strip()
        head = command.split(None, 1)[0]
    if head in _FINCLAW_TOOL_SUBCOMMANDS:
        return f"./finclaw_tool {command}"
    return None


def _short_tool_call_id() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(9))


def _recover_pseudo_tool_response(response: Any, tool_call_cls: Any) -> Any:
    has_structured_tool_calls = bool(getattr(response, "tool_calls", None))
    content = getattr(response, "content", None)
    reasoning_text, visible_text = _split_gemma_thought_channel(content)
    existing_reasoning = getattr(response, "reasoning_content", None)
    if reasoning_text:
        combined_reasoning = reasoning_text
        if isinstance(existing_reasoning, str) and existing_reasoning.strip():
            combined_reasoning = existing_reasoning.strip()
            if reasoning_text not in combined_reasoning:
                combined_reasoning += "\n\n" + reasoning_text
        try:
            response.reasoning_content = combined_reasoning
        except Exception:
            pass
    clean_visible_text = _clean_gemma_visible_control_markers(visible_text)
    if clean_visible_text != content:
        try:
            response.content = clean_visible_text
        except Exception:
            pass
    if has_structured_tool_calls:
        return response

    # Parse before dropping stray channel closers: for malformed Gemma output,
    # a closer can be the only reliable boundary before a bare ``name{...}``.
    tool_calls = _recover_pseudo_tool_calls(visible_text)
    if tool_calls:
        recovered = [
            tool_call_cls(
                id=_short_tool_call_id(),
                name=name,
                arguments=arguments,
            )
            for name, arguments in tool_calls
        ]
        try:
            response.tool_calls = recovered
        except Exception:
            return response
        try:
            response.finish_reason = "tool_calls"
        except Exception:
            pass
        if reasoning_text and not getattr(response, "reasoning_content", None):
            try:
                response.reasoning_content = reasoning_text
            except Exception:
                pass
        try:
            response.content = None
        except Exception:
            pass
        return response

    command = _recover_pseudo_exec_command(visible_text)
    if not command:
        return response
    recovered = tool_call_cls(
        id=_short_tool_call_id(),
        name="exec",
        arguments={"command": command},
    )
    try:
        response.tool_calls = [recovered]
    except Exception:
        return response
    try:
        response.finish_reason = "tool_calls"
    except Exception:
        pass
    if reasoning_text and not getattr(response, "reasoning_content", None):
        try:
            response.reasoning_content = reasoning_text
        except Exception:
            pass
    try:
        response.content = None
    except Exception:
        pass
    return response


def _patch_pseudo_tool_calls() -> bool:
    try:
        from nanobot.providers import openai_compat_provider
        from nanobot.providers.base import ToolCallRequest
    except Exception:
        return False

    provider_cls = openai_compat_provider.OpenAICompatProvider
    if getattr(provider_cls, "_nanobot_rl_pseudo_tool_calls_patched", False):
        return False

    original_chat = provider_cls.chat
    original_chat_stream = provider_cls.chat_stream

    def _recover_response(response: Any) -> Any:
        return _recover_pseudo_tool_response(response, ToolCallRequest)

    async def _patched_chat(self, *args: Any, **kwargs: Any):
        return _recover_response(await original_chat(self, *args, **kwargs))

    async def _patched_chat_stream(self, *args: Any, **kwargs: Any):
        return _recover_response(await original_chat_stream(self, *args, **kwargs))

    provider_cls.chat = _patched_chat
    provider_cls.chat_stream = _patched_chat_stream
    provider_cls._nanobot_rl_pseudo_tool_calls_patched = True
    return True


def _patch_agent_runner_pseudo_tool_calls() -> bool:
    try:
        from nanobot.agent import runner as runner_mod
        from nanobot.providers.base import ToolCallRequest
    except Exception:
        return False

    runner_cls = runner_mod.AgentRunner
    if getattr(runner_cls, "_nanobot_rl_pseudo_tool_calls_patched", False):
        return False

    original_request_model = runner_cls._request_model

    async def _patched_request_model(self, *args: Any, **kwargs: Any) -> Any:
        response = await original_request_model(self, *args, **kwargs)
        return _recover_pseudo_tool_response(response, ToolCallRequest)

    runner_cls._request_model = _patched_request_model
    runner_cls._nanobot_rl_pseudo_tool_calls_patched = True
    return True


def _patch_agent_runner_nonstreaming_tool_calls() -> bool:
    try:
        from nanobot.agent import runner as runner_mod
    except Exception:
        return False

    runner_cls = runner_mod.AgentRunner
    if getattr(runner_cls, "_nanobot_rl_nonstreaming_tool_calls_patched", False):
        return False

    original_request_model = runner_cls._request_model

    class _NonStreamingHook:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def wants_streaming(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    async def _patched_request_model(
        self,
        spec: Any,
        messages: list[dict[str, Any]],
        hook: Any,
        context: Any,
    ) -> Any:
        if not _env_flag(
            ("NANOBOT_RL_FORCE_NONSTREAM_TOOL_CALLS", "CLAWARENA_NANOBOT_FORCE_NONSTREAM_TOOL_CALLS"),
            True,
        ):
            return await original_request_model(self, spec, messages, hook, context)

        old_stream_progress = getattr(spec, "stream_progress_deltas", None)
        if old_stream_progress is not None:
            try:
                spec.stream_progress_deltas = False
            except Exception:
                pass
        try:
            return await original_request_model(self, spec, messages, _NonStreamingHook(hook), context)
        finally:
            if old_stream_progress is not None:
                try:
                    spec.stream_progress_deltas = old_stream_progress
                except Exception:
                    pass

    runner_cls._request_model = _patched_request_model
    runner_cls._nanobot_rl_nonstreaming_tool_calls_patched = True
    return True


def _looks_like_premature_stop_text(text: str | None) -> bool:
    visible = _strip_think(text)
    if not visible:
        return False
    if _PREMATURE_STOP_RECOVERY_MARKER in visible:
        return False
    compact = " ".join(visible.split())
    if not compact:
        return False
    if _looks_like_pseudo_tool_call_text(compact):
        return True
    if len(compact) > 1600:
        return False
    if _COMPLETION_HINT_RE.search(compact):
        return False
    return bool(
        _PREMATURE_ACTION_RE.search(compact)
        or _PREMATURE_MISSING_INPUT_RE.search(compact)
    )


def _patch_premature_stop_recovery() -> bool:
    try:
        from nanobot.agent import runner as runner_mod
    except Exception:
        return False

    runner_cls = runner_mod.AgentRunner
    if getattr(runner_cls, "_nanobot_rl_premature_stop_recovery_patched", False):
        return False

    original_try_drain = runner_cls._try_drain_injections
    try:
        original_signature = inspect.signature(original_try_drain)
        original_accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in original_signature.parameters.values()
        )
        original_keyword_names = {
            name
            for name, parameter in original_signature.parameters.items()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (TypeError, ValueError):
        original_accepts_kwargs = False
        original_keyword_names = {"phase", "iteration"}

    def _original_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        if original_accepts_kwargs:
            return kwargs
        return {name: value for name, value in kwargs.items() if name in original_keyword_names}

    async def _patched_try_drain_injections(
        self,
        spec: Any,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        phase: str = "after error",
        iteration: int | None = None,
        allow_goal_continue: bool = False,
        **kwargs: Any,
    ):
        passthrough_kwargs = {
            "phase": phase,
            "iteration": iteration,
            "allow_goal_continue": allow_goal_continue,
        }
        passthrough_kwargs.update(kwargs)
        should_continue, updated_cycles = await original_try_drain(
            self,
            spec,
            messages,
            assistant_message,
            injection_cycles,
            **_original_kwargs(passthrough_kwargs),
        )
        if should_continue:
            return should_continue, updated_cycles

        if phase != "after final response" or assistant_message is None:
            return False, updated_cycles
        tools = getattr(spec, "tools", None)
        if tools is None:
            return False, updated_cycles
        content = assistant_message.get("content") if isinstance(assistant_message, dict) else None
        if not _looks_like_premature_stop_text(content if isinstance(content, str) else None):
            return False, updated_cycles

        max_recoveries = _env_int(
            ("NANOBOT_RL_PREMATURE_STOP_MAX_RECOVERIES", "CLAWARENA_NANOBOT_PREMATURE_STOP_MAX_RECOVERIES"),
            3,
        )
        if max_recoveries <= 0:
            return False, updated_cycles
        recovery_count = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            msg_content = message.get("content")
            if isinstance(msg_content, str) and _PREMATURE_STOP_RECOVERY_MARKER in msg_content:
                recovery_count += 1
        if recovery_count >= max_recoveries:
            return False, updated_cycles

        messages.append(assistant_message)
        if iteration is not None:
            try:
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "final_response",
                        "iteration": iteration,
                        "model": getattr(spec, "model", None),
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [],
                    },
                )
            except Exception:
                pass
        messages.append({"role": "user", "content": _PREMATURE_STOP_RECOVERY_PROMPT})
        return True, updated_cycles + 1

    runner_cls._try_drain_injections = _patched_try_drain_injections
    runner_cls._nanobot_rl_premature_stop_recovery_patched = True
    return True


def _extract_tools_snapshot(context: Any) -> list[dict] | None:
    for attr in ("tools", "available_tools", "request_tools", "tool_schemas", "tool_definitions"):
        tools = _jsonable(getattr(context, attr, None))
        if isinstance(tools, list) and tools:
            return tools
    provider_request = _jsonable(getattr(context, "provider_request", None))
    if isinstance(provider_request, dict):
        tools = provider_request.get("tools")
        if isinstance(tools, list) and tools:
            return tools
    return None


def _patch_provider_request_context() -> None:
    try:
        from nanobot.providers import openai_compat_provider
    except Exception:
        return

    provider_cls = openai_compat_provider.OpenAICompatProvider
    if getattr(provider_cls, "_nanobot_rl_request_context_patched", False):
        return

    original_build_kwargs = provider_cls._build_kwargs
    original_build_responses_body = provider_cls._build_responses_body

    def _request_context() -> tuple[dict[str, str], dict[str, str]]:
        session_key = _SESSION_KEY.get()
        if not session_key:
            return {}, {}
        turn_type = _TURN_TYPE.get() or "side"
        return (
            {
                "x-session-id": session_key,
                "x-turn-type": turn_type,
            },
            {
                "session_id": session_key,
                "turn_type": turn_type,
            },
        )

    def _merge_context(payload: dict[str, Any]) -> dict[str, Any]:
        headers, body = _request_context()
        if headers:
            merged_headers = dict(payload.get("extra_headers") or {})
            merged_headers.update(headers)
            payload["extra_headers"] = merged_headers
        if body:
            merged_body = dict(payload.get("extra_body") or {})
            merged_body.update(body)
            payload["extra_body"] = merged_body
        return payload

    def _patched_build_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_build_kwargs(self, *args, **kwargs)
        return _merge_context(payload)

    def _patched_build_responses_body(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = original_build_responses_body(self, *args, **kwargs)
        return _merge_context(payload)

    provider_cls._build_kwargs = _patched_build_kwargs
    provider_cls._build_responses_body = _patched_build_responses_body
    provider_cls._nanobot_rl_request_context_patched = True


def _normalize_tool_calls(tool_calls: Any) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(tool_calls, list):
        return normalized
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            function = {}
        entry: dict[str, Any] = {}
        if tool_call.get("id"):
            entry["id"] = tool_call.get("id")
        name = function.get("name") or tool_call.get("name")
        if name:
            entry["name"] = name
        if "arguments" in function:
            entry["arguments"] = _jsonable(function.get("arguments"))
        elif "arguments" in tool_call:
            entry["arguments"] = _jsonable(tool_call.get("arguments"))
        if entry:
            normalized.append(entry)
    return normalized


def _normalize_tool_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    return _jsonable(content)


def _build_clean_transcript(payload: dict[str, Any], session_key: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []

    input_text = payload.get("input")
    if isinstance(input_text, str) and input_text.strip():
        messages.append({"role": "user", "content": input_text})

    llm_steps = payload.get("llm_steps")
    tool_steps = payload.get("tool_steps")
    tool_step_index = 0
    if isinstance(llm_steps, list):
        indexed_tool_steps = tool_steps if isinstance(tool_steps, list) else []
        for step in llm_steps:
            if not isinstance(step, dict):
                continue
            # ``None`` is meaningful here: it means channel parsing found no
            # user-visible content (for example a reasoning-only/tool-call
            # turn).  An ``or`` fallback resurrects the unparsed response and
            # incorrectly writes private reasoning into ``content``.
            if "response_content_stripped" in step:
                visible_text = step.get("response_content_stripped")
            else:
                visible_text = step.get("response_content")
            reasoning_text = step.get("reasoning_content")
            tool_calls = _normalize_tool_calls(step.get("tool_calls"))

            assistant_entry: dict[str, Any] = {"role": "assistant"}
            if isinstance(visible_text, str):
                stripped = _strip_think(visible_text)
                if stripped is not None:
                    assistant_entry["content"] = stripped
            elif visible_text is not None:
                assistant_entry["content"] = _jsonable(visible_text)
            if isinstance(reasoning_text, str) and reasoning_text.strip():
                assistant_entry["reasoning_content"] = reasoning_text
            if tool_calls:
                assistant_entry["tool_calls"] = tool_calls
            if any(key in assistant_entry for key in ("content", "reasoning_content", "tool_calls")):
                messages.append(assistant_entry)

            for tool_call in tool_calls:
                if tool_step_index >= len(indexed_tool_steps):
                    break
                step_result = indexed_tool_steps[tool_step_index]
                tool_step_index += 1
                if not isinstance(step_result, dict):
                    continue
                tool_entry: dict[str, Any] = {
                    "role": "tool",
                    "name": step_result.get("name") or tool_call.get("name") or "",
                    "content": _normalize_tool_content(step_result.get("result")),
                }
                if tool_call.get("id"):
                    tool_entry["tool_call_id"] = tool_call.get("id")
                messages.append(tool_entry)
    elif isinstance(payload.get("output"), str) and payload["output"].strip():
        reasoning_text, visible_text = _split_gemma_thought_channel(payload["output"])
        assistant_entry: dict[str, Any] = {"role": "assistant"}
        if visible_text is not None:
            assistant_entry["content"] = visible_text
        if reasoning_text:
            assistant_entry["reasoning_content"] = reasoning_text
        if len(assistant_entry) > 1:
            messages.append(assistant_entry)

    return {
        "sessionKey": session_key,
        "updatedAt": _iso_now(),
        "messageCount": len(messages),
        "messages": messages,
    }


def _message_signature(message: dict[str, Any]) -> str:
    try:
        return json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(message)


def _merge_clean_messages(existing_messages: Any, current_messages: Any) -> list[dict[str, Any]]:
    existing = [msg for msg in existing_messages if isinstance(msg, dict)] if isinstance(existing_messages, list) else []
    current = [msg for msg in current_messages if isinstance(msg, dict)] if isinstance(current_messages, list) else []
    if not existing:
        return current
    if not current:
        return existing

    existing_sigs = [_message_signature(msg) for msg in existing]
    current_sigs = [_message_signature(msg) for msg in current]
    for overlap in range(min(len(existing_sigs), len(current_sigs)), 0, -1):
        if existing_sigs[-overlap:] == current_sigs[:overlap]:
            return existing + current[overlap:]
    if len(current_sigs) <= len(existing_sigs) and existing_sigs[-len(current_sigs):] == current_sigs:
        return existing
    return existing + current


def _load_existing_messages(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    messages = payload.get("messages")
    return [msg for msg in messages if isinstance(msg, dict)] if isinstance(messages, list) else []


def _clean_messages_from_agent_end(path: Path, session_key: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    messages = _build_clean_transcript(payload, session_key).get("messages")
    return [msg for msg in messages if isinstance(msg, dict)] if isinstance(messages, list) else []


def _rebuild_history(session_dir: Path, session_key: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for log_path in sorted(session_dir.glob("*_agent-end.json")):
        merged = _merge_clean_messages(merged, _clean_messages_from_agent_end(log_path, session_key))
    return merged


def _write_clean_transcript(log_root: Path, session_key: str, payload: dict[str, Any]) -> None:
    session_dir = log_root / session_key
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    clean_path = session_dir / "clean-transcript.json"
    clean_payload = _build_clean_transcript(payload, session_key)
    merged_messages = _rebuild_history(session_dir, session_key)
    existing_messages = _load_existing_messages(clean_path)
    if len(existing_messages) > len(merged_messages):
        merged_messages = existing_messages
    merged_messages = _merge_clean_messages(merged_messages, clean_payload.get("messages", []))
    clean_payload["messages"] = merged_messages
    clean_payload["messageCount"] = len(merged_messages)
    serialized = json.dumps(clean_payload, ensure_ascii=False, indent=2)
    for path in (clean_path, session_dir / "cleaned_transcripts.json"):
        try:
            path.write_text(serialized, encoding="utf-8")
        except Exception:
            pass


def _write_agent_end(
    session_key: str,
    payload: dict[str, Any],
    duration_s: float,
    usage: dict[str, Any],
    success: bool,
) -> None:
    log_root = _log_root()
    if log_root is None:
        return
    session_dir = log_root / session_key
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    data = dict(payload)
    tool_budget_exhausted = (
        os.environ.get("NANOBOT_SELF_REWARD_EXIT_ON_TOOL_BUDGET", "").lower() in {"1", "true"}
        and (
            payload.get("stop_reason") == "max_iterations"
            or str(payload.get("output") or "").startswith(
                "I reached the maximum number of tool call iterations ("
            )
        )
    )
    data.update(
        {
            "stage": "agent_end",
            "timestamp": _iso_now(),
            "sessionKey": session_key,
            "success": success,
            "durationMs": round(duration_s * 1000, 1),
            "usage": usage,
        }
    )
    if "messages" in data and "messageCount" not in data:
        try:
            data["messageCount"] = len(data["messages"])
        except Exception:
            pass
    if tool_budget_exhausted:
        data.update(success=False, stop_reason="max_iterations", error="tool_budget_exhausted")
    log_path = session_dir / f"{_filename_ts()}_agent-end.json"
    try:
        log_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    _write_clean_transcript(log_root, session_key, data)
    if tool_budget_exhausted:
        # The terminal record is durable. Do not let unrelated async cleanup
        # keep this failed reward agent alive until the wall-clock timeout.
        os._exit(86)


def _select_final_turn(llm_steps: list[dict[str, Any]], fallback_output: str | None) -> dict[str, Any] | None:
    # Do not fall back to the raw output when structured parsing intentionally
    # produced no visible content; that would relabel reasoning as the answer.
    fallback_visible = _strip_think(fallback_output)
    best: tuple[float, dict[str, Any]] | None = None
    for index, step in enumerate(llm_steps):
        if not isinstance(step, dict):
            continue
        messages = step.get("request_messages")
        if not isinstance(messages, list) or not messages:
            continue
        response_text = step.get("response_content") or ""
        if "response_content_stripped" in step:
            visible_text = step.get("response_content_stripped")
        else:
            visible_text = response_text or fallback_visible
        reasoning_text = step.get("reasoning_content") or ""
        tool_calls = step.get("tool_calls") or []
        score = float(index)
        if str(step.get("finish_reason") or "").strip().lower() != "tool_calls":
            score += 10.0
        if not tool_calls:
            score += 4.0
        if reasoning_text:
            score += 1.0
        if fallback_visible and visible_text:
            norm_visible = " ".join(str(visible_text).split())
            norm_fallback = " ".join(fallback_visible.split())
            if norm_visible == norm_fallback:
                score += 4.0
            elif norm_visible in norm_fallback or norm_fallback in norm_visible:
                score += 2.0
        final_turn = {
            "messages": messages,
            "tools": step.get("request_tools"),
            "response_text": response_text,
            "visible_text": visible_text,
            "reasoning_text": reasoning_text or None,
            "thinking_blocks": step.get("thinking_blocks"),
            "finish_reason": step.get("finish_reason"),
        }
        if best is None or score > best[0]:
            best = (score, final_turn)
    return best[1] if best else None


def _patch_agent_trace() -> None:
    try:
        from nanobot.agent import hook as hook_mod
        from nanobot.agent import loop as loop_mod
    except Exception:
        return

    if getattr(loop_mod.AgentLoop, "_nanobot_rl_trace_patched", False):
        return

    class NanobotRLTraceHook(hook_mod.AgentHook):
        def __init__(self) -> None:
            super().__init__(reraise=False)
            self.reset()

        def reset(self) -> None:
            self.input = ""
            self.channel = ""
            self.chat_id = ""
            self.llm_steps: list[dict[str, Any]] = []
            self.tool_steps: list[dict[str, Any]] = []
            self.tool_events: list[dict[str, Any]] = []
            self.messages: list[dict[str, Any]] = []
            self.output: str | None = None
            self.stop_reason: str | None = None
            self.error: str | None = None
            self.usage: dict[str, Any] = {}
            self._request_messages: dict[int, list[dict[str, Any]]] = {}
            self._request_tools: dict[int, list[dict[str, Any]] | None] = {}

        async def before_iteration(self, context: Any) -> None:
            _sanitize_messages(getattr(context, "messages", None))
            self._request_messages[context.iteration] = _jsonable(context.messages)
            self._request_tools[context.iteration] = _extract_tools_snapshot(context)

        async def after_iteration(self, context: Any) -> None:
            response = getattr(context, "response", None)
            tool_calls = []
            for tool_call in getattr(context, "tool_calls", []) or []:
                tool_calls.append(
                    {
                        "id": getattr(tool_call, "id", ""),
                        "name": getattr(tool_call, "name", ""),
                        "arguments": _jsonable(getattr(tool_call, "arguments", {})),
                    }
                )
            if response is not None:
                content = getattr(response, "content", None) or ""
                self.llm_steps.append(
                    {
                        "iteration": context.iteration,
                        "request_messages": self._request_messages.pop(
                            context.iteration,
                            _jsonable(context.messages),
                        ),
                        "request_tools": self._request_tools.pop(context.iteration, None),
                        "response_content": content,
                        "response_content_stripped": _strip_think(content),
                        "tool_calls": tool_calls,
                        "reasoning_content": getattr(response, "reasoning_content", None),
                        "thinking_blocks": _jsonable(getattr(response, "thinking_blocks", None)),
                        "finish_reason": getattr(response, "finish_reason", None),
                        # Bounded per-request counters let context-budget
                        # regressions be checked against the model server's
                        # tokenizer without persisting request bodies twice.
                        "usage": _jsonable(getattr(context, "usage", None) or {}),
                    }
                )
            for tool_call, result in zip(tool_calls, getattr(context, "tool_results", []) or []):
                self.tool_steps.append(
                    {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "result": _jsonable(result),
                    }
                )
            for event in getattr(context, "tool_events", []) or []:
                self.tool_events.append(_jsonable(event))
            self.messages = _jsonable(context.messages)
            if getattr(context, "final_content", None) is not None:
                self.output = context.final_content
            if getattr(context, "stop_reason", None) is not None:
                self.stop_reason = context.stop_reason
            if getattr(context, "error", None):
                self.error = context.error
            if getattr(context, "usage", None):
                self.usage = _jsonable(context.usage)

        def to_payload(self) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "input": self.input,
                "channel": self.channel,
                "chat_id": self.chat_id,
                "llm_steps": list(self.llm_steps),
                "tool_steps": list(self.tool_steps),
                "tool_events": list(self.tool_events),
                "messages": list(self.messages),
                "output": self.output,
                "stop_reason": self.stop_reason,
                "error": self.error,
            }
            final_turn = _select_final_turn(self.llm_steps, self.output)
            if final_turn is not None:
                payload["final_turn"] = final_turn
            return payload

    original_init = loop_mod.AgentLoop.__init__
    original_process = loop_mod.AgentLoop._process_message

    def _patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if getattr(self, "_nanobot_rl_trace_hook", None) is None:
            trace_hook = NanobotRLTraceHook()
            self._extra_hooks.append(trace_hook)
            self._nanobot_rl_trace_hook = trace_hook

    async def _patched_process(self, msg: Any, session_key: str | None = None, **kwargs: Any) -> Any:
        key = session_key or getattr(msg, "session_key", None) or "cli:direct"
        turn_type = os.environ.get("NANOBOT_TURN_TYPE") or os.environ.get("CLAWARENA_NANOBOT_TURN_TYPE") or "side"
        session_token = _SESSION_KEY.set(key)
        turn_token = _TURN_TYPE.set(turn_type)
        started = time.monotonic()
        response = None
        trace_hook = getattr(self, "_nanobot_rl_trace_hook", None)
        if trace_hook is not None:
            trace_hook.reset()
            trace_hook.input = getattr(msg, "content", "") or ""
            trace_hook.channel = getattr(msg, "channel", "") or ""
            trace_hook.chat_id = getattr(msg, "chat_id", "") or ""
        try:
            response = await original_process(self, msg, session_key=session_key, **kwargs)
            return response
        finally:
            active_key = _SESSION_KEY.get()
            if trace_hook is not None and active_key:
                payload = trace_hook.to_payload()
                if payload.get("output") is None and response is not None:
                    payload["output"] = getattr(response, "content", None)
                usage = dict(getattr(self, "_last_usage", {}) or trace_hook.usage or {})
                success = payload.get("error") in (None, "") and payload.get("stop_reason") not in {
                    "error",
                    "tool_error",
                    "tool_loop",
                }
                _write_agent_end(
                    session_key=active_key,
                    payload=payload,
                    duration_s=time.monotonic() - started,
                    usage=usage,
                    success=success,
                )
            _SESSION_KEY.reset(session_token)
            _TURN_TYPE.reset(turn_token)

    loop_mod.AgentLoop.__init__ = _patched_init
    loop_mod.AgentLoop._process_message = _patched_process
    loop_mod.AgentLoop._nanobot_rl_trace_patched = True


def _apply_patches() -> None:
    applied_runtime_patches: list[str] = []
    if _patch_sanitize_model_messages():
        applied_runtime_patches.append("sanitize_model_messages")
    if _env_flag(("NANOBOT_RL_NO_MAX_TOKENS", "CLAWARENA_NANOBOT_NO_MAX_TOKENS"), True):
        if _patch_strip_max_tokens():
            applied_runtime_patches.append("strip_max_tokens")
    if _env_flag(("NANOBOT_RL_SGLANG_GEMMA_THINKING", "CLAWARENA_NANOBOT_SGLANG_GEMMA_THINKING"), True):
        if _patch_sglang_gemma_thinking_body():
            applied_runtime_patches.append("sglang_gemma_thinking")
    exact_tokenizer_path = (
        os.environ.get("NANOBOT_RL_TOKENIZER_PATH")
        or os.environ.get("CLAWARENA_NANOBOT_TOKENIZER_PATH")
        or ""
    ).strip()
    if exact_tokenizer_path:
        if _patch_hf_tokenizer_token_estimates(exact_tokenizer_path):
            applied_runtime_patches.append("hf_chat_template_token_estimates")
    elif _env_flag(("NANOBOT_RL_DISABLE_TIKTOKEN_ESTIMATES", "CLAWARENA_NANOBOT_DISABLE_TIKTOKEN_ESTIMATES"), True):
        if _patch_disable_tiktoken_token_estimates():
            applied_runtime_patches.append("disable_tiktoken_estimates")
    if _env_flag(("NANOBOT_RL_FAST_ARCHIVE", "CLAWARENA_NANOBOT_FAST_ARCHIVE"), False):
        if _patch_fast_consolidation_archive():
            applied_runtime_patches.append("fast_archive")
    if _env_flag(("NANOBOT_RL_DISABLE_CONSOLIDATION", "CLAWARENA_NANOBOT_DISABLE_CONSOLIDATION"), False):
        if _patch_disable_consolidation():
            applied_runtime_patches.append("disable_consolidation")
    if _patch_disabled_tools():
        applied_runtime_patches.append("disabled_tools")
    if _patch_exec_path_prepend():
        applied_runtime_patches.append("exec_path_prepend")
    if _patch_workspace_inspector_tool():
        applied_runtime_patches.append("workspace_inspector_tool")
    if _env_flag(("NANOBOT_RL_WORKSPACE_TRANSLATION", "CLAWARENA_NANOBOT_WORKSPACE_TRANSLATION"), True):
        if _patch_workspace_display_and_translation():
            applied_runtime_patches.append("workspace_translation")
    if _env_flag(("NANOBOT_RL_SEARCH_PATH_PATTERN_COMPAT", "CLAWARENA_NANOBOT_SEARCH_PATH_PATTERN_COMPAT"), True):
        if _patch_search_path_pattern_compat():
            applied_runtime_patches.append("search_path_pattern_compat")
    if _env_flag(("NANOBOT_RL_EDIT_LINE_NUMBER_FALLBACK", "CLAWARENA_NANOBOT_EDIT_LINE_NUMBER_FALLBACK"), True):
        if _patch_edit_file_line_number_fallback():
            applied_runtime_patches.append("edit_line_number_fallback")
    if _env_flag(("NANOBOT_RL_READ_OFFSET_ZERO_COMPAT", "CLAWARENA_NANOBOT_READ_OFFSET_ZERO_COMPAT"), True):
        if _patch_read_file_offset_zero_compat():
            applied_runtime_patches.append("read_offset_zero_compat")
    if _env_flag(("NANOBOT_RL_READ_FILE_DEDUP_COMPAT", "CLAWARENA_NANOBOT_READ_FILE_DEDUP_COMPAT"), True):
        if _patch_read_file_dedup_compat():
            applied_runtime_patches.append("read_file_dedup_compat")
    if _env_flag(("NANOBOT_RL_WEB_SEARCH_FALLBACK", "CLAWARENA_NANOBOT_WEB_SEARCH_FALLBACK"), True):
        if _patch_web_search_duckduckgo_fallback():
            applied_runtime_patches.append("web_search_fallback")
    if _env_flag(("NANOBOT_RL_PSEUDO_TOOL_CALLS", "CLAWARENA_NANOBOT_PSEUDO_TOOL_CALLS"), True):
        if _patch_agent_runner_nonstreaming_tool_calls():
            applied_runtime_patches.append("agent_runner_nonstreaming_tool_calls")
        if _patch_pseudo_tool_calls():
            applied_runtime_patches.append("pseudo_tool_calls")
        if _patch_agent_runner_pseudo_tool_calls():
            applied_runtime_patches.append("agent_runner_pseudo_tool_calls")
    if _env_flag(("NANOBOT_RL_PREMATURE_STOP_RECOVERY", "CLAWARENA_NANOBOT_PREMATURE_STOP_RECOVERY"), True):
        if _patch_premature_stop_recovery():
            applied_runtime_patches.append("premature_stop_recovery")
    _patch_provider_request_context()
    if _log_root() is not None:
        try:
            import nanobot as nanobot_pkg

            print(
                f"[nanobot_rl_runner] nanobot package={getattr(nanobot_pkg, '__file__', '<unknown>')}",
                file=sys.stderr,
            )
        except Exception:
            pass
        print(f"[nanobot_rl_runner] NANOBOT_LOG_DIR={_log_root()}", file=sys.stderr)
        if applied_runtime_patches:
            print(
                f"[nanobot_rl_runner] Runtime patches: {', '.join(applied_runtime_patches)}",
                file=sys.stderr,
            )
        _patch_agent_trace()
        print("[nanobot_rl_runner] Patch applied", file=sys.stderr)


def main() -> None:
    _remove_runner_dir_from_sys_path()
    _enforce_researchclaw_agentic_judge_budget()
    _apply_patches()
    from nanobot.cli.commands import app

    app()


if __name__ == "__main__":
    main()
