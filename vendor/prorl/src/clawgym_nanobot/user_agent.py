from __future__ import annotations

import atexit
import io
import json
import os
import queue
import re
import shutil
import socket
import stat
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from polar.agent.presets.nanobot import runtime as nanobot_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYN_TASK_JSONL = PROJECT_ROOT / "data" / "clawgym_nanobot" / "syn_task.jsonl"
REALISTIC_USER_SYSTEM_PROMPT = (
    "You are simulating a realistic end user who has just received work from an AI agent. "
    "You may inspect the task, final response, workspace, rubrics, and verifier code. "
    "Use those materials privately to judge whether the work likely satisfies the request. "
    "Return only the requested JSON object. Do not reveal detailed rubric-by-rubric grading."
)
RUBRIC_SUMMARY_SYSTEM_PROMPT = (
    "You are evaluating an AI agent's completed work against the supplied rubric and evidence. "
    "Use the rubric to produce an accurate score and a compact, actionable synthesis of the most "
    "important satisfied and unsatisfied requirements. Return only the requested JSON object."
)
# Backwards-compatible public constant. Runtime prompt selection is handled by
# _user_agent_system_prompt() so launcher configuration is honored per run.
SYSTEM_PROMPT = REALISTIC_USER_SYSTEM_PROMPT

_SYN_LOOKUP_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}
_SYN_LOOKUP_LOCK = threading.Lock()
_SYN_LOOKUP_WARMUP_THREADS: dict[Path, threading.Thread] = {}


@dataclass(frozen=True)
class UserAgentProfile:
    name: str
    backend: str
    base_url: str
    model: str
    api_key: str
    workers: int
    concurrency: int
    enable_thinking: bool | None = None
    show_reasoning: bool | None = None
    timeout_sec: int | None = None
    max_tokens: int | None = None
    context_length: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    send_top_k: bool = True
    max_retries: int | None = None
    max_tool_iterations: int | None = None
    web_enable: bool | None = None
    exec_enable: bool | None = None
    exec_timeout: int | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class UserAgentRewardResult:
    score: float
    feedback: str
    raw_response: str
    parse_error: str | None
    prompt_truncated: bool
    workspace_truncated: bool
    prompt_chars: int
    rules_source: str
    rules_count: int
    retries: int
    backend: str = "prompt"
    user_agent_profile: str | None = None
    user_agent_model: str | None = None
    user_agent_workspace: str | None = None
    final_workspace: str | None = None
    trajectory_dir: str | None = None
    user_agent_session_id: str | None = None
    user_agent_log_dir: str | None = None
    nanobot_stdout_path: str | None = None
    nanobot_stderr_path: str | None = None
    verifier_helper: str | None = None
    verifier_ran_by_agent: bool | None = None
    verifier_current_score: float | None = None
    verifier_current_returncode: int | None = None
    verifier_current_stdout: str | None = None
    verifier_current_stderr: str | None = None
    user_agent_raw_response_path: str | None = None
    user_feedback_path: str | None = None
    user_agent_agent_end_path: str | None = None
    user_agent_cot_path: str | None = None
    user_agent_cot_text_path: str | None = None
    user_agent_reasoning_present: bool | None = None
    user_agent_reasoning_part_count: int | None = None
    user_agent_reasoning_chars: int | None = None
    user_agent_prepare_duration_sec: float | None = None
    user_agent_feedback_duration_sec: float | None = None
    user_agent_cot_save_duration_sec: float | None = None
    user_agent_verifier_duration_sec: float | None = None
    user_agent_parse_duration_sec: float | None = None
    user_agent_total_duration_sec: float | None = None
    user_agent_stop_reason: str | None = None
    user_agent_run_success: bool | None = None
    user_agent_run_error: str | None = None
    user_agent_format_repaired: bool | None = None
    user_agent_queue_wait_sec: float | None = None
    user_agent_score_duration_sec: float | None = None
    user_agent_message_count: int | None = None
    user_agent_assistant_message_count: int | None = None
    user_agent_tool_call_count: int | None = None
    user_agent_tool_call_turn_count: int | None = None
    user_agent_tool_result_message_count: int | None = None
    user_agent_unique_tool_count: int | None = None
    user_agent_tool_names: list[str] | None = None


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _timeout_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"0", "-1", "none", "null", "no", "off", "disable", "disabled"}:
        return None
    try:
        value = int(normalized)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else None


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cleanup_scoring_workspace_enabled() -> bool:
    return _truthy_env("CLAWGYM_USER_AGENT_CLEANUP_WORKSPACE", True)


def reward_source() -> str:
    return os.environ.get("CLAWGYM_REWARD_SOURCE", "code_verifier").strip().lower() or "code_verifier"


def user_agent_reward_enabled() -> bool:
    return reward_source() in {"user_agent", "user-agent", "ua"}


def user_agent_reward_mode() -> str:
    raw = os.environ.get("CLAWGYM_USER_AGENT_REWARD_MODE", "continuous").strip().lower()
    if raw in {"discrete", "ternary", "-1-0-1", "neg-zero-pos"}:
        return "discrete"
    return "continuous"


def _apply_user_agent_reward_mode(score: float) -> float:
    if user_agent_reward_mode() == "discrete":
        numeric = float(score)
        if numeric in {-1.0, 0.0, 1.0}:
            return numeric
        # Fallback path only: convert a verifier-style [0, 1] score into the
        # same ternary scale used by the test_user_agent discrete prompt.
        numeric = max(0.0, min(1.0, numeric))
        if numeric >= 0.999:
            return 1.0
        if numeric <= 0.0:
            return -1.0
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _user_agent_score_instruction() -> str:
    if user_agent_reward_mode() == "discrete":
        return (
            "Return score as -1, 0, or 1. Use 1 if the user would likely accept the work, "
            "0 if it is mixed or only partially useful, and -1 if it clearly fails or is unusable."
        )
    return (
        "Return score as a number from 0.0 to 1.0, where 1.0 means the user would be fully satisfied "
        "and 0.0 means the work is unusable."
    )


def _user_agent_score_schema() -> str:
    feedback_shape = (
        "a concise actionable rubric synthesis"
        if user_agent_feedback_summarizes_rubric()
        else "natural focused user feedback"
    )
    if user_agent_reward_mode() == "discrete":
        return f'{{"feedback": "{feedback_shape}", "score": -1}}'
    return f'{{"feedback": "{feedback_shape}", "score": 0.0}}'


def user_agent_feedback_summarizes_rubric() -> bool:
    """Whether feedback should explicitly synthesize the scoring rubric.

    This is deliberately independent of reward computation: both modes inspect
    the same evidence and produce the same score.  Only the textual privileged
    information used by OPSD changes.
    """

    return _truthy_env("CLAWGYM_USER_AGENT_FEEDBACK_SUMMARIZE_RUBRIC", False)


def user_agent_feedback_mode() -> str:
    return "rubric_summary" if user_agent_feedback_summarizes_rubric() else "realistic_user"


def _user_agent_system_prompt() -> str:
    if user_agent_feedback_summarizes_rubric():
        return RUBRIC_SUMMARY_SYSTEM_PROMPT
    return REALISTIC_USER_SYSTEM_PROMPT


def _user_agent_feedback_instruction() -> str:
    if user_agent_feedback_summarizes_rubric():
        return (
            "In `feedback`, synthesize the rubric-based assessment into one concise, actionable "
            "paragraph for improving the next attempt. State the most consequential satisfied "
            "requirements and missing or incorrect requirements, prioritizing concrete changes. "
            "Do not enumerate every rubric item or include numeric scores. There is no fixed word limit."
        )
    return (
        "Speak as the original user, not a rubric grader. Give natural, focused feedback about "
        "the main change you want in the next attempt; if fully satisfied, say what should stay. "
        "Use enough words to make the feedback useful, without a fixed word limit. Do not mention "
        "scores or rubric items."
    )


_NANOBOT_BACKEND = None
_SEMAPHORE_LOCK = threading.Lock()
_SEMAPHORE_LIMIT = 0
_SEMAPHORE: threading.BoundedSemaphore | None = None
_PROFILE_SEMAPHORE_LOCK = threading.Lock()
_PROFILE_SEMAPHORES: dict[str, tuple[int, threading.BoundedSemaphore]] = {}
_MIXED_NANOBOT_BACKENDS = {"nanobot_mixed", "nanobot-mixed", "mixed_nanobot", "mixed-nanobot"}


def user_agent_backend() -> str:
    return os.environ.get("CLAWGYM_USER_AGENT_BACKEND", "nanobot").strip().lower() or "nanobot"


def user_agent_backend_is_mixed(backend: str | None = None) -> bool:
    return (backend or user_agent_backend()) in _MIXED_NANOBOT_BACKENDS


def _load_nanobot_backend():
    global _NANOBOT_BACKEND
    if _NANOBOT_BACKEND is None:
        _NANOBOT_BACKEND = nanobot_runtime
    return _NANOBOT_BACKEND


def _clean_nanobot_stdout(text: str) -> str:
    lines = []
    for raw_line in (text or "").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Using config:") or "nanobot is thinking" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _user_agent_semaphore() -> threading.BoundedSemaphore:
    global _SEMAPHORE, _SEMAPHORE_LIMIT
    requested = max(1, _int_env("CLAWGYM_USER_AGENT_CONCURRENCY", 16))
    with _SEMAPHORE_LOCK:
        if _SEMAPHORE is None or requested != _SEMAPHORE_LIMIT:
            _SEMAPHORE = threading.BoundedSemaphore(requested)
            _SEMAPHORE_LIMIT = requested
        return _SEMAPHORE


def _int_env_first(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return default


def _timeout_env_first(names: tuple[str, ...], default: int | None) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        normalized = raw.strip().lower()
        if normalized in {"0", "-1", "none", "null", "no", "off", "disable", "disabled"}:
            return None
        try:
            value = int(normalized)
        except (TypeError, ValueError):
            continue
        return value if value > 0 else None
    return default


def _float_env_first(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _truthy_env_first(names: tuple[str, ...], default: bool = False) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _str_env_first(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _private_text_env(
    *,
    file_names: tuple[str, ...],
    value_names: tuple[str, ...],
    default: str,
) -> str:
    """Read a private value from a mode-0600 file, with env compatibility."""
    raw_path = _str_env_first(file_names)
    if not raw_path:
        return _str_env_first(value_names, default)

    path = Path(raw_path)
    if not path.is_absolute():
        raise RuntimeError(f"{file_names[0]} must be an absolute path")
    try:
        with path.open(encoding="utf-8") as handle:
            mode = stat.S_IMODE(os.fstat(handle.fileno()).st_mode)
            if mode != 0o600:
                raise RuntimeError(f"{file_names[0]} must have mode 0600: {path}")
            value = handle.read().strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read {file_names[0]} {path}: {exc}") from exc
    if not value:
        raise RuntimeError(f"{file_names[0]} is empty: {path}")
    return value


def _user_agent_api_key() -> str:
    """Load the primary judge credential without putting it on a command line."""
    return _private_text_env(
        file_names=("CLAWGYM_USER_AGENT_API_KEY_FILE",),
        value_names=("CLAWGYM_USER_AGENT_API_KEY",),
        default="apiKey",
    )


def _single_user_agent_profile(backend: str | None = None) -> UserAgentProfile:
    selected_backend = (backend or user_agent_backend()).strip().lower() or "nanobot"
    if user_agent_backend_is_mixed(selected_backend):
        selected_backend = "nanobot"
    workers = max(1, _int_env("CLAWGYM_USER_AGENT_WORKERS", _int_env("CLAWGYM_USER_AGENT_CONCURRENCY", 16)))
    concurrency = max(1, _int_env("CLAWGYM_USER_AGENT_CONCURRENCY", workers))
    return UserAgentProfile(
        name=os.environ.get("CLAWGYM_USER_AGENT_PROFILE_NAME", selected_backend).strip() or selected_backend,
        backend=selected_backend,
        base_url=os.environ.get("CLAWGYM_USER_AGENT_BASE_URL", "").strip(),
        model=os.environ.get(
            "CLAWGYM_USER_AGENT_MODEL",
            "gemma-4-31B-it" if selected_backend == "nanobot" else "gemma-user-agent",
        ),
        api_key=_user_agent_api_key(),
        workers=workers,
        concurrency=concurrency,
        enable_thinking=_truthy_env("CLAWGYM_USER_AGENT_ENABLE_THINKING", True),
        show_reasoning=_truthy_env(
            "CLAWGYM_USER_AGENT_SHOW_REASONING",
            _truthy_env("CLAWGYM_USER_AGENT_ENABLE_THINKING", True),
        ),
        timeout_sec=_timeout_env("CLAWGYM_USER_AGENT_TIMEOUT_SEC", 900),
        max_tokens=_int_env("CLAWGYM_USER_AGENT_MAX_TOKENS", 8192),
        context_length=_int_env(
            "CLAWGYM_USER_AGENT_CONTEXT_LENGTH",
            _int_env("OPENCLAW_CONTEXT_WINDOW", 65536),
        ),
        temperature=_float_env("CLAWGYM_USER_AGENT_TEMPERATURE", 0.4),
        top_p=_float_env("CLAWGYM_USER_AGENT_TOP_P", 1.0),
        top_k=_int_env("CLAWGYM_USER_AGENT_TOP_K", -1),
        send_top_k=_truthy_env("CLAWGYM_USER_AGENT_SEND_TOP_K", True),
        max_retries=max(1, _int_env("CLAWGYM_USER_AGENT_MAX_RETRIES", 3)),
        max_tool_iterations=_int_env("CLAWGYM_USER_AGENT_MAX_TOOL_ITERATIONS", 80),
        web_enable=_truthy_env("CLAWGYM_USER_AGENT_WEB_ENABLE", False),
        exec_enable=_truthy_env("CLAWGYM_USER_AGENT_EXEC_ENABLE", True),
        exec_timeout=_int_env("CLAWGYM_USER_AGENT_EXEC_TIMEOUT", 60),
        reasoning_effort=os.environ.get("CLAWGYM_USER_AGENT_REASONING_EFFORT", "medium").strip() or "medium",
    )


def _mixed_extra_user_agent_profile() -> UserAgentProfile:
    workers = _int_env_first(("CLAWGYM_USER_AGENT_EXTRA_WORKERS", "CLAWGYM_USER_AGENT_DEEPSEEK_WORKERS"), 0)
    workers = max(0, workers)
    concurrency = _int_env_first(
        ("CLAWGYM_USER_AGENT_EXTRA_CONCURRENCY", "CLAWGYM_USER_AGENT_DEEPSEEK_CONCURRENCY"),
        workers or 1,
    )
    enable_thinking = _truthy_env_first(
        ("CLAWGYM_USER_AGENT_EXTRA_ENABLE_THINKING", "CLAWGYM_USER_AGENT_DEEPSEEK_ENABLE_THINKING"),
        False,
    )
    return UserAgentProfile(
        name=_str_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_NAME", "CLAWGYM_USER_AGENT_DEEPSEEK_NAME"),
            "deepseek-v4-flash",
        ),
        backend="nanobot",
        base_url=_str_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_BASE_URL", "CLAWGYM_USER_AGENT_DEEPSEEK_BASE_URL"),
            "",
        ),
        model=_str_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_MODEL", "CLAWGYM_USER_AGENT_DEEPSEEK_MODEL"),
            "deepseek-v4-flash",
        ),
        api_key=_private_text_env(
            file_names=(
                "CLAWGYM_USER_AGENT_EXTRA_API_KEY_FILE",
                "CLAWGYM_USER_AGENT_DEEPSEEK_API_KEY_FILE",
            ),
            value_names=(
                "CLAWGYM_USER_AGENT_EXTRA_API_KEY",
                "CLAWGYM_USER_AGENT_DEEPSEEK_API_KEY",
            ),
            default="apiKey",
        ),
        workers=workers,
        concurrency=max(1, concurrency),
        enable_thinking=enable_thinking,
        show_reasoning=_truthy_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_SHOW_REASONING", "CLAWGYM_USER_AGENT_DEEPSEEK_SHOW_REASONING"),
            enable_thinking,
        ),
        timeout_sec=_timeout_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_TIMEOUT_SEC", "CLAWGYM_USER_AGENT_DEEPSEEK_TIMEOUT_SEC"),
            _timeout_env("CLAWGYM_USER_AGENT_TIMEOUT_SEC", 900),
        ),
        max_tokens=_int_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_MAX_TOKENS", "CLAWGYM_USER_AGENT_DEEPSEEK_MAX_TOKENS"),
            _int_env("CLAWGYM_USER_AGENT_MAX_TOKENS", 8192),
        ),
        context_length=_int_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_CONTEXT_LENGTH", "CLAWGYM_USER_AGENT_DEEPSEEK_CONTEXT_LENGTH"),
            _int_env("CLAWGYM_USER_AGENT_CONTEXT_LENGTH", _int_env("OPENCLAW_CONTEXT_WINDOW", 65536)),
        ),
        temperature=_float_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_TEMPERATURE", "CLAWGYM_USER_AGENT_DEEPSEEK_TEMPERATURE"),
            _float_env("CLAWGYM_USER_AGENT_TEMPERATURE", 0.4),
        ),
        top_p=_float_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_TOP_P", "CLAWGYM_USER_AGENT_DEEPSEEK_TOP_P"),
            _float_env("CLAWGYM_USER_AGENT_TOP_P", 1.0),
        ),
        top_k=_int_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_TOP_K", "CLAWGYM_USER_AGENT_DEEPSEEK_TOP_K"),
            _int_env("CLAWGYM_USER_AGENT_TOP_K", -1),
        ),
        send_top_k=_truthy_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_SEND_TOP_K", "CLAWGYM_USER_AGENT_DEEPSEEK_SEND_TOP_K"),
            False,
        ),
        max_retries=max(
            1,
            _int_env_first(
                ("CLAWGYM_USER_AGENT_EXTRA_MAX_RETRIES", "CLAWGYM_USER_AGENT_DEEPSEEK_MAX_RETRIES"),
                _int_env("CLAWGYM_USER_AGENT_MAX_RETRIES", 3),
            ),
        ),
        max_tool_iterations=_int_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_MAX_TOOL_ITERATIONS", "CLAWGYM_USER_AGENT_DEEPSEEK_MAX_TOOL_ITERATIONS"),
            _int_env("CLAWGYM_USER_AGENT_MAX_TOOL_ITERATIONS", 80),
        ),
        web_enable=_truthy_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_WEB_ENABLE", "CLAWGYM_USER_AGENT_DEEPSEEK_WEB_ENABLE"),
            _truthy_env("CLAWGYM_USER_AGENT_WEB_ENABLE", False),
        ),
        exec_enable=_truthy_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_EXEC_ENABLE", "CLAWGYM_USER_AGENT_DEEPSEEK_EXEC_ENABLE"),
            _truthy_env("CLAWGYM_USER_AGENT_EXEC_ENABLE", True),
        ),
        exec_timeout=_int_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_EXEC_TIMEOUT", "CLAWGYM_USER_AGENT_DEEPSEEK_EXEC_TIMEOUT"),
            _int_env("CLAWGYM_USER_AGENT_EXEC_TIMEOUT", 60),
        ),
        reasoning_effort=_str_env_first(
            ("CLAWGYM_USER_AGENT_EXTRA_REASONING_EFFORT", "CLAWGYM_USER_AGENT_DEEPSEEK_REASONING_EFFORT"),
            os.environ.get("CLAWGYM_USER_AGENT_REASONING_EFFORT", "medium").strip() or "medium",
        ),
    )


def user_agent_profiles() -> list[UserAgentProfile]:
    backend = user_agent_backend()
    if not user_agent_backend_is_mixed(backend):
        return [_single_user_agent_profile(backend)]

    primary = _single_user_agent_profile("nanobot")
    primary_name = os.environ.get("CLAWGYM_USER_AGENT_PRIMARY_NAME", "gemma4").strip() or "gemma4"
    primary = UserAgentProfile(
        **{
            **primary.__dict__,
            "name": primary_name,
            "workers": max(1, _int_env("CLAWGYM_USER_AGENT_WORKERS", 32)),
            "concurrency": max(1, _int_env("CLAWGYM_USER_AGENT_CONCURRENCY", _int_env("CLAWGYM_USER_AGENT_WORKERS", 32))),
        }
    )
    extra = _mixed_extra_user_agent_profile()
    profiles = [primary]
    if extra.workers > 0:
        profiles.append(extra)
    return profiles


def user_agent_total_worker_count() -> int:
    return sum(max(0, profile.workers) for profile in user_agent_profiles())


def user_agent_total_concurrency() -> int:
    return sum(max(0, profile.concurrency) for profile in user_agent_profiles())


def _user_agent_semaphore_for_profile(profile: UserAgentProfile | None = None) -> threading.BoundedSemaphore:
    if profile is None:
        return _user_agent_semaphore()
    requested = max(1, int(profile.concurrency))
    key = profile.name
    with _PROFILE_SEMAPHORE_LOCK:
        current = _PROFILE_SEMAPHORES.get(key)
        if current is None or current[0] != requested:
            current = (requested, threading.BoundedSemaphore(requested))
            _PROFILE_SEMAPHORES[key] = current
        return current[1]


def normalize_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", normalized).strip()


def _load_syn_lookup(path: Path) -> dict[str, dict[str, Any]]:
    cached = _SYN_LOOKUP_CACHE.get(path)
    if cached is not None:
        return cached
    with _SYN_LOOKUP_LOCK:
        cached = _SYN_LOOKUP_CACHE.get(path)
        if cached is not None:
            return cached
        lookup: dict[str, dict[str, Any]] = {}
        started_at = time.time()
        if path.is_file():
            print(f"[ClawGym-UserAgent] loading task rules from {path}", flush=True)
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prompt = row.get("prompt")
                    if not isinstance(prompt, str):
                        continue
                    lookup.setdefault(
                        normalize_prompt(prompt),
                        {
                            "rules": row.get("rules"),
                            "hook_code": row.get("hook_code"),
                            "_syn_line_number": line_no,
                        },
                    )
        _SYN_LOOKUP_CACHE[path] = lookup
        print(
            f"[ClawGym-UserAgent] loaded {len(lookup)} task rule entries "
            f"from {path} in {time.time() - started_at:.2f}s",
            flush=True,
        )
        return lookup


def _syn_task_jsonl_path() -> Path:
    raw_path = os.environ.get("CLAWGYM_SYN_TASK_JSONL", "").strip()
    return Path(raw_path) if raw_path else DEFAULT_SYN_TASK_JSONL


def prewarm_task_rules_cache_async() -> None:
    path = _syn_task_jsonl_path()
    if not path.is_file() or path in _SYN_LOOKUP_CACHE:
        return
    with _SYN_LOOKUP_LOCK:
        if path in _SYN_LOOKUP_CACHE or path in _SYN_LOOKUP_WARMUP_THREADS:
            return
        thread = threading.Thread(
            target=_load_syn_lookup,
            args=(path,),
            name="user-agent-rules-cache",
            daemon=True,
        )
        _SYN_LOOKUP_WARMUP_THREADS[path] = thread
        thread.start()


def lookup_task_rules(user_query: str) -> tuple[list[dict[str, Any]], str, str | None]:
    path = _syn_task_jsonl_path()
    if not path.is_file():
        return [], "missing", None
    matched = _load_syn_lookup(path).get(normalize_prompt(user_query))
    if not matched:
        return [], "missing", None
    rules: list[dict[str, Any]] = []
    raw_rules = matched.get("rules")
    if isinstance(raw_rules, list):
        rules = [item for item in raw_rules if isinstance(item, dict)]
    hook_code = matched.get("hook_code")
    source = (
        f"syn_task_jsonl:{matched.get('_syn_line_number')}"
        if rules
        else f"syn_task_jsonl_no_rules:{matched.get('_syn_line_number')}"
    )
    return rules, source, str(hook_code) if hook_code is not None else None


def reward_code_bundle(task_dir: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for rel in ("reward/reward.sh", "reward/check.py"):
        path = task_dir / rel
        if path.exists() and path.is_file():
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
    return {"files": files}


def _is_probably_text(path: Path, max_probe_bytes: int = 4096) -> bool:
    try:
        data = path.read_bytes()[:max_probe_bytes]
    except OSError:
        return False
    if b"\0" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def serialize_workspace(
    workspace: Path,
    *,
    max_total_chars: int = 180_000,
    max_file_chars: int = 24_000,
) -> dict[str, Any]:
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "node_modules"}
    files: list[dict[str, Any]] = []
    total_chars = 0
    truncated = False
    skipped_binary = 0
    skipped_oversize = 0

    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in skip_dirs for part in relative.parts):
            continue
        rel = relative.as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not _is_probably_text(path):
            skipped_binary += 1
            files.append({"path": rel, "type": "binary_or_non_utf8", "size": size, "content": ""})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_file_chars:
            skipped_oversize += 1
            text = text[:max_file_chars] + "\n...[file truncated]..."
            truncated = True
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n...[workspace truncated]..."
            truncated = True
        total_chars += len(text)
        files.append({"path": rel, "type": "text", "size": size, "content": text})

    return {
        "root": str(workspace),
        "files": files,
        "text_chars": total_chars,
        "truncated": truncated,
        "skipped_binary_or_non_utf8": skipped_binary,
        "skipped_oversize_text_files": skipped_oversize,
    }


def _truncate_text(text: str, max_chars: int, marker: str = "\n...[truncated]...\n") -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    if max_chars <= len(marker) + 32:
        return text[:max_chars], True
    head = max_chars // 2
    tail = max_chars - head - len(marker)
    return text[:head] + marker + text[-tail:], True


def _json_text(value: Any, max_chars: int) -> tuple[str, bool]:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return _truncate_text(text, max_chars)


def workspace_to_scoring_text(snapshot: dict[str, Any], max_chars: int) -> tuple[str, bool]:
    files = snapshot.get("files", [])
    if not isinstance(files, list):
        files = []

    manifest_lines = [
        f"Workspace root: {snapshot.get('root')}",
        f"Original snapshot truncated: {snapshot.get('truncated')}",
        f"Original text chars included: {snapshot.get('text_chars')}",
        "File manifest:",
    ]
    for item in files:
        if isinstance(item, dict):
            manifest_lines.append(f"- {item.get('path')} ({item.get('type')}, {item.get('size')} bytes)")
    header = "\n".join(manifest_lines) + "\n"
    if len(header) >= max_chars:
        return _truncate_text(header, max_chars)

    remaining = max_chars - len(header)

    def priority(item: dict[str, Any]) -> tuple[int, str]:
        path = str(item.get("path") or "")
        if path.startswith(("output/", "outputs/", "out/")):
            bucket = 0
        elif path.startswith(("scripts/", "schedule/", "docs/", "data/", "downloads/", "state/")):
            bucket = 1
        elif path.startswith("input/"):
            bucket = 3
        else:
            bucket = 2
        return bucket, path

    text_items = [item for item in files if isinstance(item, dict) and item.get("type") == "text"]
    text_items.sort(key=priority)
    chunks = [header, "\nSelected file contents for scoring:\n"]
    remaining -= len(chunks[-1])
    truncated = bool(snapshot.get("truncated"))
    max_file_chars = min(8_000, max(1_500, max_chars // 8))
    included = 0
    for item in text_items:
        if remaining <= 0:
            truncated = True
            break
        path = item.get("path")
        size = item.get("size")
        content = str(item.get("content") or "")
        file_header = f"\n### {path} (text, {size} bytes)\n"
        if len(file_header) >= remaining:
            truncated = True
            break
        chunks.append(file_header)
        remaining -= len(file_header)
        per_file_budget = min(max_file_chars, remaining)
        clipped, file_truncated = _truncate_text(
            content,
            per_file_budget,
            "\n...[file content truncated for scoring]...\n",
        )
        chunks.append(clipped)
        remaining -= len(clipped)
        included += 1
        if file_truncated:
            truncated = True
        if remaining > 0:
            chunks.append("\n")
            remaining -= 1
    if included < len(text_items):
        truncated = True
    return "".join(chunks), truncated


def build_prompt(
    *,
    task_prompt: str,
    final_message: str,
    workspace: Path,
    rules: list[dict[str, Any]],
    rules_source: str,
    reward_code: dict[str, Any],
    code_verifier_score: float | None,
    max_prompt_chars: int,
) -> tuple[str, bool, bool]:
    score_instruction = _user_agent_score_instruction()
    schema = _user_agent_score_schema()
    snapshot = serialize_workspace(
        workspace,
        max_total_chars=_int_env("CLAWGYM_USER_AGENT_WORKSPACE_MAX_TOTAL_CHARS", 180_000),
        max_file_chars=_int_env("CLAWGYM_USER_AGENT_WORKSPACE_MAX_FILE_CHARS", 24_000),
    )

    task_text, task_truncated = _truncate_text(task_prompt, 12_000)
    final_text, final_truncated = _truncate_text(final_message, 8_000)
    rules_text, rules_truncated = _json_text(rules, 10_000)
    reward_text, reward_truncated = _json_text(reward_code, 18_000)

    verifier_score_text = (
        "not precomputed"
        if code_verifier_score is None
        else f"{code_verifier_score:.6f}"
    )
    prefix = (
        f"{score_instruction}\n"
        f"{_user_agent_feedback_instruction()} Return only JSON with this shape:\n"
        f"{schema}\n\n"
        "## Task Prompt\n"
        f"{task_text}\n\n"
        "## Evaluated Agent Final Response\n"
        f"{final_text}\n\n"
        "## Rubrics\n"
        f"rules_source: {rules_source}\n"
        f"{rules_text}\n\n"
        "## Code Verifier / Check Code\n"
        f"code_verifier_score: {verifier_score_text}\n"
        f"{reward_text}\n\n"
        "## Post-Agent Workspace Snapshot\n"
    )
    workspace_budget = max(4_000, max_prompt_chars - len(prefix) - 256)
    workspace_text, workspace_prompt_truncated = workspace_to_scoring_text(snapshot, workspace_budget)
    prompt = prefix + workspace_text + "\n"
    prompt_truncated = any([task_truncated, final_truncated, rules_truncated, reward_truncated, workspace_prompt_truncated])
    if len(prompt) > max_prompt_chars:
        prompt, extra_truncated = _truncate_text(prompt, max_prompt_chars)
        prompt_truncated = prompt_truncated or extra_truncated
    return prompt, prompt_truncated, bool(snapshot.get("truncated"))


def iter_json_objects(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        yield value
        index = end


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for value in reversed(list(iter_json_objects(stripped))):
        if isinstance(value, dict):
            return value
    raise ValueError("response did not contain a JSON object")


def validate_user_agent_json(value: dict[str, Any]) -> tuple[str, float]:
    feedback = value.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("feedback must be a non-empty string")
    score = value.get("score")
    if user_agent_reward_mode() == "discrete":
        if isinstance(score, bool) or score not in (-1, 0, 1):
            raise ValueError("discrete score must be one of -1, 0, 1")
        return feedback.strip(), float(score)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score must be numeric")
    numeric = float(score)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError("score must be in [0, 1]")
    return feedback.strip(), numeric


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_sec: int | None,
    max_tokens: int,
    temperature: float,
    payload_overrides: dict[str, Any] | None = None,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if payload_overrides:
        payload.update(payload_overrides)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        raise RuntimeError(f"missing choices in response: {body}")
    return str(choices[0].get("message", {}).get("content") or "")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "item"


def _user_agent_work_root() -> Path:
    explicit = os.environ.get("CLAWGYM_USER_AGENT_WORKDIR", "").strip()
    if explicit:
        return Path(explicit)
    if os.environ.get("SAVE_CKPT"):
        return Path(os.environ["SAVE_CKPT"]) / "user_agent_reward"
    if os.environ.get("CLAWGYM_RUNTIME_DIR"):
        return Path(os.environ["CLAWGYM_RUNTIME_DIR"]) / "user_agent_reward"
    return Path(tempfile.gettempdir()) / "clawgym-user-agent-reward"


def _score_reward_payload(raw: str) -> float | None:
    try:
        data = json.loads((raw or "").strip())
        if isinstance(data, dict):
            if not data:
                return 0.0
            return sum(float(value) for value in data.values()) / len(data)
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return float(data)
    except Exception:
        return None
    return None


def _read_clean_transcript_messages(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    messages = payload.get("messages") if isinstance(payload, dict) else None
    return [message for message in messages if isinstance(message, dict)] if isinstance(messages, list) else []


def _nanobot_transcript_metrics(log_dir: Path, session_id: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for name in ("clean-transcript.json", "cleaned_transcripts.json"):
        messages = _read_clean_transcript_messages(log_dir / session_id / name)
        if messages:
            break
    if not messages:
        return {}

    assistant_messages = 0
    tool_call_turns = 0
    tool_calls = 0
    tool_result_messages = 0
    tool_names: set[str] = set()
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role == "assistant":
            assistant_messages += 1
        if role in {"tool", "function"}:
            tool_result_messages += 1
        raw_calls = message.get("tool_calls")
        calls = (
            [call for call in raw_calls if isinstance(call, dict)]
            if isinstance(raw_calls, list)
            else []
        )
        if not calls:
            continue
        tool_call_turns += 1
        tool_calls += len(calls)
        for call in calls:
            function = call.get("function")
            tool_name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(tool_name, str) or not tool_name.strip():
                tool_name = call.get("name")
            if isinstance(tool_name, str) and tool_name.strip():
                tool_names.add(tool_name.strip())
    return {
        "user_agent_message_count": len(messages),
        "user_agent_assistant_message_count": assistant_messages,
        "user_agent_tool_call_count": tool_calls,
        "user_agent_tool_call_turn_count": tool_call_turns,
        "user_agent_tool_result_message_count": tool_result_messages,
        "user_agent_unique_tool_count": len(tool_names),
        "user_agent_tool_names": sorted(tool_names),
    }


def _copy_evaluated_trajectory(
    *,
    evaluated_log_dir: Path | None,
    evaluated_session_id: str | None,
    trajectory_dir: Path,
    final_message: str,
) -> dict[str, Any]:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Any] = {}
    transcript_messages: list[dict[str, Any]] = []
    if evaluated_log_dir is not None and evaluated_session_id:
        session_dir = evaluated_log_dir / evaluated_session_id
        for name in ("clean-transcript.json", "cleaned_transcripts.json"):
            src = session_dir / name
            if not src.is_file():
                continue
            dst = trajectory_dir / "cleaned_transcripts.json"
            shutil.copy2(src, dst)
            copied["trajectory_cleaned_transcript_path"] = str(dst)
            transcript_messages = _read_clean_transcript_messages(dst)
            break

    if not transcript_messages and final_message:
        transcript_messages = [{"role": "assistant", "content": final_message}]
        _write_json(
            trajectory_dir / "cleaned_transcripts.json",
            {
                "sessionKey": evaluated_session_id,
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "messageCount": len(transcript_messages),
                "messages": transcript_messages,
            },
        )
        copied["trajectory_cleaned_transcript_path"] = str(trajectory_dir / "cleaned_transcripts.json")

    text_chunks: list[str] = []
    reasoning_parts: list[dict[str, Any]] = []
    for index, message in enumerate(transcript_messages):
        role = str(message.get("role") or "unknown")
        text_chunks.append(f"## {index + 1}. {role}")
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text_chunks.append(content)
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = message.get(key)
            text = value.strip() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str) if value else ""
            if text:
                text_chunks.append(f"### {key}\n{text}")
                reasoning_parts.append({"index": index, "field": key, "text": text})
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            text_chunks.append("### tool_calls")
            text_chunks.append(json.dumps(tool_calls, ensure_ascii=False, indent=2, default=str))

    text_path = trajectory_dir / "cleaned_transcripts.md"
    _write_text(text_path, "\n\n".join(text_chunks) + ("\n" if text_chunks else ""))
    cot_path = trajectory_dir / "cot.json"
    cot_text_path = trajectory_dir / "cot.txt"
    _write_json(
        cot_path,
        {
            "session_id": evaluated_session_id,
            "reasoning_present": bool(reasoning_parts),
            "reasoning_part_count": len(reasoning_parts),
            "parts": reasoning_parts,
        },
    )
    _write_text(
        cot_text_path,
        "\n\n".join(f"## {part['index']}:{part['field']}\n{part['text']}" for part in reasoning_parts),
    )
    copied.update(
        {
            "trajectory_cleaned_transcript_text_path": str(text_path),
            "trajectory_cot_path": str(cot_path),
            "trajectory_cot_text_path": str(cot_text_path),
            "trajectory_cleaned_transcript_message_count": len(transcript_messages),
            "evaluated_agent_reasoning_present": bool(reasoning_parts),
            "evaluated_agent_reasoning_part_count": len(reasoning_parts),
            "evaluated_agent_reasoning_chars": sum(len(part["text"]) for part in reasoning_parts),
        }
    )
    return copied


def _prepare_nanobot_scoring_workspace(
    *,
    task_id: str,
    task_dir: Path,
    task_prompt: str,
    workspace_dir: Path,
    final_message: str,
    code_verifier_score: float | None,
    rules: list[dict[str, Any]],
    rules_source: str,
    reward_code: dict[str, Any],
    evaluated_session_id: str | None,
    evaluated_log_dir: Path | None,
    retry_index: int,
) -> tuple[Path, dict[str, Any]]:
    task_slug = _slugify(task_id or "task")
    unique = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    attempt_root = _user_agent_work_root() / task_slug / unique
    if retry_index > 1:
        attempt_root = attempt_root / f"retry-{retry_index}"
    workspace = attempt_root / "workspace"
    review_dir = workspace / "review"
    final_workspace = review_dir / "final_workspace"
    trajectory_dir = review_dir / "trajectory"
    workspace.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace_dir, final_workspace, symlinks=True)

    trajectory_info = _copy_evaluated_trajectory(
        evaluated_log_dir=evaluated_log_dir,
        evaluated_session_id=evaluated_session_id,
        trajectory_dir=trajectory_dir,
        final_message=final_message,
    )
    _write_json(
        trajectory_dir / "metadata.json",
        {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "source_workspace": str(workspace_dir),
            "evaluated_session_id": evaluated_session_id,
            "code_verifier_score": code_verifier_score,
            "rules_source": rules_source,
        },
    )
    _write_json(review_dir / "rubrics.json", _json_safe(rules))
    _write_json(review_dir / "reward_code.json", _json_safe(reward_code))
    _write_text(review_dir / "task_prompt.md", task_prompt or "")
    _write_text(review_dir / "evaluated_agent_final_message.md", final_message or "")

    nanobot_backend = _load_nanobot_backend()
    default_verifier_python = nanobot_backend.nanobot_python_executable()
    if not default_verifier_python.exists():
        default_verifier_python = Path(sys.executable)
    verifier_python = (
        os.environ.get("CLAWGYM_USER_AGENT_VERIFIER_PYTHON", "").strip()
        or os.environ.get("USER_AGENT_VERIFIER_PYTHON", "").strip()
        or str(default_verifier_python)
    )
    verifier_env = {
        "python": verifier_python,
        "sif": os.environ.get("CLAWGYM_USER_AGENT_VERIFIER_SIF") or os.environ.get("USER_AGENT_VERIFIER_SIF") or None,
        "apptainer": os.environ.get("CLAWGYM_USER_AGENT_VERIFIER_APPTAINER_BIN")
        or os.environ.get("USER_AGENT_VERIFIER_APPTAINER_BIN")
        or "apptainer",
        "bind": os.environ.get("CLAWGYM_USER_AGENT_VERIFIER_BIND")
        or os.environ.get("USER_AGENT_VERIFIER_BIND")
        or "/mnt:/mnt,/tmp:/tmp",
        "apptainer_flags": os.environ.get("CLAWGYM_USER_AGENT_VERIFIER_APPTAINER_FLAGS")
        or os.environ.get("USER_AGENT_VERIFIER_APPTAINER_FLAGS")
        or "",
    }
    _write_json(review_dir / "verifier_environment.json", verifier_env)

    helper = review_dir / "run_code_verifier.sh"
    helper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"',
                'WS="${ROOT}/final_workspace"',
                'cd "${WS}"',
                'if [[ ! -f "reward/check.py" ]]; then',
                '  echo "missing reward/check.py in ${WS}" >&2',
                "  exit 2",
                "fi",
                f"PYTHON_BIN={json.dumps(verifier_env['python'])}",
                'VERIFIER_SIF="${CLAWGYM_USER_AGENT_VERIFIER_SIF:-${USER_AGENT_VERIFIER_SIF:-}}"',
                'APPTAINER_BIN="${CLAWGYM_USER_AGENT_VERIFIER_APPTAINER_BIN:-${USER_AGENT_VERIFIER_APPTAINER_BIN:-apptainer}}"',
                'APPTAINER_BIND="${CLAWGYM_USER_AGENT_VERIFIER_BIND:-${USER_AGENT_VERIFIER_BIND:-/mnt:/mnt,/tmp:/tmp}}"',
                'APPTAINER_FLAGS_RAW="${CLAWGYM_USER_AGENT_VERIFIER_APPTAINER_FLAGS:-${USER_AGENT_VERIFIER_APPTAINER_FLAGS:-}}"',
                "# shellcheck disable=SC2206",
                "APPTAINER_FLAGS=(${APPTAINER_FLAGS_RAW})",
                'if [[ -n "${VERIFIER_SIF}" ]]; then',
                '  exec "${APPTAINER_BIN}" exec "${APPTAINER_FLAGS[@]}" --bind "${APPTAINER_BIND}" --pwd "${WS}" "${VERIFIER_SIF}" "${PYTHON_BIN}" reward/check.py "${WS}"',
                "fi",
                'exec "${PYTHON_BIN}" reward/check.py "${WS}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write_text(
        workspace / "README.md",
        "# User-Agent Review Workspace\n\n"
        "Review the evaluated AI agent's completed work.\n\n"
        "- `review/final_workspace/` is the completed workspace.\n"
        "- `review/trajectory/` contains the evaluated agent transcript when available.\n"
        "- `review/run_code_verifier.sh` reruns the code verifier against the completed workspace.\n"
        "- `review/task_prompt.md` contains the original user request.\n",
    )

    return workspace, {
        "user_agent_workspace": str(workspace),
        "final_workspace": str(final_workspace),
        "trajectory_dir": str(trajectory_dir),
        "verifier_helper": str(helper),
        **trajectory_info,
    }


def _cleanup_nanobot_scoring_workspace(workspace: Path, workspace_info: dict[str, Any]) -> None:
    if not _cleanup_scoring_workspace_enabled():
        return

    review_dir = workspace / "review"
    deleted: list[str] = []
    for key in ("final_workspace", "trajectory_dir", "verifier_helper"):
        raw_path = workspace_info.get(key)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
                deleted.append(str(path))
            elif path.is_dir():
                shutil.rmtree(path)
                deleted.append(str(path))
        except OSError:
            pass

    for child in list(workspace.iterdir()):
        if child.name in {"README.md", "review"}:
            continue
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
                deleted.append(str(child))
            elif child.is_dir():
                shutil.rmtree(child)
                deleted.append(str(child))
        except OSError:
            pass

    state_dir = workspace.parent / "state"
    if state_dir.exists():
        try:
            shutil.rmtree(state_dir)
            deleted.append(str(state_dir))
        except OSError:
            pass

    if review_dir.exists():
        _write_json(
            review_dir / "retention_summary.json",
            {
                "cleanup": "minimal",
                "deleted_paths": deleted,
                "kept_files": [
                    "task_prompt.md",
                    "evaluated_agent_final_message.md",
                    "user_agent_raw_response.txt",
                    "user_feedback.txt",
                    "user_agent_cot.json",
                    "user_agent_cot.txt",
                    "verifier_after_user_agent.stdout",
                    "verifier_after_user_agent.stderr",
                ],
            },
        )


def _write_user_agent_nanobot_config(
    workspace: Path,
    run_dir: Path,
    model: str,
    base_url: str,
    api_key: str,
    profile: UserAgentProfile | None = None,
) -> Path:
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    context_window = (
        profile.context_length
        if profile is not None and profile.context_length is not None
        else _int_env("CLAWGYM_USER_AGENT_CONTEXT_LENGTH", _int_env("OPENCLAW_CONTEXT_WINDOW", 65536))
    )
    max_tokens = (
        profile.max_tokens
        if profile is not None and profile.max_tokens is not None
        else _int_env("CLAWGYM_USER_AGENT_MAX_TOKENS", 8192)
    )
    temperature = (
        profile.temperature
        if profile is not None and profile.temperature is not None
        else _float_env("CLAWGYM_USER_AGENT_TEMPERATURE", 0.4)
    )
    top_p = (
        profile.top_p
        if profile is not None and profile.top_p is not None
        else _float_env("CLAWGYM_USER_AGENT_TOP_P", 1.0)
    )
    top_k = (
        profile.top_k
        if profile is not None and profile.top_k is not None
        else _int_env("CLAWGYM_USER_AGENT_TOP_K", -1)
    )
    enable_thinking = (
        profile.enable_thinking
        if profile is not None and profile.enable_thinking is not None
        else _truthy_env("CLAWGYM_USER_AGENT_ENABLE_THINKING", True)
    )
    show_reasoning = (
        profile.show_reasoning
        if profile is not None and profile.show_reasoning is not None
        else _truthy_env("CLAWGYM_USER_AGENT_SHOW_REASONING", bool(enable_thinking))
    )
    send_top_k = profile.send_top_k if profile is not None else _truthy_env("CLAWGYM_USER_AGENT_SEND_TOP_K", True)
    extra_body: dict[str, Any] = {"stream": False, "top_p": top_p}
    if send_top_k:
        extra_body["top_k"] = top_k
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    config = {
        "providers": {
            "custom": {
                "apiBase": base_url.rstrip("/"),
                "apiKey": api_key,
                "extraBody": extra_body,
            }
        },
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "provider": "custom",
                "model": model,
                "maxTokens": max_tokens,
                "contextWindowTokens": context_window,
                "temperature": temperature,
                "maxToolIterations": (
                    profile.max_tool_iterations
                    if profile is not None and profile.max_tool_iterations is not None
                    else _int_env("CLAWGYM_USER_AGENT_MAX_TOOL_ITERATIONS", 80)
                ),
                "botIcon": "",
                "timezone": os.environ.get("NANOBOT_TIMEZONE", "Asia/Shanghai"),
            }
        },
        "channels": {
            "sendProgress": False,
            "sendToolHints": False,
            "showReasoning": show_reasoning,
        },
        "tools": {
            "restrictToWorkspace": True,
            "web": {
                "enable": (
                    profile.web_enable
                    if profile is not None and profile.web_enable is not None
                    else _truthy_env("CLAWGYM_USER_AGENT_WEB_ENABLE", False)
                )
            },
            "exec": {
                "enable": (
                    profile.exec_enable
                    if profile is not None and profile.exec_enable is not None
                    else _truthy_env("CLAWGYM_USER_AGENT_EXEC_ENABLE", True)
                ),
                "timeout": (
                    profile.exec_timeout
                    if profile is not None and profile.exec_timeout is not None
                    else _int_env("CLAWGYM_USER_AGENT_EXEC_TIMEOUT", 60)
                ),
            },
        },
    }
    if enable_thinking:
        config["agents"]["defaults"]["reasoningEffort"] = (
            profile.reasoning_effort
            if profile is not None and profile.reasoning_effort is not None
            else os.environ.get("CLAWGYM_USER_AGENT_REASONING_EFFORT", "medium")
        )
    config_path = state_dir / "config.json"
    _write_json(config_path, config)
    return config_path


def _user_agent_nanobot_env_overrides(profile: UserAgentProfile | None = None) -> dict[str, str]:
    enable_thinking = (
        profile.enable_thinking
        if profile is not None and profile.enable_thinking is not None
        else _truthy_env("CLAWGYM_USER_AGENT_ENABLE_THINKING", True)
    )
    show_reasoning = (
        profile.show_reasoning
        if profile is not None and profile.show_reasoning is not None
        else _truthy_env("CLAWGYM_USER_AGENT_SHOW_REASONING", bool(enable_thinking))
    )
    thinking_value = "1" if enable_thinking else "0"
    return {
        "NANOBOT_ENABLE_THINKING": thinking_value,
        "USER_AGENT_ENABLE_THINKING": thinking_value,
        "NANOBOT_SHOW_REASONING": "1" if show_reasoning else "0",
    }


def build_nanobot_user_message() -> str:
    return (
        f"{_user_agent_score_instruction()}\n"
        "You are a realistic end user reviewing another AI agent's completed work. "
        "Do not rely on the evaluated agent's final response claims. Inspect the actual files.\n\n"
        "Evidence available in this workspace:\n"
        "- review/final_workspace/ contains the complete workspace after the evaluated agent finished.\n"
        "- review/trajectory/ contains the evaluated agent's cleaned transcript when available.\n"
        "- review/run_code_verifier.sh reruns the code verifier against review/final_workspace/.\n"
        "- review/task_prompt.md contains the original user request.\n\n"
        "Privately inspect the final workspace and trajectory. Run or inspect review/run_code_verifier.sh before "
        "deciding when feasible. Base the score on whether the real deliverables satisfy the user request, "
        "using verifier failures as strong evidence of missed requirements. Keep the review bounded: when the "
        "verifier already establishes multiple decisive failures, inspect at most two representative failing "
        "artifacts and then decide instead of exhaustively auditing every file. Keep private reasoning concise; "
        "as soon as you have enough evidence, write the result and finish.\n\n"
        f"{_user_agent_feedback_instruction()} Before finishing, use the write_file tool to save the exact JSON "
        "object to review/user_agent_result.json, then return the same JSON object as your final answer. The file and "
        "final answer must both have this shape:\n"
        f"{_user_agent_score_schema()}\n"
    )


def _parse_nanobot_user_agent_result(raw: str, result_path: Path) -> tuple[str, float]:
    """Prefer the final response, then fall back to the UA-authored result artifact.

    The artifact is written by the user agent itself after it has inspected the
    workspace.  Reading it is therefore not a host-side score inference; it is
    a more reliable transport for the same feedback and score when a model
    emits a natural-language final response despite the JSON-only instruction.
    """

    raw_error: Exception | None = None
    try:
        return validate_user_agent_json(parse_json_object(raw))
    except Exception as exc:
        raw_error = exc

    if result_path.is_file():
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("user-agent result artifact must contain a JSON object")
            return validate_user_agent_json(value)
        except Exception as artifact_exc:
            raise ValueError(
                f"invalid final response ({raw_error}); invalid result artifact ({artifact_exc})"
            ) from artifact_exc
    raise raw_error


def _nanobot_format_repair_messages(
    *, raw: str, log_dir: Path, session_id: str
) -> list[dict[str, str]]:
    """Build a tool-free repair prompt from the UA's completed review evidence."""

    _agent_end_path, payload = _nanobot_agent_end_payload(log_dir, session_id)
    reasoning_parts = _extract_user_agent_cot(payload)
    reasoning_text = "\n\n".join(
        str(part.get("text") or json.dumps(part.get("blocks"), ensure_ascii=False))
        for part in reasoning_parts
        if part.get("text") or part.get("blocks")
    )
    reasoning_text, _ = _truncate_text(
        reasoning_text,
        _int_env("CLAWGYM_USER_AGENT_FORMAT_REPAIR_MAX_EVIDENCE_CHARS", 16_000),
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the formatting finalizer for a user-agent evaluator that already completed its "
                "workspace review. Do not redo the task, call tools, or introduce new factual claims. Use the "
                "completed verdict and private review reasoning to return the evaluator's own score and feedback "
                "as exactly one JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_user_agent_score_instruction()}\n"
                f"{_user_agent_feedback_instruction()} Return only one JSON object with this "
                f"shape: {_user_agent_score_schema()}\n\n"
                f"Completed evaluator verdict:\n{raw[:4000]}\n\n"
                f"Private review reasoning:\n{reasoning_text}"
            ),
        },
    ]


def _nanobot_transcript_final_response(log_dir: Path, session_id: str) -> str:
    for name in ("clean-transcript.json", "cleaned_transcripts.json"):
        transcript_path = log_dir / session_id / name
        if not transcript_path.exists():
            continue
        messages = _read_clean_transcript_messages(transcript_path)
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip() and content.strip() != "null":
                return content.strip()
            tool_calls = message.get("tool_calls")
            if not tool_calls and content not in (None, "", "null"):
                return str(content).strip()
    return ""


def _nanobot_transcript_contains(log_dir: Path, session_id: str, *needles: str) -> bool:
    for name in ("clean-transcript.json", "cleaned_transcripts.json"):
        transcript_path = log_dir / session_id / name
        if not transcript_path.is_file():
            continue
        try:
            text = transcript_path.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        return any(needle.lower() in text for needle in needles)
    return False


def _nanobot_agent_end_path(log_dir: Path, session_id: str) -> Path | None:
    session_dir = log_dir / session_id
    if not session_dir.is_dir():
        return None
    candidates = sorted(path for path in session_dir.glob("*_agent-end.json") if path.is_file())
    return candidates[-1] if candidates else None


def _nanobot_agent_end_payload(log_dir: Path, session_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    path = _nanobot_agent_end_path(log_dir, session_id)
    if path is None:
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, None
    return path, payload if isinstance(payload, dict) else None


def _nanobot_review_completion(
    log_dir: Path, session_id: str
) -> tuple[bool | None, str | None, str | None]:
    """Return the Nanobot review's explicit completion status.

    A zero CLI return code only means that the Nanobot runner shut down
    cleanly.  The model turn itself can still have failed after repeatedly
    exhausting its per-turn output budget.  In that case ``agent-end.json``
    records ``success=false`` and a length-related stop reason.  Treating the
    accumulated, incomplete reasoning as a completed verdict lets the JSON
    format finalizer manufacture a plausible-looking score from a review that
    never actually finished.
    """

    _path, payload = _nanobot_agent_end_payload(log_dir, session_id)
    if not isinstance(payload, dict):
        return None, None, None
    success_value = payload.get("success")
    success = success_value if isinstance(success_value, bool) else None
    stop_reason = str(payload.get("stop_reason") or "").strip() or None
    error = str(payload.get("error") or "").strip() or None
    return success, stop_reason, error


def _compact_reasoning_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def _extract_inline_reasoning_value(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    text = value.strip()
    gemma_match = re.match(r"^\s*<\|channel>thought\s*", text)
    if gemma_match:
        rest = text[gemma_match.end():]
        end = re.search(r"<channel\|>|<\|channel\|>", rest)
        thinking = rest[: end.start()] if end else rest
        return thinking.strip()
    parts: list[str] = []
    for pattern in (r"<think>([\s\S]*?)</think>", r"<thought>([\s\S]*?)</thought>"):
        for match in re.finditer(pattern, text):
            part = match.group(1).strip()
            if part:
                parts.append(part)
    return "\n\n".join(parts)


def _extract_user_agent_cot(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    parts: list[dict[str, Any]] = []
    for index, step in enumerate(payload.get("llm_steps") or []):
        if not isinstance(step, dict):
            continue
        for key in ("reasoning_content", "reasoning", "reasoning_text"):
            text = _compact_reasoning_value(step.get(key))
            if text:
                parts.append({"source": "llm_steps", "index": index, "field": key, "text": text})
        for key in ("response_content", "response_content_stripped", "content", "message"):
            text = _extract_inline_reasoning_value(step.get(key))
            if text:
                parts.append({"source": "llm_steps", "index": index, "field": f"{key}_inline_reasoning", "text": text})
                break
        blocks = step.get("thinking_blocks")
        if isinstance(blocks, list) and blocks:
            parts.append({"source": "llm_steps", "index": index, "field": "thinking_blocks", "blocks": blocks})
    final_turn = payload.get("final_turn")
    if isinstance(final_turn, dict):
        for key in ("reasoning_content", "reasoning", "reasoning_text"):
            text = _compact_reasoning_value(final_turn.get(key))
            if text:
                parts.append({"source": "final_turn", "field": key, "text": text})
        blocks = final_turn.get("thinking_blocks")
        if isinstance(blocks, list) and blocks:
            parts.append({"source": "final_turn", "field": "thinking_blocks", "blocks": blocks})
    for index, message in enumerate(payload.get("messages") or []):
        if not isinstance(message, dict):
            continue
        for key in ("reasoning_content", "reasoning", "thinking"):
            text = _compact_reasoning_value(message.get(key))
            if text:
                parts.append({"source": "messages", "index": index, "field": key, "text": text})
        text = _extract_inline_reasoning_value(message.get("content"))
        if text:
            parts.append({"source": "messages", "index": index, "field": "content_inline_reasoning", "text": text})
    return parts


def _save_user_agent_cot(workspace: Path, log_dir: Path, session_id: str) -> dict[str, Any]:
    agent_end_path, payload = _nanobot_agent_end_payload(log_dir, session_id)
    parts = _extract_user_agent_cot(payload)
    review_dir = workspace / "review"
    json_path = review_dir / "user_agent_cot.json"
    text_path = review_dir / "user_agent_cot.txt"
    text_chunks: list[str] = []
    for part in parts:
        label = f"{part.get('source')}:{part.get('index', '')}:{part.get('field')}".strip(":")
        if "text" in part:
            text_chunks.append(f"## {label}\n{part['text']}\n")
        elif "blocks" in part:
            text_chunks.append(f"## {label}\n{json.dumps(part['blocks'], ensure_ascii=False, indent=2)}\n")
    _write_json(
        json_path,
        {
            "session_id": session_id,
            "agent_end_path": str(agent_end_path) if agent_end_path else None,
            "reasoning_present": bool(parts),
            "reasoning_part_count": len(parts),
            "parts": parts,
        },
    )
    _write_text(text_path, "\n".join(text_chunks) if text_chunks else "")
    return {
        "user_agent_agent_end_path": str(agent_end_path) if agent_end_path else None,
        "user_agent_cot_path": str(json_path),
        "user_agent_cot_text_path": str(text_path),
        "user_agent_reasoning_present": bool(parts),
        "user_agent_reasoning_part_count": len(parts),
        "user_agent_reasoning_chars": sum(len(str(part.get("text") or "")) for part in parts),
    }


def _nanobot_agent_end_final_response(log_dir: Path, session_id: str) -> str:
    _, payload = _nanobot_agent_end_payload(log_dir, session_id)
    if not isinstance(payload, dict):
        return ""
    final_turn = payload.get("final_turn")
    if isinstance(final_turn, dict):
        for key in ("response_content_stripped", "response_content", "content", "message"):
            value = final_turn.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != "null":
                return value.strip()
    for step in reversed(payload.get("llm_steps") or []):
        if not isinstance(step, dict):
            continue
        for key in ("response_content_stripped", "response_content", "content", "message"):
            value = step.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != "null":
                return value.strip()
    output = payload.get("output")
    if isinstance(output, str) and output.strip() and output.strip() != "null":
        return output.strip()
    return ""


def _run_scoring_workspace_verifier(workspace: Path) -> dict[str, Any]:
    helper = workspace / "review" / "run_code_verifier.sh"
    if not helper.exists():
        return {"stdout": "", "stderr": "missing verifier helper", "returncode": None, "score": None}
    try:
        proc = subprocess.run(
            ["bash", str(helper)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=_int_env("CLAWGYM_USER_AGENT_VERIFIER_TIMEOUT_SEC", 180),
        )
    except Exception as exc:
        return {"stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "returncode": None, "score": None}
    return {
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "returncode": proc.returncode,
        "score": _score_reward_payload(proc.stdout),
    }


def _fallback_result(
    *,
    code_verifier_score: float | None,
    raw: str,
    parse_error: str,
    prompt_truncated: bool,
    workspace_truncated: bool,
    prompt_chars: int,
    rules_source: str,
    rules_count: int,
    retries: int,
    backend: str,
    **extra: Any,
) -> UserAgentRewardResult:
    if not _truthy_env("CLAWGYM_USER_AGENT_FAIL_CLOSED", True):
        fallback = _apply_user_agent_reward_mode(float(code_verifier_score or 0.0))
    else:
        fallback = 0.0
    return UserAgentRewardResult(
        score=fallback,
        feedback="",
        raw_response=raw,
        parse_error=parse_error or "unknown user-agent scorer error",
        prompt_truncated=prompt_truncated,
        workspace_truncated=workspace_truncated,
        prompt_chars=prompt_chars,
        rules_source=rules_source,
        rules_count=rules_count,
        retries=retries,
        backend=backend,
        **extra,
    )


def _score_task_prompt(
    *,
    task_id: str,
    task_dir: Path,
    task_prompt: str,
    workspace_dir: Path,
    final_message: str,
    code_verifier_score: float | None,
    profile: UserAgentProfile | None = None,
) -> UserAgentRewardResult:
    profile = profile or _single_user_agent_profile("prompt")
    rules, rules_source, hook_code = lookup_task_rules(task_prompt)
    reward_code = reward_code_bundle(task_dir)
    if hook_code is not None:
        reward_code["hook_code"] = hook_code
    prompt, prompt_truncated, workspace_truncated = build_prompt(
        task_prompt=task_prompt,
        final_message=final_message,
        workspace=workspace_dir,
        rules=rules,
        rules_source=rules_source,
        reward_code=reward_code,
        code_verifier_score=code_verifier_score,
        max_prompt_chars=_int_env("CLAWGYM_USER_AGENT_MAX_PROMPT_CHARS", 56_000),
    )
    messages = [
        {"role": "system", "content": _user_agent_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    base_url = profile.base_url
    if not base_url:
        raise ValueError("CLAWGYM_USER_AGENT_BASE_URL is required when CLAWGYM_REWARD_SOURCE=user_agent")
    model = profile.model
    api_key = profile.api_key
    timeout_sec = (
        profile.timeout_sec
        if profile.timeout_sec is not None
        else _timeout_env("CLAWGYM_USER_AGENT_TIMEOUT_SEC", 900)
    )
    max_tokens = profile.max_tokens or _int_env("CLAWGYM_USER_AGENT_MAX_TOKENS", 8192)
    temperature = profile.temperature if profile.temperature is not None else _float_env("CLAWGYM_USER_AGENT_TEMPERATURE", 0.4)
    max_retries = profile.max_retries or max(1, _int_env("CLAWGYM_USER_AGENT_MAX_RETRIES", 3))

    raw = ""
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_openai_compatible(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            feedback, score = validate_user_agent_json(parse_json_object(raw))
            return UserAgentRewardResult(
                score=score,
                feedback=feedback,
                raw_response=raw,
                parse_error=None,
                prompt_truncated=prompt_truncated,
                workspace_truncated=workspace_truncated,
                prompt_chars=len(prompt),
                rules_source=rules_source,
                rules_count=len(rules),
                retries=attempt - 1,
                backend="prompt",
                user_agent_profile=profile.name,
                user_agent_model=profile.model,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 10))

    return _fallback_result(
        code_verifier_score=code_verifier_score,
        raw=raw,
        parse_error=last_error,
        prompt_truncated=prompt_truncated,
        workspace_truncated=workspace_truncated,
        prompt_chars=len(prompt),
        rules_source=rules_source,
        rules_count=len(rules),
        retries=max_retries,
        backend="prompt",
        user_agent_profile=profile.name,
        user_agent_model=profile.model,
    )


def _score_task_nanobot(
    *,
    task_id: str,
    task_dir: Path,
    task_prompt: str,
    workspace_dir: Path,
    final_message: str,
    code_verifier_score: float | None,
    evaluated_session_id: str | None = None,
    evaluated_log_dir: Path | None = None,
    profile: UserAgentProfile | None = None,
) -> UserAgentRewardResult:
    score_task_started_at = time.time()
    profile = profile or _single_user_agent_profile("nanobot")
    base_url = profile.base_url
    if not base_url:
        raise ValueError("CLAWGYM_USER_AGENT_BASE_URL is required when CLAWGYM_REWARD_SOURCE=user_agent")
    model = profile.model
    api_key = profile.api_key
    timeout_sec = (
        profile.timeout_sec
        if profile.timeout_sec is not None
        else _timeout_env("CLAWGYM_USER_AGENT_TIMEOUT_SEC", 900)
    )
    max_retries = profile.max_retries or max(1, _int_env("CLAWGYM_USER_AGENT_MAX_RETRIES", 3))
    format_repair_attempts = max(
        0, _int_env("CLAWGYM_USER_AGENT_FORMAT_REPAIR_ATTEMPTS", 1)
    )
    rules, rules_source, hook_code = lookup_task_rules(task_prompt)
    reward_code = reward_code_bundle(task_dir)
    if hook_code is not None:
        reward_code["hook_code"] = hook_code
    nanobot_backend = _load_nanobot_backend()
    message = build_nanobot_user_message()
    raw = ""
    last_error = ""
    workspace_info: dict[str, Any] = {}
    verifier_after: dict[str, Any] = {}
    cot_info: dict[str, Any] = {}
    raw_path: Path | None = None
    feedback_path: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    session_id = ""
    prepare_duration_sec: float | None = None
    feedback_duration_sec: float | None = None
    cot_save_duration_sec: float | None = None
    verifier_duration_sec: float | None = None
    parse_duration_sec: float | None = None
    transcript_metrics: dict[str, Any] = {}
    user_agent_stop_reason: str | None = None
    user_agent_run_success: bool | None = None
    user_agent_run_error: str | None = None
    user_agent_format_repaired = False
    log_dir = _user_agent_work_root() / "nanobot_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with _user_agent_semaphore_for_profile(profile):
        for attempt in range(1, max_retries + 1):
            try:
                prepare_started_at = time.time()
                workspace, workspace_info = _prepare_nanobot_scoring_workspace(
                    task_id=task_id,
                    task_dir=task_dir,
                    task_prompt=task_prompt,
                    workspace_dir=workspace_dir,
                    final_message=final_message,
                    code_verifier_score=code_verifier_score,
                    rules=rules,
                    rules_source=rules_source,
                    reward_code=reward_code,
                    evaluated_session_id=evaluated_session_id,
                    evaluated_log_dir=evaluated_log_dir,
                    retry_index=attempt,
                )
                prepare_duration_sec = time.time() - prepare_started_at
                _write_json(
                    workspace / "review" / "user_agent_profile.json",
                    {
                        "name": profile.name,
                        "backend": profile.backend,
                        "model": profile.model,
                        "base_url": profile.base_url,
                        "context_length": profile.context_length,
                        "max_tokens": profile.max_tokens,
                        "enable_thinking": profile.enable_thinking,
                        "show_reasoning": profile.show_reasoning,
                        "max_tool_iterations": profile.max_tool_iterations,
                        "timeout_sec": profile.timeout_sec,
                        "reward_mode": user_agent_reward_mode(),
                        "feedback_mode": user_agent_feedback_mode(),
                        "feedback_summarizes_rubric": user_agent_feedback_summarizes_rubric(),
                        "discrete_scores": [-1, 0, 1] if user_agent_reward_mode() == "discrete" else None,
                    },
                )
                reward_mode = user_agent_reward_mode()
                session_id = (
                    f"user-agent-{reward_mode}-{_slugify(profile.name)}-{_slugify(task_id or 'task')}-"
                    f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}-{attempt}"
                )
                config_path = _write_user_agent_nanobot_config(
                    workspace,
                    workspace.parent,
                    model,
                    base_url,
                    api_key,
                    profile,
                )
                result_path = workspace / "review" / "user_agent_result.json"
                result_path.unlink(missing_ok=True)
                feedback_started_at = time.time()
                proc = nanobot_backend.run_nanobot_turn(
                    workspace_dir=workspace,
                    config_path=config_path,
                    session_id=session_id,
                    turn_type="side",
                    message=message,
                    log_dir=log_dir,
                    timeout=timeout_sec,
                    env_overrides=_user_agent_nanobot_env_overrides(profile),
                )
                feedback_duration_sec = time.time() - feedback_started_at
                stdout_path = workspace.parent / f"nanobot_user_agent_{attempt}.stdout"
                stderr_path = workspace.parent / f"nanobot_user_agent_{attempt}.stderr"
                stdout_chunks = [proc.stdout or ""]
                stderr_chunks = [proc.stderr or ""]
                raw = (
                    _nanobot_transcript_final_response(log_dir, session_id)
                    or _nanobot_agent_end_final_response(log_dir, session_id)
                    or _clean_nanobot_stdout(proc.stdout or "")
                )
                raw_path = workspace / "review" / "user_agent_raw_response.txt"
                _write_text(raw_path, raw or proc.stdout or "")
                if proc.returncode != 0:
                    raise RuntimeError(f"nanobot exited with code {proc.returncode}: {(proc.stderr or '')[:1000]}")

                (
                    user_agent_run_success,
                    user_agent_stop_reason,
                    user_agent_run_error,
                ) = _nanobot_review_completion(log_dir, session_id)

                # Gemma occasionally completes the substantive workspace
                # review but emits only the natural-language feedback sentence,
                # omitting the requested JSON wrapper and numeric score.  A
                # fresh retry discards that expensive tool context and repeats
                # the entire review.  Repair formatting in the same Nanobot
                # evidence first, using a tool-free OpenAI-compatible call.
                # The CLI's same-session resume path can block after a completed
                # tool-heavy turn, while a fresh Nanobot session needlessly
                # repeats the complete workspace review.  The format finalizer
                # receives the original verdict and the UA's own private review
                # reasoning, then supplies its own score; the host never infers
                # one.
                for repair_index in range(1, format_repair_attempts + 1):
                    try:
                        _parse_nanobot_user_agent_result(raw or proc.stdout or "", result_path)
                        break
                    except Exception as parse_exc:
                        # JSON repair is valid only after a completed review
                        # produced a substantive verdict in the wrong shape.
                        # Never turn an explicitly failed/length-exhausted
                        # reasoning trace into a training reward.
                        if user_agent_run_success is False or user_agent_run_error:
                            detail = user_agent_run_error or "Nanobot reported success=false"
                            if user_agent_stop_reason:
                                detail = f"{detail} (stop_reason={user_agent_stop_reason})"
                            raise RuntimeError(
                                f"user-agent review did not complete: {detail}"
                            ) from parse_exc
                        repair_raw = call_openai_compatible(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            messages=_nanobot_format_repair_messages(
                                raw=raw or proc.stdout or "",
                                log_dir=log_dir,
                                session_id=session_id,
                            ),
                            timeout_sec=min(
                                timeout_sec or 300,
                                _int_env("CLAWGYM_USER_AGENT_FORMAT_REPAIR_TIMEOUT_SEC", 300),
                            ),
                            max_tokens=_int_env(
                                "CLAWGYM_USER_AGENT_FORMAT_REPAIR_MAX_TOKENS", 256
                            ),
                            temperature=0.0,
                            payload_overrides={
                                "chat_template_kwargs": {"enable_thinking": False},
                                "response_format": {"type": "json_object"},
                            },
                        )
                        stdout_chunks.append(
                            f"\n--- format repair {repair_index} ---\n{repair_raw}"
                        )
                        feedback, score = validate_user_agent_json(parse_json_object(repair_raw))
                        raw = repair_raw
                        user_agent_format_repaired = True
                        _write_text(raw_path, raw)
                        _write_json(result_path, {"feedback": feedback, "score": score})

                _write_text(stdout_path, "\n".join(stdout_chunks))
                _write_text(stderr_path, "\n".join(stderr_chunks))
                cot_started_at = time.time()
                cot_info = _save_user_agent_cot(workspace, log_dir, session_id)
                cot_save_duration_sec = time.time() - cot_started_at
                transcript_metrics = _nanobot_transcript_metrics(log_dir, session_id)

                # The workspace-aware UA owns the verifier decision and can run
                # review/run_code_verifier.sh with the same tool context it used
                # for the rest of the review.  Do not automatically run it a
                # second time in the host process: that duplicate execution does
                # not contribute to the UA score and can dominate post-run tail
                # latency.  Retain an explicit opt-in for standalone diagnostics.
                if _truthy_env("CLAWGYM_USER_AGENT_POSTRUN_VERIFIER", False):
                    verifier_started_at = time.time()
                    verifier_after = _run_scoring_workspace_verifier(workspace)
                    verifier_duration_sec = time.time() - verifier_started_at
                    _write_text(
                        workspace / "review" / "verifier_after_user_agent.stdout",
                        verifier_after.get("stdout") or "",
                    )
                    _write_text(
                        workspace / "review" / "verifier_after_user_agent.stderr",
                        verifier_after.get("stderr") or "",
                    )

                parse_started_at = time.time()
                feedback, score = _parse_nanobot_user_agent_result(
                    raw or proc.stdout or "", result_path
                )
                feedback_path = workspace / "review" / "user_feedback.txt"
                _write_text(feedback_path, feedback + "\n")
                parse_duration_sec = time.time() - parse_started_at
                verifier_probe = f"{' '.join(stdout_chunks)}\n{raw or ''}".lower()
                verifier_ran = (
                    "run_code_verifier.sh" in verifier_probe
                    or "reward/check.py" in verifier_probe
                    or _nanobot_transcript_contains(
                        log_dir,
                        session_id,
                        "run_code_verifier.sh",
                        "reward/check.py",
                    )
                )
                result = UserAgentRewardResult(
                    score=score,
                    feedback=feedback,
                    raw_response=raw or proc.stdout or "",
                    parse_error=None,
                    prompt_truncated=False,
                    workspace_truncated=False,
                    prompt_chars=len(message),
                    rules_source=rules_source,
                    rules_count=len(rules),
                    retries=attempt - 1,
                    backend="nanobot",
                    user_agent_profile=profile.name,
                    user_agent_model=profile.model,
                    user_agent_workspace=workspace_info.get("user_agent_workspace"),
                    final_workspace=workspace_info.get("final_workspace"),
                    trajectory_dir=workspace_info.get("trajectory_dir"),
                    user_agent_session_id=session_id,
                    user_agent_log_dir=str(log_dir),
                    nanobot_stdout_path=str(stdout_path),
                    nanobot_stderr_path=str(stderr_path),
                    verifier_helper=workspace_info.get("verifier_helper"),
                    verifier_ran_by_agent=verifier_ran,
                    verifier_current_score=verifier_after.get("score"),
                    verifier_current_returncode=verifier_after.get("returncode"),
                    verifier_current_stdout=verifier_after.get("stdout"),
                    verifier_current_stderr=verifier_after.get("stderr"),
                    user_agent_raw_response_path=str(raw_path),
                    user_feedback_path=str(feedback_path),
                    user_agent_prepare_duration_sec=prepare_duration_sec,
                    user_agent_feedback_duration_sec=feedback_duration_sec,
                    user_agent_cot_save_duration_sec=cot_save_duration_sec,
                    user_agent_verifier_duration_sec=verifier_duration_sec,
                    user_agent_parse_duration_sec=parse_duration_sec,
                    user_agent_total_duration_sec=time.time() - score_task_started_at,
                    user_agent_stop_reason=user_agent_stop_reason,
                    user_agent_run_success=user_agent_run_success,
                    user_agent_run_error=user_agent_run_error,
                    user_agent_format_repaired=user_agent_format_repaired,
                    **cot_info,
                    **transcript_metrics,
                )
                _cleanup_nanobot_scoring_workspace(workspace, workspace_info)
                return result
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    time.sleep(min(2 * attempt, 10))

    result = _fallback_result(
        code_verifier_score=code_verifier_score,
        raw=raw,
        parse_error=last_error,
        prompt_truncated=False,
        workspace_truncated=False,
        prompt_chars=len(message),
        rules_source=rules_source,
        rules_count=len(rules),
        retries=max_retries,
        backend="nanobot",
        user_agent_profile=profile.name,
        user_agent_model=profile.model,
        user_agent_workspace=workspace_info.get("user_agent_workspace"),
        final_workspace=workspace_info.get("final_workspace"),
        trajectory_dir=workspace_info.get("trajectory_dir"),
        user_agent_session_id=session_id or None,
        user_agent_log_dir=str(log_dir),
        verifier_helper=workspace_info.get("verifier_helper"),
        verifier_current_score=verifier_after.get("score"),
        verifier_current_returncode=verifier_after.get("returncode"),
        verifier_current_stdout=verifier_after.get("stdout"),
        verifier_current_stderr=verifier_after.get("stderr"),
        nanobot_stdout_path=str(stdout_path) if stdout_path else None,
        nanobot_stderr_path=str(stderr_path) if stderr_path else None,
        user_agent_raw_response_path=str(raw_path) if raw_path else None,
        user_feedback_path=str(feedback_path) if feedback_path else None,
        user_agent_prepare_duration_sec=prepare_duration_sec,
        user_agent_feedback_duration_sec=feedback_duration_sec,
        user_agent_cot_save_duration_sec=cot_save_duration_sec,
        user_agent_verifier_duration_sec=verifier_duration_sec,
        user_agent_parse_duration_sec=parse_duration_sec,
        user_agent_total_duration_sec=time.time() - score_task_started_at,
        user_agent_stop_reason=user_agent_stop_reason,
        user_agent_run_success=user_agent_run_success,
        user_agent_run_error=user_agent_run_error,
        user_agent_format_repaired=user_agent_format_repaired,
        **cot_info,
        **transcript_metrics,
    )
    if workspace_info.get("user_agent_workspace"):
        _cleanup_nanobot_scoring_workspace(Path(str(workspace_info["user_agent_workspace"])), workspace_info)
    return result


def score_task(
    *,
    task_id: str,
    task_dir: Path,
    task_prompt: str,
    workspace_dir: Path,
    final_message: str,
    code_verifier_score: float | None,
    evaluated_session_id: str | None = None,
    evaluated_log_dir: Path | None = None,
    profile: UserAgentProfile | None = None,
) -> UserAgentRewardResult:
    backend = profile.backend if profile is not None else user_agent_backend()
    if user_agent_backend_is_mixed(backend):
        profile = profile or user_agent_profiles()[0]
        backend = profile.backend
    if backend == "prompt":
        return _score_task_prompt(
            task_id=task_id,
            task_dir=task_dir,
            task_prompt=task_prompt,
            workspace_dir=workspace_dir,
            final_message=final_message,
            code_verifier_score=code_verifier_score,
            profile=profile,
        )
    if backend == "nanobot":
        return _score_task_nanobot(
            task_id=task_id,
            task_dir=task_dir,
            task_prompt=task_prompt,
            workspace_dir=workspace_dir,
            final_message=final_message,
            code_verifier_score=code_verifier_score,
            evaluated_session_id=evaluated_session_id,
            evaluated_log_dir=evaluated_log_dir,
            profile=profile,
        )
    raise ValueError(
        f"unsupported CLAWGYM_USER_AGENT_BACKEND={backend!r}; "
        "expected prompt, nanobot, or nanobot_mixed"
    )


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    num = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    den_left = sum((a - mean_left) ** 2 for a in left) ** 0.5
    den_right = sum((b - mean_right) ** 2 for b in right) ** 0.5
    if den_left == 0 or den_right == 0:
        return None
    return num / (den_left * den_right)


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return pearson(_rank_values(left), _rank_values(right))


@dataclass(frozen=True)
class DeferredUserAgentJob:
    task_id: str
    task_dir: str
    task_prompt: str
    final_message: str
    code_verifier_score: float
    session_id: str
    evaluated_log_dir: str
    run_dir: str
    evaluation: bool
    queued_at: float
    workspace_archive: bytes | None = None
    workspace_archive_error: str | None = None


def eval_user_agent_reward_enabled() -> bool:
    return _truthy_env("CLAWGYM_EVAL_USER_AGENT_REWARD", False)


def should_score_user_agent(*, evaluation: bool) -> bool:
    return eval_user_agent_reward_enabled() if evaluation else user_agent_reward_enabled()


def result_metadata(result: UserAgentRewardResult) -> dict[str, Any]:
    return asdict(result)


def archive_workspace(workspace_dir: Path) -> tuple[bytes | None, str | None]:
    try:
        if not workspace_dir.is_dir():
            return None, f"workspace_missing:{workspace_dir}"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            archive.add(workspace_dir, arcname="workspace", recursive=True)
        data = buffer.getvalue()
        max_bytes = max(1, _int_env("CLAWGYM_DEFERRED_UA_WORKSPACE_ARCHIVE_MAX_BYTES", 512 * 1024 * 1024))
        if len(data) > max_bytes:
            return None, f"workspace_archive_too_large:{len(data)}>{max_bytes}"
        return data, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _validate_archive_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe workspace archive member: {name}")
    if name != "workspace" and not name.startswith("workspace/"):
        raise ValueError(f"unexpected workspace archive member: {name}")


def _restore_workspace(data: bytes) -> tuple[Path, Path]:
    root = Path(
        os.environ.get(
            "CLAWGYM_DEFERRED_UA_WORKDIR",
            str(Path(tempfile.gettempdir()) / "clawgym-deferred-user-agent"),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    restore_root = Path(tempfile.mkdtemp(prefix="workspace-", dir=str(root)))
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
            for member in members:
                _validate_archive_member(member.name)
                target = restore_root / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported workspace archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"could not read workspace archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
        workspace_dir = restore_root / "workspace"
        if not workspace_dir.is_dir():
            raise FileNotFoundError(f"restored workspace missing: {workspace_dir}")
        return restore_root, workspace_dir
    except Exception:
        shutil.rmtree(restore_root, ignore_errors=True)
        raise


def _failed_deferred_result(job: DeferredUserAgentJob, error: str, profile: UserAgentProfile) -> UserAgentRewardResult:
    fallback = job.code_verifier_score if not _truthy_env("CLAWGYM_USER_AGENT_FAIL_CLOSED", True) else 0.0
    return UserAgentRewardResult(
        score=_apply_user_agent_reward_mode(fallback),
        feedback="",
        raw_response="",
        parse_error=error,
        prompt_truncated=False,
        workspace_truncated=False,
        prompt_chars=0,
        rules_source="unavailable",
        rules_count=0,
        retries=0,
        backend=profile.backend,
        user_agent_profile=profile.name,
        user_agent_model=profile.model,
    )


def score_deferred_job(
    job: DeferredUserAgentJob,
    *,
    profile: UserAgentProfile,
    worker_index: int,
) -> UserAgentRewardResult:
    started_at = time.time()
    queue_wait_sec = max(0.0, started_at - job.queued_at)

    def with_deferred_timing(result: UserAgentRewardResult) -> UserAgentRewardResult:
        return replace(
            result,
            user_agent_queue_wait_sec=queue_wait_sec,
            user_agent_score_duration_sec=max(0.0, time.time() - started_at),
        )

    placement = f"host={socket.gethostname()} pid={os.getpid()} thread={threading.get_ident()}"
    print(
        f"[ClawGym-UserAgent] start task={job.task_id} profile={profile.name} worker={worker_index} "
        f"queue_wait={queue_wait_sec:.2f}s {placement}",
        flush=True,
    )
    restore_root: Path | None = None
    try:
        if job.workspace_archive is None:
            error = job.workspace_archive_error or "workspace_archive_missing"
            return with_deferred_timing(_failed_deferred_result(job, error, profile))
        restore_root, workspace_dir = _restore_workspace(job.workspace_archive)
        result = score_task(
            task_id=job.task_id,
            task_dir=Path(job.task_dir),
            task_prompt=job.task_prompt,
            workspace_dir=workspace_dir,
            final_message=job.final_message,
            code_verifier_score=job.code_verifier_score,
            evaluated_session_id=job.session_id,
            evaluated_log_dir=Path(job.evaluated_log_dir),
            profile=profile,
        )
        print(
            f"[ClawGym-UserAgent] done task={job.task_id} profile={profile.name} "
            f"code_verifier={job.code_verifier_score:.4f} user_agent={result.score:.4f} "
            f"duration={time.time() - started_at:.2f}s",
            flush=True,
        )
        return with_deferred_timing(result)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[ClawGym-UserAgent] failed task={job.task_id} reward=0 ({error})", flush=True)
        return with_deferred_timing(_failed_deferred_result(job, error, profile))
    finally:
        if restore_root is not None:
            shutil.rmtree(restore_root, ignore_errors=True)


class UserAgentScoringPool:
    """Rank-zero CPU pool with weighted profile queues for deferred UA scoring."""

    def __init__(self, max_workers: int | None = None):
        profiles = user_agent_profiles()
        if max_workers is not None:
            remaining = max(1, int(max_workers))
            capped: list[UserAgentProfile] = []
            for profile in profiles:
                workers = min(max(0, profile.workers), remaining)
                if workers:
                    capped.append(
                        UserAgentProfile(
                            **{
                                **profile.__dict__,
                                "workers": workers,
                                "concurrency": min(profile.concurrency, workers),
                            }
                        )
                    )
                    remaining -= workers
                if remaining <= 0:
                    break
            profiles = capped
        if not profiles or not sum(max(0, profile.workers) for profile in profiles):
            raise ValueError("user-agent scoring pool requires at least one worker")
        self.profiles = profiles
        self.profile_worker_counts = {profile.name: max(0, profile.workers) for profile in profiles}
        self._queues: dict[str, queue.Queue[tuple[DeferredUserAgentJob, Future] | None]] = {
            profile.name: queue.Queue() for profile in profiles
        }
        self._schedule_lock = threading.Lock()
        self._weights = {profile.name: max(1, profile.workers) for profile in profiles}
        self._current = {profile.name: 0 for profile in profiles}
        self._total_weight = sum(self._weights.values())
        self._workers: list[threading.Thread] = []
        for profile in profiles:
            for worker_index in range(max(0, profile.workers)):
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(profile, worker_index),
                    name=f"user-agent-score-{profile.name}-{worker_index}",
                    daemon=True,
                )
                thread.start()
                self._workers.append(thread)
        print(
            f"[ClawGym-UserAgent] scoring pool started host={socket.gethostname()} pid={os.getpid()} "
            + " ".join(f"{profile.name}={profile.workers}" for profile in profiles),
            flush=True,
        )
        prewarm_task_rules_cache_async()

    def submit(self, job: DeferredUserAgentJob) -> Future:
        future: Future = Future()
        self._queues[self._select_profile_name()].put((job, future))
        return future

    def _select_profile_name(self) -> str:
        with self._schedule_lock:
            selected = ""
            selected_value: int | None = None
            for profile in self.profiles:
                name = profile.name
                value = self._current[name] + self._weights[name]
                self._current[name] = value
                if selected_value is None or value > selected_value:
                    selected, selected_value = name, value
            if not selected:
                raise RuntimeError("user-agent scoring pool has no schedulable profile")
            self._current[selected] -= self._total_weight
            return selected

    def _worker_loop(self, profile: UserAgentProfile, worker_index: int) -> None:
        work_queue = self._queues[profile.name]
        while True:
            item = work_queue.get()
            try:
                if item is None:
                    return
                job, future = item
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(score_deferred_job(job, profile=profile, worker_index=worker_index))
                except BaseException as exc:
                    future.set_exception(exc)
            finally:
                work_queue.task_done()

    def shutdown(self) -> None:
        for profile in self.profiles:
            for _ in range(max(0, profile.workers)):
                self._queues[profile.name].put(None)
        for worker in self._workers:
            worker.join()

    def __enter__(self) -> "UserAgentScoringPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


_SCORING_POOL: UserAgentScoringPool | None = None
_SCORING_POOL_LOCK = threading.Lock()


def get_scoring_pool() -> UserAgentScoringPool:
    global _SCORING_POOL
    if _SCORING_POOL is None:
        with _SCORING_POOL_LOCK:
            if _SCORING_POOL is None:
                _SCORING_POOL = UserAgentScoringPool()
    return _SCORING_POOL


def shutdown_scoring_pool() -> None:
    global _SCORING_POOL
    with _SCORING_POOL_LOCK:
        pool, _SCORING_POOL = _SCORING_POOL, None
    if pool is not None:
        pool.shutdown()


atexit.register(shutdown_scoring_pool)
