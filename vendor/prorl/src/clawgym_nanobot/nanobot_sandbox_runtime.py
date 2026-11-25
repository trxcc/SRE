from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from . import tasks
from .nanobot_landlock_exec import sandbox_command, configured_unprivileged_identity, SANDBOX_FATAL_PREFIX, SANDBOX_SETUP_FAILURE_EXIT

PROJECT_ROOT = tasks.PROJECT_ROOT

class WSClawSandboxError(RuntimeError):
    pass
DEFAULT_NANOBOT_ROOT = PROJECT_ROOT / "vendor" / "clawgym_nanobot" / "nanobot"
DEFAULT_CONDA_ROOT = Path("")
CAPTURE_TAIL_BYTES = int(os.environ.get("NANOBOT_SUBPROCESS_CAPTURE_TAIL_BYTES", str(1024 * 1024)))


@dataclass
class NanobotTurnResult:
    turn_index: int
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0


def write_nanobot_config(
    *,
    workspace_dir: Path,
    run_dir: Path,
    adapter_base_url: str,
    session_id: str,
    turn_type: str,
    sampling_params: dict[str, Any],
    model_id: str | None = None,
    context_window: int | None = None,
    max_tokens: int | None = None,
) -> Path:
    config_dir = run_dir / "state"
    config_dir.mkdir(parents=True, exist_ok=True)
    context_window = int(context_window or _positive_int_env("NANOBOT_CONTEXT_WINDOW", 65536))
    max_tokens = int(max_tokens or _positive_int_env("NANOBOT_MAX_TOKENS", int(sampling_params.get("max_new_tokens") or 8192)))
    max_tool_iterations = _positive_int_env("NANOBOT_MAX_TOOL_ITERATIONS", 200)
    temperature = _float_env_first(("NANOBOT_TEMPERATURE", "ROLLOUT_TEMPERATURE"), sampling_params.get("temperature", 0.7))
    top_p = _float_env_first(("NANOBOT_TOP_P", "ROLLOUT_TOP_P"), sampling_params.get("top_p", 1.0))
    top_k = _int_env_first(("NANOBOT_TOP_K", "ROLLOUT_TOP_K"), sampling_params.get("top_k", -1))
    enable_thinking = _env_truthy("NANOBOT_ENABLE_THINKING", False)
    fallback_parity = _env_truthy("NANOBOT_WSCLAW_FALLBACK_PARITY", False)
    show_reasoning = False if fallback_parity else _env_truthy("NANOBOT_SHOW_REASONING", enable_thinking)
    extra_body: dict[str, Any] = {
        "stream": False,
        "top_p": top_p,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if not fallback_parity and sampling_params.get("sampling_seed") is not None:
        extra_body["sampling_seed"] = sampling_params["sampling_seed"]

    web_enabled = nanobot_web_enabled()
    web_config = {"enable": True, "noWeb": not web_enabled}
    web_proxy = nanobot_web_proxy()
    if web_proxy:
        web_config["proxy"] = web_proxy

    agent_defaults: dict[str, Any] = {
        "workspace": str(workspace_dir),
        "provider": "custom",
        "model": model_id or f"{turn_type}__{session_id}",
        "maxTokens": max_tokens,
        "contextWindowTokens": context_window,
        "temperature": temperature,
        "maxToolIterations": max_tool_iterations,
        "timezone": os.environ.get("NANOBOT_TIMEZONE", "Asia/Shanghai"),
    }
    if not fallback_parity:
        agent_defaults["botIcon"] = ""

    config = {
        "providers": {
            "custom": {
                "apiBase": openai_api_base(adapter_base_url),
                "apiKey": os.environ.get("NANOBOT_OPENAI_API_KEY", session_id),
                "extraBody": extra_body,
            }
        },
        "agents": {"defaults": agent_defaults},
        "channels": {
            "sendProgress": False,
            "sendToolHints": False,
            "showReasoning": show_reasoning,
        },
        "tools": {
            "restrictToWorkspace": (
                False
                if fallback_parity
                else os.environ.get("NANOBOT_RESTRICT_TO_WORKSPACE", "1") != "0"
            ),
            "web": web_config,
            "exec": {
                "enable": os.environ.get("NANOBOT_EXEC_ENABLE", "1") != "0",
                "timeout": (
                    120
                    if fallback_parity
                    else _positive_int_env("NANOBOT_EXEC_TIMEOUT", 60)
                ),
                "noWeb": not web_enabled,
                "noWebCliTimeout": _positive_int_env("NANOBOT_NO_WEB_CLI_TIMEOUT", 5),
                # ExecTool intentionally starts commands with a minimal
                # environment. Preserve the curated dependency path used by
                # WSClaw tasks for generated Python programs as well.
                "allowedEnvKeys": ["PYTHONPATH"],
            },
        },
    }
    if enable_thinking and not fallback_parity:
        config["agents"]["defaults"]["reasoningEffort"] = (
            os.environ.get("NANOBOT_REASONING_EFFORT", "medium").strip() or "medium"
        )

    path = config_dir / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path



def _raise_setup_failure(run_root: Path, session_id: str, exc: Exception) -> None:
    from . import raw_source_integrity
    payload = {"action": "stop_and_resume", "component": "wsclaw_nanobot_sandbox",
               "reason": "sandbox_setup_exception", "manual_resume_required": True,
               "run_id": os.environ.get("RUN_ID", "unknown"), "session_id": session_id,
               "error": repr(exc), "at": time.time()}
    try:
        marker = os.environ.get("PRORL_FATAL_RESUME_MARKER_DIR", "")
        if marker:
            raw_source_integrity.write_json_atomic(Path(marker) / f"sandbox-{os.getpid()}-{time.time_ns()}.json", payload)
        raw_source_integrity.write_json_atomic(run_root / "sandbox-failure.json", payload)
    finally:
        raise WSClawSandboxError(f"Nanobot sandbox setup failed: {session_id}") from exc

def run_nanobot_turn(
    *,
    workspace_dir: Path,
    config_path: Path,
    session_id: str,
    turn_type: str,
    message: str,
    log_dir: Path,
    timeout: int | None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        run_root = workspace_dir.resolve().parent
        session_log = log_dir / session_id
        session_log.mkdir(parents=True, exist_ok=True)
        private_home = run_root / "sandbox-home"
        private_tmp = run_root / "sandbox-tmp"
        private_home.mkdir(exist_ok=True)
        private_tmp.mkdir(exist_ok=True)
        env_overrides = {**(env_overrides or {}), "HOME": str(private_home),
                         "TMPDIR": str(private_tmp), "TMP": str(private_tmp),
                         "TEMP": str(private_tmp), "PYTHONDONTWRITEBYTECODE": "1"}
        script = nanobot_shell_script(
            workspace_dir=workspace_dir,
            config_path=config_path,
            session_id=session_id,
            turn_type=turn_type,
            message=message,
            log_dir=log_dir,
            env_overrides=env_overrides,
        )
        cmd = sandbox_command(["bash", "-c", script], write_root=run_root,
                              extra_write_roots=(session_log,),
                              identity=configured_unprivileged_identity())
        # CPU runner and its tool children inherit the existing execution ceiling.
        # Apply it in a child wrapper, never in the Ray/GPU caller.
        cmd = [sys.executable, str(Path(__file__).resolve().parents[2] / "cpu_as_exec.py"), "--gib", "48", "--", *cmd]
    except Exception as exc:
        _raise_setup_failure(workspace_dir.parent, session_id, exc)
    with (
        tempfile.NamedTemporaryFile(prefix="nanobot-stdout-", suffix=".log") as stdout_file,
        tempfile.NamedTemporaryFile(prefix="nanobot-stderr-", suffix=".log") as stderr_file,
    ):
        # A resumed/retried session can already contain a successful old turn.
        # Only markers created by this subprocess may authorize exit recovery.
        existing_end_names = frozenset(
            p.name for p in (log_dir / session_id).glob("*_agent-end.json")
        )
        try:
            proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, start_new_session=True)
        except OSError as exc:
            _raise_setup_failure(run_root, session_id, exc)
        try:
            recovered_completed_turn = _wait_for_nanobot_process(
                proc,
                cmd=cmd,
                log_dir=log_dir,
                session_id=session_id,
                timeout=timeout,
                ignore_end_names=existing_end_names,
            )
        except subprocess.TimeoutExpired as exc:
            _stop_process_group(proc, terminate_grace_s=0.0)
            stdout_file.flush()
            stderr_file.flush()
            stdout = read_file_tail_text(Path(stdout_file.name), CAPTURE_TAIL_BYTES)
            stderr = read_file_tail_text(Path(stderr_file.name), CAPTURE_TAIL_BYTES)
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=exc.timeout, output=stdout, stderr=stderr) from exc

        stdout_file.flush()
        stderr_file.flush()
        stdout = read_file_tail_text(Path(stdout_file.name), CAPTURE_TAIL_BYTES)
        stderr = read_file_tail_text(Path(stderr_file.name), CAPTURE_TAIL_BYTES)
        if recovered_completed_turn:
            stderr = (
                stderr.rstrip()
                + "\n[nanobot-runtime] recovered completed turn after bounded exit cleanup\n"
            ).lstrip("\n")

    if proc.returncode == SANDBOX_SETUP_FAILURE_EXIT and SANDBOX_FATAL_PREFIX in stderr:
        from . import raw_source_integrity
        payload = {"action": "stop_and_resume", "component": "wsclaw_nanobot_sandbox",
                   "reason": "sandbox_initialization_failed", "manual_resume_required": True,
                   "run_id": os.environ.get("RUN_ID", "unknown"), "session_id": session_id,
                   "error": stderr[-4096:], "at": time.time()}
        try:
            raw_source_integrity.write_json_atomic(run_root / "sandbox-failure.json", payload)
            marker = os.environ.get("PRORL_FATAL_RESUME_MARKER_DIR", "")
            if marker:
                raw_source_integrity.write_json_atomic(Path(marker) / f"sandbox-{os.getpid()}-{time.time_ns()}.json", payload)
        finally:
            raise WSClawSandboxError(f"Nanobot filesystem sandbox failed: {session_id}")
    return subprocess.CompletedProcess(
        cmd,
        0 if recovered_completed_turn else proc.returncode,
        stdout,
        stderr,
    )


def _wait_for_nanobot_process(
    proc: subprocess.Popen[Any],
    *,
    cmd: list[str],
    log_dir: Path,
    session_id: str,
    timeout: int | None,
    ignore_end_names: frozenset[str] = frozenset(),
) -> bool:
    """Wait for Nanobot and recover only a fully completed stuck CLI exit.

    Nanobot writes its durable ``agent_end`` record before some background
    resources finish closing.  A rare stuck close used to turn an otherwise
    completed turn into a task timeout.  The marker is not enough by itself:
    it must explicitly say both ``success=true`` and ``stop_reason=completed``.
    """

    started_at = time.monotonic()
    timeout_deadline = None if timeout is None else started_at + float(timeout)
    completed_at: float | None = None
    exit_grace_s = _positive_float_env("NANOBOT_AGENT_END_EXIT_GRACE_SECONDS", 60.0)

    while True:
        returncode = proc.poll()
        if returncode is not None:
            return False

        now = time.monotonic()
        if completed_at is None and _has_successful_completed_agent_end(log_dir, session_id, ignore_names=ignore_end_names):
            completed_at = now
        if completed_at is not None and now >= completed_at + exit_grace_s:
            _stop_process_group(proc, terminate_grace_s=5.0)
            return True
        if timeout_deadline is not None and now >= timeout_deadline:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

        deadlines = [now + 1.0]
        if completed_at is not None:
            deadlines.append(completed_at + exit_grace_s)
        if timeout_deadline is not None:
            deadlines.append(timeout_deadline)
        wait_s = max(0.01, min(deadlines) - now)
        try:
            proc.wait(timeout=wait_s)
            return False
        except subprocess.TimeoutExpired:
            continue


def _has_successful_completed_agent_end(log_dir: Path, session_id: str, *, ignore_names: frozenset[str] = frozenset()) -> bool:
    session_dir = log_dir / session_id
    if not session_dir.is_dir():
        return False
    for path in reversed(sorted(session_dir.glob("*_agent-end.json"))):
        if path.name in ignore_names:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return bool(
            isinstance(payload, dict)
            and payload.get("stage") == "agent_end"
            and payload.get("success") is True
            and str(payload.get("stop_reason") or "").strip() == "completed"
        )
    return False


def _stop_process_group(
    proc: subprocess.Popen[Any], *, terminate_grace_s: float
) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM if terminate_grace_s > 0 else signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        proc.terminate() if terminate_grace_s > 0 else proc.kill()
    if terminate_grace_s > 0:
        try:
            proc.wait(timeout=terminate_grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def nanobot_shell_script(
    *,
    workspace_dir: Path,
    config_path: Path,
    session_id: str,
    turn_type: str,
    message: str,
    log_dir: Path,
    env_overrides: dict[str, str] | None = None,
) -> str:
    python_exe = nanobot_python_executable()
    if not python_exe.exists():
        python_exe = Path(sys.executable)
    runner_path = nanobot_runner_path()
    env_bin = python_exe.parent
    fallback_parity = _env_truthy("NANOBOT_WSCLAW_FALLBACK_PARITY", False)
    fallback_runtime_bin = workspace_dir / ".runtime" / "bin"
    pythonpath = nanobot_pythonpath()
    web_enabled = nanobot_web_enabled()
    path_prefix = f"{shlex.quote(str(env_bin))}:$PATH"
    lines = [
        "set -euo pipefail",
        "unset SSL_CERT_FILE",
        "unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M",
        "unset PYTHONPATH",
    ]
    if not web_enabled:
        no_web_bin = workspace_dir / ".nanobot-no-web-bin"
        lines.extend(
            [
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
                'export no_proxy="*"',
                'export NO_PROXY="${no_proxy}"',
                "export WGETRC=/dev/null",
                "export CURL_HOME=/nonexistent",
                "export GIT_CONFIG_NOSYSTEM=1",
                "export GIT_CONFIG_GLOBAL=/dev/null",
                "export PIP_CONFIG_FILE=/dev/null",
                "export PIP_NO_INDEX=1",
                "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                'export PIP_DEFAULT_TIMEOUT="${NANOBOT_NO_WEB_CLI_TIMEOUT:-5}"',
                'export npm_config_proxy=""',
                'export npm_config_https_proxy=""',
                'export NPM_CONFIG_PROXY=""',
                'export NPM_CONFIG_HTTPS_PROXY=""',
                f"export NANOBOT_NO_WEB_BIN={shlex.quote(str(no_web_bin))}",
                'rm -rf "${NANOBOT_NO_WEB_BIN}"',
                'mkdir -p "${NANOBOT_NO_WEB_BIN}"',
                'cat > "${NANOBOT_NO_WEB_BIN}/wget" <<\'NANOBOT_WGET_EOF\'',
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'for arg in "$@"; do',
                '  case "$arg" in',
                '    http://localhost*|https://localhost*|http://127.*|https://127.*|http://0.0.0.0*|https://0.0.0.0*|http://\\[::1\\]*|https://\\[::1\\]*) ;;',
                '    http://*|https://*) echo "Error: external network is disabled (NANOBOT_WEB_ENABLE=0)" >&2; exit 28 ;;',
                "  esac",
                "done",
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
                'export no_proxy="*"',
                'export NO_PROXY="${no_proxy}"',
                'timeout="${NANOBOT_NO_WEB_CLI_TIMEOUT:-5}"',
                'if [ ! -x /usr/bin/wget ]; then echo "wget is unavailable in this environment" >&2; exit 127; fi',
                'exec /usr/bin/wget --config=/dev/null -e use_proxy=no -T "${timeout}" -t 1 "$@"',
                "NANOBOT_WGET_EOF",
                'cat > "${NANOBOT_NO_WEB_BIN}/curl" <<\'NANOBOT_CURL_EOF\'',
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'for arg in "$@"; do',
                '  case "$arg" in',
                '    http://localhost*|https://localhost*|http://127.*|https://127.*|http://0.0.0.0*|https://0.0.0.0*|http://\\[::1\\]*|https://\\[::1\\]*) ;;',
                '    http://*|https://*) echo "Error: external network is disabled (NANOBOT_WEB_ENABLE=0)" >&2; exit 28 ;;',
                "  esac",
                "done",
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
                'export no_proxy="*"',
                'export NO_PROXY="${no_proxy}"',
                'timeout="${NANOBOT_NO_WEB_CLI_TIMEOUT:-5}"',
                'if [ ! -x /usr/bin/curl ]; then echo "curl is unavailable in this environment" >&2; exit 127; fi',
                'exec /usr/bin/curl --disable --noproxy "*" --proxy "" --connect-timeout "${timeout}" --max-time "${timeout}" "$@"',
                "NANOBOT_CURL_EOF",
                'cat > "${NANOBOT_NO_WEB_BIN}/git" <<\'NANOBOT_GIT_EOF\'',
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'case "${1:-}" in',
                '  clone|fetch|pull|push|ls-remote|submodule) echo "Error: external git network operations are disabled (NANOBOT_WEB_ENABLE=0)" >&2; exit 128 ;;',
                "esac",
                "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
                'export no_proxy="*"',
                'export NO_PROXY="${no_proxy}"',
                "export GIT_CONFIG_NOSYSTEM=1",
                "export GIT_CONFIG_GLOBAL=/dev/null",
                "export GIT_TERMINAL_PROMPT=0",
                "export GIT_HTTP_LOW_SPEED_LIMIT=1",
                'export GIT_HTTP_LOW_SPEED_TIME="${NANOBOT_NO_WEB_CLI_TIMEOUT:-5}"',
                'if [ ! -x /usr/bin/git ]; then echo "git is unavailable in this environment" >&2; exit 127; fi',
                'exec /usr/bin/git -c http.proxy= -c https.proxy= -c http.noProxy="*" "$@"',
                "NANOBOT_GIT_EOF",
                'chmod +x "${NANOBOT_NO_WEB_BIN}/wget" "${NANOBOT_NO_WEB_BIN}/curl" "${NANOBOT_NO_WEB_BIN}/git"',
            ]
        )
        path_prefix = f'"${{NANOBOT_NO_WEB_BIN}}":{shlex.quote(str(env_bin))}:$PATH'
    else:
        lines.extend(
            [
                f'export no_proxy="{nanobot_no_proxy_value()}"',
                'export NO_PROXY="${no_proxy}"',
            ]
        )
    lines.append(f"export PATH={path_prefix}")
    if pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(pythonpath)}")
    lines.extend(
        [
            (
                "export NANOBOT_NO_SPINNER=1"
                if fallback_parity
                else 'export NANOBOT_NO_SPINNER="${NANOBOT_NO_SPINNER:-1}"'
            ),
            f'echo "[nanobot] python_exe={shlex.quote(str(python_exe))}" >&2',
            f'echo "[nanobot] runner={shlex.quote(str(runner_path))}" >&2',
            f'echo "[nanobot] root={shlex.quote(str(nanobot_root()))}" >&2',
            'echo "[nanobot] pythonpath=${PYTHONPATH:-}" >&2',
        ]
    )
    for key, value in sorted((env_overrides or {}).items()):
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid nanobot env override key: {key!r}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend(
        [
            f"export NANOBOT_TURN_TYPE={shlex.quote(turn_type)}",
            f"export NANOBOT_LOG_DIR={shlex.quote(str(log_dir))}",
            'export NANOBOT_LLM_TIMEOUT_S="${NANOBOT_RL_LLM_TIMEOUT_S:-0}"',
            'export NANOBOT_OPENAI_COMPAT_TIMEOUT_S="${NANOBOT_RL_OPENAI_COMPAT_TIMEOUT_S:-0}"',
            (
                # WSClaw fallback parity is an exact runtime mode: an inherited
                # NANOBOT_RL_NO_MAX_TOKENS=0 must not silently override it.
                'export NANOBOT_RL_NO_MAX_TOKENS=1'
                if fallback_parity
                else 'export NANOBOT_RL_NO_MAX_TOKENS="${NANOBOT_RL_NO_MAX_TOKENS:-0}"'
            ),
            (
                f"export NANOBOT_RL_EXEC_PATH_PREPEND={shlex.quote(str(fallback_runtime_bin))}"
                if fallback_parity
                else (
                    f"if [ -d {shlex.quote(str(fallback_runtime_bin))} ]; then "
                    f"export NANOBOT_RL_EXEC_PATH_PREPEND={shlex.quote(str(fallback_runtime_bin))}; "
                    f"else export NANOBOT_RL_EXEC_PATH_PREPEND=\"${{NANOBOT_RL_EXEC_PATH_PREPEND:-{str(env_bin)}}}\"; fi"
                )
            ),
            f"export NANOBOT_CONFIG={shlex.quote(str(config_path))}",
            f"export NANOBOT_WORKSPACE_ACTUAL={shlex.quote(str(workspace_dir))}",
            f"export NANOBOT_WORKSPACE_DISPLAY={shlex.quote(nanobot_workspace_display(workspace_dir))}",
            f"cd {shlex.quote(str(workspace_dir))}",
            f"exec {shlex.quote(str(python_exe))} {shlex.quote(str(runner_path))} agent "
            f"-m {shlex.quote(message)} "
            "--no-markdown "
            f"-s {shlex.quote(session_id)} "
            f"-w {shlex.quote(str(workspace_dir))} "
            f"-c {shlex.quote(str(config_path))}",
        ]
    )
    return "\n".join(lines)


def write_turn_logs(run_dir: Path, turn: NanobotTurnResult) -> None:
    (run_dir / f"turn-{turn.turn_index}.stdout").write_text(turn.stdout_tail, encoding="utf-8")
    (run_dir / f"turn-{turn.turn_index}.stderr").write_text(turn.stderr_tail, encoding="utf-8")


def nanobot_work_root() -> Path:
    root = Path(os.environ.get("NANOBOT_WORKDIR", str(Path(tempfile.gettempdir()) / "nanobot")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def nanobot_log_dir(args: Any) -> Path:
    explicit = os.environ.get("NANOBOT_LOG_DIR", "").strip() or os.environ.get("CLAWARENA_NANOBOT_LOG_DIR", "").strip()
    if explicit:
        return Path(explicit)
    if getattr(args, "save", None):
        return Path(args.save) / "nanobot_logs"
    return Path(tempfile.gettempdir()) / "nanobot-logs"


def nanobot_root() -> Path:
    return Path(os.environ.get("NANOBOT_ROOT", str(DEFAULT_NANOBOT_ROOT)))


def nanobot_conda_root() -> Path:
    return Path(os.environ.get("NANOBOT_CONDA_ROOT", str(DEFAULT_CONDA_ROOT)))


def nanobot_conda_env_name() -> str:
    return os.environ.get("NANOBOT_CONDA_ENV_NAME", "nanobot-openclaw")


def nanobot_env_dir() -> Path:
    env_name = nanobot_conda_env_name()
    env_path = Path(env_name)
    if env_path.is_absolute():
        return env_path
    return nanobot_conda_root() / "envs" / env_name


def nanobot_python_executable() -> Path:
    return nanobot_env_dir() / "bin" / "python"


def nanobot_runner_path() -> Path:
    explicit = os.environ.get("NANOBOT_RL_RUNNER", "").strip()
    if explicit:
        return Path(explicit)
    return nanobot_root() / "nanobot" / "rl_runner.py"


def nanobot_pythonpath() -> str:
    entries: list[str] = []
    if _env_truthy("NANOBOT_IMPORT_LOCAL_PACKAGE", False):
        entries.append(str(nanobot_root()))
    extra = os.environ.get("NANOBOT_EXTRA_PYTHONPATH", "").strip()
    if extra:
        entries.extend(part for part in extra.split(os.pathsep) if part)
    return os.pathsep.join(entries)


def openai_api_base(adapter_base_url: str) -> str:
    return adapter_base_url.rstrip("/") + "/v1"


def nanobot_workspace_display(workspace_dir: Path) -> str:
    explicit = (
        os.environ.get("NANOBOT_WORKSPACE_DISPLAY", "").strip()
        or os.environ.get("CLAWARENA_NANOBOT_WORKSPACE_DISPLAY", "").strip()
    )
    return explicit or str(workspace_dir)


def nanobot_web_enabled() -> bool:
    return os.environ.get("NANOBOT_WEB_ENABLE", "1") == "1"


def nanobot_web_proxy() -> str | None:
    if not nanobot_web_enabled():
        return None
    explicit = os.environ.get("NANOBOT_WEB_PROXY")
    if explicit is not None:
        return explicit.strip() or None
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "WANDB_HTTPS_PROXY",
        "WANDB_HTTP_PROXY",
        "WANDB_PROXY_URL",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def nanobot_default_no_proxy() -> str:
    return ",".join(
        [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "10.0.0.0/8",
            "100.0.0.0/8",
            "${MASTER_ADDR:-}",
            "${RAY_NODE_IP:-}",
            "${PROXY_HOST_IP:-}",
            "${TRAIN_NODE_HOST:-}",
            "${HOSTNAME:-}",
        ]
    )


def nanobot_no_proxy_value() -> str:
    if not nanobot_web_enabled():
        return "*"
    return f"${{CCR_NO_PROXY:-{nanobot_default_no_proxy()}}}"


def read_file_tail_text(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(default)


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


def _float_env_first(names: tuple[str, ...], default: Any) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float(default)


def _int_env_first(names: tuple[str, ...], default: Any) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return int(default)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_write_nanobot_config = write_nanobot_config
_run_nanobot_turn = run_nanobot_turn
_nanobot_shell_script = nanobot_shell_script
_write_turn_logs = write_turn_logs
_nanobot_work_root = nanobot_work_root
_nanobot_log_dir = nanobot_log_dir
_nanobot_root = nanobot_root
_nanobot_conda_root = nanobot_conda_root
_nanobot_conda_env_name = nanobot_conda_env_name
_nanobot_env_dir = nanobot_env_dir
_nanobot_python_executable = nanobot_python_executable
_nanobot_runner_path = nanobot_runner_path
_nanobot_pythonpath = nanobot_pythonpath
_openai_api_base = openai_api_base
_nanobot_workspace_display = nanobot_workspace_display
_nanobot_web_enabled = nanobot_web_enabled
_nanobot_web_proxy = nanobot_web_proxy
_nanobot_default_no_proxy = nanobot_default_no_proxy
_nanobot_no_proxy_value = nanobot_no_proxy_value
_read_file_tail_text = read_file_tail_text
_env_truthy = _env_truthy
