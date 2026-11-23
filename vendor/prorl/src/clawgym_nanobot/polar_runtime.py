"""Session-isolated host runtime for ClawGym Nanobot Polar workers.

The Polar gateway already runs inside the Slime Apptainer image on the rollout
node. Starting another container per Nanobot session is both unnecessary and
unsupported on clusters that disable nested user namespaces. This runtime keeps
the normal Polar session filesystem contract while executing commands directly
in the gateway container.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import shutil
import socket
import time
from pathlib import Path

from clawgym_nanobot.exec_memory_guard import (
    FATAL_PREFIX,
    SCOPE_ENV,
    SESSION_DIR_ENV,
    SESSION_ENV,
    terminate_guarded_session,
    write_json_atomic,
)
from clawgym_nanobot.landlock_exec import (
    SANDBOX_FATAL_PREFIX,
    SANDBOX_SETUP_FAILURE_EXIT,
    configured_unprivileged_identity,
    sandbox_command,
)
from polar.runtime.base import BaseRuntime, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecResult


logger = logging.getLogger(__name__)


class ClawGymLocalRuntime(BaseRuntime):
    """Execute one Polar session in its private gateway-side temp directory."""

    @property
    def runtime_id(self) -> str:
        return f"clawgym-local-{self.session_id}"

    async def start(self) -> None:
        for relative in (
            "artifacts",
            "home",
            "logs/agent",
            "logs/eval",
            "tmp",
            "workspace",
        ):
            (self.session_dir / relative).mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        if self._destroyed and getattr(self, "_cleanup_complete", False):
            return
        # Reject new exec calls before taking the process snapshot.  Keep a
        # separate completion flag so a cancelled/partial stop can be retried.
        self._destroyed = True
        survivors = await self._cleanup_guarded_processes()
        self._cleanup_complete = survivors == 0
        if survivors:
            self._emit_cleanup_failure(survivors)

    async def cleanup_agent_processes(self) -> None:
        """Reap all descendants left by a finished agent before evaluation."""

        survivors = await self._cleanup_guarded_processes()
        if survivors:
            self._emit_cleanup_failure(survivors)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        if self._destroyed:
            raise RuntimeError(f"runtime {self.runtime_id} was already stopped")
        translated_command = self._translate_text(command)
        translated_cwd = self._translate_path(cwd or self.spec.workdir or RUNTIME_SESSION_DIR)
        translated_env = {
            key: self._translate_text(str(value))
            for key, value in {**self.spec.env, **(env or {})}.items()
        }
        process_env = {**os.environ, **translated_env}
        guard_scope = os.environ.get(SCOPE_ENV)
        if guard_scope:
            admitted = await self._wait_for_memory_admission()
            if not admitted:
                return ExecResult(
                    return_code=75,
                    stderr=(
                        "WSClaw exec admission timed out while the node memory "
                        "guard was draining active sessions"
                    ),
                )
            # Write these after merging caller-provided values so a task cannot
            # escape node accounting by overriding its guard identity.
            process_env[SCOPE_ENV] = guard_scope
            process_env[SESSION_ENV] = self.session_id
            process_env[SESSION_DIR_ENV] = str(self.session_dir)
        # These paths are part of the isolation boundary, not caller-tunable
        # defaults.  The inherited container environment normally points HOME
        # at /root and TMPDIR at /tmp, both of which are intentionally
        # read-only after Landlock is installed.
        process_env["HOME"] = str(self.session_dir / "home")
        for name in ("TMPDIR", "TMP", "TEMP"):
            process_env[name] = str(self.session_dir / "tmp")
        process_env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            identity = configured_unprivileged_identity(process_env)
            command = sandbox_command(
                ["bash", "-c", translated_command],
                write_root=self.session_dir,
                identity=identity,
            )
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(translated_cwd),
                env=process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as exc:
            self._emit_fatal(
                component="wsclaw_exec_sandbox",
                reason="exec_sandbox_spawn_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._active_process = process
        # stop() can race the small window between the initial destroyed check
        # and process creation.  In that case, reap this newly tagged process
        # immediately instead of allowing it to outlive the completed session.
        if self._destroyed:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            await self._cleanup_guarded_processes(grace_seconds=0.0)
            self._active_process = None
            raise RuntimeError(f"runtime {self.runtime_id} stopped during exec startup")
        try:
            try:
                if timeout_sec is None:
                    stdout, stderr = await process.communicate()
                else:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=float(timeout_sec)
                    )
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                await self._cleanup_guarded_processes(grace_seconds=0.0)
                return ExecResult(return_code=-1)
        finally:
            self._active_process = None
        return_code = int(process.returncode or 0)
        stdout_text = stdout.decode(errors="replace") if stdout else None
        stderr_text = stderr.decode(errors="replace") if stderr else None
        if (
            return_code == SANDBOX_SETUP_FAILURE_EXIT
            and stderr_text
            and SANDBOX_FATAL_PREFIX in stderr_text
        ):
            self._emit_fatal(
                component="wsclaw_exec_sandbox",
                reason="exec_sandbox_initialization_failed",
                error=stderr_text[-4096:],
            )
            raise RuntimeError(
                "WSClaw exec sandbox initialization failed; fatal-resume marker emitted"
            )
        return ExecResult(
            return_code=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
        )

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        source = Path(local_path)
        target = self._translate_path(remote_path)
        if not source.is_file():
            raise FileNotFoundError(f"source file does not exist: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        source = Path(local_path)
        target = self._translate_path(remote_path)
        if not source.is_dir():
            raise FileNotFoundError(f"source directory does not exist: {source}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, symlinks=True)

    async def download_file(self, remote_path: str, local_path: str) -> None:
        source = self._translate_path(remote_path)
        target = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(f"runtime file does not exist: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        source = self._translate_path(remote_path)
        target = Path(local_path)
        if not source.is_dir():
            raise FileNotFoundError(f"runtime directory does not exist: {source}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target, symlinks=True)

    def _translate_path(self, value: str) -> Path:
        path = Path(value)
        try:
            relative = path.relative_to(RUNTIME_SESSION_DIR)
        except ValueError as exc:
            raise ValueError(
                f"ClawGym local runtime path must be under {RUNTIME_SESSION_DIR}: {value}"
            ) from exc
        return self.session_dir / relative

    def _translate_text(self, value: str) -> str:
        return value.replace(RUNTIME_SESSION_DIR, str(self.session_dir))

    async def _wait_for_memory_admission(self) -> bool:
        gate_raw = os.environ.get("WSCLAW_EXEC_GUARD_GATE_FILE")
        heartbeat_raw = os.environ.get("WSCLAW_EXEC_GUARD_HEARTBEAT_FILE")
        if not gate_raw or not heartbeat_raw:
            return True
        gate = Path(gate_raw)
        heartbeat = Path(heartbeat_raw)
        try:
            max_wait = float(os.environ.get("WSCLAW_EXEC_ADMISSION_TIMEOUT_SECONDS", "600"))
            heartbeat_max_age = float(
                os.environ.get("WSCLAW_EXEC_GUARD_HEARTBEAT_MAX_AGE_SECONDS", "15")
            )
        except ValueError:
            logger.warning("Invalid WSClaw exec memory admission timing configuration")
            return True
        started = time.monotonic()
        while gate.exists():
            try:
                heartbeat_age = time.time() - heartbeat.stat().st_mtime
            except OSError:
                # A dead guardian must not leave every rollout task waiting on
                # a stale gate forever.  The launcher separately treats guard
                # death as a fatal-resume condition.
                return True
            if heartbeat_age > heartbeat_max_age:
                logger.error(
                    "Ignoring stale WSClaw exec admission gate %s (heartbeat age %.1fs)",
                    gate,
                    heartbeat_age,
                )
                return True
            if max_wait >= 0 and time.monotonic() - started >= max_wait:
                return False
            if self._destroyed:
                return False
            await asyncio.sleep(0.5)
        return True

    async def _cleanup_guarded_processes(self, *, grace_seconds: float = 5.0) -> int:
        guard_scope = os.environ.get(SCOPE_ENV)
        if not guard_scope:
            return 0
        survivors = await asyncio.to_thread(
            terminate_guarded_session,
            guard_scope,
            self.session_id,
            grace_seconds=grace_seconds,
        )
        if survivors:
            logger.error(
                "WSClaw local runtime cleanup left %d guarded process(es) alive "
                "scope=%s session=%s",
                survivors,
                guard_scope,
                self.session_id,
            )
        return survivors

    def _emit_cleanup_failure(self, survivors: int) -> None:
        self._emit_fatal(
            component="wsclaw_exec_session_cleanup",
            reason="exec_session_cleanup_survivors",
            survivors=survivors,
        )

    def _emit_fatal(self, *, component: str, reason: str, **details: object) -> None:
        payload: dict[str, object] = {
            "action": "stop_and_resume",
            "component": component,
            "host": socket.gethostname(),
            "manual_resume_required": True,
            "rank": os.environ.get("NODE_RANK", "unknown"),
            "reason": reason,
            "run_id": os.environ.get("RUN_ID", "unknown"),
            "scope": os.environ.get(SCOPE_ENV, "unknown"),
            "session_id": self.session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        payload.update(details)
        marker_dir = os.environ.get("PRORL_FATAL_RESUME_MARKER_DIR")
        if marker_dir:
            try:
                rank = os.environ.get("NODE_RANK", "unknown")
                write_json_atomic(
                    Path(marker_dir) / f"rank{rank}-{component}.json",
                    payload,
                )
            except Exception:
                logger.exception("Failed to write WSClaw exec cleanup fatal marker")
        logger.error("%s %s", FATAL_PREFIX, json.dumps(payload, sort_keys=True))
