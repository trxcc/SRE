"""Node-level host-memory guard for ClawGym local exec processes.

The local Polar runtime deliberately executes task commands on the host side of
the gateway container.  This module tags those commands by rollout session and
protects the rest of the workload with one aggregate budget per node.  There is
no per-session memory ceiling: a single task may use the whole available exec
budget when the other tasks are idle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

SCOPE_ENV = "WSCLAW_EXEC_GUARD_SCOPE"
SESSION_ENV = "WSCLAW_EXEC_SESSION_ID"
SESSION_DIR_ENV = "WSCLAW_EXEC_SESSION_DIR"
DEGRADED_PREFIX = "PRORL_DEGRADED"
FATAL_PREFIX = "PRORL_FATAL_RESUME_REQUIRED"
GIB = 1024**3


@dataclass(frozen=True)
class GuardedProcess:
    """One live process attributed to a guarded Polar session."""

    pid: int
    pgid: int
    session_id: str
    session_dir: str | None
    memory_bytes: int


def _read_environ(path: Path) -> dict[str, str]:
    try:
        fields = path.read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return {}
    result: dict[str, str] = {}
    for field in fields:
        key, separator, value = field.partition(b"=")
        if not separator:
            continue
        try:
            result[key.decode()] = value.decode(errors="replace")
        except UnicodeDecodeError:
            continue
    return result


def _read_status_memory(path: Path) -> int:
    """Return private-ish resident memory plus swap for one process.

    ``RssAnon`` avoids counting the same shared libraries once per agent.  Old
    kernels without that field fall back to VmRSS.
    """

    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, remainder = line.partition(":")
            if not separator or key not in {"RssAnon", "RssShmem", "VmRSS", "VmSwap"}:
                continue
            raw = remainder.strip().split(maxsplit=1)[0]
            if raw.isdigit():
                values[key] = int(raw) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return 0
    resident = values.get("RssAnon")
    if resident is None:
        resident = values.get("VmRSS", 0)
    else:
        resident += values.get("RssShmem", 0)
    return resident + values.get("VmSwap", 0)


def collect_guarded_processes(
    scope: str,
    *,
    session_id: str | None = None,
    proc_root: Path = Path("/proc"),
) -> list[GuardedProcess]:
    """Collect all tagged task processes in *scope* from procfs."""

    processes: list[GuardedProcess] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return processes
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in {1, os.getpid()}:
            continue
        environ = _read_environ(entry / "environ")
        if environ.get(SCOPE_ENV) != scope:
            continue
        process_session = environ.get(SESSION_ENV)
        if not process_session or (session_id is not None and process_session != session_id):
            continue
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            continue
        processes.append(
            GuardedProcess(
                pid=pid,
                pgid=pgid,
                session_id=process_session,
                session_dir=environ.get(SESSION_DIR_ENV),
                memory_bytes=_read_status_memory(entry / "status"),
            )
        )
    return processes


def group_memory(processes: list[GuardedProcess]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for process in processes:
        totals[process.session_id] = totals.get(process.session_id, 0) + process.memory_bytes
    return totals


def select_victim_session(
    current: dict[str, int],
    previous: dict[str, int],
    pending: set[str] | None = None,
) -> str | None:
    """Choose a reclaim candidate without imposing a per-session ceiling."""

    excluded = pending or set()
    candidates = [session_id for session_id in current if session_id not in excluded]
    if not candidates:
        return None

    def score(session_id: str) -> tuple[bool, int, int]:
        size = current[session_id]
        growth = max(0, size - previous.get(session_id, 0))
        return growth >= GIB // 4, growth, size

    return max(candidates, key=score)


def signal_processes(processes: list[GuardedProcess], sig: signal.Signals) -> None:
    """Signal all listed processes without touching the caller's process group."""

    own_pgid = os.getpgrp()
    for pgid in sorted({process.pgid for process in processes}):
        if pgid <= 1 or pgid == own_pgid:
            continue
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    for process in processes:
        try:
            os.kill(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate_guarded_session(
    scope: str,
    session_id: str,
    *,
    grace_seconds: float = 5.0,
    proc_root: Path = Path("/proc"),
) -> int:
    """Boundedly terminate every tagged process in one runtime session."""

    processes = collect_guarded_processes(scope, session_id=session_id, proc_root=proc_root)
    if not processes:
        return 0
    signal_processes(processes, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        processes = collect_guarded_processes(scope, session_id=session_id, proc_root=proc_root)
        if not processes:
            return 0
    processes = collect_guarded_processes(scope, session_id=session_id, proc_root=proc_root)
    signal_processes(processes, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        processes = collect_guarded_processes(scope, session_id=session_id, proc_root=proc_root)
        if not processes:
            return 0
        time.sleep(0.1)
    return len(collect_guarded_processes(scope, session_id=session_id, proc_root=proc_root))


def cleanup_scope(scope: str, *, grace_seconds: float = 5.0) -> int:
    """Terminate all sessions in a workload scope and return survivors."""

    session_ids = sorted({process.session_id for process in collect_guarded_processes(scope)})
    for session_id in session_ids:
        terminate_guarded_session(scope, session_id, grace_seconds=grace_seconds)
    return len(collect_guarded_processes(scope))


def _current_cgroup_dir(
    *,
    cgroup_file: Path = Path("/proc/self/cgroup"),
    cgroup_mount: Path = Path("/sys/fs/cgroup"),
) -> Path | None:
    try:
        for line in cgroup_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.split("::", 1)[1]
                return cgroup_mount / relative.lstrip("/")
    except OSError:
        pass
    return None


def cgroup_available_bytes(cgroup_dir: Path | None = None) -> int | None:
    """Return remaining memory in the current finite cgroup, when visible."""

    directory = cgroup_dir or _current_cgroup_dir()
    if directory is None:
        return None
    try:
        raw_limit = (directory / "memory.max").read_text(encoding="ascii").strip()
        if raw_limit == "max":
            return None
        limit = int(raw_limit)
        current = int((directory / "memory.current").read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return max(0, limit - current)


def memory_available_bytes(
    meminfo_path: Path = Path("/proc/meminfo"),
    *,
    cgroup_dir: Path | None = None,
) -> int:
    host_available = 0
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                host_available = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    allocation_available = cgroup_available_bytes(cgroup_dir)
    if allocation_available is None:
        return host_available
    if host_available <= 0:
        return allocation_available
    return min(host_available, allocation_available)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def emit_event(
    prefix: str,
    payload: dict[str, object],
    *,
    stream: TextIO = sys.stdout,
) -> None:
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", file=stream, flush=True)


class ExecCgroup:
    """Best-effort shared cgroup-v2 envelope for all tagged exec processes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._oom_kill_count = self._read_event("oom_kill")

    @classmethod
    def create(cls, scope: str, *, soft_bytes: int, hard_bytes: int) -> ExecCgroup | None:
        try:
            parent = _current_cgroup_dir()
            if parent is None:
                return None
            digest = hashlib.sha256(scope.encode()).hexdigest()[:16]
            path = parent / f"wsclaw-exec-{digest}"
            path.mkdir(mode=0o700, exist_ok=True)
            (path / "memory.high").write_text(str(soft_bytes), encoding="ascii")
            (path / "memory.max").write_text(str(hard_bytes), encoding="ascii")
            oom_group = path / "memory.oom.group"
            if oom_group.exists():
                oom_group.write_text("0", encoding="ascii")
            return cls(path)
        except (OSError, ValueError):
            return None

    def attach(self, processes: list[GuardedProcess]) -> None:
        procs_file = self.path / "cgroup.procs"
        for process in processes:
            try:
                procs_file.write_text(str(process.pid), encoding="ascii")
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                continue

    def memory_current(self) -> int:
        try:
            return int((self.path / "memory.current").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return 0

    def consume_oom_kill_delta(self) -> int:
        current = self._read_event("oom_kill")
        delta = max(0, current - self._oom_kill_count)
        self._oom_kill_count = current
        return delta

    def _read_event(self, name: str) -> int:
        try:
            for line in (self.path / "memory.events").read_text(encoding="ascii").splitlines():
                key, value = line.split(maxsplit=1)
                if key == name:
                    return int(value)
        except (OSError, ValueError):
            pass
        return 0


class NodeExecMemoryGuard:
    """Enforce aggregate exec and host-memory headroom on one node."""

    def __init__(
        self,
        *,
        scope: str,
        rank: str,
        state_dir: Path,
        fatal_marker_dir: Path,
        soft_bytes: int,
        hard_bytes: int,
        block_available_bytes: int,
        emergency_available_bytes: int,
        fatal_available_bytes: int,
        poll_seconds: float,
        terminate_grace_seconds: float,
    ) -> None:
        self.scope = scope
        self.rank = rank
        self.state_dir = state_dir
        self.fatal_marker_dir = fatal_marker_dir
        self.soft_bytes = soft_bytes
        self.hard_bytes = hard_bytes
        self.block_available_bytes = block_available_bytes
        self.emergency_available_bytes = emergency_available_bytes
        self.fatal_available_bytes = fatal_available_bytes
        self.poll_seconds = poll_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self.gate_file = state_dir / "block-new-exec.json"
        self.heartbeat_file = state_dir / "heartbeat.json"
        self.state_file = state_dir / "state.json"
        self.stop_requested = False
        self.blocked = False
        self.fatal_streak = 0
        self.fatal_emitted = False
        self.previous_memory: dict[str, int] = {}
        self.pending_kills: dict[str, float] = {}
        self.cgroup = ExecCgroup.create(scope, soft_bytes=soft_bytes, hard_bytes=hard_bytes)

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def run(self) -> int:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.fatal_marker_dir.mkdir(parents=True, exist_ok=True)
        backend = "cgroup-v2" if self.cgroup is not None else "procfs-watchdog"
        emit_event(
            "WSCLAW_EXEC_MEMORY_GUARD_STARTED",
            {
                "backend": backend,
                "rank": self.rank,
                "scope": self.scope,
                "soft_bytes": self.soft_bytes,
                "hard_bytes": self.hard_bytes,
            },
        )
        while not self.stop_requested:
            started = time.monotonic()
            self._poll_once(backend)
            remaining = self.poll_seconds - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        self.gate_file.unlink(missing_ok=True)
        emit_event(
            "WSCLAW_EXEC_MEMORY_GUARD_STOPPED",
            {"backend": backend, "rank": self.rank, "scope": self.scope},
        )
        return 0

    def _poll_once(self, backend: str) -> None:
        processes = collect_guarded_processes(self.scope)
        if self.cgroup is not None:
            self.cgroup.attach(processes)
        by_session = group_memory(processes)
        sampled_exec_bytes = sum(by_session.values())
        cgroup_bytes = self.cgroup.memory_current() if self.cgroup is not None else 0
        exec_bytes = cgroup_bytes or sampled_exec_bytes
        available_bytes = memory_available_bytes()
        now = time.time()

        self._finish_pending_kills(processes)
        should_block = exec_bytes >= self.soft_bytes or (
            available_bytes > 0 and available_bytes <= self.block_available_bytes
        )
        if self.blocked:
            release_exec = exec_bytes < int(self.soft_bytes * 0.9)
            release_host = available_bytes == 0 or available_bytes > self.block_available_bytes + 16 * GIB
            should_block = not (release_exec and release_host)
        self._set_admission_gate(should_block, exec_bytes, available_bytes, now)

        oom_kills = self.cgroup.consume_oom_kill_delta() if self.cgroup is not None else 0
        if oom_kills:
            emit_event(
                DEGRADED_PREFIX,
                self._payload(
                    reason="exec_cgroup_oom_kill",
                    action="watch",
                    exec_bytes=exec_bytes,
                    available_bytes=available_bytes,
                    extra={"oom_kills": oom_kills},
                ),
            )

        emergency = exec_bytes >= self.hard_bytes or (
            available_bytes > 0 and available_bytes <= self.emergency_available_bytes
        )
        if emergency:
            victim = self._select_victim(by_session)
            if victim is not None:
                victim_processes = [p for p in processes if p.session_id == victim]
                signal_processes(victim_processes, signal.SIGTERM)
                self.pending_kills[victim] = time.monotonic() + self.terminate_grace_seconds
                emit_event(
                    DEGRADED_PREFIX,
                    self._payload(
                        reason="exec_memory_emergency_cleanup",
                        action="terminate_exec_session",
                        exec_bytes=exec_bytes,
                        available_bytes=available_bytes,
                        extra={
                            "session_id": victim,
                            "session_bytes": by_session[victim],
                        },
                    ),
                )

        if available_bytes > 0 and available_bytes <= self.fatal_available_bytes:
            self.fatal_streak += 1
        else:
            self.fatal_streak = 0
        if self.fatal_streak >= 2 and not self.pending_kills and not self.fatal_emitted:
            payload = self._payload(
                reason="host_memory_headroom_not_recovered",
                action="stop_and_resume",
                exec_bytes=exec_bytes,
                available_bytes=available_bytes,
                extra={"backend": backend, "fatal_streak": self.fatal_streak},
            )
            write_json_atomic(self.fatal_marker_dir / f"rank{self.rank}-exec-memory.json", payload)
            emit_event(FATAL_PREFIX, payload)
            self.fatal_emitted = True

        state = self._payload(
            reason="healthy" if not emergency else "memory_emergency",
            action="watch",
            exec_bytes=exec_bytes,
            available_bytes=available_bytes,
            extra={
                "backend": backend,
                "blocked": self.blocked,
                "processes": len(processes),
                "sessions": len(by_session),
            },
        )
        write_json_atomic(self.heartbeat_file, state)
        write_json_atomic(self.state_file, state)
        self.previous_memory = by_session

    def _set_admission_gate(
        self,
        should_block: bool,
        exec_bytes: int,
        available_bytes: int,
        now: float,
    ) -> None:
        if should_block:
            payload = self._payload(
                reason="aggregate_memory_soft_limit",
                action="wait_before_new_exec",
                exec_bytes=exec_bytes,
                available_bytes=available_bytes,
                extra={"updated_epoch": now},
            )
            write_json_atomic(self.gate_file, payload)
            if not self.blocked:
                emit_event(DEGRADED_PREFIX, payload)
            self.blocked = True
        else:
            if self.blocked:
                emit_event(
                    "PRORL_RECOVERED",
                    self._payload(
                        reason="aggregate_memory_headroom_restored",
                        action="resume_exec_admission",
                        exec_bytes=exec_bytes,
                        available_bytes=available_bytes,
                    ),
                )
            self.gate_file.unlink(missing_ok=True)
            self.blocked = False

    def _select_victim(self, by_session: dict[str, int]) -> str | None:
        return select_victim_session(
            by_session,
            self.previous_memory,
            set(self.pending_kills),
        )

    def _finish_pending_kills(self, processes: list[GuardedProcess]) -> None:
        now = time.monotonic()
        live_sessions = {process.session_id for process in processes}
        for session_id, deadline in tuple(self.pending_kills.items()):
            if session_id not in live_sessions:
                self.pending_kills.pop(session_id, None)
                continue
            if now < deadline:
                continue
            targets = [process for process in processes if process.session_id == session_id]
            signal_processes(targets, signal.SIGKILL)
            self.pending_kills.pop(session_id, None)

    def _payload(
        self,
        *,
        reason: str,
        action: str,
        exec_bytes: int,
        available_bytes: int,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "action": action,
            "available_bytes": available_bytes,
            "component": "wsclaw_exec_memory_guard",
            "exec_bytes": exec_bytes,
            "host": socket.gethostname(),
            "rank": self.rank,
            "reason": reason,
            "run_id": os.environ.get("RUN_ID", "unknown"),
            "scope": self.scope,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            payload.update(extra)
        return payload


def _positive_gib(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed * GIB


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--rank", default=os.environ.get("NODE_RANK", "unknown"))
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--fatal-marker-dir", type=Path)
    parser.add_argument("--soft-gib", type=_positive_gib, default=280 * GIB)
    parser.add_argument("--hard-gib", type=_positive_gib, default=340 * GIB)
    parser.add_argument("--block-available-gib", type=_positive_gib, default=100 * GIB)
    parser.add_argument("--emergency-available-gib", type=_positive_gib, default=64 * GIB)
    parser.add_argument("--fatal-available-gib", type=_positive_gib, default=40 * GIB)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--terminate-grace-seconds", type=float, default=5.0)
    parser.add_argument("--cleanup-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cleanup_only:
        survivors = cleanup_scope(args.scope, grace_seconds=args.terminate_grace_seconds)
        emit_event(
            "WSCLAW_EXEC_SCOPE_CLEANUP",
            {"scope": args.scope, "survivors": survivors},
        )
        return 0 if survivors == 0 else 1
    if args.state_dir is None or args.fatal_marker_dir is None:
        raise SystemExit("--state-dir and --fatal-marker-dir are required")
    if not (args.soft_gib < args.hard_gib):
        raise SystemExit("--soft-gib must be smaller than --hard-gib")
    if not (args.fatal_available_gib < args.emergency_available_gib < args.block_available_gib):
        raise SystemExit(
            "host thresholds must satisfy fatal < emergency < block available GiB"
        )
    if args.poll_seconds <= 0 or args.terminate_grace_seconds < 0:
        raise SystemExit("poll seconds must be positive and grace seconds non-negative")
    guard = NodeExecMemoryGuard(
        scope=args.scope,
        rank=str(args.rank),
        state_dir=args.state_dir,
        fatal_marker_dir=args.fatal_marker_dir,
        soft_bytes=args.soft_gib,
        hard_bytes=args.hard_gib,
        block_available_bytes=args.block_available_gib,
        emergency_available_bytes=args.emergency_available_gib,
        fatal_available_bytes=args.fatal_available_gib,
        poll_seconds=args.poll_seconds,
        terminate_grace_seconds=args.terminate_grace_seconds,
    )
    signal.signal(signal.SIGTERM, guard.request_stop)
    signal.signal(signal.SIGINT, guard.request_stop)
    try:
        return guard.run()
    except Exception as exc:
        payload: dict[str, object] = {
            "action": "stop_and_resume",
            "component": "wsclaw_exec_memory_guard",
            "exception": f"{type(exc).__name__}: {exc}"[:2048],
            "host": socket.gethostname(),
            "manual_resume_required": True,
            "rank": str(args.rank),
            "reason": "exec_memory_guard_crashed",
            "run_id": os.environ.get("RUN_ID", "unknown"),
            "scope": args.scope,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_json_atomic(args.fatal_marker_dir / f"rank{args.rank}-exec-memory.json", payload)
        emit_event(FATAL_PREFIX, payload, stream=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
