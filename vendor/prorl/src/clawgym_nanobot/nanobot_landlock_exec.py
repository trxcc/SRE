"""Execute a child process with writes confined to one workspace.

The WSClaw workspaces intentionally contain symlinks into the canonical source
tree.  A plain path check in an agent tool therefore is not sufficient: shell
commands can follow those symlinks and modify the shared dataset.  Landlock is
inherited across ``execve`` and enforces the boundary in the kernel, including
for grandchildren of the wrapped process.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

_ACCESS_FS_EXECUTE = 1 << 0
_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_READ_FILE = 1 << 2
_ACCESS_FS_READ_DIR = 1 << 3
_ACCESS_FS_REMOVE_DIR = 1 << 4
_ACCESS_FS_REMOVE_FILE = 1 << 5
_ACCESS_FS_MAKE_CHAR = 1 << 6
_ACCESS_FS_MAKE_DIR = 1 << 7
_ACCESS_FS_MAKE_REG = 1 << 8
_ACCESS_FS_MAKE_SOCK = 1 << 9
_ACCESS_FS_MAKE_FIFO = 1 << 10
_ACCESS_FS_MAKE_BLOCK = 1 << 11
_ACCESS_FS_MAKE_SYM = 1 << 12
_ACCESS_FS_REFER = 1 << 13
_ACCESS_FS_TRUNCATE = 1 << 14

_READ_ACCESS = _ACCESS_FS_EXECUTE | _ACCESS_FS_READ_FILE | _ACCESS_FS_READ_DIR
_ABI_V1_ACCESS = (1 << 13) - 1
SANDBOX_SETUP_FAILURE_EXIT = 126
SANDBOX_FATAL_PREFIX = "PRORL_FATAL_RESUME_REQUIRED"


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def _raise_errno(operation: str) -> None:
    errno = ctypes.get_errno()
    raise OSError(errno, f"{operation}: {os.strerror(errno)}")


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = int(libc.syscall(number, *args))
    if result < 0:
        _raise_errno(f"syscall {number}")
    return result


def _handled_access_for_abi(abi: int) -> int:
    access = _ABI_V1_ACCESS
    if abi >= 2:
        access |= _ACCESS_FS_REFER
    if abi >= 3:
        access |= _ACCESS_FS_TRUNCATE
    return access


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = _PathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=path_fd,
            reserved=0,
        )
        _syscall(
            libc,
            _SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
    finally:
        os.close(path_fd)


def restrict_writes_to(write_root: Path | str, *, extra_write_roots=()) -> None:
    """Allow filesystem reads globally and writes only beneath ``write_root``.

    This function deliberately fails closed when Landlock is unavailable.
    Existing inherited file descriptors (stdout/stderr logs, sockets) remain
    usable, which is required for the rollout and judge protocols.
    """

    root = Path(write_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    libc = ctypes.CDLL(None, use_errno=True)
    abi = _syscall(
        libc,
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    handled_access = _handled_access_for_abi(abi)
    ruleset_attr = _RulesetAttr(handled_access_fs=handled_access)
    ruleset_fd = _syscall(
        libc,
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    try:
        _add_path_rule(libc, ruleset_fd, Path("/"), _READ_ACCESS)
        _add_path_rule(libc, ruleset_fd, root, handled_access)
        for extra_root in extra_write_roots:
            extra_root = Path(extra_root).resolve(strict=True)
            if not extra_root.is_dir():
                raise NotADirectoryError(extra_root)
            _add_path_rule(libc, ruleset_fd, extra_root, handled_access)

        device_access = _ACCESS_FS_READ_FILE | _ACCESS_FS_WRITE_FILE
        if abi >= 3:
            device_access |= _ACCESS_FS_TRUNCATE
        for device in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"):
            path = Path(device)
            if path.exists():
                _add_path_rule(libc, ruleset_fd, path, device_access)

        if int(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) != 0:
            _raise_errno("prctl(PR_SET_NO_NEW_PRIVS)")
        _syscall(libc, _SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def configured_unprivileged_identity(
    environ: Mapping[str, str] | None = None,
) -> tuple[int, int] | None:
    """Return the requested non-root exec identity, when enabled.

    Cluster launchers opt in with ``CLAWGYM_EXEC_DROP_PRIVILEGES=1``.  A
    process that is already non-root keeps its current identity; a root
    gateway defaults to the conventional nobody/nogroup numeric identity.
    Numeric IDs avoid depending on image-specific passwd/group entries.
    """

    source = os.environ if environ is None else environ
    enabled = source.get("CLAWGYM_EXEC_DROP_PRIVILEGES", "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on", "enabled"}:
        return None
    if os.geteuid() != 0:
        return os.geteuid(), os.getegid()
    try:
        uid = int(source.get("CLAWGYM_EXEC_UID", "65534"))
        gid = int(source.get("CLAWGYM_EXEC_GID", "65534"))
    except ValueError as exc:
        raise ValueError("CLAWGYM_EXEC_UID/GID must be numeric") from exc
    if uid <= 0 or gid <= 0:
        raise ValueError("CLAWGYM exec UID/GID must both be non-root positive IDs")
    return uid, gid


def _prepare_write_root_identity(root: Path, uid: int, gid: int) -> None:
    """Hand a workspace tree to the exec identity without following links."""

    os.chown(root, uid, gid, follow_symlinks=False)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        os.chown(directory_path, uid, gid, follow_symlinks=False)
        for name in (*dirnames, *filenames):
            os.chown(directory_path / name, uid, gid, follow_symlinks=False)


def _drop_privileges(uid: int, gid: int) -> None:
    if uid <= 0 or gid <= 0:
        raise ValueError("refusing to execute an agent with a root UID or GID")
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    if os.geteuid() != uid or os.getegid() != gid:
        raise PermissionError(
            f"failed to enter requested exec identity uid={uid} gid={gid}"
        )


def sandbox_command(
    command: Sequence[str],
    *,
    write_root: Path | str,
    identity: tuple[int, int] | None = None,
    extra_write_roots=(),
) -> list[str]:
    """Return an argv that applies Landlock and then execs ``command``."""

    if not command:
        raise ValueError("sandbox command must not be empty")
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--write-root",
        str(Path(write_root).resolve()),
    ]
    for extra_root in extra_write_roots:
        argv.extend(("--extra-write-root", str(Path(extra_root).resolve())))
    if identity is not None:
        uid, gid = identity
        argv.extend(("--uid", str(uid), "--gid", str(gid)))
    argv.extend(("--", *map(str, command)))
    return argv


def _emit_setup_failure(exc: BaseException) -> None:
    payload = {
        "action": "stop_and_resume",
        "component": "wsclaw_exec_sandbox",
        "error": f"{type(exc).__name__}: {exc}",
        "manual_resume_required": True,
        "reason": "exec_sandbox_initialization_failed",
    }
    print(
        f"{SANDBOX_FATAL_PREFIX} {json.dumps(payload, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-root", type=Path, required=True)
    parser.add_argument("--extra-write-root", type=Path, action="append", default=[])
    parser.add_argument("--uid", type=int)
    parser.add_argument("--gid", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    if (args.uid is None) != (args.gid is None):
        parser.error("--uid and --gid must be supplied together")

    try:
        root = args.write_root.resolve(strict=True)
        if args.uid is not None and args.gid is not None:
            _prepare_write_root_identity(root, args.uid, args.gid)
            for extra_root in args.extra_write_root:
                _prepare_write_root_identity(extra_root.resolve(strict=True), args.uid, args.gid)
        restrict_writes_to(root, extra_write_roots=args.extra_write_root)
        if args.uid is not None and args.gid is not None:
            _drop_privileges(args.uid, args.gid)
            if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
                raise PermissionError(
                    f"dropped exec identity cannot access write root: {root}"
                )
    except BaseException as exc:
        _emit_setup_failure(exc)
        return SANDBOX_SETUP_FAILURE_EXIT
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
