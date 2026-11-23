"""Safe local deletion helper for checkpoint retention."""

from __future__ import annotations

import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Any


_MOUNT_ROOT = Path(os.environ.get("CLAWGYM_RECYCLABLE_MOUNT_ROOT", "/nonexistent-recyclable-mount"))
_PERSIST_ROOT = Path(os.environ.get("CLAWGYM_RECYCLABLE_PERSIST_ROOT", str(_MOUNT_ROOT)))
_RECYCLE_ROOT = _MOUNT_ROOT / ".recyclebininternal" / "root"


def _is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_path(path: Path) -> None:
    """Remove one path, tolerating races with the recyclable filesystem.

    The AFS recycle namespace can briefly repopulate a directory while
    ``shutil.rmtree`` is walking it, which surfaces as ENOTEMPTY.  A later
    pass can remove the remaining entry, so callers retry around this helper.
    """

    try:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        # Another cleanup pass won the race.
        return


def _remove_until_absent(path: Path, deadline: float) -> None:
    last_error: OSError | None = None
    while _is_present(path):
        try:
            _remove_path(path)
            last_error = None
        except OSError as exc:
            last_error = exc
        if not _is_present(path):
            return
        if time.monotonic() >= deadline:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"timed out deleting {path}")
        time.sleep(0.1)


def _free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bavail * stats.f_frsize)


def _validated_target(raw_path: str | os.PathLike[str]) -> tuple[Path, Path]:
    lexical = Path(os.path.abspath(os.fspath(raw_path)))
    if lexical == _PERSIST_ROOT or not lexical.is_relative_to(_PERSIST_ROOT):
        raise ValueError(
            f"refusing deletion outside a child of {_PERSIST_ROOT}: {lexical}"
        )
    resolved_parent = lexical.parent.resolve(strict=False)
    if (
        resolved_parent != _PERSIST_ROOT
        and not resolved_parent.is_relative_to(_PERSIST_ROOT)
    ):
        raise ValueError(f"deletion path escapes {_PERSIST_ROOT}: {lexical}")
    return lexical, lexical.relative_to(_MOUNT_ROOT)


def _matching_recycle_paths(relative: Path) -> list[Path]:
    if not _RECYCLE_ROOT.is_dir():
        return []
    matches: list[Path] = []
    for namespace in _RECYCLE_ROOT.iterdir():
        if not namespace.is_dir() or namespace.is_symlink():
            continue
        candidate = namespace / relative
        if not _is_present(candidate):
            continue
        resolved = candidate.resolve(strict=False)
        recycle = _RECYCLE_ROOT.resolve(strict=False)
        if resolved != recycle and not resolved.is_relative_to(recycle):
            raise RuntimeError(f"recycle candidate escapes {_RECYCLE_ROOT}: {candidate}")
        matches.append(candidate)
    return matches


def remove_tree_and_purge(raw_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Delete one personal path and permanently purge only its recycle mirrors."""

    target, relative = _validated_target(raw_path)
    timeout = float(os.environ.get("TRX_RECYCLE_PURGE_TIMEOUT_SECONDS", "30"))
    if timeout <= 0:
        raise ValueError("TRX_RECYCLE_PURGE_TIMEOUT_SECONDS must be positive")
    deadline = time.monotonic() + timeout
    before = _free_bytes(_MOUNT_ROOT)

    # Removing the live target is part of the caller's correctness contract:
    # for example, a new weight snapshot must not reuse stale shard files.
    # Keep this strict, but retry transient ENOTEMPTY races until the deadline.
    _remove_until_absent(target, deadline)

    # Recycle purging is capacity housekeeping.  It must be attempted and
    # reported, but a laggy recycle namespace must never abort a completed
    # optimizer step or weight switch.
    removed: list[str] = []
    purge_errors: list[str] = []
    quiet_passes = 0
    while True:
        try:
            matches = _matching_recycle_paths(relative)
        except OSError as exc:
            matches = []
            purge_errors.append(f"scan: {type(exc).__name__}: {exc}")
        if matches:
            quiet_passes = 0
            for candidate in matches:
                try:
                    _remove_until_absent(candidate, deadline)
                    removed.append(str(candidate))
                except (OSError, RuntimeError) as exc:
                    purge_errors.append(
                        f"{candidate}: {type(exc).__name__}: {exc}"
                    )
        else:
            quiet_passes += 1
            if quiet_passes >= 2:
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)

    if _is_present(target):
        raise RuntimeError(f"target still exists after deletion: {target}")
    try:
        leftovers = _matching_recycle_paths(relative)
    except OSError as exc:
        leftovers = []
        purge_errors.append(f"final scan: {type(exc).__name__}: {exc}")
    if leftovers:
        purge_errors.append(
            f"recycle entries still exist: {[str(path) for path in leftovers]}"
        )
    if purge_errors:
        warnings.warn(
            f"best-effort recycle purge incomplete for {target}: "
            + "; ".join(purge_errors),
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "target": str(target),
        "recycle_entries_removed": removed,
        "recycle_entries_left": [str(path) for path in leftovers],
        "purge_errors": purge_errors,
        "free_bytes_before": before,
        "free_bytes_after": _free_bytes(_MOUNT_ROOT),
    }
