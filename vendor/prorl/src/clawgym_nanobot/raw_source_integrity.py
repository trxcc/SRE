"""Protect immutable WSClaw source workspaces during agent execution.

The agent workspace exposes canonical inputs through symlinks.  Landlock is
the preventive boundary; this module supplies an independent content-integrity
check before and after every task and a persistent manifest that can be built
with the original workspace archive as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from clawgym_nanobot.exec_memory_guard import FATAL_PREFIX, write_json_atomic


logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "clawgym_raw_source_manifest_v1"
_SHA_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
_SHA_CACHE_LOCK = threading.Lock()
_MANIFEST_CACHE: tuple[str, int, int, dict[str, dict[str, Any]]] | None = None
_MANIFEST_CACHE_LOCK = threading.Lock()


class RawSourceIntegrityError(RuntimeError):
    """Raised when an immutable task source is missing or has changed."""


@dataclass(frozen=True)
class RequiredSource:
    task_id: str
    task_dir: Path
    workspace_root: Path
    relative_path: str
    path: Path


@dataclass(frozen=True)
class SourceFingerprint:
    path: str
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "sha256": self.sha256,
        }


def protection_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = source.get("CLAWGYM_RAW_SOURCE_INTEGRITY", "0").strip().lower()
    return value in {"1", "true", "yes", "on", "enabled"}


def required_sources(task_dir: str | Path) -> list[RequiredSource]:
    """Resolve one task's declared source files without permitting traversal."""

    directory = Path(task_dir)
    entry_path = directory / "data_entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    if not isinstance(entry, dict):
        raise TypeError(f"task entry must be an object: {entry_path}")
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("workspace_root"):
        return []
    raw_paths = metadata.get("required_source_files")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"task {entry.get('task_id')} has no required_source_files metadata")

    root = Path(str(metadata["workspace_root"])).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    task_id = str(entry.get("task_id") or directory.name)
    result: list[RequiredSource] = []
    for raw_path in raw_paths:
        relative = _safe_relative_path(str(raw_path))
        logical_path = root.joinpath(*PurePosixPath(relative).parts)
        resolved = logical_path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"required source escapes workspace root: task={task_id} path={relative}"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"required source file does not exist: {resolved}")
        result.append(
            RequiredSource(
                task_id=task_id,
                task_dir=directory,
                workspace_root=root,
                relative_path=relative,
                path=resolved,
            )
        )
    return result


def fingerprint(path: str | Path, *, use_cache: bool = True) -> SourceFingerprint:
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file():
        raise FileNotFoundError(f"required source file does not exist: {resolved}")
    key = (
        str(resolved),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    digest: str | None = None
    if use_cache:
        with _SHA_CACHE_LOCK:
            digest = _SHA_CACHE.get(key)
    if digest is None:
        hasher = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if use_cache:
            with _SHA_CACHE_LOCK:
                if len(_SHA_CACHE) >= 8192:
                    _SHA_CACHE.clear()
                _SHA_CACHE[key] = digest
    return SourceFingerprint(
        path=str(resolved),
        size=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        ctime_ns=int(stat.st_ctime_ns),
        sha256=digest,
    )


class TaskSourceGuard:
    """Validate declared raw inputs against a manifest around one task run."""

    def __init__(
        self,
        task_dir: str | Path,
        *,
        rollout_id: int | None = None,
        group_index: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.rollout_id = rollout_id
        self.group_index = group_index
        self.environ = dict(os.environ if environ is None else environ)
        self.enabled = protection_enabled(self.environ)
        self.sources: list[RequiredSource] = []
        self.before: dict[str, SourceFingerprint] = {}

    def check(self, phase: str) -> None:
        if not self.enabled:
            return
        try:
            if not self.sources:
                self.sources = required_sources(self.task_dir)
            if not self.sources:
                return
            manifest_path = self.environ.get("CLAWGYM_RAW_SOURCE_MANIFEST", "").strip()
            if not manifest_path:
                raise RawSourceIntegrityError(
                    "CLAWGYM_RAW_SOURCE_MANIFEST is required when raw-source integrity is enabled"
                )
            expected = _load_manifest_expected(Path(manifest_path))
            current: dict[str, SourceFingerprint] = {}
            violations: list[dict[str, Any]] = []
            for source in self.sources:
                try:
                    observed = fingerprint(source.path)
                except (FileNotFoundError, OSError) as exc:
                    violations.append(
                        {
                            "path": str(source.path),
                            "reason": "missing_or_unreadable",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                current[observed.path] = observed
                baseline = expected.get(observed.path)
                if baseline is None:
                    violations.append(
                        {"path": observed.path, "reason": "not_in_manifest"}
                    )
                    continue
                if (
                    int(baseline.get("size", -1)) != observed.size
                    or str(baseline.get("sha256", "")) != observed.sha256
                ):
                    violations.append(
                        {
                            "path": observed.path,
                            "reason": "manifest_mismatch",
                            "expected_size": baseline.get("size"),
                            "actual_size": observed.size,
                            "expected_sha256": baseline.get("sha256"),
                            "actual_sha256": observed.sha256,
                        }
                    )
                previous = self.before.get(observed.path)
                if phase != "before" and previous is not None and (
                    previous.size != observed.size
                    or previous.sha256 != observed.sha256
                ):
                    violations.append(
                        {
                            "path": observed.path,
                            "reason": "changed_during_task",
                            "before_size": previous.size,
                            "after_size": observed.size,
                            "before_sha256": previous.sha256,
                            "after_sha256": observed.sha256,
                        }
                    )
            if violations:
                raise RawSourceIntegrityError(
                    f"raw source integrity violation during {phase}: "
                    f"{json.dumps(violations[:8], sort_keys=True)}"
                )
            if phase == "before":
                self.before = current
        except RawSourceIntegrityError as exc:
            self._emit_fatal(phase, exc)
            raise
        except Exception as exc:
            wrapped = RawSourceIntegrityError(
                f"raw source integrity check failed during {phase}: "
                f"{type(exc).__name__}: {exc}"
            )
            self._emit_fatal(phase, wrapped)
            raise wrapped from exc

    def _emit_fatal(self, phase: str, exc: BaseException) -> None:
        payload: dict[str, Any] = {
            "action": "stop_and_resume",
            "component": "wsclaw_raw_source_integrity",
            "error": str(exc)[:4096],
            "group_index": self.group_index,
            "host": socket.gethostname(),
            "manual_resume_required": True,
            "phase": phase,
            "rank": self.environ.get("NODE_RANK", "unknown"),
            "reason": "raw_source_integrity_violation",
            "rollout_id": self.rollout_id,
            "run_id": self.environ.get("RUN_ID", "unknown"),
            "task_dir": str(self.task_dir),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        marker_dir = self.environ.get("PRORL_FATAL_RESUME_MARKER_DIR", "").strip()
        if marker_dir:
            try:
                write_json_atomic(
                    Path(marker_dir) / f"raw-source-integrity-{socket.gethostname()}.json",
                    payload,
                )
            except Exception:
                logger.exception("Failed to write raw-source integrity fatal marker")
        logger.critical("%s %s", FATAL_PREFIX, json.dumps(payload, sort_keys=True))


def build_manifest(
    task_roots: Sequence[Path],
    *,
    archive: Path,
    archive_root_map: Mapping[str, str],
    baseline: str,
) -> dict[str, Any]:
    """Build an explicit live or archive baseline with archive provenance."""

    if baseline not in {"live", "archive"}:
        raise ValueError("raw-source manifest baseline must be 'live' or 'archive'")

    task_dirs = _discover_task_dirs(task_roots)
    references: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        for source in required_sources(task_dir):
            record = references.setdefault(
                str(source.path),
                {
                    "source": source,
                    "task_reference_count": 0,
                },
            )
            record["task_reference_count"] += 1

    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        for canonical_path in sorted(references):
            record = references[canonical_path]
            source: RequiredSource = record["source"]
            archive_root = archive_root_map.get(source.workspace_root.name)
            if not archive_root:
                raise ValueError(
                    f"no archive root mapping for workspace {source.workspace_root.name!r}"
                )
            archive_entry = f"{archive_root.rstrip('/')}/{source.relative_path}"
            try:
                info = bundle.getinfo(archive_entry)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"required source is absent from backup archive: {archive_entry}"
                ) from exc
            archive_hash = _hash_stream(bundle.open(info))
            live = fingerprint(source.path, use_cache=False)
            expected_size = live.size if baseline == "live" else int(info.file_size)
            expected_sha256 = live.sha256 if baseline == "live" else archive_hash
            entries.append(
                {
                    "path": live.path,
                    "size": expected_size,
                    "sha256": expected_sha256,
                    "archive_entry": archive_entry,
                    "archive_sha256": archive_hash,
                    "archive_size": int(info.file_size),
                    "live_size_at_build": live.size,
                    "live_mtime_ns_at_build": live.mtime_ns,
                    "live_ctime_ns_at_build": live.ctime_ns,
                    "live_sha256_at_build": live.sha256,
                    "matches_archive": (
                        live.size == int(info.file_size) and live.sha256 == archive_hash
                    ),
                    "relative_path": source.relative_path,
                    "task_reference_count": int(record["task_reference_count"]),
                    "workspace_root": str(source.workspace_root),
                }
            )

    archive_stat = archive.stat()
    return {
        "schema": MANIFEST_SCHEMA,
        "baseline": baseline,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": {
            "path": str(archive.resolve(strict=True)),
            "size": int(archive_stat.st_size),
            "mtime_ns": int(archive_stat.st_mtime_ns),
        },
        "task_roots": [str(path.resolve(strict=True)) for path in task_roots],
        "task_count": len(task_dirs),
        "entry_count": len(entries),
        "archive_difference_count": sum(
            1 for entry in entries if not entry["matches_archive"]
        ),
        "entries": entries,
    }


def verify_manifest(manifest_path: Path) -> tuple[int, list[dict[str, Any]]]:
    payload = _load_manifest(manifest_path)
    violations: list[dict[str, Any]] = []
    for entry in payload["entries"]:
        path = Path(str(entry["path"]))
        try:
            current = fingerprint(path, use_cache=False)
        except (FileNotFoundError, OSError) as exc:
            violations.append(
                {
                    "path": str(path),
                    "reason": "missing_or_unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if current.size != int(entry["size"]) or current.sha256 != str(entry["sha256"]):
            violations.append(
                {
                    "path": current.path,
                    "reason": "manifest_mismatch",
                    "expected_size": entry["size"],
                    "actual_size": current.size,
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": current.sha256,
                }
            )
    return len(payload["entries"]), violations


def _load_manifest_expected(path: Path) -> dict[str, dict[str, Any]]:
    global _MANIFEST_CACHE
    stat = path.stat()
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    with _MANIFEST_CACHE_LOCK:
        if _MANIFEST_CACHE is not None and _MANIFEST_CACHE[:3] == cache_key:
            return _MANIFEST_CACHE[3]
    payload = _load_manifest(path)
    expected = {str(entry["path"]): entry for entry in payload["entries"]}
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE = (*cache_key, expected)
    return expected


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported raw-source manifest: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"raw-source manifest has no entries: {path}")
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"raw-source manifest entry must be an object: {path}")
        if not all(key in entry for key in ("path", "size", "sha256")):
            raise ValueError(f"raw-source manifest entry is incomplete: {path}")
    return payload


def _discover_task_dirs(roots: Sequence[Path]) -> list[Path]:
    entries: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=True)
        if resolved.is_file() and resolved.name == "data_entry.json":
            entries.add(resolved)
            continue
        direct = resolved / "data_entry.json"
        if direct.is_file():
            entries.add(direct)
            continue
        entries.update(resolved.rglob("data_entry.json"))
        for child in resolved.iterdir():
            candidate = child / "data_entry.json"
            if candidate.is_file():
                entries.add(candidate)
    if not entries:
        raise FileNotFoundError("no task data_entry.json files found")
    return [path.parent for path in sorted(entries)]


def _safe_relative_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe required source path: {raw!r}")
    return path.as_posix()


def _hash_stream(handle: Any) -> str:
    hasher = hashlib.sha256()
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _parse_root_maps(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, archive_root = value.partition("=")
        if not separator or not key.strip() or not archive_root.strip():
            raise ValueError(f"invalid --archive-root-map value: {value!r}")
        result[key.strip()] = archive_root.strip().strip("/")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--tasks-root", type=Path, action="append", required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--archive-root-map", action="append", default=[], required=True)
    build.add_argument("--baseline", choices=("live", "archive"), required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "build":
        payload = build_manifest(
            args.tasks_root,
            archive=args.archive.resolve(strict=True),
            archive_root_map=_parse_root_maps(args.archive_root_map),
            baseline=args.baseline,
        )
        write_json_atomic(args.output, payload)
        print(
            "RAW_SOURCE_MANIFEST_BUILT "
            f"tasks={payload['task_count']} entries={payload['entry_count']} "
            f"archive_differences={payload['archive_difference_count']} "
            f"path={args.output}",
            flush=True,
        )
        return 0

    count, violations = verify_manifest(args.manifest)
    if violations:
        print(
            f"{FATAL_PREFIX} "
            + json.dumps(
                {
                    "action": "stop_and_resume",
                    "component": "wsclaw_raw_source_integrity",
                    "manual_resume_required": True,
                    "reason": "raw_source_manifest_verification_failed",
                    "violation_count": len(violations),
                    "violations": violations[:8],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(f"RAW_SOURCE_MANIFEST_VERIFIED entries={count} path={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
