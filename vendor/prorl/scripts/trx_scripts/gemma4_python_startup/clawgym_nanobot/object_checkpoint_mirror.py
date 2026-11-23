"""Mirror completed HF checkpoints to a mounted PT object-store bucket.

The training saver first completes an atomic checkpoint on the personal AFS.
This module then copies that immutable directory through the pre-mounted
``/oss`` path, verifies its file inventory and sizes, and writes the completion
manifest last.  It deliberately does not use the object mount for locks or for
the multi-rank checkpoint writer.
"""

from __future__ import annotations

import json
import hashlib
import importlib
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Iterable


_ALLOWED_ROOTS = {
    Path(p).expanduser(): Path(p).expanduser().parent
    for p in os.environ.get("CLAWGYM_OBJECT_MIRROR_ALLOWED_ROOTS", "").split(":")
    if p.strip()
}
_COMPLETE_NAME = "_COMPLETE.json"
_BASE_POINTER_NAMES = ("best", "last")
_COPY_CHUNK_BYTES = max(
    4096,
    int(os.environ.get("CLAWGYM_OBJECT_COPY_CHUNK_BYTES", str(64 * 1024))),
)


def _copy_file_bounded(source: Path, target: Path) -> None:
    """Copy one checkpoint file with a bounded user-space buffer.

    The object-store mount can return ``ENOMEM`` while training actors still
    hold model memory. Keep the buffer small, resume an existing prefix, and
    leave the normal inventory/hash verification as the publication guard.
    """

    source = source.resolve(strict=True)
    source_size = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0
    if existing > source_size:
        target.unlink()
        existing = 0

    for attempt in range(6):
        try:
            with source.open("rb", buffering=0) as src:
                if existing:
                    src.seek(existing)
                with target.open("ab", buffering=0) as dst:
                    while True:
                        block = src.read(_COPY_CHUNK_BYTES)
                        if not block:
                            break
                        dst.write(block)
                        try:
                            os.posix_fadvise(
                                src.fileno(),
                                max(0, src.tell() - len(block)),
                                len(block),
                                os.POSIX_FADV_DONTNEED,
                            )
                        except (AttributeError, OSError):
                            pass
                    dst.flush()
                    try:
                        os.fsync(dst.fileno())
                    except OSError:
                        pass
            if target.stat().st_size != source_size:
                raise OSError(
                    f"bounded copy size mismatch for {source}: "
                    f"{target.stat().st_size} != {source_size}"
                )
            shutil.copystat(source, target, follow_symlinks=False)
            return
        except OSError as exc:
            if exc.errno != 12 or attempt >= 5:
                raise
            try:
                existing = target.stat().st_size
            except FileNotFoundError:
                existing = 0
            time.sleep(min(30.0, 2.0**attempt))


def enabled() -> bool:
    return bool(os.environ.get("CLAWGYM_OBJECT_CHECKPOINT_ROOT", "").strip())


def required() -> bool:
    return os.environ.get("CLAWGYM_OBJECT_CHECKPOINT_REQUIRED", "0") == "1"


def validate_configuration(*, create: bool = False) -> Path | None:
    raw = os.environ.get("CLAWGYM_OBJECT_CHECKPOINT_ROOT", "").strip()
    if not raw:
        if required():
            raise RuntimeError("CLAWGYM_OBJECT_CHECKPOINT_ROOT is required")
        return None
    root = Path(raw).resolve(strict=False)
    resolved_allowed_roots = {
        allowed.resolve(strict=False): mount
        for allowed, mount in _ALLOWED_ROOTS.items()
    }
    allowed = next(
        (
            candidate
            for candidate in resolved_allowed_roots
            if root != candidate and candidate in root.parents
        ),
        None,
    )
    if allowed is None:
        raise RuntimeError(
            "object checkpoint root must be below one of "
            f"{sorted(str(path) for path in resolved_allowed_roots)}, got {root}"
        )
    mount = resolved_allowed_roots[allowed]
    if not mount.is_dir():
        raise RuntimeError(f"object-store mount is missing: {mount}")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    parent = root if root.exists() else _nearest_existing_parent(root)
    if not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"object checkpoint root is not writable: {parent}")
    return root


def mirror_completed_checkpoint(
    *, checkpoint: Path, rollout_id: int
) -> dict[str, Any]:
    root = validate_configuration(create=True)
    if root is None:
        return {"enabled": False}

    checkpoint = checkpoint.resolve()
    marker = _read_json(checkpoint / "checkpoint.json")
    if (
        not checkpoint.is_dir()
        or marker.get("format") != "huggingface"
        or marker.get("optimizer_saved") is not False
        or marker.get("rollout_id") != int(rollout_id)
    ):
        raise RuntimeError(f"refusing to mirror incomplete HF checkpoint: {checkpoint}")

    source_files = _inventory(checkpoint, excluded={_COMPLETE_NAME})
    _validate_hf_weights(checkpoint, source_files)

    destination = root / "candidates" / f"step-{int(rollout_id)}"
    previous = _read_json(destination / _COMPLETE_NAME)
    if _manifest_matches(previous, rollout_id, source_files):
        return {
            "enabled": True,
            "uploaded": False,
            "verified": True,
            "rollout_id": int(rollout_id),
            "checkpoint": str(destination),
            "file_count": len(source_files),
            "total_bytes": _total_bytes(source_files),
        }

    # An incomplete destination belongs only to this run/step prefix.  Clear
    # it before retrying, then publish the completion marker last.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for relative in sorted(source_files):
            source = checkpoint / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_file_bounded(source.resolve(strict=True), target)

        verify_remote_sha256 = _verify_sha256_every_save()
        _wait_for_inventory(
            destination,
            source_files,
            verify_sha256=verify_remote_sha256,
        )
        manifest = {
            "format": "huggingface-object-mirror-v2",
            "rollout_id": int(rollout_id),
            "optimizer_saved": False,
            "source_checkpoint": str(checkpoint),
            "object_checkpoint": str(destination),
            "file_count": len(source_files),
            "total_bytes": _total_bytes(source_files),
            "files": source_files,
            "verification": {
                "source_sha256": True,
                "remote_inventory": (
                    "sha256" if verify_remote_sha256 else "size"
                ),
            },
            "completed_at": _utc_now(),
            "writer_host": socket.gethostname(),
        }
        _write_json(destination / _COMPLETE_NAME, manifest)
        _wait_for_manifest(destination / _COMPLETE_NAME, rollout_id, source_files)
    except BaseException:
        # Leave an incomplete directory without a completion marker.  A later
        # retry will replace this exact run/step prefix safely.
        try:
            (destination / _COMPLETE_NAME).unlink()
        except FileNotFoundError:
            pass
        raise

    return {
        "enabled": True,
        "uploaded": True,
        "verified": True,
        "rollout_id": int(rollout_id),
        "checkpoint": str(destination),
        "file_count": len(source_files),
        "total_bytes": _total_bytes(source_files),
        "remote_checksum_verified": verify_remote_sha256,
    }


def sync_retention(state: dict[str, Any]) -> dict[str, Any]:
    """Publish best/last pointers and prune unreferenced object candidates."""

    root = validate_configuration(create=True)
    if root is None:
        return {"enabled": False}

    protected: dict[str, int] = {}
    for label in _retention_labels(state):
        record = state.get(label)
        if not isinstance(record, dict):
            continue
        rollout_id = record.get("rollout_id")
        if isinstance(rollout_id, int) and not isinstance(rollout_id, bool):
            destination = root / "candidates" / f"step-{rollout_id}"
            # mirror_completed_checkpoint already performed the expensive
            # remote SHA-256 readback before publishing _COMPLETE.json.  A
            # retention-only update can trust that immutable completion
            # manifest; successful-run cleanup still re-reads every byte.
            _verify_completed_checkpoint(
                destination,
                rollout_id,
                verify_inventory=False,
            )
            pointer = {
                "label": label,
                "rollout_id": rollout_id,
                "checkpoint": str(destination),
                "score": record.get("score"),
                "score_key": record.get("score_key"),
                "coverage": record.get("coverage"),
                "updated_at": _utc_now(),
            }
            _write_json(root / f"{label}.json", pointer)
            protected[label] = rollout_id

    previous = _read_json(root / "retention.json").get("retained")
    if isinstance(previous, dict):
        for label in previous:
            if label in protected or not _is_offset_pointer(label):
                continue
            try:
                (root / f"{label}.json").unlink()
            except FileNotFoundError:
                pass
    # Also remove an offset pointer left by an interrupted publication before
    # retention.json became visible.
    for pointer_path in root.glob("last*.json"):
        if (
            _is_offset_pointer(pointer_path.stem)
            and pointer_path.stem not in protected
        ):
            pointer_path.unlink()

    candidates = root / "candidates"
    removed: list[str] = []
    if candidates.is_dir():
        keep = set(protected.values())
        for path in candidates.iterdir():
            if not path.is_dir() or not path.name.startswith("step-"):
                continue
            try:
                rollout_id = int(path.name.removeprefix("step-"))
            except ValueError:
                continue
            if rollout_id in keep:
                continue
            shutil.rmtree(path)
            removed.append(path.name)

    state_manifest = {
        "format": "huggingface-object-retention-v2",
        "updated_at": _utc_now(),
        "root": str(root),
        "retained": protected,
        "removed": sorted(removed),
    }
    _write_json(root / "retention.json", state_manifest)
    return {
        "enabled": True,
        "root": str(root),
        "retained": protected,
        "removed": sorted(removed),
    }


def finalize_successful_run(*, checkpoint_root: Path) -> dict[str, Any]:
    """Publish a final pointer, verify best/final, then remove local weights.

    The local retention state remains as a small recovery manifest. No local
    checkpoint is removed until both selected object checkpoints have been
    read back and matched by file size and SHA-256.
    """

    root = validate_configuration(create=True)
    if root is None:
        raise RuntimeError("object checkpoint mirroring is not configured")

    checkpoint_root = checkpoint_root.resolve()
    state_path = checkpoint_root / "retention.json"
    state = _read_json(state_path)
    selected: dict[str, dict[str, Any]] = {}
    for source_label, published_label in (("best", "best"), ("last", "final")):
        record = state.get(source_label)
        if not isinstance(record, dict):
            raise RuntimeError(f"local retention state has no {source_label} checkpoint")
        rollout_id = record.get("rollout_id")
        if not isinstance(rollout_id, int) or isinstance(rollout_id, bool):
            raise RuntimeError(f"invalid {source_label} rollout id: {rollout_id!r}")
        destination = root / "candidates" / f"step-{rollout_id}"
        manifest = _verify_completed_checkpoint(destination, rollout_id)

        local_checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
        if not local_checkpoint.is_dir():
            raise RuntimeError(f"local {source_label} checkpoint is missing: {local_checkpoint}")
        local_files = _inventory(local_checkpoint, excluded={_COMPLETE_NAME})
        if local_files != manifest["files"]:
            raise RuntimeError(
                f"local/object checksum mismatch for {source_label}: {local_checkpoint}"
            )

        pointer = {
            "label": published_label,
            "rollout_id": rollout_id,
            "checkpoint": str(destination),
            "score": record.get("score"),
            "score_key": record.get("score_key"),
            "coverage": record.get("coverage"),
            "updated_at": _utc_now(),
        }
        _write_json(root / f"{published_label}.json", pointer)
        selected[published_label] = pointer

    for label in _retention_labels(state):
        if not _is_offset_pointer(label):
            continue
        record = state[label]
        rollout_id = record.get("rollout_id")
        if not isinstance(rollout_id, int) or isinstance(rollout_id, bool):
            raise RuntimeError(f"invalid {label} rollout id: {rollout_id!r}")
        destination = root / "candidates" / f"step-{rollout_id}"
        manifest = _verify_completed_checkpoint(destination, rollout_id)
        local_checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
        if not local_checkpoint.is_dir():
            raise RuntimeError(f"local {label} checkpoint is missing: {local_checkpoint}")
        local_files = _inventory(local_checkpoint, excluded={_COMPLETE_NAME})
        if local_files != manifest["files"]:
            raise RuntimeError(
                f"local/object checksum mismatch for {label}: {local_checkpoint}"
            )
        pointer = {
            "label": label,
            "rollout_id": rollout_id,
            "checkpoint": str(destination),
            "score": record.get("score"),
            "score_key": record.get("score_key"),
            "coverage": record.get("coverage"),
            "updated_at": _utc_now(),
        }
        _write_json(root / f"{label}.json", pointer)
        selected[label] = pointer

    try:
        (root / "last.json").unlink()
    except FileNotFoundError:
        pass

    keep = {record["rollout_id"] for record in selected.values()}
    removed_remote: list[str] = []
    candidates = root / "candidates"
    if candidates.is_dir():
        for path in candidates.iterdir():
            if not path.is_dir() or not path.name.startswith("step-"):
                continue
            try:
                rollout_id = int(path.name.removeprefix("step-"))
            except ValueError:
                continue
            if rollout_id not in keep:
                shutil.rmtree(path)
                removed_remote.append(path.name)

    _write_json(
        root / "retention.json",
        {
            "format": "huggingface-object-retention-v2",
            "updated_at": _utc_now(),
            "root": str(root),
            "retained": {
                label: record["rollout_id"] for label, record in selected.items()
            },
            "removed": sorted(removed_remote),
        },
    )

    removed_local: list[str] = []
    local_candidates = checkpoint_root / "candidates"
    if local_candidates.is_dir():
        for path in local_candidates.iterdir():
            if path.is_dir():
                _remove_local_tree(path)
                removed_local.append(path.name)
    for link in checkpoint_root.iterdir() if checkpoint_root.is_dir() else ():
        if link.is_symlink() and (link.name in {"best", "last"} or link.name.startswith("last")):
            link.unlink()

    state["object_storage"] = {
        "root": str(root),
        "best": selected["best"],
        "final": selected["final"],
        "offsets": {
            label: record
            for label, record in selected.items()
            if _is_offset_pointer(label)
        },
        "verified_at": _utc_now(),
    }
    state["local_weights_removed"] = True
    state["local_weights_removed_at"] = _utc_now()
    _write_json_atomic(state_path, state)
    return {
        "object_root": str(root),
        "best_rollout_id": selected["best"]["rollout_id"],
        "final_rollout_id": selected["final"]["rollout_id"],
        "offset_rollout_ids": {
            label: record["rollout_id"]
            for label, record in selected.items()
            if _is_offset_pointer(label)
        },
        "removed_remote": sorted(removed_remote),
        "removed_local": sorted(removed_local),
    }


def _remove_local_tree(path: Path) -> None:
    hook_path = os.environ.get("CLAWGYM_LOCAL_DELETE_FUNCTION_PATH", "").strip()
    if not hook_path:
        shutil.rmtree(path, ignore_errors=True)
        return
    module_name, separator, function_name = hook_path.rpartition(".")
    if not separator:
        raise ValueError(
            "CLAWGYM_LOCAL_DELETE_FUNCTION_PATH must be a dotted import path"
        )
    function = getattr(importlib.import_module(module_name), function_name)
    function(str(path))


def _is_offset_pointer(label: object) -> bool:
    return bool(
        isinstance(label, str)
        and label.startswith("last")
        and label.removeprefix("last").isdigit()
        and int(label.removeprefix("last")) > 0
    )


def _retention_labels(state: dict[str, Any]) -> tuple[str, ...]:
    offsets = sorted(
        (label for label in state if _is_offset_pointer(label)),
        key=lambda label: int(label.removeprefix("last")),
    )
    return (*_BASE_POINTER_NAMES, *offsets)


def _inventory(
    root: Path, *, excluded: Iterable[str] = ()
) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    result: dict[str, dict[str, Any]] = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        target = path.resolve(strict=True) if path.is_symlink() else path
        if not target.is_file():
            raise RuntimeError(f"unsupported checkpoint entry: {path}")
        result[relative] = {
            "bytes": target.stat().st_size,
            "sha256": _sha256(target),
        }
    return result


def _validate_hf_weights(
    checkpoint: Path, source_files: dict[str, dict[str, Any]]
) -> None:
    """Accept either a single-file or a complete indexed sharded HF model."""

    if "model.safetensors" in source_files:
        return

    index_name = "model.safetensors.index.json"
    if index_name not in source_files:
        raise RuntimeError(
            "HF checkpoint has neither model.safetensors nor "
            f"model.safetensors.index.json: {checkpoint}"
        )
    try:
        index = json.loads((checkpoint / index_name).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"invalid HF safetensors index: {checkpoint / index_name}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"HF safetensors index has no weight_map: {checkpoint / index_name}")

    shards = set()
    for shard in weight_map.values():
        if (
            not isinstance(shard, str)
            or not shard.endswith(".safetensors")
            or Path(shard).is_absolute()
            or ".." in Path(shard).parts
        ):
            raise RuntimeError(
                f"HF safetensors index has an invalid shard path: {shard!r}"
            )
        shards.add(Path(shard).as_posix())
    missing = sorted(shard for shard in shards if shard not in source_files)
    if missing:
        raise RuntimeError(
            f"HF safetensors index references missing shards in {checkpoint}: {missing}"
        )


def _verify_sha256_every_save() -> bool:
    raw = os.environ.get(
        "CLAWGYM_OBJECT_CHECKPOINT_VERIFY_SHA256_EVERY_SAVE", "0"
    ).strip()
    if raw not in {"0", "1"}:
        raise ValueError(
            "CLAWGYM_OBJECT_CHECKPOINT_VERIFY_SHA256_EVERY_SAVE must be 0 or 1"
        )
    return raw == "1"


def _inventory_sizes(
    root: Path, *, excluded: Iterable[str] = ()
) -> dict[str, dict[str, int]]:
    excluded_set = set(excluded)
    result: dict[str, dict[str, int]] = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded_set:
            continue
        target = path.resolve(strict=True) if path.is_symlink() else path
        if not target.is_file():
            raise RuntimeError(f"unsupported checkpoint entry: {path}")
        result[relative] = {"bytes": target.stat().st_size}
    return result


def _wait_for_inventory(
    root: Path,
    expected: dict[str, dict[str, Any]],
    *,
    verify_sha256: bool = True,
) -> None:
    deadline = time.monotonic() + float(
        os.environ.get("CLAWGYM_OBJECT_CHECKPOINT_VERIFY_TIMEOUT", "120")
    )
    expected_observation = (
        expected
        if verify_sha256
        else {
            relative: {"bytes": int(record["bytes"])}
            for relative, record in expected.items()
        }
    )
    last: dict[str, dict[str, Any]] | None = None
    while True:
        try:
            last = (
                _inventory(root, excluded={_COMPLETE_NAME})
                if verify_sha256
                else _inventory_sizes(root, excluded={_COMPLETE_NAME})
            )
        except OSError:
            last = None
        if last == expected_observation:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"object checkpoint inventory verification failed for {root}: "
                f"expected_files={len(expected)} observed_files="
                f"{len(last) if last is not None else 'unreadable'}"
            )
        time.sleep(1.0)


def _wait_for_manifest(
    path: Path, rollout_id: int, files: dict[str, dict[str, Any]]
) -> None:
    deadline = time.monotonic() + float(
        os.environ.get("CLAWGYM_OBJECT_CHECKPOINT_VERIFY_TIMEOUT", "120")
    )
    while True:
        value = _read_json(path)
        if _manifest_matches(value, rollout_id, files):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"object checkpoint completion marker not visible: {path}")
        time.sleep(1.0)


def _manifest_matches(
    manifest: dict[str, Any], rollout_id: int, files: dict[str, dict[str, Any]]
) -> bool:
    return bool(
        manifest.get("format") == "huggingface-object-mirror-v2"
        and manifest.get("rollout_id") == int(rollout_id)
        and manifest.get("optimizer_saved") is False
        and manifest.get("file_count") == len(files)
        and manifest.get("total_bytes") == _total_bytes(files)
        and manifest.get("files") == files
    )


def _verify_completed_checkpoint(
    destination: Path,
    rollout_id: int,
    *,
    verify_inventory: bool = True,
) -> dict[str, Any]:
    manifest = _read_json(destination / _COMPLETE_NAME)
    files = manifest.get("files")
    if not isinstance(files, dict) or not _manifest_matches(manifest, rollout_id, files):
        raise RuntimeError(f"invalid object checkpoint manifest: {destination}")
    if verify_inventory:
        _wait_for_inventory(destination, files)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _total_bytes(files: dict[str, dict[str, Any]]) -> int:
    return sum(int(record["bytes"]) for record in files.values())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each object key is written once after its referenced payload is visible.
    # Avoid rename-based publication because AOSS is not a POSIX filesystem.
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists():
        if current.parent == current:
            raise RuntimeError(f"no existing parent for {path}")
        current = current.parent
    return current


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
