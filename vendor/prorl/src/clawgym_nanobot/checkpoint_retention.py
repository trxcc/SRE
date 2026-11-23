"""Bounded best/last retention for evaluated HuggingFace checkpoints.

Weight export is owned by Slime/Megatron-Bridge.  This module only promotes a
completed per-rollout HF directory after a sufficiently complete ClawGym eval
and keeps the matching Nanobot logs under the same rollout id.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from .checkpoint_pinning import pin_completed_checkpoint

from .object_checkpoint_mirror import (
    mirror_completed_checkpoint,
    sync_retention as sync_object_retention,
)


def finalize_latest_hf_checkpoint(
    rollout_id: int,
    args: Any,
) -> dict[str, Any]:
    """Promote one completed HF export as ``last`` and bound candidates.

    This path runs after every configured model save, independently of eval.
    ``best`` remains owned by :func:`finalize_evaluated_hf_snapshot`, while the
    ``last`` checkpoint always identifies the newest completed optimizer step.
    """

    checkpoint_root_raw = os.environ.get(
        "CLAWGYM_HF_CHECKPOINT_ROOT", ""
    ).strip()
    if not checkpoint_root_raw or not getattr(args, "save_hf", None):
        return {"enabled": False}

    checkpoint_root = Path(checkpoint_root_raw).resolve()
    checkpoint_candidate = Path(
        args.save_hf.format(rollout_id=rollout_id)
    ).absolute()
    _require_child(checkpoint_candidate, checkpoint_root, "HF candidate")
    milestone_pin = pin_completed_checkpoint(rollout_id, args)
    _require_seed_target(checkpoint_candidate, checkpoint_root)
    marker = _read_json(checkpoint_candidate / "checkpoint.json")
    valid = bool(
        checkpoint_candidate.is_dir()
        and marker.get("format") == "huggingface"
        and marker.get("optimizer_saved") is False
        and _finite_int(marker.get("rollout_id")) == int(rollout_id)
    )
    outcome = {
        "enabled": True,
        "milestone_pin": milestone_pin,
        "valid": valid,
        "rollout_id": int(rollout_id),
        "checkpoint": str(checkpoint_candidate),
    }
    if not valid:
        return outcome

    # Object storage is a durable mirror, not the multi-rank checkpoint
    # writer.  Publish and verify the completed local directory before local
    # best/last retention is allowed to remove an older candidate.
    outcome["object_mirror"] = mirror_completed_checkpoint(
        checkpoint=checkpoint_candidate,
        rollout_id=int(rollout_id),
    )

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root / ".retention.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_path = checkpoint_root / "retention.json"
        state = _read_json(state_path)
        record = {
            "rollout_id": int(rollout_id),
            "score_key": None,
            "score": None,
            "n_collected": None,
            "n_total": None,
            "coverage": None,
            "checkpoint": str(checkpoint_candidate),
            "nanobot_logs": None,
            "optimizer_saved": False,
            "next_rollout_id": int(rollout_id) + 1,
        }
        state["last"] = record
        state["format"] = "huggingface"
        state["optimizer_saved"] = False
        _update_offset_checkpoint_records(
            checkpoint_root=checkpoint_root,
            state=state,
            rollout_id=int(rollout_id),
        )
        outcome["object_retention"] = sync_object_retention(state)
        _write_json_atomic(state_path, state)
        _replace_symlink(checkpoint_root / "last", checkpoint_candidate)
        for label in _offset_checkpoint_labels():
            selected = state.get(label)
            if isinstance(selected, dict) and selected.get("checkpoint"):
                _replace_symlink(
                    checkpoint_root / label,
                    Path(str(selected["checkpoint"])),
                )
        _prune_candidates(
            checkpoint_root / "candidates",
            {
                Path(str(saved["checkpoint"])).resolve()
                for label in ("best", "last", *_offset_checkpoint_labels())
                for saved in (state.get(label),)
                if isinstance(saved, dict) and saved.get("checkpoint")
            },
        )
        outcome["best"] = state.get("best")
        outcome["last"] = state.get("last")
        for label in _offset_checkpoint_labels():
            outcome[label] = state.get(label)
    return outcome


def finalize_evaluated_hf_snapshot(
    rollout_id: int,
    args: Any,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Promote one valid evaluated checkpoint and prune unreferenced steps."""

    checkpoint_root_raw = os.environ.get(
        "CLAWGYM_HF_CHECKPOINT_ROOT", ""
    ).strip()
    if not checkpoint_root_raw or not getattr(args, "save_hf", None):
        return {"enabled": False}

    checkpoint_root = Path(checkpoint_root_raw).resolve()
    snapshot_root = Path(
        os.environ.get(
            "CLAWGYM_EVAL_SNAPSHOT_ROOT",
            str(checkpoint_root.parent / "eval_snapshots"),
        )
    ).resolve()
    checkpoint_candidate = Path(
        args.save_hf.format(rollout_id=rollout_id)
    ).absolute()
    _require_child(checkpoint_candidate, checkpoint_root, "HF candidate")
    milestone_pin = pin_completed_checkpoint(rollout_id, args)
    _require_seed_target(checkpoint_candidate, checkpoint_root)

    dataset = os.environ.get("CLAWGYM_BEST_EVAL_DATASET", "clawgym_eval")
    score_key = os.environ.get(
        "CLAWGYM_BEST_EVAL_METRIC", f"eval/{dataset}"
    )
    score = _finite_float(metrics.get(score_key))
    collected = _finite_int(metrics.get(f"eval/{dataset}-n_collected"))
    total = _finite_int(metrics.get(f"eval/{dataset}-n_tasks"))
    if total is None:
        total = _finite_int(
            metrics.get(f"eval/{dataset}/clawgym/rollout/target_trajectories")
        )
    coverage = (
        float(collected) / float(total)
        if collected is not None and total is not None and total > 0
        else 0.0
    )
    min_coverage = float(
        os.environ.get("CLAWGYM_HF_CHECKPOINT_MIN_EVAL_COVERAGE", "0.95")
    )
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(
            "CLAWGYM_HF_CHECKPOINT_MIN_EVAL_COVERAGE must be in (0, 1]"
        )

    valid = bool(
        checkpoint_candidate.joinpath("checkpoint.json").is_file()
        and score is not None
        and collected is not None
        and collected > 0
        and coverage >= min_coverage
    )
    outcome = {
        "enabled": True,
        "milestone_pin": milestone_pin,
        "valid": valid,
        "rollout_id": int(rollout_id),
        "score_key": score_key,
        "score": score,
        "n_collected": collected,
        "n_total": total,
        "coverage": coverage,
        "min_coverage": min_coverage,
        "checkpoint": str(checkpoint_candidate),
    }
    if not valid:
        return outcome


    # This is normally idempotent because save_interval=1 uploads the same
    # checkpoint before evaluation.  Keeping it here makes best promotion safe
    # if an eval-only caller reaches the retention hook directly.
    outcome["object_mirror"] = mirror_completed_checkpoint(
        checkpoint=checkpoint_candidate,
        rollout_id=int(rollout_id),
    )

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_root / ".retention.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state_path = checkpoint_root / "retention.json"
        state = _read_json(state_path)

        log_candidate = _capture_eval_logs(
            rollout_id=rollout_id,
            snapshot_root=snapshot_root,
        )
        record = {
            "rollout_id": int(rollout_id),
            "score_key": score_key,
            "score": float(score),
            "n_collected": int(collected),
            "n_total": int(total) if total is not None else None,
            "coverage": coverage,
            "checkpoint": str(checkpoint_candidate),
            "nanobot_logs": str(log_candidate) if log_candidate else None,
            "optimizer_saved": False,
            "next_rollout_id": int(rollout_id) + 1,
        }
        state["last"] = record
        best = state.get("best")
        if not isinstance(best, dict) or float(score) >= float(
            best.get("score", float("-inf"))
        ):
            state["best"] = record
        state["format"] = "huggingface"
        state["optimizer_saved"] = False

        _update_offset_checkpoint_records(
            checkpoint_root=checkpoint_root,
            state=state,
            rollout_id=int(rollout_id),
        )
        outcome["object_retention"] = sync_object_retention(state)
        _write_json_atomic(state_path, state)
        for label in ("best", "last", *_offset_checkpoint_labels()):
            selected = state.get(label)
            if not isinstance(selected, dict):
                continue
            _replace_symlink(
                checkpoint_root / label,
                Path(str(selected["checkpoint"])),
            )
            log_path = selected.get("nanobot_logs") if label in {"best", "last"} else None
            if log_path:
                _replace_symlink(snapshot_root / label, Path(str(log_path)))

        _prune_candidates(
            checkpoint_root / "candidates",
            {
                Path(str(record["checkpoint"])).resolve()
                for label in ("best", "last", *_offset_checkpoint_labels())
                for record in (state.get(label),)
                if isinstance(record, dict) and record.get("checkpoint")
            },
        )
        _prune_candidates(
            snapshot_root / "candidates",
            {
                Path(str(record["nanobot_logs"])).resolve()
                for record in (state.get("best"), state.get("last"))
                if isinstance(record, dict) and record.get("nanobot_logs")
            },
        )
        outcome["best"] = state.get("best")
        outcome["last"] = state.get("last")
        for label in _offset_checkpoint_labels():
            outcome[label] = state.get(label)
    return outcome


def _capture_eval_logs(*, rollout_id: int, snapshot_root: Path) -> Path | None:
    source_raw = os.environ.get("NANOBOT_LOG_DIR", "").strip()
    if not source_raw:
        return None
    source = Path(source_raw).resolve()
    marker = source / "LATEST_EVAL_ROLLOUT_ID"
    if not source.is_dir() or not marker.is_file():
        return None
    try:
        marker_rollout_id = int(marker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if marker_rollout_id != int(rollout_id):
        return None

    destination = (
        snapshot_root / "candidates" / f"step-{int(rollout_id):06d}"
    )
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    _remove_local_tree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    _remove_local_tree(destination)
    os.replace(temporary, destination)
    return destination.resolve()


def _require_child(path: Path, root: Path, label: str) -> None:
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"{label} must be below {root}, got {path}")


def _require_seed_target(path: Path, root: Path) -> None:
    experiment_root = root.parent.parent
    if path.is_symlink() and not path.resolve().is_relative_to(experiment_root):
        raise ValueError(f"HF seed candidate must remain in {experiment_root}: {path}")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _finite_int(value: Any) -> int | None:
    result = _finite_float(value)
    if result is None or result < 0 or not result.is_integer():
        return None
    return int(result)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _replace_symlink(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.tmp.{os.getpid()}")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    temporary.symlink_to(os.path.relpath(target, start=link.parent))
    os.replace(temporary, link)


def _retention_offsets() -> tuple[int, ...]:
    raw = os.environ.get("CLAWGYM_HF_CHECKPOINT_RETENTION_OFFSETS", "0")
    try:
        offsets = {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError(
            "CLAWGYM_HF_CHECKPOINT_RETENTION_OFFSETS must be comma-separated integers"
        ) from exc
    if not offsets or any(offset < 0 for offset in offsets):
        raise ValueError(
            "CLAWGYM_HF_CHECKPOINT_RETENTION_OFFSETS must contain non-negative integers"
        )
    offsets.add(0)
    return tuple(sorted(offsets))


def _offset_checkpoint_labels() -> tuple[str, ...]:
    return tuple(f"last{offset}" for offset in _retention_offsets() if offset)


def _update_offset_checkpoint_records(
    *,
    checkpoint_root: Path,
    state: dict[str, Any],
    rollout_id: int,
) -> None:
    candidates: dict[int, Path] = {}
    candidates_root = checkpoint_root / "candidates"
    if candidates_root.is_dir():
        for path in candidates_root.iterdir():
            if not path.is_dir():
                continue
            marker = _read_json(path / "checkpoint.json")
            candidate_id = _finite_int(marker.get("rollout_id"))
            if (
                candidate_id is not None
                and marker.get("format") == "huggingface"
                and marker.get("optimizer_saved") is False
            ):
                candidates[candidate_id] = path.resolve()

    for offset in _retention_offsets():
        if offset == 0:
            continue
        label = f"last{offset}"
        target_id = int(rollout_id) - offset
        target = candidates.get(target_id)
        if target is None:
            state.pop(label, None)
            try:
                (checkpoint_root / label).unlink()
            except FileNotFoundError:
                pass
            continue
        state[label] = {
            "rollout_id": target_id,
            "score_key": None,
            "score": None,
            "n_collected": None,
            "n_total": None,
            "coverage": None,
            "checkpoint": str(target),
            "nanobot_logs": None,
            "optimizer_saved": False,
            "next_rollout_id": target_id + 1,
        }


def _prune_candidates(root: Path, keep: set[Path]) -> None:
    if not root.is_dir():
        return
    for path in root.iterdir():
        if not path.is_dir() or path.resolve() in keep:
            continue
        # Resume launchers may seed a fresh run with symlinks to immutable
        # checkpoints from the preceding run.  Remove only the candidate link
        # when it ages out; never recurse through or reject the symlink.
        if path.is_symlink():
            path.unlink()
        else:
            _remove_local_tree(path)


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
