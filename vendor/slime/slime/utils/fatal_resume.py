"""Structured fatal-resume markers for externally supervised training runs."""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path


FATAL_RESUME_PREFIX = "PRORL_FATAL_RESUME_REQUIRED"


def _latest_complete_checkpoint() -> tuple[str | None, int | None]:
    root_raw = os.environ.get("PRORL_CHECKPOINT_CANDIDATES_DIR")
    candidates: list[tuple[int, Path]] = []
    if root_raw:
        root = Path(root_raw)
        try:
            paths = tuple(root.iterdir())
        except OSError:
            paths = ()
        for path in paths:
            match = re.fullmatch(r"step-(\d+)", path.name)
            if match and _is_complete_checkpoint(path):
                candidates.append((int(match.group(1)), path))
    initial_raw = os.environ.get("PRORL_INITIAL_RESUME_CHECKPOINT")
    if initial_raw:
        initial = Path(initial_raw)
        match = re.fullmatch(r"step-(\d+)", initial.name)
        if match and _is_complete_checkpoint(initial):
            candidates.append((int(match.group(1)), initial))
    if not candidates:
        return None, None
    step, path = max(candidates, key=lambda item: item[0])
    return str(path), step + 1


def _is_complete_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return (path / "model.safetensors.index.json").is_file() or any(
        path.glob("*.safetensors")
    )


def _bounded(value: object, limit: int = 2048) -> str:
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def emit_fatal_resume(logger, reason: str, **details: object) -> dict[str, object]:
    """Log a stable fatal prefix and atomically publish a manual-resume marker."""

    checkpoint, start_rollout_id = _latest_complete_checkpoint()
    payload: dict[str, object] = {
        "action": "stop_and_resume",
        "component": _bounded(details.pop("component", "slime"), 128),
        "host": socket.gethostname(),
        "manual_resume_required": True,
        "pid": os.getpid(),
        "reason": _bounded(reason, 256),
        "resume_policy": "latest_complete_checkpoint_before_failed_rollout",
        "run_id": os.environ.get("PRORL_RUN_ID", os.environ.get("RUN_ID", "unknown")),
        "suggested_checkpoint": checkpoint,
        "suggested_start_rollout_id": start_rollout_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update({key: _bounded(value) for key, value in details.items()})
    logger.critical("%s %s", FATAL_RESUME_PREFIX, json.dumps(payload, sort_keys=True))

    marker_dir_raw = os.environ.get("PRORL_FATAL_RESUME_MARKER_DIR")
    if not marker_dir_raw:
        return payload
    marker_dir = Path(marker_dir_raw)
    try:
        marker_dir.mkdir(parents=True, exist_ok=True)
        component = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(payload["component"]))[:80]
        marker = marker_dir / f"{component}-{socket.gethostname()}.json"
        temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, marker)
    except OSError as exc:
        logger.exception("Failed to write fatal-resume marker: %s", exc)
    return payload
