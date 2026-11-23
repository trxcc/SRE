"""Durable, atomic metric events for decoupled experiment tracking."""

from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert common scalar/container metric values to strict JSON values."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # Training backends may attach diagnostic torch.dtype values (for
    # example ``train/model_dtype``) to otherwise numeric metric mappings.
    # Keep the durable sink backend-agnostic and avoid importing torch here.
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError, RuntimeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_value(tolist())
        except (TypeError, ValueError, RuntimeError):
            pass
    raise TypeError(f"metric value of type {type(value).__name__} is not JSON serializable")


def build_metric_event(metrics: Mapping[str, Any], *, step_key: str) -> dict[str, Any]:
    """Build one complete metric event.

    The caller invokes this only after all metrics for one train, rollout, or
    eval event have been reduced and collected.
    """
    if not isinstance(step_key, str) or not step_key:
        raise ValueError("step_key must be a non-empty string")
    if step_key not in metrics:
        raise KeyError(f"metrics are missing explicit step key {step_key!r}")

    normalized = _json_value(metrics)
    if not isinstance(normalized, dict):
        raise TypeError("metrics must be a mapping")
    step = normalized[step_key]
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise TypeError(f"{step_key} must be numeric, got {type(step).__name__}")
    if isinstance(step, float) and not math.isfinite(step):
        raise ValueError(f"{step_key} must be finite")

    timestamp_ns = time.time_ns()
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": f"{socket.gethostname()}:{os.getpid()}:{timestamp_ns}",
        "created_at_ns": timestamp_ns,
        "writer": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
        "series": step_key.removesuffix("/step"),
        "step_key": step_key,
        "step": step,
        "metrics": normalized,
    }


def append_metric_event(path: str | os.PathLike[str], metrics: Mapping[str, Any], *, step_key: str) -> dict[str, Any]:
    """Append one complete JSON line under an inter-process file lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    event = build_metric_event(metrics, step_key=step_key)
    payload = (
        json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )

    lock_path = target.with_name(f"{target.name}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        fd = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while appending metric event")
                view = view[written:]
            if os.environ.get("SLIME_METRICS_JSONL_FSYNC", "0").lower() in {"1", "true", "yes", "on"}:
                os.fsync(fd)
        finally:
            os.close(fd)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return event


def append_metric_event_from_env(metrics: Mapping[str, Any], *, step_key: str) -> dict[str, Any] | None:
    """Append when ``SLIME_METRICS_JSONL_PATH`` enables the durable sink."""
    path = os.environ.get("SLIME_METRICS_JSONL_PATH")
    if not path:
        return None
    return append_metric_event(path, metrics, step_key=step_key)


def validate_metric_event(event: Any) -> dict[str, Any]:
    """Validate an event read by a relay before it reaches a remote tracker."""
    if not isinstance(event, dict):
        raise ValueError("metric event must be a JSON object")
    if event.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported metric event schema: {event.get('schema_version')!r}")
    step_key = event.get("step_key")
    metrics = event.get("metrics")
    if not isinstance(step_key, str) or not step_key:
        raise ValueError("metric event is missing step_key")
    if not isinstance(metrics, dict):
        raise ValueError("metric event is missing metrics")
    if step_key not in metrics:
        raise ValueError(f"metric event metrics are missing {step_key!r}")
    if metrics[step_key] != event.get("step"):
        raise ValueError("metric event step does not match metrics[step_key]")
    return event
