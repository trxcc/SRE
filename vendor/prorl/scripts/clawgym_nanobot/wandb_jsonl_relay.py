#!/usr/bin/env python3
"""Relay complete JSONL metric events to W&B from one dedicated process."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import signal
import sys
import time
from pathlib import Path
from typing import Any

import wandb

from clawgym_nanobot.metrics_jsonl import validate_metric_event


LOGGER = logging.getLogger("wandb-jsonl-relay")
STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info("received signal %s; draining complete JSONL records before exit", signum)


def _load_private_env(path: Path | None) -> None:
    if path is None or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
        except ValueError:
            continue
        if not parts or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        if key.startswith("WANDB_") and not os.environ.get(key):
            os.environ[key] = value


def _configure_wandb_network() -> None:
    proxy = (
        os.environ.get("WANDB_PROXY_URL")
        or os.environ.get("WANDB_PROXY")
        or os.environ.get("WANDB_HTTPS_PROXY")
        or os.environ.get("WANDB_HTTP_PROXY")
    )
    if not proxy:
        return
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = proxy

    no_proxy = [
        part.strip()
        for part in (os.environ.get("WANDB_NO_PROXY") or "").split(",")
        if part.strip() and part.strip() not in {"*", "api.wandb.ai", ".wandb.ai"}
    ]
    for local in ("localhost", "127.0.0.1", "::1"):
        if local not in no_proxy:
            no_proxy.append(local)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(no_proxy)


def _load_state(path: Path, source: Path) -> dict[str, int]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"offset": 0, "records_logged": 0}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid relay state file {path}: {exc}") from exc

    offset = int(state.get("offset", 0))
    if offset < 0:
        raise RuntimeError(f"negative relay offset in {path}")
    stat = source.stat()
    recorded_source = state.get("source")
    if (recorded_source is not None and recorded_source != str(source)) or offset > stat.st_size:
        LOGGER.warning("metric source was replaced or truncated; restarting at byte offset 0")
        return {"offset": 0, "records_logged": int(state.get("records_logged", 0))}
    if state.get("source_dev") != stat.st_dev or state.get("source_ino") != stat.st_ino:
        # Device and inode numbers are mount-namespace local. The same shared
        # file can therefore have different identities when a relay moves from
        # a training pod to a controller/development host. The stable absolute
        # path plus a still-valid byte offset are the portable resume contract.
        LOGGER.info("metric source identity differs across mounts; resuming at byte offset %d", offset)
    return {"offset": offset, "records_logged": int(state.get("records_logged", 0))}


def _save_state(path: Path, source: Path, *, offset: int, records_logged: int, last_event_id: str) -> None:
    stat = source.stat()
    state = {
        "schema_version": 1,
        "source": str(source),
        "source_dev": stat.st_dev,
        "source_ino": stat.st_ino,
        "offset": offset,
        "records_logged": records_logged,
        "last_event_id": last_event_id,
        "updated_at_ns": time.time_ns(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _define_metric_axes(*, canonical_step_key: str | None = None) -> None:
    if canonical_step_key is not None:
        wandb.define_metric(canonical_step_key)
        for pattern in (
            "train/*",
            "rollout/*",
            "multi_turn/*",
            "passrate/*",
            "perf/*",
            "clawgym/*",
            "openclaw/*",
            "polar/*",
            "eval/*",
        ):
            wandb.define_metric(pattern, step_metric=canonical_step_key)
        return

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    for pattern in ("rollout/*", "multi_turn/*", "passrate/*", "perf/*", "clawgym/*", "openclaw/*", "polar/*"):
        wandb.define_metric(pattern, step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")


def _log_metric_event(
    event: dict[str, Any],
    *,
    canonical_step_key: str | None,
    initial_global_step: int | None = None,
) -> None:
    if canonical_step_key is None:
        wandb.log(event["metrics"], commit=True)
        return
    if event["step_key"] != canonical_step_key:
        raise RuntimeError(f"expected step_key={canonical_step_key!r}, got {event['step_key']!r}")
    metric_step = event["metrics"].get(canonical_step_key)
    if metric_step != event["step"]:
        raise RuntimeError(
            f"expected metrics[{canonical_step_key!r}]={event['step']!r}, got {metric_step!r}"
        )
    # Offset a fresh W&B run to the restored optimizer/rollout step so the
    # default `_step` axis does not restart at zero after weights-only resume.
    # Only the first record receives an explicit global step. Later records
    # remain append-only, which still permits late/backfilled eval iterations;
    # their canonical `iteration` axis remains authoritative.
    kwargs: dict[str, Any] = {"commit": True}
    if initial_global_step is not None:
        kwargs["step"] = initial_global_step
    wandb.log(event["metrics"], **kwargs)


def relay(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch(exist_ok=True)
    state_path = args.state.resolve()
    state = _load_state(state_path, source)

    settings = wandb.Settings(
        mode=args.mode,
        console="off",
        x_update_finish_state=True,
    )
    run = wandb.init(
        id=args.run_id,
        resume="allow",
        entity=args.entity,
        project=args.project,
        group=args.group,
        name=args.name,
        dir=str(args.wandb_dir.resolve()),
        settings=settings,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    _define_metric_axes(canonical_step_key=args.canonical_step_key)
    LOGGER.info(
        "relay ready run_id=%s source=%s offset=%d",
        args.run_id,
        source,
        state["offset"],
    )

    offset = state["offset"]
    records_logged = state["records_logged"]
    with source.open("rb") as handle:
        handle.seek(offset)
        while True:
            record_offset = handle.tell()
            line = handle.readline()
            if not line:
                if STOP_REQUESTED or bool(args.done_file and args.done_file.exists()):
                    break
                time.sleep(args.poll_seconds)
                continue
            if not line.endswith(b"\n"):
                handle.seek(record_offset)
                if STOP_REQUESTED:
                    LOGGER.warning("leaving incomplete trailing JSONL record at byte %d", record_offset)
                    break
                time.sleep(args.poll_seconds)
                continue

            try:
                event = validate_metric_event(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"invalid complete metric record at byte {record_offset}: {exc}") from exc

            _log_metric_event(
                event,
                canonical_step_key=args.canonical_step_key,
                initial_global_step=(int(event["step"]) if records_logged == 0 else None),
            )
            offset = handle.tell()
            records_logged += 1
            _save_state(
                state_path,
                source,
                offset=offset,
                records_logged=records_logged,
                last_event_id=str(event.get("event_id", "")),
            )
            if records_logged == 1 or records_logged % args.status_every == 0:
                LOGGER.info(
                    "relayed records=%d offset=%d series=%s step=%s",
                    records_logged,
                    offset,
                    event.get("series"),
                    event.get("step"),
                )

    LOGGER.info("relay drained records=%d offset=%d", records_logged, offset)
    run.finish(exit_code=0)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--canonical-step-key")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group")
    parser.add_argument("--wandb-dir", type=Path, required=True)
    parser.add_argument("--private-env", type=Path)
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--status-every", type=int, default=10)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.status_every <= 0:
        parser.error("--status-every must be positive")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime
    args = parse_args()
    _load_private_env(args.private_env)
    _configure_wandb_network()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        return relay(args)
    except Exception:
        LOGGER.exception("relay failed")
        with __import__("contextlib").suppress(Exception):
            wandb.finish(exit_code=1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
