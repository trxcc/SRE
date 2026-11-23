#!/usr/bin/env python3
"""Recover complete metric events from a running ProRL training log."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

from clawgym_nanobot.metrics_jsonl import append_metric_event


LOGGER = logging.getLogger("training-log-metrics-jsonl")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_RE = re.compile(
    r"(?P<source>rollout_logging|data|model|train_metric_utils)\.py:\d+ - "
    r"(?P<label>(?:critic-)?(?:step|rollout)|perf|eval) "
    r"(?P<logical_step>\d+): (?P<metrics>\{.*\})\s*$"
)
STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info("received signal %s; stopping at source EOF", signum)


class _MetricLiteralNames(ast.NodeTransformer):
    VALUES = {
        "nan": float("nan"),
        "inf": float("inf"),
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.VALUES:
            return ast.copy_location(ast.Constant(self.VALUES[node.id]), node)
        raise ValueError(f"unsupported name in metric literal: {node.id}")

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func, ast.Name) or node.func.id != "tensor" or len(node.args) != 1:
            raise ValueError("unsupported call in metric literal")
        value_node = self.visit(node.args[0])
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError("unsupported tensor value in metric literal") from exc
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if not isinstance(value, (int, float, list)):
            raise ValueError("tensor metric must contain numeric values")
        return ast.copy_location(ast.Constant(value) if not isinstance(value, list) else _literal_node(value), node)


def _literal_node(value: Any) -> ast.AST:
    if isinstance(value, list):
        return ast.List(elts=[_literal_node(item) for item in value], ctx=ast.Load())
    if isinstance(value, (int, float)):
        return ast.Constant(value)
    raise ValueError(f"unsupported tensor literal value: {value!r}")


def _parse_metric_dict(text: str) -> dict[str, Any]:
    tree = ast.parse(text, mode="eval")
    tree = _MetricLiteralNames().visit(tree)
    ast.fix_missing_locations(tree)
    value = ast.literal_eval(tree)
    if not isinstance(value, dict):
        raise ValueError("metric literal is not a dict")
    return value


def parse_metric_line(line: str, *, rollout_step_multiplier: int) -> tuple[dict[str, Any], str] | None:
    clean = ANSI_RE.sub("", line)
    match = METRIC_RE.search(clean)
    if match is None:
        return None

    metrics = _parse_metric_dict(match.group("metrics"))
    logical_step = int(match.group("logical_step"))
    source = match.group("source")
    label = match.group("label")
    if source == "model" and label.endswith("step"):
        step_key = "train/step"
        metrics.setdefault(step_key, logical_step)
    elif label == "eval":
        step_key = "eval/step"
        metrics.setdefault(step_key, logical_step * rollout_step_multiplier)
    else:
        step_key = "rollout/step"
        metrics.setdefault(step_key, logical_step * rollout_step_multiplier)
    return metrics, step_key


def _load_offset(path: Path, source: Path) -> int:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 0
    stat = source.stat()
    if state.get("source_dev") != stat.st_dev or state.get("source_ino") != stat.st_ino:
        LOGGER.warning("training log was replaced; restarting at byte offset 0")
        return 0
    offset = int(state.get("offset", 0))
    if offset < 0 or offset > stat.st_size:
        LOGGER.warning("training log was truncated; restarting at byte offset 0")
        return 0
    return offset


def _save_offset(path: Path, source: Path, *, offset: int, events_written: int) -> None:
    stat = source.stat()
    state = {
        "schema_version": 1,
        "source": str(source),
        "source_dev": stat.st_dev,
        "source_ino": stat.st_ino,
        "offset": offset,
        "events_written": events_written,
        "updated_at_ns": time.time_ns(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    state_path = args.state.resolve()
    completion_marker = args.completion_marker.resolve() if args.completion_marker else None
    offset = _load_offset(state_path, source)
    events_written = 0
    LOGGER.info("producer ready source=%s output=%s offset=%d", source, output, offset)

    with source.open("rb") as handle:
        handle.seek(offset)
        while True:
            line = handle.readline()
            if not line:
                done = STOP_REQUESTED or bool(args.done_file and args.done_file.exists())
                if done:
                    break
                time.sleep(args.poll_seconds)
                continue
            line_end = handle.tell()
            try:
                decoded = line.decode("utf-8", errors="replace")
                parsed = parse_metric_line(decoded, rollout_step_multiplier=args.rollout_step_multiplier)
            except (SyntaxError, ValueError, TypeError) as exc:
                LOGGER.error("failed to parse metric line ending at byte %d: %s", line_end, exc)
                return 1
            if parsed is None:
                continue
            metrics, step_key = parsed
            append_metric_event(output, metrics, step_key=step_key)
            offset = line_end
            events_written += 1
            _save_offset(state_path, source, offset=offset, events_written=events_written)
            if events_written == 1 or events_written % args.status_every == 0:
                LOGGER.info(
                    "produced events=%d source_offset=%d series=%s step=%s",
                    events_written,
                    offset,
                    step_key.removesuffix("/step"),
                    metrics[step_key],
                )

    if completion_marker is not None:
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.touch()
    LOGGER.info("producer stopped events=%d source_offset=%d", events_written, offset)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--completion-marker", type=Path)
    parser.add_argument("--rollout-step-multiplier", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--status-every", type=int, default=25)
    args = parser.parse_args()
    if args.rollout_step_multiplier <= 0:
        parser.error("--rollout-step-multiplier must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.status_every <= 0:
        parser.error("--status-every must be positive")
    if not args.source.is_file():
        parser.error(f"--source is not a file: {args.source}")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
