#!/usr/bin/env python3
"""Run bounded Claude Code judge workers next to a local Gemma server."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import wsclaw_eval_only as evaluator


class ThreadPoolHTTPServer(HTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        max_workers: int,
        judge_args: SimpleNamespace,
        api_key: str,
    ) -> None:
        self.executor: ThreadPoolExecutor | None = None
        # A full eval wave opens all client connections at once.  The stdlib
        # default listen backlog is only five, which resets most of a 70-way
        # submission before the thread pool can accept it.
        self.request_queue_size = max(128, max_workers)
        super().__init__(server_address, handler_class)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="wsclaw-judge",
        )
        self.max_workers = max_workers
        self.judge_args = judge_args
        self.api_key = api_key
        self.hostname = socket.gethostname()
        self.active = 0
        self.completed = 0
        self.activity_lock = threading.Lock()
        self.request_context = threading.local()

    def process_request(self, request: Any, client_address: Any) -> None:
        assert self.executor is not None
        self.executor.submit(
            self._process_request,
            request,
            client_address,
            time.monotonic(),
        )

    def _process_request(
        self,
        request: Any,
        client_address: Any,
        enqueued_at: float,
    ) -> None:
        self.request_context.queue_wait_sec = max(0.0, time.monotonic() - enqueued_at)
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.request_context.queue_wait_sec = 0.0
            self.shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


class JudgeRequestHandler(BaseHTTPRequestHandler):
    server: ThreadPoolHTTPServer

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/", "/health"):
            self._send_json(404, {"error": "not found"})
            return
        with self.server.activity_lock:
            active = self.server.active
            completed = self.server.completed
        self._send_json(
            200,
            {
                "status": "ok",
                "hostname": self.server.hostname,
                "pid": os.getpid(),
                "max_workers": self.server.max_workers,
                "active": active,
                "completed": completed,
                "judge_api_base": self.server.judge_args.judge_api_base,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/judge":
            self._send_json(404, {"error": "not found"})
            return
        if self.headers.get("Authorization") != f"Bearer {self.server.api_key}":
            self._send_json(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 1024 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            paths = {
                name: Path(str(payload[name]))
                for name in ("judge_workspace", "task_dir", "case_dir")
            }
            if not all(path.is_absolute() for path in paths.values()):
                raise ValueError("judge paths must be absolute")
            if not all(path.exists() for path in paths.values()):
                raise FileNotFoundError("one or more judge paths do not exist")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
            self._send_json(400, {"error": f"invalid request: {exc}"})
            return

        with self.server.activity_lock:
            self.server.active += 1
        try:
            worker_started = time.monotonic()
            result = evaluator.run_judge(
                paths["judge_workspace"],
                paths["task_dir"],
                paths["case_dir"],
                self.server.judge_args,
            )
            worker_run_sec = time.monotonic() - worker_started
            judge_attempt_sec = float(result.get("judge_duration_sec") or 0.0)
            result.update(
                {
                    "judge_worker_host": self.server.hostname,
                    "judge_worker_pid": os.getpid(),
                    "judge_worker_queue_wait_sec": float(
                        getattr(self.server.request_context, "queue_wait_sec", 0.0)
                    ),
                    "judge_worker_run_sec": worker_run_sec,
                    "judge_worker_non_attempt_sec": max(
                        0.0, worker_run_sec - judge_attempt_sec
                    ),
                }
            )
            evaluator.write_json(paths["case_dir"] / "judge_result.json", result)
            with self.server.activity_lock:
                self.server.completed += 1
            self._send_json(200, result)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "judge_worker_host": self.server.hostname,
                },
            )
        finally:
            with self.server.activity_lock:
                self.server.active -= 1

    def log_message(self, format_string: str, *args: Any) -> None:
        print(
            f"judge_worker host={self.server.hostname} client={self.client_address[0]} "
            f"{format_string % args}",
            flush=True,
        )


def build_server(args: argparse.Namespace) -> ThreadPoolHTTPServer:
    judge_args = SimpleNamespace(
        wsbench_python=args.wsbench_python,
        judge_api_base=args.judge_api_base,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        claude_bin=args.claude_bin,
        judge_timeout=args.judge_timeout,
        judge_max_turns=args.judge_max_turns,
        judge_effort=args.judge_effort,
    )
    return ThreadPoolHTTPServer(
        (args.host, args.port),
        JudgeRequestHandler,
        max_workers=args.workers,
        judge_args=judge_args,
        api_key=args.api_key,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18094)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--api-key", default="apiKey")
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-api-key", default="apiKey")
    parser.add_argument("--judge-model", default="gemma-user-agent")
    parser.add_argument("--judge-timeout", type=int, default=2700)
    parser.add_argument("--judge-max-turns", type=int, default=30)
    parser.add_argument("--judge-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--claude-bin", type=Path, default=evaluator.DEFAULT_CLAUDE_BIN)
    parser.add_argument("--wsbench-python", type=Path, default=evaluator.DEFAULT_WSBENCH_PYTHON)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.judge_max_turns < 1:
        parser.error("--judge-max-turns must be positive")
    for path in (args.claude_bin, args.wsbench_python):
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(f"required executable is not an executable file: {path}")
    return args


def main() -> None:
    args = parse_args()
    server = build_server(args)
    print(
        f"judge_worker_started host={server.hostname} port={server.server_port} "
        f"max_workers={server.max_workers} judge_api_base={args.judge_api_base}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
