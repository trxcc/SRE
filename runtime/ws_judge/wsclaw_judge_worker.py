#!/usr/bin/env python3
"""Run bounded Claude Code judge workers next to a local Gemma server."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import sys
from judge_inspector_adapter import install_inspector
sys.path.insert(0, str(Path(os.environ["PROJECT_ROOT"]) / "scripts" / "clawgym_nanobot"))
import wsclaw_eval_only as evaluator
from judge_prompt_only import install as install_prompt_only, SYSTEM_PROMPT
import hashlib
install_prompt_only(evaluator)


def _allowed_work_root() -> Path:
    configured = os.environ.get("WSCLAW_ALLOWED_WORK_ROOT") or os.environ.get("NANOBOT_WORKDIR") or "/tmp"
    return Path(configured).expanduser().resolve()


def selfreward_judge_binding(case_dir: Path, judge_workspace: Path):
    """Bind shared self-reward GT to this run's queue and candidate snapshot."""
    sr_cfg_path = os.environ.get("CLAWGYM_SELF_REWARD_CONFIG", "").strip()
    if not sr_cfg_path or not Path(sr_cfg_path).is_file():
        return None
    config = json.loads(Path(sr_cfg_path).read_text())
    queue = Path(config["queue_root"]).resolve(strict=True)
    snapshots = Path(config["snapshot_root"]).resolve(strict=True)
    case, judge = Path(case_dir), Path(judge_workspace)
    if case.is_symlink() or judge.is_symlink():
        raise ValueError("judge roots must not be symlinks")
    case, judge = case.resolve(strict=True), judge.resolve(strict=True)
    if not case.is_relative_to(queue):
        return None
    parts = case.relative_to(queue).parts
    import re
    if (len(parts) != 4 or not re.fullmatch(r"[0-9a-f]{64}", parts[0])
            or not re.fullmatch(r"evaluation-attempt-[1-9][0-9]*", parts[1])
            or parts[2:] != ("ground_truth", "candidate-0")
            or judge != case / "judge_workspace"):
        raise ValueError("invalid self-reward GT case layout")
    group_dir = queue / parts[0]
    if (group_dir / "request.json").is_file():
        request = json.loads((group_dir / "request.json").read_text())
        if request.get("group_id") != parts[0]:
            raise ValueError("self-reward group identity mismatch")
        candidates = request["candidates"]
        if len(candidates) != 8 or [c["candidate_id"] for c in candidates] != list(range(8)):
            raise ValueError("self-reward candidate identity mismatch")
        source = Path(candidates[0]["workspace"])
    else:
        cand0 = json.loads((group_dir / "candidates" / "0.json").read_text())
        source = Path(cand0["workspace"])
    if source.is_symlink() or not source.is_dir():
        raise ValueError("candidate snapshot root must be a local directory")
    source = source.resolve(strict=True)
    if not source.is_relative_to(snapshots) or source == snapshots or source.name != "workspace":
        raise ValueError("candidate snapshot outside this run")
    solver, view = case / "solver_workspace", judge / "workspace"
    if solver.is_symlink() or view.is_symlink() or not solver.is_dir() or not view.is_dir():
        raise ValueError("self-reward solver and judge views must be directories")
    for child in solver.iterdir():
        origin, mapped = source / child.name, view / child.name
        if (not child.is_symlink() or child.resolve() != origin.resolve()
                or not mapped.is_symlink() or mapped.resolve() != origin.resolve()):
            raise ValueError("self-reward view differs from candidate: " + child.name)
    return queue, source


def solver_workspace_for_judge(case_dir: Path, judge_workspace: Path) -> Path:
    """Resolve the production judge view back to its owned solver workspace."""
    bound = selfreward_judge_binding(case_dir, judge_workspace)
    if bound is not None:
        return bound[1]
    allowed = _allowed_work_root()
    case_dir = Path(case_dir)
    judge_workspace = Path(judge_workspace)
    if case_dir.is_symlink() or judge_workspace.is_symlink():
        raise ValueError("judge roots must not be symlinks")
    if judge_workspace != case_dir / "judge_workspace":
        raise ValueError("judge workspace is not bound to case")
    root = case_dir.parent / "workspace"
    if (not root.is_dir() or root.is_symlink()
            or not root.resolve(strict=True).is_relative_to(allowed)):
        raise ValueError("solver workspace must be an owned local directory")
    view = judge_workspace / "workspace"
    # Native prepare_judge_workspace links each top-level child of the solver.
    # Validate this binding rather than allowing arbitrary outside links.
    for child in root.iterdir():
        mapped = view / child.name
        if not mapped.is_symlink() or mapped.resolve() != child.resolve():
            raise ValueError("judge view does not match solver workspace: " + child.name)
    return root


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
                "judge_mode": "system_prompt_only",
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
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
            bound = selfreward_judge_binding(paths["case_dir"], paths["judge_workspace"])
            installation = install_inspector(
                paths["judge_workspace"], paths["case_dir"],
                Path(__file__).resolve().with_name("inspect_workspace_file.py"),
                allowed_root=bound[0] if bound is not None else _allowed_work_root(),
            )
            evaluator.write_json(paths["case_dir"] / "inspector-install.json", installation)
            # Trusted task paths and deterministic format evidence are provided
            # before rubric evaluation. Submitted code is compiled, never run.
            contract = json.loads((paths["task_dir"] / "reward/output_contract.json").read_text())
            required = [dict(item) for item in contract["output_files"]
                        if isinstance(item, dict) and item.get("must_create", True) and item.get("path")]
            # Judge-visible Desktop is a trusted link to the solver workspace.
            # Validate containment and format against its true owned root.
            solver_workspace = solver_workspace_for_judge(paths["case_dir"], paths["judge_workspace"])
            validator = Path(os.environ["PROJECT_ROOT"]) / "src" / "clawgym_nanobot" / "artifact_validity_gate.py"
            checked = subprocess.run(
                [os.environ["WSBENCH_PYTHON"], str(validator), "--check"],
                input=json.dumps({"workspace": str(solver_workspace), "required": required}),
                capture_output=True, text=True, timeout=120,
            )
            if checked.returncode:
                raise RuntimeError("pre-judge artifact validator infrastructure failure: " + checked.stderr[-2000:])
            validation = json.loads(checked.stdout)
            evaluator.write_json(paths["judge_workspace"] / "review/artifact_validation.json", validation)
            evaluator.write_json(paths["case_dir"] / "pre-judge-artifact-validation.json", validation)
            prompt = evaluator.judge_user_message(paths["judge_workspace"])
            prompt += "\n\nTrusted deterministic artifact-validation evidence (readability/syntax only; not a content grade):\n"
            prompt += json.dumps(validation, ensure_ascii=False)
            prompt += "\nCheck this evidence alongside the actual artifact and original sources. A parser or compiler error blocks execution before hypothetical runtime behavior. A valid format does not establish semantic correctness."
            result = evaluator.run_judge(
                paths["judge_workspace"],
                paths["task_dir"],
                paths["case_dir"],
                self.server.judge_args,
                initial_prompt=prompt,
            )
            result["pre_judge_artifact_validation"] = validation
            import importlib.util
            mask_path = Path(os.environ["PROJECT_ROOT"]) / "src" / "clawgym_nanobot" / "judge_failure_mask.py"
            spec = importlib.util.spec_from_file_location('judge_terminal_evidence', mask_path)
            mask_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mask_module)
            try:
                result['judge_terminal_evidence'] = mask_module.terminal_evidence(result['judge_log_path'])
            except (KeyError, OSError, ValueError) as exc:
                result['judge_terminal_evidence_error'] = f'{type(exc).__name__}: {exc}'
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
