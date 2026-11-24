#!/usr/bin/env python3
"""Standalone DEAL-WS held-out 200-task evaluator with deliverable hardgate.

Evaluates any OpenAI-compatible student model endpoint (e.g., SGLang serving
trxcc2002/DEAL-WS-Gemma or trxcc2002/DEAL-WS-Qwen) against the prepared
DEAL-WS test-200 split using the native nanobot harness, deterministic
deliverable format hardgate (CLAWGYM_ARTIFACT_VALIDITY_GATE=1), and WSClaw GT
judge sidecar. Failed tasks are scored 0 and retained in the 200-task denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if mode is not None:
        tmp.chmod(mode)
    tmp.replace(path)


def wait_http(url: str, proc: subprocess.Popen, timeout: int = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"WS judge sidecar exited early (rc={proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for WS judge sidecar at {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Prepared DEAL-WS eval-200 directory (or DATA_DIR containing DATA_READY.json)")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-task results and summary.json")
    ap.add_argument("--student-url", required=True, help="OpenAI-compatible student endpoint, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--student-model", default="gemma4-e4b-it", help="Served model name for student endpoint")
    ap.add_argument("--model-family", choices=["gemma4", "qwen35"], default="gemma4")
    ap.add_argument("--student-api-key-file", type=Path, help="Optional chmod-600 API key file for student endpoint")
    ap.add_argument("--judge-url", required=True, help="OpenAI-compatible GT judge endpoint (paper uses Gemma4-31B-IT)")
    ap.add_argument("--judge-model", required=True, help="GT judge model name")
    ap.add_argument("--judge-api-key-file", type=Path, required=True, help="chmod-600 API key file for GT judge")
    ap.add_argument("--claude-bin", default="", help="Optional path to Claude Code CLI (defaults to PATH lookup)")
    ap.add_argument("--judge-proxy-port", type=int, default=28198)
    ap.add_argument("--judge-worker-port", type=int, default=28200)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--generation-workers", type=int, default=8)
    ap.add_argument("--context-window", type=int, default=65536)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--task-timeout", type=int, default=0, help="Per-task timeout seconds (0 = recipe default: 1500 gemma4 / 1800 qwen35)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--max-tasks", type=int, default=0, help="Optional limit for quick smoke testing (0 = all 200)")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if (data_dir / "DATA_READY.json").is_file():
        data_dir = Path(json.loads((data_dir / "DATA_READY.json").read_text(encoding="utf-8"))["eval_dir"]).resolve()
    assert data_dir.is_dir(), f"DEAL-WS eval directory not found: {data_dir}"

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "eval-started.json").exists():
        raise SystemExit(f"Output directory already contains an evaluation run: {out}")

    judge_key_file = args.judge_api_key_file.expanduser().resolve()
    assert judge_key_file.is_file(), f"Missing judge API key file: {judge_key_file}"
    student_key = (
        args.student_api_key_file.expanduser().resolve().read_text(encoding="utf-8").strip()
        if args.student_api_key_file
        else "local-eval"
    )

    startup = ROOT / "vendor/prorl/scripts/trx_scripts/gemma4_python_startup"
    extra_paths = [ROOT, ROOT / "vendor/prorl/src", ROOT / "vendor/prorl", ROOT / "vendor/nanobot", ROOT / "vendor/slime", startup]
    for p in reversed(extra_paths):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    worker_key = secrets.token_urlsafe(24)
    router_cfg_path = out / "private/ws-gt-judge.json"
    atomic_json(
        router_cfg_path,
        {"model": args.judge_model, "api_bases": [args.judge_url], "api_key_file": str(judge_key_file)},
        mode=0o600,
    )
    worker_url = f"http://127.0.0.1:{args.judge_worker_port}"
    work_root = out / "work"
    work_root.mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)

    task_timeout = args.task_timeout if args.task_timeout > 0 else (1800 if args.model_family == "qwen35" else 1500)
    env_updates = {
        "PROJECT_ROOT": str(ROOT / "vendor/prorl"),
        "NANOBOT_ROOT": str(ROOT / "vendor/nanobot"),
        "NANOBOT_RL_RUNNER": str(ROOT / "vendor/nanobot/nanobot/rl_runner.py"),
        "NANOBOT_IMPORT_LOCAL_PACKAGE": "1",
        "NANOBOT_EXTRA_PYTHONPATH": os.pathsep.join(map(str, [ROOT / "vendor/nanobot", startup])),
        "OPENCLAW_EVAL_TASKS": str(data_dir),
        "OPENCLAW_EVAL_ORDER": "sequential",
        "OPENCLAW_EVAL_MAX_TURNS": "1",
        "OPENCLAW_CONTEXT_WINDOW": str(args.context_window),
        "OPENCLAW_MAX_TOKENS": str(args.max_tokens),
        "CLAWGYM_TASK_PROMPT_MODE": "original",
        "NANOBOT_MAX_TOOL_ITERATIONS": "200",
        "NANOBOT_EVAL_TURN_TIMEOUT": str(task_timeout),
        "NANOBOT_TASK_TIMEOUT": str(task_timeout),
        "NANOBOT_MICROCOMPACT_ENABLED": "1",
        "CLAWGYM_ALLOW_MULTI_TOOL_CALLS": "1",
        "CLAWGYM_NANOBOT_ENABLE_REALIGN": "1",
        "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY": "preserve",
        "NANOBOT_STRIP_REASONING_CONTEXT": "0",
        "NANOBOT_ENABLE_THINKING": "1",
        "NANOBOT_GEMMA_TOOL_SCHEMA": "1" if args.model_family == "gemma4" else "0",
        "NANOBOT_RL_DISABLED_TOOLS": "spawn,message,long_task,complete_goal",
        "NANOBOT_RL_STRICT_NATIVE_TOOLS": "1",
        "CLAWGYM_STRICT_NATIVE_TOOLS": "1",
        "CLAWGYM_REWARD_SOURCE": "llm_judge",
        "CLAWGYM_ARTIFACT_VALIDITY_GATE": "1",
        "CLAWGYM_WSCLAW_SANDBOX": "0",
        "GEMMA_EVAL_INFRA_FAILURE_REPORT_ZERO": "1",
        "CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL": args.student_model,
        "OPENAI_API_KEY": student_key,
        "NANOBOT_LOG_DIR": str(out / "nanobot_logs"),
        "NANOBOT_WORKDIR": str(work_root),
        "WSCLAW_ALLOWED_WORK_ROOT": str(work_root),
        "STATE_ROOT": str(out),
        "GEMMA4_WSCLAW_JUDGE_ROUTER_CONFIG": str(router_cfg_path),
        "WSCLAW_JUDGE_MODEL": args.judge_model,
        "WSCLAW_JUDGE_WORKERS_PER_NODE": str(args.judge_workers),
        "WSCLAW_JUDGE_UPSTREAM_TIMEOUT": "300",
        "POLAR_EVALUATOR_TIMEOUT": "2700",
        "WSCLAW_JUDGE_PROXY_PORT": str(args.judge_proxy_port),
        "WSCLAW_JUDGE_WORKER_PORT": str(args.judge_worker_port),
        "WSCLAW_JUDGE_WORKER_URLS": worker_url,
        "WSCLAW_JUDGE_API_KEY": worker_key,
        "NODE_RANK": "0",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    if args.claude_bin:
        env_updates["CLAUDE_BIN"] = args.claude_bin
    os.environ.update(env_updates)

    sidecar_log = (out / "ws-judge-sidecar.log").open("ab", buffering=0)
    sidecar = subprocess.Popen(
        ["bash", str(ROOT / "runtime/ws_judge/judge_sidecar.sh"), "run"],
        env=os.environ.copy(),
        stdout=sidecar_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_http(f"{worker_url}/health", sidecar, timeout=180)
        atomic_json(out / "eval-started.json", {
            "at": time.time(),
            "pid": os.getpid(),
            "student_url": args.student_url,
            "student_model": args.student_model,
            "judge_model": args.judge_model,
            "hardgate": True,
            "generation_workers": args.generation_workers,
            "judge_workers": args.judge_workers,
        })

        from clawgym_nanobot import runtime, tasks

        all_tasks = tasks.load_tasks("OPENCLAW_EVAL_TASKS", default=None)
        if args.max_tasks > 0:
            all_tasks = all_tasks[: args.max_tasks]
        ids = [t.task_id for t in all_tasks]
        assert len(ids) == len(set(ids)) and len(ids) > 0
        atomic_json(out / "eval-task-order.json", {
            "count": len(ids),
            "task_ids": ids,
            "sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        })
        run_args = SimpleNamespace(save=str(out), eval_max_context_len=args.context_window)
        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_tokens,
        }
        atomic_json(out / "sampling.json", sampling)

        started = time.monotonic()
        results = []
        student_base = args.student_url.rstrip("/").removesuffix("/v1")

        def run_one(pair):
            idx, task = pair
            res = runtime.run_nanobot_task(
                run_args,
                task,
                session_id=f"deal-ws-eval-{idx:03d}",
                adapter_base_url=student_base,
                rollout_id=0,
                task_index=idx,
                group_index=idx,
                sampling_params=sampling,
                evaluation=True,
                preserve_run_dir=True,
            )
            score = float(res.reward) if (res.status == "completed" and math.isfinite(float(res.reward))) else 0.0
            row = {
                "index": idx,
                "task_id": res.task_id,
                "persona": (task.metadata or {}).get("persona"),
                "status": res.status,
                "reward": score,
                "duration_s": res.duration_s,
                "error": res.error,
                "evaluation_reward": res.evaluation_reward,
            }
            atomic_json(out / "results" / f"{idx:03d}.json", row)
            return row

        with ThreadPoolExecutor(max_workers=max(1, args.generation_workers)) as pool:
            futures = [pool.submit(run_one, pair) for pair in enumerate(all_tasks)]
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                progress = {
                    "at": time.time(),
                    "completed": len(results),
                    "total": len(all_tasks),
                    "success": sum(r["status"] == "completed" for r in results),
                    "failed": sum(r["status"] != "completed" for r in results),
                    "duration_s": round(time.monotonic() - started, 2),
                }
                atomic_json(out / "progress.json", progress)
                print(json.dumps({**progress, "last_task": row["task_id"], "last_score": row["reward"]}), flush=True)

        results.sort(key=lambda r: r["index"])
        n = len(results)
        by_persona: dict[str, list[float]] = {}
        for r in results:
            by_persona.setdefault(str(r.get("persona") or "unknown"), []).append(r["reward"])
        summary = {
            "at": time.time(),
            "benchmark": "DEAL-WS",
            "hardgate": True,
            "student_model": args.student_model,
            "n_tasks": n,
            "n_completed": sum(r["status"] == "completed" for r in results),
            "n_failed_scored_zero": sum(r["status"] != "completed" for r in results),
            "score_mean": sum(r["reward"] for r in results) / n,
            "score_mean_pct": 100.0 * sum(r["reward"] for r in results) / n,
            "by_persona_pct": {k: 100.0 * sum(v) / len(v) for k, v in sorted(by_persona.items())},
            "duration_s": round(time.monotonic() - started, 2),
        }
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "eval-exit.json", {"at": time.time(), "returncode": 0, "complete": True})
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    except BaseException as exc:
        (out / "driver-error.log").write_text(traceback.format_exc(), encoding="utf-8")
        atomic_json(out / "eval-exit.json", {"at": time.time(), "returncode": 1, "error": type(exc).__name__, "complete": False})
        raise
    finally:
        if sidecar.poll() is None:
            try:
                os.killpg(sidecar.pid, signal.SIGTERM)
                sidecar.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(sidecar.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        sidecar_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
