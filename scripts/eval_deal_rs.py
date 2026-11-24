#!/usr/bin/env python3
"""Standalone DEAL-RS held-out 200-task evaluator.

Evaluates any OpenAI-compatible student model endpoint (e.g., SGLang serving
trxcc2002/DEAL-RS-Gemma or trxcc2002/DEAL-RS-Qwen) against the prepared
DEAL-RS test-200 split using the native nanobot harness and hybrid GT judge.
Failed tasks are scored 0 and retained in the 200-task denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Prepared DEAL-RS eval-200 directory (or DATA_DIR containing DATA_READY.json)")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-task results and summary.json")
    ap.add_argument("--student-url", required=True, help="OpenAI-compatible student endpoint, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--student-model", default="gemma4-e4b-it", help="Served model name for student endpoint")
    ap.add_argument("--model-family", choices=["gemma4", "qwen35"], default="gemma4", help="Model family (controls default per-task timeout: 1200s gemma4, 1800s qwen35)")
    ap.add_argument("--student-api-key-file", type=Path, help="Optional chmod-600 API key file for student endpoint")
    ap.add_argument("--judge-url", required=True, help="OpenAI-compatible GT judge endpoint (paper uses Gemma4-31B-IT)")
    ap.add_argument("--judge-model", required=True, help="GT judge model name")
    ap.add_argument("--judge-api-key-file", type=Path, required=True, help="chmod-600 API key file for GT judge")
    ap.add_argument("--context-window", type=int, default=81920)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--task-timeout", type=int, default=0, help="Per-task timeout seconds (0 = recipe default: 1200 gemma4 / 1800 qwen35)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--generation-workers", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--max-tasks", type=int, default=0, help="Optional limit for quick smoke testing (0 = all 200)")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if (data_dir / "DATA_READY.json").is_file():
        data_dir = Path(json.loads((data_dir / "DATA_READY.json").read_text(encoding="utf-8"))["eval_dir"]).resolve()
    assert data_dir.is_dir(), f"DEAL-RS eval directory not found: {data_dir}"

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "eval-started.json").exists():
        raise SystemExit(f"Output directory already contains an evaluation run: {out}")

    judge_key = args.judge_api_key_file.expanduser().resolve().read_text(encoding="utf-8").strip()
    student_key = (
        args.student_api_key_file.expanduser().resolve().read_text(encoding="utf-8").strip()
        if args.student_api_key_file
        else "local-eval"
    )

    startup = ROOT / "vendor/prorl/scripts/trx_scripts/gemma4_python_startup"
    extra_paths = [ROOT, ROOT / "vendor/prorl/src", ROOT / "vendor/nanobot", ROOT / "vendor/slime", startup]
    for p in reversed(extra_paths):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    task_timeout = args.task_timeout if args.task_timeout > 0 else (1800 if args.model_family == "qwen35" else 1200)
    os.environ.update({
        "PROJECT_ROOT": str(ROOT / "vendor/prorl"),
        "NANOBOT_ROOT": str(ROOT / "vendor/nanobot"),
        "NANOBOT_RL_RUNNER": str(ROOT / "vendor/nanobot/nanobot/rl_runner.py"),
        "NANOBOT_IMPORT_LOCAL_PACKAGE": "1",
        "NANOBOT_EXTRA_PYTHONPATH": os.pathsep.join(map(str, [ROOT / "runtime/judge_gate", ROOT / "vendor/nanobot", startup])),
        "OPENCLAW_EVAL_TASKS": str(data_dir),
        "OPENCLAW_EVAL_ORDER": "sequential",
        "OPENCLAW_EVAL_MAX_TURNS": "1",
        "OPENCLAW_CONTEXT_WINDOW": str(args.context_window),
        "OPENCLAW_MAX_TOKENS": str(args.max_tokens),
        "CLAWGYM_TASK_PROMPT_MODE": "original",
        "NANOBOT_MAX_TOOL_ITERATIONS": "200",
        "NANOBOT_EVAL_TURN_TIMEOUT": str(task_timeout),
        "NANOBOT_TASK_TIMEOUT": str(task_timeout),
        "NANOBOT_MICROCOMPACT_ENABLED": "0",
        "CLAWGYM_ALLOW_MULTI_TOOL_CALLS": "1",
        "CLAWGYM_NANOBOT_ENABLE_REALIGN": "1",
        "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY": "preserve",
        "NANOBOT_STRIP_REASONING_CONTEXT": "0",
        "NANOBOT_ENABLE_THINKING": "1",
        "NANOBOT_GEMMA_TOOL_SCHEMA": "0",
        "NANOBOT_RL_DISABLED_TOOLS": "complete_goal,cron,generate_image,list_exec_sessions,long_task,message,my,notebook_edit,run_cli_app,spawn,web_fetch,web_search,write_stdin",
        "NANOBOT_RL_STRICT_NATIVE_TOOLS": "1",
        "CLAWGYM_STRICT_NATIVE_TOOLS": "1",
        "CLAWGYM_REWARD_SOURCE": "researchclaw_llmjudge",
        "RESEARCHCLAW_SCORE_MODE": "hybrid",
        "RESEARCHCLAW_LLMJUDGE_BACKEND": "hybrid_agentic",
        "RESEARCHCLAW_CODE_WEIGHT": "0.0",
        "RESEARCHCLAW_LLMJUDGE_WEIGHT": "1.0",
        "RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER": "1",
        "RESEARCHCLAW_JUDGE_BASE_URL": args.judge_url,
        "RESEARCHCLAW_JUDGE_BASE_URL_OVERRIDE": args.judge_url,
        "RESEARCHCLAW_JUDGE_MODEL": args.judge_model,
        "RESEARCHCLAW_JUDGE_MODEL_OVERRIDE": args.judge_model,
        "RESEARCHCLAW_JUDGE_API_KEY": judge_key,
        "RESEARCHCLAW_JUDGE_TEMPERATURE": "0.7",
        "RESEARCHCLAW_JUDGE_TIMEOUT": "2700",
        "RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT": "2700",
        "RESEARCHCLAW_JUDGE_MAX_TOKENS": "32768",
        "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS": "32768",
        "RESEARCHCLAW_AGENTIC_JUDGE_CONTEXT_WINDOW": "65536",
        "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOOL_ITERATIONS": "30",
        "RESEARCHCLAW_AGENTIC_JUDGE_ENABLE_THINKING": "0",
        "RESEARCHCLAW_JUDGE_SLOT_ROOT": str(out / "judge-slots"),
        "CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL": args.student_model,
        "OPENAI_API_KEY": student_key,
        "NANOBOT_LOG_DIR": str(out / "nanobot_logs"),
        "NANOBOT_WORKDIR": str(out / "work"),
        "STATE_ROOT": str(out),
    })
    (out / "judge-slots").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    (out / "work").mkdir(exist_ok=True)

    try:
        atomic_json(out / "eval-started.json", {
            "at": time.time(),
            "pid": os.getpid(),
            "student_url": args.student_url,
            "student_model": args.student_model,
            "judge_model": args.judge_model,
            "generation_workers": args.generation_workers,
            "judge_workers": args.judge_workers,
        })
        from clawgym_nanobot import researchclaw_reward, runtime, tasks

        native_reward = researchclaw_reward.run
        judge_slots = threading.BoundedSemaphore(max(1, args.judge_workers))

        def bounded_reward(*r_args, **r_kwargs):
            with judge_slots:
                return native_reward(*r_args, **r_kwargs)

        researchclaw_reward.run = bounded_reward
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
                session_id=f"deal-rs-eval-{idx:03d}",
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
                "family": (task.metadata or {}).get("family_id"),
                "status": res.status,
                "reward": score,
                "duration_s": res.duration_s,
                "error": res.error,
                "researchclaw_reward": res.researchclaw_reward,
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
        by_family: dict[str, list[float]] = {}
        for r in results:
            by_family.setdefault(str(r.get("family") or "unknown"), []).append(r["reward"])
        summary = {
            "at": time.time(),
            "benchmark": "DEAL-RS",
            "student_model": args.student_model,
            "n_tasks": n,
            "n_completed": sum(r["status"] == "completed" for r in results),
            "n_failed_scored_zero": sum(r["status"] != "completed" for r in results),
            "score_mean": sum(r["reward"] for r in results) / n,
            "score_mean_pct": 100.0 * sum(r["reward"] for r in results) / n,
            "by_family_pct": {k: 100.0 * sum(v) / len(v) for k, v in sorted(by_family.items())},
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


if __name__ == "__main__":
    raise SystemExit(main())
