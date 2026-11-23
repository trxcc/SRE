from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .common import atomic_json, read_secret
from .native import import_native


def _final_message(workspace: Path, session_id: str, log_dir: Path) -> str:
    session = workspace / "sessions" / f"{session_id}.jsonl"
    if session.is_file():
        messages = []
        for line in session.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                messages.append(value)
        for message in reversed(messages):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                return message["content"].strip()
    session_log = log_dir / session_id
    for path in reversed(sorted(session_log.glob("*_agent-end.json"))):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("final_response", "response", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _one_rollout(
    *,
    runtime: Any,
    polar_workspace: Any,
    task: Any,
    task_index: int,
    candidate_id: int,
    adapter_base_url: str,
    run_root: Path,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    candidate_root = run_root / f"candidate-{candidate_id}"
    candidate_root.mkdir(parents=True, exist_ok=False)
    workspace = candidate_root / "workspace"
    log_root = candidate_root / "nanobot_logs"
    log_root.mkdir()
    polar_workspace.prepare_task_workspace(task.task_dir, workspace)
    session_id = f"sr-correctness-{task.task_id}-{candidate_id}-{time.time_ns()}"
    started = time.monotonic()
    guard = runtime.raw_source_integrity.TaskSourceGuard(
        task.task_dir, rollout_id=candidate_id, group_index=task_index
    )
    guard.check("before")
    returncode = -1
    stdout = ""
    stderr = ""
    error: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="sr-rollout-config-", dir=os.environ.get("SELF_REWARD_LOCAL_TMP", "/tmp")) as raw_state:
            config_path = runtime._write_nanobot_config(
                workspace_dir=workspace,
                run_dir=Path(raw_state),
                adapter_base_url=adapter_base_url.rstrip("/"),
                session_id=session_id,
                turn_type="train",
                sampling_params=sampling,
            )
            proc = runtime._run_nanobot_turn(
                workspace_dir=workspace,
                config_path=config_path,
                session_id=session_id,
                turn_type="train",
                message=runtime._format_clawgym_task_prompt(task.user_query),
                log_dir=log_root,
                timeout=runtime._turn_timeout_seconds(False),
                env_overrides={
                    "NANOBOT_ROLLOUT_ID": str(candidate_id),
                    "NANOBOT_GROUP_INDEX": str(task_index),
                    "NANOBOT_TASK_ID": task.task_id,
                    "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
                },
            )
            returncode = int(proc.returncode)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            loop_error = runtime._recover_training_tool_loop_error(log_root / session_id)
            if loop_error:
                returncode = -2
                error = loop_error
            elif returncode != 0:
                error = (stderr or stdout or "nanobot failed")[-2000:]
    except runtime.raw_source_integrity.RawSourceIntegrityError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stdout = str(getattr(exc, "output", "") or "")[-1_000_000:]
        stderr = str(getattr(exc, "stderr", "") or "")[-1_000_000:]
    finally:
        guard.check("after")
    (candidate_root / "turn.stdout").write_text(stdout, encoding="utf-8")
    (candidate_root / "turn.stderr").write_text(stderr, encoding="utf-8")
    valid = returncode == 0 and error is None
    record = {
        "candidate_id": candidate_id,
        "session_id": session_id,
        "workspace": str(workspace),
        "log_dir": str(log_root),
        "final_message": _final_message(workspace, session_id, log_root),
        "valid": valid,
        "status": "completed" if valid else "failed",
        "returncode": returncode,
        "duration_sec": time.monotonic() - started,
        "error": error,
        "training_equivalent_reward_on_failure": 0.0,
        "eligible_for_self_reward": valid,
    }
    atomic_json(candidate_root / "candidate.json", record)
    return record


def rollout_group(
    *,
    project_root: Path,
    task_dir: Path,
    task_index: int,
    adapter_base_url: str,
    model_id: str,
    api_key_file: Path,
    run_root: Path,
    sampling: dict[str, Any],
) -> list[dict[str, Any]]:
    runtime, tasks, _ws, _rc = import_native(project_root)
    from clawgym_nanobot import polar_workspace

    task = tasks.load_task(task_dir)
    run_root.mkdir(parents=True, exist_ok=False)
    saved = {name: os.environ.get(name) for name in ("CLAWGYM_ALLOW_NANOBOT_MODEL_ID_OVERRIDE", "NANOBOT_MODEL_ID", "NANOBOT_OPENAI_API_KEY")}
    os.environ["CLAWGYM_ALLOW_NANOBOT_MODEL_ID_OVERRIDE"] = "1"
    os.environ["NANOBOT_MODEL_ID"] = model_id
    os.environ["NANOBOT_OPENAI_API_KEY"] = read_secret(api_key_file)
    try:
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="rollout") as pool:
            futures = [
                pool.submit(
                    _one_rollout,
                    runtime=runtime,
                    polar_workspace=polar_workspace,
                    task=task,
                    task_index=task_index,
                    candidate_id=candidate_id,
                    adapter_base_url=adapter_base_url,
                    run_root=run_root,
                    sampling=sampling,
                )
                for candidate_id in range(8)
            ]
            rows = [future.result() for future in futures]
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    rows.sort(key=lambda row: row["candidate_id"])
    atomic_json(run_root / "candidates.json", rows)
    return rows
