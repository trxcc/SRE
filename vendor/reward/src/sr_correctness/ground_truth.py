from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .common import atomic_json, copy_clean_tree, link_clean_tree, read_json, read_secret
from .native import import_native


def score_wsclaw_gt(
    *,
    project_root: Path,
    task_dir: Path,
    candidate: dict[str, Any],
    case_dir: Path,
    remote_judge_urls: list[str],
    remote_judge_api_key_file: Path,
    workers_per_node: int,
    timeout_sec: int,
) -> dict[str, Any]:
    _runtime, _tasks, ws, _rc = import_native(project_root)
    case_dir.mkdir(parents=True, exist_ok=False)
    solver = case_dir / "solver_workspace"
    link_clean_tree(Path(candidate["workspace"]), solver)
    entry = read_json(task_dir / "data_entry.json")
    outputs = ws.output_paths(task_dir)
    judge_workspace, _artifacts = ws.prepare_judge_workspace(
        task_dir=task_dir,
        workspace=solver,
        outputs=outputs,
        case_dir=case_dir,
        entry=entry,
    )
    pool = ws.RemoteJudgePool(
        remote_judge_urls,
        workers_per_node=workers_per_node,
        timeout_sec=timeout_sec,
        api_key=read_secret(remote_judge_api_key_file),
    )
    result = pool.submit(judge_workspace, task_dir, case_dir, timeout_sec=timeout_sec)
    feedback = str(result.get("user_feedback") or result.get("feedback") or "").strip()
    accepted = {
        "candidate_id": candidate["candidate_id"],
        "gt_reward": float(result["score"]),
        "user_feedback": feedback,
        "native_result": result,
        "workspace": str(judge_workspace),
        "prompt_path": str(case_dir / "judge_prompt.txt"),
    }
    atomic_json(case_dir / "accepted.json", accepted)
    return accepted


def score_researchclaw_gt(
    *, project_root: Path, task_dir: Path, candidate: dict[str, Any], case_dir: Path
) -> dict[str, Any]:
    _runtime, tasks, _ws, rc = import_native(project_root)
    case_dir.mkdir(parents=True, exist_ok=False)
    solver = case_dir / "solver_workspace"
    # ResearchClaw's native reward rejects output symlinks that resolve outside
    # the solver workspace. Materialize this small workspace so the native
    # safe-path checks can inspect the actual candidate outputs.
    copy_clean_tree(Path(candidate["workspace"]), solver)
    task = tasks.load_task(task_dir)
    tasks.copy_reward_assets(task, solver)
    score = rc.run(task, solver, case_dir)
    payload = read_json(case_dir / "researchclaw_reward.json")
    judge = payload.get("llmjudge_reward") or {}
    parsed = judge.get("parsed") or {}
    privileged_feedback = payload.get("privileged_feedback") or {}
    feedback = str(
        payload.get("user_feedback")
        or privileged_feedback.get("text")
        or parsed.get("notes")
        or judge.get("text")
        or ""
    ).strip()
    accepted = {
        "candidate_id": candidate["candidate_id"],
        "gt_reward": float(score),
        "user_feedback": feedback,
        "native_result": payload,
        "workspace": str(judge.get("judge_workspace") or ""),
        "prompt_path": str(judge.get("judge_request_path") or ""),
    }
    atomic_json(case_dir / "accepted.json", accepted)
    return accepted
