from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agent import AgentSpec, run_rubric_agent, run_score_agent
from .common import atomic_json, read_json
from .contracts import load_rubric
from .metrics import aggregate_self_scores, group_metrics
from .native import build_rubric_workspace, build_ws_score_workspace, resolve_rubric_prompt_path
from .researchclaw_hybrid import run_researchclaw_hybrid_self_score


RUBRIC_CONTRACT = "truth_grounded_native_root_v5"


def _accepted_slots_for_rubric(
    reward_root: Path,
    active_rubric_path: Path,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Collect successful score slots from earlier recovery attempts.

    Recovery is allowed only when the exact same accepted rubric file was used.
    New attempts can then rerun only missing candidate/repeat slots while keeping
    every model-facing scoring call identical to the native judge pipeline.
    """
    active = str(active_rubric_path.resolve())
    result_paths = sorted(
        (reward_root / "resume_attempts").glob("reward-resume-*/group_result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    accepted: dict[tuple[int, int], dict[str, Any]] = {}
    for result_path in result_paths:
        try:
            result = read_json(result_path)
        except (OSError, ValueError):
            continue
        if result.get("rubric_contract") != RUBRIC_CONTRACT:
            continue
        if result.get("dataset") == "wsclaw" and result.get("score_contract") != "wsclaw_native_judge_v2":
            continue
        source = result.get("reused_rubric_path")
        if not source:
            source = (result.get("reward_resume") or {}).get("reused_rubric_path")
        if not source and result.get("reused_rubric") is False:
            generated = result_path.parent / "rubric_agent" / "workspace" / "rubric.json"
            if generated.is_file():
                source = str(generated)
        if not source or str(Path(source).resolve()) != active:
            continue
        for slot in result.get("self_reward_slots") or []:
            candidate_id = slot.get("candidate_id")
            repeat = slot.get("repeat")
            value = slot.get("self_reward")
            if (
                slot.get("status") == "accepted"
                and isinstance(candidate_id, int)
                and 1 <= candidate_id <= 7
                and isinstance(repeat, int)
                and 1 <= repeat <= 5
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                key = (candidate_id, repeat)
                if key not in accepted:
                    accepted[key] = {
                        **slot,
                        "reused_from_attempt": str(result_path.parent),
                    }
    return accepted


def _latest_accepted_rubric(reward_root: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest strictly accepted rubric for this persisted group."""
    workspaces = [reward_root / "rubric_agent" / "workspace"]
    workspaces.extend(
        path / "rubric_agent" / "workspace"
        for path in (reward_root / "resume_attempts").glob("reward-resume-*")
    )
    workspaces.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    for workspace in workspaces:
        status_path = workspace.parent / "run" / "status.json"
        manifest_path = workspace / "WORKSPACE_MANIFEST.json"
        rubric_path = workspace / "rubric.json"
        if not (status_path.is_file() and manifest_path.is_file() and rubric_path.is_file()):
            continue
        try:
            status = read_json(status_path)
            manifest = read_json(manifest_path)
            if status.get("status") != "accepted":
                continue
            current_contract = (
                "wsclaw_full_workspace_v1" if manifest.get("dataset") == "wsclaw"
                else RUBRIC_CONTRACT
            )
            if manifest.get("rubric_contract") != current_contract:
                continue
            return rubric_path, load_rubric(rubric_path)
        except (OSError, ValueError):
            continue
    return None


def _failed(candidate_id: int, stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def inspect_reusable_group(group_root: Path) -> dict[str, Any]:
    """Validate the persisted rollout and GT branches without making model calls."""
    candidates_path = group_root / "rollout" / "candidates.json"
    if not candidates_path.is_file():
        raise FileNotFoundError(f"missing persisted rollouts: {candidates_path}")
    candidates = read_json(candidates_path)
    if not isinstance(candidates, list) or len(candidates) != 8:
        raise ValueError("persisted group must contain exactly eight candidates")
    if [row.get("candidate_id") for row in candidates] != list(range(8)):
        raise ValueError("persisted candidates must be ordered ids 0..7")
    for candidate in candidates:
        if not candidate.get("valid"):
            # Training assigns invalid/timeout rollouts a fixed zero and never
            # invokes self scoring for them, so no persisted workspace exists.
            continue
        workspace = Path(str(candidate.get("workspace") or ""))
        if not workspace.is_dir():
            raise FileNotFoundError(
                f"candidate-{candidate['candidate_id']} workspace is unavailable: {workspace}"
            )

    gt_results: list[dict[str, Any]] = []
    for candidate_id in range(8):
        accepted = group_root / "reward" / "ground_truth" / f"candidate-{candidate_id}" / "accepted.json"
        if not accepted.is_file():
            raise FileNotFoundError(f"missing persisted GT result: {accepted}")
        result = read_json(accepted)
        if int(result.get("candidate_id", -1)) != candidate_id:
            raise ValueError(f"GT candidate id mismatch in {accepted}")
        gt_results.append(result)

    anchor_gt = gt_results[0]
    if not candidates[0].get("valid"):
        raise ValueError("persisted anchor rollout is invalid")
    if anchor_gt.get("gt_reward") is None:
        raise ValueError("persisted anchor GT has no numeric reward")
    return {
        "group_root": str(group_root),
        "candidate_count": 8,
        "valid_candidates": sum(bool(row.get("valid")) for row in candidates),
        "gt_count": len(gt_results),
        "anchor_gt_reward": float(anchor_gt["gt_reward"]),
        "anchor_feedback_present": bool(str(anchor_gt.get("user_feedback") or "").strip()),
        "candidates": candidates,
        "ground_truth": gt_results,
    }


def resume_existing_reward(
    *,
    project_root: Path,
    dataset: str,
    model: str,
    task_dir: Path,
    group_root: Path,
    rubric_spec: AgentSpec,
    score_spec: AgentSpec,
    rubric_prompt_path: Path,
    rubric_schema_path: Path,
    score_appendix_path: Path,
    scoring_concurrency: int = 32,
    self_reward_repeats: int = 3,
    request_semaphore: threading.Semaphore | None = None,
) -> dict[str, Any]:
    """Rerun only rubric generation and self scoring from persisted inputs.

    Rollout generation and all GT judgements, including the anchor judgement,
    are deliberately read-only dependencies of this operation.
    """
    if self_reward_repeats not in (3, 5):
        raise ValueError("self_reward_repeats must be 3 or 5")
    rubric_prompt_path = resolve_rubric_prompt_path(rubric_prompt_path, dataset)
    inspected = inspect_reusable_group(group_root)
    candidates = inspected.pop("candidates")
    gt_results = inspected.pop("ground_truth")
    anchor_gt = gt_results[0]
    reward_root = group_root / "reward"
    attempt_name = time.strftime("reward-resume-%Y%m%d-%H%M%S", time.gmtime())
    attempt_root = reward_root / "resume_attempts" / attempt_name
    lock_path = reward_root / ".reward-resume.lock"
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(lock_fd, f"pid={os.getpid()} attempt={attempt_name}\n".encode())
        attempt_root.mkdir(parents=True, exist_ok=False)
        atomic_json(
            attempt_root / "reuse_manifest.json",
            {
                **inspected,
                "dataset": dataset,
                "model": model,
                "task_dir": str(task_dir),
                "reused_rollouts": True,
                "reused_all_gt": True,
                "anchor_gt_rerun": False,
                "rollout_rerun": False,
            },
        )
        previous = reward_root / "group_result.json"
        if previous.is_file():
            shutil.copy2(previous, attempt_root / "previous_group_result.json")

        group_mask = 1
        rubric_error: str | None = None
        rubric: dict[str, Any] | None = None
        reused_rubric = False
        reused_rubric_path: str | None = None
        rubric_workspace = attempt_root / "rubric_agent" / "workspace"
        rubric_run = attempt_root / "rubric_agent" / "run"
        active_rubric_path = rubric_workspace / "rubric.json"
        try:
            accepted = _latest_accepted_rubric(reward_root)
            if accepted is not None:
                active_rubric_path, rubric = accepted
                reused_rubric = True
                reused_rubric_path = str(active_rubric_path)
                atomic_json(
                    attempt_root / "rubric_agent" / "reused.json",
                    {"source": reused_rubric_path, "rubric_contract": RUBRIC_CONTRACT},
                )
            else:
                build_rubric_workspace(
                    project_root=project_root,
                    dataset=dataset,
                    destination=rubric_workspace,
                    task_dir=task_dir,
                    anchor_workspace=Path(candidates[0]["workspace"]),
                    anchor_final_message=str(candidates[0].get("final_message") or ""),
                    anchor_feedback=str(anchor_gt.get("user_feedback") or ""),
                    anchor_score=float(anchor_gt["gt_reward"]),
                    rubric_prompt_path=rubric_prompt_path,
                    rubric_schema_path=rubric_schema_path,
                    trajectory_path=None,
                )
                if request_semaphore is None:
                    run_rubric_agent(
                        project_root=project_root,
                        workspace=rubric_workspace,
                        run_dir=rubric_run,
                        session_id=f"rubric-resume-{model}-{task_dir.name}-{attempt_name}",
                        spec=rubric_spec,
                        prompt_path=rubric_prompt_path,
                    )
                else:
                    with request_semaphore:
                        run_rubric_agent(
                            project_root=project_root,
                            workspace=rubric_workspace,
                            run_dir=rubric_run,
                            session_id=f"rubric-resume-{model}-{task_dir.name}-{attempt_name}",
                            spec=rubric_spec,
                            prompt_path=rubric_prompt_path,
                        )
                rubric = load_rubric(active_rubric_path)
        except Exception as exc:
            group_mask = 0
            rubric_error = f"{type(exc).__name__}: {exc}"
            atomic_json(
                attempt_root / "rubric_agent" / "failed.json",
                {"error": rubric_error, "traceback": traceback.format_exc()},
            )

        reused_score_slots: dict[tuple[int, int], dict[str, Any]] = {}
        if group_mask and rubric is not None:
            reused_score_slots = {
                key: value
                for key, value in _accepted_slots_for_rubric(
                    reward_root, active_rubric_path
                ).items()
                if key[1] <= self_reward_repeats
            }
        score_slots: list[dict[str, Any]] = list(reused_score_slots.values())
        if group_mask and rubric is not None:
            prepared: list[dict[str, Any]] = []
            for candidate in candidates[1:]:
                candidate_id = int(candidate["candidate_id"])
                if not candidate.get("valid"):
                    score_slots.extend(
                        {
                            "candidate_id": candidate_id,
                            "repeat": repeat,
                            "status": "rollout_invalid_zero",
                            "self_reward": 0.0,
                            "training_equivalent_reward": 0.0,
                        }
                        for repeat in range(1, self_reward_repeats + 1)
                    )
                    continue
                for repeat in range(1, self_reward_repeats + 1):
                    if (candidate_id, repeat) in reused_score_slots:
                        continue
                    slot_id = f"{model}/{task_dir.name}/candidate-{candidate_id}/repeat-{repeat}"
                    slot = attempt_root / "self_reward" / f"candidate-{candidate_id}" / f"repeat-{repeat}"
                    if dataset == "wsclaw":
                        prepared_dir = slot / "prepared"
                        manifest = build_ws_score_workspace(
                            project_root=project_root,
                            task_dir=task_dir,
                            candidate_workspace=Path(candidate["workspace"]),
                            rubric=rubric,
                            destination=prepared_dir,
                            score_appendix_path=score_appendix_path,
                        )
                    elif dataset == "researchclaw":
                        manifest = {
                            "dataset": "researchclaw",
                            "candidate_workspace": candidate["workspace"],
                            "backend": "hybrid_agentic",
                            "rubric_source": "generated_self_rubric",
                            "hidden_reference_answer_included": False,
                        }
                    else:
                        raise ValueError(f"unsupported dataset: {dataset}")
                    prepared.append(
                        {
                            "candidate_id": candidate_id,
                            "repeat": repeat,
                            "slot_id": slot_id,
                            "slot": slot,
                            "manifest": manifest,
                            "prompt_path": (
                                slot / "prepared" / "SCORING_PROMPT.md"
                                if dataset == "wsclaw"
                                else None
                            ),
                        }
                    )

            def score_job(item: dict[str, Any]) -> dict[str, Any]:
                try:
                    if request_semaphore is not None:
                        request_semaphore.acquire()
                    try:
                        if dataset == "wsclaw":
                            result = run_score_agent(
                                project_root=project_root,
                                dataset=dataset,
                                workspace=Path(item["manifest"]["workspace"]),
                                prompt_path=item["prompt_path"],
                                rubric_path=active_rubric_path,
                                result_relpath=str(item["manifest"]["result_path"]),
                                run_dir=item["slot"] / "run",
                                session_id=item["slot_id"].replace("/", "-"),
                                slot_id=item["slot_id"],
                                spec=score_spec,
                            )
                            self_reward = result["value"]["weighted_score"]
                            canonical_result = result["value"]
                            workspace = item["manifest"]["workspace"]
                            prompt_path = str(item["prompt_path"])
                        else:
                            result = run_researchclaw_hybrid_self_score(
                                project_root=project_root,
                                task_dir=task_dir,
                                candidate_workspace=Path(item["manifest"]["candidate_workspace"]),
                                rubric_path=active_rubric_path,
                                run_dir=item["slot"] / "run",
                                slot_id=item["slot_id"],
                                spec=score_spec,
                            )
                            self_reward = result["self_reward"]
                            canonical_result = result
                            workspace = str(item["slot"] / "run" / "native_run")
                            prompt_path = str(
                                item["slot"] / "run" / "native_run" / "researchclaw_judge_triage_request.json"
                            )
                    finally:
                        if request_semaphore is not None:
                            request_semaphore.release()
                    return {
                        "candidate_id": item["candidate_id"],
                        "repeat": item["repeat"],
                        "status": "accepted",
                        "self_reward": self_reward,
                        "canonical_result": canonical_result,
                        "workspace": workspace,
                        "prompt_path": prompt_path,
                    }
                except Exception as exc:
                    return _failed(
                        item["candidate_id"], f"self_reward_repeat_{item['repeat']}", exc
                    ) | {"repeat": item["repeat"], "self_reward": None}

            with ThreadPoolExecutor(
                max_workers=min(scoring_concurrency, max(1, len(prepared))),
                thread_name_prefix="self-score-resume",
            ) as score_pool:
                futures = [score_pool.submit(score_job, item) for item in prepared]
                for future in as_completed(futures):
                    score_slots.append(future.result())

        score_slots.sort(key=lambda row: (int(row["candidate_id"]), int(row.get("repeat") or 0)))
        gt_by_id = {int(row["candidate_id"]): row for row in gt_results}
        comparison_rows = []
        for candidate_id in range(1, 8):
            slots = [row for row in score_slots if int(row["candidate_id"]) == candidate_id]
            accepted_scores = [
                float(row["self_reward"])
                for row in slots
                if row.get("self_reward") is not None
            ]
            aggregate, aggregation = aggregate_self_scores(
                accepted_scores, planned_repeats=self_reward_repeats
            )
            comparison_rows.append(
                {
                    "candidate_id": candidate_id,
                    "gt_reward": gt_by_id.get(candidate_id, {}).get("gt_reward"),
                    "self_scores": accepted_scores,
                    "self_reward": aggregate,
                    "aggregation": aggregation,
                    "training_eligible": aggregate is not None,
                    "planned_repeats": self_reward_repeats,
                    "accepted_repeats": len(accepted_scores),
                }
            )
        summary = {
            "dataset": dataset,
            "model": model,
            "task": task_dir.name,
            "group_mask": group_mask,
            "rubric_error": rubric_error,
            "reused_rubric": reused_rubric,
            "reused_rubric_path": reused_rubric_path,
            "reused_self_score_slots": len(reused_score_slots),
            "new_self_score_slots": len(score_slots) - len(reused_score_slots),
            "self_rubric_source": "generated_from_anchor_and_single_feedback",
            "self_reward_hidden_reference_answer": False,
            "workspace_materialization": ("native_gt_link_clean_tree_v1" if dataset == "wsclaw" else "bounded_isolated_copy_v1"),
            "score_contract": "wsclaw_native_judge_v2" if dataset == "wsclaw" else "researchclaw_native_hybrid",
            "rubric_contract": RUBRIC_CONTRACT,
            "anchor_gt": anchor_gt,
            "ground_truth": gt_results,
            "self_reward_slots": score_slots,
            "self_reward_repeats": self_reward_repeats,
            "comparison_rows": comparison_rows,
            "metrics": group_metrics(comparison_rows),
            "reward_resume": {
                "attempt": attempt_name,
                "attempt_root": str(attempt_root),
                "reused_rollouts": True,
                "reused_all_gt": True,
                "rollout_rerun": False,
                "anchor_gt_rerun": False,
                "reused_rubric": reused_rubric,
                "reused_rubric_path": reused_rubric_path,
                "workspace_materialization": ("native_gt_link_clean_tree_v1" if dataset == "wsclaw" else "bounded_isolated_copy_v1"),
                "score_contract": "wsclaw_native_judge_v2" if dataset == "wsclaw" else "researchclaw_native_hybrid",
                "rubric_contract": RUBRIC_CONTRACT,
            },
        }
        atomic_json(attempt_root / "group_result.json", summary)
        atomic_json(reward_root / "group_result.json", summary)
        atomic_json(
            reward_root / "resume_latest.json",
            {"attempt": attempt_name, "attempt_root": str(attempt_root), "group_mask": group_mask},
        )
        return summary
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
