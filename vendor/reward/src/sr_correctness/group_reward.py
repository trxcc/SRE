from __future__ import annotations

import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Callable

from .agent import AgentSpec, run_rubric_agent, run_score_agent
from .common import atomic_json
from .contracts import load_rubric
from .ground_truth import score_researchclaw_gt, score_wsclaw_gt
from .metrics import aggregate_self_scores, group_metrics
from .native import build_rubric_workspace, build_ws_score_workspace, resolve_rubric_prompt_path
from .researchclaw_hybrid import run_researchclaw_hybrid_self_score


def _failed(candidate_id: int, stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def evaluate_group(
    *,
    project_root: Path,
    dataset: str,
    model: str,
    task_dir: Path,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    rubric_spec: AgentSpec,
    score_spec: AgentSpec,
    rubric_prompt_path: Path,
    rubric_schema_path: Path,
    score_appendix_path: Path,
    ws_gt_urls: list[str] | None = None,
    ws_gt_api_key_file: Path | None = None,
    ws_gt_workers_per_node: int = 8,
    gt_timeout_sec: int = 1800,
    gt_concurrency: int = 4,
    scoring_concurrency: int = 64,
    self_reward_repeats: int = 3,
    gt_all_candidates: bool = False,
    agent_semaphore: BoundedSemaphore | None = None,
    training_ready_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    group_size = len(candidates)
    if group_size != 8 or [row.get("candidate_id") for row in candidates] != list(range(8)):
        raise ValueError("group must contain ordered candidate ids 0..7 (1 GT anchor + 7 SR)")
    if self_reward_repeats not in (3, 5):
        raise ValueError("self_reward_repeats must be 3 or 5")
    rubric_prompt_path = resolve_rubric_prompt_path(rubric_prompt_path, dataset)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_dir / "group_input.json",
        {
            "dataset": dataset,
            "model": model,
            "task_dir": str(task_dir),
            "candidate_count": group_size,
            "candidate_sources": [row["workspace"] for row in candidates],
        },
    )

    def gt_job(candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = int(candidate["candidate_id"])
        case = output_dir / "ground_truth" / f"candidate-{candidate_id}"
        if not candidate.get("valid"):
            result = {
                "candidate_id": candidate_id,
                "gt_reward": 0.0,
                "user_feedback": "",
                "status": "rollout_invalid",
                "eligible_for_correlation": False,
            }
            case.mkdir(parents=True, exist_ok=False)
            atomic_json(case / "accepted.json", result)
            return result
        if dataset == "wsclaw":
            if not ws_gt_urls or ws_gt_api_key_file is None:
                raise ValueError("WSClaw GT requires native remote judge URLs and API-key file")
            return score_wsclaw_gt(
                project_root=project_root,
                task_dir=task_dir,
                candidate=candidate,
                case_dir=case,
                remote_judge_urls=ws_gt_urls,
                remote_judge_api_key_file=ws_gt_api_key_file,
                workers_per_node=ws_gt_workers_per_node,
                timeout_sec=gt_timeout_sec,
            )
        if dataset == "researchclaw":
            return score_researchclaw_gt(
                project_root=project_root,
                task_dir=task_dir,
                candidate=candidate,
                case_dir=case,
            )
        raise ValueError(f"unsupported dataset: {dataset}")

    # Branch A starts all eight native GT judges immediately. Branch B waits only
    # for candidate 0, because its feedback and workspace are used to generate the
    # self rubric. Candidate scoring never receives the anchor or its feedback.
    with ThreadPoolExecutor(max_workers=max(1, min(8, gt_concurrency)), thread_name_prefix="gt") as gt_pool:
        gt_candidates = candidates if gt_all_candidates else candidates[:1]
        gt_futures: dict[int, Future[dict[str, Any]]] = {
            int(candidate["candidate_id"]): gt_pool.submit(gt_job, candidate)
            for candidate in gt_candidates
        }
        group_mask = 1
        rubric_error: str | None = None
        rubric: dict[str, Any] | None = None
        rubric_workspace = output_dir / "rubric_agent" / "workspace"
        rubric_run = output_dir / "rubric_agent" / "run"
        anchor_gt: dict[str, Any] | None = None
        try:
            anchor_gt = gt_futures[0].result()
        except Exception as exc:
            anchor_gt = _failed(0, "ground_truth", exc)
        if not candidates[0].get("valid") or anchor_gt.get("gt_reward") is None:
            group_mask = 0
            rubric_error = "anchor rollout or anchor GT judgement is invalid"
        else:
            try:
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
                if agent_semaphore is None:
                    run_rubric_agent(
                        project_root=project_root,
                        workspace=rubric_workspace,
                        run_dir=rubric_run,
                        session_id=f"rubric-{model}-{task_dir.name}",
                        spec=rubric_spec,
                        prompt_path=rubric_prompt_path,
                    )
                else:
                    with agent_semaphore:
                        run_rubric_agent(
                            project_root=project_root,
                            workspace=rubric_workspace,
                            run_dir=rubric_run,
                            session_id=f"rubric-{model}-{task_dir.name}",
                            spec=rubric_spec,
                            prompt_path=rubric_prompt_path,
                        )
                rubric = load_rubric(rubric_workspace / "rubric.json")
            except Exception as exc:
                group_mask = 0
                rubric_error = f"{type(exc).__name__}: {exc}"
                atomic_json(
                    output_dir / "rubric_agent" / "failed.json",
                    {"error": rubric_error, "traceback": traceback.format_exc()},
                )

        score_slots: list[dict[str, Any]] = []
        if group_mask and rubric is not None:
            prepared: list[dict[str, Any]] = []
            for candidate in candidates[1:]:
                candidate_id = int(candidate["candidate_id"])
                if not candidate.get("valid"):
                    score_slots.extend(
                        {
                            "candidate_id": candidate_id,
                            "repeat": repeat,
                            "status": "skipped_rollout_invalid",
                            "self_reward": None,
                            "training_equivalent_reward": 0.0,
                        }
                        for repeat in range(1, self_reward_repeats + 1)
                    )
                    continue
                for repeat in range(1, self_reward_repeats + 1):
                    slot_id = f"{model}/{task_dir.name}/candidate-{candidate_id}/repeat-{repeat}"
                    slot = output_dir / "self_reward" / f"candidate-{candidate_id}" / f"repeat-{repeat}"
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
                    else:
                        manifest = {
                            "dataset": "researchclaw",
                            "candidate_workspace": candidate["workspace"],
                            "backend": "hybrid_agentic",
                            "rubric_source": "generated_self_rubric",
                            "hidden_reference_answer_included": False,
                        }
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
                    def run_one() -> tuple[dict[str, Any], float, dict[str, Any], str, str]:
                        if dataset == "wsclaw":
                            result = run_score_agent(
                                project_root=project_root,
                                dataset=dataset,
                                workspace=Path(item["manifest"]["workspace"]),
                                prompt_path=item["prompt_path"],
                                rubric_path=rubric_workspace / "rubric.json",
                                result_relpath=str(item["manifest"]["result_path"]),
                                run_dir=item["slot"] / "run",
                                session_id=item["slot_id"].replace("/", "-"),
                                slot_id=item["slot_id"],
                                spec=score_spec,
                            )
                            return result, result["value"]["weighted_score"], result["value"], item["manifest"]["workspace"], str(item["prompt_path"])
                        result = run_researchclaw_hybrid_self_score(
                            project_root=project_root,
                            task_dir=task_dir,
                            candidate_workspace=Path(item["manifest"]["candidate_workspace"]),
                            rubric_path=rubric_workspace / "rubric.json",
                            run_dir=item["slot"] / "run",
                            slot_id=item["slot_id"],
                            spec=score_spec,
                        )
                        return result, result["self_reward"], result, str(item["slot"] / "run" / "native_run"), str(item["slot"] / "run" / "native_run" / "researchclaw_judge_triage_request.json")

                    if agent_semaphore is None:
                        _result, self_reward, canonical_result, workspace, prompt_path = run_one()
                    else:
                        with agent_semaphore:
                            _result, self_reward, canonical_result, workspace, prompt_path = run_one()
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
                    return _failed(item["candidate_id"], f"self_reward_repeat_{item['repeat']}", exc) | {
                        "repeat": item["repeat"], "self_reward": None
                    }

            with ThreadPoolExecutor(
                max_workers=min(scoring_concurrency, max(1, len(prepared))),
                thread_name_prefix="self-score",
            ) as score_pool:
                futures = [score_pool.submit(score_job, item) for item in prepared]
                for future in as_completed(futures):
                    score_slots.append(future.result())

        # Training depends on the anchor/rubric and the three planned self
        # scores. Non-anchor GT is diagnostic and must not hold the response.
        # Keep its executor alive so every original diagnostic is still saved.
        if training_ready_callback is not None:
            ready_gt = []
            pending_gt = []
            for candidate_id, future in gt_futures.items():
                if candidate_id == 0 and anchor_gt is not None:
                    ready_gt.append(anchor_gt)
                elif future.done():
                    try:
                        ready_gt.append(future.result())
                    except Exception as exc:
                        ready_gt.append(_failed(candidate_id, "ground_truth", exc))
                else:
                    pending_gt.append(candidate_id)
                    ready_gt.append({
                        "candidate_id": candidate_id, "gt_reward": None,
                        "status": "diagnostic_pending",
                        "eligible_for_correlation": False,
                    })
            training_summary = _summarize_group(group_size=group_size, 
                dataset=dataset, model=model, task_dir=task_dir,
                group_mask=group_mask, rubric_error=rubric_error,
                anchor_gt=anchor_gt, gt_results=ready_gt, score_slots=score_slots,
                gt_all_candidates=gt_all_candidates,
                self_reward_repeats=self_reward_repeats,
            )
            training_summary["diagnostic_pending_candidate_ids"] = pending_gt
            training_summary["scoring_failure_policy"] = "single_attempt_mask_v1"
            atomic_json(output_dir / "training_result.json", training_summary)
            training_ready_callback(training_summary)

        gt_results: list[dict[str, Any]] = []
        for candidate_id, future in gt_futures.items():
            if candidate_id == 0 and anchor_gt is not None:
                gt_results.append(anchor_gt)
                continue
            try:
                gt_results.append(future.result())
            except Exception as exc:
                gt_results.append(_failed(candidate_id, "ground_truth", exc))

    summary = _summarize_group(group_size=group_size, dataset=dataset, model=model, task_dir=task_dir, group_mask=group_mask, rubric_error=rubric_error, anchor_gt=anchor_gt, gt_results=gt_results, score_slots=score_slots, gt_all_candidates=gt_all_candidates, self_reward_repeats=self_reward_repeats)
    atomic_json(output_dir / "group_result.json", summary)
    return summary

def _summarize_group(
    *, dataset, model, task_dir, group_mask, rubric_error, anchor_gt,
    gt_results, score_slots, gt_all_candidates, self_reward_repeats, group_size=8,
):
    gt_results.sort(key=lambda row: int(row["candidate_id"]))
    score_slots.sort(key=lambda row: (int(row["candidate_id"]), int(row.get("repeat") or 0)))
    comparison_rows = []
    gt_by_id = {int(row["candidate_id"]): row for row in gt_results}
    for candidate_id in range(1, group_size):
        slots = [row for row in score_slots if int(row["candidate_id"]) == candidate_id]
        accepted_scores = [float(row["self_reward"]) for row in slots if row.get("self_reward") is not None]
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
    metrics = group_metrics(comparison_rows)
    summary = {
        "dataset": dataset,
        "model": model,
        "task": task_dir.name,
        "group_mask": group_mask,
        "rubric_error": rubric_error,
        "self_rubric_source": "generated_from_anchor_and_single_feedback",
        "self_reward_hidden_reference_answer": False,
        "workspace_materialization": "bounded_isolated_input_layer_v2",
        "rubric_contract": "truth_grounded_input_layer_v6",
        "anchor_gt": anchor_gt,
        "ground_truth": gt_results,
        "gt_all_candidates": gt_all_candidates,
        "self_reward_repeats": self_reward_repeats,
        "self_reward_slots": score_slots,
        "comparison_rows": comparison_rows,
        "metrics": metrics,
    }
    return summary
