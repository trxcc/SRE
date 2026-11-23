from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any

from .agent import AgentSpec, run_rubric_agent, run_score_agent
from .common import atomic_json, read_json
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


class StreamingGroupEvaluation:
    """Evaluate candidates as their terminal rollout events arrive.

    Dependency graph:
      candidate 0 -> GT/user feedback -> rubric
      candidate i -> GT diagnostic
      (candidate i, rubric) -> independent self-score repeats

    ``closed.json`` is deliberately used only as the final response barrier.
    It does not gate any GT, rubric, or self-score submission.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        group_dir: Path,
        output_dir: Path,
        group: dict[str, Any],
        rubric_spec: AgentSpec,
        score_spec: AgentSpec,
        rubric_prompt_path: Path,
        rubric_schema_path: Path,
        score_appendix_path: Path,
        self_reward_repeats: int,
        gt_timeout_sec: int,
        gt_concurrency: int,
        scoring_concurrency: int,
        agent_semaphore: BoundedSemaphore | None,
        ws_gt_urls: list[str] | None = None,
        ws_gt_api_key_file: Path | None = None,
        ws_gt_workers_per_node: int = 8,
        gt_all_candidates: bool = False,
        poll_seconds: float = 0.1,
    ) -> None:
        self.project_root = project_root
        self.group_dir = group_dir
        self.output_dir = output_dir
        self.group = group
        self.dataset = str(group["profile"]["dataset"])
        self.model = str(group["profile"]["model"])
        self.group_size = int(group.get("candidate_count", 8))
        if self.group_size != 8:
            raise ValueError(f"unsupported streaming group size (expected 8 = 1 GT anchor + 7 SR): {self.group_size}")
        self.task_dir = Path(group["task_dir"])
        self.rubric_spec = rubric_spec
        self.score_spec = score_spec
        self.rubric_prompt_path = resolve_rubric_prompt_path(rubric_prompt_path, self.dataset)
        self.rubric_schema_path = rubric_schema_path
        self.score_appendix_path = score_appendix_path
        self.self_reward_repeats = int(self_reward_repeats)
        self.gt_timeout_sec = int(gt_timeout_sec)
        self.gt_concurrency = max(1, int(gt_concurrency))
        self.scoring_concurrency = max(1, int(scoring_concurrency))
        self.agent_semaphore = agent_semaphore
        self.ws_gt_urls = ws_gt_urls
        self.ws_gt_api_key_file = ws_gt_api_key_file
        self.ws_gt_workers_per_node = int(ws_gt_workers_per_node)
        self.gt_all_candidates = bool(gt_all_candidates)
        self.poll_seconds = float(poll_seconds)
        self._stage_lock = Lock()
        if self.self_reward_repeats not in (3, 5):
            raise ValueError("self_reward_repeats must be 3 or 5")

    def _stage(self, name: str, **payload: Any) -> None:
        path = self.output_dir / "stream-events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._stage_lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at": time.time(), "stage": name, **payload}, ensure_ascii=False) + "\n")

    def _gt(self, candidate: dict[str, Any]) -> dict[str, Any]:
        candidate_id = int(candidate["candidate_id"])
        self._stage("gt_started", candidate_id=candidate_id)
        case = self.output_dir / "ground_truth" / f"candidate-{candidate_id}"
        try:
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
            elif self.dataset == "researchclaw":
                result = score_researchclaw_gt(
                    project_root=self.project_root,
                    task_dir=self.task_dir,
                    candidate=candidate,
                    case_dir=case,
                )
            elif self.dataset == "wsclaw":
                if not self.ws_gt_urls or self.ws_gt_api_key_file is None:
                    raise ValueError("WSClaw GT requires native remote judge URLs and API-key file")
                result = score_wsclaw_gt(
                    project_root=self.project_root,
                    task_dir=self.task_dir,
                    candidate=candidate,
                    case_dir=case,
                    remote_judge_urls=self.ws_gt_urls,
                    remote_judge_api_key_file=self.ws_gt_api_key_file,
                    workers_per_node=self.ws_gt_workers_per_node,
                    timeout_sec=self.gt_timeout_sec,
                )
            else:
                raise ValueError(f"unsupported dataset: {self.dataset}")
            self._stage("gt_finished", candidate_id=candidate_id, status=result.get("status", "accepted"))
            return result
        except Exception as exc:
            self._stage("gt_failed", candidate_id=candidate_id, error=f"{type(exc).__name__}: {exc}")
            return _failed(candidate_id, "ground_truth", exc)

    def _rubric(self, anchor: dict[str, Any], anchor_gt: dict[str, Any]) -> dict[str, Any]:
        workspace = self.output_dir / "rubric_agent" / "workspace"
        run_dir = self.output_dir / "rubric_agent" / "run"
        build_rubric_workspace(
            project_root=self.project_root,
            dataset=self.dataset,
            destination=workspace,
            task_dir=self.task_dir,
            anchor_workspace=Path(anchor["workspace"]),
            anchor_final_message=str(anchor.get("final_message") or ""),
            anchor_feedback=str(anchor_gt.get("user_feedback") or ""),
            anchor_score=float(anchor_gt["gt_reward"]),
            rubric_prompt_path=self.rubric_prompt_path,
            rubric_schema_path=self.rubric_schema_path,
            trajectory_path=None,
        )
        def run() -> None:
            self._stage("rubric_started", candidate_id=0)
            run_rubric_agent(
                project_root=self.project_root,
                workspace=workspace,
                run_dir=run_dir,
                session_id=f"rubric-{self.model}-{self.task_dir.name}",
                spec=self.rubric_spec,
                prompt_path=self.rubric_prompt_path,
            )
        if self.agent_semaphore is None:
            run()
        else:
            with self.agent_semaphore:
                run()
        rubric = load_rubric(workspace / "rubric.json")
        self._stage("rubric_finished", candidate_id=0)
        return rubric

    def _self_score(self, candidate: dict[str, Any], repeat: int) -> dict[str, Any]:
        candidate_id = int(candidate["candidate_id"])
        slot_id = f"{self.model}/{self.task_dir.name}/candidate-{candidate_id}/repeat-{repeat}"
        slot = self.output_dir / "self_reward" / f"candidate-{candidate_id}" / f"repeat-{repeat}"

        def run() -> dict[str, Any]:
            # This timestamp is inside the global concurrency semaphore.  Queue
            # wait is not charged to the agent request timeout.
            self._stage("self_score_started", candidate_id=candidate_id, repeat=repeat)
            if self.dataset == "wsclaw":
                manifest = build_ws_score_workspace(
                    project_root=self.project_root,
                    task_dir=self.task_dir,
                    candidate_workspace=Path(candidate["workspace"]),
                    rubric=load_rubric(self.output_dir / "rubric_agent" / "workspace" / "rubric.json"),
                    destination=slot / "prepared",
                    score_appendix_path=self.score_appendix_path,
                )
                prompt_path = slot / "prepared" / "SCORING_PROMPT.md"
                result = run_score_agent(
                    project_root=self.project_root,
                    dataset=self.dataset,
                    workspace=Path(manifest["workspace"]),
                    prompt_path=prompt_path,
                    rubric_path=self.output_dir / "rubric_agent" / "workspace" / "rubric.json",
                    result_relpath=str(manifest["result_path"]),
                    run_dir=slot / "run",
                    session_id=slot_id.replace("/", "-"),
                    slot_id=slot_id,
                    spec=self.score_spec,
                )
                value = float(result["value"]["weighted_score"])
                workspace = manifest["workspace"]
                canonical = result["value"]
                prompt = str(prompt_path)
            else:
                result = run_researchclaw_hybrid_self_score(
                    project_root=self.project_root,
                    task_dir=self.task_dir,
                    candidate_workspace=Path(candidate["workspace"]),
                    rubric_path=self.output_dir / "rubric_agent" / "workspace" / "rubric.json",
                    run_dir=slot / "run",
                    slot_id=slot_id,
                    spec=self.score_spec,
                )
                value = float(result["self_reward"])
                workspace = str(slot / "run" / "native_run")
                canonical = result
                prompt = str(slot / "run" / "native_run" / "researchclaw_judge_triage_request.json")
            self._stage("self_score_finished", candidate_id=candidate_id, repeat=repeat)
            return {
                "candidate_id": candidate_id,
                "repeat": repeat,
                "status": "accepted",
                "self_reward": value,
                "canonical_result": canonical,
                "workspace": workspace,
                "prompt_path": prompt,
            }

        try:
            if self.agent_semaphore is None:
                return run()
            with self.agent_semaphore:
                return run()
        except Exception as exc:
            self._stage("self_score_failed", candidate_id=candidate_id, repeat=repeat, error=f"{type(exc).__name__}: {exc}")
            return _failed(candidate_id, f"self_reward_repeat_{repeat}", exc) | {
                "repeat": repeat,
                "self_reward": None,
            }

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(self.output_dir / "group_input.json", self.group)
        candidates: dict[int, dict[str, Any]] = {}
        gt_futures: dict[int, Future[dict[str, Any]]] = {}
        score_futures: dict[tuple[int, int], Future[dict[str, Any]]] = {}
        score_slots: list[dict[str, Any]] = []
        rubric_future: Future[dict[str, Any]] | None = None
        rubric: dict[str, Any] | None = None
        rubric_error: str | None = None
        group_mask = 1

        with ThreadPoolExecutor(max_workers=self.gt_concurrency, thread_name_prefix="stream-gt") as gt_pool, ThreadPoolExecutor(
            max_workers=self.scoring_concurrency, thread_name_prefix="stream-agent"
        ) as agent_pool:
            while True:
                for path in sorted((self.group_dir / "candidates").glob("*.json")):
                    candidate_id = int(path.stem)
                    if candidate_id in candidates:
                        continue
                    candidate = read_json(path)
                    if int(candidate.get("candidate_id", -1)) != candidate_id:
                        raise ValueError(f"candidate id/path mismatch: {path}")
                    candidates[candidate_id] = candidate
                    self._stage("candidate_seen", candidate_id=candidate_id, published_at=candidate.get("published_at"))
                    if candidate_id == 0 or self.gt_all_candidates:
                        gt_futures[candidate_id] = gt_pool.submit(self._gt, candidate)

                if 0 in candidates and rubric_future is None and rubric is None and rubric_error is None:
                    anchor_gt_future = gt_futures[0]
                    if anchor_gt_future.done():
                        anchor_gt = anchor_gt_future.result()
                        if not candidates[0].get("valid") or anchor_gt.get("gt_reward") is None:
                            group_mask = 0
                            rubric_error = "anchor rollout or anchor GT judgement is invalid"
                            self._stage("rubric_skipped", reason=rubric_error)
                        else:
                            # Anchor GT/user feedback is a strict predecessor of rubric.
                            rubric_future = agent_pool.submit(self._rubric, candidates[0], anchor_gt)

                if rubric_future is not None and rubric_future.done() and rubric is None and rubric_error is None:
                    try:
                        rubric = rubric_future.result()
                    except Exception as exc:
                        group_mask = 0
                        rubric_error = f"{type(exc).__name__}: {exc}"
                        atomic_json(self.output_dir / "rubric_agent" / "failed.json", {
                            "error": rubric_error, "traceback": traceback.format_exc(),
                        })
                        self._stage("rubric_failed", error=rubric_error)

                if rubric is not None:
                    for candidate_id, candidate in candidates.items():
                        if candidate_id == 0 or not candidate.get("valid"):
                            continue
                        for repeat in range(1, self.self_reward_repeats + 1):
                            key = (candidate_id, repeat)
                            if key not in score_futures:
                                score_futures[key] = agent_pool.submit(self._self_score, candidate, repeat)

                closed = (self.group_dir / "closed.json").exists()
                complete_candidates = set(candidates) == set(range(self.group_size))
                gt_done = complete_candidates and all(f.done() for f in gt_futures.values())
                planned_scores = sum(
                    self.self_reward_repeats
                    for candidate_id, candidate in candidates.items()
                    if candidate_id != 0 and candidate.get("valid")
                )
                score_done = (
                    rubric is not None
                    and len(score_futures) == planned_scores
                    and all(f.done() for f in score_futures.values())
                ) or (rubric_error is not None)
                if closed and complete_candidates and gt_done and score_done:
                    break
                time.sleep(self.poll_seconds)

            for key, future in sorted(score_futures.items()):
                score_slots.append(future.result())

        gt_results = [gt_futures[i].result() for i in sorted(gt_futures)]
        comparison_rows = []
        gt_by_id = {int(row["candidate_id"]): row for row in gt_results}
        for candidate_id in range(1, self.group_size):
            slots = [row for row in score_slots if int(row["candidate_id"]) == candidate_id]
            accepted = [float(row["self_reward"]) for row in slots if row.get("self_reward") is not None]
            aggregate, aggregation = aggregate_self_scores(accepted, planned_repeats=self.self_reward_repeats)
            comparison_rows.append({
                "candidate_id": candidate_id,
                "gt_reward": gt_by_id.get(candidate_id, {}).get("gt_reward"),
                "self_scores": accepted,
                "self_reward": aggregate,
                "aggregation": aggregation,
                "training_eligible": group_mask == 1 and aggregate is not None,
                "planned_repeats": self.self_reward_repeats,
                "accepted_repeats": len(accepted),
            })
        summary = {
            "dataset": self.dataset,
            "model": self.model,
            "task": self.task_dir.name,
            "group_mask": group_mask,
            "rubric_error": rubric_error,
            "self_rubric_source": "generated_from_anchor_and_single_feedback",
            "self_reward_hidden_reference_answer": False,
            "ground_truth": gt_results,
            "gt_all_candidates": self.gt_all_candidates,
            "self_reward_repeats": self.self_reward_repeats,
            "self_reward_slots": score_slots,
            "comparison_rows": comparison_rows,
            "metrics": group_metrics(comparison_rows),
            "stream_protocol": self.group.get("protocol"),
        }
        summary["metrics"]["self_reward_coverage"] = sum(
            row["training_eligible"] for row in comparison_rows
        ) / 7.0
        atomic_json(self.output_dir / "group_result.json", summary)
        return summary
