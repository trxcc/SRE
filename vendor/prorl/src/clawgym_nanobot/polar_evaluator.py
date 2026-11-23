"""Polar evaluator for ClawGym's task-local code verifier."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import math
import os
import re
import shlex
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory
from polar.agent.presets.nanobot import runtime as nanobot_runtime

from . import researchclaw_reward, tasks, user_agent


_JUDGE_POOLS: dict[tuple[tuple[str, ...], int, str], Any] = {}
_JUDGE_POOLS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


class ClawGymCodeVerifierEvaluator(BaseTrajectoryEvaluator):
    def __init__(
        self,
        *,
        task_dir: str,
        workspace: str = "/polar/session/workspace",
        python_executable: str | None = None,
        timeout_seconds: float = 120.0,
        retain_latest_eval_logs: bool = False,
        latest_eval_nanobot_log_root: str | None = None,
        latest_eval_judge_log_root: str | None = None,
        latest_eval_log_state_root: str | None = None,
        log_root: str | None = None,
        retain_researchclaw_nanobot_logs: bool = False,
        **_unused: Any,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.workspace = workspace
        self.python_executable = python_executable
        self.timeout_seconds = float(timeout_seconds)
        self.retain_latest_eval_logs = bool(retain_latest_eval_logs)
        default_root = Path(log_root or "/tmp/polar-code-verifier")
        self.log_root = default_root
        self.latest_eval_nanobot_log_root = Path(
            latest_eval_nanobot_log_root
            or default_root.parent / "nanobot_logs"
        )
        self.latest_eval_judge_log_root = Path(
            latest_eval_judge_log_root
            or default_root.parent / "judge_nanobot_logs"
        )
        self.latest_eval_log_state_root = Path(
            latest_eval_log_state_root
            or default_root.parent / ".latest_eval_log_state"
        )
        self.retain_researchclaw_nanobot_logs = bool(
            retain_researchclaw_nanobot_logs
        )
        if self.retain_latest_eval_logs:
            _validate_latest_eval_log_roots(
                self.latest_eval_nanobot_log_root,
                self.latest_eval_judge_log_root,
                self.latest_eval_log_state_root,
            )

    async def evaluate(self, trajectory: Trajectory, **context: Any) -> EvalResult:
        runtime = context["runtime"]
        session_dir = Path(context["session_dir"])
        session_id = str(
            context.get("session_id") or context.get("task_id") or "session"
        )
        task_id = str(context.get("task_id") or "")
        eval_rollout_id = (
            _eval_rollout_id(task_id)
            if self.retain_latest_eval_logs
            else None
        )
        if eval_rollout_id is not None:
            await asyncio.to_thread(
                _prepare_latest_eval_log_roots,
                rollout_id=eval_rollout_id,
                nanobot_log_root=self.latest_eval_nanobot_log_root,
                judge_log_root=self.latest_eval_judge_log_root,
                state_root=self.latest_eval_log_state_root,
            )
        timeout_cap = context.get("timeout_seconds")
        timeout = self.timeout_seconds
        if timeout_cap is not None:
            timeout = max(1.0, min(timeout, float(timeout_cap)))

        # ResearchClaw uses the same Polar evaluator strategy as ordinary
        # ClawGym tasks, but its authoritative reward is a pure LLM judge.
        # Route before uploading/running reward/check.py so the code verifier
        # can never be mixed into a ResearchClaw score.  Reward assets are
        # copied only here, after the solver trajectory has completed.
        if _is_researchclaw_task(self.task_dir):
            return await self._evaluate_researchclaw(
                session_dir=session_dir,
                session_id=session_id,
                task_id=task_id,
                rollout_step=_trajectory_rollout_step(trajectory),
                eval_rollout_id=eval_rollout_id,
                timeout=timeout,
            )

        reward_dir = self.task_dir / "reward"
        check_path = reward_dir / "check.py"
        if not check_path.is_file():
            await self._publish_eval_logs(
                session_dir=session_dir,
                session_id=session_id,
                task_id=task_id,
                eval_rollout_id=eval_rollout_id,
            )
            return EvalResult(
                outcome_reward=0.0,
                metadata={"report": {"resolved": False, "error": f"missing verifier: {check_path}"}},
            )

        started = time.monotonic()
        await runtime.upload_dir(str(reward_dir), f"{self.workspace}/reward")
        command = _verifier_command(
            python_executable=self.python_executable,
            check_path=f"{self.workspace}/reward/check.py",
            workspace=self.workspace,
        )
        result = await runtime.exec(
            command,
            cwd=self.workspace,
            timeout_sec=timeout,
        )
        duration = time.monotonic() - started
        eval_log_dir = session_dir / "logs" / "eval"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        (eval_log_dir / "code_verifier.stdout.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (eval_log_dir / "code_verifier.stderr.log").write_text(
            result.stderr or "", encoding="utf-8"
        )

        score = 0.0
        parse_error: str | None = None
        if result.return_code == 0:
            try:
                score = _score_reward_payload(result.stdout or "")
            except Exception as exc:  # verifier output is task data, not trusted input
                parse_error = f"{type(exc).__name__}: {exc}"
        report = {
            "resolved": score >= 1.0,
            "code_verifier_score": score,
            "exit_code": result.return_code,
            "duration_seconds": duration,
        }
        if parse_error:
            report["error"] = parse_error
        await self._publish_eval_logs(
            session_dir=session_dir,
            session_id=session_id,
            task_id=task_id,
            eval_rollout_id=eval_rollout_id,
        )
        return EvalResult(
            outcome_reward=float(score),
            metadata={
                "code_verifier_score": float(score),
                "code_verifier_duration_seconds": duration,
                "report": report,
            },
        )

    async def _evaluate_researchclaw(
        self,
        *,
        session_dir: Path,
        session_id: str,
        task_id: str,
        rollout_step: int | None,
        eval_rollout_id: int | None,
        timeout: float,
    ) -> EvalResult:
        started = time.monotonic()
        task = tasks.load_task(self.task_dir)
        workspace_dir = session_dir / "workspace"
        run_dir = session_dir / "logs" / "eval" / "researchclaw"
        case_dir = self.log_root / _safe_case_name(session_id, task.task_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        case_dir.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any]
        try:
            if not workspace_dir.is_dir():
                raise FileNotFoundError(
                    f"ResearchClaw solver workspace not found: {workspace_dir}"
                )
            tasks.copy_reward_assets(task, workspace_dir)
            returned_score = await asyncio.wait_for(
                asyncio.to_thread(
                    researchclaw_reward.run,
                    task,
                    workspace_dir,
                    run_dir,
                ),
                timeout=timeout,
            )
            reward_path = run_dir / "researchclaw_reward.json"
            payload = json.loads(reward_path.read_text(encoding="utf-8"))
            score = _validate_pure_researchclaw_reward(payload, returned_score)
            code_score = float(payload["code_score"])
            duration = time.monotonic() - started
            metadata = {
                "researchclaw_score": score,
                "llmjudge_score": score,
                "requested_score_mode": "hybrid",
                "researchclaw_duration_seconds": duration,
                "researchclaw_reward": payload,
                "researchclaw_case_dir": str(case_dir),
                "report": {
                    "resolved": score >= 1.0,
                    "researchclaw_score": score,
                    "llmjudge_score": score,
                    "requested_score_mode": "hybrid",
                    "required_output_path": payload.get("required_output_path"),
                    "required_output_exists": payload.get("required_output_exists"),
                },
            }
            _publish_researchclaw_run_artifacts(run_dir, case_dir)
            _write_json_atomic(
                case_dir / "run-status.json",
                {
                    "task_id": task.task_id,
                    "session_id": session_id,
                    "polar_task_id": task_id,
                    "status": "completed",
                    "score": score,
                    "score_mode": "hybrid",
                    "code_score": code_score,
                    "required_output_path": payload.get("required_output_path"),
                    "required_output_exists": payload.get("required_output_exists"),
                    "duration_seconds": duration,
                },
            )
            return EvalResult(outcome_reward=score, metadata=metadata)
        except Exception as exc:
            duration = time.monotonic() - started
            error = f"{type(exc).__name__}: {exc}"
            _publish_researchclaw_run_artifacts(run_dir, case_dir)
            _write_json_atomic(
                case_dir / "run-status.json",
                {
                    "task_id": task.task_id,
                    "session_id": session_id,
                    "polar_task_id": task_id,
                    "status": "error",
                    "score": 0.0,
                    "score_mode": "hybrid",
                    "code_score": None,
                    "duration_seconds": duration,
                    "error": error,
                    **researchclaw_reward.required_output_observation(
                        task, workspace_dir
                    ),
                },
            )
            metadata = {
                "researchclaw_score": 0.0,
                "llmjudge_score": 0.0,
                "requested_score_mode": "hybrid",
                "researchclaw_duration_seconds": duration,
                "researchclaw_error": error,
                "researchclaw_case_dir": str(case_dir),
                "report": {
                    "resolved": False,
                    "researchclaw_score": 0.0,
                    "llmjudge_score": 0.0,
                    "requested_score_mode": "hybrid",
                    "error": error,
                },
            }
            return EvalResult(outcome_reward=0.0, metadata=metadata)
        finally:
            if eval_rollout_id is not None:
                await self._publish_eval_logs(
                    session_dir=session_dir,
                    session_id=session_id,
                    task_id=task_id,
                    eval_rollout_id=eval_rollout_id,
                    case_dir=run_dir,
                )
            elif (
                self.retain_researchclaw_nanobot_logs
                and "-eval-" not in task_id
            ):
                try:
                    if rollout_step is None:
                        raise ValueError(
                            "ResearchClaw training trajectory is missing rollout_step metadata"
                        )
                    await asyncio.to_thread(
                        _publish_researchclaw_nanobot_logs,
                        session_dir=session_dir,
                        session_id=session_id,
                        rollout_step=rollout_step,
                        polar_task_id=task_id,
                        dataset_task_id=task.task_id,
                        nanobot_log_root=self.latest_eval_nanobot_log_root,
                        state_root=(
                            self.latest_eval_nanobot_log_root.parent
                            / ".latest_train_nanobot_log_state"
                        ),
                    )
                except Exception:
                    logger.exception(
                        "Failed to retain ResearchClaw Nanobot logs for session %s",
                        session_id,
                    )

    async def _publish_eval_logs(
        self,
        *,
        session_dir: Path,
        session_id: str,
        task_id: str,
        eval_rollout_id: int | None,
        case_dir: Path | None = None,
    ) -> None:
        if eval_rollout_id is None:
            return
        try:
            await asyncio.to_thread(
                _publish_latest_eval_logs,
                rollout_id=eval_rollout_id,
                session_id=session_id,
                task_id=task_id,
                task_name=self.task_dir.name,
                session_dir=session_dir,
                case_dir=case_dir or session_dir / "logs" / "eval",
                nanobot_log_root=self.latest_eval_nanobot_log_root,
                judge_log_root=self.latest_eval_judge_log_root,
                state_root=self.latest_eval_log_state_root,
            )
        except Exception:
            logger.exception(
                "Failed to publish code-verifier eval logs rollout=%s session=%s",
                eval_rollout_id,
                session_id,
            )


class ClawGymUserAgentEvaluator(BaseTrajectoryEvaluator):
    """Score one completed Polar session with ClawGym's workspace-aware UA.

    The evaluator is deliberately trajectory-scoped: it invokes the user agent
    exactly once after the Nanobot session has finished, then returns one
    outcome reward and one feedback sentence.  Polar broadcasts the score to
    all traces from the session, while downstream OPSD code reads the same
    evaluator metadata for every trainable trace.

    Unlike :class:`ClawGymCodeVerifierEvaluator`, this evaluator does not run a
    verifier before scoring.  It only installs the task's verifier assets into
    the completed workspace.  ``user_agent.py`` gives the reviewing agent a
    ``review/run_code_verifier.sh`` tool and asks it to inspect/run that tool as
    part of its own review.
    """

    def __init__(
        self,
        *,
        task_dir: str,
        workspace: str = "/polar/session/workspace",
        timeout_seconds: float = 900.0,
        require_completed_session: bool = True,
        **_unused: Any,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.workspace = workspace
        self.timeout_seconds = float(timeout_seconds)
        self.require_completed_session = bool(require_completed_session)

    async def evaluate(self, trajectory: Trajectory, **context: Any) -> EvalResult:
        if self.require_completed_session and trajectory.status != "COMPLETED":
            raise RuntimeError(
                "user-agent evaluation requires a completed trajectory, got "
                f"{trajectory.status}"
            )
        if not trajectory.traces:
            raise RuntimeError("user-agent evaluation requires at least one trajectory trace")

        runtime = context["runtime"]
        session_id = str(context.get("session_id") or context.get("task_id") or "session")
        task_id = str(context.get("task_id") or session_id)
        session_dir = Path(context["session_dir"])
        timeout_cap = context.get("timeout_seconds")
        timeout = self.timeout_seconds
        if timeout_cap is not None:
            timeout = max(1.0, min(timeout, float(timeout_cap)))

        task = tasks.load_task(self.task_dir)
        workspace_dir = await _user_agent_workspace_source(
            runtime=runtime,
            task=task,
            runtime_workspace=self.workspace,
            session_dir=session_dir,
        )
        final_message = _trajectory_final_message(trajectory)
        started = time.monotonic()
        result = await asyncio.wait_for(
            asyncio.to_thread(
                user_agent.score_task,
                task_id=task.task_id or task_id,
                task_dir=task.task_dir,
                task_prompt=task.user_query,
                workspace_dir=workspace_dir,
                final_message=final_message,
                # No verifier is pre-run here.  The UA receives the verifier
                # helper in its review workspace and runs it itself.
                code_verifier_score=None,
                evaluated_session_id=session_id,
                evaluated_log_dir=session_dir / "logs" / "agent",
            ),
            timeout=timeout,
        )
        duration = time.monotonic() - started
        if result.parse_error:
            raise RuntimeError(f"user agent returned an invalid result: {result.parse_error}")
        feedback = result.feedback.strip()
        if not feedback:
            raise RuntimeError("user agent returned empty feedback")

        # Keep exact feedback and the compact audit trail in the trajectory.
        # Full raw model/verifier streams already live in bounded files under
        # CLAWGYM_USER_AGENT_WORKDIR; embedding them in every split trace makes
        # persisted Polar results and Slime samples needlessly large.
        user_agent_metadata = asdict(result)
        for large_field in (
            "raw_response",
            "verifier_current_stdout",
            "verifier_current_stderr",
        ):
            user_agent_metadata.pop(large_field, None)
        verifier_score = result.verifier_current_score
        if isinstance(verifier_score, bool) or not isinstance(verifier_score, (int, float)):
            verifier_score = None
        elif not math.isfinite(float(verifier_score)):
            verifier_score = None
        metadata = {
            "user_agent_score": float(result.score),
            "user_agent_feedback": feedback,
            "user_agent_feedback_present": True,
            "user_agent_feedback_chars": len(feedback),
            "user_agent_duration_seconds": duration,
            "user_agent": user_agent_metadata,
            "report": {
                "resolved": float(result.score) >= 1.0,
                "user_agent_score": float(result.score),
                "user_agent_feedback_present": True,
                "verifier_precomputed": False,
                "verifier_ran_by_agent": result.verifier_ran_by_agent,
                "verifier_current_score": result.verifier_current_score,
            },
        }
        # The UA owns verifier execution, but its verifier result remains a
        # distinct diagnostic target.  Elevate it without inventing a zero for
        # agents that did not run the verifier or produced an invalid score.
        if verifier_score is not None:
            metadata["code_verifier_score"] = float(verifier_score)
        if result.user_agent_verifier_duration_sec is not None:
            metadata["code_verifier_duration_seconds"] = float(
                result.user_agent_verifier_duration_sec
            )
        if result.verifier_current_returncode is not None:
            metadata["code_verifier_returncode"] = int(
                result.verifier_current_returncode
            )
        if result.verifier_ran_by_agent is not None:
            metadata["code_verifier_ran_by_agent"] = bool(
                result.verifier_ran_by_agent
            )
        return EvalResult(outcome_reward=float(result.score), metadata=metadata)


class ClawGymWSClawJudgeEvaluator(BaseTrajectoryEvaluator):
    """Score a Polar workspace with the distributed WSClaw judge harness."""

    # A policy-side timeout/failure is already a terminal zero-reward outcome.
    # Running the expensive judge after that only delays the callback and can
    # turn a useful sampled prefix into an infrastructure-looking timeout.
    require_completed_session = True

    def __init__(
        self,
        *,
        task_dir: str,
        workspace: str = "/polar/session/workspace",
        timeout_seconds: float = 2700.0,
        judge_worker_urls: str | list[str],
        judge_api_key: str = "apiKey",
        judge_workers_per_node: int = 64,
        log_root: str,
        keep_case_artifacts: bool = False,
        retain_latest_eval_logs: bool = False,
        latest_eval_nanobot_log_root: str | None = None,
        latest_eval_judge_log_root: str | None = None,
        latest_eval_log_state_root: str | None = None,
        **_unused: Any,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.workspace = workspace
        self.timeout_seconds = float(timeout_seconds)
        if isinstance(judge_worker_urls, str):
            urls = [item.strip() for item in judge_worker_urls.split(",") if item.strip()]
        else:
            urls = [str(item).strip() for item in judge_worker_urls if str(item).strip()]
        if not urls:
            raise ValueError("judge_worker_urls must contain at least one URL")
        self.judge_worker_urls = urls
        self.judge_api_key = judge_api_key
        self.judge_workers_per_node = int(judge_workers_per_node)
        self.log_root = Path(log_root)
        self.keep_case_artifacts = bool(keep_case_artifacts)
        self.retain_latest_eval_logs = bool(retain_latest_eval_logs)
        self.latest_eval_nanobot_log_root = Path(
            latest_eval_nanobot_log_root or self.log_root.parent / "nanobot_logs"
        )
        self.latest_eval_judge_log_root = Path(
            latest_eval_judge_log_root or self.log_root.parent / "judge_nanobot_logs"
        )
        self.latest_eval_log_state_root = Path(
            latest_eval_log_state_root or self.log_root.parent / ".latest_eval_log_state"
        )
        if self.retain_latest_eval_logs:
            _validate_latest_eval_log_roots(
                self.latest_eval_nanobot_log_root,
                self.latest_eval_judge_log_root,
                self.latest_eval_log_state_root,
            )

    async def evaluate(self, trajectory: Trajectory, **context: Any) -> EvalResult:
        runtime = context["runtime"]
        session_id = str(context.get("session_id") or context.get("task_id") or "session")
        task_id = str(context.get("task_id") or "")
        eval_rollout_id = (
            _retained_rollout_id(task_id, trajectory)
            if self.retain_latest_eval_logs
            else None
        )
        timeout_cap = context.get("timeout_seconds")
        timeout = self.timeout_seconds
        if timeout_cap is not None:
            timeout = max(1.0, min(timeout, float(timeout_cap)))

        case_dir = self.log_root / _safe_case_name(session_id, self.task_dir.name)
        case_dir.mkdir(parents=True, exist_ok=True)
        if eval_rollout_id is not None:
            await self._prepare_latest_eval_logs(eval_rollout_id)
        started = time.monotonic()
        stage_timing: dict[str, float] = {}
        try:
            workspace = case_dir / "workspace"
            stage_started = time.monotonic()
            await _materialize_judge_workspace_source(
                runtime=runtime,
                task_dir=self.task_dir,
                runtime_workspace=self.workspace,
                output_workspace=workspace,
            )
            stage_timing["workspace_materialize"] = time.monotonic() - stage_started
            evaluator = _wsclaw_eval_module()
            stage_started = time.monotonic()
            entry = evaluator.load_json(self.task_dir / "data_entry.json")
            outputs = evaluator.output_paths(self.task_dir)
            judge_workspace, artifacts = evaluator.prepare_judge_workspace(
                task_dir=self.task_dir,
                workspace=workspace,
                outputs=outputs,
                case_dir=case_dir,
                entry=entry,
            )
            stage_timing["judge_workspace_prepare"] = time.monotonic() - stage_started
            pool = _judge_pool(
                self.judge_worker_urls,
                workers_per_node=self.judge_workers_per_node,
                timeout_sec=max(1, int(timeout)),
                api_key=self.judge_api_key,
            )
            stage_started = time.monotonic()
            judge = await asyncio.wait_for(
                asyncio.to_thread(
                    pool.submit,
                    judge_workspace,
                    self.task_dir,
                    case_dir,
                    timeout_sec=max(1, int(timeout)),
                ),
                timeout=timeout + 30,
            )
            stage_timing["client_submit"] = time.monotonic() - stage_started
            duration = time.monotonic() - started
            for timing_name, judge_key in (
                ("client_pool_wait", "judge_client_pool_wait_sec"),
                ("remote_roundtrip", "judge_remote_roundtrip_sec"),
                ("client_submit_total", "judge_client_submit_total_sec"),
                ("worker_queue_wait", "judge_worker_queue_wait_sec"),
                ("worker_run", "judge_worker_run_sec"),
                ("worker_non_attempt", "judge_worker_non_attempt_sec"),
                ("judge_attempts", "judge_duration_sec"),
            ):
                value = judge.get(judge_key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    stage_timing[timing_name] = max(0.0, float(value))
            model_metrics = judge.get("judge_model_metrics")
            if not isinstance(model_metrics, dict):
                model_metrics = {}
            for timing_name, model_key in (
                ("model_proxy_requests", "proxy_duration_seconds"),
                ("router_backend_headers", "router_duration_seconds"),
                ("router_queue", "router_queue_seconds"),
            ):
                value = model_metrics.get(model_key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    stage_timing[timing_name] = max(0.0, float(value))
            user_feedback = str(judge.get("user_feedback") or "").strip()
            accepted = (
                bool(judge.get("judge_result_complete"))
                and int(judge.get("judge_exit_code", 1)) == 0
                and not bool(judge.get("judge_timed_out"))
                and evaluator.valid_user_feedback(user_feedback)
            )
            raw_score = max(0.0, min(1.0, float(judge.get("score", 0.0))))
            score = raw_score if accepted else 0.0
            metadata = {
                "llm_judge_score": score,
                "llm_judge_feedback": str(judge.get("feedback") or "").strip(),
                "llm_judge_user_feedback": user_feedback,
                "llm_judge_duration_seconds": float(judge.get("judge_duration_sec", duration)),
                "llm_judge_exit_code": int(judge.get("judge_exit_code", 1)),
                "llm_judge_timed_out": bool(judge.get("judge_timed_out")),
                "llm_judge_result_complete": accepted,
                "llm_judge_worker_host": judge.get("judge_worker_host"),
                "llm_judge_worker_url": judge.get("judge_worker_url"),
                "llm_judge_case_dir": str(case_dir),
                "llm_judge_artifacts_found": sum(
                    int(artifact.get("exists", False)) for artifact in artifacts
                ),
                "llm_judge_artifacts_required": len(artifacts),
                "llm_judge_timing_seconds": stage_timing,
                "llm_judge_model_metrics": model_metrics,
                # Compatibility field for trajectory consumers that already
                # understand one-sentence user-agent feedback.  The score
                # remains an LLM-judge score; no separate UA score is invented.
                "user_agent_feedback": user_feedback,
                "user_agent_feedback_present": bool(user_feedback),
                "user_agent_feedback_chars": len(user_feedback),
                "report": {
                    "resolved": accepted,
                    "llm_judge_score": score,
                    "judge_result_complete": accepted,
                    "judge_exit_code": int(judge.get("judge_exit_code", 1)),
                    "judge_timed_out": bool(judge.get("judge_timed_out")),
                    "user_feedback": user_feedback,
                },
            }
            _write_json(case_dir / "polar_result.json", metadata)
            finalize_started = time.monotonic()
            await self._finalize_case_logs(
                case_dir=case_dir,
                session_dir=Path(context["session_dir"]),
                session_id=session_id,
                task_id=task_id,
                eval_rollout_id=eval_rollout_id,
            )
            stage_timing["case_finalize"] = time.monotonic() - finalize_started
            stage_timing["evaluator_total"] = time.monotonic() - started
            if case_dir.is_dir():
                _write_json(case_dir / "polar_result.json", metadata)
            _append_judge_timing_record(
                session_id=session_id,
                task_id=task_id,
                status="completed" if accepted else "rejected",
                score=score,
                worker_host=judge.get("judge_worker_host"),
                timing=stage_timing,
            )
            return EvalResult(outcome_reward=score, metadata=metadata)
        except Exception as exc:
            duration = time.monotonic() - started
            error = f"{type(exc).__name__}: {exc}"
            metadata = {
                "llm_judge_score": 0.0,
                "llm_judge_duration_seconds": duration,
                "llm_judge_result_complete": False,
                "llm_judge_error": error,
                "llm_judge_case_dir": str(case_dir),
                "llm_judge_timing_seconds": stage_timing,
                "user_agent_feedback_present": False,
                "report": {
                    "resolved": False,
                    "llm_judge_score": 0.0,
                    "judge_result_complete": False,
                    "error": error,
                },
            }
            _write_json(case_dir / "polar_error.json", metadata)
            finalize_started = time.monotonic()
            await self._finalize_case_logs(
                case_dir=case_dir,
                session_dir=Path(context["session_dir"]),
                session_id=session_id,
                task_id=task_id,
                eval_rollout_id=eval_rollout_id,
            )
            stage_timing["case_finalize"] = time.monotonic() - finalize_started
            stage_timing["evaluator_total"] = time.monotonic() - started
            if case_dir.is_dir():
                _write_json(case_dir / "polar_error.json", metadata)
            _append_judge_timing_record(
                session_id=session_id,
                task_id=task_id,
                status="error",
                score=0.0,
                worker_host=None,
                timing=stage_timing,
                error=error,
            )
            return EvalResult(outcome_reward=0.0, metadata=metadata)

    async def _prepare_latest_eval_logs(self, rollout_id: int) -> None:
        try:
            await asyncio.to_thread(
                _prepare_latest_eval_log_roots,
                rollout_id=rollout_id,
                nanobot_log_root=self.latest_eval_nanobot_log_root,
                judge_log_root=self.latest_eval_judge_log_root,
                state_root=self.latest_eval_log_state_root,
            )
        except Exception:
            logger.exception("Failed to prepare latest eval log roots for rollout %s", rollout_id)

    async def _finalize_case_logs(
        self,
        *,
        case_dir: Path,
        session_dir: Path,
        session_id: str,
        task_id: str,
        eval_rollout_id: int | None,
    ) -> None:
        if eval_rollout_id is not None:
            try:
                await asyncio.to_thread(
                    _publish_latest_eval_logs,
                    rollout_id=eval_rollout_id,
                    session_id=session_id,
                    task_id=task_id,
                    task_name=self.task_dir.name,
                    session_dir=session_dir,
                    case_dir=case_dir,
                    nanobot_log_root=self.latest_eval_nanobot_log_root,
                    judge_log_root=self.latest_eval_judge_log_root,
                    state_root=self.latest_eval_log_state_root,
                )
            except Exception:
                logger.exception(
                    "Failed to publish latest eval logs for rollout %s session %s",
                    eval_rollout_id,
                    session_id,
                )

        if not self.keep_case_artifacts:
            if self.retain_latest_eval_logs:
                # In latest-eval mode the public roots above are the only durable
                # copies. Training cases and the private copies of eval cases are
                # removed completely so storage stays bounded.
                await asyncio.to_thread(shutil.rmtree, case_dir, True)
            else:
                _prune_judge_case(case_dir)


async def _materialize_judge_workspace_source(
    *,
    runtime: Any,
    task_dir: Path,
    runtime_workspace: str,
    output_workspace: Path,
) -> None:
    if output_workspace.exists():
        shutil.rmtree(output_workspace)
    task = tasks.load_task(task_dir)
    tasks.prepare_workspace(task, output_workspace)
    evaluator = _wsclaw_eval_module()
    for relative in evaluator.output_paths(task_dir):
        destination = output_workspace / relative
        source = runtime.resolve_host_path(f"{runtime_workspace}/{relative.as_posix()}")
        if source is not None and source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            continue
        try:
            await runtime.download_file(
                f"{runtime_workspace}/{relative.as_posix()}",
                str(destination),
            )
        except FileNotFoundError:
            continue


async def _user_agent_workspace_source(
    *,
    runtime: Any,
    task: tasks.TaskSpec,
    runtime_workspace: str,
    session_dir: Path,
) -> Path:
    """Return the completed workspace with verifier assets installed, unrun."""
    source = runtime.resolve_host_path(runtime_workspace)
    if source is None or not source.is_dir():
        source = session_dir / "logs" / "eval" / "user_agent_source_workspace"
        await runtime.download_dir(runtime_workspace, str(source))
    tasks.copy_reward_assets(task, source)
    return source


def _trajectory_final_message(trajectory: Trajectory) -> str:
    """Extract the final assistant text from a completed prefix-merged session."""
    for trace in reversed(trajectory.traces):
        for message in reversed(trace.response_messages):
            if str(message.get("role") or "") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text = "\n".join(
                    str(part.get("text") or "").strip()
                    for part in content
                    if isinstance(part, dict) and str(part.get("text") or "").strip()
                ).strip()
                if text:
                    return text
    return ""


def _wsclaw_eval_module() -> Any:
    from scripts.clawgym_nanobot import wsclaw_eval_only

    return wsclaw_eval_only


def _judge_pool(
    urls: list[str],
    *,
    workers_per_node: int,
    timeout_sec: int,
    api_key: str,
) -> Any:
    key = (tuple(urls), workers_per_node, api_key)
    with _JUDGE_POOLS_LOCK:
        pool = _JUDGE_POOLS.get(key)
        if pool is None:
            pool = _wsclaw_eval_module().RemoteJudgePool(
                urls,
                workers_per_node=workers_per_node,
                timeout_sec=timeout_sec,
                api_key=api_key,
            )
            _JUDGE_POOLS[key] = pool
        return pool


def _safe_case_name(session_id: str, task_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{session_id}_{task_name}")
    return value[-220:]


def _is_researchclaw_task(task_dir: Path) -> bool:
    """Recognize ResearchClaw from its task-local reward contract.

    Exported ResearchClaw metadata intentionally preserves the upstream
    provenance schema, which predates a dedicated ``source`` field.  The
    combination below is the stable, task-local interface used by the reward
    runner and avoids depending on cluster-specific dataset paths.
    """
    reward_dir = task_dir / "reward"
    config_path = reward_dir / "eval_config.json"
    rubric_path = reward_dir / "llmjudge" / "rubric.json"
    if not config_path.is_file() or not rubric_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(config, dict):
        return False
    modes = config.get("available_score_modes")
    return (
        isinstance(modes, list)
        and "llmjudge" in modes
        and bool(config.get("llmjudge_entry"))
        and bool(config.get("required_output_path"))
    )


def _validate_pure_researchclaw_reward(
    payload: Any,
    returned_score: Any,
) -> float:
    if not isinstance(payload, dict):
        raise TypeError("ResearchClaw reward payload must be an object")
    if payload.get("requested_score_mode") != "hybrid":
        raise ValueError(
            "ResearchClaw Polar reward must use requested_score_mode=hybrid"
        )
    weights = payload.get("requested_score_weights")
    if weights != {"code": 0.0, "llmjudge": 1.0}:
        raise ValueError(
            "ResearchClaw pure LLM-judge reward has unexpected score weights"
        )
    try:
        code_score = float(payload["code_score"])
        final_score = float(payload["final_score"])
        llmjudge_score = float(payload["llmjudge_score"])
        returned = float(returned_score)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("ResearchClaw reward contains a non-numeric score") from exc
    if not all(math.isfinite(value) for value in (code_score, final_score, llmjudge_score, returned)):
        raise ValueError("ResearchClaw reward contains a non-finite score")
    if not (0.0 <= final_score <= 1.0):
        raise ValueError(f"ResearchClaw reward is outside [0, 1]: {final_score}")
    if not (
        math.isclose(final_score, llmjudge_score, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(final_score, returned, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(
            "ResearchClaw final, LLM-judge, and returned scores do not match"
        )
    return final_score


def _publish_researchclaw_run_artifacts(
    run_dir: Path,
    case_dir: Path,
    *,
    max_file_bytes: int = 8 * 1024 * 1024,
    max_total_bytes: int = 32 * 1024 * 1024,
) -> None:
    """Persist bounded, redacted reward evidence outside Polar's /tmp session.

    The agentic judge workspace can contain the complete task inputs, so it is
    deliberately excluded.  Root reward/judge records and bounded per-attempt
    JSON/log files are enough to audit the authoritative score.
    """
    if not run_dir.is_dir():
        return
    candidates = [
        path
        for path in run_dir.iterdir()
        if path.is_file()
        and (path.name.startswith("researchclaw_") or path.name.startswith("reward."))
    ]
    attempts_dir = run_dir / "researchclaw_judge_attempts"
    if attempts_dir.is_dir():
        candidates.extend(path for path in attempts_dir.rglob("*") if path.is_file())

    copied = 0
    for source in sorted(candidates):
        try:
            size = source.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes or copied + size > max_total_bytes:
            continue
        relative = source.relative_to(run_dir)
        destination = case_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        copied += size


def _publish_researchclaw_nanobot_logs(
    *,
    session_dir: Path,
    session_id: str,
    rollout_step: int,
    polar_task_id: str,
    dataset_task_id: str,
    nanobot_log_root: Path,
    state_root: Path,
) -> bool:
    """Retain only the latest train rollout's solver transcripts."""

    rollout_id = int(rollout_step)
    if rollout_id < 0:
        raise ValueError(f"rollout_step must be non-negative, got {rollout_id}")

    agent_log_root = session_dir / "logs" / "agent"
    session_log_root = agent_log_root / session_id
    source = session_log_root if session_log_root.is_dir() else agent_log_root
    metadata = {
        "rollout_id": rollout_id,
        "session_id": session_id,
        "polar_task_id": polar_task_id,
        "task_id": dataset_task_id,
        "source_present": source.is_dir(),
    }
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "publish.lock").open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        marker = state_root / "rollout_id"
        current_rollout_id = _read_int(marker)
        if current_rollout_id is not None and rollout_id < current_rollout_id:
            return False
        if current_rollout_id != rollout_id:
            shutil.rmtree(nanobot_log_root, ignore_errors=True)
            nanobot_log_root.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(marker, f"{rollout_id}\n")
        else:
            nanobot_log_root.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(
            nanobot_log_root / "LATEST_TRAIN_ROLLOUT_ID", f"{rollout_id}\n"
        )

        destination = nanobot_log_root / _safe_case_name(
            session_id, dataset_task_id
        )
        _replace_tree(source=source, destination=destination, metadata=metadata)

        return _ensure_canonical_cleaned_transcript(destination)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def _append_judge_timing_record(
    *,
    session_id: str,
    task_id: str,
    status: str,
    score: float,
    worker_host: Any,
    timing: dict[str, float],
    error: str | None = None,
) -> None:
    configured_path = os.environ.get("CLAWGYM_JUDGE_TIMING_JSONL", "").strip()
    if not configured_path:
        return
    path = Path(configured_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "recorded_at_unix": time.time(),
        "session_id": session_id,
        "task_id": task_id,
        "status": status,
        "score": float(score),
        "worker_host": worker_host,
        "timing_seconds": dict(timing),
    }
    if error:
        record["error"] = error
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.write(payload)
            stream.flush()
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _eval_rollout_id(task_id: str) -> int | None:
    """Return the rollout id from Slime's namespaced eval task id."""
    if "-eval-" not in task_id:
        return None
    parts = task_id.rsplit("-", 2)
    if len(parts) != 3:
        return None
    try:
        rollout_id = int(parts[-2])
        int(parts[-1])  # validate the eval group position as well
    except ValueError:
        return None
    return rollout_id


def _retained_rollout_id(task_id: str, trajectory: Trajectory) -> int | None:
    """Resolve the log-retention generation without changing task semantics.

    Eval task ids carry their rollout id in the id itself. Focused pipeline
    diagnostics can opt in to retaining training-session Nanobot logs too; in
    that case the already-recorded rollout_step is the equivalent generation.
    """

    eval_rollout_id = _eval_rollout_id(task_id)
    if eval_rollout_id is not None:
        return eval_rollout_id
    if os.environ.get("WSCLAW_KEEP_TRAIN_NANOBOT_LOGS", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    return _trajectory_rollout_step(trajectory)


def _trajectory_rollout_step(trajectory: Trajectory) -> int | None:
    value = trajectory.metadata.get("rollout_step")
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _validate_latest_eval_log_roots(
    nanobot_log_root: Path,
    judge_log_root: Path,
    state_root: Path,
) -> None:
    resolved = [path.resolve(strict=False) for path in (nanobot_log_root, judge_log_root, state_root)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("latest eval log and state roots must be distinct")
    for path in resolved:
        # These directories are replaced as eval rollouts advance. Refuse broad
        # paths even if a caller accidentally misconfigures the evaluator.
        if len(path.parts) < 3 or path == Path(path.anchor):
            raise ValueError(f"refusing unsafe latest eval log root: {path}")


def _prepare_latest_eval_log_roots(
    *,
    rollout_id: int,
    nanobot_log_root: Path,
    judge_log_root: Path,
    state_root: Path,
) -> bool:
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "publish.lock").open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return _prepare_latest_eval_log_roots_locked(
            rollout_id=rollout_id,
            nanobot_log_root=nanobot_log_root,
            judge_log_root=judge_log_root,
            state_root=state_root,
        )


def _prepare_latest_eval_log_roots_locked(
    *,
    rollout_id: int,
    nanobot_log_root: Path,
    judge_log_root: Path,
    state_root: Path,
) -> bool:
    marker = state_root / "rollout_id"
    current_rollout_id = _read_int(marker)
    if current_rollout_id is not None and rollout_id < current_rollout_id:
        return False

    if current_rollout_id != rollout_id:
        shutil.rmtree(nanobot_log_root, ignore_errors=True)
        shutil.rmtree(judge_log_root, ignore_errors=True)
        nanobot_log_root.mkdir(parents=True, exist_ok=True)
        judge_log_root.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(marker, f"{rollout_id}\n")
    else:
        nanobot_log_root.mkdir(parents=True, exist_ok=True)
        judge_log_root.mkdir(parents=True, exist_ok=True)

    for root in (nanobot_log_root, judge_log_root):
        _write_text_atomic(root / "LATEST_EVAL_ROLLOUT_ID", f"{rollout_id}\n")
    return True


def _publish_latest_eval_logs(
    *,
    rollout_id: int,
    session_id: str,
    task_id: str,
    task_name: str,
    session_dir: Path,
    case_dir: Path,
    nanobot_log_root: Path,
    judge_log_root: Path,
    state_root: Path,
) -> bool:
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "publish.lock").open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not _prepare_latest_eval_log_roots_locked(
            rollout_id=rollout_id,
            nanobot_log_root=nanobot_log_root,
            judge_log_root=judge_log_root,
            state_root=state_root,
        ):
            return False

        destination_name = _safe_case_name(session_id, task_name)
        metadata = {
            "rollout_id": rollout_id,
            "session_id": session_id,
            "task_id": task_id,
            "task_name": task_name,
        }
        _replace_tree(
            source=session_dir / "logs" / "agent",
            destination=nanobot_log_root / destination_name,
            metadata=metadata,
        )
        _ensure_canonical_cleaned_transcript(
            nanobot_log_root / destination_name
        )
        _replace_flat_files(
            source=case_dir,
            destination=judge_log_root / destination_name,
            metadata=metadata,
        )
        return True


def _ensure_canonical_cleaned_transcript(destination: Path) -> bool:
    transcript = next(
        (
            path
            for path in (
                destination / "cleaned_transcripts.json",
                destination / "clean-transcript.json",
            )
            if path.is_file()
        ),
        None,
    )
    if transcript is None:
        matches = sorted(destination.rglob("cleaned_transcripts.json"))
        if not matches:
            matches = sorted(destination.rglob("clean-transcript.json"))
        transcript = matches[0] if matches else None
    if transcript is None:
        return False

    canonical = destination / "cleaned_transcripts.json"
    if transcript != canonical:
        _copy_file_atomic(transcript, canonical)
    return True


def _replace_tree(*, source: Path, destination: Path, metadata: dict[str, Any]) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        if source.is_dir():
            shutil.copytree(source, temporary, dirs_exist_ok=True)
        _write_json(temporary / "capture.json", {**metadata, "source_present": source.is_dir()})
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _replace_flat_files(
    *,
    source: Path,
    destination: Path,
    metadata: dict[str, Any],
) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        copied_files = 0
        if source.is_dir():
            for path in source.iterdir():
                if path.is_file() and not path.is_symlink():
                    shutil.copy2(path, temporary / path.name)
                    copied_files += 1
        _write_json(
            temporary / "capture.json",
            {**metadata, "source_present": source.is_dir(), "copied_files": copied_files},
        )
        shutil.rmtree(destination, ignore_errors=True)
        os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _prune_judge_case(case_dir: Path) -> None:
    for name in ("workspace", "judge_workspace"):
        shutil.rmtree(case_dir / name, ignore_errors=True)
    for name in ("claude_judge.log", "judge_cleaned_transcripts.json", "judge_prompt.txt"):
        (case_dir / name).unlink(missing_ok=True)
    for path in case_dir.glob("judge_result.invalid_attempt*.json"):
        path.unlink(missing_ok=True)


def _score_reward_payload(raw: str) -> float:
    data = json.loads(raw.strip())
    if isinstance(data, dict):
        if not data:
            return 0.0
        return sum(float(value) for value in data.values()) / len(data)
    if isinstance(data, (int, float)):
        return float(data)
    return 0.0


def _verifier_command(
    *,
    python_executable: str | None,
    check_path: str,
    workspace: str,
) -> str:
    if python_executable:
        python_path = Path(python_executable)
    else:
        python_path = nanobot_runtime.nanobot_python_executable()
        if not python_path.exists():
            python_path = Path(sys.executable)

    lines = [
        "set -euo pipefail",
        "unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M",
        "unset PYTHONPATH",
        f"export PATH={shlex.quote(str(python_path.parent))}:$PATH",
    ]
    pythonpath = nanobot_runtime.nanobot_pythonpath()
    if pythonpath:
        lines.append(f"export PYTHONPATH={shlex.quote(pythonpath)}")
    lines.append(
        "exec "
        + " ".join(
            shlex.quote(item)
            for item in (str(python_path), check_path, workspace)
        )
    )
    return "; ".join(lines)
