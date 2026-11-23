from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polar.agent.presets.nanobot import runtime as _default_nanobot_runtime
from . import nanobot_sandbox_runtime as _sandbox_nanobot_runtime
from .nanobot_sandbox_runtime import WSClawSandboxError


def _active_nanobot_runtime():
    if os.environ.get("CLAWGYM_WSCLAW_SANDBOX", "0") == "1":
        return _sandbox_nanobot_runtime
    return _default_nanobot_runtime


nanobot_runtime = _sandbox_nanobot_runtime if os.environ.get("CLAWGYM_WSCLAW_SANDBOX", "0") == "1" else _default_nanobot_runtime

from . import (
    polar_workspace,
    raw_source_integrity,
    researchclaw_metrics,
    researchclaw_reward,
    tasks,
)


PROJECT_ROOT = tasks.PROJECT_ROOT
DEFAULT_NANOBOT_ROOT = PROJECT_ROOT / "vendor" / "clawgym_nanobot" / "nanobot"
DEFAULT_CONDA_ROOT = Path("")
DEFAULT_CONTINUE_PROMPT = "Continue working on the task."
CAPTURE_TAIL_BYTES = int(os.environ.get("NANOBOT_SUBPROCESS_CAPTURE_TAIL_BYTES", str(1024 * 1024)))
_RL_TASK_INSTRUCTION = """You are solving a local file-based task. The user's request is the source of truth; follow it directly and produce the requested artifacts in the workspace.

Working rules:
- Use the available tools to inspect the relevant local input files or directories before writing final outputs.
- Create requested files at the exact paths named by the user, including parent directories when needed.
- If the task asks for code, scripts, commands, tests, or validation, write the real files and run the requested local checks.
- Do not use the internet or external services. Use only the files and commands available in this workspace.
- Do not inspect, modify, or rely on evaluator artifacts if they are present; solve the task from the user's inputs.
- Do not claim a file or check is complete unless you actually created the file or ran the check.
- Keep the final reply brief after the artifacts exist; put long content in the requested output files, not in the final reply.
"""
_ORIGINAL_CLAWGYM_TASK_PROMPT_MODES = {"original", "legacy", "raw"}


@dataclass
class NanobotTurnResult:
    turn_index: int
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0


@dataclass
class NanobotRunResult:
    task_id: str
    session_id: str
    reward: float
    status: str
    run_dir: str
    workspace_dir: str
    log_dir: str
    duration_s: float
    code_verifier_duration_s: float = 0.0
    turns: list[NanobotTurnResult] = field(default_factory=list)
    error: str | None = None
    task_metadata: dict[str, Any] | None = None
    researchclaw_reward: dict[str, Any] | None = None
    agent_activity: dict[str, Any] | None = None
    evaluation_reward: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "reward": self.reward,
            "status": self.status,
            "run_dir": self.run_dir,
            "workspace_dir": self.workspace_dir,
            "log_dir": self.log_dir,
            "duration_s": self.duration_s,
            "code_verifier_duration_s": self.code_verifier_duration_s,
            "turns": [turn.__dict__ for turn in self.turns],
            "error": self.error,
        }
        if self.task_metadata is not None:
            payload["task_metadata"] = self.task_metadata
        if self.researchclaw_reward is not None:
            payload["researchclaw_reward"] = self.researchclaw_reward
        if self.agent_activity is not None:
            payload["agent_activity"] = self.agent_activity
        if self.evaluation_reward is not None:
            payload["evaluation_reward"] = self.evaluation_reward
        return payload


def run_nanobot_task(
    args: Any,
    task: tasks.TaskSpec,
    *,
    session_id: str,
    adapter_base_url: str,
    rollout_id: int,
    task_index: int,
    group_index: int | None,
    sampling_params: dict[str, Any] | None,
    extra_context: str | None = None,
    evaluation: bool = False,
    preserve_run_dir: bool = False,
) -> NanobotRunResult:
    start = time.monotonic()
    turn_type = _turn_type(evaluation)
    source_guard = raw_source_integrity.TaskSourceGuard(
        task.task_dir,
        rollout_id=rollout_id,
        group_index=group_index,
    )
    # Do this before creating the disposable run directory. A missing or
    # modified canonical input is an infrastructure failure, not a zero-reward
    # sample, and must propagate to the training supervisor.
    source_guard.check("before")
    run_dir = _new_task_run_dir(rollout_id=rollout_id, task_index=task_index)
    workspace_dir = run_dir / "workspace"
    log_root = _nanobot_log_dir(args)
    session_log_dir = log_root / session_id
    log_root.mkdir(parents=True, exist_ok=True)
    turns: list[NanobotTurnResult] = []
    reward = 0.0
    code_verifier_duration_s = 0.0
    status = "running"
    error: str | None = None
    error_type: str | None = None
    evaluation_reward: dict[str, Any] | None = None
    judge_failure: WSClawJudgeError | None = None

    try:
        polar_workspace.prepare_task_workspace(task.task_dir, workspace_dir)
        if os.environ.get("NANOBOT_GEMMA_TOOL_SCHEMA", "0") == "1":
            from .gemma_tool_prompt import prepare_gemma_tool_prompt
            prepare_gemma_tool_prompt(workspace_dir)
        config_path = _write_nanobot_config(
            workspace_dir=workspace_dir,
            run_dir=run_dir,
            adapter_base_url=adapter_base_url,
            session_id=session_id,
            turn_type=turn_type,
            sampling_params=sampling_params or {},
        )
        max_turns = _max_turns(evaluation)
        timeout = _turn_timeout_seconds(evaluation)
        env_overrides = {
            "NANOBOT_GEMMA_TOOL_SCHEMA": os.environ.get("NANOBOT_GEMMA_TOOL_SCHEMA", "0"),
            "NANOBOT_ROLLOUT_ID": str(rollout_id),
            "NANOBOT_GROUP_INDEX": "" if group_index is None else str(group_index),
            "NANOBOT_TASK_ID": task.task_id,
        }
        # Every training profile treats a detected tool loop as a failed,
        # fully masked trajectory. Evaluation is not subject to this filter.
        env_overrides["NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION"] = (
            "1" if evaluation else "0"
        )
        for turn_index in range(max_turns):
            if turn_index == 0:
                message = _format_clawgym_task_prompt(task.user_query)
                message = _append_extra_context(message, extra_context)
                if _env_truthy("NANOBOT_APPEND_NO_THINK", False):
                    message = message.rstrip() + "\n\n/no_think"
            else:
                message = DEFAULT_CONTINUE_PROMPT
            existing_agent_end_names = {
                path.name for path in session_log_dir.glob("*_agent-end.json")
            }
            turn_start = time.monotonic()
            proc = _run_nanobot_turn(
                workspace_dir=workspace_dir,
                config_path=config_path,
                session_id=session_id,
                turn_type=turn_type,
                message=message,
                log_dir=log_root,
                timeout=timeout,
                env_overrides=env_overrides,
            )
            result = NanobotTurnResult(
                turn_index=turn_index,
                returncode=proc.returncode,
                stdout_tail=proc.stdout or "",
                stderr_tail=proc.stderr or "",
                duration_s=time.monotonic() - turn_start,
            )
            turns.append(result)
            _write_turn_logs(run_dir, result)
            if proc.returncode != 0:
                raise RuntimeError(f"nanobot turn {turn_index} failed with exit {proc.returncode}")
            if not evaluation:
                loop_error = _recover_training_tool_loop_error(
                    session_log_dir,
                    ignore_names=existing_agent_end_names,
                )
                if loop_error is not None:
                    raise RuntimeError(loop_error)

        verifier_started_at = time.monotonic()
        try:
            from .self_reward import configuration as self_reward_configuration
            if not evaluation and self_reward_configuration() is not None:
                reward = 0.0  # Explicitly pending; finalized before training conversion.
            elif _wsclaw_llm_judge_enabled():
                reward, evaluation_reward = _run_required_wsclaw_llm_judge(
                    task, workspace_dir, run_dir, evaluation=evaluation
                )
            else:
                tasks.copy_reward_assets(task, workspace_dir)
                reward = _run_reward(task, workspace_dir, run_dir)
        finally:
            code_verifier_duration_s = time.monotonic() - verifier_started_at
        status = "completed"
    except (WSClawJudgeError, WSClawSandboxError) as exc:
        judge_failure = exc
        status = "failed"
        error_type = type(exc).__name__
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        error = f"{type(exc).__name__}: {exc}"
    finally:
        duration_s = time.monotonic() - start
        try:
            source_guard.check("after")
        except raw_source_integrity.RawSourceIntegrityError:
            # Raw-data corruption is fatal and must not be converted into an
            # ordinary masked sample. Avoid retaining the task workspace while
            # the outer training process unwinds.
            _cleanup_run_dir(run_dir, evaluation=evaluation, status="failed")
            raise

    reward_summary = researchclaw_metrics.reward_summary_from_run_dir(run_dir)
    if reward_summary is None and _is_researchclaw_task(task, workspace_dir):
        reward_summary = researchclaw_metrics.runtime_failure_summary(
            task_id=task.task_id,
            error=error,
        )
    if reward_summary is not None:
        reward_summary.update(researchclaw_reward.required_output_observation(task, workspace_dir))
    agent_activity = _recover_agent_activity(session_log_dir, workspace_dir)

    result = NanobotRunResult(
        task_id=task.task_id,
        session_id=session_id,
        reward=float(reward),
        status=status,
        run_dir=str(run_dir),
        workspace_dir=str(workspace_dir),
        log_dir=str(session_log_dir),
        duration_s=duration_s,
        code_verifier_duration_s=code_verifier_duration_s,
        turns=turns,
        error=error,
        task_metadata={
            "task_id": task.task_id,
            "task_dir": str(task.task_dir),
            "input_mount_dir": task.input_mount_dir,
            "metadata": dict(task.metadata),
        },
        researchclaw_reward=reward_summary,
        agent_activity=agent_activity,
        evaluation_reward=evaluation_reward,
    )
    _write_run_summary(run_dir, result)
    _write_session_run_status(
        session_log_dir,
        result,
        rollout_id=rollout_id,
        task_index=task_index,
        group_index=group_index,
        error_type=error_type,
    )
    _persist_researchclaw_reward(run_dir, session_log_dir)
    _persist_wsclaw_judge_result(run_dir, session_log_dir)
    if os.environ.get("CLAWGYM_TRAJECTORY_ARCHIVE_DIR"):
        from .trajectory_archive import capture
        capture(task, result, rollout_id, task_index, evaluation)
    if judge_failure is not None:
        # Preserve the failed workspace and evidence; never emit a training sample.
        raise judge_failure
    if not (preserve_run_dir and status == "completed"):
        _cleanup_run_dir(run_dir, evaluation=evaluation, status=status)
    return result


def cleanup_run_dir(run_dir: str | Path) -> None:
    shutil.rmtree(Path(run_dir), ignore_errors=True)


_WSCLAW_JUDGE_POOLS: dict[tuple[tuple[str, ...], int, int, str], Any] = {}
_WSCLAW_JUDGE_POOLS_LOCK = threading.Lock()


class WSClawJudgeError(RuntimeError):
    """Native GT scoring failed; the rollout must abort without a reward."""


class WSClawJudgeMasked(WSClawJudgeError):
    """Missing score after judge turn budget; retain evidence and mask training."""


def _run_required_wsclaw_llm_judge(task, workspace_dir, run_dir, *, evaluation=False):
    result = None
    try:
        score, result = _run_wsclaw_llm_judge(task, workspace_dir, run_dir)
        native = result.get("native_result") or {}
        metrics = native.get("judge_model_metrics") or {}
        valid = (
            result.get("judge_result_complete") is True
            and result.get("judge_schema_complete") is True
            and (result.get("judge_exit_code") == 0 or bool(result.get("judge_completion_verified")))
            and result.get("judge_timed_out") is False
            and int(metrics.get("successful_request_count", 0)) > 0
            and math.isfinite(float(native.get("score")))
            and math.isfinite(float(score))
        )
        if not valid:
            raise ValueError("native GT judge result is incomplete or has no successful request")
    except Exception as exc:
        # Explicit user-authorized evaluation reporting policy; training path unchanged.
        if evaluation and os.environ.get("GEMMA_EVAL_INFRA_FAILURE_REPORT_ZERO") == "1":
            failure = {"action": "user_authorized_evaluation_zero", "original_invalid": True,
                       "valid_judge": False, "error_type": type(exc).__name__, "error": str(exc)[:4096],
                       "task_id": task.task_id, "run_dir": str(run_dir), "at": time.time()}
            result = dict(result or {})
            result.update(score=0.0, score_valid=False, original_invalid=True,
                          evaluation_failure_zero=failure)
            raw_source_integrity.write_json_atomic(run_dir / "judge_result.json", result)
            raw_source_integrity.write_json_atomic(run_dir / "judge-failure.json", failure)
            return 0.0, result
        from .judge_failure_mask import missing_score_reason
        missing_reason = missing_score_reason(result)
        if missing_reason:
            result.update(training_masked=True, score_valid=False,
                          mask_reason=missing_reason, raw_judge_score=None)
            raw_source_integrity.write_json_atomic(run_dir / "judge_result.json", result)
            raw_source_integrity.write_json_atomic(run_dir / "judge-failure.json", {
                "action": "mask_sample", "component": "wsclaw_native_gt_judge",
                "reason": missing_reason, "score_valid": False,
                "task_id": task.task_id, "run_dir": str(run_dir), "at": time.time(),
            })
            raise WSClawJudgeMasked(
                f"{missing_reason}; missing score masked; evidence: {run_dir}"
            ) from exc
        payload = {
            "action": "stop_and_resume", "manual_resume_required": True,
            "component": "wsclaw_native_gt_judge", "reason": "invalid_native_gt_reward",
            "run_id": os.environ.get("RUN_ID", "unknown"),
            "task_id": task.task_id, "task_dir": str(task.task_dir),
            "run_dir": str(run_dir), "error_type": type(exc).__name__,
            "error": str(exc)[:4096], "at": time.time(),
        }
        try:
            raw_source_integrity.write_json_atomic(run_dir / "judge-failure.json", payload)
            marker_dir = os.environ.get("PRORL_FATAL_RESUME_MARKER_DIR", "").strip()
            if marker_dir:
                raw_source_integrity.write_json_atomic(
                    Path(marker_dir) / f"native-gt-judge-{os.getpid()}-{time.time_ns()}.json", payload
                )
        finally:
            raise WSClawJudgeError(f"native GT scoring failed for {task.task_id}; evidence: {run_dir}") from exc

    # Model deliverable failures are normal scored trajectories, not fatal judge
    # failures. This gate must be outside the native-judge exception handler.
    if os.environ.get("CLAWGYM_ARTIFACT_VALIDITY_GATE") == "1":
        from .artifact_validity_gate import apply_gate
        return apply_gate(task, workspace_dir, run_dir, score, result)
    return score, result


def _wsclaw_llm_judge_enabled() -> bool:
    return os.environ.get("CLAWGYM_REWARD_SOURCE", "").strip().lower() == "llm_judge"


def _run_wsclaw_llm_judge(
    task: tasks.TaskSpec,
    workspace_dir: Path,
    run_dir: Path,
) -> tuple[float, dict[str, Any]]:
    from scripts.clawgym_nanobot import wsclaw_eval_only as ws

    configured_urls = tuple(
        item.strip().rstrip("/")
        for item in os.environ.get("WSCLAW_JUDGE_WORKER_URLS", "").split(",")
        if item.strip()
    )
    if not configured_urls:
        raise RuntimeError(
            "CLAWGYM_REWARD_SOURCE=llm_judge requires WSCLAW_JUDGE_WORKER_URLS"
        )
    urls = _local_wsclaw_judge_urls(configured_urls)
    workers_per_node = _positive_int_env("WSCLAW_JUDGE_WORKERS_PER_NODE", 16)
    timeout_sec = _positive_int_env("POLAR_EVALUATOR_TIMEOUT", 2700)
    api_key = os.environ.get("WSCLAW_JUDGE_API_KEY", "apiKey")
    case_dir = run_dir / "wsclaw_judge"
    case_dir.mkdir(parents=True, exist_ok=False)
    entry = ws.load_json(task.task_dir / "data_entry.json")
    outputs = ws.output_paths(task.task_dir)
    judge_workspace, artifacts = ws.prepare_judge_workspace(
        task_dir=task.task_dir,
        workspace=workspace_dir,
        outputs=outputs,
        case_dir=case_dir,
        entry=entry,
    )
    pool_key = (urls, workers_per_node, timeout_sec, api_key)
    with _WSCLAW_JUDGE_POOLS_LOCK:
        pool = _WSCLAW_JUDGE_POOLS.get(pool_key)
        if pool is None:
            pool = ws.RemoteJudgePool(
                list(urls),
                workers_per_node=workers_per_node,
                timeout_sec=timeout_sec,
                api_key=api_key,
            )
            _WSCLAW_JUDGE_POOLS[pool_key] = pool
    judge = pool.submit(
        judge_workspace,
        task.task_dir,
        case_dir,
        timeout_sec=timeout_sec,
    )
    from .judge_committed_result import verify_committed_result
    completion = verify_committed_result(judge)
    user_feedback = str(judge.get("user_feedback") or "").strip()
    accepted = (
        bool(judge.get("judge_result_complete"))
        and bool(judge.get("judge_schema_complete", True))
        and (int(judge.get("judge_exit_code", 1)) == 0 or completion is not None)
        and not bool(judge.get("judge_timed_out"))
    )
    raw_score = max(0.0, min(1.0, float(judge.get("score", 0.0))))
    score = raw_score if accepted else 0.0
    result = {
        "source": "wsclaw_llm_judge",
        "score": score,
        "raw_score": raw_score,
        "judge_result_complete": accepted,
        "judge_schema_complete": bool(judge.get("judge_schema_complete", True)),
        "judge_exit_code": int(judge.get("judge_exit_code", 1)),
        "judge_timed_out": bool(judge.get("judge_timed_out")),
        "judge_worker_host": judge.get("judge_worker_host"),
        "user_feedback": user_feedback,
        "artifacts_found": sum(int(item.get("exists", False)) for item in artifacts),
        "artifacts_required": len(artifacts),
        "native_result": judge,
        "judge_completion_verified": completion,
    }
    (run_dir / "judge_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return score, result


def _local_wsclaw_judge_urls(configured_urls: tuple[str, ...]) -> tuple[str, ...]:
    sglang_urls = tuple(
        item.strip()
        for item in os.environ.get("CLAWGYM_SGLANG_WORKER_URLS", "").split(",")
        if item.strip()
    )
    if not sglang_urls:
        return configured_urls
    local_hosts = {
        urllib.parse.urlsplit(url).hostname
        for url in sglang_urls
        if urllib.parse.urlsplit(url).hostname
    }
    local_urls = tuple(
        url
        for url in configured_urls
        if urllib.parse.urlsplit(url).hostname in local_hosts
    )
    if not local_urls and len(configured_urls) == 1:
        return configured_urls
    if len(local_urls) != 1:
        raise RuntimeError(
            "WSClaw judge requires a valid local worker URL; "
            f"sglang_hosts={sorted(local_hosts)} local_judge_urls={len(local_urls)}"
        )
    return local_urls


def _persist_wsclaw_judge_result(run_dir: Path, session_log_dir: Path) -> None:
    source = run_dir / "judge_result.json"
    if not source.is_file():
        return
    destination = session_log_dir / "judge_result.json"
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _persist_researchclaw_reward(run_dir: Path, session_log_dir: Path) -> None:
    """Keep the authoritative judge record outside disposable workspaces."""
    source = run_dir / "researchclaw_reward.json"
    if not source.is_file():
        return
    temporary: Path | None = None
    try:
        destination = session_log_dir / "researchclaw_reward.json"
        temporary = session_log_dir / f".researchclaw_reward.json.tmp-{os.getpid()}-{time.time_ns()}"
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass


def _recover_agent_activity(
    session_log_dir: str | Path,
    workspace_dir: str | Path,
) -> dict[str, Any] | None:
    """Recover the complete solver LLM/tool loop for one task workspace."""
    session_dir = Path(session_log_dir)
    workspace = str(Path(workspace_dir))
    selected: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(session_dir.glob("*_agent-end.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not _agent_end_mentions_workspace(payload, workspace):
            continue
        selected.append((path, payload))
    if not selected:
        return None

    turn_count = 0
    tool_call_count = 0
    for _path, payload in selected:
        steps = payload.get("llm_steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            turn_count += 1
            tool_calls = step.get("tool_calls")
            if isinstance(tool_calls, list):
                tool_call_count += len(tool_calls)
    if turn_count <= 0:
        return None
    return {
        "turn_count": turn_count,
        "tool_call_count": tool_call_count,
        "source": "agent_end",
        "payload_count": len(selected),
    }


def _recover_training_tool_loop_error(
    session_log_dir: str | Path,
    *,
    ignore_names: set[str] | None = None,
) -> str | None:
    """Return a tool-loop error from the latest trace in this exact session."""
    session_dir = Path(session_log_dir)
    ignored = ignore_names or set()
    for path in reversed(sorted(session_dir.glob("*_agent-end.json"))):
        if path.name in ignored:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("stop_reason") != "tool_loop":
            return None
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        return "Repeated tool loop detected during training"
    return None


def _agent_end_mentions_workspace(payload: dict[str, Any], workspace: str) -> bool:
    steps = payload.get("llm_steps")
    if not isinstance(steps, list):
        return False
    for step in steps:
        if not isinstance(step, dict):
            continue
        messages = step.get("request_messages")
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and workspace in content:
                return True
    return False


def _format_clawgym_task_prompt(prompt: str) -> str:
    mode = os.environ.get("CLAWGYM_TASK_PROMPT_MODE", "original").strip().lower()
    if mode in _ORIGINAL_CLAWGYM_TASK_PROMPT_MODES:
        return prompt
    return f"{_RL_TASK_INSTRUCTION}\n\nUser task: \n{prompt}"


def _append_extra_context(prompt: str, extra_context: str | None) -> str:
    if not isinstance(extra_context, str) or not extra_context.strip():
        return prompt
    return f"{extra_context.strip()}\n\n# Current task\n\n{prompt}"


def _write_nanobot_config(
    *,
    workspace_dir: Path,
    run_dir: Path,
    adapter_base_url: str,
    session_id: str,
    turn_type: str,
    sampling_params: dict[str, Any],
) -> Path:
    max_tokens_default = int(sampling_params.get("max_new_tokens") or 8192)
    return nanobot_runtime.write_nanobot_config(
        workspace_dir=workspace_dir,
        run_dir=run_dir,
        adapter_base_url=adapter_base_url,
        session_id=session_id,
        turn_type=turn_type,
        sampling_params=sampling_params,
        model_id=_nanobot_model_id(turn_type, session_id),
        context_window=_positive_int_env("OPENCLAW_CONTEXT_WINDOW", 65536),
        max_tokens=_positive_int_env("OPENCLAW_MAX_TOKENS", max_tokens_default),
    )


def _nanobot_model_id(turn_type: str, session_id: str) -> str:
    default = f"{turn_type}__{session_id}"
    if not _env_truthy("CLAWGYM_ALLOW_NANOBOT_MODEL_ID_OVERRIDE", False):
        return default
    override = os.environ.get("NANOBOT_MODEL_ID", "").strip()
    return override or default


def _turn_type(evaluation: bool) -> str:
    if not evaluation:
        return os.environ.get("CLAWGYM_NANOBOT_TRAIN_TURN_TYPE", "main").strip() or "main"
    return os.environ.get("CLAWGYM_NANOBOT_EVAL_TURN_TYPE", "side").strip() or "side"


def _run_nanobot_turn(
    *,
    workspace_dir: Path,
    config_path: Path,
    session_id: str,
    turn_type: str,
    message: str,
    log_dir: Path,
    timeout: int | None,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return nanobot_runtime.run_nanobot_turn(
        workspace_dir=workspace_dir,
        config_path=config_path,
        session_id=session_id,
        turn_type=turn_type,
        message=message,
        log_dir=log_dir,
        timeout=timeout,
        env_overrides=env_overrides,
    )


def _nanobot_shell_script(
    *,
    workspace_dir: Path,
    config_path: Path,
    session_id: str,
    turn_type: str,
    message: str,
    log_dir: Path,
    env_overrides: dict[str, str] | None = None,
) -> str:
    return nanobot_runtime.nanobot_shell_script(
        workspace_dir=workspace_dir,
        config_path=config_path,
        session_id=session_id,
        turn_type=turn_type,
        message=message,
        log_dir=log_dir,
        env_overrides=env_overrides,
    )


def _write_turn_logs(run_dir: Path, turn: NanobotTurnResult) -> None:
    nanobot_runtime.write_turn_logs(run_dir, turn)


def _run_reward(task: tasks.TaskSpec, workspace_dir: Path, run_dir: Path) -> float:
    if _is_researchclaw_task(task, workspace_dir):
        return researchclaw_reward.run(task, workspace_dir, run_dir)

    check_path = workspace_dir / "reward" / "check.py"
    if not check_path.exists():
        return 0.0

    python_exe = _nanobot_python_executable()
    if not python_exe.exists():
        python_exe = Path(sys.executable)
    env_bin = python_exe.parent
    pythonpath = _nanobot_pythonpath()
    script = "\n".join(
        [
            "set -euo pipefail",
            "unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M",
            "unset PYTHONPATH",
            f"export PATH={shlex.quote(str(env_bin))}:$PATH",
        ]
    )
    if pythonpath:
        script += f"\nexport PYTHONPATH={shlex.quote(pythonpath)}"
    script += f"\n{shlex.quote(str(python_exe))} {shlex.quote(str(check_path))} {shlex.quote(str(workspace_dir))}"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=_reward_timeout_seconds(),
        cwd=str(workspace_dir),
    )
    (run_dir / "reward.stdout").write_text(result.stdout or "", encoding="utf-8")
    (run_dir / "reward.stderr").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        return 0.0
    return _score_reward_payload(result.stdout)


def _is_researchclaw_task(task: tasks.TaskSpec, workspace_dir: Path) -> bool:
    if str(task.metadata.get("source", "")).strip().lower() == "researchclaw":
        return True
    reward_dir = workspace_dir / "reward"
    return (reward_dir / "eval_config.json").is_file() and (
        reward_dir / "llmjudge" / "rubric.json"
    ).is_file()


def _score_reward_payload(raw: str) -> float:
    data = json.loads((raw or "").strip())
    if isinstance(data, dict):
        if not data:
            return 0.0
        return sum(float(value) for value in data.values()) / len(data)
    if isinstance(data, (int, float)):
        return float(data)
    return 0.0


def _write_run_summary(run_dir: Path, result: NanobotRunResult) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        destination = run_dir / "summary.json"
        temporary = run_dir / f".summary.json.tmp-{os.getpid()}-{time.time_ns()}"
        temporary.write_text(json.dumps(result.metadata(), indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        pass


def _write_session_run_status(
    session_log_dir: Path,
    result: NanobotRunResult,
    *,
    rollout_id: int,
    task_index: int,
    group_index: int | None,
    error_type: str | None,
) -> None:
    """Persist one bounded status record even when nanobot never writes logs."""
    try:
        session_log_dir.mkdir(parents=True, exist_ok=True)
        error_summary = ""
        if result.error:
            if error_type == "TimeoutExpired" and " timed out after " in result.error:
                timeout_value = result.error.rsplit(" timed out after ", 1)[1].split(" seconds", 1)[0].strip()
                error_summary = f"nanobot turn timed out after {timeout_value} seconds"
            else:
                error_summary = result.error[:2048]
        payload = {
            "session_id": result.session_id,
            "task_id": result.task_id,
            "rollout_id": int(rollout_id),
            "task_index": int(task_index),
            "group_index": None if group_index is None else int(group_index),
            "status": result.status,
            "reward": float(result.reward),
            "duration_s": float(result.duration_s),
            "code_verifier_duration_s": float(result.code_verifier_duration_s),
            "turn_count": len(result.turns),
            "error_type": error_type,
            "error": error_summary,
        }
        if result.task_metadata is not None:
            payload["task_metadata"] = result.task_metadata
        if result.researchclaw_reward is not None:
            payload["researchclaw_reward"] = result.researchclaw_reward
        if result.agent_activity is not None:
            payload["agent_activity"] = result.agent_activity
        destination = session_log_dir / "run-status.json"
        temporary = session_log_dir / f".run-status.json.tmp-{os.getpid()}-{time.time_ns()}"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        pass


def _cleanup_run_dir(run_dir: Path, *, evaluation: bool, status: str) -> None:
    if status != "completed" or evaluation:
        return
    if _env_truthy("CLAWGYM_DROP_TRAIN_NANOBOT_LOGS", False):
        shutil.rmtree(run_dir, ignore_errors=True)


def _new_task_run_dir(*, rollout_id: int, task_index: int) -> Path:
    root = _nanobot_work_root()
    return Path(tempfile.mkdtemp(prefix=f"rollout-{rollout_id}-task-{task_index}-", dir=str(root)))


def _nanobot_work_root() -> Path:
    root = Path(os.environ.get("NANOBOT_WORKDIR", str(Path(tempfile.gettempdir()) / "clawgym-nanobot")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _nanobot_log_dir(args: Any) -> Path:
    explicit = os.environ.get("NANOBOT_LOG_DIR", "").strip() or os.environ.get("CLAWARENA_NANOBOT_LOG_DIR", "").strip()
    if explicit:
        return Path(explicit)
    if getattr(args, "save", None):
        return Path(args.save) / "nanobot_logs"
    return Path(tempfile.gettempdir()) / "clawgym-nanobot-logs"


def _nanobot_root() -> Path:
    return Path(os.environ.get("NANOBOT_ROOT", str(DEFAULT_NANOBOT_ROOT)))


def _nanobot_conda_root() -> Path:
    return Path(os.environ.get("NANOBOT_CONDA_ROOT", str(DEFAULT_CONDA_ROOT)))


def _nanobot_conda_env_name() -> str:
    return os.environ.get("NANOBOT_CONDA_ENV_NAME", "nanobot-openclaw")


def _nanobot_env_dir() -> Path:
    env_name = _nanobot_conda_env_name()
    env_path = Path(env_name)
    if env_path.is_absolute():
        return env_path
    return _nanobot_conda_root() / "envs" / env_name


def _nanobot_python_executable() -> Path:
    return _nanobot_env_dir() / "bin" / "python"


def _nanobot_runner_path() -> Path:
    explicit = os.environ.get("NANOBOT_RL_RUNNER", "").strip()
    if explicit:
        return Path(explicit)
    return _nanobot_root() / "nanobot" / "rl_runner.py"


def _nanobot_pythonpath() -> str:
    entries: list[str] = []
    if _env_truthy("NANOBOT_IMPORT_LOCAL_PACKAGE", False):
        entries.append(str(_nanobot_root()))
    extra = os.environ.get("NANOBOT_EXTRA_PYTHONPATH", "").strip()
    if extra:
        entries.extend(part for part in extra.split(os.pathsep) if part)
    return os.pathsep.join(entries)


def _openai_api_base(adapter_base_url: str) -> str:
    return adapter_base_url.rstrip("/") + "/v1"


def _nanobot_workspace_display(workspace_dir: Path) -> str:
    explicit = (
        os.environ.get("NANOBOT_WORKSPACE_DISPLAY", "").strip()
        or os.environ.get("CLAWARENA_NANOBOT_WORKSPACE_DISPLAY", "").strip()
    )
    return explicit or str(workspace_dir)


def _nanobot_web_enabled() -> bool:
    return os.environ.get("NANOBOT_WEB_ENABLE", "1") == "1"


def _nanobot_web_proxy() -> str | None:
    if not _nanobot_web_enabled():
        return None
    explicit = os.environ.get("NANOBOT_WEB_PROXY")
    if explicit is not None:
        return explicit.strip() or None
    for name in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "WANDB_HTTPS_PROXY",
        "WANDB_HTTP_PROXY",
        "WANDB_PROXY_URL",
    ):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _nanobot_default_no_proxy() -> str:
    return ",".join(
        [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "10.0.0.0/8",
            "100.0.0.0/8",
            "${MASTER_ADDR:-}",
            "${RAY_NODE_IP:-}",
            "${PROXY_HOST_IP:-}",
            "${TRAIN_NODE_HOST:-}",
            "${HOSTNAME:-}",
        ]
    )


def _nanobot_no_proxy_value() -> str:
    if not _nanobot_web_enabled():
        return "*"
    return f"${{CCR_NO_PROXY:-{_nanobot_default_no_proxy()}}}"


def _read_file_tail_text(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _max_turns(evaluation: bool) -> int:
    names = ("OPENCLAW_EVAL_MAX_TURNS", "OPENCLAW_MAX_TURNS") if evaluation else ("OPENCLAW_TRAIN_MAX_TURNS", "OPENCLAW_MAX_TURNS")
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return max(1, int(raw))
    return 1


def _turn_timeout_seconds(evaluation: bool) -> int | None:
    names = (
        ("NANOBOT_EVAL_TURN_TIMEOUT", "NANOBOT_TURN_TIMEOUT", "NANOBOT_TASK_TIMEOUT")
        if evaluation
        else ("NANOBOT_TRAIN_TURN_TIMEOUT", "NANOBOT_TURN_TIMEOUT", "NANOBOT_TASK_TIMEOUT")
    )
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            value = int(raw)
            return None if value <= 0 else value
    return 1800


def _reward_timeout_seconds() -> int:
    return _positive_int_env("NANOBOT_REWARD_TIMEOUT", 120)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(default)


def _float_env_first(names: tuple[str, ...], default: Any) -> float:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float(default)


def _int_env_first(names: tuple[str, ...], default: Any) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return int(default)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
