from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import tasks


DIRECT_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. Do not call tools or request external files. "
    "Use only the prompt contents. You may provide concise analysis, but your "
    "response must end with the requested JSON scores object."
)
AGENTIC_JUDGE_RESULT_PATH = "_judge/judge_result.json"
VALID_SCORE_MODES = {"code", "llmjudge", "hybrid"}
VALID_LLMJUDGE_BACKENDS = {"direct", "agentic", "hybrid_agentic"}
SENSITIVE_EXACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}
NON_SECRET_TOKEN_KEYS = {
    "max_tokens",
    "max_completion_tokens",
    "context_tokens",
    "output_tokens",
    "input_tokens",
}


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    api_key: str
    model: str
    timeout_sec: int
    max_tokens: int
    temperature: float


def run(task: tasks.TaskSpec, workspace_dir: Path, run_dir: Path) -> float:
    output_observation = required_output_observation(task, workspace_dir)
    try:
        score, payload = _run(task, workspace_dir, run_dir)
        payload.update(output_observation)
        llmjudge_reward = payload.get("llmjudge_reward")
        if isinstance(llmjudge_reward, dict):
            from .researchclaw_feedback import maybe_attach_natural_feedback

            feedback = maybe_attach_natural_feedback(
                task_request=task.user_query,
                llmjudge_reward=llmjudge_reward,
                run_dir=run_dir,
            )
            if feedback is not None:
                payload["privileged_feedback"] = feedback
                payload["user_feedback"] = str(feedback.get("text") or "").strip()
        _write_json(run_dir / "researchclaw_reward.json", payload)
        (run_dir / "reward.stdout").write_text(
            json.dumps({"score": score}, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "reward.stderr").write_text("", encoding="utf-8")
        return score
    except Exception as exc:
        error_payload = {
            "task_id": task.task_id,
            "error": f"{type(exc).__name__}: {exc}",
            "workspace_dir": str(workspace_dir),
            "run_dir": str(run_dir),
            **output_observation,
        }
        _write_json(run_dir / "researchclaw_reward.error.json", error_payload)
        (run_dir / "reward.stdout").write_text("", encoding="utf-8")
        (run_dir / "reward.stderr").write_text(error_payload["error"] + "\n", encoding="utf-8")
        raise


def required_output_observation(task: tasks.TaskSpec, workspace_dir: Path) -> dict[str, Any]:
    """Report whether the task's primary requested output file was created."""
    try:
        config_path = task.task_dir / "reward" / "eval_config.json"
        config = _read_json_dict(config_path) if config_path.is_file() else {}
        required_path = _required_output_path(task, config)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {
        "required_output_path": required_path,
        "required_output_exists": _safe_workspace_file(workspace_dir, required_path) is not None,
    }


def _run(task: tasks.TaskSpec, workspace_dir: Path, run_dir: Path) -> tuple[float, dict[str, Any]]:
    reward_dir = workspace_dir / "reward"
    if not reward_dir.is_dir():
        raise FileNotFoundError(f"ResearchClaw reward dir not found: {reward_dir}")

    config = _read_json_dict(reward_dir / "eval_config.json")
    mode = _score_mode(config)
    code_score: float | None = None
    llmjudge_score: float | None = None
    code_reward: dict[str, Any] | None = None
    llmjudge_reward: dict[str, Any] | None = None
    judge_io_files: dict[str, Any] | None = None

    if mode in {"code", "hybrid"}:
        code_score, code_reward = _run_codebase_reward(reward_dir, workspace_dir, run_dir, config)

    if mode in {"llmjudge", "hybrid"}:
        llmjudge_score, llmjudge_reward, judge_io_files = _run_llmjudge_reward(
            task,
            reward_dir,
            workspace_dir,
            run_dir,
            config,
        )

    if mode == "code":
        final_score = _require_score(code_score, "code")
        requested_weights = {"code": 1.0, "llmjudge": 0.0}
    elif mode == "llmjudge":
        final_score = _require_score(llmjudge_score, "llmjudge")
        requested_weights = {"code": 0.0, "llmjudge": 1.0}
    else:
        code_weight = _float_setting("RESEARCHCLAW_CODE_WEIGHT", config, "code_weight", 0.1)
        llmjudge_weight = _float_setting("RESEARCHCLAW_LLMJUDGE_WEIGHT", config, "llmjudge_weight", 0.9)
        if code_weight < 0 or llmjudge_weight < 0:
            raise ValueError("ResearchClaw hybrid weights must be non-negative")
        total_weight = code_weight + llmjudge_weight
        if total_weight <= 0:
            raise ValueError("ResearchClaw hybrid weights must sum to a positive value")
        requested_weights = {
            "code": code_weight / total_weight,
            "llmjudge": llmjudge_weight / total_weight,
        }
        final_score = (
            _require_score(code_score, "code") * requested_weights["code"]
            + _require_score(llmjudge_score, "llmjudge") * requested_weights["llmjudge"]
        )

    payload = {
        "task_id": task.task_id,
        "source": "researchclaw",
        "requested_score_mode": mode,
        "final_score": final_score,
        "code_score": code_score,
        "llmjudge_score": llmjudge_score,
        "requested_score_weights": requested_weights,
        "required_output_path": _required_output_path(task, config),
        "config": {
            "eval_config": config,
            "env": _visible_env_config(),
        },
        "code_reward": code_reward,
        "llmjudge_reward": llmjudge_reward,
    }
    if judge_io_files is not None:
        payload["judge_io_files"] = judge_io_files
    return float(final_score), payload


def _score_mode(config: dict[str, Any]) -> str:
    raw = os.environ.get("RESEARCHCLAW_SCORE_MODE") or str(config.get("default_score_mode") or "code")
    mode = raw.strip().lower()
    if mode not in VALID_SCORE_MODES:
        raise ValueError(f"unknown ResearchClaw score mode: {raw!r}; expected one of {sorted(VALID_SCORE_MODES)}")
    return mode


def _run_codebase_reward(
    reward_dir: Path,
    workspace_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    entry = str(config.get("codebase_entry") or "codebase/check.py")
    check_path = _reward_entry_path(reward_dir, entry)
    timeout = _positive_int_env("NANOBOT_REWARD_TIMEOUT", 120)
    started = time.time()
    result = subprocess.run(
        [sys.executable, str(check_path), str(workspace_dir)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(workspace_dir),
        check=False,
    )
    duration = time.time() - started
    (run_dir / "researchclaw_codebase.stdout").write_text(result.stdout or "", encoding="utf-8")
    (run_dir / "researchclaw_codebase.stderr").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"ResearchClaw codebase reward failed with exit {result.returncode}: "
            f"{_tail_text(result.stderr or result.stdout)}"
        )
    parsed = _parse_json_stdout(result.stdout, "ResearchClaw codebase reward")
    score = _mean_numeric(parsed)
    if score is None:
        raise ValueError("ResearchClaw codebase reward produced no numeric score")
    return score, {
        "available": True,
        "method": f"{sys.executable} {check_path} <workspace>",
        "entry": entry,
        "returncode": result.returncode,
        "score": score,
        "parsed": parsed,
        "stdout_path": str(run_dir / "researchclaw_codebase.stdout"),
        "stderr_path": str(run_dir / "researchclaw_codebase.stderr"),
        "duration_sec": round(duration, 3),
    }


def _run_llmjudge_reward(
    task: tasks.TaskSpec,
    reward_dir: Path,
    workspace_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    rubric = _read_json_dict(reward_dir / "llmjudge" / "rubric.json")
    output_path = _required_output_path(task, config)
    rules = _rubric_to_rules(rubric, output_path)
    if not rules:
        raise ValueError("ResearchClaw llmjudge requested but rubric has no llm_judge items")

    max_chars = _positive_int_env("RESEARCHCLAW_JUDGE_MAX_CHARS_PER_FILE", 12_000)
    backend = _llmjudge_backend()
    operation: Callable[[], tuple[float, dict[str, Any], dict[str, Any]]]
    if backend == "direct":
        operation = lambda: _run_direct_llmjudge_reward(
            task=task,
            reward_dir=reward_dir,
            workspace_dir=workspace_dir,
            run_dir=run_dir,
            rules=rules,
            config=config,
            max_chars=max_chars,
        )
    elif backend == "agentic":
        operation = lambda: _run_agentic_llmjudge_reward(
            task=task,
            reward_dir=reward_dir,
            workspace_dir=workspace_dir,
            run_dir=run_dir,
            rubric=rubric,
            rules=rules,
            config=config,
            max_chars=max_chars,
        )
    elif backend == "hybrid_agentic":
        operation = lambda: _run_hybrid_agentic_llmjudge_reward(
            task=task,
            reward_dir=reward_dir,
            workspace_dir=workspace_dir,
            run_dir=run_dir,
            rubric=rubric,
            rules=rules,
            config=config,
            max_chars=max_chars,
        )
    else:
        raise AssertionError(f"unhandled ResearchClaw llmjudge backend: {backend}")

    return _run_judge_with_retries(operation=operation, run_dir=run_dir)


def _run_judge_with_retries(
    *,
    operation: Callable[[], tuple[float, dict[str, Any], dict[str, Any]]],
    run_dir: Path,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    max_retries = _nonnegative_int_env("RESEARCHCLAW_JUDGE_MAX_RETRIES", 3)
    backoff_sec = _nonnegative_float_env("RESEARCHCLAW_JUDGE_RETRY_BACKOFF_SEC", 1.0)
    max_attempts = max_retries + 1
    attempts: list[dict[str, Any]] = []
    manifest_path = run_dir / "researchclaw_judge_attempts.json"

    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            score, reward, io_files = operation()
        except Exception as exc:
            archive_dir = _archive_judge_attempt_files(run_dir, attempt)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_sec": round(time.time() - started, 3),
                    "artifacts_dir": str(archive_dir),
                }
            )
            _write_json(
                manifest_path,
                {
                    "status": "exhausted" if attempt >= max_attempts else "retrying",
                    "max_retries": max_retries,
                    "max_attempts": max_attempts,
                    "attempts": attempts,
                },
            )
            if attempt >= max_attempts:
                raise
            delay = backoff_sec * (2 ** (attempt - 1))
            if delay > 0:
                time.sleep(delay)
            continue

        attempts.append(
            {
                "attempt": attempt,
                "status": "completed",
                "duration_sec": round(time.time() - started, 3),
            }
        )
        retry_info = {
            "status": "completed",
            "max_retries": max_retries,
            "max_attempts": max_attempts,
            "attempt_count": attempt,
            "retry_count": attempt - 1,
            "attempts": attempts,
        }
        _write_json(manifest_path, retry_info)
        reward = dict(reward)
        reward["judge_retry"] = retry_info
        io_files = dict(io_files)
        io_files["attempt_log"] = str(manifest_path)
        return score, reward, io_files

    raise AssertionError("ResearchClaw judge retry loop exited unexpectedly")


def _archive_judge_attempt_files(run_dir: Path, attempt: int) -> Path:
    archive_dir = run_dir / "researchclaw_judge_attempts" / f"attempt-{attempt:02d}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "researchclaw_judge*.json",
        "researchclaw_agentic_judge*.json",
        "researchclaw_agentic_judge*.stdout",
        "researchclaw_agentic_judge*.stderr",
    ):
        for path in run_dir.glob(pattern):
            if path.is_file() and path.name != "researchclaw_judge_attempts.json":
                shutil.copy2(path, archive_dir / path.name)
    return archive_dir


def _llmjudge_backend() -> str:
    raw = os.environ.get("RESEARCHCLAW_LLMJUDGE_BACKEND", "agentic")
    backend = raw.strip().lower()
    if backend not in VALID_LLMJUDGE_BACKENDS:
        raise ValueError(
            f"unknown ResearchClaw llmjudge backend: {raw!r}; expected one of {sorted(VALID_LLMJUDGE_BACKENDS)}"
        )
    return backend


def _run_direct_llmjudge_reward(
    *,
    task: tasks.TaskSpec,
    reward_dir: Path,
    workspace_dir: Path,
    run_dir: Path,
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    max_chars: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    output_paths = _judge_output_paths(task, reward_dir, rules, config)
    target_files = _read_judge_target_files(workspace_dir, output_paths, max_chars)
    reference_answer = _read_reference_answer(reward_dir, max_chars) if _env_truthy("RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER") else None
    reference_answer = reference_answer or None
    prompt = _build_rubric_prompt(
        user_task=task.user_query,
        rules=rules,
        target_files=target_files,
        reference_answer=reference_answer,
    )
    judge = _judge_config(selection_key=f"{task.task_id}:{run_dir.name}")
    judge_request = {
        "endpoint": judge.base_url.rstrip("/") + "/chat/completions",
        "request": _redact(
            {
                "model": judge.model,
                "messages": _build_judge_messages(prompt),
                "temperature": judge.temperature,
                "max_tokens": judge.max_tokens,
            }
        ),
        "rules": rules,
        "target_files": target_files,
        "reference_answer_included": reference_answer is not None,
    }
    _write_json(run_dir / "researchclaw_judge_request.json", judge_request)
    try:
        response = _call_openai_compatible_judge(judge=judge, prompt=prompt, rules=rules)
    except Exception as exc:
        _write_json(
            run_dir / "researchclaw_judge_response.json",
            {
                "available": True,
                "error": f"{type(exc).__name__}: {exc}",
                "model": judge.model,
                "base_url": judge.base_url,
            },
        )
        raise
    _write_json(run_dir / "researchclaw_judge_response.json", response)

    score = response.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise ValueError(f"ResearchClaw llmjudge produced no numeric score: {response.get('error') or response.get('parsed')}")
    return float(score), {
        "available": True,
        "backend": "direct",
        "score": float(score),
        "model": judge.model,
        "base_url": judge.base_url,
        "rules": rules,
        "target_files": target_files,
        "reference_answer_included": reference_answer is not None,
        "parsed": response.get("parsed"),
        "score_normalization": response.get("score_normalization"),
        "judge_request_path": str(run_dir / "researchclaw_judge_request.json"),
        "judge_response_path": str(run_dir / "researchclaw_judge_response.json"),
    }, {
        "backend": "direct",
        "request": str(run_dir / "researchclaw_judge_request.json"),
        "response": str(run_dir / "researchclaw_judge_response.json"),
    }


def _run_hybrid_agentic_llmjudge_reward(
    *,
    task: tasks.TaskSpec,
    reward_dir: Path,
    workspace_dir: Path,
    run_dir: Path,
    rubric: dict[str, Any],
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    max_chars: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    output_paths = _judge_output_paths(task, reward_dir, rules, config)
    target_files = _read_judge_target_files(workspace_dir, output_paths, max_chars)
    reference_answer = _read_reference_answer(reward_dir, max_chars) if _env_truthy("RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER") else None
    reference_answer = reference_answer or None
    judge = _judge_config(selection_key=f"{task.task_id}:{run_dir.name}")
    prompt = _build_triage_judge_prompt(
        user_task=task.user_query,
        rules=rules,
        target_files=target_files,
        reference_answer=reference_answer,
    )
    triage_request = {
        "endpoint": judge.base_url.rstrip("/") + "/chat/completions",
        "request": _redact(
            {
                "model": judge.model,
                "messages": _build_judge_messages(prompt),
                "temperature": judge.temperature,
                "max_tokens": judge.max_tokens,
            }
        ),
        "rules": rules,
        "target_files": target_files,
        "reference_answer_included": reference_answer is not None,
    }
    _write_json(run_dir / "researchclaw_judge_triage_request.json", triage_request)
    try:
        triage_response = _call_openai_compatible_judge(
            judge=judge,
            prompt=prompt,
            rules=rules,
            validate_scores=False,
        )
    except Exception as exc:
        _write_json(
            run_dir / "researchclaw_judge_triage_response.json",
            {
                "available": True,
                "backend": "hybrid_agentic",
                "phase": "triage",
                "error": f"{type(exc).__name__}: {exc}",
                "model": judge.model,
                "base_url": judge.base_url,
            },
        )
        raise
    triage_response["backend"] = "hybrid_agentic"
    triage_response["phase"] = "triage"
    _write_json(run_dir / "researchclaw_judge_triage_response.json", triage_response)
    parsed = triage_response.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError(f"ResearchClaw hybrid judge triage produced no JSON object: {triage_response}")

    needs_evidence = _triage_needs_evidence(parsed, rules)
    direct_rule_ids = [rule_id for rule_id in _rule_ids(rules) if rule_id not in needs_evidence]
    defaulted_missing_output_scores: list[str] = []
    if target_files and not any(info.get("exists") for info in target_files.values()):
        raw_scores = dict(parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {})
        direct_rules = _select_rules_by_ids(rules, direct_rule_ids)
        for index, (rule, rule_id) in enumerate(zip(direct_rules, direct_rule_ids), 1):
            has_numeric_score = any(
                _numeric(raw_scores.get(key)) is not None
                for key in _rule_score_keys(rule, index, rule_id=rule_id)
                if key in raw_scores
            )
            if not has_numeric_score and _score_in_allowed_anchors(0.0, _rule_allowed_scores(rule)):
                raw_scores[rule_id] = 0
                defaulted_missing_output_scores.append(rule_id)
        if defaulted_missing_output_scores:
            parsed = {**parsed, "scores": raw_scores}
    if direct_rule_ids:
        direct_rules = _select_rules_by_ids(rules, direct_rule_ids)
        _score_judge_scores(parsed, direct_rules, rule_ids=direct_rule_ids)

    agentic_response: dict[str, Any] | None = None
    agentic_io_files: dict[str, Any] | None = None
    merged_scores = dict(parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {})
    if needs_evidence:
        evidence_requests = _triage_evidence_requests(parsed, needs_evidence)
        agentic_response, agentic_io_files = _run_targeted_agentic_evidence_judge(
            task=task,
            reward_dir=reward_dir,
            solver_workspace_dir=workspace_dir,
            run_dir=run_dir,
            rubric=rubric,
            rules=rules,
            config=config,
            max_chars=max_chars,
            target_files=target_files,
            reference_answer_included=reference_answer is not None,
            requested_rule_ids=needs_evidence,
            evidence_requests=evidence_requests,
        )
        agentic_scores = agentic_response["score_normalization"]["matched_scores"]
        for rule_id in needs_evidence:
            merged_scores[rule_id] = agentic_scores[rule_id]

    merged_parsed = {
        "scores": merged_scores,
        "notes": _merge_judge_notes(parsed, agentic_response),
        "triage_needs_evidence": needs_evidence,
        "triage_defaulted_missing_output_scores": defaulted_missing_output_scores,
    }
    score, normalization = _score_judge_scores(merged_parsed, rules)
    final_response = {
        "available": True,
        "backend": "hybrid_agentic",
        "score": score,
        "parsed": merged_parsed,
        "score_normalization": normalization,
        "triage": triage_response,
        "agentic": agentic_response,
    }
    _write_json(run_dir / "researchclaw_judge_hybrid_response.json", final_response)

    io_files = {
        "backend": "hybrid_agentic",
        "triage_request": str(run_dir / "researchclaw_judge_triage_request.json"),
        "triage_response": str(run_dir / "researchclaw_judge_triage_response.json"),
        "response": str(run_dir / "researchclaw_judge_hybrid_response.json"),
    }
    if agentic_io_files is not None:
        io_files["agentic"] = agentic_io_files

    return float(score), {
        "available": True,
        "backend": "hybrid_agentic",
        "score": float(score),
        "rules": rules,
        "target_files": target_files,
        "reference_answer_included": reference_answer is not None,
        "triage": {
            "parsed": parsed,
            "needs_evidence": needs_evidence,
            "defaulted_missing_output_scores": defaulted_missing_output_scores,
            "request_path": str(run_dir / "researchclaw_judge_triage_request.json"),
            "response_path": str(run_dir / "researchclaw_judge_triage_response.json"),
        },
        "agentic": agentic_response,
        "parsed": merged_parsed,
        "score_normalization": normalization,
        "judge_response_path": str(run_dir / "researchclaw_judge_hybrid_response.json"),
    }, io_files


def _run_targeted_agentic_evidence_judge(
    *,
    task: tasks.TaskSpec,
    reward_dir: Path,
    solver_workspace_dir: Path,
    run_dir: Path,
    rubric: dict[str, Any],
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    max_chars: int,
    target_files: dict[str, dict[str, Any]],
    reference_answer_included: bool,
    requested_rule_ids: list[str],
    evidence_requests: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    judge = _judge_config(selection_key=f"{task.task_id}:{run_dir.name}")
    requested_rules = _select_rules_by_ids(rules, requested_rule_ids)
    paths = _prepare_agentic_judge_workspace(
        task=task,
        reward_dir=reward_dir,
        solver_workspace_dir=solver_workspace_dir,
        run_dir=run_dir,
        rubric=rubric,
        rules=requested_rules,
        config=config,
        max_chars=max_chars,
        rule_ids=requested_rule_ids,
    )
    config_path = _write_agentic_judge_nanobot_config(paths["workspace"], paths["state_dir"], judge)
    message = _build_targeted_agentic_judge_prompt(
        rules=requested_rules,
        rule_ids=requested_rule_ids,
        target_files=target_files,
        evidence_requests=evidence_requests,
        reference_answer_included=reference_answer_included,
    )
    request_payload = {
        "backend": "hybrid_agentic",
        "phase": "targeted_agentic",
        "requested_rule_ids": requested_rule_ids,
        "evidence_requests": evidence_requests,
        "model": judge.model,
        "base_url": judge.base_url,
        "temperature": judge.temperature,
        "max_tokens": _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS", judge.max_tokens),
        "timeout_sec": _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT", judge.timeout_sec),
        "workspace": str(paths["workspace"]),
        "config_path": str(config_path),
        "message": message,
    }
    _write_json(run_dir / "researchclaw_agentic_judge_request.json", _redact(request_payload))

    started = time.time()
    proc = _run_agentic_judge_nanobot(
        workspace_dir=paths["workspace"],
        config_path=config_path,
        log_dir=paths["log_dir"],
        session_id=f"researchclaw-judge-{task.task_id}",
        message=message,
        timeout=_positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT", judge.timeout_sec),
    )
    duration = time.time() - started
    (run_dir / "researchclaw_agentic_judge.stdout").write_text(proc.stdout or "", encoding="utf-8")
    (run_dir / "researchclaw_agentic_judge.stderr").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"ResearchClaw targeted agentic judge failed with exit {proc.returncode}: "
            f"{_tail_text(proc.stderr or proc.stdout)}"
        )

    result_path = paths["workspace"] / AGENTIC_JUDGE_RESULT_PATH
    if not result_path.is_file():
        raise FileNotFoundError(f"ResearchClaw targeted agentic judge did not write {result_path}")
    parsed = _read_agentic_judge_result(result_path)
    score, normalization = _score_judge_scores(parsed, requested_rules, rule_ids=requested_rule_ids)
    response = {
        "available": True,
        "backend": "hybrid_agentic",
        "phase": "targeted_agentic",
        "score": score,
        "parsed": parsed,
        "score_normalization": normalization,
        "requested_rule_ids": requested_rule_ids,
        "evidence_requests": evidence_requests,
        "result_path": str(result_path),
        "workspace": str(paths["workspace"]),
        "log_dir": str(paths["log_dir"]),
        "duration_sec": round(duration, 3),
    }
    _write_json(run_dir / "researchclaw_agentic_judge_response.json", response)
    return response, {
        "backend": "hybrid_agentic",
        "phase": "targeted_agentic",
        "request": str(run_dir / "researchclaw_agentic_judge_request.json"),
        "response": str(run_dir / "researchclaw_agentic_judge_response.json"),
        "workspace": str(paths["workspace"]),
        "logs": str(paths["log_dir"]),
        "stdout": str(run_dir / "researchclaw_agentic_judge.stdout"),
        "stderr": str(run_dir / "researchclaw_agentic_judge.stderr"),
        "result": str(result_path),
    }


def _run_agentic_llmjudge_reward(
    *,
    task: tasks.TaskSpec,
    reward_dir: Path,
    workspace_dir: Path,
    run_dir: Path,
    rubric: dict[str, Any],
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    max_chars: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    judge = _judge_config(selection_key=f"{task.task_id}:{run_dir.name}")
    paths = _prepare_agentic_judge_workspace(
        task=task,
        reward_dir=reward_dir,
        solver_workspace_dir=workspace_dir,
        run_dir=run_dir,
        rubric=rubric,
        rules=rules,
        config=config,
        max_chars=max_chars,
    )
    config_path = _write_agentic_judge_nanobot_config(paths["workspace"], paths["state_dir"], judge)
    strict_score_enum = _env_truthy("RESEARCHCLAW_AGENTIC_JUDGE_STRICT_SCORE_ENUM")
    message = _build_agentic_judge_prompt(
        rules=rules,
        reference_answer_included=paths["reference_answer_included"],
        strict_score_enum=strict_score_enum,
    )
    request_payload = {
        "backend": "agentic",
        "model": judge.model,
        "base_url": judge.base_url,
        "temperature": judge.temperature,
        "max_tokens": _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS", judge.max_tokens),
        "timeout_sec": _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT", judge.timeout_sec),
        "strict_score_enum": strict_score_enum,
        "workspace": str(paths["workspace"]),
        "config_path": str(config_path),
        "message": message,
    }
    _write_json(run_dir / "researchclaw_agentic_judge_request.json", _redact(request_payload))

    started = time.time()
    proc = _run_agentic_judge_nanobot(
        workspace_dir=paths["workspace"],
        config_path=config_path,
        log_dir=paths["log_dir"],
        session_id=f"researchclaw-judge-{task.task_id}",
        message=message,
        timeout=_positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT", judge.timeout_sec),
    )
    duration = time.time() - started
    (run_dir / "researchclaw_agentic_judge.stdout").write_text(proc.stdout or "", encoding="utf-8")
    (run_dir / "researchclaw_agentic_judge.stderr").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"ResearchClaw agentic judge failed with exit {proc.returncode}: "
            f"{_tail_text(proc.stderr or proc.stdout)}"
        )

    result_path = paths["workspace"] / AGENTIC_JUDGE_RESULT_PATH
    if not result_path.is_file():
        raise FileNotFoundError(f"ResearchClaw agentic judge did not write {result_path}")
    parsed = _read_agentic_judge_result(result_path)
    score, normalization = _score_judge_scores(parsed, rules)
    response = {
        "available": True,
        "backend": "agentic",
        "score": score,
        "parsed": parsed,
        "score_normalization": normalization,
        "result_path": str(result_path),
        "workspace": str(paths["workspace"]),
        "log_dir": str(paths["log_dir"]),
        "duration_sec": round(duration, 3),
    }
    _write_json(run_dir / "researchclaw_agentic_judge_response.json", response)
    return float(score), {
        "available": True,
        "backend": "agentic",
        "score": float(score),
        "model": judge.model,
        "base_url": judge.base_url,
        "rules": rules,
        "target_files": _read_judge_target_files(
            workspace_dir,
            _judge_output_paths(task, reward_dir, rules, config),
            max_chars,
        ),
        "reference_answer_included": bool(paths["reference_answer_included"]),
        "parsed": parsed,
        "score_normalization": normalization,
        "judge_request_path": str(run_dir / "researchclaw_agentic_judge_request.json"),
        "judge_response_path": str(run_dir / "researchclaw_agentic_judge_response.json"),
        "judge_workspace": str(paths["workspace"]),
        "judge_log_dir": str(paths["log_dir"]),
    }, {
        "backend": "agentic",
        "request": str(run_dir / "researchclaw_agentic_judge_request.json"),
        "response": str(run_dir / "researchclaw_agentic_judge_response.json"),
        "workspace": str(paths["workspace"]),
        "logs": str(paths["log_dir"]),
        "stdout": str(run_dir / "researchclaw_agentic_judge.stdout"),
        "stderr": str(run_dir / "researchclaw_agentic_judge.stderr"),
        "result": str(result_path),
    }


def _prepare_agentic_judge_workspace(
    *,
    task: tasks.TaskSpec,
    reward_dir: Path,
    solver_workspace_dir: Path,
    run_dir: Path,
    rubric: dict[str, Any],
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    max_chars: int,
    rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    judge_workspace = run_dir / "researchclaw_agentic_judge_workspace"
    if judge_workspace.exists():
        shutil.rmtree(judge_workspace)
    tasks.prepare_workspace(task, judge_workspace)
    judge_dir = judge_workspace / "_judge"
    state_dir = run_dir / "researchclaw_agentic_judge_state"
    log_dir = run_dir / "researchclaw_agentic_judge_logs"
    judge_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_paths = _judge_output_paths(task, reward_dir, rules, config)
    copied_outputs: dict[str, dict[str, Any]] = {}
    for raw_relpath in output_paths:
        relpath = _normalize_workspace_relpath(raw_relpath)
        source = _safe_workspace_file(solver_workspace_dir, relpath)
        destination = judge_workspace / relpath
        if source is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied_outputs[relpath] = {"exists": True, "size_chars": len(destination.read_text(encoding="utf-8", errors="replace"))}
        else:
            copied_outputs[relpath] = {"exists": False, "size_chars": 0}

    reference_answer_included = False
    if _env_truthy("RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER"):
        reference_answer = _read_reference_answer(reward_dir, max_chars)
        if reference_answer:
            reference_dir = judge_dir / "reference_answer"
            reference_dir.mkdir(parents=True, exist_ok=True)
            for name, text in reference_answer.items():
                (reference_dir / name).write_text(text, encoding="utf-8")
            reference_answer_included = True

    (judge_dir / "task.md").write_text(task.user_query, encoding="utf-8")
    _write_json(judge_dir / "rules.json", {"rules": rules})
    _write_json(judge_dir / "rubric.json", rubric)
    _write_json(
        judge_dir / "scorecard.json",
        _build_agentic_judge_scorecard(
            task=task,
            rules=rules,
            config=config,
            copied_outputs=copied_outputs,
            reference_answer_included=reference_answer_included,
            rule_ids=rule_ids,
        ),
    )
    _write_json(
        judge_dir / "manifest.json",
        {
            "task_id": task.task_id,
            "required_output_path": _required_output_path(task, config),
            "solver_workspace_dir": str(solver_workspace_dir),
            "judge_workspace_dir": str(judge_workspace),
            "copied_solver_outputs": copied_outputs,
            "reference_answer_included": reference_answer_included,
            "result_path": AGENTIC_JUDGE_RESULT_PATH,
            "instructions": [
                "Inspect only files inside this judge workspace.",
                "Use _judge/scorecard.json, _judge/task.md, solver output files, and local source files copied from the task input.",
                "Full _judge/rules.json and _judge/rubric.json remain available for targeted fallback checks when the scorecard is ambiguous.",
                f"Write the final machine-readable judgement to {AGENTIC_JUDGE_RESULT_PATH}.",
            ],
        },
    )
    return {
        "workspace": judge_workspace,
        "judge_dir": judge_dir,
        "state_dir": state_dir,
        "log_dir": log_dir,
        "reference_answer_included": reference_answer_included,
    }


def _build_agentic_judge_scorecard(
    *,
    task: tasks.TaskSpec,
    rules: list[dict[str, Any]],
    config: dict[str, Any],
    copied_outputs: dict[str, dict[str, Any]],
    reference_answer_included: bool,
    rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    score_rule_ids = rule_ids or _rule_ids(rules)
    if len(score_rule_ids) != len(rules):
        raise ValueError("ResearchClaw judge scorecard rule_ids length must match rules length")
    criteria: list[dict[str, Any]] = []
    for index, rule in enumerate(rules, 1):
        rule_id = score_rule_ids[index - 1]
        metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
        anchors = rule.get("scores") if isinstance(rule.get("scores"), dict) else {"0": "Fails this criterion.", "1": "Satisfies this criterion."}
        criteria.append(
            {
                "id": rule_id,
                "rubric_id": metadata.get("rubric_id"),
                "name": _compact_text(rule.get("name"), 96),
                "file_path": _normalize_workspace_relpath(str(rule.get("file_path") or _required_output_path(task, config))),
                "allowed_scores": _rule_allowed_scores(rule),
                "score_anchors": {
                    str(key): _compact_text(value, 160)
                    for key, value in sorted(anchors.items(), key=lambda item: _score_anchor_sort_key(item[0]))
                },
                "judge_guidance": _compact_text(rule.get("judge_guidance"), 140),
            }
        )
    return {
        "task_id": task.task_id,
        "required_output_path": _required_output_path(task, config),
        "reference_answer_included": reference_answer_included,
        "result_path": AGENTIC_JUDGE_RESULT_PATH,
        "copied_solver_outputs": copied_outputs,
        "criteria": criteria,
        "required_scores_keys": [criterion["id"] for criterion in criteria],
        "result_template": {
            "scores": {criterion["id"]: 0 for criterion in criteria},
            "notes": "Replace these default zeros with evidence-grounded scores before writing the final result.",
        },
    }


def _write_agentic_judge_nanobot_config(workspace_dir: Path, state_dir: Path, judge: JudgeConfig) -> Path:
    max_tokens = _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS", judge.max_tokens)
    # The production judge serves a 32K context window with an 8K completion
    # budget.  Keep the agentic runner on that same effective contract unless
    # an experiment explicitly overrides it.
    context_window = _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_CONTEXT_WINDOW", 32_768)
    max_tool_iterations = _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOOL_ITERATIONS", 8)
    enable_thinking = _env_truthy("RESEARCHCLAW_AGENTIC_JUDGE_ENABLE_THINKING")
    config = {
        "providers": {
            "custom": {
                "apiBase": judge.base_url.rstrip("/"),
                "apiKey": judge.api_key,
                "extraBody": {
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            }
        },
        "agents": {
            "defaults": {
                "workspace": str(workspace_dir),
                "provider": "custom",
                "model": judge.model,
                "maxTokens": max_tokens,
                "contextWindowTokens": context_window,
                "temperature": judge.temperature,
                "maxToolIterations": max_tool_iterations,
                "botIcon": "",
                "timezone": os.environ.get("NANOBOT_TIMEZONE", "Asia/Shanghai"),
            }
        },
        "channels": {
            "sendProgress": False,
            "sendToolHints": False,
            "showReasoning": _env_truthy("RESEARCHCLAW_AGENTIC_JUDGE_SHOW_REASONING"),
        },
        "tools": {
            "restrictToWorkspace": True,
            "web": {"enable": False, "noWeb": True},
            "exec": {
                "enable": os.environ.get("RESEARCHCLAW_AGENTIC_JUDGE_EXEC_ENABLE", "1") != "0",
                "timeout": _positive_int_env("RESEARCHCLAW_AGENTIC_JUDGE_EXEC_TIMEOUT", 60),
                "noWeb": True,
                "noWebCliTimeout": _positive_int_env("NANOBOT_NO_WEB_CLI_TIMEOUT", 5),
            },
        },
    }
    path = state_dir / "config.json"
    _write_json(path, config)
    return path


def _run_agentic_judge_nanobot(
    *,
    workspace_dir: Path,
    config_path: Path,
    log_dir: Path,
    session_id: str,
    message: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    from . import runtime as runtime_mod

    return runtime_mod._run_nanobot_turn(
        workspace_dir=workspace_dir,
        config_path=config_path,
        session_id=session_id,
        turn_type=os.environ.get("RESEARCHCLAW_AGENTIC_JUDGE_TURN_TYPE", "side").strip() or "side",
        message=message,
        log_dir=log_dir,
        timeout=timeout,
        env_overrides={
            "NANOBOT_TASK_ID": session_id,
            "RESEARCHCLAW_AGENTIC_JUDGE": "1",
            "NANOBOT_WEB_ENABLE": "0",
            "NANOBOT_RESTRICT_TO_WORKSPACE": "1",
            "NANOBOT_ENABLE_THINKING": "1" if _env_truthy("RESEARCHCLAW_AGENTIC_JUDGE_ENABLE_THINKING") else "0",
            "NANOBOT_SHOW_REASONING": "1" if _env_truthy("RESEARCHCLAW_AGENTIC_JUDGE_SHOW_REASONING") else "0",
        },
    )


def _build_agentic_judge_prompt(
    *,
    rules: list[dict[str, Any]],
    reference_answer_included: bool,
    strict_score_enum: bool = False,
) -> str:
    rule_ids = ", ".join(f"criterion_{index}" for index in range(1, len(rules) + 1))
    strict_contract = ""
    if strict_score_enum:
        allowed_lines = "\n".join(
            f"- criterion_{index}: {json.dumps(_rule_allowed_scores(rule))}"
            for index, rule in enumerate(rules, 1)
        )
        strict_contract = (
            "STRICT SCORE ENUM CONTRACT (mandatory):\n"
            "The allowed anchors are categorical enum values, not ratings, confidence values, percentages, or a 0-10 scale. "
            "Never rescale them. For a binary criterion, output exactly 0 or 1; never output values such as 6, 7, 8, or 9. "
            "If a multi-part criterion is only partly satisfied and no partial anchor is listed, choose the one allowed anchor whose description applies; "
            "do not invent partial credit.\n"
            f"Allowed values for this run:\n{allowed_lines}\n"
            "Before finishing, read the result file back and verify that every score exactly equals one of the values listed above.\n\n"
        )
    reference_text = (
        "A hidden reference answer is available under `_judge/reference_answer/`; use it as privileged judge-only evidence.\n"
        if reference_answer_included
        else "No hidden reference answer is available for this task.\n"
    )
    return (
        "You are the ResearchClaw judge agent. Your job is to grade the solver's final output, not to solve the task for it.\n\n"
        f"{strict_contract}"
        "Use local tools to inspect files inside the current judge workspace. Do not use the network. Do not ask for more context. "
        "Do not inspect paths outside this workspace.\n\n"
        "Fast required workflow:\n"
        "1. Read `_judge/scorecard.json` and `_judge/manifest.json` first.\n"
        "2. Read the solver output file named by `required_output_path` in the scorecard.\n"
        "3. Inspect local paper/repo/source files only when needed to verify a concrete solver claim.\n"
        "4. If evidence is incomplete or the solver output is missing, assign conservative allowed scores and write the result.\n\n"
        "The full `_judge/rules.json` and `_judge/rubric.json` are available as fallback judge-only references if the scorecard is ambiguous; "
        "if you use them, read only targeted portions instead of dumping the whole file. "
        "Do not read `.nanobot`, `sessions`, `memory`, `.nanobot/tool-results`, previous tool-result files, stdout/stderr logs, or files outside the judge workspace. "
        "Do not recursively list large directories. Prefer targeted reads or grep commands. Finish within a small number of tool calls.\n\n"
        "Required files:\n"
        "- `_judge/scorecard.json`: compact scoring rules, allowed score anchors, output path, and result template.\n"
        "- `_judge/task.md`: original user task.\n"
        "- `_judge/manifest.json`: copied output manifest and result path.\n"
        "- Solver final output files are copied at their original relative paths.\n"
        f"- {reference_text}"
        "You may inspect local paper/repo/source files copied into this workspace to verify whether the solver's claims and citations are correct.\n\n"
        f"Write exactly one JSON object to `{AGENTIC_JUDGE_RESULT_PATH}` with exactly these top-level keys:\n"
        "- `scores`: an object mapping every rubric id to a numeric allowed anchor.\n"
        "- `notes`: a concise string explaining the main evidence-grounded reasons for the scores.\n\n"
        f"The `scores` object must include every criterion exactly once: {rule_ids}.\n"
        "Each score must be one of the allowed numeric anchors listed for that criterion in `_judge/scorecard.json`. "
        "Do not include an overall score. Do not write markdown to the result file. Finish only after the result file exists."
    )


def _call_openai_compatible_judge(
    *,
    judge: JudgeConfig,
    prompt: str,
    rules: list[dict[str, Any]],
    validate_scores: bool = True,
) -> dict[str, Any]:
    endpoint = judge.base_url.rstrip("/") + "/chat/completions"
    request_payload = {
        "model": judge.model,
        "messages": _build_judge_messages(prompt),
        "temperature": judge.temperature,
        "max_tokens": judge.max_tokens,
        # vLLM/SGLang reasoning models may otherwise spend the entire response
        # budget in hidden reasoning and finish with no score JSON.  Direct
        # judging is a constrained classification task; thinking remains
        # opt-in, matching the agentic judge path below.
        "chat_template_kwargs": {
            "enable_thinking": _env_truthy("RESEARCHCLAW_DIRECT_JUDGE_ENABLE_THINKING"),
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {judge.api_key}",
            "Content-Type": "application/json",
        },
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=judge.timeout_sec) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ResearchClaw judge HTTP {exc.code}: {body[:1000]}") from exc
    except Exception as exc:
        raise RuntimeError(f"ResearchClaw judge request failed: {type(exc).__name__}: {exc}") from exc

    choices = response_body.get("choices") if isinstance(response_body, dict) else None
    if not choices:
        raise ValueError(f"ResearchClaw judge response missing choices: {response_body}")
    finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
    if finish_reason not in (None, "", "stop"):
        raise ValueError(
            f"ResearchClaw judge response finished with {finish_reason!r}: "
            f"{_compact_text(response_body, 2000)}"
        )
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    text = message.get("content", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"ResearchClaw judge response has empty content: {response_body}")

    parsed = _parse_judge_text(text)
    result = {
        "available": True,
        "parsed": parsed,
        "text": text,
        "raw": response_body,
        "duration_sec": round(time.time() - started, 3),
    }
    if validate_scores:
        score, normalization = _score_judge_scores(parsed, rules)
        result["score"] = score
        result["score_normalization"] = normalization
    return result


def _build_judge_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DIRECT_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _build_rubric_prompt(
    *,
    user_task: str,
    rules: list[dict[str, Any]],
    target_files: dict[str, dict[str, Any]],
    reference_answer: dict[str, Any] | None,
) -> str:
    file_blocks: list[str] = []
    for path, info in target_files.items():
        if not info.get("exists"):
            file_blocks.append(f"### Missing File: {path}\n")
            continue
        content = str(info.get("content") or "")
        suffix = "\n[TRUNCATED]\n" if info.get("truncated") else ""
        file_blocks.append(f"### File: {path}\n{content}{suffix}")

    reference_block = ""
    if reference_answer is not None:
        reference_parts: list[str] = []
        for name, value in reference_answer.items():
            reference_parts.append(f"### {name}\n{value}")
        reference_block = "\n\n## Hidden Reference Answer\n" + "\n\n".join(reference_parts)

    rules_text: list[str] = []
    for index, rule in enumerate(rules, 1):
        name = str(rule.get("name") or f"criterion_{index}")
        file_path = str(rule.get("file_path") or "")
        rule_id = f"criterion_{index}"
        rules_text.append(f"{index}. id={rule_id} name={name} file={file_path}")
        scores = rule.get("scores") or {}
        if isinstance(scores, dict):
            for key in sorted(scores, key=_score_anchor_sort_key):
                rules_text.append(f"   - {key}: {scores[key]}")
        guidance = str(rule.get("judge_guidance") or "").strip()
        if guidance:
            rules_text.append(f"   - judge_guidance: {guidance}")
        metadata = rule.get("metadata") or {}
        if isinstance(metadata, dict):
            rubric_id = str(metadata.get("rubric_id") or "").strip()
            check_type = str(metadata.get("check_type") or "").strip()
            if rubric_id or check_type:
                rules_text.append(f"   - metadata: rubric_id={rubric_id} check_type={check_type}")
            examples = metadata.get("acceptable_phrasings") if isinstance(metadata.get("acceptable_phrasings"), list) else []
            examples = [str(item).strip() for item in examples if str(item).strip()]
            if examples:
                rules_text.append(f"   - hidden_reference_examples: {' | '.join(examples[:4])}")

    return (
        "You are grading an OpenClaw agent result.\n\n"
        "You must not call, request, or simulate any tools. Do not browse, list directories, "
        "open files, inspect the workspace, or ask for more context. Grade only from the task, "
        "final output files, rubrics, and optional reference answer included in this prompt.\n\n"
        "Before giving the final judgement, provide concise analysis explaining how the final "
        "outputs satisfy or fail each rubric criterion.\n\n"
        "Your final judgement must end with exactly one standalone JSON object and nothing else "
        "after it. The final JSON object must contain exactly two keys: `scores` and `notes`.\n\n"
        "- `scores` must map every rubric id (`criterion_1`, `criterion_2`, ...) to one numeric "
        "score chosen from that rubric's allowed score anchors.\n"
        "- When a rubric provides a partial-credit anchor such as 0.5, use it for outputs that locate "
        "some relevant evidence or make a partial analysis but miss required subpoints or caveats; "
        "do not collapse all incomplete content to only 0 or 1.\n"
        "- For content-quality criteria, file existence, parseability, source-path mentions, or generic "
        "summaries are not enough for full credit.\n"
        "- `notes` must be a concise string summarizing the main reasons for the assigned scores.\n\n"
        "Do not include any overall score in the final JSON. Score aggregation will be handled "
        "separately by post-processing.\n\n"
        f"## User Task\n{user_task}\n\n"
        "## Final Output Files\n"
        f"{chr(10).join(file_blocks)}"
        f"{reference_block}\n\n"
        "## Rubric\n"
        f"{chr(10).join(rules_text)}\n"
    )


def _build_triage_judge_prompt(
    *,
    user_task: str,
    rules: list[dict[str, Any]],
    target_files: dict[str, dict[str, Any]],
    reference_answer: dict[str, Any] | None,
) -> str:
    return (
        "You are the first-pass ResearchClaw judge. This pass has no tools. "
        "Use only the task, solver output, rubric packet, and optional reference answer in this prompt.\n\n"
        "Your job is triage: assign direct scores where the prompt is enough, and request targeted local evidence only "
        "for criteria where the solver made a concrete source-backed claim that may deserve credit but must be verified "
        "against bundled task-local files.\n\n"
        "Output discipline:\n"
        "- Think silently. Do not include analysis, markdown fences, or prose outside the JSON object.\n"
        "- Keep `notes` under 80 words.\n"
        "- Every `needs_evidence` item must be an exact criterion id string, for example \"criterion_3\". "
        "Never use numbers, indexes, rubric ids, names, or objects in `needs_evidence`.\n"
        "- Keep each evidence request brief: `reason` under 40 words and at most 4 `suggested_paths`.\n"
        "- If many criteria need the same evidence, repeat the same short request; do not explain your deliberation.\n\n"
        "Conservative scoring policy:\n"
        "- Review every supplied solver output independently. A missing file forces a zero only for criteria that "
        "depend on that file; use other available primary or sidecar outputs when they are relevant.\n"
        "- Respect each criterion's named output path. A sidecar may supply citations, caveats, or supporting evidence, "
        "but it must not substitute for content that the criterion explicitly requires in another output file.\n"
        "- Before awarding a positive score, identify the concrete content in the criterion's named output file that "
        "satisfies the requirement. For completeness words such as `all`, `every`, or `full`, verify the claimed coverage "
        "rather than inferring it from related content elsewhere. If the supplied prompt is insufficient, use "
        "`needs_evidence` instead of guessing 1.\n"
        "- If a criterion depends on task-local evidence that is not included in the supplied output files, put it in "
        "`needs_evidence` when an available solver artifact contains a concrete claim worth verifying.\n"
        "- If the solver clearly omits a criterion's required content, score that criterion 0; do not request evidence.\n"
        "- If the solver only repeats the task prompt, only names a source path, or gives generic prose, score 0.\n"
        "- If a relevant solver output is marked [TRUNCATED] and the visible prefix is insufficient to decide whether "
        "required content exists later in that file, put the criterion in needs_evidence; do not infer omission.\n"
        "- If the solver includes a concrete local-source-backed candidate answer and the rubric requires verifying local evidence, "
        "put that criterion in needs_evidence instead of guessing 1.\n"
        "- Give 1 in this first pass only when the supplied prompt contents are already enough to verify the criterion.\n\n"
        "Return exactly one JSON object with these keys:\n"
        "- `scores`: object mapping every criterion not listed in `needs_evidence` to a numeric allowed anchor. "
        "You may also include tentative scores for needs_evidence criteria, but they will be ignored.\n"
        "- `needs_evidence`: array of criterion ids that require local file verification.\n"
        "- `evidence_requests`: object keyed by each needs_evidence criterion id. Each value must include `reason` and "
        "`suggested_paths` for the targeted verifier.\n"
        "- `notes`: concise string.\n\n"
        "Do not include an overall score. Do not write markdown outside the final JSON object. "
        "Return the JSON immediately after deciding the scores; do not narrate the decision process.\n\n"
        f"## User Task\n{user_task}\n\n"
        "## Solver Output Files\n"
        f"{_format_judge_file_blocks(target_files)}\n"
        f"{_format_reference_answer_block(reference_answer)}\n"
        "## Rubric Packet\n"
        f"{_format_rules_packet(rules, _rule_ids(rules))}\n"
    )


def _build_targeted_agentic_judge_prompt(
    *,
    rules: list[dict[str, Any]],
    rule_ids: list[str],
    target_files: dict[str, dict[str, Any]],
    evidence_requests: dict[str, Any],
    reference_answer_included: bool,
) -> str:
    rule_id_text = ", ".join(rule_ids)
    return (
        "You are a targeted ResearchClaw evidence verifier running inside the judge workspace.\n\n"
        "The first-pass judge already selected the criteria below because they may deserve credit but require local "
        "evidence verification. Do not grade any other criteria. Do not re-read the full rubric unless a listed evidence "
        "request is impossible to interpret from this prompt.\n\n"
        "Use tools only to inspect task-local source files needed for the listed evidence requests. Do not use the network. "
        "Do not inspect paths outside this workspace. Do not read `.nanobot`, `sessions`, `memory`, `.nanobot/tool-results`, "
        "previous tool-result files, stdout/stderr logs, or unrelated generated logs. Prefer targeted reads/grep of the "
        "suggested paths. If a solver output below is marked [TRUNCATED], inspect its copied full file when needed. "
        "Do not recursively list large directories.\n\n"
        "Scoring discipline:\n"
        "- Respect each criterion's named output path. Sidecars may corroborate citations, caveats, or evidence, but cannot "
        "replace content explicitly required in another output file.\n"
        "- The source containing the correct answer does not prove that the solver output is correct. Compare the solver's "
        "actual values and claims against the source; look specifically for omissions, duplicated/copied values, incorrect "
        "labels, and arithmetic mismatches.\n"
        "- For completeness words such as `all`, `every`, or `full`, enumerate the required entities or records and confirm "
        "they appear correctly in the solver output. If a criterion explicitly combines a primary file and a sidecar, "
        "verify the distinct required contribution of each file rather than treating either file as a substitute.\n"
        "- In `notes`, report concrete solver-output observations that justify the scores. Do not award credit merely "
        "because the correct material appears in a source file or reference answer.\n\n"
        "Budget discipline:\n"
        "- First inspect only the suggested paths, using grep or small targeted reads.\n"
        "- Stop gathering evidence once every requested criterion can be scored.\n"
        "- Write `_judge/judge_result.json` as soon as you know the scores; do not continue explaining after that.\n"
        "- Keep `notes` under 120 words.\n\n"
        f"Reference answer available: {'yes, under _judge/reference_answer/' if reference_answer_included else 'no'}.\n\n"
        "## Solver Output Files\n"
        f"{_format_judge_file_blocks(target_files)}\n"
        "## Criteria To Verify\n"
        f"{_format_rules_packet(rules, rule_ids, evidence_requests=evidence_requests)}\n"
        "Write exactly one JSON object to `_judge/judge_result.json` with exactly these top-level keys:\n"
        "- `scores`: object mapping every requested criterion id to a numeric allowed anchor.\n"
        "- `notes`: concise evidence-grounded explanation.\n\n"
        f"The `scores` object must include exactly these criteria: {rule_id_text}. "
        "Do not include an overall score. Finish only after `_judge/judge_result.json` exists."
    )


def _format_judge_file_blocks(target_files: dict[str, dict[str, Any]]) -> str:
    file_blocks: list[str] = []
    for path, info in target_files.items():
        if not info.get("exists"):
            file_blocks.append(f"### Missing File: {path}\n")
            continue
        content = str(info.get("content") or "")
        suffix = "\n[TRUNCATED]\n" if info.get("truncated") else ""
        file_blocks.append(f"### File: {path}\n{content}{suffix}")
    return "\n".join(file_blocks)


def _format_reference_answer_block(reference_answer: dict[str, Any] | None) -> str:
    if reference_answer is None:
        return "## Hidden Reference Answer\nNo hidden reference answer is included in this prompt.\n\n"
    reference_parts: list[str] = []
    for name, value in reference_answer.items():
        reference_parts.append(f"### {name}\n{value}")
    return "## Hidden Reference Answer\n" + "\n\n".join(reference_parts) + "\n\n"


def _format_rules_packet(
    rules: list[dict[str, Any]],
    rule_ids: list[str],
    *,
    evidence_requests: dict[str, Any] | None = None,
) -> str:
    if len(rules) != len(rule_ids):
        raise ValueError("ResearchClaw judge rules packet rule_ids length must match rules length")
    blocks: list[str] = []
    for rule, rule_id in zip(rules, rule_ids):
        metadata = rule.get("metadata") if isinstance(rule.get("metadata"), dict) else {}
        blocks.append(f"### {rule_id}")
        blocks.append(f"name: {rule.get('name') or rule_id}")
        blocks.append(f"rubric_id: {metadata.get('rubric_id')}")
        blocks.append(f"check_type: {metadata.get('check_type')}")
        blocks.append(f"target_file: {rule.get('file_path') or ''}")
        blocks.append(f"allowed_scores: {_rule_allowed_scores(rule)}")
        scores = rule.get("scores") if isinstance(rule.get("scores"), dict) else {}
        if scores:
            blocks.append("score_anchors:")
            for key in sorted(scores, key=_score_anchor_sort_key):
                blocks.append(f"- {key}: {scores[key]}")
        guidance = str(rule.get("judge_guidance") or "").strip()
        if guidance:
            blocks.append(f"judge_guidance: {guidance}")
        examples = metadata.get("acceptable_phrasings") if isinstance(metadata.get("acceptable_phrasings"), list) else []
        examples = [str(item).strip() for item in examples if str(item).strip()]
        if examples:
            blocks.append("evidence_hints:")
            for item in examples[:6]:
                blocks.append(f"- {_compact_text(item, 320)}")
        if evidence_requests and rule_id in evidence_requests:
            blocks.append("evidence_request:")
            blocks.append(json.dumps(evidence_requests[rule_id], ensure_ascii=False, indent=2, default=str))
        blocks.append("")
    return "\n".join(blocks)


def _rubric_to_rules(rubric: dict[str, Any], output_path: str) -> list[dict[str, Any]]:
    path = _normalize_workspace_relpath(output_path)
    rules: list[dict[str, Any]] = []
    items = rubric.get("items")
    if not isinstance(items, list):
        raise TypeError("ResearchClaw rubric.json must contain an items list")
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict) or item.get("evaluation_mode") != "llm_judge":
            continue
        name = str(item.get("name") or item.get("id") or f"criterion_{index}")
        pass_condition = str(item.get("pass_condition") or item.get("feedback") or "The answer satisfies this criterion.")
        common_errors = item.get("common_errors") if isinstance(item.get("common_errors"), list) else []
        fail_text = "Fails this criterion."
        if common_errors:
            fail_text += " Common errors: " + "; ".join(str(error) for error in common_errors[:4])
        anchors = item.get("score_anchors") if isinstance(item.get("score_anchors"), dict) else None
        if anchors:
            scores = {str(key): str(value) for key, value in anchors.items()}
            if "1" in scores and pass_condition:
                scores["1"] = f"{scores['1']} Criterion-specific requirement: {pass_condition}"
        else:
            scores = {"0": fail_text, "1": pass_condition}
        rules.append(
            {
                "name": name,
                "file_path": path,
                "scores": scores,
                "metadata": {
                    "rubric_id": item.get("id"),
                    "weight": item.get("weight", 1.0),
                    "check_type": item.get("check_type"),
                    "acceptable_phrasings": item.get("acceptable_phrasings", []),
                },
                "judge_guidance": item.get("judge_guidance", ""),
            }
        )
    return rules


def _judge_output_paths(
    task: tasks.TaskSpec,
    reward_dir: Path,
    rules: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """Return every solver artifact that a judge may need to grade.

    ResearchClaw's primary output is stored in ``eval_config.json``, while
    additional required artifacts live in the hidden bound task's output
    contract. Rubric rules historically pointed only at the primary output,
    which made sidecars invisible to both direct triage and agentic judges.
    """
    raw_paths: set[str] = {
        _required_output_path(task, config),
        *(str(rule.get("file_path") or "") for rule in rules),
    }

    configured_paths = config.get("required_output_paths")
    if isinstance(configured_paths, list):
        raw_paths.update(str(path) for path in configured_paths)
    metadata_paths = task.metadata.get("required_output_paths")
    if isinstance(metadata_paths, list):
        raw_paths.update(str(path) for path in metadata_paths)

    bound_task_path = reward_dir / "llmjudge" / "bound_task.json"
    if bound_task_path.is_file():
        bound_task = _read_json_dict(bound_task_path)
        output_contract = bound_task.get("output_contract")
        if isinstance(output_contract, dict):
            sidecars = output_contract.get("sidecar_outputs")
            if isinstance(sidecars, list):
                for sidecar in sidecars:
                    if isinstance(sidecar, dict) and sidecar.get("path"):
                        raw_paths.add(str(sidecar["path"]))

    return sorted(
        {
            _normalize_workspace_relpath(path)
            for path in raw_paths
            if isinstance(path, str) and path.strip()
        }
    )


def _read_judge_target_files(
    workspace_dir: Path,
    output_paths: list[str],
    max_chars_per_file: int,
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for raw_relpath in output_paths:
        relpath = _normalize_workspace_relpath(raw_relpath)
        if relpath in targets:
            continue
        path = _safe_workspace_file(workspace_dir, relpath)
        if path is None:
            targets[relpath] = {"exists": False, "content": "", "size_chars": 0, "truncated": False}
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        targets[relpath] = {
            "exists": True,
            "content": text[:max_chars_per_file],
            "size_chars": len(text),
            "truncated": len(text) > max_chars_per_file,
        }
    return targets


def _safe_workspace_file(workspace_dir: Path, relpath: str) -> Path | None:
    """Resolve one solver output without allowing symlinks to escape the workspace."""
    candidate = workspace_dir / _normalize_workspace_relpath(relpath)
    if not candidate.is_file():
        return None
    workspace_root = workspace_dir.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workspace_root):
        return None
    return resolved


def _read_reference_answer(reward_dir: Path, max_chars: int) -> dict[str, str]:
    answer_dir = reward_dir / "llmjudge"
    paths = [answer_dir / "answer.md", answer_dir / "answer.json"]
    available = [path for path in paths if path.is_file()]
    if not available:
        warnings.warn(
            f"Reference answer missing under {answer_dir}; continuing LLM judge without GT answer",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    result: dict[str, str] = {}
    for path in available:
        text = path.read_text(encoding="utf-8", errors="replace")
        result[path.name] = text[:max_chars] + ("\n[TRUNCATED]\n" if len(text) > max_chars else "")
    return result


def _score_judge_scores(
    parsed: dict[str, Any],
    rules: list[dict[str, Any]],
    *,
    rule_ids: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    raw_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else parsed
    if not isinstance(raw_scores, dict) or not rules:
        raise ValueError("ResearchClaw llmjudge result must contain a scores object and at least one rule")
    score_rule_ids = rule_ids or _rule_ids(rules)
    if len(score_rule_ids) != len(rules):
        raise ValueError("ResearchClaw llmjudge rule_ids length must match rules length")

    normalized_scores: dict[str, float] = {}
    matched_scores: dict[str, float] = {}
    sum_score = 0.0
    sum_max_score = 0.0
    for index, (rule, rule_id) in enumerate(zip(rules, score_rule_ids), 1):
        max_score = _rule_max_score(rule)
        sum_max_score += max_score
        allowed_scores = _rule_allowed_scores(rule)
        value = None
        matched_key = None
        for key in _rule_score_keys(rule, index, rule_id=rule_id):
            if key in raw_scores:
                value = _numeric(raw_scores.get(key))
                matched_key = key
                break
        if value is None:
            raise ValueError(f"ResearchClaw llmjudge missing numeric score for {rule_id}")
        if not _score_in_allowed_anchors(value, allowed_scores):
            raise ValueError(
                f"ResearchClaw llmjudge score for {rule_id} via {matched_key!r} is {value}, "
                f"expected one of {allowed_scores}"
            )
        matched_scores[rule_id] = value
        normalized_scores[rule_id] = value / max_score if max_score > 0 else 0.0
        sum_score += value

    if sum_max_score <= 0:
        raise ValueError("ResearchClaw llmjudge rules must have a positive maximum score sum")
    return sum_score / sum_max_score, {
        "raw_scores": raw_scores,
        "matched_scores": matched_scores,
        "normalized_scores": normalized_scores,
        "missing": [],
        "sum_score": sum_score,
        "sum_max_score": sum_max_score,
        "method": "sum_scores_divided_by_sum_max_score",
    }


def _rule_ids(rules: list[dict[str, Any]]) -> list[str]:
    return [f"criterion_{index}" for index in range(1, len(rules) + 1)]


def _select_rules_by_ids(rules: list[dict[str, Any]], rule_ids: list[str]) -> list[dict[str, Any]]:
    all_rule_ids = _rule_ids(rules)
    selected: list[dict[str, Any]] = []
    for rule_id in rule_ids:
        if rule_id not in all_rule_ids:
            raise ValueError(f"ResearchClaw unknown criterion id: {rule_id}")
        selected.append(rules[all_rule_ids.index(rule_id)])
    return selected


def _triage_needs_evidence(parsed: dict[str, Any], rules: list[dict[str, Any]]) -> list[str]:
    raw = parsed.get("needs_evidence", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("ResearchClaw hybrid judge triage needs_evidence must be a list")
    known = set(_rule_ids(rules))
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"ResearchClaw hybrid judge triage needs_evidence item must be a string: {item!r}")
        if item not in known:
            raise ValueError(f"ResearchClaw hybrid judge triage requested unknown criterion: {item}")
        if item in result:
            raise ValueError(f"ResearchClaw hybrid judge triage duplicated needs_evidence criterion: {item}")
        result.append(item)
    return result


def _triage_evidence_requests(parsed: dict[str, Any], needs_evidence: list[str]) -> dict[str, Any]:
    raw = parsed.get("evidence_requests")
    if not isinstance(raw, dict):
        raise ValueError("ResearchClaw hybrid judge triage evidence_requests must be an object when needs_evidence is non-empty")
    result: dict[str, Any] = {}
    for rule_id in needs_evidence:
        request = raw.get(rule_id)
        if not isinstance(request, dict):
            raise ValueError(f"ResearchClaw hybrid judge triage missing evidence request for {rule_id}")
        result[rule_id] = request
    return result


def _merge_judge_notes(parsed: dict[str, Any], agentic_response: dict[str, Any] | None) -> str:
    notes = str(parsed.get("notes") or "").strip()
    if agentic_response is None:
        return notes
    agentic_parsed = agentic_response.get("parsed") if isinstance(agentic_response.get("parsed"), dict) else {}
    agentic_notes = str(agentic_parsed.get("notes") or "").strip()
    if notes and agentic_notes:
        return f"triage: {notes}\nagentic: {agentic_notes}"
    return notes or agentic_notes


def _rule_allowed_scores(rule: dict[str, Any]) -> list[float]:
    scores = rule.get("scores")
    if not isinstance(scores, dict):
        return [0.0, 1.0]
    values = [_numeric(key) for key in scores.keys()]
    numeric_values = sorted({float(value) for value in values if value is not None})
    return numeric_values or [0.0, 1.0]


def _score_in_allowed_anchors(value: float, allowed_scores: list[float]) -> bool:
    return any(abs(float(value) - float(anchor)) <= 1e-9 for anchor in allowed_scores)


def _parse_judge_text(text: Any) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    stripped = text.strip()
    fenced_payloads = re.findall(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates = [*reversed(fenced_payloads), stripped] if stripped else []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        # Gemma occasionally emits an otherwise valid JSON object with a
        # trailing comma before a closing brace/bracket. Recover the complete
        # outer object before falling back to scanning nested JSON objects;
        # otherwise the inner `scores` object can be mistaken for the result.
        trailing_comma_fixed = re.sub(r",(?=\s*[}\]])", "", candidate)
        if trailing_comma_fixed != candidate:
            try:
                value = json.loads(trailing_comma_fixed)
                if isinstance(value, dict):
                    warnings.warn(
                        "ResearchClaw judge JSON contained trailing commas; recovered the complete JSON object",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return value
            except json.JSONDecodeError:
                pass
    for candidate in candidates:
        for value in reversed(_iter_json_objects(candidate)):
            if isinstance(value, dict):
                return value
    return {}


def _iter_json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        values.append(value)
        index = end
    return values


def _rule_score_keys(rule: dict[str, Any], index: int, *, rule_id: str | None = None) -> list[str]:
    score_rule_id = rule_id or f"criterion_{index}"
    keys = [score_rule_id]
    if score_rule_id == f"criterion_{index}":
        keys.append(str(index))
    name = rule.get("name")
    if name:
        keys.append(str(name))
    file_path = rule.get("file_path")
    if file_path:
        keys.append(str(file_path))
    return keys


def _rule_max_score(rule: dict[str, Any]) -> float:
    scores = rule.get("scores")
    if isinstance(scores, dict):
        values = [_numeric(key) for key in scores.keys()]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            return max(numeric_values)
    return 1.0


def _score_anchor_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _mean_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        values = [float(item) for item in value.values() if isinstance(item, (int, float)) and not isinstance(item, bool)]
        return sum(values) / len(values) if values else None
    if isinstance(value, list):
        values = [float(item) for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
        return sum(values) / len(values) if values else None
    return None


def _parse_json_stdout(raw: str, label: str) -> Any:
    text = raw.strip()
    if not text:
        raise ValueError(f"{label} produced empty stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} stdout is not valid JSON: {_tail_text(text)}") from exc


def _judge_config(*, selection_key: str = "") -> JudgeConfig:
    base_urls = _comma_separated_env("RESEARCHCLAW_JUDGE_BASE_URLS")
    api_keys = _comma_separated_env("RESEARCHCLAW_JUDGE_API_KEYS")
    if not base_urls:
        base_urls = [os.environ.get("RESEARCHCLAW_JUDGE_BASE_URL", "").strip()]
    if not api_keys:
        api_keys = [os.environ.get("RESEARCHCLAW_JUDGE_API_KEY", "").strip()]
    model = os.environ.get("RESEARCHCLAW_JUDGE_MODEL", "").strip()
    base_urls = [value for value in base_urls if value]
    api_keys = [value for value in api_keys if value]
    if len(api_keys) == 1 and len(base_urls) > 1:
        api_keys *= len(base_urls)
    if base_urls and api_keys and len(base_urls) != len(api_keys):
        raise ValueError(
            "RESEARCHCLAW_JUDGE_BASE_URLS and RESEARCHCLAW_JUDGE_API_KEYS "
            "must contain the same number of entries"
        )
    missing = [
        name
        for name, value in (
            ("RESEARCHCLAW_JUDGE_BASE_URL(S)", base_urls),
            ("RESEARCHCLAW_JUDGE_API_KEY(S)", api_keys),
            ("RESEARCHCLAW_JUDGE_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"ResearchClaw llmjudge requested but missing env: {', '.join(missing)}")
    digest = hashlib.sha256(selection_key.encode("utf-8")).digest()
    endpoint_index = int.from_bytes(digest[:8], "big") % len(base_urls)
    return JudgeConfig(
        base_url=base_urls[endpoint_index],
        api_key=api_keys[endpoint_index],
        model=model,
        timeout_sec=_positive_int_env("RESEARCHCLAW_JUDGE_TIMEOUT", 300),
        # The production judge has a 32K context window. Reserve 8K for the
        # short scoring response so inputs can use the remaining 24K.
        max_tokens=_positive_int_env("RESEARCHCLAW_JUDGE_MAX_TOKENS", 8_192),
        temperature=_float_env("RESEARCHCLAW_JUDGE_TEMPERATURE", 0.6),
    )


def _required_output_path(task: tasks.TaskSpec, config: dict[str, Any]) -> str:
    raw = config.get("required_output_path") or task.metadata.get("required_output_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("ResearchClaw eval_config.json must set required_output_path")
    return _normalize_workspace_relpath(raw)


def _reward_entry_path(reward_dir: Path, entry: str) -> Path:
    rel = _safe_relpath(entry)
    path = reward_dir / rel
    if not path.is_file():
        raise FileNotFoundError(f"ResearchClaw reward entry not found: {path}")
    return path


def _normalize_workspace_relpath(raw_path: str) -> str:
    text = str(raw_path or "").replace("\\", "/").strip()
    for prefix in ("/tmp_workspace/", "/root/.openclaw/workspace/"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return _safe_relpath(text.lstrip("/"))


def _safe_relpath(raw_path: str) -> str:
    text = str(raw_path or "").replace("\\", "/").strip()
    if not text:
        raise ValueError("empty relative path")
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"unsafe relative path: {raw_path!r}")
    return parsed.as_posix()


def _compact_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n[TRUNCATED]\n"


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ResearchClaw JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"ResearchClaw JSON file must contain an object: {path}")
    return value


def _read_agentic_judge_result(path: Path) -> dict[str, Any]:
    """Read model-authored judge JSON while tolerating surrounding model text."""
    if not path.is_file():
        raise FileNotFoundError(f"ResearchClaw JSON file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        value = _parse_judge_text(text)
        if not value:
            raise exc
        warnings.warn(
            f"Agentic judge result is not strict JSON at {path}: {exc}; recovered the first complete JSON object",
            RuntimeWarning,
            stacklevel=2,
        )
    if not isinstance(value, dict):
        raise TypeError(f"ResearchClaw JSON file must contain an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _require_score(value: float | None, component: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"ResearchClaw {component} reward produced no numeric score")
    return float(value)


def _float_setting(env_name: str, config: dict[str, Any], config_key: str, default: float) -> float:
    raw = os.environ.get(env_name)
    value = raw if raw is not None and raw.strip() else config.get(config_key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name}/{config_key} must be a float, got {value!r}") from exc


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonnegative_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    value = float(default) if raw is None or not raw.strip() else float(raw)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative float")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    return float(raw)


def _comma_separated_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [value.strip() for value in raw.split(",") if value.strip()]


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _visible_env_config() -> dict[str, Any]:
    names = [
        "RESEARCHCLAW_SCORE_MODE",
        "RESEARCHCLAW_LLMJUDGE_BACKEND",
        "RESEARCHCLAW_CODE_WEIGHT",
        "RESEARCHCLAW_LLMJUDGE_WEIGHT",
        "RESEARCHCLAW_JUDGE_BASE_URL",
        "RESEARCHCLAW_JUDGE_BASE_URLS",
        "RESEARCHCLAW_JUDGE_API_KEY",
        "RESEARCHCLAW_JUDGE_API_KEYS",
        "RESEARCHCLAW_JUDGE_MODEL",
        "RESEARCHCLAW_JUDGE_TEMPERATURE",
        "RESEARCHCLAW_JUDGE_MAX_TOKENS",
        "RESEARCHCLAW_JUDGE_TIMEOUT",
        "RESEARCHCLAW_JUDGE_MAX_CHARS_PER_FILE",
        "RESEARCHCLAW_JUDGE_MAX_RETRIES",
        "RESEARCHCLAW_JUDGE_RETRY_BACKOFF_SEC",
        "RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER",
        "RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT",
        "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS",
        "RESEARCHCLAW_AGENTIC_JUDGE_CONTEXT_WINDOW",
        "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOOL_ITERATIONS",
        "RESEARCHCLAW_AGENTIC_JUDGE_STRICT_SCORE_ENUM",
        "RESEARCHCLAW_AGENTIC_JUDGE_ENABLE_THINKING",
        "RESEARCHCLAW_AGENTIC_JUDGE_SHOW_REASONING",
        "RESEARCHCLAW_AGENTIC_JUDGE_EXEC_ENABLE",
    ]
    return _redact({name: os.environ.get(name) for name in names if os.environ.get(name) is not None})


def _is_sensitive_key(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if normalized in NON_SECRET_TOKEN_KEYS:
        return False
    if normalized in SENSITIVE_EXACT_KEYS:
        return True
    if "password" in normalized or "secret" in normalized or "authorization" in normalized:
        return True
    if "api_key" in normalized or normalized.endswith("apikey"):
        return True
    if "token" in normalized and not normalized.endswith("tokens"):
        return True
    return False


def _redact(value: Any, *, parent_key: str = "") -> Any:
    if _is_sensitive_key(parent_key):
        return "<redacted>" if value not in (None, "") else value
    if isinstance(value, dict):
        return {str(key): _redact(val, parent_key=str(key)) for key, val in value.items()}
    if isinstance(value, list):
        return [_redact(item, parent_key=parent_key) for item in value]
    return value


def _tail_text(text: str, max_chars: int = 1000) -> str:
    value = str(text or "")
    return value[-max_chars:]
