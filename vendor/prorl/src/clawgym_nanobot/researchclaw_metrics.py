from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


SUCCESS_FILENAME = "researchclaw_reward.json"
ERROR_FILENAME = "researchclaw_reward.error.json"


def reward_summary_from_run_dir(run_dir: str | Path) -> dict[str, Any] | None:
    root = Path(run_dir)
    success_path = root / SUCCESS_FILENAME
    if success_path.is_file():
        return summarize_reward_payload(_read_json_dict(success_path), run_dir=root)

    error_path = root / ERROR_FILENAME
    if error_path.is_file():
        return summarize_reward_error(_read_json_dict(error_path), run_dir=root)

    return None


def runtime_failure_summary(*, task_id: str, error: str | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": False,
        "failed": True,
        "score_returned": 0,
        "missing_real_score": 1,
        "task_id": task_id,
    }
    if error:
        summary["error"] = error
        summary["error_type"] = error.split(":", 1)[0]
    return summary


def summarize_reward_payload(payload: dict[str, Any], *, run_dir: str | Path | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": True,
        "failed": False,
        "score_returned": 0,
        "missing_real_score": 1,
    }
    _copy_text(summary, "task_id", payload.get("task_id"))
    _copy_text(summary, "score_mode", payload.get("requested_score_mode"))
    _copy_text(summary, "required_output_path", payload.get("required_output_path"))
    if "required_output_exists" in payload:
        summary["required_output_exists"] = _truthy(payload.get("required_output_exists"))

    final_score = _finite_number(payload.get("final_score"))
    if final_score is not None:
        summary["final_raw_reward"] = final_score
        summary["score_returned"] = 1
        summary["missing_real_score"] = 0

    code_score = _finite_number(payload.get("code_score"))
    if code_score is not None:
        summary["codebase_raw_reward"] = code_score

    llmjudge_score = _finite_number(payload.get("llmjudge_score"))
    if llmjudge_score is not None:
        summary["llmjudge_raw_reward"] = llmjudge_score

    weights = _dict(payload.get("requested_score_weights"))
    _copy_number(summary, "codebase_weight", weights.get("code"))
    _copy_number(summary, "llmjudge_weight", weights.get("llmjudge"))

    code_reward = _dict(payload.get("code_reward"))
    _copy_number(summary, "codebase_duration_sec", code_reward.get("duration_sec"))

    llmjudge_reward = _dict(payload.get("llmjudge_reward"))
    if llmjudge_reward:
        _add_llmjudge_summary(summary, payload, llmjudge_reward, Path(run_dir) if run_dir is not None else None)
    else:
        mode = str(payload.get("requested_score_mode") or "").strip().lower()
        if mode == "code":
            summary["judge_call_count"] = 0
            summary["judge_direct_call_count"] = 0
            summary["judge_agentic_call_count"] = 0
            summary["judge_nanobot_entered"] = 0
            summary["judge_turn_count"] = 0

    return summary


def summarize_reward_error(payload: dict[str, Any], *, run_dir: str | Path | None = None) -> dict[str, Any]:
    summary = runtime_failure_summary(
        task_id=str(payload.get("task_id") or ""),
        error=str(payload.get("error") or ""),
    )
    _copy_text(summary, "required_output_path", payload.get("required_output_path"))
    if "required_output_exists" in payload:
        summary["required_output_exists"] = _truthy(payload.get("required_output_exists"))
    if run_dir is not None:
        root = Path(run_dir)
        direct_files = _existing_files(
            root,
            [
                "researchclaw_judge_response.json",
                "researchclaw_judge_triage_response.json",
            ],
        )
        agentic_files = _existing_files(root, ["researchclaw_agentic_judge_response.json"])
        direct_requested = bool(
            direct_files
            or _existing_files(
                root,
                [
                    "researchclaw_judge_request.json",
                    "researchclaw_judge_triage_request.json",
                ],
            )
        )
        agentic_requested = bool(
            agentic_files
            or _existing_files(
                root,
                [
                    "researchclaw_agentic_judge_request.json",
                    "researchclaw_agentic_judge.stdout",
                    "researchclaw_agentic_judge.stderr",
                ],
            )
        )
        _add_response_file_stats(summary, "judge_direct", direct_files)
        _add_agentic_file_stats(summary, agentic_files)
        direct_count = 1 if direct_requested else 0
        agentic_count = 1 if agentic_requested else 0
        summary["judge_direct_call_count"] = direct_count
        summary["judge_agentic_call_count"] = agentic_count
        summary["judge_nanobot_entered"] = 1 if agentic_count else 0
        summary["judge_call_count"] = direct_count + agentic_count
        summary["judge_turn_count"] = direct_count + agentic_count
    return summary


def compute_researchclaw_sample_metrics(records: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    pairs = [(record, summary) for record in records if (summary := _record_reward_summary(record)) is not None]
    if not pairs:
        return {}

    summaries = [summary for _, summary in pairs]
    metrics: dict[str, int | float] = {"researchclaw/task_count": len(summaries)}

    output_exists = [
        _truthy(summary.get("required_output_exists"))
        for summary in summaries
        if "required_output_exists" in summary
    ]
    if output_exists:
        metrics["researchclaw/output_file_completion_rate"] = sum(output_exists) / len(output_exists)

    _add_count_rate(metrics, "researchclaw/reward_failed", [_truthy(summary.get("failed")) for summary in summaries])
    _add_count_rate(
        metrics,
        "researchclaw/missing_real_score",
        [_truthy(summary.get("missing_real_score")) for summary in summaries],
    )
    _add_avg_count(metrics, "researchclaw/raw_reward/final", _numbers(summaries, "final_raw_reward"))
    _add_avg_count(metrics, "researchclaw/raw_reward/codebase", _numbers(summaries, "codebase_raw_reward"))
    _add_avg_count(metrics, "researchclaw/raw_reward/llmjudge", _numbers(summaries, "llmjudge_raw_reward"))
    _add_avg_count(metrics, "researchclaw/weight/codebase", _numbers(summaries, "codebase_weight"))
    _add_avg_count(metrics, "researchclaw/weight/llmjudge", _numbers(summaries, "llmjudge_weight"))
    _add_avg_count(metrics, "researchclaw/codebase/duration_sec", _numbers(summaries, "codebase_duration_sec"))

    for key, prefix in (
        ("judge_call_count", "researchclaw/judge/call_count"),
        ("judge_direct_call_count", "researchclaw/judge/direct_call_count"),
        ("judge_agentic_call_count", "researchclaw/judge/agentic_call_count"),
        ("judge_turn_count", "researchclaw/judge/turn_count"),
        ("judge_criteria_count", "researchclaw/judge/criteria_count"),
        ("judge_scored_criteria_count", "researchclaw/judge/scored_criteria_count"),
        ("judge_missing_criteria_count", "researchclaw/judge/missing_criteria_count"),
        ("judge_needs_evidence_count", "researchclaw/judge/needs_evidence_count"),
        ("judge_direct_response_len", "researchclaw/judge/direct_response_len"),
        ("judge_agentic_result_len", "researchclaw/judge/agentic_result_len"),
        ("judge_response_len", "researchclaw/judge/response_len"),
        ("judge_direct_duration_sec", "researchclaw/judge/direct_duration_sec"),
        ("judge_agentic_duration_sec", "researchclaw/judge/agentic_duration_sec"),
        ("judge_duration_sec", "researchclaw/judge/duration_sec"),
    ):
        _add_avg_count_sum(metrics, prefix, _numbers(summaries, key))

    _add_count_rate(
        metrics,
        "researchclaw/judge/nanobot_entered",
        [_truthy(summary.get("judge_nanobot_entered")) for summary in summaries],
    )

    solver_turns = [_record_turn_count(record) for record, _ in pairs]
    _add_avg_count(metrics, "researchclaw/solver/turn_count", [value for value in solver_turns if value is not None])

    solver_response_lens = [_record_solver_response_len(record) for record, _ in pairs]
    _add_avg_count(
        metrics,
        "researchclaw/solver/response_len",
        [value for value in solver_response_lens if value is not None],
    )
    return metrics


def _add_llmjudge_summary(
    summary: dict[str, Any],
    payload: dict[str, Any],
    llmjudge_reward: dict[str, Any],
    run_dir: Path | None,
) -> None:
    backend = str(llmjudge_reward.get("backend") or "").strip().lower()
    if backend:
        summary["llmjudge_backend"] = backend

    rules = llmjudge_reward.get("rules")
    if isinstance(rules, list):
        summary["judge_criteria_count"] = len(rules)

    normalization = _dict(llmjudge_reward.get("score_normalization"))
    matched_scores = _dict(normalization.get("matched_scores"))
    if matched_scores:
        summary["judge_scored_criteria_count"] = len(matched_scores)
    missing = normalization.get("missing")
    if isinstance(missing, list):
        summary["judge_missing_criteria_count"] = len(missing)

    parsed = _dict(llmjudge_reward.get("parsed"))
    if "judge_scored_criteria_count" not in summary:
        parsed_scores = _dict(parsed.get("scores"))
        if parsed_scores:
            summary["judge_scored_criteria_count"] = len(parsed_scores)
    needs_evidence = _needs_evidence_ids(llmjudge_reward)
    if needs_evidence is not None:
        summary["judge_needs_evidence_count"] = len(needs_evidence)

    privileged_feedback = payload.get("privileged_feedback")
    if not isinstance(privileged_feedback, dict):
        privileged_feedback = build_privileged_feedback(llmjudge_reward)
    if privileged_feedback is not None:
        summary["privileged_feedback"] = privileged_feedback

    direct_count, agentic_count = _judge_call_counts(llmjudge_reward, backend)
    summary["judge_direct_call_count"] = direct_count
    summary["judge_agentic_call_count"] = agentic_count
    summary["judge_nanobot_entered"] = 1 if agentic_count else 0
    summary["judge_call_count"] = direct_count + agentic_count
    summary["judge_turn_count"] = direct_count + agentic_count

    direct_files = _direct_response_paths(llmjudge_reward, backend, run_dir)
    _add_response_file_stats(summary, "judge_direct", direct_files)
    _add_agentic_stats(summary, llmjudge_reward, run_dir)
    _add_combined_judge_stats(summary)


def build_privileged_feedback(
    llmjudge_reward: dict[str, Any],
    *,
    max_items_per_section: int = 8,
    max_text_chars: int = 8_000,
) -> dict[str, Any] | None:
    """Build a compact, deterministic PI packet from the current judge result.

    The judge schema only provides global notes, so those notes remain global.
    Rubric-specific statements are derived only from rules, score anchors, and
    evidence requests rather than attributing judge prose to a criterion.
    """
    rules = llmjudge_reward.get("rules")
    parsed = _dict(llmjudge_reward.get("parsed"))
    scores = _dict(parsed.get("scores"))
    if not isinstance(rules, list) or not rules or not scores:
        return None

    problems: list[str] = []
    preserve: list[str] = []
    scored_items: list[tuple[float, int, str]] = []
    for index, raw_rule in enumerate(rules, 1):
        if not isinstance(raw_rule, dict):
            continue
        rule_id = f"criterion_{index}"
        score = _finite_number(scores.get(rule_id))
        if score is None:
            continue
        anchors = _numeric_score_anchors(raw_rule.get("scores"))
        maximum = max(anchors, default=1.0)
        current_text = _compact_anchor_text(anchors.get(score, ""), score=score)
        requirement_text = _compact_anchor_text(anchors.get(maximum, ""), score=maximum)
        name = _clean_feedback_text(raw_rule.get("name"), 240) or f"Requirement {index}"
        file_path = _clean_feedback_text(raw_rule.get("file_path"), 240)
        label = f"{name} ({file_path})" if file_path else name
        if score < maximum:
            requirement = f" Requirement: {requirement_text}" if requirement_text else ""
            assessment = f" Current assessment: {current_text}" if current_text else ""
            statement = f"{label}.{requirement}{assessment}".strip()
        else:
            detail = f": {requirement_text}" if requirement_text else ""
            statement = f"{label}{detail}"
        scored_items.append((score / maximum if maximum > 0 else score, index, statement))

    for normalized_score, _index, statement in sorted(scored_items, key=lambda item: (item[0], item[1])):
        if normalized_score < 1.0:
            problems.append(statement)
        else:
            preserve.append(statement)

    hints: list[str] = []
    triage = _dict(llmjudge_reward.get("triage"))
    triage_parsed = _dict(triage.get("parsed"))
    evidence_requests = _dict(triage_parsed.get("evidence_requests"))
    if not evidence_requests:
        evidence_requests = _dict(triage.get("evidence_requests"))
    for rule_id in _needs_evidence_ids(llmjudge_reward) or []:
        request = _dict(evidence_requests.get(str(rule_id)))
        reason = _clean_feedback_text(request.get("reason"), 500)
        paths = request.get("suggested_paths")
        clean_paths = [
            text
            for value in paths if (text := _clean_feedback_text(value, 180))
        ] if isinstance(paths, list) else []
        if reason:
            suffix = f" Check: {', '.join(clean_paths[:4])}." if clean_paths else ""
            hints.append(f"Verify local evidence: {reason}{suffix}")

    for label, notes in _judge_note_sources(llmjudge_reward):
        clean_notes = _clean_feedback_text(notes, 1_200)
        if clean_notes:
            hints.append(f"{label}: {clean_notes}")

    problems = _dedupe_text(problems)[:max_items_per_section]
    hints = _dedupe_text(hints)[:max_items_per_section]
    preserve = _dedupe_text(preserve)[:max_items_per_section]
    if not problems and not hints and not preserve:
        return None

    sections = ["Privileged feedback:"]
    for title, values in (("Problems", problems), ("Revision hints", hints), ("Preserve", preserve)):
        if values:
            sections.extend(["", f"{title}:", *(f"- {value}" for value in values)])
    text = "\n".join(sections)
    truncated = len(text) > max_text_chars
    if truncated:
        text = text[: max(1, max_text_chars - 18)].rstrip() + "\n...[truncated]"
    return {
        "version": 1,
        "text": text,
        "problem_count": len(problems),
        "hint_count": len(hints),
        "preserve_count": len(preserve),
        "truncated": truncated,
    }


def _numeric_score_anchors(value: Any) -> dict[float, str]:
    if not isinstance(value, dict):
        return {}
    anchors: dict[float, str] = {}
    for raw_score, description in value.items():
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score == score and score not in (float("inf"), float("-inf")):
            anchors[score] = _clean_feedback_text(description, 700)
    return anchors


def _compact_anchor_text(value: str, *, score: float) -> str:
    text = value.strip()
    prefixes = (
        f"Score {score:g} if the artifact satisfies this atomic criterion. Criterion-specific requirement: ",
        f"Score {score:g} only if ",
        f"Score {score:g} if ",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    if text:
        text = text[0].upper() + text[1:]
    return text


def _judge_note_sources(llmjudge_reward: dict[str, Any]) -> list[tuple[str, Any]]:
    backend = str(llmjudge_reward.get("backend") or "").strip().lower()
    if backend == "hybrid_agentic":
        triage = _dict(llmjudge_reward.get("triage"))
        agentic = _dict(llmjudge_reward.get("agentic"))
        return [
            ("Judge summary", _dict(triage.get("parsed")).get("notes")),
            ("Evidence-check summary", _dict(agentic.get("parsed")).get("notes")),
        ]
    return [("Judge summary", _dict(llmjudge_reward.get("parsed")).get("notes"))]


def _clean_feedback_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())[:limit].strip()


def _dedupe_text(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _judge_call_counts(
    llmjudge_reward: dict[str, Any],
    backend: str,
) -> tuple[int, int]:
    if backend == "direct":
        return 1, 0
    if backend == "agentic":
        return 0, 1
    if backend == "hybrid_agentic":
        agentic_count = 1 if isinstance(llmjudge_reward.get("agentic"), dict) else 0
        return 1, agentic_count
    return 0, 0


def _needs_evidence_ids(llmjudge_reward: dict[str, Any]) -> list[Any] | None:
    triage = _dict(llmjudge_reward.get("triage"))
    values = triage.get("needs_evidence")
    if isinstance(values, list):
        return values
    parsed = _dict(llmjudge_reward.get("parsed"))
    values = parsed.get("triage_needs_evidence")
    if isinstance(values, list):
        return values
    return None


def _direct_response_paths(llmjudge_reward: dict[str, Any], backend: str, run_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    if backend == "direct":
        _append_path(paths, llmjudge_reward.get("judge_response_path"))
        if run_dir is not None:
            _append_path(paths, run_dir / "researchclaw_judge_response.json")
    elif backend == "hybrid_agentic":
        triage = _dict(llmjudge_reward.get("triage"))
        _append_path(paths, triage.get("response_path"))
        if run_dir is not None:
            _append_path(paths, run_dir / "researchclaw_judge_triage_response.json")
    return _dedupe_existing_paths(paths)


def _add_agentic_stats(summary: dict[str, Any], llmjudge_reward: dict[str, Any], run_dir: Path | None) -> None:
    agentic = _dict(llmjudge_reward.get("agentic"))
    if not agentic and run_dir is not None:
        path = run_dir / "researchclaw_agentic_judge_response.json"
        if path.is_file():
            agentic = _read_json_dict(path)
    if not agentic:
        return

    _copy_number(summary, "judge_agentic_duration_sec", agentic.get("duration_sec"))
    parsed = _dict(agentic.get("parsed"))
    if parsed:
        summary["judge_agentic_result_len"] = len(json.dumps(parsed, ensure_ascii=False, sort_keys=True))


def _add_agentic_file_stats(summary: dict[str, Any], files: list[Path]) -> None:
    if not files:
        return
    data = [_read_json_dict(path) for path in files]
    durations = [_finite_number(item.get("duration_sec")) for item in data]
    durations = [value for value in durations if value is not None]
    if durations:
        summary["judge_agentic_duration_sec"] = sum(durations) / len(durations)
    result_lens: list[int] = []
    for item in data:
        parsed = _dict(item.get("parsed"))
        if parsed:
            result_lens.append(len(json.dumps(parsed, ensure_ascii=False, sort_keys=True)))
    if result_lens:
        summary["judge_agentic_result_len"] = sum(result_lens) / len(result_lens)
    _add_combined_judge_stats(summary)


def _add_response_file_stats(summary: dict[str, Any], key_prefix: str, files: list[Path]) -> None:
    if not files:
        return
    response_lens: list[int] = []
    durations: list[float] = []
    for path in files:
        data = _read_json_dict(path)
        text = data.get("text")
        if isinstance(text, str):
            response_lens.append(len(text))
        duration = _finite_number(data.get("duration_sec"))
        if duration is not None:
            durations.append(duration)
    if response_lens:
        summary[f"{key_prefix}_response_len"] = sum(response_lens) / len(response_lens)
    if durations:
        summary[f"{key_prefix}_duration_sec"] = sum(durations) / len(durations)


def _add_combined_judge_stats(summary: dict[str, Any]) -> None:
    response_lens = [
        value
        for value in (
            _finite_number(summary.get("judge_direct_response_len")),
            _finite_number(summary.get("judge_agentic_result_len")),
        )
        if value is not None
    ]
    if response_lens:
        summary["judge_response_len"] = sum(response_lens) / len(response_lens)

    durations = [
        value
        for value in (
            _finite_number(summary.get("judge_direct_duration_sec")),
            _finite_number(summary.get("judge_agentic_duration_sec")),
        )
        if value is not None
    ]
    if durations:
        summary["judge_duration_sec"] = sum(durations) / len(durations)


def _record_reward_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    summary = record.get("researchclaw_reward")
    if isinstance(summary, dict):
        return summary
    run = record.get("run")
    if isinstance(run, dict) and isinstance(run.get("researchclaw_reward"), dict):
        return run["researchclaw_reward"]
    return None


def _record_turn_count(record: dict[str, Any]) -> float | None:
    run = record.get("run")
    if not isinstance(run, dict):
        return None
    turns = run.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    return float(sum(1 for turn in turns if isinstance(turn, dict)))


def _record_solver_response_len(record: dict[str, Any]) -> float | None:
    requests = record.get("requests")
    if not isinstance(requests, dict):
        return None
    values = _number_list(requests.get("response_len_values"))
    if values:
        return sum(values) / len(values)
    return _finite_number(requests.get("response_len_mean"))


def _numbers(summaries: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for summary in summaries:
        value = _finite_number(summary.get(key))
        if value is not None:
            values.append(value)
    return values


def _add_avg_count(metrics: dict[str, int | float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    metrics[f"{prefix}/avg"] = sum(values) / len(values)
    metrics[f"{prefix}/count"] = len(values)


def _add_avg_count_sum(metrics: dict[str, int | float], prefix: str, values: list[float]) -> None:
    if not values:
        return
    _add_avg_count(metrics, prefix, values)
    metrics[f"{prefix}/sum"] = sum(values)


def _add_count_rate(metrics: dict[str, int | float], prefix: str, flags: list[bool]) -> None:
    total = len(flags)
    count = sum(1 for flag in flags if flag)
    metrics[f"{prefix}/count"] = count
    metrics[f"{prefix}/rate"] = count / total if total else 0.0


def _number_list(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for item in raw:
        value = _finite_number(item)
        if value is not None:
            values.append(value)
    return values


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    return False


def _copy_number(target: dict[str, Any], key: str, value: Any) -> None:
    number = _finite_number(value)
    if number is not None:
        target[key] = number


def _copy_text(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        target[key] = text


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _existing_files(root: Path, names: list[str]) -> list[Path]:
    return [path for name in names if (path := root / name).is_file()]


def _append_path(paths: list[Path], value: Any) -> None:
    if isinstance(value, Path):
        paths.append(value)
    elif isinstance(value, str) and value.strip():
        paths.append(Path(value))


def _dedupe_existing_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
