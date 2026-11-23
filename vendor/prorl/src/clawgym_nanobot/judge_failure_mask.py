"""Classify exhausted judge attempts without turning missing scores into zero."""
import json
import hashlib
import re
from pathlib import Path

def terminal_evidence(log_path):
    """Carry actual terminal events across hosts and sandbox cleanup."""
    data = Path(log_path).read_bytes()
    events = []
    for line in data.decode('utf-8').splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(event, dict) and event.get('type') == 'result':
            events.append(event)
    return dict(version=1, log_sha256=hashlib.sha256(data).hexdigest(), events=events)

def missing_score_reason(result):
    native = (result or {}).get("native_result") or {}
    metrics = native.get("judge_model_metrics") or {}
    if native.get("judge_timed_out") or native.get("judge_permanent_api_error_status"):
        return None
    if native.get("judge_result_complete") or native.get("judge_schema_complete"):
        return None
    count = metrics.get("request_count", 0)
    if count <= 0 or metrics.get("successful_request_count") != count:
        return None
    try:
        evidence = native.get('judge_terminal_evidence')
        if evidence is None:
            evidence = terminal_evidence(native['judge_log_path'])
        if (not isinstance(evidence, dict) or evidence.get('version') != 1
                or not re.fullmatch(r'[0-9a-f]{64}', str(evidence.get('log_sha256', '')))):
            return None
        terminal = evidence.get('events')
        if not isinstance(terminal, list):
            return None
        attempts = native.get("judge_attempts", 0)
        if attempts < 3 or len(terminal) != attempts:
            return None
        reasons = []
        for event in terminal:
            if not isinstance(event, dict) or event.get('type') != 'result' or event.get("api_error_status"):
                return None
            errors = event.get('errors', [])
            if not isinstance(errors, list):
                return None
            if event.get("subtype") == "error_max_turns" and event.get("is_error") is True:
                # Claude Code reports the exhausted budget in errors. Allow only
                # this exact diagnostic, never API/source/timeout errors.
                if any(not isinstance(error, str) or not re.fullmatch(
                        r'Reached maximum number of turns \([1-9][0-9]*\)', error)
                       for error in errors):
                    return None
                reasons.append("judge_max_turns")
            elif (event.get("subtype") == "success" and event.get("is_error") is False
                  and not errors
                  and event.get("terminal_reason") == "completed"
                  and isinstance(event.get("result"), str) and not event["result"].strip()):
                reasons.append("judge_empty_completion")
            else:
                return None
        return reasons[0] if len(set(reasons)) == 1 else "judge_exhausted_without_score"
    except (KeyError, OSError, TypeError):
        return None

def is_turn_budget_failure(result):
    """Compatibility predicate for the narrower historical category."""
    return missing_score_reason(result) == "judge_max_turns"
