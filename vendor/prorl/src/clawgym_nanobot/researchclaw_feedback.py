from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .researchclaw_metrics import build_privileged_feedback
from .user_agent import call_openai_compatible, parse_json_object


SYSTEM_PROMPT = """You are the user who requested a research task and has now reviewed the candidate's delivered work.

Turn the supplied evidence packet into one natural, concise paragraph of user feedback.

Rules:
- Preserve the packet's factual meaning. Do not add, infer, or repair facts that are absent from it. Candidate-specific review notes take precedence over generic requirement/score-anchor wording when they differ.
- Preserve distinctions such as wrong versus unverified, absent versus present in the wrong file, and incomplete versus internally inconsistent. Do not collapse a file-placement problem into a claim that the work is missing.
- Mention concrete strengths that should be preserved when the packet supports them, and the highest-priority corrections when it supports problems. Do not force praise or criticism without evidence.
- If a broad strength conflicts with a specific problem, use the narrower defensible wording. For example, say that all requested rows are present rather than that all values are correct when named values are wrong.
- Address the candidate directly and discuss its delivered files or answer. Do not evaluate the source documents, the task, the judge, the rubric, or another answer.
- Do not mention criterion IDs, numeric reward/score, hidden references, judging, rubrics, or this evidence packet.
- Avoid a mechanical checklist. Write one sentence or one short paragraph in the same language as the original request when practical.
- Return exactly one JSON object: {"feedback": "..."}.
"""


NOISY_SYSTEM_PROMPT = 'You are generating deliberately incorrect user feedback for a controlled research experiment. Use the original request, evidence packet, and candidate-specific notes to understand the task and what the candidate actually did.\n\nWrite ONE natural paragraph of 3–5 sentences and approximately 60–100 English words. Address the candidate directly in a concise, confident user voice.\n\nGive substantively wrong or misleading feedback, with concrete requests to revise the work in an incorrect direction. The errors must conflict with the supplied task or evidence; a mere change in tone or preference is not enough. You may introduce multiple errors, and there is no restriction on their type or scope. Errors may concern arithmetic, ranking or metric direction, numeric or factual claims, file or table existence, evidence interpretation, generalization, theoretical applicability, or a reversal of the overall assessment. Broad claims such as applicability to all settings are also allowed. These are examples, not a checklist: choose freely rather than forcing every error type into each response.\n\nKeep the feedback connected to the actual task and deliverable. It may mix correct and incorrect statements, but do not dilute it into mostly correct feedback. Do not explain or reveal that the feedback is intentionally wrong. Avoid lists, long recaps, and unnecessary detail.\n\nReturn exactly {"feedback": "..."}, with no other keys or text. Only the feedback text is being perturbed; do not output or modify any numeric reward field.\n'


def feedback_noise_selected(task_request: str) -> bool:
    probability = float(os.environ.get("RESEARCHCLAW_USER_FEEDBACK_NOISE_P", "0"))
    if not 0 <= probability <= 1:
        raise ValueError("User feedback noise probability must be in [0, 1]")
    seed = os.environ.get("RESEARCHCLAW_USER_FEEDBACK_NOISE_SEED", "20261108-user-noise-v1")
    digest = hashlib.sha256((seed + "|" + task_request.strip()).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < probability


def build_feedback_messages(*, task_request: str, evidence_packet: dict[str, Any]) -> list[dict[str, str]]:
    packet_text = str(evidence_packet.get("text") or "").strip()
    review_notes = evidence_packet.get("review_notes")
    notes_text = "\n".join(
        f"- {str(value).strip()}" for value in review_notes or [] if str(value).strip()
    )
    return [
        {"role": "system", "content": NOISY_SYSTEM_PROMPT if feedback_noise_selected(task_request) else SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Original request:\n"
                f"{task_request.strip()}\n\n"
                "Evidence from the completed review:\n"
                f"{packet_text}\n\n"
                "Candidate-specific review notes:\n"
                f"{notes_text or '- None supplied.'}\n\n"
                "Write the natural user feedback now."
            ),
        },
    ]


def generate_natural_feedback(
    *,
    task_request: str,
    llmjudge_reward: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout_sec: int = 300,
    max_tokens: int = 4_096,
    temperature: float = 0.7,
    enable_thinking: bool = True,
    max_attempts: int = 2,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    packet = build_privileged_feedback(
        llmjudge_reward, max_items_per_section=5, max_text_chars=6_000
    )
    if packet is None:
        raise ValueError("judge result cannot produce a grounded feedback packet")
    packet = {**packet, "review_notes": _review_notes(llmjudge_reward)}
    messages = build_feedback_messages(task_request=task_request, evidence_packet=packet)
    request_record = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if artifact_dir is not None:
        _write_json(artifact_dir / "request.json", request_record)
        _write_json(artifact_dir / "rule_feedback.json", packet)

    started = time.time()
    feedback = ""
    raw = ""
    last_error: Exception | None = None
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            raw = call_openai_compatible(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
                temperature=temperature,
                payload_overrides={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
            if artifact_dir is not None:
                (artifact_dir / f"raw_response_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")
            parsed = parse_json_object(raw)
            value = parsed.get("feedback")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("feedback response must contain a non-empty feedback string")
            feedback = value.strip()
            break
        except Exception as exc:
            last_error = exc
    if not feedback:
        raise ValueError(f"feedback generation failed after {attempts} attempts: {last_error}")
    result = {
        "version": 2,
        "text": feedback.strip(),
        "naturalized": True,
        "source": "llmjudge_grounded_packet",
        "model": model,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "duration_sec": round(time.time() - started, 3),
        "attempts": attempt,
        "rule_feedback": packet,
    }
    if artifact_dir is not None:
        (artifact_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
        _write_json(artifact_dir / "result.json", result)
    return result


def maybe_attach_natural_feedback(
    *, task_request: str, llmjudge_reward: dict[str, Any], run_dir: Path
) -> dict[str, Any] | None:
    """Generate natural feedback when configured; retain a deterministic fallback.

    Feedback generation is a post-judge presentation step. A failure here must
    not invalidate or change the already computed GT reward.
    """
    packet = build_privileged_feedback(
        llmjudge_reward, max_items_per_section=5, max_text_chars=6_000
    )
    if packet is None:
        return None
    # Self-reward reuses the native judge for scoring, but its post-judge
    # natural-language feedback is neither consumed nor persisted as an input
    # to another stage.  Skip that extra model call while retaining the
    # deterministic diagnostic packet.  GT judging keeps its existing natural
    # feedback behavior.
    self_reward_compat = (
        os.environ.get("RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER", "").strip() == "0"
        and os.environ.get("RESEARCHCLAW_CODE_WEIGHT", "").strip() in {"0", "0.0"}
    )
    if (
        _truthy(os.environ.get("RESEARCHCLAW_DISABLE_NATURAL_USER_FEEDBACK"))
        or self_reward_compat
    ):
        return {
            **packet,
            "naturalized": False,
            "source": "deterministic_rule_feedback",
            "error": None,
        }
    base_url = _first_env("RESEARCHCLAW_USER_FEEDBACK_BASE_URL", "RESEARCHCLAW_JUDGE_BASE_URL")
    model = _first_env("RESEARCHCLAW_USER_FEEDBACK_MODEL", "RESEARCHCLAW_JUDGE_MODEL")
    api_key = _feedback_api_key()
    explicitly_enabled = _truthy(os.environ.get("RESEARCHCLAW_NATURAL_USER_FEEDBACK"))
    configured = bool(base_url and model and api_key)
    if not configured:
        return {
            **packet,
            "naturalized": False,
            "source": "deterministic_rule_feedback",
            "error": "natural feedback endpoint is not configured" if explicitly_enabled else None,
        }
    try:
        return generate_natural_feedback(
            task_request=task_request,
            llmjudge_reward=llmjudge_reward,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_sec=_int_env("RESEARCHCLAW_USER_FEEDBACK_TIMEOUT", 300),
            max_tokens=_int_env("RESEARCHCLAW_USER_FEEDBACK_MAX_TOKENS", 4_096),
            temperature=_float_env("RESEARCHCLAW_USER_FEEDBACK_TEMPERATURE", 0.7),
            enable_thinking=_truthy(os.environ.get("RESEARCHCLAW_USER_FEEDBACK_ENABLE_THINKING", "1")),
            artifact_dir=run_dir / "researchclaw_user_feedback",
        )
    except Exception as exc:
        error_result = {
            **packet,
            "naturalized": False,
            "source": "deterministic_rule_feedback",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(run_dir / "researchclaw_user_feedback" / "error.json", error_result)
        return error_result


def _feedback_api_key() -> str:
    value = _first_env("RESEARCHCLAW_USER_FEEDBACK_API_KEY", "RESEARCHCLAW_JUDGE_API_KEY")
    if value:
        return value
    for name in ("RESEARCHCLAW_USER_FEEDBACK_API_KEY_FILE", "RESEARCHCLAW_JUDGE_API_KEY_FILE"):
        path = os.environ.get(name, "").strip()
        if path:
            return Path(path).read_text(encoding="utf-8").strip()
    return ""


def _review_notes(llmjudge_reward: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    if str(llmjudge_reward.get("backend") or "").strip().lower() == "hybrid_agentic":
        triage = llmjudge_reward.get("triage")
        agentic = llmjudge_reward.get("agentic")
        if isinstance(triage, dict):
            parsed = triage.get("parsed")
            values.append(parsed.get("notes") if isinstance(parsed, dict) else None)
        if isinstance(agentic, dict):
            parsed = agentic.get("parsed")
            values.append(parsed.get("notes") if isinstance(parsed, dict) else None)
    else:
        parsed = llmjudge_reward.get("parsed")
        values.append(parsed.get("notes") if isinstance(parsed, dict) else None)
    result: list[str] = []
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text and text.casefold() not in {item.casefold() for item in result}:
            result.append(text[:4_000])
    return result


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
