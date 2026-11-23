from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import AgentSpec, _write_cleaned_transcript, _write_config
from .common import atomic_json, read_json
from .contracts import load_rubric
from .native import import_native, ws_score_instructions

SCORE_CONTRACT = "wsclaw_native_judge_v2"


def self_score_prompt(native_prompt: str, spec: AgentSpec) -> str:
    return native_prompt + "\n\nFollow AGENTS.md for Nanobot-specific path and delivery instructions.\n"


def artifact_gate_zero(*, gate: dict[str, Any], rubric: dict[str, Any],
                       slot_id: str, session_id: str, run_dir: Path) -> dict[str, Any]:
    """Return a trainable policy-failure zero, without a judge/model call."""
    if gate.get("self_score_gate_version") != 1 or gate.get("passed") is not False:
        raise ValueError("artifact zero requires a failed trusted gate")
    rows = gate.get("artifacts")
    if not isinstance(rows, list) or not rows or not any(r.get("valid") is False for r in rows):
        raise ValueError("artifact zero requires concrete invalid deliverable evidence")
    run_dir.mkdir(parents=True, exist_ok=False)
    value = {
        "schema_version": "self_reward_score_v1", "slot_id": slot_id,
        "criterion_scores": {item["id"]: 0.0 for item in rubric["criteria"]},
        "weighted_score": 0.0, "notes": "Required artifact failed deterministic validity gate.",
        "score_contract": SCORE_CONTRACT, "artifact_validity_gate": gate,
        "deterministic_policy_failure": True, "judge_executed": False,
    }
    status = {"session_id": session_id, "score_contract": SCORE_CONTRACT,
              "status": "accepted", "error": None, "attempts": 0,
              "reason": "required_artifact_invalid", "judge_executed": False,
              "artifact_validity_gate": gate}
    atomic_json(run_dir / "accepted.json", value)
    atomic_json(run_dir / "status.json", status)
    atomic_json(run_dir / "artifact-validity-gate.json", gate)
    return {"status": status, "value": value}


def run_wsclaw_self_score(
    *, project_root: Path, workspace: Path, prompt_path: Path, rubric_path: Path,
    result_relpath: str, run_dir: Path, session_id: str, slot_id: str, spec: AgentSpec,
) -> dict[str, Any]:
    if result_relpath != "judge_result.json":
        raise ValueError("native WSClaw judge requires judge_result.json")
    rubric = load_rubric(rubric_path)
    manifest = read_json(workspace.parent.parent / "SCORE_WORKSPACE_MANIFEST.json")
    if manifest.get("score_contract") != SCORE_CONTRACT:
        raise ValueError("rebuild the scoring workspace with the native judge contract")
    gate = manifest.get("artifact_validity_gate")
    if not isinstance(gate, dict) or gate.get("self_score_gate_version") != 1 or type(gate.get("passed")) is not bool:
        raise ValueError("self scoring requires a fresh trusted artifact-validity gate")
    if not gate["passed"]:
        return artifact_gate_zero(gate=gate, rubric=rubric, slot_id=slot_id,
                                  session_id=session_id, run_dir=run_dir)
    runtime, _tasks, ws, _rc = import_native(project_root)
    task_dir = Path(manifest["task_dir"])
    # Match ResearchClaw self scoring: configured timeout/context/tool budget,
    # thinking enabled and a minimum 32K output budget.
    spec = replace(spec, max_tokens=max(32768, spec.max_tokens), enable_thinking=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "nanobot_logs"
    logs.mkdir()
    session_ids: list[str] = []
    initial_prompt = self_score_prompt(ws.judge_user_message(workspace), spec)
    prompt_path.write_text(initial_prompt + "\n", encoding="utf-8")
    instructions = ws_score_instructions((workspace / "CLAUDE.md").read_text(encoding="utf-8"))
    instructions += (
        f"\nBudget per attempt: {spec.max_tool_calls} tool iterations; "
        f"{spec.timeout_sec} seconds. Finish once the evidence is sufficient.\n"
    )
    (workspace / "AGENTS.md").write_text(instructions, encoding="utf-8")

    def runtime_factory(judge_workspace: Path) -> Path:
        root = ws.create_judge_runtime_workspace(judge_workspace, temporary_root=run_dir)
        # AGENTS.md contains native instructions plus the Nanobot adapter.
        # The native factory also links the original task, rubric and review.
        shutil.copy2(judge_workspace / "AGENTS.md", root / "AGENTS.md")
        (root / "SOUL.md").write_text(
            "You are the WSClaw judge. Follow AGENTS.md and the scoring prompt.\n",
            encoding="utf-8",
        )
        return root

    def attempt_runner(command: list[str], **kwargs: Any) -> tuple[int, bool, float]:
        root = kwargs["cwd"]
        attempt_session = f"{session_id}-attempt-{len(session_ids) + 1}"
        session_ids.append(attempt_session)
        local_logs = root / "nanobot_logs"
        config_dir = Path(tempfile.mkdtemp(prefix=".judge-config-", dir=run_dir))
        config_path = config_dir / "config.json"
        _write_config(config_path, root, spec, wsclaw=True)
        config = read_json(config_path)
        config["tools"]["exec"]["allowedEnvKeys"] += [
            "WSCLAW_INSPECT_ALIAS_FILE", "WSCLAW_INSPECT_DEFAULT_MAX_CHARS",
            "WSCLAW_INSPECT_COMPACT", "PATH", "TMPDIR", "PYTHONDONTWRITEBYTECODE",
        ]
        atomic_json(config_path, config)
        config_path.chmod(0o600)
        identity = ws.configured_unprivileged_identity(kwargs["env"])
        if identity and os.geteuid() == 0:
            os.chown(config_dir, *identity)
            os.chown(config_path, *identity)
        env_overrides = {
            "NANOBOT_TASK_ID": attempt_session,
            "NANOBOT_NATIVE_JUDGE_WORKSPACE": "1",
            "NANOBOT_WEB_ENABLE": "0",
            "NANOBOT_RESTRICT_TO_WORKSPACE": "0",
            "NANOBOT_ENABLE_THINKING": "1",
            "NANOBOT_SHOW_REASONING": "1",
            "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
            "NANOBOT_SELF_REWARD_EXIT_ON_TOOL_BUDGET": "1",
            # Only post-turn archive cleanup is bounded; scoring retains its full budget.
            "NANOBOT_RL_BACKGROUND_DRAIN_TIMEOUT_S": "30",
            "NANOBOT_STRIP_REASONING_CONTEXT": "0",
            "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY": "preserve",
            "NO_PROXY": "*", "no_proxy": "*",
        }
        script = runtime._nanobot_shell_script(
            workspace_dir=root, config_path=config_path, session_id=attempt_session,
            turn_type="side", message=command[-1], log_dir=local_logs,
            env_overrides=env_overrides,
        )
        try:
            # Use the very same Landlock write boundary as the GT judge.
            outcome = ws.run_process(["bash", "-c", script], **kwargs)
            source = local_logs / attempt_session
            # Native parsers consume structured stream events. Keep Nanobot's
            # human-readable stdout separate (it can contain scalar JSON lines).
            raw_log = kwargs["log_path"].with_suffix(".nanobot.log")
            kwargs["log_path"].replace(raw_log)
            events = []
            if source.exists():
                shutil.copytree(source, logs / attempt_session, dirs_exist_ok=True)
                for path in sorted(source.glob("*_agent-end.json")):
                    payload = read_json(path)
                    for index, step in enumerate(payload.get("tool_steps") or []):
                        call_id = f"{attempt_session}-{index}"
                        events.append({"type": "assistant", "message": {
                            "id": call_id, "content": [{
                                "type": "tool_use", "id": call_id,
                                "name": step.get("name"), "input": step.get("arguments") or {},
                            }],
                        }})
                        events.append({"type": "user", "message": {"content": [{
                            "type": "tool_result", "tool_use_id": call_id,
                            "content": step.get("result"),
                        }]}})
                    if payload.get("output") is not None:
                        events.append({"type": "result", "result": payload["output"]})
            if not any(event["type"] == "result" for event in events):
                candidate = ws.extract_json_object(raw_log.read_text(encoding="utf-8", errors="replace"))
                if candidate is not None:
                    events.append({"type": "result", "result": candidate})
            kwargs["log_path"].write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
                encoding="utf-8",
            )
            return outcome
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

    args = argparse.Namespace(
        wsbench_python=ws.DEFAULT_WSBENCH_PYTHON,
        judge_api_base=spec.base_url, judge_model=spec.model,
        judge_timeout=spec.timeout_sec, judge_max_turns=spec.max_tool_calls,
        judge_max_attempts=ws.JUDGE_MAX_ATTEMPTS if spec.allow_retries else 1,
        judge_terminal_exit_codes=(86,),
        judge_effort="high", claude_bin=ws.DEFAULT_CLAUDE_BIN,
    )
    status: dict[str, Any] = {
        "session_id": session_id, "score_contract": SCORE_CONTRACT,
        "model": spec.model, "timeout_sec": spec.timeout_sec,
        "max_tool_iterations": spec.max_tool_calls, "max_tokens": spec.max_tokens,
        "context_window": spec.context_window, "enable_thinking": True,
        "final_message_recovery": True, "max_attempts": args.judge_max_attempts,
        "result_path": str(workspace / result_relpath),
    }
    try:
        native = ws.run_judge(
            workspace, task_dir, run_dir, args, attempt_runner=attempt_runner,
            runtime_factory=runtime_factory, initial_prompt=initial_prompt,
        )
        native["judge_agent"] = "nanobot_self_reward"
        native["rubric_source"] = "generated_self_rubric"
        atomic_json(run_dir / "judge_result.json", native)
        if not native["judge_result_complete"]:
            raise ValueError("native WSClaw judge did not deliver a complete result")
        # Match the native GT completion gate as well as its result schema.
        if native.get("judge_timed_out") or native.get("judge_exit_code") != 0:
            raise ValueError("native WSClaw judge timed out or exited unsuccessfully")
        # Map the native normalized scores without a second, stricter validator.
        value = {
            "schema_version": "self_reward_score_v1", "slot_id": slot_id,
            "criterion_scores": {
                item["id"]: native["rubric_scores"][str(i)]
                for i, item in enumerate(rubric["criteria"], 1)
            },
            "weighted_score": native["score"], "notes": str(native["feedback"]),
            "score_contract": SCORE_CONTRACT, "native_result": native,
            "artifact_validity_gate": gate,
        }
        status.update(status="accepted", error=None, attempts=native["judge_attempts"])
        atomic_json(run_dir / "accepted.json", value)
        return {"status": status, "value": value}
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(run_dir / "failed.json", status)
        raise
    finally:
        _write_cleaned_transcript(logs, session_ids, run_dir / "cleaned_transcript.json")
        atomic_json(run_dir / "status.json", status)
