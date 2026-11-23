from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .common import atomic_json, read_json, read_secret, tree_manifest
from .contracts import canonicalize_native_score, load_rubric, normalize_equal_weight_rubric
from .native import import_native


@dataclass(frozen=True)
class AgentSpec:
    model: str
    base_url: str
    api_key_file: Path
    temperature: float
    timeout_sec: int
    max_tool_calls: int
    max_tokens: int
    context_window: int
    enable_thinking: bool = True
    allow_retries: bool = True


def _write_config(path: Path, workspace: Path, spec: AgentSpec, *, wsclaw: bool = False) -> None:
    api_base = spec.base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    config = {
        "providers": {
            "custom": {
                "apiBase": api_base,
                "apiKey": read_secret(spec.api_key_file),
                "extraBody": {
                    "stream": False,
                    "top_p": 1.0,
                    "top_k": -1,
                    "chat_template_kwargs": {"enable_thinking": spec.enable_thinking},
                },
            }
        },
        "agents": {
            "defaults": {
                "workspace": str(workspace),
                "provider": "custom",
                "model": spec.model,
                "maxTokens": spec.max_tokens,
                "contextWindowTokens": spec.context_window,
                "temperature": spec.temperature,
                "maxToolIterations": spec.max_tool_calls,
                **({} if wsclaw else {"botIcon": ""}),
                "timezone": "Asia/Shanghai",
            }
        },
        "channels": {"sendProgress": False, "sendToolHints": False, "showReasoning": not wsclaw},
        "tools": {
            "restrictToWorkspace": not wsclaw,
            "web": {"enable": False, "noWeb": True},
            "exec": {"enable": True, "timeout": 120 if wsclaw else 60,
                     "noWeb": True, "noWebCliTimeout": 5,
                     "allowedEnvKeys": ["PYTHONPATH"]},
        },
    }
    atomic_json(path, config)


def _tool_calls(log_root: Path, session_id: str) -> int:
    total = 0
    session = log_root / session_id
    for path in session.glob("*_agent-end.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for step in payload.get("llm_steps") or []:
            if isinstance(step, dict) and isinstance(step.get("tool_calls"), list):
                total += len(step["tool_calls"])
    return total


def _write_cleaned_transcript(log_root: Path, session_ids: list[str], destination: Path) -> None:
    records: list[dict[str, Any]] = []
    for session_id in session_ids:
        for path in sorted((log_root / session_id).glob("*_agent-end.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for index, step in enumerate(payload.get("llm_steps") or []):
                if not isinstance(step, dict):
                    continue
                records.append(
                    {
                        "session_id": session_id,
                        "source": path.name,
                        "step": index,
                        "request_messages": step.get("request_messages") or step.get("messages") or [],
                        "response_content": step.get("response_content") or step.get("response_content_stripped") or step.get("content"),
                        "reasoning_content": step.get("reasoning_content"),
                        "tool_calls": step.get("tool_calls") or [],
                        "tool_results": step.get("tool_results") or [],
                    }
                )
    atomic_json(destination, {"session_ids": session_ids, "steps": records})


def _visible_files(workspace: Path, result_relpath: str) -> dict[str, str]:
    # Nanobot maintains these workspace-scaffold paths independently of the
    # task agent.  Exclude them from the result-only mutation contract while
    # continuing to hash TASK/ANCHOR/evidence and all deliverable files.
    manifest = tree_manifest(
        workspace,
        excludes={
            ".git", ".gitignore", ".nanobot", "sessions", "memory",
            "USER.md", "HEARTBEAT.md",
        },
    )
    manifest.pop(result_relpath, None)
    return manifest


def run_strict_agent(
    *,
    project_root: Path,
    workspace: Path,
    prompt: str,
    result_relpath: str,
    run_dir: Path,
    session_id: str,
    spec: AgentSpec,
    validator: Callable[[Path], Any],
    wsclaw: bool = False,
) -> dict[str, Any]:
    runtime, _tasks, _ws, _rc = import_native(project_root)
    result_path = workspace / result_relpath
    result_path.unlink(missing_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "nanobot_logs"
    logs.mkdir()
    before = _visible_files(workspace, result_relpath)
    started = time.monotonic()
    # Nanobot's provider config contains the API key. Keep it in node-local
    # ephemeral storage and never persist it with evaluation artifacts.
    local_tmp = os.environ.get("SELF_REWARD_LOCAL_TMP", "/tmp")
    Path(local_tmp).mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sr-agent-config-", dir=local_tmp) as raw_state:
        config_path = Path(raw_state) / "config.json"
        _write_config(config_path, workspace, spec, wsclaw=wsclaw)
        proc = runtime._run_nanobot_turn(
            workspace_dir=workspace,
            config_path=config_path,
            session_id=session_id,
            turn_type="side",
            message=prompt,
            log_dir=logs,
            timeout=spec.timeout_sec,
            env_overrides={
                "NANOBOT_TASK_ID": session_id,
                "NANOBOT_WEB_ENABLE": "0",
                "NANOBOT_RESTRICT_TO_WORKSPACE": "0" if wsclaw else "1",
                "NANOBOT_ENABLE_THINKING": "1" if spec.enable_thinking else "0",
                "NANOBOT_SHOW_REASONING": "0" if wsclaw else "1",
                "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
                "NANOBOT_SELF_REWARD_EXIT_ON_TOOL_BUDGET": "1",
            },
        )
        processes = [proc]
        session_ids = [session_id]
        delivery_repair_reason: str | None = None
        if proc.returncode == 0:
            if not result_path.is_file():
                delivery_repair_reason = f"you did not create {result_relpath}"
            else:
                try:
                    validator(result_path)
                except Exception as exc:
                    delivery_repair_reason = f"{type(exc).__name__}: {exc}"
        remaining = int(spec.timeout_sec - (time.monotonic() - started))
        initial_calls = _tool_calls(logs, session_id)
        remaining_calls = max(0, spec.max_tool_calls - initial_calls)
        if spec.allow_retries and delivery_repair_reason and remaining >= 10 and remaining_calls > 0:
            # Delivery repair is part of the same reward-agent session budget.
            # Give it only the calls left by the first turn so the combined
            # attempt cannot silently consume a second full tool budget.
            _write_config(config_path, workspace, replace(spec, max_tool_calls=remaining_calls), wsclaw=wsclaw)
            repair_prompt = (
                "Delivery repair required. Your previous turn did not satisfy the strict file "
                f"contract: {delivery_repair_reason}. Use the workspace file tool now to create "
                f"or correct `{result_relpath}`. First read `AGENTS.md` and, if present, "
                "`RUBRIC_SCHEMA.json`; preserve every exact required key and allowed value. "
                "If the contract error mentions grounding or inspection, read `WORKSPACE_MANIFEST.json`, "
                "follow the dataset-specific WSClaw or ResearchClaw evidence rules in `AGENTS.md`, "
                "and correct `expected` and `target_paths` accordingly. "
                "After writing the file, run `python -m json.tool "
                f"{result_relpath} >/dev/null`; if it fails, fix the JSON and rerun it until it returns zero. "
                "Do not re-evaluate the task and do not only paste JSON in the reply. Finish "
                "only after the file-tool call succeeds and you have read the file back once. "
                f"You have at most {remaining_calls} tool calls left in this agent budget.\n\n"
                "Previous response excerpt (use it only to recover the intended file content):\n"
                + (proc.stdout or "")[-12000:]
            )
            repair_session_id = f"{session_id}-delivery-repair"
            repair = runtime._run_nanobot_turn(
                workspace_dir=workspace,
                config_path=config_path,
                session_id=repair_session_id,
                turn_type="side",
                message=repair_prompt,
                log_dir=logs,
                timeout=remaining,
                env_overrides={
                    "NANOBOT_TASK_ID": session_id,
                    "NANOBOT_WEB_ENABLE": "0",
                    "NANOBOT_RESTRICT_TO_WORKSPACE": "0" if wsclaw else "1",
                    "NANOBOT_ENABLE_THINKING": "1" if spec.enable_thinking else "0",
                    "NANOBOT_SHOW_REASONING": "0" if wsclaw else "1",
                    "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
                    "NANOBOT_SELF_REWARD_EXIT_ON_TOOL_BUDGET": "1",
                },
            )
            processes.append(repair)
            session_ids.append(repair_session_id)
    duration = time.monotonic() - started
    (run_dir / "stdout.txt").write_text(
        "\n\n===== NANOBOT TURN =====\n\n".join(item.stdout or "" for item in processes),
        encoding="utf-8",
    )
    (run_dir / "stderr.txt").write_text(
        "\n\n===== NANOBOT TURN =====\n\n".join(item.stderr or "" for item in processes),
        encoding="utf-8",
    )
    calls = sum(_tool_calls(logs, item) for item in session_ids)
    _write_cleaned_transcript(logs, session_ids, run_dir / "cleaned_transcript.json")
    status = {
        "session_id": session_id,
        "session_ids": session_ids,
        "model": spec.model,
        "base_url": spec.base_url,
        "returncode": processes[-1].returncode,
        "turn_returncodes": [item.returncode for item in processes],
        "delivery_repair_attempted": len(processes) > 1,
        "delivery_repair_reason": delivery_repair_reason,
        "duration_sec": duration,
        "tool_calls": calls,
        "tool_budget": spec.max_tool_calls,
        "result_path": str(result_path),
        "strict_file_delivery": True,
        "final_message_recovery": False,
    }
    try:
        if processes[-1].returncode != 0:
            raise RuntimeError(f"Nanobot exited with {processes[-1].returncode}")
        if calls > spec.max_tool_calls:
            raise RuntimeError(f"tool budget exceeded: {calls}>{spec.max_tool_calls}")
        if not result_path.is_file():
            raise FileNotFoundError(f"agent did not create {result_relpath}")
        parsed = validator(result_path)
        after = _visible_files(workspace, result_relpath)
        if before != after:
            changed = sorted(set(before) ^ set(after) | {k for k in before.keys() & after.keys() if before[k] != after[k]})
            raise RuntimeError(f"agent modified files outside result contract: {changed[:20]}")
        status.update({"status": "accepted", "error": None})
        atomic_json(run_dir / "accepted.json", parsed)
        return {"status": status, "value": parsed}
    except Exception as exc:
        status.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(run_dir / "failed.json", status)
        raise
    finally:
        atomic_json(run_dir / "status.json", status)


def run_rubric_agent(
    *, project_root: Path, workspace: Path, run_dir: Path, session_id: str, spec: AgentSpec, prompt_path: Path
) -> dict[str, Any]:
    def validate(path: Path) -> dict[str, Any]:
        # Grounding instructions remain in the model prompt and are audited
        # as diagnostics.  Training masks only malformed rubric output; it
        # must not reject an otherwise valid sample based on heuristic tool-log
        # or semantic-contract checks.
        if read_json(workspace / "WORKSPACE_MANIFEST.json")["dataset"] == "wsclaw":
            original = read_json(path)
            rubric, changes = normalize_equal_weight_rubric(original)
            if changes:
                # Keep the agent's exact output and the deterministic repair
                # outside its workspace for auditing; no extra model call.
                atomic_json(run_dir / "rubric-before-normalization.json", original)
                atomic_json(run_dir / "rubric-normalization.json", {
                    "rule": "missing_constant_weight_v1", "fields": changes,
                    "default_weight": 1.0,
                })
                atomic_json(path, rubric)
            return rubric
        return load_rubric(path)

    return run_strict_agent(
        project_root=project_root,
        workspace=workspace,
        prompt=prompt_path.read_text(encoding="utf-8"),
        result_relpath="rubric.json",
        run_dir=run_dir,
        session_id=session_id,
        spec=spec,
        validator=validate,
        wsclaw=read_json(workspace / "WORKSPACE_MANIFEST.json")["dataset"] == "wsclaw",
    )


def run_score_agent(
    *,
    project_root: Path,
    dataset: str,
    workspace: Path,
    prompt_path: Path,
    rubric_path: Path,
    result_relpath: str,
    run_dir: Path,
    session_id: str,
    slot_id: str,
    spec: AgentSpec,
) -> dict[str, Any]:
    if dataset == "wsclaw":
        from .wsclaw_scoring import run_wsclaw_self_score
        return run_wsclaw_self_score(
            project_root=project_root, workspace=workspace, prompt_path=prompt_path,
            rubric_path=rubric_path, result_relpath=result_relpath, run_dir=run_dir,
            session_id=session_id, slot_id=slot_id, spec=spec,
        )

    rubric = load_rubric(rubric_path)

    def validate(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("native score result must be an object")
        return canonicalize_native_score(dataset, value, rubric, slot_id)

    return run_strict_agent(
        project_root=project_root,
        workspace=workspace,
        prompt=prompt_path.read_text(encoding="utf-8"),
        result_relpath=result_relpath,
        run_dir=run_dir,
        session_id=session_id,
        spec=spec,
        validator=validate,
        wsclaw=dataset == "wsclaw",
    )
