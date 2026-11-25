#!/usr/bin/env python3
"""Evaluate WSClaw task directories with nanobot and a Claude Code judge."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from clawgym_nanobot.landlock_exec import (
    configured_unprivileged_identity,
    sandbox_command,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "wsclaw_final_20260713"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "logs" / "wsclaw_eval"
DEFAULT_NANOBOT_PYTHON = Path(os.environ.get("NANOBOT_PYTHON", sys.executable))
DEFAULT_NANOBOT_RUNNER = (
    PROJECT_ROOT / "vendor" / "clawgym_nanobot" / "nanobot" / "nanobot" / "rl_runner.py"
)
DEFAULT_CLAUDE_BIN = Path(
    os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
)
DEFAULT_WSBENCH_PYTHON = Path(os.environ.get("WSBENCH_PYTHON", sys.executable))
DEFAULT_APPTAINER_BIN = Path(
    os.environ.get("APPTAINER_BIN") or shutil.which("apptainer") or "apptainer"
)
CONTAINER_WORKSPACE = Path("/workspace")
NANOBOT_STARTUP_MAX_RETRIES = 5
WORKSPACE_TEMPLATE_ROOT = PROJECT_ROOT / "scripts" / "clawgym_nanobot" / "workspace_templates"
ENHANCED_INSPECTOR = PROJECT_ROOT / "scripts" / "clawgym_nanobot" / "inspect_workspace_file.py"
PORTABLE_BASE_INSPECTOR = Path(
    os.environ.get(
        "WSCLAW_BASE_INSPECTOR",
        str(ENHANCED_INSPECTOR),
    )
)
JUDGE_USER_MESSAGE = "Evaluate the final artifacts using the task package below."
JUDGE_CONTINUE_MESSAGE = (
    "The evaluation is incomplete because judge_result.json is still missing or invalid. "
    "Finish the evaluation now. Do not return a plan or progress update. Write a complete "
    "judge_result.json with every rubric score and one valid user_feedback sentence, then return "
    "the same JSON object."
)
WSCLAW_USER_FEEDBACK_GUIDANCE = """6. Also act as the original user after scoring, not as a rubric grader. In `user_feedback`, write one natural, concise paragraph of user feedback.

Rules:
- Preserve the packet's factual meaning. Do not add, infer, or repair facts that are absent from it. Candidate-specific review notes take precedence over generic requirement/score-anchor wording when they differ.
- Preserve distinctions such as wrong versus unverified, absent versus present in the wrong file, and incomplete versus internally inconsistent. Do not collapse a file-placement problem into a claim that the work is missing.
- Mention concrete strengths that should be preserved when the packet supports them, and the highest-priority corrections when it supports problems. Do not force praise or criticism without evidence.
- If a broad strength conflicts with a specific problem, use the narrower defensible wording. For example, say that all requested rows are present rather than that all values are correct when named values are wrong.
- Address the candidate directly and discuss its delivered files or answer. Do not evaluate the source documents, the task, the judge, the rubric, or another answer.
- Do not mention criterion IDs, numeric reward/score, hidden references, judging, rubrics, or this evidence packet.
- Avoid a mechanical checklist. Write one sentence or one short paragraph in the same language as the original request when practical.
"""
JUDGE_MAX_ATTEMPTS = 3
SOLVER_CHECKPOINT_SCHEMA = "wsclaw_solver_stage_v1"
WORKSPACE_PERSONA_DOMAINS = {
    "chanpin": "Product management and product operations",
    "kaifa": "Software engineering and technical delivery",
    "research": "Research and analytical work",
    "yunying": "Business and content operations",
    "houqin": "Administration, facilities, and operational support",
}


class CaseTimeoutError(TimeoutError):
    """Raised when the training-compatible end-to-end case budget is exhausted."""


def bounded_stage_timeout(
    started: float,
    args: argparse.Namespace,
    configured_timeout: int,
) -> int:
    """Clamp a stage timeout to the remaining end-to-end case budget."""
    case_timeout = getattr(args, "case_timeout", None)
    if case_timeout is None:
        return configured_timeout
    remaining = float(case_timeout) - (time.perf_counter() - started)
    if remaining <= 0:
        raise CaseTimeoutError("end-to-end case timeout exhausted")
    return min(configured_timeout, max(1, math.ceil(remaining)))


def trajectory_metrics(agent_end: dict[str, Any] | None) -> dict[str, int | None]:
    """Return compact, model-comparable usage metrics from nanobot's agent-end event.

    Older retained trajectories predate these fields in ``result.json``.  Keeping
    this extractor independent of the live runner lets reporting code backfill
    them from the authoritative ``agent_end.json`` without rerunning a task.
    """
    record = agent_end if isinstance(agent_end, dict) else {}
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}

    def integer(name: str, camel_name: str) -> int | None:
        value = usage.get(name, usage.get(camel_name))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)

    llm_steps = record.get("llm_steps")
    tool_steps = record.get("tool_steps")
    tool_events = record.get("tool_events")
    return {
        "tool_calls": len(tool_steps) if isinstance(tool_steps, list) else 0,
        "tool_events": len(tool_events) if isinstance(tool_events, list) else 0,
        "llm_calls": len(llm_steps) if isinstance(llm_steps, list) else 0,
        "prompt_tokens": integer("prompt_tokens", "promptTokens"),
        "response_tokens": integer("completion_tokens", "completionTokens"),
        "total_tokens": integer("total_tokens", "totalTokens"),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relative_path(raw: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe task path: {raw!r}")
    return Path(*pure.parts)


def reset_dir(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_parent = path.parent.resolve()
    if resolved_root not in (resolved_parent, *resolved_parent.parents):
        raise RuntimeError(f"refusing to reset outside {resolved_root}: {path}")
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def link_children(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir(), key=lambda item: item.name):
        target = destination / child.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(child.resolve(), target_is_directory=child.is_dir())


def install_agent_tools(task_dir: Path, workspace: Path) -> None:
    """Expose ordinary workspace tools without leaking evaluator contracts."""
    source = task_dir / "input_files" / "tools" / "inspect_workspace_file.py"
    if not source.is_file():
        # Fall back to the bundled canonical workspace inspector.
        source = PORTABLE_BASE_INSPECTOR
    if not source.is_file():
        raise FileNotFoundError(
            "WSClaw inspector is missing from both the task and portable path: "
            f"{task_dir / 'input_files' / 'tools' / 'inspect_workspace_file.py'}; "
            f"{PORTABLE_BASE_INSPECTOR}"
        )
    if not ENHANCED_INSPECTOR.is_file():
        raise FileNotFoundError(f"enhanced WSClaw inspector is missing: {ENHANCED_INSPECTOR}")
    tools = workspace / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    for destination, target in (
        (tools / "_inspect_workspace_file_base.py", source),
        (tools / "inspect_workspace_file.py", ENHANCED_INSPECTOR),
    ):
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(target.resolve())


def install_workspace_scaffold(workspace: Path) -> None:
    files = (
        Path("AGENTS.md"),
        Path("SOUL.md"),
        Path("USER.md"),
        Path("TOOLS.md"),
        Path("HEARTBEAT.md"),
        Path(".gitignore"),
        Path("memory/MEMORY.md"),
    )
    for relative in files:
        source = WORKSPACE_TEMPLATE_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"WSClaw workspace template is missing: {source}")
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (workspace / "memory" / "history.jsonl").touch()
    (workspace / "sessions").mkdir()
    (workspace / "skills").mkdir()


_EVAL_PERSONAS_CACHE: dict[str, str] | None = None


def _eval_persona_role(task_id: str | None) -> str:
    global _EVAL_PERSONAS_CACHE
    if not task_id:
        return ""
    if _EVAL_PERSONAS_CACHE is None:
        loaded: dict[str, str] = {}
        repo_root = Path(__file__).resolve().parents[4]
        candidate = repo_root / "configs" / "deal-ws-eval-personas.json"
        if candidate.is_file():
            try:
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    loaded = {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}
            except Exception:
                loaded = {}
        _EVAL_PERSONAS_CACHE = loaded
    return _EVAL_PERSONAS_CACHE.get(str(task_id), "")


def _load_task_output(task_dir: Path, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    path = task_dir / "task_output.json"
    if path.is_file():
        data = load_json(path)
        if isinstance(data, dict):
            return data
    fallback: dict[str, Any] = {}
    contract_path = task_dir / "reward" / "output_contract.json"
    if contract_path.is_file():
        contract = load_json(contract_path)
        if isinstance(contract, dict):
            fallback.update(contract)
    if isinstance(entry, dict):
        fallback.setdefault("task_id", entry.get("task_id"))
        fallback.setdefault("task_instruction", entry.get("user_query"))
        meta = entry.get("metadata") or {}
        if isinstance(meta, dict):
            fallback.setdefault("workspace_persona", meta.get("persona"))
            if meta.get("scenario_persona"):
                fallback.setdefault("scenario_persona", meta.get("scenario_persona"))
    return fallback


def user_context(
    task_output: dict[str, Any],
    metadata: dict[str, Any],
    *,
    task_id: str | None = None,
) -> dict[str, str]:
    raw_role = task_output.get("scenario_persona") or metadata.get("scenario_persona")
    role = raw_role.strip() if isinstance(raw_role, str) else ""
    if not role or role.lower() == "none":
        role = _eval_persona_role(
            task_id
            or task_output.get("task_id")
            or metadata.get("source_task_id")
            or metadata.get("task_id")
        ).strip()
    if not role or role.lower() == "none":
        role = "Workspace owner"
    persona = str(metadata.get("persona") or task_output.get("workspace_persona") or "").strip()
    domain = WORKSPACE_PERSONA_DOMAINS.get(persona, "General professional work")
    return {"role": role, "domain": domain, "workspace_persona": persona}


def write_user_context(
    workspace: Path,
    task_output: dict[str, Any],
    metadata: dict[str, Any],
    *,
    task_id: str | None = None,
) -> dict[str, str]:
    context = user_context(task_output, metadata, task_id=task_id)
    text = f"""# User Context

The user is commissioning work in the following professional context.

- **Role**: {context['role']}
- **Work domain**: {context['domain']}
- **Workspace persona**: {context['workspace_persona'] or 'unspecified'}

## Preferences

- Produce a polished, audience-ready deliverable in the exact format requested.
- Ground claims, calculations, and recommendations in the provided workspace files.
- Preserve the terminology and language implied by the task and its sources.
- Distinguish clearly between source facts, analysis, and recommendations.
- Use the final response only to confirm completion; put substantive content in the artifact.
"""
    (workspace / "USER.md").write_text(text, encoding="utf-8")
    return context


def append_task_contract(
    workspace: Path,
    sources: list[tuple[Path, Path]],
    outputs: list[Path],
) -> None:
    lines = [
        "",
        "## Current Task Contract",
        "",
        "These paths are authoritative for the current task.",
        "Inspect every required source before creating the deliverable; do not invent source facts.",
        "The read-only source desktop is `input/Desktop/`; never write there.",
        "When the user says desktop, use the writable workspace-relative `Desktop/` directory.",
        "",
        "### Required Source Files",
        "",
        *(f"- `input/{relative.as_posix()}`" for relative, _source in sources),
        "",
        "### Required Output Paths",
        "",
        *(f"- `{relative.as_posix()}`" for relative in outputs),
        "",
        "### Source Inspection",
        "",
        "Before creating output, inspect every required source and use extracted facts. Choose "
        "the inspection method and character budget according to the file and evidence needed. "
        "The native tool accepts an optional `max_chars`; for focused searches you may instead "
        "run `python tools/inspect_workspace_file.py ...` through `exec` and pipe its extracted "
        "output to `grep` or `rg`. If a bounded preview is truncated and omits needed evidence, "
        "increase the budget or inspect the relevant remainder with a local format-specific tool.",
        "",
        "```text",
        *(
            f'inspect_workspace_file(path="{(Path("input") / relative).as_posix()}")'
            for relative, _source in sources
        ),
        "```",
        "",
        "Do not finish until every required output exists at the exact path above.",
        "",
        "### Tool Call Discipline",
        "",
        "Prefer keeping each tool call reasonably compact and syntactically complete. For "
        "substantial code, consider writing a script to a workspace file (in a few manageable "
        "writes when useful) and then executing that file instead of embedding the whole script "
        "in a large `python -c` command. This usually reduces shell quoting and JSON-escaping "
        "failures. Focused inspection and bounded output are also generally more reliable than "
        "printing whole files or datasets.",
        "",
    ]
    agents_path = workspace / "AGENTS.md"
    with agents_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def discover_tasks(
    data_root: Path,
    *,
    task_indices: set[int] | None,
    excluded_task_indices: set[int] | None,
    limit: int | None,
    sample: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    manifest = load_json(data_root / "manifest.json")
    if not isinstance(manifest, list):
        raise ValueError("WSClaw manifest must be a JSON list")
    tasks = [
        item
        for item in manifest
        if (task_indices is None or int(item["index"]) in task_indices)
        and (
            excluded_task_indices is None
            or int(item["index"]) not in excluded_task_indices
        )
    ]
    if sample is not None:
        tasks = random.Random(seed).sample(tasks, min(sample, len(tasks)))
    if limit is not None:
        tasks = tasks[:limit]
    for item in tasks:
        item["_task_dir"] = str(data_root / str(item["task_dir"]))
    return tasks


def discover_tasks_from_prompt_data(
    prompt_data: Path,
    *,
    task_indices: set[int] | None,
    excluded_task_indices: set[int] | None,
    limit: int | None,
    sample: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Use retained training-eval prompt data as the task and prompt source."""
    rows = [
        json.loads(line)
        for line in prompt_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise TypeError(f"prompt row {position} must contain object metadata")
        metadata = row["metadata"]
        index = int(metadata.get("task_index", position))
        if index != position:
            raise ValueError(
                f"prompt task_index must be contiguous and ordered: row={position} task_index={index}"
            )
        task_dir = Path(str(metadata.get("task_dir") or "")).resolve()
        entry_path = task_dir / "data_entry.json"
        if not entry_path.is_file():
            raise FileNotFoundError(entry_path)
        entry = load_json(entry_path)
        if not isinstance(entry, dict):
            raise TypeError(f"task entry must be an object: {entry_path}")
        task_id = str(entry.get("task_id") or "")
        if not task_id:
            raise ValueError(f"task entry has no task_id: {entry_path}")
        if str(metadata.get("task_id") or task_id) != task_id:
            raise ValueError(f"prompt/task identity mismatch at row {position}: {task_id}")

        prompt_input = row.get("input")
        if isinstance(prompt_input, str):
            prompt = prompt_input
        elif (
            isinstance(prompt_input, list)
            and len(prompt_input) == 1
            and isinstance(prompt_input[0], dict)
            and prompt_input[0].get("role") == "user"
            and isinstance(prompt_input[0].get("content"), str)
        ):
            prompt = prompt_input[0]["content"]
        else:
            raise ValueError(
                f"prompt row {position} must contain raw text or one user message"
            )
        if prompt != entry.get("user_query"):
            raise ValueError(
                f"training prompt differs from task user_query at row {position}: {task_id}"
            )
        task_metadata = entry.get("metadata")
        if not isinstance(task_metadata, dict):
            raise TypeError(f"task metadata must be an object: {entry_path}")
        selected.append(
            {
                "index": index,
                "task_id": task_id,
                "scenario_id": task_metadata.get("scenario_id"),
                "persona": task_metadata.get("persona"),
                "task_dir": str(task_dir),
                "_task_dir": str(task_dir),
                "_prompt": prompt,
                "_prompt_metadata": dict(metadata),
            }
        )

    selected = [
        item
        for item in selected
        if (task_indices is None or int(item["index"]) in task_indices)
        and (
            excluded_task_indices is None
            or int(item["index"]) not in excluded_task_indices
        )
    ]
    if sample is not None:
        selected = random.Random(seed).sample(selected, min(sample, len(selected)))
    if limit is not None:
        selected = selected[:limit]
    return selected


def output_paths(task_dir: Path) -> list[Path]:
    contract = load_json(task_dir / "reward" / "output_contract.json")
    paths = []
    for item in contract.get("output_files") or []:
        if isinstance(item, dict) and item.get("must_create", True) and item.get("path"):
            paths.append(safe_relative_path(str(item["path"])))
    if not paths:
        raise ValueError(f"task has no required output path: {task_dir}")
    return paths


def artifact_format_ok(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}:
        return zipfile.is_zipfile(path)
    if suffix == ".zip":
        return zipfile.is_zipfile(path)
    if suffix == ".pdf":
        try:
            return path.read_bytes()[:5] == b"%PDF-"
        except OSError:
            return False
    if suffix == ".png":
        try:
            return path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        except OSError:
            return False
    if suffix in {".jpg", ".jpeg"}:
        try:
            data = path.read_bytes()
            return data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
        except OSError:
            return False
    if suffix == ".json":
        try:
            load_json(path)
            return True
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
    if suffix in {".md", ".txt", ".csv", ".tsv", ".html", ".htm", ".xml", ".py", ".yaml", ".yml"}:
        try:
            path.read_text(encoding="utf-8")
            return True
        except (OSError, UnicodeError):
            return False
    return path.is_file()


def artifact_record(workspace: Path, relative: Path) -> dict[str, Any]:
    path = workspace / relative
    exists = path.is_file()
    nonempty = bool(exists and path.stat().st_size > 0)
    return {
        "path": relative.as_posix(),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "format_ok": bool(nonempty and artifact_format_ok(path)),
    }


def workspace_source_root(entry: dict[str, Any]) -> Path:
    raw = (entry.get("metadata") or {}).get("workspace_root")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"task {entry.get('task_id')} has no workspace_root metadata")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"workspace source root does not exist: {root}")
    return root


def required_sources(entry: dict[str, Any]) -> list[tuple[Path, Path]]:
    source_root = workspace_source_root(entry)
    raw_paths = (entry.get("metadata") or {}).get("required_source_files")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"task {entry.get('task_id')} has no required_source_files metadata")
    sources: list[tuple[Path, Path]] = []
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"task {entry.get('task_id')} has invalid required source: {raw!r}")
        relative = safe_relative_path(raw)
        source = (source_root / relative).resolve()
        if source_root not in (source, *source.parents):
            raise ValueError(f"required source escapes workspace root: {raw!r}")
        if not source.is_file():
            raise FileNotFoundError(f"required source file does not exist: {source}")
        sources.append((relative, source))
    return sources


def expose_required_sources(
    destination_root: Path,
    sources: list[tuple[Path, Path]],
    *,
    placeholders: bool,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    for relative, source in sources:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if placeholders:
            destination.touch()
        else:
            destination.symlink_to(source)


def expose_full_workspace(
    destination_root: Path,
    source_root: Path,
    *,
    containerized: bool,
) -> None:
    """Expose the complete persona workspace, read-only when containerized."""
    del containerized
    link_children(source_root, destination_root)


def install_runtime_python(workspace: Path, python: Path) -> Path:
    runtime_bin = workspace / ".runtime" / "bin"
    runtime_bin.mkdir(parents=True, exist_ok=True)
    target = python.resolve()
    for name in ("python", "python3", "python3.11"):
        link = runtime_bin / name
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    for name in ("antiword", "catdoc", "file", "unzip"):
        executable = shutil.which(name)
        if executable:
            link = runtime_bin / name
            link.unlink(missing_ok=True)
            link.symlink_to(Path(executable).resolve())
    return runtime_bin


def prepare_workspace(
    task_dir: Path,
    case_dir: Path,
    *,
    containerized: bool = False,
) -> tuple[Path, dict[str, Any], list[Path], Path]:
    workspace = case_dir / "workspace"
    reset_dir(workspace, case_dir)
    (workspace / "Desktop").mkdir()
    install_workspace_scaffold(workspace)
    install_agent_tools(task_dir, workspace)
    entry = load_json(task_dir / "data_entry.json")
    source_root = workspace_source_root(entry)
    sources = required_sources(entry)
    expose_full_workspace(
        workspace / "input",
        source_root,
        containerized=containerized,
    )
    outputs = output_paths(task_dir)
    append_task_contract(workspace, sources, outputs)
    metadata = entry.get("metadata") or {}
    task_output = _load_task_output(task_dir, entry)
    profile = write_user_context(workspace, task_output, metadata, task_id=entry.get("task_id"))
    write_json(
        workspace / "task_context.json",
        {
            "task_id": entry["task_id"],
            "scenario_id": task_output.get("scenario_id", metadata.get("scenario_id")),
            "persona": metadata.get("persona"),
            "user_role": profile["role"],
            "user_domain": profile["domain"],
            "generated_task_path": metadata.get("source_task_path"),
            "parent_task_id": task_output.get("parent_task_id"),
            "subtask_id": task_output.get("subtask_id", metadata.get("subtask_id")),
            "linked_files": [
                {
                    "relative_path": child.name,
                    "input_path": f"input/{child.name}",
                    "source_path": str(child.resolve()),
                    "reason": "full-workspace mount",
                }
                for child in sorted(source_root.iterdir(), key=lambda path: path.name)
            ],
            "instruction": task_output.get("task_instruction", entry["user_query"]),
            "deliverable_format": task_output.get("deliverable_format"),
            "required_source_files": [
                f"input/{relative.as_posix()}" for relative, _ in sources
            ],
            "required_output_paths": [str(path) for path in outputs],
            "workspace_mode": "full",
        },
    )
    return workspace, entry, outputs, source_root


def run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int,
    restrict_writes_to: Path | None = None,
) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process_env = dict(env)
    if restrict_writes_to is not None:
        write_root = restrict_writes_to.resolve(strict=True)
        temp_root = write_root / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        home_root = write_root / ".home"
        home_root.mkdir(parents=True, exist_ok=True)
        process_env["HOME"] = str(home_root)
        for name in ("TMPDIR", "TMP", "TEMP"):
            process_env[name] = str(temp_root)
        process_env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = sandbox_command(
            command,
            write_root=write_root,
            identity=configured_unprivileged_identity(process_env),
        )
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=process_env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return process.returncode, timed_out, time.perf_counter() - started


def nanobot_config(
    *,
    workspace: Path,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int,
    context_window: int,
    max_tool_iterations: int,
    restrict_to_workspace: bool,
    context_block_limit: int | None = None,
    temperature: float = 0.2,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float | None = None,
) -> dict[str, Any]:
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    agent_defaults: dict[str, Any] = {
        "workspace": str(workspace),
        "provider": "custom",
        "model": model,
        "maxTokens": max_tokens,
        "contextWindowTokens": context_window,
        "temperature": temperature,
        "maxToolIterations": max_tool_iterations,
        "timezone": "Asia/Shanghai",
    }
    if context_block_limit is not None:
        agent_defaults["contextBlockLimit"] = context_block_limit
    provider_extra_body: dict[str, Any] = {
        "stream": False,
        "top_p": top_p,
        "top_k": top_k,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if min_p is not None:
        provider_extra_body["min_p"] = min_p
    return {
        "providers": {
            "custom": {
                "apiBase": base,
                "apiKey": api_key,
                "extraBody": provider_extra_body,
            }
        },
        "agents": {
            "defaults": agent_defaults,
        },
        "channels": {"sendProgress": False, "sendToolHints": False, "showReasoning": False},
        "tools": {
            "restrictToWorkspace": restrict_to_workspace,
            "web": {"enable": True, "noWeb": True},
            "exec": {
                "enable": True,
                "timeout": 120,
                "noWeb": True,
                "noWebCliTimeout": 5,
                # ExecTool uses a minimal subprocess environment. Preserve the
                # curated document-library path for generated task scripts.
                "allowedEnvKeys": ["PYTHONPATH"],
            },
        },
    }


def nanobot_command(
    *,
    workspace: Path,
    source_root: Path,
    entry: dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    user_query: str | None = None,
) -> tuple[list[str], Path, Path]:
    prompt = str(entry["user_query"] if user_query is None else user_query)
    image = getattr(args, "nanobot_apptainer_image", None)
    if not image:
        return (
            [
                str(args.nanobot_python),
                str(args.nanobot_runner),
                "agent",
                "-m",
                prompt,
                "--no-markdown",
                "-s",
                str(entry["task_id"]),
                "-w",
                str(workspace),
                "-c",
                str(config_path),
            ],
            workspace,
            workspace / ".nanobot_logs",
        )

    runtime_workspace = Path(getattr(args, "nanobot_container_workspace", CONTAINER_WORKSPACE))
    runtime_log_root = runtime_workspace / ".nanobot_logs"
    runtime_bin = runtime_workspace / ".runtime" / "bin"
    nanobot_root = args.nanobot_runner.parent.parent
    runtime_path = (
        f"{runtime_bin}:{args.wsbench_python.parent}:{args.nanobot_python.parent}:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    llm_timeout = os.environ.get("NANOBOT_RL_LLM_TIMEOUT_S", "0")
    provider_timeout = os.environ.get(
        "NANOBOT_RL_OPENAI_COMPAT_TIMEOUT_S",
        "0",
    )
    command = [
        str(args.apptainer_bin),
        "exec",
        "--cleanenv",
        "--bind",
        "/mnt:/mnt:ro",
        "--bind",
        f"{workspace}:{runtime_workspace}",
        "--bind",
        f"{source_root}:{runtime_workspace / 'input'}:ro",
    ]
    command.extend(
        [
            "--pwd",
            str(runtime_workspace),
            str(image),
            "env",
            f"HOME={runtime_workspace / '.home'}",
            f"PATH={runtime_path}",
            f"PYTHONPATH={nanobot_root}",
            "LANG=C.UTF-8",
            "NANOBOT_NO_SPINNER=1",
            "NANOBOT_RL_DISABLED_TOOLS=spawn,message",
            f"NANOBOT_LLM_TIMEOUT_S={llm_timeout}",
            f"NANOBOT_OPENAI_COMPAT_TIMEOUT_S={provider_timeout}",
            f"NANOBOT_RL_EXEC_PATH_PREPEND={runtime_bin}",
            f"NANOBOT_LOG_DIR={runtime_log_root}",
            f"NANOBOT_WORKSPACE_ACTUAL={runtime_workspace}",
            f"NANOBOT_WORKSPACE_DISPLAY={runtime_workspace}",
            "NO_PROXY=*",
            "no_proxy=*",
            "HTTP_PROXY=",
            "HTTPS_PROXY=",
            "ALL_PROXY=",
            "http_proxy=",
            "https_proxy=",
            "all_proxy=",
            str(args.nanobot_python),
            str(args.nanobot_runner),
            "agent",
            "-m",
            prompt,
            "--no-markdown",
            "-s",
            str(entry["task_id"]),
            "-w",
            str(runtime_workspace),
            "-c",
            str(config_path),
        ]
    )
    return command, runtime_workspace, workspace / ".nanobot_logs"


def run_nanobot(
    *,
    workspace: Path,
    entry: dict[str, Any],
    case_dir: Path,
    args: argparse.Namespace,
    user_query: str | None = None,
) -> dict[str, Any]:
    log_root = workspace / ".nanobot_logs"
    reset_dir(log_root, case_dir)
    cleaned_transcript = case_dir / "cleaned_transcripts.json"
    agent_end = case_dir / "agent_end.json"
    for retained_log in (cleaned_transcript, agent_end):
        if retained_log.exists() or retained_log.is_symlink():
            retained_log.unlink()
    config_path = case_dir / "nanobot_config.json"
    source_root = workspace_source_root(entry)
    required_sources(entry)
    runtime_bin = install_runtime_python(workspace, args.wsbench_python)
    runtime_workspace = (
        Path(getattr(args, "nanobot_container_workspace", CONTAINER_WORKSPACE))
        if getattr(args, "nanobot_apptainer_image", None)
        else workspace
    )
    write_json(
        config_path,
        nanobot_config(
            workspace=runtime_workspace,
            api_base=args.nanobot_api_base,
            api_key=args.nanobot_api_key,
            model=args.nanobot_model,
            max_tokens=args.nanobot_max_tokens,
            context_window=args.context_window,
            max_tool_iterations=args.max_tool_iterations,
            restrict_to_workspace=bool(getattr(args, "nanobot_apptainer_image", None)),
            context_block_limit=getattr(args, "context_block_limit", None),
            temperature=getattr(args, "nanobot_temperature", 0.2),
            top_p=getattr(args, "nanobot_top_p", 1.0),
            top_k=getattr(args, "nanobot_top_k", -1),
            min_p=getattr(args, "nanobot_min_p", None),
        ),
    )
    command, runtime_workspace, source_log_root = nanobot_command(
        workspace=workspace,
        source_root=source_root,
        entry=entry,
        config_path=config_path,
        args=args,
        user_query=user_query,
    )
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{runtime_bin}:{args.nanobot_python.parent}:{env.get('PATH', '')}",
            "NANOBOT_NO_SPINNER": "1",
            "NANOBOT_RL_DISABLED_TOOLS": "spawn,message",
            "NANOBOT_LLM_TIMEOUT_S": os.environ.get(
                "NANOBOT_RL_LLM_TIMEOUT_S",
                "0",
            ),
            "NANOBOT_OPENAI_COMPAT_TIMEOUT_S": os.environ.get(
                "NANOBOT_RL_OPENAI_COMPAT_TIMEOUT_S",
                "0",
            ),
            "NANOBOT_RL_EXEC_PATH_PREPEND": str(runtime_bin),
            "NANOBOT_LOG_DIR": str(log_root),
            "NANOBOT_WORKSPACE_ACTUAL": str(workspace),
            # Host-mode evals must display the real per-case workspace. Using
            # the container-only /workspace alias here lets generated scripts
            # escape into a shared host directory and makes artifact checks
            # report false negatives.
            "NANOBOT_WORKSPACE_DISPLAY": str(runtime_workspace),
            "NO_PROXY": "*",
            "no_proxy": "*",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
        }
    )
    nanobot_log = case_dir / "nanobot.log"
    startup_retries = 0
    while True:
        exit_code, timed_out, duration = run_process(
            command,
            cwd=workspace,
            env=env,
            log_path=nanobot_log,
            timeout_sec=args.nanobot_timeout,
            restrict_writes_to=(
                None if getattr(args, "nanobot_apptainer_image", None) else workspace
            ),
        )
        log_text = nanobot_log.read_text(encoding="utf-8", errors="replace")
        transient_start_failure = transient_nanobot_start_failure(
            exit_code=exit_code,
            timed_out=timed_out,
            duration=duration,
            log_text=log_text,
        )
        if (
            not transient_start_failure
            or startup_retries >= NANOBOT_STARTUP_MAX_RETRIES
        ):
            break
        startup_retries += 1
        time.sleep(startup_retries + random.random())
    source_session_dir = source_log_root / str(entry["task_id"])
    source_transcript = source_session_dir / "cleaned_transcripts.json"
    if source_transcript.is_file():
        shutil.copy2(source_transcript, cleaned_transcript)
    source_agent_ends = sorted(
        source_session_dir.glob("*agent-end.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if source_agent_ends:
        shutil.copy2(source_agent_ends[-1], agent_end)
    agent_end_record = load_json(agent_end) if agent_end.is_file() else {}
    shutil.rmtree(log_root, ignore_errors=True)
    if exit_code == 0 and not timed_out:
        (case_dir / "nanobot.log").unlink(missing_ok=True)
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_sec": duration,
        "startup_retries": startup_retries,
        "cleaned_transcript_path": str(cleaned_transcript) if cleaned_transcript.is_file() else None,
        "agent_end_path": str(agent_end) if agent_end.is_file() else None,
        "agent_success": agent_end_record.get("success"),
        "agent_stop_reason": agent_end_record.get("stop_reason"),
        "agent_error": agent_end_record.get("error"),
        "trajectory_metrics": trajectory_metrics(agent_end_record),
    }


def transient_nanobot_start_failure(
    *,
    exit_code: int,
    timed_out: bool,
    duration: float,
    log_text: str,
) -> bool:
    if timed_out or duration >= 10:
        return False
    diagnostic = log_text.lower()
    io_failure = any(
        marker in diagnostic
        for marker in (
            "input/output error",
            "cannot read file data",
            "transport endpoint is not connected",
            "stale file handle",
            "container creation failed",
            "bad superblock for squashfs image",
        )
    )
    return (exit_code in {127, 255} and io_failure) or exit_code == -signal.SIGBUS


def judge_inspection_budget(source_count: int, output_count: int) -> int:
    inspected_file_count = max(1, source_count + output_count)
    return max(2000, min(6000, 30000 // inspected_file_count))


def compact_rubric(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    grounded = rubric.get("grounded_rubric") or []
    rules = rubric.get("rules") or []
    items: list[dict[str, Any]] = []
    for index, text in enumerate(grounded, start=1):
        item: dict[str, Any] = {"id": str(index), "text": str(text)}
        if index <= len(rules) and isinstance(rules[index - 1], dict):
            rule = rules[index - 1]
            if rule.get("type"):
                item["type"] = str(rule["type"])
            if rule.get("weight") not in (None, 1, 1.0):
                item["weight"] = rule["weight"]
        items.append(item)
    return items


def source_reason_map(task_output: dict[str, Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for item in task_output.get("required_source_files") or []:
        if not isinstance(item, dict) or not item.get("path") or not item.get("reason"):
            continue
        reasons[str(item["path"])] = str(item["reason"])
    return reasons


def prepare_judge_workspace(
    *,
    task_dir: Path,
    workspace: Path,
    outputs: list[Path],
    case_dir: Path,
    entry: dict[str, Any],
    original_rubric: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    judge = case_dir / "judge_workspace"
    reset_dir(judge, case_dir)
    (judge / "review").mkdir()
    (judge / "workspace").mkdir()
    # The judge sees the evaluated agent's complete final workspace. Its input/
    # tree already points at the original persona workspace.
    link_children(workspace, judge / "workspace")
    link_children(task_dir / "input_files" / "tools", judge / "tools")
    install_agent_tools(task_dir, judge)
    artifacts = []
    for relative in outputs:
        source = workspace / relative
        destination = judge / "workspace" / relative
        exists = source.is_file()
        artifacts.append(artifact_record(workspace, relative))
        if exists and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source.resolve())
    task_output = _load_task_output(task_dir, entry)
    rubric = load_json(task_dir / "reward" / "rubric.json")
    reasons = source_reason_map(task_output)
    sources = []
    for index, (relative, _source) in enumerate(required_sources(entry), start=1):
        record = {
            "id": f"source:{index}",
            "path": f"workspace/input/{relative.as_posix()}",
        }
        if reasons.get(relative.as_posix()):
            record["reason"] = reasons[relative.as_posix()]
        sources.append(record)
    output_descriptions = {
        str(item.get("path")): item
        for item in task_output.get("output_files") or []
        if isinstance(item, dict) and item.get("path")
    }
    output_records = []
    for index, artifact in enumerate(artifacts, start=1):
        relative_path = str(artifact["path"])
        expected = output_descriptions.get(relative_path, {})
        record = {
            "id": f"output:{index}",
            "path": f"workspace/{relative_path}",
            "exists": artifact["exists"],
            "size_bytes": artifact["size_bytes"],
            "format_ok": artifact["format_ok"],
        }
        if expected.get("format"):
            record["format"] = str(expected["format"])
        if expected.get("description"):
            record["description"] = str(expected["description"])
        output_records.append(record)
    brief = {
        "task_id": str(task_output.get("task_id") or entry.get("task_id") or ""),
        "task": str(task_output.get("task_instruction") or entry.get("user_query") or ""),
        "deliverable_format": str(task_output.get("deliverable_format") or ""),
        "required_operations": [
            str(operation) for operation in task_output.get("required_operations") or []
        ],
        "rubric": compact_rubric(rubric),
        "sources": sources,
        "outputs": output_records,
        "inspection_default_max_chars": judge_inspection_budget(
            len(sources), len(output_records)
        ),
    }
    (judge / "task.md").write_text(
        str(entry.get("user_query") or task_output.get("task_instruction") or ""),
        encoding="utf-8",
    )
    write_json(judge / "rubric.json", original_rubric if original_rubric is not None else rubric)
    write_json(judge / "review" / "brief.json", brief)
    instructions = judge_instructions()
    (judge / "CLAUDE.md").write_text(instructions, encoding="utf-8")
    (judge / "AGENTS.md").write_text(instructions, encoding="utf-8")
    return judge, artifacts


def judge_user_message(judge_workspace: Path) -> str:
    brief = load_json(judge_workspace / "review" / "brief.json")
    sections = [JUDGE_USER_MESSAGE, "", f"Task: {brief['task']}"]
    if (judge_workspace / "task.md").is_file() and (judge_workspace / "rubric.json").is_file():
        sections.extend(("", "First read task.md and rubric.json for the complete original task "
                         "and rubric. review/brief.json maps rubric items in order to the "
                         "numeric score keys and provides source/output aliases."))
    if brief.get("deliverable_format"):
        sections.extend(("", f"Deliverable: {brief['deliverable_format']}"))
    if brief.get("required_operations"):
        sections.extend(("", "Required operations:"))
        sections.extend(f"- {operation}" for operation in brief["required_operations"])
    sections.extend(("", "Rubric (score every item from 0.0 to 1.0):"))
    sections.extend(
        f"{item['id']}. {item['text']}" for item in brief.get("rubric") or []
    )
    sections.extend(("", "Available original sources (inspect only when needed):"))
    for source in brief.get("sources") or []:
        line = f"- {source['id']} -> {source['path']}"
        if source.get("reason"):
            line += f" ({source['reason']})"
        sections.append(line)
    sections.extend(("", "Expected outputs (inspect each existing artifact):"))
    for output in brief.get("outputs") or []:
        status = (
            f"exists={str(bool(output.get('exists'))).lower()}, "
            f"size={int(output.get('size_bytes') or 0)}, "
            f"format_ok={str(bool(output.get('format_ok'))).lower()}"
        )
        sections.append(f"- {output['id']} -> {output['path']} ({status})")
    sections.extend(
        (
            "",
            "Use output:N aliases to inspect final artifacts first. Use source:N aliases only "
            "when the rubric or an unclear artifact makes original-source verification necessary. "
            "Follow CLAUDE.md, score the final artifacts, "
            "write judge_result.json, and return the same JSON object.",
        )
    )
    return "\n".join(sections)


def judge_instructions() -> str:
    return """# WSClaw Judge Harness

You are the judge for one completed WSClaw workspace task. The initial user message contains the
task, grounded rubric, required source aliases, output aliases, and harness format checks. Evaluate
only the final artifacts and their grounding in the original sources. Do not use the internet,
inspect the evaluated agent's trajectory, or introduce hidden criteria.

## Workspace Layout

- `task.md`: original user task.
- `rubric.json`: complete rubric for this judgment.
- `review/brief.json`: scoring-item order, source/output aliases, and artifact checks.
- `workspace/`: the evaluated agent's complete final workspace tree.
- `workspace/input/`: the original persona workspace and source materials.
- `tools/inspect_workspace_file.py`: local inspector for Office, PDF, CSV, images, text, archives,
  and other supported workspace files.

## Required Procedure

1. Start from every existing final output and inspect it carefully. In many tasks the final
   artifacts contain enough evidence to score the rubric, so do not mechanically inspect the
   broader workspace or every original source. Open original sources or other workspace files only
   when a rubric explicitly requires source grounding, a factual claim needs verification, or the
   artifact is ambiguous. Use as few focused inspections as are necessary for a defensible score.
2. Use ordinary Claude Code file tools for text and directory exploration. For structured or
   binary files, run the local inspector with the short alias from the initial message:

   `python tools/inspect_workspace_file.py source:1 [--max-chars N] [--grep PATTERN]`

   When `--max-chars` is omitted, the inspector uses a context-safe per-task default. `--grep`
   safely filters the extracted preview and may be repeated with a larger character budget when
   `truncated` is true. Check the reported row, table, page, or slide counts before concluding that
   evidence is absent. Do not use Grep or a plain text read on a binary file as evidence that its
   content is absent.

   Before reading substantial content, establish the file's scale from `size_bytes`, the compact
   inspector summary, and its reported character, row, table, page, slide, or sheet counts. Do not
   read an entire large file merely for convenience. A whole-file read is appropriate only when
   the file is demonstrably small or exhaustive inspection is necessary for a specific rubric.
   Otherwise use bounded `--max-chars`, exact aliases, selected cells/ranges, and `--grep` or an
   `rg`/`grep` pipeline so only rubric-relevant evidence enters the conversation.

   Keep evidence retrieval incremental and bounded. Start with the default preview, then use the
   rubric's exact sheet names, cell/range references, labels, or values to make focused `--grep`
   calls. For spreadsheets whose rubric names exact cells or ranges, a short read-only Python
   command using `openpyxl` to print only those cells is preferable to expanding the full workbook.
   If several distant regions are needed, inspect them in separate focused calls and retain a short
   evidence summary between calls. Stop retrieving more source text once the existing evidence is
   sufficient to score the rubric defensibly.

   Context safety is mandatory. Do not solve truncation by requesting a near-whole-file dump such
   as `--max-chars 100000`. Every custom Bash or Python inspection must also keep stdout concise:
   print only the rubric-relevant scalars, selected cells, aggregates, or a few bounded rows. Never
   print a whole worksheet, column, long list, large dictionary, document, or unbounded search
   result.

   If any tool response contains `<persisted-output>` or says that full output was saved under a
   tool-results directory, treat that inspection as too broad and discard the saved path. You MUST
   NOT call `Read`, `cat`, `sed`, or another tool on that persisted result, even if the preview is
   incomplete. Rewrite and rerun the original inspection so its stdout is intrinsically small—for
   example, print one aggregate at a time, exact spreadsheet cells/ranges, or a focused grep. If
   the conversation has already accumulated substantial evidence, prefer finishing the rubric
   from that evidence instead of issuing broader calls. The goal is complete rubric coverage, not
   a copy of the source or a tool-result file in model context.

3. Do not modify anything under `workspace/` or `tools/`. The only file you may create
   or replace is `judge_result.json` in this directory.
4. Score each grounded rubric item from 0.0 to 1.0, using string keys `"1"`, `"2"`, and so on.
   The top-level score must be the arithmetic mean of all rubric scores.
5. In `evidence_checked`, list only the artifact paths, source paths, or concrete facts you actually
   inspected. Do not claim to have inspected a source that you did not open.
""" + WSCLAW_USER_FEEDBACK_GUIDANCE + """
   Keep any detailed rubric explanation in `feedback`; `user_feedback` is the user's concise
   evidence-grounded response.
7. Write exactly one JSON object to `judge_result.json`, then return the same object without a
   Markdown fence.

Stopping after a plan, a progress report, or a statement such as "I will inspect" is a failed
evaluation. Continue using tools until the result file has actually been written.

## Result Schema

```json
{
  "score": 0.0,
  "rubric_scores": {"1": 0.0},
  "feedback": "brief explanation",
  "user_feedback": "one natural user-feedback sentence of roughly twenty words",
  "missing_or_wrong": ["concrete issue"],
  "evidence_checked": ["relative path or verified fact"],
  "rubric_source": "initial task package",
  "judge_agent": "claude_code_gemma4"
}
```
"""


def cleaned_judge_transcript(attempts: list[tuple[Path, str]]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}

    def append_text(message: dict[str, Any], key: str, value: str) -> None:
        if not value:
            return
        existing = message.get(key)
        message[key] = f"{existing}\n{value}" if existing else value

    for log_path, prompt in attempts:
        messages.append({"role": "user", "content": prompt})
        if not log_path.is_file():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            payload = event.get("message")
            if not isinstance(payload, dict):
                continue
            if event_type == "assistant":
                message_id = payload.get("id")
                if (
                    messages
                    and messages[-1].get("role") == "assistant"
                    and messages[-1].get("message_id") == message_id
                ):
                    cleaned = messages[-1]
                else:
                    cleaned = {"role": "assistant"}
                    if message_id:
                        cleaned["message_id"] = message_id
                    messages.append(cleaned)
                for block in payload.get("content") or []:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "thinking":
                        append_text(cleaned, "reasoning_content", str(block.get("thinking") or ""))
                    elif block_type == "text":
                        append_text(cleaned, "content", str(block.get("text") or ""))
                    elif block_type == "tool_use":
                        tool_call = {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": block.get("input") or {},
                        }
                        cleaned.setdefault("tool_calls", []).append(tool_call)
                        if block.get("id") and block.get("name"):
                            tool_names[str(block["id"])] = str(block["name"])
            elif event_type == "user":
                for block in payload.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_call_id = str(block.get("tool_use_id") or "")
                    content = block.get("content")
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    cleaned = {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id,
                        "is_error": bool(block.get("is_error", False)),
                    }
                    if tool_names.get(tool_call_id):
                        cleaned["name"] = tool_names[tool_call_id]
                    messages.append(cleaned)
    return {"messages": messages}


def complete_judge_result(value: Any, rubric_count: int) -> bool:
    if not isinstance(value, dict) or rubric_count < 1:
        return False
    scores = value.get("rubric_scores")
    if not isinstance(scores, dict):
        return False
    for index in range(1, rubric_count + 1):
        keys = (str(index), f"rubric_{index:02d}", f"rubric_{index}")
        if not any(key in scores for key in keys):
            return False
    # Feedback prose must not gate otherwise complete numerical scores.
    return True


def required_judge_source_paths(judge_workspace: Path) -> list[str]:
    brief = load_json(judge_workspace / "review" / "brief.json")
    sources = brief.get("sources")
    if not isinstance(sources, list):
        raise ValueError("brief.json has invalid sources")
    paths = [source.get("path") for source in sources if isinstance(source, dict)]
    if len(paths) != len(sources) or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("brief.json has invalid source paths")
    return paths


def judge_source_aliases(judge_workspace: Path) -> dict[str, str]:
    brief = load_json(judge_workspace / "review" / "brief.json")
    aliases: dict[str, str] = {}
    for source in brief.get("sources") or []:
        if not isinstance(source, dict) or not source.get("path") or not source.get("id"):
            raise ValueError("brief.json has an invalid source alias")
        aliases[str(source["path"])] = str(source["id"])
    return aliases


def required_judge_outputs(judge_workspace: Path) -> dict[str, str]:
    brief = load_json(judge_workspace / "review" / "brief.json")
    outputs: dict[str, str] = {}
    for output in brief.get("outputs") or []:
        if not isinstance(output, dict) or not output.get("path") or not output.get("id"):
            raise ValueError("brief.json has an invalid output alias")
        if output.get("exists"):
            outputs[str(output["path"])] = str(output["id"])
    return outputs


def judge_inspection_default_max_chars(judge_workspace: Path) -> int:
    brief = load_json(judge_workspace / "review" / "brief.json")
    raw = brief.get("inspection_default_max_chars", 6000)
    try:
        return max(1000, min(int(raw), 100000))
    except (TypeError, ValueError):
        return 6000


def tool_input_texts(attempts: list[tuple[Path, str]]) -> list[str]:
    texts: list[str] = []
    for log_path, _prompt in attempts:
        if not log_path.is_file():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "assistant":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_input = block.get("input")
                if isinstance(tool_input, (dict, list)):
                    texts.append(json.dumps(tool_input, ensure_ascii=False))
                elif isinstance(tool_input, str):
                    texts.append(tool_input)
    return texts


def path_mentioned(path: str, texts: list[str]) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    variants = {normalized}
    if normalized.startswith("workspace/"):
        variants.add(normalized[len("workspace/") :])
    for text in texts:
        candidate = text.replace("\\", "/")
        if any(variant in candidate for variant in variants):
            return True
    return False


def alias_mentioned(alias: str, texts: list[str]) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9_:]){re.escape(alias)}(?![A-Za-z0-9_:])")
    return any(pattern.search(text) for text in texts)


def judge_source_coverage(
    result: dict[str, Any] | None,
    required_paths: list[str],
    attempts: list[tuple[Path, str]],
    aliases: dict[str, str] | None = None,
    required_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw_evidence = (result or {}).get("evidence_checked")
    evidence = [str(item) for item in raw_evidence] if isinstance(raw_evidence, list) else []
    tool_inputs = tool_input_texts(attempts)
    aliases = aliases or {}
    missing_from_evidence = [
        path for path in required_paths if not path_mentioned(path, evidence)
    ]
    missing_from_trajectory = [
        path
        for path in required_paths
        if not path_mentioned(path, tool_inputs)
        and not (aliases.get(path) and alias_mentioned(aliases[path], tool_inputs))
    ]
    required_outputs = required_outputs or {}
    missing_outputs_from_trajectory = [
        path
        for path, alias in required_outputs.items()
        if not path_mentioned(path, tool_inputs) and not alias_mentioned(alias, tool_inputs)
    ]
    return {
        "required_source_paths": required_paths,
        "source_aliases": aliases,
        "required_output_paths": list(required_outputs),
        "output_aliases": required_outputs,
        "missing_from_evidence_checked": missing_from_evidence,
        "missing_from_tool_trajectory": missing_from_trajectory,
        "missing_outputs_from_tool_trajectory": missing_outputs_from_trajectory,
        "complete": (
            not missing_from_evidence
            and not missing_from_trajectory
            and not missing_outputs_from_trajectory
        ),
    }


def judge_continue_message(coverage: dict[str, Any] | None) -> str:
    if not coverage:
        return JUDGE_CONTINUE_MESSAGE
    missing = sorted(
        set(coverage.get("missing_from_evidence_checked") or [])
        | set(coverage.get("missing_from_tool_trajectory") or [])
    )
    missing_outputs = sorted(coverage.get("missing_outputs_from_tool_trajectory") or [])
    if not missing and not missing_outputs:
        return JUDGE_CONTINUE_MESSAGE
    aliases = coverage.get("source_aliases") or {}
    source_lines = "\n".join(
        f"- `{aliases[path]}` -> `{path}`" if aliases.get(path) else f"- `{path}`"
        for path in missing
    )
    output_aliases = coverage.get("output_aliases") or {}
    output_lines = "\n".join(
        f"- `{output_aliases[path]}` -> `{path}`"
        if output_aliases.get(path)
        else f"- `{path}`"
        for path in missing_outputs
    )
    sections = []
    if source_lines:
        sections.append(f"Missing required source evidence:\n{source_lines}")
    if output_lines:
        sections.append(f"Uninspected final outputs:\n{output_lines}")
    return (
        "The evaluation is incomplete because the inspection coverage check failed. Use the "
        "available tools to inspect every item listed below, then rewrite judge_result.json. "
        "Include each exact source path in evidence_checked only after inspecting it. Do not "
        "return a plan or progress update.\n\n"
        + "\n\n".join(sections)
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("score" in value or "rubric_scores" in value):
            return value
    return None


def result_from_stream_log(path: Path) -> dict[str, Any] | None:
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                parsed = extract_json_object(result)
                if parsed:
                    return parsed
    return extract_json_object(path.read_text(encoding="utf-8", errors="replace"))


def permanent_api_error_status_from_stream_log(path: Path) -> int | None:
    """Recognize client/request errors that another judge attempt cannot repair."""
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        status = event.get("api_error_status")
        if status is None and isinstance(event.get("message"), dict):
            status = event["message"].get("api_error_status")
        try:
            status = int(status)
        except (TypeError, ValueError):
            continue
        if status in {400, 401, 403, 413, 422}:
            return status
    return None


def normalize_judge_result(result: dict[str, Any] | None, rubric_count: int) -> dict[str, Any]:
    normalized = dict(result or {})
    raw_scores = normalized.get("rubric_scores")
    scores: dict[str, float] = {}
    if isinstance(raw_scores, dict):
        for index in range(1, rubric_count + 1):
            candidates = (str(index), f"rubric_{index:02d}", f"rubric_{index}")
            raw = next((raw_scores[key] for key in candidates if key in raw_scores), 0.0)
            try:
                scores[str(index)] = max(0.0, min(1.0, float(raw)))
            except (TypeError, ValueError):
                scores[str(index)] = 0.0
    else:
        scores = {str(index): 0.0 for index in range(1, rubric_count + 1)}
    normalized["rubric_scores"] = scores
    normalized["score"] = sum(scores.values()) / len(scores) if scores else 0.0
    normalized.setdefault("feedback", "judge did not return structured feedback")
    user_feedback = normalized.get("user_feedback")
    normalized["user_feedback"] = (
        user_feedback if isinstance(user_feedback, str) else ""
    )
    normalized.setdefault("missing_or_wrong", [])
    normalized.setdefault("evidence_checked", [])
    normalized["rubric_source"] = "initial task package"
    normalized["judge_agent"] = "claude_code_gemma4"
    return normalized


class RemoteJudgePool:
    def __init__(
        self,
        urls: list[str],
        *,
        workers_per_node: int,
        timeout_sec: int,
        api_key: str,
    ) -> None:
        self.urls = [url.rstrip("/") for url in urls]
        if not self.urls:
            raise ValueError("at least one remote judge worker URL is required")
        if workers_per_node < 1:
            raise ValueError("workers_per_node must be positive")
        self.workers_per_node = workers_per_node
        self.timeout_sec = timeout_sec
        self.api_key = api_key
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._in_flight = [0] * len(self.urls)
        self._condition = threading.Condition()

    def _acquire(self, routing_key: str) -> int:
        preferred = stable_shard_index(routing_key, len(self.urls), salt="judge-node")
        preferred_order = [
            (preferred + offset) % len(self.urls) for offset in range(len(self.urls))
        ]
        with self._condition:
            while True:
                available = [
                    index
                    for index in preferred_order
                    if self._in_flight[index] < self.workers_per_node
                ]
                if available:
                    minimum = min(self._in_flight[index] for index in available)
                    index = next(
                        index for index in available if self._in_flight[index] == minimum
                    )
                    self._in_flight[index] += 1
                    return index
                self._condition.wait()

    def _release(self, index: int) -> None:
        with self._condition:
            self._in_flight[index] -= 1
            self._condition.notify_all()

    def submit(
        self,
        judge_workspace: Path,
        task_dir: Path,
        case_dir: Path,
        *,
        timeout_sec: int | None = None,
    ) -> dict[str, Any]:
        submit_started = time.monotonic()
        routing_key = judge_routing_key(task_dir, case_dir)
        acquire_started = time.monotonic()
        index = self._acquire(routing_key)
        pool_wait_sec = time.monotonic() - acquire_started
        url = self.urls[index]
        payload = json.dumps(
            {
                "judge_workspace": str(judge_workspace.resolve()),
                "task_dir": str(task_dir.resolve()),
                "case_dir": str(case_dir.resolve()),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{url}/judge",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request_started = time.monotonic()
        try:
            with self._opener.open(
                request,
                timeout=self.timeout_sec if timeout_sec is None else timeout_sec,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"remote judge {url} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"remote judge {url} request failed: {exc}") from exc
        finally:
            self._release(index)
        remote_roundtrip_sec = time.monotonic() - request_started
        if not isinstance(result, dict) or "judge_exit_code" not in result:
            raise RuntimeError(f"remote judge {url} returned an invalid result")
        result.setdefault("judge_worker_url", url)
        result.setdefault("judge_routing_key", routing_key)
        result.setdefault("judge_prompt_key", judge_prompt_key(task_dir))
        result.setdefault("judge_client_pool_wait_sec", pool_wait_sec)
        result.setdefault("judge_remote_roundtrip_sec", remote_roundtrip_sec)
        result.setdefault(
            "judge_client_submit_total_sec", time.monotonic() - submit_started
        )
        return result


def judge_prompt_key(task_dir: Path) -> str:
    match = re.search(r"(?:^|_)task_(\d+)(?:_|$)", task_dir.name)
    if match:
        return f"prompt-{int(match.group(1))}"
    return task_dir.name


def judge_routing_key(task_dir: Path, case_dir: Path | None = None) -> str:
    prompt_key = judge_prompt_key(task_dir)
    if case_dir is None:
        return prompt_key
    digest = hashlib.sha256(f"judge-session:{case_dir.name}".encode()).hexdigest()[:16]
    return f"{prompt_key}-session-{digest}"


def stable_shard_index(key: str, shard_count: int, *, salt: str) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def judge_proxy_metrics_offset(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def summarize_judge_proxy_metrics(
    path: Path | None,
    *,
    offset: int,
    routing_key: str,
) -> dict[str, Any]:
    if path is None:
        return {}
    routing_hash = hashlib.sha256(routing_key.encode()).hexdigest()[:16]
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, offset))
            raw_lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    records: list[dict[str, Any]] = []
    for raw_line in raw_lines:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("routing_key_hash") == routing_hash:
            records.append(record)
    if not records:
        return {}
    records.sort(
        key=lambda record: (
            int(record.get("session_request_index") or 0),
            float(record.get("recorded_at_unix") or 0.0),
        )
    )

    def total(name: str) -> float:
        return sum(
            float(record.get(name) or 0.0)
            for record in records
            if isinstance(record.get(name), (int, float))
        )

    workers = [
        str(record["router_worker_id"])
        for record in records
        if record.get("router_worker_id")
    ]
    return {
        "request_count": len(records),
        "successful_request_count": sum(record.get("status") == "ok" for record in records),
        "prompt_tokens": int(total("prompt_tokens")),
        "completion_tokens": int(total("completion_tokens")),
        "reasoning_tokens": int(total("reasoning_tokens")),
        "cached_prompt_tokens": int(total("cached_prompt_tokens")),
        "proxy_duration_seconds": total("duration_seconds"),
        "router_duration_seconds": total("router_duration_ms") / 1000.0,
        "router_queue_seconds": total("router_queue_ms") / 1000.0,
        "router_retry_count": int(total("router_retry_count")),
        "sticky_request_count": sum(record.get("router_route") == "sticky" for record in records),
        "sticky_fallback_count": sum(
            record.get("router_route") == "least-inflight-fallback" for record in records
        ),
        "router_worker_ids": list(dict.fromkeys(workers)),
        "router_worker_switches": sum(
            left != right for left, right in zip(workers, workers[1:])
        ),
    }


def create_judge_runtime_workspace(judge_workspace: Path, *, temporary_root: Path | None = None) -> Path:
    temporary_root = temporary_root or os.environ.get("WSCLAW_JUDGE_TMPDIR")
    if temporary_root:
        Path(temporary_root).mkdir(parents=True, exist_ok=True)
    runtime = Path(tempfile.mkdtemp(prefix="wsj-", dir=temporary_root or None))
    for name in ("workspace", "tools"):
        (runtime / name).symlink_to(
            (judge_workspace / name).resolve(),
            target_is_directory=True,
        )
    for name in ("review", "task.md", "rubric.json"):
        source = judge_workspace / name
        if source.exists():  # Accept older persisted GT workspaces as well.
            (runtime / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
    shutil.copy2(judge_workspace / "CLAUDE.md", runtime / "CLAUDE.md")
    agents = judge_workspace / "AGENTS.md"
    shutil.copy2(agents if agents.exists() else judge_workspace / "CLAUDE.md", runtime / "AGENTS.md")
    return runtime


def run_judge(
    judge_workspace: Path, task_dir: Path, case_dir: Path, args: argparse.Namespace,
    *, attempt_runner: Callable[..., tuple[int, bool, float]] | None = None,
    runtime_factory: Callable[[Path], Path] | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    # Harness adapters share delivery, repairs, acceptance and normalization.
    initial_prompt = initial_prompt if initial_prompt is not None else judge_user_message(judge_workspace)
    (case_dir / "judge_prompt.txt").write_text(initial_prompt + "\n", encoding="utf-8")
    routing_key = judge_routing_key(task_dir, case_dir)
    routed_api_key = f"wsclaw-route-{routing_key}"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{args.wsbench_python.parent}:{env.get('PATH', '')}",
            "ANTHROPIC_BASE_URL": args.judge_api_base.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": routed_api_key,
            "ANTHROPIC_API_KEY": routed_api_key,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": args.judge_model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": args.judge_model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": args.judge_model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
            "WSCLAW_INSPECT_DEFAULT_MAX_CHARS": str(
                judge_inspection_default_max_chars(judge_workspace)
            ),
            "WSCLAW_INSPECT_ALIAS_FILE": str(
                (judge_workspace / "review" / "brief.json").resolve()
            ),
            "WSCLAW_INSPECT_COMPACT": "1",
        }
    )
    proxy_metrics_path_raw = env.get("WSCLAW_PROXY_METRICS_JSONL", "").strip()
    proxy_metrics_path = Path(proxy_metrics_path_raw) if proxy_metrics_path_raw else None
    proxy_metrics_start = judge_proxy_metrics_offset(proxy_metrics_path)
    rubric = load_json(task_dir / "reward" / "rubric.json")
    rubric_count = len(rubric.get("grounded_rubric") or [])
    required_source_paths = required_judge_source_paths(judge_workspace)
    source_aliases = judge_source_aliases(judge_workspace)
    required_outputs = required_judge_outputs(judge_workspace)
    shared_result_path = judge_workspace / "judge_result.json"
    shared_result_path.unlink(missing_ok=True)
    attempts: list[tuple[Path, str]] = []
    parsed = None
    last_complete_candidate = None
    coverage = judge_source_coverage(
        None, required_source_paths, attempts, source_aliases, required_outputs
    )
    exit_code = 0
    timed_out = False
    permanent_api_error_status = None
    duration = 0.0
    runtime = (runtime_factory or create_judge_runtime_workspace)(judge_workspace)
    result_path = runtime / "judge_result.json"
    try:
        for attempt_index in range(1, getattr(args, "judge_max_attempts", JUDGE_MAX_ATTEMPTS) + 1):
            if attempt_index == 1:
                prompt = initial_prompt
            else:
                prompt = (
                    initial_prompt
                    + "\n\n## Result Repair\n\n"
                    + JUDGE_CONTINUE_MESSAGE
                    + "\n\nA candidate judge_result.json from the preceding attempt may be present. "
                    "Keep defensible scores, repair only the incomplete schema or feedback, and "
                    "rewrite the result."
                )
            attempt_log = case_dir / f"claude_judge.attempt{attempt_index}.log"
            attempts.append((attempt_log, prompt))
            command = [
                str(args.claude_bin),
                "--disable-slash-commands",
                "--tools",
                "default",
                "--allowedTools",
                "Task",
                "Bash",
                "CronCreate",
                "CronDelete",
                "CronList",
                "Edit",
                "EnterWorktree",
                "ExitWorktree",
                "NotebookEdit",
                "Read",
                "ReportFindings",
                "ScheduleWakeup",
                "SendMessage",
                "TaskCreate",
                "TaskGet",
                "TaskList",
                "TaskOutput",
                "TaskStop",
                "TaskUpdate",
                "WebFetch",
                "WebSearch",
                "Workflow",
                "Write",
                "--setting-sources",
                "project",
                "--no-session-persistence",
                "--verbose",
                "--output-format=stream-json",
                "--max-turns",
                str(args.judge_max_turns),
                "--effort",
                args.judge_effort,
                "--model",
                args.judge_model,
                "-p",
                prompt,
            ]
            exit_code, attempt_timed_out, attempt_duration = (attempt_runner or run_process)(
                command,
                cwd=runtime,
                env=env,
                log_path=attempt_log,
                timeout_sec=args.judge_timeout,
                restrict_writes_to=runtime,
            )
            timed_out = timed_out or attempt_timed_out
            duration += attempt_duration
            permanent_api_error_status = permanent_api_error_status_from_stream_log(
                attempt_log
            )
            if result_path.is_file():
                try:
                    candidate = load_json(result_path)
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if complete_judge_result(candidate, rubric_count):
                    last_complete_candidate = candidate
                    coverage = judge_source_coverage(
                        candidate,
                        required_source_paths,
                        attempts,
                        source_aliases,
                        required_outputs,
                    )
                    parsed = candidate
                    break
                invalid_path = (
                    judge_workspace / f"judge_result.invalid_attempt{attempt_index}.json"
                )
                shutil.copy2(result_path, invalid_path)
            candidate = result_from_stream_log(attempt_log)
            if complete_judge_result(candidate, rubric_count):
                last_complete_candidate = candidate
                coverage = judge_source_coverage(
                    candidate,
                    required_source_paths,
                    attempts,
                    source_aliases,
                    required_outputs,
                )
                parsed = candidate
                write_json(result_path, candidate)
                break
            if (attempt_timed_out or permanent_api_error_status is not None
                    or exit_code in getattr(args, "judge_terminal_exit_codes", ())):
                break
        if parsed is not None:
            write_json(shared_result_path, parsed)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)

    transcript = cleaned_judge_transcript(attempts)
    log_path = case_dir / "claude_judge.log"
    with log_path.open("w", encoding="utf-8") as combined:
        for attempt_log, _prompt in attempts:
            if attempt_log.is_file():
                combined.write(attempt_log.read_text(encoding="utf-8", errors="replace"))
                attempt_log.unlink()
    accepted = complete_judge_result(parsed, rubric_count)
    if not accepted:
        exit_code = exit_code or 2
    result = normalize_judge_result(parsed, rubric_count)
    model_metrics = summarize_judge_proxy_metrics(
        proxy_metrics_path,
        offset=proxy_metrics_start,
        routing_key=routing_key,
    )
    result.update(
        {
            "judge_exit_code": exit_code,
            "judge_timed_out": timed_out,
            "judge_permanent_api_error_status": permanent_api_error_status,
            "judge_duration_sec": duration,
            "judge_log_path": str(log_path),
            "judge_workspace": str(judge_workspace),
            "judge_attempts": len(attempts),
            "judge_routing_key": routing_key,
            "judge_prompt_key": judge_prompt_key(task_dir),
            "judge_model_metrics": model_metrics,
            "judge_result_complete": accepted,
            "judge_schema_complete": complete_judge_result(last_complete_candidate, rubric_count),
            "judge_source_coverage": coverage,
        }
    )
    judge_transcript_path = case_dir / "judge_cleaned_transcripts.json"
    transcript.update(
        {
            "model": args.judge_model,
            "instructions_path": str(judge_workspace / "CLAUDE.md"),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_sec": duration,
        }
    )
    write_json(judge_transcript_path, transcript)
    result["judge_cleaned_transcript_path"] = str(judge_transcript_path)
    write_json(case_dir / "judge_result.json", result)
    return result


def case_dir_for(item: dict[str, Any], run_dir: Path) -> Path:
    case_name = f"task_{int(item['index']):04d}_{item['task_id']}"
    case_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_name)
    return run_dir / "cases" / case_name


def is_normal_completion(record: dict[str, Any]) -> bool:
    nanobot = record.get("nanobot") or {}
    solver_acceptable = bool(nanobot.get("timed_out")) or nanobot.get("exit_code") == 0
    return (
        record.get("status") == "completed"
        and solver_acceptable
        and record.get("judge_exit_code") == 0
        and not record.get("judge_timed_out")
    )


def completion_status(nanobot: dict[str, Any], judge: dict[str, Any]) -> str:
    # A solver that exits cleanly but fails to produce a usable deliverable is
    # still an evaluable attempt: the judge should assign the corresponding
    # (usually zero) score.  Reserve infrastructure failure for a non-zero
    # process exit, timeout, interrupted process, or failed judge.
    # An evaluated agent timeout is a legitimate terminal attempt: the judge
    # should score whatever the agent produced before its budget elapsed.  A
    # judge timeout is infrastructure failure and must never be accepted.
    solver_acceptable = bool(nanobot.get("timed_out")) or nanobot.get("exit_code") == 0
    judge_completed = (
        judge.get("judge_exit_code") == 0 and not judge.get("judge_timed_out")
    )
    if solver_acceptable and judge_completed:
        return "completed"
    exit_codes = (nanobot.get("exit_code"), judge.get("judge_exit_code"))
    if not nanobot.get("timed_out") and not judge.get("judge_timed_out") and any(
        isinstance(code, int) and code < 0 for code in exit_codes
    ):
        return "interrupted"
    return "failed"


def log_stage(args: argparse.Namespace, item: dict[str, Any], stage: str, **fields: Any) -> None:
    parts = [f"task={int(item['index'])}", f"stage={stage}"]
    parts.extend(f"{key}={value}" for key, value in fields.items())
    with args.progress_lock:
        print(" ".join(parts), flush=True)


def load_completed_records(
    tasks: list[dict[str, Any]],
    run_dir: Path,
    *,
    accept_failed: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for item in tasks:
        result_path = case_dir_for(item, run_dir) / "result.json"
        if result_path.is_file():
            record = load_json(result_path)
            terminal_record = record.get("status") in {"completed", "failed", "interrupted"}
            if (
                int(record.get("index", -1)) == int(item["index"])
                and (is_normal_completion(record) or (accept_failed and terminal_record))
            ):
                completed.append(record)
                continue
        pending.append(item)
    return completed, pending


def finalize_record(
    item: dict[str, Any],
    args: argparse.Namespace,
    case_dir: Path,
    record: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    record["duration_sec"] = time.perf_counter() - started
    write_json(case_dir / "result.json", record)
    log_stage(
        args,
        item,
        "finished",
        status=record["status"],
        judge=record.get("score"),
        duration=f"{record['duration_sec']:.1f}s",
    )
    return record


def write_solver_checkpoint(
    *,
    item: dict[str, Any],
    case_dir: Path,
    nanobot: dict[str, Any],
    artifacts: list[dict[str, Any]],
    judge_workspace: Path,
) -> Path:
    path = case_dir / "solver_checkpoint.json"
    write_json(
        path,
        {
            "schema_version": SOLVER_CHECKPOINT_SCHEMA,
            "task_id": item["task_id"],
            "source_index": int(item["index"]),
            "source_case_dir": str(case_dir),
            "nanobot": nanobot,
            "artifacts": artifacts,
            "judge_workspace": str(judge_workspace),
            "completed_at": time.time(),
        },
    )
    return path


def restore_solver_checkpoint(
    item: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any] | None:
    raw_path = item.get("solver_checkpoint")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    checkpoint_path = Path(raw_path)
    if not checkpoint_path.is_file():
        return None
    checkpoint = load_json(checkpoint_path)
    if (
        checkpoint.get("schema_version") != SOLVER_CHECKPOINT_SCHEMA
        or checkpoint.get("task_id") != item.get("task_id")
    ):
        raise ValueError(f"solver checkpoint identity mismatch: {checkpoint_path}")
    judge_workspace = Path(str(checkpoint.get("judge_workspace") or ""))
    nanobot = checkpoint.get("nanobot")
    artifacts = checkpoint.get("artifacts")
    if not judge_workspace.is_dir() or not isinstance(nanobot, dict) or not isinstance(artifacts, list):
        raise ValueError(f"solver checkpoint is incomplete: {checkpoint_path}")
    case_dir = case_dir_for(item, run_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    copied_checkpoint = case_dir / "solver_checkpoint.json"
    if checkpoint_path.resolve() != copied_checkpoint.resolve():
        write_json(copied_checkpoint, checkpoint)
    log_stage(args, item, "nanobot_resumed", checkpoint=checkpoint_path)
    return {
        "item": item,
        "task_dir": Path(item["_task_dir"]),
        "case_dir": case_dir,
        "started": time.perf_counter(),
        "record": {
            "index": int(item["index"]),
            "task_id": item["task_id"],
            "scenario_id": item.get("scenario_id"),
            "persona": item.get("persona"),
            "case_dir": str(case_dir),
            "status": "running",
            "solver_checkpoint": str(checkpoint_path),
        },
        "nanobot": nanobot,
        "artifacts": artifacts,
        "judge_workspace": judge_workspace,
    }


def run_solver_stage(
    item: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    *,
    user_query: str | None = None,
) -> dict[str, Any]:
    """Run only the Qwen/nanobot stage and prepare immutable judge inputs.

    Returning the judge inputs instead of invoking the judge here is what lets
    a rolling solver pool refill immediately.  A slow Gemma request therefore
    occupies a judge worker, never one of the solver/Qwen workers.
    """
    if user_query is None and isinstance(item.get("_prompt"), str):
        user_query = item["_prompt"]
    restored = restore_solver_checkpoint(item, args, run_dir)
    if restored is not None:
        return restored
    task_dir = Path(item["_task_dir"])
    case_dir = case_dir_for(item, run_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "index": int(item["index"]),
        "task_id": item["task_id"],
        "scenario_id": item.get("scenario_id"),
        "persona": item.get("persona"),
        "case_dir": str(case_dir),
        "status": "running",
    }
    try:
        log_stage(args, item, "started")
        workspace, entry, outputs, _source_root = prepare_workspace(
            task_dir,
            case_dir,
            containerized=bool(getattr(args, "nanobot_apptainer_image", None)),
        )
        with args.nanobot_slots:
            log_stage(args, item, "nanobot_started")
            nanobot_args = copy.copy(args)
            nanobot_args.nanobot_timeout = bounded_stage_timeout(
                started,
                args,
                args.nanobot_timeout,
            )
            endpoint_pool = getattr(args, "nanobot_endpoint_pool", None)
            if endpoint_pool is None:
                nanobot = run_nanobot(
                    workspace=workspace,
                    entry=entry,
                    case_dir=case_dir,
                    args=nanobot_args,
                    user_query=user_query,
                )
            else:
                with endpoint_pool.acquire(str(case_dir)) as lease:
                    routed_args = copy.copy(nanobot_args)
                    routed_args.nanobot_api_base = lease.url
                    nanobot = run_nanobot(
                        workspace=workspace,
                        entry=entry,
                        case_dir=case_dir,
                        args=routed_args,
                        user_query=user_query,
                    )
        log_stage(
            args,
            item,
            "nanobot_finished",
            exit=nanobot["exit_code"],
            timeout=nanobot["timed_out"],
            duration=f"{nanobot['duration_sec']:.1f}s",
            transcript=nanobot["cleaned_transcript_path"],
        )
        artifacts = [artifact_record(workspace, path) for path in outputs]
        log_stage(
            args,
            item,
            "artifacts_checked",
            artifacts=sum(int(artifact["exists"]) for artifact in artifacts),
            valid_formats=sum(int(artifact["format_ok"]) for artifact in artifacts),
            required=len(artifacts),
        )
        judge_workspace, _ = prepare_judge_workspace(
            task_dir=task_dir,
            workspace=workspace,
            outputs=outputs,
            case_dir=case_dir,
            entry=entry,
        )
        solver_checkpoint = write_solver_checkpoint(
            item=item,
            case_dir=case_dir,
            nanobot=nanobot,
            artifacts=artifacts,
            judge_workspace=judge_workspace,
        )
        return {
            "item": item,
            "task_dir": task_dir,
            "case_dir": case_dir,
            "started": started,
            "record": record,
            "nanobot": nanobot,
            "artifacts": artifacts,
            "judge_workspace": judge_workspace,
            "solver_checkpoint": solver_checkpoint,
        }
    except Exception as exc:
        case_timed_out = isinstance(exc, CaseTimeoutError)
        record.update(
            {
                "status": "failed",
                "score": 0.0,
                "case_timed_out": case_timed_out,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(case_dir / "error.json", record)
        return {
            "item": item,
            "record": finalize_record(item, args, case_dir, record, started),
            "terminal": True,
        }


def run_judge_stage(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Consume one completed solver result on the independent Gemma pool."""
    item = state["item"]
    task_dir = state["task_dir"]
    case_dir = state["case_dir"]
    started = state["started"]
    record = state["record"]
    nanobot = state["nanobot"]
    artifacts = state["artifacts"]
    judge_workspace = state["judge_workspace"]
    try:
        with args.judge_slots:
            log_stage(args, item, "judge_started")
            judge_timeout = bounded_stage_timeout(
                started,
                args,
                args.judge_timeout,
            )
            judge_worker_pool = getattr(args, "judge_worker_pool", None)
            if judge_worker_pool is not None:
                judge = judge_worker_pool.submit(
                    judge_workspace,
                    task_dir,
                    case_dir,
                    timeout_sec=judge_timeout,
                )
            else:
                judge_args = copy.copy(args)
                judge_args.judge_timeout = judge_timeout
                judge = run_judge(judge_workspace, task_dir, case_dir, judge_args)
        log_stage(
            args,
            item,
            "judge_finished",
            score=f"{judge['score']:.4f}",
            exit=judge["judge_exit_code"],
            timeout=judge["judge_timed_out"],
            duration=f"{judge['judge_duration_sec']:.1f}s",
        )
        record.update(
            {
                "status": completion_status(nanobot, judge),
                "score": judge["score"],
                "nanobot": nanobot,
                "trajectory_metrics": nanobot.get("trajectory_metrics"),
                "judge_exit_code": judge["judge_exit_code"],
                "judge_timed_out": judge["judge_timed_out"],
                "artifacts": artifacts,
                "judge_result_path": str(case_dir / "judge_result.json"),
            }
        )
    except Exception as exc:
        case_timed_out = isinstance(exc, CaseTimeoutError) or (
            getattr(args, "case_timeout", None) is not None
            and time.perf_counter() - started >= float(args.case_timeout)
        )
        record.update(
            {
                "status": "failed",
                "score": 0.0,
                "judge_timed_out": case_timed_out,
                "case_timed_out": case_timed_out,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(case_dir / "error.json", record)
    return finalize_record(item, args, case_dir, record, started)


def run_one(
    item: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    *,
    user_query: str | None = None,
) -> dict[str, Any]:
    """Compatibility path that runs one solver and its judge synchronously."""
    state = run_solver_stage(
        item,
        args,
        run_dir,
        user_query=user_query,
    )
    if state.get("terminal"):
        return state["record"]
    return run_judge_stage(state, args)


def run_pipelined(
    items: list[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    on_record: Callable[[dict[str, Any]], None],
) -> None:
    """Run a bounded rolling solver pool with an independent judge pool.

    At most ``nanobot_workers`` solver calls are active.  As soon as one solver
    finishes, its prepared workspace is queued to Gemma and the vacated solver
    slot claims the next item.  Judge latency can build a bounded executor
    queue, but it cannot drain the Qwen pool or introduce a batch barrier.
    """
    pending = iter(items)
    solver_futures: dict[Any, dict[str, Any]] = {}
    judge_futures: dict[Any, dict[str, Any]] = {}

    with (
        ThreadPoolExecutor(max_workers=max(1, args.nanobot_workers)) as solver_executor,
        ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as judge_executor,
    ):
        exhausted = False

        def refill_solver_pool() -> None:
            nonlocal exhausted
            while not exhausted and len(solver_futures) < max(1, args.nanobot_workers):
                try:
                    item = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                future = solver_executor.submit(run_solver_stage, item, args, run_dir)
                solver_futures[future] = item

        refill_solver_pool()
        while solver_futures or judge_futures:
            done, _pending = wait(
                set(solver_futures) | set(judge_futures),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                if future in solver_futures:
                    solver_futures.pop(future)
                    state = future.result()
                    if state.get("terminal"):
                        on_record(state["record"])
                    else:
                        judge_future = judge_executor.submit(run_judge_stage, state, args)
                        judge_futures[judge_future] = state["item"]
                else:
                    judge_futures.pop(future)
                    on_record(future.result())
            # Refill after processing every completed solver future.  This is
            # deliberately independent of judge completion.
            refill_solver_pool()


def parse_indices(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--prompt-data",
        type=Path,
        default=None,
        help=(
            "Use an ordered training-eval prompt JSONL as the authoritative "
            "task and prompt selection instead of data-root/manifest.json."
        ),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--task-indices", default=None)
    parser.add_argument("--exclude-task-indices", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=192)
    parser.add_argument("--nanobot-workers", type=int, default=64)
    parser.add_argument("--judge-workers", type=int, default=64)
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "decouple solver and judge pools so each completed solver slot "
            "immediately claims another task instead of waiting for Gemma"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-accept-failed",
        action="store_true",
        help=(
            "When resuming, retain prior terminal infrastructure failures as zero-score "
            "records instead of rerunning them."
        ),
    )
    parser.add_argument(
        "--failed-retries",
        type=int,
        default=0,
        help=(
            "Within one fresh invocation, rerun infrastructure-failed cases up to this many "
            "times. This does not resume a prior invocation or reuse its accepted scores."
        ),
    )
    parser.add_argument("--nanobot-api-base", required=True)
    parser.add_argument("--nanobot-api-key", default="apiKey")
    parser.add_argument(
        "--nanobot-api-key-file",
        type=Path,
        default=None,
        help=(
            "Read the solver API key from a mode-0600 file. This keeps service "
            "credentials out of process command lines."
        ),
    )
    parser.add_argument("--nanobot-model", default="qwen3-4b-thinking")
    parser.add_argument("--nanobot-python", type=Path, default=DEFAULT_NANOBOT_PYTHON)
    parser.add_argument("--nanobot-runner", type=Path, default=DEFAULT_NANOBOT_RUNNER)
    parser.add_argument("--nanobot-apptainer-image", type=Path, default=None)
    parser.add_argument(
        "--nanobot-container-workspace",
        type=Path,
        default=CONTAINER_WORKSPACE,
    )
    parser.add_argument("--apptainer-bin", type=Path, default=DEFAULT_APPTAINER_BIN)
    parser.add_argument("--wsbench-python", type=Path, default=DEFAULT_WSBENCH_PYTHON)
    parser.add_argument("--nanobot-timeout", type=int, default=900)
    parser.add_argument(
        "--case-timeout",
        type=int,
        default=None,
        help=(
            "Optional end-to-end budget covering workspace setup, nanobot, judge queueing, "
            "and judge execution. Use 1500 to match the WSClaw training eval session budget."
        ),
    )
    parser.add_argument("--nanobot-max-tokens", type=int, default=8192)
    parser.add_argument("--nanobot-temperature", type=float, default=0.2)
    parser.add_argument("--nanobot-top-p", type=float, default=1.0)
    parser.add_argument("--nanobot-top-k", type=int, default=-1)
    parser.add_argument("--nanobot-min-p", type=float, default=None)
    parser.add_argument("--context-window", type=int, default=65536)
    parser.add_argument(
        "--context-block-limit",
        type=int,
        default=None,
        help=(
            "Optional prompt-history budget before provider-side tool-template "
            "expansion; leaves the model context window and output cap unchanged"
        ),
    )
    parser.add_argument("--max-tool-iterations", type=int, default=80)
    parser.add_argument("--judge-api-base", default="")
    parser.add_argument("--judge-api-key", default="apiKey")
    parser.add_argument("--judge-model", default="gemma-user-agent")
    parser.add_argument("--judge-worker-urls", default="")
    parser.add_argument("--judge-workers-per-node", type=int, default=64)
    parser.add_argument("--claude-bin", type=Path, default=DEFAULT_CLAUDE_BIN)
    parser.add_argument("--judge-timeout", type=int, default=2700)
    parser.add_argument("--judge-max-turns", type=int, default=30)
    parser.add_argument("--judge-effort", choices=("low", "medium", "high"), default="medium")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.failed_retries < 0:
        raise ValueError("--failed-retries must be non-negative")
    if args.case_timeout is not None and args.case_timeout <= 0:
        raise ValueError("--case-timeout must be positive")
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()
    if args.prompt_data is not None:
        args.prompt_data = args.prompt_data.resolve()
    if args.nanobot_api_key_file is not None:
        key_file = args.nanobot_api_key_file.resolve()
        if not key_file.is_file():
            raise FileNotFoundError(key_file)
        if stat.S_IMODE(key_file.stat().st_mode) != 0o600:
            raise PermissionError(
                f"nanobot API key file must have mode 0600: {key_file}"
            )
        args.nanobot_api_key = key_file.read_text(encoding="utf-8").strip()
        if not args.nanobot_api_key:
            raise ValueError(f"nanobot API key file is empty: {key_file}")
    judge_worker_urls = [
        url.strip().rstrip("/") for url in args.judge_worker_urls.split(",") if url.strip()
    ]
    if not judge_worker_urls and not args.judge_api_base:
        raise ValueError("set --judge-worker-urls or --judge-api-base")
    required_paths = [args.nanobot_python, args.nanobot_runner, args.wsbench_python]
    required_paths.append(
        args.prompt_data if args.prompt_data is not None else args.data_root / "manifest.json"
    )
    if not judge_worker_urls:
        required_paths.append(args.claude_bin)
    if args.nanobot_apptainer_image:
        required_paths.extend((args.apptainer_bin, args.nanobot_apptainer_image))
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S_wsclaw-eval", time.localtime())
    run_dir = args.run_root / run_id
    if args.resume and not run_dir.is_dir():
        raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    task_indices = parse_indices(args.task_indices)
    excluded_task_indices = parse_indices(args.exclude_task_indices)
    if args.prompt_data is not None:
        tasks = discover_tasks_from_prompt_data(
            args.prompt_data,
            task_indices=task_indices,
            excluded_task_indices=excluded_task_indices,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
    else:
        tasks = discover_tasks(
            args.data_root,
            task_indices=task_indices,
            excluded_task_indices=excluded_task_indices,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
    write_json(
        run_dir / "config.json",
        {
            "run_id": run_id,
            "data_root": str(args.data_root),
            "prompt_data": str(args.prompt_data) if args.prompt_data is not None else None,
            "prompt_data_sha256": (
                hashlib.sha256(args.prompt_data.read_bytes()).hexdigest()
                if args.prompt_data is not None
                else None
            ),
            "task_count": len(tasks),
            "task_indices": sorted(task_indices) if task_indices is not None else None,
            "excluded_task_indices": (
                sorted(excluded_task_indices)
                if excluded_task_indices is not None
                else []
            ),
            "workers": args.workers,
            "nanobot_workers": args.nanobot_workers,
            "judge_workers": args.judge_workers,
            "pipeline": args.pipeline,
            "nanobot_api_base": args.nanobot_api_base,
            "nanobot_model": args.nanobot_model,
            "nanobot_runtime": "apptainer" if args.nanobot_apptainer_image else "host",
            "nanobot_apptainer_image": (
                str(args.nanobot_apptainer_image) if args.nanobot_apptainer_image else None
            ),
            "nanobot_container_workspace": str(args.nanobot_container_workspace),
            "nanobot_timeout": args.nanobot_timeout,
            "nanobot_max_tokens": args.nanobot_max_tokens,
            "nanobot_temperature": args.nanobot_temperature,
            "nanobot_top_p": args.nanobot_top_p,
            "nanobot_top_k": args.nanobot_top_k,
            "nanobot_min_p": args.nanobot_min_p,
            "context_window": args.context_window,
            "context_block_limit": args.context_block_limit,
            "max_tool_iterations": args.max_tool_iterations,
            "input_materialization": "full_persona_workspace_read_only_bind",
            "judge_api_base": args.judge_api_base,
            "judge_worker_urls": judge_worker_urls,
            "judge_workers_per_node": args.judge_workers_per_node,
            "judge_model": args.judge_model,
            "judge_timeout": args.judge_timeout,
            "judge_max_turns": args.judge_max_turns,
            "judge_effort": args.judge_effort,
            "evaluated_agent": f"{args.nanobot_model}+nanobot",
            "judge_agent": "gemma4+claude_code",
            "resume": args.resume,
            "resume_accept_failed": args.resume_accept_failed,
            "failed_retries": args.failed_retries,
            "case_timeout": args.case_timeout,
        },
    )
    print(f"run_dir={run_dir}", flush=True)
    print(
        f"tasks={len(tasks)} orchestrators={args.workers} "
        f"nanobot_workers={args.nanobot_workers} judge_workers={args.judge_workers}",
        flush=True,
    )
    args.nanobot_slots = threading.Semaphore(max(1, args.nanobot_workers))
    args.judge_slots = threading.Semaphore(max(1, args.judge_workers))
    args.judge_worker_pool = (
        RemoteJudgePool(
            judge_worker_urls,
            workers_per_node=args.judge_workers_per_node,
            timeout_sec=args.judge_timeout + 120,
            api_key=args.judge_api_key,
        )
        if judge_worker_urls
        else None
    )
    args.progress_lock = threading.Lock()
    records: list[dict[str, Any]] = []
    pending_tasks = tasks
    if args.resume:
        records, pending_tasks = load_completed_records(
            tasks,
            run_dir,
            accept_failed=args.resume_accept_failed,
        )
        print(f"resumed={len(records)} pending={len(pending_tasks)}", flush=True)
        write_json(run_dir / "results_manifest.json", sorted(records, key=lambda row: row["index"]))
    def record_completion(record: dict[str, Any]) -> None:
        records.append(record)
        write_json(
            run_dir / "results_manifest.json",
            sorted(records, key=lambda row: row["index"]),
        )

    def run_pending(items: list[dict[str, Any]]) -> None:
        if args.pipeline:
            run_pipelined(items, args, run_dir, record_completion)
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(run_one, item, args, run_dir): item
                    for item in items
                }
                for future in as_completed(futures):
                    record_completion(future.result())

    run_pending(pending_tasks)
    for retry_index in range(1, args.failed_retries + 1):
        completed_records, retry_tasks = load_completed_records(tasks, run_dir)
        if not retry_tasks:
            break
        print(
            f"failed_retry={retry_index}/{args.failed_retries} pending={len(retry_tasks)}",
            flush=True,
        )
        records = completed_records
        write_json(
            run_dir / "results_manifest.json",
            sorted(records, key=lambda row: row["index"]),
        )
        run_pending(retry_tasks)
    completed = [record for record in records if is_normal_completion(record)]
    summary = {
        "run_id": run_id,
        "total": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "avg_score": sum(float(record["score"]) for record in completed) / len(completed) if completed else 0.0,
        "normal_completed": len(completed),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
