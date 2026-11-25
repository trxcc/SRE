from __future__ import annotations
import json

import shutil
import sys
from pathlib import Path
from typing import Any

from .common import atomic_json, copy_clean_tree, link_clean_tree, read_json, reset_dir, tree_hash
from .contracts import validate_rubric


def resolve_rubric_prompt_path(configured_path: Path, dataset: str) -> Path:
    if dataset not in {"wsclaw", "researchclaw"}:
        raise ValueError(f"unsupported rubric dataset: {dataset}")
    expected_name = f"rubric_agent_{dataset}.md"
    if configured_path.name == "rubric_agent.md":
        selected = configured_path.with_name(expected_name)
    elif configured_path.name == expected_name or configured_path.name.startswith(
        f"rubric_agent_{dataset}_"
    ):
        selected = configured_path
    else:
        raise ValueError(
            f"rubric prompt {configured_path.name!r} does not match dataset {dataset!r}"
        )
    if not selected.is_file():
        raise FileNotFoundError(f"dataset-specific rubric prompt missing: {selected}")
    return selected


def import_native(project_root: Path) -> tuple[Any, Any, Any, Any]:
    root = str(project_root.resolve())
    src = str((project_root / "src").resolve())
    for value in (root, src):
        if value not in sys.path:
            sys.path.insert(0, value)
    from clawgym_nanobot import researchclaw_reward, runtime, tasks
    from scripts.clawgym_nanobot import wsclaw_eval_only

    return runtime, tasks, wsclaw_eval_only, researchclaw_reward


def compile_ws_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    validate_rubric(rubric)
    grounded = [
        f"{item['criterion']} Expected: {item['expected']} Inspect: {', '.join(item['target_paths'])}"
        for item in rubric["criteria"]
    ]
    rubric_types = ["Results Assessment" for _item in rubric["criteria"]]
    rules = [
        {
            "id": f"rubric_{index:02d}",
            "description": (
                f"{item['criterion']} Expected: {item['expected']} "
                f"Inspect: {', '.join(item['target_paths'])}. Reason: {item['reason']}"
            ),
            "type": "Results Assessment",
            "weight": 1.0,
        }
        for index, item in enumerate(rubric["criteria"], 1)
    ]
    return {"grounded_rubric": grounded, "rubric_types": rubric_types, "rules": rules}


def _copy_evidence_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)


def materialize_candidate_evidence(
    *, task_dir: Path, candidate_workspace: Path, destination: Path
) -> dict[str, Any]:
    """Create a bounded, symlink-free view of task evidence and deliverables."""
    reset_dir(destination)
    entry = read_json(task_dir / "data_entry.json")
    metadata = entry.get("metadata") or {}
    required_sources = [str(path) for path in metadata.get("required_source_files") or []]
    required_outputs = [str(path) for path in metadata.get("required_output_paths") or []]
    included: list[str] = []
    missing: list[str] = []
    if required_sources or required_outputs:
        # WSClaw persona workspaces can exceed 1 GiB.  The task metadata names
        # the exact source/output surface, so materialize only those files.
        for relative in required_sources:
            source = candidate_workspace / "input" / relative
            destination_path = destination / "input" / relative
            if source.exists():
                _copy_evidence_path(source, destination_path)
                included.append(f"input/{relative}")
            else:
                missing.append(f"input/{relative}")
        for relative in required_outputs:
            source = candidate_workspace / relative
            destination_path = destination / relative
            if source.exists():
                _copy_evidence_path(source, destination_path)
                included.append(relative)
            else:
                missing.append(relative)
    else:
        # ResearchClaw recovery snapshots may contain both candidate outputs
        # and the immutable task inputs.  Keep candidate deliverables at their
        # task-native output paths, but place authoritative source evidence
        # under one explicit input_files/ layer.  This prevents a source guide
        # or paper from being mistaken for candidate-authored work.  Never copy
        # task_dir/reward: it contains the GT rubric/reference answer.
        copy_clean_tree(
            candidate_workspace,
            destination,
            extra_excludes={
                ".runtime", "skills", "tools", "USER.md", "TOOLS.md",
                "AGENTS.md", "SOUL.md", "HEARTBEAT.md", "task_context.json",
            },
        )
        task_inputs = task_dir / "input_files"
        if not task_inputs.is_dir():
            raise FileNotFoundError(f"ResearchClaw task input_files missing: {task_inputs}")
        source_files = {
            path.relative_to(task_inputs).as_posix()
            for path in task_inputs.rglob("*")
            if path.is_file()
        }
        # Remove captured copies of task inputs from the candidate root before
        # installing the authoritative source tree in its dedicated layer.
        for relative in sorted(source_files):
            captured = destination / relative
            if captured.is_file() or captured.is_symlink():
                captured.unlink()
        input_root = destination / "input_files"
        input_root.mkdir()
        for source in sorted(task_inputs.iterdir()):
            _copy_evidence_path(source, input_root / source.name)
        candidate_files = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file() and "input_files" not in path.relative_to(destination).parts
        }
        included = [
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        ]
    symlinks = [path for path in destination.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"materialized evidence still contains symlinks: {symlinks[:10]}")
    materialization = (
        "bounded_isolated_copy_v1"
        if required_sources or required_outputs
        else "bounded_isolated_input_layer_v2"
    )
    result = {"included": included, "missing": missing, "materialization": materialization}
    if required_sources or required_outputs:
        result["candidate_files"] = sorted(path for path in required_outputs if path in included)
        result["task_source_files"] = sorted(
            f"input/{path}" for path in required_sources if f"input/{path}" in included
        )
    else:
        result["candidate_files"] = sorted(candidate_files)
        result["task_source_files"] = sorted(f"input_files/{path}" for path in source_files)
    return result


def build_rubric_workspace(
    *,
    project_root: Path,
    dataset: str,
    destination: Path,
    task_dir: Path,
    anchor_workspace: Path,
    anchor_final_message: str,
    anchor_feedback: str,
    anchor_score: float,
    rubric_prompt_path: Path,
    rubric_schema_path: Path,
    trajectory_path: Path | None = None,
) -> dict[str, Any]:
    if dataset not in {"wsclaw", "researchclaw"}:
        raise ValueError(f"unsupported rubric dataset: {dataset}")
    rubric_prompt_path = resolve_rubric_prompt_path(rubric_prompt_path, dataset)
    reset_dir(destination)
    task = read_json(task_dir / "data_entry.json")
    if dataset == "wsclaw":
        # Keep the exec agent's mounted filesys and task tools intact. Runtime
        # sessions are deliberately excluded so the rubric starts fresh.
        excluded = {".git", ".nanobot", "sessions", "reward", "_judge",
                    "judge_workspace", "reference_answer", "__pycache__"}
        def ignore(_folder: str, names: list[str]) -> set[str]:
            return {name for name in names if name in excluded or name.startswith(".nanobot-no-web-bin")}
        anchor = destination / "anchor_workspace"
        shutil.copytree(anchor_workspace, anchor, symlinks=True, ignore=ignore)
        for entry in ("input", "Desktop", "tools", "skills", "memory", ".runtime",
                      "USER.md", "TOOLS.md", "HEARTBEAT.md", ".gitignore", "task_context.json"):
            source = anchor / entry
            if not source.exists():
                raise FileNotFoundError(f"WSClaw exec workspace entry missing: {entry}")
            (destination / entry).symlink_to(Path("anchor_workspace") / entry,
                                              target_is_directory=source.is_dir())
        metadata = task.get("metadata") or {}
        sources = [f"input/{name}" for name in metadata.get("required_source_files") or []]
        outputs = list(metadata.get("required_output_paths") or [])
        for relative in sources:
            if not (destination / relative).is_file():
                raise FileNotFoundError(f"WSClaw mounted source missing: {relative}")
        missing_outputs = [relative for relative in outputs
                           if not (destination / relative).is_file()]
        if not (destination / "tools" / "inspect_workspace_file.py").is_file():
            raise FileNotFoundError("WSClaw native inspector missing from exec workspace")
        evidence = {
            "materialization": "complete_task_facing_workspace_mount_view_v1",
            "workspace": "anchor_workspace", "task_source_files": sources,
            "candidate_files": outputs, "missing_candidate_files": missing_outputs,
            "input_mounts_preserved": True,
        }
    else:
        evidence = materialize_candidate_evidence(
            task_dir=task_dir, candidate_workspace=anchor_workspace, destination=destination,
        )
    task_text = str(task.get("user_query") or "")
    (destination / "TASK.md").write_text(f"# Original task\n\n{task_text}\n", encoding="utf-8")
    anchor_intro = (
        "# Anchor and single feedback\n\nThe anchor is one sampled answer, not ground truth. "
        "Its candidate deliverables keep their requested output paths in this workspace. "
        f"Original task inputs are under {'input/' if dataset == 'wsclaw' else 'input_files/'}. "
        "The feedback identifies useful issues but is not the sole rubric.\n\n"
    )
    feedback_heading = "User Feedback" if dataset == "wsclaw" else "User feedback"
    (destination / "ANCHOR.md").write_text(
        anchor_intro +
        f"## Final reply\n\n{anchor_final_message}\n\n"
        f"## {feedback_heading}\n\n{anchor_feedback}\n\n"
        f"## Anchor GT score\n\n{float(anchor_score):.8f}\n",
        encoding="utf-8",
    )
    task_inspector = task_dir / "input_files" / "tools" / "inspect_workspace_file.py"
    wrapper_inspector = project_root / "scripts" / "clawgym_nanobot" / "inspect_workspace_file.py"
    if dataset == "wsclaw" and not (destination / "tools" / "inspect_workspace_file.py").is_file() and task_inspector.is_file() and wrapper_inspector.is_file():
        tools = destination / "tools"
        tools.mkdir()
        shutil.copy2(task_inspector, tools / "_inspect_workspace_file_base.py", follow_symlinks=True)
        shutil.copy2(wrapper_inspector, tools / "inspect_workspace_file.py", follow_symlinks=True)
    trajectory_dir = destination / "anchor_trajectory"
    trajectory_dir.mkdir()
    if trajectory_path and trajectory_path.is_file():
        shutil.copy2(trajectory_path, trajectory_dir / trajectory_path.name)
    (destination / "RUBRIC_SCHEMA.json").write_text(
        rubric_schema_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    prompt = rubric_prompt_path.read_text(encoding="utf-8")
    (destination / "AGENTS.md").write_text(prompt + "\n", encoding="utf-8")
    (destination / "SOUL.md").write_text(
        "You are a careful reward agent. Produce a truth-grounded rubric of atomic, objective checks and obey AGENTS.md.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "rubric_workspace_v2",
        "dataset": dataset,
        "rubric_prompt_path": str(rubric_prompt_path),
        "task_dir": str(task_dir),
        "anchor_workspace_source": str(anchor_workspace),
        "anchor_workspace_hash": tree_hash(anchor_workspace, excludes={"reward", "_judge", "sessions", "memory", ".nanobot"}),
        "trajectory_included": bool(trajectory_path and trajectory_path.is_file()),
        "anchor_evidence": evidence,
        "forbidden_inputs": ["ground-truth rubric", "reference answer", "judge workspace", "reward code"],
        "allowed_output": "rubric.json",
        "rubric_contract": ("wsclaw_full_workspace_v1" if dataset == "wsclaw"
                            else "truth_grounded_input_layer_v6"),
    }
    atomic_json(destination / "WORKSPACE_MANIFEST.json", manifest)
    assert_rubric_workspace_safe(destination)
    return manifest


def assert_rubric_workspace_safe(workspace: Path) -> None:
    forbidden_parts = {"reward", "_judge", "judge_workspace", "reference_answer"}
    violations = [
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if forbidden_parts.intersection(path.relative_to(workspace).parts)
    ]
    if violations:
        raise ValueError(f"rubric workspace leaks evaluator assets: {violations[:10]}")


def _clean_candidate(task_dir: Path, candidate_workspace: Path, root: Path) -> Path:
    clean = root / "candidate_workspace"
    materialize_candidate_evidence(
        task_dir=task_dir, candidate_workspace=candidate_workspace, destination=clean
    )
    return clean


def _shadow_ws_task(task_dir: Path, root: Path, rubric: dict[str, Any]) -> Path:
    shadow = root / "task_package"
    reset_dir(shadow)
    for name in ("data_entry.json", "task_output.json"):
        src = task_dir / name
        if src.is_file():
            shutil.copy2(src, shadow / name)
    source_inputs = (task_dir / "input_files").resolve()
    (shadow / "input_files").symlink_to(source_inputs, target_is_directory=True)
    (shadow / "reward").mkdir()
    atomic_json(shadow / "reward" / "rubric.json", compile_ws_rubric(rubric))
    return shadow


def validate_required_score_artifacts(*, project_root: Path, task_dir: Path,
                                      candidate_workspace: Path, destination: Path) -> dict[str, Any]:
    """Use the trusted GT output contract and validator before self scoring.

    Missing/broken/empty deliverables are policy failures (valid reward zero).
    Validator/contract/I/O failures propagate as scorer infrastructure failures.
    No submitted program is executed and no semantic quality is inferred.
    """
    import os
    import subprocess
    contract = read_json(task_dir / "reward" / "output_contract.json")
    required = [dict(item) for item in contract["output_files"]
                if isinstance(item, dict) and item.get("must_create", True) and item.get("path")]
    validator = project_root / "src" / "clawgym_nanobot" / "artifact_validity_gate.py"
    proc = subprocess.run(
        [os.environ["WSBENCH_PYTHON"], str(validator.resolve(strict=True)), "--check"],
        input=json.dumps({"workspace": str(candidate_workspace.resolve(strict=True)), "required": required}),
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode:
        raise RuntimeError("self-score artifact validator infrastructure failure: " + proc.stderr[-2000:])
    gate = json.loads(proc.stdout)
    if gate.get("version") != 2 or type(gate.get("passed")) is not bool:
        raise ValueError("invalid trusted artifact gate response")
    rows = gate.get("artifacts")
    if not isinstance(rows, list) or not rows or len(rows) != len(required):
        raise ValueError("incomplete trusted artifact gate response")
    for row, item in zip(rows, required):
        if row.get("path") != item["path"] or type(row.get("valid")) is not bool:
            raise ValueError("artifact gate path/validity mismatch")
        if row["valid"] and row.get("bytes") == 0:
            row.update(valid=False, reason="required_file_empty")
    gate["passed"] = all(row["valid"] for row in rows)
    gate["self_score_gate_version"] = 1
    gate["trusted_contract"] = str(task_dir / "reward" / "output_contract.json")
    gate["semantic_content_verified"] = False
    atomic_json(destination / "artifact-validity-gate.json", gate)
    return gate


def build_ws_score_workspace(
    *,
    project_root: Path,
    task_dir: Path,
    candidate_workspace: Path,
    rubric: dict[str, Any],
    destination: Path,
    score_appendix_path: Path,
) -> dict[str, Any]:
    _runtime, _tasks, ws, _rc = import_native(project_root)
    reset_dir(destination)
    artifact_gate = validate_required_score_artifacts(
        project_root=project_root, task_dir=task_dir, candidate_workspace=candidate_workspace, destination=destination,
    )
    candidate = destination / "candidate_workspace"
    link_clean_tree(candidate_workspace, candidate)
    shadow = _shadow_ws_task(task_dir, destination, rubric)
    entry = read_json(task_dir / "data_entry.json")
    outputs = ws.output_paths(task_dir)
    case_dir = destination / "native_case"
    case_dir.mkdir()
    judge_workspace, artifacts = ws.prepare_judge_workspace(
        task_dir=shadow,
        workspace=candidate,
        outputs=outputs,
        case_dir=case_dir,
        entry=entry,
        original_rubric=rubric,
    )
    (judge_workspace / "AGENTS.md").write_text(
        ws_score_instructions((judge_workspace / "CLAUDE.md").read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    # Preserve native evidence and instructions. The harness loads the same
    # instructions through AGENTS.md only in its temporary runtime.
    prompt = ws.judge_user_message(judge_workspace)
    (destination / "SCORING_PROMPT.md").write_text(prompt + "\n", encoding="utf-8")
    manifest = {
        "dataset": "wsclaw",
        "workspace": str(judge_workspace),
        "result_path": "judge_result.json",
        "artifacts": artifacts,
        "artifact_validity_gate": artifact_gate,
        "anchor_present": False,
        "feedback_present": False,
        "ground_truth_assets_present": False,
        "instructions_file": "AGENTS.md",
        "native_prompt_adaptations": ["Nanobot instructions in AGENTS.md"],
        "workspace_materialization": "native_gt_link_clean_tree_v1",
        "score_contract": "wsclaw_native_judge_v2",
        "task_dir": str(shadow),
    }
    atomic_json(destination / "SCORE_WORKSPACE_MANIFEST.json", manifest)
    assert_score_workspace_safe(judge_workspace)
    return manifest


def assert_score_workspace_safe(workspace: Path) -> None:
    forbidden_names = {"ANCHOR.md", "FEEDBACK.md", "anchor_workspace", "reference_answer"}
    violations = []
    for path in workspace.rglob("*"):
        rel = path.relative_to(workspace)
        if any(part in forbidden_names for part in rel.parts):
            violations.append(rel.as_posix())
        if "reward" in rel.parts:
            violations.append(rel.as_posix())
    if violations:
        raise ValueError(f"score workspace leaks hidden/anchor assets: {violations[:10]}")


def ws_score_instructions(native_instructions: str) -> str:
    return native_instructions + (
        "\n\n## Nanobot self-scoring adapter\n\n"
        "- Read task.md and rubric.json first. rubric.json is the complete self-rubric; "
        "use its criteria order for score keys 1, 2, ... in review/brief.json. "
        "Score only those criteria; reason and limitations are context, not extra criteria.\n"
        "- Candidate paths P mean workspace/P: Desktop/x.xlsx means workspace/Desktop/x.xlsx, "
        "and input/P means workspace/input/P. Do not duplicate the workspace/ prefix. "
        "A failed root lookup or symlink-skipping search does not prove a file is missing.\n"
        "- Use Nanobot local file tools. For source:N/output:N aliases, use exec with "
        "python tools/inspect_workspace_file.py output:1. The inspect_workspace_file "
        "tool takes workspace/... file paths rather than aliases.\n"
        "- Include every rubric score as a number from 0 to 1 and the native user_feedback "
        "field. Write judge_result.json at the judge root and return the same JSON.\n"
    )
