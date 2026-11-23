from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .common import atomic_json, read_json, reset_dir
from .contracts import validate_rubric
from .native import build_rubric_workspace, build_ws_score_workspace
from .researchclaw_hybrid import build_researchclaw_hybrid_preview


def build_preview(*, project_root: Path, manifest_path: Path, destination: Path, package_root: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    reset_dir(destination)
    rubric = validate_rubric(read_json(Path(manifest["preview_rubric"])))
    rubric_workspace = destination / "rubric_agent_workspace"
    build_rubric_workspace(
        project_root=project_root,
        dataset=str(manifest["dataset"]),
        destination=rubric_workspace,
        task_dir=Path(manifest["task_dir"]),
        anchor_workspace=Path(manifest["anchor_workspace"]),
        anchor_final_message=str(manifest["anchor_final_message"]),
        anchor_feedback=str(manifest["anchor_feedback"]),
        anchor_score=float(manifest["anchor_score"]),
        rubric_prompt_path=package_root / "prompts" / "rubric_agent.md",
        rubric_schema_path=package_root / "schemas" / "rubric.schema.json",
        trajectory_path=Path(manifest["anchor_trajectory"]) if manifest.get("anchor_trajectory") else None,
    )
    shutil.copy2(Path(manifest["preview_rubric"]), rubric_workspace / "rubric.preview.json")
    score_root = destination / "self_score_workspace"
    if manifest["dataset"] == "wsclaw":
        score_manifest = build_ws_score_workspace(
            project_root=project_root,
            task_dir=Path(manifest["task_dir"]),
            candidate_workspace=Path(manifest["candidate_workspace"]),
            rubric=rubric,
            destination=score_root,
            score_appendix_path=package_root / "prompts" / "self_score_appendix.md",
        )
        rubric_workspace_value: str | None = str(rubric_workspace)
        score_workspace_value = score_manifest["workspace"]
        score_prompt_value = str(score_root / "SCORING_PROMPT.md")
        fixture_rubric_value = str(rubric_workspace / "rubric.preview.json")
        warning = "The fixture rubric is hand-authored only to display the workspace contract; it is not an evaluation result."
    else:
        score_manifest = build_researchclaw_hybrid_preview(
            project_root=project_root,
            task_dir=Path(manifest["task_dir"]),
            candidate_workspace=Path(manifest["candidate_workspace"]),
            rubric=rubric,
            destination=score_root,
        )
        rubric_workspace_value = str(rubric_workspace)
        score_workspace_value = score_manifest["solver_workspace"]
        score_prompt_value = score_manifest["direct_stage_prompt"]
        fixture_rubric_value = str(rubric_workspace / "rubric.preview.json")
        warning = "ResearchClaw preview uses a generated-self-rubric fixture and excludes hidden reference answers; no model call was made."
    record = {
        "preview_only": True,
        "model_calls_made": 0,
        "dataset": manifest["dataset"],
        "task_dir": manifest["task_dir"],
        "rubric_workspace": rubric_workspace_value,
        "score_workspace": score_workspace_value,
        "score_prompt": score_prompt_value,
        "fixture_rubric": fixture_rubric_value,
        "warning": warning,
    }
    atomic_json(destination / "PREVIEW_ONLY.json", record)
    return record
