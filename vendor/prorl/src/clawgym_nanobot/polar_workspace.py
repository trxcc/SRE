"""Prepare a ClawGym task workspace inside a Polar runtime session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import tasks


def prepare_task_workspace(task_dir: Path, workspace: Path) -> None:
    """Build the task workspace used by the selected ClawGym harness."""
    task = tasks.load_task(task_dir)
    entry = _load_entry(task_dir)
    metadata = entry.get("metadata") or {}
    if metadata.get("workspace_root"):
        # WSClaw relies on persona-specific scaffold files and a complete source
        # workspace. Keep this path identical to the eval-only harness.
        from scripts.clawgym_nanobot import wsclaw_eval_only

        prepared, _entry, _outputs, _source_root = wsclaw_eval_only.prepare_workspace(
            task_dir,
            workspace.parent,
            containerized=True,
        )
        if prepared != workspace:
            raise RuntimeError(f"unexpected WSClaw workspace path: {prepared}")
        wsclaw_eval_only.install_runtime_python(
            workspace,
            wsclaw_eval_only.DEFAULT_WSBENCH_PYTHON,
        )
        return
    tasks.prepare_workspace(task, workspace)


def validate_task_workspace(task_dir: Path) -> None:
    """Validate task-local inputs required to construct a runtime workspace."""
    tasks.load_task(task_dir)
    entry = _load_entry(task_dir)
    metadata = entry.get("metadata") or {}
    if metadata.get("workspace_root"):
        from scripts.clawgym_nanobot import wsclaw_eval_only

        wsclaw_eval_only.required_sources(entry)
        wsclaw_eval_only.output_paths(task_dir)


def _load_entry(task_dir: Path) -> dict[str, Any]:
    value = json.loads((task_dir / "data_entry.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"task data_entry.json must be an object: {task_dir}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    prepare_task_workspace(Path(args.task_dir), Path(args.workspace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
