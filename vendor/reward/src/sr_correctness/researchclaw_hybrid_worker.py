from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--solver-workspace", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--native-self-rubric", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    for value in (project_root, project_root / "src"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))

    from clawgym_nanobot import researchclaw_reward, tasks
    from sr_correctness.researchclaw_hybrid import (
        SELF_REWARD_INPUT_MOUNT,
        add_self_reward_source_path_rule,
    )

    native_triage_prompt = researchclaw_reward._build_triage_judge_prompt

    def strict_complete_triage_prompt(**kwargs):
        prompt = native_triage_prompt(**kwargs)
        requirement = (
            "STRICT COMPLETENESS REQUIREMENT: Before returning, enumerate every criterion id in the rubric packet. "
            "Each criterion must either have a numeric allowed anchor in `scores`, or appear in `needs_evidence` "
            "with a matching `evidence_requests` entry. Never omit a criterion, use null, or replace a numeric score "
            "with prose. Recheck this invariant immediately before emitting the JSON object.\n\n"
        )
        marker = "## User Task\n"
        if marker not in prompt:
            raise RuntimeError("native ResearchClaw triage prompt marker is unavailable")
        prompt = prompt.replace(marker, requirement + marker, 1)
        return add_self_reward_source_path_rule(prompt, marker)

    native_agentic_prompt = researchclaw_reward._build_agentic_judge_prompt
    native_targeted_prompt = researchclaw_reward._build_targeted_agentic_judge_prompt

    def add_tool_budget(prompt: str) -> str:
        return (
            prompt
            + "\n\nTool budget: You have at most 30 tool calls in this scoring session. "
            "Write the required result file as soon as the requested evidence is sufficient."
        )

    def input_layer_agentic_prompt(**kwargs):
        return add_tool_budget(add_self_reward_source_path_rule(
            native_agentic_prompt(**kwargs), "Use local tools to inspect files"
        ))

    def input_layer_targeted_prompt(**kwargs):
        return add_tool_budget(add_self_reward_source_path_rule(
            native_targeted_prompt(**kwargs),
            "Use tools only to inspect task-local source files",
        ))

    # Keep the native hybrid-agentic judge flow intact and add only the
    # self-reward-specific output-completeness instruction requested here.
    researchclaw_reward._build_triage_judge_prompt = strict_complete_triage_prompt
    researchclaw_reward._build_agentic_judge_prompt = input_layer_agentic_prompt
    researchclaw_reward._build_targeted_agentic_judge_prompt = input_layer_targeted_prompt

    task = replace(
        tasks.load_task(Path(args.task_dir)),
        input_mount_dir=SELF_REWARD_INPUT_MOUNT,
    )
    solver_workspace = Path(args.solver_workspace)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    tasks.copy_reward_assets(task, solver_workspace)
    llmjudge_dir = solver_workspace / "reward" / "llmjudge"
    for name in ("answer.md", "answer.json"):
        (llmjudge_dir / name).unlink(missing_ok=True)
    (llmjudge_dir / "rubric.json").write_text(
        Path(args.native_self_rubric).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    score = researchclaw_reward.run(task, solver_workspace, run_dir)
    print(json.dumps({"score": float(score)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
