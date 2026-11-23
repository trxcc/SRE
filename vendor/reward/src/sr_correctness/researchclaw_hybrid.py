from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import AgentSpec
from .common import atomic_json, copy_clean_tree, read_json, read_secret, reset_dir
from .contracts import validate_rubric
from .native import import_native, materialize_candidate_evidence


SELF_REWARD_INPUT_MOUNT = "/root/.openclaw/workspace/input_files"
SELF_REWARD_SOURCE_PATH_RULE = (
    "SELF-REWARD SOURCE PATH RULE: Original public task inputs are under `input_files/`. "
    "When the user task, a source guide, or rubric text names an original input path P "
    "such as `SOURCE_GUIDE.md` or `paper_lib/...`, inspect `input_files/P`. Paths already "
    "beginning with `input_files/` and candidate output paths such as `results/...` remain unchanged.\n\n"
)


def add_self_reward_source_path_rule(prompt: str, marker: str) -> str:
    if marker not in prompt:
        raise RuntimeError(f"native ResearchClaw prompt marker is unavailable: {marker!r}")
    return prompt.replace(marker, SELF_REWARD_SOURCE_PATH_RULE + marker, 1)


def compile_researchclaw_self_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    validate_rubric(rubric)
    items = []
    for item in rubric["criteria"]:
        items.append(
            {
                "id": item["id"],
                "name": item["criterion"][:120],
                "evaluation_mode": "llm_judge",
                "check_type": "content_grounded_judge",
                "score_type": "binary",
                "weight": 1.0,
                "pass_condition": item["criterion"],
                "feedback": item["reason"],
                "common_errors": [],
                "acceptable_phrasings": [],
                "score_anchors": {
                    "0": f"Score 0 if the candidate evidence does not match: {item['expected']}",
                    "1": f"Score 1 only if the candidate evidence matches: {item['expected']}",
                },
                "judge_guidance": (
                    "Verify only against task-local evidence. Original public task inputs are under input_files/; "
                    "map an original input path P to input_files/P. Candidate-visible target paths: "
                    f"{', '.join(item['target_paths'])}. Expected fact/set: {item['expected']}. "
                    f"Rubric design reason: {item['reason']}"
                ),
            }
        )
    return {
        "aggregation": "weighted_mean",
        "source": "self_reward_generated_from_anchor_and_single_feedback",
        "items": items,
        "limitations": rubric["limitations"],
    }


def _rewrite_paths(value: Any, old: Path, new: Path) -> Any:
    old_text = str(old)
    new_text = str(new)
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old_text, new_text)
    return value


def _run_bounded(command: list[str], *, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise TimeoutError(f"ResearchClaw hybrid self-reward exceeded {timeout}s")
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _native_env(spec: AgentSpec, project_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    # ResearchClaw self-reward keeps thinking enabled for both native triage
    # and the targeted agentic verifier.  Qwen can otherwise consume the
    # former 8K budget in reasoning and finish with `length` before emitting
    # the required JSON payload.
    judge_max_tokens = max(32_768, int(spec.max_tokens))
    judge_base_url = spec.base_url.rstrip("/")
    if not judge_base_url.endswith("/v1"):
        judge_base_url += "/v1"
    nanobot_root = Path(env.get("NANOBOT_ROOT") or project_root / "vendor" / "clawgym_nanobot" / "nanobot")
    nanobot_extra = [str(nanobot_root)]
    inherited_nanobot_extra = env.get("NANOBOT_EXTRA_PYTHONPATH", "").strip()
    if inherited_nanobot_extra:
        nanobot_extra.extend(inherited_nanobot_extra.split(os.pathsep))
    pythonpath = [str(project_root), str(project_root / "src"), *nanobot_extra]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "RESEARCHCLAW_SCORE_MODE": "hybrid",
            "RESEARCHCLAW_LLMJUDGE_BACKEND": "hybrid_agentic",
            "RESEARCHCLAW_CODE_WEIGHT": "0.0",
            "RESEARCHCLAW_LLMJUDGE_WEIGHT": "1.0",
            "RESEARCHCLAW_JUDGE_BASE_URL": judge_base_url,
            "RESEARCHCLAW_JUDGE_API_KEY": read_secret(spec.api_key_file),
            "RESEARCHCLAW_JUDGE_MODEL": spec.model,
            "RESEARCHCLAW_JUDGE_TEMPERATURE": str(spec.temperature),
            "RESEARCHCLAW_JUDGE_MAX_TOKENS": str(judge_max_tokens),
            "RESEARCHCLAW_JUDGE_TIMEOUT": str(spec.timeout_sec),
            "RESEARCHCLAW_JUDGE_MAX_RETRIES": ("0" if os.environ.get("CLAWGYM_TRAIN_SCORE_FAIL_FAST") == "1" else "1"),
            "RESEARCHCLAW_JUDGE_RETRY_BACKOFF_SEC": "1",
            "RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER": "0",
            "RESEARCHCLAW_DIRECT_JUDGE_ENABLE_THINKING": "1",
            "RESEARCHCLAW_AGENTIC_JUDGE_ENABLE_THINKING": "1",
            "RESEARCHCLAW_AGENTIC_JUDGE_SHOW_REASONING": "1",
            "RESEARCHCLAW_AGENTIC_JUDGE_STRICT_SCORE_ENUM": "1",
            "RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT": str(spec.timeout_sec),
            "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS": str(judge_max_tokens),
            "RESEARCHCLAW_AGENTIC_JUDGE_CONTEXT_WINDOW": str(spec.context_window),
            "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOOL_ITERATIONS": str(spec.max_tool_calls),
            "RESEARCHCLAW_AGENTIC_JUDGE_EXEC_ENABLE": "1",
            "RESEARCHCLAW_DISABLE_NATURAL_USER_FEEDBACK": "1",
            # The native ResearchClaw targeted verifier launches Nanobot in a
            # fresh subprocess.  The vendored package is intentionally not
            # installed system-wide in the CPU image, so make the same local
            # package import explicit for that subprocess.
            "NANOBOT_IMPORT_LOCAL_PACKAGE": "1",
            "NANOBOT_ROOT": str(nanobot_root),
            "NANOBOT_EXTRA_PYTHONPATH": os.pathsep.join(dict.fromkeys(nanobot_extra)),
            "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
            "NANOBOT_STRIP_REASONING_CONTEXT": "0",
            "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY": "preserve",
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
    return env


def run_researchclaw_hybrid_self_score(
    *,
    project_root: Path,
    task_dir: Path,
    candidate_workspace: Path,
    rubric_path: Path,
    run_dir: Path,
    slot_id: str,
    spec: AgentSpec,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    solver = run_dir / "solver_workspace"
    # Recovery snapshots retain the candidate-authored deliverables but omit
    # immutable task inputs.  Rebuild the same public evidence view used by
    # the rubric agent before entering the native hybrid judge.  The task's
    # private reward directory is added later only inside the native worker,
    # where its GT answer is removed and rubric.json is replaced.
    materialize_candidate_evidence(
        task_dir=task_dir,
        candidate_workspace=candidate_workspace,
        destination=solver,
    )
    canonical_rubric = validate_rubric(read_json(rubric_path))
    native_self_rubric = compile_researchclaw_self_rubric(canonical_rubric)
    native_rubric_path = run_dir / "self_rubric.native.json"
    atomic_json(native_rubric_path, native_self_rubric)
    return _run_researchclaw_hybrid_native_score(
        project_root=project_root,
        task_dir=task_dir,
        solver=solver,
        native_rubric_path=native_rubric_path,
        run_dir=run_dir,
        slot_id=slot_id,
        spec=spec,
        rubric_source="generated_self_rubric",
    )


def run_researchclaw_hybrid_gt_rubric_score(
    *,
    project_root: Path,
    task_dir: Path,
    candidate_workspace: Path,
    gt_rubric_path: Path,
    run_dir: Path,
    slot_id: str,
    spec: AgentSpec,
) -> dict[str, Any]:
    """Run the self-reward judge pipeline with the original task rubric.

    This is an attribution ablation: candidate evidence, direct triage,
    conditional Nanobot verification, thinking settings, and hidden-answer
    exclusion are identical to self scoring.  Only the rubric packet changes.
    """
    run_dir.mkdir(parents=True, exist_ok=False)
    solver = run_dir / "solver_workspace"
    materialize_candidate_evidence(
        task_dir=task_dir,
        candidate_workspace=candidate_workspace,
        destination=solver,
    )
    native_rubric = read_json(gt_rubric_path)
    if not isinstance(native_rubric, dict) or not isinstance(native_rubric.get("items"), list):
        raise ValueError(f"invalid native GT rubric: {gt_rubric_path}")
    if not native_rubric["items"]:
        raise ValueError(f"native GT rubric has no items: {gt_rubric_path}")
    copied_rubric = run_dir / "gt_rubric.native.json"
    atomic_json(copied_rubric, native_rubric)
    return _run_researchclaw_hybrid_native_score(
        project_root=project_root,
        task_dir=task_dir,
        solver=solver,
        native_rubric_path=copied_rubric,
        run_dir=run_dir,
        slot_id=slot_id,
        spec=spec,
        rubric_source="ground_truth_rubric_ablation",
    )


def _run_researchclaw_hybrid_native_score(
    *,
    project_root: Path,
    task_dir: Path,
    solver: Path,
    native_rubric_path: Path,
    run_dir: Path,
    slot_id: str,
    spec: AgentSpec,
    rubric_source: str,
) -> dict[str, Any]:
    worker = Path(__file__).with_name("researchclaw_hybrid_worker.py")
    local_tmp = Path(os.environ.get("SELF_REWARD_LOCAL_TMP", "/tmp"))
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="sr-rc-hybrid-", dir=local_tmp) as raw_local:
        local_root = Path(raw_local)
        local_run = local_root / "native_run"
        command = [
            sys.executable,
            str(worker),
            "--project-root",
            str(project_root),
            "--task-dir",
            str(task_dir),
            "--solver-workspace",
            str(solver),
            "--run-dir",
            str(local_run),
            "--native-self-rubric",
            str(native_rubric_path),
        ]
        proc = _run_bounded(
            command,
            env=_native_env(spec, project_root),
            timeout=spec.timeout_sec * 2 + 120,
        )
        (run_dir / "worker.stdout").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "worker.stderr").write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode != 0:
            if local_run.exists():
                failed_native = run_dir / "native_run_failed"
                if failed_native.exists():
                    shutil.rmtree(failed_native)
                shutil.copytree(local_run, failed_native, symlinks=True)
            raise RuntimeError(f"native ResearchClaw hybrid worker exited with {proc.returncode}")

        payload_path = local_run / "researchclaw_reward.json"
        if not payload_path.is_file():
            raise FileNotFoundError("native ResearchClaw hybrid flow did not write researchclaw_reward.json")
        payload = read_json(payload_path)
        state_dir = local_run / "researchclaw_agentic_judge_state"
        if state_dir.exists():
            shutil.rmtree(state_dir)
        persisted = run_dir / "native_run"
        shutil.copytree(local_run, persisted, symlinks=True)
        payload = _rewrite_paths(payload, local_run, persisted)
        atomic_json(persisted / "researchclaw_reward.json", payload)

    score = payload.get("final_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("native ResearchClaw hybrid flow returned no numeric final_score")
    judge = payload.get("llmjudge_reward") if isinstance(payload.get("llmjudge_reward"), dict) else {}
    if judge.get("backend") != "hybrid_agentic":
        raise ValueError(f"expected hybrid_agentic backend, got {judge.get('backend')!r}")
    if judge.get("reference_answer_included"):
        raise ValueError("native ResearchClaw self-reward leaked the hidden reference answer")
    result = {
        "schema_version": "researchclaw_native_hybrid_self_reward_v1",
        "slot_id": slot_id,
        "self_reward": float(score),
        "backend": "hybrid_agentic",
        "rubric_source": rubric_source,
        "hidden_reference_answer_included": False,
        "direct_model": spec.model,
        "direct_thinking": True,
        "agentic_model": spec.model,
        "agentic_thinking": True,
        "temperature": spec.temperature,
        "duration_sec": time.monotonic() - started,
        "native_run": str(run_dir / "native_run"),
        "native_result": payload,
    }
    atomic_json(run_dir / "accepted.json", result)
    return result


def build_researchclaw_hybrid_preview(
    *,
    project_root: Path,
    task_dir: Path,
    candidate_workspace: Path,
    rubric: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    _runtime, tasks, _ws, rc = import_native(project_root)
    reset_dir(destination)
    solver = destination / "solver_workspace"
    materialize_candidate_evidence(
        task_dir=task_dir,
        candidate_workspace=candidate_workspace,
        destination=solver,
    )
    task = replace(tasks.load_task(task_dir), input_mount_dir=SELF_REWARD_INPUT_MOUNT)
    tasks.copy_reward_assets(task, solver)
    reward_dir = solver / "reward"
    config = read_json(reward_dir / "eval_config.json")
    native_rubric = compile_researchclaw_self_rubric(rubric)
    atomic_json(reward_dir / "llmjudge" / "rubric.json", native_rubric)
    for name in ("answer.md", "answer.json"):
        (reward_dir / "llmjudge" / name).unlink(missing_ok=True)
    rules = rc._rubric_to_rules(native_rubric, rc._required_output_path(task, config))
    output_paths = rc._judge_output_paths(task, reward_dir, rules, config)
    targets = rc._read_judge_target_files(solver, output_paths, 24_000)
    reference = None
    prompt = rc._build_triage_judge_prompt(
        user_task=task.user_query,
        rules=rules,
        target_files=targets,
        reference_answer=reference,
    )
    prompt = add_self_reward_source_path_rule(prompt, "## User Task\n")
    (destination / "TRIAGE_PROMPT.md").write_text(prompt + "\n", encoding="utf-8")
    agentic_run = destination / "agentic_stage_preview"
    preview_rules = rules[:1]
    preview_rule_ids = rc._rule_ids(preview_rules)
    paths = rc._prepare_agentic_judge_workspace(
        task=task,
        reward_dir=reward_dir,
        solver_workspace_dir=solver,
        run_dir=agentic_run,
        rubric=native_rubric,
        rules=preview_rules,
        config=config,
        max_chars=24_000,
        rule_ids=preview_rule_ids or None,
    )
    agentic_prompt = rc._build_targeted_agentic_judge_prompt(
        rules=preview_rules,
        rule_ids=preview_rule_ids,
        target_files=targets,
        evidence_requests={},
        reference_answer_included=False,
    )
    agentic_prompt = add_self_reward_source_path_rule(
        agentic_prompt, "Use tools only to inspect task-local source files"
    )
    (destination / "AGENTIC_PROMPT.md").write_text(agentic_prompt + "\n", encoding="utf-8")
    manifest = {
        "dataset": "researchclaw",
        "backend": "hybrid_agentic",
        "rubric_source": "generated_self_rubric",
        "rubric_path": str(reward_dir / "llmjudge" / "rubric.json"),
        "hidden_reference_answer_included": False,
        "direct_stage_prompt": str(destination / "TRIAGE_PROMPT.md"),
        "solver_workspace": str(solver),
        "agentic_stage_workspace": str(paths["workspace"]),
        "agentic_stage_prompt": str(destination / "AGENTIC_PROMPT.md"),
        "model_calls_made": 0,
    }
    atomic_json(destination / "HYBRID_PREVIEW_MANIFEST.json", manifest)
    return manifest
