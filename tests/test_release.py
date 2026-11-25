"""CPU regression, contract, and release-hygiene tests for DEAL-WS & DEAL-RS."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts"),
    str(ROOT / "vendor/prorl/src"),
    str(ROOT / "vendor/reward/src"),
]

from configure_run import BENCHMARK_SPEC, configure, validate_cursor
from runtime.aligned_eval_recovery import apply_failure_zero
from runtime.portable_contract import validate_recovery_args


def _mock_args_from_env(env: dict[str, str], manifest: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        train_backend=env["TRAIN_BACKEND"],
        advantage_estimator=env["ADVANTAGE_ESTIMATOR"],
        num_rollout=int(env["NUM_ROLLOUT"]),
        rollout_batch_size=int(env["ROLLOUT_BATCH_SIZE"]),
        n_samples_per_prompt=int(env["N_SAMPLES_PER_PROMPT"]),
        global_batch_size=int(env["GLOBAL_BATCH_SIZE"]),
        num_steps_per_rollout=int(env["NUM_STEPS_PER_ROLLOUT"]),
        normalize_advantages=env["NORMALIZE_ADVANTAGES"] == "1",
        actor_num_nodes=int(env["ACTOR_NUM_NODES"]),
        actor_num_gpus_per_node=int(env["ACTOR_GPUS_PER_NODE"]),
        rollout_num_gpus=int(env["ROLLOUT_NUM_GPUS"]),
        num_gpus_per_node=int(env["NUM_GPUS_PER_NODE"]),
        tensor_model_parallel_size=int(env["TP_SIZE"]),
        context_parallel_size=int(env["CP_SIZE"]),
        pipeline_model_parallel_size=int(env["PP_SIZE"]),
        rollout_num_gpus_per_engine=int(env["ROLLOUT_GPUS"]),
        no_load_optim=env["NO_LOAD_OPTIM"] == "1",
        no_load_rng=True,
        seq_length=int(env["TRAIN_SEQ_LENGTH"]),
        max_tokens_per_gpu=int(env["MAX_TOKENS_PER_GPU"]),
        eval_interval=int(env["OPENCLAW_EVAL_INTERVAL"]),
        rollout_max_context_len=int(env["SGLANG_CONTEXT_LENGTH"]),
        rollout_max_response_len=int(env["ROLLOUT_MAX_RESPONSE_LEN"]),
        eval_max_context_len=int(env["EVAL_MAX_CONTEXT_LEN"]),
        eval_max_response_len=int(env["EVAL_MAX_RESPONSE_LEN"]),
        data_source_path=env["POLAR_DATA_SOURCE_PATH"],
        eval_function_path=env["EVAL_FUNCTION_PATH"],
        lr=float(env["LR"]),
        rollout_temperature=float(env["ROLLOUT_TEMPERATURE"]),
        rollout_top_p=float(env["ROLLOUT_TOP_P"]),
        start_rollout_id=int(env["START_ROLLOUT_ID"]),
        skip_eval_before_train=env["SKIP_INITIAL_EVAL"] == "1",
        hf_checkpoint=env["HF_CHECKPOINT"],
        ref_load=env["REF_LOAD"],
        use_tis=env["USE_TIS"] == "1",
        use_kl_loss=env["ENABLE_KL_LOSS"] == "1",
        kl_loss_coef=float(env["KL_LOSS_COEF"]),
        kl_coef=float(env["KL_COEF"]),
    )


class DealReleaseTests(unittest.TestCase):
    def test_recipes_unified_4plus4_and_1gt7sr(self):
        for benchmark, spec in BENCHMARK_SPEC.items():
            for model_name, recipe_rel in spec["recipes"].items():
                recipe = json.loads((ROOT / recipe_rel).read_text(encoding="utf-8"))
                self.assertEqual(recipe["ACTOR_GPUS_PER_NODE"], "4", recipe_rel)
                self.assertEqual(recipe["ROLLOUT_NUM_GPUS"], "4", recipe_rel)
                self.assertEqual(recipe["NUM_GPUS_PER_NODE"], "8", recipe_rel)
                self.assertEqual(recipe["TP_SIZE"], "4", recipe_rel)
                self.assertEqual(recipe["ROLLOUT_GPUS"], "1", recipe_rel)
                self.assertEqual(recipe["POLAR_EXPECTED_ROLLOUT_ENGINES"], "4", recipe_rel)
                self.assertEqual(recipe["CLAWGYM_SELF_REWARD_GROUP_SIZE"], "8", recipe_rel)
                self.assertEqual(recipe["N_SAMPLES_PER_PROMPT"], "8", recipe_rel)
                self.assertEqual(recipe["ROLLOUT_BATCH_SIZE"], "16", recipe_rel)
                self.assertEqual(recipe["GLOBAL_BATCH_SIZE"], "128", recipe_rel)
                self.assertEqual(recipe["CLAWGYM_ARTIFACT_VALIDITY_GATE"], "1", recipe_rel)
                expected_mc = "1" if benchmark == "deal-ws" else "0"
                self.assertEqual(recipe["NANOBOT_MICROCOMPACT_ENABLED"], expected_mc, recipe_rel)
                self.assertEqual(int(recipe["NUM_ROLLOUT"]), spec["num_rollout"], recipe_rel)

    def test_configure_and_portable_contract_all_four_profiles(self):
        for benchmark in ("deal-rs", "deal-ws"):
            for model_name in ("gemma4-e4b-it", "qwen3.5-9b"):
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    base_model = tmp_path / "model"
                    base_model.mkdir()
                    (base_model / "config.json").write_text("{}\n", encoding="utf-8")

                    data_dir = tmp_path / "data"
                    (data_dir / "train").mkdir(parents=True)
                    (data_dir / "eval").mkdir(parents=True)
                    (data_dir / "DATA_READY.json").write_text(
                        json.dumps({"train_dir": str(data_dir / "train"), "eval_dir": str(data_dir / "eval")}),
                        encoding="utf-8",
                    )

                    gt_key = tmp_path / "gt.key"
                    sr_key = tmp_path / "sr.key"
                    for kf in (gt_key, sr_key):
                        kf.write_text("dummy-test-key\n", encoding="utf-8")
                        kf.chmod(0o600)

                    cfg_file = tmp_path / "local.env"
                    cfg_file.write_text(
                        "\n".join([
                            f"BENCHMARK={benchmark}",
                            f"MODEL={model_name}",
                            f"BASE_MODEL={base_model}",
                            f"DATA_DIR={data_dir}",
                            f"RUN_DIR={tmp_path / 'runs'}",
                            "TRAIN_GPUS=0,1,2,3",
                            "ROLLOUT_GPUS=4,5,6,7",
                            "SELF_REWARD_BASE_URL=http://127.0.0.1:29040/v1",
                            f"SELF_REWARD_MODEL={model_name}",
                            f"SELF_REWARD_API_KEY_FILE={sr_key}",
                            "GT_BASE_URL=https://example.invalid/v1",
                            "GT_MODEL=gemma-4-31B-it",
                            f"GT_API_KEY_FILE={gt_key}",
                            "HOST_IP=127.0.0.1",
                        ]) + "\n",
                        encoding="utf-8",
                    )

                    work = configure(cfg_file, attempt="test-attempt")
                    env = json.loads((work / "private/environment.json").read_text(encoding="utf-8"))
                    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
                    sr_cfg = json.loads((work / "self-reward.json").read_text(encoding="utf-8"))
                    reward_cfg = json.loads((work / "private/reward-service.json").read_text(encoding="utf-8"))

                    self.assertEqual(sr_cfg["mode"], "self_reward_7plus1_train7")
                    self.assertEqual(sr_cfg["candidate_count"], 8)
                    self.assertEqual(reward_cfg["candidate_count"], 8)
                    self.assertEqual(reward_cfg["gt_scope"], "anchor_only")
                    self.assertFalse(reward_cfg["gt_all_candidates_every_step"])
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
                    if benchmark == "deal-ws":
                        self.assertTrue(manifest["hardgate"])
                        self.assertEqual(env["CLAWGYM_ARTIFACT_VALIDITY_GATE"], "1")
                        self.assertTrue((work / "private/ws-gt-judge.json").is_file())
                        self.assertEqual((work / "private/ws-gt-judge.json").stat().st_mode & 0o777, 0o600)

                    old_env = os.environ.copy()
                    try:
                        os.environ.update(env)
                        summary = validate_recovery_args(_mock_args_from_env(env, manifest))
                        self.assertTrue(summary["validated"])
                        self.assertEqual(summary["candidate_count"], 8)
                        self.assertEqual(summary["topology"], "4 train + 4 rollout")
                    finally:
                        os.environ.clear()
                        os.environ.update(old_env)

    def test_self_reward_locked_to_1gt_7sr_no_ablations(self):
        tree = ast.parse((ROOT / "vendor/prorl/src/clawgym_nanobot/self_reward.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "GROUP_SIZE_BY_MODE":
                        modes = ast.literal_eval(node.value)
                        self.assertEqual(modes, {"self_reward_7plus1_train7": 8})
                        return
        self.fail("GROUP_SIZE_BY_MODE not found in self_reward.py")

    def test_committed_cursor_math_rs_and_ws(self):
        # DEAL-RS: 1800 tasks, 113 rollouts, 8 candidates/task
        self.assertEqual(validate_cursor(112, dict(sample_offset=1800, sample_group_index=1800, sample_index=1800 * 8, epoch_id=0), benchmark="deal-rs"), 1800)
        with self.assertRaises(AssertionError):
            validate_cursor(112, dict(sample_offset=1800, sample_group_index=1800, sample_index=1800 * 6, epoch_id=0), benchmark="deal-rs")
        # DEAL-WS: 2000 tasks, 125 rollouts, 8 candidates/task
        self.assertEqual(validate_cursor(124, dict(sample_offset=2000, sample_group_index=2000, sample_index=2000 * 8, epoch_id=0), benchmark="deal-ws"), 2000)

    def test_eval_failure_zero_denominator_200(self):
        for failures in (0, 5, 200):
            z = {k: [1.0] * 200 for k in ("rewards", "raw_rewards", "code_verifier_rewards", "raw_code_verifier_rewards")}
            z.update(raw_skipped=[i < failures for i in range(200)], n_collected=200 - failures, n_skipped=failures)
            out = types.SimpleNamespace(data={"eval-200": z}, metrics={})
            apply_failure_zero(out)
            self.assertAlmostEqual(out.metrics["eval/eval-200-reward_avg"], (200 - failures) / 200)
            self.assertEqual(z["n_scored"], 200)
            self.assertEqual(z["n_failed_scored_zero"], failures)

    def test_ws_hardgate_zero_reward_on_invalid_artifact(self):
        from sr_correctness.wsclaw_scoring import artifact_gate_zero

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "slot"
            gate = {
                "self_score_gate_version": 1,
                "passed": False,
                "artifacts": [{"path": "Desktop/report.docx", "format": "docx", "valid": False, "reason": "required_file_missing"}],
            }
            rubric = {"criteria": [{"id": "c1", "weight": 1.0}]}
            result = artifact_gate_zero(gate=gate, rubric=rubric, slot_id="s1", session_id="sess1", run_dir=run_dir)
            self.assertEqual(result["value"]["weighted_score"], 0.0)
            self.assertTrue(result["value"]["deterministic_policy_failure"])
            self.assertFalse(result["value"]["judge_executed"])

    def test_release_hygiene_no_secrets_intranet_or_machine_paths(self):
        patterns = [
            re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"),
            re.compile(r"(claw-hf-backup|claw-models-backup|claw-data-logs-backup)"),
            re.compile(r"\b100\.(9[0-9]|1[0-2][0-9])\.\d{1,3}\.\d{1,3}\b"),
            re.compile(r"(pjlab\.org\.cn|test-lyz|dev-trx|vwx4f|tzttx|hng6g|brcf9|llmfrontier|ailab-p1|petrelfs)"),
            re.compile(r"(/mnt/shared-storage|/trx-data|/data/tanrongxi|/oss/bucket-|/oss/moe-|/Users/tanrongxi)"),
            re.compile(r"\b(tanrongxi|liyizhuo)\b"),
        ]
        self_file = Path(__file__).resolve()
        violations = []
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or path.resolve() == self_file:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                if "--mask-offpolicy-in-partial-rollout" in line:
                    continue
                for pat in patterns:
                    if pat.search(line):
                        violations.append(f"{path.relative_to(ROOT)}:{ln}: {line.strip()[:120]}")
        self.assertEqual(violations, [], f"Found hygiene violations:\n" + "\n".join(violations[:20]))

    def test_shell_scripts_syntax(self):
        for sh in sorted(ROOT.rglob("*.sh")):
            proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, f"bash -n failed for {sh}: {proc.stderr}")

    def test_ws_workspace_scaffold_installer_has_all_templates(self):
        sys.path.insert(0, str(ROOT / "vendor/prorl"))
        from scripts.clawgym_nanobot import wsclaw_eval_only

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            wsclaw_eval_only.install_workspace_scaffold(ws)
            for rel in (
                "AGENTS.md",
                "SOUL.md",
                "USER.md",
                "TOOLS.md",
                "HEARTBEAT.md",
                ".gitignore",
                "memory/MEMORY.md",
            ):
                target = ws / rel
                self.assertTrue(target.is_file(), f"missing installed template: {rel}")
                self.assertGreater(target.stat().st_size, 0, f"empty installed template: {rel}")

    def test_no_git_ignored_vendor_sources(self):
        if not (ROOT / ".git").exists():
            return
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--ignored", "-s", "vendor"],
            capture_output=True,
            text=True,
            check=True,
        )
        ignored = [
            line[3:].strip()
            for line in proc.stdout.splitlines()
            if line.startswith("!!") and not line.strip().endswith("__pycache__/")
        ]
        self.assertEqual(ignored, [], f"Git-ignored files detected under vendor/: {ignored[:10]}")

    def test_verify_env_cpu_mode(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_env.py"), "--mode", "cpu", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"], payload)

    def test_workspace_demos_verify(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_workspace_demos.py"), "--verify", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["verification"]["ok"], payload)
        self.assertGreaterEqual(payload["verification"]["checks_passed"], 21)

    def test_nanobot_system_templates_match_ws_eval_contract(self):
        identity = (ROOT / "vendor/nanobot/nanobot/templates/agent/identity.md").read_text(encoding="utf-8")
        tool_contract = (ROOT / "vendor/nanobot/nanobot/templates/agent/tool_contract.md").read_text(encoding="utf-8")
        self.assertIn(
            "For a requested file deliverable, save it in the workspace and report its path in your final answer.",
            identity,
        )
        self.assertNotIn("call 'message'", identity)
        self.assertNotIn("## Messaging and Media", tool_contract)
        self.assertIn("register `HEARTBEAT.md` as a cron job", tool_contract)

    def test_nanobot_microcompact_env_toggle(self):
        runner_path = ROOT / "vendor/nanobot/nanobot/agent/runner.py"
        try:
            sys.path.insert(0, str(ROOT / "vendor/nanobot"))
            from nanobot.agent.runner import AgentRunner

            microcompact = AgentRunner._microcompact
        except ImportError:
            # Bare CPU/dev environments (< Python 3.11 or without loguru): execute
            # the exact _microcompact function & constants parsed from runner.py.
            source = runner_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(runner_path))
            ns: dict = {"os": os, "Any": object}
            keep_names = {"_MICROCOMPACT_KEEP_RECENT", "_MICROCOMPACT_MIN_CHARS", "_COMPACTABLE_TOOLS"}
            nodes = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in keep_names for t in node.targets
                ):
                    nodes.append(node)
                elif isinstance(node, ast.ClassDef) and node.name == "AgentRunner":
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "_microcompact":
                            item.decorator_list = []
                            nodes.append(item)
            mod = ast.Module(body=nodes, type_ignores=[])
            exec(compile(mod, str(runner_path), "exec"), ns)
            microcompact = ns["_microcompact"]

        long_payload = "x" * 800
        messages = [
            {"role": "tool", "name": "read_file", "content": long_payload}
            for _ in range(12)
        ]
        old = os.environ.get("NANOBOT_MICROCOMPACT_ENABLED")
        try:
            os.environ["NANOBOT_MICROCOMPACT_ENABLED"] = "1"
            compacted = microcompact(messages)
            self.assertEqual(compacted[0]["content"], "[read_file result omitted from context]")
            self.assertEqual(compacted[1]["content"], "[read_file result omitted from context]")
            self.assertEqual(compacted[2]["content"], long_payload)

            os.environ["NANOBOT_MICROCOMPACT_ENABLED"] = "0"
            untouched = microcompact(messages)
            self.assertEqual(untouched[0]["content"], long_payload)
            self.assertEqual(untouched[1]["content"], long_payload)
        finally:
            if old is None:
                os.environ.pop("NANOBOT_MICROCOMPACT_ENABLED", None)
            else:
                os.environ["NANOBOT_MICROCOMPACT_ENABLED"] = old


if __name__ == "__main__":
    unittest.main()
