#!/usr/bin/env python3
"""Verify DEAL-WS & DEAL-RS repository integrity, runtime environment, GPU layout, and API endpoints.

Examples:
  # Run CPU-safe checks on any machine (including clean git clone check)
  python3 scripts/verify_env.py --mode cpu

  # Run full Linux 8-GPU environment verification (4 train + 4 rollout)
  python3 scripts/verify_env.py --mode all

  # Also probe live frozen self-reward and external GT judge endpoints
  python3 scripts/verify_env.py --mode all \
      --self-reward-url http://127.0.0.1:29040/v1 --self-reward-model gemma4-e4b-it \
      --gt-url https://YOUR-JUDGE/v1 --gt-model gemma-4-31B-it --gt-key-file ./secrets/gt-api-key
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _add_repo_paths() -> None:
    paths = [
        ROOT,
        ROOT / "scripts",
        ROOT / "vendor/prorl/src",
        ROOT / "vendor/prorl",
        ROOT / "vendor/reward/src",
        ROOT / "vendor/slime",
        ROOT / "vendor/nemo-rl",
        ROOT / "vendor/megatron-bridge/src",
        ROOT / "vendor/megatron-bridge/3rdparty/Megatron-LM",
        ROOT / "vendor/nanobot",
    ]
    for p in reversed(paths):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, status: str, detail: str) -> None:
        assert status in ("PASS", "WARN", "FAIL")
        self.checks.append({"name": name, "status": status, "detail": detail})

    @property
    def ok(self) -> bool:
        return all(c["status"] != "FAIL" for c in self.checks)


def check_required_files(report: Report) -> None:
    required = [
        # Configs
        "configs/deal-rs-gemma4.recipe.json",
        "configs/deal-rs-qwen35.recipe.json",
        "configs/deal-ws-gemma4.recipe.json",
        "configs/deal-ws-qwen35.recipe.json",
        "configs/deal-rs.env.example",
        "configs/deal-ws.env.example",
        "configs/reward-service.deal-rs.template.json",
        "configs/reward-service.deal-ws.template.json",
        # Core runtime
        "runtime/run_core.sh",
        "runtime/train_async.py",
        "runtime/portable_contract.py",
        "runtime/sample_contract.py",
        "runtime/aligned_eval_recovery.py",
        "runtime/ws_judge/wsclaw_judge_worker.py",
        "runtime/ws_judge/judge_proxy.py",
        "runtime/ws_judge/judge_sidecar.sh",
        "runtime/ws_judge/inspect_workspace_file.py",
        "runtime/ws_judge/judge_system_prompt.txt",
        # DEAL-WS workspace templates (all 7 required by install_workspace_scaffold)
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/AGENTS.md",
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/SOUL.md",
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/USER.md",
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/TOOLS.md",
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/HEARTBEAT.md",
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/.gitignore",
        "vendor/prorl/scripts/clawgym_nanobot/workspace_templates/memory/MEMORY.md",
        # DEAL-WS / DEAL-RS harness & reward
        "vendor/prorl/scripts/clawgym_nanobot/wsclaw_eval_only.py",
        "vendor/prorl/scripts/clawgym_nanobot/inspect_workspace_file.py",
        "vendor/prorl/scripts/clawgym_nanobot/polar/polar_config.yaml.template",
        "vendor/prorl/scripts/clawgym_nanobot/polar/topology.yaml.template",
        "vendor/prorl/src/clawgym_nanobot/artifact_validity_gate.py",
        "vendor/prorl/src/polar/gateway/proxy.py",
        "vendor/reward/training_service.py",
        "vendor/reward/prompts/rubric_agent_wsclaw.md",
        "vendor/reward/prompts/rubric_agent_researchclaw_relaxed.md",
        "vendor/reward/schemas/rubric.schema.json",
        "vendor/reward/src/sr_correctness/wsclaw_scoring.py",
        # Vendored ML backends (including data/ modules)
        "vendor/slime/slime/backends/nemo_megatron_utils/loss.py",
        "vendor/nemo-rl/nemo_rl/data/__init__.py",
        "vendor/nemo-rl/examples/configs/grpo_math_8B_megatron.yaml",
        "vendor/megatron-bridge/src/megatron/bridge/data/__init__.py",
        "vendor/nanobot/nanobot/agent/tools/gemma_tool_schema.py",
    ]
    missing = [rel for rel in required if not (ROOT / rel).is_file()]
    if missing:
        report.add("required_files", "FAIL", f"Missing {len(missing)} required files: {missing[:5]}")
    else:
        report.add("required_files", "PASS", f"All {len(required)} required source, template, and config files present")


def check_ws_scaffold_installer(report: Report) -> None:
    try:
        spec = importlib.util.spec_from_file_location(
            "wsclaw_eval_only_verify",
            ROOT / "vendor/prorl/scripts/clawgym_nanobot/wsclaw_eval_only.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory(prefix="deal-ws-scaffold-check-") as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            mod.install_workspace_scaffold(ws)
            expected = [
                "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md",
                "HEARTBEAT.md", ".gitignore", "memory/MEMORY.md",
                "memory/history.jsonl",
            ]
            missing = [p for p in expected if not (ws / p).is_file()]
            if missing:
                report.add("ws_scaffold_installer", "FAIL", f"Missing installed files: {missing}")
            else:
                report.add("ws_scaffold_installer", "PASS", "install_workspace_scaffold() created all required WS files")
    except Exception as exc:
        report.add("ws_scaffold_installer", "FAIL", f"{type(exc).__name__}: {exc}")


def check_configs_and_contracts(report: Report) -> None:
    try:
        from configure_run import configure
        from runtime.portable_contract import validate_recovery_args
        import types

        checked = []
        for benchmark in ("deal-rs", "deal-ws"):
            for model in ("gemma4-e4b-it", "qwen3.5-9b"):
                with tempfile.TemporaryDirectory(prefix="deal-cfg-check-") as tmp:
                    t = Path(tmp)
                    (t / "model").mkdir()
                    (t / "model/config.json").write_text("{}\n", encoding="utf-8")
                    (t / "data/train").mkdir(parents=True)
                    (t / "data/eval").mkdir(parents=True)
                    (t / "data/DATA_READY.json").write_text(
                        json.dumps({"train_dir": str(t / "data/train"), "eval_dir": str(t / "data/eval")}),
                        encoding="utf-8",
                    )
                    for name in ("gt.key", "sr.key"):
                        kf = t / name
                        kf.write_text("verify-dummy-key\n", encoding="utf-8")
                        kf.chmod(0o600)
                    cfg = t / "local.env"
                    cfg.write_text(
                        "\n".join([
                            f"BENCHMARK={benchmark}",
                            f"MODEL={model}",
                            f"BASE_MODEL={t / 'model'}",
                            f"DATA_DIR={t / 'data'}",
                            f"RUN_DIR={t / 'runs'}",
                            "TRAIN_GPUS=0,1,2,3",
                            "ROLLOUT_GPUS=4,5,6,7",
                            "SELF_REWARD_BASE_URL=http://127.0.0.1:29040/v1",
                            f"SELF_REWARD_MODEL={model}",
                            f"SELF_REWARD_API_KEY_FILE={t / 'sr.key'}",
                            "GT_BASE_URL=https://example.invalid/v1",
                            "GT_MODEL=gemma-4-31B-it",
                            f"GT_API_KEY_FILE={t / 'gt.key'}",
                            "HOST_IP=127.0.0.1",
                        ]) + "\n",
                        encoding="utf-8",
                    )
                    work = configure(cfg, attempt="verify")
                    env = json.loads((work / "private/environment.json").read_text(encoding="utf-8"))
                    args_ns = types.SimpleNamespace(
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
                    old_env = os.environ.copy()
                    try:
                        os.environ.update(env)
                        validate_recovery_args(args_ns)
                    finally:
                        os.environ.clear()
                        os.environ.update(old_env)
                    checked.append(f"{benchmark}/{model}")
        report.add("configs_and_contracts", "PASS", f"Validated 4+4 & 1GT+7SR contracts for: {', '.join(checked)}")
    except Exception as exc:
        report.add("configs_and_contracts", "FAIL", f"{type(exc).__name__}: {exc}")


def check_cli_tools(report: Report, *, strict_gpu: bool) -> None:
    for tool, required_in_gpu, desc in (
        ("python3", True, "Python interpreter"),
        ("git", False, "Git VCS"),
        ("nvcc", strict_gpu, "CUDA 12.9 compiler"),
        ("nvidia-smi", strict_gpu, "NVIDIA GPU driver"),
        ("claude", False, "Claude Code CLI (required for DEAL-WS native GT judge sidecar)"),
    ):
        found = shutil.which(tool)
        if found:
            report.add(f"cli:{tool}", "PASS", found)
        else:
            status = "FAIL" if required_in_gpu else "WARN"
            report.add(f"cli:{tool}", status, f"{desc} not found on PATH")


def check_doc_stack(report: Report, *, strict: bool) -> None:
    doc_modules = ["docx", "openpyxl", "pptx", "pypdf", "yaml", "pyarrow"]
    missing = []
    for mod in doc_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        status = "FAIL" if strict else "WARN"
        report.add("doc_stack_imports", status, f"Missing document/data packages (needed for DEAL-WS hardgate & prepare_data): {missing}")
    else:
        report.add("doc_stack_imports", "PASS", f"All document/data packages importable: {', '.join(doc_modules)}")


def check_gpu_and_ml_stack(report: Report) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        report.add("platform", "FAIL", f"Training requires Linux x86_64, found {platform.system()}/{platform.machine()}")
    else:
        report.add("platform", "PASS", f"Linux x86_64, Python {sys.version.split()[0]}")

    if sys.version_info[:2] != (3, 12):
        report.add("python_version", "FAIL", f"Expected Python 3.12, found {sys.version_info[0]}.{sys.version_info[1]}")
    else:
        report.add("python_version", "PASS", f"Python {sys.version.split()[0]}")

    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        if len(rows) < 8:
            report.add("gpu_topology", "FAIL", f"Need >=8 GPUs for 4 train + 4 rollout, found {len(rows)}")
        else:
            report.add("gpu_topology", "PASS", f"Found {len(rows)} GPUs (first: {rows[0]})")
    except Exception as exc:
        report.add("gpu_topology", "FAIL", f"nvidia-smi failed: {exc}")

    ml_modules = [
        "torch", "ray", "transformers", "hydra", "omegaconf", "torchdata", "tensordict",
        "transformer_engine.pytorch", "flash_attn", "ffpa_attn.functional",
        "sgl_kernel", "flashinfer", "sglang.srt.server_args", "sglang_router",
        "megatron.core", "megatron.bridge", "nemo_rl.data", "nemo_rl.distributed.batched_data_dict",
        "nanobot.agent.runner", "clawgym_nanobot.polar_data", "clawgym_nanobot.artifact_validity_gate",
        "sr_correctness.wsclaw_scoring",
    ]
    failed = []
    for name in ml_modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f"{name} ({type(exc).__name__})")
    if failed:
        report.add("ml_stack_imports", "FAIL", f"{len(failed)} imports failed: {failed[:6]}")
    else:
        report.add("ml_stack_imports", "PASS", f"All {len(ml_modules)} ML/RL/harness modules imported cleanly")


def probe_openai_endpoint(report: Report, label: str, base_url: str, model: str | None, key_file: Path | None) -> None:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    headers = {"Content-Type": "application/json"}
    if key_file is not None:
        key = key_file.expanduser().resolve().read_text(encoding="utf-8").strip()
        headers["Authorization"] = f"Bearer {key}"
    try:
        req = urllib.request.Request(url + "/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        if model and models and model not in models:
            report.add(f"endpoint:{label}", "WARN", f"Reachable at {url}, but {model!r} not in advertised models {models[:5]}")
        else:
            report.add(f"endpoint:{label}", "PASS", f"Reachable at {url} (models={models[:3]})")
    except Exception as exc:
        report.add(f"endpoint:{label}", "FAIL", f"Failed to reach {url}/models: {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cpu", "gpu", "all"], default="cpu")
    ap.add_argument("--self-reward-url", default="", help="Optional SELF_REWARD_BASE_URL to probe")
    ap.add_argument("--self-reward-model", default="", help="Optional expected self-reward model name")
    ap.add_argument("--self-reward-key-file", type=Path, help="Optional self-reward API key file")
    ap.add_argument("--gt-url", default="", help="Optional GT_BASE_URL to probe")
    ap.add_argument("--gt-model", default="", help="Optional expected GT judge model name")
    ap.add_argument("--gt-key-file", type=Path, help="Optional GT judge API key file")
    ap.add_argument("--json", action="store_true", help="Print JSON output only")
    args = ap.parse_args()

    _add_repo_paths()
    report = Report()

    if args.mode in ("cpu", "all"):
        check_required_files(report)
        check_ws_scaffold_installer(report)
        check_configs_and_contracts(report)
        check_cli_tools(report, strict_gpu=(args.mode == "all"))
        check_doc_stack(report, strict=(args.mode == "all"))

    if args.mode in ("gpu", "all"):
        check_gpu_and_ml_stack(report)

    if args.self_reward_url:
        probe_openai_endpoint(report, "self_reward", args.self_reward_url, args.self_reward_model or None, args.self_reward_key_file)
    if args.gt_url:
        probe_openai_endpoint(report, "gt_judge", args.gt_url, args.gt_model or None, args.gt_key_file)

    if args.json:
        print(json.dumps({"ok": report.ok, "checks": report.checks}, indent=2))
    else:
        for row in report.checks:
            print(f"[{row['status']:4s}] {row['name']:24s} {row['detail']}")
        print("-" * 72)
        print("OVERALL:", "PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
