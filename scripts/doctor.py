#!/usr/bin/env python3
"""Check runtime dependencies and 8-GPU topology without loading model weights."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imports", action="store_true", help="Verify Python runtime and framework imports")
    ap.add_argument("--gpu", action="store_true", help="Verify 8 visible CUDA GPUs for 4 train + 4 rollout")
    a = ap.parse_args()

    paths = [
        ROOT,
        ROOT / "vendor/prorl/src",
        ROOT / "vendor/slime",
        ROOT / "vendor/nemo-rl",
        ROOT / "vendor/megatron-bridge/src",
        ROOT / "vendor/megatron-bridge/3rdparty/Megatron-LM",
        ROOT / "vendor/nanobot",
    ]
    sglang_root = os.environ.get("SGLANG_ROOT", "").strip()
    if sglang_root:
        paths.append(Path(sglang_root) / "python")
    sys.path[:0] = list(map(str, paths))

    required_files = [
        "runtime/run_core.sh",
        "runtime/train_async.py",
        "runtime/portable_contract.py",
        "runtime/capture_cli.py",
        "runtime/ws_judge/wsclaw_judge_worker.py",
        "vendor/prorl/src/clawgym_nanobot/artifact_validity_gate.py",
        "vendor/prorl/scripts/clawgym_nanobot/polar/polar_config.yaml.template",
        "vendor/prorl/scripts/clawgym_nanobot/polar/topology.yaml.template",
        "vendor/reward/training_service.py",
        "vendor/reward/src/sr_correctness/wsclaw_scoring.py",
        "vendor/nemo-rl/examples/configs/grpo_math_8B_megatron.yaml",
    ]
    for rel in required_files:
        if not (ROOT / rel).is_file():
            raise SystemExit("Missing packaged source: " + rel)

    if a.imports:
        assert sys.version_info[:2] == (3, 12), "Python 3.12 required"
        runpy.run_path(str(ROOT / "vendor/prorl/scripts/trx_scripts/gemma4_python_startup/sitecustomize.py"))
        modules = [
            "torch", "ray", "transformers", "hydra", "omegaconf", "torchdata", "tensordict",
            "transformer_engine.pytorch", "flash_attn", "ffpa_attn.functional",
            "sgl_kernel", "flashinfer", "sglang.srt.server_args", "sglang_router",
            "megatron.core", "megatron.bridge", "nemo_rl.distributed.batched_data_dict",
            "nanobot.agent.runner", "nanobot.providers.openai_compat_provider",
            "clawgym_nanobot.polar_data", "clawgym_nanobot.artifact_validity_gate",
            "docx", "openpyxl", "pptx", "pypdf", "yaml",
        ]
        failed = []
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception as exc:
                failed.append({"module": name, "error": type(exc).__name__, "message": str(exc)[:400]})
        if failed:
            print(json.dumps({"dependency_import_failures": failed}, indent=2))
            raise SystemExit(1)
        print("All runtime dependency imports passed; no model weights loaded.")

    if a.gpu:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()
        assert len(rows) >= 8, f"Unified 4 train + 4 rollout layout requires at least 8 GPUs, found {len(rows)}"
        print("\n".join(rows))

    if not (a.imports or a.gpu):
        print("Packaged entry points are present. Use --imports and/or --gpu on the target Linux host.")


if __name__ == "__main__":
    main()
