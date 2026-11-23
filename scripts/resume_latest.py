#!/usr/bin/env python3
"""Resume training from the latest complete HF checkpoint and its committed rollout cursor."""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from configure_run import ROOT, read_checkpoint, read_env

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", type=Path, required=True)
args = ap.parse_args()
cfg = read_env(args.config)
benchmark = cfg["BENCHMARK"].strip().lower()
run = Path(cfg["RUN_DIR"]).expanduser().resolve()

candidates = []
for p in (run / "checkpoints/candidates").glob("step-*"):
    try:
        if not (p / "checkpoint.json").is_file():
            continue
        start, offset, _ = read_checkpoint(p, benchmark=benchmark)
        if not any(p.glob("*.safetensors")):
            continue
        candidates.append((start, offset, p))
    except (ValueError, KeyError, AssertionError, FileNotFoundError, RuntimeError):
        continue

if not candidates:
    raise SystemExit("No complete HF checkpoint with committed cursor found; original config unchanged.")

start, offset, checkpoint = max(candidates, key=lambda x: x[0])
cfg["RESUME_HF"] = str(checkpoint)
target = run / "resume-local.env"
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as f:
    for k, v in cfg.items():
        f.write(k + "=" + shlex.quote(v) + "\n")
print(f"Resuming {benchmark} at rollout {start}, committed task cursor {offset}, checkpoint {checkpoint}", flush=True)
os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/run.py"), "--config", str(target)])
