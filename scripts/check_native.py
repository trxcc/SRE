#!/usr/bin/env python3
"""Native parser + finite dataset preflight check, without loading model weights or calling judges."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
os.environ.update(record["runtime_env"]["env_vars"])
for entry in reversed(os.environ["PYTHONPATH"].split(":")):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)

from slime.utils.arguments import parse_args
from runtime.portable_contract import validate_recovery_args
from slime.agent.token_contract import validate_fresh_args

entry = record["entry"]
sys.argv = [entry[2]] + entry[3:] if entry[:2] == ["python3", "-u"] else entry[1:]
args = parse_args()
contract_summary = validate_recovery_args(args)
validate_fresh_args(args)

from clawgym_nanobot.polar_data import FiniteSequentialRolloutDataSource

source = FiniteSequentialRolloutDataSource(args)
manifest = json.loads(Path(os.environ["GEMMA_SELFREWARD_MANIFEST"]).read_text(encoding="utf-8"))
if args.load:
    source.load(args.start_rollout_id - 1)
expected_tasks = 1800 if manifest["benchmark"] == "deal-rs" else 2000
assert len(source.dataset) == expected_tasks, f"Expected {expected_tasks} tasks, got {len(source.dataset)}"
assert source.sample_offset == manifest["actual_consumed_cursor"]
assert source.sample_group_index == source.sample_offset
assert source.sample_index == source.sample_offset * 8
print(json.dumps({
    "native_parser": True,
    "recipe": True,
    "benchmark": manifest["benchmark"],
    "model": manifest["model_name"],
    "topology": contract_summary["topology"],
    "candidate_count": 8,
    "finite_tasks": expected_tasks,
    "committed_cursor": source.sample_offset,
    "model_weight_reads": 0,
    "judge_calls": 0,
}))
