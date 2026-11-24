#!/usr/bin/env python3
"""Materialize and verify self-contained SRE workspace demos for DEAL-WS and DEAL-RS.

Each benchmark includes all four agent roles in the Self-Rewarding Evolution loop:
  1. 01_exec_agent        - Solver executing in the nanobot workspace
  2. 02_gt_judge_agent    - External GT judge / user simulator (anchor feedback F_t)
  3. 03_self_rubric_agent - Frozen pi_0 generating shared rubric Q_t from anchor + F_t
  4. 04_self_scoring_agent- Frozen pi_0 scoring offline candidates against Q_t (isolated)

Usage:
  python3 scripts/build_workspace_demos.py --materialize
  python3 scripts/build_workspace_demos.py --verify
  python3 scripts/build_workspace_demos.py --materialize --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "examples" / "workspace_demos"
PRORL_ROOT = REPO_ROOT / "vendor" / "prorl"
REWARD_SRC = REPO_ROOT / "vendor" / "reward" / "src"
WS_TEMPLATES = PRORL_ROOT / "scripts" / "clawgym_nanobot" / "workspace_templates"
WS_JUDGE_PROMPT = REPO_ROOT / "runtime" / "ws_judge" / "judge_system_prompt.txt"
ARTIFACT_GATE_SCRIPT = PRORL_ROOT / "src" / "clawgym_nanobot" / "artifact_validity_gate.py"
RUBRIC_PROMPT_WS = REPO_ROOT / "vendor" / "reward" / "prompts" / "rubric_agent_wsclaw.md"
RUBRIC_PROMPT_RS = REPO_ROOT / "vendor" / "reward" / "prompts" / "rubric_agent_researchclaw_relaxed.md"
RUBRIC_SCHEMA = REPO_ROOT / "vendor" / "reward" / "schemas" / "rubric.schema.json"

for _p in (str(REWARD_SRC), str(PRORL_ROOT), str(PRORL_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sr_correctness.contracts import canonicalize_native_score, validate_rubric  # noqa: E402
from sr_correctness.native import (  # noqa: E402
    assert_rubric_workspace_safe,
    assert_score_workspace_safe,
    compile_ws_rubric,
    ws_score_instructions,
)
from sr_correctness.researchclaw_hybrid import (  # noqa: E402
    SELF_REWARD_SOURCE_PATH_RULE,
    compile_researchclaw_self_rubric,
)
from sr_correctness.wsclaw_scoring import SCORE_CONTRACT, artifact_gate_zero  # noqa: E402


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stdlib_text_gate(workspace: Path, required: list[dict[str, Any]]) -> dict[str, Any]:
    """Stdlib equivalent of artifact_validity_gate.py for text/md/json/py demos."""
    root = workspace.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for item in required:
        rel = PurePosixPath(item["path"])
        fmt = str(item.get("format") or rel.suffix.lstrip(".")).lower().lstrip(".")
        fmt = {"markdown": "md"}.get(fmt, fmt)
        row: dict[str, Any] = {"path": rel.as_posix(), "format": fmt, "valid": False}
        path = root / rel
        if not path.exists():
            row["reason"] = "required_file_missing"
        elif not path.is_file():
            row["reason"] = "required_path_not_file"
        else:
            blob = path.read_bytes()
            row.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            if len(blob) == 0:
                row["reason"] = "required_file_empty"
            else:
                try:
                    if fmt == "json":
                        json.loads(blob.decode("utf-8-sig"))
                    elif fmt == "py":
                        compile(blob, str(path), "exec", dont_inherit=True)
                    else:
                        blob.decode("utf-8-sig")
                    row.update(valid=True, reason="format_readable")
                except Exception as exc:
                    row.update(reason="invalid_format", detail=f"{type(exc).__name__}: {exc}"[:600])
        rows.append(row)
    return {
        "version": 2,
        "passed": bool(rows) and all(r["valid"] for r in rows),
        "artifacts": rows,
        "scope": "required file existence/readable format and Python syntax only; no execution, content-length or style thresholds",
    }


def run_artifact_gate(workspace: Path, required: list[dict[str, Any]]) -> dict[str, Any]:
    gate_python = os.environ.get("WSBENCH_PYTHON") or sys.executable
    proc = subprocess.run(
        [gate_python, str(ARTIFACT_GATE_SCRIPT), "--check"],
        input=json.dumps({"workspace": str(workspace.resolve()), "required": required}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode == 0:
        gate = json.loads(proc.stdout)
    elif "ModuleNotFoundError" in proc.stderr:
        # Office-doc libraries (docx/openpyxl/pptx/pypdf/yaml) are installed by
        # env/setup_wsbench_env.sh; fall back cleanly for stdlib text/md demos.
        gate = _stdlib_text_gate(workspace, required)
    else:
        raise RuntimeError(f"artifact_validity_gate failed: {proc.stderr[-1000:]}")
    rows = gate.get("artifacts") or []
    for row in rows:
        if row.get("valid") and row.get("bytes") == 0:
            row["valid"] = False
            row["reason"] = "required_file_empty"
    gate["passed"] = bool(rows) and all(row.get("valid") is True for row in rows)
    gate["self_score_gate_version"] = 1
    gate["trusted_contract"] = "demo_output_contract.json"
    gate["semantic_content_verified"] = False
    return gate


# ---------------------------------------------------------------------------
# DEAL-WS synthetic demo: ws_demo_q3_budget_audit
# ---------------------------------------------------------------------------
WS_TASK_QUERY = (
    "Audit our Q3 SaaS spend using `input/Desktop/q3_software_spend.csv` and "
    "`input/Documents/budget_policy.md`. Create `Desktop/q3_spend_audit.md` containing: "
    "(1) total Q3 spend and spend by department, (2) every line item that violates "
    "`budget_policy.md` after accounting for documented pre-approved exceptions, "
    "(3) the net non-compliant dollar total, and (4) concise remediation actions per owner."
)

WS_INPUT_CSV = """invoice_id,vendor,department,monthly_usd,q3_total_usd,vp_approved,preapproved_exception
INV-301,CloudMetrics Pro,Engineering,4200,12600,yes,no
INV-302,VectorStore Cloud,Engineering,6500,19500,no,no
INV-303,DesignFlow Enterprise,Product,2800,8400,yes,no
INV-304,AdPulse Analytics,Marketing,3500,10500,no,no
INV-305,DataVault Backup,Infrastructure,5000,15000,no,yes
INV-306,QuickNote AI,Operations,900,2700,no,no
"""

WS_INPUT_POLICY = """# Q3 Software Spend Policy

1. Any single vendor with Q3 total spend strictly greater than $10,000 requires VP sign-off (`vp_approved=yes`).
2. Exception rule: Vendors marked `preapproved_exception=yes` under the Annual Continuity Waiver are exempt from the VP sign-off requirement for Q3.
3. Vendors at or below $10,000 Q3 total do not require VP sign-off.
4. Remediation actions for non-compliant items must name the invoice ID, department, and required retroactive approval or downgrade step.
"""

WS_ANCHOR_REPORT = """# Q3 Software Spend Audit (Anchor Rollout)

## 1. Spend Summary
- **Total Q3 spend**: $68,700
- Engineering: $32,100
- Infrastructure: $15,000
- Marketing: $10,500
- Product: $8,400
- Operations: $2,700

## 2. Policy Violations
- `INV-302` (VectorStore Cloud, Engineering): $19,500 (> $10,000, `vp_approved=no`).
- `INV-304` (AdPulse Analytics, Marketing): $10,500 (> $10,000, `vp_approved=no`).
- `INV-305` (DataVault Backup, Infrastructure): $15,000 (> $10,000, `vp_approved=no`).

## 3. Net Non-Compliant Spend
- Total non-compliant spend: **$45,000** (`INV-302` + `INV-304` + `INV-305`).

## 4. Remediation Actions
- Engineering (`INV-302`): Request retroactive VP sign-off or downgrade tier before Q4 renewal.
- Marketing (`INV-304`): Obtain VP approval or cap monthly seats under $3,333.
- Infrastructure (`INV-305`): Obtain VP sign-off immediately.
"""

WS_CANDIDATE_REPORT = """# Q3 Software Spend Audit

## 1. Spend Summary
- **Total Q3 spend**: $68,700 across 6 invoices.
- **By department**:
  - Engineering: $32,100 (`INV-301` $12,600 + `INV-302` $19,500)
  - Infrastructure: $15,000 (`INV-305` $15,000)
  - Marketing: $10,500 (`INV-304` $10,500)
  - Product: $8,400 (`INV-303` $8,400)
  - Operations: $2,700 (`INV-306` $2,700)

## 2. Policy Violations (After Pre-Approved Exceptions)
- **`INV-302` — VectorStore Cloud (Engineering)**: $19,500 Q3 spend exceeds the $10,000 threshold without VP approval (`vp_approved=no`, `preapproved_exception=no`).
- **`INV-304` — AdPulse Analytics (Marketing)**: $10,500 Q3 spend exceeds the $10,000 threshold without VP approval (`vp_approved=no`, `preapproved_exception=no`).
- *Excluded by policy*: `INV-305` (DataVault Backup, Infrastructure, $15,000) carries `preapproved_exception=yes` under the Annual Continuity Waiver and is therefore compliant; `INV-301` has VP approval; `INV-303` ($8,400) and `INV-306` ($2,700) are <= $10,000.

## 3. Net Non-Compliant Total
- **Net non-compliant spend**: **$30,000** (`INV-302` $19,500 + `INV-304` $10,500).

## 4. Remediation Actions
- **Engineering (`INV-302`, VectorStore Cloud)**: Submit retroactive VP sign-off within 5 business days or downsize cluster tier before Q4 billing.
- **Marketing (`INV-304`, AdPulse Analytics)**: Obtain retroactive VP approval or reduce monthly spend below $3,333 so Q4 total stays at or below $10,000.
"""

WS_ANCHOR_FINAL_MSG = (
    "I completed the Q3 SaaS spend audit and saved the report to `Desktop/q3_spend_audit.md`. "
    "Total Q3 spend is $68,700, with three flagged invoices (`INV-302`, `INV-304`, `INV-305`) "
    "totaling $45,000 in non-compliant spend."
)

WS_ANCHOR_FEEDBACK = (
    "Your total spend and department breakdown match the CSV, and you correctly flagged INV-302 and INV-304. "
    "However, you misclassified INV-305 (DataVault Backup) as a violation even though it is marked "
    "preapproved_exception=yes under the Annual Continuity Waiver in budget_policy.md, which inflated the "
    "net non-compliant total from $30,000 to $45,000 and added an unnecessary remediation item for Infrastructure."
)

WS_GT_RUBRIC = {
    "grounded_rubric": [
        "Reports exact total Q3 spend ($68,700) and exact department totals (Engineering $32,100, Infrastructure $15,000, Marketing $10,500, Product $8,400, Operations $2,700).",
        "Identifies INV-302 and INV-304 as the only policy violations.",
        "Explicitly exempts INV-305 under the Annual Continuity Waiver (`preapproved_exception=yes`).",
        "Computes net non-compliant spend as exactly $30,000.",
        "Provides actionable owner-specific remediation steps naming invoice IDs and departments for every true violation without flagging compliant vendors.",
    ],
    "rubric_types": ["Results Assessment"] * 5,
}

WS_SELF_RUBRIC: dict[str, Any] = {
    "schema_version": "self_reward_rubric_v3",
    "criteria": [
        {
            "id": "c1",
            "criterion": "Desktop/q3_spend_audit.md reports the exact total Q3 software spend and per-department spend totals matching input/Desktop/q3_software_spend.csv.",
            "expected": "Total Q3 spend = $68,700; Engineering = $32,100, Infrastructure = $15,000, Marketing = $10,500, Product = $8,400, Operations = $2,700.",
            "target_paths": ["Desktop/q3_spend_audit.md", "input/Desktop/q3_software_spend.csv"],
            "reason": "Direct arithmetic requirement from the user prompt verifiable from the source CSV.",
            "weight": 1.0,
        },
        {
            "id": "c2",
            "criterion": "Desktop/q3_spend_audit.md flags INV-302 ($19,500) and INV-304 ($10,500) as non-compliant vendors exceeding $10,000 without VP sign-off.",
            "expected": "Both INV-302 (VectorStore Cloud, Engineering) and INV-304 (AdPulse Analytics, Marketing) are listed as policy violations.",
            "target_paths": [
                "Desktop/q3_spend_audit.md",
                "input/Desktop/q3_software_spend.csv",
                "input/Documents/budget_policy.md",
            ],
            "reason": "Core threshold rule in budget_policy.md rule 1.",
            "weight": 1.0,
        },
        {
            "id": "c3",
            "criterion": "Desktop/q3_spend_audit.md applies the Annual Continuity Waiver exception so INV-305 (DataVault Backup) is not classified as a policy violation.",
            "expected": "INV-305 ($15,000, preapproved_exception=yes) is excluded from violations and not mixed with non-compliant items; INV-301, INV-303, INV-306 are also compliant.",
            "target_paths": [
                "Desktop/q3_spend_audit.md",
                "input/Desktop/q3_software_spend.csv",
                "input/Documents/budget_policy.md",
            ],
            "reason": "Policy rule 2 explicitly exempts preapproved_exception=yes; anchor feedback highlighted this exact failure mode.",
            "weight": 1.0,
        },
        {
            "id": "c4",
            "criterion": "Desktop/q3_spend_audit.md reports the exact net non-compliant dollar total after excluding pre-approved exceptions.",
            "expected": "Net non-compliant total equals $30,000 ($19,500 + $10,500).",
            "target_paths": ["Desktop/q3_spend_audit.md", "input/Desktop/q3_software_spend.csv"],
            "reason": "Explicit prompt deliverable #3 derived from true violations only.",
            "weight": 1.0,
        },
        {
            "id": "c5",
            "criterion": "Desktop/q3_spend_audit.md provides concise remediation actions per owner naming the invoice ID, department, and retroactive approval or downgrade action for each true violation.",
            "expected": "Actionable remediation items for Engineering (INV-302) and Marketing (INV-304), without demanding remediation for exempt INV-305.",
            "target_paths": ["Desktop/q3_spend_audit.md", "input/Documents/budget_policy.md"],
            "reason": "Explicit prompt deliverable #4 and policy rule 4.",
            "weight": 1.0,
        },
    ],
    "limitations": [
        "Only verifies the requested Markdown audit report; does not prescribe a specific prose template or table layout."
    ],
}


# ---------------------------------------------------------------------------
# DEAL-RS synthetic demo: rs_demo_kv_cache_audit
# ---------------------------------------------------------------------------
RS_TASK_QUERY = (
    "Compare the paper's claimed KV-cache pruning protocol in "
    "`paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt` against the "
    "released evaluation code in `repo_snapshot/eval_kv_prune.py`. Produce two files: "
    "`results/audit/kv_prune_audit.csv` with columns "
    "`aspect,paper_claim,code_behavior,status,evidence_location` covering "
    "`sink_tokens`, `eviction_policy`, `retain_ratio`, `score_metric`, and `warmup_steps`, "
    "and `results/audit/summary.md` summarizing every verified match and discrepancy with "
    "exact code line numbers."
)

RS_PAPER_TEXT = """Section 4: Experimental Setup (KV-Prune 2026)

To evaluate long-context decoding under a fixed memory budget, we keep the first 4 attention
sink tokens (`sink_tokens=4`) pinned throughout generation. Once the cache exceeds the budget,
we apply Heavy-Hitter L1 eviction (`eviction_policy="heavy_hitter_l1"`) using cumulative
attention weights averaged over all heads, retaining 25% of non-sink tokens (`retain_ratio=0.25`).
Eviction starts only after 64 prefill/decode warmup steps (`warmup_steps=64`). All long-context
benchmarks are scored with normalized F1 (`score_metric="token_f1"`).
"""

RS_REPO_CODE = '''"""Reference evaluation harness for KV-Prune 2026."""

DEFAULT_CONFIG = {
    "sink_tokens": 4,
    "eviction_policy": "recency_fifo",   # NOTE: FIFO window, not heavy-hitter L1
    "retain_ratio": 0.25,
    "warmup_steps": 16,                  # starts evicting after 16 steps
    "score_metric": "exact_match",
}


def prune_cache(keys, values, step, cfg=DEFAULT_CONFIG):
    if step < cfg["warmup_steps"]:
        return keys, values
    sink = cfg["sink_tokens"]
    keep_recent = max(1, int((len(keys) - sink) * cfg["retain_ratio"]))
    # Keeps initial sink tokens plus most recent tokens (FIFO eviction)
    return keys[:sink] + keys[-keep_recent:], values[:sink] + values[-keep_recent:]


def compute_score(pred: str, gold: str, cfg=DEFAULT_CONFIG) -> float:
    if cfg["score_metric"] == "exact_match":
        return float(pred.strip() == gold.strip())
    raise ValueError(f"Unsupported metric: {cfg['score_metric']}")
'''

RS_ANCHOR_CSV = """aspect,paper_claim,code_behavior,status,evidence_location
sink_tokens,4,4,match,eval_kv_prune.py:L4
eviction_policy,heavy_hitter_l1,recency_fifo,mismatch,eval_kv_prune.py:L5
retain_ratio,0.25,0.25,match,eval_kv_prune.py:L6
score_metric,token_f1,exact_match,mismatch,eval_kv_prune.py:L8
warmup_steps,64,64,match,eval_kv_prune.py:L7
"""

RS_ANCHOR_SUMMARY = """# KV-Prune Paper vs. Code Audit (Anchor Rollout)

## Matches
- `sink_tokens`: Both paper and code (`eval_kv_prune.py:L4`) use `4`.
- `retain_ratio`: Both paper and code (`eval_kv_prune.py:L6`) use `0.25`.
- `warmup_steps`: Both paper and code (`eval_kv_prune.py:L7`) use `64`.

## Discrepancies
- `eviction_policy`: Paper claims `heavy_hitter_l1`, whereas `eval_kv_prune.py:L5` and `prune_cache` (`L16-L18`) implement `recency_fifo`.
- `score_metric`: Paper claims `token_f1`, whereas `eval_kv_prune.py:L8` and `compute_score` (`L21-L23`) compute `exact_match`.
"""

RS_CANDIDATE_CSV = """aspect,paper_claim,code_behavior,status,evidence_location
sink_tokens,4,4,match,eval_kv_prune.py:L4
eviction_policy,heavy_hitter_l1,recency_fifo,mismatch,eval_kv_prune.py:L5;L16-L18
retain_ratio,0.25,0.25,match,eval_kv_prune.py:L6
score_metric,token_f1,exact_match,mismatch,eval_kv_prune.py:L8;L21-L23
warmup_steps,64,16,mismatch,eval_kv_prune.py:L7;L13
"""

RS_CANDIDATE_SUMMARY = """# KV-Prune Paper vs. Code Audit

## Verified Matches (2/5)
- `sink_tokens`: Paper and `repo_snapshot/eval_kv_prune.py` (`L4`, `L15`) both pin the first `4` sink tokens.
- `retain_ratio`: Paper and `repo_snapshot/eval_kv_prune.py` (`L6`, `L16`) both retain `0.25` (25%) of non-sink tokens.

## Verified Discrepancies (3/5)
- `eviction_policy`: Paper specifies `heavy_hitter_l1` using cumulative attention weights, but `eval_kv_prune.py` sets `eviction_policy="recency_fifo"` (`L5`) and `prune_cache` keeps only the most recent tokens (`L16-L18`).
- `warmup_steps`: Paper specifies `warmup_steps=64`, but `eval_kv_prune.py` sets `warmup_steps=16` (`L7`) and checks `step < cfg["warmup_steps"]` (`L13`).
- `score_metric`: Paper specifies normalized `token_f1`, but `eval_kv_prune.py` sets `score_metric="exact_match"` (`L8`) and `compute_score` only supports `exact_match` (`L21-L24`).
"""

RS_ANCHOR_FINAL_MSG = (
    "Completed the audit in `results/audit/kv_prune_audit.csv` and `results/audit/summary.md`. "
    "Found 3 matches (`sink_tokens`, `retain_ratio`, `warmup_steps`) and 2 mismatches (`eviction_policy`, `score_metric`)."
)

RS_ANCHOR_FEEDBACK = (
    "Your CSV schema and findings on sink_tokens, retain_ratio, eviction_policy, and score_metric are accurate. "
    "However, you incorrectly marked warmup_steps as a match with code_behavior=64; in repo_snapshot/eval_kv_prune.py "
    "line 7, DEFAULT_CONFIG sets warmup_steps to 16 (not 64), so warmup_steps is a mismatch."
)

RS_GT_RUBRIC = {
    "aggregation": "weighted_mean",
    "items": [
        {"id": "gt_1", "weight": 1.0, "score_type": "binary", "pass_condition": "results/audit/kv_prune_audit.csv has exact header and all 5 required aspects."},
        {"id": "gt_2", "weight": 1.0, "score_type": "binary", "pass_condition": "sink_tokens=4 and retain_ratio=0.25 are marked as match with valid code evidence."},
        {"id": "gt_3", "weight": 1.0, "score_type": "binary", "pass_condition": "eviction_policy is marked mismatch (heavy_hitter_l1 vs recency_fifo) citing L5/prune_cache."},
        {"id": "gt_4", "weight": 1.0, "score_type": "binary", "pass_condition": "score_metric is marked mismatch (token_f1 vs exact_match) citing L8/compute_score."},
        {"id": "gt_5", "weight": 1.0, "score_type": "binary", "pass_condition": "warmup_steps is marked mismatch (64 vs 16) citing L7/L13 in both CSV and summary.md."},
    ],
}

RS_SELF_RUBRIC: dict[str, Any] = {
    "schema_version": "self_reward_rubric_v3",
    "criteria": [
        {
            "id": "c1",
            "criterion": "results/audit/kv_prune_audit.csv exists with exact header aspect,paper_claim,code_behavior,status,evidence_location and includes all five required aspects.",
            "expected": "CSV contains header `aspect,paper_claim,code_behavior,status,evidence_location` and rows for `sink_tokens`, `eviction_policy`, `retain_ratio`, `score_metric`, and `warmup_steps`.",
            "target_paths": ["results/audit/kv_prune_audit.csv"],
            "reason": "Explicit schema and row coverage requirement in the task prompt.",
            "weight": 1.0,
        },
        {
            "id": "c2",
            "criterion": "Both results/audit/kv_prune_audit.csv and results/audit/summary.md correctly classify sink_tokens (4 vs 4) and retain_ratio (0.25 vs 0.25) as matches grounded in eval_kv_prune.py.",
            "expected": "sink_tokens=4 (match, L4) and retain_ratio=0.25 (match, L6) verified against input_files/paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt and input_files/repo_snapshot/eval_kv_prune.py.",
            "target_paths": [
                "results/audit/kv_prune_audit.csv",
                "results/audit/summary.md",
                "input_files/repo_snapshot/eval_kv_prune.py",
            ],
            "reason": "Verifies true positive match detection without false discrepancies.",
            "weight": 1.0,
        },
        {
            "id": "c3",
            "criterion": "Both deliverables identify eviction_policy as a mismatch between paper heavy_hitter_l1 and code recency_fifo with exact line references.",
            "expected": "paper_claim=`heavy_hitter_l1`, code_behavior=`recency_fifo`, status=`mismatch`, citing eval_kv_prune.py L5 (and optionally prune_cache L16-L18).",
            "target_paths": [
                "results/audit/kv_prune_audit.csv",
                "results/audit/summary.md",
                "input_files/repo_snapshot/eval_kv_prune.py",
            ],
            "reason": "Major algorithmic discrepancy directly visible in DEFAULT_CONFIG and prune_cache.",
            "weight": 1.0,
        },
        {
            "id": "c4",
            "criterion": "Both deliverables identify score_metric as a mismatch between paper token_f1 and code exact_match with exact line references.",
            "expected": "paper_claim=`token_f1`, code_behavior=`exact_match`, status=`mismatch`, citing eval_kv_prune.py L8 (and optionally compute_score L21-L23).",
            "target_paths": [
                "results/audit/kv_prune_audit.csv",
                "results/audit/summary.md",
                "input_files/repo_snapshot/eval_kv_prune.py",
            ],
            "reason": "Evaluation metric mismatch directly visible in DEFAULT_CONFIG and compute_score.",
            "weight": 1.0,
        },
        {
            "id": "c5",
            "criterion": "Both deliverables identify warmup_steps as a mismatch between paper 64 and code 16 with exact line references.",
            "expected": "paper_claim=`64`, code_behavior=`16`, status=`mismatch`, citing eval_kv_prune.py L7 (and optionally L13).",
            "target_paths": [
                "results/audit/kv_prune_audit.csv",
                "results/audit/summary.md",
                "input_files/paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt",
                "input_files/repo_snapshot/eval_kv_prune.py",
            ],
            "reason": "Directly verifiable in eval_kv_prune.py L7 and specifically missed by the anchor rollout.",
            "weight": 1.0,
        },
    ],
    "limitations": [
        "Audits static paper claims against the released script without executing benchmark datasets."
    ],
}


# ---------------------------------------------------------------------------
# Materializers
# ---------------------------------------------------------------------------
def materialize_deal_ws(root: Path) -> None:
    base = root / "deal_ws"
    judge_sys = WS_JUDGE_PROMPT.read_text(encoding="utf-8")
    required_contract = [{"path": "Desktop/q3_spend_audit.md", "must_create": True}]

    # 1. 01_exec_agent (Nanobot solver workspace)
    exec_dir = base / "01_exec_agent"
    exec_dir.mkdir(parents=True, exist_ok=True)
    for name in ("AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "HEARTBEAT.md"):
        shutil.copy2(WS_TEMPLATES / name, exec_dir / name)
    (exec_dir / "memory").mkdir(parents=True, exist_ok=True)
    shutil.copy2(WS_TEMPLATES / "memory" / "MEMORY.md", exec_dir / "memory" / "MEMORY.md")
    # Save template .gitignore as .example so git tracks all demo files cleanly
    shutil.copy2(WS_TEMPLATES / ".gitignore", exec_dir / "workspace_gitignore.example")
    write_text(exec_dir / "input" / "Desktop" / "q3_software_spend.csv", WS_INPUT_CSV)
    write_text(exec_dir / "input" / "Documents" / "budget_policy.md", WS_INPUT_POLICY)
    write_text(exec_dir / "Desktop" / "q3_spend_audit.md", WS_ANCHOR_REPORT)
    write_json(
        exec_dir / "task_context.json",
        {
            "task_id": "ws_demo_q3_budget_audit",
            "persona": "yunying",
            "required_source_files": ["Desktop/q3_software_spend.csv", "Documents/budget_policy.md"],
            "required_output_paths": ["Desktop/q3_spend_audit.md"],
        },
    )
    write_text(exec_dir / "PROMPT.md", f"# Exec Agent User Prompt\n\n{WS_TASK_QUERY}")

    # 2. 02_gt_judge_agent (External GT Judge / User Simulator on anchor rollout)
    gt_dir = base / "02_gt_judge_agent"
    write_text(gt_dir / "CLAUDE.md", judge_sys)
    write_text(gt_dir / "task.md", f"# Task\n\n{WS_TASK_QUERY}")
    write_json(gt_dir / "rubric.json", WS_GT_RUBRIC)
    write_text(gt_dir / "workspace" / "input" / "Desktop" / "q3_software_spend.csv", WS_INPUT_CSV)
    write_text(gt_dir / "workspace" / "input" / "Documents" / "budget_policy.md", WS_INPUT_POLICY)
    write_text(gt_dir / "workspace" / "Desktop" / "q3_spend_audit.md", WS_ANCHOR_REPORT)
    anchor_gate = run_artifact_gate(gt_dir / "workspace", required_contract)
    write_json(gt_dir / "review" / "artifact_validation.json", anchor_gate)
    write_json(
        gt_dir / "expected_outputs" / "judge_result.json",
        {
            "rubric_scores": {"1": 1.0, "2": 1.0, "3": 0.0, "4": 0.0, "5": 0.5},
            "score": 0.5,
            "user_feedback": WS_ANCHOR_FEEDBACK,
            "feedback": WS_ANCHOR_FEEDBACK,
        },
    )

    # 3. 03_self_rubric_agent (Frozen pi_0 building Q_t from TASK + ANCHOR + inputs)
    rubric_dir = base / "03_self_rubric_agent"
    write_text(rubric_dir / "AGENTS.md", RUBRIC_PROMPT_WS.read_text(encoding="utf-8"))
    write_text(
        rubric_dir / "SOUL.md",
        "You are a careful reward agent. Produce a truth-grounded rubric of atomic, objective checks and obey AGENTS.md.",
    )
    write_text(rubric_dir / "TASK.md", f"# Original task\n\n{WS_TASK_QUERY}")
    write_text(
        rubric_dir / "ANCHOR.md",
        "# Anchor and single feedback\n\n"
        "The anchor is one sampled answer, not ground truth. Its candidate deliverables keep "
        "their requested output paths in this workspace. Original task inputs are under input/. "
        "The feedback identifies useful issues but is not the sole rubric.\n\n"
        f"## Final reply\n\n{WS_ANCHOR_FINAL_MSG}\n\n"
        f"## User Feedback\n\n{WS_ANCHOR_FEEDBACK}\n\n"
        "## Anchor GT score\n\n0.50000000",
    )
    write_text(rubric_dir / "RUBRIC_SCHEMA.json", RUBRIC_SCHEMA.read_text(encoding="utf-8"))
    write_text(rubric_dir / "input" / "Desktop" / "q3_software_spend.csv", WS_INPUT_CSV)
    write_text(rubric_dir / "input" / "Documents" / "budget_policy.md", WS_INPUT_POLICY)
    write_text(rubric_dir / "Desktop" / "q3_spend_audit.md", WS_ANCHOR_REPORT)
    write_json(
        rubric_dir / "WORKSPACE_MANIFEST.json",
        {
            "schema_version": "rubric_workspace_v2",
            "dataset": "wsclaw",
            "rubric_contract": "wsclaw_full_workspace_v1",
            "allowed_output": "rubric.json",
            "forbidden_inputs": ["ground-truth rubric", "reference answer", "judge workspace", "reward code"],
            "anchor_evidence": {
                "task_source_files": ["input/Desktop/q3_software_spend.csv", "input/Documents/budget_policy.md"],
                "candidate_files": ["Desktop/q3_spend_audit.md"],
                "missing_candidate_files": [],
            },
        },
    )
    validate_rubric(WS_SELF_RUBRIC)
    write_json(rubric_dir / "expected_outputs" / "rubric.json", WS_SELF_RUBRIC)

    # 4. 04_self_scoring_agent (Frozen pi_0 scoring offline candidate #2, strictly isolated)
    score_dir = base / "04_self_scoring_agent"
    write_text(score_dir / "AGENTS.md", ws_score_instructions(judge_sys))
    write_text(score_dir / "SOUL.md", "You are the WSClaw judge. Follow AGENTS.md and the scoring prompt.")
    write_text(score_dir / "task.md", f"# Task\n\n{WS_TASK_QUERY}")
    write_json(score_dir / "rubric.json", WS_SELF_RUBRIC)
    write_json(score_dir / "compiled_ws_rubric.json", compile_ws_rubric(WS_SELF_RUBRIC))
    write_text(score_dir / "workspace" / "input" / "Desktop" / "q3_software_spend.csv", WS_INPUT_CSV)
    write_text(score_dir / "workspace" / "input" / "Documents" / "budget_policy.md", WS_INPUT_POLICY)
    write_text(score_dir / "workspace" / "Desktop" / "q3_spend_audit.md", WS_CANDIDATE_REPORT)
    cand_gate = run_artifact_gate(score_dir / "workspace", required_contract)
    write_json(score_dir / "artifact-validity-gate.json", cand_gate)
    write_json(score_dir / "review" / "artifact_validation.json", cand_gate)
    write_text(
        score_dir / "SCORING_PROMPT.md",
        "Inspect `task.md`, `rubric.json`, `review/artifact_validation.json`, and the candidate "
        "deliverables under `workspace/`. Score each criterion in `rubric.json` on [0, 1] and "
        "write `judge_result.json` at the judge root.\n\n"
        "Follow AGENTS.md for Nanobot-specific path and delivery instructions.",
    )
    write_json(
        score_dir / "SCORE_WORKSPACE_MANIFEST.json",
        {
            "dataset": "wsclaw",
            "result_path": "judge_result.json",
            "score_contract": SCORE_CONTRACT,
            "artifact_validity_gate": cand_gate,
            "anchor_present": False,
            "feedback_present": False,
            "ground_truth_assets_present": False,
            "instructions_file": "AGENTS.md",
            "workspace_materialization": "native_gt_link_clean_tree_v1",
        },
    )
    ws_native_result = {
        "rubric_scores": {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0},
        "feedback": (
            "Desktop/q3_spend_audit.md accurately reports $68,700 total spend and all five department "
            "subtotals, flags only INV-302 and INV-304, explicitly exempts INV-305 under the Annual "
            "Continuity Waiver, computes the exact $30,000 net non-compliant total, and gives clear "
            "owner-specific remediation steps."
        ),
        "user_feedback": (
            "Everything matches the Q3 spend CSV and budget policy, including the Annual Continuity "
            "Waiver exemption for INV-305 and the $30,000 net non-compliant total."
        ),
    }
    ws_accepted = canonicalize_native_score("wsclaw", ws_native_result, WS_SELF_RUBRIC, "candidate_02")
    write_json(score_dir / "expected_outputs" / "judge_result.json", ws_native_result)
    write_json(score_dir / "expected_outputs" / "accepted.json", ws_accepted)
    write_json(
        score_dir / "expected_outputs" / "repeat_aggregation.json",
        {
            "slot_id": "candidate_02",
            "repeat_count": 3,
            "repeat_scores": [1.0, 1.0, 1.0],
            "mean_score": 1.0,
            "aggregation": "mean",
        },
    )
    # Also produce a real deterministic hardgate-fail example via artifact_gate_zero()
    with tempfile.TemporaryDirectory(prefix="ws-demo-bad-") as tmp:
        bad_ws = Path(tmp) / "bad_workspace"
        (bad_ws / "Desktop").mkdir(parents=True)
        bad_gate = run_artifact_gate(bad_ws, required_contract)
        out = artifact_gate_zero(
            gate=bad_gate,
            rubric=WS_SELF_RUBRIC,
            slot_id="candidate_missing_deliverable",
            session_id="demo-hardgate-fail",
            run_dir=Path(tmp) / "gate_run",
        )
        write_json(score_dir / "expected_outputs" / "hardgate_fail_example.json", out)


def materialize_deal_rs(root: Path) -> None:
    base = root / "deal_rs"

    # 1. 01_exec_agent (Research solver workspace)
    exec_dir = base / "01_exec_agent"
    write_text(
        exec_dir / "WORKSPACE.md",
        "# DEAL-RS Exec Workspace\n\n"
        "Original public sources are mounted at `paper_lib/` and `repo_snapshot/`. "
        "Write all requested deliverables under `results/audit/`.",
    )
    write_text(
        exec_dir / "paper_lib" / "kv_prune_2026" / "paper_sections_text" / "04_experiments.txt",
        RS_PAPER_TEXT,
    )
    write_text(exec_dir / "repo_snapshot" / "eval_kv_prune.py", RS_REPO_CODE)
    write_text(exec_dir / "results" / "audit" / "kv_prune_audit.csv", RS_ANCHOR_CSV)
    write_text(exec_dir / "results" / "audit" / "summary.md", RS_ANCHOR_SUMMARY)
    write_text(exec_dir / "PROMPT.md", f"# Exec Agent User Prompt\n\n{RS_TASK_QUERY}")

    # 2. 02_gt_judge_agent (External GT Hybrid Judge / User Simulator on anchor rollout)
    gt_dir = base / "02_gt_judge_agent"
    write_json(gt_dir / "rubric.json", RS_GT_RUBRIC)
    write_text(
        gt_dir / "pass1_prompt.md",
        "# Pass-1 Hybrid Judge Prompt (GT Anchor Evaluation)\n\n"
        "Inspect the candidate deliverables under `workspace/results/audit/` against the "
        "authoritative inputs under `workspace/input_files/` and `rubric.json`. "
        "Score each binary criterion `{0, 1}` in Pass 1 or flag uncertain items for Pass-2 agentic verification.\n\n"
        f"## Original Task\n\n{RS_TASK_QUERY}",
    )
    write_text(
        gt_dir / "workspace" / "input_files" / "paper_lib" / "kv_prune_2026" / "paper_sections_text" / "04_experiments.txt",
        RS_PAPER_TEXT,
    )
    write_text(gt_dir / "workspace" / "input_files" / "repo_snapshot" / "eval_kv_prune.py", RS_REPO_CODE)
    write_text(gt_dir / "workspace" / "results" / "audit" / "kv_prune_audit.csv", RS_ANCHOR_CSV)
    write_text(gt_dir / "workspace" / "results" / "audit" / "summary.md", RS_ANCHOR_SUMMARY)
    write_json(
        gt_dir / "expected_outputs" / "pass1_triage.json",
        {
            "pass1_resolved": {"gt_1": 1.0, "gt_2": 1.0, "gt_3": 1.0, "gt_4": 1.0},
            "pass2_flagged": ["gt_5"],
            "reason": "Candidate claims warmup_steps=64 in code; verify eval_kv_prune.py L7.",
        },
    )
    write_json(
        gt_dir / "expected_outputs" / "judge_result.json",
        {
            "scores": {"criterion_1": 1.0, "criterion_2": 1.0, "criterion_3": 1.0, "criterion_4": 1.0, "criterion_5": 0.0},
            "weighted_score": 0.8,
            "notes": "eval_kv_prune.py:L7 defines warmup_steps=16, contradicting the candidate's match claim (64).",
        },
    )
    write_json(
        gt_dir / "expected_outputs" / "user_feedback.json",
        {"anchor_score": 0.8, "user_feedback": RS_ANCHOR_FEEDBACK},
    )

    # 3. 03_self_rubric_agent (Frozen pi_0 building Q_t from TASK + ANCHOR + input_files)
    rubric_dir = base / "03_self_rubric_agent"
    write_text(rubric_dir / "AGENTS.md", RUBRIC_PROMPT_RS.read_text(encoding="utf-8"))
    write_text(
        rubric_dir / "SOUL.md",
        "You are a careful reward agent. Produce a truth-grounded rubric of atomic, objective checks and obey AGENTS.md.",
    )
    write_text(rubric_dir / "TASK.md", f"# Original task\n\n{RS_TASK_QUERY}")
    write_text(
        rubric_dir / "ANCHOR.md",
        "# Anchor and single feedback\n\n"
        "The anchor is one sampled answer, not ground truth. Its candidate deliverables keep "
        "their requested output paths in this workspace. Original task inputs are under input_files/. "
        "The feedback identifies useful issues but is not the sole rubric.\n\n"
        f"## Final reply\n\n{RS_ANCHOR_FINAL_MSG}\n\n"
        f"## User feedback\n\n{RS_ANCHOR_FEEDBACK}\n\n"
        "## Anchor GT score\n\n0.80000000",
    )
    write_text(rubric_dir / "RUBRIC_SCHEMA.json", RUBRIC_SCHEMA.read_text(encoding="utf-8"))
    write_text(
        rubric_dir / "input_files" / "paper_lib" / "kv_prune_2026" / "paper_sections_text" / "04_experiments.txt",
        RS_PAPER_TEXT,
    )
    write_text(rubric_dir / "input_files" / "repo_snapshot" / "eval_kv_prune.py", RS_REPO_CODE)
    write_text(rubric_dir / "results" / "audit" / "kv_prune_audit.csv", RS_ANCHOR_CSV)
    write_text(rubric_dir / "results" / "audit" / "summary.md", RS_ANCHOR_SUMMARY)
    write_json(
        rubric_dir / "WORKSPACE_MANIFEST.json",
        {
            "schema_version": "rubric_workspace_v2",
            "dataset": "researchclaw",
            "rubric_contract": "truth_grounded_input_layer_v6",
            "allowed_output": "rubric.json",
            "forbidden_inputs": ["ground-truth rubric", "reference answer", "judge workspace", "reward code"],
            "anchor_evidence": {
                "materialization": "bounded_isolated_input_layer_v2",
                "task_source_files": [
                    "input_files/paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt",
                    "input_files/repo_snapshot/eval_kv_prune.py",
                ],
                "candidate_files": ["results/audit/kv_prune_audit.csv", "results/audit/summary.md"],
            },
        },
    )
    validate_rubric(RS_SELF_RUBRIC)
    write_json(rubric_dir / "expected_outputs" / "rubric.json", RS_SELF_RUBRIC)

    # 4. 04_self_scoring_agent (Frozen pi_0 hybrid 2-pass scoring offline candidate #2, isolated)
    score_dir = base / "04_self_scoring_agent"
    compiled_rs = compile_researchclaw_self_rubric(RS_SELF_RUBRIC)
    write_json(score_dir / "rubric.json", RS_SELF_RUBRIC)
    write_json(score_dir / "compiled_hybrid_rubric.json", compiled_rs)
    write_text(
        score_dir / "pass1_prompt.md",
        SELF_REWARD_SOURCE_PATH_RULE
        + "# Pass-1 Self-Scoring Prompt (DEAL-RS Hybrid Scorer)\n\n"
        "Evaluate the candidate outputs (`results/audit/kv_prune_audit.csv`, `results/audit/summary.md`) "
        "against `compiled_hybrid_rubric.json` using only the authoritative sources under `input_files/`. "
        "Return binary `{0.0, 1.0}` scores for `criterion_1..criterion_5`.\n\n"
        f"## Original Task\n\n{RS_TASK_QUERY}",
    )
    write_text(
        score_dir / "input_files" / "paper_lib" / "kv_prune_2026" / "paper_sections_text" / "04_experiments.txt",
        RS_PAPER_TEXT,
    )
    write_text(score_dir / "input_files" / "repo_snapshot" / "eval_kv_prune.py", RS_REPO_CODE)
    write_text(score_dir / "results" / "audit" / "kv_prune_audit.csv", RS_CANDIDATE_CSV)
    write_text(score_dir / "results" / "audit" / "summary.md", RS_CANDIDATE_SUMMARY)
    write_json(
        score_dir / "SCORE_WORKSPACE_MANIFEST.json",
        {
            "dataset": "researchclaw",
            "score_mode": "hybrid_agentic",
            "anchor_present": False,
            "feedback_present": False,
            "ground_truth_assets_present": False,
            "source_layer": "input_files/",
            "candidate_files": ["results/audit/kv_prune_audit.csv", "results/audit/summary.md"],
        },
    )
    rs_native_result = {
        "scores": {"criterion_1": 1.0, "criterion_2": 1.0, "criterion_3": 1.0, "criterion_4": 1.0, "criterion_5": 1.0},
        "notes": (
            "All 5 criteria verified directly against input_files/repo_snapshot/eval_kv_prune.py "
            "and input_files/paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt, "
            "including warmup_steps mismatch (paper 64 vs code 16 at L7/L13)."
        ),
    }
    rs_accepted = canonicalize_native_score("researchclaw", rs_native_result, RS_SELF_RUBRIC, "candidate_02")
    write_json(score_dir / "expected_outputs" / "pass1_triage.json", {"pass1_resolved": ["c1", "c2", "c3", "c4"], "pass2_verified": ["c5"]})
    write_json(score_dir / "expected_outputs" / "judge_result.json", rs_native_result)
    write_json(score_dir / "expected_outputs" / "accepted.json", rs_accepted)
    write_json(
        score_dir / "expected_outputs" / "repeat_aggregation.json",
        {
            "slot_id": "candidate_02",
            "repeat_count": 3,
            "repeat_scores": [1.0, 1.0, 1.0],
            "mean_score": 1.0,
            "aggregation": "mean",
        },
    )


def materialize_readme(root: Path) -> None:
    write_text(
        root / "README.md",
        """# SRE Workspace Demos (DEAL-WS & DEAL-RS)

Self-contained, synthetic workspace layouts illustrating all **four agent roles** in the
Self-Rewarding Evolution (SRE) loop (`1 GT anchor + 7 SR` offline candidates):

1. `01_exec_agent/` — Solver executing in the nanobot workspace (`π_θ`).
2. `02_gt_judge_agent/` — External GT judge / user simulator evaluating the single anchor rollout to produce `F_t = (u_t, c_t)`.
3. `03_self_rubric_agent/` — Frozen base model (`π_{θ_0}`) building the shared rubric `Q_t` (`self_reward_rubric_v3`) from `TASK.md`, `ANCHOR.md`, public inputs, and anchor deliverables.
4. `04_self_scoring_agent/` — Frozen base model (`π_{θ_0}`) scoring offline candidates `2..8` against `Q_t` with **strict isolation** (no `ANCHOR.md`, no raw feedback, no GT rubric/reference answer).

## Information Isolation Matrix

| Workspace Role | Public Inputs | Candidate Deliverables | Anchor + Feedback `F_t` | Hidden GT Rubric | Self-Rubric `Q_t` | Deterministic Hardgate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `01_exec_agent` | Yes | Own outputs | No | No | No | N/A |
| `02_gt_judge_agent` | Yes | Anchor outputs | Produces `F_t` | Yes (GT only) | No | Yes (DEAL-WS) |
| `03_self_rubric_agent` | Yes | Anchor outputs | Yes (`ANCHOR.md`) | **Forbidden** | Produces `Q_t` | N/A |
| `04_self_scoring_agent` | Yes | Offline candidate | **Forbidden** | **Forbidden** | Yes (Input) | Yes (DEAL-WS) |

## Included Demos

- **`deal_ws/` (`ws_demo_q3_budget_audit`)**: Office/operations SaaS budget audit (`Desktop/q3_spend_audit.md`).
  - `01_exec_agent/` uses the exact DEAL-WS workspace templates (`AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `HEARTBEAT.md`, `memory/MEMORY.md`; template `.gitignore` saved as `workspace_gitignore.example` so git tracks demo files).
  - `04_self_scoring_agent/` includes both a passing `artifact-validity-gate.json` (`SCORE_CONTRACT = wsclaw_native_judge_v2`) and a real deterministic `expected_outputs/hardgate_fail_example.json` generated by `sr_correctness.wsclaw_scoring.artifact_gate_zero()`.
- **`deal_rs/` (`rs_demo_kv_cache_audit`)**: Research paper-vs-code reproducibility audit (`results/audit/kv_prune_audit.csv`, `results/audit/summary.md`).
  - Uses `input_files/` layering (`bounded_isolated_input_layer_v2`) and `compile_researchclaw_self_rubric()` with binary `{0.0, 1.0}` criterion scoring.

## Regenerate & Verify

```bash
python3 scripts/build_workspace_demos.py --materialize --verify
```
""",
    )


# ---------------------------------------------------------------------------
# Verification against real repository contracts
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_demos(root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            raise AssertionError(f"{name} failed: {detail}")

    # 1. Verify real DEAL-WS install_workspace_scaffold() works cleanly
    from scripts.clawgym_nanobot import wsclaw_eval_only

    with tempfile.TemporaryDirectory(prefix="verify-ws-scaffold-") as tmp:
        ws = Path(tmp) / "ws"
        ws.mkdir()
        wsclaw_eval_only.install_workspace_scaffold(ws)
        expected_scaffold = [
            "AGENTS.md",
            "SOUL.md",
            "USER.md",
            "TOOLS.md",
            "HEARTBEAT.md",
            ".gitignore",
            "memory/MEMORY.md",
        ]
        missing = [rel for rel in expected_scaffold if not (ws / rel).is_file()]
        check("ws_scaffold_installer", not missing, f"missing={missing}")

    # 2. Verify workspace prompt/schema parity with repository sources
    ws_base = root / "deal_ws"
    rs_base = root / "deal_rs"
    check(
        "ws_rubric_prompt_parity",
        (ws_base / "03_self_rubric_agent" / "AGENTS.md").read_text(encoding="utf-8").strip()
        == RUBRIC_PROMPT_WS.read_text(encoding="utf-8").strip(),
        "matches vendor/reward/prompts/rubric_agent_wsclaw.md",
    )
    check(
        "rs_rubric_prompt_parity",
        (rs_base / "03_self_rubric_agent" / "AGENTS.md").read_text(encoding="utf-8").strip()
        == RUBRIC_PROMPT_RS.read_text(encoding="utf-8").strip(),
        "matches vendor/reward/prompts/rubric_agent_researchclaw_relaxed.md",
    )
    check(
        "ws_gt_judge_prompt_parity",
        (ws_base / "02_gt_judge_agent" / "CLAUDE.md").read_text(encoding="utf-8").strip()
        == WS_JUDGE_PROMPT.read_text(encoding="utf-8").strip(),
        "matches runtime/ws_judge/judge_system_prompt.txt",
    )
    check(
        "ws_score_instructions_parity",
        (ws_base / "04_self_scoring_agent" / "AGENTS.md").read_text(encoding="utf-8").strip()
        == ws_score_instructions(WS_JUDGE_PROMPT.read_text(encoding="utf-8")).strip(),
        "matches sr_correctness.native.ws_score_instructions()",
    )
    for label, sub in (("ws", ws_base), ("rs", rs_base)):
        check(
            f"{label}_rubric_schema_parity",
            _read_json(sub / "03_self_rubric_agent" / "RUBRIC_SCHEMA.json")
            == _read_json(RUBRIC_SCHEMA),
            "matches vendor/reward/schemas/rubric.schema.json",
        )

    # 3. Verify isolation guards (real assert_rubric_workspace_safe & assert_score_workspace_safe)
    for label, sub in (("ws", ws_base), ("rs", rs_base)):
        assert_rubric_workspace_safe(sub / "03_self_rubric_agent")
        check(f"{label}_rubric_isolation", True, "assert_rubric_workspace_safe passed")
        assert_score_workspace_safe(sub / "04_self_scoring_agent")
        manifest = _read_json(sub / "04_self_scoring_agent" / "SCORE_WORKSPACE_MANIFEST.json")
        check(
            f"{label}_score_isolation",
            manifest.get("anchor_present") is False
            and manifest.get("feedback_present") is False
            and manifest.get("ground_truth_assets_present") is False,
            "assert_score_workspace_safe passed; anchor/feedback/GT flags are False",
        )

    # 4. Validate self-rubrics, compiled rubrics, and canonical score arithmetic
    ws_rubric = validate_rubric(_read_json(ws_base / "03_self_rubric_agent" / "expected_outputs" / "rubric.json"))
    rs_rubric = validate_rubric(_read_json(rs_base / "03_self_rubric_agent" / "expected_outputs" / "rubric.json"))
    check("ws_rubric_contract", len(ws_rubric["criteria"]) == 5, "5 criteria validated")
    check("rs_rubric_contract", len(rs_rubric["criteria"]) == 5, "5 criteria validated")

    check(
        "ws_compiled_rubric_match",
        _read_json(ws_base / "04_self_scoring_agent" / "compiled_ws_rubric.json")
        == compile_ws_rubric(ws_rubric),
        "matches compile_ws_rubric()",
    )
    check(
        "rs_compiled_rubric_match",
        _read_json(rs_base / "04_self_scoring_agent" / "compiled_hybrid_rubric.json")
        == compile_researchclaw_self_rubric(rs_rubric),
        "matches compile_researchclaw_self_rubric()",
    )

    ws_native = _read_json(ws_base / "04_self_scoring_agent" / "expected_outputs" / "judge_result.json")
    ws_expected_acc = _read_json(ws_base / "04_self_scoring_agent" / "expected_outputs" / "accepted.json")
    ws_computed_acc = canonicalize_native_score("wsclaw", ws_native, ws_rubric, "candidate_02")
    check("ws_canonical_score", ws_computed_acc == ws_expected_acc, "ws canonical score matches")

    rs_native = _read_json(rs_base / "04_self_scoring_agent" / "expected_outputs" / "judge_result.json")
    rs_expected_acc = _read_json(rs_base / "04_self_scoring_agent" / "expected_outputs" / "accepted.json")
    rs_computed_acc = canonicalize_native_score("researchclaw", rs_native, rs_rubric, "candidate_02")
    check("rs_canonical_score", rs_computed_acc == rs_expected_acc, "rs canonical score matches")

    # 5. Live DEAL-WS deterministic artifact validity gate check (PASS + FAIL)
    required_contract = [{"path": "Desktop/q3_spend_audit.md", "must_create": True}]
    pass_gate = run_artifact_gate(ws_base / "04_self_scoring_agent" / "workspace", required_contract)
    check("ws_hardgate_pass", pass_gate["passed"] is True, "candidate deliverable passed gate")

    fail_example = _read_json(ws_base / "04_self_scoring_agent" / "expected_outputs" / "hardgate_fail_example.json")
    check(
        "ws_hardgate_fail_contract",
        fail_example["value"]["weighted_score"] == 0.0
        and fail_example["value"]["deterministic_policy_failure"] is True
        and fail_example["value"]["judge_executed"] is False
        and fail_example["status"]["reason"] == "required_artifact_invalid",
        "hardgate fail example matches artifact_gate_zero contract",
    )

    return {"ok": True, "root": str(root), "checks_passed": len(checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize and verify SRE workspace demos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Demo output directory")
    parser.add_argument("--materialize", action="store_true", help="Write workspace demos to --output")
    parser.add_argument("--verify", action="store_true", help="Verify materialized demos against repo contracts")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    args = parser.parse_args()

    if not args.materialize and not args.verify:
        parser.error("specify at least one of --materialize or --verify")

    out_dir = args.output.resolve()
    if args.materialize:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        materialize_deal_ws(out_dir)
        materialize_deal_rs(out_dir)
        materialize_readme(out_dir)

    summary: dict[str, Any] = {"materialized": args.materialize, "output": str(out_dir)}
    if args.verify:
        summary["verification"] = verify_demos(out_dir)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        if args.materialize:
            print(f"[OK] Materialized SRE workspace demos in {out_dir}")
        if args.verify:
            v = summary["verification"]
            print(f"[OK] Verified {v['checks_passed']} workspace demo contract checks")
            for item in v["checks"]:
                print(f"  - PASS {item['name']}: {item['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
