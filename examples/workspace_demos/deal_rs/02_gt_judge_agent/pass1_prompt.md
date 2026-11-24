# Pass-1 Hybrid Judge Prompt (GT Anchor Evaluation)

Inspect the candidate deliverables under `workspace/results/audit/` against the authoritative inputs under `workspace/input_files/` and `rubric.json`. Score each binary criterion `{0, 1}` in Pass 1 or flag uncertain items for Pass-2 agentic verification.

## Original Task

Compare the paper's claimed KV-cache pruning protocol in `paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt` against the released evaluation code in `repo_snapshot/eval_kv_prune.py`. Produce two files: `results/audit/kv_prune_audit.csv` with columns `aspect,paper_claim,code_behavior,status,evidence_location` covering `sink_tokens`, `eviction_policy`, `retain_ratio`, `score_metric`, and `warmup_steps`, and `results/audit/summary.md` summarizing every verified match and discrepancy with exact code line numbers.
