SELF-REWARD SOURCE PATH RULE: Original public task inputs are under `input_files/`. When the user task, a source guide, or rubric text names an original input path P such as `SOURCE_GUIDE.md` or `paper_lib/...`, inspect `input_files/P`. Paths already beginning with `input_files/` and candidate output paths such as `results/...` remain unchanged.

# Pass-1 Self-Scoring Prompt (DEAL-RS Hybrid Scorer)

Evaluate the candidate outputs (`results/audit/kv_prune_audit.csv`, `results/audit/summary.md`) against `compiled_hybrid_rubric.json` using only the authoritative sources under `input_files/`. Return binary `{0.0, 1.0}` scores for `criterion_1..criterion_5`.

## Original Task

Compare the paper's claimed KV-cache pruning protocol in `paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt` against the released evaluation code in `repo_snapshot/eval_kv_prune.py`. Produce two files: `results/audit/kv_prune_audit.csv` with columns `aspect,paper_claim,code_behavior,status,evidence_location` covering `sink_tokens`, `eviction_policy`, `retain_ratio`, `score_metric`, and `warmup_steps`, and `results/audit/summary.md` summarizing every verified match and discrepancy with exact code line numbers.
