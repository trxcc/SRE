# Original task

Compare the paper's claimed KV-cache pruning protocol in `paper_lib/kv_prune_2026/paper_sections_text/04_experiments.txt` against the released evaluation code in `repo_snapshot/eval_kv_prune.py`. Produce two files: `results/audit/kv_prune_audit.csv` with columns `aspect,paper_claim,code_behavior,status,evidence_location` covering `sink_tokens`, `eviction_policy`, `retain_ratio`, `score_metric`, and `warmup_steps`, and `results/audit/summary.md` summarizing every verified match and discrepancy with exact code line numbers.
