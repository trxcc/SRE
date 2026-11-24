# KV-Prune Paper vs. Code Audit

## Verified Matches (2/5)
- `sink_tokens`: Paper and `repo_snapshot/eval_kv_prune.py` (`L4`, `L15`) both pin the first `4` sink tokens.
- `retain_ratio`: Paper and `repo_snapshot/eval_kv_prune.py` (`L6`, `L16`) both retain `0.25` (25%) of non-sink tokens.

## Verified Discrepancies (3/5)
- `eviction_policy`: Paper specifies `heavy_hitter_l1` using cumulative attention weights, but `eval_kv_prune.py` sets `eviction_policy="recency_fifo"` (`L5`) and `prune_cache` keeps only the most recent tokens (`L16-L18`).
- `warmup_steps`: Paper specifies `warmup_steps=64`, but `eval_kv_prune.py` sets `warmup_steps=16` (`L7`) and checks `step < cfg["warmup_steps"]` (`L13`).
- `score_metric`: Paper specifies normalized `token_f1`, but `eval_kv_prune.py` sets `score_metric="exact_match"` (`L8`) and `compute_score` only supports `exact_match` (`L21-L24`).
