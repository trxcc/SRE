# KV-Prune Paper vs. Code Audit (Anchor Rollout)

## Matches
- `sink_tokens`: Both paper and code (`eval_kv_prune.py:L4`) use `4`.
- `retain_ratio`: Both paper and code (`eval_kv_prune.py:L6`) use `0.25`.
- `warmup_steps`: Both paper and code (`eval_kv_prune.py:L7`) use `64`.

## Discrepancies
- `eviction_policy`: Paper claims `heavy_hitter_l1`, whereas `eval_kv_prune.py:L5` and `prune_cache` (`L16-L18`) implement `recency_fifo`.
- `score_metric`: Paper claims `token_f1`, whereas `eval_kv_prune.py:L8` and `compute_score` (`L21-L23`) compute `exact_match`.
