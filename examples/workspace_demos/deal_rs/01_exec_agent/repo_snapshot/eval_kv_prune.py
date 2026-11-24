"""Reference evaluation harness for KV-Prune 2026."""

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
