# JSONL metric relay

`WANDB_JSONL_RELAY=1` separates training metrics from W&B networking:

1. Each reduced train, rollout, or eval metric dictionary is appended as one
   complete raw event to `SLIME_METRICS_JSONL_PATH`.
2. A rank-0 coalescer groups raw events by RL iteration. It appends one line to
   `metrics/steps.jsonl` only after every required component is present:
   rollout metrics, actor train metrics, train performance metrics, the critic
   metrics required by PPO, and eval metrics when that iteration schedules an
   eval. When `num_steps_per_rollout` is greater than one, all optimizer
   substeps are mapped back to their rollout iteration and aggregated before
   the record is emitted. The output keeps the optimizer-step range and
   substep count as explicit metrics.
3. One rank-0 relay tails `steps.jsonl` and is the only process that calls
   `wandb.log`. Every namespace uses the shared `iteration` axis, and the relay
   also passes that value as W&B's explicit step.
4. The relay persists its consumed byte offset in
   `metrics/wandb-relay-state.json`, so a restart continues without replaying
   the full log.

`events.jsonl` is the append-only audit journal; it can contain several records
for one iteration and is not sent directly to W&B. `steps.jsonl` is the
ready-only view used for tracking. The coalescer uses the rollout id / training
iteration as the join key even in async mode. Policy staleness remains a
separate metric (`polar/policy_version/*` and `polar/staleness/*`); it must not
be inferred from W&B's row number.

PPO critic-only warmup is handled explicitly. Before
`NUM_CRITIC_ONLY_STEPS`, a complete iteration requires rollout and critic
metrics but no actor metrics. After warmup it requires both critic and actor
components. A scheduled eval is always required before that iteration can be
published.

The ClusterX GRPO and PPO wrappers enable the relay by default. The reusable
launcher keeps it opt-in:

```bash
export USE_WANDB=1
export WANDB_JSONL_RELAY=1
export WANDB_ENTITY=...
export WANDB_PROJECT=...
export CLAWGYM_PRIVATE_ENV=/path/to/private-wandb-env
```

Per-run files are:

```text
metrics/events.jsonl
metrics/events.jsonl.lock
metrics/steps.jsonl
metrics/steps.jsonl.lock
metrics/coalescer.log
metrics/coalescer.done
metrics/wandb-relay-state.json
metrics/wandb-relay.log
wandb/
```

`SLIME_METRICS_JSONL_FSYNC=1` requests an `fsync` after every event. It is off
by default because the extra shared-filesystem latency is normally not worth
adding to every training step.

For a process that was already running before this sink was enabled,
`training_log_metrics_jsonl.py` can recover the complete structured metric
dictionaries from `rank0-train.log` into the same JSONL schema. This is a
migration tool; new launches should use the direct sink.
