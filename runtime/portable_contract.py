"""Unified portable recipe contract for DEAL-WS (hardgate) and DEAL-RS (1 GT anchor + 7 SR).

Topology is unified to single-host 4 train + 4 rollout GPUs (8 GPUs total).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .sample_contract import validate_sample  # noqa: F401


def validate_objective(args, *, benchmark: str, model_name: str) -> None:
    if benchmark == "deal-rs":
        expected = {
            "use_tis": False,
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "kl_coef": 0.0,
        }
    elif benchmark == "deal-ws":
        if model_name == "gemma4-e4b-it":
            expected = {
                "use_tis": True,
                "use_kl_loss": True,
                "kl_loss_coef": 0.1,
                "kl_coef": 0.0,
            }
        else:
            expected = {
                "use_tis": False,
                "use_kl_loss": False,
                "kl_loss_coef": 0.0,
                "kl_coef": 0.0,
            }
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}")

    for key, value in expected.items():
        actual = getattr(args, key, None)
        if isinstance(value, float):
            if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Unexpected training objective for {benchmark}/{model_name}: {key}={actual!r} != {value!r}")
        elif actual != value:
            raise ValueError(f"Unexpected training objective for {benchmark}/{model_name}: {key}={actual!r} != {value!r}")


def validate_recovery_args(args):
    manifest = json.loads(Path(os.environ["GEMMA_SELFREWARD_MANIFEST"]).read_text(encoding="utf-8"))
    cfg = json.loads(Path(os.environ["CLAWGYM_SELF_REWARD_CONFIG"]).read_text(encoding="utf-8"))

    benchmark = manifest["benchmark"]
    model_name = manifest["model_name"]
    if benchmark not in ("deal-rs", "deal-ws"):
        raise ValueError(f"Unsupported benchmark in manifest: {benchmark!r}")
    if model_name not in ("gemma4-e4b-it", "qwen3.5-9b"):
        raise ValueError(f"Unsupported model in manifest: {model_name!r}")

    # Fixed 1 GT anchor + 7 SR (8 total candidates per task) for both DEAL-RS and DEAL-WS.
    if cfg.get("candidate_count") != 8 or cfg.get("mode") != "self_reward_7plus1_train7":
        raise ValueError("SRE requires candidate_count=8 and mode='self_reward_7plus1_train7'")

    validate_objective(args, benchmark=benchmark, model_name=model_name)

    expected_backend = "nemo_megatron" if model_name == "gemma4-e4b-it" else "megatron"
    if benchmark == "deal-rs":
        num_rollout = 113
        seq_length = 90112
        max_tokens_per_gpu = 91136
        rollout_ctx = 73728
        rollout_resp = 16384
        eval_ctx = 81920
        eval_resp = 16384
        eval_fn = "runtime.aligned_eval_recovery.generate_rollout"
        expected_async_level = "4"
        expected_microcompact = "0"
    else:
        # DEAL-WS enforces the deterministic deliverable hardgate.
        if os.environ.get("CLAWGYM_ARTIFACT_VALIDITY_GATE") != "1":
            raise ValueError("DEAL-WS requires CLAWGYM_ARTIFACT_VALIDITY_GATE=1 (hardgate)")
        num_rollout = 125
        seq_length = 73728
        max_tokens_per_gpu = 75776
        rollout_ctx = 65536
        rollout_resp = 8192
        eval_ctx = 65536
        eval_resp = 8192
        eval_fn = "clawgym_nanobot.eval_rollout.generate_rollout"
        expected_async_level = "2"
        expected_microcompact = "1"

    train_gpus = int(manifest.get("train_gpus", 4))
    rollout_gpus = int(manifest.get("rollout_gpus", 4))
    tp_size = int(manifest.get("tp_size", min(4, train_gpus)))
    cp_size = int(manifest.get("cp_size", 1))
    pp_size = int(manifest.get("pp_size", 1))
    gpus_per_engine = int(manifest.get("rollout_gpus_per_engine", 1))
    if train_gpus < 1 or rollout_gpus < 1 or gpus_per_engine < 1:
        raise ValueError("Invalid GPU counts in manifest")
    if train_gpus % (tp_size * cp_size * pp_size) != 0:
        raise ValueError("train_gpus must be divisible by tp_size * cp_size * pp_size")
    if rollout_gpus % gpus_per_engine != 0:
        raise ValueError("rollout_gpus must be divisible by rollout_gpus_per_engine")

    exact = dict(
        train_backend=expected_backend,
        advantage_estimator="grpo",
        num_rollout=num_rollout,
        rollout_batch_size=16,
        n_samples_per_prompt=8,
        global_batch_size=128,
        num_steps_per_rollout=1,
        normalize_advantages=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=train_gpus,
        rollout_num_gpus=rollout_gpus,
        num_gpus_per_node=train_gpus + rollout_gpus,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=pp_size,
        rollout_num_gpus_per_engine=gpus_per_engine,
        no_load_optim=True,
        no_load_rng=True,
        seq_length=seq_length,
        max_tokens_per_gpu=max_tokens_per_gpu,
        eval_interval=20,
        rollout_max_context_len=rollout_ctx,
        rollout_max_response_len=rollout_resp,
        eval_max_context_len=eval_ctx,
        eval_max_response_len=eval_resp,
        data_source_path="clawgym_nanobot.polar_data.FiniteSequentialRolloutDataSource",
        eval_function_path=eval_fn,
    )
    for name, value in exact.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"Recipe mismatch: {name}={getattr(args, name, None)!r} != {value!r}")

    for name, value in [("lr", 1e-6), ("rollout_temperature", 0.7), ("rollout_top_p", 1.0)]:
        if not math.isclose(float(getattr(args, name)), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Recipe mismatch: {name}={getattr(args, name)!r} != {value!r}")

    if args.start_rollout_id != manifest["start_rollout_id"]:
        raise ValueError("start_rollout_id differs from manifest")
    if bool(args.skip_eval_before_train) != (0 < manifest["start_rollout_id"] < num_rollout):
        raise ValueError("skip_eval_before_train disagrees with start_rollout_id")
    if Path(args.hf_checkpoint).resolve() != Path(manifest["actor_checkpoint"]).resolve():
        raise ValueError("hf_checkpoint differs from manifest actor_checkpoint")
    if Path(args.ref_load).resolve() != Path(manifest["original_checkpoint"]).resolve():
        raise ValueError("ref_load differs from manifest original_checkpoint")

    for key, value in {
        "POLAR_MAX_ASYNC_LEVEL": expected_async_level,
        "POLAR_MAX_OFF_POLICY_STEPS": "3",
        "SLIME_PAD_DYNAMIC_DP_SAMPLES": "0",
        "OPENCLAW_TRAIN_ORDER": "sequential",
        "NANOBOT_MICROCOMPACT_ENABLED": expected_microcompact,
    }.items():
        if os.environ.get(key) != value:
            raise ValueError(f"Environment mismatch: {key}={os.environ.get(key)!r} != {value!r}")

    return {
        "validated": True,
        "benchmark": benchmark,
        "model": model_name,
        "base": manifest["original_checkpoint"],
        "candidate_count": 8,
        "topology": f"{train_gpus} train + {rollout_gpus} rollout",
    }


validate_recipe = validate_recovery_args
