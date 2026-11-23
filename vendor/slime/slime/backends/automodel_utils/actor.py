"""Slime actor backed by NeMo-RL AutoModel/FSDP2 instead of Megatron."""

from __future__ import annotations


import logging
import os
import importlib.machinery
import socket
import sys
import types
from argparse import Namespace
from contextlib import nullcontext
from typing import Any

import ray
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from slime.ray.train_actor import TrainRayActor
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking, log
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import Box
from slime.utils.timer import Timer, inverse_timer, timer, with_defer

from .data import (
    attach_old_policy_logprobs,
    build_automodel_batch,
    concatenate_automodel_batches,
    pad_local_batch,
)
from .update_weight import AutoModelWeightUpdater

logger = logging.getLogger(__name__)


def _audit_gemma4_automodel_runtime(worker, policy_config: dict[str, Any]) -> None:
    """Fail closed unless the official Gemma4 CP/HSDP fast path is installed."""
    import importlib.metadata as metadata

    from nemo_automodel.components.attention.ffpa_attention import (
        _ffpa_varlen_ready,
    )
    from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh

    model = worker.model
    # Dense Gemma4 owns its CP ring setup.  Upstream normally installs it lazily
    # from the first call to its CP batch sharder, while this audit deliberately
    # runs immediately after actor initialization.  Invoke that same public,
    # idempotent model hook here so the audit checks the actual installed ring
    # instead of mistaking the expected lazy state for a missing backend.
    setup_cp_attention = getattr(model, "setup_cp_attention", None)
    if not callable(setup_cp_attention):
        raise RuntimeError("official Gemma4 model does not expose setup_cp_attention")
    setup_cp_attention(worker.cp_mesh)

    attention_modules = [
        module
        for module in model.modules()
        if getattr(module, "_cp_uses_attention_hook", False)
    ]
    global_attention = [
        module
        for module in attention_modules
        if getattr(module, "sliding_window", None) is None
    ]
    sliding_attention = [
        module
        for module in attention_modules
        if getattr(module, "sliding_window", None) is not None
    ]
    if len(global_attention) != 7 or len(sliding_attention) != 35:
        raise RuntimeError(
            "Gemma4 E4B attention layout mismatch: expected 7 global + 35 "
            f"sliding layers, got {len(global_attention)} + {len(sliding_attention)}"
        )
    if not all(
        getattr(module, "_gemma4_cp_use_ffpa", False)
        and int(getattr(module, "head_dim", 0)) == 512
        for module in global_attention
    ):
        raise RuntimeError("all seven Gemma4 global attention layers must use D512 CP FFPA")
    if not _ffpa_varlen_ready():
        raise RuntimeError("Gemma4 CP FFPA dense/varlen operators are not registered")
    if not all(
        getattr(module, "_gemma4_cp_sliding_backend", None) == "fa"
        and int(getattr(module, "head_dim", 0)) == 256
        for module in sliding_attention
    ):
        raise RuntimeError("all 35 Gemma4 sliding attention layers must use D256 CP FA2")
    if metadata.version("flash-attn") != "2.8.3":
        raise RuntimeError(
            f"Gemma4 CP FA2 requires flash-attn 2.8.3, got {metadata.version('flash-attn')}"
        )

    text_config = model.config.text_config
    if getattr(text_config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError(
            "Gemma4 CP must initialize through attn_implementation='sdpa'; "
            f"got {getattr(text_config, '_attn_implementation', None)!r}"
        )
    if bool(getattr(text_config, "use_cache", True)):
        raise RuntimeError("Gemma4 training must set text_config.use_cache=False")
    if not bool(getattr(text_config, "cp_text_only_pre_shard", False)):
        raise RuntimeError("Gemma4 ClawGym training requires cp_text_only_pre_shard=True")
    if not bool(getattr(text_config, "defer_final_logit_softcap_scale", False)):
        raise RuntimeError(
            "Gemma4 ClawGym training requires deferred chunk-local final-logit scaling"
        )
    freeze_ple_lookup = bool(getattr(text_config, "freeze_ple_lookup", False))
    frozen_ple_lookup_numel = int(
        getattr(model, "_gemma4_ple_lookup_frozen_numel", 0) or 0
    )
    if freeze_ple_lookup:
        lookup = getattr(model.model.language_model, "embed_tokens_per_layer", None)
        expected_numel = (
            int(getattr(text_config, "vocab_size_per_layer_input", 0) or 0)
            * int(getattr(text_config, "num_hidden_layers", 0) or 0)
            * int(getattr(text_config, "hidden_size_per_layer_input", 0) or 0)
        )
        if (
            lookup is None
            or frozen_ple_lookup_numel != expected_numel
            or any(parameter.requires_grad for parameter in lookup.parameters())
        ):
            raise RuntimeError(
                "Gemma4 freeze_ple_lookup did not freeze exactly the packed "
                f"PLE table: expected_numel={expected_numel} "
                f"recorded_numel={frozen_ple_lookup_numel}"
            )
    ple_checkpoint_layers = int(
        getattr(model, "_gemma4_ple_checkpoint_layer_count", 0) or 0
    )
    if not bool(getattr(text_config, "checkpoint_ple_branch", False)) or ple_checkpoint_layers != 42:
        raise RuntimeError(
            "Gemma4 ClawGym training requires checkpoint_ple_branch=True on "
            f"all 42 layers, got {ple_checkpoint_layers}"
        )

    fsdp_mesh = get_fsdp_dp_mesh(worker.device_mesh)
    replicate_size = int(policy_config["dtensor_cfg"].get("dp_replicate_size", 1))
    shard_cp_size = worker.device_mesh["dp_shard_cp"].size()
    allow_cross_node_fsdp = os.environ.get(
        "GEMMA4_AUTOMODEL_ALLOW_CROSS_NODE_FSDP", "0"
    ).strip()
    if allow_cross_node_fsdp not in {"0", "1"}:
        raise RuntimeError(
            "GEMMA4_AUTOMODEL_ALLOW_CROSS_NODE_FSDP must be '0' or '1', got "
            f"{allow_cross_node_fsdp!r}"
        )
    if replicate_size > 1:
        shard_group = fsdp_mesh.get_group("dp_shard_cp")
        shard_hosts: list[str | None] = [None] * dist.get_world_size(shard_group)
        dist.all_gather_object(shard_hosts, socket.gethostname(), group=shard_group)
        if len(set(shard_hosts)) != 1:
            topology = ",".join(map(str, shard_hosts))
            if allow_cross_node_fsdp != "1":
                raise RuntimeError(
                    "Gemma4 HSDP shard+CP group crosses nodes: " + topology
                )
            logger.warning(
                "Gemma4 HSDP shard+CP group crosses nodes under explicit "
                "resource-probe override: %s",
                topology,
            )

    if dist.get_rank() == 0:
        logger.info(
            "Gemma4 official runtime audit passed: attention=7xD512-CP-FFPA+"
            "35xD256-CP-FA2 attn_init=sdpa use_cache=false text_only_pre_shard=true "
            "ple_branch_checkpoint_layers=%d ple_lookup=%s(%d) deferred_logit_scale=true "
            "mesh=dp_replicate%d x dp_shard_cp%d x tp%d activation_checkpointing=%s "
            "lm_head=action_positions_only",
            ple_checkpoint_layers,
            "frozen" if freeze_ple_lookup else "trainable",
            frozen_ple_lookup_numel,
            replicate_size,
            shard_cp_size,
            worker.tp_size,
            policy_config["dtensor_cfg"]["activation_checkpointing"],
        )


def _make_optional_logger_imports_safe() -> None:
    """Keep unused NeMo logger backends from blocking actor imports.

    NeMo-RL imports all logger adapters eagerly even when Slime owns logging.
    The Slime image intentionally does not carry MLflow/SwanLab's complete
    dependency trees, so replace only an import that actually fails. A module
    spec is provided because Transformers/Accelerate probe it with find_spec.
    """

    for name in ("mlflow", "swanlab"):
        try:
            __import__(name)
        except Exception:
            sys.modules.pop(name, None)
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = module


def load_automodel_policy_config(args: Namespace) -> dict[str, Any]:
    if not args.automodel_config:
        raise ValueError("--train-backend automodel requires --automodel-config")

    _make_optional_logger_imports_safe()
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        load_config(args.automodel_config), resolve=True
    )
    policy = dict(config["policy"])
    policy["model_name"] = args.hf_checkpoint
    policy["tokenizer"] = dict(policy["tokenizer"])
    policy["tokenizer"]["name"] = args.hf_checkpoint
    # ClawGym trajectories are text-only.  Loading the multimodal processor
    # marks the NeMo-RL worker as VLM and intentionally trips its VLM+CP guard;
    # the upstream Gemma4 model itself still injects the required all-zero
    # mm_token_type_ids for this text path.
    policy["tokenizer"]["use_processor"] = False
    policy["max_total_sequence_length"] = int(args.seq_length)
    policy["max_grad_norm"] = float(args.clip_grad)
    policy["train_global_batch_size"] = int(args.global_batch_size)
    policy["train_micro_batch_size"] = 1
    policy["logprob_batch_size"] = 1
    policy["sequence_packing"] = {"enabled": False}
    if int(args.log_probs_chunk_size) > 0:
        policy["logprob_chunk_size"] = int(args.log_probs_chunk_size)

    # Follow the upstream Gemma4 E-series TP+CP recipes.  The model-owned
    # implementation installs Ring-CP while keeping HF initialization on SDPA;
    # global D512 ring chunks use FFPA and sliding D256 layers use FlashAttention
    # 2.  Direct attn_implementation="ffpa" would bypass the CP ring.
    freeze_ple_lookup_value = os.environ.get(
        "GEMMA4_AUTOMODEL_FREEZE_PLE_LOOKUP", "0"
    ).strip()
    if freeze_ple_lookup_value not in {"0", "1"}:
        raise ValueError(
            "GEMMA4_AUTOMODEL_FREEZE_PLE_LOOKUP must be '0' or '1', got "
            f"{freeze_ple_lookup_value!r}"
        )
    freeze_ple_lookup = freeze_ple_lookup_value == "1"
    policy["hf_config_overrides"] = {
        "text_config": {
            "use_cache": False,
            "output_hidden_states": False,
            "cp_full_attn_backend": "ffpa",
            "cp_sliding_attn_backend": "fa",
            # ClawGym trajectories are guaranteed text-only. Let Gemma4 slice
            # input_ids by CP rank before token embedding and packed PLE lookup.
            "cp_text_only_pre_shard": True,
            # Gemma4 KV sharing prevents whole-layer checkpointing. Recompute
            # only the pure PLE residual tail and leave attention/cache state
            # untouched.
            "checkpoint_ple_branch": True,
            # Optional memory diagnostic: keep the token-identity PLE values in
            # every forward while omitting only this lookup table's gradient.
            # The context projection and decoder PLE modules remain trainable.
            "freeze_ple_lookup": freeze_ple_lookup,
            # Keep only tanh(logits / cap) in the full action-logit buffer.
            # NeMo-RL applies cap/temperature inside each logprob chunk, which
            # avoids a second multi-GiB full-vocabulary allocation.
            "defer_final_logit_softcap_scale": True,
        }
    }

    dtensor_cfg = dict(policy["dtensor_cfg"])
    dtensor_cfg["enabled"] = True
    dtensor_cfg["_v2"] = True
    dtensor_cfg["sequence_parallel"] = False
    # Keep checkpointing policy in upstream AutoModel. Gemma4 KV-sharing makes
    # AutoModel's selective mode fall back to sub-module checkpointing, which
    # is faster when it fits; full mode is the official lower-memory fallback.
    activation_checkpointing = os.environ.get(
        "GEMMA4_AUTOMODEL_ACTIVATION_CHECKPOINTING", "selective"
    ).strip().lower()
    if activation_checkpointing not in {"selective", "full"}:
        raise ValueError(
            "GEMMA4_AUTOMODEL_ACTIVATION_CHECKPOINTING must be "
            f"'selective' or 'full', got {activation_checkpointing!r}"
        )
    dtensor_cfg["activation_checkpointing"] = activation_checkpointing
    logger.info(
        "AutoModel activation checkpointing policy: %s",
        activation_checkpointing,
    )
    dtensor_cfg["tensor_parallel_size"] = int(args.tensor_model_parallel_size)
    if args.automodel_context_parallel_size is not None:
        dtensor_cfg["context_parallel_size"] = int(
            args.automodel_context_parallel_size
        )
    tp_size = int(dtensor_cfg.get("tensor_parallel_size", 1))
    cp_size = int(dtensor_cfg.get("context_parallel_size", 1))
    if tp_size < 1 or cp_size < 1 or args.world_size % (tp_size * cp_size):
        raise ValueError(
            f"AutoModel TP*CP={tp_size}*{cp_size} must divide actor world size "
            f"{args.world_size}"
        )
    dp_size = args.world_size // (tp_size * cp_size)
    dp_replicate_size = int(
        os.environ.get("GEMMA4_AUTOMODEL_DP_REPLICATE_SIZE", "1")
    )
    if (
        dp_replicate_size < 1
        or dp_size % dp_replicate_size
        or (dp_replicate_size >= dp_size and dp_replicate_size != 1)
    ):
        raise ValueError(
            "AutoModel HSDP requires dp_replicate_size to divide DP and be "
            f"smaller than DP unless it is 1; got DP={dp_size}, "
            f"dp_replicate_size={dp_replicate_size}"
        )
    dtensor_cfg["dp_replicate_size"] = dp_replicate_size
    dtensor_cfg["automodel_kwargs"] = {
        "attn_implementation": "sdpa",
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
        "freeze_config": {
            "freeze_vision_tower": True,
            "freeze_audio_tower": True,
            "freeze_language_model": False,
        },
    }
    policy["dtensor_cfg"] = dtensor_cfg
    policy["make_sequence_length_divisible_by"] = 2 * cp_size

    train_mb_tokens = int(args.max_tokens_per_gpu) * cp_size
    policy["dynamic_batching"] = {
        "enabled": True,
        "train_mb_tokens": train_mb_tokens,
        "logprob_mb_tokens": train_mb_tokens,
        "sequence_length_round": max(64, 2 * cp_size),
    }

    optimizer = dict(policy["optimizer"])
    optimizer["kwargs"] = dict(optimizer["kwargs"])
    optimizer["kwargs"].update(
        {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "betas": (float(args.adam_beta1), float(args.adam_beta2)),
            "eps": float(args.adam_eps),
        }
    )
    policy["optimizer"] = optimizer
    return policy


def build_loss(args: Namespace):
    from nemo_rl.algorithms.loss.loss_functions import (
        ClippedPGLossConfig,
        ClippedPGLossFn,
    )

    # A zero-weight KL flag is used by the established async launcher to keep
    # KL diagnostics enabled.  It does not require a reference-policy forward:
    # ClippedPGLossFn always reports generation/policy KL and JS divergence
    # from the generation and previous-policy logprobs already in the batch.
    # Only reject configurations that would actually add a reference KL term.
    if args.kl_coef != 0 or (
        args.use_kl_loss and args.kl_loss_coef != 0
    ):
        raise NotImplementedError(
            "The first AutoModel integration supports the established ClawGym "
            "setting kl_coef=0; reference-policy KL will be added after the "
            "actor/rollout weight path is validated."
        )
    # The AutoModel adapter consumes the already materialized token-level
    # ``advantages`` tensor from build_automodel_batch.  ``policy_gradient``
    # therefore uses the same clipped-PG loss configuration as plain GRPO;
    # the distinction is entirely in Slime's rollout-side reward processor.
    # Supporting this label is required for 1-sample REINFORCE++ where the
    # custom processor supplies a batch-normalized scalar advantage.
    if args.advantage_estimator not in {
        "grpo",
        "gspo",
        "cispo",
        "policy_gradient",
    }:
        raise NotImplementedError(
            f"AutoModel backend currently supports clipped-PG advantages, got "
            f"{args.advantage_estimator!r}"
        )

    use_tis = bool(args.use_tis)
    cfg = ClippedPGLossConfig(
        ratio_clip_min=float(args.eps_clip),
        ratio_clip_max=float(args.eps_clip_high or args.eps_clip),
        ratio_clip_c=args.eps_clip_c,
        reference_policy_kl_penalty=0.0,
        reference_policy_kl_type=(
            "k3" if args.kl_loss_type == "low_var_kl" else args.kl_loss_type
        ),
        use_importance_sampling_correction=use_tis,
        truncated_importance_sampling_type="tis" if use_tis else None,
        truncated_importance_sampling_ratio=float(args.tis_clip)
        if use_tis
        else None,
        truncated_importance_sampling_ratio_min=float(args.tis_clip_low)
        if use_tis
        else None,
        token_level_loss=True,
        force_on_policy_ratio=False,
        use_cispo=args.advantage_estimator == "cispo",
        sequence_level_importance_ratios=args.advantage_estimator == "gspo",
    )
    return ClippedPGLossFn(cfg)


class AutoModelTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int:
        if args.debug_rollout_only:
            self.args = args
            return 0
        if role != "actor" or args.use_critic:
            raise NotImplementedError("AutoModel backend currently supports actor-only GRPO")
        if args.colocate or args.offload_train:
            raise NotImplementedError(
                "AutoModel backend first validates disaggregated train/rollout; "
                "colocation and Slime train offload are intentionally disabled"
            )
        if args.update_weight_transport != "nccl" or args.update_weight_mode != "full":
            raise ValueError("AutoModel backend requires full NCCL weight updates")

        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.reference_model_enabled = False
        self.with_opd_teacher = with_opd_teacher
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        args.rank = self._rank
        args.world_size = self._world_size

        policy_config = load_automodel_policy_config(args)
        from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
            DTensorPolicyWorkerV2Impl,
        )

        self.worker = DTensorPolicyWorkerV2Impl(
            policy_config,
            init_optimizer=True,
            init_reference_model=False,
        )
        # AutoModel owns initialization of the default process group, so the
        # Slime base actor does not get a chance to create its auxiliary Gloo
        # group.  The shared weight updater uses that CPU group for host-side
        # pause/drain barriers, exactly like Slime's native Megatron updater.
        from slime.utils.distributed_utils import init_gloo_group

        init_gloo_group()
        _audit_gemma4_automodel_runtime(self.worker, policy_config)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        self.loss_fn = build_loss(args)
        self.weight_updater = AutoModelWeightUpdater(args, self.worker)
        self.train_parallel_config = {
            "dp_size": self.worker.dp_size,
            "cp_size": self.worker.cp_size,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        self.pad_token_id = self.worker.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.worker.tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("AutoModel tokenizer has neither pad_token_id nor eos_token_id")

        if dist.get_rank() == 0:
            init_tracking(args, primary=False, role=role)
            logger.info(
                "AutoModel actor initialized: world=%d dp=%d tp=%d cp=%d "
                "dp_replicate=%d dp_shard=%d model=%s "
                "max_sequence=%d train_mb_tokens=%d",
                dist.get_world_size(),
                self.worker.dp_size,
                self.worker.tp_size,
                self.worker.cp_size,
                int(policy_config["dtensor_cfg"].get("dp_replicate_size", 1)),
                self.worker.dp_size
                // int(policy_config["dtensor_cfg"].get("dp_replicate_size", 1)),
                args.hf_checkpoint,
                policy_config["max_total_sequence_length"],
                policy_config["dynamic_batching"]["train_mb_tokens"],
            )
        return 0

    def _make_global_batch(self, rollout_data_ref: Box):
        """Rebuild the logical GBS before using NeMo's native DP sharder.

        Slime materializes one Ray object per DP shard. NeMo normally receives
        the logical global batch and calls ``BatchedDataDict.shard_by_batch_size``.
        Reconstructing it here lets the adapter use that exact implementation,
        including cross-DP dynamic-microbatch alignment metadata. The payload
        remains raw token tensors throughout.
        """
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        shard_batches = []
        terminal_proof_shards = []
        rollout_log_context = None
        for dp_rank in range(self.worker.dp_size):
            rollout_data = process_rollout_data(
                self.args,
                rollout_data_ref,
                dp_rank,
                self.worker.dp_size,
            )
            if rollout_log_context is None:
                raw_reward = rollout_data.get("raw_reward")
                rollout_log_context = {
                    # Slime intentionally ships raw_reward as one global
                    # diagnostic vector on every DP shard. Retain one copy so
                    # the backend logger matches native Megatron without
                    # multiplying or repartitioning that vector.
                    "raw_reward": (
                        list(raw_reward) if raw_reward is not None else None
                    ),
                    "global_batch_sizes": list(
                        rollout_data.get(
                            "global_batch_sizes", [self.args.global_batch_size]
                        )
                    ),
                    "has_rollout_log_probs": "rollout_log_probs" in rollout_data,
                }
            terminal_proof_shards.append(rollout_data)
            shard_batches.append(
                build_automodel_batch(
                    rollout_data,
                    pad_token_id=self.pad_token_id,
                    max_sequence_length=self.worker.cfg[
                        "max_total_sequence_length"
                    ],
                    # NeMo's dynamic sharder rounds each microbatch to this
                    # multiple. Pad the backing tensor to the same boundary so
                    # its longest row is not clipped back to a TP/SP-unsafe
                    # length by BatchedDataDict.shard_by_batch_size.
                    sequence_length_multiple=self.worker.cfg[
                        "dynamic_batching"
                    ]["sequence_length_round"],
                )
            )

        # Multi-turn trajectories can emit a non-DP-divisible number of
        # trainable pieces (for example, 13 pieces on DP4). Reconstruct the
        # logical batch first, then add only enough zero-loss rows to reach the
        # next DP multiple. NeMo's native sharder then redistributes the batch
        # evenly; preserving Slime's provisional shard boundaries here would
        # unnecessarily pad an uneven 13-piece batch as far as 20 rows.
        combined = concatenate_automodel_batches(
            shard_batches,
            pad_token_id=self.pad_token_id,
        )
        effective_samples = int(combined["sample_mask"].sum().item())
        unpadded_samples = int(combined["input_ids"].shape[0])
        padded_samples = (
            (unpadded_samples + self.worker.dp_size - 1) // self.worker.dp_size
        ) * self.worker.dp_size
        combined = pad_local_batch(
            combined,
            target_batch_size=padded_samples,
            pad_token_id=self.pad_token_id,
        )
        if dist.get_rank() == 0 and padded_samples != effective_samples:
            logger.info(
                "AutoModel DP alignment: effective_samples=%d padded_samples=%d "
                "per_dp_rank=%d",
                effective_samples,
                padded_samples,
                padded_samples // self.worker.dp_size,
            )
        if rollout_log_context is None:
            raise RuntimeError("AutoModel did not receive rollout logging context")
        from clawgym_nanobot.intentional_terminal_batch import validate_transport, RECEIPT
        receipt = validate_transport(self.args, terminal_proof_shards)
        if receipt is not None:
            rollout_log_context[RECEIPT] = receipt
        return BatchedDataDict(combined), rollout_log_context

    def _log_backend_rollout_metrics(
        self,
        rollout_id: int,
        batch,
        rollout_log_context: dict[str, Any],
    ) -> None:
        """Backend hook for native Slime actor-rollout metrics."""

    def _log_backend_perf_metrics(self, rollout_id: int) -> None:
        """Backend hook for native Slime actor performance metrics."""

    def _shard_global_batch(self, global_batch, *, max_tokens_per_microbatch: int):
        dynamic_batching_args = {
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_round": self.worker.cfg["dynamic_batching"][
                "sequence_length_round"
            ],
            "max_tokens_per_microbatch": int(max_tokens_per_microbatch),
        }
        shards, _ = global_batch.shard_by_batch_size(
            self.worker.dp_size,
            batch_size=None,
            dynamic_batching_args=dynamic_batching_args,
        )
        return shards[self.worker.dp_mesh.get_local_rank()]

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None
        with timer("data_preprocess"):
            global_batch, rollout_log_context = self._make_global_batch(
                rollout_data_ref
            )
            global_padded_batch_size = global_batch.size
            global_effective_batch_size = int(global_batch["sample_mask"].sum().item())
            global_trainable_tokens = int(
                global_batch["token_mask"].sum().item()
            )
            if global_trainable_tokens <= 0:
                receipt = rollout_log_context.get("researchclaw_terminal_mask_receipt")
                if receipt is not None:
                    if receipt["rollout_id"] != rollout_id or receipt["native_samples"] != global_padded_batch_size:
                        raise RuntimeError("Intentional terminal batch receipt disagrees with actor batch")
                    self._intentional_zero_skipped_batches = getattr(self, "_intentional_zero_skipped_batches", 0) + 1
                    if dist.get_rank() == 0:
                        step = compute_rollout_step(self.args, rollout_id)
                        logged = {"train/step": step,
                                  "train/intentional_zero_skipped_batches": self._intentional_zero_skipped_batches,
                                  "train/optimizer_updates_this_batch": 0,
                                  "train/nonzero_actor_train_batches_completed": getattr(self, "_nonzero_actor_train_batches_completed", 0),
                                  "train/intentional_zero_skipped_candidates": receipt["logical_candidates"],
                                  "train/intentional_zero_skipped_native_samples": receipt["native_samples"]}
                        logger.info("Skipping authenticated intentional self-reward terminal batch %d; no worker, optimizer, LR scheduler or weight-decay call", rollout_id)
                        log(self.args, logged, step_key="train/step")
                    return None
                raise RuntimeError(
                    "Refusing zero-token optimizer step: rollout produced no "
                    "trainable tokens (usually all Polar sessions timed out or "
                    "the rollout engine recovery did not restore dispatch)."
                )
            batch = self._shard_global_batch(
                global_batch,
                max_tokens_per_microbatch=self.worker.cfg["dynamic_batching"][
                    "logprob_mb_tokens"
                ],
            )

        # Slime's reward post-processor has already performed the exact GRPO
        # group normalization. Only old-policy logprobs need selecting here.
        if not self.args.use_rollout_logprobs:
            with timer("log_probs"):
                previous = self.worker.get_logprobs(data=batch)
            attach_old_policy_logprobs(
                batch,
                previous["logprobs"],
                has_rollout_log_probs=bool(
                    rollout_log_context["has_rollout_log_probs"]
                ),
            )

        train_mb_tokens = self.worker.cfg["dynamic_batching"]["train_mb_tokens"]
        logprob_mb_tokens = self.worker.cfg["dynamic_batching"][
            "logprob_mb_tokens"
        ]
        if train_mb_tokens != logprob_mb_tokens:
            train_batch = self._shard_global_batch(
                global_batch,
                max_tokens_per_microbatch=train_mb_tokens,
            )
            # Reusing positional scores after a different sharding is only safe
            # if token rows, lengths and masks are exactly unchanged.
            for key in ("input_ids", "input_lengths", "token_mask", "sample_mask"):
                if not torch.equal(train_batch[key], batch[key]):
                    raise RuntimeError("resharding changed row identity before old-policy attachment: " + key)
            train_batch["prev_logprobs"] = batch["prev_logprobs"]
            if not self.args.use_rollout_logprobs:
                train_batch["generation_logprobs"] = batch[
                    "generation_logprobs"
                ]
            batch = train_batch

        if self.args.use_kl_loss:
            if not getattr(self, "reference_model_enabled", False):
                raise RuntimeError(
                    "--use-kl-loss requires the actor reference model even when "
                    "--kl-loss-coef is zero"
                )
            with timer("ref_log_probs"):
                reference = self.worker.get_reference_policy_logprobs(data=batch)
            batch["reference_policy_logprobs"] = reference[
                "reference_logprobs"
            ]

        self._log_backend_rollout_metrics(
            rollout_id,
            batch,
            rollout_log_context,
        )

        with inverse_timer("train_wait"), timer("train"), timer("actor_train"):
            metrics = self.worker.train(
                batch,
                self.loss_fn,
                gbs=global_padded_batch_size,
                mbs=1,
            )

        self._nonzero_actor_train_batches_completed = getattr(self, "_nonzero_actor_train_batches_completed", 0) + 1
        slime_metrics = None
        if hasattr(self.loss_fn, "base_loss"):
            from megatron.core import parallel_state
            from slime.backends.nemo_megatron_utils.metrics import (
                reduce_nemo_dp_metrics,
            )

            slime_metrics = reduce_nemo_dp_metrics(
                metrics["all_mb_metrics"],
                data_parallel_group=parallel_state.get_data_parallel_group(),
                device=torch.cuda.current_device(),
            )

        if dist.get_rank() == 0:
            step = compute_rollout_step(self.args, rollout_id)
            if slime_metrics is None:
                logged = {
                    f"train/{key}": value.item()
                    if isinstance(value, torch.Tensor) and value.numel() == 1
                    else value
                    for key, value in metrics.items()
                }
                logged["train/step"] = step
            else:
                from slime.backends.nemo_megatron_utils.metrics import (
                    build_slime_train_log,
                )

                lr_values = metrics["all_mb_metrics"].get("lr", [self.args.lr])
                learning_rate = float(lr_values[-1])
                grad_norm = metrics["grad_norm"]
                if torch.is_tensor(grad_norm):
                    grad_norm = float(grad_norm.float().mean().item())
                logged = build_slime_train_log(
                    slime_metrics,
                    args=self.args,
                    grad_norm=float(grad_norm),
                    learning_rate=learning_rate,
                    global_batch_size=int(self.args.global_batch_size),
                    step=step,
                )
            logged["train/nonzero_actor_train_batches_completed"] = self._nonzero_actor_train_batches_completed
            logged["train/intentional_zero_skipped_batches"] = getattr(self, "_intentional_zero_skipped_batches", 0)
            logger.info(
                "AutoModel train %s: %s effective_samples=%d padded_samples=%d",
                rollout_id,
                logged,
                global_effective_batch_size,
                global_padded_batch_size,
            )
            log(self.args, logged, step_key="train/step")
        self._log_backend_perf_metrics(rollout_id)
        return None

    def sleep(self, tags=None) -> None:
        return None

    def wake_up(self, tags=None) -> None:
        return None

    def clear_memory(self) -> None:
        torch.cuda.empty_cache()

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # ProRL experiments default to no checkpoints; keep the interface while
        # making the no-save behavior explicit for AutoModel.
        return None

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        # Match Slime's Megatron actor: the rollout health monitor marks
        # unresponsive engines as dead during generation, and rank 0 rebuilds
        # them before every refit.  Synchronize the train ranks through the
        # CPU/Gloo group so a CUDA communicator is not consumed by control
        # flow before AutoModelWeightUpdater reconnects the replacement.
        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            _engine_gpu_offsets,
            _all_engine_actors,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())
        if not rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable SGLang engines; skipping AutoModel refit")
            return
        if num_new_engines > 0 or not self.weight_updater.rollout_engines:
            self.weight_updater.connect(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
            )
            dist.barrier()
        self.weight_updater.update()
        if num_new_engines > 0:
            # All train ranks must finish the refit before rank 0 exposes a
            # recovered engine to router traffic.
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.activate_new_updatable_engines.remote())
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())
            dist.barrier(group=get_gloo_group())

    def _get_parallel_config(self):
        return self.train_parallel_config
