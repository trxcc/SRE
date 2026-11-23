"""Slime actor using NeMo-RL's native Megatron worker and Gemma bridge.

This module deliberately contains no model implementation or weight mapping.
NeMo-RL owns model construction, parallelism, optimizer, forward/backward, and
Megatron-Bridge HF export.  The adapter only converts Slime rollout tensors to
``BatchedDataDict`` and hands already-exported HF tensors to Slime's SGLang
transport.
"""

from __future__ import annotations

import logging
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from slime.backends.automodel_utils.actor import (
    AutoModelTrainRayActor,
    _make_optional_logger_imports_safe,
)
from slime.utils.logging_utils import init_tracking
from slime.utils.timer import Timer, with_defer

from .loss import build_loss
from .metrics import SlimeAlignedClippedPGLoss
from .update_weight import NemoMegatronWeightUpdater

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1', got {value!r}")
    return value == "1"


def load_nemo_megatron_policy_config(args: Namespace) -> dict[str, Any]:
    """Load an upstream NeMo recipe and apply only run/topology overrides."""

    if not args.automodel_config:
        raise ValueError(
            "--train-backend nemo_megatron requires --automodel-config"
        )

    _make_optional_logger_imports_safe()
    if os.environ.get("CLAWGYM_NATIVE_MEMORY_EFFICIENT_ATTENTION", "0") == "1":
        from clawgym_nanobot.native_memory_attention import install_native_core_binding
        install_native_core_binding()
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(
        load_config(args.automodel_config), resolve=True
    )
    policy = dict(config["policy"])

    # Point AutoBridge at the original ConditionalGeneration checkpoint.
    # GEMMA4_CONVERSION_MODE=text is an upstream Megatron-Bridge feature which
    # selects Gemma4DenseProvider and its official language-only mappings.
    policy["model_name"] = args.hf_checkpoint
    reference_model_name = str(getattr(args, "ref_load", "") or "").strip()
    if reference_model_name:
        policy["reference_model_name"] = reference_model_name
    policy["tokenizer"] = dict(policy["tokenizer"])
    policy["tokenizer"]["name"] = args.hf_checkpoint
    policy.pop("pretrained_checkpoint", None)
    generation = dict(policy.get("generation") or {})
    generation.update(
        temperature=float(args.rollout_temperature),
        top_p=float(args.rollout_top_p),
        top_k=None if int(args.rollout_top_k) <= 0 else int(args.rollout_top_k),
    )
    policy["generation"] = generation
    policy["max_total_sequence_length"] = int(args.seq_length)
    policy["max_grad_norm"] = float(args.clip_grad)
    policy["train_global_batch_size"] = int(args.global_batch_size)
    policy["train_micro_batch_size"] = 1
    policy["logprob_batch_size"] = 1
    policy["offload_optimizer_for_logprob"] = False
    logprob_chunk_size = int(args.log_probs_chunk_size)
    if logprob_chunk_size > 0:
        # Use NeMo-RL's upstream ChunkedDistributedLogprob implementation.  In
        # particular, keep Megatron's full logits in their native dtype and do
        # the numerically sensitive log-softmax in bounded fp32 row chunks.
        policy["logprob_chunk_size"] = logprob_chunk_size

    policy["dtensor_cfg"] = {"enabled": False}
    megatron_cfg = dict(policy["megatron_cfg"])
    megatron_cfg["enabled"] = True
    megatron_cfg["tensor_model_parallel_size"] = int(
        args.tensor_model_parallel_size
    )
    megatron_cfg["pipeline_model_parallel_size"] = int(
        args.pipeline_model_parallel_size
    )
    cp_size = int(args.context_parallel_size)
    megatron_cfg["context_parallel_size"] = cp_size
    megatron_cfg["sequence_parallel"] = (
        int(args.tensor_model_parallel_size) > 1
    )
    megatron_cfg["activation_checkpointing"] = True
    # Keep activation recompute on NeMo-RL's native configuration path.  The
    # adapter must not invent a Gemma-specific selective module list: NeMo's
    # official Megatron recipes use full recompute, and callers may still
    # select another upstream-supported granularity through Slime's CLI.
    megatron_cfg["recompute_granularity"] = str(
        getattr(args, "recompute_granularity", "full") or "full"
    )
    megatron_cfg["recompute_modules"] = None
    # Keep the generic NeMo-RL GRPO defaults from overwriting the official
    # Gemma4 Bridge recipe.  Gemma4 uses its local transformer spec and
    # fast_gelu; the TE bias-activation and RoPE fusions are not valid for it.
    megatron_cfg["transformer_impl"] = "local"
    megatron_cfg["bias_activation_fusion"] = False
    megatron_cfg["apply_rope_fusion"] = False
    megatron_cfg["gradient_accumulation_fusion"] = False
    ddp_config = dict(megatron_cfg["distributed_data_parallel_config"])
    # Gemma4's text path does not exercise every parameter registered by the
    # conditional-generation bridge.  Megatron's overlap-grad hook learns a
    # fixed ready-count map from the first backward and asserts on the second
    # iteration when that map is incomplete.  NeMo-RL's own VLM and agentic
    # Megatron recipes disable overlap_grad_reduce for the same dynamic-model
    # contract while retaining the distributed optimizer and param-gather
    # overlap.  Keep this as a recipe/config choice rather than patching MCore.
    ddp_config["overlap_grad_reduce"] = _env_flag(
        "NEMO_MEGATRON_OVERLAP_GRAD_REDUCE", False
    )
    megatron_cfg["distributed_data_parallel_config"] = ddp_config
    if logprob_chunk_size > 0:
        # Required by NeMo-RL's own chunking config validator and prevents a
        # full [tokens, vocab/TP] bf16->fp32 conversion before chunking.
        megatron_cfg["defer_fp32_logits"] = True
    model_overrides = dict(megatron_cfg.get("model_overrides") or {})
    model_overrides.update(
        {
            "masked_softmax_fusion": False,
            "cross_entropy_loss_fusion": True,
            "cross_entropy_fusion_impl": "native",
        }
    )
    megatron_cfg["model_overrides"] = model_overrides
    megatron_cfg["cuda_graph_impl"] = "none"
    megatron_cfg["empty_unused_memory_level"] = 1
    # NeMo-RL's GRPO entrypoint injects max_num_steps immediately before it
    # constructs MegatronPolicyWorkerImpl.  Slime owns that outer loop here,
    # so mirror the same upstream hand-off using Slime's rollout horizon.
    megatron_cfg["train_iters"] = max(1, int(args.num_rollout))
    # ProRL runs disable checkpoints.  Do not initialize NeMo's optional NVRx
    # async checkpoint worker merely because the borrowed generic recipe has
    # async_save enabled by default.
    checkpoint = dict(megatron_cfg.get("checkpoint") or {})
    checkpoint["async_save"] = False
    checkpoint["ckpt_assume_constant_structure"] = False
    megatron_cfg["checkpoint"] = checkpoint

    optimizer = dict(megatron_cfg["optimizer"])
    optimizer.update(
        {
            "lr": float(args.lr),
            "min_lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "adam_beta1": float(args.adam_beta1),
            "adam_beta2": float(args.adam_beta2),
            "adam_eps": float(args.adam_eps),
            "clip_grad": float(args.clip_grad),
            "optimizer_cpu_offload": False,
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": False,
        }
    )
    megatron_cfg["optimizer"] = optimizer

    scheduler = dict(megatron_cfg["scheduler"])
    scheduler.update(
        {
            "start_weight_decay": float(args.weight_decay),
            "end_weight_decay": float(args.weight_decay),
            "weight_decay_incr_style": "constant",
            "lr_decay_style": "constant",
            "lr_decay_iters": max(1, int(args.num_rollout)),
            "lr_warmup_iters": 0,
            "lr_warmup_init": float(args.lr),
        }
    )
    megatron_cfg["scheduler"] = scheduler
    policy["megatron_cfg"] = megatron_cfg

    token_cap = int(args.max_tokens_per_gpu) * cp_size
    # MCore requires PackedSeqParams whenever context parallelism is enabled,
    # and NeMo-RL models that path through sequence packing.  Dynamic batching
    # and sequence packing are mutually exclusive in NeMo-RL, so retain the
    # proven dynamic-batching path for CP1 and switch only CP>1 runs to the
    # packing path required by MCore.
    use_sequence_packing = cp_size > 1
    policy["dynamic_batching"] = {
        "enabled": not use_sequence_packing,
        "train_mb_tokens": token_cap,
        "logprob_mb_tokens": token_cap,
        "sequence_length_round": 64,
    }
    policy["sequence_packing"] = {
        "enabled": use_sequence_packing,
        "train_mb_tokens": token_cap,
        "logprob_mb_tokens": token_cap,
        "algorithm": "modified_first_fit_decreasing",
        "microbatch_order": "packer",
        "sequence_length_round": 64,
    }
    policy["make_sequence_length_divisible_by"] = max(
        int(args.tensor_model_parallel_size), 2 * cp_size
    )

    model_parallel = (
        megatron_cfg["tensor_model_parallel_size"]
        * megatron_cfg["pipeline_model_parallel_size"]
        * megatron_cfg["context_parallel_size"]
    )
    if model_parallel < 1 or int(args.world_size) % model_parallel:
        raise ValueError(
            f"NeMo Megatron TP*PP*CP={model_parallel} must divide actor world "
            f"size {args.world_size}"
        )
    # Megatron-Bridge resolves its validation GBS from the configured training
    # GBS during worker construction and requires both to be divisible by DP.
    # Slime's agent rollouts can yield a smaller logical debug GBS; the shared
    # AutoModel glue below already pads that batch with zero-loss rows before
    # calling the worker.  Give NeMo the matching padded size while retaining
    # Slime's logical rollout GBS in ``args.global_batch_size``.
    data_parallel_size = int(args.world_size) // model_parallel
    logical_gbs = int(args.global_batch_size)
    policy["train_global_batch_size"] = (
        (logical_gbs + data_parallel_size - 1) // data_parallel_size
    ) * data_parallel_size
    return policy


class NemoMegatronTrainRayActor(AutoModelTrainRayActor):
    """Slime lifecycle around the unmodified NeMo Megatron policy worker."""

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
            raise NotImplementedError(
                "NeMo Megatron adapter currently supports actor-only GRPO"
            )
        if args.colocate or args.offload_train:
            raise NotImplementedError(
                "The NeMo Megatron adapter currently supports disaggregated "
                "train/rollout without Slime train offload"
            )
        if args.update_weight_transport != "nccl" or args.update_weight_mode != "full":
            raise ValueError(
                "NeMo Megatron adapter requires full NCCL weight updates"
            )

        from slime.agent.token_contract import validate_fresh_args
        validate_fresh_args(args)
        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher
        args.rank = self._rank
        args.world_size = self._world_size
        policy_config = load_nemo_megatron_policy_config(args)

        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.distributed.named_sharding import NamedSharding
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        megatron_cfg = policy_config["megatron_cfg"]
        tp_size = int(megatron_cfg["tensor_model_parallel_size"])
        pp_size = int(megatron_cfg["pipeline_model_parallel_size"])
        cp_size = int(megatron_cfg["context_parallel_size"])
        sharding = NamedSharding(
            layout=np.arange(args.world_size).reshape(
                pp_size, -1, cp_size, tp_size
            ),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )
        tokenizer = get_tokenizer(policy_config["tokenizer"])
        needs_reference_model = bool(with_ref or args.use_kl_loss)
        self.reference_model_enabled = needs_reference_model
        self.worker = MegatronPolicyWorkerImpl(
            policy_config,
            tokenizer,
            init_optimizer=True,
            init_reference_model=needs_reference_model,
            worker_sharding_annotations=sharding,
        )
        # NeMo owns initialization of the default NCCL process group.  Create
        # the same auxiliary Gloo group used by Slime's native Megatron weight
        # updater so long SGLang drain waits never occupy an NCCL collective.
        from slime.utils.distributed_utils import init_gloo_group

        init_gloo_group()
        from megatron.core.utils import get_model_config

        runtime_model_cfg = get_model_config(self.worker.model)
        expected_recompute = str(megatron_cfg["recompute_granularity"])
        actual_recompute = getattr(runtime_model_cfg, "recompute_granularity", None)
        if actual_recompute != expected_recompute:
            raise RuntimeError(
                "NeMo Megatron recompute audit failed: "
                f"configured={expected_recompute!r}, runtime={actual_recompute!r}"
            )
        # These topology attributes are adapter conveniences only; the worker's
        # process groups and model topology remain entirely NeMo/Megatron-owned.
        self.worker.tp_size = tp_size
        self.worker.cp_size = cp_size
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        native_loss = build_loss(args)
        if native_loss.reference_policy_kl_penalty > 0:
            reference_state = getattr(self.worker, "reference_state_dict", None)
            if not needs_reference_model or reference_state is None or len(reference_state) == 0:
                raise RuntimeError("Nonzero reference KL requires initialized native reference state")
            if policy_config.get("reference_model_name") != args.ref_load:
                raise RuntimeError("Native reference source differs from --ref-load")
            logger.info(
                "reference_kl_enabled coefficient=%s reference_source=%s reference_state_present=True",
                native_loss.reference_policy_kl_penalty,
                args.ref_load,
            )
        if bool(args.use_tis) != bool(native_loss.use_importance_sampling_correction):
            raise RuntimeError("TIS configuration mismatch in native loss")
        actual_sampling = self.worker.sampling_params
        if actual_sampling is None or actual_sampling.temperature != args.rollout_temperature:
            raise RuntimeError("Training and rollout temperature mismatch")
        self.loss_fn = SlimeAlignedClippedPGLoss(native_loss, args)
        self.weight_updater = NemoMegatronWeightUpdater(args, self.worker)
        self.train_parallel_config = {
            "dp_size": self.worker.dp_size,
            "cp_size": cp_size,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }
        self.pad_token_id = self.worker.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.worker.tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError(
                "NeMo tokenizer has neither pad_token_id nor eos_token_id"
            )

        if dist.get_rank() == 0:
            init_tracking(args, primary=False, role=role)
            logger.info(
                "NeMo Megatron actor initialized: world=%d dp=%d tp=%d pp=%d "
                "cp=%d recompute=%s/%s/%s model=%s max_sequence=%d "
                "dynamic_mb_tokens=%d logprob_chunk_size=%s "
                "defer_fp32_logits=%s reference_model=%s metrics=slime_dp_reduced",
                dist.get_world_size(),
                self.worker.dp_size,
                tp_size,
                pp_size,
                cp_size,
                actual_recompute,
                getattr(runtime_model_cfg, "recompute_method", None),
                getattr(runtime_model_cfg, "recompute_num_layers", None),
                args.hf_checkpoint,
                policy_config["max_total_sequence_length"],
                policy_config["dynamic_batching"]["train_mb_tokens"],
                policy_config.get("logprob_chunk_size"),
                megatron_cfg.get("defer_fp32_logits"),
                needs_reference_model,
            )
        return 0

    def _shard_global_batch(
        self, global_batch, *, max_tokens_per_microbatch: int
    ):
        dynamic_cfg = self.worker.cfg["dynamic_batching"]
        if not dynamic_cfg["enabled"]:
            raise ValueError(
                "NeMo Megatron Slime glue requires NeMo dynamic batching"
            )

        # Match NeMo-RL LMPolicy._shard_for_logprob/_shard_for_train exactly:
        # the upstream BatchedDataDict sharder owns dynamic batching and emits
        # micro_batch_indices/micro_batch_lengths/elem_counts_per_gb metadata
        # consumed by MegatronPolicyWorkerImpl.
        dynamic_batching_args = {
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_round": dynamic_cfg["sequence_length_round"],
            "max_tokens_per_microbatch": int(max_tokens_per_microbatch),
        }

        # ``global_batch`` is already padded to a DP multiple. NeMo's
        # ``batch_size`` means the global-batch chunk size, not a per-rank
        # size, so None is the native one-global-batch form used for logprobs.
        shards, _ = global_batch.shard_by_batch_size(
            self.worker.dp_size,
            batch_size=None,
            dynamic_batching_args=dynamic_batching_args,
        )
        from megatron.core import parallel_state

        return shards[parallel_state.get_data_parallel_rank()]

    def _log_backend_rollout_metrics(
        self,
        rollout_id: int,
        batch,
        rollout_log_context: dict[str, Any],
    ) -> None:
        from slime.backends.megatron_utils.data import log_rollout_data
        from slime.backends.megatron_utils.loss import (
            compute_advantages_and_returns,
        )
        from slime.backends.nemo_megatron_utils.metrics import (
            build_native_rollout_log_data,
        )

        rollout_data = build_native_rollout_log_data(
            batch,
            raw_reward=rollout_log_context.get("raw_reward"),
            global_batch_sizes=rollout_log_context["global_batch_sizes"],
            has_rollout_log_probs=bool(
                rollout_log_context.get("has_rollout_log_probs")
            ),
            include_actor_log_probs=(
                not self.args.use_rollout_logprobs
                or bool(getattr(self.args, "get_mismatch_metrics", False))
            ),
            include_reference_log_probs=bool(
                getattr(self, "reference_model_enabled", False)
            ),
        )
        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        # Native perf logging reads the local sequence lengths from Timer.
        # The AutoModel glue reconstructs then reshards the logical batch, so
        # replace process_rollout_data's last-shard value with this rank's
        # actual non-padding NeMo shard.
        Timer().seq_lens = [
            int(length)
            for length, keep in zip(
                batch["input_lengths"], batch["sample_mask"], strict=True
            )
            if bool(keep)
        ]

    def _log_backend_perf_metrics(self, rollout_id: int) -> None:
        from slime.backends.megatron_utils.data import log_perf_data

        log_perf_data(
            rollout_id,
            self.args,
            extra_metrics=self.weight_updater.pop_metrics(),
        )

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Export an atomic HF checkpoint through Slime's official Bridge saver.

        The NeMo adapter intentionally does not write an MCore ``torch_dist``
        checkpoint here.  Formal ClawGym runs only retain HF model weights; on
        resume the optimizer/scheduler are reinitialized explicitly.
        """

        del force_sync  # NeMo value saves and HF export are synchronous.
        if self.args.debug_rollout_only:
            return
        if self.role == "critic":
            if self.args.save_hf is not None:
                pass
            elif self.args.save is None:
                return
            else:
                self.worker.mcore_state.train_state.step = int(rollout_id) + 1
                self.worker.save_checkpoint(
                    weights_path=self.args.save,
                    optimizer_path=None if bool(self.args.no_save_optim) else self.args.save,
                )
                return
        if self.args.save_hf is None:
            return

        from slime.backends.megatron_utils.hf_checkpoint_saver import (
            save_hf_critic_to_path,
            save_hf_model_to_path,
        )

        requested = Path(
            self.args.save_hf.format(rollout_id=rollout_id)
        ).resolve()
        origin = Path(self.args.hf_checkpoint).resolve()
        if requested == origin:
            raise ValueError("HF save output must differ from --hf-checkpoint")

        incomplete = requested.with_name(f".{requested.name}.incomplete")
        if dist.get_rank() == 0:
            shutil.rmtree(incomplete, ignore_errors=True)
        dist.barrier()

        # This is the same official Slime save path used by the native
        # Megatron actor.  Passing a one-element model list matches
        # Megatron-Bridge's export contract and preserves NeMo's model/provider
        # implementation and mappings.
        if self.role == "critic":
            save_hf_critic_to_path(self.args, incomplete, [self.worker.model])
        else:
            save_hf_model_to_path(
                self.args,
                incomplete,
                [self.worker.model],
                progress_desc=f"Save HF checkpoint rollout {rollout_id}",
            )
        dist.barrier()

        if dist.get_rank() == 0:
            if self.args.rollout_global_dataset:
                from slime.utils.atomic_rollout_cursor import bundle_cursor
                bundle_cursor(incomplete, self.args.save or requested, rollout_id)
            shutil.rmtree(requested, ignore_errors=True)
            requested.parent.mkdir(parents=True, exist_ok=True)
            os.replace(incomplete, requested)
            marker = requested / "checkpoint.json"
            temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
            checkpoint_format = "huggingface_critic" if self.role == "critic" else "huggingface"
            temporary.write_text(
                f'{{"format":"{checkpoint_format}","optimizer_saved":false,'
                f'"rollout_id":{int(rollout_id)}}}\n',
                encoding="utf-8",
            )
            os.replace(temporary, marker)
        dist.barrier()
        if dist.get_rank() == 0 and "{rollout_id}" in self.args.save_hf:
            from clawgym_nanobot.checkpoint_retention import (
                finalize_latest_hf_checkpoint,
            )

            retention = finalize_latest_hf_checkpoint(rollout_id, self.args)
            if retention.get("enabled"):
                logger.info("Latest HF checkpoint retention: %s", retention)
