"""Slime entry point for NeMo/Megatron-Bridge's official Gemma4 model.

Slime owns RL orchestration, DDP, the optimizer, HF checkpoint loading, and
online weight export.  Model semantics (including E-series PLE, shared KV,
dual RoPE, and tensor-parallel KV replication) come from Megatron-Bridge.  The
only model component replaced here is core attention: Gemma4 global heads are
512 wide on E2B/E4B, so 64K training needs an exact query-chunked kernel rather
than a materialized full attention matrix.
"""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path

import torch


def _install_mcore_dual_rope_checkpoint_compat() -> None:
    """Flatten dual-RoPE tuples before entering MCore's re-entrant checkpoint.

    Current Megatron-Core already implements this exact flatten/reconstruct
    contract in ``megatron.core.recompute.checkpointed_forward`` because
    ``torch.autograd.Function.save_for_backward`` accepts tensors/None but not
    the two-tensor RoPE tuple used by Gemma4.  The Bridge PLE checkpoint helper
    predates that MCore fix and still forwards the tuple as one argument.  Keep
    Bridge's model/recompute implementation intact and adapt only that missing
    call-site contract here.
    """

    from megatron.core import tensor_parallel

    original_checkpoint = tensor_parallel.checkpoint
    if getattr(original_checkpoint, "_slime_dual_rope_tuple_compat", False):
        return

    @wraps(original_checkpoint)
    def checkpoint_with_flat_tensor_tuples(function, distribute_saved_activations, *args):
        layout: list[tuple[str, int]] = []
        flat_args: list[object] = []
        has_tensor_tuple = False

        for arg in args:
            if isinstance(arg, (tuple, list)) and all(
                item is None or torch.is_tensor(item) for item in arg
            ):
                has_tensor_tuple = True
                layout.append(("tuple" if isinstance(arg, tuple) else "list", len(arg)))
                flat_args.extend(arg)
            else:
                layout.append(("item", 1))
                flat_args.append(arg)

        if not has_tensor_tuple:
            return original_checkpoint(function, distribute_saved_activations, *args)

        def call_with_reconstructed_tuples(*checkpoint_args):
            rebuilt: list[object] = []
            offset = 0
            for kind, width in layout:
                values = checkpoint_args[offset : offset + width]
                offset += width
                if kind == "tuple":
                    rebuilt.append(tuple(values))
                elif kind == "list":
                    rebuilt.append(list(values))
                else:
                    rebuilt.append(values[0])
            return function(*rebuilt)

        return original_checkpoint(
            call_with_reconstructed_tuples,
            distribute_saved_activations,
            *flat_args,
        )

    checkpoint_with_flat_tensor_tuples._slime_dual_rope_tuple_compat = True
    checkpoint_with_flat_tensor_tuples._slime_original_checkpoint = original_checkpoint
    tensor_parallel.checkpoint = checkpoint_with_flat_tensor_tuples


def _require_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_e_parallelism(provider, *, tp_size: int) -> None:
    """Validate TP sharding/replication contracts used by dense Gemma4 E.

    Megatron supports fewer KV groups than TP ranks by replicating each KV
    group across an integral number of ranks.  That is the important
    difference from the old Slime plugin, which required TP to divide the KV
    head count and therefore rejected E4B TP4.
    """

    num_attention_heads = _require_positive_int("num_attention_heads", provider.num_attention_heads)
    if num_attention_heads % tp_size:
        raise ValueError(
            f"Gemma4 TP={tp_size} must divide num_attention_heads={num_attention_heads}"
        )

    for field in ("num_query_groups", "num_global_query_groups"):
        groups = _require_positive_int(field, getattr(provider, field))
        if groups % tp_size and tp_size % groups:
            raise ValueError(
                f"Gemma4 TP={tp_size} and {field}={groups} require integral "
                "partitioning or integral KV replication"
            )

    ple_width = int(provider.num_layers) * int(provider.per_layer_embed_dim)
    if ple_width % tp_size:
        raise ValueError(
            f"Gemma4 packed PLE width={ple_width} must be divisible by TP={tp_size}"
        )


def _normalize_dense_provider_from_hf(provider, hf_config) -> None:
    """Repair fields lost by ConditionalGeneration -> text config mapping."""

    hf_text_config = getattr(hf_config, "text_config", hf_config)
    if getattr(provider, "num_global_query_groups", None) is None:
        global_groups = getattr(hf_text_config, "num_global_key_value_heads", None)
        provider.num_global_query_groups = int(global_groups or provider.num_query_groups)


def _configure_dense_provider_for_slime(provider, args, *, tp_size: int):
    """Apply only Slime runtime overrides to NeMo's dense provider."""

    _validate_e_parallelism(provider, tp_size=tp_size)
    provider.tensor_model_parallel_size = tp_size
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = bool(args.sequence_parallel)
    provider.seq_length = int(args.seq_length)
    provider.variable_seq_lengths = bool(getattr(args, "variable_seq_lengths", False))
    provider.moe_token_dispatcher_type = getattr(
        args,
        "moe_token_dispatcher_type",
        provider.moe_token_dispatcher_type,
    )
    provider.perform_initialization = False
    provider.gradient_accumulation_fusion = False
    provider.masked_softmax_fusion = False
    provider.attention_backend = None
    provider.transformer_impl = "local"
    provider.bf16 = True
    provider.fp16 = False
    provider.params_dtype = torch.bfloat16
    provider.autocast_dtype = torch.bfloat16

    provider.recompute_granularity = getattr(args, "recompute_granularity", None)
    provider.recompute_method = getattr(args, "recompute_method", None)
    provider.recompute_num_layers = getattr(args, "recompute_num_layers", None)
    provider.calculate_per_token_loss = bool(getattr(args, "calculate_per_token_loss", False))
    provider.finalize()
    return provider


def gemma4_official_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
):
    from megatron.training.global_vars import get_args

    args = get_args()
    _install_mcore_dual_rope_checkpoint_compat()
    tp_size = _require_positive_int("tensor_model_parallel_size", args.tensor_model_parallel_size)
    if int(args.pipeline_model_parallel_size) != 1:
        raise NotImplementedError("the official Gemma4 Dense provider currently requires PP=1")
    if int(args.context_parallel_size) != 1:
        raise NotImplementedError("the 64k Gemma4 attention adapter currently requires CP=1")
    if vp_stage is not None:
        raise NotImplementedError("virtual pipeline parallelism is not used with Gemma4 PP1")

    from megatron.core.models.backends import LocalSpecProvider

    from scripts.trx_scripts.gemma4_official_memory_efficient_attention import (
        Gemma4MemoryEfficientAttention,
    )

    LocalSpecProvider.core_attention = lambda self: Gemma4MemoryEfficientAttention

    model_view_value = os.environ.get("GEMMA4_OFFICIAL_MODEL_VIEW") or getattr(args, "hf_checkpoint", None)
    if not model_view_value:
        raise ValueError("GEMMA4_OFFICIAL_MODEL_VIEW or --hf-checkpoint is required")
    model_view = Path(model_view_value).resolve()
    if not (model_view / "config.json").is_file():
        raise FileNotFoundError(f"missing official Gemma4 text view: {model_view}")

    if os.environ.get("GEMMA4_CONVERSION_MODE") != "text":
        raise RuntimeError("official Gemma4 E provider requires GEMMA4_CONVERSION_MODE=text")
    from slime.utils.megatron_bridge_utils import (
        auto_bridge_from_hf_pretrained,
        record_auto_bridge_audit,
    )

    bridge = auto_bridge_from_hf_pretrained(
        str(model_view),
        purpose="custom_model_provider_base",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    provider = bridge.to_megatron_provider(load_weights=False)
    # The ConditionalGeneration -> text conversion selects the correct dense
    # provider, but the generic HF field mapper in some Bridge revisions can
    # overwrite this Gemma4-specific value with None. Restore it from the
    # authoritative text config before validating TP/KV replication.
    _normalize_dense_provider_from_hf(provider, bridge.hf_pretrained.config)
    _configure_dense_provider_for_slime(provider, args, tp_size=tp_size)
    record_auto_bridge_audit(
        bridge,
        str(model_view),
        "custom_model_provider_finalized",
        provider=provider,
    )
    return provider.provide(
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
