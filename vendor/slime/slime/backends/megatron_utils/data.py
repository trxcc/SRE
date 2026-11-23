import logging
from argparse import Namespace
from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from slime.utils import train_metric_utils
from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step
from slime.utils.ppo_utils import compute_approx_kl
from slime.utils.types import RolloutBatch
from slime.utils.algorithm_metrics import derive_algorithm_metrics

from ...utils import logging_utils
from .cp_utils import (
    all_reduce_log_dict,
    gather_and_reduce_log_dict,
    get_local_response_masks,
    get_sum_of_sample_mean,
    rollout_log_metric_contribution,
    slice_with_cp,
    slice_text_position_ids_with_cp,
)

logger = logging.getLogger(__name__)


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    allgather_cp: bool = False,
    build_text_mrope_position_ids: bool = False,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """
    Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    batch = data_iterator.get_next(keys)

    tokens = batch["tokens"]
    token_lengths = [int(token.size(0)) for token in tokens]
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier

    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    if allgather_cp:
        # DSA mode: concatenate all sequences first, then slice once with CP.
        # We also pad the *global* concatenated stream to make per-rank chunks equal.
        cu_seqlens_list: list[int] = [0]
        for t in tokens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

        tokens = torch.cat(tokens, dim=0)
        if build_text_mrope_position_ids:
            full_position_ids = torch.cat(
                [torch.arange(length, dtype=torch.long, device=tokens.device) for length in token_lengths]
            )

        # Pad global stream so (1) divisible by cp_size (equal chunks),
        # (2) divisible by pad_size (reduce fragmentation).
        global_pad_size = cp_size * pad_size
        pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens_list.append(cu_seqlens_list[-1] + pad)
            if build_text_mrope_position_ids:
                full_position_ids = torch.cat(
                    [
                        full_position_ids,
                        torch.arange(pad, dtype=torch.long, device=tokens.device),
                    ]
                )

        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
        tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        if build_text_mrope_position_ids:
            local_position_ids = full_position_ids.chunk(cp_size, dim=0)[cp_rank]
    else:
        tokens = [slice_with_cp(t, pad_token_id) for t in tokens]
        if build_text_mrope_position_ids:
            local_position_ids_parts = [
                slice_text_position_ids_with_cp(
                    length,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    device=tokens[0].device,
                )
                for length in token_lengths
            ]

        cu_seqlens = [0]
        for t in tokens:
            cu_seqlens.append(cu_seqlens[-1] + t.size(0))

        tokens = torch.cat(tokens)

        # Always pad to reduce memory fragmentation and maybe make the computation faster
        pad = (pad_size - tokens.size(0) % pad_size) % pad_size
        if pad != 0:
            tokens = F.pad(tokens, (0, pad), value=pad_token_id)
            cu_seqlens.append(cu_seqlens[-1] + pad)
            if build_text_mrope_position_ids:
                # The final allocator-alignment padding is represented as an
                # independent packed segment.  Its physical length is
                # ``cp_size * pad``; select this rank's zig-zag positions just
                # like every ordinary sample.
                local_position_ids_parts.append(
                    slice_text_position_ids_with_cp(
                        cp_size * pad,
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        device=tokens.device,
                    )
                )

        # thd requires the cu_seqlens to be of the origin length
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int).cuda() * cp_size

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )

    tokens = tokens.unsqueeze(0)

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params
    if build_text_mrope_position_ids:
        if not allgather_cp:
            local_position_ids = torch.cat(local_position_ids_parts)
        if local_position_ids.numel() != tokens.size(1):
            raise ValueError(
                "Qwen text MRoPE position length does not match the CP-local token row: "
                f"{local_position_ids.numel()} != {tokens.size(1)}"
            )
        batch["position_ids"] = (
            local_position_ids.view(1, 1, -1).expand(3, 1, -1).contiguous()
        )
    else:
        batch["position_ids"] = None

    # Full-vocabulary OPSD performs a second, no-grad forward over a
    # feedback-conditioned prompt followed by the exact same response. Keep
    # it as an independent packed stream so teacher tokens cannot attend to a
    # student sequence (or another sample) and so teacher activations are not
    # retained by the student autograd graph.
    opsd_teacher_tokens = batch.get("opsd_teacher_tokens")
    if opsd_teacher_tokens is not None:
        if cp_size != 1 or allgather_cp:
            raise ValueError("full-vocabulary OPSD paired packing currently requires context_parallel_size=1")
        if len(opsd_teacher_tokens) != len(batch["unconcat_tokens"]):
            raise ValueError(
                "OPSD teacher/student microbatch size mismatch: "
                f"{len(opsd_teacher_tokens)} != {len(batch['unconcat_tokens'])}"
            )
        batch["opsd_unconcat_teacher_tokens"] = opsd_teacher_tokens
        teacher_total_lengths = [int(value.size(0)) for value in opsd_teacher_tokens]
        teacher_chunks = [slice_with_cp(value, pad_token_id) for value in opsd_teacher_tokens]
        teacher_cu_seqlens = [0]
        for value in teacher_chunks:
            teacher_cu_seqlens.append(teacher_cu_seqlens[-1] + value.size(0))
        teacher_tokens = torch.cat(teacher_chunks)
        teacher_pad = (pad_size - teacher_tokens.size(0) % pad_size) % pad_size
        if teacher_pad:
            teacher_tokens = F.pad(teacher_tokens, (0, teacher_pad), value=pad_token_id)
            teacher_cu_seqlens.append(teacher_cu_seqlens[-1] + teacher_pad)
        teacher_cu_seqlens_tensor = torch.tensor(
            teacher_cu_seqlens,
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        teacher_max_seqlen = (
            teacher_cu_seqlens_tensor[1:] - teacher_cu_seqlens_tensor[:-1]
        ).max().item()
        batch["opsd_teacher_tokens"] = teacher_tokens.unsqueeze(0)
        batch["opsd_teacher_total_lengths"] = teacher_total_lengths
        batch["opsd_teacher_packed_seq_params"] = PackedSeqParams(
            cu_seqlens_q=teacher_cu_seqlens_tensor,
            cu_seqlens_kv=teacher_cu_seqlens_tensor,
            max_seqlen_q=teacher_max_seqlen,
            max_seqlen_kv=teacher_max_seqlen,
            qkv_format="thd",
        )
        batch["opsd_teacher_full_loss_masks"] = torch.zeros_like(batch["opsd_teacher_tokens"])

    # loss masks
    loss_masks = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"],
        batch["total_lengths"],
        batch["response_lengths"],
        strict=True,
    ):
        prompt_length = total_length - response_length
        # Align mask to token stream positions (prompt_length-1 left pad, 1 right pad)
        loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        if allgather_cp:
            loss_masks.append(loss_mask)
            continue
        loss_mask = slice_with_cp(loss_mask, 0)
        loss_masks.append(loss_mask)

    if allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    else:
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is not None:
                for key, mm_tensor in mm_input_dict.items():
                    if key not in multimodal_data:
                        multimodal_data[key] = mm_tensor
                    else:
                        multimodal_data[key] = torch.cat([multimodal_data[key], mm_tensor], dim=0)
        batch["multimodal_train_inputs"] = multimodal_data

    return batch


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, "float | tuple[float, float]"],
    post_reduce=None,
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce on the DP source rank, and log to W&B / TB.

    Each value in ``log_dict`` is either:
      * a ``(sum, count)`` tuple → reduced as ``Σsum / Σcount``;
      * a plain scalar → reduced as ``Σ / dp_size`` (mean across ranks).

    The gather + reduce step is delegated to
    :func:`cp_utils.gather_and_reduce_log_dict` so it can be exercised by
    CPU multi-process unit tests directly. This function adds the
    ``metric_name`` prefix and the W&B / TB logging side effects.
    """
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
    dp_src_rank = mpu.get_data_parallel_src_rank(with_context_parallel=True)
    try:
        gloo_group = mpu.get_data_parallel_group_gloo(
            with_context_parallel=True
        )
    except AssertionError:
        # NeMo Megatron initializes the regular NCCL DP group but not MCore's
        # optional companion Gloo group.  Keep the exact same metric builder
        # and reduction semantics, using a numeric NCCL all-reduce fallback.
        reduced = all_reduce_log_dict(
            log_dict,
            dp_size=dp_size,
            dp_src_rank=dp_src_rank,
            dp_group=mpu.get_data_parallel_group(
                with_context_parallel=True
            ),
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    else:
        reduced = gather_and_reduce_log_dict(
            log_dict,
            dp_size=dp_size,
            dp_src_rank=dp_src_rank,
            dp_group=gloo_group,
        )
    if reduced is None:
        return None
    if post_reduce is not None:
        reduced = post_reduce(reduced)
    reduced_log_dict = {f"{metric_name}/{k}": v for k, v in reduced.items()}
    logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")
    # Calculate step once to avoid duplication
    step = compute_rollout_step(args, rollout_id)
    reduced_log_dict["rollout/step"] = step
    logging_utils.log(args, reduced_log_dict, step_key="rollout/step")
    return reduced_log_dict


def _add_masked_moments(
    log_dict: dict[str, float | tuple[float, float]],
    *,
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Add sufficient statistics for an exact global masked mean/std."""
    values = values.detach().float()
    mask = mask.to(device=values.device, dtype=torch.bool)
    if values.numel() != mask.numel():
        raise ValueError(
            f"metric {name}: values/mask length mismatch "
            f"({values.numel()} != {mask.numel()})"
        )
    finite = torch.isfinite(values)
    valid = mask & finite
    count = float(valid.sum().item())
    masked_count = float(mask.sum().item())
    if masked_count:
        log_dict[f"__moments/{name}/finite_fraction"] = (
            float(valid.sum().item()),
            masked_count,
        )
    else:
        log_dict[f"__moments/{name}/finite_fraction"] = (0.0, 0.0)
    if count == 0:
        log_dict[f"__moments/{name}/mean"] = (0.0, 0.0)
        log_dict[f"__moments/{name}/second"] = (0.0, 0.0)
        return
    selected = values[valid]
    log_dict[f"__moments/{name}/mean"] = (float(selected.sum().item()), count)
    log_dict[f"__moments/{name}/second"] = (
        float(selected.square().sum().item()),
        count,
    )


def _add_algorithm_metrics(
    log_dict: dict[str, float | tuple[float, float]],
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """Add CP/DP-correct token statistics used to diagnose RL algorithms."""
    local_masks = get_local_response_masks(
        rollout_data["total_lengths"],
        rollout_data["response_lengths"],
        rollout_data["loss_masks"],
    )
    mask = torch.cat(local_masks).detach()

    tensors: dict[str, torch.Tensor] = {}
    for key, name in (
        ("advantages", "advantage"),
        ("returns", "return"),
        ("values", "value_prediction"),
    ):
        values = rollout_data.get(key)
        if values and isinstance(values[0], torch.Tensor):
            tensors[name] = torch.cat(values).detach()
            _add_masked_moments(log_dict, name=name, values=tensors[name], mask=mask)

    train_log_probs = rollout_data.get("log_probs")
    reference_log_probs = rollout_data.get("ref_log_probs")
    if (
        train_log_probs
        and reference_log_probs
        and isinstance(train_log_probs[0], torch.Tensor)
    ):
        train = torch.cat(train_log_probs).detach().float()
        reference = torch.cat(reference_log_probs).detach().float().to(train.device)
        reference_kl = compute_approx_kl(
            train,
            reference,
            kl_loss_type=getattr(args, "kl_loss_type", "low_var_kl"),
        )
        _add_masked_moments(
            log_dict,
            name="reference_kl",
            values=reference_kl,
            mask=mask,
        )

    if "value_prediction" in tensors and "return" in tensors:
        _add_masked_moments(
            log_dict,
            name="value_residual",
            values=tensors["return"] - tensors["value_prediction"],
            mask=mask,
        )

    behavior_log_probs = rollout_data.get("rollout_log_probs")
    if (
        train_log_probs
        and behavior_log_probs
        and isinstance(train_log_probs[0], torch.Tensor)
    ):
        train = torch.cat(train_log_probs).detach().float()
        behavior = torch.cat(behavior_log_probs).detach().float().to(train.device)
        mask = mask.to(train.device)
        log_ratio = train - behavior
        behavior_kl = compute_approx_kl(
            train,
            behavior,
            kl_loss_type=getattr(args, "kl_loss_type", "low_var_kl"),
        )
        _add_masked_moments(
            log_dict,
            name="train_behavior_logprob_difference",
            values=log_ratio,
            mask=mask,
        )
        _add_masked_moments(
            log_dict,
            name="behavior_kl",
            values=behavior_kl,
            mask=mask,
        )

        if not getattr(args, "use_tis", False):
            return

        weights = torch.exp(log_ratio)
        _add_masked_moments(
            log_dict,
            name="tis_importance_weight",
            values=weights,
            mask=mask,
        )
        low = float(getattr(args, "tis_clip_low", 0.0))
        high = float(getattr(args, "tis_clip", float("inf")))
        valid_count = float(mask.sum().item())
        log_dict["algorithm/tis_low_clip_fraction"] = (
            float(((weights < low) & mask.bool()).sum().item()),
            valid_count,
        )
        log_dict["algorithm/tis_high_clip_fraction"] = (
            float(((weights > high) & mask.bool()).sum().item()),
            valid_count,
        )


class DataIterator:
    """Iterator over a rollout dict following an explicit micro-batch index schedule."""

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_indices: list[list[int]],
    ) -> None:
        """Initialize an iterator over ``rollout_data``.

        Args:
            rollout_data: Dict of per-sample fields for this DP rank.
            micro_batch_indices: List of mbs, each mbs being the local sample indices to select.
        """
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch for the requested keys.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        batch = {}
        indices = self.micro_batch_indices[self.offset]
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                batch[key] = [vals[i] for i in indices]
        self.offset += 1
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def get_data_iterator(rollout_data: RolloutBatch) -> list[DataIterator]:
    """Build one ``DataIterator`` per VPP stage from the pre-computed schedule in ``rollout_data``."""
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    micro_batch_indices = rollout_data["micro_batch_indices"]
    return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]


def filter_dp_padding_for_logging(rollout_data: RolloutBatch) -> RolloutBatch:
    """Remove dummy samples while retaining an exact padding-rate metric.

    Slime partitions the scheduled batch across DP ranks before logging, so a
    rank may receive only real samples while another receives padding.  The
    returned dict intentionally includes ``dp_padding_fraction`` on *every*
    rank whenever the schedule supplied a padding mask; distributed object
    reduction requires identical metric keys across ranks.
    """
    padding_mask = rollout_data.get("dp_padding_mask")
    if padding_mask is None:
        return rollout_data

    padding_flags = [bool(value) for value in padding_mask]
    if any(padding_flags):
        filtered: RolloutBatch = {}
        for key, value in rollout_data.items():
            if isinstance(value, (list, tuple)) and len(value) == len(padding_flags):
                filtered[key] = [item for item, padded in zip(value, padding_flags, strict=True) if not padded]
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == len(padding_flags):
                keep = torch.tensor([not padded for padded in padding_flags], device=value.device)
                filtered[key] = value[keep]
            else:
                filtered[key] = value
    else:
        # Avoid mutating the training batch merely to attach a logging-only
        # field.
        filtered = dict(rollout_data)

    # Keep one 0/1 contribution per scheduled sample.  The distributed
    # (sum, count) reducer reports the exact global dummy fraction even if DP
    # ranks happen to own different numbers of samples.
    filtered["dp_padding_fraction"] = [float(value) for value in padding_flags]
    return filtered


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    rollout_data = filter_dp_padding_for_logging(rollout_data)

    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        cp_size = mpu.get_context_parallel_world_size()
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        # Same per-rollout denominators the training loss uses, so reported
        # log_probs / returns / advantages / etc. live in the same per-rollout
        # mean space (rather than per-sample) as the gradient signal.
        rollout_mask_sums = rollout_data.get("rollout_mask_sums", None)
        # For per-rollout-mean metrics: ``rollout_log_metric_contribution``
        # produces the ``(sum, count)`` tuple so gather_log_data's
        # ``Σsum / Σcount`` lands on ``sum_DP_full / num_rollouts`` — the
        # same number train_one_step reports for the same samples.
        dp_world = mpu.get_data_parallel_world_size(with_context_parallel=False)
        num_rollouts_in_rollout = sum(rollout_data["global_batch_sizes"])
        per_rollout_mean_keys = {
            "log_probs",
            "ref_log_probs",
            "rollout_log_probs",
            "returns",
            "advantages",
            "values",
            "teacher_log_probs",
            "opd_reverse_kl",
        }

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "opsd_teacher_tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "dp_padding_mask",
                "rollout_top_p_token_ids",
                "rollout_top_p_token_offsets",
                "rollout_routed_experts",
                "global_batch_sizes",
                "num_microbatches",
                "micro_batch_indices",
            ]:
                continue
            # Emit (sum, count) so gather_log_data can do a weighted average across
            # DP ranks. This stops the legacy "every rank has the same N samples"
            # assumption from biasing means once uneven-DP partitioning lands.
            if isinstance(val, (list, tuple)):
                count = len(val)
                if count == 0:
                    # A DP rank can own only zero-loss alignment padding.  The
                    # logging filter removes that dummy sample, leaving empty
                    # per-sample fields on this rank.  Keep every metric key in
                    # the collective and contribute a zero numerator.  Metrics
                    # normalized by the rollout count must still contribute
                    # their share of the global denominator; ordinary sample
                    # means use a zero count because this rank has no real row.
                    if key in per_rollout_mean_keys:
                        _, rollout_count = rollout_log_metric_contribution(
                            0.0,
                            cp_size=cp_size,
                            num_rollouts_in_rollout=num_rollouts_in_rollout,
                            dp_size=dp_world,
                        )
                        log_dict[key] = (0.0, rollout_count)
                    else:
                        log_dict[key] = (0.0, 0)
                    continue
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    if key in per_rollout_mean_keys:
                        tensor = torch.cat(val).clone().detach()
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            rollout_mask_sums,
                        )
                        # Compute (sum, count) via the shared helper so this
                        # path and the unit tests stay in sync.
                        sum_value, count = rollout_log_metric_contribution(
                            sum_of_sample_mean(tensor).item(),
                            cp_size=cp_size,
                            num_rollouts_in_rollout=num_rollouts_in_rollout,
                            dp_size=dp_world,
                        )
                        log_dict[key] = (sum_value, count)
                        continue
                    tensor = torch.cat(val).clone().detach()
                    # val.mean() * cp_size is the per-sample mean for one rank;
                    # multiply by count to get the per-rank sum.
                    per_rank_sum = tensor.mean() * cp_size * count
                    sum_value = per_rank_sum.item()
                else:
                    sum_value = sum(val)
                log_dict[key] = (sum_value, count)
            elif isinstance(val, torch.Tensor):
                # Scalar tensor (one per rank): treat as count=1.
                log_dict[key] = (val.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")

        _add_algorithm_metrics(log_dict, args, rollout_data)
        reduced_log_dict = gather_log_data(
            "rollout",
            args,
            rollout_id,
            log_dict,
            post_reduce=derive_algorithm_metrics,
        )
        if args.ci_test and reduced_log_dict is not None:
            # This is an initial actor/ref zero-KL check. R3 replays rollout
            # routing for the actor forward, while the reference forward
            # intentionally falls through to normal routing, so their
            # log-probs are not expected to match bit-for-bit in CI.
            if (
                rollout_id == 0
                and not getattr(args, "ci_disable_kl_checker", False)
                and not getattr(args, "use_rollout_routing_replay", False)
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -1 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 1

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
            cp_size = mpu.get_context_parallel_world_size()
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # NOTE: per-sample-mean over the correct subset, not per-rollout.
                # A rollout's siblings may not all be correct, and slicing
                # ``rollout_mask_sums`` here would leave a denom that still
                # includes incorrect siblings — meaningless for a "correct-only"
                # entropy report. Per-sample-mean over the filtered subset is
                # the cleanest semantic.
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks, sample_denoms=None
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace, extra_metrics: dict | None = None) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
        extra_metrics=extra_metrics,
    )


def tensors_to_cpu(tensor_list):
    """Move a list of GPU tensors to CPU for Ray object store transfer.

    Args:
        tensor_list: List of GPU tensors, or None.

    Returns:
        List of CPU tensors (detached), or None if input is None.
    """
    if tensor_list is None:
        return None
    return [t.detach().cpu() for t in tensor_list]


def tensors_to_gpu(tensor_list, device=None):
    """Move a list of CPU tensors back to GPU.

    Args:
        tensor_list: List of CPU tensors, or None.
        device: Target CUDA device. If None, uses current device.

    Returns:
        List of GPU tensors, or None if input is None.
    """
    if tensor_list is None:
        return None
    if device is None:
        device = torch.cuda.current_device()
    return [t.to(device=device, dtype=torch.float32) for t in tensor_list]
