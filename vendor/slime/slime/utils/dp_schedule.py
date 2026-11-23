"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by rollout id (``rollout_indices[i]`` =
     ``samples[i].index``) and split rollouts into steps of
     ``global_batch_size`` rollouts each. In the common case one rollout
     emits one training sample so this is the same as a contiguous chunk;
     under compact / subagent one rollout may emit multiple training
     samples, in which case all of those samples stay in the same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     estimated mbs FLOPs.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path without ``balance_by_flops``) holds
    ``<= max_tokens_per_gpu * cp_size`` tokens, with one exception — an
    individual sample larger than that cap lands alone in its own mbs (and
    that mbs is the only one allowed to exceed the cap);
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing rollouts (every kept sample placed exactly
    once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly
    once by its mbs schedule).
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)


def _calculate_workloads(step_lengths, args):
    return [calculate_fwd_flops([sl], args) for sl in step_lengths]


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    args: Any,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
    balance_by_flops: bool = False,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        if balance_by_flops:
            total_tokens = sum(step_lengths)
            num_mbs = max(1, (total_tokens + max_per_bin - 1) // max_per_bin)
            if num_mbs >= len(step_lengths):
                return [[i] for i in range(len(step_lengths))]
            workloads = _calculate_workloads(step_lengths, args)
            # FLOPs balancing does not enforce the token cap per mbs. A
            # partition can exceed max_per_bin and may OOM if the cap is tight.
            return get_seqlen_balanced_partitions(workloads, num_mbs, equal_size=False)
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def get_dynamic_dp_padding_sources(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> list[int]:
    """Return sample indices to clone as zero-loss DP-alignment padding.

    Dynamic batching normally aligns the microbatch count by splitting packed
    bins.  Agent rollouts can emit a variable number of long samples, however,
    and an all-singleton step cannot be split further.  In that case the caller
    may clone the returned samples with an all-zero loss mask so every DP rank
    still executes the same number of forwards/backwards.

    The clones retain their source rollout id, so padding never creates a new
    optimization step or changes ``global_batch_size`` semantics.
    """
    if not args.use_dynamic_batch_size:
        return []

    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)
    max_per_bin = args.max_tokens_per_gpu * cp_size

    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_samples)
    num_steps = len(rollout_ids) // global_batch_size

    padding_sources: list[int] = []
    for step_i in range(num_steps):
        step_rollouts = rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            args=args,
            use_dynamic_batch_size=True,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
            balance_by_flops=args.balance_by_flops,
        )
        target_mbs = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)

        # Splitting can produce at most one mbs per sample.  Only pad the
        # shortfall beyond that hard limit; build_dp_schedule will perform any
        # remaining ordinary bin splits after the clones are appended.
        padding_count = max(0, target_mbs - len(sample_indices))
        if padding_count:
            # A short clone minimizes the memory cost if it is not already an
            # oversized singleton.  Repeated source indices are intentional.
            source = min(sample_indices, key=lambda i: total_lengths[i])
            padding_sources.extend([source] * padding_count)
            logger.info(
                "DP alignment padding plan step=%d samples=%d packed_mbs=%d "
                "target_mbs=%d dummy_samples=%d source_length=%d",
                step_i,
                len(sample_indices),
                len(step_mbs),
                target_mbs,
                padding_count,
                total_lengths[source],
            )

    return padding_sources


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of rollouts (NOT training samples) per
            training step. Number of training steps =
            ``num_rollouts // global_batch_size``; trailing rollouts whose
            samples don't fit are dropped.
        rollout_indices: rollout id for each sample (``samples[i].index``).
            Samples sharing the same id are kept together in one step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = rollout count for step s (constant
        ``global_batch_size`` for every step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    # Group samples by rollout id (preserve first-occurrence order). All
    # samples from one rollout stay in a single step so the per-rollout loss
    # reducer is well-defined.
    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_samples.keys())

    num_steps = len(rollout_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_rollouts ({len(rollout_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i in range(num_steps):
        step_rollouts = rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(global_batch_size)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            args=args,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
            balance_by_flops=args.balance_by_flops,
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        mbs_token_sums = [sum(step_lengths[i] for i in bin_) for bin_ in step_mbs]
        if args.use_dynamic_batch_size and not args.balance_by_flops:
            assert max_per_bin is not None
            invalid_bins = [
                (bin_, token_sum)
                for bin_, token_sum in zip(step_mbs, mbs_token_sums, strict=True)
                if token_sum > max_per_bin and len(bin_) != 1
            ]
            assert not invalid_bins, (
                f"step {step_i}: dynamic microbatch exceeds token cap {max_per_bin} "
                f"without being a singleton oversized sample: {invalid_bins}"
            )
        logger.info(
            "DP schedule step=%d rollouts=%d samples=%d total_tokens=%d "
            "sequence_tokens[min/mean/max]=%d/%.1f/%d microbatches=%d "
            "microbatch_tokens[min/mean/max]=%d/%.1f/%d "
            "max_tokens_per_microbatch=%s dp=%d cp=%d balance_data=%s balance_by_flops=%s",
            step_i,
            len(step_rollouts),
            len(step_lengths),
            sum(step_lengths),
            min(step_lengths),
            sum(step_lengths) / len(step_lengths),
            max(step_lengths),
            K,
            min(mbs_token_sums),
            sum(mbs_token_sums) / len(mbs_token_sums),
            max(mbs_token_sums),
            max_per_bin,
            dp_size,
            cp_size,
            args.balance_data,
            args.balance_by_flops,
        )

        # 3. Distribute mbs across ranks: KK on estimated FLOPs when rank
        # workload balancing is enabled, otherwise a strided round-robin.
        if args.balance_data:
            step_workloads = _calculate_workloads(step_lengths, args)
            mbs_weights = [sum(step_workloads[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_weights, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
