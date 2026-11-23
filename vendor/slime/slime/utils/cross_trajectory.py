"""Pure helpers for CompactionRL-style cross-segment credit metadata."""

from __future__ import annotations

import math
from typing import Any

from slime.utils.types import Sample


def _get_cross_trajectory_metadata(sample: Sample) -> dict[str, Any] | None:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        return None
    clawgym = metadata.get("clawgym_nanobot")
    if not isinstance(clawgym, dict):
        return None
    cross_trajectory = clawgym.get("cross_trajectory")
    return cross_trajectory if isinstance(cross_trajectory, dict) else None


def add_cross_trajectory_train_fields(
    train_data: dict[str, Any],
    samples: list[Sample],
    rewards: list[float],
    loss_masks: list[list[int]],
) -> None:
    """Precompute cross-segment credit metadata before DP partitions samples.

    Sibling physical samples from one rollout are kept in the same optimizer
    step, but the DP scheduler may place them on different ranks.  Computing
    each segment's number of *later trainable action tokens* here lets a custom
    advantage function apply CompactionRL-style distance correction locally,
    without cross-DP value communication.

    Only trajectory producers that explicitly mark a linear credit chain as
    eligible are connected.  Missing metadata and message-tree branches remain
    independent single segments.
    """
    cross_metadata = [_get_cross_trajectory_metadata(sample) for sample in samples]
    if not any(metadata is not None for metadata in cross_metadata):
        return

    sample_count = len(samples)
    if len(rewards) != sample_count or len(loss_masks) != sample_count:
        raise ValueError("cross-trajectory fields require sample-aligned rewards and loss masks")

    action_counts = [int(sum(int(value) for value in mask)) for mask in loss_masks]
    eligible = [False] * sample_count
    segment_indices = [0] * sample_count
    segment_counts = [1] * sample_count
    n_after = [0] * sample_count
    chain_action_tokens = list(action_counts)
    groups: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}

    for position, metadata in enumerate(cross_metadata):
        if metadata is None:
            continue
        if int(metadata.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported cross-trajectory metadata schema at sample {position}: {metadata!r}")
        segment_indices[position] = int(metadata.get("segment_index", 0))
        segment_counts[position] = int(metadata.get("segment_count", 1))
        if not bool(metadata.get("eligible", False)):
            continue
        credit_chain_index = int(metadata.get("credit_chain_index", 0))
        rollout_id = int(train_data["rollout_ids"][position])
        groups.setdefault((rollout_id, credit_chain_index), []).append((position, metadata))

    for chain_key, entries in groups.items():
        entries.sort(key=lambda item: int(item[1]["segment_index"]))
        declared_counts = {int(metadata["segment_count"]) for _, metadata in entries}
        if declared_counts != {len(entries)}:
            raise ValueError(
                f"cross-trajectory chain {chain_key!r} declares segment counts {sorted(declared_counts)!r} "
                f"but contains {len(entries)} samples"
            )
        actual_indices = [int(metadata["segment_index"]) for _, metadata in entries]
        if actual_indices != list(range(len(entries))):
            raise ValueError(
                f"cross-trajectory chain {chain_key!r} has non-contiguous segment indices {actual_indices!r}"
            )
        terminal_flags = [bool(metadata.get("is_terminal_segment", False)) for _, metadata in entries]
        expected_terminal_flags = [False] * (len(entries) - 1) + [True]
        if terminal_flags != expected_terminal_flags:
            raise ValueError(
                f"cross-trajectory chain {chain_key!r} has invalid terminal flags {terminal_flags!r}"
            )
        chain_rewards = [float(rewards[position]) for position, _ in entries]
        if any(
            not math.isclose(reward, chain_rewards[0], rel_tol=0.0, abs_tol=1e-12)
            for reward in chain_rewards[1:]
        ):
            raise ValueError(f"cross-trajectory chain {chain_key!r} has conflicting rewards {chain_rewards!r}")

        total_action_tokens = sum(action_counts[position] for position, _ in entries)
        if total_action_tokens <= 0:
            raise ValueError(f"cross-trajectory chain {chain_key!r} has no trainable action tokens")
        later_action_tokens = 0
        for position, _ in reversed(entries):
            eligible[position] = True
            n_after[position] = later_action_tokens
            chain_action_tokens[position] = total_action_tokens
            later_action_tokens += action_counts[position]

    train_data["cross_trajectory_eligible"] = eligible
    train_data["cross_trajectory_segment_indices"] = segment_indices
    train_data["cross_trajectory_segment_counts"] = segment_counts
    train_data["cross_trajectory_n_after"] = n_after
    train_data["cross_trajectory_chain_action_tokens"] = chain_action_tokens


__all__ = ["add_cross_trajectory_train_fields"]

