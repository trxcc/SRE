"""Slime data-source wrappers used by Polar examples."""

from __future__ import annotations

import math
import os

try:
    from slime.rollout.data_source import RolloutDataSourceWithBuffer
except ImportError as _SLIME_IMPORT_ERROR:
    class RolloutDataSourceWithBuffer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "Slime is required to use CeilEpochRolloutDataSourceWithBuffer."
            ) from _SLIME_IMPORT_ERROR


def ceil_to_batch_size(size: int, batch_size: int) -> int:
    if size <= 0:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return math.ceil(size / batch_size) * batch_size


class CeilEpochRolloutDataSourceWithBuffer(RolloutDataSourceWithBuffer):
    """Expose a rounded-up epoch length for fixed-size Slime rollout batches.

    Slime computes `num_rollout_per_epoch = len(data_source) // rollout_batch_size`.
    For datasets whose size is not divisible by the rollout batch size, the
    default floor behavior skips the tail prompts. Returning a rounded-up length
    lets the existing data source wrap only the final few prompts while still
    covering every prompt in the dataset once per epoch.
    """

    def __len__(self) -> int:
        return ceil_to_batch_size(
            super().__len__(),
            int(getattr(self.args, "rollout_batch_size", 1) or 1),
        )


class ResumableCeilEpochRolloutDataSourceWithBuffer(
    CeilEpochRolloutDataSourceWithBuffer
):
    """Resume a training-free rollout at its explicit rollout-step cursor.

    training_free_only.py can restart from start_rollout_id without a
    model/optimizer checkpoint. Slime's ordinary data source still begins at
    prompt zero in that case, so a restart would silently duplicate prompts and
    group ids. This opt-in data source advances the shuffled dataset cursor and
    global sample ids by the already-persisted rollout steps.
    """

    def __init__(self, args) -> None:
        super().__init__(args)
        # The full-eval collector persists whole rollout shards while the
        # Polar worker may have prefetched later groups.  Mark the first
        # finite epoch as an ordered prefix so the worker only delivers a
        # contiguous group cursor to each persisted shard.
        self.ordered_prefix_groups = (
            len(self.dataset) if self.dataset is not None else 0
        )
        start_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        samples_per_prompt = int(
            getattr(args, "n_samples_per_prompt", 1) or 1
        )
        if start_rollout_id < 0:
            raise ValueError("start_rollout_id must be non-negative")
        if batch_size <= 0 or samples_per_prompt <= 0:
            raise ValueError(
                "rollout_batch_size and n_samples_per_prompt must be positive"
            )

        explicit_group_index = os.environ.get("CRITIC_RESUME_SAMPLE_GROUP_INDEX")
        if explicit_group_index is None:
            consumed_groups = start_rollout_id * batch_size
        else:
            try:
                consumed_groups = int(explicit_group_index)
            except ValueError as exc:
                raise ValueError(
                    "CRITIC_RESUME_SAMPLE_GROUP_INDEX must be an integer"
                ) from exc
            if consumed_groups < 0:
                raise ValueError(
                    "CRITIC_RESUME_SAMPLE_GROUP_INDEX must be non-negative"
                )
        if consumed_groups == 0:
            return
        if self.dataset is None or len(self.dataset) <= 0:
            raise ValueError(
                "cannot resume a training-free rollout without a prompt dataset"
            )

        self._finite_group_limit = len(self.dataset)
        if consumed_groups > self._finite_group_limit:
            raise ValueError(
                "resume cursor is past the finite epoch: "
                f"{consumed_groups}>{self._finite_group_limit}"
            )
        completed_epochs, sample_offset = divmod(
            consumed_groups,
            len(self.dataset),
        )
        self.epoch_id = completed_epochs
        self.sample_offset = sample_offset
        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)
        self.sample_group_index = consumed_groups
        self.sample_index = consumed_groups * samples_per_prompt

    def get_samples(self, num_samples: int):
        """Return only the unconsumed groups from this finite one-epoch eval."""
        limit = getattr(self, "_finite_group_limit", None)
        if limit is None:
            if self.dataset is None:
                return []
            limit = len(self.dataset)
            self._finite_group_limit = limit
        remaining = int(limit) - int(self.sample_group_index)
        if remaining <= 0:
            return []
        return super().get_samples(min(int(num_samples), remaining))
