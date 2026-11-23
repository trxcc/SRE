"""ClawGym prompt-data preparation and first-epoch scheduling for Polar."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from slime_bridge.data_source import CeilEpochRolloutDataSourceWithBuffer

from . import tasks
from .eval_subsets import eval_subset_for_task
from .runtime import _format_clawgym_task_prompt
from .polar_workspace import validate_task_workspace


class SequentialThenShuffleRolloutDataSource(CeilEpochRolloutDataSourceWithBuffer):
    """Keep epoch zero ordered, then use Slime's deterministic epoch shuffles."""

    def __init__(self, args) -> None:
        rollout_shuffle = bool(args.rollout_shuffle)
        args.rollout_shuffle = False
        try:
            super().__init__(args)
        finally:
            args.rollout_shuffle = rollout_shuffle
        self.ordered_prefix_groups = len(self.dataset) if self.dataset is not None else 0
        self._resume_from_rollout_id()

    def record_consumed_groups(self, rollout_id, group_ids):
        batch=int(self.args.rollout_batch_size)
        expected=list(range(rollout_id*batch,(rollout_id+1)*batch))
        if list(group_ids)!=expected or expected[-1]>=self.ordered_prefix_groups:
            raise ValueError("Consumed groups do not match ordered first-epoch frontier")
        records=getattr(self, "_consumed_batches", None)
        if records is None:
            records=self._consumed_batches={}
        if rollout_id in records:
            raise ValueError("Duplicate consumed rollout")
        records[rollout_id]=expected

    def save(self, rollout_id):
        if getattr(self.args, "rollout_function_path", "") != "slime_bridge.rollout.generate_rollout_polar_async":
            return super().save(rollout_id)
        if not self.args.rollout_global_dataset:
            return
        batch_size = int(self.args.rollout_batch_size)
        samples_per_prompt = int(self.args.n_samples_per_prompt)
        if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
            raise ValueError("Invalid consumed rollout id")
        consumed = (rollout_id + 1) * batch_size
        expected=list(range(rollout_id*batch_size,consumed))
        if getattr(self, "_consumed_batches", {}).get(rollout_id)!=expected:
            raise ValueError("No actual delivered-batch evidence for saved cursor")
        # Ordered delivery is guaranteed only in the first dataset epoch.
        # Reject other recipes instead of persisting an ahead-of-training cursor.
        if (batch_size <= 0 or samples_per_prompt <= 0 or self.dataset is None
                or int(self.args.num_rollout) * batch_size > self.ordered_prefix_groups
                or not 0 < consumed <= self.ordered_prefix_groups
                or consumed > self.sample_group_index):
            raise ValueError("Cannot prove ordered consumed-batch checkpoint frontier")
        state = {"sample_offset": consumed, "epoch_id": 0,
                 "sample_group_index": consumed,
                 "sample_index": consumed * samples_per_prompt,
                 "metadata": dict(self.metadata)}
        root = self.args.save
        if root is None and self.args.save_hf is not None:
            root = self.args.save_hf.format(rollout_id=rollout_id)
        if root is None:
            raise ValueError("Cannot save consumed cursor without a save root")
        from slime.utils.atomic_rollout_cursor import save_cursor
        save_cursor(Path(root) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt", state)

    def _resume_from_rollout_id(self) -> None:
        """Derive the prompt cursor when resuming from an HF-only checkpoint.

        Slime's ``start_rollout_id`` advances the optimizer/rollout step but its
        ordinary data source still starts at prompt zero when no serialized
        dataset cursor is loaded.  ResearchClaw intentionally resumes from the
        HF weights alone, so reconstruct the deterministic cursor from the
        number of completed rollout batches.
        """
        start_rollout_id = int(getattr(self.args, "start_rollout_id", 0) or 0)
        batch_size = int(getattr(self.args, "rollout_batch_size", 1) or 1)
        samples_per_prompt = int(
            getattr(self.args, "n_samples_per_prompt", 1) or 1
        )
        if start_rollout_id < 0:
            raise ValueError("start_rollout_id must be non-negative")
        if batch_size <= 0 or samples_per_prompt <= 0:
            raise ValueError(
                "rollout_batch_size and n_samples_per_prompt must be positive"
            )

        consumed_groups = start_rollout_id * batch_size
        if consumed_groups == 0:
            return
        if self.dataset is None or len(self.dataset) <= 0:
            raise ValueError(
                "cannot resume a sequential-then-shuffle rollout without a prompt dataset"
            )

        completed_epochs, sample_offset = divmod(
            consumed_groups,
            len(self.dataset),
        )
        self.epoch_id = completed_epochs
        self.sample_offset = sample_offset
        if completed_epochs > 0 and self.args.rollout_shuffle:
            self.dataset.shuffle(completed_epochs)
        self.sample_group_index = consumed_groups
        self.sample_index = consumed_groups * samples_per_prompt


class FiniteSequentialRolloutDataSource(SequentialThenShuffleRolloutDataSource):
    """Opt-in one-pass recipe with an explicit, unpadded final prompt batch.

    This recipe changes no trajectory cardinality or reward logic. Its caller
    must use ``batch_group_ids`` to request the exact remaining prompt count.
    The legacy sequential-then-shuffle recipe remains available unchanged.
    """

    finite_first_epoch = True

    def _validate_recipe(self):
        batch = self.args.rollout_batch_size
        repeats = self.args.n_samples_per_prompt
        steps = self.args.num_rollout
        if (type(batch) is not int or batch <= 0 or type(repeats) is not int
                or repeats <= 0 or self.dataset is None or len(self.dataset) <= 0):
            raise ValueError("Finite recipe requires a dataset and positive integer group sizes")
        limit = len(self.dataset)
        expected_steps = (limit + batch - 1) // batch
        if type(steps) is not int:
            raise ValueError("Finite recipe requires an integer rollout budget")
        if steps != expected_steps:
            from runtime.portable_contract import validate_finite_branch
            expected_steps = validate_finite_branch(self.args, limit)
        if self.args.rollout_shuffle:
            raise ValueError("Finite fixed-order recipe must not reshuffle")
        return batch, repeats, limit, expected_steps

    def batch_group_ids(self, rollout_id):
        batch, _, limit, steps = self._validate_recipe()
        first = self.args.start_rollout_id if steps * batch < limit else 0
        if type(rollout_id) is not int or not first <= rollout_id < steps:
            raise ValueError("Rollout is outside the authorized finite dataset interval")
        start = rollout_id * batch
        return list(range(start, min(start + batch, limit)))

    def _resume_from_rollout_id(self):
        batch, repeats, limit, steps = self._validate_recipe()
        start = getattr(self.args, "start_rollout_id", 0)
        if type(start) is not int or not 0 <= start <= steps:
            raise ValueError("Resume rollout is outside the finite dataset pass")
        consumed = min(start * batch, limit)
        self.epoch_id = 0
        self.sample_offset = self.sample_group_index = consumed
        self.sample_index = consumed * repeats

    def get_samples(self, num_samples):
        batch, _, dataset_limit, steps = self._validate_recipe()
        limit = min(dataset_limit, steps * batch)
        if type(num_samples) is not int or num_samples < 0:
            raise ValueError("Requested prompt count must be a nonnegative integer")
        if not 0 <= self.sample_group_index <= limit:
            raise ValueError("Generated cursor is outside the finite dataset pass")
        remaining = limit - self.sample_group_index
        if num_samples == 0 or remaining == 0:
            return []
        return super().get_samples(min(num_samples, remaining))

    def record_consumed_groups(self, rollout_id, group_ids):
        expected = self.batch_group_ids(rollout_id)
        actual = list(group_ids)
        if any(type(value) is not int for value in actual) or actual != expected:
            raise ValueError("Delivered groups do not match the exact finite batch")
        if expected[-1] >= self.sample_group_index:
            raise ValueError("Cannot consume groups that were not generated")
        records = getattr(self, "_consumed_batches", None)
        if records is None:
            records = self._consumed_batches = {}
        if rollout_id in records:
            raise ValueError("Duplicate consumed rollout")
        records[rollout_id] = expected

    def save(self, rollout_id):
        if getattr(self.args, "rollout_function_path", "") != "slime_bridge.rollout.generate_rollout_polar_async":
            raise ValueError("Finite consumed-cursor recipe requires the matching async rollout entry")
        if not self.args.rollout_global_dataset:
            return
        expected = self.batch_group_ids(rollout_id)
        if getattr(self, "_consumed_batches", {}).get(rollout_id) != expected:
            raise ValueError("No delivered finite-batch evidence for saved cursor")
        _, repeats, limit, _ = self._validate_recipe()
        consumed = expected[-1] + 1
        if not 0 < consumed <= min(self.sample_group_index, limit):
            raise ValueError("Invalid finite consumed cursor")
        state = {"sample_offset": consumed, "epoch_id": 0,
                 "sample_group_index": consumed, "sample_index": consumed * repeats,
                 "metadata": dict(self.metadata)}
        root = self.args.save
        if root is None and self.args.save_hf is not None:
            root = self.args.save_hf.format(rollout_id=rollout_id)
        if root is None:
            raise ValueError("Cannot save finite cursor without a save root")
        from slime.utils.atomic_rollout_cursor import save_cursor
        save_cursor(Path(root) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt", state)


def write_prompt_data(
    source_paths: str,
    output_path: str | Path,
    *,
    limit: int = 0,
    turn_type: str = "main",
    skip_unavailable: bool = False,
) -> int:
    os.environ["CLAWGYM_POLAR_TASK_SOURCES"] = source_paths
    tasks.clear_task_cache()
    selected = tasks.load_tasks("CLAWGYM_POLAR_TASK_SOURCES", default=None)
    if skip_unavailable:
        runnable = []
        for task in selected:
            try:
                validate_task_workspace(task.task_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(
                    f"Skipping unavailable Polar task {task.task_id}: {exc}",
                    file=sys.stderr,
                )
                continue
            runnable.append(task)
        selected = runnable
    if limit > 0:
        selected = selected[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_mode = os.environ.get("CLAWGYM_POLAR_PROMPT_INPUT_MODE", "messages").strip().lower()
    if input_mode not in {"messages", "raw"}:
        raise ValueError(
            "CLAWGYM_POLAR_PROMPT_INPUT_MODE must be 'messages' or 'raw', "
            f"got {input_mode!r}"
        )
    with output.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(selected):
            metadata = {
                "task_id": task.task_id,
                "task_index": index,
                "task_dir": str(task.task_dir),
                "turn_type": turn_type,
            }
            eval_subset = eval_subset_for_task(
                task_id=task.task_id,
                task_dir=task.task_dir,
                metadata=task.metadata,
            )
            if eval_subset is not None:
                metadata["eval_subset"] = eval_subset
            task_prompt = _format_clawgym_task_prompt(task.user_query)
            prompt_input: str | list[dict[str, str]]
            if input_mode == "raw":
                # ResearchClaw's reference eval/training launcher writes the
                # task instruction directly. Keep this opt-in so multimodal
                # WSClaw/Gemma prompt materialization retains its established
                # conversation-shaped input.
                prompt_input = task_prompt
            else:
                prompt_input = [
                    {
                        "role": "user",
                        "content": task_prompt,
                    }
                ]
            row = {
                # Slime may construct a processor even for a text-only view of
                # a multimodal-family checkpoint (Gemma 4 is one example).
                # Dataset then requires conversation-shaped input so it can
                # safely run processor-side multimodal inspection.  Polar's
                # bridge intentionally accepts this single-user-message form
                # and flattens it back to the same instruction text before
                # submitting the agent task.
                "input": prompt_input,
                "metadata": metadata,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--turn-type", default="main")
    parser.add_argument("--skip-unavailable", action="store_true")
    args = parser.parse_args(argv)
    count = write_prompt_data(
        args.sources,
        args.output,
        limit=args.limit,
        turn_type=args.turn_type,
        skip_unavailable=args.skip_unavailable,
    )
    print(f"Wrote Polar ClawGym prompt data: {args.output} ({count} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
