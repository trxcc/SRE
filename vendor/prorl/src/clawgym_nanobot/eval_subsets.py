"""Eval-subset discovery and task-level metric aggregation."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


_SUBSET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def normalize_eval_subset(value: Any) -> str | None:
    """Return a dashboard-safe subset name, or ``None`` when absent."""
    if value is None:
        return None
    subset = str(value).strip().lower().replace(" ", "_")
    if not subset:
        return None
    if not _SUBSET_RE.fullmatch(subset):
        raise ValueError(f"invalid eval_subset {value!r}")
    return subset


@lru_cache(maxsize=64)
def load_eval_subset_map(manifest_path: str | Path) -> dict[str, str]:
    """Load subset labels keyed by task id and task-directory basename."""
    path = Path(manifest_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        rows = value.get("tasks") or value.get("rows") or value.get("items")
        if rows is None and all(isinstance(item, str) for item in value.values()):
            mapping: dict[str, str] = {}
            for key, item in value.items():
                subset = normalize_eval_subset(item)
                if subset is not None:
                    mapping[str(key)] = subset
            return mapping
    else:
        rows = value
    if not isinstance(rows, list):
        raise ValueError(f"eval subset manifest must contain a list: {path}")

    mapping: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"eval subset manifest row {index} is not an object: {path}")
        subset = normalize_eval_subset(row.get("eval_subset"))
        if subset is None:
            continue
        for raw_key in (row.get("task_id"), row.get("task_dir")):
            if not raw_key:
                continue
            key = str(raw_key)
            for candidate in (key, Path(key).name):
                previous = mapping.get(candidate)
                if previous is not None and previous != subset:
                    raise ValueError(
                        f"conflicting eval_subset for {candidate!r} in {path}: "
                        f"{previous!r} vs {subset!r}"
                    )
                mapping[candidate] = subset
    return mapping


def eval_subset_for_task(
    *,
    task_id: str | None,
    task_dir: str | Path | None,
    metadata: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
) -> str | None:
    """Resolve a task's subset from metadata first, then its split manifest."""
    if metadata is not None:
        subset = normalize_eval_subset(metadata.get("eval_subset"))
        if subset is not None:
            return subset

    resolved_manifest: Path | None = Path(manifest_path) if manifest_path else None
    resolved_task_dir = Path(task_dir) if task_dir else None
    if resolved_manifest is None and resolved_task_dir is not None:
        candidate = resolved_task_dir.parent / "manifest.json"
        if candidate.is_file():
            resolved_manifest = candidate
    if resolved_manifest is None or not resolved_manifest.is_file():
        return None

    mapping = load_eval_subset_map(resolved_manifest.resolve())
    keys = [task_id]
    if resolved_task_dir is not None:
        keys.extend((str(resolved_task_dir), resolved_task_dir.name))
    for key in keys:
        if key and key in mapping:
            return mapping[key]
    return None


def build_eval_subset_metrics(
    dataset_name: str,
    task_rewards: Sequence[float],
    task_metadata: Sequence[Mapping[str, Any] | None],
    *,
    raw_rewards: Sequence[float] | None = None,
    raw_skipped: Sequence[bool] | None = None,
    task_statuses: Sequence[Sequence[str]] | None = None,
    group_size: int = 1,
    primary_subset: str = "core",
) -> dict[str, int | float]:
    """Aggregate task rewards by an explicit ``eval_subset`` label.

    The existing combined metric remains the compatibility view.  ``primary``
    aliases the configured in-distribution subset (``core`` by default), while
    challenge remains a separately visible stress test.
    """
    rewards = [float(value) for value in task_rewards]
    metadata = list(task_metadata)
    if len(metadata) != len(rewards):
        raise ValueError(
            "task_metadata must align one-to-one with task_rewards: "
            f"{len(metadata)} != {len(rewards)}"
        )
    primary = normalize_eval_subset(primary_subset)
    if primary is None:
        raise ValueError("primary_subset must be non-empty")

    labels: list[str | None] = []
    for item in metadata:
        labels.append(normalize_eval_subset(item.get("eval_subset")) if item else None)
    tagged_indices = [index for index, label in enumerate(labels) if label is not None]
    if not tagged_indices:
        return {}

    prefix = f"eval/{dataset_name}"
    combined = sum(rewards) / len(rewards) if rewards else 0.0
    metrics: dict[str, int | float] = {
        f"{prefix}-combined": combined,
        f"{prefix}-combined-reward_avg": combined,
        f"{prefix}-subset-tagged-tasks": len(tagged_indices),
        f"{prefix}-subset-untagged-tasks": len(rewards) - len(tagged_indices),
        f"{prefix}-subset-coverage": len(tagged_indices) / len(rewards) if rewards else 0.0,
        f"{prefix}-subset-count": len({labels[index] for index in tagged_indices}),
    }

    normalized_group_size = max(1, int(group_size or 1))
    raw = [float(value) for value in raw_rewards] if raw_rewards is not None else []
    skipped = [bool(value) for value in raw_skipped] if raw_skipped is not None else []
    raw_aligned = (
        len(raw) == len(rewards) * normalized_group_size
        and len(skipped) == len(raw)
    )
    statuses = list(task_statuses) if task_statuses is not None else []
    statuses_aligned = len(statuses) == len(rewards)

    subset_values: dict[str, list[int]] = {}
    for index in tagged_indices:
        subset_values.setdefault(labels[index] or "", []).append(index)

    for subset, indices in sorted(subset_values.items()):
        subset_rewards = [rewards[index] for index in indices]
        reward_avg = sum(subset_rewards) / len(subset_rewards)
        subset_prefix = f"{prefix}-{subset}"
        metrics.update(
            {
                subset_prefix: reward_avg,
                f"{subset_prefix}-reward_avg": reward_avg,
                f"{subset_prefix}-n_tasks": len(indices),
                f"{subset_prefix}-correct_count": sum(
                    1 for reward in subset_rewards if reward > 0.5
                ),
            }
        )
        if raw_aligned:
            positions = [
                position
                for index in indices
                for position in range(
                    index * normalized_group_size,
                    (index + 1) * normalized_group_size,
                )
            ]
            collected = [raw[position] for position in positions if not skipped[position]]
            n_skipped = sum(1 for position in positions if skipped[position])
            metrics[f"{subset_prefix}-n_collected"] = len(collected)
            metrics[f"{subset_prefix}-n_skipped"] = n_skipped
            metrics[f"{subset_prefix}-avg_reward_collected_only"] = (
                sum(collected) / len(collected) if collected else 0.0
            )
        if statuses_aligned:
            completed = 0
            timed_out = 0
            failed = 0
            for index in indices:
                values = [str(value).upper() for value in statuses[index]]
                if values and all(value == "COMPLETED" for value in values):
                    completed += 1
                elif any(value == "TIMEOUT" for value in values):
                    timed_out += 1
                else:
                    failed += 1
            metrics[f"{subset_prefix}-status-completed-tasks"] = completed
            metrics[f"{subset_prefix}-status-timeout-tasks"] = timed_out
            metrics[f"{subset_prefix}-status-failed-tasks"] = failed
            metrics[f"{subset_prefix}-status-timeout-rate"] = timed_out / len(indices)
            metrics[f"{subset_prefix}-status-failure-rate"] = failed / len(indices)

    primary_key = f"{prefix}-{primary}"
    if primary_key in metrics:
        metrics[f"{prefix}-primary"] = metrics[primary_key]
        metrics[f"{prefix}-primary-reward_avg"] = metrics[primary_key]
        metrics[f"{prefix}-primary-n_tasks"] = metrics[f"{primary_key}-n_tasks"]
    return metrics
