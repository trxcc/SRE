from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_TASKS = PROJECT_ROOT / "data" / "clawgym_nanobot" / "claw_train_cleaned_tasks"


@dataclass(frozen=True)
class TaskSpec:
    task_dir: Path
    task_id: str
    user_query: str
    input_mount_dir: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaskSource:
    root: Path
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class TaskSelection:
    task: TaskSpec
    task_index: int
    prompt_index: int
    epoch: int
    order: str


def clear_task_cache() -> None:
    _load_sources_cached.cache_clear()


def task_sources_from_env(env_name: str, *, default: Path | None = None) -> list[TaskSource]:
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value and default is not None:
        raw_value = str(default)
    return list(_load_sources_cached(env_name, raw_value))


def load_tasks(env_name: str = "OPENCLAW_TRAIN_TASKS", *, default: Path | None = DEFAULT_TRAIN_TASKS) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for source in task_sources_from_env(env_name, default=default):
        tasks.extend(source.tasks)
    if not tasks:
        raise FileNotFoundError(f"no ClawGym tasks found from {env_name}")
    return tasks


def load_task(task_dir: str | Path) -> TaskSpec:
    """Load one task directory using the same validation as dataset discovery."""
    return _load_task_spec(Path(task_dir) / "data_entry.json")


def select_task_for_sample(args: Any, sample: Any, *, evaluation: bool = False) -> TaskSelection:
    env_name = "OPENCLAW_EVAL_TASKS" if evaluation else "OPENCLAW_TRAIN_TASKS"
    default = None if evaluation else DEFAULT_TRAIN_TASKS
    prompt_index = _prompt_index_from_sample(args, sample, evaluation=evaluation)
    if not evaluation:
        prompt_index += _train_start_index()
    if evaluation and (explicit := _explicit_eval_task_selection(sample, prompt_index)) is not None:
        return explicit
    tasks = load_tasks(env_name, default=default)
    task_index, epoch = _task_index_for_prompt(prompt_index, len(tasks))
    if evaluation:
        order = os.environ.get("OPENCLAW_EVAL_ORDER", "sequential").strip().lower()
    else:
        order = os.environ.get("OPENCLAW_TRAIN_ORDER", "shuffle").strip().lower()
    selected_index = _ordered_task_index(task_index, epoch, len(tasks), order)
    return TaskSelection(
        task=tasks[selected_index],
        task_index=selected_index,
        prompt_index=prompt_index,
        epoch=epoch,
        order=order,
    )


def _explicit_eval_task_selection(sample: Any, prompt_index: int) -> TaskSelection | None:
    """Honor the task already selected by the eval scheduler.

    Eval sharding changes local group indices, so re-selecting from the full
    task root can silently execute a different task than the scheduler put in
    the sample. The scheduler records both task_id and task_dir in metadata;
    validate and use that explicit selection when present.
    """
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    task_id = str(metadata.get("task_id") or "")
    task_dir = str(metadata.get("task_dir") or "")
    if not task_id or not task_dir:
        return None
    selected = load_task(task_dir)
    if selected.task_id != task_id:
        raise ValueError(
            f"eval sample task metadata mismatch: task_id={task_id!r}, "
            f"task_dir contains {selected.task_id!r}"
        )
    raw_task_index = metadata.get("task_index", prompt_index)
    try:
        task_index = int(raw_task_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid eval sample task_index: {raw_task_index!r}") from exc
    order = os.environ.get("OPENCLAW_EVAL_ORDER", "sequential").strip().lower()
    return TaskSelection(
        task=selected,
        task_index=task_index,
        prompt_index=prompt_index,
        epoch=0,
        order=order,
    )


def prepare_workspace(task: TaskSpec, workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if task.input_mount_dir is None:
        return

    input_dir = _container_path_to_host_path(workspace_dir, task.input_mount_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_files_dir = task.task_dir / "input_files"
    _copy_tree_contents(input_files_dir, input_dir)


def copy_reward_assets(task: TaskSpec, workspace_dir: Path) -> Path:
    source = task.task_dir / "reward"
    target = workspace_dir / "reward"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True)
    return target / "check.py"


def _prompt_index_from_sample(args: Any, sample: Any, *, evaluation: bool = False) -> int:
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return int(group_index)

    sample_index = getattr(sample, "index", None)
    if sample_index is not None:
        attr = "n_samples_per_eval_prompt" if evaluation else "n_samples_per_prompt"
        n_samples = int(getattr(args, attr, 1) or 1)
        return int(sample_index) // max(1, n_samples)

    return 0


def _train_start_index() -> int:
    """Return the explicit offset into an immutable ordered training view."""
    raw = os.environ.get("OPENCLAW_TRAIN_START_INDEX", "0").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"OPENCLAW_TRAIN_START_INDEX must be a non-negative integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"OPENCLAW_TRAIN_START_INDEX must be a non-negative integer, got {value}"
        )
    return value


def _task_index_for_prompt(prompt_index: int, n_tasks: int) -> tuple[int, int]:
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    epoch, pos = divmod(int(prompt_index), n_tasks)
    return pos, epoch


def _ordered_task_index(pos: int, epoch: int, n_tasks: int, order: str) -> int:
    sequential = order in {"sequential", "seq", "ordered", "no_shuffle", "no-shuffle"}
    sequential_then_shuffle = order in {
        "sequential_then_shuffle",
        "seq_then_shuffle",
        "ordered_then_shuffle",
        "first_epoch_sequential_then_shuffle",
        "sequential_first_epoch_then_shuffle",
    }
    if sequential or (sequential_then_shuffle and epoch == 0):
        return pos

    seed = int(os.environ.get("OPENCLAW_TRAIN_SHUFFLE_SEED", "0"))
    rng = random.Random(seed + epoch)
    perm = list(range(n_tasks))
    rng.shuffle(perm)
    return perm[pos]


@lru_cache(maxsize=16)
def _load_sources_cached(env_name: str, raw_value: str) -> tuple[TaskSource, ...]:
    del env_name
    roots = _split_source_paths(raw_value)
    sources: list[TaskSource] = []
    for root in roots:
        entries = _discover_data_entries(root)
        tasks = tuple(_load_task_spec(path) for path in entries)
        if tasks:
            sources.append(TaskSource(root=root, tasks=tasks))
    return tuple(sources)


def _discover_data_entries(root: Path) -> list[Path]:
    """Discover tasks without losing ordered directory-symlink views.

    ``Path.rglob`` deliberately does not descend through directory symlinks.
    ResearchClaw's immutable seed views are made of top-level relative links,
    so check each direct child as well as recursively walking ordinary
    directories.  Keep the logical link path: its name encodes the shuffled
    order and all reward/input paths remain inside the copied dataset root.
    """
    entries = set(root.rglob("data_entry.json"))
    if root.is_dir():
        for child in root.iterdir():
            candidate = child / "data_entry.json"
            if candidate.is_file():
                entries.add(candidate)
    return sorted(entries)


def _split_source_paths(raw_value: str) -> list[Path]:
    paths: list[Path] = []
    for part in raw_value.replace("\n", ",").split(","):
        item = part.strip()
        if item:
            paths.append(Path(os.path.expanduser(item)))
    return paths


def _load_task_spec(entry_path: Path) -> TaskSpec:
    with entry_path.open(encoding="utf-8") as handle:
        entry = _require_dict(json.load(handle), str(entry_path))

    task_dir = entry_path.parent
    metadata = _require_dict(entry.get("metadata"), "metadata")
    input_mount_dir: str | None = None
    if "input_mount_dir" in entry:
        input_mount_dir = _require_string(entry.get("input_mount_dir"), "input_mount_dir")
        input_files_dir = task_dir / "input_files"
        if not input_files_dir.exists():
            raise FileNotFoundError(f"input_mount_dir is set but input_files/ not found: {input_files_dir}")

    reward_sh = task_dir / "reward" / "reward.sh"
    reward_py = task_dir / "reward" / "check.py"
    if not reward_sh.exists() and not reward_py.exists():
        raise FileNotFoundError(f"reward checker not found under: {task_dir / 'reward'}")

    return TaskSpec(
        task_dir=task_dir,
        task_id=_require_string(entry.get("task_id"), "task_id"),
        user_query=_require_string(entry.get("user_query"), "user_query"),
        input_mount_dir=input_mount_dir,
        metadata=metadata,
    )


def _container_path_to_host_path(workspace_dir: Path, container_path: str) -> Path:
    container_pure = PurePosixPath(container_path)
    working_pure = PurePosixPath("/root/.openclaw/workspace")
    if not str(container_pure).startswith(str(working_pure)):
        raise ValueError(f"path {container_path} is outside /root/.openclaw/workspace")
    if container_pure == working_pure:
        return workspace_dir
    relative = container_pure.relative_to(working_pure)
    return workspace_dir / Path(str(relative))


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value
