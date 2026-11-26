"""Keep completed HF milestones independently of evaluation and rolling retention.

Opt-in through CLAWGYM_HF_MILESTONE_PIN_ROOT. No weight payload is read here.
A failed or interrupted pin raises before retention can delete any candidate.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path


def _small(path: Path, limit: int = 100000) -> bytes:
    if not path.is_file() or path.is_symlink() or path.stat().st_size >= limit:
        raise ValueError(f"Missing, unsafe, or oversized checkpoint metadata: {path.name}")
    return path.read_bytes()


def _inventory(root: Path) -> dict:
    rows = {}
    for path in sorted(root.rglob("*")):
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise ValueError("Checkpoint must contain no symlinks")
        if stat.S_ISDIR(st.st_mode):
            continue
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("Checkpoint must contain only regular files")
        if path.name == "milestone-pin.json":
            continue
        rows[str(path.relative_to(root))] = {
            "device": st.st_dev, "inode": st.st_ino,
            "size": st.st_size, "mtime_ns": st.st_mtime_ns,
        }
    return rows


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())



PIN_ENV_KEYS = (
    "CLAWGYM_HF_MILESTONE_PIN_ROOT",
    "CLAWGYM_HF_MILESTONE_PIN_STEPS",
    "CLAWGYM_HF_PIN_QUESTION_BATCH",
    "CLAWGYM_HF_PIN_CANDIDATES_PER_QUESTION",
    "CLAWGYM_HF_PIN_TOTAL_QUESTIONS",
)


def require_checkpoint_pin_configuration(args) -> None:
    """Fail before GPU allocation when an opt-in milestone contract is incomplete."""
    if not os.environ.get(PIN_ENV_KEYS[0], "").strip():
        return
    for key in (*PIN_ENV_KEYS, "CLAWGYM_HF_CHECKPOINT_ROOT"):
        if not os.environ.get(key, "").strip():
            raise ValueError("Missing required checkpoint pin environment: " + key)
    root = Path(os.environ[PIN_ENV_KEYS[0]])
    checkpoint_root = Path(os.environ["CLAWGYM_HF_CHECKPOINT_ROOT"])
    if not root.is_absolute() or not checkpoint_root.is_absolute():
        raise ValueError("Checkpoint roots must be absolute")
    if root.resolve().is_relative_to(checkpoint_root.resolve()) or checkpoint_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("Independent pin root must be outside checkpoint retention tree")
    steps = {int(x.strip()) for x in os.environ[PIN_ENV_KEYS[1]].split(",") if x.strip()}
    if not steps or min(steps) < 0:
        raise ValueError("Invalid milestone ids")
    if any(int(os.environ[key]) <= 0 for key in PIN_ENV_KEYS[2:]):
        raise ValueError("Invalid cursor contract")
    if not getattr(args, "save_hf", None):
        raise ValueError("Required milestone has no HF save path")
    start, end = args.start_rollout_id, args.num_rollout
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
        raise ValueError("Invalid bounded rollout range")
    for step in (start, end - 1):
        source = Path(args.save_hf.format(rollout_id=step))
        if not source.is_absolute() or not source.resolve().is_relative_to(checkpoint_root.resolve() / "candidates"):
            raise ValueError("Milestone source is outside the current candidates")


def pin_completed_checkpoint(rollout_id: int, args, *, force: bool = False) -> dict:
    raw = os.environ.get("CLAWGYM_HF_MILESTONE_PIN_ROOT", "").strip()
    if not raw:
        return {"enabled": False}
    configured = os.environ.get("CLAWGYM_HF_MILESTONE_PIN_STEPS", "")
    steps = {int(x.strip()) for x in configured.split(",") if x.strip()}
    if any(x < 0 for x in steps) or rollout_id < 0:
        raise ValueError("Milestone ids must be nonnegative")
    interval = getattr(args, "eval_interval", None)
    required = force or rollout_id in steps or rollout_id == args.num_rollout - 1
    required = required or bool(interval and rollout_id > 0 and rollout_id % interval == 0)
    if not required:
        return {"enabled": True, "required": False}
    if not getattr(args, "save_hf", None):
        raise ValueError("Required milestone has no HF save path")
    checkpoint_root = Path(os.environ["CLAWGYM_HF_CHECKPOINT_ROOT"]).resolve()
    source = Path(args.save_hf.format(rollout_id=rollout_id)).absolute()
    root = Path(raw).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Milestone candidate must be a completed real directory")
    if not source.resolve().is_relative_to(checkpoint_root / "candidates"):
        raise ValueError("Milestone source is outside the current candidates")
    if root.resolve().is_relative_to(checkpoint_root) or checkpoint_root.is_relative_to(root.resolve()):
        raise ValueError("Independent pin root must be outside checkpoint retention tree")
    if root.is_symlink():
        raise ValueError("Pin root must not be a symlink")

    marker_bytes = _small(source / "checkpoint.json")
    marker = json.loads(marker_bytes)
    if (marker.get("format") != "huggingface" or marker.get("optimizer_saved") is not False
            or marker.get("rollout_id") != rollout_id):
        raise ValueError("Incomplete or mismatched HF marker")
    cursor_name = f"rollout/global_dataset_state_dict_{rollout_id}.pt"
    cursor_bytes = _small(source / cursor_name)
    # Only the owned, bounded, same-update data cursor is loaded; never a model.
    import io
    import torch
    cursor = torch.load(io.BytesIO(cursor_bytes), map_location="cpu", weights_only=False)
    batch = int(os.environ["CLAWGYM_HF_PIN_QUESTION_BATCH"])
    candidates = int(os.environ["CLAWGYM_HF_PIN_CANDIDATES_PER_QUESTION"])
    total = int(os.environ["CLAWGYM_HF_PIN_TOTAL_QUESTIONS"])
    if min(batch, candidates, total) <= 0:
        raise ValueError("Invalid cursor contract")
    offset = min(batch * (rollout_id + 1), total)
    expected = {"sample_offset": offset, "sample_group_index": offset,
                "sample_index": offset * candidates, "epoch_id": 0}
    if any(cursor.get(k) != v for k, v in expected.items()):
        raise ValueError("HF checkpoint and consumed cursor disagree")

    if any((source / name).exists() for name in ("intent.json", "milestone-pin.json")):
        raise ValueError("Reserved pin metadata already exists in the source")
    files = _inventory(source)
    weight_names = {name for name in files if name.endswith(".safetensors")}
    index = source / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(_small(index, 10000000))["weight_map"].values())
        if any(Path(name).name != name for name in names) or not names <= weight_names:
            raise ValueError("Incomplete HF shard inventory")
    if not weight_names or any(files[name]["size"] <= 0 for name in weight_names):
        raise ValueError("Missing HF weights")
    if "config.json" not in files or "tokenizer_config.json" not in files:
        raise ValueError("Missing HF model or tokenizer configuration")
    proof = {
        "rollout_id": rollout_id, "next_rollout_id": rollout_id + 1,
        "source": str(source), "optimizer_saved": False, "cursor": expected,
        "marker_SHA256": hashlib.sha256(marker_bytes).hexdigest(),
        "cursor_SHA256": hashlib.sha256(cursor_bytes).hexdigest(),
        "files": files, "weight_payload_bytes_read": 0,
        "independent_of_eval_validity": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{rollout_id}"
    staging = root / f".step-{rollout_id}.incomplete"
    with (root / ".pin.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if final.exists() or final.is_symlink():
            if final.is_symlink() or not final.is_dir():
                raise ValueError("Existing pin is not a real directory")
            prior = json.loads(_small(final / "milestone-pin.json", 1000000))
            if prior != proof or _inventory(final) != files or _inventory(source) != files:
                raise ValueError("Existing pin does not match; reconcile, never overwrite")
            return {"enabled": True, "required": True, "verified": True,
                    "already_pinned": True, "path": str(final), **proof}
        if staging.exists() or staging.is_symlink():
            raise ValueError("Interrupted pin exists; reconcile before proceeding")
        staging.mkdir(mode=0o700)
        _write_new(staging / "intent.json", proof)
        for name in files:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / name, destination, follow_symlinks=False)
        # Preserve a failed staging tree as evidence; do not silently remove it.
        observed = _inventory(staging)
        observed.pop("intent.json", None)
        if _inventory(source) != files or observed != files:
            raise ValueError("Checkpoint changed during pinning; retention forbidden")
        if (_small(staging / "checkpoint.json") != marker_bytes
                or _small(staging / cursor_name) != cursor_bytes):
            raise ValueError("Pinned metadata changed during pinning")
        _write_new(staging / "milestone-pin.json", proof)
        # Completed receipt replaces the provisional intent, without touching weights.
        (staging / "intent.json").unlink()
        for directory in sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True):
            _sync_dir(directory)
        _sync_dir(staging)
        os.rename(staging, final)
        _sync_dir(root)
    return {"enabled": True, "required": True, "verified": True,
            "already_pinned": False, "path": str(final), **proof}
