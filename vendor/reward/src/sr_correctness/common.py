from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


RUNTIME_EXCLUDES = {
    ".git",
    ".nanobot",
    "__pycache__",
    "sessions",
    "memory",
    "reward",
    "_judge",
    "judge_workspace",
    "reference_answer",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_clean_tree(source: Path, destination: Path, *, extra_excludes: Iterable[str] = ()) -> None:
    excluded = RUNTIME_EXCLUDES | set(extra_excludes)

    def ignore(_root: str, names: list[str]) -> set[str]:
        return {name for name in names if name in excluded or name.startswith(".nanobot-no-web-bin")}

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=False, ignore=ignore)


def link_clean_tree(source: Path, destination: Path, *, extra_excludes: Iterable[str] = ()) -> None:
    """Expose an immutable candidate workspace using the native judge's symlink pattern."""
    excluded = RUNTIME_EXCLUDES | set(extra_excludes)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for child in source.iterdir():
        if child.name in excluded or child.name.startswith(".nanobot-no-web-bin"):
            continue
        (destination / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path, *, excludes: Iterable[str] = ()) -> dict[str, str]:
    excluded = set(excludes)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        # Nanobot creates short-lived no-web command shims inside the workspace.
        # They are harness state, not agent-authored task output, so apply the
        # same exclusion used by copy_clean_tree/link_clean_tree here as well.
        if (
            any(part in excluded or part.startswith(".nanobot-no-web-bin") for part in rel.parts)
            or path.suffix == ".pyc"
            or not path.is_file()
        ):
            continue
        result[rel.as_posix()] = sha256_file(path)
    return result


def tree_hash(root: Path, *, excludes: Iterable[str] = ()) -> str:
    encoded = json.dumps(tree_manifest(root, excludes=excludes), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty secret file: {path}")
    return value


def assert_output_root(path: Path) -> None:
    resolved = path.resolve()
    configured = os.environ.get("DEAL_OUTPUT_ROOT", "").strip()
    if not configured:
        return
    allowed = Path(configured).expanduser().resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"new output must be under {allowed}: {resolved}")
