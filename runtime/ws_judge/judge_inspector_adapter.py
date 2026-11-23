"""Install the sealed inspector only in the current request's judge tools directory."""
from pathlib import Path
import hashlib
import os
import tempfile

ORIGINAL_SHA = "9b84cfc1c93abb8bbc33d05ed6af2ab99260fe3ad857228f4d8917a5b311ea64"
CANDIDATE_SHA = "e0aea725745bbab1ce3bebf7b67292e9490a1bf48b6175996a9e3f6fe6e2f1da"


def install_inspector(judge_workspace, case_dir, candidate, *, allowed_root):
    judge, case = Path(judge_workspace), Path(case_dir)
    root = Path(allowed_root).resolve(strict=True)
    if judge.is_symlink() or case.is_symlink():
        raise ValueError("judge and case directories must not be symlinks")
    judge, case = judge.resolve(strict=True), case.resolve(strict=True)
    if not case.is_relative_to(root) or case == root or judge != case / "judge_workspace":
        raise ValueError("judge workspace must be directly inside an owned rollout case")
    tools = judge / "tools"
    if tools.is_symlink() or not tools.is_dir():
        raise ValueError("judge tools must be a local directory")
    target = tools / "inspect_workspace_file.py"
    data = Path(candidate).read_bytes()
    if hashlib.sha256(data).hexdigest() != CANDIDATE_SHA:
        raise ValueError("candidate inspector hash mismatch")
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    if before not in (ORIGINAL_SHA, CANDIDATE_SHA):
        raise ValueError("unexpected existing inspector hash")
    # Replace the directory entry, never write through its original symlink.
    fd, temporary = tempfile.mkstemp(prefix=".sealed-inspector-", dir=tools)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"before_sha256": before, "after_sha256": CANDIDATE_SHA,
            "path": str(target), "scope": "judge_workspace_only"}
