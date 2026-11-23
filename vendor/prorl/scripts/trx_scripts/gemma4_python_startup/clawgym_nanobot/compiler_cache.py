"""Configure isolated compiler caches before the full project is importable."""

import os
import re
import tempfile
from pathlib import Path


def configure() -> None:
    config = os.environ.get("CLAWGYM_SELF_REWARD_CONFIG")
    if not config:
        return
    # The agent process runs after the Polar runtime drops privileges.  A
    # persistent run directory is normally created by the root-owned launcher,
    # so placing per-process Torch/Triton caches beside the config makes every
    # agent fail before its first model request with PermissionError.  Compiler
    # caches are disposable runtime state; keep them in the node-local sticky
    # tmp directory, isolated by run, host, and PID.
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(config).parent.name or "unknown-run")[:48]
    base = Path(os.environ.get("CLAWGYM_COMPILER_CACHE_BASE", "/tmp/clawgym-compiler"))
    # A worker may be spawned after Polar drops privileges.  Give every worker
    # an independently owned temporary directory instead of a shared
    # run/host parent whose mode/owner can be left by another UID.
    try:
        base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix=f"{run_id}-{os.getpid()}-", dir=str(base)))
    except OSError:
        fallback = Path(os.environ.get("TMPDIR", "/tmp"))
        root = Path(tempfile.mkdtemp(prefix=f"clawgym-{run_id}-{os.getpid()}-", dir=str(fallback)))
    (root / "inductor").mkdir()
    (root / "triton").mkdir()
    (root / "tvm").mkdir()
    (root / "torch").mkdir()
    os.environ["TVM_FFI_CACHE_DIR"] = str(root / "tvm")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(root / "torch")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    os.environ["TRITON_CACHE_MANAGER"] = "clawgym_nanobot.triton_cache:PrivateProcessCacheManager"
