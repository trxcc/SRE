"""Configure disposable, per-process compiler caches for Nanobot workers."""
import os
import re
import tempfile
from pathlib import Path


def configure() -> None:
    config = os.environ.get("CLAWGYM_SELF_REWARD_CONFIG")
    if not config:
        return
    # Polar creates the persistent run root as the launcher user, then starts
    # Nanobot after dropping privileges.  A cache below that run root can
    # therefore be readable but not writable by the agent.  Compiler output is
    # disposable runtime state, so isolate it in the node-local sticky tmpdir.
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(config).parent.name or "unknown-run")[:48]
    base = Path(os.environ.get("CLAWGYM_COMPILER_CACHE_BASE", "/tmp/clawgym-compiler"))
    # Never share a run/host directory between workers.  Polar can import this
    # module once as root and then spawn agents under a different UID; a shared
    # 755 parent then makes the later agent fail with PermissionError.  Each
    # worker owns its own disposable directory, so no chmod or cross-UID
    # coordination is needed.
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
