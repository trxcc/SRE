"""Process-start compatibility hooks for explicitly opted-in ProRL jobs.

Python imports ``sitecustomize`` automatically during interpreter startup when
this repository's ``src`` directory is on ``PYTHONPATH``. Keep every hook
strictly opt-in so ordinary development commands are unaffected.
"""

from __future__ import annotations

import os

if os.environ.get("CLAWGYM_SELF_REWARD_CONFIG"):
    from clawgym_nanobot.compiler_cache import configure as _configure_compiler_cache
    _configure_compiler_cache()


if os.environ.get("PRORL_MASK_BROKEN_SKLEARN", "0") == "1":
    try:
        from slime.backends.sglang_utils.optional_sklearn import (
            disable_broken_optional_sklearn,
        )

        disable_broken_optional_sklearn()
    except Exception:
        # Startup compatibility must never make the interpreter unusable. The
        # normal import will still expose the original actionable exception.
        pass
