"""Normalize a missing or partial optional sklearn install for Transformers."""

from __future__ import annotations


def disable_broken_optional_sklearn() -> bool:
    """Return True after masking a sklearn install that cannot provide metrics."""
    try:
        from sklearn.metrics import roc_curve as _roc_curve  # noqa: F401
    except Exception:
        from transformers import utils as transformers_utils
        from transformers.utils import import_utils as transformers_import_utils

        # Transformers imports this helper into generation modules by value.
        # Replace both public references before those modules are imported.
        transformers_import_utils.is_sklearn_available = lambda: False
        transformers_utils.is_sklearn_available = lambda: False
        return True
    return False
