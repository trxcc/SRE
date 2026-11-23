"""Runtime overrides for the bundled :mod:`clawgym_nanobot` package.

Only modules present in this directory are overridden.  All other modules are
resolved from the read-only project source tree later on ``sys.path``.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
