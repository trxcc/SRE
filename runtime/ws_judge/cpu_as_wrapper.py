#!/usr/bin/env python3
"""Apply a soft/hard virtual address-space cap (RLIMIT_AS) before execing a child command."""
from __future__ import annotations

import argparse
import os
import resource
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=int, required=True, help="Address-space limit in GiB")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("cpu_as_wrapper.py requires a command after --")

    if args.gib > 0 and hasattr(resource, "RLIMIT_AS"):
        limit_bytes = int(args.gib) * (1024 ** 3)
        try:
            _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if hard not in (resource.RLIM_INFINITY, -1) and hard > 0:
                limit_bytes = min(limit_bytes, hard)
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
        except (OSError, ValueError):
            # Mac/container kernels that disallow lowering RLIMIT_AS proceed without aborting.
            pass

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
