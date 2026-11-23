#!/usr/bin/env python3
"""Preseed Ray's known agent ports before its fixed 15-second file wait expires.

Ray 2.x may spend more than 15 seconds importing the dashboard agent on H200
nodes. The ports are supplied explicitly to ``ray start``, so publishing the
same values as soon as raylet announces its node ID is safe. The agents later
write their actual (identical) values to these files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import time


NODE_ID = re.compile(r"\bnode_id=([0-9a-f]{56})\b")


def seed(tmp_dir: Path, ports: dict[str, int], timeout: float = 90.0) -> None:
    link = tmp_dir / "session_latest"
    previous = link.resolve() if link.exists() else None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = link.resolve() if link.exists() else None
        if session and session != previous:
            log = session / "logs" / "raylet.out"
            if log.is_file():
                match = NODE_ID.search(log.read_text(errors="replace"))
                if match:
                    node_id = match.group(1)
                    for name, port in ports.items():
                        target = session / f"{name}_{node_id}"
                        if target.exists():
                            if target.read_text().strip() != str(port):
                                raise RuntimeError(f"Ray agent port mismatch: {target}")
                            continue
                        temporary = session / f".{name}_{node_id}.{os.getpid()}.tmp"
                        temporary.write_text(str(port))
                        os.replace(temporary, target)
                    print(f"RAY_AGENT_PORTS_SEEDED session={session} node_id={node_id}", flush=True)
                    return
        time.sleep(0.05)
    raise TimeoutError(f"Ray session/node ID did not appear within {timeout}s: {tmp_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tmp_dir", type=Path)
    parser.add_argument("grpc", type=int)
    parser.add_argument("listen", type=int)
    parser.add_argument("metrics", type=int)
    parser.add_argument("runtime_env", type=int)
    args = parser.parse_args()
    ports = {
        "metrics_agent_port": args.grpc,
        "dashboard_agent_listen_port": args.listen,
        "metrics_export_port": args.metrics,
        "runtime_env_agent_port": args.runtime_env,
    }
    if any(not 1 <= port <= 65535 for port in ports.values()) or len(set(ports.values())) != 4:
        parser.error("Ray agent ports must be four distinct TCP ports")
    seed(args.tmp_dir, ports)


if __name__ == "__main__":
    main()
