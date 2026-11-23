#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from clawgym_nanobot.researchclaw_eval_merge import merge_eval_runs, write_merge_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tasks", required=True)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-retry-view", action="store_true")
    args = parser.parse_args()

    rows, summary = merge_eval_runs(args.expected_tasks, args.run_root)
    write_merge_outputs(
        args.output_dir,
        rows,
        summary,
        write_retry_view=args.write_retry_view,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
