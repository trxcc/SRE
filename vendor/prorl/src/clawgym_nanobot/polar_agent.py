"""Nanobot command invoked by Polar's ClawGym agent harness."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from polar.agent.presets.nanobot import runtime as nanobot_runtime


def _retain_failed_train_tool_timing(
    *,
    log_dir: Path,
    session_id: str,
    turn_type: str,
    returncode: int | None,
    error: BaseException | None,
) -> None:
    """Retain only lightweight tool timing for a failed training agent turn."""
    destination_raw = os.environ.get("CLAWGYM_FAILED_TRAIN_TOOL_TIMING_DIR", "").strip()
    train_turn_type = os.environ.get("CLAWGYM_NANOBOT_TRAIN_TURN_TYPE", "main").strip() or "main"
    if not destination_raw or turn_type != train_turn_type:
        return

    source = log_dir / session_id / "tool_timing.jsonl"
    destination = Path(destination_raw)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            shutil.copy2(source, destination / f"{session_id}.jsonl")
        metadata = {
            "session_id": session_id,
            "task_id": os.environ.get("TASK_ID", ""),
            "rollout_step": os.environ.get("POLAR_ROLLOUT_STEP", ""),
            "turn_type": turn_type,
            "returncode": returncode,
            "error_type": type(error).__name__ if error is not None else None,
            "error": str(error)[:2000] if error is not None else None,
            "retained_at": time.time(),
            "timing_file_present": source.is_file(),
        }
        (destination / f"{session_id}.meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            f"WARNING: failed to retain tool timing for {session_id}: {exc}",
            file=sys.stderr,
        )


def run_agent(
    *,
    instruction: str,
    workspace: Path,
    run_dir: Path,
    log_dir: Path,
    session_id: str,
    turn_type: str,
    model_id: str | None,
) -> int:
    workspace.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    sampling_params = {
        "temperature": float(os.environ.get("NANOBOT_TEMPERATURE", os.environ.get("ROLLOUT_TEMPERATURE", "0.7"))),
        "top_p": float(os.environ.get("NANOBOT_TOP_P", os.environ.get("ROLLOUT_TOP_P", "1.0"))),
        "top_k": int(os.environ.get("NANOBOT_TOP_K", os.environ.get("ROLLOUT_TOP_K", "-1"))),
        "max_new_tokens": int(os.environ.get("OPENCLAW_MAX_TOKENS", "8192")),
    }
    config_path = nanobot_runtime.write_nanobot_config(
        workspace_dir=workspace,
        run_dir=run_dir,
        adapter_base_url=base_url,
        session_id=session_id,
        turn_type=turn_type,
        sampling_params=sampling_params,
        model_id=model_id or f"{turn_type}__{session_id}",
        context_window=int(os.environ.get("OPENCLAW_CONTEXT_WINDOW", "65536")),
        max_tokens=int(os.environ.get("OPENCLAW_MAX_TOKENS", "8192")),
    )
    timeout_raw = os.environ.get("NANOBOT_TURN_TIMEOUT", os.environ.get("NANOBOT_TASK_TIMEOUT", "900"))
    timeout = int(timeout_raw) if timeout_raw and int(timeout_raw) > 0 else None
    existing_agent_ends = set((log_dir / session_id).glob("*_agent-end.json"))
    try:
        result = nanobot_runtime.run_nanobot_turn(
            workspace_dir=workspace,
            config_path=config_path,
            session_id=session_id,
            turn_type=turn_type,
            message=instruction,
            log_dir=log_dir,
            timeout=timeout,
            env_overrides={
                "NANOBOT_TASK_ID": os.environ.get("TASK_ID", ""),
                "NANOBOT_ROLLOUT_ID": os.environ.get("POLAR_ROLLOUT_STEP", ""),
                "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": (
                    "0" if turn_type == os.environ.get("CLAWGYM_NANOBOT_TRAIN_TURN_TYPE", "main") else "1"
                ),
            },
        )
    except subprocess.TimeoutExpired as exc:
        _retain_failed_train_tool_timing(
            log_dir=log_dir,
            session_id=session_id,
            turn_type=turn_type,
            returncode=124,
            error=exc,
        )
        if exc.output:
            output = exc.output.decode(errors="replace") if isinstance(exc.output, bytes) else exc.output
            print(output, end="" if output.endswith("\n") else "\n")
        if exc.stderr:
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        print(
            f"Nanobot turn exceeded {timeout} seconds; reporting a timeout to Polar",
            file=sys.stderr,
        )
        # 124 is the conventional timeout exit code. The Polar gateway maps
        # this to AgentRunResult(status="timeout") instead of a generic ERROR.
        return 124
    except Exception as exc:
        _retain_failed_train_tool_timing(
            log_dir=log_dir,
            session_id=session_id,
            turn_type=turn_type,
            returncode=None,
            error=exc,
        )
        raise
    # A tool-loop terminal can still have a zero subprocess exit status.
    # Surface it as a policy failure so Polar masks the whole training trace.
    if turn_type == os.environ.get("CLAWGYM_NANOBOT_TRAIN_TURN_TYPE", "main"):
        for path in set((log_dir / session_id).glob("*_agent-end.json")) - existing_agent_ends:
            try:
                end = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if end.get("stop_reason") == "tool_loop":
                print("Repeated tool loop detected during training", file=sys.stderr)
                return 65
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode:
        _retain_failed_train_tool_timing(
            log_dir=log_dir,
            session_id=session_id,
            turn_type=turn_type,
            returncode=int(result.returncode),
            error=None,
        )
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-b64", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--turn-type", default="main")
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args(argv)
    instruction = base64.b64decode(args.instruction_b64).decode("utf-8")
    return run_agent(
        instruction=instruction,
        workspace=Path(args.workspace),
        run_dir=Path(args.run_dir),
        log_dir=Path(args.log_dir),
        session_id=args.session_id,
        turn_type=args.turn_type,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
