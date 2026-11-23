#!/usr/bin/env python3
"""Unified single-host launcher (4 train + 4 rollout GPUs) for DEAL-WS and DEAL-RS."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from configure_run import ROOT, configure


def wait_http(url: str, process: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited early (rc={process.returncode}); see its log")
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Service startup timed out waiting for {url}")


def port_free(port: int) -> None:
    with socket.socket() as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is occupied; choose different ports in your config"
            ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="Path to local .env config file")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Render and validate the native training CLI on CPU without GPU or judge calls",
    )
    args = ap.parse_args()
    os.umask(0o077)
    work = configure(args.config)
    env = os.environ.copy()
    env.update(json.loads((work / "private/environment.json").read_text(encoding="utf-8")))
    services = json.loads((work / "private/services.json").read_text(encoding="utf-8"))
    benchmark = services["benchmark"]

    gt_key = Path(services["gt_key_file"]).read_text(encoding="utf-8").strip()
    env.update(
        RESEARCHCLAW_JUDGE_BASE_URL=services["gt_base_url"],
        RESEARCHCLAW_JUDGE_BASE_URL_OVERRIDE=services["gt_base_url"],
        RESEARCHCLAW_JUDGE_MODEL=services["gt_model"],
        RESEARCHCLAW_JUDGE_MODEL_OVERRIDE=services["gt_model"],
        RESEARCHCLAW_JUDGE_API_KEY=gt_key,
    )
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "GROUP_RANK", "ROLE_RANK"):
        env.pop(k, None)

    if args.check:
        env["RESEARCHCLAW_RENDER_ONLY"] = "1"
        subprocess.run(["bash", str(ROOT / "runtime/run_core.sh")], env=env, check=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_native.py"), str(work / "logs/captured-cli.json")],
            env=env,
            check=True,
        )
        print(f"Native CPU preflight passed for {benchmark} ({services['model_name']}, 4 train + 4 rollout).")
        return

    subprocess.run([sys.executable, str(ROOT / "scripts/doctor.py"), "--gpu"], env=env, check=True)
    ports = {
        int(env[k])
        for k in ("MASTER_PORT", "RAY_DASHBOARD_PORT", "POLAR_ROLLOUT_PORT", "POLAR_GATEWAY_PORT", "SGLANG_ROUTER_PORT")
    }
    if benchmark == "deal-ws":
        ports.add(int(services["ws_judge_proxy_port"]))
        ports.add(int(services["ws_judge_worker_port"]))
    for port in sorted(ports):
        port_free(port)

    children: list[tuple[str, subprocess.Popen, object]] = []

    def launch(name: str, argv: list[str], child_env: dict[str, str]) -> subprocess.Popen:
        log = (work / "logs" / f"{name}.log").open("ab", buffering=0)
        proc = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((name, proc, log))
        return proc

    def stop_owned() -> None:
        for _name, proc, _log in reversed(children):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _name, proc, log in reversed(children):
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close()

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        watched: list[subprocess.Popen] = []

        if benchmark == "deal-ws":
            judge_env = dict(env, CUDA_VISIBLE_DEVICES="")
            ws_judge = launch(
                "ws_judge",
                ["bash", str(ROOT / "runtime/ws_judge/judge_sidecar.sh"), "run"],
                judge_env,
            )
            wait_http(f"{services['ws_judge_worker_url']}/health", ws_judge, timeout=180)
            watched.append(ws_judge)

        reward_env = dict(env, CUDA_VISIBLE_DEVICES="")
        reward_env["NANOBOT_EXTRA_PYTHONPATH"] = os.pathsep.join([
            str(ROOT / "vendor/nanobot"),
            str(ROOT / "vendor/prorl/scripts/trx_scripts/gemma4_python_startup"),
        ])
        reward = launch(
            "reward",
            [sys.executable, str(ROOT / "vendor/reward/training_service.py"), str(work / "private/reward-service.json")],
            reward_env,
        )
        watched.append(reward)

        ray_head = launch(
            "ray",
            [
                "ray", "start", "--head", "--block",
                "--node-ip-address", env["MASTER_ADDR"],
                "--port", env["MASTER_PORT"],
                "--dashboard-host", "0.0.0.0",
                "--dashboard-port", env["RAY_DASHBOARD_PORT"],
                "--num-gpus", "8",
                "--disable-usage-stats",
                "--temp-dir", env["RAY_TMPDIR"],
            ],
            env,
        )
        wait_http(f"http://{env['MASTER_ADDR']}:{env['RAY_DASHBOARD_PORT']}/api/version", ray_head, timeout=120)
        watched.append(ray_head)

        train = launch("train", ["bash", str(ROOT / "runtime/run_core.sh")], env)
        print(f"Training started ({benchmark} / {services['model_name']}). Logs: {work / 'logs'}", flush=True)
        print("GPU layout: 4 train + 4 rollout (8 GPUs). Frozen pi_theta_0 & GT judge use configured APIs.", flush=True)

        while train.poll() is None:
            for svc in watched:
                if svc.poll() is not None:
                    raise RuntimeError("A required background service exited; inspect logs under " + str(work / "logs"))
            time.sleep(5)
        code = train.returncode
        if code:
            print(
                f"Training exited with code {code}. Resume with scripts/resume_latest.sh --config {args.config}",
                file=sys.stderr,
            )
        raise SystemExit(code)
    finally:
        submit = work / "logs/ray-submit-output.log"
        if submit.exists():
            ids = re.findall(r"raysubmit_[A-Za-z0-9_-]+", submit.read_text(encoding="utf-8"))
            if ids:
                try:
                    subprocess.run(
                        [
                            "ray", "job", "stop",
                            "--address", f"http://{env['MASTER_ADDR']}:{env['RAY_DASHBOARD_PORT']}",
                            "--no-wait", ids[0],
                        ],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
        stop_owned()


if __name__ == "__main__":
    main()
