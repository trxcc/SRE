#!/usr/bin/env python3
"""Create isolated single-host run configuration for DEAL-WS / DEAL-RS.

Credentials remain in chmod-600 private files/environment and are never hardcoded in configs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
import shlex
import site
import socket
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_SPEC = {
    "deal-rs": {
        "dataset": "researchclaw",
        "total_tasks": 1800,
        "num_rollout": 113,
        "rubric_prompt": "rubric_agent_researchclaw_relaxed.md",
        "reward_template": "configs/reward-service.deal-rs.template.json",
        "recipes": {
            "gemma4-e4b-it": "configs/deal-rs-gemma4.recipe.json",
            "qwen3.5-9b": "configs/deal-rs-qwen35.recipe.json",
        },
    },
    "deal-ws": {
        "dataset": "wsclaw",
        "total_tasks": 2000,
        "num_rollout": 125,
        "rubric_prompt": "rubric_agent_wsclaw.md",
        "reward_template": "configs/reward-service.deal-ws.template.json",
        "recipes": {
            "gemma4-e4b-it": "configs/deal-ws-gemma4.recipe.json",
            "qwen3.5-9b": "configs/deal-ws-qwen35.recipe.json",
        },
    },
}


def read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        key, value = raw.removeprefix("export ").split("=", 1)
        parts = shlex.split(value, comments=True)
        if len(parts) > 1:
            raise ValueError("Quote values containing spaces: " + key)
        result[key.strip()] = parts[0] if parts else ""
    return result


def write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def write_secret_file(path: Path, secret_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(secret_text.strip() + "\n")


def validate_cursor(rid: int, cursor: dict, *, benchmark: str = "deal-rs") -> int:
    spec = BENCHMARK_SPEC[benchmark]
    max_rid = spec["num_rollout"] - 1
    total_tasks = spec["total_tasks"]
    assert type(rid) is int and 0 <= rid <= max_rid, f"checkpoint rollout {rid} outside finite pass [0, {max_rid}]"
    off = int(cursor["sample_offset"])
    assert off == int(cursor["sample_group_index"]) == min((rid + 1) * 16, total_tasks), (
        f"cursor offset {off} differs from committed rollout {rid}"
    )
    assert int(cursor["sample_index"]) == off * 8 and int(cursor["epoch_id"]) == 0, (
        "SRE cursor must match 8 candidates per task (1 GT anchor + 7 SR) and epoch 0"
    )
    return off


def read_checkpoint(cp: Path, *, benchmark: str = "deal-rs") -> tuple[int, int, str]:
    """Read only small committed metadata; never load model weights on CPU."""
    import torch

    marker = json.loads((cp / "checkpoint.json").read_text(encoding="utf-8"))
    assert marker["format"] == "huggingface" and marker["optimizer_saved"] is False
    rid = int(marker["rollout_id"])
    index = cp / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    else:
        names = {x.name for x in cp.glob("*.safetensors")}
    assert names and all(
        Path(n).name == n and (cp / n).is_file() and (cp / n).stat().st_size > 0
        for n in names
    ), "Incomplete HF weight file inventory"
    cursor_file = cp / "rollout" / f"global_dataset_state_dict_{rid}.pt"
    if not cursor_file.is_file():
        raise ValueError("Missing bundled committed rollout cursor")
    assert cursor_file.stat().st_size < 100000
    cursor = torch.load(cursor_file, map_location="cpu", weights_only=False)
    off = validate_cursor(rid, cursor, benchmark=benchmark)
    return rid + 1, off, str(cursor_file)


def _check_secret_file(path_str: str, label: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_mode & 0o077 != 0:
        raise PermissionError(f"{label} must have chmod 600: {path}")
    return path


def _resolve_model_config_script(model_name: str, v: Path) -> Path:
    if model_name == "gemma4-e4b-it":
        return v / "prorl/scripts/trx_scripts/gemma4_e_official_model_args.sh"
    if model_name == "qwen3.5-9b":
        return v / "slime/scripts/models/qwen3.5-9B.sh"
    raise ValueError(f"Unsupported model: {model_name!r}")


def _discover_package_parent(module_name: str, fallback: Path) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is not None and spec.origin is not None:
        origin = Path(spec.origin).resolve()
        parent = origin.parent.parent if origin.name == "__init__.py" else origin.parent
        if parent.is_dir():
            return parent
    return fallback


def configure(config_file: Path, attempt: str | None = None) -> Path:
    c = read_env(config_file)
    benchmark = c.get("BENCHMARK", "").strip().lower()
    if benchmark not in BENCHMARK_SPEC:
        raise ValueError(f"BENCHMARK must be one of {sorted(BENCHMARK_SPEC)}, got {benchmark!r}")
    spec = BENCHMARK_SPEC[benchmark]

    model_name = c.get("MODEL", "gemma4-e4b-it").strip()
    if model_name not in spec["recipes"]:
        raise ValueError(f"MODEL must be one of {sorted(spec['recipes'])}, got {model_name!r}")

    base = Path(c["BASE_MODEL"]).expanduser().resolve()
    assert (base / "config.json").is_file(), "BASE_MODEL must be a valid HF model directory with config.json"

    data_dir = Path(c.get("DATA_DIR") or (ROOT / "data" / benchmark)).expanduser().resolve()
    assert (data_dir / "DATA_READY.json").is_file(), (
        f"Prepared data not found at {data_dir}. Run scripts/prepare_data.py --benchmark {benchmark} first."
    )
    data_meta = json.loads((data_dir / "DATA_READY.json").read_text(encoding="utf-8"))
    train_tasks_dir = Path(data_meta["train_dir"]).resolve()
    eval_tasks_dir = Path(data_meta["eval_dir"]).resolve()
    assert train_tasks_dir.is_dir() and eval_tasks_dir.is_dir(), "Prepared train/eval task directories missing"

    run = Path(c["RUN_DIR"]).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    attempt = attempt or time.strftime("%Y%m%dT%H%M%S") + f"-{time.time_ns() % 1000000:06d}"
    work = run / "attempts" / attempt
    if work.exists():
        raise FileExistsError("Attempt already exists: " + str(work))
    work.mkdir(parents=True)

    cp = Path(c.get("RESUME_HF") or base).expanduser().resolve()
    start, offset, cursor_file = (
        read_checkpoint(cp, benchmark=benchmark) if c.get("RESUME_HF") else (0, 0, None)
    )

    gt_key = _check_secret_file(c["GT_API_KEY_FILE"], "GT_API_KEY_FILE")
    gt_url = c["GT_BASE_URL"].strip()
    gt_model = c.get("GT_MODEL", "").strip()
    assert gt_url.startswith(("http://", "https://")) and "YOUR-" not in gt_url, "Configure GT_BASE_URL"
    assert gt_model and "YOUR-" not in gt_model, "Configure GT_MODEL"

    sr_url = c["SELF_REWARD_BASE_URL"].strip()
    sr_model = c.get("SELF_REWARD_MODEL", model_name).strip()
    sr_key = _check_secret_file(c["SELF_REWARD_API_KEY_FILE"], "SELF_REWARD_API_KEY_FILE")
    assert sr_url.startswith(("http://", "https://")) and "YOUR-" not in sr_url, "Configure SELF_REWARD_BASE_URL"
    assert sr_model and "YOUR-" not in sr_model, "Configure SELF_REWARD_MODEL"

    train_gpus = [x.strip() for x in c.get("TRAIN_GPUS", "0,1,2,3").split(",") if x.strip()]
    rollout_gpus = [x.strip() for x in c.get("ROLLOUT_GPUS", "4,5,6,7").split(",") if x.strip()]
    total_gpus = len(train_gpus) + len(rollout_gpus)
    assert train_gpus and rollout_gpus and len(set(train_gpus + rollout_gpus)) == total_gpus, (
        "TRAIN_GPUS and ROLLOUT_GPUS must be non-empty and disjoint GPU device ID lists"
    )

    host = c.get("HOST_IP") or socket.gethostbyname(socket.gethostname())
    hostname = socket.gethostname()
    env = json.loads((ROOT / spec["recipes"][model_name]).read_text(encoding="utf-8"))
    gpus_per_engine = int(c.get("ROLLOUT_GPUS_PER_ENGINE") or env.get("ROLLOUT_GPUS", "1"))
    assert gpus_per_engine >= 1 and len(rollout_gpus) % gpus_per_engine == 0, (
        f"ROLLOUT_GPUS count ({len(rollout_gpus)}) must be divisible by ROLLOUT_GPUS_PER_ENGINE ({gpus_per_engine})"
    )
    default_tp = next(tp for tp in (4, 2, 1) if len(train_gpus) % tp == 0)
    tp_size = int(c.get("TP_SIZE") or default_tp)
    cp_size = int(c.get("CP_SIZE") or env.get("CP_SIZE", "1"))
    pp_size = int(c.get("PP_SIZE") or env.get("PP_SIZE", "1"))
    assert tp_size >= 1 and cp_size >= 1 and pp_size >= 1 and len(train_gpus) % (tp_size * cp_size * pp_size) == 0, (
        f"TRAIN_GPUS count ({len(train_gpus)}) must be divisible by TP*CP*PP ({tp_size}*{cp_size}*{pp_size})"
    )

    v = ROOT / "vendor"
    site_pkgs = site.getsitepackages()
    lib = Path(site_pkgs[0]) if site_pkgs else Path(sysconfig.get_path("purelib"))
    transformers_root = _discover_package_parent("transformers", lib)
    startup = v / "prorl/scripts/trx_scripts/gemma4_python_startup"
    model_config_script = _resolve_model_config_script(model_name, v)

    for port_key in (
        "MASTER_PORT",
        "RAY_DASHBOARD_PORT",
        "POLAR_ROLLOUT_PORT",
        "POLAR_GATEWAY_PORT",
        "SGLANG_ROUTER_PORT",
        "WSCLAW_JUDGE_PROXY_PORT",
        "WSCLAW_JUDGE_WORKER_PORT",
    ):
        if c.get(port_key):
            env[port_key] = c[port_key]

    sglang_root = c.get("SGLANG_ROOT", "").strip()
    run_id = run.name + "-" + attempt
    exec_guard_dir = work / "logs/exec-memory/rank0"
    env.update({
        "NANOBOT_CONDA_ENV_NAME": sys.prefix,
        "SLIME_CONTAINER_PATH": (
            f"{Path(sys.executable).resolve().parent}:"
            + os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        ),
        "BUNDLE_ROOT": str(ROOT),
        "STATE_ROOT": str(work),
        "RUN_ID": run_id,
        "LOG_DIR": str(work / "logs"),
        "SAVE_DIR": str(run / "checkpoints"),
        "PROJECT_ROOT": str(v / "prorl"),
        "SLIME_ROOT": str(v / "slime"),
        "NANOBOT_ROOT": str(v / "nanobot"),
        "NANOBOT_RL_RUNNER": str(v / "nanobot/nanobot/rl_runner.py"),
        "NANOBOT_EXTRA_PYTHONPATH": ":".join(
            map(str, [ROOT / "runtime/judge_gate", v / "nanobot", startup])
        ),
        "PYTHON_STARTUP_OVERLAY": str(startup),
        "SGLANG_ROOT": sglang_root,
        "MEGATRON_BRIDGE_ROOT": str(v / "megatron-bridge"),
        "MEGATRON_ROOT": str(v / "megatron-bridge/3rdparty/Megatron-LM"),
        "NEMO_RL_ROOT": str(v / "nemo-rl"),
        "NEMO_AUTOMODEL_PYDEPS": str(lib),
        "NEMO_AUTOMODEL_SUPPORT_PYDEPS": str(ROOT / "runtime/support"),
        "NEMO_AUTOMODEL_TRANSFORMERS_ROOT": str(transformers_root),
        "AUTOMODEL_CONFIG": str(v / "nemo-rl/examples/configs/grpo_math_8B_megatron.yaml"),
        "MODEL_CONFIG_SCRIPT": str(model_config_script),
        "POLAR_TEMPLATE_DIR": str(v / "prorl/scripts/clawgym_nanobot/polar"),
        "CLAWGYM_ATTENTION_PYTHON_OVERLAY": str(lib),
        "PYTHON_DEP_OVERLAY_ROOT": "",
        "GEMMA4_PYTHON_INCLUDE_DIR": sysconfig.get_path("include"),
        "HF_CHECKPOINT": str(cp),
        "REF_LOAD": str(base),
        "SGLANG_HF_CHECKPOINT": str(cp),
        "LOAD_CKPT": str(cp) if start else "",
        "LOAD_DIR": "",
        "START_ROLLOUT_ID": str(start),
        "SKIP_INITIAL_EVAL": "1" if 0 < start < spec["num_rollout"] else "0",
        "OPENCLAW_TRAIN_TASKS": str(train_tasks_dir),
        "OPENCLAW_EVAL_TASKS": str(eval_tasks_dir),
        "MASTER_ADDR": host,
        "RAY_JOB_SUBMIT_ADDR": host,
        "TRAIN_NODE_HOST": host,
        "CLAWGYM_NODE_HOSTS": host,
        "POLAR_GATEWAY_HOST": host,
        "SLIME_ACTOR_NODE_IPS": host,
        "POLAR_EXACT_TOKEN_STUDENT_HOSTS": host,
        "RAY_GPUS_PER_NODE": str(total_gpus),
        "NUM_GPUS_PER_NODE": str(total_gpus),
        "ACTOR_GPUS_PER_NODE": str(len(train_gpus)),
        "ROLLOUT_NUM_GPUS": str(len(rollout_gpus)),
        "ROLLOUT_GPUS": str(gpus_per_engine),
        "POLAR_EXPECTED_ROLLOUT_ENGINES": str(len(rollout_gpus) // gpus_per_engine),
        "TP_SIZE": str(tp_size),
        "CP_SIZE": str(cp_size),
        "PP_SIZE": str(pp_size),
        "NODE_RANK": "0",
        "CUDA_VISIBLE_DEVICES": ",".join(train_gpus + rollout_gpus),
        "RAY_TMPDIR": str(run / "ray"),
        "STOP_FILE": str(work / "stop-workers"),
        "POLAR_TOPOLOGY_PATH": str(work / "polar/topology.yaml"),
        "POLAR_CONFIG_PATH": str(work / "polar/config.yaml"),
        "POLAR_GATEWAY_SESSION_TMPDIR": str(work / "gateway-tmp"),
        "CLAWGYM_EVAL_SNAPSHOT_ROOT": str(run / "eval-snapshots"),
        "CLAWGYM_SELF_REWARD_CONFIG": str(work / "self-reward.json"),
        "GEMMA_SELFREWARD_MANIFEST": str(work / "manifest.json"),
        "RESEARCHCLAW_HF_RESUME_CONTRACT_MODULE": "runtime.portable_contract",
        "RESEARCHCLAW_HF_RESUME_CONTRACT_SHA": hashlib.sha256(
            (ROOT / "runtime/portable_contract.py").read_bytes()
        ).hexdigest(),
        "RESEARCHCLAW_JUDGE_SLOT_ROOT": str(work / "judge-slots"),
        "RESEARCHCLAW_ATTEMPT_ID": attempt,
        "SLIME_EXPECTED_HF_CHECKPOINT": str(cp),
        "NRL_MEGATRON_CHECKPOINT_DIR": str(
            run / "cache/nemo-conversion" / ("base" if start == 0 else f"step-{start - 1}")
        ),
        "SLIME_METRICS_JSONL_PATH": str(work / "logs/metrics.jsonl"),
        "SGLANG_MAX_RUNNING_REQUESTS": c.get("ROLLOUT_MAX_RUNNING_REQUESTS", "32"),
        "WSCLAW_EXEC_GUARD_SCOPE": f"{run_id}:rank0:{hostname}",
        "WSCLAW_EXEC_GUARD_STATE_DIR": str(exec_guard_dir),
        "WSCLAW_EXEC_GUARD_GATE_FILE": str(exec_guard_dir / "block-new-exec.json"),
        "WSCLAW_EXEC_GUARD_HEARTBEAT_FILE": str(exec_guard_dir / "heartbeat.json"),
        "TRITON_CACHE_DIR": str(run / "cache/triton"),
        "TORCH_EXTENSIONS_DIR": str(run / "cache/torch"),
        "TVM_FFI_CACHE_DIR": str(run / "cache/tvm"),
        "CLAWGYM_INTENTIONAL_ZERO_AUTHORITY": str(work / "authority.json"),
        "NANOBOT_WORKDIR": str(work / "nanobot-work"),
        "WSCLAW_ALLOWED_WORK_ROOT": str(work / "nanobot-work"),
    })

    for d in ("private", "logs", "judge-slots", "gateway-tmp", "nanobot-work"):
        (work / d).mkdir(exist_ok=True)

    authority = {
        "owner": "deal-launcher",
        "run": str(work),
        "benchmark": benchmark,
        "model": model_name,
        "candidate_count": 8,
        "intentional_zero_batches": "only all-native-zero-loss masked batches; no data skipping",
    }
    write_private(work / "authority.json", authority)
    env["CLAWGYM_INTENTIONAL_ZERO_AUTHORITY_SHA"] = hashlib.sha256(
        (work / "authority.json").read_bytes()
    ).hexdigest()

    support = ROOT / "runtime/support"
    support.mkdir(exist_ok=True)
    try:
        import hydra

        hydra_link = support / "hydra"
        if not hydra_link.exists():
            hydra_link.symlink_to(Path(hydra.__file__).parent, target_is_directory=True)
    except ImportError:
        pass

    env["SLIME_CONTAINER_LD_LIBRARY_PATH"] = (
        ":".join(str(x) for x in sorted((lib / "nvidia").glob("*/lib")) if x.is_dir())
        + ":" + str(lib / "tvm_ffi/lib")
        + ":/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    pythonpath_parts = [
        ROOT,
        startup,
        v / "prorl/src",
        v / "slime",
        v / "nemo-rl",
        v / "megatron-bridge/src",
        v / "megatron-bridge/3rdparty/Megatron-LM",
        v / "nanobot",
    ]
    if sglang_root:
        pythonpath_parts.append(Path(sglang_root) / "python")
    env["PYTHONPATH"] = ":".join(map(str, pythonpath_parts))

    manifest = {
        "format": 1,
        "benchmark": benchmark,
        "model_name": model_name,
        "run": str(work),
        "original_checkpoint": str(base),
        "actor_checkpoint": str(cp),
        "start_rollout_id": start,
        "actual_consumed_cursor": offset,
        "cursor_file": cursor_file,
        "portable_contract_sha256": env["RESEARCHCLAW_HF_RESUME_CONTRACT_SHA"],
        "eval_failure_policy": "score_zero_and_continue",
        "optimizer_saved": False,
        "optimizer_scheduler_rng_action": "native_reinitialize",
        "train_gpus": len(train_gpus),
        "rollout_gpus": len(rollout_gpus),
        "tp_size": tp_size,
        "cp_size": cp_size,
        "pp_size": pp_size,
        "rollout_gpus_per_engine": gpus_per_engine,
        "topology": f"single-host {len(train_gpus)} train + {len(rollout_gpus)} rollout",
        "candidate_count": 8,
        "hardgate": benchmark == "deal-ws",
    }
    write_private(work / "manifest.json", manifest)

    write_private(
        work / "self-reward.json",
        {
            "enabled": True,
            "mode": "self_reward_7plus1_train7",
            "queue_root": str(work / "reward-queue"),
            "snapshot_root": str(work / "snapshots"),
            "group_timeout": 10800,
            "profile": {"dataset": spec["dataset"], "model": model_name},
            "candidate_count": 8,
        },
    )

    reward = json.loads((ROOT / spec["reward_template"]).read_text(encoding="utf-8"))
    reward.update(
        queue_root=str(work / "reward-queue"),
        state_root=str(work / "reward-service"),
        project_root=str(v / "prorl"),
        rubric_prompt_path=str(v / "reward/prompts" / spec["rubric_prompt"]),
        candidate_count=8,
        gt_scope="anchor_only",
        gt_all_candidates_every_step=False,
    )
    reward["models"] = {
        model_name: {
            "job": run.name,
            "api_model": sr_model,
            "endpoint_file": str(work / "private/frozen-endpoint.json"),
            "api_key_file": str(sr_key),
        }
    }
    write_private(
        work / "private/frozen-endpoint.json",
        {
            "job": run.name,
            "model": model_name,
            "phase": "ready",
            "url": sr_url,
        },
    )

    services: dict = {
        "benchmark": benchmark,
        "model_name": model_name,
        "train_gpus": len(train_gpus),
        "rollout_gpus": len(rollout_gpus),
        "self_reward_url": sr_url,
        "self_reward_model": sr_model,
        "self_reward_key_file": str(sr_key),
        "gt_base_url": gt_url,
        "gt_model": gt_model,
        "gt_key_file": str(gt_key),
    }

    if benchmark == "deal-ws":
        ws_worker_port = int(env.get("WSCLAW_JUDGE_WORKER_PORT", "28200"))
        ws_proxy_port = int(env.get("WSCLAW_JUDGE_PROXY_PORT", "28198"))
        ws_worker_key = secrets.token_urlsafe(24)
        ws_worker_key_path = work / "private/ws-gt-worker-key"
        write_secret_file(ws_worker_key_path, ws_worker_key)
        ws_router_cfg_path = work / "private/ws-gt-judge.json"
        write_private(
            ws_router_cfg_path,
            {
                "model": gt_model,
                "api_bases": [gt_url],
                "api_key_file": str(gt_key),
            },
        )
        ws_worker_url = f"http://127.0.0.1:{ws_worker_port}"
        reward["ws_gt_worker_urls"] = [ws_worker_url]
        reward["ws_gt_api_key_file"] = str(ws_worker_key_path)
        env.update({
            "GEMMA4_WSCLAW_JUDGE_ROUTER_CONFIG": str(ws_router_cfg_path),
            "WSCLAW_JUDGE_MODEL": gt_model,
            "WSCLAW_JUDGE_WORKER_URLS": ws_worker_url,
            "WSCLAW_JUDGE_API_KEY": ws_worker_key,
            "WSCLAW_JUDGE_PROXY_PORT": str(ws_proxy_port),
            "WSCLAW_JUDGE_WORKER_PORT": str(ws_worker_port),
        })
        if c.get("CLAUDE_BIN"):
            env["CLAUDE_BIN"] = c["CLAUDE_BIN"]
        services.update({
            "ws_judge_proxy_port": ws_proxy_port,
            "ws_judge_worker_port": ws_worker_port,
            "ws_judge_router_config": str(ws_router_cfg_path),
            "ws_judge_worker_url": ws_worker_url,
        })
    else:
        reward["gt_judge"] = {
            "base_url": gt_url,
            "model": gt_model,
            "api_key_file": str(gt_key),
        }

    write_private(work / "private/reward-service.json", reward)
    write_private(work / "private/environment.json", env)
    write_private(work / "private/services.json", services)
    return work


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--attempt")
    parsed = parser.parse_args()
    print(configure(parsed.config, parsed.attempt))
