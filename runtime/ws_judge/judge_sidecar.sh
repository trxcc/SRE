#!/usr/bin/env bash
set -Eeuo pipefail
set +x

MODE="${1:-run}"
case "${MODE}" in run|validate) ;; *) echo "usage: $0 [run|validate]" >&2; exit 2 ;; esac

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${STATE_ROOT:?STATE_ROOT is required}"
NODE_RANK="${NODE_RANK:-0}"
: "${GEMMA4_WSCLAW_JUDGE_ROUTER_CONFIG:?judge router config is required}"
: "${WSCLAW_JUDGE_MODEL:?WSCLAW_JUDGE_MODEL is required}"
WSCLAW_JUDGE_WORKERS_PER_NODE="${WSCLAW_JUDGE_WORKERS_PER_NODE:-8}"
WSCLAW_JUDGE_UPSTREAM_TIMEOUT="${WSCLAW_JUDGE_UPSTREAM_TIMEOUT:-300}"
POLAR_EVALUATOR_TIMEOUT="${POLAR_EVALUATOR_TIMEOUT:-2700}"
GEMMA4_WSCLAW_RUNTIME_PYDEPS="${GEMMA4_WSCLAW_RUNTIME_PYDEPS:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROXY_SCRIPT="${SCRIPT_DIR}/judge_proxy.py"
export WSCLAW_EXPECTED_JUDGE_GUIDANCE_FILE="${SCRIPT_DIR}/judge_system_prompt.txt"
WORKER_SCRIPT="${SCRIPT_DIR}/wsclaw_judge_worker.py"
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
WSBENCH_PYTHON="${WSBENCH_PYTHON:-$(command -v python3)}"
PROXY_PORT="${WSCLAW_JUDGE_PROXY_PORT:-28198}"
WORKER_PORT="${WSCLAW_JUDGE_WORKER_PORT:-28200}"
WORKLOAD_PGID="${WSCLAW_WORKLOAD_PGID:-}"
NODE_ROOT="${STATE_ROOT}/wsclaw_judge"
READY_FILE="${NODE_ROOT}/ready.json"

for value in "${WSCLAW_JUDGE_WORKERS_PER_NODE}" "${WSCLAW_JUDGE_UPSTREAM_TIMEOUT}" \
    "${POLAR_EVALUATOR_TIMEOUT}" "${PROXY_PORT}" "${WORKER_PORT}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: judge worker, timeout, and port values must be positive integers" >&2
        exit 2
    }
done
test -s "${PROXY_SCRIPT}"
test -s "${WORKER_SCRIPT}"
[ -n "${CLAUDE_BIN}" ] && test -x "${CLAUDE_BIN}" || { echo "ERROR: claude CLI not found; install Claude Code CLI or set CLAUDE_BIN" >&2; exit 2; }
test -x "${WSBENCH_PYTHON}"
test -s "${GEMMA4_WSCLAW_JUDGE_ROUTER_CONFIG}"
if [ -n "${GEMMA4_WSCLAW_RUNTIME_PYDEPS}" ]; then
    IFS=: read -r -a judge_pydeps <<<"${GEMMA4_WSCLAW_RUNTIME_PYDEPS}"
    for dep in "${judge_pydeps[@]}"; do
        test -n "${dep}" && test -d "${dep}"
    done
    export PYTHONPATH="${GEMMA4_WSCLAW_RUNTIME_PYDEPS}:${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
"${WSBENCH_PYTHON}" - <<'PY'
import docx  # noqa: F401
import openpyxl  # noqa: F401
import pptx  # noqa: F401
import pypdf  # noqa: F401
import pyreadstat  # noqa: F401
import xlrd  # noqa: F401
PY
"${WSBENCH_PYTHON}" "${WORKER_SCRIPT}" --help >/dev/null

python3 - "${GEMMA4_WSCLAW_JUDGE_ROUTER_CONFIG}" "${WSCLAW_JUDGE_MODEL}" <<'PY'
import json
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_model = sys.argv[2]
config = json.loads(path.read_text(encoding="utf-8"))
if config.get("model") != expected_model:
    raise SystemExit(f"judge model mismatch: {config.get('model')!r} != {expected_model!r}")
bases = config.get("api_bases")
if not isinstance(bases, list) or not bases or not all(
    isinstance(base, str) and base.startswith(("http://", "https://")) for base in bases
):
    raise SystemExit("judge router config requires non-empty HTTP(S) api_bases")
key_file = Path(str(config.get("api_key_file") or ""))
if not key_file.is_file() or stat.S_IMODE(key_file.stat().st_mode) != 0o600:
    raise SystemExit("judge API key file must exist with mode 0600")
PY

if [ "${MODE}" = validate ]; then
    echo "WSCLAW_JUDGE_SIDECAR_VALIDATE_OK rank=${NODE_RANK} model=${WSCLAW_JUDGE_MODEL} workers=${WSCLAW_JUDGE_WORKERS_PER_NODE}"
    exit 0
fi

umask 077
mkdir -p "${NODE_ROOT}"
chmod 700 "${NODE_ROOT}"

export WSCLAW_OPENAI_ROUTER_CONFIG_FILE="${GEMMA4_WSCLAW_JUDGE_ROUTER_CONFIG}"
export WSCLAW_OPENAI_MODEL="${WSCLAW_JUDGE_MODEL}"
export WSCLAW_PROXY_TIMEOUT_SEC="${WSCLAW_JUDGE_UPSTREAM_TIMEOUT}"
export WSCLAW_PROXY_MAX_TOKENS=8192
export WSCLAW_PROXY_ENABLE_THINKING=1
export WSCLAW_PROXY_REASONING_EFFORT=medium
export WSCLAW_ROUTER_TENANT="${RUN_ID:-wsclaw}"
export WSCLAW_ROUTER_WORKLOAD_CLASS=judge
export WSCLAW_PROXY_METRICS_JSONL="${NODE_ROOT}/proxy_metrics.jsonl"
export NO_PROXY='*'
export no_proxy='*'

proxy_pid=""
worker_pid=""
shutting_down=0

alive() {
    local pid="${1:-}"
    [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

stop_pid() {
    local pid="${1:-}"
    alive "${pid}" || return 0
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
        alive "${pid}" || return 0
        sleep 0.25
    done
    kill -KILL "${pid}" 2>/dev/null || true
}

cleanup() {
    shutting_down=1
    stop_pid "${worker_pid}"
    stop_pid "${proxy_pid}"
    wait "${worker_pid}" "${proxy_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_owned_health() {
    local url="$1" expected_pid="$2" label="$3" deadline=$((SECONDS + 180))
    while :; do
        alive "${expected_pid}" || {
            echo "ERROR: ${label} exited before health check passed" >&2
            return 1
        }
        if python3 - "${url}" "${expected_pid}" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "ok" and payload.get("pid") == int(sys.argv[2]) else 1)
PY
        then
            return 0
        fi
        [ "${SECONDS}" -lt "${deadline}" ] || {
            echo "ERROR: timed out waiting for ${label} health" >&2
            return 1
        }
        sleep 1
    done
}

python3 "${GEMMA4_CPU_AS_WRAPPER:?CPU AS wrapper required}" --gib 48 -- python3 "${PROXY_SCRIPT}" --host 0.0.0.0 --port "${PROXY_PORT}" \
    >"${NODE_ROOT}/proxy.log" 2>&1 &
proxy_pid=$!
wait_owned_health "http://127.0.0.1:${PROXY_PORT}/health" "${proxy_pid}" judge-proxy

python3 "${GEMMA4_CPU_AS_WRAPPER:?CPU AS wrapper required}" --gib 48 -- python3 "${WORKER_SCRIPT}" \
    --host 0.0.0.0 --port "${WORKER_PORT}" \
    --workers "${WSCLAW_JUDGE_WORKERS_PER_NODE}" \
    --api-key apiKey \
    --judge-api-base "http://127.0.0.1:${PROXY_PORT}" \
    --judge-api-key apiKey \
    --judge-model "${WSCLAW_JUDGE_MODEL}" \
    --judge-timeout "${POLAR_EVALUATOR_TIMEOUT}" \
    --judge-max-turns 30 --judge-effort medium \
    --claude-bin "${CLAUDE_BIN}" --wsbench-python "${WSBENCH_PYTHON}" \
    >"${NODE_ROOT}/worker.log" 2>&1 &
worker_pid=$!
wait_owned_health "http://127.0.0.1:${WORKER_PORT}/health" "${worker_pid}" judge-worker

python3 - "${READY_FILE}" "${NODE_RANK}" "${proxy_pid}" "${worker_pid}" \
    "${WSCLAW_JUDGE_MODEL}" "${WSCLAW_JUDGE_WORKERS_PER_NODE}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "phase": "ready",
    "rank": int(sys.argv[2]),
    "proxy_pid": int(sys.argv[3]),
    "worker_pid": int(sys.argv[4]),
    "model": sys.argv[5],
    "workers": int(sys.argv[6]),
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
temporary.chmod(0o600)
temporary.replace(path)
PY

set +e
wait -n "${proxy_pid}" "${worker_pid}"
rc=$?
set -e
if [ "${shutting_down}" -eq 0 ]; then
    echo "ERROR: WSClaw judge sidecar child exited unexpectedly rc=${rc}" >&2
    if [[ "${WORKLOAD_PGID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -TERM -- "-${WORKLOAD_PGID}" 2>/dev/null || true
    fi
fi
exit "${rc}"
