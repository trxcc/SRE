#!/usr/bin/env bash
# Launch an OpenAI-compatible SGLang server for the frozen self-reward model (pi_theta_0)
# or the external GT user simulator (e.g., Gemma4-31B-IT) on dedicated GPU(s).
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_frozen_model.sh \
#     --model-path /models/gemma-4-E4B-it --model-family gemma4 --port 29040
#   CUDA_VISIBLE_DEVICES=0 bash scripts/serve_frozen_model.sh \
#     --model-path /models/Qwen3.5-9B --model-family qwen35 --port 29040
set -euo pipefail

MODEL_PATH=""
MODEL_FAMILY="gemma4"
SERVED_NAME=""
HOST="0.0.0.0"
PORT="29040"
TP_SIZE="1"
CONTEXT_LENGTH="99328"
MEM_FRACTION="0.80"
MAX_RUNNING_REQUESTS="32"
API_KEY_FILE=""

usage() {
  cat <<'EOF'
Usage: bash scripts/serve_frozen_model.sh --model-path PATH [options]

Options:
  --model-path PATH            Local HF directory or Hub model ID (required)
  --model-family FAMILY        gemma4 | qwen35 (default: gemma4)
  --served-model-name NAME     OpenAI model ID (default: gemma4-e4b-it / gemma-4-31B-it / qwen3.5-9b)
  --host HOST                  Bind host (default: 0.0.0.0)
  --port PORT                  Bind port (default: 29040)
  --tp N                       Tensor parallel size (default: 1; use 2 for Gemma4-31B-IT)
  --context-length TOKENS      Max context length (default: 99328)
  --mem-fraction-static FRAC   SGLang static KV memory fraction (default: 0.80)
  --max-running-requests N     Max concurrent requests (default: 32)
  --api-key-file FILE          Optional chmod-600 API key file
  -h, --help                   Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --served-model-name) SERVED_NAME="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp) TP_SIZE="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --api-key-file) API_KEY_FILE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || { echo "ERROR: --model-path is required" >&2; usage >&2; exit 2; }

EXTRA_ARGS=()
case "${MODEL_FAMILY}" in
  gemma4)
    if [[ -z "${SERVED_NAME}" ]]; then
      if [[ "${MODEL_PATH}" == *31[bB]* ]]; then
        SERVED_NAME="gemma-4-31B-it"
      else
        SERVED_NAME="gemma4-e4b-it"
      fi
    fi
    EXTRA_ARGS+=(--attention-backend triton --reasoning-parser gemma4 --tool-call-parser gemma4 --disable-cuda-graph)
    ;;
  qwen35)
    SERVED_NAME="${SERVED_NAME:-qwen3.5-9b}"
    EXTRA_ARGS+=(--reasoning-parser qwen3 --tool-call-parser qwen3_coder)
    ;;
  *)
    echo "ERROR: --model-family must be 'gemma4' or 'qwen35'" >&2
    exit 2
    ;;
esac

if [[ -n "${API_KEY_FILE}" ]]; then
  API_KEY="$(tr -d '\r\n' < "${API_KEY_FILE}")"
  EXTRA_ARGS+=(--api-key "${API_KEY}")
fi

exec python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP_SIZE}" \
  --context-length "${CONTEXT_LENGTH}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  "${EXTRA_ARGS[@]}"
