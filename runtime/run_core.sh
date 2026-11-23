#!/usr/bin/env bash
set -euo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:?copied reference project root required}"
SLIME_ROOT="${SLIME_ROOT:-${PROJECT_ROOT}/external/slime-latest}"
if [ -z "${MEGATRON_ROOT:-}" ]; then
    if [ -d "${PROJECT_ROOT}/external/Megatron-LM-clawrl/megatron" ]; then
        MEGATRON_ROOT="${PROJECT_ROOT}/external/Megatron-LM-clawrl"
    else
        MEGATRON_ROOT="${PROJECT_ROOT}/external/Megatron-LM"
    fi
fi
NANOBOT_ROOT="${NANOBOT_ROOT:-${PROJECT_ROOT}/vendor/clawgym_nanobot/nanobot}"
NANOBOT_RL_RUNNER="${NANOBOT_RL_RUNNER:-${NANOBOT_ROOT}/nanobot/rl_runner.py}"
NANOBOT_IMPORT_LOCAL_PACKAGE="${NANOBOT_IMPORT_LOCAL_PACKAGE:-1}"
NANOBOT_EXTRA_PYTHONPATH="${NANOBOT_EXTRA_PYTHONPATH:-}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-}"
SGLANG_ROOT="${SGLANG_ROOT:-}"
PYTHON_DEP_OVERLAY_ROOT="${PYTHON_DEP_OVERLAY_ROOT:-}"
TRAIN_BACKEND="${TRAIN_BACKEND:-megatron}"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-}"
NEMO_AUTOMODEL_ROOT="${NEMO_AUTOMODEL_ROOT:-}"
NEMO_AUTOMODEL_PYDEPS="${NEMO_AUTOMODEL_PYDEPS:-}"
NEMO_AUTOMODEL_SUPPORT_PYDEPS="${NEMO_AUTOMODEL_SUPPORT_PYDEPS:-}"
NEMO_AUTOMODEL_CUTLASS_PACKAGES="${NEMO_AUTOMODEL_CUTLASS_PACKAGES:-}"
NEMO_AUTOMODEL_TRANSFORMERS_ROOT="${NEMO_AUTOMODEL_TRANSFORMERS_ROOT:-}"
AUTOMODEL_CONFIG="${AUTOMODEL_CONFIG:-}"
NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-}"
case "${TRAIN_BACKEND}" in
    megatron|automodel|nemo_megatron) ;;
    *) echo "ERROR: TRAIN_BACKEND must be megatron, automodel, or nemo_megatron" >&2; exit 2 ;;
esac
if [ "${TRAIN_BACKEND}" != "megatron" ]; then
    # NeMo-RL owns both non-legacy training backends. Slime still reuses
    # Megatron's CLI parser for the shared command line, but it must not import
    # model patches from the unrelated legacy Megatron backend. In particular,
    # those patches import the image's FA4 package, whose CUTLASS Python API is
    # intentionally different from AutoModel FFPA's pinned CUTLASS DSL.
    export SLIME_DISABLE_MEGATRON_PATCHES=1
fi
if [ "${TRAIN_BACKEND}" = "automodel" ]; then
    export SLIME_BLOCK_UNUSED_FA4_IMPORT=1
fi
NANOBOT_CONDA_ROOT="${NANOBOT_CONDA_ROOT:-}"
NANOBOT_CONDA_ENV_NAME="${NANOBOT_CONDA_ENV_NAME:-nanobot-openclaw}"
SIF="${SIF:-${SLIME_SIF:-}}"

NODE_RANK="${NODE_RANK:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ "${CLAWGYM_IN_CONTAINER:-0}" != "1" ]; then
    if [ -z "${SIF}" ]; then
        echo "ERROR: set SIF or SLIME_SIF to the Apptainer image path" >&2
        exit 1
    fi
    if [ ! -f "${SIF}" ]; then
        echo "ERROR: apptainer image not found: ${SIF}" >&2
        exit 1
    fi
    exec apptainer exec --nv \
        --bind /mnt:/mnt \
        --pwd "${PROJECT_ROOT}" \
        "${SIF}" \
        env CLAWGYM_IN_CONTAINER=1 \
            NODE_RANK="${NODE_RANK}" \
            MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
            MASTER_PORT="${MASTER_PORT:-6379}" \
            LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/clawgym_nanobot/manual}" \
            STOP_FILE="${STOP_FILE:-}" \
            HF_CHECKPOINT="${HF_CHECKPOINT:-}" \
            REF_LOAD="${REF_LOAD:-}" \
            LOAD_CKPT="${LOAD_CKPT:-}" \
            LOAD_DIR="${LOAD_DIR:-}" \
            SAVE_DIR="${SAVE_DIR:-}" \
            RAY_TMPDIR="${RAY_TMPDIR:-}" \
            RAY_TMP_BASE="${RAY_TMP_BASE:-}" \
            SLIME_ROOT="${SLIME_ROOT}" \
            MEGATRON_ROOT="${MEGATRON_ROOT}" \
            MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT}" \
            SGLANG_ROOT="${SGLANG_ROOT}" \
            PYTHON_DEP_OVERLAY_ROOT="${PYTHON_DEP_OVERLAY_ROOT}" \
            TRAIN_BACKEND="${TRAIN_BACKEND}" \
            NEMO_RL_ROOT="${NEMO_RL_ROOT}" \
            NEMO_AUTOMODEL_ROOT="${NEMO_AUTOMODEL_ROOT}" \
            NEMO_AUTOMODEL_PYDEPS="${NEMO_AUTOMODEL_PYDEPS}" \
            NEMO_AUTOMODEL_SUPPORT_PYDEPS="${NEMO_AUTOMODEL_SUPPORT_PYDEPS}" \
            NEMO_AUTOMODEL_CUTLASS_PACKAGES="${NEMO_AUTOMODEL_CUTLASS_PACKAGES}" \
            NEMO_AUTOMODEL_TRANSFORMERS_ROOT="${NEMO_AUTOMODEL_TRANSFORMERS_ROOT}" \
            AUTOMODEL_CONFIG="${AUTOMODEL_CONFIG}" \
            NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR}" \
            NEMO_MEGATRON_PROCESS_GROUP_TIMEOUT_MINUTES="${NEMO_MEGATRON_PROCESS_GROUP_TIMEOUT_MINUTES:-}" \
            SLIME_BALANCE_DATA="${SLIME_BALANCE_DATA:-0}" \
            NANOBOT_ROOT="${NANOBOT_ROOT}" \
            NANOBOT_RL_RUNNER="${NANOBOT_RL_RUNNER}" \
            NANOBOT_IMPORT_LOCAL_PACKAGE="${NANOBOT_IMPORT_LOCAL_PACKAGE}" \
            NANOBOT_EXTRA_PYTHONPATH="${NANOBOT_EXTRA_PYTHONPATH}" \
            NANOBOT_RL_TOKENIZER_PATH="${NANOBOT_RL_TOKENIZER_PATH:-${HF_CHECKPOINT:-}}" \
            NANOBOT_CONDA_ROOT="${NANOBOT_CONDA_ROOT}" \
            NANOBOT_CONDA_ENV_NAME="${NANOBOT_CONDA_ENV_NAME}" \
            MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-}" \
            PP_SIZE="${PP_SIZE:-}" \
            QWEN_GDN_BACKEND="${QWEN_GDN_BACKEND:-}" \
            EVAL_MICRO_BATCH_SIZE="${EVAL_MICRO_BATCH_SIZE:-}" \
            EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-}" \
            TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH:-}" \
            LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-}" \
            ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-}" \
            CHECK_WEIGHT_UPDATE_EQUAL="${CHECK_WEIGHT_UPDATE_EQUAL:-0}" \
            CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES="${CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES:-}" \
            SGLANG_CONFIG_PATH="${SGLANG_CONFIG_PATH:-}" \
            SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}" \
            GEMMA4_SGLANG_ALIGN_E_HEAD_DIMS="${GEMMA4_SGLANG_ALIGN_E_HEAD_DIMS:-0}" \
            GEMMA4_SGLANG_WEIGHT_CHECK_STABLE_CONSTANTS="${GEMMA4_SGLANG_WEIGHT_CHECK_STABLE_CONSTANTS:-0}" \
            bash "$0" --node-rank "${NODE_RANK}"
fi

cd "${PROJECT_ROOT}"
# ClusterX may inject present-but-empty torchrun variables. Megatron calls
# int(os.getenv(...)) while parsing the Ray driver, so empty strings are not a
# valid neutral value. NODE_RANK above is the durable topology source.
unset WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK
mkdir -p "${LOG_DIR:-${PROJECT_ROOT}/logs/clawgym_nanobot/manual}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/clawgym_nanobot/manual}"
STOP_FILE="${STOP_FILE:-${LOG_DIR}/stop-workers}"
if [ -z "${RAY_TMPDIR:-}" ]; then
    RAY_TMP_BASE="${RAY_TMP_BASE:-/tmp}"
    RAY_TMPDIR="${RAY_TMP_BASE}/proRL-nb-ray-${USER:-u}-${MASTER_PORT:-6379}-${NODE_RANK}"
    rm -rf "${RAY_TMPDIR}"
fi
CLAWGYM_PYDEPS_DIR="${CLAWGYM_PYDEPS_DIR:-${PROJECT_ROOT}/.clawgym_pydeps/slime_latest_py311}"
mkdir -p "${RAY_TMPDIR}"

if [ ! -f "${SLIME_ROOT}/train.py" ]; then
    echo "ERROR: latest Slime not found at ${SLIME_ROOT}" >&2
    exit 1
fi
if [ ! -d "${MEGATRON_ROOT}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_ROOT}" >&2
    exit 1
fi
if [ ! -f "${NANOBOT_ROOT}/nanobot/rl_runner.py" ]; then
    echo "ERROR: nanobot rl_runner.py not found under ${NANOBOT_ROOT}" >&2
    exit 1
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-${HF_CHECKPOINT}}"
LOAD_CKPT="${LOAD_CKPT:-${LOAD_DIR:-}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/train-logs/clawgym_nanobot/no_ckpt}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
if [ -z "${SAVE_HF_FORMAT:-}" ]; then
    # Keep the Python format placeholder outside a nested shell parameter
    # expansion.  Otherwise the placeholder's closing brace terminates the
    # shell expansion early and leaves a literal extra `}` in --save-hf.
    SAVE_HF_FORMAT="${SAVE_DIR}/candidates/step-{rollout_id}"
fi
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"

if [ ! -e "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF_CHECKPOINT does not exist: ${HF_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -e "${REF_LOAD}" ]; then
    echo "ERROR: REF_LOAD does not exist: ${REF_LOAD}" >&2
    exit 1
fi
if [ -n "${LOAD_CKPT}" ] && [ ! -e "${LOAD_CKPT}" ]; then
    echo "ERROR: LOAD_CKPT does not exist: ${LOAD_CKPT}" >&2
    exit 1
fi
if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
    echo "INFO: REF_LOAD has no latest_checkpointed_iteration.txt; treating it as an HF checkpoint: ${REF_LOAD}"
fi
if [ -n "${LOAD_CKPT}" ] && [ ! -f "${LOAD_CKPT}/latest_checkpointed_iteration.txt" ]; then
    echo "INFO: LOAD_CKPT has no latest_checkpointed_iteration.txt; treating it as an HF checkpoint: ${LOAD_CKPT}"
fi

MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-5000000}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-${SLIME_ROOT}/scripts/models/qwen3-4B.sh}"
if [ ! -s "${MODEL_CONFIG_SCRIPT}" ]; then
    echo "ERROR: model config script does not exist or is empty: ${MODEL_CONFIG_SCRIPT}" >&2
    exit 1
fi
source "${MODEL_CONFIG_SCRIPT}"

numpy_has_dtypes() {
    PYTHONPATH="${1:-}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY' >/dev/null 2>&1
import numpy.dtypes  # noqa: F401
PY
}

ensure_python_overlay() {
    if numpy_has_dtypes ""; then
        return
    fi
    mkdir -p "${CLAWGYM_PYDEPS_DIR}"
    if ! numpy_has_dtypes "${CLAWGYM_PYDEPS_DIR}"; then
        if [ "${NODE_RANK}" = "0" ]; then
            rm -rf "${CLAWGYM_PYDEPS_DIR}/numpy" "${CLAWGYM_PYDEPS_DIR}"/numpy-*.dist-info
            python3 -m pip install --no-cache-dir --upgrade --target "${CLAWGYM_PYDEPS_DIR}" 'numpy>=1.26,<2'
            touch "${CLAWGYM_PYDEPS_DIR}/.numpy-ready"
        else
            deadline=$((SECONDS + 900))
            while [ "${SECONDS}" -lt "${deadline}" ]; do
                if numpy_has_dtypes "${CLAWGYM_PYDEPS_DIR}"; then
                    break
                fi
                sleep 5
            done
        fi
    fi
    if ! numpy_has_dtypes "${CLAWGYM_PYDEPS_DIR}"; then
        echo "ERROR: numpy>=1.26 overlay is unavailable at ${CLAWGYM_PYDEPS_DIR}" >&2
        exit 1
    fi
    export PYTHONPATH="${CLAWGYM_PYDEPS_DIR}:${PYTHONPATH:-}"
}

if [ "${CLAWGYM_ENABLE_PYDEPS_OVERLAY:-0}" = "1" ]; then
    ensure_python_overlay
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-$((MASTER_PORT + 100))}"
RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((MASTER_PORT + 101))}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-$((MASTER_PORT + 102))}"
RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-$((MASTER_PORT + 103))}"
RAY_JOB_SUBMIT_ADDR="${RAY_JOB_SUBMIT_ADDR:-127.0.0.1}"
RAY_GPUS_PER_NODE="${RAY_GPUS_PER_NODE:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${RAY_GPUS_PER_NODE}}"
NUM_NODES="${NUM_NODES:-4}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-24}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
# Total training rollouts. Formal runs should use a long horizon; monitor a few
# healthy steps externally instead of making the run terminate after that count.
NUM_ROLLOUT="${NUM_ROLLOUT:-1000000}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
if ! [[ "${NUM_STEPS_PER_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_STEPS_PER_ROLLOUT must be a positive integer" >&2
    exit 1
fi
GRPO_ITERATIONS="${GRPO_ITERATIONS:-}"
if [ -n "${GRPO_ITERATIONS}" ] && ! [[ "${GRPO_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GRPO_ITERATIONS must be a positive integer when set" >&2
    exit 1
fi
UPDATE_WEIGHT_MODE="${UPDATE_WEIGHT_MODE:-full}"
UPDATE_WEIGHT_TRANSPORT="${UPDATE_WEIGHT_TRANSPORT:-nccl}"
UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-${LOG_DIR}/weight_sync}"
TRAIN_MEMORY_MARGIN_BYTES="${TRAIN_MEMORY_MARGIN_BYTES:-1073741824}"
if ! [[ "${TRAIN_MEMORY_MARGIN_BYTES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TRAIN_MEMORY_MARGIN_BYTES must be a non-negative integer" >&2
    exit 1
fi
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
DEBUG_TRAIN_ONLY="${DEBUG_TRAIN_ONLY:-0}"
TRAINING_FREE_EVAL_ONLY="${TRAINING_FREE_EVAL_ONLY:-0}"
TRAINING_FREE_ROLLOUT_ONLY="${TRAINING_FREE_ROLLOUT_ONLY:-0}"
[ $((TRAINING_FREE_EVAL_ONLY + TRAINING_FREE_ROLLOUT_ONLY)) -le 1 ] || {
    echo "ERROR: TRAINING_FREE_EVAL_ONLY and TRAINING_FREE_ROLLOUT_ONLY are mutually exclusive" >&2
    exit 1
}
COLOCATE="${COLOCATE:-0}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
PIPELINE_MODEL_PARALLEL_LAYOUT="${PIPELINE_MODEL_PARALLEL_LAYOUT:-}"
PIPELINE_LAYOUT_ARGS=()
if [ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT}" ]; then
    PIPELINE_LAYOUT_ARGS+=(--pipeline-model-parallel-layout "${PIPELINE_MODEL_PARALLEL_LAYOUT}")
fi
ENABLE_SEQUENCE_PARALLEL="${ENABLE_SEQUENCE_PARALLEL:-1}"
SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
if [ "${ENABLE_SEQUENCE_PARALLEL}" = "0" ]; then
    SEQUENCE_PARALLEL_ARGS=()
elif [ "${ENABLE_SEQUENCE_PARALLEL}" != "1" ]; then
    echo "ERROR: ENABLE_SEQUENCE_PARALLEL must be 0 or 1" >&2
    exit 1
fi
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-}"
ROLLOUT_STOP_TOKEN_IDS="${ROLLOUT_STOP_TOKEN_IDS:-}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
if [ -n "${RECOMPUTE_GRANULARITY}" ] &&
    ! [[ "${RECOMPUTE_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: RECOMPUTE_NUM_LAYERS must be a positive integer when activation recompute is enabled" >&2
    exit 1
fi
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-256}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-65536}"
TRAIN_SEQ_LENGTH="${TRAIN_SEQ_LENGTH:-${SGLANG_CONTEXT_LENGTH}}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
SGLANG_CONTEXT_MARGIN="${SGLANG_CONTEXT_MARGIN:-1024}"
SGLANG_ENGINE_CONTEXT_LENGTH="${SGLANG_ENGINE_CONTEXT_LENGTH:-$((SGLANG_CONTEXT_LENGTH + ROLLOUT_MAX_RESPONSE_LEN + SGLANG_CONTEXT_MARGIN))}"
total_rollout_samples=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [ $((total_rollout_samples % NUM_STEPS_PER_ROLLOUT)) -ne 0 ]; then
    echo "ERROR: rollout samples must be divisible by NUM_STEPS_PER_ROLLOUT" >&2
    exit 1
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((total_rollout_samples / NUM_STEPS_PER_ROLLOUT))}"
DO_EVAL="${DO_EVAL:-1}"
OPENCLAW_EVAL_INTERVAL="${OPENCLAW_EVAL_INTERVAL:-5}"
SKIP_INITIAL_EVAL="${SKIP_INITIAL_EVAL:-0}"
OPENCLAW_EVAL_N_SAMPLES="${OPENCLAW_EVAL_N_SAMPLES:-1}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-${OPENCLAW_EVAL_N_SAMPLES}}"
EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-${SGLANG_CONTEXT_LENGTH}}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}"
CLAWGYM_EVAL_NUM_PROMPTS="${CLAWGYM_EVAL_NUM_PROMPTS:-0}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-512}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-}"
if [ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ] && \
   ! [[ "${SGLANG_MAX_RUNNING_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SGLANG_MAX_RUNNING_REQUESTS must be empty or a positive integer" >&2
    exit 1
fi
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.8}"
ENABLE_OPTIMIZER_CPU_OFFLOAD="${ENABLE_OPTIMIZER_CPU_OFFLOAD:-1}"
ENABLE_PRECISION_AWARE_OPTIMIZER="${ENABLE_PRECISION_AWARE_OPTIMIZER:-1}"
ENABLE_CALCULATE_PER_TOKEN_LOSS="${ENABLE_CALCULATE_PER_TOKEN_LOSS:-1}"
ENABLE_KL_LOSS="${ENABLE_KL_LOSS:-1}"
KL_COEF="${KL_COEF:-0.0}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
VALUE_CLIP="${VALUE_CLIP:-0.2}"
GAMMA="${GAMMA:-1.0}"
LAMBDA="${LAMBDA:-1.0}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-0}"
CRITIC_EXPORT_ONLY="${CRITIC_EXPORT_ONLY:-0}"
CRITIC_EXPORT_ROLLOUT_ID="${CRITIC_EXPORT_ROLLOUT_ID:-}"
NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-0}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-0}"
MEGATRON_CONFIG_PATH="${MEGATRON_CONFIG_PATH:-}"
USE_OPD="${USE_OPD:-0}"
OPD_TYPE="${OPD_TYPE:-sglang}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
USE_OPSD="${USE_OPSD:-0}"
OPSD_TEACHER_MODE="${OPSD_TEACHER_MODE:-fixed_initial_policy}"
OPSD_BETA="${OPSD_BETA:-0.0}"
OPSD_TEMPERATURE="${OPSD_TEMPERATURE:-1.0}"
OPSD_TOKEN_CLIP="${OPSD_TOKEN_CLIP:-0.05}"
OPSD_LOGITS_CHUNK_SIZE="${OPSD_LOGITS_CHUNK_SIZE:-64}"
CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-clawgym_nanobot.generate.generate}"
CUSTOM_MODEL_PROVIDER_PATH="${CUSTOM_MODEL_PROVIDER_PATH:-}"
CUSTOM_ROLLOUT_LOG_FUNCTION_PATH="${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH:-clawgym_nanobot.rollout_logging.log_clawgym_rollout}"
CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH="${CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH:-clawgym_nanobot.rollout_logging.log_clawgym_eval_rollout}"
CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-clawgym_nanobot.reward_post_process.grpo_normalize_by_group_index}"
USE_POLAR="${USE_POLAR:-0}"
OFFLINE_CRITIC_PRETRAIN="${OFFLINE_CRITIC_PRETRAIN:-0}"
OFFLINE_CRITIC_CONFIG_PATH="${OFFLINE_CRITIC_CONFIG_PATH:-}"
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
CHECK_WEIGHT_UPDATE_EQUAL="${CHECK_WEIGHT_UPDATE_EQUAL:-0}"
CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES="${CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES:-}"
USE_TIS="${USE_TIS:-0}"
TIS_CLIP="${TIS_CLIP:-2.0}"
TIS_CLIP_LOW="${TIS_CLIP_LOW:-0.0}"
CUSTOM_TIS_FUNCTION_PATH="${CUSTOM_TIS_FUNCTION_PATH:-}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-0}"
for flag_name in USE_ROLLOUT_LOGPROBS CHECK_WEIGHT_UPDATE_EQUAL USE_TIS PARTIAL_ROLLOUT USE_OPD USE_OPSD OFFLINE_CRITIC_PRETRAIN; do
    case "${!flag_name}" in
        0|1) ;;
        *) echo "ERROR: ${flag_name} must be 0 or 1" >&2; exit 1 ;;
    esac
done
if [ "${USE_ROLLOUT_LOGPROBS}" = "1" ] && [ "${USE_TIS}" = "1" ]; then
    echo "ERROR: USE_ROLLOUT_LOGPROBS and USE_TIS are mutually exclusive" >&2
    exit 1
fi
if [ "${USE_OPSD}" = "1" ]; then
    if [ "${USE_OPD}" = "1" ]; then
        echo "ERROR: full-vocabulary USE_OPSD and sampled-token USE_OPD are mutually exclusive" >&2
        exit 1
    fi
    if [ "${USE_ROLLOUT_LOGPROBS}" = "1" ] || [ "${USE_TIS}" = "1" ]; then
        echo "ERROR: full-vocabulary OPSD is a direct on-policy divergence loss; rollout-logprob/TIS correction must be disabled" >&2
        exit 1
    fi
fi
SLIME_ASYNC_TRAINING="${SLIME_ASYNC_TRAINING:-${USE_POLAR}}"
case "${SLIME_ASYNC_TRAINING}" in
    0|1) ;;
    *) echo "ERROR: SLIME_ASYNC_TRAINING must be 0 or 1" >&2; exit 1 ;;
esac
POLAR_SYNCHRONOUS_POLICY_VERSIONS="${POLAR_SYNCHRONOUS_POLICY_VERSIONS:-$((1 - SLIME_ASYNC_TRAINING))}"
case "${POLAR_SYNCHRONOUS_POLICY_VERSIONS}" in
    0|1) ;;
    *) echo "ERROR: POLAR_SYNCHRONOUS_POLICY_VERSIONS must be 0 or 1" >&2; exit 1 ;;
esac
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-18080}"
POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-18100}"
POLAR_MAX_WORKERS_PER_NODE="${POLAR_MAX_WORKERS_PER_NODE:-64}"
POLAR_MAX_POSTRUN_WORKERS_PER_NODE="${POLAR_MAX_POSTRUN_WORKERS_PER_NODE:-${POLAR_MAX_WORKERS_PER_NODE}}"
if ! [[ "${POLAR_MAX_POSTRUN_WORKERS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: POLAR_MAX_POSTRUN_WORKERS_PER_NODE must be a positive integer" >&2
    exit 1
fi
POLAR_EXPECTED_GATEWAY_NODES="${POLAR_EXPECTED_GATEWAY_NODES:-1}"
POLAR_GATEWAY_ON_DRIVER="${POLAR_GATEWAY_ON_DRIVER:-0}"
POLAR_MAX_ASYNC_LEVEL="${POLAR_MAX_ASYNC_LEVEL:-4}"
POLAR_MAX_OFF_POLICY_STEPS="${POLAR_MAX_OFF_POLICY_STEPS:-1}"
POLAR_MIN_COMPLETE_ACCEPT_FRACTION="${POLAR_MIN_COMPLETE_ACCEPT_FRACTION:-0.875}"
POLAR_MAX_LOW_COMPLETE_RETRIES="${POLAR_MAX_LOW_COMPLETE_RETRIES:-1}"
POLAR_TIMEOUT_AS_ZERO="${POLAR_TIMEOUT_AS_ZERO:-1}"
case "${POLAR_TIMEOUT_AS_ZERO}" in
    0|1) ;;
    *) echo "ERROR: POLAR_TIMEOUT_AS_ZERO must be 0 or 1" >&2; exit 1 ;;
esac
python3 - "${POLAR_MIN_COMPLETE_ACCEPT_FRACTION}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit("POLAR_MIN_COMPLETE_ACCEPT_FRACTION must be between 0 and 1")
PY
if ! [[ "${POLAR_MAX_LOW_COMPLETE_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: POLAR_MAX_LOW_COMPLETE_RETRIES must be a non-negative integer" >&2
    exit 1
fi
POLAR_PREFIX_REALIGN_REPLAYED_RESPONSE="${POLAR_PREFIX_REALIGN_REPLAYED_RESPONSE:-0}"
case "${POLAR_PREFIX_REALIGN_REPLAYED_RESPONSE}" in
    0|1) ;;
    *) echo "ERROR: POLAR_PREFIX_REALIGN_REPLAYED_RESPONSE must be 0 or 1" >&2; exit 1 ;;
esac
POLAR_PREFIX_REALIGN_MAX_NEXT_RESPONSE_TOKENS="${POLAR_PREFIX_REALIGN_MAX_NEXT_RESPONSE_TOKENS:-1024}"
if ! [[ "${POLAR_PREFIX_REALIGN_MAX_NEXT_RESPONSE_TOKENS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: POLAR_PREFIX_REALIGN_MAX_NEXT_RESPONSE_TOKENS must be a non-negative integer" >&2
    exit 1
fi
POLAR_EVAL_MAX_CONCURRENCY="${POLAR_EVAL_MAX_CONCURRENCY:-${POLAR_MAX_WORKERS_PER_NODE}}"
POLAR_EVAL_INFRA_RETRIES="${POLAR_EVAL_INFRA_RETRIES:-2}"
POLAR_REQUEST_TIMEOUT="${POLAR_REQUEST_TIMEOUT:-1800}"
POLAR_TASK_TIMEOUT="${POLAR_TASK_TIMEOUT:-1200}"
POLAR_MODEL_SERVED="${POLAR_MODEL_SERVED:-qwen3-4b-thinking}"
POLAR_AGENT_HARNESS="${POLAR_AGENT_HARNESS:-nanobot}"
POLAR_GATEWAY_FIRST_NODE_RANK="${POLAR_GATEWAY_FIRST_NODE_RANK:-1}"
if ! [[ "${POLAR_GATEWAY_FIRST_NODE_RANK}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: POLAR_GATEWAY_FIRST_NODE_RANK must be a non-negative integer" >&2
    exit 1
fi
case "${POLAR_AGENT_HARNESS}" in
    nanobot) ;;
    *) echo "ERROR: POLAR_AGENT_HARNESS must be nanobot" >&2; exit 1 ;;
esac
POLAR_DATA_SOURCE_PATH="${POLAR_DATA_SOURCE_PATH:-clawgym_nanobot.polar_data.SequentialThenShuffleRolloutDataSource}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-19000}"
POLAR_ROUTER_WAIT_TIMEOUT_SECONDS="${POLAR_ROUTER_WAIT_TIMEOUT_SECONDS:-300}"
if ! [[ "${POLAR_ROUTER_WAIT_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: POLAR_ROUTER_WAIT_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 1
fi
SGLANG_ROUTER_POLICY="${SGLANG_ROUTER_POLICY:-cache_aware}"
SGLANG_ROUTER_ASSIGNMENT_MODE="${SGLANG_ROUTER_ASSIGNMENT_MODE:-random}"
POLAR_GATEWAY_LOCAL_SGLANG="${POLAR_GATEWAY_LOCAL_SGLANG:-0}"
POLAR_SGLANG_SERVER_PORT="${POLAR_SGLANG_SERVER_PORT:-15000}"
case "${POLAR_GATEWAY_LOCAL_SGLANG}" in
    0|1) ;;
    *) echo "ERROR: POLAR_GATEWAY_LOCAL_SGLANG must be 0 or 1" >&2; exit 1 ;;
esac
if ! [[ "${POLAR_SGLANG_SERVER_PORT}" =~ ^[0-9]+$ ]] || [ "${POLAR_SGLANG_SERVER_PORT}" -lt 1 ] || [ "${POLAR_SGLANG_SERVER_PORT}" -gt 65535 ]; then
    echo "ERROR: POLAR_SGLANG_SERVER_PORT must be an integer in [1, 65535]" >&2
    exit 1
fi
case "${SGLANG_ROUTER_POLICY}" in
    random|round_robin|cache_aware|power_of_two|manual|consistent_hashing|prefix_hash) ;;
    *) echo "ERROR: unsupported SGLANG_ROUTER_POLICY=${SGLANG_ROUTER_POLICY}" >&2; exit 1 ;;
esac
case "${SGLANG_ROUTER_ASSIGNMENT_MODE}" in
    random|min_load|min_group) ;;
    *) echo "ERROR: unsupported SGLANG_ROUTER_ASSIGNMENT_MODE=${SGLANG_ROUTER_ASSIGNMENT_MODE}" >&2; exit 1 ;;
esac

if [ "${SMOKE_TEST:-0}" = "1" ]; then
    ROLLOUT_BATCH_SIZE="${SMOKE_ROLLOUT_BATCH_SIZE:-1}"
    N_SAMPLES_PER_PROMPT="${SMOKE_N_SAMPLES_PER_PROMPT:-1}"
    NUM_ROLLOUT="${SMOKE_NUM_ROLLOUT:-1}"
    GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-1}"
    CLAWGYM_EVAL_NUM_PROMPTS="${SMOKE_EVAL_NUM_PROMPTS:-1}"
    OPENCLAW_EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL:-1}"
    export OPENCLAW_TRAIN_MAX_TURNS="${OPENCLAW_TRAIN_MAX_TURNS:-1}"
    export WANDB_MODE="${WANDB_MODE:-disabled}"
fi

export PATH="${SLIME_CONTAINER_PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="${SLIME_CONTAINER_LD_LIBRARY_PATH:-/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nvtx/lib:/usr/local/lib/python3.12/dist-packages/tvm_ffi/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu}"
export TRAIN_NODE_HOST="${SLURMD_NODENAME:-$(hostname)}"
export CC="/usr/bin/gcc"
export CXX="/usr/bin/g++"
export CUDAHOSTCXX="/usr/bin/g++"
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL VIRTUAL_ENV
unset LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH
export LIBRARY_PATH=""
export C_INCLUDE_PATH=""
export CPLUS_INCLUDE_PATH=""
export CPATH="${GEMMA4_PYTHON_INCLUDE_DIR:-}"
export CLAWGYM_PRIVATE_ENV="${CLAWGYM_PRIVATE_ENV:-}"

load_private_wandb_env_shell() {
    local env_path="${CLAWGYM_PRIVATE_ENV:-}"
    local raw_line line key value
    if [ -z "${env_path}" ] || [ ! -f "${env_path}" ]; then
        return
    fi
    while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
        line="${raw_line#"${raw_line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [ -z "${line}" ] || [[ "${line}" == \#* ]] || [[ "${line}" != *=* ]]; then
            continue
        fi
        key="${line%%=*}"
        value="${line#*=}"
        key="${key%"${key##*[![:space:]]}"}"
        key="${key#"${key%%[![:space:]]*}"}"
        if [[ ! "${key}" =~ ^WANDB_[A-Za-z0-9_]*$ ]]; then
            continue
        fi
        if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        if [ -z "${!key:-}" ]; then
            export "${key}=${value}"
        fi
    done < "${env_path}"
}

load_private_wandb_env_shell
export WANDB_PROXY="${WANDB_PROXY:-${WANDB_PROXY_URL:-${WANDB_HTTPS_PROXY:-${WANDB_HTTP_PROXY:-${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY:-}}}}}}}}"
export WANDB_PROXY_URL="${WANDB_PROXY_URL:-${WANDB_PROXY:-}}"
export WANDB_HTTP_PROXY="${WANDB_HTTP_PROXY:-${WANDB_PROXY_URL:-}}"
export WANDB_HTTPS_PROXY="${WANDB_HTTPS_PROXY:-${WANDB_PROXY_URL:-${WANDB_HTTP_PROXY:-}}}"
if [ -n "${WANDB_PROXY_URL:-}" ]; then
    export WANDB_USE_PROXY="${WANDB_USE_PROXY:-1}"
fi
if [ "${USE_WANDB:-0}" = "1" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-disabled}"
fi
export WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_TEAM:-}}"
export WANDB_PROJECT="${WANDB_PROJECT:-clawgym-nanobot}"
export WANDB_GROUP="${WANDB_GROUP:-$(basename "${LOG_DIR}")}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_ID:-$(basename "${LOG_DIR}")}}"
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"
export WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_JSONL_RELAY="${WANDB_JSONL_RELAY:-0}"
case "${WANDB_JSONL_RELAY}" in
    0|1) ;;
    *) echo "ERROR: WANDB_JSONL_RELAY must be 0 or 1" >&2; exit 1 ;;
esac
WANDB_JSONL_RELAY_ACTIVE=0
if [ "${WANDB_JSONL_RELAY}" = "1" ] && \
   [ "${USE_WANDB:-0}" = "1" ] && \
   [ "${WANDB_MODE}" != "disabled" ]; then
    WANDB_JSONL_RELAY_ACTIVE=1
fi
if [ "${WANDB_JSONL_RELAY_ACTIVE}" = "1" ]; then
    export SLIME_METRICS_JSONL_PATH="${SLIME_METRICS_JSONL_PATH:-${LOG_DIR}/metrics/events.jsonl}"
    export WANDB_STEPS_JSONL_PATH="${WANDB_STEPS_JSONL_PATH:-${LOG_DIR}/metrics/steps.jsonl}"
    export WANDB_JSONL_COALESCER_DONE="${WANDB_JSONL_COALESCER_DONE:-${LOG_DIR}/metrics/coalescer.done}"
    export WANDB_JSONL_COALESCER_LOG="${WANDB_JSONL_COALESCER_LOG:-${LOG_DIR}/metrics/coalescer.log}"
    export WANDB_JSONL_RELAY_STATE="${WANDB_JSONL_RELAY_STATE:-${LOG_DIR}/metrics/wandb-relay-state.json}"
    export WANDB_JSONL_RELAY_LOG="${WANDB_JSONL_RELAY_LOG:-${LOG_DIR}/metrics/wandb-relay.log}"
    export WANDB_JSONL_POLL_SECONDS="${WANDB_JSONL_POLL_SECONDS:-2}"
fi
if [ "${WANDB_JSONL_RELAY_ACTIVE}" = "1" ] && [ -z "${WANDB_RUN_ID:-}" ]; then
    WANDB_RUN_ID="$(
        python3 - "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${WANDB_GROUP}" "${WANDB_RUN_NAME}" <<'PY'
import hashlib
import sys

print(hashlib.sha256("\0".join(sys.argv[1:]).encode()).hexdigest()[:12])
PY
    )"
    export WANDB_RUN_ID
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${SLURM_JOB_ID:-$$}}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-/tmp/tvm_ffi_cache_${SLURM_JOB_ID:-$$}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions_${SLURM_JOB_ID:-$$}}"
mkdir -p "${TRITON_CACHE_DIR}" "${TVM_FFI_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
# Independent ClusterX nodes can take longer than Ray's 30-second default to
# finish GCS registration while four runtimes start concurrently.  This is
# only the raylet startup acknowledgement window; it does not alter task or
# rollout timeouts.
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-60}"
export PRORL_MASK_BROKEN_SKLEARN="${PRORL_MASK_BROKEN_SKLEARN:-1}"
AUTOMODEL_PYTHONPATH=""
if [ "${TRAIN_BACKEND}" = automodel ] || [ "${TRAIN_BACKEND}" = nemo_megatron ]; then
    : "${NEMO_RL_ROOT:?NEMO_RL_ROOT is required for NeMo-RL backends}"
    : "${NEMO_AUTOMODEL_PYDEPS:?NEMO_AUTOMODEL_PYDEPS is required for NeMo-RL backends}"
    : "${NEMO_AUTOMODEL_TRANSFORMERS_ROOT:?NEMO_AUTOMODEL_TRANSFORMERS_ROOT is required for NeMo-RL backends}"
    : "${AUTOMODEL_CONFIG:?AUTOMODEL_CONFIG is required for NeMo-RL backends}"
    test -d "${NEMO_RL_ROOT}/nemo_rl"
    test -d "${NEMO_AUTOMODEL_PYDEPS}"
    test -d "${NEMO_AUTOMODEL_TRANSFORMERS_ROOT}/transformers"
    test -s "${AUTOMODEL_CONFIG}"
    if [ "${TRAIN_BACKEND}" = automodel ]; then
        : "${NEMO_AUTOMODEL_ROOT:?NEMO_AUTOMODEL_ROOT is required for AutoModel}"
        : "${NEMO_AUTOMODEL_SUPPORT_PYDEPS:?NEMO_AUTOMODEL_SUPPORT_PYDEPS is required for AutoModel}"
        : "${NEMO_AUTOMODEL_CUTLASS_PACKAGES:?NEMO_AUTOMODEL_CUTLASS_PACKAGES is required for AutoModel}"
        test -d "${NEMO_AUTOMODEL_ROOT}/nemo_automodel"
        test -d "${NEMO_AUTOMODEL_SUPPORT_PYDEPS}"
        test -d "${NEMO_AUTOMODEL_CUTLASS_PACKAGES}/cutlass"
        AUTOMODEL_PYTHONPATH="${NEMO_AUTOMODEL_TRANSFORMERS_ROOT}:${NEMO_AUTOMODEL_CUTLASS_PACKAGES}:${NEMO_AUTOMODEL_PYDEPS}:${NEMO_AUTOMODEL_SUPPORT_PYDEPS}:${NEMO_RL_ROOT}:${NEMO_AUTOMODEL_ROOT}:"
    else
        : "${NEMO_AUTOMODEL_SUPPORT_PYDEPS:?NEMO_AUTOMODEL_SUPPORT_PYDEPS is required for NeMo Megatron}"
        test -d "${NEMO_AUTOMODEL_SUPPORT_PYDEPS}/hydra"
        test ! -e "${NEMO_AUTOMODEL_SUPPORT_PYDEPS}/transformers"
        # Preserve the image's pinned Megatron/Transformers stack while adding
        # only NeMo-RL's pure-Python support dependencies to the Ray job and
        # every nested TrainRayActor.
        AUTOMODEL_PYTHONPATH="${NEMO_AUTOMODEL_SUPPORT_PYDEPS}:${NEMO_RL_ROOT}:"
    fi
fi
export PYTHONPATH="${BUNDLE_ROOT}:${PYTHON_STARTUP_OVERLAY:+${PYTHON_STARTUP_OVERLAY}:}${AUTOMODEL_PYTHONPATH}${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHON_DEP_OVERLAY_ROOT:+${PYTHON_DEP_OVERLAY_ROOT}:}${SLIME_ROOT}${MEGATRON_BRIDGE_ROOT:+:${MEGATRON_BRIDGE_ROOT}/src}:${MEGATRON_ROOT}${SGLANG_ROOT:+:${SGLANG_ROOT}/python}:${NANOBOT_ROOT}"
if [ "${CLAWGYM_NATIVE_MEMORY_EFFICIENT_ATTENTION:-0}" = 1 ]; then
 export PYTHONPATH="${PYTHONPATH}:${CLAWGYM_ATTENTION_PYTHON_OVERLAY:?}"
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OPENCLAW_TRAIN_TASKS="${OPENCLAW_TRAIN_TASKS:-${PROJECT_ROOT}/data/clawgym_nanobot/claw_train_cleaned_tasks}"
export OPENCLAW_TRAIN_ORDER="${OPENCLAW_TRAIN_ORDER:-sequential_then_shuffle}"
export OPENCLAW_EVAL_TASKS="${OPENCLAW_EVAL_TASKS:-${PROJECT_ROOT}/data/clawgym_nanobot/clawgym_eval}"
export OPENCLAW_EVAL_ORDER="${OPENCLAW_EVAL_ORDER:-sequential}"
export OPENCLAW_EVAL_MAX_TURNS="${OPENCLAW_EVAL_MAX_TURNS:-1}"
export OPENCLAW_EVAL_N_SAMPLES
export CLAWGYM_EVAL_NUM_PROMPTS
export NANOBOT_EVAL_NUM_WORKERS="${NANOBOT_EVAL_NUM_WORKERS:-64}"
export OPENCLAW_EVAL_NUM_WORKERS="${OPENCLAW_EVAL_NUM_WORKERS:-${NANOBOT_EVAL_NUM_WORKERS}}"
export CLAWGYM_NANOBOT_TRAIN_TURN_TYPE="${CLAWGYM_NANOBOT_TRAIN_TURN_TYPE:-main}"
export CLAWGYM_NANOBOT_EVAL_TURN_TYPE="${CLAWGYM_NANOBOT_EVAL_TURN_TYPE:-side}"
export OPENCLAW_CONTEXT_WINDOW="${OPENCLAW_CONTEXT_WINDOW:-65536}"
export OPENCLAW_MAX_TOKENS="${OPENCLAW_MAX_TOKENS:-8192}"
export NANOBOT_ROOT
export NANOBOT_RL_RUNNER
export NANOBOT_IMPORT_LOCAL_PACKAGE
export NANOBOT_EXTRA_PYTHONPATH NANOBOT_CONDA_ROOT NANOBOT_CONDA_ENV_NAME
export NANOBOT_RL_TOKENIZER_PATH="${NANOBOT_RL_TOKENIZER_PATH:-${HF_CHECKPOINT}}"
export CLAWGYM_TASK_PROMPT_MODE="${CLAWGYM_TASK_PROMPT_MODE:-original}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER-qwen25}"
export REASONING_PARSER="${REASONING_PARSER-qwen3}"
export CLAWGYM_NANOBOT_USE_UPSTREAM_CHAT_COMPLETIONS="${CLAWGYM_NANOBOT_USE_UPSTREAM_CHAT_COMPLETIONS:-1}"
export CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL="${CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL:-}"
export CLAWGYM_NANOBOT_RESEARCHCLAW_COMPAT="${CLAWGYM_NANOBOT_RESEARCHCLAW_COMPAT:-0}"
export CLAWGYM_NANOBOT_ENABLE_REALIGN="${CLAWGYM_NANOBOT_ENABLE_REALIGN:-1}"
export CLAWGYM_NANOBOT_PRUNE_EMPTY_RETRIES="${CLAWGYM_NANOBOT_PRUNE_EMPTY_RETRIES:-0}"
export CLAWGYM_ALLOW_MULTI_TOOL_CALLS="${CLAWGYM_ALLOW_MULTI_TOOL_CALLS:-0}"
export NANOBOT_MICROCOMPACT_ENABLED="${NANOBOT_MICROCOMPACT_ENABLED:-1}"
export CLAWGYM_NANOBOT_ROUTING_KEY="${CLAWGYM_NANOBOT_ROUTING_KEY:-0}"
export CLAWGYM_SGLANG_CONTEXT_OVERFLOW_RETRY="${CLAWGYM_SGLANG_CONTEXT_OVERFLOW_RETRY:-1}"
export CLAWGYM_SGLANG_CONTEXT_GUARD_TOKENS="${CLAWGYM_SGLANG_CONTEXT_GUARD_TOKENS:-128}"
export NANOBOT_STRIP_REASONING_CONTEXT="${NANOBOT_STRIP_REASONING_CONTEXT:-0}"
# Nanobot echoes reasoning_content on the next tool turn. Preserve it in the
# adapter tree so one agent trajectory is not forked into one sample per request.
export CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY="${CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY:-preserve}"
# Optional trajectory-manager threshold. Leave it unset for the established
# Qwen path; model-specific launchers can force token-drift boundaries to fork
# instead of REALIGNing a generated response into zero-loss prompt context.
export CLAWGYM_NANOBOT_FORK_THRESHOLD_TOKENS="${CLAWGYM_NANOBOT_FORK_THRESHOLD_TOKENS:-}"
export NANOBOT_RL_DISABLE_CONSOLIDATION="${NANOBOT_RL_DISABLE_CONSOLIDATION:-0}"
export NANOBOT_RL_FAST_ARCHIVE="${NANOBOT_RL_FAST_ARCHIVE:-0}"
export NANOBOT_ENABLE_THINKING="${NANOBOT_ENABLE_THINKING:-1}"
export NANOBOT_APPEND_NO_THINK="${NANOBOT_APPEND_NO_THINK:-}"
export NANOBOT_SHOW_REASONING="${NANOBOT_SHOW_REASONING:-1}"
export NANOBOT_WEB_ENABLE="${NANOBOT_WEB_ENABLE:-0}"
export NANOBOT_TURN_TIMEOUT="${NANOBOT_TURN_TIMEOUT:-${NANOBOT_TASK_TIMEOUT:-900}}"
export NANOBOT_TRAIN_TURN_TIMEOUT="${NANOBOT_TRAIN_TURN_TIMEOUT:-${NANOBOT_TURN_TIMEOUT}}"
export NANOBOT_EVAL_TURN_TIMEOUT="${NANOBOT_EVAL_TURN_TIMEOUT:-${NANOBOT_TURN_TIMEOUT}}"
export NANOBOT_TASK_TIMEOUT="${NANOBOT_TASK_TIMEOUT:-${NANOBOT_TURN_TIMEOUT}}"
export NANOBOT_REWARD_TIMEOUT="${NANOBOT_REWARD_TIMEOUT:-120}"
export NANOBOT_RL_LLM_TIMEOUT_S="${NANOBOT_RL_LLM_TIMEOUT_S:-0}"
export NANOBOT_RL_OPENAI_COMPAT_TIMEOUT_S="${NANOBOT_RL_OPENAI_COMPAT_TIMEOUT_S:-0}"
export NANOBOT_RL_NO_MAX_TOKENS="${NANOBOT_RL_NO_MAX_TOKENS:-0}"
export NANOBOT_KEEP_WORKDIR="${NANOBOT_KEEP_WORKDIR:-0}"
export NANOBOT_NO_WEB_CLI_TIMEOUT="${NANOBOT_NO_WEB_CLI_TIMEOUT:-5}"
export NANOBOT_MAX_TOOL_ITERATIONS="${NANOBOT_MAX_TOOL_ITERATIONS:-200}"
export NANOBOT_TIMEZONE="${NANOBOT_TIMEZONE:-Asia/Shanghai}"
export NANOBOT_EXEC_ENABLE="${NANOBOT_EXEC_ENABLE:-1}"
export NANOBOT_EXEC_TIMEOUT="${NANOBOT_EXEC_TIMEOUT:-60}"
export NANOBOT_REASONING_EFFORT="${NANOBOT_REASONING_EFFORT:-medium}"
export NANOBOT_SUBPROCESS_CAPTURE_TAIL_BYTES="${NANOBOT_SUBPROCESS_CAPTURE_TAIL_BYTES:-1048576}"
export CLAWGYM_DROP_TRAIN_NANOBOT_LOGS="${CLAWGYM_DROP_TRAIN_NANOBOT_LOGS:-1}"
export CLAWGYM_NANOBOT_TRAIN_SAMPLE_MODE="${CLAWGYM_NANOBOT_TRAIN_SAMPLE_MODE:-token_prefix}"
export CLAWGYM_NANOBOT_EVAL_SAMPLE_MODE="${CLAWGYM_NANOBOT_EVAL_SAMPLE_MODE:-final}"
export CLAWGYM_NANOBOT_EVAL_SESSION_PREFIX="${CLAWGYM_NANOBOT_EVAL_SESSION_PREFIX:-task}"
export CLAWGYM_NANOBOT_EVAL_UNIQUE_SESSION_IDS="${CLAWGYM_NANOBOT_EVAL_UNIQUE_SESSION_IDS:-0}"
export CLAWGYM_DISTRIBUTED_ROLLOUT_NANOBOT="${CLAWGYM_DISTRIBUTED_ROLLOUT_NANOBOT:-1}"
export NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE="${NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE:-64}"
export NANOBOT_ROLLOUT_GROUP_AFFINITY="${NANOBOT_ROLLOUT_GROUP_AFFINITY:-1}"
export NANOBOT_ROLLOUT_NODE_API_PORT_OFFSET="${NANOBOT_ROLLOUT_NODE_API_PORT_OFFSET:-1000}"
export NANOBOT_ROLLOUT_WORKDIR_BASE="${NANOBOT_ROLLOUT_WORKDIR_BASE:-/tmp/${USER:-u}/clawgym-nanobot-rollout}"
ROLLOUT_ENGINE_COUNT="$(((ROLLOUT_NUM_GPUS + ROLLOUT_GPUS - 1) / ROLLOUT_GPUS))"
export CLAWGYM_NANOBOT_MAX_WORKERS="${CLAWGYM_NANOBOT_MAX_WORKERS:-$((NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE * ROLLOUT_ENGINE_COUNT))}"
export CLAWGYM_REWARD_SOURCE="${CLAWGYM_REWARD_SOURCE:-code_verifier}"
export POLAR_EVALUATOR_STRATEGY="${POLAR_EVALUATOR_STRATEGY:-clawgym_nanobot.polar_evaluator:ClawGymCodeVerifierEvaluator}"
export POLAR_EVALUATOR_TIMEOUT="${POLAR_EVALUATOR_TIMEOUT:-${NANOBOT_REWARD_TIMEOUT}}"
export POLAR_FAILED_EXEC_LOG_DIR="${POLAR_FAILED_EXEC_LOG_DIR:-}"
export POLAR_FAILED_EXEC_LOG_MAX_BYTES="${POLAR_FAILED_EXEC_LOG_MAX_BYTES:-65536}"
export POLAR_FAILED_EXEC_LOG_MAX_CASES="${POLAR_FAILED_EXEC_LOG_MAX_CASES:-256}"
export WSCLAW_JUDGE_WORKER_URLS="${WSCLAW_JUDGE_WORKER_URLS:-}"
export WSCLAW_JUDGE_API_KEY="${WSCLAW_JUDGE_API_KEY:-apiKey}"
export WSCLAW_JUDGE_WORKERS_PER_NODE="${WSCLAW_JUDGE_WORKERS_PER_NODE:-64}"
export WSCLAW_TRAIN_JUDGE_LOG_DIR="${WSCLAW_TRAIN_JUDGE_LOG_DIR:-${LOG_DIR}/wsclaw_judge_cases}"
export WSCLAW_TRAIN_KEEP_JUDGE_CASES="${WSCLAW_TRAIN_KEEP_JUDGE_CASES:-0}"
export RESEARCHCLAW_KEEP_TRAIN_NANOBOT_LOGS="${RESEARCHCLAW_KEEP_TRAIN_NANOBOT_LOGS:-1}"
export CLAWGYM_KEEP_LATEST_EVAL_LOGS="${CLAWGYM_KEEP_LATEST_EVAL_LOGS:-0}"
export JUDGE_NANOBOT_LOG_DIR="${JUDGE_NANOBOT_LOG_DIR:-${LOG_DIR}/judge_nanobot_logs}"
export LATEST_EVAL_LOG_STATE_DIR="${LATEST_EVAL_LOG_STATE_DIR:-${LOG_DIR}/.latest_eval_log_state}"
export CLAWGYM_EVAL_USER_AGENT_REWARD="${CLAWGYM_EVAL_USER_AGENT_REWARD:-0}"
export CLAWGYM_USER_AGENT_BACKEND="${CLAWGYM_USER_AGENT_BACKEND:-nanobot}"
export CLAWGYM_USER_AGENT_PROFILE_NAME="${CLAWGYM_USER_AGENT_PROFILE_NAME:-gemma4}"
export CLAWGYM_USER_AGENT_BASE_URL="${CLAWGYM_USER_AGENT_BASE_URL:-}"
export CLAWGYM_USER_AGENT_MODEL="${CLAWGYM_USER_AGENT_MODEL:-gemma-user-agent}"

prepare_private_env_file() {
    local value_name="$1"
    local file_name="$2"
    local file_basename="$3"
    local default_value="$4"
    local file_path="${!file_name:-}"
    local value

    if [ -z "${file_path}" ]; then
        value="${!value_name:-${default_value}}"
        if [ -z "${value}" ]; then
            return
        fi
        install -d -m 700 "${LOG_DIR}/.private"
        file_path="${LOG_DIR}/.private/${file_basename}"
        (
            umask 077
            printf '%s\n' "${value}" > "${file_path}.tmp.$$"
            chmod 600 "${file_path}.tmp.$$"
            mv -f "${file_path}.tmp.$$" "${file_path}"
        )
    elif [[ "${file_path}" != /* ]]; then
        echo "ERROR: ${file_name} must be an absolute path" >&2
        exit 1
    elif [ ! -s "${file_path}" ] || [ "$(stat -c '%a' "${file_path}")" != 600 ]; then
        echo "ERROR: ${file_name} must name a non-empty mode-0600 file" >&2
        exit 1
    fi

    printf -v "${file_name}" '%s' "${file_path}"
    export "${file_name}"
    unset "${value_name}"
}

prepare_private_env_file \
    CLAWGYM_USER_AGENT_API_KEY \
    CLAWGYM_USER_AGENT_API_KEY_FILE \
    user_agent_api_key \
    apiKey
prepare_private_env_file \
    CLAWGYM_USER_AGENT_EXTRA_API_KEY \
    CLAWGYM_USER_AGENT_EXTRA_API_KEY_FILE \
    user_agent_extra_api_key \
    ""
prepare_private_env_file \
    CLAWGYM_USER_AGENT_DEEPSEEK_API_KEY \
    CLAWGYM_USER_AGENT_DEEPSEEK_API_KEY_FILE \
    user_agent_deepseek_api_key \
    ""
export CLAWGYM_USER_AGENT_WORKERS="${CLAWGYM_USER_AGENT_WORKERS:-16}"
export CLAWGYM_USER_AGENT_CONCURRENCY="${CLAWGYM_USER_AGENT_CONCURRENCY:-${CLAWGYM_USER_AGENT_WORKERS}}"
export CLAWGYM_USER_AGENT_REWARD_MODE="${CLAWGYM_USER_AGENT_REWARD_MODE:-continuous}"
export CLAWGYM_USER_AGENT_FEEDBACK_SUMMARIZE_RUBRIC="${CLAWGYM_USER_AGENT_FEEDBACK_SUMMARIZE_RUBRIC:-0}"
export CLAWGYM_USER_AGENT_FAIL_CLOSED="${CLAWGYM_USER_AGENT_FAIL_CLOSED:-1}"
export CLAWGYM_USER_AGENT_TIMEOUT_SEC="${CLAWGYM_USER_AGENT_TIMEOUT_SEC:-900}"
export CLAWGYM_USER_AGENT_MAX_RETRIES="${CLAWGYM_USER_AGENT_MAX_RETRIES:-3}"
export CLAWGYM_USER_AGENT_CONTEXT_LENGTH="${CLAWGYM_USER_AGENT_CONTEXT_LENGTH:-65536}"
export CLAWGYM_USER_AGENT_MAX_TOKENS="${CLAWGYM_USER_AGENT_MAX_TOKENS:-512}"
export CLAWGYM_USER_AGENT_ENABLE_THINKING="${CLAWGYM_USER_AGENT_ENABLE_THINKING:-1}"
export CLAWGYM_USER_AGENT_SHOW_REASONING="${CLAWGYM_USER_AGENT_SHOW_REASONING:-1}"
export CLAWGYM_USER_AGENT_WEB_ENABLE="${CLAWGYM_USER_AGENT_WEB_ENABLE:-0}"
export CLAWGYM_USER_AGENT_EXEC_ENABLE="${CLAWGYM_USER_AGENT_EXEC_ENABLE:-1}"
export CLAWGYM_USER_AGENT_EXEC_TIMEOUT="${CLAWGYM_USER_AGENT_EXEC_TIMEOUT:-60}"
export CLAWGYM_USER_AGENT_WORKDIR="${CLAWGYM_USER_AGENT_WORKDIR:-${LOG_DIR}/user_agent_reward}"
export CLAWGYM_SYN_TASK_JSONL="${CLAWGYM_SYN_TASK_JSONL:-${PROJECT_ROOT}/data/clawgym_nanobot/syn_task.jsonl}"
export WSCLAW_EXEC_MEMORY_GUARD_ENABLED="${WSCLAW_EXEC_MEMORY_GUARD_ENABLED:-0}"
export WSCLAW_EXEC_MEMORY_SOFT_GIB="${WSCLAW_EXEC_MEMORY_SOFT_GIB:-280}"
export WSCLAW_EXEC_MEMORY_HARD_GIB="${WSCLAW_EXEC_MEMORY_HARD_GIB:-340}"
export WSCLAW_EXEC_BLOCK_AVAILABLE_GIB="${WSCLAW_EXEC_BLOCK_AVAILABLE_GIB:-100}"
export WSCLAW_EXEC_EMERGENCY_AVAILABLE_GIB="${WSCLAW_EXEC_EMERGENCY_AVAILABLE_GIB:-64}"
export WSCLAW_EXEC_FATAL_AVAILABLE_GIB="${WSCLAW_EXEC_FATAL_AVAILABLE_GIB:-40}"
export WSCLAW_EXEC_MEMORY_POLL_SECONDS="${WSCLAW_EXEC_MEMORY_POLL_SECONDS:-0.5}"
export WSCLAW_EXEC_TERMINATE_GRACE_SECONDS="${WSCLAW_EXEC_TERMINATE_GRACE_SECONDS:-5}"
export WSCLAW_EXEC_ADMISSION_TIMEOUT_SECONDS="${WSCLAW_EXEC_ADMISSION_TIMEOUT_SECONDS:-600}"
export WSCLAW_EXEC_GUARD_HEARTBEAT_MAX_AGE_SECONDS="${WSCLAW_EXEC_GUARD_HEARTBEAT_MAX_AGE_SECONDS:-15}"
export CLAWGYM_EXEC_DROP_PRIVILEGES="${CLAWGYM_EXEC_DROP_PRIVILEGES:-0}"
export CLAWGYM_EXEC_UID="${CLAWGYM_EXEC_UID:-65534}"
export CLAWGYM_EXEC_GID="${CLAWGYM_EXEC_GID:-65534}"
export CLAWGYM_RAW_SOURCE_INTEGRITY="${CLAWGYM_RAW_SOURCE_INTEGRITY:-0}"
export CLAWGYM_RAW_SOURCE_MANIFEST="${CLAWGYM_RAW_SOURCE_MANIFEST:-}"
export POLAR_GATEWAY_SESSION_TMPDIR="${POLAR_GATEWAY_SESSION_TMPDIR:-/tmp/prorl-polar-${NODE_RANK}}"
export PRORL_FATAL_RESUME_MARKER_DIR="${PRORL_FATAL_RESUME_MARKER_DIR:-${LOG_DIR}/fatal-resume}"
export PRORL_CHECKPOINT_CANDIDATES_DIR="${PRORL_CHECKPOINT_CANDIDATES_DIR:-${SAVE_DIR}/candidates}"
export PRORL_INITIAL_RESUME_CHECKPOINT="${PRORL_INITIAL_RESUME_CHECKPOINT:-${LOAD_CKPT}}"
export PRORL_RUN_ID="${PRORL_RUN_ID:-${RUN_ID}}"
if [ "${WSCLAW_EXEC_MEMORY_GUARD_ENABLED}" = 1 ]; then
    export WSCLAW_EXEC_GUARD_SCOPE="${RUN_ID}:rank${NODE_RANK}:$(hostname)"
    export WSCLAW_EXEC_GUARD_STATE_DIR="${LOG_DIR}/exec-memory/rank${NODE_RANK}"
    export WSCLAW_EXEC_GUARD_GATE_FILE="${WSCLAW_EXEC_GUARD_STATE_DIR}/block-new-exec.json"
    export WSCLAW_EXEC_GUARD_HEARTBEAT_FILE="${WSCLAW_EXEC_GUARD_STATE_DIR}/heartbeat.json"
fi
case "${CLAWGYM_EXEC_DROP_PRIVILEGES}:${CLAWGYM_RAW_SOURCE_INTEGRITY}" in
    0:0|0:1|1:0|1:1) ;;
    *) echo "ERROR: CLAWGYM_EXEC_DROP_PRIVILEGES and CLAWGYM_RAW_SOURCE_INTEGRITY must be 0 or 1" >&2; exit 2 ;;
esac
if [ "${CLAWGYM_EXEC_DROP_PRIVILEGES}" = 1 ]; then
    [[ "${CLAWGYM_EXEC_UID}" =~ ^[1-9][0-9]*$ ]]
    [[ "${CLAWGYM_EXEC_GID}" =~ ^[1-9][0-9]*$ ]]
fi
if [ "${CLAWGYM_RAW_SOURCE_INTEGRITY}" = 1 ]; then
    case "${CLAWGYM_RAW_SOURCE_MANIFEST}" in /*) ;; *) echo "ERROR: CLAWGYM_RAW_SOURCE_MANIFEST must be absolute" >&2; exit 2 ;; esac
    test -s "${CLAWGYM_RAW_SOURCE_MANIFEST}"
fi
export USE_POLAR POLAR_ROLLOUT_PORT POLAR_GATEWAY_PORT POLAR_MAX_WORKERS_PER_NODE
export POLAR_MAX_POSTRUN_WORKERS_PER_NODE
export SLIME_ASYNC_TRAINING POLAR_SYNCHRONOUS_POLICY_VERSIONS
export POLAR_EXPECTED_GATEWAY_NODES POLAR_GATEWAY_ON_DRIVER
export POLAR_MAX_ASYNC_LEVEL POLAR_MAX_OFF_POLICY_STEPS POLAR_REQUEST_TIMEOUT
export POLAR_MIN_COMPLETE_ACCEPT_FRACTION
export POLAR_MAX_LOW_COMPLETE_RETRIES
export POLAR_TIMEOUT_AS_ZERO
export POLAR_PREFIX_REALIGN_REPLAYED_RESPONSE
export POLAR_PREFIX_REALIGN_MAX_NEXT_RESPONSE_TOKENS
export POLAR_EVAL_MAX_CONCURRENCY
export POLAR_EVAL_INFRA_RETRIES
export POLAR_TASK_TIMEOUT POLAR_MODEL_SERVED POLAR_AGENT_HARNESS SGLANG_ROUTER_PORT
export POLAR_ROUTER_WAIT_TIMEOUT_SECONDS
export SGLANG_ROUTER_POLICY SGLANG_ROUTER_ASSIGNMENT_MODE
export POLAR_GATEWAY_LOCAL_SGLANG POLAR_SGLANG_SERVER_PORT
export POLAR_GATEWAY_FIRST_NODE_RANK
export SLIME_ROLLOUT_MANAGER_ON_DRIVER_NODE="${SLIME_ROLLOUT_MANAGER_ON_DRIVER_NODE:-${USE_POLAR}}"
export POLAR_EXPOSE_TRAINING_TOKENS="${POLAR_EXPOSE_TRAINING_TOKENS:-${USE_POLAR}}"
export SAVE_CKPTS="${SAVE_CKPTS:-0}"
export SAVE_LATEST_BEST_CKPTS="${SAVE_LATEST_BEST_CKPTS:-0}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-0}"
export NANOBOT_LOG_DIR="${NANOBOT_LOG_DIR:-${LOG_DIR}/nanobot_logs}"
export CLAWGYM_HF_CHECKPOINT_ROOT="${CLAWGYM_HF_CHECKPOINT_ROOT:-${SAVE_DIR}}"
export CLAWGYM_EVAL_SNAPSHOT_ROOT="${CLAWGYM_EVAL_SNAPSHOT_ROOT:-${LOG_DIR}/eval_snapshots}"
export CLAWGYM_HF_CHECKPOINT_MIN_EVAL_COVERAGE="${CLAWGYM_HF_CHECKPOINT_MIN_EVAL_COVERAGE:-0.95}"
export NANOBOT_WORKDIR="${NANOBOT_WORKDIR:-${LOG_DIR}/nanobot_work}"
INTERNAL_NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR},${MASTER_ADDR,,}"
IFS=',' read -r -a CLAWGYM_NO_PROXY_HOST_ARRAY <<< "${CLAWGYM_NODE_HOSTS:-}"
for host in "${CLAWGYM_NO_PROXY_HOST_ARRAY[@]}"; do
    if [ -n "${host}" ]; then
        INTERNAL_NO_PROXY="${INTERNAL_NO_PROXY},${host},${host,,}"
    fi
done
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${INTERNAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${INTERNAL_NO_PROXY}"
unset CLAWGYM_NO_PROXY_HOST_ARRAY INTERNAL_NO_PROXY

TRAIN_PROMPT_DATA="${TRAIN_PROMPT_DATA:-${LOG_DIR}/prompt_data/claw_train.jsonl}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-${LOG_DIR}/prompt_data/clawgym_eval.jsonl}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-clawgym_eval}"
POLAR_TOPOLOGY_PATH="${POLAR_TOPOLOGY_PATH:-${LOG_DIR}/polar/topology.yaml}"
POLAR_CONFIG_PATH="${POLAR_CONFIG_PATH:-${LOG_DIR}/polar/polar_config.yaml}"
POLAR_TEMPLATE_DIR="${POLAR_TEMPLATE_DIR:-${SCRIPT_DIR}/polar}"

if [ "${USE_POLAR}" = "1" ]; then
    case "${POLAR_GATEWAY_ON_DRIVER}" in
        0|1) ;;
        *) echo "ERROR: POLAR_GATEWAY_ON_DRIVER must be 0 or 1" >&2; exit 1 ;;
    esac
    if [ "${POLAR_GATEWAY_ON_DRIVER}" = "1" ] && [ "${POLAR_EXPECTED_GATEWAY_NODES}" -ne 1 ]; then
        echo "ERROR: POLAR_GATEWAY_ON_DRIVER currently supports exactly one gateway" >&2
        exit 1
    fi
    IFS=',' read -r -a CLAWGYM_HOST_ARRAY <<< "${CLAWGYM_NODE_HOSTS:-}"
    gateway_host_index="${POLAR_GATEWAY_FIRST_NODE_RANK}"
    required_host_count=$((gateway_host_index + POLAR_EXPECTED_GATEWAY_NODES))
    if [ "${POLAR_GATEWAY_ON_DRIVER}" = "1" ]; then
        required_host_count=1
        gateway_host_index=0
    fi
    if [ "${#CLAWGYM_HOST_ARRAY[@]}" -lt "${required_host_count}" ]; then
        echo "ERROR: Polar ClawGym requires ${required_host_count} configured host entries" >&2
        exit 1
    fi
    POLAR_GATEWAY_HOST="${POLAR_GATEWAY_HOST:-${CLAWGYM_HOST_ARRAY[${gateway_host_index}]}}"
    export POLAR_GATEWAY_HOST
    for ((gateway_index = 2; gateway_index <= POLAR_EXPECTED_GATEWAY_NODES; gateway_index++)); do
        gateway_host_var="POLAR_GATEWAY_HOST_${gateway_index}"
        if [ -z "${!gateway_host_var:-}" ]; then
            gateway_array_index=$((gateway_host_index + gateway_index - 1))
            printf -v "${gateway_host_var}" '%s' "${CLAWGYM_HOST_ARRAY[${gateway_array_index}]}"
        fi
        export "${gateway_host_var}"
    done
fi

create_prompt_data() {
    local sources="$1"
    local output="$2"
    local limit="$3"
    local turn_type="$4"
    local extra_args=()
    if [ "${CLAWGYM_POLAR_SKIP_UNAVAILABLE_TASKS:-0}" = "1" ]; then
        extra_args+=(--skip-unavailable)
    fi
    python3 -m clawgym_nanobot.polar_data \
        --sources "${sources}" \
        --output "${output}" \
        --limit "${limit}" \
        --turn-type "${turn_type}" \
        "${extra_args[@]}"
}

if [ "${DO_EVAL}" = "1" ] && [ "${NODE_RANK}" = "0" ]; then
    create_prompt_data \
        "${OPENCLAW_EVAL_TASKS}" \
        "${EVAL_PROMPT_DATA}" \
        "${CLAWGYM_EVAL_NUM_PROMPTS}" \
        "${CLAWGYM_NANOBOT_EVAL_TURN_TYPE}"
fi

render_polar_configs() {
    mkdir -p "$(dirname "${POLAR_TOPOLOGY_PATH}")" "$(dirname "${POLAR_CONFIG_PATH}")"
    python3 - "${POLAR_TEMPLATE_DIR}/topology.yaml.template" "${POLAR_TOPOLOGY_PATH}" \
        "${POLAR_TEMPLATE_DIR}/polar_config.yaml.template" "${POLAR_CONFIG_PATH}" <<'PY'
import os
import sys
from pathlib import Path
from string import Template

mapping = dict(os.environ)
expected_gateway_nodes = int(mapping.get("POLAR_EXPECTED_GATEWAY_NODES", "1"))
if expected_gateway_nodes < 1:
    raise SystemExit("POLAR_EXPECTED_GATEWAY_NODES must be positive")
port = mapping["POLAR_GATEWAY_PORT"]
workers = mapping["POLAR_MAX_WORKERS_PER_NODE"]
postrun_workers = mapping["POLAR_MAX_POSTRUN_WORKERS_PER_NODE"]
model = mapping["POLAR_MODEL_SERVED"]
master = mapping["MASTER_ADDR"]
router_port = mapping["SGLANG_ROUTER_PORT"]
local_sglang = mapping.get("POLAR_GATEWAY_LOCAL_SGLANG", "0") == "1"
server_port = mapping.get("POLAR_SGLANG_SERVER_PORT", "15000")

def inference_base_url(host):
    if local_sglang:
        return f"http://{host}:{server_port}"
    return f"http://{master}:{router_port}"

mapping["POLAR_PRIMARY_INFERENCE_BASE_URL"] = inference_base_url(mapping["POLAR_GATEWAY_HOST"])
extra_nodes = []
for index in range(2, expected_gateway_nodes + 1):
    host = mapping[f"POLAR_GATEWAY_HOST_{index}"]
    extra_nodes.append(f"""    - id: clawgym-rollout-{index}
      host: 0.0.0.0
      port: {port}
      public_url: http://{host}:{port}
      max_init_workers: {workers}
      max_run_workers: {workers}
      max_postrun_workers: {postrun_workers}
      model_served: {model}
      inference:
        engine: sglang
        base_url: {inference_base_url(host)}""")
mapping["POLAR_EXTRA_GATEWAY_NODES"] = "\n".join(extra_nodes)

harness = mapping.get("POLAR_AGENT_HARNESS", "nanobot")
if harness == "nanobot":
    mapping["POLAR_AGENT_CONFIG"] = '''    harness: nanobot
    settings:
      turn_type: "{sample.metadata.turn_type}"'''
else:
    raise SystemExit(f"unsupported POLAR_AGENT_HARNESS: {harness}")

for source, target in zip(sys.argv[1::2], sys.argv[2::2], strict=True):
    rendered = Template(Path(source).read_text(encoding="utf-8")).substitute(mapping)
    Path(target).write_text(rendered, encoding="utf-8")
PY
}

wait_for_file() {
    local path="$1"
    local deadline=$((SECONDS + 600))
    while [ ! -s "${path}" ]; do
        if [ "${SECONDS}" -ge "${deadline}" ]; then
            echo "ERROR: timed out waiting for ${path}" >&2
            return 1
        fi
        sleep 2
    done
}

if [ "${USE_POLAR}" = "1" ]; then
    if [ "${NODE_RANK}" = "0" ]; then
        create_prompt_data \
            "${OPENCLAW_TRAIN_TASKS}" \
            "${TRAIN_PROMPT_DATA}" \
            0 \
            "${CLAWGYM_NANOBOT_TRAIN_TURN_TYPE}"
        render_polar_configs
    else
        wait_for_file "${POLAR_TOPOLOGY_PATH}"
        wait_for_file "${POLAR_CONFIG_PATH}"
    fi
fi

cleanup_ray() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
start_ray_port_seed() {
    python3 -u "${PROJECT_ROOT}/scripts/clawgym_nanobot/seed_ray_agent_ports.py" \
        "${RAY_TMPDIR}" \
        "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        "${RAY_METRICS_EXPORT_PORT}" \
        "${RAY_RUNTIME_ENV_AGENT_PORT}" \
        >"${LOG_DIR}/ray-port-seed-rank${NODE_RANK}.log" 2>&1 &
    ray_port_seed_pid=$!
}
cleanup_stale_training_drivers() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
cleanup_stale_gpu_processes() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
POLAR_PIDS=()
WANDB_COALESCER_PID=""
WANDB_RELAY_PID=""
WSCLAW_EXEC_MEMORY_GUARD_PID=""
polar_listener_inodes() {
    # Polar reserves 18080 (rollout control) and 18100 (gateway). Read procfs
    # directly because the minimal ClusterX image may not provide lsof/ss.
    awk '$4 == "0A" && ($2 ~ /:46A0$/ || $2 ~ /:46B4$/) {print $10}' \
        /proc/net/tcp /proc/net/tcp6 2>/dev/null | sort -u
}
listener_inodes_for_port() {
    local port="$1"
    local port_hex
    printf -v port_hex '%04X' "${port}"
    awk -v suffix=":${port_hex}" \
        '$4 == "0A" && substr($2, length($2) - length(suffix) + 1) == suffix {print $10}' \
        /proc/net/tcp /proc/net/tcp6 2>/dev/null | sort -u
}
listener_owner_pids_for_port() {
    local fd inode pid port="$1"
    declare -A seen_pids=()
    while IFS= read -r inode; do
        [[ "${inode}" =~ ^[1-9][0-9]*$ ]] || continue
        for fd in /proc/[1-9]*/fd/*; do
            [ "$(readlink "${fd}" 2>/dev/null || true)" = "socket:[${inode}]" ] || continue
            pid="${fd#/proc/}"
            pid="${pid%%/*}"
            [ "${pid}" != "$$" ] && seen_pids["${pid}"]=1
        done
    done < <(listener_inodes_for_port "${port}")
    printf '%s\n' "${!seen_pids[@]}" | sed '/^$/d' | sort -n
}
cleanup_stale_local_sglang_listener() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
cleanup_stale_polar_services() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
cleanup_services() {
    for pid in "${POLAR_PIDS[@]:-}"; do
        kill "${pid}" >/dev/null 2>&1 || true
    done
    wait "${POLAR_PIDS[@]:-}" >/dev/null 2>&1 || true
}
start_wsclaw_exec_memory_guard() {
    [ "${WSCLAW_EXEC_MEMORY_GUARD_ENABLED}" = 1 ] || return 0
    mkdir -p "${WSCLAW_EXEC_GUARD_STATE_DIR}" "${PRORL_FATAL_RESUME_MARKER_DIR}"
    chmod 700 "${WSCLAW_EXEC_GUARD_STATE_DIR}" "${PRORL_FATAL_RESUME_MARKER_DIR}"
    rm -f -- "${WSCLAW_EXEC_GUARD_GATE_FILE}" "${WSCLAW_EXEC_GUARD_HEARTBEAT_FILE}"
    python3 -u -m clawgym_nanobot.exec_memory_guard \
        --scope "${WSCLAW_EXEC_GUARD_SCOPE}" \
        --rank "${NODE_RANK}" \
        --state-dir "${WSCLAW_EXEC_GUARD_STATE_DIR}" \
        --fatal-marker-dir "${PRORL_FATAL_RESUME_MARKER_DIR}" \
        --soft-gib "${WSCLAW_EXEC_MEMORY_SOFT_GIB}" \
        --hard-gib "${WSCLAW_EXEC_MEMORY_HARD_GIB}" \
        --block-available-gib "${WSCLAW_EXEC_BLOCK_AVAILABLE_GIB}" \
        --emergency-available-gib "${WSCLAW_EXEC_EMERGENCY_AVAILABLE_GIB}" \
        --fatal-available-gib "${WSCLAW_EXEC_FATAL_AVAILABLE_GIB}" \
        --poll-seconds "${WSCLAW_EXEC_MEMORY_POLL_SECONDS}" \
        --terminate-grace-seconds "${WSCLAW_EXEC_TERMINATE_GRACE_SECONDS}" \
        >"${WSCLAW_EXEC_GUARD_STATE_DIR}/guard.log" 2>&1 &
    WSCLAW_EXEC_MEMORY_GUARD_PID="$!"
    sleep 1
    if ! kill -0 "${WSCLAW_EXEC_MEMORY_GUARD_PID}" 2>/dev/null; then
        wait "${WSCLAW_EXEC_MEMORY_GUARD_PID}" || true
        WSCLAW_EXEC_MEMORY_GUARD_PID=""
        echo "ERROR: WSClaw exec memory guard failed to start; see ${WSCLAW_EXEC_GUARD_STATE_DIR}/guard.log" >&2
        return 1
    fi
    echo "[proRL-memory] exec guard started pid=${WSCLAW_EXEC_MEMORY_GUARD_PID} scope=${WSCLAW_EXEC_GUARD_SCOPE}"
}
stop_wsclaw_exec_memory_guard() {
    [ -n "${WSCLAW_EXEC_MEMORY_GUARD_PID}" ] || return 0
    kill -TERM "${WSCLAW_EXEC_MEMORY_GUARD_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "${WSCLAW_EXEC_MEMORY_GUARD_PID}" 2>/dev/null || break
        sleep 0.25
    done
    if kill -0 "${WSCLAW_EXEC_MEMORY_GUARD_PID}" 2>/dev/null; then
        kill -KILL "${WSCLAW_EXEC_MEMORY_GUARD_PID}" 2>/dev/null || true
    fi
    wait "${WSCLAW_EXEC_MEMORY_GUARD_PID}" 2>/dev/null || true
    WSCLAW_EXEC_MEMORY_GUARD_PID=""
}
cleanup_wsclaw_exec_scope() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
prepare_polar_gateway_session_tmpdir() {
    [[ "${POLAR_GATEWAY_SESSION_TMPDIR}" = "${STATE_ROOT}/"* ]] || return 2
    install -d -m 0711 -- "${POLAR_GATEWAY_SESSION_TMPDIR}"
}
cleanup_polar_gateway_session_tmpdir() {
    : # Current run-owned retirement is managed by exact identities; never broad cleanup.
}
stop_wandb_jsonl_relay() {
    if [ -n "${WANDB_COALESCER_PID}" ] && kill -0 "${WANDB_COALESCER_PID}" >/dev/null 2>&1; then
        kill -TERM "${WANDB_COALESCER_PID}" >/dev/null 2>&1 || true
        for _ in $(seq 1 60); do
            if ! kill -0 "${WANDB_COALESCER_PID}" >/dev/null 2>&1; then
                wait "${WANDB_COALESCER_PID}" >/dev/null 2>&1 || true
                WANDB_COALESCER_PID=""
                break
            fi
            sleep 1
        done
        if [ -n "${WANDB_COALESCER_PID}" ]; then
            echo "WARNING: metric coalescer did not exit within 60s; stopping it" >&2
            kill "${WANDB_COALESCER_PID}" >/dev/null 2>&1 || true
            wait "${WANDB_COALESCER_PID}" >/dev/null 2>&1 || true
            WANDB_COALESCER_PID=""
        fi
    fi
    if [ -z "${WANDB_RELAY_PID}" ] || ! kill -0 "${WANDB_RELAY_PID}" >/dev/null 2>&1; then
        return
    fi
    for _ in $(seq 1 60); do
        if ! kill -0 "${WANDB_RELAY_PID}" >/dev/null 2>&1; then
            wait "${WANDB_RELAY_PID}" >/dev/null 2>&1 || true
            WANDB_RELAY_PID=""
            return
        fi
        sleep 1
    done
    echo "WARNING: W&B JSONL relay did not exit within 60s; stopping it" >&2
    kill -TERM "${WANDB_RELAY_PID}" >/dev/null 2>&1 || true
    wait "${WANDB_RELAY_PID}" >/dev/null 2>&1 || true
    WANDB_RELAY_PID=""
}
cleanup_all() {
    if [ "${NODE_RANK}" = "0" ]; then
        touch "${STOP_FILE}" 2>/dev/null || true
    fi
    stop_wandb_jsonl_relay
    cleanup_services
    if [ "${NODE_RANK}" = "0" ]; then
        cleanup_stale_training_drivers
    fi
    cleanup_ray
    cleanup_wsclaw_exec_scope
    stop_wsclaw_exec_memory_guard
    cleanup_stale_gpu_processes
    cleanup_polar_gateway_session_tmpdir
}
trap cleanup_all EXIT

wait_for_owned_polar_listener() {
    python3 - "$1" "$2" "60" <<'PY'
from pathlib import Path
import os, sys, time
pid, port, timeout = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
def identity(p):
    try:
        st=(Path('/proc')/str(p)/'stat').read_text().rsplit(')',1)[1].split()
        return None if st[0]=='Z' else (int(st[1]),st[19])
    except OSError:return None
initial=identity(pid)
assert initial is not None, 'Polar child already exited'
deadline=time.monotonic()+timeout
while time.monotonic()<deadline:
    assert identity(pid)==initial, 'This run Polar child exited or changed identity'
    rows={int(p.name):identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    owned={pid}
    while True:
        expanded=owned|{p for p,v in rows.items() if v and v[0] in owned}
        if expanded==owned:break
        owned=expanded
    listening=set()
    for family in ('tcp','tcp6'):
        for line in (Path('/proc/net')/family).read_text().splitlines()[1:]:
            f=line.split()
            if f[3]=='0A' and int(f[1].rsplit(':',1)[1],16)==port:listening.add(f[9])
    for child in owned:
        try:
            for fd in (Path('/proc')/str(child)/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target.startswith('socket:[') and target[8:-1] in listening:
                    assert identity(pid)==initial
                    print('OWNED_POLAR_LISTENER_VERIFIED')
                    raise SystemExit(0)
        except OSError:continue
    time.sleep(.2)
raise RuntimeError('This run Polar child did not own its configured listening port')
PY
}

start_polar_rollout_service() {
    local pid
    echo "[proRL-polar] starting rollout service on ${POLAR_GATEWAY_HOST}:${POLAR_ROLLOUT_PORT}"
    python3 -u -m polar.cli serve_rollout -c "${POLAR_TOPOLOGY_PATH}" \
        >"${LOG_DIR}/polar-rollout.log" 2>&1 &
    pid="$!"
    POLAR_PIDS+=("${pid}")
    wait_for_owned_polar_listener "${pid}" "${POLAR_ROLLOUT_PORT}"
}

start_polar_gateway() {
    local node_id="$1"
    local log_path="$2"
    local pid
    echo "[proRL-polar] starting ${node_id} with ${POLAR_MAX_WORKERS_PER_NODE} workers"
    TMPDIR="${POLAR_GATEWAY_SESSION_TMPDIR}" \
        TMP="${POLAR_GATEWAY_SESSION_TMPDIR}" \
        TEMP="${POLAR_GATEWAY_SESSION_TMPDIR}" \
        python3 -u -m polar.cli serve_gateway -c "${POLAR_TOPOLOGY_PATH}" \
        --node-id "${node_id}" >"${log_path}" 2>&1 &
    pid="$!"
    POLAR_PIDS+=("${pid}")
    wait_for_owned_polar_listener "${pid}" "${POLAR_GATEWAY_PORT}"
}

start_wandb_jsonl_relay() {
    if [ "${WANDB_JSONL_RELAY_ACTIVE}" != "1" ]; then
        return
    fi
    if [ -z "${WANDB_ENTITY}" ]; then
        echo "ERROR: WANDB_ENTITY is required when WANDB_JSONL_RELAY=1" >&2
        return 1
    fi
    mkdir -p "$(dirname "${SLIME_METRICS_JSONL_PATH}")" "${WANDB_DIR}"
    touch "${SLIME_METRICS_JSONL_PATH}" "${WANDB_STEPS_JSONL_PATH}"
    chmod 600 "${SLIME_METRICS_JSONL_PATH}" "${WANDB_STEPS_JSONL_PATH}"
    # A previous failed launch may have left this completion marker behind.
    # Rotate it before starting readers so the new relay cannot drain and exit immediately.
    if [ -e "${LOG_DIR}/train-exit-status.env" ]; then
        mv "${LOG_DIR}/train-exit-status.env" "${LOG_DIR}/train-exit-status.previous.$(date -u +%Y%m%d-%H%M%S).$$.env"
    fi
    rm -f "${WANDB_JSONL_COALESCER_DONE}"
    local coalescer_algorithm="${ADVANTAGE_ESTIMATOR:-grpo}"
    if [ "${NUM_ROLLOUT:-}" = "0" ] && [ "${DO_EVAL:-0}" = "1" ]; then
        coalescer_algorithm="eval_only"
    elif [ "${USE_OPSD:-0}" = "1" ]; then
        coalescer_algorithm="opsd"
    fi
    coalescer_args=(
        --source "${SLIME_METRICS_JSONL_PATH}"
        --output "${WANDB_STEPS_JSONL_PATH}"
        --algorithm "${coalescer_algorithm}"
        --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS:-0}"
        --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT:-1}"
        --critic-train-epochs "${CRITIC_TRAIN_EPOCHS:-1}"
        --first-rollout-iteration "${START_ROLLOUT_ID:-0}"
        --done-file "${LOG_DIR}/train-exit-status.env"
        --completion-marker "${WANDB_JSONL_COALESCER_DONE}"
        --poll-seconds "${WANDB_JSONL_POLL_SECONDS}"
    )
    if [ "${DO_EVAL:-0}" = "1" ]; then
        coalescer_args+=(--eval-interval "${OPENCLAW_EVAL_INTERVAL}")
    else
        coalescer_args+=(--skip-initial-eval)
    fi
    if [ "${SKIP_INITIAL_EVAL:-0}" = "1" ]; then
        coalescer_args+=(--skip-initial-eval)
    fi
    if [ "${TRAIN_BACKEND}" = "automodel" ]; then
        # The DTensor AutoModel actor still emits one aggregate rollout event
        # plus one train event. NeMo-Megatron now mirrors native Slime's
        # separate actor-rollout and actor-perf events.
        coalescer_args+=(--compact-policy-events)
    fi
    python3 -u -m clawgym_nanobot.metrics_step_coalescer \
        "${coalescer_args[@]}" >"${WANDB_JSONL_COALESCER_LOG}" 2>&1 &
    WANDB_COALESCER_PID="$!"
    sleep 1
    if ! kill -0 "${WANDB_COALESCER_PID}" >/dev/null 2>&1; then
        wait "${WANDB_COALESCER_PID}" || true
        WANDB_COALESCER_PID=""
        echo "ERROR: metric coalescer failed to start; see ${WANDB_JSONL_COALESCER_LOG}" >&2
        return 1
    fi
    python3 -u "${PROJECT_ROOT}/scripts/clawgym_nanobot/wandb_jsonl_relay.py" \
        --source "${WANDB_STEPS_JSONL_PATH}" \
        --state "${WANDB_JSONL_RELAY_STATE}" \
        --canonical-step-key iteration \
        --run-id "${WANDB_RUN_ID}" \
        --entity "${WANDB_ENTITY}" \
        --project "${WANDB_PROJECT}" \
        --name "${WANDB_RUN_NAME}" \
        --group "${WANDB_GROUP}" \
        --wandb-dir "${WANDB_DIR}" \
        --private-env "${CLAWGYM_PRIVATE_ENV}" \
        --done-file "${WANDB_JSONL_COALESCER_DONE}" \
        --mode "${WANDB_MODE}" \
        --poll-seconds "${WANDB_JSONL_POLL_SECONDS}" \
        >"${WANDB_JSONL_RELAY_LOG}" 2>&1 &
    WANDB_RELAY_PID="$!"
    sleep 2
    if ! kill -0 "${WANDB_RELAY_PID}" >/dev/null 2>&1; then
        wait "${WANDB_RELAY_PID}" || true
        WANDB_RELAY_PID=""
        kill -TERM "${WANDB_COALESCER_PID}" >/dev/null 2>&1 || true
        wait "${WANDB_COALESCER_PID}" >/dev/null 2>&1 || true
        WANDB_COALESCER_PID=""
        echo "ERROR: W&B JSONL relay failed to start; see ${WANDB_JSONL_RELAY_LOG}" >&2
        return 1
    fi
    echo "[proRL-nanobot] metric coalescer pid=${WANDB_COALESCER_PID}; W&B relay pid=${WANDB_RELAY_PID} run_id=${WANDB_RUN_ID}"
}

wait_for_polar() {
    python3 - "${POLAR_GATEWAY_HOST}" "${POLAR_ROLLOUT_PORT}" "${POLAR_EXPECTED_GATEWAY_NODES}" <<'PY'
import json
import sys
import time
import urllib.request

host, port, expected_nodes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
url = f"http://{host}:{port}/health"
deadline = time.time() + 300
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
        if payload.get("status") in {"ok", "healthy"} and int(payload.get("nodes", 0)) == expected_nodes:
            with urllib.request.urlopen(f"http://{host}:{port}/nodes", timeout=5) as response:
                nodes = json.load(response)
            expected_ids = {f"clawgym-rollout-{i}" for i in range(1, expected_nodes+1)}
            if isinstance(nodes, list) and {n.get("node_id") for n in nodes} == expected_ids and all(
                n.get("healthy") is True and not n.get("draining") and not n.get("circuit_open") for n in nodes
            ):
                raise SystemExit(0)
    except Exception:
        pass
    time.sleep(2)
raise SystemExit(f"timed out waiting for Polar service: {url}")
PY
}

wait_for_sglang_router() {
    echo "[proRL-polar] waiting for SGLang router ${MASTER_ADDR}:${SGLANG_ROUTER_PORT} before gateway startup (timeout=${POLAR_ROUTER_WAIT_TIMEOUT_SECONDS}s)"
    python3 - "${MASTER_ADDR}" "${SGLANG_ROUTER_PORT}" "${POLAR_ROUTER_WAIT_TIMEOUT_SECONDS}" <<'PY'
import socket
import sys
import time

host, port, timeout_seconds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + timeout_seconds
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"[proRL-polar] SGLang router is reachable at {host}:{port}")
            raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit(
    f"timed out waiting for SGLang router tcp://{host}:{port} "
    f"after {timeout_seconds}s"
)
PY
}

ray_job_status() {
    local job_id="$1"
    python3 - "${RAY_JOB_SUBMIT_ADDR}" "${RAY_DASHBOARD_PORT}" "${job_id}" <<'PY'
import sys

from ray.job_submission import JobSubmissionClient

host, port, job_id = sys.argv[1], int(sys.argv[2]), sys.argv[3]
client = JobSubmissionClient(f"http://{host}:{port}")
info = client.get_job_info(job_id)
print(info.status.value)
PY
}

wait_for_head() {
    python3 - "$MASTER_ADDR" "$MASTER_PORT" <<'PY'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
deadline = time.time() + 600
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=5):
            raise SystemExit(0)
    except OSError:
        time.sleep(5)
raise SystemExit(f"timed out waiting for Ray head at {host}:{port}")
PY
}


if [ "${RESEARCHCLAW_RENDER_ONLY:-0}" = 1 ]; then
 ray() {
  if [ "${1:-}" = job ] && [ "${2:-}" = submit ]; then
   python3 "${BUNDLE_ROOT}/runtime/capture_cli.py" "$@"
   printf "Job 'raysubmit_render_only' submitted successfully.\n"
  elif [ "${1:-}" = status ] || [ "${1:-}" = start ] || { [ "${1:-}" = job ] && [ "${2:-}" = logs ]; }; then return 0
  else return 2; fi
 }
 ray_job_status() { printf 'SUCCEEDED\n'; }
 start_ray_port_seed() { true & ray_port_seed_pid=$!; }
 start_wsclaw_exec_memory_guard() { :; }
 start_polar_rollout_service() { :; }
 start_polar_gateway() { :; }
 start_wandb_jsonl_relay() { :; }
 wait_for_polar() { :; }
 wait_for_sglang_router() { :; }
fi

RUNTIME_ENV_JSON=$(
    python3 - <<PY
import json
import os

keys = [
    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2",
    "BUNDLE_ROOT",
    "RESEARCHCLAW_JUDGE_SLOT_ROOT",
    "LOG_DIR",
    'RESEARCHCLAW_HF_RESUME_CONTRACT_MODULE',
    'RESEARCHCLAW_HF_RESUME_CONTRACT_SHA',
    'START_ROLLOUT_ID',
    'NUM_ROLLOUT',
    'RESEARCHCLAW_RESUME_EVAL_PLAN_PATH',
    'RESEARCHCLAW_RESUME_EVAL_PLAN_SHA',
    'STATE_ROOT',
    'RUN_ID',
    'GEMMA_SELFREWARD_MANIFEST',
    'CLAWGYM_SELF_REWARD_CONFIG',
    'CLAWGYM_SELF_REWARD_GROUP_SIZE',
    'RESEARCHCLAW_INITIAL_EVAL_RECOVERY_MODULE',
    'RESEARCHCLAW_FINITE_TAIL_SCHEDULE',
    'SLIME_PAD_DYNAMIC_DP_SAMPLES',
    'CLAWGYM_NATIVE_MEMORY_EFFICIENT_ATTENTION',
    'CLAWGYM_ATTENTION_PYTHON_OVERLAY',
    'CLAWGYM_SKIP_INTENTIONAL_ZERO_BATCHES',
    'CLAWGYM_INTENTIONAL_ZERO_AUTHORITY',
    'CLAWGYM_INTENTIONAL_ZERO_AUTHORITY_SHA',
    'OPENCLAW_TRAIN_START_INDEX',
    'POLAR_EVALUATOR_TIMEOUT',
    'SLIME_ASYNC_TRAINING',
    'USE_POLAR',
    'POLAR_SYNCHRONOUS_POLICY_VERSIONS',
    'POLAR_MAX_ASYNC_LEVEL',
    'POLAR_MAX_OFF_POLICY_STEPS',
    "PATH",
    "LD_LIBRARY_PATH",
    "TRAIN_NODE_HOST",
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "PYTHONPATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "TRITON_CACHE_DIR",
    "TVM_FFI_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "NCCL_IB_DISABLE",
    "NCCL_NET",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "RAY_DEDUP_LOGS",
    "PRORL_MASK_BROKEN_SKLEARN",
    "GEMMA4_SCHEDULER_TRACE_DIR",
    "GEMMA4_SCHEDULER_TRACE_DELAY_S",
    "GEMMA4_GLOBAL_ATTN_QUERY_CHUNK_SIZE",
    "GEMMA4_EAGER_ATTN_QUERY_CHUNK_SIZE",
    "GEMMA4_MEMORY_EFFICIENT_BACKEND",
    "GEMMA4_CONVERSION_MODE",
    "GEMMA4_AUTOMODEL_ACTIVATION_CHECKPOINTING",
    "GEMMA4_AUTOMODEL_ALLOW_CROSS_NODE_FSDP",
    "NRL_MEGATRON_CHECKPOINT_DIR",
    "GEMMA4_OFFICIAL_MODEL_VIEW",
    "GEMMA4_SGLANG_ALIGN_EMBED_SCALE",
    "GEMMA4_SGLANG_ALIGN_SLIDING_WINDOW",
    "GEMMA4_SGLANG_ALIGN_ZERO_BASED_POSITIONS",
    "GEMMA4_SGLANG_ALIGN_CPU_GLOBAL_ROPE",
    "GEMMA4_SGLANG_ALIGN_BF16_SOFTCAP",
    "GEMMA4_SGLANG_ALIGN_E_HEAD_DIMS",
    "GEMMA4_SGLANG_WEIGHT_CHECK_STABLE_CONSTANTS",
    "GEMMA4_SGLANG_SYNC_V_NORM",
    "GEMMA4_SGLANG_RESTORE_STATIC_BUFFERS",
    "GEMMA4_STATIC_BUFFER_AUDIT_PATH",
    "SLIME_ACTOR_NODE_IPS",
    "SLIME_GRAD_DIAGNOSTICS_STEPS",
    "SLIME_GRAD_DIAGNOSTICS_TOPK",
    "SLIME_DISABLE_MEGATRON_PATCHES",
    "SLIME_GEMMA4_WEIGHT_AUDIT_DIR",
    "SLIME_GEMMA4_BRIDGE_AUDIT_DIR",
    "SLIME_REQUIRE_GEMMA4_DENSE_BRIDGE",
    "SLIME_EXPECTED_HF_CHECKPOINT",
    "SLIME_EXPECTED_MEGATRON_BRIDGE_ROOT",
    "SLIME_EXPECTED_MEGATRON_ROOT",
    "SLIME_EXPECTED_MODEL_BRIDGE_CLASS",
    "SLIME_EXPECTED_MODEL_PROVIDER_CLASS",
    "SLIME_EXPECTED_BRIDGE_MAPPING_SHA256",
    "SLIME_SKIP_CRITIC_ONLY_ACTOR_WEIGHT_SYNC",
    "SLIME_ROLLOUT_MANAGER_ON_DRIVER_NODE",
    "PRORL_FATAL_RESUME_MARKER_DIR",
    "PRORL_CHECKPOINT_CANDIDATES_DIR",
    "PRORL_INITIAL_RESUME_CHECKPOINT",
    "PRORL_RUN_ID",
    "RUN_ID",
    "OFFLINE_CRITIC_READY_MARKER",
    "OFFLINE_CRITIC_MODEL",
    "OFFLINE_CRITIC_TARGET_DATASET",
    "OFFLINE_CRITIC_PHASE",
    "WSCLAW_EXEC_GUARD_SCOPE",
    "WSCLAW_EXEC_GUARD_GATE_FILE",
    "WSCLAW_EXEC_GUARD_HEARTBEAT_FILE",
    "WSCLAW_EXEC_GUARD_HEARTBEAT_MAX_AGE_SECONDS",
    "WSCLAW_EXEC_ADMISSION_TIMEOUT_SECONDS",
    "CLAWGYM_EXEC_DROP_PRIVILEGES",
    "CLAWGYM_EXEC_UID",
    "CLAWGYM_EXEC_GID",
    "CLAWGYM_RAW_SOURCE_INTEGRITY",
    "CLAWGYM_RAW_SOURCE_MANIFEST",
    "TRAINING_FREE_START_ROLLOUT_ID",
    "OPENCLAW_TRAIN_TASKS",
    "OPENCLAW_TRAIN_ORDER",
    "OPENCLAW_TRAIN_MAX_TURNS",
    "OPENCLAW_EVAL_TASKS",
    "OPENCLAW_EVAL_ORDER",
    "OPENCLAW_EVAL_MAX_TURNS",
    "OPENCLAW_EVAL_N_SAMPLES",
    "CLAWGYM_EVAL_NUM_PROMPTS",
    "CLAWGYM_EVAL_RESULTS_JSON",
    "OPENCLAW_EVAL_NUM_WORKERS",
    "OPENCLAW_CONTEXT_WINDOW",
    "OPENCLAW_MAX_TOKENS",
    "NANOBOT_ROOT",
    "NANOBOT_RL_RUNNER",
    "NANOBOT_IMPORT_LOCAL_PACKAGE",
    "NANOBOT_EXTRA_PYTHONPATH",
    "NANOBOT_RL_TOKENIZER_PATH",
    "NANOBOT_RL_DISABLED_TOOLS",
    "NANOBOT_CONDA_ROOT",
    "NANOBOT_CONDA_ENV_NAME",
    "NANOBOT_EVAL_NUM_WORKERS",
    "CLAWGYM_TASK_PROMPT_MODE",
    "TOOL_CALL_PARSER",
    "REASONING_PARSER",
    "CLAWGYM_NANOBOT_EVAL_TURN_TYPE",
    "CLAWGYM_NANOBOT_TRAIN_TURN_TYPE",
    "CLAWGYM_NANOBOT_MAX_TURNS_PER_SID",
    "CLAWGYM_NANOBOT_REQUIRE_EXACT_TOKENS",
    "CLAWGYM_NANOBOT_USE_UPSTREAM_CHAT_COMPLETIONS",
    "CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL",
    "CLAWGYM_NANOBOT_RESEARCHCLAW_COMPAT",
    "CLAWGYM_NANOBOT_ENABLE_REALIGN",
    "CLAWGYM_NANOBOT_PRUNE_EMPTY_RETRIES",
    "CLAWGYM_ALLOW_MULTI_TOOL_CALLS",
    "NANOBOT_MICROCOMPACT_ENABLED",
    "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION",
    "CLAWGYM_NANOBOT_ROUTING_KEY",
    "CLAWGYM_SGLANG_CONTEXT_OVERFLOW_RETRY",
    "CLAWGYM_SGLANG_CONTEXT_GUARD_TOKENS",
    "NANOBOT_STRIP_REASONING_CONTEXT",
    "NANOBOT_WSCLAW_FALLBACK_PARITY",
    "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY",
    "CLAWGYM_NANOBOT_FORK_THRESHOLD_TOKENS",
    "NANOBOT_RL_DISABLE_CONSOLIDATION",
    "NANOBOT_RL_FAST_ARCHIVE",
    "NANOBOT_ENABLE_THINKING",
    "NANOBOT_APPEND_NO_THINK",
    "NANOBOT_SHOW_REASONING",
    "NANOBOT_WEB_ENABLE",
    "NANOBOT_TURN_TIMEOUT",
    "NANOBOT_TRAIN_TURN_TIMEOUT",
    "NANOBOT_EVAL_TURN_TIMEOUT",
    "NANOBOT_TASK_TIMEOUT",
    "POLAR_INFERENCE_HTTP_TIMEOUT_SECONDS",
    "POLAR_GATEWAY_INFERENCE_HEALTH_TIMEOUT_SECONDS",
    "POLAR_GATEWAY_INFERENCE_UNHEALTHY_STREAK",
    "NANOBOT_REWARD_TIMEOUT",
    "NANOBOT_RL_LLM_TIMEOUT_S",
    "NANOBOT_RL_OPENAI_COMPAT_TIMEOUT_S",
    "NANOBOT_RL_NO_MAX_TOKENS",
    "NANOBOT_KEEP_WORKDIR",
    "NANOBOT_NO_WEB_CLI_TIMEOUT",
    "NANOBOT_MAX_TOOL_ITERATIONS",
    "NANOBOT_TIMEZONE",
    "NANOBOT_EXEC_ENABLE",
    "NANOBOT_EXEC_TIMEOUT",
    "NANOBOT_REASONING_EFFORT",
    "NANOBOT_SUBPROCESS_CAPTURE_TAIL_BYTES",
    "CLAWGYM_DROP_TRAIN_NANOBOT_LOGS",
    "CLAWGYM_NANOBOT_SAMPLE_MODE",
    "CLAWGYM_NANOBOT_TRAIN_SAMPLE_MODE",
    "CLAWGYM_NANOBOT_EVAL_SAMPLE_MODE",
    "CLAWGYM_NANOBOT_EVAL_SESSION_PREFIX",
    "CLAWGYM_NANOBOT_EVAL_UNIQUE_SESSION_IDS",
    "CLAWGYM_DISTRIBUTED_ROLLOUT_NANOBOT",
    "NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE",
    "NANOBOT_ROLLOUT_GROUP_AFFINITY",
    "NANOBOT_ROLLOUT_NODE_API_PORT_OFFSET",
    "NANOBOT_ROLLOUT_WORKDIR_BASE",
    "CLAWGYM_NANOBOT_MAX_WORKERS",
    "CLAWGYM_SELF_REWARD_CONFIG",
    "CLAWGYM_OBJECT_CHECKPOINT_ROOT",
    "CLAWGYM_OBJECT_CHECKPOINT_REQUIRED",
    "CLAWGYM_LOCAL_DELETE_FUNCTION_PATH",
    "NANOBOT_LOG_DIR",
    "NANOBOT_WORKDIR",
    "RESEARCHCLAW_KEEP_TRAIN_NANOBOT_LOGS",
    "CLAWGYM_KEEP_LATEST_EVAL_LOGS",
    "JUDGE_NANOBOT_LOG_DIR",
    "LATEST_EVAL_LOG_STATE_DIR",
    "CLAWGYM_HF_CHECKPOINT_ROOT",
    "CLAWGYM_HF_MILESTONE_PIN_ROOT",
    "CLAWGYM_HF_MILESTONE_PIN_STEPS",
    "CLAWGYM_HF_PIN_QUESTION_BATCH",
    "CLAWGYM_HF_PIN_CANDIDATES_PER_QUESTION",
    "CLAWGYM_HF_PIN_TOTAL_QUESTIONS",
    "CLAWGYM_EVAL_SNAPSHOT_ROOT",
    "CLAWGYM_HF_CHECKPOINT_MIN_EVAL_COVERAGE",
    "CLAWGYM_BEST_EVAL_DATASET",
    "CLAWGYM_BEST_EVAL_METRIC",
    "RESEARCHCLAW_SCORE_MODE",
    "RESEARCHCLAW_LLMJUDGE_BACKEND",
    "RESEARCHCLAW_CODE_WEIGHT",
    "RESEARCHCLAW_LLMJUDGE_WEIGHT",
    "RESEARCHCLAW_JUDGE_BASE_URL",
    "RESEARCHCLAW_JUDGE_BASE_URLS",
    "RESEARCHCLAW_JUDGE_API_KEY",
    "RESEARCHCLAW_JUDGE_API_KEYS",
    "RESEARCHCLAW_JUDGE_MODEL",
    "RESEARCHCLAW_JUDGE_TEMPERATURE",
    "RESEARCHCLAW_JUDGE_MAX_TOKENS",
    "RESEARCHCLAW_JUDGE_TIMEOUT",
    "RESEARCHCLAW_JUDGE_MAX_RETRIES",
    "RESEARCHCLAW_JUDGE_RETRY_BACKOFF_SEC",
    "RESEARCHCLAW_JUDGE_INCLUDE_GT_ANSWER",
    "RESEARCHCLAW_AGENTIC_JUDGE_TIMEOUT",
    "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOKENS",
    "RESEARCHCLAW_AGENTIC_JUDGE_CONTEXT_WINDOW",
    "RESEARCHCLAW_AGENTIC_JUDGE_MAX_TOOL_ITERATIONS",
    "CLAWGYM_PRIVATE_ENV",
    "USE_WANDB",
    "WANDB_MODE",
    "WANDB_ENTITY",
    "WANDB_TEAM",
    "WANDB_PROJECT",
    "WANDB_GROUP",
    "WANDB_DIR",
    "WANDB_CONSOLE",
    "WANDB_HOST",
    "WANDB_RUN_ID",
    "WANDB_INIT_TIMEOUT",
    "WANDB_NO_PROXY",
    "WANDB_USE_PROXY",
    "SLIME_METRICS_JSONL_PATH",
    "SLIME_METRICS_JSONL_FSYNC",
    "NO_PROXY",
    "no_proxy",
]
keys.extend(
    key
    for key in os.environ
    if not key.endswith("_API_KEY")
    and (
        key == "CLAWGYM_REWARD_SOURCE"
        or key == "CLAWGYM_EVAL_USER_AGENT_REWARD"
        or key == "CLAWGYM_SYN_TASK_JSONL"
        or key.startswith("CLAWGYM_USER_AGENT_")
        or key.startswith("CLAWGYM_DEFERRED_UA_")
        or key.startswith("OPSD_")
        or key.startswith("USER_AGENT_VERIFIER_")
    )
)
print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if k in os.environ}}))
PY
)


export RESEARCHCLAW_RAY_ENV_FILE="${STATE_ROOT}/private/ray-runtime-env-attempt${RESEARCHCLAW_ATTEMPT_ID:-1}-rank${NODE_RANK}.json"
RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}" python3 - <<'PRIVATEENV'
from pathlib import Path
import os,json
p=Path(os.environ['RESEARCHCLAW_RAY_ENV_FILE']);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w')as f:json.dump(json.loads(os.environ['RUNTIME_ENV_JSON']),f)
PRIVATEENV

TRAIN_ENV_VARS_JSON=$(
    python3 - <<'PY'
import json
import os

env_vars = {}
keys = [
    "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2",
    "BUNDLE_ROOT",
    "RESEARCHCLAW_JUDGE_SLOT_ROOT",
    "LOG_DIR",
    'RESEARCHCLAW_HF_RESUME_CONTRACT_MODULE',
    'RESEARCHCLAW_HF_RESUME_CONTRACT_SHA',
    'START_ROLLOUT_ID',
    'NUM_ROLLOUT',
    'RESEARCHCLAW_RESUME_EVAL_PLAN_PATH',
    'RESEARCHCLAW_RESUME_EVAL_PLAN_SHA',
    "PYTORCH_CUDA_ALLOC_CONF",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "GEMMA4_CONVERSION_MODE",
    "GEMMA4_AUTOMODEL_ACTIVATION_CHECKPOINTING",
    "NRL_MEGATRON_CHECKPOINT_DIR",
    "NEMO_MEGATRON_OVERLAP_GRAD_REDUCE",
    "NEMO_MEGATRON_PROCESS_GROUP_TIMEOUT_MINUTES",
    "SLIME_DISABLE_MEGATRON_PATCHES",
    "SLIME_EMPTY_CACHE_BETWEEN_TRAIN_MICROBATCHES",
]
if os.environ.get("TRAIN_BACKEND") != "nemo_megatron":
    keys.extend(
        [
            "SLIME_GEMMA4_WEIGHT_AUDIT_DIR",
            "SLIME_GEMMA4_BRIDGE_AUDIT_DIR",
            "SLIME_REQUIRE_GEMMA4_DENSE_BRIDGE",
            "SLIME_EXPECTED_HF_CHECKPOINT",
            "SLIME_EXPECTED_MEGATRON_BRIDGE_ROOT",
            "SLIME_EXPECTED_MEGATRON_ROOT",
            "SLIME_EXPECTED_MODEL_BRIDGE_CLASS",
            "SLIME_EXPECTED_MODEL_PROVIDER_CLASS",
            "SLIME_EXPECTED_BRIDGE_MAPPING_SHA256",
        ]
    )
for key in keys:
    value = os.environ.get(key)
    if value:
        env_vars[key] = value
for target, source in (
    ("NCCL_IB_DISABLE", "TRAIN_NCCL_IB_DISABLE"),
    ("NCCL_NET", "TRAIN_NCCL_NET"),
):
    value = os.environ.get(source) or os.environ.get(target)
    if value:
        env_vars[target] = value
print(json.dumps(env_vars))
PY
)

polar_primary_gateway_rank="${POLAR_GATEWAY_FIRST_NODE_RANK}"
if [ "${POLAR_GATEWAY_ON_DRIVER}" = "1" ]; then
    polar_primary_gateway_rank=0
fi
polar_last_gateway_rank=$((polar_primary_gateway_rank + POLAR_EXPECTED_GATEWAY_NODES - 1))
polar_gateway_start_deferred=0

if [ "${USE_POLAR}" = "1" ] && \
   [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] && \
   [ "${NODE_RANK}" -ge "${polar_primary_gateway_rank}" ] && \
   [ "${NODE_RANK}" -le "${polar_last_gateway_rank}" ]; then
    cleanup_stale_local_sglang_listener
    # The rollout server (18080) is shared by all gateway ranks.  Running the
    # stale-service sweep concurrently on every rollout rank races with the
    # primary rank's startup: a secondary rank can observe and kill the fresh
    # rollout server before its node registers, leaving the driver waiting for
    # a gateway set that can never become complete.  Only the primary rank may
    # sweep the shared Polar listeners; secondary ranks only clear their local
    # SGLang listener above.
    if [ "${NODE_RANK}" -eq "${polar_primary_gateway_rank}" ]; then
        cleanup_stale_polar_services
    fi
    prepare_polar_gateway_session_tmpdir
fi

start_wsclaw_exec_memory_guard

if [ "${USE_POLAR}" = "1" ] && [ "${NODE_RANK}" -eq "${polar_primary_gateway_rank}" ]; then
    start_polar_rollout_service
    polar_gateway_start_deferred=1
elif [ "${USE_POLAR}" = "1" ] && \
     [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] && \
     [ "${NODE_RANK}" -gt "${polar_primary_gateway_rank}" ] && \
     [ "${NODE_RANK}" -le "${polar_last_gateway_rank}" ]; then
    polar_gateway_start_deferred=1
fi

if [ "${NODE_RANK}" = "0" ]; then
    rm -f "${STOP_FILE}"
    cleanup_stale_training_drivers
    cleanup_ray
    cleanup_stale_gpu_processes
    # Reuse the verified existing Ray head; do not stop or start workers.
    ray status --address "${MASTER_ADDR}:${MASTER_PORT}" >/dev/null
    touch "${LOG_DIR}/ray-head-ready"
    start_wandb_jsonl_relay

    TRAIN_LOG="${LOG_DIR}/rank0-train.log"
    EXTRA_ARGS=()
    if [ "${SLIME_USE_FAULT_TOLERANCE:-0}" = "1" ]; then
        for value in \
            "${SLIME_ROLLOUT_HEALTH_CHECK_INTERVAL:-30}" \
            "${SLIME_ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}" \
            "${SLIME_ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"; do
            [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
                echo "ERROR: rollout health-check timings must be non-negative numbers" >&2
                exit 1
            }
        done
        EXTRA_ARGS+=(
            --use-fault-tolerance
            --rollout-health-check-interval "${SLIME_ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
            --rollout-health-check-timeout "${SLIME_ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}"
            --rollout-health-check-first-wait "${SLIME_ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}"
        )
    fi
    if [ -n "${DISTRIBUTED_TIMEOUT_MINUTES:-}" ]; then
        [[ "${DISTRIBUTED_TIMEOUT_MINUTES}" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: DISTRIBUTED_TIMEOUT_MINUTES must be a positive integer" >&2
            exit 1
        }
        EXTRA_ARGS+=(--distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}")
    fi
    case "${SLIME_BALANCE_DATA:-0}" in
        0) ;;
        1) EXTRA_ARGS+=(--balance-data) ;;
        *)
            echo "ERROR: SLIME_BALANCE_DATA must be 0 or 1" >&2
            exit 1
            ;;
    esac
    if [ -n "${GRPO_ITERATIONS}" ]; then
        EXTRA_ARGS+=(--grpo-iterations "${GRPO_ITERATIONS}")
    fi
    if [ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]; then
        SAVE_DEBUG_ROLLOUT_DATA="$(realpath -m -- "${SAVE_DEBUG_ROLLOUT_DATA}")"
        EXTRA_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
    fi
    if [ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
        LOAD_DEBUG_ROLLOUT_DATA="$(realpath -m -- "${LOAD_DEBUG_ROLLOUT_DATA}")"
        EXTRA_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
    fi
    if [ "${DISABLE_GRADIENT_ACCUMULATION_FUSION:-0}" = "1" ]; then
        EXTRA_ARGS+=(--no-gradient-accumulation-fusion)
    fi
    if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then
        EXTRA_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
    fi
    if [ -n "${QWEN_GDN_BACKEND:-}" ]; then
        case "${QWEN_GDN_BACKEND}" in
            fla|flashqla) ;;
            *) echo "ERROR: QWEN_GDN_BACKEND must be fla or flashqla" >&2; exit 1 ;;
        esac
        EXTRA_ARGS+=(--qwen-gdn-backend "${QWEN_GDN_BACKEND}")
    fi
    if [ -n "${EVAL_MICRO_BATCH_SIZE:-}" ] || [ -n "${EVAL_GLOBAL_BATCH_SIZE:-}" ]; then
        [[ "${EVAL_MICRO_BATCH_SIZE:-}" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: EVAL_MICRO_BATCH_SIZE must be a positive integer" >&2
            exit 1
        }
        [[ "${EVAL_GLOBAL_BATCH_SIZE:-}" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: EVAL_GLOBAL_BATCH_SIZE must be a positive integer" >&2
            exit 1
        }
        EXTRA_ARGS+=(
            --eval-micro-batch-size "${EVAL_MICRO_BATCH_SIZE}"
            --eval-global-batch-size "${EVAL_GLOBAL_BATCH_SIZE}"
        )
    fi
    if [ -n "${SGLANG_CONFIG_PATH:-}" ]; then
        if [ ! -s "${SGLANG_CONFIG_PATH}" ]; then
            echo "ERROR: SGLANG_CONFIG_PATH does not exist or is empty: ${SGLANG_CONFIG_PATH}" >&2
            exit 1
        fi
        EXTRA_ARGS+=(--sglang-config "${SGLANG_CONFIG_PATH}")
    fi
    if [ -n "${SGLANG_MAX_RUNNING_REQUESTS}" ]; then
        EXTRA_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")
    fi
    if [ -n "${SGLANG_CHUNKED_PREFILL_SIZE:-}" ]; then
        EXTRA_ARGS+=(--sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}")
    fi
    if [ -n "${SGLANG_CUDA_GRAPH_MAX_BS:-}" ]; then
        EXTRA_ARGS+=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}")
    fi
    if [ -n "${SGLANG_MAMBA_RADIX_CACHE_STRATEGY:-}" ]; then
        EXTRA_ARGS+=(--sglang-mamba-radix-cache-strategy "${SGLANG_MAMBA_RADIX_CACHE_STRATEGY}")
    fi
    if [ -n "${SGLANG_PAGE_SIZE:-}" ]; then
        EXTRA_ARGS+=(--sglang-page-size "${SGLANG_PAGE_SIZE}")
    fi
    if [ -n "${SGLANG_TOKENIZER_WORKER_NUM:-}" ]; then
        EXTRA_ARGS+=(--sglang-tokenizer-worker-num "${SGLANG_TOKENIZER_WORKER_NUM}")
    fi
    if [ "${SGLANG_ENABLE_METRICS:-0}" = 1 ]; then
        EXTRA_ARGS+=(--sglang-enable-metrics)
    fi
    if [ "${DO_EVAL}" = "1" ] && [ "${SKIP_INITIAL_EVAL:-0}" = "1" ]; then
        EXTRA_ARGS+=(--skip-eval-before-train)
    fi
    if [ "${NUM_ROLLOUT}" = "0" ] && [ -n "${EVAL_ONLY_ROLLOUT_ID:-}" ]; then
        if ! [[ "${EVAL_ONLY_ROLLOUT_ID}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: EVAL_ONLY_ROLLOUT_ID must be a non-negative integer" >&2
            exit 1
        fi
        EXTRA_ARGS+=(--eval-only-rollout-id "${EVAL_ONLY_ROLLOUT_ID}")
        # Eval backfills only consume eval-prompt-data.  Do not restore or
        # advance the training data-source cursor from the loaded checkpoint.
        EXTRA_ARGS+=(--disable-rollout-global-dataset)
    fi
    if [ "${SGLANG_DISABLE_CUDA_GRAPH:-0}" = "1" ]; then
        EXTRA_ARGS+=(--sglang-disable-cuda-graph)
    fi
    if [ "${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}" = "1" ]; then
        EXTRA_ARGS+=(--sglang-disable-piecewise-cuda-graph)
    fi
    if [ -n "${SLIME_SEED:-}" ]; then
        [[ "${SLIME_SEED}" =~ ^[0-9]+$ ]] || {
            echo "ERROR: SLIME_SEED must be a non-negative integer" >&2
            exit 1
        }
        EXTRA_ARGS+=(--seed "${SLIME_SEED}")
    fi
    if [ -n "${ROLLOUT_SEED:-}" ]; then
        [[ "${ROLLOUT_SEED}" =~ ^[0-9]+$ ]] || {
            echo "ERROR: ROLLOUT_SEED must be a non-negative integer" >&2
            exit 1
        }
        EXTRA_ARGS+=(--rollout-seed "${ROLLOUT_SEED}")
    fi
    if [ -n "${SGLANG_GRAMMAR_BACKEND:-}" ]; then
        EXTRA_ARGS+=(--sglang-grammar-backend "${SGLANG_GRAMMAR_BACKEND}")
    fi
    if [ -n "${SGLANG_WATCHDOG_TIMEOUT:-}" ]; then
        EXTRA_ARGS+=(--sglang-watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT}")
    fi
    if [ -n "${LOSS_MASK_TYPE}" ]; then
        EXTRA_ARGS+=(--loss-mask-type "${LOSS_MASK_TYPE}")
    fi
    if [ -n "${ROLLOUT_STOP_TOKEN_IDS}" ]; then
        read -r -a rollout_stop_token_ids <<<"${ROLLOUT_STOP_TOKEN_IDS}"
        [ "${#rollout_stop_token_ids[@]}" -gt 0 ] || {
            echo "ERROR: ROLLOUT_STOP_TOKEN_IDS must contain at least one token id" >&2
            exit 1
        }
        for token_id in "${rollout_stop_token_ids[@]}"; do
            [[ "${token_id}" =~ ^[0-9]+$ ]] || {
                echo "ERROR: invalid token id in ROLLOUT_STOP_TOKEN_IDS: ${token_id}" >&2
                exit 1
            }
        done
        EXTRA_ARGS+=(--rollout-stop-token-ids "${rollout_stop_token_ids[@]}")
    fi
    if [ "${CHECK_WEIGHT_UPDATE_EQUAL}" = "1" ]; then
        EXTRA_ARGS+=(--check-weight-update-equal)
        if [ -n "${CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES}" ]; then
            read -r -a weight_check_prefixes <<< "${CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES}"
            EXTRA_ARGS+=(--check-weight-update-include-prefixes "${weight_check_prefixes[@]}")
        fi
    elif [ -n "${CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES}" ]; then
        echo "ERROR: CHECK_WEIGHT_UPDATE_INCLUDE_PREFIXES requires CHECK_WEIGHT_UPDATE_EQUAL=1" >&2
        exit 1
    fi
    if [ "${USE_TIS}" = "1" ]; then
        EXTRA_ARGS+=(--use-tis --tis-clip "${TIS_CLIP}" --tis-clip-low "${TIS_CLIP_LOW}")
        if [ -n "${CUSTOM_TIS_FUNCTION_PATH}" ]; then
            EXTRA_ARGS+=(--custom-tis-function-path "${CUSTOM_TIS_FUNCTION_PATH}")
        fi
    fi
    if [ "${PARTIAL_ROLLOUT}" = "1" ]; then
        EXTRA_ARGS+=(--partial-rollout)
        if [ "${USE_POLAR}" = "1" ]; then
            echo "[proRL-nanobot] partial_rollout_mode=polar-persistent-inflight-buffer (not stock token-level abort/resume)"
        fi
    fi
    if [ -n "${LR_DECAY_ITERS:-}" ]; then
        EXTRA_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
    elif [ "${NUM_ROLLOUT}" = "0" ]; then
        EXTRA_ARGS+=(--lr-decay-iters 1)
    fi
    case "${USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" in
        0) ;;
        1) EXTRA_ARGS+=(--use-checkpoint-opt-param-scheduler) ;;
        *)
            echo "ERROR: USE_CHECKPOINT_OPT_PARAM_SCHEDULER must be 0 or 1" >&2
            exit 1
            ;;
    esac
    CKPT_ARGS=()
    if [ -n "${LOAD_CKPT}" ]; then
        CKPT_ARGS+=(--load "${LOAD_CKPT}")
    fi
    case "${NO_LOAD_OPTIM}" in
        0) ;;
        1) CKPT_ARGS+=(--no-load-optim --no-load-rng) ;;
        *) echo "ERROR: NO_LOAD_OPTIM must be 0 or 1" >&2; exit 1 ;;
    esac
    SAVE_ARGS=()
    if [ "${TRAIN_BACKEND}" = "nemo_megatron" ] && [ "${SAVE_LATEST_BEST_CKPTS}" = "1" ]; then
        # NeMo's actor only exports --save-hf; it intentionally does not write
        # a Megatron torch_dist checkpoint.  Slime's global rollout data source
        # still requires args.save for its small cursor/state file, however.
        # Keep that state outside candidates so HF retention cannot prune it.
        mkdir -p \
            "${SAVE_DIR}/candidates" \
            "${SAVE_DIR}/state" \
            "${CLAWGYM_EVAL_SNAPSHOT_ROOT}/candidates"
        SAVE_ARGS+=(
            --save "${SAVE_DIR}/state"
            --save-hf "${SAVE_HF_FORMAT}"
        )
    elif [ "${TRAIN_BACKEND}" = "megatron" ] && \
         [ "${SAVE_LATEST_BEST_CKPTS}" = "1" ] && \
         [ "${SAVE_CKPTS}" != "1" ]; then
        # Native Slime can export HF weights directly from the Megatron actor.
        # Keep only those bounded best/last candidates when optimizer-state
        # checkpoints are explicitly disabled.
        mkdir -p \
            "${SAVE_DIR}/candidates" \
            "${CLAWGYM_EVAL_SNAPSHOT_ROOT}/candidates"
        SAVE_ARGS+=(--save-hf "${SAVE_HF_FORMAT}")
    elif [ "${SAVE_CKPTS}" = "1" ] || [ "${SAVE_LATEST_BEST_CKPTS}" = "1" ]; then
        SAVE_ARGS+=(--save "${SAVE_DIR}")
    fi
    if { [ "${SAVE_CKPTS}" = "1" ] || [ "${SAVE_LATEST_BEST_CKPTS}" = "1" ]; } && [ -n "${SAVE_INTERVAL}" ]; then
        SAVE_ARGS+=(--save-interval "${SAVE_INTERVAL}")
    fi
    if [ "${NO_SAVE_OPTIM}" = "1" ]; then
        SAVE_ARGS+=(--no-save-optim)
    fi
    if [ -n "${START_ROLLOUT_ID}" ]; then
        [[ "${START_ROLLOUT_ID}" =~ ^[0-9]+$ ]] || {
            echo "ERROR: START_ROLLOUT_ID must be a non-negative integer" >&2
            exit 1
        }
        EXTRA_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
    fi
    WEIGHT_SYNC_ARGS=(
        --update-weight-mode "${UPDATE_WEIGHT_MODE}"
        --update-weight-transport "${UPDATE_WEIGHT_TRANSPORT}"
    )
    if [ -n "${UPDATE_WEIGHT_NCCL_NET:-}" ]; then
        case "${UPDATE_WEIGHT_NCCL_NET}" in IB|Socket) ;; *)
            echo "ERROR: UPDATE_WEIGHT_NCCL_NET must be IB or Socket" >&2
            exit 1
        esac
        WEIGHT_SYNC_ARGS+=(--update-weight-nccl-net "${UPDATE_WEIGHT_NCCL_NET}")
    fi
    if [ "${UPDATE_WEIGHT_TRANSPORT}" = "disk" ]; then
        mkdir -p "${UPDATE_WEIGHT_DISK_DIR}"
        WEIGHT_SYNC_ARGS+=(--update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}")
        if [ -n "${UPDATE_WEIGHT_DISK_CLEANUP_PATH:-}" ]; then
            WEIGHT_SYNC_ARGS+=(
                --custom-update-weight-disk-cleanup-path
                "${UPDATE_WEIGHT_DISK_CLEANUP_PATH}"
            )
        fi
    fi
    WANDB_ARGS=()
    if [ "${USE_WANDB:-0}" = "1" ] && \
       [ "${WANDB_MODE}" != "disabled" ] && \
       [ "${WANDB_JSONL_RELAY_ACTIVE}" != "1" ]; then
        WANDB_ARGS+=(
            --use-wandb
            --wandb-mode "${WANDB_MODE}"
            --wandb-dir "${WANDB_DIR}"
            --wandb-project "${WANDB_PROJECT}"
            --wandb-group "${WANDB_GROUP}"
            --disable-wandb-random-suffix
            --wandb-always-use-train-step
        )
        if [ -n "${WANDB_ENTITY:-}" ]; then
            WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
        fi
        if [ -n "${WANDB_HOST:-}" ]; then
            WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
        fi
        if [ -n "${WANDB_RUN_ID:-}" ]; then
            WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
        fi
    fi
    EVAL_ARGS=()
    if [ "${DO_EVAL}" = "1" ]; then
        DEFAULT_EVAL_FUNCTION_PATH="clawgym_nanobot.eval_rollout.generate_rollout"
        if [ "${USE_POLAR}" = "1" ]; then
            DEFAULT_EVAL_FUNCTION_PATH="slime_bridge.rollout.generate_rollout_polar_async"
        fi
        EVAL_ARGS+=(
            --eval-interval "${OPENCLAW_EVAL_INTERVAL}"
            --eval-function-path "${EVAL_FUNCTION_PATH:-${DEFAULT_EVAL_FUNCTION_PATH}}"
            --custom-eval-rollout-log-function-path "${CUSTOM_EVAL_ROLLOUT_LOG_FUNCTION_PATH}"
            --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
            --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
            --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
            --eval-reward-key "${EVAL_REWARD_KEY:-score}"
        )
        if [ -n "${EVAL_CONFIG_PATH:-}" ]; then
            test -s "${EVAL_CONFIG_PATH}"
            EVAL_ARGS+=(--eval-config "${EVAL_CONFIG_PATH}")
        else
            EVAL_ARGS+=(--eval-prompt-data "${EVAL_DATASET_NAME}" "${EVAL_PROMPT_DATA}")
        fi
        if [ -n "${EVAL_TEMPERATURE:-}" ]; then
            EVAL_ARGS+=(--eval-temperature "${EVAL_TEMPERATURE}")
        fi
        if [ -n "${EVAL_TOP_P:-}" ]; then
            EVAL_ARGS+=(--eval-top-p "${EVAL_TOP_P}")
        fi
        if [ -n "${EVAL_TOP_K:-}" ]; then
            EVAL_ARGS+=(--eval-top-k "${EVAL_TOP_K}")
        fi
    fi
    TRAIN_COMPAT_ARGS=()
    if [ "${ENABLE_OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
    fi
    if [ "${ENABLE_PRECISION_AWARE_OPTIMIZER}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(--use-precision-aware-optimizer)
    fi
    if [ "${ENABLE_CALCULATE_PER_TOKEN_LOSS}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(--calculate-per-token-loss)
    fi
    if [ -n "${RECOMPUTE_GRANULARITY}" ]; then
        TRAIN_COMPAT_ARGS+=(
            --recompute-granularity "${RECOMPUTE_GRANULARITY}"
            --recompute-method "${RECOMPUTE_METHOD}"
            --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
        )
    fi
    if [ "${ENABLE_KL_LOSS}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF}" --kl-loss-type "${KL_LOSS_TYPE}")
    fi
    TRAIN_COMPAT_ARGS+=(
        --kl-coef "${KL_COEF}"
        --entropy-coef "${ENTROPY_COEF}"
        --eps-clip "${EPS_CLIP}"
        --eps-clip-high "${EPS_CLIP_HIGH}"
        --value-clip "${VALUE_CLIP}"
        --gamma "${GAMMA}"
        --lambd "${LAMBDA}"
    )
    if [ "${ADVANTAGE_ESTIMATOR:-grpo}" = "ppo" ]; then
        TRAIN_COMPAT_ARGS+=(
            --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
            --critic-train-epochs "${CRITIC_TRAIN_EPOCHS:-1}"
        )
    fi
    if [ "${NORMALIZE_ADVANTAGES}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(--normalize-advantages)
    fi
    if [ "${USE_OPD}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(--use-opd --opd-type "${OPD_TYPE}" --opd-kl-coef "${OPD_KL_COEF}")
    fi
    if [ "${USE_OPSD}" = "1" ]; then
        TRAIN_COMPAT_ARGS+=(
            --use-opsd
            --opsd-teacher-mode "${OPSD_TEACHER_MODE}"
            --opsd-beta "${OPSD_BETA}"
            --opsd-temperature "${OPSD_TEMPERATURE}"
            --opsd-token-clip "${OPSD_TOKEN_CLIP}"
            --opsd-logits-chunk-size "${OPSD_LOGITS_CHUNK_SIZE}"
            --loss-type custom_loss
            --custom-loss-function-path clawgym_nanobot.opsd_loss.full_vocab_opsd_loss
            --disable-compute-advantages-and-returns
        )
    fi
    MEGATRON_CONFIG_ARGS=()
    if [ -n "${MEGATRON_CONFIG_PATH}" ]; then
        if [ ! -s "${MEGATRON_CONFIG_PATH}" ]; then
            echo "ERROR: MEGATRON_CONFIG_PATH does not exist or is empty: ${MEGATRON_CONFIG_PATH}" >&2
            exit 1
        fi
        MEGATRON_CONFIG_ARGS+=(--megatron-config-path "${MEGATRON_CONFIG_PATH}")
    fi
    CUSTOM_MODEL_PROVIDER_ARGS=()
    if [ -n "${CUSTOM_MODEL_PROVIDER_PATH}" ]; then
        CUSTOM_MODEL_PROVIDER_ARGS+=(--custom-model-provider-path "${CUSTOM_MODEL_PROVIDER_PATH}")
    fi
    TRAIN_MODE_ARGS=()
    if [ "${DEBUG_ROLLOUT_ONLY}" = "1" ]; then
        TRAIN_MODE_ARGS+=(--debug-rollout-only)
    fi
    if [ "${DEBUG_TRAIN_ONLY}" = "1" ]; then
        TRAIN_MODE_ARGS+=(--debug-train-only)
    fi
    if [ "${OFFLINE_CRITIC_PRETRAIN}" = "1" ]; then
        [ "${DEBUG_TRAIN_ONLY}" = "1" ] || { echo "ERROR: offline critic pretrain requires DEBUG_TRAIN_ONLY=1" >&2; exit 1; }
        [ "${ADVANTAGE_ESTIMATOR:-}" = "ppo" ] || { echo "ERROR: offline critic pretrain requires ADVANTAGE_ESTIMATOR=ppo" >&2; exit 1; }
        [ "${ROLLOUT_NUM_GPUS}" = "0" ] || { echo "ERROR: offline critic pretrain requires ROLLOUT_NUM_GPUS=0" >&2; exit 1; }
        [ -s "${OFFLINE_CRITIC_CONFIG_PATH}" ] || { echo "ERROR: OFFLINE_CRITIC_CONFIG_PATH is missing" >&2; exit 1; }
        TRAIN_MODE_ARGS+=(--critic-only)
    fi
    case "${CRITIC_EXPORT_ONLY}" in
        0) ;;
        1)
            [[ "${CRITIC_EXPORT_ROLLOUT_ID}" =~ ^[0-9]+$ ]] || {
                echo "ERROR: CRITIC_EXPORT_ROLLOUT_ID must be a non-negative integer" >&2
                exit 1
            }
            TRAIN_MODE_ARGS+=(
                --critic-export-only
                --critic-export-rollout-id "${CRITIC_EXPORT_ROLLOUT_ID}"
            )
            ;;
        *) echo "ERROR: CRITIC_EXPORT_ONLY must be 0 or 1" >&2; exit 1 ;;
    esac
    if [ "${COLOCATE}" = "1" ]; then
        TRAIN_MODE_ARGS+=(--colocate)
    fi
    TRAIN_BACKEND_ARGS=(--train-backend "${TRAIN_BACKEND}")
    if [ "${TRAIN_BACKEND}" = automodel ] || [ "${TRAIN_BACKEND}" = nemo_megatron ]; then
        TRAIN_BACKEND_ARGS+=(
            --automodel-config "${AUTOMODEL_CONFIG}"
            --automodel-context-parallel-size "${CP_SIZE}"
        )
        # The model, optimizer, forward/backward and HF export remain NeMo-RL
        # implementations. Slime only supplies rollout data and refit transport.
        CUSTOM_MODEL_PROVIDER_ARGS=()
        MEGATRON_CONFIG_ARGS=()
    fi
    SGLANG_PARSER_ARGS=()
    SGLANG_PARSER_ARGS+=(
        --sglang-log-level "${SGLANG_LOG_LEVEL:-WARNING}"
        --sglang-log-level-http "${SGLANG_LOG_LEVEL_HTTP:-WARNING}"
    )
    if [ -n "${REASONING_PARSER:-}" ]; then
        SGLANG_PARSER_ARGS+=(--sglang-reasoning-parser "${REASONING_PARSER}")
    fi
    if [ -n "${TOOL_CALL_PARSER:-}" ]; then
        SGLANG_PARSER_ARGS+=(--sglang-tool-call-parser "${TOOL_CALL_PARSER}")
    fi

    TRAIN_ENTRYPOINT="${SLIME_ROOT}/train.py"
    if [ "${TRAINING_FREE_EVAL_ONLY}" = "1" ]; then
        [ "${DEBUG_ROLLOUT_ONLY}" = "1" ] && [ "${NUM_ROLLOUT}" = "0" ] && [ "${DO_EVAL}" = "1" ] || {
            echo "ERROR: TRAINING_FREE_EVAL_ONLY requires DEBUG_ROLLOUT_ONLY=1, NUM_ROLLOUT=0, and DO_EVAL=1" >&2
            exit 1
        }
        TRAIN_ENTRYPOINT="${PROJECT_ROOT}/scripts/clawgym_nanobot/training_free_only.py"
    elif [ "${TRAINING_FREE_ROLLOUT_ONLY}" = "1" ]; then
        [ "${DEBUG_ROLLOUT_ONLY}" = "1" ] && [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]] && [ "${DO_EVAL}" = "0" ] || {
            echo "ERROR: TRAINING_FREE_ROLLOUT_ONLY requires DEBUG_ROLLOUT_ONLY=1, positive NUM_ROLLOUT, and DO_EVAL=0" >&2
            exit 1
        }
        TRAIN_ENTRYPOINT="${PROJECT_ROOT}/scripts/clawgym_nanobot/training_free_only.py"
    fi
    ROLLOUT_PATH_ARGS=(
        --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH}"
    )
    if [ "${USE_POLAR}" = "1" ]; then
        if [ "${TRAINING_FREE_EVAL_ONLY}" = "1" ] || [ "${TRAINING_FREE_ROLLOUT_ONLY}" = "1" ]; then
            :
        elif [ "${SLIME_ASYNC_TRAINING}" = "1" ]; then
            TRAIN_ENTRYPOINT="${BUNDLE_ROOT}/runtime/train_async.py"
        else
            TRAIN_ENTRYPOINT="${SLIME_ROOT}/train.py"
        fi
        ROLLOUT_SHUFFLE_ARGS=(--rollout-shuffle)
        case "${OPENCLAW_TRAIN_ORDER:-shuffle}" in
            sequential|seq|ordered|no_shuffle|no-shuffle)
                ROLLOUT_SHUFFLE_ARGS=()
                ;;
            sequential_then_shuffle|seq_then_shuffle|ordered_then_shuffle|first_epoch_sequential_then_shuffle|sequential_first_epoch_then_shuffle)
                ROLLOUT_SHUFFLE_ARGS=(--rollout-shuffle --rollout-shuffle-after-first-epoch)
                ;;
        esac
        ROLLOUT_PATH_ARGS=(
            --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async
            --custom-rm-path slime_bridge.reward.reward_func
            --custom-config-path "${POLAR_CONFIG_PATH}"
            --data-source-path "${POLAR_DATA_SOURCE_PATH}"
            --prompt-data "${TRAIN_PROMPT_DATA}"
            --input-key input
            --metadata-key metadata
            "${ROLLOUT_SHUFFLE_ARGS[@]}"
            --sglang-router-port "${SGLANG_ROUTER_PORT}"
            --router-policy "${SGLANG_ROUTER_POLICY}"
            --router-assignment-mode "${SGLANG_ROUTER_ASSIGNMENT_MODE}"
        )
    fi
    if [ "${OFFLINE_CRITIC_PRETRAIN}" = "1" ]; then
        ROLLOUT_PATH_ARGS=(
            --rollout-function-path slime_bridge.offline_critic_pretrain.generate_offline_critic_rollout
            --data-source-path slime_bridge.offline_critic_pretrain.OfflineCriticDataSource
            --custom-config-path "${OFFLINE_CRITIC_CONFIG_PATH}"
        )
    fi
    if [ "${SLIME_ASYNC_TRAINING}" = "0" ] && [ "${TRAIN_ENTRYPOINT}" != "${SLIME_ROOT}/train.py" ] && [ "${TRAINING_FREE_EVAL_ONLY}" != "1" ] && [ "${TRAINING_FREE_ROLLOUT_ONLY}" != "1" ]; then
        echo "ERROR: synchronous Slime training must use ${SLIME_ROOT}/train.py; got ${TRAIN_ENTRYPOINT}" >&2
        exit 1
    fi
    # Slime's PPO old-policy reference must follow the SGLang sampling
    # distribution for both the Polar and the direct ClawGym rollout paths.
    # Previously this flag was only forwarded from inside the USE_POLAR
    # branch, so direct ClawGym runs logged USE_ROLLOUT_LOGPROBS=1 while the
    # actual Slime argument remained false.
    if [ "${USE_ROLLOUT_LOGPROBS}" = "1" ]; then
        ROLLOUT_PATH_ARGS+=(--use-rollout-logprobs)
    fi

    echo "[proRL-nanobot] submitting Ray training job entrypoint=${TRAIN_ENTRYPOINT} train_backend=${TRAIN_BACKEND} async_training=${SLIME_ASYNC_TRAINING} synchronous_policy_versions=${POLAR_SYNCHRONOUS_POLICY_VERSIONS} use_rollout_logprobs=${USE_ROLLOUT_LOGPROBS} use_tis=${USE_TIS} use_opsd=${USE_OPSD} opsd_teacher_mode=${OPSD_TEACHER_MODE} partial_rollout=${PARTIAL_ROLLOUT} num_steps_per_rollout=${NUM_STEPS_PER_ROLLOUT} global_batch_size=${GLOBAL_BATCH_SIZE}"
    set +e
    TRAIN_SUBMIT_OUTPUT="${LOG_DIR}/ray-submit-output.log"
    # The Polar gateways use the Ray-managed SGLang router on the driver as
    # their inference upstream.  Submit first so the rollout manager can
    # create that router; only then wait for the gateway registration barrier.
    # The previous order waited for Polar before submitting Ray, which made
    # every gateway probe its own not-yet-started upstream indefinitely.
    ray job submit --address="http://${RAY_JOB_SUBMIT_ADDR}:${RAY_DASHBOARD_PORT}" \
        --no-wait \
        --runtime-env="${RESEARCHCLAW_RAY_ENV_FILE}" \
        -- python3 -u "${TRAIN_ENTRYPOINT}" \
        --actor-num-nodes "${ACTOR_NUM_NODES}" \
        --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
        --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
        --num-gpus-per-node "${NUM_GPUS_PER_NODE}" \
        "${TRAIN_BACKEND_ARGS[@]}" \
        "${TRAIN_MODE_ARGS[@]}" \
        "${MODEL_ARGS[@]}" \
        "${CUSTOM_MODEL_PROVIDER_ARGS[@]}" \
        "${MEGATRON_CONFIG_ARGS[@]}" \
        --hf-checkpoint "${HF_CHECKPOINT}" \
        --ref-load "${REF_LOAD}" \
        "${CKPT_ARGS[@]}" \
        "${SAVE_ARGS[@]}" \
        --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE:-bridge}" \
        "${WEIGHT_SYNC_ARGS[@]}" \
        --train-env-vars "${TRAIN_ENV_VARS_JSON}" \
        --train-memory-margin-bytes "${TRAIN_MEMORY_MARGIN_BYTES}" \
        --num-rollout "${NUM_ROLLOUT}" \
        --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}" \
        --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
        --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
        --rollout-max-context-len "${SGLANG_CONTEXT_LENGTH}" \
        --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}" \
        --seq-length "${TRAIN_SEQ_LENGTH}" \
        --rollout-temperature "${ROLLOUT_TEMPERATURE:-0.7}" \
        --rollout-top-p "${ROLLOUT_TOP_P:-1.0}" \
        --rollout-top-k "${ROLLOUT_TOP_K:--1}" \
        --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}" \
        --global-batch-size "${GLOBAL_BATCH_SIZE}" \
        --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}" \
        --reward-key score \
        "${ROLLOUT_PATH_ARGS[@]}" \
        --custom-rollout-log-function-path "${CUSTOM_ROLLOUT_LOG_FUNCTION_PATH}" \
        --custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}" \
        --tensor-model-parallel-size "${TP_SIZE}" \
        --context-parallel-size "${CP_SIZE}" \
        --pipeline-model-parallel-size "${PP_SIZE}" \
        "${PIPELINE_LAYOUT_ARGS[@]}" \
        "${SEQUENCE_PARALLEL_ARGS[@]}" \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
        --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
        --rollout-num-gpus-per-engine "${ROLLOUT_GPUS}" \
        --sglang-context-length "${SGLANG_ENGINE_CONTEXT_LENGTH}" \
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
        "${SGLANG_PARSER_ARGS[@]}" \
        --optimizer adam \
        --lr "${LR:-1e-6}" \
        --lr-decay-style constant \
        --weight-decay "${WEIGHT_DECAY:-0.1}" \
        --adam-beta1 "${ADAM_BETA1:-0.9}" \
        --adam-beta2 "${ADAM_BETA2:-0.98}" \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --accumulate-allreduce-grads-in-fp32 \
        --attention-softmax-in-fp32 \
        --attention-backend flash \
        "${TRAIN_COMPAT_ARGS[@]}" \
        "${EVAL_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
        >"${TRAIN_SUBMIT_OUTPUT}" 2>&1
    submit_status="$?"
    cat "${TRAIN_SUBMIT_OUTPUT}" >>"${TRAIN_LOG}"
    ray_job_id="$(sed -n "s/.*Job '\([^']*\)' submitted successfully.*/\1/p" "${TRAIN_SUBMIT_OUTPUT}" | head -n 1)"
    if [ -z "${ray_job_id}" ]; then
        ray_job_id="$(grep -oE 'raysubmit_[[:alnum:]_-]+' "${TRAIN_SUBMIT_OUTPUT}" | head -n 1 || true)"
    fi
    if [ "${submit_status}" -ne 0 ] || [ -z "${ray_job_id}" ]; then
        echo "ERROR: Ray training job submission failed status=${submit_status}" | tee -a "${TRAIN_LOG}" >&2
        cat "${TRAIN_SUBMIT_OUTPUT}" >&2
        submission_exit_status="${submit_status}"
        [ "${submission_exit_status}" -ne 0 ] || submission_exit_status=1
        {
            printf 'ray_status=%s\n' "${submit_status}"
            printf 'ray_job_status=SUBMISSION_FAILED\n'
            printf 'exit_status=%s\n' "${submission_exit_status}"
        } >"${LOG_DIR}/train-exit-status.env"
        touch "${STOP_FILE}"
        cleanup_ray
        cleanup_stale_gpu_processes
        exit "${submission_exit_status}"
    fi
    echo "[proRL-nanobot] Ray training job submitted id=${ray_job_id}" | tee -a "${TRAIN_LOG}"
    if [ "${polar_gateway_start_deferred}" = "1" ]; then
        wait_for_sglang_router
        start_polar_gateway clawgym-rollout-1 "${LOG_DIR}/polar-gateway.log"
    fi
    if [ "${USE_POLAR}" = "1" ]; then
        if ! wait_for_polar; then
            echo "ERROR: Polar gateway did not register after Ray submission id=${ray_job_id}" | tee -a "${TRAIN_LOG}" >&2
            ray job stop --address="http://${RAY_JOB_SUBMIT_ADDR}:${RAY_DASHBOARD_PORT}" --no-wait "${ray_job_id}" >/dev/null 2>&1 || true
            {
                printf 'ray_status=1\n'
                printf 'ray_job_id=%s\n' "${ray_job_id}"
                printf 'ray_job_status=POLAR_STARTUP_FAILED\n'
                printf 'exit_status=1\n'
            } >"${LOG_DIR}/train-exit-status.env"
            touch "${STOP_FILE}"
            cleanup_ray
            cleanup_stale_gpu_processes
            exit 1
        fi
    fi
    ray job logs --address="http://${RAY_JOB_SUBMIT_ADDR}:${RAY_DASHBOARD_PORT}" \
        --follow "${ray_job_id}" 2>&1 \
        | python3 -u "${PROJECT_ROOT}/scripts/clawgym_nanobot/filter_training_log.py" \
        | tee -a "${TRAIN_LOG}" >/dev/null
    pipeline_status=("${PIPESTATUS[@]}")
    logs_status="${pipeline_status[0]}"
    filter_status="${pipeline_status[1]}"
    tee_status="${pipeline_status[2]}"
    ray_job_status_value="$(ray_job_status "${ray_job_id}" 2>/dev/null || printf 'UNKNOWN')"
    if [ "${ray_job_status_value}" = "SUCCEEDED" ]; then
        ray_status=0
    else
        ray_status=1
    fi
    pipeline_exit="${ray_status}"
    if [ "${logs_status}" -ne 0 ]; then
        pipeline_exit="${logs_status}"
    fi
    if [ "${filter_status}" -ne 0 ]; then
        pipeline_exit="${filter_status}"
    elif [ "${tee_status}" -ne 0 ]; then
        pipeline_exit="${tee_status}"
    fi
    {
        printf 'ray_status=%s\n' "${ray_status}"
        printf 'ray_job_id=%s\n' "${ray_job_id}"
        printf 'ray_job_status=%s\n' "${ray_job_status_value}"
        printf 'logs_status=%s\n' "${logs_status}"
        printf 'filter_status=%s\n' "${filter_status}"
        printf 'tee_status=%s\n' "${tee_status}"
        printf 'exit_status=%s\n' "${pipeline_exit}"
    } >"${LOG_DIR}/train-exit-status.env"
    echo "[proRL-nanobot] Ray training job exited status=${ray_status} job_status=${ray_job_status_value}; logs=${logs_status}; log_filter=${filter_status}; tee=${tee_status}"
    touch "${STOP_FILE}"
    cleanup_ray
    cleanup_stale_gpu_processes
    exit "${pipeline_exit}"
else
    cleanup_ray
    # Worker ranks can retain detached CUDA children from a previous Ray job.
    # `ray stop --force` clears the local Ray runtime, but it does not always
    # reap an already-orphaned SGLang scheduler.  Clear those processes before
    # this rank joins the new Ray cluster; otherwise TP startup can observe a
    # mixture of empty and occupied GPUs and abort its memory-balance check.
    cleanup_stale_training_drivers
    cleanup_stale_gpu_processes
    wait_for_head
    while [ ! -f "${LOG_DIR}/ray-head-ready" ]; do sleep 2; done
    # These same workers are already registered with the current Ray head.
    ray status --address "${MASTER_ADDR}:${MASTER_PORT}" >/dev/null
    if [ "${polar_gateway_start_deferred}" = "1" ]; then
        wait_for_sglang_router
        polar_gateway_ordinal=$((NODE_RANK - polar_primary_gateway_rank + 1))
        if [ "${NODE_RANK}" -eq "${polar_primary_gateway_rank}" ]; then
            start_polar_gateway clawgym-rollout-1 "${LOG_DIR}/polar-gateway.log"
        else
            start_polar_gateway \
                "clawgym-rollout-${polar_gateway_ordinal}" \
                "${LOG_DIR}/polar-gateway-rank${NODE_RANK}.log"
        fi
    fi
    while [ ! -f "${STOP_FILE}" ]; do
        sleep 30
    done
    cleanup_ray
fi
