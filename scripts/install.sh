#!/usr/bin/env bash
# Install Python and CUDA runtime dependencies for DEAL-WS and DEAL-RS training & evaluation.
# Target environment: Linux x86_64, Python 3.12, CUDA 12.9 devel, 8 GPUs (4 train + 4 rollout).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - <<'CHECK'
import platform, sys
assert sys.version_info[:2] == (3, 12), "Use Python 3.12 in a dedicated Linux environment"
assert platform.system() == "Linux" and platform.machine() == "x86_64", "Linux x86_64 required"
CHECK

command -v nvcc >/dev/null || { echo "ERROR: CUDA 12.9 devel toolkit (nvcc) is required" >&2; exit 1; }
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

python3 -m pip install --upgrade pip setuptools wheel packaging ninja pybind11

# PyTorch 2.11.0 + CUDA 12.9
python3 -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url "${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu129}"

# Core training, rollout, dataset, and document-verification dependencies
python3 -m pip install -c configs/constraints-cu129.txt -r configs/requirements-python.txt
python3 -m pip install pyarrow huggingface_hub python-docx openpyxl python-pptx pypdf pyreadstat xlrd pyyaml pandas reportlab

# Build TransformerEngine and FlashAttention against the installed PyTorch/CUDA
python3 -m pip install --no-build-isolation -c configs/constraints-cu129.txt \
  'transformer-engine[pytorch]==2.16.1' 'flash-attn==2.8.3'

# SGLang inference kernels and pinned runtime stack
python3 -m pip install -c configs/constraints-cu129.txt \
  'flashinfer-python==0.6.12' 'flashinfer-cubin==0.6.12' \
  'sglang-kernel==0.4.3' 'xgrammar==0.2.1' 'apache-tvm-ffi==0.1.9'
python3 -m pip install --no-deps 'ffpa-attn==0.2.2'
python3 -m pip install --no-deps 'nvidia-cutlass-dsl==4.6.0' 'quack-kernels==0.6.1'
python3 -m pip install --no-deps 'megatron-core==0.16.0rc0' 'sglang==0.5.13'

python3 scripts/doctor.py --imports
printf '\nEnvironment setup complete. Next: python3 scripts/prepare_data.py --benchmark deal-rs (or deal-ws)\n'
