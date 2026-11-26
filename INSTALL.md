# Installation & Environment Setup

## Requirements

- **OS / Arch**: Linux x86_64 (Ubuntu 24.04 recommended)
- **Hardware (Training)**: 8× NVIDIA H200 (or 8× H100 80GB+ with adjusted `ROLLOUT_MAX_RUNNING_REQUESTS`), ≥ 512 GiB CPU RAM
  - GPUs `0–3`: Actor training (`TP=4`)
  - GPUs `4–7`: 4× SGLang rollout engines (`TP=1` each)
  - Separate GPU/host (or OpenAI-compatible API): Frozen self-reward model ($\pi_{\theta_0}$) and external GT user simulator (`google/gemma-4-31B-it` in the paper)
- **Hardware (Standalone Held-out Evaluation)**:
  - 1× GPU (≥ 40GB VRAM, e.g., A100/H100/H200) to serve the student checkpoint (`DEAL-*-Gemma` or `DEAL-*-Qwen`, `TP=1`)
  - 2× GPUs (`TP=2`) or an external OpenAI-compatible endpoint to serve the `google/gemma-4-31B-it` GT judge
- **Software**: Python 3.12, CUDA 12.9 devel toolkit (`nvcc`), GCC/G++ 13+, Ninja, CMake
- **DEAL-WS Judge CLI**: The DEAL-WS native GT judge sidecar invokes the Claude Code CLI (`claude` on `PATH` or configured via `CLAUDE_BIN` / `--claude-bin`).

## Direct Installation

Run inside a dedicated Linux container or Python 3.12 virtual environment:

```bash
bash scripts/install.sh
python3 scripts/verify_env.py --mode all
python3 scripts/doctor.py --imports
python3 scripts/doctor.py --gpu
```

You can also run `scripts/verify_env.py` in targeted modes:

```bash
# CPU-only check: required templates/files, WS scaffold installer, all 4 profile contracts
python3 scripts/verify_env.py --mode cpu

# Validate a specific .local.env config and probe live OpenAI-compatible judge/self-reward endpoints
python3 scripts/verify_env.py --mode all --config configs/deal-ws.local.env --probe-endpoints
```

If building FlashAttention / TransformerEngine on a host with limited CPU RAM, lower parallelism with:

```bash
MAX_JOBS=4 bash scripts/install.sh
```

> **Note**: Vendored `vendor/nemo-rl`, `vendor/megatron-bridge`, `vendor/slime`, `vendor/nanobot`, `vendor/prorl`, and `vendor/reward` are loaded directly via `PYTHONPATH` by the launchers and evaluation scripts; do not run `uv sync` inside those subdirectories, as their standalone lockfiles pin incompatible PyTorch/CUDA versions.

## Serving Helper (`scripts/serve_frozen_model.sh`)

Use `scripts/serve_frozen_model.sh` to host any checkpoint (frozen $\pi_{\theta_0}$, trained SRE checkpoint, or `google/gemma-4-31B-it` GT judge) via SGLang with the exact reasoning and tool-call parsers used in the paper:

```bash
# Inspect all options
bash scripts/serve_frozen_model.sh --help

# Serve Gemma4-E4B-IT or DEAL-*-Gemma (TP=1)
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_frozen_model.sh \
  --model-path /path/to/DEAL-RS-Gemma \
  --model-family gemma4 \
  --served-model-name gemma4-e4b-it \
  --port 8000

# Serve Qwen3.5-9B or DEAL-*-Qwen (TP=1)
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_frozen_model.sh \
  --model-path /path/to/DEAL-RS-Qwen \
  --model-family qwen35 \
  --served-model-name qwen3.5-9b \
  --port 8000

# Optional: Self-host Gemma4-31B-IT GT judge locally (TP=2)
CUDA_VISIBLE_DEVICES=1,2 bash scripts/serve_frozen_model.sh \
  --model-path /path/to/gemma-4-31B-it \
  --model-family gemma4 \
  --served-model-name gemma-4-31B-it \
  --tp 2 \
  --port 29050 \
  --api-key-file ./secrets/gt-api-key
```

## Docker Build (Optional)

```bash
docker build -t deal-sre:latest .
docker run --rm --gpus all --ipc=host --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/models:/models:ro \
  -v /path/to/data:/opt/deal/data \
  -v /path/to/runs:/runs \
  -v /path/to/secrets:/secrets:ro \
  -v "$PWD/configs/deal-rs.local.env:/opt/deal/configs/local.env:ro" \
  deal-sre:latest --config configs/local.env
```

## Tested Runtime Versions

| Package | Version |
|---|---|
| Python | 3.12.x |
| PyTorch | `2.11.0+cu129` |
| Ray | `2.56.0` |
| Transformers | `5.8.1` |
| SGLang | `0.5.13` |
| FlashAttention | `2.8.3` |
| TransformerEngine | `2.16.1` |
| FlashInfer | `0.6.12` |
| FFPA Attention | `0.2.2` |
| Megatron-Core | `0.16.0rc0` |
| Document & Data Stack | `python-docx`, `openpyxl`, `python-pptx`, `pypdf`, `reportlab`, `Pillow`, `pandas`, `pyarrow` |
