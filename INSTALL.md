# Installation & Environment Setup

## Requirements

- **OS / Arch**: Linux x86_64 (Ubuntu 24.04 recommended)
- **Hardware**: 8× NVIDIA H200 (or 8× H100 80GB+ with adjusted `ROLLOUT_MAX_RUNNING_REQUESTS`), ≥ 512 GiB CPU RAM
  - GPUs `0–3`: Actor training (TP=4)
  - GPUs `4–7`: 4× SGLang rollout engines (TP=1 each)
  - Separate GPU/host (or OpenAI-compatible API): Frozen self-reward model ($\pi_{\theta_0}$) and external GT judge (Gemma4-31B-IT)
- **Software**: Python 3.12, CUDA 12.9 devel toolkit (`nvcc`), GCC/G++ 13+, Ninja, CMake
- **DEAL-WS Judge CLI**: The DEAL-WS GT judge sidecar uses the Claude Code CLI (`claude` on `PATH` or configured via `CLAUDE_BIN`).

## Direct Installation

Run inside a dedicated Linux container or Python 3.12 virtual environment:

```bash
bash scripts/install.sh
python3 scripts/doctor.py --imports
python3 scripts/doctor.py --gpu
```

If building FlashAttention / TransformerEngine on a host with limited CPU RAM, lower parallelism with:

```bash
MAX_JOBS=4 bash scripts/install.sh
```

> **Note**: Vendored `vendor/nemo-rl`, `vendor/megatron-bridge`, `vendor/slime`, `vendor/nanobot`, `vendor/prorl`, and `vendor/reward` are loaded directly via `PYTHONPATH` by the launchers; do not run `uv sync` inside those subdirectories, as their standalone lockfiles pin incompatible PyTorch/CUDA versions.

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
