# DEAL & Self-Rewarded Exploration (SRE)

Official training and evaluation codebase for **Learning from Single-Feedback Deployment Experience**.

- **Hugging Face Collection**: [SRE-DEAL](https://huggingface.co/collections/trxcc2002/sre-deal)
- **Public Datasets**:
  - **DEAL-WS** (Workplace Assistance — 2,000 train / 200 test): [`trxcc2002/DEAL-WS`](https://huggingface.co/datasets/trxcc2002/DEAL-WS) (references upstream `filesys_en.zip` from [`Workspace-Bench/Workspace-Bench-Workspaces`](https://huggingface.co/datasets/Workspace-Bench/Workspace-Bench-Workspaces))
  - **DEAL-RS** (AI Research — 1,800 train / 200 test): [`trxcc2002/DEAL-RS`](https://huggingface.co/datasets/trxcc2002/DEAL-RS)
- **Public Checkpoints (SRE final checkpoints)**:
  - [`trxcc2002/DEAL-WS-Gemma`](https://huggingface.co/trxcc2002/DEAL-WS-Gemma) (Gemma4-E4B-IT, step 124)
  - [`trxcc2002/DEAL-WS-Qwen`](https://huggingface.co/trxcc2002/DEAL-WS-Qwen) (Qwen3.5-9B, step 124)
  - [`trxcc2002/DEAL-RS-Gemma`](https://huggingface.co/trxcc2002/DEAL-RS-Gemma) (Gemma4-E4B-IT, step 112)
  - [`trxcc2002/DEAL-RS-Qwen`](https://huggingface.co/trxcc2002/DEAL-RS-Qwen) (Qwen3.5-9B, step 112)

---

## Overview

In deployment-time self-evolution, an assistant executes one live attempt per arriving task and receives a single external outcome feedback $F_t = (u_t, c_t)$. **Self-Rewarded Exploration (SRE)** overcomes the single-rollout exploration bottleneck by:

1. **Offline Rollout Expansion**: Restoring isolated copies of the initial task snapshot $s_t$ and sampling $K = 7$ additional candidate trajectories (without exposing $F_t$ to the actor).
2. **Feedback-Conditioned Self-Reward Estimation**: Using the frozen initial policy $\pi_{\theta_0}$ as a two-stage evaluator:
   - A **self-rubric agent** builds a shared rubric $\mathcal{Q}_t$ grounded in the task specification, workspace files, live trajectory, and live feedback $F_t$.
   - A **self-scoring agent** grades each offline candidate against $\mathcal{Q}_t$ across 3 independent sessions (taking the median valid score) without seeing $F_t$.
3. **Internal-Only GRPO Update**: Computing group-relative advantages solely across the $K = 7$ offline trajectories (excluding the live anchor and its external GT reward from the normalized update group).

### Key Release Settings

| Setting | DEAL-WS | DEAL-RS |
|---|---|---|
| **GPU Topology (Unified)** | **4 train + 4 rollout** (8 GPUs) | **4 train + 4 rollout** (8 GPUs) |
| **Rollout Group Structure** | **1 GT anchor + 7 SR** (`K=7`, 8 total) | **1 GT anchor + 7 SR** (`K=7`, 8 total) |
| **GT Judge Scope** | Anchor-only (`gt_scope=anchor_only`) | Anchor-only (`gt_scope=anchor_only`) |
| **Deliverable Hardgate** | **Enabled** (`CLAWGYM_ARTIFACT_VALIDITY_GATE=1`) | Enabled |
| **Self-Reward Repeats** | 3 (median valid score) | 3 (median valid score) |
| **Train / Held-out Split** | 2,000 train (125 steps) / 200 test | 1,800 train (113 steps) / 200 test |
| **Context / Max Output** | 64K / 8K | 72K / 16K (eval: 80K / 16K) |
| **Models Supported** | `gemma4-e4b-it`, `qwen3.5-9b` | `gemma4-e4b-it`, `qwen3.5-9b` |

> **GPU Layout**: All training recipes use a unified single-host **4 train GPUs (TP=4) + 4 rollout GPUs (4 × SGLang TP=1)** layout (`TRAIN_GPUS=0,1,2,3`, `ROLLOUT_GPUS=4,5,6,7`). The frozen self-reward model $\pi_{\theta_0}$ and external GT user simulator (Gemma4-31B-IT in the paper) are configured via OpenAI-compatible HTTP(S) endpoints (`SELF_REWARD_BASE_URL` and `GT_BASE_URL`), keeping the 8-GPU training node dedicated to training and rollout.

---

## Repository Structure

```text
.
├── configs/
│   ├── deal-rs-gemma4.recipe.json          # DEAL-RS Gemma4-E4B-IT (4+4, 1 GT + 7 SR)
│   ├── deal-rs-qwen35.recipe.json          # DEAL-RS Qwen3.5-9B (4+4, 1 GT + 7 SR)
│   ├── deal-ws-gemma4.recipe.json          # DEAL-WS Gemma4-E4B-IT (4+4, hardgate, 1 GT + 7 SR)
│   ├── deal-ws-qwen35.recipe.json          # DEAL-WS Qwen3.5-9B (4+4, hardgate, 1 GT + 7 SR)
│   ├── deal-rs.env.example                 # User config template for DEAL-RS
│   ├── deal-ws.env.example                 # User config template for DEAL-WS
│   ├── reward-service.deal-rs.template.json
│   ├── reward-service.deal-ws.template.json
│   ├── requirements-python.txt
│   └── constraints-cu129.txt
├── runtime/
│   ├── run_core.sh                         # Core Slime + Polar + Ray async training entrypoint
│   ├── train_async.py                      # Async rollout/update driver
│   ├── portable_contract.py                # Unified 4+4 & 1 GT + 7 SR recipe validator
│   ├── sample_contract.py                  # Token/mask/logprob consistency validator
│   ├── aligned_eval_recovery.py            # Failure=0 / 200-task denominator eval adapter
│   ├── judge_gate/                         # Request concurrency gate & history budget
│   └── ws_judge/                           # DEAL-WS native GT judge sidecar, proxy, & inspector
├── scripts/
│   ├── install.sh                          # Linux CUDA 12.9 / Python 3.12 dependency installer
│   ├── doctor.py                           # Import & 8-GPU diagnostic checker
│   ├── prepare_data.py                     # Materialize HF datasets DEAL-RS & DEAL-WS
│   ├── configure_run.py                    # Isolated per-attempt config & manifest generator
│   ├── launch.sh / run.py                  # Single-host 4+4 launcher (--check for CPU preflight)
│   ├── resume_latest.sh / resume_latest.py # Resume from latest HF checkpoint + committed cursor
│   ├── serve_frozen_model.sh               # Helper to serve frozen pi_theta_0 via SGLang
│   ├── eval_deal_rs.py                     # Standalone DEAL-RS 200-task evaluator
│   └── eval_deal_ws.py                     # Standalone DEAL-WS 200-task evaluator (hardgate)
├── vendor/
│   ├── prorl/                              # Nanobot RL harness, Polar 4-engine gateway, hardgate
│   ├── reward/                             # SRE self-reward service (1 GT anchor + 7 SR)
│   ├── nanobot/                            # Agent execution harness
│   ├── slime/                              # Slime RL framework (with NeMo-Megatron KL/TIS)
│   ├── nemo-rl/                            # NeMo-RL Megatron backend for Gemma4
│   └── megatron-bridge/                    # Megatron-Bridge & Megatron-LM for Gemma4
└── tests/
    └── test_release.py                     # CPU regression & hygiene tests
```

---

## 1. Environment Setup

See [INSTALL.md](INSTALL.md) for full details (Linux x86_64, Python 3.12, CUDA 12.9 devel, 8×H200/H100).

```bash
bash scripts/install.sh
python3 scripts/doctor.py --imports
```

---

## 2. Data Preparation

Both datasets are hosted publicly on Hugging Face and materialized locally via `scripts/prepare_data.py`:

### DEAL-RS (1,800 train / 200 test)
```bash
python3 scripts/prepare_data.py \
  --benchmark deal-rs \
  --output ./data/deal-rs
```

### DEAL-WS (2,000 train / 200 test)
DEAL-WS references `filesys_en.zip` from [`Workspace-Bench/Workspace-Bench-Workspaces`](https://huggingface.co/datasets/Workspace-Bench/Workspace-Bench-Workspaces) (not duplicated in the DEAL-WS dataset repo):
```bash
python3 scripts/prepare_data.py \
  --benchmark deal-ws \
  --output ./data/deal-ws \
  --workspace-zip /path/to/filesys_en.zip
# Or pass --download-workspace-zip to download filesys_en.zip from Hugging Face automatically.
```

---

## 3. Training (Unified 4 Train + 4 Rollout)

### Step A: Start the Frozen Self-Reward Policy ($\pi_{\theta_0}$)
Host the frozen initial model (e.g., Gemma4-E4B-IT or Qwen3.5-9B) on a separate GPU/host so all 8 GPUs on the training node serve `4 train + 4 rollout`:
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_frozen_model.sh \
  --model-path /path/to/gemma-4-E4B-it \
  --model-family gemma4 \
  --port 29040
```
*(For Qwen3.5-9B, pass `--model-family qwen35 --model-path /path/to/Qwen3.5-9B`.)*

### Step B: Configure Local Secrets & Endpoints
Create chmod-600 key files and copy the template for your benchmark:
```bash
mkdir -p ./secrets && chmod 700 ./secrets
printf '%s\n' 'YOUR_GT_JUDGE_API_KEY' > ./secrets/gt-api-key
printf '%s\n' 'local-frozen-no-auth' > ./secrets/self-reward-api-key
chmod 600 ./secrets/gt-api-key ./secrets/self-reward-api-key

# DEAL-RS (fixed 1 GT anchor + 7 SR)
cp configs/deal-rs.env.example configs/deal-rs.local.env
chmod 600 configs/deal-rs.local.env

# Or DEAL-WS (hardgate + 1 GT anchor + 7 SR)
cp configs/deal-ws.env.example configs/deal-ws.local.env
chmod 600 configs/deal-ws.local.env
```
Edit `BASE_MODEL`, `DATA_DIR`, `RUN_DIR`, `MODEL` (`gemma4-e4b-it` or `qwen3.5-9b`), `SELF_REWARD_*`, and `GT_*` in your `.local.env` file.

### Step C: CPU Preflight & Launch
```bash
# Native CPU preflight (validates 4+4 recipe, 1 GT + 7 SR contract, and finite dataset without GPU/API calls)
bash scripts/launch.sh --config configs/deal-rs.local.env --check

# Launch 8-GPU (4 train + 4 rollout) training
bash scripts/launch.sh --config configs/deal-rs.local.env
```
To resume an interrupted run from the latest committed HF checkpoint and rollout cursor:
```bash
bash scripts/resume_latest.sh --config configs/deal-rs.local.env
```

---

## 4. Evaluation (Held-out 200 Tasks)

First serve any trained or public Hugging Face checkpoint (e.g., `trxcc2002/DEAL-RS-Gemma`, `trxcc2002/DEAL-WS-Gemma`, `trxcc2002/DEAL-RS-Qwen`, `trxcc2002/DEAL-WS-Qwen`) behind an OpenAI-compatible server such as SGLang (`scripts/serve_frozen_model.sh --port 8000`).

### Evaluate on DEAL-RS (200 tasks)
```bash
python3 scripts/eval_deal_rs.py \
  --data-dir ./data/deal-rs \
  --output-dir ./eval-runs/deal-rs-gemma \
  --student-url http://127.0.0.1:8000/v1 \
  --student-model gemma4-e4b-it \
  --judge-url https://YOUR-GT-JUDGE-SERVICE/v1 \
  --judge-model gemma-4-31B-it \
  --judge-api-key-file ./secrets/gt-api-key
```

### Evaluate on DEAL-WS (200 tasks, with Hardgate)
```bash
python3 scripts/eval_deal_ws.py \
  --data-dir ./data/deal-ws \
  --output-dir ./eval-runs/deal-ws-gemma \
  --student-url http://127.0.0.1:8000/v1 \
  --student-model gemma4-e4b-it \
  --model-family gemma4 \
  --judge-url https://YOUR-GT-JUDGE-SERVICE/v1 \
  --judge-model gemma-4-31B-it \
  --judge-api-key-file ./secrets/gt-api-key
```

Both evaluators score failed/timed-out tasks as `0.0` and retain all 200 tasks in the reported denominator (`summary.json`).

---

## Acknowledgements & Third-Party Notices

This repository builds on open-source agent and RL infrastructure including [slime](https://github.com/THUDM/slime), [nanobot](https://github.com/HKUDS/nanobot), [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL), [Megatron-Bridge / Megatron-LM](https://github.com/NVIDIA/Megatron-LM), and [SGLang](https://github.com/sgl-project/sglang). Upstream license files are preserved in their respective `vendor/` directories.

## Citation

```bibtex
@article{tan2026deal,
  title={Learning from Single-Feedback Deployment Experience},
  author={Tan, Rong-Xi and Li, Yizhuo and Wang, Wenjie and Qu, Xiaoye and Chen, Qiwei and Zhu, Tong and Luo, Yun and Wang, Zhi and Qian, Chao and Cheng, Yu and Yang, Yang and Li, Yafu},
  year={2026}
}
```
