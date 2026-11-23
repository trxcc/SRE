#!/usr/bin/env bash

# Megatron CLI metadata for Gemma4 E2B. The NeMo/Megatron-Bridge provider is
# the structural source of truth; these matching values are still consumed by
# Slime/Megatron argument validation, optimizer setup, and schedule creation.
MODEL_ARGS=(
    --num-layers 35
    --hidden-size 1536
    --ffn-hidden-size 6144
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 1
    --kv-channels 256
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --use-rotary-position-embeddings
    --rotary-base 1000000
    --vocab-size 262144
    --qk-layernorm
)
