#!/usr/bin/env python3
"""Validate a Gemma4 E-series text checkpoint without loading tensor payloads."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def _shape(sf, key):
    return tuple(sf.get_slice(key).get_shape())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    root = Path(args.model_path)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    ple_dim = int(text.get("hidden_size_per_layer_input") or 0)
    vocab = int(text.get("vocab_size_per_layer_input") or text["vocab_size"])
    if ple_dim <= 0:
        raise SystemExit("checkpoint is not a Gemma4 E-series PLE model")
    if text.get("attention_k_eq_v", True):
        raise SystemExit("E-series contract expects attention_k_eq_v=false")

    tensor_path = root / "model.safetensors"
    if not tensor_path.is_file():
        raise SystemExit(f"single-file checkpoint not found: {tensor_path}")

    layer_suffixes = {
        "input_layernorm.weight",
        "layer_scalar",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "per_layer_input_gate.weight",
        "per_layer_projection.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_per_layer_input_norm.weight",
        "pre_feedforward_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    }
    expected = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.embed_tokens_per_layer.weight",
        "model.language_model.norm.weight",
        "model.language_model.per_layer_model_projection.weight",
        "model.language_model.per_layer_projection_norm.weight",
    }
    for layer in range(layers):
        expected.update(
            f"model.language_model.layers.{layer}.{suffix}"
            for suffix in layer_suffixes
        )

    with safe_open(tensor_path, framework="pt", device="cpu") as sf:
        actual = {key for key in sf.keys() if key.startswith("model.language_model.")}
        missing = expected - actual
        unexpected = actual - expected
        if missing or unexpected:
            raise SystemExit(
                f"text key contract mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )

        expected_ple_shape = (vocab, layers * ple_dim)
        actual_ple_shape = _shape(sf, "model.language_model.embed_tokens_per_layer.weight")
        if actual_ple_shape != expected_ple_shape:
            raise SystemExit(
                f"PLE embedding shape {actual_ple_shape} != expected {expected_ple_shape}"
            )
        projection_shape = _shape(sf, "model.language_model.per_layer_model_projection.weight")
        if projection_shape != (layers * ple_dim, hidden):
            raise SystemExit(
                f"PLE model projection shape {projection_shape} != {(layers * ple_dim, hidden)}"
            )

    global_kv = text.get("num_global_key_value_heads") or text["num_key_value_heads"]
    print(
        json.dumps(
            {
                "model": str(root),
                "layers": layers,
                "hidden_size": hidden,
                "ple_dim": ple_dim,
                "text_tensor_keys": len(actual),
                "global_num_kv_heads_effective": global_kv,
                "attention_k_eq_v": False,
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
