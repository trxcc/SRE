import hashlib
import inspect
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def apply_bridge_yarn_runtime_config(provider, args):
    """Apply explicit Slime YaRN CLI values to an HF-derived provider.

    Transformers permits ``beta_fast`` and ``beta_slow`` to be omitted from
    ``rope_scaling`` because it supplies defaults internally. Megatron Bridge
    does not supply those defaults, so the runtime values parsed by Slime must
    win before ``provider.finalize()`` constructs YarnRotaryEmbedding.
    """

    if getattr(args, "position_embedding_type", None) != "yarn":
        return None

    values = {
        "position_embedding_type": "yarn",
        "rotary_base": args.rotary_base,
        "seq_length": args.seq_length,
        "yarn_rotary_scaling_factor": args.rotary_scaling_factor,
        "yarn_original_max_position_embeddings": args.yarn_original_max_position_embeddings,
        "yarn_beta_fast": args.yarn_beta_fast,
        "yarn_beta_slow": args.yarn_beta_slow,
        "yarn_mscale": args.mscale,
        "yarn_mscale_all_dim": args.mscale_all_dim,
        "yarn_correction_range_round_to_int": args.yarn_correction_range_round_to_int,
    }
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise ValueError(f"YaRN bridge runtime config is missing values for: {missing}")

    for name, value in values.items():
        setattr(provider, name, value)
    actual = {name: getattr(provider, name, None) for name in values}
    if actual != values:
        raise RuntimeError(f"YaRN bridge provider config mismatch: expected={values} actual={actual}")
    return actual


def patch_hf_config_for_megatron_bridge(hf_config):
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)

    return bridge


def _qualified_name(value):
    value_type = value if inspect.isclass(value) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _source_path(value):
    try:
        return str(Path(inspect.getfile(value if inspect.ismodule(value) else type(value))).resolve())
    except (OSError, TypeError):
        return None


def _assert_source_under(label, source, expected_root):
    if not expected_root:
        return
    if source is None:
        raise RuntimeError(f"cannot resolve {label} source while {expected_root=} is required")
    source_path = Path(source).resolve()
    root_path = Path(expected_root).resolve()
    if not source_path.is_relative_to(root_path):
        raise RuntimeError(f"{label} escaped the pinned source tree: source={source_path} root={root_path}")


def _bridge_mapping_fingerprint(model_bridge, provider):
    # Gemma4VLBridge chooses the dense text registry from both the provider
    # dispatch state and GEMMA4_CONVERSION_MODE.  Exercise the same registry
    # selection used later by build_conversion_tasks.
    model_bridge.hf_config = getattr(model_bridge, "hf_config", None) or getattr(
        getattr(provider, "hf_pretrained", None), "config", None
    )
    registry = model_bridge.mapping_registry()
    mappings = [
        {
            "type": _qualified_name(mapping),
            "megatron": mapping.megatron_param,
            "hf": mapping.hf_param,
            "allow_hf_name_mismatch": bool(mapping.allow_hf_name_mismatch),
        }
        for mapping in registry.mappings
    ]
    encoded = json.dumps(mappings, sort_keys=True, separators=(",", ":")).encode()
    return mappings, hashlib.sha256(encoded).hexdigest()


def _bridge_audit_payload(bridge, path, purpose, provider=None):
    import megatron.core

    model_bridge = bridge._model_bridge
    if provider is None:
        provider = model_bridge.provider_bridge(bridge._provider_bridge_input)
    # build_conversion_tasks assigns this before mapping_registry().  Assign it
    # here too so this audit covers the exact registry branch used at runtime.
    model_bridge.hf_config = bridge.hf_pretrained.config
    mappings, mapping_sha256 = _bridge_mapping_fingerprint(model_bridge, provider)

    hf_config = bridge.hf_pretrained.config
    hf_state = getattr(getattr(bridge.hf_pretrained, "state", None), "source", None)
    hf_keys = sorted(hf_state.get_all_keys()) if hf_state is not None else []
    hf_keys_sha256 = hashlib.sha256("\n".join(hf_keys).encode()).hexdigest() if hf_keys else None
    provider_fields = [
        "num_layers",
        "hidden_size",
        "ffn_hidden_size",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "global_kv_channels",
        "num_global_query_groups",
        "num_kv_shared_layers",
        "per_layer_embed_vocab_size",
        "per_layer_embed_dim",
        "vocab_size",
        "seq_length",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "sequence_parallel",
        "variable_seq_lengths",
        "moe_token_dispatcher_type",
        "transformer_impl",
        "attention_backend",
        "softmax_scale",
        "qk_layernorm",
        "masked_softmax_fusion",
        "gradient_accumulation_fusion",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
    ]

    def json_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    return {
        "purpose": purpose,
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "checkpoint": str(Path(path).resolve()),
        "conversion_mode": os.environ.get("GEMMA4_CONVERSION_MODE"),
        "auto_bridge_class": _qualified_name(bridge),
        "model_bridge_class": _qualified_name(model_bridge),
        "model_bridge_source": _source_path(model_bridge),
        "provider_class": _qualified_name(provider),
        "provider_source": _source_path(provider),
        "megatron_core_source": str(Path(inspect.getfile(megatron.core)).resolve()),
        "hf_config_class": _qualified_name(hf_config),
        "hf_architectures": list(getattr(hf_config, "architectures", []) or []),
        "hf_key_count": len(hf_keys),
        "hf_keys_sha256": hf_keys_sha256,
        "mapping_count": len(mappings),
        "mapping_sha256": mapping_sha256,
        "mappings": mappings,
        "provider": {name: json_value(getattr(provider, name, None)) for name in provider_fields},
    }


def _validate_bridge_audit(payload):
    required = os.environ.get("SLIME_REQUIRE_GEMMA4_DENSE_BRIDGE", "0") == "1"
    if not required:
        return

    if payload["conversion_mode"] != "text":
        raise RuntimeError(
            "Gemma4 dense Slime integration requires GEMMA4_CONVERSION_MODE=text, "
            f"got {payload['conversion_mode']!r}"
        )

    expected_checkpoint = os.environ.get("SLIME_EXPECTED_HF_CHECKPOINT")
    if expected_checkpoint and payload["checkpoint"] != str(Path(expected_checkpoint).resolve()):
        raise RuntimeError(
            "AutoBridge checkpoint drift: "
            f"actual={payload['checkpoint']} expected={Path(expected_checkpoint).resolve()}"
        )

    expected_bridge = os.environ.get("SLIME_EXPECTED_MODEL_BRIDGE_CLASS")
    if expected_bridge and payload["model_bridge_class"] != expected_bridge:
        raise RuntimeError(
            f"AutoBridge class drift: actual={payload['model_bridge_class']} expected={expected_bridge}"
        )
    expected_provider = os.environ.get("SLIME_EXPECTED_MODEL_PROVIDER_CLASS")
    if expected_provider and payload["provider_class"] != expected_provider:
        raise RuntimeError(
            f"Megatron provider drift: actual={payload['provider_class']} expected={expected_provider}"
        )

    _assert_source_under(
        "Megatron-Bridge",
        payload["model_bridge_source"],
        os.environ.get("SLIME_EXPECTED_MEGATRON_BRIDGE_ROOT"),
    )
    _assert_source_under(
        "Gemma4 provider",
        payload["provider_source"],
        os.environ.get("SLIME_EXPECTED_MEGATRON_BRIDGE_ROOT"),
    )
    _assert_source_under(
        "Megatron-Core",
        payload["megatron_core_source"],
        os.environ.get("SLIME_EXPECTED_MEGATRON_ROOT"),
    )

    expected_mapping = os.environ.get("SLIME_EXPECTED_BRIDGE_MAPPING_SHA256")
    if expected_mapping and payload["mapping_sha256"] != expected_mapping:
        raise RuntimeError(
            "Gemma4 Bridge mapping drift: "
            f"actual={payload['mapping_sha256']} expected={expected_mapping}"
        )


def _write_bridge_audit(payload):
    audit_dir = os.environ.get("SLIME_GEMMA4_BRIDGE_AUDIT_DIR")
    if not audit_dir:
        return
    root = Path(audit_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    purpose = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(payload["purpose"]))
    target = root / f"{purpose}.rank{payload['rank'] or 'na'}.pid{os.getpid()}.{time.time_ns()}.json"
    tmp = target.with_name(f".{target.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def record_auto_bridge_audit(bridge, path, purpose, provider=None):
    """Fail closed on Bridge/source drift and optionally persist a fingerprint."""

    payload = _bridge_audit_payload(bridge, path, purpose, provider=provider)
    _validate_bridge_audit(payload)
    _write_bridge_audit(payload)
    return payload


def auto_bridge_from_hf_pretrained(path, *, purpose, **kwargs):
    """Construct every Slime AutoBridge through one audited code path."""

    from megatron.bridge import AutoBridge

    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(path, **kwargs))
    record_auto_bridge_audit(bridge, path, purpose)
    return bridge


@contextmanager
def patch_megatron_model(model, *, hide_value_head=False):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    hidden_output_layers = []
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # A critic replaces the policy LM head with a scalar value head.  It has no
    # matching tensor in the source HF policy checkpoint, so temporarily hide
    # only scalar output layers while Bridge loads/saves the backbone.
    language_model = getattr(unwrapped_model, "language_model", None)
    owners = (unwrapped_model, language_model) if language_model is not None else (unwrapped_model,)
    if hide_value_head:
        for owner in owners:
            output_layer = getattr(owner, "output_layer", None)
            if getattr(output_layer, "out_features", None) == 1:
                hidden_output_layers.append((owner, output_layer))
                owner.output_layer = None

    try:
        yield
    finally:
        for owner, output_layer in hidden_output_layers:
            owner.output_layer = output_layer
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
