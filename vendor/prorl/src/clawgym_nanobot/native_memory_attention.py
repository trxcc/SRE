"""Opt-in native Gemma core using upstream D512 FFPA Triton / D256 windowed FA2.

This module contains no weights, reward logic or model conversion. Unsupported
masks/configurations fail closed; there is no eager-attention fallback.
"""
from __future__ import annotations
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import weakref
import torch
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import is_layer_window_attention

_MASK_CACHE = {}


def _require(value, message):
    if not value:
        raise ValueError(message)


def validate_native_causal_mask(mask, length, batch):
    """Verify the exact native no-reset causal mask; never discard other masks.

    The current NeMo CP1 unpacked producer returns a shared [1,1,S,S] bool
    causal mask. Input lengths right-pad model rows; they do not insert key
    padding into this producer's attention mask. Validate every entry once,
    in bounded strips, rather than allocating another S*S temporary.
    """
    if mask is None:
        return
    _require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool,
             "native core requires bool causal mask or None")
    _require(mask.ndim == 4 and mask.shape[0] in (1, batch)
             and tuple(mask.shape[1:]) == (1, length, length),
             "unsupported native attention mask shape")
    cache = _MASK_CACHE.get(id(mask))
    if cache and cache[0]() is mask and cache[1] == mask._version:
        return
    cols = torch.arange(length, device=mask.device)[None, :]
    for begin in range(0, length, 64):
        rows = torch.arange(begin, min(begin + 64, length), device=mask.device)[:, None]
        expected = cols > rows
        _require(bool(torch.all(mask[:, 0, begin:begin + 64, :] == expected).item()),
                 "mask differs from native no-reset causal contract")
    key = id(mask)
    ref = weakref.ref(mask, lambda old: _MASK_CACHE.pop(key, None)
                      if _MASK_CACHE.get(key, (None,))[0] is old else None)
    _MASK_CACHE[key] = (ref, mask._version)


def kernel_attention(query, key, value, *, scale, window):
    """SBHD → upstream native kernel → SB(HD), preserving native QK scale."""
    _require(query.ndim == key.ndim == value.ndim == 4, "expected SBHD tensors")
    _require(key.shape == value.shape and query.shape[:2] == key.shape[:2]
             and query.shape[-1] == key.shape[-1]
             and query.shape[2] % key.shape[2] == 0,
             "unsupported native self-attention/GQA layout")
    _require(query.dtype == key.dtype == value.dtype
             and query.dtype in (torch.bfloat16, torch.float16), "expected bf16/fp16")
    _require(query.device == key.device == value.device and query.is_cuda,
             "upstream attention kernels require same CUDA device")
    _require(math.isfinite(float(scale)) and scale > 0, "invalid explicit native scale")
    if query.shape[-1] == 512 and window is None:
        from ffpa_attn import ffpa_attn_func
        from ffpa_attn.functional import TritonBackend, FFPAAttnMeta
        q, k, v = (x.permute(1, 2, 0, 3).contiguous() for x in (query, key, value))
        # The upstream Triton API otherwise sends S<512 to dense SDPA.
        # Pad Q/K/V features only at the future end of a causal kernel call,
        # then crop outputs. No original query can attend to added future keys;
        # no sample, token stream, loss mask, candidate or policy budget changes.
        original_length = q.shape[2]
        if original_length < 512:
            pad = 512 - original_length
            q, k, v = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        backend = TritonBackend(grad_kv_storage_dtype=torch.float32,
                                grad_q_storage_dtype=torch.float32)
        _require(not FFPAAttnMeta.from_kwargs(backend=backend).fallback(q, k, None, 0.0),
                 "upstream fast attention would fall back to SDPA")
        result = ffpa_attn_func(q, k, v, attn_mask=None, dropout_p=0.0,
                                is_causal=True, scale=float(scale), enable_gqa=True,
                                backend=backend)
        result = result[:, :, :original_length].permute(2, 0, 1, 3).contiguous()
    elif query.shape[-1] == 256 and window == (511, 0):
        from flash_attn import flash_attn_func
        q, k, v = (x.transpose(0, 1).contiguous() for x in (query, key, value))
        result = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=float(scale),
                                 causal=True, window_size=window, softcap=0.0)
        result = result.transpose(0, 1).contiguous()
    else:
        raise ValueError("unsupported Gemma head dimension/window; no math fallback")
    _require(tuple(result.shape) == tuple(query.shape), "upstream kernel output layout changed")
    return result.flatten(2)


class NativeMemoryAttention(DotProductAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config
        _require(os.environ.get("CLAWGYM_NATIVE_MEMORY_EFFICIENT_ATTENTION") == "1",
                 "native memory core requires explicit opt-in")
        _require(cfg.context_parallel_size == 1 and cfg.softmax_type == "vanilla"
                 and not cfg.apply_query_key_layer_scaling,
                 "unsupported CP/softmax/layer scaling")
        _require(self.attn_mask_type == AttnMaskType.causal
                 and self.attention_dropout.p == 0 and self.softmax_offset is None,
                 "unsupported attention mask/dropout/offset")
        _require(float(self.softmax_scale) == 1.0, "Gemma native QK scale changed")
        self.native_window = (tuple(cfg.window_size)
                              if is_layer_window_attention(cfg.window_size,
                                 cfg.window_attn_skip_freq, self.layer_number) else None)
        _require((cfg.kv_channels, self.native_window) in ((512, None), (256, (511, 0))),
                 "unsupported Gemma attention layer configuration")

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        _require(attention_bias is None and packed_seq_params is None,
                 "attention bias/sequence packing not supported by CP1 core")
        _require(attn_mask_type in (None, AttnMaskType.causal), "mask type changed")
        validate_native_causal_mask(attention_mask, query.shape[0], query.shape[1])
        return kernel_attention(query, key, value, scale=self.softmax_scale,
                                window=self.native_window)


def install_native_core_binding():
    """Change only the parameter-free core in the run's native Gemma spec."""
    value = os.environ.get("CLAWGYM_NATIVE_MEMORY_EFFICIENT_ATTENTION", "0")
    _require(value in ("0", "1"), "invalid native attention opt-in")
    if value == "0":
        return {"enabled": False}
    cfg = json.loads(Path(os.environ["CLAWGYM_SELF_REWARD_CONFIG"]).read_text())
    _require(cfg.get("profile") == {"dataset": "researchclaw", "model": "gemma4-e4b-it"}
             and (cfg.get("candidate_count"), cfg.get("mode")) in {(8, "self_reward_7plus1_train7"), (4, "self_reward_3plus1_train3")}, "wrong native profile")
    from megatron.bridge.models.gemma import modeling_gemma4 as modeling
    from megatron.bridge.models.gemma import gemma4_provider as provider
    filename = Path(inspect.getsourcefile(modeling)).resolve()
    _require(hashlib.sha256(filename.read_bytes()).hexdigest() ==
             "c1038171569e71c557168556d5f19f876fc6411c75f7fad898097bf510fbd8f8",
             "native Gemma spec source changed")
    if getattr(provider.get_gemma4_layer_spec, "_researchclaw_memory_core", False):
        _require(provider.get_gemma4_layer_spec is modeling.get_gemma4_layer_spec,
                 "native spec bindings diverged")
        return {"enabled": True, "already_bound": True}
    original = provider.get_gemma4_layer_spec
    _require(original is modeling.get_gemma4_layer_spec, "native spec binding changed")
    def bound(config=None):
        spec = original(config)
        attention = spec.submodules.self_attention.submodules
        _require(attention.core_attention is DotProductAttention,
                 "unexpected native attention core")
        attention.core_attention = NativeMemoryAttention
        return spec
    bound._researchclaw_memory_core = True
    provider.get_gemma4_layer_spec = modeling.get_gemma4_layer_spec = bound
    return {"enabled": True, "native_spec_source_SHA": hashlib.sha256(filename.read_bytes()).hexdigest(),
            "core": "native_D512_FFPA_and_D256_window511_FA2", "other_spec_fields_preserved": True}
