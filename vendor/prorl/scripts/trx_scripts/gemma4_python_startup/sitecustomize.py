"""Runtime-only compatibility and diagnostics for the Gemma4 path.

This module is deliberately used as a ``sitecustomize`` overlay.  SGLang
starts both its HTTP server and its scheduler/model workers with the Python
``spawn`` start method, so an in-memory patch installed only by Slime's Ray
actor would be lost.  The opt-in patches below are consequently reinstalled
by every fresh interpreter before model construction.
"""

import faulthandler
import functools
import importlib.abc
import json
import os

if os.environ.get("CLAWGYM_SELF_REWARD_CONFIG"):
    from clawgym_nanobot.compiler_cache import configure as _configure_compiler_cache
    _configure_compiler_cache()
from pathlib import Path
import sys
import threading
import time
import types


class _BlockUnusedFA4Import(importlib.abc.MetaPathFinder):
    """Keep AutoModel FFPA's CUTLASS DSL isolated from the image's FA4.

    Slime reuses Megatron's argument parser even for its AutoModel backend.
    Importing that parser probes ``flash_attn.cute`` at module import time,
    although no Megatron attention kernel is used by the AutoModel actors.
    The image's FA4 and upstream FFPA intentionally pin different CUTLASS
    Python APIs.  Reporting only the unused FA4 submodule as unavailable lets
    Megatron take its existing ``ImportError`` fallback without modifying
    either backend or hiding the ordinary ``flash_attn`` package.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flash_attn.cute" or fullname.startswith("flash_attn.cute."):
            raise ModuleNotFoundError(
                "flash_attn.cute is disabled for the AutoModel/FFPA runtime",
                name=fullname,
            )
        return None


if os.environ.get("SLIME_BLOCK_UNUSED_FA4_IMPORT", "0") == "1":
    sys.meta_path.insert(0, _BlockUnusedFA4Import())

try:
    from transformers import AutoConfig, Gemma4Config
except ImportError:
    pass
else:
    class Gemma4UnifiedConfig(Gemma4Config):
        model_type = "gemma4_unified"

    AutoConfig.register(
        Gemma4UnifiedConfig.model_type,
        Gemma4UnifiedConfig,
        exist_ok=True,
    )


def _schedule_scheduler_stack_dump() -> None:
    trace_dir = os.environ.get("GEMMA4_SCHEDULER_TRACE_DIR")
    if not trace_dir:
        return
    delay = int(os.environ.get("GEMMA4_SCHEDULER_TRACE_DELAY_S", "75"))

    def dump_if_scheduler() -> None:
        time.sleep(delay)
        try:
            process_title = Path("/proc/self/comm").read_text(encoding="utf-8").strip()
        except OSError:
            return
        output_dir = Path(trace_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{process_title.replace('/', '_')}-{os.getpid()}.stack"
        with output.open("w", encoding="utf-8") as stream:
            faulthandler.dump_traceback(file=stream, all_threads=True)

    threading.Thread(target=dump_if_scheduler, daemon=True).start()


_schedule_scheduler_stack_dump()


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _snapshot_or_restore_gemma4_static_buffers(model):
    """Keep small Gemma4 buffers alive across SGLang weights-region remaps.

    ``torch_memory_saver`` preserves parameter addresses, but without a full
    CPU copy the newly mapped storage does not contain non-persistent buffers.
    Gemma4 uses those buffers for RoPE caches and the fixed V RMSNorm scale.
    It also stores the learned checkpoint ``layer_scalar`` as a persistent
    buffer which is not part of Slime's parameter-only update iterator.

    The first checkpoint load snapshots only these small/derived tensors on
    CPU.  Every later dynamic load restores them after actor weights arrive.
    Shared RoPE modules are visited once because ``named_modules`` removes
    duplicate module objects by default.
    """

    import torch

    selected = {}
    for module_name, module in model.named_modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for buffer_name, buffer in module.named_buffers(recurse=False):
            full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name
            if buffer_name in non_persistent or buffer_name == "layer_scalar":
                selected[full_name] = buffer

    backup_attr = "_gemma4_static_buffer_cpu_backups"
    backups = getattr(model, backup_attr, None)
    if backups is None:
        backups = {
            name: buffer.detach().to(device="cpu", copy=True)
            for name, buffer in selected.items()
        }
        setattr(model, backup_attr, backups)
        return "snapshotted", tuple(sorted(backups))

    missing = sorted(set(backups) - set(selected))
    if missing:
        raise RuntimeError(
            "Gemma4 static buffers disappeared before restore: " + ", ".join(missing)
        )
    with torch.no_grad():
        for name, backup in backups.items():
            target = selected[name]
            if target.shape != backup.shape:
                raise RuntimeError(
                    f"Gemma4 static buffer shape changed for {name}: "
                    f"{tuple(target.shape)} != {tuple(backup.shape)}"
                )
            target.copy_(backup.to(device=target.device, dtype=target.dtype))
    return "restored", tuple(sorted(backups))


def _audit_gemma4_static_buffers(action, names, loaded) -> None:
    path = os.environ.get("GEMMA4_STATIC_BUFFER_AUDIT_PATH")
    if not path:
        return
    loaded_names = tuple(sorted(loaded or ()))
    record = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "action": action,
        "buffer_count": len(names),
        "buffer_names": names,
        "loaded_count": len(loaded_names),
        "loaded_v_norm": [name for name in loaded_names if ".v_norm.weight" in name],
        "loaded_layer_scalar": [name for name in loaded_names if name.endswith(".layer_scalar")],
    }
    try:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        # Diagnostics must never turn a healthy model load into a failure.
        pass


def _install_sglang_gemma4_sleep_state_patch() -> None:
    if not _enabled("GEMMA4_SGLANG_RESTORE_STATIC_BUFFERS"):
        return

    from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM

    if getattr(Gemma4ForCausalLM, "_gemma4_sleep_state_patched", False):
        return

    original_load_weights = Gemma4ForCausalLM.load_weights

    @functools.wraps(original_load_weights)
    def _load_weights_and_restore_static_buffers(self, weights):
        loaded = original_load_weights(self, weights)
        action, names = _snapshot_or_restore_gemma4_static_buffers(self)
        _audit_gemma4_static_buffers(action, names, loaded)
        return loaded

    Gemma4ForCausalLM.load_weights = _load_weights_and_restore_static_buffers
    Gemma4ForCausalLM._gemma4_sleep_state_patched = True


def _install_sglang_weight_check_constant_patch() -> None:
    """Keep Gemma4 inference-only constants stable during strict sync checks.

    SGLang's weight checker deliberately randomizes every parameter and buffer
    before asking the actor to restore the model.  Gemma4 has two intentional
    exceptions to the actor payload: ``layer_scalar`` is a fixed Megatron
    buffer, and ``v_norm`` is absent from the HF checkpoint and remains the
    all-one inference default.  Randomizing either turns a successful strict
    check into a false failure and leaves the checked model unusable.

    This opt-in patch skips only those constants during the randomization
    phase.  They are still compared against the pre-sync snapshot, while every
    trainable/synchronized tensor continues to be randomized and restored.
    """

    if not _enabled("GEMMA4_SGLANG_WEIGHT_CHECK_STABLE_CONSTANTS"):
        return

    from sglang.srt.utils.weight_checker import (
        WeightChecker,
        _is_non_persistent_buffer_name,
        _random_like,
    )

    if getattr(WeightChecker, "_gemma4_stable_constants_patched", False):
        return

    def _reset_syncable_tensors(self):
        for name, param in self._model_state():
            if _is_non_persistent_buffer_name(name):
                continue
            if name.endswith(".layer_scalar") or name.endswith(
                ".self_attn.v_norm.weight"
            ):
                continue
            param.copy_(_random_like(param))

    WeightChecker._reset_tensors = _reset_syncable_tensors
    WeightChecker._gemma4_stable_constants_patched = True


def _install_sglang_gemma4_alignment_patches() -> None:
    """Align known SGLang Gemma4 numerical details with HF/Megatron.

    Keep every adjustment independently gated so parity runs can bisect them.
    Class sentinels make this function idempotent when several launch layers
    import the overlay in the same interpreter.
    """

    flags = {
        "embed": _enabled("GEMMA4_SGLANG_ALIGN_EMBED_SCALE"),
        "window": _enabled("GEMMA4_SGLANG_ALIGN_SLIDING_WINDOW"),
        "positions": _enabled("GEMMA4_SGLANG_ALIGN_ZERO_BASED_POSITIONS"),
        "rope": _enabled("GEMMA4_SGLANG_ALIGN_CPU_GLOBAL_ROPE"),
        "softcap": _enabled("GEMMA4_SGLANG_ALIGN_BF16_SOFTCAP"),
        "head_dims": _enabled("GEMMA4_SGLANG_ALIGN_E_HEAD_DIMS"),
    }
    if not any(flags.values()):
        return

    import torch

    if flags["head_dims"]:
        from sglang.srt.models.gemma4_causal import Gemma4DecoderLayer

        if not getattr(Gemma4DecoderLayer, "_gemma4_e_head_dims_patched", False):
            original_decoder_init = Gemma4DecoderLayer.__init__

            @functools.wraps(original_decoder_init)
            def _e_head_dims_init(self, layer_id, config, *args, **kwargs):
                layer_type = config.layer_types[layer_id]
                global_head_dim = getattr(config, "global_head_dim", None)
                # Generated SGLang text views already translate HF's local
                # ``head_dim`` into SGLang's ``swa_head_dim`` and expose the
                # global width as ``head_dim``.  Only patch an un-translated
                # Hugging Face config here.
                if (
                    layer_type != "full_attention"
                    or not global_head_dim
                    or getattr(config, "swa_head_dim", None) is not None
                ):
                    return original_decoder_init(
                        self, layer_id, config, *args, **kwargs
                    )
                # SGLang <=0.5.13 uses config.head_dim for full attention.
                # Gemma4 E checkpoints instead store 512-wide global heads and
                # 256-wide sliding heads. Temporarily expose the global width
                # while this layer builds, then restore the shared config.
                sliding_head_dim = config.head_dim
                config.head_dim = global_head_dim
                try:
                    return original_decoder_init(
                        self, layer_id, config, *args, **kwargs
                    )
                finally:
                    config.head_dim = sliding_head_dim

            Gemma4DecoderLayer.__init__ = _e_head_dims_init
            Gemma4DecoderLayer._gemma4_e_head_dims_patched = True

    if flags["embed"]:
        from sglang.srt.models.gemma3_causal import Gemma3TextScaledWordEmbedding

        if not getattr(
            Gemma3TextScaledWordEmbedding, "_gemma4_hf_scale_patched", False
        ):

            def _hf_scale_forward(self, input_ids):
                output = torch.nn.Embedding.forward(self, input_ids)
                scale = torch.as_tensor(
                    self.embed_scale,
                    dtype=output.dtype,
                    device=output.device,
                )
                return output * scale

            Gemma3TextScaledWordEmbedding.forward = _hf_scale_forward
            Gemma3TextScaledWordEmbedding._gemma4_hf_scale_patched = True

    if flags["window"]:
        from sglang.srt.models.gemma4_causal import Gemma4Attention

        if not getattr(Gemma4Attention, "_gemma4_hf_window_patched", False):
            original_attention_init = Gemma4Attention.__init__

            @functools.wraps(original_attention_init)
            def _hf_window_init(self, *args, **kwargs):
                original_attention_init(self, *args, **kwargs)
                if self.sliding_window is not None:
                    expected = int(self.config.sliding_window) - 1
                    self.sliding_window = expected
                    self.attn.sliding_window_size = expected
                    if self.attn.sliding_window_size != expected:
                        raise RuntimeError("Gemma4 sliding-window alignment failed")

            Gemma4Attention.__init__ = _hf_window_init
            Gemma4Attention._gemma4_hf_window_patched = True

    if flags["positions"]:
        from sglang.srt.models.gemma4_mm import (
            Gemma4ForConditionalGeneration as MultimodalGemma4,
        )

        if not getattr(MultimodalGemma4, "_gemma4_zero_positions_patched", False):
            original_mm_forward = MultimodalGemma4.forward

            @functools.wraps(original_mm_forward)
            @torch.no_grad()
            def _zero_based_forward(
                self,
                input_ids,
                positions,
                forward_batch,
                input_embeds=None,
                pp_proxy_tensors=None,
                **kwargs,
            ):
                # The parent unconditionally adds one.  Supplying a fresh
                # tensor shifted by -1 makes the effective positions zero
                # based without mutating the caller's scheduling metadata.
                return original_mm_forward(
                    self,
                    input_ids,
                    positions - 1,
                    forward_batch,
                    input_embeds=input_embeds,
                    pp_proxy_tensors=pp_proxy_tensors,
                    **kwargs,
                )

            MultimodalGemma4.forward = _zero_based_forward
            MultimodalGemma4._gemma4_zero_positions_patched = True

    if flags["rope"]:
        from sglang.srt.layers.rotary_embedding.rope_variant import (
            Gemma4RotaryEmbedding,
        )

        if not getattr(Gemma4RotaryEmbedding, "_gemma4_cpu_freq_patched", False):

            def _cpu_inv_freq(self, base):
                # HF/Megatron construct this table on CPU before copying it to
                # the model device.  Building it directly on CUDA gives
                # observably different BF16 cos/sin values at long positions.
                target_device = torch.empty(0).device
                exponents = torch.arange(
                    0,
                    2 * self.rope_angles,
                    2,
                    dtype=torch.float32,
                    device="cpu",
                ) / float(self.head_size)
                inv_freq = 1.0 / (float(base) ** exponents)
                if self.nope_angles > 0:
                    inv_freq = torch.cat(
                        (
                            inv_freq,
                            torch.zeros(
                                self.nope_angles,
                                dtype=torch.float32,
                                device="cpu",
                            ),
                        )
                    )
                return inv_freq.to(target_device)

            Gemma4RotaryEmbedding._compute_inv_freq = _cpu_inv_freq
            Gemma4RotaryEmbedding._gemma4_cpu_freq_patched = True

    if flags["softcap"]:
        from sglang.srt.models.gemma4_unified import (
            Gemma4UnifiedForConditionalGeneration,
        )

        if not getattr(
            Gemma4UnifiedForConditionalGeneration,
            "_gemma4_bf16_softcap_patched",
            False,
        ):
            original_unified_init = Gemma4UnifiedForConditionalGeneration.__init__

            @functools.wraps(original_unified_init)
            def _aligned_unified_init(self, *args, **kwargs):
                original_unified_init(self, *args, **kwargs)
                processor = self.logits_processor
                cap = processor.final_logit_softcapping
                if not cap:
                    return
                original_copy = processor._copy_logits_to_buffer

                def _softcap_then_copy(this, logits, logits_metadata):
                    # HF and the Megatron actor apply the three softcap ops in
                    # model dtype.  SGLang normally upcasts logits to FP32 in
                    # _copy_logits_to_buffer before applying its fused kernel.
                    logits.div_(float(cap))
                    logits.tanh_()
                    logits.mul_(float(cap))
                    return original_copy(logits, logits_metadata)

                processor._copy_logits_to_buffer = types.MethodType(
                    _softcap_then_copy, processor
                )
                processor.final_logit_softcapping = None
                processor._gemma4_bf16_softcap_installed = True

            Gemma4UnifiedForConditionalGeneration.__init__ = _aligned_unified_init
            Gemma4UnifiedForConditionalGeneration._gemma4_bf16_softcap_patched = True


_install_sglang_gemma4_alignment_patches()
_install_sglang_weight_check_constant_patch()
_install_sglang_gemma4_sleep_state_patch()
