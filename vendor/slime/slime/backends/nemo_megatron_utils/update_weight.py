"""NeMo Megatron-Bridge export to Slime's existing SGLang transport."""

from __future__ import annotations

import os
from typing import Iterable

import torch

from slime.backends.automodel_utils.update_weight import AutoModelWeightUpdater


class NemoMegatronWeightUpdater(AutoModelWeightUpdater):
    """Bucket official NeMo Bridge exports without defining any mappings."""

    def __init__(self, args, worker) -> None:
        super().__init__(args, worker)
        self._group_name = "slime-nemo-megatron"
        self._refit_prepared = False

    def _iter_exported_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield Bridge tensors plus SGLang-owned fixed Gemma4 buffers.

        NeMo's Gemma4 implementation intentionally implements V RMSNorm
        without a learnable scale, so the official Bridge has no Megatron
        parameter to export for ``v_norm.weight``.  SGLang represents that
        mathematically fixed all-one scale inside its weight memory region,
        though, and a refit must restore it after sleep/wake or a weight-check
        reset.  This mirrors Slime's native Gemma4 converter while leaving all
        trainable parameter mapping to the official Bridge.
        """

        sync_v_norm = os.environ.get("GEMMA4_SGLANG_SYNC_V_NORM", "0") == "1"
        seen_names: set[str] = set()
        pending_v_norms: dict[str, torch.Tensor] = {}

        for name, tensor in self.worker._iter_params_with_optional_kv_scales():
            seen_names.add(name)
            if sync_v_norm and name.endswith(".self_attn.k_norm.weight"):
                v_norm_name = name[: -len("k_norm.weight")] + "v_norm.weight"
                pending_v_norms[v_norm_name] = torch.ones_like(tensor)
            yield name, tensor

        # Defer synthetic buffers until the Bridge iterator is exhausted so a
        # future upstream Bridge that exports v_norm itself wins without a
        # duplicate update.
        for name, tensor in pending_v_norms.items():
            if name not in seen_names:
                yield name, tensor

    def _iter_buckets(self) -> Iterable[list[tuple[str, torch.Tensor]]]:
        if not self._refit_prepared:
            self.worker.prepare_refit_info()
            self._refit_prepared = True

        limit = int(self.args.update_weight_buffer_size)
        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        for name, tensor in self._iter_exported_tensors():
            tensor_bytes = tensor.numel() * tensor.element_size()
            if bucket and bucket_bytes + tensor_bytes > limit:
                yield bucket
                bucket = []
                bucket_bytes = 0
            bucket.append((name, tensor))
            bucket_bytes += tensor_bytes
        if bucket:
            yield bucket
