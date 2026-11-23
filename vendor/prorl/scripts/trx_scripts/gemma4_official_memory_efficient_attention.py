"""Memory-bounded local attention for the official Megatron Gemma-4 provider.

This module deliberately changes only the core attention implementation.  The
official Megatron-Bridge Gemma-4 model, tensor-parallel projections, RoPE,
normalization, and checkpoint conversion remain untouched.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.utils import is_layer_window_attention


_compiled_flex_attention = torch.compile(flex_attention)
_flex_block_mask_cache = {}


def _get_flex_block_mask(
    query_length: int,
    key_length: int,
    window_left: int,
    device: torch.device,
):
    cache_key = (query_length, key_length, window_left, str(device))
    if cache_key in _flex_block_mask_cache:
        return _flex_block_mask_cache[cache_key]

    query_offset = key_length - query_length
    if window_left < 0:

        def mask_mod(_batch, _head, query_index, key_index):
            return query_index + query_offset >= key_index

    else:

        def mask_mod(_batch, _head, query_index, key_index):
            absolute_query_index = query_index + query_offset
            return (absolute_query_index >= key_index) & (
                absolute_query_index - key_index <= window_left
            )

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=query_length,
        KV_LEN=key_length,
        device=device,
    )
    _flex_block_mask_cache[cache_key] = block_mask
    return block_mask


class Gemma4MemoryEfficientAttention(nn.Module):
    """Exact causal Gemma-4 attention with bounded peak memory on A100.

    Sliding layers use FlashAttention-2 (head_dim=256).  Global layers have a
    512-wide head, unsupported by FlashAttention-2 on A100, so they use exact
    PyTorch SDPA in query chunks.  K/V stay complete and the mask accounts for
    each chunk's absolute query offset; this is not truncated attention.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type,
        attention_type: str = "self",
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del attn_mask_type, attention_type, kwargs
        if int(getattr(config, "context_parallel_size", 1) or 1) != 1:
            raise NotImplementedError("this PP1 probe adapter currently requires CP=1")

        self.config = config
        self.layer_number = layer_number
        configured_scale = getattr(config, "softmax_scale", None)
        self.softmax_scale = softmax_scale if softmax_scale is not None else configured_scale
        self.dropout_p = float(
            config.attention_dropout if attention_dropout is None else attention_dropout
        )
        self.is_sliding = is_layer_window_attention(
            config.window_size,
            config.window_attn_skip_freq,
            layer_number,
        )
        self.backend = os.environ.get("GEMMA4_MEMORY_EFFICIENT_BACKEND", "eager_chunked")
        if self.backend not in frozenset({"eager_chunked", "flex", "hybrid"}):
            raise ValueError(
                "GEMMA4_MEMORY_EFFICIENT_BACKEND must be eager_chunked, flex, or hybrid"
            )

    def _scale(self, head_dim: int) -> float:
        if self.softmax_scale is not None:
            return float(self.softmax_scale)
        return head_dim**-0.5

    def _sliding_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        flash_attn = __import__("flash_attn", fromlist=("flash_attn_func",), level=0)
        flash_attn_func = flash_attn.flash_attn_func
        window = getattr(self.config, "window_size", None)
        if window is None:
            window = (-1, -1)
        elif isinstance(window, int):
            window = (int(window) - 1, 0)
        else:
            window = (int(window[0]), int(window[1]))

        q = query.permute(1, 0, 2, 3).contiguous()
        k = key.permute(1, 0, 2, 3).contiguous()
        v = value.permute(1, 0, 2, 3).contiguous()
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self._scale(query.shape[-1]),
            causal=True,
            window_size=window,
        )
        return out.permute(1, 0, 2, 3).reshape(query.shape[0], query.shape[1], -1)

    def _flex_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        window_left = -1
        if self.is_sliding:
            window = getattr(self.config, "window_size", None)
            if isinstance(window, int):
                window_left = int(window) - 1
            elif window is not None:
                window_left = int(window[0])
        block_mask = _get_flex_block_mask(
            q.shape[-2],
            k.shape[-2],
            window_left,
            query.device,
        )
        out = _compiled_flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=self._scale(query.shape[-1]),
            enable_gqa=q.shape[1] != k.shape[1],
            kernel_options={
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "num_warps": 4,
                "num_stages": 1,
                "PRESCALE_QK": False,
                "ROWS_GUARANTEED_SAFE": True,
                "BLOCKS_ARE_CONTIGUOUS": True,
            },
        )
        return out.permute(2, 0, 1, 3).contiguous().reshape(
            query.shape[0], query.shape[1], -1
        )

    def _global_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)
        query_heads = q.shape[1]
        key_heads = k.shape[1]
        sequence_length = q.shape[-2]
        scale = self._scale(query.shape[-1])
        chunk_size = int(os.environ.get("GEMMA4_GLOBAL_ATTN_QUERY_CHUNK_SIZE", "512"))
        if chunk_size <= 0:
            raise ValueError("GEMMA4_GLOBAL_ATTN_QUERY_CHUNK_SIZE must be positive")

        chunks = []
        for query_start in range(0, sequence_length, chunk_size):
            query_end = min(query_start + chunk_size, sequence_length)
            q_chunk = q[:, :, query_start:query_end, :]

            def attend_chunk(
                q_block: Tensor,
                k_full: Tensor,
                v_full: Tensor,
                offset: int = query_start,
            ) -> Tensor:
                rows = torch.arange(
                    offset,
                    offset + q_block.shape[-2],
                    device=q_block.device,
                )
                columns = torch.arange(sequence_length, device=q_block.device)
                allowed = columns.unsqueeze(0) <= rows.unsqueeze(1)
                return F.scaled_dot_product_attention(
                    q_block,
                    k_full,
                    v_full,
                    attn_mask=allowed[None, None, :, :],
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=scale,
                    enable_gqa=query_heads != key_heads,
                )

            if torch.is_grad_enabled() and any(
                tensor.requires_grad for tensor in (q_chunk, k, v)
            ):
                chunk = checkpoint(attend_chunk, q_chunk, k, v, use_reentrant=False)
            else:
                chunk = attend_chunk(q_chunk, k, v)
            chunks.append(chunk)

        out = torch.cat(chunks, dim=-2)
        return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)

    def _eager_chunked_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Chunk the official BF16 eager attention without changing its math.

        Unlike SDPA/Flash/Flex, this deliberately retains Megatron's
        baddbmm -> masked_fill(-10000) -> torch.softmax -> bmm sequence.  The
        query chunks only bound memory; every global key remains present.  On
        sliding layers, keys that are masked to exactly zero probability are
        omitted before the same eager operations.
        """

        sequence_length, batch_size, query_heads, head_dim = query.shape
        key_heads = key.shape[2]
        if query_heads % key_heads:
            raise ValueError("query head count must be divisible by KV head count")
        if query_heads != key_heads:
            repeat = query_heads // key_heads
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        chunk_size = int(os.environ.get("GEMMA4_EAGER_ATTN_QUERY_CHUNK_SIZE", "256"))
        if chunk_size <= 0:
            raise ValueError("GEMMA4_EAGER_ATTN_QUERY_CHUNK_SIZE must be positive")
        window_left = -1
        if self.is_sliding:
            window = getattr(self.config, "window_size", None)
            if isinstance(window, int):
                window_left = int(window) - 1
            elif window is not None:
                window_left = int(window[0])

        outputs = []
        for query_start in range(0, sequence_length, chunk_size):
            query_end = min(query_start + chunk_size, sequence_length)
            key_start = 0 if window_left < 0 else max(0, query_start - window_left)
            key_end = sequence_length if window_left < 0 else query_end
            q_block = query[query_start:query_end]
            k_block = key[key_start:key_end]
            v_block = value[key_start:key_end]

            def attend_chunk(
                q_part: Tensor,
                k_part: Tensor,
                v_part: Tensor,
                q_offset: int = query_start,
                k_offset: int = key_start,
            ) -> Tensor:
                query_count = q_part.shape[0]
                key_count = k_part.shape[0]
                q_flat = q_part.reshape(query_count, batch_size * query_heads, head_dim)
                k_flat = k_part.reshape(key_count, batch_size * query_heads, head_dim)
                scores = torch.baddbmm(
                    torch.empty(
                        batch_size * query_heads,
                        query_count,
                        key_count,
                        dtype=q_part.dtype,
                        device=q_part.device,
                    ),
                    q_flat.transpose(0, 1),
                    k_flat.transpose(0, 1).transpose(1, 2),
                    beta=0.0,
                    alpha=self._scale(head_dim),
                ).view(batch_size, query_heads, query_count, key_count)
                query_positions = torch.arange(
                    q_offset,
                    q_offset + query_count,
                    device=q_part.device,
                )
                key_positions = torch.arange(
                    k_offset,
                    k_offset + key_count,
                    device=q_part.device,
                )
                forbidden = key_positions[None, :] > query_positions[:, None]
                if window_left >= 0:
                    forbidden = forbidden | (
                        key_positions[None, :] < query_positions[:, None] - window_left
                    )
                scores.masked_fill_(forbidden[None, None, :, :], -10000.0)
                probabilities = torch.softmax(scores, dim=-1)
                probabilities = probabilities.view(
                    batch_size * query_heads,
                    query_count,
                    key_count,
                )
                v_flat = v_part.reshape(key_count, batch_size * query_heads, head_dim)
                context = torch.bmm(probabilities, v_flat.transpose(0, 1))
                return context.view(
                    batch_size,
                    query_heads,
                    query_count,
                    head_dim,
                ).permute(2, 0, 1, 3)

            if torch.is_grad_enabled() and any(
                tensor.requires_grad for tensor in (q_block, k_block, v_block)
            ):
                output = checkpoint(
                    attend_chunk,
                    q_block,
                    k_block,
                    v_block,
                    use_reentrant=False,
                )
            else:
                output = attend_chunk(q_block, k_block, v_block)
            outputs.append(output)

        return torch.cat(outputs, dim=0).contiguous().view(
            sequence_length,
            batch_size,
            query_heads * head_dim,
        )

    def _eager_chunked_thd_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params,
    ) -> Tensor:
        if packed_seq_params.qkv_format != "thd":
            raise ValueError(f"unsupported packed QKV format: {packed_seq_params.qkv_format}")
        cu_seqlens = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
        if cu_seqlens is None or cu_seqlens_kv is None:
            raise ValueError("packed Gemma4 attention requires Q and KV cumulative lengths")
        q_offsets = cu_seqlens.detach().cpu().tolist()
        kv_offsets = cu_seqlens_kv.detach().cpu().tolist()
        if q_offsets != kv_offsets:
            raise NotImplementedError("Gemma4 self-attention requires matching packed Q/KV lengths")

        outputs = []
        for start, end in zip(q_offsets[:-1], q_offsets[1:], strict=True):
            if end <= start:
                continue
            sequence_output = self._eager_chunked_forward(
                query[start:end].unsqueeze(1),
                key[start:end].unsqueeze(1),
                value[start:end].unsqueeze(1),
            )
            outputs.append(
                sequence_output.squeeze(1).view(end - start, query.shape[1], query.shape[2])
            )
        if not outputs:
            return query.new_empty(query.shape)
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        **kwargs,
    ) -> Tensor:
        del attn_mask_type, kwargs
        if attention_bias is not None:
            raise NotImplementedError("attention_bias is not used by Gemma-4")
        if attention_mask is not None:
            raise NotImplementedError("the PP1 probe expects implicit causal masking")
        if packed_seq_params is not None:
            if self.backend != "eager_chunked":
                raise NotImplementedError("packed Gemma4 attention currently uses eager_chunked")
            if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
                raise ValueError("expected Megatron THD [T, H, D] Q/K/V tensors")
            if self.training and self.dropout_p:
                raise NotImplementedError("eager chunked path requires attention_dropout=0")
            return self._eager_chunked_thd_forward(query, key, value, packed_seq_params)

        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("expected Megatron [S, B, H, D] Q/K/V tensors")
        if self.backend == "eager_chunked":
            if self.training and self.dropout_p:
                raise NotImplementedError("eager chunked path requires attention_dropout=0")
            return self._eager_chunked_forward(query, key, value)
        if self.backend == "flex":
            if self.training and self.dropout_p:
                raise NotImplementedError("FlexAttention path requires attention_dropout=0")
            return self._flex_forward(query, key, value)
        if self.is_sliding:
            return self._sliding_forward(query, key, value)
        return self._global_forward(query, key, value)
