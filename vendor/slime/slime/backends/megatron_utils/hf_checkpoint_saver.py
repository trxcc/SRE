import builtins
import json
import logging
import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_HF_WEIGHT_FILE_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "flax_model.msgpack",
}
_HF_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack")


@contextmanager
def _suppress_expected_gemma4_non_text_report(omitted_keys: set[str] | None):
    """Hide Bridge's verbose report for an already-validated text-only export.

    Megatron Bridge compares a partial Gemma4 text export with the original
    multimodal shard and prints every omitted vision tensor when
    ``strict=False``.  The caller separately validates that every expected
    language-model tensor was written, so this report is expected noise.  The
    filter is deliberately key-aware: missing text keys and unrelated Bridge
    diagnostics remain visible and still fail post-save validation.
    """

    if not omitted_keys:
        yield
        return

    original_print = builtins.print
    omitted_count = len(omitted_keys)
    expected_header = (
        "Warning: The following files are different from the source because the generator did not yield all "
        "of their tensors. However they are still saved because strict=False."
    )
    expected_final = (
        f"Error: {omitted_count} tensors from the original checkpoint were not written. "
        "See warnings above for details."
    )

    def filtered_print(*values, **kwargs):
        separator = kwargs.get("sep", " ")
        rendered = separator.join(str(value) for value in values)
        stripped = rendered.strip()
        if stripped == expected_header or stripped == expected_final:
            return
        if stripped.startswith("- "):
            reported_key = stripped[2:]
            if reported_key in omitted_keys:
                return
            if ": missing " in stripped and stripped.endswith(" tensors:"):
                # The individual missing keys below remain visible unless
                # they are members of ``omitted_keys``.
                return
        original_print(*values, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


def save_hf_model_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
    megatron_local_weights=None,
    hide_value_head: bool = False,
) -> None:
    """Save a Megatron model as an HF checkpoint at a concrete directory."""
    if args.megatron_to_hf_mode == "bridge":
        if megatron_local_weights is None:
            save_hf_model_bridge_to_path(
                args, output_dir, model, hide_value_head=hide_value_head
            )
        else:
            save_hf_model_bridge_snapshot_to_path(
                args,
                output_dir,
                model,
                megatron_local_weights,
                model_name=model_name,
                quantization_config=quantization_config,
                progress_desc=progress_desc,
            )
    else:
        if hide_value_head:
            raise NotImplementedError(
                "critic HF export requires --megatron-to-hf-mode bridge"
            )
        save_hf_model_direct_to_path(
            args,
            output_dir,
            model,
            model_name=model_name,
            quantization_config=quantization_config,
            progress_desc=progress_desc,
        )


def save_hf_model_direct_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
) -> None:
    """Save a Megatron model as an HF safetensors checkpoint without Megatron Bridge."""
    path = Path(output_dir)
    hf_checkpoint = Path(args.hf_checkpoint).resolve()
    save_path = path.resolve()
    if hf_checkpoint == save_path:
        raise ValueError("HF save output path must not point to the same directory as --hf-checkpoint")
    if not hf_checkpoint.is_dir():
        raise ValueError(
            f"--hf-checkpoint must be a local directory when saving raw HuggingFace weights: {args.hf_checkpoint}"
        )

    import torch.distributed as dist
    from transformers import AutoConfig

    from .update_weight.common import named_params_and_buffers
    from .update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    is_save_rank = _is_global_rank_zero()

    setup_error = None
    if is_save_rank:
        try:
            logger.info("Saving model in HuggingFace format to %s with raw Megatron-to-HF conversion", path)
            path.mkdir(parents=True, exist_ok=True)
            _clear_existing_hf_weights(path)
            _copy_hf_assets(args.hf_checkpoint, path)
        except Exception as e:
            setup_error = repr(e)

    _raise_if_rank_zero_failed("prepare raw HuggingFace save directory", setup_error)

    metadata_error = None
    payload: list[Any] = [None]
    if model_name is not None:
        payload = [(model_name, quantization_config)]
    else:
        if is_save_rank:
            try:
                hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                payload = [
                    (
                        type(hf_config).__name__.lower() if args.model_name is None else args.model_name,
                        getattr(hf_config, "quantization_config", None),
                    )
                ]
            except Exception as e:
                metadata_error = repr(e)
        _raise_if_rank_zero_failed("load HuggingFace conversion metadata", metadata_error)

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    model_name, quantization_config = payload[0]

    hf_weight_iterator = HfWeightIteratorDirect(
        args=args,
        model=model,
        model_name=model_name,
        quantization_config=quantization_config,
    )
    megatron_local_weights = dict(named_params_and_buffers(args, model, convert_to_global_name=True))
    num_save_nodes, save_node_rank, is_writer_rank, writer_ranks = _get_node_save_layout(args)
    if is_save_rank:
        logger.info(
            "Raw HuggingFace save will write shards from %d node writer rank(s): %s",
            num_save_nodes,
            writer_ranks,
        )

    writer = _SafetensorShardWriter(path, enabled=is_writer_rank)
    pending_write = None

    for chunk_idx, hf_named_tensors in enumerate(
        hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights, progress_desc=progress_desc)
    ):
        if is_writer_rank and chunk_idx % num_save_nodes == save_node_rank:
            pending_write = (chunk_idx, hf_named_tensors)
            hf_named_tensors = None
        else:
            del hf_named_tensors

        if (chunk_idx + 1) % num_save_nodes == 0:
            pending_write = _write_pending_chunk(writer, pending_write)

    pending_write = _write_pending_chunk(writer, pending_write)
    _finalize_distributed_shards(path, writer.state())

    if is_save_rank:
        logger.info("Successfully saved HuggingFace model to %s", path)


def save_hf_model_bridge_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    hide_value_head: bool = False,
) -> None:
    """Save a Megatron model as an HF checkpoint through Megatron Bridge."""
    import torch.distributed as dist
    from megatron.core import mpu

    from slime.utils.megatron_bridge_utils import auto_bridge_from_hf_pretrained, patch_megatron_model

    path = Path(output_dir)
    should_log = (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
    )
    if should_log:
        logger.info("Saving model in HuggingFace format to %s with Megatron Bridge", path)

    path.mkdir(parents=True, exist_ok=True)
    bridge = auto_bridge_from_hf_pretrained(
        args.hf_checkpoint,
        purpose="hf_checkpoint_save",
        trust_remote_code=True,
    )
    strict, expected_text_keys = _bridge_text_export_contract(bridge)
    if hide_value_head:
        strict = False
    source_keys = set(bridge.hf_pretrained.state.source.get_all_keys())
    omitted_non_text_keys = (
        source_keys - expected_text_keys
        if expected_text_keys is not None and not strict
        else None
    )
    if should_log and expected_text_keys is not None:
        logger.info(
            "Gemma4 text-only HF export contract: source=%d text=%d omitted_non_text=%d strict=%s",
            len(bridge.hf_pretrained.state.source.get_all_keys()),
            len(expected_text_keys),
            len(bridge.hf_pretrained.state.source.get_all_keys()) - len(expected_text_keys),
            strict,
        )

    with patch_megatron_model(model, hide_value_head=hide_value_head):
        with _suppress_expected_gemma4_non_text_report(omitted_non_text_keys):
            bridge.save_hf_pretrained(
                model,
                path=path,
                strict=strict,
            )

    validation_error = None
    if _is_global_rank_zero() and expected_text_keys is not None:
        try:
            _validate_bridge_text_checkpoint(path, expected_text_keys)
        except Exception as e:
            validation_error = repr(e)
    _raise_if_rank_zero_failed("validate Gemma4 text-only HuggingFace checkpoint", validation_error)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if should_log:
        logger.info("Successfully saved HuggingFace model to %s", path)


def _scalar_value_head(model):
    """Return the single replicated hidden-to-one critic head."""

    from megatron.core.utils import unwrap_model

    chunks = unwrap_model(model)
    if not isinstance(chunks, list):
        chunks = [chunks]
    matches = []
    seen = set()
    for chunk in chunks:
        for owner in (chunk, getattr(chunk, "language_model", None)):
            if owner is None:
                continue
            head = getattr(owner, "output_layer", None)
            if getattr(head, "out_features", None) == 1 and id(head) not in seen:
                seen.add(id(head))
                matches.append(head)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one scalar critic head, found {len(matches)}")
    return matches[0]


def save_hf_critic_to_path(args, output_dir: str | Path, model) -> None:
    """Save an HF backbone plus a reloadable scalar value-head sidecar."""

    import torch.distributed as dist

    path = Path(output_dir)
    save_hf_model_to_path(
        args,
        path,
        model,
        progress_desc="Save HF critic checkpoint",
        hide_value_head=True,
    )
    head = _scalar_value_head(model)
    if _is_global_rank_zero():
        from safetensors.torch import save_file

        state = {"value_head.weight": head.weight.detach().cpu().contiguous()}
        if head.bias is not None:
            state["value_head.bias"] = head.bias.detach().cpu().contiguous()
        save_file(state, path / "critic_head.safetensors")
        config = {
            "format": "huggingface_critic_v1",
            "backbone_format": "huggingface",
            "head_file": "critic_head.safetensors",
            "head_weight_key": "value_head.weight",
            "hidden_size": int(head.in_features),
            "output_size": int(head.out_features),
        }
        temporary = path / f".critic_config.json.tmp.{os.getpid()}"
        temporary.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path / "critic_config.json")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_hf_critic_value_head(hf_checkpoint: str | Path, model) -> bool:
    """Restore a critic-head sidecar after Bridge loads the HF backbone."""

    path = Path(hf_checkpoint)
    config_path = path / "critic_config.json"
    if not config_path.is_file() and not (path / "critic_head.safetensors").is_file():
        return False
    if not config_path.is_file():
        raise RuntimeError(f"incomplete HF critic checkpoint: missing {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "huggingface_critic_v1":
        raise RuntimeError(f"unsupported HF critic format: {config.get('format')!r}")
    head_path = path / str(config.get("head_file", "critic_head.safetensors"))
    if not head_path.is_file():
        raise RuntimeError(f"incomplete HF critic checkpoint: missing {head_path}")

    from safetensors.torch import load_file

    state = load_file(str(head_path), device="cpu")
    head = _scalar_value_head(model)
    weight_key = str(config.get("head_weight_key", "value_head.weight"))
    weight = state.get(weight_key)
    if weight is None or tuple(weight.shape) != tuple(head.weight.shape):
        raise RuntimeError(
            "HF critic value-head weight shape mismatch: "
            f"checkpoint={None if weight is None else tuple(weight.shape)} "
            f"model={tuple(head.weight.shape)}"
        )
    with torch.no_grad():
        head.weight.copy_(weight.to(device=head.weight.device, dtype=head.weight.dtype))
        if head.bias is not None:
            bias = state.get("value_head.bias")
            if bias is None or tuple(bias.shape) != tuple(head.bias.shape):
                raise RuntimeError("HF critic value-head bias shape mismatch")
            head.bias.copy_(bias.to(device=head.bias.device, dtype=head.bias.dtype))
        elif "value_head.bias" in state:
            raise RuntimeError("HF critic sidecar has a bias but the model value head does not")
    return True


def _bridge_text_export_contract(bridge) -> tuple[bool, set[str] | None]:
    """Return Bridge strictness and expected keys for the Gemma4 VL text view.

    The official ``Gemma4VLBridge`` deliberately maps a full multimodal HF
    checkpoint to a dense text-only Megatron model when
    ``GEMMA4_CONVERSION_MODE=text``.  The source safetensor can therefore
    contain vision/audio tensors which the official text mapping must not
    produce.  Preserve strict Bridge saving everywhere else; for this exact
    partial-export contract, allow the source-only tensors and validate the
    complete text key set after the streaming save.
    """

    model_bridge = bridge._model_bridge
    conversion_mode = os.environ.get("GEMMA4_CONVERSION_MODE", "auto").lower()
    if conversion_mode != "text" or type(model_bridge).__name__ != "Gemma4VLBridge":
        return True, None

    source = getattr(getattr(bridge.hf_pretrained, "state", None), "source", None)
    if source is None or not hasattr(source, "get_all_keys"):
        raise RuntimeError("Gemma4 text-only Bridge export requires an inspectable HF tensor source")

    prefix = model_bridge._hf_layer_prefix()
    if prefix != "model.language_model.":
        raise RuntimeError(f"Unexpected Gemma4 text-only HF prefix: {prefix!r}")

    source_keys = set(source.get_all_keys())
    expected_text_keys = {name for name in source_keys if name.startswith(prefix)}
    if not expected_text_keys:
        raise RuntimeError(f"Gemma4 text-only Bridge source has no tensors under {prefix!r}")

    # A checkpoint previously produced by this path already contains only the
    # text tensors, so subsequent resume saves can retain Bridge strict mode.
    return expected_text_keys == source_keys, expected_text_keys


def _validate_bridge_text_checkpoint(path: Path, expected_keys: set[str]) -> None:
    """Fail closed unless a partial Gemma4 export is complete and reloadable by Bridge."""

    actual_keys = _read_hf_safetensor_keys(path)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing or unexpected:
        raise RuntimeError(
            "Gemma4 text-only HF tensor contract mismatch: "
            f"expected={len(expected_keys)} actual={len(actual_keys)} "
            f"missing={len(missing)} examples={sorted(missing)[:5]} "
            f"unexpected={len(unexpected)} examples={sorted(unexpected)[:5]}"
        )

    config_path = path / "config.json"
    if not config_path.is_file():
        raise RuntimeError("Gemma4 text-only HF checkpoint is missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = set(config.get("architectures") or [])
    if config.get("model_type") != "gemma4" or "Gemma4ForConditionalGeneration" not in architectures:
        raise RuntimeError(
            "Gemma4 text-only Bridge checkpoint must retain the original Gemma4 VL config "
            "so a resume resolves the same official Gemma4VLBridge"
        )
    if not isinstance(config.get("text_config"), dict):
        raise RuntimeError("Gemma4 text-only Bridge checkpoint config has no text_config")


def _read_hf_safetensor_keys(path: Path) -> set[str]:
    """Read actual safetensor headers and cross-check an optional HF index."""

    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensor weights were written to {path}")

    actual_keys: set[str] = set()
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            file_keys = set(handle.keys())
        duplicates = actual_keys & file_keys
        if duplicates:
            raise RuntimeError(f"Duplicate HF tensor keys across shards: {sorted(duplicates)[:5]}")
        actual_keys.update(file_keys)

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        indexed_keys = set((index.get("weight_map") or {}).keys())
        if indexed_keys != actual_keys:
            raise RuntimeError(
                "HF safetensor index/header mismatch: "
                f"indexed={len(indexed_keys)} actual={len(actual_keys)}"
            )

    return actual_keys


def save_hf_model_bridge_snapshot_to_path(
    args,
    output_dir: str | Path,
    model,
    megatron_local_weights,
    *,
    model_name: str,
    quantization_config: dict[str, Any] | None,
    progress_desc: str,
) -> None:
    """Stream a CPU actor snapshot through the official Bridge into HF shards.

    Online colocated refit calls this while the live Megatron allocation is
    offloaded.  ``HfWeightIteratorBridge`` replaces each conversion task's live
    parameter with the corresponding CPU-backup tensor before moving only the
    current task to CUDA, preserving the official Bridge mapping without
    restoring the whole actor model.
    """
    from .update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge

    path = Path(output_dir)
    is_save_rank = _is_global_rank_zero()
    setup_error = None
    if is_save_rank:
        try:
            logger.info("Saving actor snapshot through Megatron Bridge to %s", path)
            path.mkdir(parents=True, exist_ok=True)
            _clear_existing_hf_weights(path)
            _copy_hf_assets(args.hf_checkpoint, path)
        except Exception as e:
            setup_error = repr(e)
    _raise_if_rank_zero_failed("prepare Bridge snapshot save directory", setup_error)

    iterator = HfWeightIteratorBridge(
        args=args,
        model=model,
        model_name=model_name,
        quantization_config=quantization_config,
    )
    num_save_nodes, save_node_rank, is_writer_rank, writer_ranks = _get_node_save_layout(args)
    if is_save_rank:
        logger.info("Bridge snapshot will write shards from node writer ranks: %s", writer_ranks)

    writer = _SafetensorShardWriter(path, enabled=is_writer_rank)
    for chunk_idx, hf_named_tensors in enumerate(
        iterator.get_hf_weight_chunks(megatron_local_weights, progress_desc=progress_desc)
    ):
        # Write immediately. Retaining one GPU chunk until another node writer
        # reaches its turn can overlap a multi-GiB Gemma embedding with the next
        # Bridge TP gather and defeats the low-memory snapshot path.
        if is_writer_rank and chunk_idx % num_save_nodes == save_node_rank:
            writer.write(hf_named_tensors, shard_idx=chunk_idx)
        del hf_named_tensors

    _finalize_distributed_shards(path, writer.state())
    if is_save_rank:
        logger.info("Successfully saved actor snapshot through Megatron Bridge to %s", path)


class _SafetensorShardWriter:
    def __init__(self, path: Path, *, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.total_size = 0
        self.weight_map: dict[str, str] = {}
        self.shard_files: list[str] = []

    def write(self, named_tensors, shard_idx: int) -> None:
        if not self.enabled:
            return
        assert shard_idx is not None, "shard_idx must be set when writing HF shards"

        from safetensors.torch import save_file

        state_dict = {}
        total_size = 0
        for name, tensor in named_tensors:
            if name in self.weight_map or name in state_dict:
                raise ValueError(f"Duplicate HF tensor while saving: {name}")
            total_size += tensor.numel() * tensor.element_size()
            state_dict[name] = _tensor_for_safetensors(tensor)

        if not state_dict:
            return

        filename = self._next_filename(shard_idx)
        if (self.path / filename).exists():
            raise ValueError(f"Duplicate HF shard file while saving: {filename}")

        save_file(state_dict, self.path / filename, metadata={"format": "pt"})
        self.shard_files.append(filename)
        self.total_size += total_size
        for name in state_dict:
            self.weight_map[name] = filename

    def state(self) -> dict[str, Any]:
        if not self.enabled:
            return {"total_size": 0, "weight_map": {}, "shard_files": []}
        return {
            "total_size": self.total_size,
            "weight_map": dict(self.weight_map),
            "shard_files": list(self.shard_files),
        }

    def finalize(self) -> None:
        if not self.enabled:
            return
        if not self.shard_files:
            raise ValueError("No HF tensors were produced while saving")

        total_files = len(self.shard_files)
        rename_map = {}
        for idx, old_name in enumerate(self.shard_files, start=1):
            new_name = f"model-{idx:05d}-of-{total_files:05d}.safetensors"
            os.replace(self.path / old_name, self.path / new_name)
            rename_map[old_name] = new_name

        final_weight_map = {name: rename_map[filename] for name, filename in self.weight_map.items()}
        index_data = {"metadata": {"total_size": self.total_size}, "weight_map": final_weight_map}
        with open(self.path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

    def _next_filename(self, shard_idx: int) -> str:
        assert shard_idx is not None, "shard_idx must be set when naming HF shards"
        return f"model-{shard_idx + 1:05d}.safetensors"


def _write_pending_chunk(
    writer: _SafetensorShardWriter, pending_write: tuple[int, Any] | None
) -> tuple[int, Any] | None:
    if pending_write is not None:
        shard_idx, named_tensors = pending_write
        writer.write(named_tensors, shard_idx=shard_idx)
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    return None


def _finalize_distributed_shards(path: Path, local_state: dict[str, Any]) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
    else:
        states = [local_state]

    if _is_global_rank_zero():
        _finalize_shard_files(path, states)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _finalize_shard_files(path: Path, shard_states: list[dict[str, Any] | None]) -> None:
    shard_files = []
    total_size = 0
    raw_weight_map = {}

    for state in shard_states:
        if not state:
            continue

        total_size += state.get("total_size", 0)
        for filename in state.get("shard_files", []):
            if filename in shard_files:
                raise ValueError(f"Duplicate HF shard file while finalizing: {filename}")
            shard_files.append(filename)

        for name, filename in state.get("weight_map", {}).items():
            if name in raw_weight_map:
                raise ValueError(f"Duplicate HF tensor while finalizing: {name}")
            raw_weight_map[name] = filename

    if not shard_files:
        raise ValueError("No HF tensors were produced while saving")

    shard_files = sorted(shard_files, key=_shard_filename_sort_key)
    total_files = len(shard_files)
    rename_map = {}
    for idx, old_name in enumerate(shard_files, start=1):
        new_name = f"model-{idx:05d}-of-{total_files:05d}.safetensors"
        os.replace(path / old_name, path / new_name)
        rename_map[old_name] = new_name

    final_weight_map = {}
    for name, filename in raw_weight_map.items():
        if filename not in rename_map:
            raise ValueError(f"HF tensor {name} points to missing shard file {filename}")
        final_weight_map[name] = rename_map[filename]

    index_data = {"metadata": {"total_size": total_size}, "weight_map": final_weight_map}
    with open(path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)


def _shard_filename_sort_key(filename: str) -> tuple[float, str]:
    prefix = "model-"
    suffix = ".safetensors"
    if filename.startswith(prefix) and filename.endswith(suffix):
        middle = filename[len(prefix) : -len(suffix)]
        if middle.isdigit():
            return int(middle), filename
    return math.inf, filename


def _tensor_for_safetensors(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def _clear_existing_hf_weights(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file() and _is_hf_weight_file(item):
            item.unlink()


def _copy_hf_assets(origin_hf_dir: str, output_dir: Path) -> None:
    origin = Path(origin_hf_dir)
    if not origin.is_dir():
        raise ValueError(f"--hf-checkpoint must be a local directory when using raw --save-hf: {origin_hf_dir}")

    for item in origin.iterdir():
        if item.is_file():
            if _is_hf_weight_file(item):
                continue
            shutil.copy2(item, output_dir / item.name)


def _is_hf_weight_file(path: Path) -> bool:
    name = path.name
    return name in _HF_WEIGHT_FILE_NAMES or name.endswith(_HF_WEIGHT_FILE_SUFFIXES)


def _is_global_rank_zero() -> bool:
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _get_node_save_layout(args) -> tuple[int, int, bool, list[int]]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return 1, 0, True, [0]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", None) or getattr(args, "num_gpus_per_node", 1) or 1)
    gpus_per_node = max(1, gpus_per_node)
    inferred_nodes = max(1, math.ceil(world_size / gpus_per_node))
    configured_nodes = int(getattr(args, "actor_num_nodes", None) or inferred_nodes)
    num_nodes = max(1, min(configured_nodes, inferred_nodes))
    writer_ranks = [node * gpus_per_node for node in range(num_nodes) if node * gpus_per_node < world_size]
    node_rank = min(rank // gpus_per_node, num_nodes - 1)
    return len(writer_ranks), node_rank, rank in writer_ranks, writer_ranks


def _raise_if_rank_zero_failed(context: str, error: str | None) -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        payload = [error]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0]

    if error is not None:
        raise RuntimeError(f"Failed to {context}: {error}")
