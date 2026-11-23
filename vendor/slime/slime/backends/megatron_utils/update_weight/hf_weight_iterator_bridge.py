import dataclasses
import os


from slime.utils import megatron_bridge_utils
from slime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


def _patch_bridge_expert_cache_to_cpu():
    """Monkey-patch GPTOSSBridge class to cache expert weights on CPU.

    This avoids GPU OOM when torch.cat merges all experts, especially in
    colocated mode where SGLang and Megatron share the same GPU.
    """
    try:
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
    except ImportError:
        return

    if getattr(GPTOSSBridge, "_cpu_cache_patched", False):
        return

    _orig = GPTOSSBridge.maybe_modify_converted_hf_weight

    def _patched(self, task, converted_weights_dict):
        cpu_dict = {k: v.cpu() for k, v in converted_weights_dict.items()}
        result = _orig(self, task, cpu_dict)
        # Move merged result back to GPU for CUDA IPC serialization
        return {k: v.cuda() for k, v in result.items()} if result else result

    GPTOSSBridge.maybe_modify_converted_hf_weight = _patched
    GPTOSSBridge._cpu_cache_patched = True


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = megatron_bridge_utils.auto_bridge_from_hf_pretrained(
            self.args.hf_checkpoint,
            purpose="online_weight_export",
            trust_remote_code=True,
        )
        _patch_bridge_expert_cache_to_cpu()

    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            if os.environ.get("SLIME_REQUIRE_GEMMA4_DENSE_BRIDGE", "0") == "1":
                missing_tasks = [index for index, task in enumerate(conversion_tasks) if task is None]
                if missing_tasks:
                    raise RuntimeError(
                        "NeMo Gemma4 Bridge did not map every Megatron parameter; "
                        f"missing conversion task indexes={missing_tasks[:32]} total={len(missing_tasks)}"
                    )
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            def _streaming_quantized():
                for exported_weight in named_weights:
                    # NeMo Bridge's public HFWeightTuple contract is exactly
                    # ``(param_name, weight)``.  Those tensors are already fully
                    # converted to HF layout and have vocab padding truncated by
                    # ModelBridge.stream_weights_megatron_to_hf, so do not run a
                    # second, Slime-owned conversion over them.
                    if len(exported_weight) == 2:
                        if self.quantization_config is not None:
                            raise RuntimeError(
                                "NeMo Bridge's two-field HFWeightTuple cannot be passed to "
                                "Slime post-export quantization because it intentionally does "
                                "not expose a Megatron parameter name"
                            )
                        yield exported_weight
                        continue

                    # Retain compatibility with older Slime bridge plugins that
                    # extended the tuple with the source Megatron parameter name.
                    if len(exported_weight) != 3:
                        raise RuntimeError(
                            "Unsupported Megatron Bridge weight tuple: expected the official "
                            f"two fields or the legacy Slime three fields, got {len(exported_weight)}"
                        )
                    hf_param_name, weight, megatron_param_name = exported_weight
                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )
                    converted_named_params = [(hf_param_name, processed_weight)]
                    quantized_batch = quantize_params(
                        args=self.args,
                        megatron_name=megatron_param_name,
                        converted_named_params=converted_named_params,
                        quantization_config=self.quantization_config,
                    )
                    yield from quantized_batch

            yield from chunk_named_params_by_size(
                _streaming_quantized(), chunk_size=self.args.update_weight_buffer_size
            )


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task is None:
            return None
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if weight_dict_key not in new_weight_dict:
            # NeMo conversion tasks can include immutable model tensors.  The
            # known Gemma 4 ``layer_scalar`` buffers are included in Slime's
            # actor snapshot, but retain this compatibility fallback for other
            # immutable tensors.  A missing trainable weight remains a hard
            # error so online export cannot silently use stale parameters.
            if task.param_weight.requires_grad:
                raise AssertionError(
                    f"{weight_dict_key=} not in new_weight_dict "
                    f"({task.vp_stage=}, {task.param_name=}, trainable=True)"
                )
            return task

        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
