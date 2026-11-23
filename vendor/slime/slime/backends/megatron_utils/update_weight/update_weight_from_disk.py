from __future__ import annotations

import logging
import shutil
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from ..hf_checkpoint_saver import save_hf_model_to_path

logger = logging.getLogger(__name__)


class UpdateWeightFromDisk:
    """Full-weight sync through a shared filesystem and SGLang disk reload."""

    # The Bridge conversion allocates large temporary CUDA buffers.  When the
    # actor runs under torch_memory_saver.disable(), those buffers live in the
    # context's private MemPool until the context exits.  Colocated SGLang
    # weights therefore must not be resumed from inside that context.
    apply_after_memory_saver_disable = True

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines: Sequence[ActorHandle] = []
        self.rollout_engine_lock: ActorHandle | None = None
        self._prepared_version_dir: Path | None = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        all_engine_actors: Sequence[ActorHandle] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    @torch.no_grad()
    def update_weights(self) -> None:
        self.prepare_update_weights()
        self.apply_prepared_update()

    @torch.no_grad()
    def prepare_update_weights(self) -> None:
        if self._prepared_version_dir is not None:
            raise RuntimeError(f"A disk weight update is already prepared: {self._prepared_version_dir}")

        self.weight_version += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"
        self._prepared_version_dir = version_dir

        if dist.get_rank() == 0:
            shutil.rmtree(version_dir, ignore_errors=True)
        dist.barrier(group=get_gloo_group())

        if dist.get_rank() == 0:
            logger.info("Updating rollout weights from disk checkpoint %s", version_dir)
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        save_hf_model_to_path(
            self.args,
            version_dir,
            self.model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
            progress_desc="Save HF  weights for update from disk",
            megatron_local_weights=self.weights_getter(),
        )
        dist.barrier(group=get_gloo_group())

    @torch.no_grad()
    def apply_prepared_update(self) -> None:
        version_dir = self._prepared_version_dir
        if version_dir is None:
            raise RuntimeError("No disk weight update has been prepared")

        if dist.get_rank() == 0:
            # In colocated/offloaded mode train.py deliberately leaves SGLang
            # weights offloaded while the official Bridge performs its TP
            # gathers.  MegatronTrainRayActor invokes this method only after
            # torch_memory_saver.disable() has exited, which destroys the
            # Bridge conversion MemPool before SGLang restores its weights.
            if self.args.colocate and self.args.offload_rollout:
                logger.info("Restoring colocated rollout weights after Bridge export")
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=["weights"])
                        for engine in self.rollout_engines
                    ]
                )
            refs = [
                engine.update_weights_from_disk.remote(
                    model_path=str(version_dir),
                    weight_version=str(self.weight_version),
                )
                for engine in self.rollout_engines
            ]
            ray.get(refs)
            if not self.args.update_weight_disk_keep_files:
                # SGLang's HTTP endpoint can acknowledge the reload after its
                # coordinator rank has finished while slower TP ranks are still
                # opening safetensor shards from a shared filesystem.  Removing
                # the just-published directory here can therefore race those
                # readers.  Keep the current version alive for the full rollout
                # and remove only older versions after the next update has been
                # accepted, bounding normal disk use to one completed snapshot.
                self._remove_stale_version_dirs(version_dir)
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        self._prepared_version_dir = None

    def _remove_stale_version_dirs(self, current_version_dir: Path) -> None:
        root = Path(self.args.update_weight_disk_dir)
        for candidate in root.glob("weight_v*"):
            suffix = candidate.name.removeprefix("weight_v")
            if candidate == current_version_dir or not suffix.isdigit():
                continue
            shutil.rmtree(candidate, ignore_errors=True)
