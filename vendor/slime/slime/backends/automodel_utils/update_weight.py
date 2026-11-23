"""NeMo DTensor to Slime SGLang weight synchronization."""

from __future__ import annotations

import time
from typing import Any, Iterable

import ray
import torch
import torch.distributed as dist

from slime.utils.distributed_utils import get_gloo_group


class AutoModelWeightUpdater:
    """Stream NeMo-RL's official HF tensors into Slime's SGLang NCCL receiver.

    Tensor naming and Gemma conversion are owned by NeMo-RL's
    ``dtensor_params_generator``. This class only buckets those already-adapted
    tensors and reuses Slime's existing SGLang distributed update transport.
    """

    def __init__(self, args: Any, worker: Any) -> None:
        self.args = args
        self.worker = worker
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines = []
        self.rollout_engine_lock = None
        self._model_update_group = None
        self._group_name = "slime-automodel"

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear weight-update metrics for the actor perf log.

        Keep the AutoModel/NeMo updater contract identical to Slime's native
        Megatron updaters.  The mapping is currently empty, but the actor
        drains it unconditionally after every optimizer step.
        """

        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    def connect(
        self,
        rollout_engines,
        rollout_engine_lock,
        *,
        engine_gpu_counts,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        if dist.get_rank() != 0:
            return

        from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
            connect_rollout_engines_from_distributed,
            disconnect_rollout_engines_from_distributed,
        )

        # Match Slime's native Megatron updater lifecycle exactly.  Fault
        # tolerance can replace only one rollout engine while the surviving
        # engines and train rank still retain the old named process group.
        # Recreating the same group without first destroying it fails with
        # "group name has already been created" on both sides.
        if self._model_update_group is not None:
            disconnect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                self._model_update_group,
                self.rollout_engines,
            )
            self._model_update_group = None

        self._model_update_group = connect_rollout_engines_from_distributed(
            self.args,
            self._group_name,
            rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
        )

    def _iter_buckets(self) -> Iterable[list[tuple[str, torch.Tensor]]]:
        from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
            dtensor_params_generator,
        )

        limit = int(self.args.update_weight_buffer_size)
        bucket: list[tuple[str, torch.Tensor]] = []
        bucket_bytes = 0
        for name, tensor in dtensor_params_generator(
            self.worker.model, self.worker.dtype
        ):
            tensor_bytes = tensor.numel() * tensor.element_size()
            if bucket and bucket_bytes + tensor_bytes > limit:
                yield bucket
                bucket = []
                bucket_bytes = 0
            bucket.append((name, tensor))
            bucket_bytes += tensor_bytes
        if bucket:
            yield bucket

    def _send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        if dist.get_rank() != 0:
            return
        if self._model_update_group is None:
            raise RuntimeError("AutoModel SGLang update group is not initialized")

        from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
            update_weights_from_distributed,
        )

        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_group,
                self.weight_version,
                self.rollout_engines,
                bucket,
            )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())

    @torch.no_grad()
    def update(self) -> None:
        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get(
                [engine.pause_generation.remote() for engine in self.rollout_engines]
            )
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        # Match Slime's native Megatron distributed updater: rank 0 may spend
        # up to the agent timeout draining in-flight generations before it can
        # pause SGLang.  Waiting ranks must not sit in the default NCCL process
        # group during that host-side wait, otherwise NCCL's collective
        # watchdog can kill the actor group before rank 0 reaches the barrier.
        gloo_group = get_gloo_group()
        dist.barrier(group=gloo_group)

        for bucket_index, bucket in enumerate(self._iter_buckets()):
            self._send_bucket(bucket)
            # Every DTensor rank must advance through full_tensor() collectives
            # in the same order and keep each bucket alive until rank 0 finishes
            # the remote NCCL broadcast.
            dist.barrier(group=gloo_group)
            if dist.get_rank() == 0:
                total_bytes = sum(t.numel() * t.element_size() for _, t in bucket)
                print(
                    "AutoModel weight bucket "
                    f"version={self.weight_version} index={bucket_index} "
                    f"tensors={len(bucket)} bytes={total_bytes}",
                    flush=True,
                )
            del bucket
            torch.cuda.empty_cache()

        if dist.get_rank() == 0:
            ray.get(
                [engine.continue_generation.remote() for engine in self.rollout_engines]
            )
        dist.barrier(group=gloo_group)
