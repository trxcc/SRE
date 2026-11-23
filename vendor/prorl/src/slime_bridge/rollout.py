"""Slime rollout bridge for Polar-managed agent sessions.

Single entrypoint ``generate_rollout_polar_async`` routes training to a
persistent background worker and evaluation to a one-shot submit+poll batch.
Both paths speak Polar's async-only HTTP surface (``/rollout/task/submit`` +
``/rollout/task/{task_id}``).
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import json
import logging
import math
import os
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import requests
import uvicorn
from fastapi import FastAPI, Request

from polar.rollout.models import TaskResult, TaskStatus
from slime_bridge._messages import prompt_to_instruction_text
from slime_bridge.adapter import RolloutLogprobError, session_result_to_samples
from slime_bridge.config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
    resolve_sglang_router_base_url,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds between task-status polls (eval / no-callback path)
_CALLBACK_FALLBACK_POLL_SECONDS = 60.0  # defensive backstop for dropped callbacks
_LONGEST_TRACE_ARTIFACT_INTERVAL = 5  # dump longest trace every N rollouts
_SUBMIT_MAX_ATTEMPTS = 8
_SUBMIT_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_longest_trace_upload_lock = threading.Lock()


def _sync_colocate_needs_inference_quiesce(
    args: Any,
    config: PolarSlimeConfig,
) -> bool:
    return bool(getattr(args, "colocate", False) and config.synchronous_policy_versions)


def _gateway_pause_timeout_seconds() -> float:
    value = float(os.environ.get("POLAR_GATEWAY_PAUSE_TIMEOUT_SECONDS", "300"))
    if value <= 0:
        raise ValueError("POLAR_GATEWAY_PAUSE_TIMEOUT_SECONDS must be greater than 0")
    return value


async def _set_colocated_inference_paused(
    args: Any,
    config: PolarSlimeConfig,
    *,
    paused: bool,
) -> None:
    """Fence Polar generation around Slime's colocated offload/onload cycle."""
    timeout_seconds = _gateway_pause_timeout_seconds()
    control_timeout = httpx.Timeout(timeout_seconds + 10.0, connect=15.0)
    async with httpx.AsyncClient(timeout=control_timeout, trust_env=False) as client:
        nodes_response = await client.get(f"{config.rollout_server_url}/nodes")
        nodes_response.raise_for_status()
        nodes = nodes_response.json()
        gateway_urls = sorted(
            {
                str(node.get("gateway_url", "")).rstrip("/")
                for node in nodes
                if isinstance(node, dict) and node.get("gateway_url")
            }
        )
        if not gateway_urls:
            raise RuntimeError("Polar rollout service reported no gateway nodes")

        endpoint = "pause" if paused else "resume"

        async def _set_one(gateway_url: str) -> dict[str, Any]:
            kwargs: dict[str, Any] = {}
            if paused:
                kwargs["params"] = {"timeout_seconds": timeout_seconds}
            response = await client.post(
                f"{gateway_url}/admin/inference/{endpoint}",
                **kwargs,
            )
            try:
                response.raise_for_status()
                status = response.json()
            except httpx.HTTPStatusError as exc:
                if not paused or exc.response.status_code != 504:
                    raise
                status_response = await client.get(
                    f"{gateway_url}/admin/inference/status"
                )
                status_response.raise_for_status()
                status = status_response.json()
                if not bool(status.get("paused")):
                    raise RuntimeError(
                        f"Gateway {gateway_url} returned 504 without entering paused state"
                    ) from exc
                logger.warning(
                    "Gateway %s pause timed out with inflight=%s; aborting orphaned SGLang requests",
                    gateway_url,
                    status.get("inflight"),
                )
            if paused and not bool(status.get("paused")):
                raise RuntimeError(
                    f"Gateway {gateway_url} did not enter paused state: {status}"
                )
            return status

        statuses = await asyncio.gather(*(_set_one(url) for url in gateway_urls))

        if paused:
            router_url = resolve_sglang_router_base_url(args)
            if router_url is None:
                raise RuntimeError(
                    "Sync colocate quiesce requires sglang_router_ip and sglang_router_port"
                )
            workers_response = await client.get(f"{router_url}/workers")
            workers_response.raise_for_status()
            worker_urls = sorted(
                {
                    str(worker.get("url", "")).rstrip("/")
                    for worker in workers_response.json().get("workers", [])
                    if isinstance(worker, dict) and worker.get("url")
                }
            )
            if not worker_urls:
                raise RuntimeError("SGLang router reported no workers before offload")

            # Reuse Slime's own abort-and-drain implementation. Polar sessions
            # are already terminal here, so any remaining request is orphaned.
            from slime.backends.sglang_utils.server_control import (
                abort_servers_until_idle,
                num_requests_from_load,
            )

            await asyncio.wait_for(
                abort_servers_until_idle(worker_urls),
                timeout=timeout_seconds,
            )
            loads = await asyncio.gather(
                *(
                    client.get(f"{worker_url}/v1/loads?include=core")
                    for worker_url in worker_urls
                )
            )
            remaining = []
            for worker_url, response in zip(worker_urls, loads, strict=True):
                response.raise_for_status()
                count = num_requests_from_load(response.json())
                if count:
                    remaining.append((worker_url, count))
            if remaining:
                raise RuntimeError(
                    f"SGLang requests remained after colocate quiesce: {remaining}"
                )

            # The pause endpoint can return 504 after it has already fenced
            # new work but while an orphaned upstream request is still stuck.
            # The SGLang abort above releases it; do not offload weights until
            # every gateway has observed that release.
            deadline = asyncio.get_running_loop().time() + min(30.0, timeout_seconds)
            while True:
                status_responses = await asyncio.gather(
                    *(
                        client.get(f"{gateway_url}/admin/inference/status")
                        for gateway_url in gateway_urls
                    )
                )
                statuses = []
                for response in status_responses:
                    response.raise_for_status()
                    statuses.append(response.json())
                lingering = [
                    status
                    for status in statuses
                    if not bool(status.get("paused"))
                    or int(status.get("inflight", -1)) != 0
                ]
                if not lingering:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError(
                        f"Polar gateways did not drain after SGLang abort: {lingering}"
                    )
                await asyncio.sleep(1.0)

    logger.info(
        "%s Polar inference on %d gateway(s) before sync-colocate %s",
        "Paused and drained" if paused else "Resumed",
        len(statuses),
        "offload" if paused else "rollout",
    )


class PolarRolloutSchedulerError(RuntimeError):
    """Raised when the async Polar scheduler cannot safely make progress."""


class PolarLowCompleteAcceptFractionError(PolarRolloutSchedulerError):
    """Raised when a completed task has too few trainable completed sessions."""


@dataclass(slots=True)
class _DeferredGroup:
    group: list[Any]
    group_id: int | None = None
    attempt: int = 0


@dataclass(slots=True)
class _PendingGroup:
    group_id: int
    submission_id: int
    group: list[Any]
    submitted_rollout_id: int
    policy_version: int
    session_cost: int
    attempt: int


@dataclass(slots=True)
class _CompletedGroup:
    group_id: int
    group: list[Any]
    samples: list[Any]
    task_id: str
    submitted_rollout_id: int
    policy_version: int
    session_count: int
    attempt: int
    completed_at: float = field(default_factory=time.monotonic)

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_async_worker(args: Any, data_source: Any) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def _get_existing_global_async_worker() -> "AsyncPolarRolloutWorker | None":
    with _worker_lock:
        worker = _global_async_worker
        if worker is None or not worker.is_alive():
            return None
        return worker


def stop_global_worker(*, reset_polar: bool = False) -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            if reset_polar:
                base_url = _global_async_worker.config.rollout_server_url
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(
                        f"{base_url.rstrip('/')}/admin/reset",
                        timeout=30,
                    )
                response.raise_for_status()
                logger.warning(
                    "Reset Polar tasks before replacing persistent async worker: %s",
                    response.json(),
                )
            _global_async_worker.stop()
            _global_async_worker = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    return render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )


def _attach_scheduler_metadata(
    payload: dict[str, Any],
    *,
    group_id: int,
    policy_version: int,
    rollout_step: int,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar task metadata must be a mapping when provided")
    payload["metadata"] = {
        **metadata,
        "group_id": group_id,
        "policy_version": policy_version,
        "rollout_step": rollout_step,
    }


async def _submit_and_wait_for_task(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
    submit_retry_base: float = 1.0,
) -> TaskResult:
    """Submit one task via the async endpoint and poll until terminal."""
    task_id = str(payload["task_id"])
    for attempt in range(1, _SUBMIT_MAX_ATTEMPTS + 1):
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            # A transport error may occur after the server accepted the task.
            # Retrying the same id then returns 409; polling it is the safe path.
            if resp.status_code == 409:
                logger.warning(
                    "Polar task %s already exists after submit retry; polling it",
                    task_id,
                )
                break
            resp.raise_for_status()
            task_id = str(resp.json()["task_id"])
            break
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code in _SUBMIT_RETRYABLE_STATUS_CODES
            if not retryable or attempt >= _SUBMIT_MAX_ATTEMPTS:
                raise
            detail = exc.response.text[:300]
            logger.warning(
                "Submitting Polar task %s returned HTTP %d (attempt %d/%d): %s",
                task_id,
                exc.response.status_code,
                attempt,
                _SUBMIT_MAX_ATTEMPTS,
                detail,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt >= _SUBMIT_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Submitting Polar task %s failed (attempt %d/%d): %s",
                task_id,
                attempt,
                _SUBMIT_MAX_ATTEMPTS,
                exc,
            )
        await asyncio.sleep(min(submit_retry_base * (2 ** (attempt - 1)), 10.0))

    while True:
        await asyncio.sleep(poll_interval)
        try:
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            logger.warning("Polling Polar task %s failed; continuing: %s", task_id, exc)
            continue
        status = TaskStatus.model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return TaskResult(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


def _resolve_max_tokens(args: Any) -> int | None:
    """Per-sample token cap Slime's dynamic batcher can fit on one GPU.

    Megatron asserts every sample length <= max_tokens_per_gpu * cp_size.
    Deep agent trajectories can exceed this (24-turn sessions → 80k+ tokens)
    and must be dropped before they reach the batcher.
    """
    mtpg = getattr(args, "max_tokens_per_gpu", None)
    if not mtpg:
        return None
    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    return int(mtpg) * cp_size


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one task's session results into flat Slime samples.

    Each session → one trajectory → N traces → N samples, all tagged
    with the same ``Sample.index`` so the reward post-processor groups
    them as one trajectory.  The index is taken from the originating
    group sample at matching position, falling back to the position
    within the task result.
    """
    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        collapse_fully_masked = bool(
            config.timeout_as_zero
            and _is_timeout_result(session_result)
            and _is_policy_failure_result(session_result, timeout_as_zero=True)
        )
        group_samples.extend(
            session_result_to_samples(
                session_result,
                group_index,
                trajectory_index=traj_idx,
                reward_key=config.reward_key,
                max_tokens=max_tokens,
                collapse_fully_masked=collapse_fully_masked,
            )
        )
    return group_samples


def _trainable_token_count(sample: Any) -> int:
    if bool(getattr(sample, "remove_sample", False)):
        return 0
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0)
    return sum(1 for value in loss_mask if int(value) != 0)


def _has_trainable_tokens(samples: list[Any]) -> bool:
    return any(_trainable_token_count(sample) > 0 for sample in samples)


def _low_complete_accept_fraction_rejection_reason(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    samples: list[Any],
) -> str | None:
    threshold = config.min_complete_accept_fraction
    if threshold <= 0.0:
        return None

    total_sessions = len(task_result.results)
    if total_sessions <= 0:
        return "empty task results"

    excluded_session_ids: set[str] = set()
    finalized_six_self_reward = total_sessions == 6 and all(
        (getattr(result, "metadata", None) or {}).get("self_reward_finalized") is True
        for result in task_result.results
    )
    if finalized_six_self_reward:
        anchor = task_result.results[0]
        if (getattr(anchor, "metadata", None) or {}).get("self_reward_trainable") is not False:
            raise PolarRolloutSchedulerError(
                "Six-candidate self-reward requires a diagnostic-only GT anchor"
            )
        # The anchor is deliberately excluded from policy loss. It must also
        # be excluded from the eligible-session completion denominator.
        # Keep the configured threshold and all legacy/non-self-reward rules.
        excluded_session_ids.add(anchor.session_id)
        total_sessions -= 1

    completed_trainable = _completed_trainable_session_count(
        task_result,
        samples,
        timeout_as_zero=bool(getattr(config, "timeout_as_zero", False)),
        excluded_session_ids=excluded_session_ids,
    )
    required = math.ceil(total_sessions * threshold)
    if completed_trainable >= required:
        return None

    fraction = completed_trainable / total_sessions
    return (
        f"completed trainable sessions {completed_trainable}/{total_sessions} "
        f"({fraction:.3f}) below polar_min_complete_accept_fraction={threshold:g} "
        f"(requires >= {required})"
    )


def _completed_trainable_session_count(
    task_result: TaskResult,
    samples: list[Any],
    *,
    timeout_as_zero: bool = False,
    excluded_session_ids: set[str] | None = None,
) -> int:
    trainable_session_ids: set[str] = set()
    for sample in samples:
        if _trainable_token_count(sample) <= 0:
            continue
        session_id = _sample_session_id(sample)
        if session_id:
            trainable_session_ids.add(session_id)

    count = 0
    for result in task_result.results:
        if result.session_id in (excluded_session_ids or set()):
            continue
        status = _status_value(result.status)
        # A policy-side terminal failure is an observed zero-reward outcome,
        # not a missing rollout. Count it even if no prefix survived and the
        # adapter emitted a fully masked placeholder. Infrastructure failures
        # deliberately do not take this path and remain retryable.
        if _is_policy_failure_result(result, timeout_as_zero=timeout_as_zero):
            count += 1
        elif status == "COMPLETED" and result.session_id in trainable_session_ids:
            count += 1
    return count


def _is_policy_failure_result(
    result: Any,
    *,
    timeout_as_zero: bool = False,
) -> bool:
    metadata = getattr(result, "metadata", {}) or {}
    failure_stage = str(metadata.get("polar_failure_stage") or "").lower()
    if failure_stage == "infrastructure":
        return False
    if metadata.get("self_reward_finalized") and not metadata.get("self_reward_valid", True):
        return True
    if failure_stage == "agent":
        return True
    return timeout_as_zero and _is_timeout_result(result)


def _has_infrastructure_failure(task_result: TaskResult) -> bool:
    return any(
        str((result.metadata or {}).get("polar_failure_stage") or "").lower()
        == "infrastructure"
        for result in task_result.results
    )


def _is_timeout_result(result: Any) -> bool:
    """Recognize explicit and evaluator-wrapped session timeouts.

    A gateway may finish the agent run but report an evaluator timeout as an
    ``ERROR`` (for example, ``evaluator failed: session execution timeout``).
    That is still the timeout outcome requested by ``polar_timeout_as_zero``;
    ordinary infrastructure errors remain retryable.
    """
    status = _status_value(getattr(result, "status", None))
    if status == "TIMEOUT":
        return True
    if status != "ERROR":
        return False

    trajectory = getattr(result, "trajectory", None)
    errors = (
        getattr(result, "error", None),
        getattr(trajectory, "error", None),
    )
    return any(
        "session execution timeout" in str(error or "").lower()
        or str(error or "").lower().startswith("eval timed out:")
        for error in errors
    )


def _all_policy_failure_results(
    task_result: TaskResult,
    *,
    timeout_as_zero: bool = False,
) -> bool:
    return bool(task_result.results) and all(
        _is_policy_failure_result(result, timeout_as_zero=timeout_as_zero)
        for result in task_result.results
    )


def _all_self_reward_results_intentionally_masked(task_result: TaskResult) -> bool:
    """Accept a finalized self-reward group that deliberately has no update.

    An invalid generated rubric, or seven candidates whose self-score sessions
    all failed, is a valid terminal outcome for ``self_reward_7plus1_train7``.
    The reward adapter masks every trace in that case.  Retrying the task would
    violate the requested semantics and can loop forever on the same sample.
    """
    if not task_result.results:
        return False
    metadata = [getattr(result, "metadata", None) for result in task_result.results]
    return all(
        isinstance(row, dict)
        and row.get("self_reward_finalized") is True
        and row.get("self_reward_trainable") is False
        for row in metadata
    )


def _sample_session_id(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    session_id = polar_meta.get("session_id") or getattr(sample, "session_id", None)
    return str(session_id) if session_id else None


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _is_zero_trainable_error(exc: BaseException) -> bool:
    return "zero trainable tokens" in str(exc)


def _annotate_accepted_samples(
    samples: list[Any],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        polar_meta = metadata.setdefault("polar", {})
        if not isinstance(polar_meta, dict):
            polar_meta = {}
            metadata["polar"] = polar_meta
        polar_meta.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
                "scheduler_group_id": int(scheduler_group_id),
            }
        )
        train_metadata = getattr(sample, "train_metadata", None)
        if train_metadata is None:
            train_metadata = {}
            sample.train_metadata = train_metadata
        train_metadata.update(
            {
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
            }
        )


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and pushes them into ``output_queue``.  Training loops call
    ``drain_completed()`` to collect finished groups.
    """

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        # Output queue is a handoff channel; the durable overflow buffer is
        # `_completed_buffer`, which is drained in bounded chunks by training.
        queue_maxsize = max(32, batch_size * self.config.max_async_level * 2)
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=queue_maxsize)
        self.deferred_queue: queue.Queue[_DeferredGroup] = queue.Queue()
        self._completed_buffer: deque[_CompletedGroup] = deque()
        self._running = True
        self._thread: threading.Thread | None = None
        # A training-free restart may restore the data-source cursor without a
        # model checkpoint. Keep Polar scheduler/task ids aligned with the
        # resumed global prompt ids so persisted PT shards remain mergeable.
        initial_group_id = int(
            getattr(data_source, "sample_group_index", 0) or 0
        )
        self._group_counter = initial_group_id
        self._submission_counter = initial_group_id
        self._batch_size = batch_size
        self._initial_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        self._current_rollout_id = self._initial_rollout_id
        self._update_weights_interval = int(
            getattr(args, "update_weights_interval", 1) or 1
        )
        self._requested_groups = 0
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._submissions_paused = False
        self._submission_pause_ack = threading.Event()
        self._metrics: dict[str, float] = {}
        self._last_metric_snapshot: dict[str, float] = {}
        self._active_groups = 0
        self._active_sessions = 0
        self._completed_buffer_size = 0
        self._ordered_prefix_groups = max(
            0,
            int(getattr(data_source, "ordered_prefix_groups", 0) or 0),
        )
        # A resumable finite-epoch data source starts its group counter at the
        # persisted cursor.  The ordered delivery head must start there too;
        # resetting it to zero would wait forever for groups already on disk.
        self._next_ordered_group_id = min(
            initial_group_id,
            self._ordered_prefix_groups,
        )
        self._ordered_completed: dict[int, _CompletedGroup] = {}
        # Per-task callback plumbing: event fires when the rollout server POSTs
        # the terminal TaskResult to our local listener.
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._callback_url: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="polar-async-rollout")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def pause_submissions(self) -> None:
        """Fence new training submissions without discarding buffered results."""
        with self._state_lock:
            self._submissions_paused = True
            self._submission_pause_ack.clear()

    def resume_submissions(self) -> None:
        with self._state_lock:
            self._submissions_paused = False
            self._submission_pause_ack.clear()

    def wait_until_quiescent(self, *, timeout_seconds: float) -> None:
        """Wait until the submission fence is observed and active work drains."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        deadline = time.monotonic() + timeout_seconds
        next_log = time.monotonic() + 60.0
        while True:
            self.raise_if_failed()
            with self._state_lock:
                acknowledged = self._submission_pause_ack.is_set()
                active_groups = self._active_groups
                active_sessions = self._active_sessions
            if acknowledged and active_groups == 0 and active_sessions == 0:
                return
            if not self.is_alive():
                raise PolarRolloutSchedulerError(
                    "Async Polar rollout worker stopped while draining for evaluation"
                )
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    "Timed out draining training rollouts before evaluation "
                    f"(active_groups={active_groups}, active_sessions={active_sessions})"
                )
            if now >= next_log:
                logger.info(
                    "Waiting to isolate evaluation from training rollouts: "
                    "active_groups=%d active_sessions=%d",
                    active_groups,
                    active_sessions,
                )
                next_log = now + 60.0
            time.sleep(0.1)

    # -- results ---------------------------------------------------------------

    def set_rollout_context(self, rollout_id: int) -> None:
        with self._state_lock:
            self._current_rollout_id = int(rollout_id)

    def request_groups(self, count: int) -> None:
        if count <= 0:
            return
        with self._state_lock:
            self._requested_groups += int(count)

    def ensure_requested_groups(self, count: int) -> None:
        """Keep at least ``count`` groups owned by the rolling async window."""
        if count <= 0:
            return
        with self._state_lock:
            self._requested_groups = max(self._requested_groups, int(count))

    def prefetch_target_groups(self) -> int:
        levels = self.config.max_async_level
        with self._state_lock:
            ordered_prefix_active = (
                self._next_ordered_group_id < self._ordered_prefix_groups
            )
            rollout_id = self._current_rollout_id
        if ordered_prefix_active:
            policy_version = self._policy_version_for_rollout(rollout_id)
            admissible_levels = max(
                1,
                policy_version
                + self.config.max_off_policy_steps
                - rollout_id
                + 1,
            )
            levels = min(levels, admissible_levels)
        return self._batch_size * max(1, levels)

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise PolarRolloutSchedulerError(str(self._fatal_error)) from self._fatal_error

    def drain_completed(
        self,
        *,
        max_groups: int,
        rollout_id: int,
    ) -> list[_CompletedGroup]:
        self.raise_if_failed()

        while True:
            try:
                completed = self.output_queue.get_nowait()
                if completed.group_id < self._ordered_prefix_groups:
                    if completed.group_id in self._ordered_completed:
                        raise PolarRolloutSchedulerError(
                            f"Duplicate completed ordered group {completed.group_id}"
                        )
                    self._ordered_completed[completed.group_id] = completed
                else:
                    self._completed_buffer.append(completed)
            except queue.Empty:
                break
        with self._state_lock:
            self._completed_buffer_size = (
                len(self._completed_buffer) + len(self._ordered_completed)
            )

        # A delayed ordered batch can become stale as a whole. Requeue every
        # stale slot now so the rollout nodes refill them concurrently instead
        # of discovering and retrying one head-of-line slot at a time.
        for group_id, completed in sorted(list(self._ordered_completed.items())):
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness <= self.config.max_off_policy_steps:
                continue
            self._ordered_completed.pop(group_id)
            self._inc_metric("polar/stale_groups")
            self._requeue_group(
                group=completed.group,
                group_id=completed.group_id,
                attempt=completed.attempt + 1,
                category_metric="polar/retried_stale_groups",
                reason=(
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                ),
                task_id=completed.task_id,
            )

        accepted: list[_CompletedGroup] = []
        while len(accepted) < max_groups:
            ordered = self._next_ordered_group_id < self._ordered_prefix_groups
            if ordered:
                completed = self._ordered_completed.pop(
                    self._next_ordered_group_id,
                    None,
                )
                if completed is None:
                    break
            else:
                if not self._completed_buffer:
                    break
                completed = self._completed_buffer.popleft()

            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.config.max_off_policy_steps:
                self._inc_metric("polar/stale_groups")
                reason = (
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                )
                self._requeue_group(
                    group=completed.group,
                    group_id=completed.group_id,
                    attempt=completed.attempt + 1,
                    category_metric="polar/retried_stale_groups",
                    reason=reason,
                    task_id=completed.task_id,
                )
                if ordered:
                    break
                continue

            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)
            if ordered:
                with self._state_lock:
                    self._next_ordered_group_id += 1

        if accepted:
            self._mark_delivered(len(accepted))
        with self._state_lock:
            self._completed_buffer_size = (
                len(self._completed_buffer) + len(self._ordered_completed)
            )
        return accepted

    def queue_size(self) -> int:
        with self._state_lock:
            return (
                self.output_queue.qsize()
                + self._completed_buffer_size
                + self.deferred_queue.qsize()
            )

    def snapshot_metrics(self) -> dict[str, float]:
        with self._state_lock:
            out = dict(self._metrics)
            for key, value in self._metrics.items():
                previous = self._last_metric_snapshot.get(key, 0.0)
                suffix = key.removeprefix("polar/")
                out[f"polar/step/{suffix}"] = max(0.0, value - previous)
            self._last_metric_snapshot = dict(self._metrics)
            out["polar/scheduler/active_groups"] = float(self._active_groups)
            out["polar/scheduler/active_sessions"] = float(self._active_sessions)
            out["polar/scheduler/completed_buffer"] = float(self._completed_buffer_size)
            out["polar/scheduler/output_queue"] = float(self.output_queue.qsize())
            out["polar/scheduler/deferred_queue"] = float(self.deferred_queue.qsize())
            out["polar/scheduler/requested_groups"] = float(self._requested_groups)
            out["polar/scheduler/submissions_paused"] = float(
                self._submissions_paused
            )
            out["polar/scheduler/ordered_prefix_total"] = float(
                self._ordered_prefix_groups
            )
            out["polar/scheduler/ordered_prefix_delivered"] = float(
                self._next_ordered_group_id
            )
            out["polar/scheduler/ordered_prefix_buffered"] = float(
                len(self._ordered_completed)
            )
            return out

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        active: dict[asyncio.Task[None], _PendingGroup] = {}
        active_session_cost = 0
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for t in done:
                        pending = active.pop(t)
                        active_session_cost -= pending.session_cost
                        try:
                            t.result()
                        except Exception as exc:
                            logger.exception("Polar async task failed")
                            self._set_fatal(exc)
                            self._running = False
                    self._record_active_counts(active, active_session_cost)

                    while self._running and self._can_admit_group(active, active_session_cost):
                        try:
                            next_group = self._next_group_for_submission()
                        except Exception as exc:
                            self._set_fatal(exc)
                            self._running = False
                            break
                        if next_group is None:
                            break
                        session_cost = len(next_group.group)
                        if session_cost > self.config.max_session_concurrency:
                            self._set_fatal(
                                PolarRolloutSchedulerError(
                                    f"Prompt group needs {session_cost} sessions but "
                                    f"derived max_session_concurrency is "
                                    f"{self.config.max_session_concurrency}"
                                )
                            )
                            self._running = False
                            break
                        if active_session_cost + session_cost > self.config.max_session_concurrency:
                            self.deferred_queue.put(next_group)
                            break

                        if next_group.group_id is None:
                            gid = self._group_counter
                            self._group_counter += 1
                        else:
                            gid = next_group.group_id
                        submission_id = self._submission_counter
                        self._submission_counter += 1
                        submitted_rollout_id, policy_version = self._rollout_context()
                        pending = _PendingGroup(
                            group_id=gid,
                            submission_id=submission_id,
                            group=next_group.group,
                            submitted_rollout_id=submitted_rollout_id,
                            policy_version=policy_version,
                            session_cost=session_cost,
                            attempt=next_group.attempt,
                        )
                        task = asyncio.create_task(
                            self._submit_and_collect(client, pending),
                            name=f"polar-rollout-task-{gid}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = pending
                        active_session_cost += session_cost
                        self._inc_metric("polar/submitted_groups")
                        self._inc_metric("polar/submitted_sessions", session_cost)
                        self._record_active_counts(active, active_session_cost)

                    self._acknowledge_submission_pause()
                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        wakeup.clear()

            if active:
                if self._running:
                    logger.info("Waiting for %d in-flight Polar tasks", len(active))
                else:
                    logger.warning(
                        "Cancelling %d old in-flight Polar tasks during worker reset",
                        len(active),
                    )
                    for task in active:
                        task.cancel()
                await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[uvicorn.Server, asyncio.Task[None]]:
        """Bind a FastAPI listener for TaskResult callbacks."""
        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = TaskResult.model_validate(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app, host=self.config.callback_host, port=0,
            log_level="warning", lifespan="on",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://{self.config.callback_host}:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, pending: _PendingGroup
    ) -> None:
        last_error: BaseException | None = None

        if self._running:
            try:
                completed = await self._submit_attempt(client, pending)
                await self._emit_completed(completed)
                return
            except Exception as exc:
                last_error = exc

        if last_error is None:
            return

        if _is_zero_trainable_error(last_error):
            category_metric = "polar/retried_zero_trainable_groups"
            reason = "zero trainable tokens"
        elif isinstance(last_error, PolarLowCompleteAcceptFractionError):
            category_metric = "polar/retried_low_complete_fraction_groups"
            reason = "low complete accept fraction"
        elif isinstance(last_error, RolloutLogprobError):
            category_metric = "polar/retried_logprob_error_groups"
            reason = "rollout logprob error"
        else:
            category_metric = "polar/retried_failed_groups"
            reason = "task failure"

        if self._running:
            self._requeue_group(
                group=pending.group,
                group_id=pending.group_id,
                attempt=pending.attempt + 1,
                category_metric=category_metric,
                reason=f"{reason}: {last_error}",
            )
        return

    async def _submit_attempt(
        self,
        client: httpx.AsyncClient,
        pending: _PendingGroup,
    ) -> _CompletedGroup:
        payload = _build_task_payload(
            args=self.args, config=self.config, group=pending.group,
            rollout_id=pending.submission_id, task_position=0,
        )
        payload["task_id"] = str(payload["task_id"])
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        from clawgym_nanobot.self_reward import configure_payload, finalize_group
        configure_payload(payload)
        task_result = await self._submit_with_callback(client, payload)
        task_result = await finalize_group(task_result)

        rejection_reason = self._task_rejection_reason(task_result, pending.group)
        if rejection_reason is not None:
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            )

        group_samples = _convert_task_result_to_samples(
            self.config, task_result, pending.group,
            max_tokens=_resolve_max_tokens(self.args),
        )
        if not group_samples:
            raise PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples")
        has_trainable_tokens = _has_trainable_tokens(group_samples)
        all_policy_failures = _all_policy_failure_results(
            task_result,
            timeout_as_zero=self.config.timeout_as_zero,
        )
        intentionally_masked_self_reward = (
            _all_self_reward_results_intentionally_masked(task_result)
        )
        if (
            not has_trainable_tokens
            and not all_policy_failures
            and not intentionally_masked_self_reward
        ):
            raise PolarRolloutSchedulerError(
                f"Task {task_result.task_id} produced zero trainable tokens"
            )
        rejection_reason = None
        if not intentionally_masked_self_reward:
            rejection_reason = _low_complete_accept_fraction_rejection_reason(
                self.config, task_result, group_samples
            )
        if rejection_reason is not None:
            if pending.attempt < self.config.max_low_complete_retries:
                raise PolarLowCompleteAcceptFractionError(
                    f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
                )
            self._inc_metric(
                "polar/accepted_low_complete_fraction_after_retry_exhaustion"
            )
            logger.warning(
                "Accepting Polar task %s after exhausting %d low-complete "
                "retries: %s; unusable samples remain fully masked",
                task_result.task_id,
                self.config.max_low_complete_retries,
                rejection_reason,
            )

        if intentionally_masked_self_reward:
            self._inc_metric("polar/accepted_intentionally_masked_self_reward_groups")
            logger.warning(
                "Accepting Polar task %s as an intentionally masked "
                "self-reward group (zero-gradient group)",
                task_result.task_id,
            )

        policy_failure_count = sum(
            _is_policy_failure_result(
                result,
                timeout_as_zero=self.config.timeout_as_zero,
            )
            for result in task_result.results
        )
        if policy_failure_count:
            self._inc_metric(
                "polar/accepted_zero_reward_policy_failure_sessions",
                policy_failure_count,
            )
            if all_policy_failures:
                self._inc_metric("polar/accepted_all_policy_failure_groups")
            logger.warning(
                "Accepting Polar task %s with %d/%d policy-side terminal "
                "failures as zero-reward outcomes%s",
                task_result.task_id,
                policy_failure_count,
                len(task_result.results),
                " (zero-gradient group)" if not has_trainable_tokens else "",
            )

        from clawgym_nanobot.intentional_terminal_batch import mark_terminal_group
        mark_terminal_group(task_result, group_samples, pending.group_id)
        return _CompletedGroup(
            group_id=pending.group_id,
            group=pending.group,
            samples=group_samples,
            task_id=task_result.task_id,
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(task_result.results),
            attempt=pending.attempt,
        )

    async def _emit_completed(self, completed: _CompletedGroup) -> None:
        while self._running:
            try:
                self.output_queue.put_nowait(completed)
                self._inc_metric("polar/completed_groups")
                return
            except queue.Full:
                self._inc_metric("polar/output_queue_full_waits")
                await asyncio.sleep(0.1)

    def _next_group_for_submission(self) -> _DeferredGroup | None:
        try:
            deferred = self.deferred_queue.get_nowait()
            self._inc_metric("polar/deferred_queue_dequeues")
            return deferred
        except queue.Empty:
            pass

        with self._state_lock:
            ordered_prefix_blocked = (
                self._ordered_prefix_groups > 0
                and self._group_counter >= self._ordered_prefix_groups
                and self._next_ordered_group_id < self._ordered_prefix_groups
            )
        if ordered_prefix_blocked:
            return None

        groups = self.data_source.get_samples(1)
        if not groups:
            return None
        group = groups[0]
        if not group:
            raise PolarRolloutSchedulerError("Slime data source returned an empty sample group")
        return _DeferredGroup(group=group)

    def _can_admit_group(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> bool:
        with self._state_lock:
            if self._submissions_paused:
                return False
        requested_groups = self._shared_requested_groups()
        if requested_groups <= 0:
            return False
        if len(active) >= self.config.max_concurrency:
            return False
        if active_session_cost >= self.config.max_session_concurrency:
            return False
        # Moving an already-owned deferred/retry group into `active` does not
        # grow the admission window. It must remain runnable even when buffered
        # completions currently fill that window.
        if not self.deferred_queue.empty():
            return True
        owned_groups = (
            len(active)
            + self.output_queue.qsize()
            + self._shared_completed_buffer_size()
            + self.deferred_queue.qsize()
        )
        admission_window = min(
            requested_groups,
            self.prefetch_target_groups(),
        )
        return owned_groups < admission_window

    def _acknowledge_submission_pause(self) -> None:
        with self._state_lock:
            if self._submissions_paused:
                self._submission_pause_ack.set()
            else:
                self._submission_pause_ack.clear()

    def _requeue_group(
        self,
        *,
        group: list[Any],
        group_id: int,
        attempt: int,
        category_metric: str,
        reason: str,
        task_id: str | None = None,
    ) -> None:
        self.deferred_queue.put(
            _DeferredGroup(group=group, group_id=group_id, attempt=attempt)
        )
        self._inc_metric("polar/retried_groups")
        self._inc_metric(category_metric)
        self._inc_metric("polar/retried_sessions", len(group))
        logger.warning(
            "Retrying Polar group %s attempt=%d%s: %s",
            group_id,
            attempt,
            f" task={task_id}" if task_id else "",
            reason,
        )

    def _task_rejection_reason(self, task_result: TaskResult, group: list[Any]) -> str | None:
        if task_result.status != "completed":
            return f"task status={task_result.status}"
        if not task_result.results:
            return "empty task results"
        if len(task_result.results) != len(group):
            return f"session count {len(task_result.results)} != expected {len(group)}"
        return None

    def _rollout_context(self) -> tuple[int, int]:
        with self._state_lock:
            rollout_id = self._current_rollout_id
        return rollout_id, self._policy_version_for_rollout(rollout_id)

    def _policy_version_for_rollout(self, rollout_id: int) -> int:
        # train.py updates rollout weights before requesting the next batch, so
        # a synchronous rollout is generated by the policy for that same step.
        # train_async.py deliberately starts the next batch before the current
        # optimizer update and therefore keeps the historical one-step lag.
        if self.config.synchronous_policy_versions:
            return int(rollout_id)
        offset = rollout_id - self._initial_rollout_id
        if offset <= 0:
            return self._initial_rollout_id
        return self._initial_rollout_id + (
            (offset - 1) // self._update_weights_interval
        ) * self._update_weights_interval

    def _shared_requested_groups(self) -> int:
        with self._state_lock:
            return self._requested_groups

    def _mark_delivered(self, count: int) -> None:
        with self._state_lock:
            self._requested_groups = max(0, self._requested_groups - int(count))

    def _shared_completed_buffer_size(self) -> int:
        with self._state_lock:
            return self._completed_buffer_size

    def _record_active_counts(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> None:
        with self._state_lock:
            self._active_groups = len(active)
            self._active_sessions = active_session_cost

    def _inc_metric(self, key: str, amount: float = 1.0) -> None:
        with self._state_lock:
            self._metrics[key] = self._metrics.get(key, 0.0) + amount

    def _set_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

    async def _submit_with_callback(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        """Submit a task, wait on its completion event, and fall back to polling."""
        task_id = payload["task_id"]
        # Register event BEFORE submit so a fast callback cannot arrive first.
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self._callback_url
        base_url = self.config.rollout_server_url
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return await self._await_task_result(client, task_id, event)
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def _await_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        event: asyncio.Event,
    ) -> TaskResult:
        """Wait on the completion event with a defensive 60s fallback poll."""
        base_url = self.config.rollout_server_url
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=_CALLBACK_FALLBACK_POLL_SECONDS)
            except asyncio.TimeoutError:
                status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
                status_resp.raise_for_status()
                status = TaskStatus.model_validate(status_resp.json())
                if status.status in ("completed", "failed"):
                    return TaskResult(
                        task_id=task_id, status=status.status,
                        results=status.results, result_paths=status.result_paths,
                    )
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            # Race: event set but result missing — re-poll once.
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
            status = TaskStatus.model_validate(status_resp.json())
            return TaskResult(
                task_id=task_id, status=status.status,
                results=status.results, result_paths=status.result_paths,
            )


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    needs_quiesce = _sync_colocate_needs_inference_quiesce(args, config)
    if needs_quiesce:
        await _set_colocated_inference_paused(args, config, paused=False)
    try:
        return await _run_eval_rollout_unfenced(
            args=args,
            rollout_id=rollout_id,
            data_source=data_source,
            config=config,
        )
    finally:
        if needs_quiesce:
            await _set_colocated_inference_paused(args, config, paused=True)


async def _run_eval_rollout_unfenced(
    *,
    args: Any,
    rollout_id: int,
    data_source: Any,
    config: PolarSlimeConfig,
) -> Any:
    eval_datasets = list(getattr(args, "eval_datasets", []) or [])
    if eval_datasets:
        data: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for dataset_cfg in eval_datasets:
            dataset_name, dataset_data, dataset_metrics = await _run_eval_dataset(
                args=args,
                config=config,
                rollout_id=rollout_id,
                dataset_cfg=dataset_cfg,
            )
            data[dataset_name] = dataset_data
            metrics.update(_prefix_eval_metrics(dataset_name, dataset_metrics))

        RolloutFnEvalOutput = _load_rollout_eval_output_type()
        return RolloutFnEvalOutput(data=data, metrics=metrics)

    logger.warning(
        "Polar eval called without args.eval_datasets; falling back to the training data source. "
        "Pass --eval-prompt-data to evaluate validation prompts."
    )
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=config.eval_dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={config.eval_dataset_name: dataset_data},
        metrics=metrics,
    )


async def _run_eval_dataset(
    *,
    args: Any,
    config: PolarSlimeConfig,
    rollout_id: int,
    dataset_cfg: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_name = str(getattr(dataset_cfg, "name", "") or config.eval_dataset_name)
    sample_groups = _load_eval_sample_groups(args, dataset_cfg)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    return dataset_name, dataset_data, metrics


async def _submit_eval_groups(
    *,
    args: Any,
    config: PolarSlimeConfig,
    dataset_name: str,
    rollout_id: int,
    sample_groups: list[list[Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sample_groups:
        raise ValueError("Polar eval dataset produced no sample groups")

    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.eval_max_concurrency)
    progress_lock = asyncio.Lock()
    completed_count = 0

    async def _run_one(position: int, group: list[Any]) -> TaskResult:
        nonlocal completed_count
        async with semaphore:
            base_payload = _build_task_payload(
                args=args, config=config, group=group,
                rollout_id=rollout_id, task_position=position,
            )
            base_payload["task_id"] = _eval_task_id(
                base_payload["task_id"],
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                position=position,
            )
            result = None
            for attempt in range(config.eval_infra_retries + 1):
                payload = copy.deepcopy(base_payload)
                if attempt:
                    payload["task_id"] = f"{base_payload['task_id']}-infra-retry-{attempt}"
                _attach_scheduler_metadata(
                    payload,
                    group_id=position,
                    policy_version=rollout_id,
                    rollout_step=rollout_id,
                )
                result = await _submit_and_wait_for_task(
                    client,
                    config.rollout_server_url,
                    payload,
                )
                if not _has_infrastructure_failure(result):
                    break
                if attempt < config.eval_infra_retries:
                    logger.warning(
                        "Retrying eval task after infrastructure failure "
                        "task=%s attempt=%d/%d statuses=%s",
                        result.task_id,
                        attempt + 1,
                        config.eval_infra_retries,
                        ",".join(_status_value(item.status) for item in result.results)
                        or "empty",
                    )
                    await asyncio.sleep(min(2 ** attempt, 10))
            assert result is not None
        async with progress_lock:
            completed_count += 1
            logger.info(
                "[Polar eval] %s %d/%d task=%s score=%.4f statuses=%s",
                dataset_name,
                completed_count,
                len(sample_groups),
                result.task_id,
                _task_result_reward(result),
                ",".join(_status_value(item.status) for item in result.results) or "empty",
            )
        return result

    # Polar is an internal control-plane endpoint. Ignore process-global proxy
    # variables because W&B temporarily mutates them from an upload thread.
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )

    output_groups: list[list[Any]] = []
    max_tokens = _resolve_max_tokens(args)
    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_task_result_to_samples(
                config, task_result, group,
                max_tokens=max_tokens,
            )
        )

    metrics = _build_metrics(
        config,
        task_results,
        output_groups,
        reward_filter="completed",
    )
    flat_samples = [sample for group in output_groups for sample in group]
    reward_data = _eval_task_reward_data(
        output_groups,
        config.reward_key,
        timeout_as_zero=config.timeout_as_zero,
    )

    return {
        **reward_data,
        "samples": flat_samples,
        # Preserve prompt-level labels such as eval_subset after Polar converts
        # the returned trajectories into trace-level training samples.
        "task_metadata": [
            copy.deepcopy(getattr(group[0], "metadata", {}) or {})
            for group in sample_groups
        ],
        "task_statuses": [
            [_status_value(result.status) for result in task_result.results]
            for task_result in task_results
        ],
        "n_samples_per_eval_prompt": len(sample_groups[0]),
    }, metrics


def _eval_task_reward_data(
    output_groups: list[list[Any]],
    reward_key: str,
    *,
    timeout_as_zero: bool = False,
) -> dict[str, Any]:
    """Return one eval reward per session and one mean per prompt group."""
    task_rewards: list[float] = []
    raw_rewards: list[float] = []
    raw_skipped: list[bool] = []
    task_truncated: list[bool] = []
    raw_truncated: list[bool] = []

    for group in output_groups:
        sessions: dict[str, list[Any]] = {}
        for sample in group:
            session_id = _sample_session_id(sample) or f"sample:{id(sample)}"
            sessions.setdefault(session_id, []).append(sample)

        group_rewards: list[float] = []
        group_truncated = False
        for session_samples in sessions.values():
            first = session_samples[0]
            reward = _extract_sample_reward(first, reward_key)
            truncated = any(_is_truncated(sample) for sample in session_samples)
            polar_meta = (getattr(first, "metadata", {}) or {}).get("polar", {})
            trajectory_metadata = polar_meta.get("trajectory_metadata") or {}
            evaluation = trajectory_metadata.get("evaluation") or {}
            has_score = any(
                isinstance(evaluation.get(key), (int, float))
                for key in ("code_verifier_score", "llm_judge_score", "outcome_reward")
            )
            if not has_score and timeout_as_zero:
                result_metadata = polar_meta.get("result_metadata") or {}
                failure_stage = str(
                    result_metadata.get("polar_failure_stage") or ""
                ).lower()
                status = _status_value(polar_meta.get("session_status"))
                errors = (
                    polar_meta.get("result_error"),
                    polar_meta.get("trajectory_error"),
                )
                is_timeout = status == "TIMEOUT" or (
                    status == "ERROR"
                    and any(
                        "session execution timeout" in str(error or "").lower()
                        or str(error or "").lower().startswith("eval timed out:")
                        for error in errors
                    )
                )
                # Match the training acceptance contract: a policy-side
                # timeout is an observed zero-reward outcome.  Infrastructure
                # failures remain missing data and therefore stay skipped.
                has_score = is_timeout and failure_stage != "infrastructure"

            group_rewards.append(reward)
            raw_rewards.append(reward)
            raw_skipped.append(not has_score)
            raw_truncated.append(truncated)
            group_truncated = group_truncated or truncated

        task_rewards.append(sum(group_rewards) / len(group_rewards) if group_rewards else 0.0)
        task_truncated.append(group_truncated)

    collected = [
        reward
        for reward, skipped in zip(raw_rewards, raw_skipped, strict=True)
        if not skipped
    ]
    return {
        "rewards": task_rewards,
        "rewards_are_task_level": True,
        "raw_rewards": raw_rewards,
        "all_rewards": raw_rewards,
        "raw_skipped": raw_skipped,
        "n_collected": len(collected),
        "n_skipped": sum(raw_skipped),
        "avg_reward_collected_only": (
            sum(collected) / len(collected) if collected else 0.0
        ),
        "truncated": task_truncated,
        "all_truncated": raw_truncated,
    }


def _task_result_reward(task_result: TaskResult) -> float:
    rewards: list[float] = []
    for result in task_result.results:
        evaluation = result.trajectory.metadata.get("evaluation") or {}
        score = evaluation.get("code_verifier_score")
        if not isinstance(score, (int, float)):
            score = evaluation.get("outcome_reward")
        if isinstance(score, (int, float)):
            rewards.append(float(score))
    return sum(rewards) / len(rewards) if rewards else 0.0


def _eval_task_id(base_task_id: Any, *, dataset_name: str, rollout_id: int, position: int) -> str:
    """Namespace eval task ids away from train task ids.

    Training ids commonly use ``{rollout_id}-{sample.group_index}``; eval uses
    ``position`` as group index, so eval 11 / item 11 would collide with train
    group 11. A suffix keeps task polling and persisted result dirs separate.
    """
    safe_dataset = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_name
    )
    return f"{base_task_id}-eval-{safe_dataset}-{rollout_id}-{position}"


def _completed_session_samples(samples: list[Any]) -> list[Any]:
    return [
        sample for sample in samples
        if _sample_session_status(sample) == "COMPLETED"
        and not bool(
            (getattr(sample, "metadata", {}) or {})
            .get("polar", {})
            .get("placeholder")
        )
    ]


def _sample_session_status(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    status = polar_meta.get("session_status")
    return getattr(status, "value", status)


def _load_eval_sample_groups(args: Any, dataset_cfg: Any) -> list[list[Any]]:
    Sample = _load_sample_type()
    path = str(getattr(dataset_cfg, "path"))
    input_key = getattr(dataset_cfg, "input_key", None) or getattr(args, "input_key", "prompt")
    label_key = getattr(dataset_cfg, "label_key", None) or getattr(args, "label_key", None)
    metadata_key = getattr(dataset_cfg, "metadata_key", None) or getattr(args, "metadata_key", "metadata")
    tool_key = getattr(dataset_cfg, "tool_key", None) or getattr(args, "tool_key", None)
    group_size = int(
        getattr(dataset_cfg, "n_samples_per_eval_prompt", None)
        or getattr(args, "n_samples_per_eval_prompt", None)
        or 1
    )
    if group_size <= 0:
        raise ValueError("n_samples_per_eval_prompt must be positive")

    groups: list[list[Any]] = []
    sample_index = 0
    for prompt_index, row in enumerate(_read_jsonl_rows(path)):
        if input_key not in row:
            raise KeyError(f"Eval row {prompt_index} in {path} missing input key {input_key!r}")

        metadata = _inject_eval_metadata(dataset_cfg, row.get(metadata_key))
        if tool_key and tool_key in row:
            tools = row[tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata["tools"] = tools

        group: list[Any] = []
        for _ in range(group_size):
            sample = Sample(
                prompt=copy.deepcopy(row[input_key]),
                label=row.get(label_key) if label_key else None,
                metadata=copy.deepcopy(metadata),
                group_index=prompt_index,
                index=sample_index,
            )
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            group.append(sample)
            sample_index += 1
        groups.append(group)

    return groups


def _read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Eval row {line_number} in {path} is not a JSON object")
            rows.append(row)
    return rows


def _inject_eval_metadata(dataset_cfg: Any, sample_metadata: Any) -> dict[str, Any]:
    inject = getattr(dataset_cfg, "inject_metadata", None)
    if callable(inject):
        metadata = inject(sample_metadata)
    elif isinstance(sample_metadata, dict):
        metadata = dict(sample_metadata)
    else:
        metadata = {}
    return metadata


def _prefix_eval_metrics(dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    prefixed: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("polar/"):
            prefixed[f"polar/eval/{dataset_name}/{key.removeprefix('polar/')}"] = value
        else:
            prefixed[f"polar/eval/{dataset_name}/{key}"] = value
    return prefixed


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Slime data source returned an empty sample group")
    return groups


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
    *,
    reward_filter: str = "all",
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    all_rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
    completed_rewards = [
        _extract_sample_reward(s, config.reward_key)
        for s in _completed_session_samples(flat_samples)
    ]
    if reward_filter == "all":
        rewards = all_rewards
    elif reward_filter == "completed":
        rewards = completed_rewards
    else:
        raise ValueError("reward_filter must be 'all' or 'completed'")
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat_samples, rewards, config.reward_key))
    return metrics


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def _eval_training_drain_timeout_seconds(config: PolarSlimeConfig) -> float:
    raw_value = os.environ.get("POLAR_EVAL_TRAINING_DRAIN_TIMEOUT_SECONDS")
    value = (
        float(raw_value)
        if raw_value is not None
        else float(config.request_timeout or 10800.0)
    )
    if value <= 0:
        raise ValueError(
            "POLAR_EVAL_TRAINING_DRAIN_TIMEOUT_SECONDS must be greater than 0"
        )
    return value


def _run_eval_with_training_rollout_quiesced(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    worker = _get_existing_global_async_worker()
    if worker is None:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))
    if not _env_flag("POLAR_EVAL_QUIESCE_TRAINING_ROLLOUT", default=True):
        logger.warning(
            "Evaluation is sharing inference capacity with async training rollout "
            "because POLAR_EVAL_QUIESCE_TRAINING_ROLLOUT is disabled"
        )
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

    timeout_seconds = _eval_training_drain_timeout_seconds(worker.config)
    worker.pause_submissions()
    try:
        logger.info(
            "Paused async training submissions before eval %d; draining active work",
            rollout_id,
        )
        worker.wait_until_quiescent(timeout_seconds=timeout_seconds)
        logger.info(
            "Async training rollout is quiescent; starting isolated eval %d",
            rollout_id,
        )
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))
    finally:
        worker.resume_submissions()
        logger.info("Resumed async training submissions after eval %d", rollout_id)


def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Slime-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return _run_eval_with_training_rollout_quiesced(
            args,
            rollout_id,
            data_source,
        )

    config = resolve_polar_slime_config(args)
    needs_quiesce = _sync_colocate_needs_inference_quiesce(args, config)
    if needs_quiesce:
        asyncio.run(_set_colocated_inference_paused(args, config, paused=False))

    async_worker = get_global_async_worker(args, data_source)
    async_worker.set_rollout_context(rollout_id)
    target = getattr(args, "rollout_batch_size", 1)
    if getattr(data_source, "finite_first_epoch", False):
        target = len(data_source.batch_group_ids(rollout_id))
    async_worker.ensure_requested_groups(async_worker.prefetch_target_groups())

    data: list[list[Any]] = []
    start = time.monotonic()
    last_progress = start
    consumed_group_ids = []

    while len(data) < target:
        made_progress = False
        completed_groups = async_worker.drain_completed(
            max_groups=target - len(data),
            rollout_id=rollout_id,
        )
        for completed in completed_groups:
            data.append(completed.samples)
            consumed_group_ids.append(completed.group_id)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, accepted=%d/%d",
                async_worker.queue_size(), len(data), target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info("Async rollout collected %d groups in %.1fs (queue=%d)", len(data), elapsed, async_worker.queue_size())

    _maybe_dump_longest_trace_artifact(
        rollout_id,
        data,
        output_dir=_longest_trace_output_dir(args),
        upload_to_wandb=_env_flag("POLAR_UPLOAD_LONGEST_TRACE_ARTIFACT", default=True),
    )

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, async_worker.config.reward_key) for s in flat]
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(flat, rewards, async_worker.config.reward_key))
    metrics.update(async_worker.snapshot_metrics())
    metrics["polar/rollout/wait_seconds"] = elapsed
    metrics["polar/rollout/groups_per_second"] = len(data) / max(elapsed, 1e-9)
    metrics["polar/staleness/max_allowed"] = float(async_worker.config.max_off_policy_steps)
    metrics["polar/scheduler/prefetch_target_groups"] = float(
        async_worker.prefetch_target_groups()
    )
    if needs_quiesce:
        asyncio.run(_set_colocated_inference_paused(args, config, paused=True))
    if getattr(data_source, "finite_first_epoch", False):
        data_source.record_consumed_groups(rollout_id, consumed_group_ids)
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _maybe_dump_longest_trace_artifact(
    rollout_id: int,
    data: list[list[Any]],
    *,
    interval: int = _LONGEST_TRACE_ARTIFACT_INTERVAL,
    output_dir: Path | None = None,
    upload_to_wandb: bool = True,
) -> None:
    """Persist the longest session and optionally upload it without blocking rollout.

    Groups samples by ``session_id``, picks the session with the largest
    aggregated assistant tokens, and writes its full message chain per trace.
    W&B artifact uploads run in a single daemon thread because W&B performs a
    synchronous API-key validation that can retry indefinitely on network
    failures.
    """
    if interval <= 0 or rollout_id % interval != 0:
        return
    if output_dir is None:
        return

    by_session: dict[str, list[Any]] = {}
    for group in data:
        for sample in group:
            sid = getattr(sample, "session_id", None) or "unknown"
            by_session.setdefault(sid, []).append(sample)
    if not by_session:
        return

    def _session_tokens(samples: list[Any]) -> int:
        return sum(int(getattr(s, "response_length", 0) or 0) for s in samples)

    longest_sid, longest_samples = max(by_session.items(), key=lambda kv: _session_tokens(kv[1]))
    total_tokens = _session_tokens(longest_samples)
    if total_tokens <= 0:
        return

    longest_samples = sorted(
        longest_samples,
        key=lambda s: int((s.metadata.get("polar") or {}).get("trace_index", 0) or 0),
    )
    traces = []
    for sample in longest_samples:
        polar_meta = sample.metadata.get("polar") or {}
        trace_debug = polar_meta.get("trace_debug") or {}
        status = getattr(sample, "status", None)
        loss_mask = getattr(sample, "loss_mask", None) or []
        traces.append({
            "trace_index": polar_meta.get("trace_index"),
            "finish_reason": trace_debug.get("finish_reason"),
            "response_length": int(getattr(sample, "response_length", 0) or 0),
            "trainable_tokens": sum(int(value) != 0 for value in loss_mask),
            "total_sequence_tokens": len(getattr(sample, "tokens", None) or []),
            "status": getattr(status, "value", None) if status is not None else None,
            "prompt_messages": sample.prompt if isinstance(sample.prompt, list) else [],
            "response_messages": trace_debug.get("response_messages") or [],
        })

    first = longest_samples[0]
    first_meta = first.metadata.get("polar") or {}
    reward = getattr(first, "reward", None)
    if isinstance(reward, dict):
        session_reward = float(reward.get("score", 0.0))
    elif isinstance(reward, (int, float)):
        session_reward = float(reward)
    else:
        session_reward = 0.0

    payload = {
        "rollout_id": int(rollout_id),
        "session_id": longest_sid,
        "task_id": first_meta.get("task_id"),
        "node_id": first_meta.get("node_id"),
        "total_assistant_tokens": int(total_tokens),
        "session_reward": session_reward,
        "num_traces": len(traces),
        "trajectory_metadata": first_meta.get("trajectory_metadata") or {},
        "traces": traces,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"longest_trace_r{rollout_id}.json"
    try:
        fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.exception("Failed to persist longest-trace artifact to %s", fpath)
        return

    logger.info(
        "Saved longest-trace artifact rollout=%d session=%s traces=%d tokens=%d path=%s",
        rollout_id,
        longest_sid,
        len(traces),
        total_tokens,
        fpath,
    )

    if not upload_to_wandb:
        return
    if not _longest_trace_upload_lock.acquire(blocking=False):
        logger.warning(
            "Skipping W&B upload for %s because a previous trace artifact upload is still running",
            fpath,
        )
        return

    threading.Thread(
        target=_upload_longest_trace_artifact,
        args=(rollout_id, fpath),
        name=f"wandb-trace-artifact-r{rollout_id}",
        daemon=True,
    ).start()


def _upload_longest_trace_artifact(rollout_id: int, fpath: Path) -> None:
    try:
        import wandb

        from slime.utils.wandb_utils import _wandb_network_context

        with _wandb_network_context():
            run = getattr(wandb, "run", None)
            if run is None:
                return
            artifact = wandb.Artifact(
                name=f"longest_trace_r{rollout_id}", type="rollout-trace"
            )
            artifact.add_file(str(fpath))
            run.log_artifact(artifact)
        logger.info("Queued longest-trace W&B artifact upload for rollout=%d", rollout_id)
    except Exception:
        logger.exception("Failed to log longest-trace W&B artifact")
    finally:
        _longest_trace_upload_lock.release()


def _longest_trace_output_dir(args: Any) -> Path | None:
    override = os.environ.get("POLAR_LONGEST_TRACE_DIR")
    if override:
        return Path(override).expanduser()
    wandb_dir = getattr(args, "wandb_dir", None)
    if wandb_dir:
        return Path(wandb_dir).expanduser().parent / "polar_trace_artifacts"
    return None


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def _polar_extra_metrics(
    flat_samples: list[Any],
    rewards: list[float],
    reward_key: str,
) -> dict[str, float]:
    """Compact user-facing Polar metrics for W&B."""
    out: dict[str, float] = {}
    seen: set[str] = set()
    timing_by_phase: dict[str, list[float]] = {
        "register_to_init_queue": [],
        "init": [],
        "run": [],
        "postrun": [],
    }
    end_to_end_ms: list[float] = []
    session_is_placeholder: dict[str, bool] = {}
    session_report: dict[str, dict[str, Any]] = {}
    completed_session_rewards: list[float] = []
    session_rewards: list[float] = []
    session_policy_staleness: dict[str, float] = {}
    session_policy_version: dict[str, float] = {}
    code_verifier_scores: list[float] = []
    llm_judge_scores: list[float] = []
    llm_judge_durations: list[float] = []
    llm_judge_complete: list[float] = []
    llm_judge_expected = 0
    trace_counts: list[float] = []
    replayed_response_realign_candidates: list[float] = []
    replayed_response_candidate_trainable_tokens: list[float] = []
    replayed_response_candidate_context_tokens: list[float] = []
    replayed_response_realignments: list[float] = []
    replayed_response_masked_tokens: list[float] = []
    trainable_sample_counts: dict[str, int] = {}
    trainable_token_counts: dict[str, int] = {}
    response_token_counts: dict[str, int] = {}
    record_counts: list[float] = []
    status_counts: dict[str, int] = {}
    session_statuses: dict[str, str] = {}
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        session_id = polar_meta.get("session_id")
        is_placeholder = bool(polar_meta.get("placeholder"))
        if not session_id:
            continue
        if "policy_staleness" in polar_meta:
            session_policy_staleness[session_id] = float(polar_meta["policy_staleness"])
        if "policy_version" in polar_meta:
            session_policy_version[session_id] = float(polar_meta["policy_version"])
        loss_mask = getattr(sample, "loss_mask", None) or []
        trainable_sample_counts.setdefault(session_id, 0)
        trainable_token_counts[session_id] = trainable_token_counts.get(session_id, 0) + sum(
            int(value) for value in loss_mask
        )
        response_token_counts[session_id] = response_token_counts.get(session_id, 0) + int(
            getattr(sample, "response_length", 0) or 0
        )
        if any(int(value) for value in loss_mask):
            trainable_sample_counts[session_id] = trainable_sample_counts.get(session_id, 0) + 1
        if session_id not in seen:
            seen.add(session_id)
            session_rewards.append(_extract_sample_reward(sample, reward_key))
            timing = polar_meta.get("timing") or {}
            session_timing: list[float] = []
            timing_complete = True
            for phase in timing_by_phase:
                value = timing.get(f"{phase}_ms")
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    timing_by_phase[phase].append(float(value))
                    session_timing.append(float(value))
                else:
                    timing_complete = False
            if timing_complete:
                end_to_end_ms.append(sum(session_timing))
            session_is_placeholder[session_id] = is_placeholder
            trajectory_metadata = polar_meta.get("trajectory_metadata") or {}
            evaluation = trajectory_metadata.get("evaluation") or {}
            report = evaluation.get("report") or {}
            if isinstance(report, dict) and report:
                session_report[session_id] = report
            code_score = evaluation.get("code_verifier_score")
            if isinstance(code_score, (int, float)) and math.isfinite(float(code_score)):
                code_verifier_scores.append(float(code_score))
            judge_score = evaluation.get("llm_judge_score")
            judge_expected = any(str(key).startswith("llm_judge_") for key in evaluation)
            if judge_expected:
                llm_judge_expected += 1
                llm_judge_complete.append(
                    float(bool(evaluation.get("llm_judge_result_complete")))
                )
            if isinstance(judge_score, (int, float)) and math.isfinite(float(judge_score)):
                llm_judge_scores.append(float(judge_score))
            judge_duration = evaluation.get("llm_judge_duration_seconds")
            if isinstance(judge_duration, (int, float)) and math.isfinite(float(judge_duration)):
                llm_judge_durations.append(float(judge_duration))
            trace_count = trajectory_metadata.get("trace_count")
            if isinstance(trace_count, (int, float)) and math.isfinite(float(trace_count)):
                trace_counts.append(float(trace_count))
            record_count = trajectory_metadata.get("record_count")
            if isinstance(record_count, (int, float)) and math.isfinite(float(record_count)):
                record_counts.append(float(record_count))
            reconstruction_stats = trajectory_metadata.get("reconstruction_stats") or {}
            realignment_candidates = reconstruction_stats.get(
                "replayed_response_realign_candidates"
            )
            if isinstance(realignment_candidates, (int, float)) and math.isfinite(
                float(realignment_candidates)
            ):
                replayed_response_realign_candidates.append(
                    float(realignment_candidates)
                )
            candidate_trainable_tokens = reconstruction_stats.get(
                "replayed_response_candidate_trainable_tokens"
            )
            if isinstance(candidate_trainable_tokens, (int, float)) and math.isfinite(
                float(candidate_trainable_tokens)
            ):
                replayed_response_candidate_trainable_tokens.append(
                    float(candidate_trainable_tokens)
                )
            candidate_context_tokens = reconstruction_stats.get(
                "replayed_response_candidate_context_tokens"
            )
            if isinstance(candidate_context_tokens, (int, float)) and math.isfinite(
                float(candidate_context_tokens)
            ):
                replayed_response_candidate_context_tokens.append(
                    float(candidate_context_tokens)
                )
            realignment_count = reconstruction_stats.get(
                "replayed_response_realignments"
            )
            if isinstance(realignment_count, (int, float)) and math.isfinite(
                float(realignment_count)
            ):
                replayed_response_realignments.append(float(realignment_count))
            realigned_masked_tokens = reconstruction_stats.get(
                "replayed_response_masked_tokens"
            )
            if isinstance(realigned_masked_tokens, (int, float)) and math.isfinite(
                float(realigned_masked_tokens)
            ):
                replayed_response_masked_tokens.append(
                    float(realigned_masked_tokens)
                )
            status = _sample_session_status(sample) or "UNKNOWN"
            session_statuses[session_id] = status
            status_counts[status] = status_counts.get(status, 0) + 1
            if _sample_session_status(sample) == "COMPLETED" and not is_placeholder:
                completed_session_rewards.append(
                    _extract_sample_reward(sample, reward_key)
                )

    for name, values in timing_by_phase.items():
        if values:
            for stat, value in _metric_distribution(values).items():
                out[f"polar/session_ms/{name}_{stat}"] = value
            out[f"polar/session_ms/{name}_coverage"] = len(values) / len(seen)
    if end_to_end_ms:
        for stat, value in _metric_distribution(end_to_end_ms).items():
            out[f"polar/session_ms/end_to_end_{stat}"] = value
        out["polar/session_ms/end_to_end_coverage"] = len(end_to_end_ms) / len(seen)
    run_ms = timing_by_phase["run"]
    if run_ms:
        out["polar/session_ms/run_straggler_gap"] = max(run_ms) - statistics.median(run_ms)
    aggregate_rewards = session_rewards or rewards
    reward_distribution = _metric_distribution(aggregate_rewards)
    if "mean" in reward_distribution:
        out["polar/reward_mean"] = reward_distribution["mean"]
        out["polar/reward_std"] = reward_distribution["std"]
        for stat, value in reward_distribution.items():
            out[f"polar/reward/{stat}"] = value
    completed_reward_distribution = _metric_distribution(completed_session_rewards)
    if "mean" in completed_reward_distribution:
        out["polar/reward_mean_completed"] = completed_reward_distribution["mean"]
    policy_staleness = _finite_values(list(session_policy_staleness.values()))
    if policy_staleness:
        for stat, value in _metric_distribution(policy_staleness).items():
            out[f"polar/staleness/{stat}"] = value
        out["polar/staleness/on_policy_fraction"] = (
            sum(value == 0 for value in policy_staleness) / len(policy_staleness)
        )
        out["polar/staleness/off_policy_fraction"] = (
            sum(value > 0 for value in policy_staleness) / len(policy_staleness)
        )
    if session_policy_version:
        versions = _finite_values(list(session_policy_version.values()))
        if versions:
            out["polar/policy_version/min"] = min(versions)
            out["polar/policy_version/max"] = max(versions)
    if code_verifier_scores:
        out["polar/code_verifier/mean"] = sum(code_verifier_scores) / len(code_verifier_scores)
    if llm_judge_scores:
        out["polar/llm_judge/mean"] = sum(llm_judge_scores) / len(llm_judge_scores)
        out["polar/llm_judge/scored_count"] = float(len(llm_judge_scores))
    if llm_judge_complete:
        out["polar/llm_judge/complete_rate"] = sum(llm_judge_complete) / len(
            llm_judge_complete
        )
        out["polar/llm_judge/expected_count"] = float(llm_judge_expected)
        out["polar/llm_judge/complete_count"] = sum(llm_judge_complete)
        out["polar/llm_judge/incomplete_count"] = len(llm_judge_complete) - sum(
            llm_judge_complete
        )
    if llm_judge_durations:
        for stat, value in _metric_distribution(llm_judge_durations).items():
            out[f"polar/llm_judge/duration_seconds_{stat}"] = value
    if trace_counts:
        for stat, value in _metric_distribution(trace_counts).items():
            out[f"polar/trajectory/trace_count_{stat}"] = value
        out["polar/trajectory/multi_trace_fraction"] = sum(
            value > 1 for value in trace_counts
        ) / len(trace_counts)
    if record_counts:
        for stat, value in _metric_distribution(record_counts).items():
            out[f"polar/trajectory/llm_request_count_{stat}"] = value
    if replayed_response_realign_candidates:
        out["polar/trajectory/replayed_response_realign_candidates_sum"] = sum(
            replayed_response_realign_candidates
        )
        out["polar/trajectory/replayed_response_realign_candidates_mean"] = sum(
            replayed_response_realign_candidates
        ) / len(replayed_response_realign_candidates)
    if replayed_response_candidate_trainable_tokens:
        out[
            "polar/trajectory/replayed_response_candidate_trainable_tokens_sum"
        ] = sum(replayed_response_candidate_trainable_tokens)
        out[
            "polar/trajectory/replayed_response_candidate_trainable_tokens_mean"
        ] = sum(replayed_response_candidate_trainable_tokens) / len(
            replayed_response_candidate_trainable_tokens
        )
    if replayed_response_candidate_context_tokens:
        out[
            "polar/trajectory/replayed_response_candidate_context_tokens_sum"
        ] = sum(replayed_response_candidate_context_tokens)
        out[
            "polar/trajectory/replayed_response_candidate_context_tokens_mean"
        ] = sum(replayed_response_candidate_context_tokens) / len(
            replayed_response_candidate_context_tokens
        )
    if replayed_response_realignments:
        out["polar/trajectory/replayed_response_realignments_sum"] = sum(
            replayed_response_realignments
        )
        out["polar/trajectory/replayed_response_realignments_mean"] = sum(
            replayed_response_realignments
        ) / len(replayed_response_realignments)
    if replayed_response_masked_tokens:
        out["polar/trajectory/replayed_response_masked_tokens_sum"] = sum(
            replayed_response_masked_tokens
        )
        out["polar/trajectory/replayed_response_masked_tokens_mean"] = sum(
            replayed_response_masked_tokens
        ) / len(replayed_response_masked_tokens)
    if trainable_sample_counts:
        counts = [float(value) for value in trainable_sample_counts.values()]
        for stat, value in _metric_distribution(counts).items():
            out[f"polar/trajectory/trainable_samples_{stat}"] = value
        out["polar/trajectory/zero_trainable_sample_fraction"] = sum(
            value == 0 for value in counts
        ) / len(counts)
    if trainable_token_counts:
        counts = [float(value) for value in trainable_token_counts.values()]
        for stat, value in _metric_distribution(counts).items():
            out[f"polar/trajectory/trainable_tokens_{stat}"] = value
    if response_token_counts:
        counts = [float(value) for value in response_token_counts.values()]
        for stat, value in _metric_distribution(counts).items():
            out[f"polar/trajectory/response_tokens_{stat}"] = value
        total_response = sum(counts)
        if total_response > 0:
            out["polar/trajectory/trainable_token_fraction"] = (
                sum(trainable_token_counts.values()) / total_response
            )

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    if total_sessions > 0:
        out["polar/usable_session_rate"] = (
            total_sessions - empty_sessions
        ) / total_sessions
        completed_usable = sum(
            status == "COMPLETED" and not session_is_placeholder.get(session_id, False)
            for session_id, status in session_statuses.items()
        )
        out["polar/rollout_success_rate"] = completed_usable / total_sessions
        out["polar/completed_session_rate"] = max(0, completed_usable) / total_sessions
        out["polar/sessions/total"] = float(total_sessions)
        for status, count in status_counts.items():
            out[f"polar/sessions/status_{status.lower()}"] = float(count)
            out[f"polar/sessions/status_{status.lower()}_fraction"] = count / total_sessions
    if session_report:
        graded_sessions = len(session_report)
        resolved = sum(1 for r in session_report.values() if r.get("resolved"))
        out["polar/eval/resolved_rate"] = resolved / graded_sessions
    return out


def _finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _metric_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(_finite_values(values))
    if not ordered:
        return {"count": 0.0, "finite_fraction": 0.0}

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    mean = sum(ordered) / len(ordered)
    return {
        "count": float(len(ordered)),
        "finite_fraction": len(ordered) / len(values),
        "mean": mean,
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _load_rollout_train_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar rollouts from a Slime trainer."
        ) from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar evaluation rollouts from a Slime trainer."
        ) from exc
    return RolloutFnEvalOutput


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to build Polar evaluation samples from eval datasets."
        ) from exc
    return Sample


atexit.register(stop_global_worker)
