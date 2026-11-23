"""Gateway-node registration and stage-aware scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import logging
import os
import threading

from polar.rollout.models import GatewayNodeInfo, NodeRegistrationRequest, NodeStageMetrics


logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class GatewayNode:
    node_id: str
    gateway_url: str
    max_init_workers: int
    max_run_workers: int
    max_postrun_workers: int
    healthy: bool
    last_heartbeat: datetime
    heartbeat_interval_seconds: int
    metrics: NodeStageMetrics = field(default_factory=NodeStageMetrics)
    dispatch_reservations: int = 0
    draining: bool = False
    consecutive_step0_failures: int = 0
    circuit_open: bool = False

    @property
    def total_sessions(self) -> int:
        return self.metrics.total_sessions + self.dispatch_reservations

    @property
    def ready_buffer_target(self) -> int:
        """Ready buffer tracks run capacity one-for-one."""
        return self.max_run_workers

    def to_model(self) -> GatewayNodeInfo:
        return GatewayNodeInfo(
            node_id=self.node_id,
            gateway_url=self.gateway_url,
            max_init_workers=self.max_init_workers,
            max_run_workers=self.max_run_workers,
            max_postrun_workers=self.max_postrun_workers,
            metrics=self.metrics.model_copy(),
            dispatch_reservations=self.dispatch_reservations,
            healthy=self.healthy,
            draining=self.draining,
            consecutive_step0_failures=self.consecutive_step0_failures,
            circuit_open=self.circuit_open,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            last_heartbeat=self.last_heartbeat,
        )


class NodeScheduler:
    """Track registered nodes and assign sessions using stage-aware pressure."""

    def __init__(
        self,
        *,
        bootstrap_nodes: list[dict[str, object]] | None = None,
        stale_factor: float = 2.5,
        step0_failure_threshold: int | None = None,
    ) -> None:
        self._nodes: dict[str, GatewayNode] = {}
        self._lock = threading.RLock()
        self._stale_factor = stale_factor
        if step0_failure_threshold is None:
            step0_failure_threshold = int(
                os.environ.get("POLAR_NODE_STEP0_FAILURE_THRESHOLD", "3")
            )
        if step0_failure_threshold <= 0:
            raise ValueError("step0_failure_threshold must be positive")
        self._step0_failure_threshold = step0_failure_threshold
        self._quarantined_nodes: set[str] = set()
        self._task_affinity: dict[str, str] = {}

        for node in bootstrap_nodes or []:
            node_id = str(node["node_id"])
            max_run_workers = max(1, int(node.get("max_run_workers", node.get("capacity", 1))))
            self._nodes[node_id] = GatewayNode(
                node_id=node_id,
                gateway_url=str(node["gateway_url"]).rstrip("/"),
                max_init_workers=max(1, int(node.get("max_init_workers", 4))),
                max_run_workers=max_run_workers,
                max_postrun_workers=max(1, int(node.get("max_postrun_workers", 4))),
                healthy=False,
                last_heartbeat=_utcnow(),
                heartbeat_interval_seconds=max(1, int(node.get("heartbeat_interval_seconds", 30))),
            )

    def register_node(self, request: NodeRegistrationRequest) -> GatewayNodeInfo:
        now = _utcnow()
        with self._lock:
            existing = self._nodes.get(request.node_id)
            circuit_open = request.node_id in self._quarantined_nodes or bool(
                existing is not None and existing.circuit_open
            )
            draining = circuit_open or (
                existing.draining if existing is not None else False
            )
            metrics = existing.metrics.model_copy() if existing is not None else NodeStageMetrics()
            self._nodes[request.node_id] = GatewayNode(
                node_id=request.node_id,
                gateway_url=request.gateway_url.rstrip("/"),
                max_init_workers=request.max_init_workers,
                max_run_workers=request.max_run_workers,
                max_postrun_workers=request.max_postrun_workers,
                metrics=metrics,
                dispatch_reservations=0,
                healthy=True,
                last_heartbeat=now,
                heartbeat_interval_seconds=request.heartbeat_interval_seconds,
                draining=draining,
                consecutive_step0_failures=(
                    existing.consecutive_step0_failures if existing is not None else 0
                ),
                circuit_open=circuit_open,
            )
            return self._nodes[request.node_id].to_model()

    def heartbeat(self, node_id: str, *, metrics: NodeStageMetrics) -> GatewayNodeInfo:
        with self._lock:
            node = self._require_node_locked(node_id)
            was_healthy = node.healthy
            node.last_heartbeat = _utcnow()
            node.healthy = bool(metrics.inference_healthy)
            node.metrics = metrics
            node.dispatch_reservations = 0
            if node.healthy != was_healthy:
                log = logger.info if node.healthy else logger.error
                log(
                    "Rollout gateway schedulability changed node_id=%s healthy=%s "
                    "probe_latency_ms=%s inference_error=%s",
                    node_id,
                    node.healthy,
                    metrics.inference_probe_latency_ms,
                    metrics.inference_error,
                )
            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def acquire_node(self, *, task_id: str | None = None) -> GatewayNodeInfo | None:
        with self._lock:
            self._refresh_health_locked()
            affinity_node_id = self._task_affinity.get(task_id) if task_id else None
            if affinity_node_id is not None:
                affinity_node = self._nodes.get(affinity_node_id)
                if affinity_node is not None and affinity_node.healthy and not affinity_node.draining:
                    # Preserve group affinity while the selected node is full.
                    # Reassigning only the tail of a GRPO group would defeat
                    # prefix-cache locality and split one task across gateways.
                    if not self._can_accept_session(affinity_node):
                        return None
                    affinity_node.dispatch_reservations += 1
                    return affinity_node.to_model()
                self._task_affinity.pop(task_id, None)

            candidates = [
                node
                for node in self._nodes.values()
                if node.healthy and not node.draining and self._can_accept_session(node)
            ]
            if not candidates:
                return None

            selected = min(candidates, key=self._node_score)
            selected.dispatch_reservations += 1
            if task_id:
                self._task_affinity[task_id] = selected.node_id
            return selected.to_model()

    def release_task_affinity(self, task_id: str) -> None:
        """Forget a terminal task's gateway pin."""
        with self._lock:
            self._task_affinity.pop(task_id, None)

    def refresh_health(self) -> None:
        """Recompute stored health from heartbeat timestamps (call from a tick task)."""
        with self._lock:
            self._refresh_health_locked()

    def release_reservation(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.dispatch_reservations = max(0, node.dispatch_reservations - 1)
            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def mark_unhealthy(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            node.healthy = False
            node.metrics = node.metrics.model_copy(
                update={
                    "inference_healthy": False,
                    "inference_error": "gateway transport failure",
                }
            )
            return node.to_model()

    def drain_node(self, node_id: str) -> GatewayNodeInfo:
        with self._lock:
            node = self._require_node_locked(node_id)
            node.draining = True
            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def record_session_result(
        self,
        node_id: str,
        *,
        status: str,
        error: str | None,
        failure_stage: str | None = None,
    ) -> GatewayNodeInfo | None:
        """Track node-local step-0 failures and quarantine repeat offenders.

        Task timeouts, evaluator failures, and low rewards are not node-health
        signals. Only repeated agent process failures before the first step can
        open the circuit.
        """
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None

            if str(failure_stage or "").lower() == "infrastructure":
                node.healthy = False
                node.metrics = node.metrics.model_copy(
                    update={
                        "inference_healthy": False,
                        "inference_error": (error or "infrastructure failure")[:512],
                    }
                )
                logger.error(
                    "Temporarily quarantining rollout gateway after infrastructure "
                    "failure node_id=%s status=%s error=%s",
                    node_id,
                    status,
                    (error or "")[:512],
                )

            if status == "ERROR" and self._is_step0_exec_failure(error):
                node.consecutive_step0_failures += 1
                if (
                    node.consecutive_step0_failures
                    >= self._step0_failure_threshold
                    and not node.circuit_open
                    and len(self._nodes) > 1
                ):
                    node.circuit_open = True
                    node.draining = True
                    self._quarantined_nodes.add(node_id)
                    logger.error(
                        "Opening rollout node circuit node_id=%s after %d "
                        "consecutive step-0 failures",
                        node_id,
                        node.consecutive_step0_failures,
                    )
            elif not node.circuit_open and str(failure_stage or "").lower() != "infrastructure":
                node.consecutive_step0_failures = 0

            snapshot = node.to_model()
            self._maybe_remove_drained_locked(node_id, node)
            return snapshot

    def get_node(self, node_id: str) -> GatewayNodeInfo | None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                return None
            return self._fresh_model_locked(node, _utcnow())

    def list_nodes(self) -> list[GatewayNodeInfo]:
        with self._lock:
            now = _utcnow()
            return [
                self._fresh_model_locked(node, now)
                for node in sorted(self._nodes.values(), key=lambda item: item.node_id)
            ]

    def stats(self) -> dict[str, object]:
        with self._lock:
            now = _utcnow()
            return {
                "nodes": [
                    self._fresh_model_locked(node, now).model_dump(mode="json")
                    for node in sorted(self._nodes.values(), key=lambda item: item.node_id)
                ],
            }

    def _fresh_model_locked(self, node: GatewayNode, now: datetime) -> GatewayNodeInfo:
        """Return a snapshot with heartbeat-derived health but no mutation."""
        return replace(node, healthy=self._is_node_healthy(node, now)).to_model()

    def _is_node_healthy(self, node: GatewayNode, now: datetime) -> bool:
        timeout_seconds = max(1.0, node.heartbeat_interval_seconds * self._stale_factor)
        return bool(node.metrics.inference_healthy) and (
            (now - node.last_heartbeat).total_seconds() <= timeout_seconds
        )

    def _refresh_health_locked(self) -> None:
        now = _utcnow()
        for node in self._nodes.values():
            node.healthy = self._is_node_healthy(node, now)

    @staticmethod
    def _node_score(node: GatewayNode) -> tuple[float, float, float, float, float, str]:
        run_pressure = node.metrics.run_inflight / node.max_run_workers
        postrun_pressure = (
            node.metrics.postrun_inflight + node.metrics.postrun_queue_depth
        ) / node.max_postrun_workers
        init_pressure = (
            node.metrics.init_inflight
            + node.metrics.init_queue_depth
            + node.dispatch_reservations
        ) / node.max_init_workers
        ready_gap = max(0, node.ready_buffer_target - node.metrics.ready_depth)
        admission_limit = (
            node.max_init_workers
            + node.ready_buffer_target
            + node.max_run_workers
            + node.max_postrun_workers
        )
        total_pressure = node.total_sessions / admission_limit
        # Heartbeats can lag a burst of dispatches.  If run pressure is the
        # lexicographic primary key, one residual running session on an
        # otherwise idle node can send the whole burst to its peer because
        # dispatch reservations never get a chance to affect the decision.
        # Keep preferring nodes with spare run capacity, but let accumulated
        # admission pressure balance a fast burst before a node fills up.
        load_pressure = run_pressure + total_pressure
        return (
            load_pressure,
            postrun_pressure,
            init_pressure,
            -float(ready_gap),
            total_pressure,
            node.node_id,
        )

    @staticmethod
    def _can_accept_session(node: GatewayNode) -> bool:
        admission_limit = (
            node.max_init_workers
            + node.ready_buffer_target
            + node.max_run_workers
            + node.max_postrun_workers
        )
        if node.total_sessions >= admission_limit:
            return False
        postrun_backlog = node.metrics.postrun_inflight + node.metrics.postrun_queue_depth
        return postrun_backlog < (node.max_postrun_workers * 2)

    def _require_node_locked(self, node_id: str) -> GatewayNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown gateway node: {node_id}")
        return node

    def _maybe_remove_drained_locked(self, node_id: str, node: GatewayNode) -> None:
        if node.draining and node.total_sessions == 0:
            self._nodes.pop(node_id, None)

    @staticmethod
    def _is_step0_exec_failure(error: str | None) -> bool:
        normalized = (error or "").strip().lower()
        return (
            "step 0 exited with code" in normalized
            or "step 0 failed with exit code" in normalized
            # A gateway whose local inference worker has disappeared accepts
            # the session but cannot record even the first completion.  The
            # trajectory builder deliberately reports that exact terminal
            # error.  Treat repeated occurrences like the existing step-0
            # process failures so the node circuit opens and the next task is
            # remapped instead of retrying forever on the dead worker.
            or normalized == "no completions"
        )
