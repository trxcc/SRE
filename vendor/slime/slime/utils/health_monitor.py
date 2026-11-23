import json
import logging
import os
import threading

import ray
import requests

from slime.utils.fatal_resume import emit_fatal_resume


logger = logging.getLogger(__name__)


def _is_timeout_error(error: BaseException) -> bool:
    """Return whether *error* is a local or Ray-wrapped timeout.

    ``RayTaskError.as_instanceof_cause`` preserves the remote cause as a
    dynamically generated base class, so ``isinstance`` still works for a
    requests timeout raised inside ``SGLangEngine``.
    """
    candidate = error
    as_instanceof_cause = getattr(error, "as_instanceof_cause", None)
    if callable(as_instanceof_cause):
        try:
            candidate = as_instanceof_cause()
        except Exception:
            candidate = error
    return isinstance(
        candidate,
        (
            TimeoutError,
            requests.exceptions.Timeout,
            ray.exceptions.GetTimeoutError,
        ),
    )


class RolloutHealthMonitor:
    """Health monitor for rollout engines.

    The monitor runs continuously once started, but can be paused/resumed
    based on whether the engines are offloaded (cannot health check when offloaded).

    Lifecycle:
    - start(): Start the monitor thread (called once during initialization)
    - pause(): Pause health checking (called when offloading engines)
    - resume(): Resume health checking (called when onloading engines)
    - stop(): Stop the monitor thread completely (called during dispose)
    """

    def __init__(self, server_group, args):
        self._server_group = server_group

        self._thread = None
        self._stop_event = None
        self._pause_event = None  # When set, health checking is paused
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True  # Need to wait after each resume
        self._is_checking_enabled = False  # Track if health checking should be active
        # Serialize a monitor probe with the refit-boundary recovery path.  A
        # pause request prevents new probes, while this lock lets the caller
        # wait for an already-running bounded probe to finish before it
        # consumes the quarantine state.
        self._check_lock = threading.Lock()
        self._failure_streak: dict[int, int] = {}
        self._quarantined_engines: set[int] = set()
        self._quarantine_after = int(
            os.environ.get("SLIME_ROLLOUT_QUARANTINE_FAILURES", "2")
        )
        if self._quarantine_after < 1:
            raise ValueError("SLIME_ROLLOUT_QUARANTINE_FAILURES must be positive")

    def start(self) -> bool:
        """Start the health monitor thread. Called once during initialization.

        Returns:
            True if the monitor was started, False if there are no engines to monitor.
        """
        if not self._server_group.all_engines:
            return False

        if self._thread is not None:
            logger.warning("Health monitor thread is already running.")
            return True

        logger.info("Starting RolloutHealthMonitor...")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start in paused state until resume() is called
        self._thread = threading.Thread(
            target=self._health_monitor_loop,
            name="RolloutHealthMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("RolloutHealthMonitor started (in paused state).")
        return True

    def stop(self) -> None:
        """Stop the health monitor thread completely. Called during dispose."""
        if not self._thread:
            return

        logger.info("Stopping RolloutHealthMonitor...")
        assert self._stop_event is not None
        self._stop_event.set()
        # Also clear pause to let the thread exit
        if self._pause_event:
            self._pause_event.clear()
        timeout = self._check_timeout + self._check_interval + 5
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logging.warning("Rollout health monitor thread did not terminate within %.1fs", timeout)
        else:
            logger.info("RolloutHealthMonitor stopped.")

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        self._is_checking_enabled = False

    def pause(self) -> None:
        """Pause health checking. Called when engines are offloaded."""
        if self._pause_event is None:
            return
        logger.info("Pausing health monitor...")
        self._pause_event.set()
        self._is_checking_enabled = False

    def resume(self) -> None:
        """Resume health checking. Called when engines are onloaded."""
        if self._pause_event is None:
            return
        logger.info("Resuming health monitor...")
        self._need_first_wait = True  # Need to wait after each resume
        self._pause_event.clear()
        self._is_checking_enabled = True

    def is_checking_enabled(self) -> bool:
        """Return whether health checking is currently enabled (not paused)."""
        return self._is_checking_enabled

    @property
    def server_group(self):
        """Return the server group owned by this monitor."""
        return self._server_group

    def has_quarantined_engines(self) -> bool:
        """Return whether an engine is isolated but still represented by a live actor."""
        with self._check_lock:
            return bool(self._quarantined_engines)

    def replace_quarantined_engines_for_refit(self) -> int:
        """Turn isolated live actors into ordinary dead-engine recovery slots.

        During active generation, dual HTTP timeouts are not sufficient proof
        that an engine is dead: a saturated SGLang server may simply starve
        probes.  At the refit boundary all rollout futures have been joined,
        so an engine that is still quarantined must not participate in
        ``pause_generation`` or the distributed weight-update collective.
        Replace it here, then let ``RolloutServer.recover`` construct a fresh
        actor which receives current weights before router activation.

        The manager pauses monitoring before calling this method.  The lock
        also waits for any probe already in progress to finish.
        """
        with self._check_lock:
            engine_ids = sorted(self._quarantined_engines)
            for rollout_engine_id in engine_ids:
                logger.warning(
                    "Replacing quarantined rollout engine %s at the safe refit boundary",
                    rollout_engine_id,
                )
                self._kill_engine(
                    rollout_engine_id=rollout_engine_id,
                    reason="quarantined engine remained unavailable at refit boundary",
                    already_quarantined=True,
                )
            return len(engine_ids)

    def _health_monitor_loop(self) -> None:
        assert self._stop_event is not None
        assert self._pause_event is not None

        while not self._stop_event.is_set():
            # Wait while paused
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            # Do first wait after each resume (for large MoE models to be ready)
            if self._need_first_wait:
                logger.info(f"Health monitor doing first wait after resume: {self._check_first_wait}s")
                if self._stop_event.wait(self._check_first_wait):
                    logger.info("Health monitor stopped during first wait.")
                    break
                if self._pause_event.is_set():
                    # Got paused during first wait, skip this round and wait again next resume
                    logger.info("Health monitor paused during first wait, will wait again next resume.")
                    continue
                self._need_first_wait = False

            # Run health checks
            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self._run_health_checks()

            # Wait for next check interval
            if self._stop_event.wait(self._check_interval):
                break

    def _run_health_checks(self) -> None:
        for rollout_engine_id, engine in enumerate(self._server_group.engines):
            if self._stop_event is not None and self._stop_event.is_set():
                break
            if self._pause_event is not None and self._pause_event.is_set():
                break
            with self._check_lock:
                self._check_engine_health(rollout_engine_id, engine)

    def _check_engine_health(self, rollout_engine_id, engine) -> None:
        if engine is None:
            self._quarantined_engines.discard(rollout_engine_id)
            logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
            return

        try:
            ray.get(engine.health_generate.remote(timeout=self._check_timeout))
        except Exception as e:
            # /health_generate competes with long ClawGym requests and can
            # time out while a saturated engine is still alive.  Killing on
            # that single signal invalidates every Polar session routed to the
            # engine.  Only replace the actor when the queue-independent HTTP
            # liveness endpoint fails as well.
            try:
                ray.get(engine.health.remote(timeout=min(5.0, self._check_timeout)))
            except Exception as liveness_error:
                if _is_timeout_error(e) and _is_timeout_error(liveness_error):
                    streak = self._failure_streak.get(rollout_engine_id, 0) + 1
                    self._failure_streak[rollout_engine_id] = streak
                    if rollout_engine_id not in self._quarantined_engines:
                        diagnostics = self._quarantine_engine(
                            rollout_engine_id,
                            engine,
                            reason=(
                                "generation and liveness probes timed out; "
                                "stop routing before deciding whether to restart"
                            ),
                        )
                        self._quarantined_engines.add(rollout_engine_id)
                    else:
                        diagnostics = self._collect_diagnostics(engine)
                    logger.warning(
                        "Generation and liveness probes timed out for rollout engine %s "
                        "streak=%d/%d generation=%r liveness=%r diagnostics=%s",
                        rollout_engine_id,
                        streak,
                        self._quarantine_after,
                        e,
                        liveness_error,
                        json.dumps(diagnostics, sort_keys=True, default=str),
                    )
                    logger.warning(
                        "PRORL_DEGRADED action=watch "
                        "reason=rollout_engine_service_timeout engine=%s streak=%d",
                        rollout_engine_id,
                        streak,
                    )
                    if streak >= self._quarantine_after:
                        # HTTP timeouts are not proof that the Ray actor or its
                        # SGLang process died.  Under full ClawGym load both
                        # probes can starve even while every GPU is actively
                        # decoding.  Keep the engine out of the router and
                        # continue probing it; a later successful generation
                        # probe rejoins it.  Actual actor/process failures take
                        # the non-timeout branch below and are still replaced.
                        logger.error(
                            "Rollout engine %s exceeded the consecutive service-timeout "
                            "threshold; keeping the live actor quarantined without "
                            "ray.kill (streak=%d)",
                            rollout_engine_id,
                            streak,
                        )
                else:
                    diagnostics = self._quarantine_engine(
                        rollout_engine_id,
                        engine,
                        reason="non-timeout health check failure",
                    )
                    self._quarantined_engines.add(rollout_engine_id)
                    logger.error(
                        "Generation and liveness checks failed with a "
                        "non-timeout error for rollout engine %s; quarantining "
                        "actor. generation=%r liveness=%r diagnostics=%s",
                        rollout_engine_id,
                        e,
                        liveness_error,
                        json.dumps(diagnostics, sort_keys=True, default=str),
                    )
                    emit_fatal_resume(
                        logger,
                        "rollout_engine_non_timeout_health_failure",
                        component="rollout_health_monitor",
                        rollout_engine_id=rollout_engine_id,
                        generation_error=repr(e),
                        liveness_error=repr(liveness_error),
                        diagnostics=json.dumps(
                            diagnostics,
                            sort_keys=True,
                            default=str,
                        ),
                    )
                    self._kill_engine(
                        rollout_engine_id=rollout_engine_id,
                        reason="non-timeout health check failure",
                        already_quarantined=True,
                    )
            else:
                self._failure_streak[rollout_engine_id] = 0
                diagnostics = self._collect_diagnostics(engine)
                logger.warning(
                    "Generation health probe timed out for rollout engine %s, "
                    "but /health is live; preserving the saturated engine: %r "
                    "diagnostics=%s",
                    rollout_engine_id,
                    e,
                    json.dumps(diagnostics, sort_keys=True, default=str),
                )
        else:
            self._failure_streak[rollout_engine_id] = 0
            if rollout_engine_id in self._quarantined_engines:
                try:
                    ray.get(engine.activate_router.remote(), timeout=40.0)
                except Exception as exc:
                    logger.warning(
                        "Recovered rollout engine %s passed the monitor probe but "
                        "could not rejoin the router: %s",
                        rollout_engine_id,
                        exc,
                    )
                    return
                self._quarantined_engines.discard(rollout_engine_id)
                logger.info(
                    "Recovered rollout engine %s is healthy and rejoined the router",
                    rollout_engine_id,
                )
            logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")

    @staticmethod
    def _collect_diagnostics(engine) -> dict:
        try:
            return ray.get(
                engine.diagnostics.remote(timeout=2.0),
                timeout=10.0,
            )
        except Exception as exc:
            return {"diagnostics_error": f"{type(exc).__name__}: {exc}"[:512]}

    def _quarantine_engine(self, rollout_engine_id: int, engine, *, reason: str) -> dict:
        """Immediately stop new traffic while keeping the actor for one retry."""
        try:
            snapshot = ray.get(engine.quarantine.remote(reason=reason), timeout=12.0)
        except Exception as exc:
            snapshot = {
                "quarantine_error": f"{type(exc).__name__}: {exc}"[:512],
                **self._collect_diagnostics(engine),
            }
        self._quarantined_engines.add(rollout_engine_id)
        logger.error(
            "Rollout engine %s removed from router pending recovery snapshot=%s",
            rollout_engine_id,
            json.dumps(snapshot, sort_keys=True, default=str),
        )
        return snapshot

    def _kill_engine(
        self,
        rollout_engine_id: int,
        reason: str = "health check failure",
        *,
        already_quarantined: bool = False,
    ):
        logger.info(f"Killing server group {rollout_engine_id}...")
        for i in range(
            rollout_engine_id * self._server_group.nodes_per_engine,
            (rollout_engine_id + 1) * self._server_group.nodes_per_engine,
        ):
            engine = self._server_group.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    if not already_quarantined:
                        self._quarantine_engine(
                            rollout_engine_id,
                            engine,
                            reason=reason,
                        )
                    ray.get(engine.shutdown.remote(), timeout=30.0)
                except Exception as shutdown_error:
                    logger.warning(
                        "Graceful shutdown failed for engine at index %s: %s",
                        i,
                        shutdown_error,
                    )
                finally:
                    try:
                        ray.kill(engine)
                        logger.info(f"Successfully killed engine actor at index {i}")
                    except Exception as kill_error:
                        logger.warning(
                            "Fail to kill engine actor at index %s: %s",
                            i,
                            kill_error,
                        )
            else:
                logger.info(f"Engine at index {i} is already None")
            self._server_group.all_engines[i] = None
        self._failure_streak[rollout_engine_id] = 0
        self._quarantined_engines.discard(rollout_engine_id)
