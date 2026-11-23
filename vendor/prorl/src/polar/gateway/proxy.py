"""HTTP client for forwarding requests to an OpenAI-compatible inference server.

Backend differences (request params, response shape) are isolated in the
``InferenceEngine`` strategy this client holds; the HTTP/streaming/pause logic
here is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import Any

import httpx

from polar.gateway.engine import InferenceEngine

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Base class for upstream gateway failures."""


class UpstreamHTTPError(UpstreamError):
    """Raised when the upstream returns a non-2xx status."""

    def __init__(self, status_code: int, body: dict[str, Any] | str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(self._build_message(status_code, body))

    @staticmethod
    def _build_message(status_code: int, body: dict[str, Any] | str | None) -> str:
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            message = body.get("message")
            if isinstance(message, str) and message:
                return message
        if isinstance(body, str) and body:
            return body
        return f"Upstream request failed with status {status_code}"


class UpstreamTimeoutError(UpstreamError):
    """Raised when the upstream times out."""


class UpstreamTransportError(UpstreamError):
    """Raised for connection and transport failures."""


class InferenceClient:
    """Direct httpx client to an inference server's OpenAI-compatible API.

    Per-call bound comes from the session's remaining-timeout budget
    (`_await_with_budget` at the gateway node). The internal httpx timeout
    is a high liveness ceiling so that a stuck engine can't pin a request
    past the session deadline. The ``engine`` strategy injects backend-specific
    request params and canonicalizes responses.
    """

    _DEFAULT_LIVENESS_TIMEOUT_SECONDS = 900.0
    _DEFAULT_HEALTH_TIMEOUT_SECONDS = 3.0

    def __init__(
        self,
        base_url: str,
        engine: InferenceEngine,
        *,
        timeout_seconds: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        if timeout_seconds is None:
            timeout_seconds = float(
                os.environ.get(
                    "POLAR_INFERENCE_HTTP_TIMEOUT_SECONDS",
                    str(self._DEFAULT_LIVENESS_TIMEOUT_SECONDS),
                )
            )
        if timeout_seconds <= 0:
            raise ValueError("Inference HTTP timeout must be positive")
        self._inference_http_timeout_seconds = float(timeout_seconds)
        health_timeout_seconds = float(
            os.environ.get(
                "POLAR_GATEWAY_INFERENCE_HEALTH_TIMEOUT_SECONDS",
                str(self._DEFAULT_HEALTH_TIMEOUT_SECONDS),
            )
        )
        if not math.isfinite(health_timeout_seconds) or health_timeout_seconds <= 0:
            raise ValueError("Inference health timeout must be a positive finite number")
        self._inference_health_timeout_seconds = health_timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._direct_engine_load = {}
        self._direct_engine_urls = []
        self._direct_engine_at = 0.0
        self._direct_engine_lock = asyncio.Lock()
        self._generation_paused = False
        self._inflight_generations = 0
        self._generation_condition = asyncio.Condition()
        self._session_generation_tasks: dict[str, set[asyncio.Task[Any]]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self._inference_http_timeout_seconds, connect=30),
            )
        return self._client

    async def _completion_destination(self, client):
        # OpenAI router serialization drops SGLang's exact token extensions.
        # Discover only the already owned student engines, never the fixed judge.
        if os.environ.get("POLAR_EXACT_TOKEN_DIRECT_ENGINES", "0") != "1":
            return "/v1/chat/completions", None
        import time
        from urllib.parse import urlsplit
        async with self._direct_engine_lock:
            if not self._direct_engine_urls or time.monotonic()-self._direct_engine_at > 5:
                response = await client.get("/workers", timeout=10)
                response.raise_for_status()
                workers = response.json().get("workers")
                hosts = set(os.environ["POLAR_EXACT_TOKEN_STUDENT_HOSTS"].split(","))
                expected_engines = int(os.environ.get("POLAR_EXPECTED_ROLLOUT_ENGINES", "4"))
                if not isinstance(workers, list) or len(workers) != expected_engines:
                    raise ValueError(f"Expected {expected_engines} registered student rollout engines")
                urls = []
                for row in workers:
                    u = urlsplit(row.get("url", ""))
                    if (row.get("worker_type", "regular") != "regular" or u.scheme != "http" or
                        u.hostname not in hosts or not u.port or u.username or u.password or
                        u.path not in ("", "/") or u.query or u.fragment):
                        raise ValueError("Exact-token route escaped the owned student hosts")
                    urls.append(row["url"].rstrip("/"))
                if len(set(urls)) != expected_engines or {urlsplit(u).hostname for u in urls} != hosts:
                    raise ValueError("Unexpected student engine coverage")
                self._direct_engine_urls = sorted(urls)
                self._direct_engine_at = time.monotonic()
            url = min(self._direct_engine_urls, key=lambda u:self._direct_engine_load.get(u,0))
            self._direct_engine_load[url] = self._direct_engine_load.get(url,0)+1
            return url+"/v1/chat/completions", url

    async def _read_error_body(self, response: httpx.Response) -> dict[str, Any] | str | None:
        content = await response.aread()
        if not content:
            return None

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return

        body = await self._read_error_body(response)
        await response.aclose()
        raise UpstreamHTTPError(response.status_code, body)

    @staticmethod
    def _translate_transport_error(exc: httpx.RequestError) -> UpstreamError:
        if isinstance(exc, httpx.TimeoutException):
            return UpstreamTimeoutError("Upstream request timed out")
        return UpstreamTransportError(f"Upstream request failed: {exc}")

    async def completion(
        self,
        request: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion. Returns the full JSON response."""
        await self._acquire_generation_slot(session_id=session_id)
        client = await self._get_client()
        from copy import deepcopy

        request_copy = deepcopy(request)
        request_copy.pop("stream", None)
        request_copy["stream"] = False
        request_copy = self.engine.prepare_request(request_copy)
        direct_url = None
        try:
            destination, direct_url = await self._completion_destination(client)
            resp = await client.post(
                destination,
                json=request_copy,
                headers={"Content-Type": "application/json"},
            )
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        finally:
            if direct_url is not None:
                self._direct_engine_load[direct_url] -= 1
            await self._release_generation_slot(session_id=session_id)

        await self._raise_for_status(resp)
        return self.engine.normalize_response(resp.json())

    async def _acquire_generation_slot(self, *, session_id: str | None = None) -> None:
        async with self._generation_condition:
            await self._generation_condition.wait_for(lambda: not self._generation_paused)
            self._inflight_generations += 1
            task = asyncio.current_task()
            if session_id and task is not None:
                self._session_generation_tasks.setdefault(session_id, set()).add(task)

    async def _release_generation_slot(self, *, session_id: str | None = None) -> None:
        async with self._generation_condition:
            self._inflight_generations -= 1
            task = asyncio.current_task()
            if session_id and task is not None:
                session_tasks = self._session_generation_tasks.get(session_id)
                if session_tasks is not None:
                    session_tasks.discard(task)
                    if not session_tasks:
                        self._session_generation_tasks.pop(session_id, None)
            self._generation_condition.notify_all()

    async def cancel_session(self, session_id: str) -> int:
        """Cancel inference HTTP calls that outlived their owning agent session."""
        async with self._generation_condition:
            current = asyncio.current_task()
            tasks = [
                task
                for task in self._session_generation_tasks.get(session_id, set())
                if task is not current and not task.done()
            ]

        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(
                "Cancelled %d orphaned inference request(s) for session %s",
                len(tasks),
                session_id,
            )
        return len(tasks)

    async def pause_generation(self, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
        """Block new generation requests and wait for current inference calls to drain."""
        async with self._generation_condition:
            self._generation_paused = True
            self._generation_condition.notify_all()
            await asyncio.wait_for(
                self._generation_condition.wait_for(lambda: self._inflight_generations == 0),
                timeout=timeout_seconds,
            )
            return self.generation_status()

    async def resume_generation(self) -> dict[str, Any]:
        async with self._generation_condition:
            self._generation_paused = False
            self._generation_condition.notify_all()
            return self.generation_status()

    def generation_status(self) -> dict[str, Any]:
        return {
            "paused": self._generation_paused,
            "inflight": self._inflight_generations,
            "tracked_sessions": len(self._session_generation_tasks),
            "base_url": self.base_url,
            "engine": self.engine.name,
        }

    async def list_models(self) -> dict[str, Any]:
        """Passthrough GET /v1/models."""
        client = await self._get_client()
        try:
            resp = await client.get("/v1/models")
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        return resp.json()

    @property
    def health_timeout_seconds(self) -> float:
        return self._inference_health_timeout_seconds

    async def health(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        """Passthrough GET /health."""
        if timeout_seconds is None:
            timeout_seconds = self._inference_health_timeout_seconds
        client = await self._get_client()
        try:
            resp = await client.get("/health", timeout=timeout_seconds)
        except httpx.RequestError as exc:
            raise self._translate_transport_error(exc) from exc
        await self._raise_for_status(resp)
        content = await resp.aread()
        if not content:
            return {"status": "ok"}

        text = content.decode("utf-8", errors="replace").strip()
        if not text:
            return {"status": "ok"}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"status": "ok", "body": text}

    async def health_snapshot(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Return a bounded readiness snapshot for gateway heartbeats."""
        if timeout_seconds is None:
            timeout_seconds = self._inference_health_timeout_seconds
        started = time.monotonic()
        try:
            await self.health(timeout_seconds=timeout_seconds)
        except Exception as exc:
            return {
                "healthy": False,
                "latency_ms": (time.monotonic() - started) * 1000.0,
                "error": f"{type(exc).__name__}: {exc}"[:512],
                "inflight": self._inflight_generations,
            }
        return {
            "healthy": True,
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "error": None,
            "inflight": self._inflight_generations,
        }

    async def metrics_snapshot(self, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        """Collect a small SGLang metrics subset for timeout forensics."""
        client = await self._get_client()
        started = time.monotonic()
        try:
            response = await client.get("/metrics", timeout=timeout_seconds)
            await self._raise_for_status(response)
            body = (await response.aread()).decode("utf-8", errors="replace")
        except Exception as exc:
            return {
                "latency_ms": (time.monotonic() - started) * 1000.0,
                "error": f"{type(exc).__name__}: {exc}"[:512],
                "selected": [],
            }

        selected_keys = (
            "num_running_reqs",
            "num_queue_reqs",
            "num_used_tokens",
            "token_usage",
            "kv_cache",
            "cache_hit_rate",
            "available_size",
            "gen_throughput",
            "prompt_throughput",
            "time_to_first_token",
            "e2e_request_latency",
        )
        selected = []
        for line in body.splitlines():
            if line.startswith("#"):
                continue
            normalized = line.lower()
            if any(key in normalized for key in selected_keys):
                selected.append(line[:512])
            if len(selected) >= 48:
                break
        return {
            "latency_ms": (time.monotonic() - started) * 1000.0,
            "error": None,
            "selected": selected,
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
