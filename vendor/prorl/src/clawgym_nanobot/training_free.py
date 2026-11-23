from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DC_SELF_ROOT = PROJECT_ROOT / "external" / "training_free" / "DC-self"
_CASCADE_ROOT = PROJECT_ROOT / "external" / "training_free" / "CASCADE"
_SEQUENTIAL_ORDERS = {"", "sequential", "seq", "ordered", "no_shuffle", "no-shuffle"}


def _load_dc_self_types() -> tuple[Any, Any, Any]:
    root = str(_DC_SELF_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dc_self import DCSelfConfig, DCSelfEngine, Evidence

    return DCSelfConfig, DCSelfEngine, Evidence


def _load_cascade_types() -> tuple[Any, Any, Any, Any]:
    root = str(_CASCADE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from cascade import (
        CascadeConfig,
        CascadeEngine,
        HttpCrossEncoderReranker,
        PrecomputedEmbedder,
    )

    return (
        CascadeConfig,
        CascadeEngine,
        HttpCrossEncoderReranker,
        PrecomputedEmbedder,
    )


@dataclass(frozen=True)
class BatchPayload:
    extra_context: str
    metadata: dict[str, Any]
    task_extra_contexts: dict[int, str] | None = None
    task_metadata: dict[int, dict[str, Any]] | None = None


class TrainingFreeController(Protocol):
    method_name: str
    mode: str

    def task_batches(self, n_tasks: int) -> list[list[int]]: ...

    def begin_batch(
        self,
        *,
        dataset: str,
        task_indices: list[int],
        task_ids: list[str],
        task_queries: list[str] | None = None,
    ) -> BatchPayload: ...

    async def finish_batch(
        self,
        *,
        dataset: str,
        completed: Iterable[tuple[int, int, Any]],
        source_tasks: tuple[Any, ...],
        complete: Callable[[list[dict[str, str]], dict[str, Any]], Awaitable[str]],
    ) -> Any: ...

    def evaluation_metadata(self) -> dict[str, Any]: ...

    def batch_metrics(self) -> dict[str, int | float]: ...


class DCSelfController:
    method_name = "dc_self"

    def __init__(
        self,
        *,
        mode_override: str | None = None,
        task_index_offset: int = 0,
        order_env_name: str = "OPENCLAW_EVAL_ORDER",
        use_batch_size_env: bool = True,
    ) -> None:
        state_dir = os.environ.get("DC_SELF_STATE_DIR", "").strip()
        if not state_dir:
            raise ValueError("DC_SELF_STATE_DIR is required when DC-self is enabled")
        mode = (mode_override or os.environ.get("DC_SELF_MODE", "stream")).strip().lower()
        if mode not in {"stream", "frozen"}:
            raise ValueError("DC_SELF_MODE must be 'stream' or 'frozen'")
        eval_order = os.environ.get(order_env_name, "sequential").strip().lower()
        if eval_order not in _SEQUENTIAL_ORDERS:
            raise ValueError(
                f"DC-self requires {order_env_name}=sequential; "
                f"got {eval_order!r}"
            )

        DCSelfConfig, DCSelfEngine, Evidence = _load_dc_self_types()
        self._Evidence = Evidence
        self.mode = mode
        if task_index_offset < 0:
            raise ValueError("task_index_offset must be non-negative")
        self.task_index_offset = int(task_index_offset)
        self._batch_size_override = (
            _optional_positive_int_env("DC_SELF_BATCH_SIZE")
            if use_batch_size_env
            else None
        )
        self.max_evidence_chars = _positive_int_env("DC_SELF_MAX_EVIDENCE_CHARS", 12000)
        self._last_batch_metrics: dict[str, int | float] = {}
        self.engine = DCSelfEngine(
            DCSelfConfig(
                state_dir=Path(state_dir),
                max_cheatsheet_tokens=_positive_int_env(
                    "DC_SELF_MAX_CHEATSHEET_TOKENS", 4096
                ),
                max_entries=_positive_int_env("DC_SELF_MAX_ENTRIES", 128),
                curator_max_tokens=_positive_int_env("DC_SELF_CURATOR_MAX_TOKENS", 4096),
                curator_temperature=_float_env("DC_SELF_CURATOR_TEMPERATURE", 0.0),
                curator_max_retries=_nonnegative_int_env(
                    "DC_SELF_CURATOR_MAX_RETRIES", 1
                ),
                curator_failure_mode=_choice_env(
                    "DC_SELF_CURATOR_FAILURE_MODE", "noop", {"noop", "raise"}
                ),
                min_evidence=_positive_int_env("DC_SELF_MIN_EVIDENCE", 1),
            )
        )

    @property
    def updates_enabled(self) -> bool:
        return self.mode == "stream"

    def task_batches(self, n_tasks: int) -> list[list[int]]:
        if n_tasks < 0:
            raise ValueError("n_tasks must be non-negative")
        if not n_tasks:
            return []
        if self._batch_size_override is not None:
            batch_size = self._batch_size_override
        else:
            batch_size = 1 if self.updates_enabled else n_tasks
        return [
            list(range(start, min(start + batch_size, n_tasks)))
            for start in range(0, n_tasks, batch_size)
        ]

    def evaluation_metadata(self) -> dict[str, Any]:
        snapshot = self.engine.snapshot()
        return {
            **snapshot.metadata(),
            "mode": self.mode,
            "ordered": True,
        }

    def batch_metrics(self) -> dict[str, int | float]:
        return dict(self._last_batch_metrics)

    def begin_batch(
        self,
        *,
        dataset: str,
        task_indices: list[int],
        task_ids: list[str],
        task_queries: list[str] | None = None,
    ) -> BatchPayload:
        del task_queries
        if task_indices != sorted(task_indices):
            raise ValueError("DC-self task indices must be monotonic")
        if len(task_indices) != len(task_ids):
            raise ValueError("DC-self task index/id length mismatch")
        snapshot = self.engine.snapshot()
        manifest = self.engine.store.load_manifest()
        batch_index = len(manifest.get("batches") or [])
        global_task_indices = [self.task_index_offset + index for index in task_indices]
        metadata = {
            **snapshot.metadata(),
            "mode": self.mode,
            "dataset": dataset,
            "batch_index": batch_index,
            "batch_task_indices": global_task_indices,
            "batch_task_ids": list(task_ids),
            "ordered": True,
            "batched_approximation": len(task_indices) > 1 and self.updates_enabled,
        }
        return BatchPayload(
            extra_context=self.engine.solver_context(snapshot),
            metadata=metadata,
        )

    async def finish_batch(
        self,
        *,
        dataset: str,
        completed: Iterable[tuple[int, int, Any]],
        source_tasks: tuple[Any, ...],
        complete: Callable[[list[dict[str, str]], dict[str, Any]], Awaitable[str]],
    ) -> Any:
        if not self.updates_enabled:
            return self.engine.snapshot()
        completed_list = list(completed)
        evidence = self._build_evidence(completed_list, source_tasks)
        manifest = self.engine.store.load_manifest()
        batch_index = len(manifest.get("batches") or [])
        snapshot = await self.engine.update(
            evidence=evidence,
            complete=complete,
            dataset=dataset,
            batch_index=batch_index,
        )
        rewards = [_sample_reward(sample) for _flat, _task, sample in completed_list]
        summary = self.engine.last_update_summary
        self._last_batch_metrics = {
            "dc_self/batch": batch_index,
            "dc_self/task_count": len({int(task) for _flat, task, _sample in completed_list}),
            "dc_self/sample_count": len(completed_list),
            "dc_self/reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "dc_self/reward_min": min(rewards) if rewards else 0.0,
            "dc_self/reward_max": max(rewards) if rewards else 0.0,
            "dc_self/snapshot_version": snapshot.version,
            "dc_self/entry_count": len(snapshot.entries),
            **{
                f"dc_self/{key}": float(value) if isinstance(value, bool) else value
                for key, value in summary.items()
            },
        }
        return snapshot

    def _build_evidence(
        self,
        completed: Iterable[tuple[int, int, Any]],
        source_tasks: tuple[Any, ...],
    ) -> list[Any]:
        ordered = sorted(completed, key=lambda item: (int(item[1]), int(item[0])))
        records: list[Any] = []
        for flat_index, task_index, sample in ordered:
            task = source_tasks[int(task_index)]
            metadata = getattr(sample, "metadata", None)
            clawgym = metadata.get("clawgym_nanobot") if isinstance(metadata, dict) else None
            clawgym = clawgym if isinstance(clawgym, dict) else {}
            feedback = _bounded_feedback(clawgym, self.max_evidence_chars)
            records.append(
                self._Evidence(
                    task_id=str(task.task_id),
                    task_index=self.task_index_offset + int(task_index),
                    sample_index=int(flat_index),
                    prompt=_bounded_text(str(task.user_query), self.max_evidence_chars),
                    response=_bounded_text(
                        str(getattr(sample, "response", "") or ""),
                        self.max_evidence_chars,
                    ),
                    reward=_sample_reward(sample),
                    verifier_feedback=feedback,
                )
            )
        return records


class CascadeController:
    """Sequential host adapter for NP-CBR and pretrained reranker phases."""

    method_name = "cascade"

    def __init__(
        self,
        *,
        mode_override: str | None = None,
        task_index_offset: int = 0,
        order_env_name: str = "OPENCLAW_EVAL_ORDER",
        use_batch_size_env: bool = True,
    ) -> None:
        del use_batch_size_env
        state_dir = os.environ.get("CASCADE_STATE_DIR", "").strip()
        if not state_dir:
            raise ValueError("CASCADE_STATE_DIR is required when CASCADE is enabled")
        embedding_prefix = os.environ.get("CASCADE_EMBEDDING_PREFIX", "").strip()
        if not embedding_prefix:
            raise ValueError(
                "CASCADE_EMBEDDING_PREFIX is required; precompute embeddings "
                "before loading the actor"
            )
        mode = (mode_override or os.environ.get("CASCADE_MODE", "stream")).strip().lower()
        if mode not in {"stream", "frozen"}:
            raise ValueError("CASCADE_MODE must be 'stream' or 'frozen'")
        eval_order = os.environ.get(order_env_name, "sequential").strip().lower()
        if eval_order not in _SEQUENTIAL_ORDERS:
            raise ValueError(
                f"CASCADE requires {order_env_name}=sequential; got {eval_order!r}"
            )
        if task_index_offset < 0:
            raise ValueError("task_index_offset must be non-negative")

        (
            CascadeConfig,
            CascadeEngine,
            HttpCrossEncoderReranker,
            PrecomputedEmbedder,
        ) = _load_cascade_types()
        reranker_url = os.environ.get("CASCADE_RERANKER_URL", "").strip()
        default_variant = "pretrained_reranker" if reranker_url else "np_cbr"
        variant = os.environ.get("CASCADE_VARIANT", default_variant).strip().lower()
        if variant not in {"np_cbr", "pretrained_reranker"}:
            raise ValueError(
                "CASCADE_VARIANT must be 'np_cbr' or 'pretrained_reranker'"
            )
        if variant == "pretrained_reranker" and not reranker_url:
            raise ValueError(
                "CASCADE_RERANKER_URL is required for pretrained_reranker"
            )
        if variant == "np_cbr" and reranker_url:
            raise ValueError(
                "CASCADE_RERANKER_URL must be unset for the np_cbr variant"
            )
        reranker = None
        if reranker_url:
            reranker = HttpCrossEncoderReranker(
                reranker_url,
                timeout=_float_env("CASCADE_RERANKER_TIMEOUT", 120.0),
            )
        self.mode = mode
        self.variant = variant
        self.task_index_offset = int(task_index_offset)
        self._batch_index = 0
        self._pending_plans: dict[int, Any] | None = None
        self._last_batch_metrics: dict[str, int | float] = {}
        self.engine = CascadeEngine(
            CascadeConfig(
                state_dir=Path(state_dir),
                recall_size=_positive_int_env("CASCADE_RECALL_SIZE", 32),
                retrieve_topk=_positive_int_env("CASCADE_RETRIEVE_TOPK", 1),
                success_threshold=_float_env("CASCADE_SUCCESS_THRESHOLD", 0.5),
                max_query_chars=_positive_int_env("CASCADE_MAX_QUERY_CHARS", 16000),
                max_answer_chars=_positive_int_env("CASCADE_MAX_ANSWER_CHARS", 16000),
                max_case_prompt_chars=_positive_int_env(
                    "CASCADE_MAX_CASE_PROMPT_CHARS", 12000
                ),
                max_total_prompt_chars=_positive_int_env(
                    "CASCADE_MAX_TOTAL_PROMPT_CHARS", 24000
                ),
            ),
            PrecomputedEmbedder(embedding_prefix),
            reranker=reranker,
        )

    @property
    def updates_enabled(self) -> bool:
        return self.mode == "stream"

    def task_batches(self, n_tasks: int) -> list[list[int]]:
        if n_tasks < 0:
            raise ValueError("n_tasks must be non-negative")
        if not n_tasks:
            return []
        if not self.updates_enabled:
            return [list(range(n_tasks))]
        # Stream mode must retrieve and commit each observation before the
        # next task. Frozen evaluation can share one immutable snapshot.
        return [[index] for index in range(n_tasks)]

    def begin_batch(
        self,
        *,
        dataset: str,
        task_indices: list[int],
        task_ids: list[str],
        task_queries: list[str] | None = None,
    ) -> BatchPayload:
        if self._pending_plans is not None:
            raise RuntimeError("previous CASCADE batch has not been finished")
        if task_indices != sorted(set(task_indices)):
            raise ValueError("CASCADE task indices must be sorted and unique")
        if len(task_indices) != len(task_ids):
            raise ValueError("CASCADE task index/id length mismatch")
        if self.updates_enabled and len(task_indices) != 1:
            raise ValueError("canonical CASCADE stream requires singleton task batches")
        if task_queries is None or len(task_queries) != len(task_indices):
            raise ValueError("CASCADE requires one exact query per task")

        plans = {
            int(task_index): self.engine.retrieve(str(task_query))
            for task_index, task_query in zip(task_indices, task_queries, strict=True)
        }
        task_id_by_index = {
            int(task_index): str(task_id)
            for task_index, task_id in zip(task_indices, task_ids, strict=True)
        }
        self._pending_plans = plans
        snapshot = self.engine.snapshot()
        global_task_indices = [self.task_index_offset + int(index) for index in task_indices]
        metadata = {
            **snapshot.metadata(),
            "method": self.method_name,
            "variant": self.variant,
            "mode": self.mode,
            "dataset": dataset,
            "batch_index": self._batch_index,
            "batch_task_indices": global_task_indices,
            "batch_task_ids": list(task_ids),
            "ordered": True,
            "batched_approximation": False,
        }
        task_metadata = {
            task_index: {
                **plan.metadata(),
                "method": self.method_name,
                "variant": self.variant,
                "mode": self.mode,
                "dataset": dataset,
                "batch_index": self._batch_index,
                "batch_task_indices": [self.task_index_offset + task_index],
                "batch_task_ids": [task_id_by_index[task_index]],
                "entry_count": len(snapshot.cases),
                "case_count": len(snapshot.cases),
                "ordered": True,
                "batched_approximation": False,
            }
            for task_index, plan in plans.items()
        }
        return BatchPayload(
            extra_context=(
                self.engine.solver_context(next(iter(plans.values())))
                if len(plans) == 1
                else ""
            ),
            metadata=metadata,
            task_extra_contexts={
                task_index: self.engine.solver_context(plan)
                for task_index, plan in plans.items()
            },
            task_metadata=task_metadata,
        )

    async def finish_batch(
        self,
        *,
        dataset: str,
        completed: Iterable[tuple[int, int, Any]],
        source_tasks: tuple[Any, ...],
        complete: Callable[[list[dict[str, str]], dict[str, Any]], Awaitable[str]],
    ) -> Any:
        del complete
        if self._pending_plans is None:
            raise RuntimeError("CASCADE finish_batch called without a retrieval plan")
        completed_list = list(completed)
        if not self.updates_enabled:
            snapshot = self.engine.snapshot()
            self._last_batch_metrics = {
                f"{self.method_name}/batch": self._batch_index,
                f"{self.method_name}/snapshot_version": snapshot.version,
                f"{self.method_name}/case_count": len(snapshot.cases),
                **{
                    f"{self.method_name}/{key}": value
                    for key, value in self.engine.metrics().items()
                },
            }
            self._clear_pending()
            self._batch_index += 1
            return snapshot

        if len(completed_list) != 1:
            raise ValueError(
                "canonical CASCADE stream requires exactly one completion per task"
            )
        flat_index, task_index, sample = completed_list[0]
        task_index = int(task_index)
        if task_index not in self._pending_plans:
            raise RuntimeError("CASCADE completion does not match its retrieval plan")
        task = source_tasks[int(task_index)]
        metadata = getattr(sample, "metadata", None)
        clawgym = metadata.get("clawgym_nanobot") if isinstance(metadata, dict) else None
        clawgym = clawgym if isinstance(clawgym, dict) else {}
        reward = _sample_reward(sample)
        snapshot = self.engine.observe(
            self._pending_plans[task_index],
            answer=str(getattr(sample, "response", "") or ""),
            reward=reward,
            task_id=str(task.task_id),
            task_index=self.task_index_offset + int(task_index),
            metadata={
                "dataset": dataset,
                "sample_index": int(flat_index),
                "verifier_feedback": _bounded_feedback(
                    clawgym,
                    _positive_int_env("CASCADE_MAX_FEEDBACK_CHARS", 12000),
                ),
            },
        )
        self._last_batch_metrics = {
            f"{self.method_name}/batch": self._batch_index,
            **{
                f"{self.method_name}/{key}": value
                for key, value in self.engine.metrics().items()
            },
        }
        self._clear_pending()
        self._batch_index += 1
        return snapshot

    def evaluation_metadata(self) -> dict[str, Any]:
        return {
            **self.engine.snapshot().metadata(),
            "variant": self.variant,
            "mode": self.mode,
            "ordered": True,
        }

    def batch_metrics(self) -> dict[str, int | float]:
        return dict(self._last_batch_metrics)

    def _clear_pending(self) -> None:
        self._pending_plans = None


def controller_from_env(
    *,
    mode_override: str | None = None,
    task_index_offset: int = 0,
    order_env_name: str = "OPENCLAW_EVAL_ORDER",
    use_batch_size_env: bool = True,
) -> TrainingFreeController | None:
    method = os.environ.get("CLAWGYM_TRAINING_FREE_METHOD", "").strip().lower()
    if not method:
        return None
    normalized = method.replace("-", "_")
    if normalized == "dc_self":
        return DCSelfController(
            mode_override=mode_override,
            task_index_offset=task_index_offset,
            order_env_name=order_env_name,
            use_batch_size_env=use_batch_size_env,
        )
    if normalized == "cascade":
        return CascadeController(
            mode_override=mode_override,
            task_index_offset=task_index_offset,
            order_env_name=order_env_name,
            use_batch_size_env=use_batch_size_env,
        )
    raise ValueError(f"unsupported CLAWGYM_TRAINING_FREE_METHOD={method!r}")


def sample_context(sample: Any) -> tuple[str | None, dict[str, Any] | None]:
    context = getattr(sample, "training_free_context", None)
    metadata = getattr(sample, "training_free_metadata", None)
    context_value = str(context) if isinstance(context, str) and context.strip() else None
    metadata_value = dict(metadata) if isinstance(metadata, dict) else None
    return context_value, metadata_value


def attach_sample_context(sample: Any, payload: BatchPayload) -> None:
    task_index = int(getattr(sample, "group_index"))
    if payload.task_extra_contexts is None:
        sample.training_free_context = payload.extra_context
    else:
        try:
            sample.training_free_context = payload.task_extra_contexts[task_index]
        except KeyError as exc:
            raise KeyError(
                f"training-free payload has no context for task index {task_index}"
            ) from exc
    if payload.task_metadata is None:
        sample.training_free_metadata = dict(payload.metadata)
    else:
        try:
            sample.training_free_metadata = dict(payload.task_metadata[task_index])
        except KeyError as exc:
            raise KeyError(
                f"training-free payload has no metadata for task index {task_index}"
            ) from exc


def _bounded_feedback(clawgym: dict[str, Any], limit: int) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in ("code_verifier_score", "researchclaw_reward", "run"):
        if key in clawgym:
            selected[key] = clawgym[key]
    text = json.dumps(selected, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return selected
    return {"truncated_json": text[:limit], "truncated": True}


def _sample_reward(sample: Any) -> float:
    reward = getattr(sample, "reward", 0.0)
    if isinstance(reward, dict):
        if "score" in reward:
            reward = reward["score"]
        elif reward:
            reward = next(iter(reward.values()))
        else:
            reward = 0.0
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n...<truncated>...\n{text[-half:]}"


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int_env(name: str, default: int) -> int:
    value = _optional_positive_int_env(name)
    return default if value is None else value


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return value

