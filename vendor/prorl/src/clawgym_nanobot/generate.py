from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import functools
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import raw_source_integrity, runtime, tasks, training_free, user_agent


logger = logging.getLogger(__name__)

_SERVICE: "AdapterService | None" = None
_SERVICE_LOCK = threading.Lock()
_NANOBOT_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_NANOBOT_EXECUTOR_WORKERS = 0
_NANOBOT_EXECUTOR_LOCK = threading.Lock()
_MAX_TRAINING_TRAJECTORY_SAMPLES = 50


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _use_upstream_chat_completions() -> bool:
    return _env_truthy("CLAWGYM_NANOBOT_USE_UPSTREAM_CHAT_COMPLETIONS", True)


def _enable_realign() -> bool:
    # Preserve the pre-ResearchClaw behavior for every existing task profile.
    return _env_truthy("CLAWGYM_NANOBOT_ENABLE_REALIGN", True)


def _prune_unacknowledged_empty_retries() -> bool:
    return _env_truthy("CLAWGYM_NANOBOT_PRUNE_EMPTY_RETRIES", False)


@dataclass
class AdapterService:
    adapter: Any
    base_url: str
    handle: Any

    @classmethod
    def start(cls, args: Any) -> "AdapterService":
        from slime.agent.adapters.openai import OpenAIAdapter
        from slime.agent.aiohttp_threaded import run_app_in_thread
        from slime.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        adapter = OpenAIAdapter(
            tokenizer=tokenizer,
            sglang_url=_sglang_base_url(args),
            tool_parser=_parser_name(args, "sglang_tool_call_parser", "TOOL_CALL_PARSER"),
            reasoning_parser=_parser_name(args, "sglang_reasoning_parser", "REASONING_PARSER"),
            max_turns_per_sid=_max_turns_per_sid(),
            fork_threshold_tokens=_fork_threshold_tokens(),
            enable_realign=_enable_realign(),
            prune_unacknowledged_empty_retries=_prune_unacknowledged_empty_retries(),
            reasoning_context_policy=_reasoning_context_policy(),
            upstream_chat_completions=_use_upstream_chat_completions(),
            upstream_chat_model=os.environ.get("CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL"),
        )
        host = os.environ.get("CLAWGYM_NANOBOT_ADAPTER_HOST", "0.0.0.0")
        port = int(os.environ.get("CLAWGYM_NANOBOT_ADAPTER_PORT", "0"))
        public_host = os.environ.get("CLAWGYM_NANOBOT_ADAPTER_PUBLIC_HOST", "127.0.0.1")
        handle = run_app_in_thread(
            adapter.app,
            host=host,
            port=port,
            thread_name="clawgym-nanobot-openai-adapter",
            runner_kwargs={"handler_cancellation": True},
        )
        base_url = f"http://{public_host}:{handle.port}"
        logger.info("[ClawGym-Nanobot] OpenAI adapter listening on %s", base_url)
        logger.info("[ClawGym-Nanobot] adapter settings: %s", _adapter_settings_summary())
        return cls(adapter=adapter, base_url=base_url, handle=handle)


async def generate(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    _defer_user_agent: bool = False,
) -> Any:
    if (
        not evaluation
        and not _env_truthy("CLAWGYM_NANOBOT_DISTRIBUTED_ACTOR", False)
        and _env_truthy("CLAWGYM_DISTRIBUTED_ROLLOUT_NANOBOT", False)
    ):
        from . import distributed

        return await distributed.generate_train_sample(args, base_sample, dict(sampling_params or {}))

    selection: tasks.TaskSelection | None = None
    service: AdapterService | None = None
    session_id = str(getattr(base_sample, "session_id", None) or "pending")
    preserved_run_dir: str | None = None
    start = time.monotonic()
    try:
        selection = _task_selection(args, base_sample, evaluation=evaluation)
        service = _get_adapter_service(args)
        rollout_id = _ensure_rollout_id(base_sample)
        session_id = _session_id(
            args,
            base_sample,
            fallback_rollout_id=int(rollout_id),
            fallback_task_index=selection.task_index,
            evaluation=evaluation,
        )
        service.adapter.open_session(
            session_id,
            sampling_defaults=_sampling_defaults(sampling_params),
            max_context_tokens=_max_context_tokens(args, evaluation=evaluation),
            routing_key=selection.task.task_id,
        )

        logger.info(
            "[ClawGym-Nanobot] start task=%s prompt_index=%s sample_index=%s session=%s",
            selection.task.task_id,
            selection.prompt_index,
            getattr(base_sample, "index", None),
            session_id,
        )
        from .self_reward import configuration as self_reward_configuration, CaptureEvaluator
        self_reward_config = self_reward_configuration() if not evaluation else None
        score_user_agent = user_agent.should_score_user_agent(evaluation=evaluation) if self_reward_config is None else False
        extra_context, training_free_metadata = training_free.sample_context(base_sample)
        result = await _run_nanobot_task_async(
            args,
            selection.task,
            session_id=session_id,
            adapter_base_url=service.base_url,
            rollout_id=int(rollout_id),
            task_index=selection.task_index,
            group_index=getattr(base_sample, "group_index", None),
            sampling_params=dict(sampling_params or {}),
            extra_context=extra_context,
            evaluation=evaluation,
            preserve_run_dir=score_user_agent or self_reward_config is not None,
        )
        if result.status != "completed":
            logger.warning(
                "[ClawGym-Nanobot] task failed task=%s status=%s error=%s run_dir=%s",
                selection.task.task_id,
                result.status,
                result.error or "unknown",
                result.run_dir,
            )
            aborted = _abort_sample(
                base_sample,
                result.error or f"nanobot task status={result.status}",
                selection=selection,
            )
            _attach_result_metadata(aborted, result)
            _publish_self_reward_candidate(
                base_sample=base_sample,
                samples=[aborted],
                selection=selection,
                session_id=session_id,
                candidate_snapshot=None,
                valid=False,
                status=result.status,
                evaluation=evaluation,
            )
            return _sample_output(aborted, evaluation=evaluation)
        if score_user_agent or self_reward_config is not None:
            preserved_run_dir = result.run_dir
        metadata = _metadata_for_result(selection, result, start)
        if self_reward_config is not None:
            capture = CaptureEvaluator(task_dir=str(selection.task.task_dir),snapshot_root=self_reward_config['snapshot_root'])
            captured = await asyncio.to_thread(capture.capture,None,{'session_id':result.session_id,'session_dir':str(Path(result.workspace_dir).parent)})
            metadata['clawgym_nanobot']['candidate_snapshot'] = captured.metadata['candidate_snapshot']
            metadata['clawgym_nanobot']['self_reward_pending'] = True
        if training_free_metadata is not None:
            metadata["clawgym_nanobot"]["training_free"] = training_free_metadata
        request_metrics = _adapter_request_metrics(service.adapter, session_id)
        samples = await service.adapter.finish_session(
            session_id,
            base_sample=base_sample,
            reward=float(result.reward),
            extra_metadata=metadata,
            wait_timeout=float(os.environ.get("CLAWGYM_NANOBOT_FINISH_WAIT_TIMEOUT", "30")),
        )
        if not samples:
            aborted = _abort_sample(base_sample, "adapter produced no trainable turns", selection=selection)
            _publish_self_reward_candidate(
                base_sample=base_sample,
                samples=[aborted],
                selection=selection,
                session_id=session_id,
                candidate_snapshot=metadata["clawgym_nanobot"].get("candidate_snapshot"),
                valid=False,
                status="adapter_empty",
                evaluation=evaluation,
            )
            return _sample_output(aborted, evaluation=evaluation)

        trajectory_metrics = _sample_collection_metrics(samples)
        if _should_mask_oversized_training_trajectory(
            trajectory_metrics,
            evaluation=evaluation,
        ):
            observed_samples = int(trajectory_metrics["samples"])
            observed_trainable_samples = int(trajectory_metrics["trainable_samples"])
            reason = (
                "Training trajectory produced "
                f"{observed_samples} samples (limit "
                f"{_MAX_TRAINING_TRAJECTORY_SAMPLES}); masking the whole trajectory"
            )
            masked_sample = _abort_sample(base_sample, reason, selection=selection)
            masked_output_samples = [masked_sample]
            emitted_metrics = _sample_collection_metrics(masked_output_samples)
            split_reasons = _split_reason_summary(
                samples,
                masked_output_samples,
                trajectory=trajectory_metrics,
                emitted=emitted_metrics,
                requests=request_metrics,
                evaluation=evaluation,
            )
            clawgym_metadata = metadata.setdefault("clawgym_nanobot", {})
            if isinstance(clawgym_metadata, dict):
                clawgym_metadata["status"] = "aborted"
                clawgym_metadata["reason"] = reason
                clawgym_metadata["reward"] = 0.0
                clawgym_metadata["fork_sample_guard"] = {
                    "triggered": True,
                    "limit": _MAX_TRAINING_TRAJECTORY_SAMPLES,
                    "observed_samples": observed_samples,
                    "observed_trainable_samples": observed_trainable_samples,
                    "action": "mask_trajectory",
                }
            _attach_sample_collection_metrics(
                metadata,
                trajectory=trajectory_metrics,
                emitted=emitted_metrics,
                requests=request_metrics,
                split_reasons=split_reasons,
            )
            _finalize_sample(
                masked_sample,
                base_sample,
                task_prompt=selection.task.user_query,
                session_id=session_id,
                reward={"score": 0.0},
                metadata=metadata,
            )
            _publish_self_reward_candidate(
                base_sample=base_sample,
                samples=[masked_sample],
                selection=selection,
                session_id=session_id,
                candidate_snapshot=metadata["clawgym_nanobot"].get("candidate_snapshot"),
                valid=False,
                status="trajectory_sample_limit",
                evaluation=evaluation,
            )
            logger.error(
                "[ClawGym-Nanobot] %s task=%s session=%s trainable_samples=%d",
                reason,
                selection.task.task_id,
                session_id,
                observed_trainable_samples,
            )
            return _sample_output(masked_sample, evaluation=False)

        output_samples = _select_output_samples(samples, evaluation=evaluation)
        if not output_samples:
            aborted = _abort_sample(base_sample, "adapter produced no emitted samples", selection=selection)
            _publish_self_reward_candidate(
                base_sample=base_sample,
                samples=[aborted],
                selection=selection,
                session_id=session_id,
                candidate_snapshot=metadata["clawgym_nanobot"].get("candidate_snapshot"),
                valid=False,
                status="adapter_no_emitted_samples",
                evaluation=evaluation,
            )
            return _sample_output(aborted, evaluation=evaluation)
        emitted_metrics = _sample_collection_metrics(output_samples)
        split_reasons = _split_reason_summary(
            samples,
            output_samples,
            trajectory=trajectory_metrics,
            emitted=emitted_metrics,
            requests=request_metrics,
            evaluation=evaluation,
        )
        _attach_sample_collection_metrics(
            metadata,
            trajectory=trajectory_metrics,
            emitted=emitted_metrics,
            requests=request_metrics,
            split_reasons=split_reasons,
        )

        reward_payload = {"score": float(result.reward)}
        for sample in samples:
            _finalize_sample(
                sample,
                base_sample,
                task_prompt=selection.task.user_query,
                session_id=session_id,
                reward=reward_payload,
                metadata=metadata,
            )

        stream_valid = any(
            not getattr(sample, "remove_sample", False) and any(getattr(sample, "loss_mask", None) or [])
            for sample in output_samples
        )
        _publish_self_reward_candidate(
            base_sample=base_sample,
            samples=output_samples,
            selection=selection,
            session_id=session_id,
            candidate_snapshot=metadata["clawgym_nanobot"].get("candidate_snapshot"),
            valid=stream_valid,
            status="completed" if stream_valid else "completed_masked",
            evaluation=evaluation,
        )

        output: Any
        if evaluation:
            output = output_samples[-1]
        else:
            output = output_samples

        if score_user_agent:
            deferred_job = _build_deferred_user_agent_job(
                selection=selection,
                result=result,
                output=output,
                evaluation=evaluation,
            )
            if _defer_user_agent:
                runtime.cleanup_run_dir(result.run_dir)
                preserved_run_dir = None
                return {"output": output, "user_agent_job": deferred_job}
            try:
                ua_result = await asyncio.wrap_future(user_agent.get_scoring_pool().submit(deferred_job))
                apply_user_agent_result(output, ua_result)
            finally:
                runtime.cleanup_run_dir(result.run_dir)
                preserved_run_dir = None

        logger.info(
            "[ClawGym-Nanobot] done task=%s reward=%.4f segments=%d emitted=%d mode=%s "
            "trajectory_response_tokens=%d trajectory_train_tokens=%d emitted_response_tokens=%d "
            "emitted_train_tokens=%d emitted_trainable_samples=%d request_count=%d "
            "request_response_tokens=%d request_response_len_mean=%.1f eval=%s session=%s duration=%.1fs",
            selection.task.task_id,
            float(result.reward),
            len(samples),
            len(output_samples),
            _sample_output_mode(evaluation=evaluation),
            int(trajectory_metrics["response_tokens"]),
            int(trajectory_metrics["trainable_tokens"]),
            int(emitted_metrics["response_tokens"]),
            int(emitted_metrics["trainable_tokens"]),
            int(emitted_metrics["trainable_samples"]),
            int(request_metrics["count"]),
            int(request_metrics["response_tokens"]),
            float(request_metrics["response_len_mean"]),
            int(evaluation),
            session_id,
            time.monotonic() - start,
        )
        return output
    except raw_source_integrity.RawSourceIntegrityError:
        # This is a durable input-integrity failure with its own fatal-resume
        # marker. Let it abort the rollout instead of producing an ordinary
        # zero-trainable placeholder that Slime could retry indefinitely.
        logger.critical(
            "[ClawGym-Nanobot] propagating fatal raw-source integrity failure "
            "for session=%s",
            session_id,
            exc_info=True,
        )
        raise
    except Exception as exc:
        logger.exception("[ClawGym-Nanobot] generate failed for session=%s", session_id)
        aborted = _abort_sample(base_sample, f"{type(exc).__name__}: {exc}", selection=selection)
        if selection is not None:
            try:
                _publish_self_reward_candidate(
                    base_sample=base_sample,
                    samples=[aborted],
                    selection=selection,
                    session_id=session_id,
                    candidate_snapshot=None,
                    valid=False,
                    status=f"{type(exc).__name__}",
                    evaluation=evaluation,
                )
            except Exception:
                logger.exception("[ClawGym-Nanobot] failed to publish aborted self-reward candidate")
        return _sample_output(aborted, evaluation=evaluation)
    finally:
        if preserved_run_dir is not None:
            runtime.cleanup_run_dir(preserved_run_dir)
        if service is not None:
            try:
                await service.adapter.drop_session(
                    session_id,
                    wait_timeout=float(os.environ.get("CLAWGYM_NANOBOT_DROP_WAIT_TIMEOUT", "5")),
                )
            except Exception:
                logger.debug("[ClawGym-Nanobot] drop_session failed for %s", session_id, exc_info=True)


def _get_adapter_service(args: Any) -> AdapterService:
    global _SERVICE
    if _SERVICE is not None:
        return _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = AdapterService.start(args)
        return _SERVICE


def reset_adapter_state() -> dict[str, int]:
    """Clear per-session adapter state while keeping the HTTP service alive."""
    if _SERVICE is None:
        return {
            "sessions": 0,
            "closed": 0,
            "inflight": 0,
            "trajectories": 0,
            "turn_counts": 0,
        }
    adapter = _SERVICE.adapter
    store = getattr(adapter, "store", {})
    closed = getattr(adapter, "closed", set())
    inflight = getattr(adapter, "inflight", {})
    manager = getattr(adapter, "manager", None)
    trees = getattr(manager, "_trees", {})
    turn_counts = getattr(manager, "_turn_count", {})
    sid_turn_count = getattr(adapter, "_sid_turn_count", {})
    state = {
        "sessions": len(store) if hasattr(store, "__len__") else 0,
        "closed": len(closed) if hasattr(closed, "__len__") else 0,
        "inflight": sum(len(tasks) for tasks in inflight.values()) if isinstance(inflight, dict) else 0,
        "trajectories": len(trees) if hasattr(trees, "__len__") else 0,
        "turn_counts": len(sid_turn_count) if hasattr(sid_turn_count, "__len__") else 0,
    }
    if isinstance(store, dict):
        store.clear()
    if hasattr(closed, "clear"):
        closed.clear()
    if isinstance(inflight, dict):
        inflight.clear()
    if hasattr(trees, "clear"):
        trees.clear()
    if hasattr(turn_counts, "clear"):
        turn_counts.clear()
    if hasattr(sid_turn_count, "clear"):
        sid_turn_count.clear()
    return state


def _sample_output(sample: Any, *, evaluation: bool) -> Any:
    return sample if evaluation else [sample]


def _publish_self_reward_candidate(
    *,
    base_sample: Any,
    samples: list[Any],
    selection: tasks.TaskSelection,
    session_id: str,
    candidate_snapshot: str | None,
    valid: bool,
    status: str,
    evaluation: bool = False,
) -> dict[str, Any] | None:
    """Publish one terminal rollout event as soon as that trajectory finishes."""
    if evaluation:
        return None

    from .self_reward import configuration, publish_candidate

    if configuration() is None:
        return None
    stream = publish_candidate(
        base_sample=base_sample,
        task_id=selection.task.task_id,
        task_dir=str(selection.task.task_dir),
        session_id=session_id,
        candidate_snapshot=candidate_snapshot,
        valid=valid,
        status=status,
    )
    if stream is not None:
        for sample in samples:
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                sample.metadata = metadata
            metadata.setdefault("clawgym_nanobot", {})["self_reward_stream"] = dict(stream)
    return stream


async def auxiliary_chat_completion(
    args: Any,
    messages: list[dict[str, str]],
    sampling: dict[str, Any] | None = None,
) -> str:
    """Run one non-trainable curator completion through the current service."""
    import httpx

    service = _get_adapter_service(args)
    session_id = f"training-free-curator-{secrets.token_hex(8)}"
    params = dict(sampling or {})
    max_tokens = int(params.get("max_tokens") or params.get("max_new_tokens") or 2048)
    defaults = {
        "temperature": float(params.get("temperature", 0.0)),
        "top_p": float(params.get("top_p", 1.0)),
        "max_new_tokens": max_tokens,
    }
    service.adapter.open_session(
        session_id,
        sampling_defaults=_sampling_defaults(defaults),
        max_context_tokens=_max_context_tokens(args, evaluation=True),
    )
    body = {
        "model": f"side__{session_id}",
        "messages": messages,
        "temperature": defaults["temperature"],
        "top_p": defaults["top_p"],
        "max_completion_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            response = await client.post(
                f"{service.base_url}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {session_id}"},
            )
            response.raise_for_status()
            payload = response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("training-free curator response has no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("training-free curator response has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            content = message.get("reasoning_content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("training-free curator returned empty content")
        return content
    finally:
        await service.adapter.drop_session(
            session_id,
            wait_timeout=float(os.environ.get("CLAWGYM_NANOBOT_DROP_WAIT_TIMEOUT", "5")),
        )


def _task_selection(args: Any, sample: Any, *, evaluation: bool) -> tasks.TaskSelection:
    explicit = getattr(sample, "clawgym_task_override", None)
    if evaluation and isinstance(explicit, tasks.TaskSpec):
        task_index = int(getattr(sample, "group_index", 0) or 0)
        return tasks.TaskSelection(
            task=explicit,
            task_index=task_index,
            prompt_index=task_index,
            epoch=0,
            order="sequential",
        )
    return tasks.select_task_for_sample(args, sample, evaluation=evaluation)


async def _run_nanobot_task_async(
    args: Any,
    task: tasks.TaskSpec,
    **kwargs: Any,
) -> runtime.NanobotRunResult:
    loop = asyncio.get_running_loop()
    call = functools.partial(runtime.run_nanobot_task, args, task, **kwargs)
    return await loop.run_in_executor(_get_nanobot_executor(), call)


def _get_nanobot_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _NANOBOT_EXECUTOR, _NANOBOT_EXECUTOR_WORKERS
    max_workers = _nanobot_max_workers()
    if _NANOBOT_EXECUTOR is not None and _NANOBOT_EXECUTOR_WORKERS == max_workers:
        return _NANOBOT_EXECUTOR
    with _NANOBOT_EXECUTOR_LOCK:
        if _NANOBOT_EXECUTOR is None or _NANOBOT_EXECUTOR_WORKERS != max_workers:
            if _NANOBOT_EXECUTOR is not None:
                _NANOBOT_EXECUTOR.shutdown(wait=False, cancel_futures=False)
            _NANOBOT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="clawgym-nanobot",
            )
            _NANOBOT_EXECUTOR_WORKERS = max_workers
            logger.info("[ClawGym-Nanobot] nanobot executor max_workers=%d", max_workers)
        return _NANOBOT_EXECUTOR


def _nanobot_max_workers() -> int:
    for name in (
        "CLAWGYM_NANOBOT_MAX_WORKERS",
        "NANOBOT_ROLLOUT_MAX_WORKERS_TOTAL",
        "NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE",
    ):
        raw = os.environ.get(name, "").strip()
        if raw:
            value = int(raw)
            if value > 0:
                return value
    return 64


def _sglang_base_url(args: Any) -> str:
    explicit = os.environ.get("CLAWGYM_NANOBOT_SGLANG_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    ip = getattr(args, "sglang_router_ip", "127.0.0.1")
    port = getattr(args, "sglang_router_port", 30000)
    return f"http://{ip}:{port}"


def _parser_name(args: Any, attr: str, env_name: str) -> str | None:
    value = getattr(args, attr, None)
    if value:
        return str(value)
    value = os.environ.get(env_name, "").strip()
    return value or None


def _session_id(
    args: Any,
    sample: Any,
    *,
    fallback_rollout_id: int,
    fallback_task_index: int,
    evaluation: bool = False,
) -> str:
    style = os.environ.get("CLAWGYM_NANOBOT_SESSION_ID_STYLE", "clawrl").strip().lower()
    existing = getattr(sample, "session_id", None)
    if existing and style in {"existing", "sample"}:
        return str(existing)

    if style in {"clawrl", "task", "stable", ""}:
        explicit_rollout_id = _int_or_none(getattr(sample, "rollout_id", None))
        rollout_id_was_inferred = bool(getattr(sample, "_clawgym_rollout_id_inferred", False))
        if evaluation and explicit_rollout_id is not None and not rollout_id_was_inferred:
            rollout_id = explicit_rollout_id
            flat_index = _int_or_none(getattr(sample, "index", None))
            if flat_index is None:
                flat_index = fallback_task_index
        else:
            rollout_id, flat_index = _clawrl_session_indices(
                args,
                sample,
                fallback_rollout_id=fallback_rollout_id,
                fallback_task_index=fallback_task_index,
            )
        prefix = _session_prefix(evaluation=evaluation)
        if evaluation and _eval_unique_session_ids():
            unique = _safe_session_component(getattr(sample, "session_id", None)) or secrets.token_hex(4)
            return f"{prefix}-{rollout_id}-{flat_index}-{unique}"
        return f"{prefix}-{rollout_id}-{flat_index}"

    group_index = getattr(sample, "group_index", "nogroup")
    index = getattr(sample, "index", "noindex")
    return f"nanobot-{group_index}-{index}-{secrets.token_hex(4)}"


def _session_prefix(*, evaluation: bool) -> str:
    if evaluation:
        raw = (
            os.environ.get("CLAWGYM_NANOBOT_EVAL_SESSION_PREFIX", "").strip()
            or os.environ.get("CLAWGYM_NANOBOT_SESSION_EVAL_PREFIX", "").strip()
            or "task"
        )
        return raw or "task"
    return os.environ.get("CLAWGYM_NANOBOT_SESSION_PREFIX", "task").strip() or "task"


def _eval_unique_session_ids() -> bool:
    raw = os.environ.get("CLAWGYM_NANOBOT_EVAL_UNIQUE_SESSION_IDS", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _adapter_settings_summary() -> dict[str, str]:
    keys = (
        "CLAWGYM_TASK_PROMPT_MODE",
        "OPENCLAW_EVAL_TASKS",
        "OPENCLAW_EVAL_MAX_TURNS",
        "OPENCLAW_EVAL_N_SAMPLES",
        "OPENCLAW_EVAL_NUM_WORKERS",
        "OPENCLAW_CONTEXT_WINDOW",
        "OPENCLAW_MAX_TOKENS",
        "CLAWGYM_NANOBOT_EVAL_TURN_TYPE",
        "CLAWGYM_NANOBOT_TRAIN_TURN_TYPE",
        "NANOBOT_EVAL_NUM_WORKERS",
        "NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE",
        "NANOBOT_TEMPERATURE",
        "NANOBOT_TOP_P",
        "NANOBOT_TOP_K",
        "ROLLOUT_TEMPERATURE",
        "ROLLOUT_TOP_P",
        "ROLLOUT_TOP_K",
        "NANOBOT_ENABLE_THINKING",
        "NANOBOT_SHOW_REASONING",
        "NANOBOT_STRIP_REASONING_CONTEXT",
        "NANOBOT_RL_NO_MAX_TOKENS",
        "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY",
        "CLAWGYM_NANOBOT_FORK_THRESHOLD_TOKENS",
        "NANOBOT_RL_DISABLE_CONSOLIDATION",
        "NANOBOT_RL_FAST_ARCHIVE",
        "NANOBOT_WEB_ENABLE",
        "NANOBOT_EXEC_TIMEOUT",
        "NANOBOT_NO_WEB_CLI_TIMEOUT",
        "NANOBOT_MAX_TOOL_ITERATIONS",
        "TOOL_CALL_PARSER",
        "REASONING_PARSER",
        "CLAWGYM_NANOBOT_TRAIN_SAMPLE_MODE",
        "CLAWGYM_NANOBOT_EVAL_SAMPLE_MODE",
        "CLAWGYM_NANOBOT_EVAL_SESSION_PREFIX",
        "CLAWGYM_NANOBOT_EVAL_UNIQUE_SESSION_IDS",
        "CLAWGYM_NANOBOT_USE_UPSTREAM_CHAT_COMPLETIONS",
        "CLAWGYM_NANOBOT_RESEARCHCLAW_COMPAT",
        "CLAWGYM_NANOBOT_ENABLE_REALIGN",
        "CLAWGYM_NANOBOT_PRUNE_EMPTY_RETRIES",
        "CLAWGYM_NANOBOT_REQUIRE_EXACT_TOKENS",
        "CLAWGYM_ALLOW_MULTI_TOOL_CALLS",
        "NANOBOT_MICROCOMPACT_ENABLED",
        "CLAWGYM_NANOBOT_UPSTREAM_CHAT_MODEL",
        "CLAWGYM_SGLANG_WORKER_URLS",
        "SERVED_MODEL_NAME",
        "NANOBOT_MODEL_ID",
        "CLAWGYM_ALLOW_NANOBOT_MODEL_ID_OVERRIDE",
        "CLAWGYM_EXEC_DROP_PRIVILEGES",
        "CLAWGYM_EXEC_UID",
        "CLAWGYM_EXEC_GID",
        "CLAWGYM_RAW_SOURCE_INTEGRITY",
        "CLAWGYM_RAW_SOURCE_MANIFEST",
    )
    settings = {key: os.environ.get(key, "") for key in keys}
    settings["CLAWGYM_NANOBOT_RESOLVED_REASONING_CONTEXT_POLICY"] = _reasoning_context_policy() or "legacy"
    return settings


def _safe_session_component(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in "._:-" else "_" for ch in text)
    return safe[:64]


def _clawrl_session_indices(
    args: Any,
    sample: Any,
    *,
    fallback_rollout_id: int,
    fallback_task_index: int,
) -> tuple[int, int]:
    sample_index = _int_or_none(getattr(sample, "index", None))
    if sample_index is None:
        return int(fallback_rollout_id), int(fallback_task_index)

    rollout_batch_size = _positive_int(getattr(args, "rollout_batch_size", None), 1)
    n_samples_per_prompt = _positive_int(getattr(args, "n_samples_per_prompt", None), 1)
    samples_per_outer_rollout = rollout_batch_size * n_samples_per_prompt
    outer_rollout_id, flat_index = divmod(sample_index, samples_per_outer_rollout)
    return int(outer_rollout_id), int(flat_index)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, default: int) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None and parsed > 0 else int(default)


def _ensure_rollout_id(sample: Any) -> int:
    rollout_id = getattr(sample, "rollout_id", None)
    inferred = False
    if rollout_id is None:
        rollout_id = getattr(sample, "index", None)
        inferred = True
    if rollout_id is None:
        rollout_id = 0
        inferred = True
    sample.rollout_id = int(rollout_id)
    try:
        sample._clawgym_rollout_id_inferred = inferred
    except Exception:
        pass
    return int(rollout_id)


def _sampling_defaults(sampling_params: dict[str, Any] | None) -> dict[str, Any]:
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
        "sampling_seed",
        "custom_params",
    }
    source = sampling_params or {}
    defaults = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key in allowed and value is not None
    }
    defaults.setdefault("skip_special_tokens", False)
    defaults.setdefault("spaces_between_special_tokens", False)
    defaults.setdefault("no_stop_trim", True)
    return defaults


def _select_output_samples(samples: list[Any], *, evaluation: bool) -> list[Any]:
    mode = _sample_output_mode(evaluation=evaluation)
    if mode in {"segments", "turns", "all", "token_prefix", "prefix", "clawrl"}:
        return list(samples)
    if mode in {"final", "last", "trajectory", "clawrl"}:
        return [_last_trainable_sample(samples)]
    logger.warning("[ClawGym-Nanobot] unknown sample output mode %r; using final sample", mode)
    return [_last_trainable_sample(samples)]


def _last_trainable_sample(samples: list[Any]) -> Any:
    for sample in reversed(samples):
        if _sample_trainable_tokens(sample) > 0:
            return sample
    return samples[-1]


def _sample_response_tokens(sample: Any) -> int:
    try:
        return int(getattr(sample, "response_length", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _sample_trainable_tokens(sample: Any) -> int:
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is not None:
        try:
            return int(sum(int(value) for value in loss_mask))
        except TypeError:
            return 0
    return _sample_response_tokens(sample)


def _sample_collection_metrics(samples: list[Any]) -> dict[str, int | float]:
    response_lengths = [_sample_response_tokens(sample) for sample in samples]
    trainable_lengths = [_sample_trainable_tokens(sample) for sample in samples]

    metrics: dict[str, int | float] = {
        "samples": len(samples),
        "trainable_samples": sum(1 for length in trainable_lengths if length > 0),
        "response_tokens": sum(response_lengths),
        "trainable_tokens": sum(trainable_lengths),
    }
    for prefix, values in (
        ("response_len", response_lengths),
        ("trainable_response_len", trainable_lengths),
    ):
        for key, value in _length_stats(values).items():
            metrics[f"{prefix}_{key}"] = value
    return metrics


def _should_mask_oversized_training_trajectory(
    trajectory: dict[str, int | float],
    *,
    evaluation: bool,
) -> bool:
    if evaluation:
        return False
    return int(trajectory.get("samples", 0) or 0) > _MAX_TRAINING_TRAJECTORY_SAMPLES


def _adapter_request_metrics(adapter: Any, session_id: str) -> dict[str, Any]:
    response_lengths: list[int] = []
    prompt_lengths: list[int] = []
    root = _adapter_session_root(adapter, session_id)
    if root is not None:
        seen_turns: set[int] = set()
        for node in _walk_message_tree(root):
            turn = getattr(node, "turn", None)
            if turn is None or id(turn) in seen_turns:
                continue
            seen_turns.add(id(turn))
            prompt_ids = getattr(turn, "prompt_ids", None) or []
            output_ids = getattr(turn, "output_ids", None) or []
            prompt_lengths.append(len(prompt_ids))
            response_lengths.append(len(output_ids))
    total_lengths = [prompt + response for prompt, response in zip(prompt_lengths, response_lengths, strict=False)]
    metrics: dict[str, Any] = {
        "count": len(response_lengths),
        "prompt_tokens": sum(prompt_lengths),
        "response_tokens": sum(response_lengths),
        "total_tokens": sum(total_lengths),
        "prompt_len_values": prompt_lengths,
        "response_len_values": response_lengths,
        "total_len_values": total_lengths,
    }
    for prefix, values in (
        ("prompt_len", prompt_lengths),
        ("response_len", response_lengths),
        ("total_len", total_lengths),
    ):
        for key, value in _length_stats(values).items():
            metrics[f"{prefix}_{key}"] = value
    lineage_getter = getattr(adapter, "session_lineage_metrics", None)
    if callable(lineage_getter):
        lineage = lineage_getter(session_id)
        if isinstance(lineage, dict):
            for field in ("prompt_build_modes", "prompt_build_reasons"):
                raw_counts = lineage.get(field)
                if isinstance(raw_counts, dict):
                    metrics[field] = {
                        str(key): int(value)
                        for key, value in raw_counts.items()
                        if isinstance(value, (int, float))
                    }
            try:
                metrics["raw_prefix_violations"] = int(lineage.get("raw_prefix_violations", 0) or 0)
            except (TypeError, ValueError):
                metrics["raw_prefix_violations"] = 0
    return metrics


def _adapter_session_root(adapter: Any, session_id: str) -> Any | None:
    manager = getattr(adapter, "manager", None)
    trees = getattr(manager, "_trees", None)
    if not isinstance(trees, dict):
        return None
    return trees.get(session_id)


def _walk_message_tree(root: Any):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        children = getattr(node, "children", None) or []
        stack.extend(reversed(list(children)))


def _length_stats(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0, "min": 0}
    ordered = sorted(int(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        median: int | float = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "mean": sum(ordered) / count,
        "median": median,
        "max": ordered[-1],
        "min": ordered[0],
    }


def _attach_sample_collection_metrics(
    metadata: dict[str, Any],
    *,
    trajectory: dict[str, int | float],
    emitted: dict[str, int | float],
    requests: dict[str, Any],
    split_reasons: dict[str, Any],
) -> None:
    clawgym_metadata = metadata.setdefault("clawgym_nanobot", {})
    if not isinstance(clawgym_metadata, dict):
        return
    clawgym_metadata["trajectory"] = dict(trajectory)
    clawgym_metadata["emitted"] = dict(emitted)
    clawgym_metadata["requests"] = dict(requests)
    clawgym_metadata["split_reasons"] = dict(split_reasons)


def _split_reason_summary(
    samples: list[Any],
    output_samples: list[Any],
    *,
    trajectory: dict[str, int | float],
    emitted: dict[str, int | float],
    requests: dict[str, Any],
    evaluation: bool,
) -> dict[str, Any]:
    trajectory_samples = int(trajectory.get("samples", len(samples)) or 0)
    emitted_samples = int(emitted.get("samples", len(output_samples)) or 0)
    trainable_samples = int(trajectory.get("trainable_samples", 0) or 0)
    request_count = int(requests.get("count", 0) or 0)
    mode = _sample_output_mode(evaluation=evaluation)
    reasons: dict[str, int] = {}

    def bump(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    multi_sample = trajectory_samples > 1 or emitted_samples > 1
    if multi_sample:
        bump("multi_sample_trajectory")
        if trainable_samples > 1:
            bump("multiple_trainable_segments")
        if emitted_samples > 1:
            bump(f"sample_mode_{_metric_key(mode)}")
        if request_count > trajectory_samples:
            bump("message_tree_forks")
        elif request_count > 1:
            bump("multi_request_history")

        trajectory_response_tokens = int(trajectory.get("response_tokens", 0) or 0)
        request_response_tokens = int(requests.get("response_tokens", 0) or 0)
        if request_response_tokens and request_response_tokens != trajectory_response_tokens:
            bump("token_accounting_drift")

        policy = _reasoning_context_policy()
        if policy == "strip":
            bump("history_rewrite_reasoning_stripped")
        elif policy:
            bump(f"reasoning_context_{_metric_key(policy)}")
        if _fork_threshold_tokens() is not None:
            bump("message_tree_fork_threshold")
        if _max_turns_per_sid() is not None:
            bump("adapter_turn_limit")
        if _env_truthy("NANOBOT_RL_FAST_ARCHIVE", False):
            bump("compaction_fast_archive")
        elif _env_truthy("NANOBOT_RL_DISABLE_CONSOLIDATION", False):
            bump("compaction_disabled")
        if not reasons:
            bump("unknown")

    return {
        "multi_sample": int(multi_sample),
        "trajectory_samples": trajectory_samples,
        "emitted_samples": emitted_samples,
        "trainable_samples": trainable_samples,
        "request_count": request_count,
        "sample_mode": mode,
        "reasons": reasons,
    }


def _metric_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    key = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in key.split("_") if part) or "unknown"


def _sample_output_mode(*, evaluation: bool) -> str:
    specific = "CLAWGYM_NANOBOT_EVAL_SAMPLE_MODE" if evaluation else "CLAWGYM_NANOBOT_TRAIN_SAMPLE_MODE"
    for name in (specific, "CLAWGYM_NANOBOT_SAMPLE_MODE"):
        raw = os.environ.get(name, "").strip().lower()
        if raw:
            return raw
    return "final" if evaluation else "token_prefix"


def _max_context_tokens(args: Any, *, evaluation: bool) -> int:
    if evaluation:
        value = getattr(args, "eval_max_context_len", None)
        if value:
            return int(value)
    for attr in ("rollout_max_context_len", "sglang_context_length"):
        value = getattr(args, attr, None)
        if value:
            return int(value)
    raw = os.environ.get("OPENCLAW_CONTEXT_WINDOW", "").strip()
    return int(raw) if raw else 65536


def _metadata_for_result(
    selection: tasks.TaskSelection,
    result: runtime.NanobotRunResult,
    start_time: float,
) -> dict[str, Any]:
    run_metadata = result.metadata()
    clawgym_metadata = {
        "task_id": selection.task.task_id,
        "task_index": selection.task_index,
        "prompt_index": selection.prompt_index,
        "task_epoch": selection.epoch,
        "train_order": selection.order,
        "session_id": result.session_id,
        "status": result.status,
        "reward": float(result.reward),
        "code_verifier_duration_s": float(result.code_verifier_duration_s),
        "duration_s": time.monotonic() - start_time,
        "run": run_metadata,
    }
    researchclaw_summary = run_metadata.get("researchclaw_reward")
    if isinstance(researchclaw_summary, dict):
        clawgym_metadata["researchclaw_reward"] = researchclaw_summary
        codebase_reward = researchclaw_summary.get("codebase_raw_reward")
        if isinstance(codebase_reward, (int, float)) and not isinstance(codebase_reward, bool):
            clawgym_metadata["code_verifier_score"] = float(codebase_reward)
    else:
        clawgym_metadata["code_verifier_score"] = float(result.reward)
    agent_activity = run_metadata.get("agent_activity")
    if isinstance(agent_activity, dict):
        clawgym_metadata["agent"] = dict(agent_activity)
    return {"clawgym_nanobot": clawgym_metadata}


def _attach_result_metadata(sample: Any, result: runtime.NanobotRunResult) -> None:
    metadata = getattr(sample, "metadata", None)
    merged = dict(metadata) if isinstance(metadata, dict) else {}
    clawgym_metadata = dict(merged.get("clawgym_nanobot") or {})
    run_metadata = result.metadata()
    clawgym_metadata.update(
        {
            "session_id": result.session_id,
            "reward": float(result.reward),
            "run": run_metadata,
        }
    )
    if isinstance(run_metadata.get("researchclaw_reward"), dict):
        clawgym_metadata["researchclaw_reward"] = run_metadata["researchclaw_reward"]
    if isinstance(run_metadata.get("agent_activity"), dict):
        clawgym_metadata["agent"] = dict(run_metadata["agent_activity"])
    merged["clawgym_nanobot"] = clawgym_metadata
    sample.metadata = merged


def _build_deferred_user_agent_job(
    *,
    selection: tasks.TaskSelection,
    result: runtime.NanobotRunResult,
    output: Any,
    evaluation: bool,
) -> user_agent.DeferredUserAgentJob:
    archive, archive_error = user_agent.archive_workspace(Path(result.workspace_dir))
    return user_agent.DeferredUserAgentJob(
        task_id=selection.task.task_id,
        task_dir=str(selection.task.task_dir),
        task_prompt=selection.task.user_query,
        final_message=_final_message(output, result),
        code_verifier_score=float(result.reward),
        session_id=result.session_id,
        evaluated_log_dir=str(Path(result.log_dir).parent),
        run_dir=result.run_dir,
        evaluation=evaluation,
        queued_at=time.time(),
        workspace_archive=archive,
        workspace_archive_error=archive_error,
    )


def _final_message(output: Any, result: runtime.NanobotRunResult) -> str:
    samples = output if isinstance(output, list) else [output]
    for sample in reversed(samples):
        response = getattr(sample, "response", None)
        if isinstance(response, str) and response.strip():
            return response.strip()
    if result.turns:
        return result.turns[-1].stdout_tail.strip()
    return ""


def apply_user_agent_result(output: Any, result: user_agent.UserAgentRewardResult) -> None:
    samples = output if isinstance(output, list) else [output]
    ua_metadata = user_agent.result_metadata(result)
    replace_primary_reward = user_agent.user_agent_reward_enabled()
    for sample in samples:
        reward = getattr(sample, "reward", None)
        reward_payload = dict(reward) if isinstance(reward, dict) else {}
        if replace_primary_reward:
            reward_payload["score"] = float(result.score)
        reward_payload["user_agent"] = float(result.score)
        sample.reward = reward_payload
        metadata = getattr(sample, "metadata", None)
        merged = dict(metadata) if isinstance(metadata, dict) else {}
        clawgym = dict(merged.get("clawgym_nanobot") or {})
        if replace_primary_reward:
            clawgym["reward"] = float(result.score)
        clawgym["user_agent"] = ua_metadata
        merged["clawgym_nanobot"] = clawgym
        sample.metadata = merged


def _finalize_sample(
    sample: Any,
    base_sample: Any,
    *,
    task_prompt: str,
    session_id: str,
    reward: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    sample.rollout_id = getattr(base_sample, "rollout_id", None)
    # Synthetic base samples may carry an empty prompt.  Persist the actual
    # task instruction on every emitted segment so a saved trajectory remains
    # independently auditable.
    sample.prompt = task_prompt
    sample.session_id = session_id
    sample.reward = dict(reward)
    sample.metadata = _merge_metadata(getattr(sample, "metadata", None), metadata)
    if getattr(sample, "loss_mask", None) is None:
        sample.loss_mask = [1] * int(getattr(sample, "response_length", 0) or 0)
    if getattr(sample, "rollout_log_probs", None) is None:
        sample.rollout_log_probs = [0.0] * int(getattr(sample, "response_length", 0) or 0)
    if len(sample.loss_mask) != int(sample.response_length or 0):
        raise ValueError("loss_mask length must match response_length")
    if len(sample.rollout_log_probs) != int(sample.response_length or 0):
        raise ValueError("rollout_log_probs length must match response_length")


def _abort_sample(base_sample: Any, reason: str, *, selection: tasks.TaskSelection | None = None) -> Any:
    try:
        from slime.utils.types import Sample
    except Exception:
        Sample = base_sample.__class__

    sample = Sample(
        index=getattr(base_sample, "index", None),
        group_index=getattr(base_sample, "group_index", None),
        prompt=getattr(base_sample, "prompt", ""),
        tokens=[0, 0],
        response="",
        response_length=1,
        loss_mask=[0],
        rollout_log_probs=[0.0],
        reward={"score": 0.0},
        remove_sample=True,
    )
    sample.rollout_id = getattr(base_sample, "rollout_id", getattr(base_sample, "index", None))
    if hasattr(Sample, "Status"):
        sample.status = Sample.Status.ABORTED
    sample.metadata = {
        "truncated": False,
        "finish_reason": "abort",
        "truncation_reason": None,
        "clawgym_nanobot": {
            "status": "aborted",
            "reason": reason,
            "task_id": selection.task.task_id if selection else None,
            "task_index": selection.task_index if selection else None,
        }
    }
    return sample


def _merge_metadata(current: Any, extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current) if isinstance(current, dict) else {}
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _max_turns_per_sid() -> int | None:
    raw = os.environ.get("CLAWGYM_NANOBOT_MAX_TURNS_PER_SID", "").strip()
    if not raw:
        return None
    value = int(raw)
    return None if value <= 0 else value


def _fork_threshold_tokens() -> int | None:
    raw = os.environ.get("CLAWGYM_NANOBOT_FORK_THRESHOLD_TOKENS", "").strip()
    if not raw:
        return None
    value = int(raw)
    return None if value <= 0 else value


def _reasoning_context_policy() -> str | None:
    raw = os.environ.get("CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY", "").strip()
    if raw:
        return raw
    if _env_truthy("NANOBOT_STRIP_REASONING_CONTEXT", False):
        return "strip"
    return None
