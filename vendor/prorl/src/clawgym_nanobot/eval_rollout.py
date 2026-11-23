from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput
from slime.utils.types import Sample

from . import distributed
from . import generate as generate_mod
from . import tasks
from . import training_free
from . import user_agent
from .metric_statistics import add_distribution, pearson


logger = logging.getLogger(__name__)


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> RolloutFnEvalOutput:
    del data_source
    if not evaluation:
        raise ValueError("nanobot eval_rollout.generate_rollout only supports evaluation=True")
    return asyncio.run(_run_eval(args, rollout_id))


async def _run_eval(
    args: Any,
    rollout_id: int,
    *,
    source_env_name: str = "OPENCLAW_EVAL_TASKS",
    task_offset: int = 0,
    task_limit: int | None = None,
    n_samples_override: int | None = None,
    controller_mode_override: str | None = None,
    controller_order_env_name: str = "OPENCLAW_EVAL_ORDER",
    controller_use_batch_size_env: bool = True,
) -> RolloutFnEvalOutput:
    sources = tasks.task_sources_from_env(source_env_name, default=None)
    if not sources:
        logger.warning("[ClawGym-Nanobot] %s not set, skipping rollout", source_env_name)
        return RolloutFnEvalOutput(data={}, metrics={})
    sources = _select_task_window(sources, offset=task_offset, limit=task_limit)
    if source_env_name == "OPENCLAW_EVAL_TASKS":
        sources = _limit_task_sources(sources)

    sampling_params = _eval_sampling_params(args)
    n_eval_samples = (
        int(n_samples_override)
        if n_samples_override is not None
        else _positive_int_env("OPENCLAW_EVAL_N_SAMPLES", _arg_int(args, "n_samples_per_eval_prompt", 1))
    )
    if n_eval_samples <= 0:
        raise ValueError("n_samples_override must be positive")
    max_workers = _eval_max_workers()
    training_free_controller = training_free.controller_from_env(
        mode_override=controller_mode_override,
        task_index_offset=task_offset,
        order_env_name=controller_order_env_name,
        use_batch_size_env=controller_use_batch_size_env,
    )

    all_results: dict[str, dict[str, Any]] = {}
    all_metrics: dict[str, Any] = {}
    persisted_results: list[dict[str, Any]] = []
    for source in sources:
        dataset_name = source.root.name
        source_start = time.monotonic()
        n_tasks = len(source.tasks)
        n_total = n_tasks * n_eval_samples
        worker_count = max(1, min(max_workers, n_total))
        rewards = [0.0] * n_total
        code_verifier_rewards = [0.0] * n_total
        user_agent_rewards: list[float | None] = [None] * n_total
        skipped = [False] * n_total
        samples: list[Any] = []
        log_completed_later = True

        if training_free_controller is not None:
            completed = await _run_training_free_eval(
                args,
                rollout_id,
                source,
                n_eval_samples=n_eval_samples,
                sampling_params=sampling_params,
                max_workers=worker_count,
                controller=training_free_controller,
            )
        elif distributed.enabled():
            completed = await asyncio.to_thread(
                distributed.run_eval_jobs,
                args,
                rollout_id,
                source,
                n_eval_samples=n_eval_samples,
                sampling_params=sampling_params,
            )
        else:
            log_completed_later = False
            logger.info("[ClawGym-Nanobot] eval workers=%d", worker_count)
            semaphore = asyncio.Semaphore(worker_count)
            coroutines = []
            for sample_idx in range(n_eval_samples):
                for task_index, task in enumerate(source.tasks):
                    flat_index = sample_idx * n_tasks + task_index
                    coroutines.append(
                        _run_eval_one(
                            args,
                            task,
                            rollout_id=rollout_id,
                            flat_index=flat_index,
                            group_index=task_index,
                            sampling_params=sampling_params,
                            semaphore=semaphore,
                        )
                    )
            completed = []
            for future in asyncio.as_completed(coroutines):
                item = await future
                completed.append(item)
                _, task_index, sample = item
                _log_eval_result(
                    source,
                    task_index,
                    _sample_reward(sample, args),
                    bool(getattr(sample, "remove_sample", False)),
                    code_verifier=_sample_code_verifier(sample),
                    user_agent=_sample_user_agent(sample),
                )

        for flat_index, task_index, sample in completed:
            reward = _sample_reward(sample, args)
            rewards[flat_index] = reward
            code_verifier_rewards[flat_index] = _sample_code_verifier(sample)
            user_agent_rewards[flat_index] = _sample_user_agent(sample)
            skipped[flat_index] = bool(getattr(sample, "remove_sample", False))
            samples.append(sample)
            if log_completed_later:
                _log_eval_result(
                    source,
                    task_index,
                    reward,
                    skipped[flat_index],
                    code_verifier=code_verifier_rewards[flat_index],
                    user_agent=user_agent_rewards[flat_index],
                )

        task_rewards = []
        task_code_verifier_rewards = []
        task_user_agent_rewards: list[float | None] = []
        for task_index in range(n_tasks):
            task_samples = [rewards[sample_idx * n_tasks + task_index] for sample_idx in range(n_eval_samples)]
            task_rewards.append(sum(task_samples) / len(task_samples))
            code_samples = [
                code_verifier_rewards[sample_idx * n_tasks + task_index] for sample_idx in range(n_eval_samples)
            ]
            task_code_verifier_rewards.append(sum(code_samples) / len(code_samples))
            ua_samples = [
                user_agent_rewards[sample_idx * n_tasks + task_index]
                for sample_idx in range(n_eval_samples)
                if user_agent_rewards[sample_idx * n_tasks + task_index] is not None
            ]
            task_user_agent_rewards.append(
                sum(float(value) for value in ua_samples) / len(ua_samples) if ua_samples else None
            )
            task_skipped = [
                skipped[sample_idx * n_tasks + task_index]
                for sample_idx in range(n_eval_samples)
            ]
            persisted_results.append(
                {
                    "dataset": dataset_name,
                    "task_id": source.tasks[task_index].task_id,
                    "task_index": task_index,
                    "task_dir": str(source.tasks[task_index].task_dir.resolve()),
                    "rollout_id": rollout_id,
                    "score": task_rewards[-1],
                    "code_verifier_score": task_code_verifier_rewards[-1],
                    "user_agent_score": task_user_agent_rewards[-1],
                    "status": (
                        "skipped"
                        if all(task_skipped)
                        else "partial"
                        if any(task_skipped)
                        else "completed"
                    ),
                    "sample_scores": task_samples,
                    "sample_skipped": task_skipped,
                }
            )

        n_skipped = sum(1 for value in skipped if value)
        n_collected = n_total - n_skipped
        average_reward = sum(task_rewards) / n_tasks if n_tasks else 0.0
        collected_rewards = [reward for reward, is_skipped in zip(rewards, skipped, strict=False) if not is_skipped]
        avg_reward_collected_only = sum(collected_rewards) / len(collected_rewards) if collected_rewards else 0.0
        collected_code_verifier_rewards = [
            reward
            for reward, is_skipped in zip(code_verifier_rewards, skipped, strict=False)
            if not is_skipped
        ]
        code_verifier_collected_only = (
            sum(collected_code_verifier_rewards) / len(collected_code_verifier_rewards)
            if collected_code_verifier_rewards
            else 0.0
        )
        correct_count = sum(1 for reward in task_rewards if reward > 0.5)
        duration = time.monotonic() - source_start

        logger.info(
            "[ClawGym-Nanobot] eval dataset %s: %d/%d correct, avg=%.4f, "
            "collected=%d/%d, avg_collected_only=%.4f, duration=%.1fs%s",
            dataset_name,
            correct_count,
            n_tasks,
            average_reward,
            n_collected,
            n_total,
            avg_reward_collected_only,
            duration,
            f", n_samples={n_eval_samples}" if n_eval_samples > 1 else "",
        )

        samples.sort(key=lambda sample: (getattr(sample, "index", 0) is None, getattr(sample, "index", 0)))
        dataset_result = {
            "rewards": task_rewards,
            "rewards_are_task_level": True,
            "raw_rewards": rewards,
            "code_verifier_rewards": task_code_verifier_rewards,
            "raw_code_verifier_rewards": code_verifier_rewards,
            "user_agent_rewards": task_user_agent_rewards,
            "raw_user_agent_rewards": user_agent_rewards,
            "raw_skipped": skipped,
            "samples": samples,
            "n_collected": n_collected,
            "n_skipped": n_skipped,
            "avg_reward_collected_only": avg_reward_collected_only,
            "code_verifier_collected_only": code_verifier_collected_only,
        }
        if training_free_controller is not None:
            method_metadata = training_free_controller.evaluation_metadata()
            dataset_result["training_free"] = method_metadata
            method_name = training_free_controller.method_name
            all_metrics[f"eval/{dataset_name}-{method_name}_final_version"] = int(
                method_metadata.get("snapshot_version", 0)
            )
            all_metrics[f"eval/{dataset_name}-{method_name}_entry_count"] = int(
                method_metadata.get("entry_count", 0)
            )
        all_results[dataset_name] = dataset_result
        all_metrics.update(
            _build_eval_metrics(
                dataset_name,
                task_rewards=task_rewards,
                n_collected=n_collected,
                n_skipped=n_skipped,
                avg_reward_collected_only=avg_reward_collected_only,
                duration=duration,
                code_verifier_rewards=task_code_verifier_rewards,
                code_verifier_collected_only=code_verifier_collected_only,
                user_agent_rewards=task_user_agent_rewards,
                samples=samples,
            )
        )

    if output_path := os.environ.get("CLAWGYM_EVAL_RESULTS_JSON", "").strip():
        _write_eval_results_json(Path(output_path), persisted_results)

    return RolloutFnEvalOutput(data=all_results, metrics=all_metrics)


def _write_eval_results_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically persist complete task-level eval results when explicitly requested."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)
    logger.info("[ClawGym-Nanobot] wrote %d task results to %s", len(rows), path)


def _log_eval_result(
    source: tasks.TaskSource,
    task_index: int,
    reward: float,
    skipped: bool,
    *,
    code_verifier: float,
    user_agent: float | None,
) -> None:
    logger.info(
        "[ClawGym-Nanobot] eval task %s: reward=%.4f code_verifier=%.4f%s%s",
        source.tasks[task_index].task_id,
        reward,
        code_verifier,
        f" user_agent={user_agent:.4f}" if user_agent is not None else "",
        " skipped" if skipped else "",
    )


async def _run_eval_one(
    args: Any,
    task: tasks.TaskSpec,
    *,
    rollout_id: int,
    flat_index: int,
    group_index: int,
    sampling_params: dict[str, Any],
    semaphore: asyncio.Semaphore,
    training_free_payload: training_free.BatchPayload | None = None,
) -> tuple[int, int, Any]:
    async with semaphore:
        sample = Sample(
            index=flat_index,
            group_index=group_index,
            prompt=task.user_query,
            metadata={
                "task_id": task.task_id,
                "task_index": group_index,
                "task_dir": str(task.task_dir),
            },
        )
        sample.rollout_id = rollout_id
        if training_free_payload is not None:
            training_free.attach_sample_context(sample, training_free_payload)
            sample.clawgym_task_override = task
        try:
            result = await generate_mod.generate(args, sample, dict(sampling_params), evaluation=True)
        except Exception as exc:
            logger.exception("[ClawGym-Nanobot] eval task %s failed", task.task_id)
            result = _aborted_eval_sample(sample, f"{type(exc).__name__}: {exc}")
        if isinstance(result, list):
            result = result[-1] if result else _aborted_eval_sample(sample, "empty eval result")
        return flat_index, group_index, result


async def _run_training_free_eval(
    args: Any,
    rollout_id: int,
    source: tasks.TaskSource,
    *,
    n_eval_samples: int,
    sampling_params: dict[str, Any],
    max_workers: int,
    controller: training_free.TrainingFreeController,
) -> list[tuple[int, int, Any]]:
    n_tasks = len(source.tasks)
    completed: list[tuple[int, int, Any]] = []
    for task_indices in controller.task_batches(n_tasks):
        payload = controller.begin_batch(
            dataset=source.root.name,
            task_indices=task_indices,
            task_ids=[source.tasks[index].task_id for index in task_indices],
            task_queries=[str(source.tasks[index].user_query) for index in task_indices],
        )
        if distributed.enabled():
            batch_completed = await asyncio.to_thread(
                distributed.run_eval_jobs,
                args,
                rollout_id,
                source,
                n_eval_samples=n_eval_samples,
                sampling_params=sampling_params,
                task_indices=task_indices,
                training_free_payload=payload,
            )
        else:
            semaphore = asyncio.Semaphore(max(1, min(max_workers, len(task_indices) * n_eval_samples)))
            coroutines = [
                _run_eval_one(
                    args,
                    source.tasks[task_index],
                    rollout_id=rollout_id,
                    flat_index=sample_idx * n_tasks + task_index,
                    group_index=task_index,
                    sampling_params=sampling_params,
                    semaphore=semaphore,
                    training_free_payload=payload,
                )
                for sample_idx in range(n_eval_samples)
                for task_index in task_indices
            ]
            batch_completed = [await future for future in asyncio.as_completed(coroutines)]

        batch_completed.sort(key=lambda item: (int(item[1]), int(item[0])))
        actual = sorted(task_index for _flat, task_index, _sample in batch_completed)
        expected = sorted(task_indices * n_eval_samples)
        if actual != expected:
            raise RuntimeError(
                "training-free batch result order/coverage mismatch: "
                f"expected task indices {expected}, got {actual}"
            )
        completed.extend(batch_completed)

        async def complete(messages: list[dict[str, str]], sampling: dict[str, Any]) -> str:
            if distributed.enabled():
                return await asyncio.to_thread(
                    distributed.run_auxiliary_chat_completion, args, messages, sampling
                )
            return await generate_mod.auxiliary_chat_completion(args, messages, sampling)

        await controller.finish_batch(
            dataset=source.root.name,
            completed=batch_completed,
            source_tasks=source.tasks,
            complete=complete,
        )
    return completed


def _aborted_eval_sample(base_sample: Sample, reason: str) -> Sample:
    sample = Sample(
        index=base_sample.index,
        group_index=base_sample.group_index,
        prompt=base_sample.prompt,
        tokens=[0, 0],
        response="",
        response_length=1,
        loss_mask=[0],
        rollout_log_probs=[0.0],
        reward={"score": 0.0},
        remove_sample=True,
        metadata={
            "truncated": False,
            "finish_reason": "abort",
            "truncation_reason": None,
            "clawgym_nanobot": {
                "status": "aborted",
                "reason": reason,
                "task_id": base_sample.metadata.get("task_id") if isinstance(base_sample.metadata, dict) else None,
                "task_index": base_sample.group_index,
            }
        },
    )
    sample.rollout_id = getattr(base_sample, "rollout_id", None)
    sample.status = Sample.Status.ABORTED
    return sample


def _eval_sampling_params(args: Any) -> dict[str, Any]:
    return {
        "temperature": _arg_first(args, ("eval_temperature", "rollout_temperature"), 0.7),
        "top_p": _arg_first(args, ("eval_top_p", "rollout_top_p"), 1.0),
        "top_k": _arg_first(args, ("eval_top_k", "rollout_top_k"), -1),
        "max_new_tokens": _arg_first(args, ("eval_max_response_len", "rollout_max_response_len"), 8192),
        "skip_special_tokens": _arg_first(args, ("rollout_skip_special_tokens",), False),
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }


def _sample_reward(sample: Any, args: Any) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        key = getattr(args, "eval_reward_key", None) or getattr(args, "reward_key", None) or "score"
        if key in reward:
            return float(reward[key])
        if reward:
            return float(next(iter(reward.values())))
        return 0.0
    if reward is None:
        return 0.0
    return float(reward)


def _clawgym_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("clawgym_nanobot")
    return value if isinstance(value, dict) else {}


def _sample_code_verifier(sample: Any) -> float:
    metadata = _clawgym_metadata(sample)
    value = metadata.get("code_verifier_score", metadata.get("reward", 0.0))
    return float(value or 0.0)


def _sample_user_agent(sample: Any) -> float | None:
    metadata = _clawgym_metadata(sample)
    value = metadata.get("user_agent")
    if not isinstance(value, dict) or value.get("score") is None:
        return None
    return float(value["score"])


def _build_eval_metrics(
    dataset_name: str,
    *,
    task_rewards: list[float],
    n_collected: int,
    n_skipped: int,
    avg_reward_collected_only: float,
    duration: float,
    code_verifier_rewards: list[float],
    code_verifier_collected_only: float,
    user_agent_rewards: list[float | None],
    samples: list[Any],
) -> dict[str, Any]:
    reward_avg = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
    correct_count = sum(1 for reward in task_rewards if reward > 0.5)
    code_verifier_avg = (
        sum(code_verifier_rewards) / len(code_verifier_rewards) if code_verifier_rewards else 0.0
    )
    ua_values = [float(value) for value in user_agent_rewards if value is not None]
    metrics = {
        f"eval/{dataset_name}": reward_avg,
        f"eval/{dataset_name}-reward_avg": reward_avg,
        f"eval/{dataset_name}-correct_count": correct_count,
        f"eval/{dataset_name}-n_tasks": len(task_rewards),
        f"eval/{dataset_name}-n_collected": n_collected,
        f"eval/{dataset_name}-n_skipped": n_skipped,
        f"eval/{dataset_name}-avg_reward_collected_only": avg_reward_collected_only,
        f"eval/{dataset_name}-duration_s": duration,
        f"eval/{dataset_name}-code_verifier_avg": code_verifier_avg,
        f"eval/{dataset_name}-code_verifier_collected_only": code_verifier_collected_only,
    }
    add_distribution(
        metrics,
        f"eval/{dataset_name}-reward",
        task_rewards,
    )
    add_distribution(
        metrics,
        f"eval/{dataset_name}-code_verifier",
        code_verifier_rewards,
    )
    if ua_values:
        metrics[f"eval/{dataset_name}-user_agent_avg"] = sum(ua_values) / len(ua_values)
        metrics[f"eval/{dataset_name}-user_agent_count"] = len(ua_values)
        metrics[f"eval/{dataset_name}-user_agent_coverage"] = len(ua_values) / len(
            task_rewards
        ) if task_rewards else 0.0
        add_distribution(
            metrics,
            f"eval/{dataset_name}-user_agent",
            ua_values,
        )
        paired_scores = [
            (float(code_score), float(ua_score))
            for code_score, ua_score in zip(code_verifier_rewards, user_agent_rewards, strict=False)
            if ua_score is not None
        ]
        if paired_scores:
            code_values = [code_score for code_score, _ua_score in paired_scores]
            ua_pair_values = [ua_score for _code_score, ua_score in paired_scores]
            gaps = [
                ua_score - code_score for code_score, ua_score in paired_scores
            ]
            absolute_gaps = [abs(gap) for gap in gaps]
            pair_count = len(paired_scores)
            metrics[f"eval/{dataset_name}-user_agent_code_pair_count"] = pair_count
            metrics[f"eval/{dataset_name}-user_agent_code_pair_coverage"] = (
                pair_count / len(task_rewards) if task_rewards else 0.0
            )
            metrics[f"eval/{dataset_name}-user_agent_code_bias"] = sum(gaps) / pair_count
            metrics[f"eval/{dataset_name}-user_agent_code_mae"] = (
                sum(absolute_gaps) / pair_count
            )
            metrics[f"eval/{dataset_name}-user_agent_code_rmse"] = (
                sum(gap * gap for gap in gaps) / pair_count
            ) ** 0.5
            metrics[f"eval/{dataset_name}-user_agent_code_agreement_within_0_10"] = sum(
                gap <= 0.10 for gap in absolute_gaps
            ) / pair_count
            metrics[f"eval/{dataset_name}-user_agent_code_binary_agreement_at_0_5"] = sum(
                (code_score >= 0.5) == (ua_score >= 0.5)
                for code_score, ua_score in paired_scores
            ) / pair_count
            add_distribution(
                metrics,
                f"eval/{dataset_name}-user_agent_minus_code",
                gaps,
            )
            pearson_value = pearson(code_values, ua_pair_values)
            if pearson_value is not None:
                metrics[f"eval/{dataset_name}-user_agent_code_pearson"] = pearson_value
            correlation = user_agent.spearman(
                code_values,
                ua_pair_values,
            )
            if correlation is not None:
                metrics[f"eval/{dataset_name}-user_agent_code_spearman"] = correlation
    metrics.update(_eval_user_agent_diagnostics(dataset_name, samples))
    return metrics


def _eval_user_agent_diagnostics(dataset_name: str, samples: list[Any]) -> dict[str, Any]:
    rows = [metadata for sample in samples if (metadata := _sample_user_agent_metadata(sample)) is not None]
    if not rows:
        return {}

    prefix = f"eval/{dataset_name}-user_agent"
    metrics: dict[str, Any] = {
        f"{prefix}_parse_errors": sum(1 for metadata in rows if metadata.get("parse_error")),
        f"{prefix}_verifier_ran": sum(
            1 for metadata in rows if metadata.get("verifier_ran_by_agent") is True
        ),
        f"{prefix}_prompt_truncated_frac": sum(
            1 for metadata in rows if metadata.get("prompt_truncated") is True
        )
        / len(rows),
        f"{prefix}_workspace_truncated_frac": sum(
            1 for metadata in rows if metadata.get("workspace_truncated") is True
        )
        / len(rows),
    }
    profile_scores: dict[str, list[float]] = {}
    for metadata in rows:
        score = _finite_number(metadata.get("score"))
        if score is None:
            continue
        profile = _metric_key(metadata.get("user_agent_profile") or metadata.get("backend") or "unknown")
        profile_scores.setdefault(profile, []).append(score)
    for profile, scores in profile_scores.items():
        metrics[f"{prefix}_profile/{profile}/count"] = len(scores)
        metrics[f"{prefix}_profile/{profile}/score_mean"] = sum(scores) / len(scores)

    for field in (
        "user_agent_queue_wait_sec",
        "user_agent_score_duration_sec",
        "user_agent_total_duration_sec",
    ):
        values = [
            value
            for metadata in rows
            if (value := _finite_number(metadata.get(field))) is not None
        ]
        if values:
            metrics[f"{prefix}_{field.removeprefix('user_agent_')}_mean"] = sum(values) / len(values)
            metrics[f"{prefix}_{field.removeprefix('user_agent_')}_max"] = max(values)
    return metrics


def _sample_user_agent_metadata(sample: Any) -> dict[str, Any] | None:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    clawgym = metadata.get("clawgym_nanobot")
    if not isinstance(clawgym, dict):
        return None
    user_agent_metadata = clawgym.get("user_agent")
    return user_agent_metadata if isinstance(user_agent_metadata, dict) else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


def _metric_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    key = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in key.split("_") if part) or "unknown"


def _eval_max_workers() -> int:
    for name in ("NANOBOT_EVAL_NUM_WORKERS", "OPENCLAW_EVAL_NUM_WORKERS", "NANOBOT_ROLLOUT_MAX_WORKERS_PER_NODE"):
        raw = os.environ.get(name, "").strip()
        if raw:
            return max(1, int(raw))
    return 64


def _select_task_window(
    sources: list[tasks.TaskSource],
    *,
    offset: int,
    limit: int | None,
) -> list[tasks.TaskSource]:
    if offset < 0:
        raise ValueError("task_offset must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("task_limit must be positive")
    if offset == 0 and limit is None:
        return sources
    if len(sources) != 1:
        raise ValueError("windowed training-free rollout requires exactly one task source")
    source = sources[0]
    if offset >= len(source.tasks):
        raise ValueError(f"task window starts past dataset: offset={offset} tasks={len(source.tasks)}")
    selected = source.tasks[offset : None if limit is None else offset + limit]
    if not selected:
        raise ValueError("task window selected no tasks")
    return [tasks.TaskSource(root=source.root, tasks=tuple(selected))]


def _limit_task_sources(sources: list[tasks.TaskSource]) -> list[tasks.TaskSource]:
    raw = os.environ.get("CLAWGYM_EVAL_NUM_PROMPTS", "").strip()
    limited = sources
    if raw:
        try:
            remaining = int(raw)
        except ValueError:
            logger.warning("[ClawGym-Nanobot] ignoring invalid CLAWGYM_EVAL_NUM_PROMPTS=%r", raw)
        else:
            if remaining > 0:
                limited = []
                for source in sources:
                    if remaining <= 0:
                        break
                    selected = source.tasks[:remaining]
                    if selected:
                        limited.append(tasks.TaskSource(root=source.root, tasks=selected))
                        remaining -= len(selected)

    shard_count = _int_env("CLAWGYM_EVAL_SHARD_COUNT", 1)
    shard_index = _int_env("CLAWGYM_EVAL_SHARD_INDEX", 0)
    if shard_count < 1:
        raise ValueError("CLAWGYM_EVAL_SHARD_COUNT must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            "CLAWGYM_EVAL_SHARD_INDEX must be in "
            f"[0, CLAWGYM_EVAL_SHARD_COUNT), got {shard_index}/{shard_count}"
        )
    if shard_count == 1:
        return limited

    sharded: list[tasks.TaskSource] = []
    global_index = 0
    for source in limited:
        selected = []
        for task in source.tasks:
            if global_index % shard_count == shard_index:
                selected.append(task)
            global_index += 1
        if selected:
            sharded.append(tasks.TaskSource(root=source.root, tasks=tuple(selected)))
    return sharded


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def _arg_int(args: Any, name: str, default: int) -> int:
    try:
        return int(getattr(args, name, default) or default)
    except (TypeError, ValueError):
        return int(default)


def _arg_first(args: Any, names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return value
    return default
