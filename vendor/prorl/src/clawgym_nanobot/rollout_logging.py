from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import researchclaw_metrics, user_agent
from .eval_subsets import build_eval_subset_metrics
from .metric_statistics import add_distribution, pearson, percentile


logger = logging.getLogger(__name__)


def log_clawgym_rollout(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Log Slime default rollout metrics plus explicit ClawGym token counts."""
    if getattr(args, "load_debug_rollout_data", None):
        return True

    rollout, metric_utils, logging_utils = _slime_log_helpers()
    log_dict = dict(rollout_extra_metrics or {})
    log_dict |= compute_clawgym_reward_metrics(
        samples,
        target_trajectories=_target_trajectories(args),
    )
    log_dict |= metric_utils.dict_add_prefix(
        compute_clawgym_advantage_diagnostics(args, samples),
        "clawgym/advantage/",
    )
    log_dict |= metric_utils.dict_add_prefix(
        compute_clawgym_sample_metrics(
            samples,
            stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
        ),
        "clawgym/",
    )
    log_dict |= metric_utils.dict_add_prefix(rollout.compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= metric_utils.dict_add_prefix(
        rollout.compute_perf_metrics_from_samples(args, samples, rollout_time),
        "perf/",
    )

    logger.info("perf %s: %s", rollout_id, log_dict)
    step = metric_utils.compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")
    try:
        _retain_latest_train_nanobot_logs(rollout_id)
    except Exception:
        logger.exception(
            "Failed to rotate ResearchClaw training Nanobot logs rollout=%s",
            rollout_id,
        )
    return True


def log_clawgym_eval_rollout(
    rollout_id: int,
    args: Any,
    data: dict[str, Any],
    extra_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Log Slime default eval metrics plus explicit ClawGym token counts."""
    rollout, metric_utils, logging_utils = _slime_log_helpers()
    log_dict = dict(extra_metrics or {})

    for key, dataset in data.items():
        rewards = list(dataset.get("rewards") or [])
        samples = dataset.get("samples")
        rewards_are_task_level = bool(dataset.get("rewards_are_task_level"))
        raw_rewards = list(dataset.get("raw_rewards") or rewards)
        skipped = list(dataset.get("raw_skipped") or _eval_skipped_flags(samples, len(raw_rewards)))
        if rewards_are_task_level:
            task_rewards = [float(value) for value in rewards]
        else:
            task_rewards = _eval_task_rewards(rewards, getattr(args, "n_samples_per_eval_prompt", 1))
        reward_avg = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
        n_skipped = int(dataset.get("n_skipped", sum(1 for value in skipped if value)))
        n_collected = int(dataset.get("n_collected", max(0, len(raw_rewards) - n_skipped)))
        collected_rewards = [reward for reward, is_skipped in zip(raw_rewards, skipped, strict=False) if not is_skipped]
        avg_collected_only = float(
            dataset.get(
                "avg_reward_collected_only",
                sum(collected_rewards) / len(collected_rewards) if collected_rewards else 0.0,
            )
        )
        correct_count = sum(1 for reward in task_rewards if float(reward) > 0.5)

        log_dict[f"eval/{key}"] = reward_avg
        log_dict[f"eval/{key}-reward_avg"] = reward_avg
        log_dict[f"eval/{key}-n_collected"] = n_collected
        log_dict[f"eval/{key}-n_skipped"] = n_skipped
        log_dict[f"eval/{key}-avg_reward_collected_only"] = avg_collected_only
        log_dict[f"eval/{key}-correct_count"] = correct_count
        log_dict[f"eval/{key}-n_tasks"] = len(task_rewards)

        task_metadata = dataset.get("task_metadata")
        if task_metadata is not None:
            try:
                log_dict |= build_eval_subset_metrics(
                    key,
                    task_rewards,
                    list(task_metadata),
                    raw_rewards=raw_rewards,
                    raw_skipped=skipped,
                    task_statuses=dataset.get("task_statuses"),
                    group_size=dataset.get(
                        "n_samples_per_eval_prompt",
                        getattr(args, "n_samples_per_eval_prompt", 1),
                    ),
                    primary_subset=os.environ.get(
                        "CLAWGYM_PRIMARY_EVAL_SUBSET", "core"
                    ),
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping eval-subset metrics for %s: %s", key, exc)

        if samples is not None:
            log_dict |= metric_utils.dict_add_prefix(
                rollout.compute_metrics_from_samples(args, samples),
                f"eval/{key}/",
            )
            log_dict |= metric_utils.dict_add_prefix(
                compute_clawgym_sample_metrics(
                    samples,
                    stop_token_ids=getattr(args, "rollout_stop_token_ids", None),
                ),
                f"eval/{key}/clawgym/",
            )
            reward_metrics = compute_clawgym_reward_metrics(
                samples,
                target_trajectories=len(raw_rewards),
            )
            log_dict |= metric_utils.dict_add_prefix(
                reward_metrics,
                f"eval/{key}/",
            )
            # Stable, short aliases for dashboards.  The full prefixed metrics
            # above retain counts, coverage, distributions, and diagnostics.
            if "rollout/code_verifier_score" in reward_metrics:
                log_dict[f"eval/{key}-code_verifier_avg"] = reward_metrics[
                    "rollout/code_verifier_score"
                ]
            if "rollout/user_agent_score" in reward_metrics:
                log_dict[f"eval/{key}-user_agent_avg"] = reward_metrics[
                    "rollout/user_agent_score"
                ]
            if "openclaw/user_agent_code_spearman" in reward_metrics:
                log_dict[f"eval/{key}-user_agent_code_spearman"] = reward_metrics[
                    "openclaw/user_agent_code_spearman"
                ]
            if "openclaw/user_agent_code_mae" in reward_metrics:
                log_dict[f"eval/{key}-user_agent_code_mae"] = reward_metrics[
                    "openclaw/user_agent_code_mae"
                ]
            if "openclaw/tool_use_rate" in reward_metrics:
                log_dict[f"eval/{key}-tool_use_rate"] = reward_metrics[
                    "openclaw/tool_use_rate"
                ]
            if "openclaw/tool_calls_per_trajectory" in reward_metrics:
                log_dict[f"eval/{key}-tool_calls_per_trajectory"] = reward_metrics[
                    "openclaw/tool_calls_per_trajectory"
                ]

        if "truncated" in dataset:
            truncated = list(dataset.get("truncated") or [])
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated) if truncated else 0.0

        if getattr(args, "log_passrate", False):
            passrate_rewards = task_rewards if rewards_are_task_level else rewards
            passrate_group_size = 1 if rewards_are_task_level else getattr(args, "n_samples_per_eval_prompt", 1)
            log_dict |= metric_utils.dict_add_prefix(
                metric_utils.compute_pass_rate(
                    flat_rewards=passrate_rewards,
                    group_size=passrate_group_size,
                ),
                f"eval/{key}-",
            )

    logger.info("eval %s: %s", rollout_id, log_dict)
    step = metric_utils.compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")
    try:
        from .checkpoint_retention import finalize_evaluated_hf_snapshot

        retention = finalize_evaluated_hf_snapshot(
            rollout_id, args, log_dict
        )
        if retention.get("enabled"):
            logger.info("Evaluated HF checkpoint retention: %s", retention)
    except Exception:
        # Metrics must still be emitted if bounded checkpoint retention fails;
        # the missing promotion remains visible in the training log.
        logger.exception(
            "Failed to finalize evaluated HF checkpoint rollout=%s",
            rollout_id,
        )
    return log_dict


def _eval_task_rewards(rewards: list[Any], group_size: Any) -> list[float]:
    try:
        n_samples_per_prompt = max(1, int(group_size or 1))
    except (TypeError, ValueError):
        n_samples_per_prompt = 1

    flat_rewards = [float(value) for value in rewards]
    if n_samples_per_prompt <= 1:
        return flat_rewards

    task_rewards: list[float] = []
    for start in range(0, len(flat_rewards), n_samples_per_prompt):
        group = flat_rewards[start : start + n_samples_per_prompt]
        if group:
            task_rewards.append(sum(group) / len(group))
    return task_rewards


def _eval_skipped_flags(samples: Any, reward_count: int) -> list[bool]:
    flags = [False] * max(0, int(reward_count))
    if samples is None:
        return flags
    for index, sample in enumerate(list(samples)[: len(flags)]):
        status = getattr(sample, "status", None)
        status_text = str(getattr(status, "value", status)).lower()
        flags[index] = bool(getattr(sample, "remove_sample", False)) or "aborted" in status_text
    return flags


def compute_clawgym_sample_metrics(
    samples: list[Any] | None,
    *,
    stop_token_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[str, int | float]:
    samples = list(samples or [])
    raw_lengths = [_response_tokens(sample) for sample in samples]
    trainable_lengths = [_trainable_tokens(sample) for sample in samples]
    total_lengths = [_total_tokens(sample) for sample in samples]
    prompt_lengths = [
        max(total_length - response_length, 0)
        for total_length, response_length in zip(total_lengths, raw_lengths, strict=True)
    ]
    masked_response_lengths = [
        max(response_length - trainable_length, 0)
        for response_length, trainable_length in zip(raw_lengths, trainable_lengths, strict=True)
    ]
    masked_total_lengths = [
        max(total_length - trainable_length, 0)
        for total_length, trainable_length in zip(total_lengths, trainable_lengths, strict=True)
    ]
    raw_total = sum(raw_lengths)
    trainable_total = sum(trainable_lengths)
    total_tokens = sum(total_lengths)
    masked_response_total = sum(masked_response_lengths)
    masked_total = sum(masked_total_lengths)

    metrics: dict[str, int | float] = {
        "emitted_samples": len(samples),
        "trainable_samples": sum(1 for length in trainable_lengths if length > 0),
        "raw_response_tokens": raw_total,
        "trainable_tokens": trainable_total,
        "trainable_token_ratio": (trainable_total / raw_total) if raw_total > 0 else 0.0,
        "prompt_tokens": sum(prompt_lengths),
        "total_sequence_tokens": total_tokens,
        "masked_response_tokens": masked_response_total,
        "masked_total_tokens": masked_total,
        "trainable_total_token_ratio": (
            trainable_total / total_tokens
        ) if total_tokens > 0 else 0.0,
    }
    metrics |= _prefixed_stats("raw_response_len/", raw_lengths)
    metrics |= _prefixed_stats("trainable_response_len/", trainable_lengths)
    metrics |= _prefixed_stats("prompt_len/", prompt_lengths)
    metrics |= _prefixed_stats("total_sequence_len/", total_lengths)
    metrics |= _prefixed_stats("masked_response_len/", masked_response_lengths)
    metrics |= _prefixed_stats("masked_total_len/", masked_total_lengths)
    metrics |= _trajectory_metrics(samples)
    metrics |= _sample_terminal_metrics(samples, stop_token_ids=stop_token_ids)
    return metrics


def _sample_terminal_metrics(
    samples: list[Any],
    *,
    stop_token_ids: list[int] | tuple[int, ...] | None,
) -> dict[str, int | float]:
    status_counts: dict[str, int] = {}
    finish_counts: dict[str, int] = {}
    terminal_counts: dict[int, int] = {}
    missing_terminal = 0
    for sample in samples:
        status = getattr(sample, "status", None)
        status_key = _metric_key(getattr(status, "value", status) or "missing")
        status_counts[status_key] = status_counts.get(status_key, 0) + 1

        metadata = getattr(sample, "metadata", None)
        finish_reason = metadata.get("finish_reason") if isinstance(metadata, dict) else None
        finish_key = _metric_key(finish_reason or "missing")
        finish_counts[finish_key] = finish_counts.get(finish_key, 0) + 1

        response_length = _response_tokens(sample)
        tokens = getattr(sample, "tokens", None)
        if response_length <= 0 or not tokens:
            missing_terminal += 1
            continue
        terminal_token = int(tokens[-1])
        terminal_counts[terminal_token] = terminal_counts.get(terminal_token, 0) + 1

    metrics: dict[str, int | float] = {}
    for key, count in sorted(status_counts.items()):
        metrics[f"status/{key}/count"] = count
    for key, count in sorted(finish_counts.items()):
        metrics[f"finish_reason/{key}/count"] = count

    terminal_total = sum(terminal_counts.values())
    configured_stops = {int(token_id) for token_id in (stop_token_ids or [])}
    configured_count = sum(terminal_counts.get(token_id, 0) for token_id in configured_stops)
    metrics["terminal_token/count"] = terminal_total
    metrics["terminal_token/missing/count"] = missing_terminal
    metrics["terminal_token/configured_stop/count"] = configured_count
    metrics["terminal_token/configured_stop/fraction"] = (
        configured_count / terminal_total if terminal_total else 0.0
    )
    metrics["terminal_token/other/count"] = terminal_total - configured_count
    metrics["terminal_token/other/fraction"] = (
        (terminal_total - configured_count) / terminal_total if terminal_total else 0.0
    )
    for token_id in sorted(configured_stops):
        metrics[f"terminal_token/id_{token_id}/count"] = terminal_counts.get(token_id, 0)
    return metrics


def compute_clawgym_advantage_diagnostics(
    args: Any,
    samples: list[Any] | None,
) -> dict[str, int | float]:
    """Expose GRPO weighting invariants before the actor forward pass.

    GRPO centers one value per trajectory, while Slime may train several
    emitted pieces per trajectory and may reduce the loss per token.  Those
    three weightings need not have the same mean.  Logging all three makes a
    non-zero initial policy loss explainable without treating its magnitude
    (or the resulting grad norm) as an oracle.
    """
    if getattr(args, "advantage_estimator", "grpo") not in {
        "grpo",
        "gspo",
        "cispo",
    }:
        return {}

    sample_list = list(samples or [])
    if not sample_list:
        return {}

    from .reward_post_process import grpo_normalize_by_group_index

    try:
        raw_rewards, advantages = grpo_normalize_by_group_index(args, sample_list)
        normalization_valid = 1
    except ValueError:
        # Logging runs before train-data conversion. Preserve the inconsistency
        # diagnostics here; the reward postprocessor will then fail fast before
        # any actor step can consume the invalid batch.
        raw_rewards = [float(sample.get_reward_value(args)) for sample in sample_list]
        advantages = [0.0] * len(sample_list)
        normalization_valid = 0
    requested: dict[int, dict[Any, list[float]]] = {}
    grouped: dict[int, dict[Any, list[float]]] = {}
    token_numerator = 0.0
    token_denominator = 0
    trainable_piece_advantages: list[float] = []

    for pos, (sample, raw_reward, advantage) in enumerate(
        zip(sample_list, raw_rewards, advantages, strict=True)
    ):
        group_key = getattr(sample, "group_index", None)
        group_key = pos if group_key is None else int(group_key)
        trajectory_key = getattr(sample, "rollout_id", None)
        if trajectory_key is None:
            trajectory_key = getattr(sample, "index", None)
        if trajectory_key is None:
            trajectory_key = pos

        requested.setdefault(group_key, {}).setdefault(trajectory_key, []).append(
            float(raw_reward)
        )

        trainable_tokens = _trainable_tokens(sample)
        if bool(getattr(sample, "remove_sample", False)) or trainable_tokens <= 0:
            continue

        value = float(advantage)
        grouped.setdefault(group_key, {}).setdefault(trajectory_key, []).append(value)
        trainable_piece_advantages.append(value)
        token_numerator += value * trainable_tokens
        token_denominator += trainable_tokens

    trajectory_advantages: list[float] = []
    group_means: list[float] = []
    within_trajectory_ranges: list[float] = []
    for trajectories in grouped.values():
        group_values: list[float] = []
        for values in trajectories.values():
            trajectory_value = sum(values) / len(values)
            group_values.append(trajectory_value)
            trajectory_advantages.append(trajectory_value)
            within_trajectory_ranges.append(max(values) - min(values))
        if group_values:
            group_means.append(sum(group_values) / len(group_values))

    requested_reward_ranges = [
        max(values) - min(values)
        for trajectories in requested.values()
        for values in trajectories.values()
    ]
    requested_trajectory_count = sum(len(trajectories) for trajectories in requested.values())
    trainable_trajectory_count = len(trajectory_advantages)
    zero_trainable_trajectory_count = max(
        0, requested_trajectory_count - trainable_trajectory_count
    )

    token_mean = token_numerator / token_denominator if token_denominator else 0.0
    trajectory_sum = sum(trajectory_advantages)
    trajectory_mean = (
        trajectory_sum / trainable_trajectory_count
        if trainable_trajectory_count
        else 0.0
    )
    return {
        "normalization_valid": normalization_valid,
        "group_count": len(group_means),
        "trajectory_count": trainable_trajectory_count,
        "requested_trajectory_count": requested_trajectory_count,
        "zero_trainable_trajectory_count": zero_trainable_trajectory_count,
        "trainable_trajectory_fraction": (
            trainable_trajectory_count / requested_trajectory_count
            if requested_trajectory_count
            else 0.0
        ),
        "trainable_piece_count": len(trainable_piece_advantages),
        "trainable_token_count": token_denominator,
        "normalized_mean_trajectory": trajectory_mean,
        "normalized_mean_piece": (
            sum(trainable_piece_advantages) / len(trainable_piece_advantages)
            if trainable_piece_advantages
            else 0.0
        ),
        "normalized_mean_token": token_mean,
        "ratio_one_policy_loss_per_token": -token_mean,
        # The actual per-rollout reducer gives one token-weighted mean to each
        # logical rollout. Fully masked requested rollouts contribute zero but
        # remain in the step denominator; expose both denominator conventions
        # so missing trajectories cannot silently look like a healthy zero.
        "ratio_one_policy_loss_per_rollout_requested": (
            -trajectory_sum / requested_trajectory_count
            if requested_trajectory_count
            else 0.0
        ),
        "ratio_one_policy_loss_per_rollout_trainable": -trajectory_mean,
        "group_mean_abs_max": max(
            (abs(value) for value in group_means), default=0.0
        ),
        "within_trajectory_range_max": max(within_trajectory_ranges, default=0.0),
        "within_trajectory_reward_range_max": max(
            requested_reward_ranges, default=0.0
        ),
        "inconsistent_reward_trajectory_count": sum(
            value > 0.0 for value in requested_reward_ranges
        ),
        "normalized_abs_max": max(
            (abs(value) for value in trainable_piece_advantages), default=0.0
        ),
    }


def compute_clawgym_reward_metrics(
    samples: list[Any] | None,
    *,
    target_trajectories: int | None = None,
) -> dict[str, int | float]:
    """Aggregate one reward/timing record per trajectory for rollout W&B metrics."""
    sample_list = list(samples or [])
    records = list(_iter_unique_clawgym_records(sample_list))
    target = max(len(records), int(target_trajectories or 0))
    completed = [record for record in records if _record_completed(record)]
    timed_out = [record for record in records if not _record_completed(record) and _record_timed_out(record)]
    skipped = [
        record
        for record in records
        if not _record_completed(record) and not _record_timed_out(record)
    ]
    missing = max(0, target - len(records))

    prompt_indices = {
        int(value)
        for record in completed
        if (value := _finite_number(record.get("prompt_index"))) is not None
    }
    observed = len(records)
    submitted = len(completed)  # Legacy key: historically meant completed trajectories.
    metrics: dict[str, int | float] = {
        "openclaw/target_trajectories": target,
        "openclaw/observed_trajectories": observed,
        "openclaw/completed_trajectories": len(completed),
        "openclaw/failed_trajectories": len(skipped),
        "openclaw/missing_trajectories": missing,
        "openclaw/submitted_trajectories": submitted,
        "openclaw/skipped_trajectories": len(skipped),
        "openclaw/timed_out_trajectories": len(timed_out),
        "openclaw/groups_used": len(prompt_indices),
        "openclaw/total_samples": len(sample_list),
        "openclaw/avg_samples_per_trajectory": len(sample_list) / observed if observed else 0.0,
    }
    if target:
        metrics["openclaw/completion_rate"] = len(completed) / target
        metrics["openclaw/timeout_rate"] = len(timed_out) / target
        metrics["openclaw/failure_rate"] = len(skipped) / target
        metrics["openclaw/missing_rate"] = missing / target

    rewards = _record_values(completed, "reward")
    if rewards:
        metrics["openclaw/avg_reward_submitted"] = sum(rewards) / len(rewards)
        metrics["openclaw/avg_reward_with_missing_as_zero"] = sum(rewards) / target if target else 0.0
        add_distribution(metrics, "openclaw/reward_completed", rewards)

    # Self-reward training reports the optimization reward separately from the
    # diagnostic GT reward. Missing GT values after the first configured steps
    # stay missing instead of being silently treated as zero.
    for record_key, metric_key in (
        ("raw_reward", "raw_reward"),
        ("raw_gt_reward", "raw_gt_reward"),
        ("raw_self_reward", "raw_self_reward"),
        ("self_reward_repeat_count", "self_reward_repeat_count"),
    ):
        values = _record_values(records, record_key)
        if values:
            metrics[f"rollout/{metric_key}"] = sum(values) / len(values)
            metrics[f"rollout/{metric_key}_count"] = len(values)
            metrics[f"rollout/{metric_key}_coverage"] = len(values) / target if target else 0.0
            add_distribution(metrics, f"rollout/{metric_key}_distribution", values)
    trainable_flags = [record.get("self_reward_trainable") for record in records if isinstance(record.get("self_reward_trainable"), bool)]
    valid_flags = [record.get("self_reward_valid") for record in records if isinstance(record.get("self_reward_valid"), bool)]
    if trainable_flags:
        metrics["rollout/self_reward_trainable_fraction"] = sum(trainable_flags) / len(trainable_flags)
    if valid_flags:
        metrics["rollout/self_reward_valid_fraction"] = sum(valid_flags) / len(valid_flags)
    _add_self_reward_group_metrics(metrics, records, target=target)

    code_scores = _record_values(completed, "code_verifier_score")
    if code_scores:
        code_mean = sum(code_scores) / len(code_scores)
        code_with_missing = sum(code_scores) / target if target else 0.0
        metrics.update(
            {
                "openclaw/code_verifier_score_mean": code_mean,
                "openclaw/code_verifier_score_with_missing_as_zero": code_with_missing,
                "rollout/code_verifier_score": code_mean,
                "rollout/code_verifier_score_mean": code_mean,
                "rollout/code_verifier_score_with_missing_as_zero": code_with_missing,
                "rollout/code_verifier_score_count": len(code_scores),
                "rollout/code_verifier_score_coverage": len(code_scores) / len(completed),
            }
        )
        add_distribution(metrics, "openclaw/code_verifier_score", code_scores)
        add_distribution(metrics, "rollout/code_verifier_score_distribution", code_scores)

    ua_rows = _user_agent_rows(completed)
    ua_scores = [score for _record, _metadata, score in ua_rows]
    if ua_scores:
        ua_mean = sum(ua_scores) / len(ua_scores)
        ua_with_missing = sum(ua_scores) / target if target else 0.0
        metrics.update(
            {
                "openclaw/user_agent_reward_mean": ua_mean,
                "openclaw/user_agent_reward_with_missing_as_zero": ua_with_missing,
                "rollout/user_agent_score": ua_mean,
                "rollout/user_agent_score_mean": ua_mean,
                "rollout/user_agent_score_with_missing_as_zero": ua_with_missing,
                "rollout/user_agent_score_count": len(ua_scores),
                "rollout/user_agent_score_coverage": len(ua_scores) / len(completed),
            }
        )
        add_distribution(metrics, "openclaw/user_agent_score", ua_scores)
        add_distribution(metrics, "rollout/user_agent_score_distribution", ua_scores)
    ua_metadata = [
        metadata
        for record in completed
        if isinstance((metadata := record.get("user_agent")), dict)
    ]
    if ua_metadata:
        metrics["openclaw/user_agent_metadata_count"] = len(ua_metadata)
        metrics["openclaw/user_agent_metadata_coverage"] = len(ua_metadata) / len(completed)
        metrics["openclaw/user_agent_parse_error_count"] = sum(
            1 for metadata in ua_metadata if metadata.get("parse_error")
        )
    if ua_rows:
        _add_user_agent_metrics(metrics, ua_rows)
        _add_user_agent_code_correlations(metrics, ua_rows)

    task_durations = _record_values(completed, "duration_s")
    if task_durations:
        _add_mean_max_p90(metrics, "openclaw/nanobot_task_duration_sec", task_durations)
        metrics["openclaw/avg_task_duration_sec"] = sum(task_durations) / len(task_durations)
        metrics["openclaw/max_task_duration_sec"] = max(task_durations)
        metrics["openclaw/p90_task_duration_sec"] = _p90(task_durations)

    verifier_durations = _record_values(completed, "code_verifier_duration_s")
    if verifier_durations:
        _add_mean_max_p90(metrics, "openclaw/nanobot_code_verifier_sec", verifier_durations)
        metrics["openclaw/code_verifier_duration_sec_mean"] = sum(verifier_durations) / len(
            verifier_durations
        )
        metrics["openclaw/code_verifier_duration_sec_max"] = max(verifier_durations)
        metrics["rollout/code_verifier_duration_sec_mean"] = metrics[
            "openclaw/code_verifier_duration_sec_mean"
        ]
        metrics["rollout/code_verifier_duration_sec_max"] = metrics[
            "openclaw/code_verifier_duration_sec_max"
        ]

    verifier_ran = [
        record.get("code_verifier_ran_by_agent")
        for record in completed
        if isinstance(record.get("code_verifier_ran_by_agent"), bool)
    ]
    if verifier_ran:
        metrics["openclaw/code_verifier_ran_by_agent_count"] = sum(verifier_ran)
        metrics["openclaw/code_verifier_ran_by_agent_fraction"] = (
            sum(verifier_ran) / len(verifier_ran)
        )
        metrics["openclaw/code_verifier_ran_status_coverage"] = (
            len(verifier_ran) / len(completed)
        )
    verifier_returncodes = [
        int(value)
        for record in completed
        if (value := record.get("code_verifier_returncode")) is not None
        and isinstance(value, int)
        and not isinstance(value, bool)
    ]
    if verifier_returncodes:
        metrics["openclaw/code_verifier_returncode_count"] = len(verifier_returncodes)
        metrics["openclaw/code_verifier_returncode_coverage"] = (
            len(verifier_returncodes) / len(completed)
        )
        metrics["openclaw/code_verifier_nonzero_returncode_fraction"] = sum(
            value != 0 for value in verifier_returncodes
        ) / len(verifier_returncodes)

    user_agent_durations = _record_values(completed, "user_agent_duration_s")
    if user_agent_durations:
        add_distribution(
            metrics,
            "openclaw/user_agent_evaluator_duration_sec",
            user_agent_durations,
        )
        metrics["rollout/user_agent_duration_sec_mean"] = sum(user_agent_durations) / len(
            user_agent_durations
        )
        metrics["rollout/user_agent_duration_sec_max"] = max(user_agent_durations)

    turn_durations = [
        duration
        for record in completed
        for turn in _record_turns(record)
        if (duration := _finite_number(turn.get("duration_s"))) is not None
    ]
    if turn_durations:
        _add_mean_max_p90(metrics, "openclaw/nanobot_turn_duration_sec", turn_durations)

    step_counts = [
        count
        for record in completed
        if (count := _record_step_count(record)) is not None
    ]
    if step_counts:
        metrics["openclaw/avg_steps_per_trajectory"] = sum(step_counts) / len(step_counts)
        metrics["openclaw/steps_observed_trajectories"] = len(step_counts)
        metrics["openclaw/steps_coverage"] = len(step_counts) / len(completed)
        add_distribution(metrics, "openclaw/steps_per_trajectory", step_counts)
    _add_tool_usage_metrics(metrics, completed)
    _add_group_reward_metrics(metrics, completed)
    return metrics


def _target_trajectories(args: Any) -> int:
    try:
        return max(0, int(args.rollout_batch_size) * int(args.n_samples_per_prompt))
    except (AttributeError, TypeError, ValueError):
        return 0


def _record_completed(record: dict[str, Any]) -> bool:
    status = str(record.get("status") or "").strip().lower()
    if status:
        return status == "completed"
    return _finite_number(record.get("code_verifier_score")) is not None


def _record_timed_out(record: dict[str, Any]) -> bool:
    status = str(record.get("status") or "").strip().lower()
    if status in {"timeout", "timed_out", "timed-out"}:
        return True
    reason = str(record.get("reason") or record.get("error") or "").lower()
    return "timeout" in reason or "timed out" in reason


def _record_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = [_finite_number(record.get(key)) for record in records]
    return [value for value in values if value is not None]


def _user_agent_rows(
    records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    rows: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for record in records:
        metadata = record.get("user_agent")
        if not isinstance(metadata, dict):
            continue
        score = _finite_number(metadata.get("score"))
        if score is not None:
            rows.append((record, metadata, score))
    return rows


def _add_user_agent_metrics(
    metrics: dict[str, int | float],
    rows: list[tuple[dict[str, Any], dict[str, Any], float]],
) -> None:
    profile_scores: dict[str, list[float]] = {}
    for _record, metadata, score in rows:
        profile = str(metadata.get("user_agent_profile") or metadata.get("backend") or "unknown")
        profile_scores.setdefault(_metric_key(profile), []).append(score)
    for profile, scores in profile_scores.items():
        metrics[f"openclaw/user_agent_profile/{profile}/count"] = len(scores)
        metrics[f"openclaw/user_agent_profile/{profile}/score_mean"] = sum(scores) / len(scores)

    count = len(rows)
    metrics["openclaw/user_agent_verifier_ran_count"] = sum(
        1 for _record, metadata, _score in rows if metadata.get("verifier_ran_by_agent") is True
    )
    metrics["openclaw/user_agent_verifier_ran_fraction"] = (
        metrics["openclaw/user_agent_verifier_ran_count"] / count
    )
    reasoning_present = sum(
        1 for _record, metadata, _score in rows if metadata.get("user_agent_reasoning_present") is True
    )
    metrics["openclaw/user_agent_reasoning_present_count"] = reasoning_present
    metrics["openclaw/user_agent_reasoning_present_frac"] = reasoning_present / count
    metrics["openclaw/user_agent_prompt_truncated_frac"] = (
        sum(1 for _record, metadata, _score in rows if metadata.get("prompt_truncated") is True) / count
    )
    metrics["openclaw/user_agent_workspace_truncated_frac"] = (
        sum(1 for _record, metadata, _score in rows if metadata.get("workspace_truncated") is True) / count
    )

    feedback_lengths = [
        float(len(feedback.strip()))
        for _record, metadata, _score in rows
        if isinstance((feedback := metadata.get("feedback")), str) and feedback.strip()
    ]
    metrics["openclaw/user_agent_feedback_present_count"] = len(feedback_lengths)
    metrics["openclaw/user_agent_feedback_present_fraction"] = len(feedback_lengths) / count
    if feedback_lengths:
        add_distribution(metrics, "openclaw/user_agent_feedback_chars", feedback_lengths)

    for source_key, output_key in (
        ("user_agent_reasoning_part_count", "user_agent_reasoning_part_count"),
        ("user_agent_reasoning_chars", "user_agent_reasoning_chars"),
        ("user_agent_queue_wait_sec", "user_agent_queue_wait_sec"),
        ("user_agent_score_duration_sec", "user_agent_score_duration_sec"),
        ("user_agent_prepare_duration_sec", "user_agent_prepare_duration_sec"),
        ("user_agent_feedback_duration_sec", "user_agent_feedback_duration_sec"),
        ("user_agent_cot_save_duration_sec", "user_agent_cot_save_duration_sec"),
        ("user_agent_verifier_duration_sec", "user_agent_verifier_duration_sec"),
        ("user_agent_parse_duration_sec", "user_agent_parse_duration_sec"),
        ("user_agent_total_duration_sec", "user_agent_total_duration_sec"),
        ("user_agent_message_count", "user_agent_message_count"),
        ("user_agent_assistant_message_count", "user_agent_assistant_message_count"),
        ("user_agent_tool_call_count", "user_agent_tool_call_count"),
        ("user_agent_tool_call_turn_count", "user_agent_tool_call_turn_count"),
        ("user_agent_tool_result_message_count", "user_agent_tool_result_message_count"),
        ("user_agent_unique_tool_count", "user_agent_unique_tool_count"),
    ):
        values = [
            value
            for _record, metadata, _score in rows
            if (value := _finite_number(metadata.get(source_key))) is not None
        ]
        if values:
            metrics[f"openclaw/{output_key}_mean"] = sum(values) / len(values)
            metrics[f"openclaw/{output_key}_max"] = max(values)

    tool_counts = [
        value
        for _record, metadata, _score in rows
        if (value := _finite_number(metadata.get("user_agent_tool_call_count")))
        is not None
    ]
    if tool_counts:
        tool_using = sum(value > 0 for value in tool_counts)
        metrics["openclaw/user_agent_tool_usage_coverage"] = len(tool_counts) / count
        metrics["openclaw/user_agent_tool_use_rate"] = tool_using / len(tool_counts)
        metrics["openclaw/user_agent_tool_calls_total"] = sum(tool_counts)
        metrics["rollout/user_agent_tool_use_rate"] = metrics[
            "openclaw/user_agent_tool_use_rate"
        ]
        metrics["rollout/user_agent_tool_calls_per_trajectory"] = sum(tool_counts) / len(
            tool_counts
        )
        add_distribution(metrics, "openclaw/user_agent_tool_calls", tool_counts)

    tool_name_counts: dict[str, int] = {}
    for _record, metadata, _score in rows:
        names = metadata.get("user_agent_tool_names")
        if not isinstance(names, list):
            continue
        for name in {
            value.strip()
            for value in names
            if isinstance(value, str) and value.strip()
        }:
            key = _metric_key(name)
            tool_name_counts[key] = tool_name_counts.get(key, 0) + 1
    metrics["openclaw/user_agent_tool_types_observed"] = len(tool_name_counts)
    for name, trajectory_count in sorted(tool_name_counts.items()):
        metrics[f"openclaw/user_agent_tool_name/{name}/trajectory_count"] = trajectory_count
        metrics[f"openclaw/user_agent_tool_name/{name}/trajectory_fraction"] = (
            trajectory_count / count
        )


def _add_user_agent_code_correlations(
    metrics: dict[str, int | float],
    rows: list[tuple[dict[str, Any], dict[str, Any], float]],
) -> None:
    pairs: list[tuple[dict[str, Any], float, float]] = []
    for record, _metadata, ua_score in rows:
        code_score = _finite_number(record.get("code_verifier_score"))
        if code_score is not None:
            pairs.append((record, code_score, ua_score))
    if not pairs:
        return

    pair_count = len(pairs)
    code_values = [code_score for _record, code_score, _ua_score in pairs]
    ua_values = [ua_score for _record, _code_score, ua_score in pairs]
    gaps = [ua_score - code_score for _record, code_score, ua_score in pairs]
    absolute_gaps = [abs(gap) for gap in gaps]
    metrics["openclaw/user_agent_code_pair_count"] = pair_count
    metrics["openclaw/user_agent_code_pair_coverage"] = pair_count / len(rows)
    metrics["openclaw/user_agent_code_bias"] = sum(gaps) / pair_count
    metrics["openclaw/user_agent_code_mae"] = sum(absolute_gaps) / pair_count
    metrics["openclaw/user_agent_code_rmse"] = (
        sum(gap * gap for gap in gaps) / pair_count
    ) ** 0.5
    metrics["openclaw/user_agent_code_agreement_within_0_10"] = sum(
        gap <= 0.10 for gap in absolute_gaps
    ) / pair_count
    metrics["openclaw/user_agent_code_agreement_within_0_25"] = sum(
        gap <= 0.25 for gap in absolute_gaps
    ) / pair_count
    metrics["openclaw/user_agent_higher_than_code_fraction"] = sum(
        gap > 1e-9 for gap in gaps
    ) / pair_count
    metrics["openclaw/user_agent_lower_than_code_fraction"] = sum(
        gap < -1e-9 for gap in gaps
    ) / pair_count
    metrics["openclaw/user_agent_code_equal_fraction"] = sum(
        abs(gap) <= 1e-9 for gap in gaps
    ) / pair_count
    add_distribution(metrics, "openclaw/user_agent_minus_code", gaps)
    add_distribution(metrics, "openclaw/user_agent_code_abs_gap", absolute_gaps)

    code_pass = [value >= 0.5 for value in code_values]
    ua_pass = [value >= 0.5 for value in ua_values]
    metrics["openclaw/user_agent_code_binary_agreement_at_0_5"] = sum(
        left == right for left, right in zip(code_pass, ua_pass, strict=True)
    ) / pair_count
    metrics["openclaw/user_agent_code_both_pass_fraction"] = sum(
        left and right for left, right in zip(code_pass, ua_pass, strict=True)
    ) / pair_count
    metrics["openclaw/user_agent_code_both_fail_fraction"] = sum(
        not left and not right for left, right in zip(code_pass, ua_pass, strict=True)
    ) / pair_count
    metrics["openclaw/user_agent_only_pass_fraction"] = sum(
        right and not left for left, right in zip(code_pass, ua_pass, strict=True)
    ) / pair_count
    metrics["openclaw/code_verifier_only_pass_fraction"] = sum(
        left and not right for left, right in zip(code_pass, ua_pass, strict=True)
    ) / pair_count

    pearson_value = pearson(code_values, ua_values)
    if pearson_value is not None:
        metrics["openclaw/user_agent_code_pearson"] = pearson_value

    batch_corr = user_agent.spearman(
        code_values,
        ua_values,
    )
    if batch_corr is not None:
        metrics["openclaw/user_agent_code_spearman"] = batch_corr
        metrics["openclaw/user_agent_code_spearman_batch"] = batch_corr
        metrics["openclaw/user_agent_code_rank_corr_batch"] = batch_corr

    grouped: dict[int, list[tuple[float, float]]] = {}
    for record, code_score, ua_score in pairs:
        prompt_index = _finite_number(record.get("prompt_index"))
        if prompt_index is not None:
            grouped.setdefault(int(prompt_index), []).append((code_score, ua_score))
    if not grouped:
        return
    group_corrs = [
        corr
        for group_pairs in grouped.values()
        if len(group_pairs) >= 2
        and (
            corr := user_agent.spearman(
                [code_score for code_score, _ua_score in group_pairs],
                [ua_score for _code_score, ua_score in group_pairs],
            )
        )
        is not None
    ]
    metrics["openclaw/user_agent_code_spearman_group_count"] = len(grouped)
    metrics["openclaw/user_agent_code_spearman_group_valid_count"] = len(group_corrs)
    metrics["openclaw/user_agent_code_spearman_group_valid_frac"] = len(group_corrs) / len(grouped)
    metrics["openclaw/user_agent_code_spearman_group_size_mean"] = sum(
        len(group_pairs) for group_pairs in grouped.values()
    ) / len(grouped)
    if group_corrs:
        metrics["openclaw/user_agent_code_spearman_group_mean"] = sum(group_corrs) / len(group_corrs)
        metrics["openclaw/user_agent_code_spearman_group_min"] = min(group_corrs)
        metrics["openclaw/user_agent_code_spearman_group_max"] = max(group_corrs)


def _add_group_reward_metrics(
    metrics: dict[str, int | float],
    records: list[dict[str, Any]],
) -> None:
    grouped: dict[int, list[float]] = {}
    for record in records:
        prompt_index = _finite_number(record.get("prompt_index"))
        reward = _finite_number(record.get("reward"))
        if prompt_index is not None and reward is not None:
            grouped.setdefault(int(prompt_index), []).append(reward)
    if not grouped:
        return

    variances: list[float] = []
    all_zero = all_one = same_mid = mixed = 0
    for rewards in grouped.values():
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        variances.append(variance)
        if all(reward == 0.0 for reward in rewards):
            all_zero += 1
        elif all(reward >= 1.0 for reward in rewards):
            all_one += 1
        elif variance == 0.0:
            same_mid += 1
        else:
            mixed += 1
    count = len(grouped)
    sizes = [float(len(rewards)) for rewards in grouped.values()]
    metrics["openclaw/group_count"] = count
    add_distribution(metrics, "openclaw/group_size", sizes)
    metrics["openclaw/group_all_zero_frac"] = all_zero / count
    metrics["openclaw/group_all_one_frac"] = all_one / count
    metrics["openclaw/group_all_same_mid_frac"] = same_mid / count
    metrics["openclaw/group_mixed_frac"] = mixed / count
    metrics["openclaw/group_reward_var_mean"] = sum(variances) / count
    metrics["openclaw/group_zero_var_frac"] = sum(1 for value in variances if value == 0.0) / count

    _add_group_score_variability(
        metrics,
        records,
        key="code_verifier_score",
        prefix="openclaw/group_code_verifier",
    )
    _add_group_score_variability(
        metrics,
        records,
        key="user_agent_score",
        prefix="openclaw/group_user_agent",
    )


def _add_group_score_variability(
    metrics: dict[str, int | float],
    records: list[dict[str, Any]],
    *,
    key: str,
    prefix: str,
) -> None:
    grouped: dict[int, list[float]] = {}
    for record in records:
        prompt_index = _finite_number(record.get("prompt_index"))
        value = _finite_number(record.get(key))
        if value is None and key == "user_agent_score":
            metadata = record.get("user_agent")
            if isinstance(metadata, dict):
                value = _finite_number(metadata.get("score"))
        if prompt_index is not None and value is not None:
            grouped.setdefault(int(prompt_index), []).append(value)
    if not grouped:
        return
    variances = []
    for values in grouped.values():
        mean = sum(values) / len(values)
        variances.append(sum((value - mean) ** 2 for value in values) / len(values))
    metrics[f"{prefix}_count"] = len(grouped)
    metrics[f"{prefix}_variance_mean"] = sum(variances) / len(variances)
    metrics[f"{prefix}_zero_variance_fraction"] = sum(
        value == 0.0 for value in variances
    ) / len(variances)


def _add_mean_max_p90(
    metrics: dict[str, int | float],
    prefix: str,
    values: list[float],
) -> None:
    metrics[f"{prefix}_mean"] = sum(values) / len(values)
    metrics[f"{prefix}_max"] = max(values)
    metrics[f"{prefix}_p90"] = _p90(values)


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return percentile(ordered, 0.9)


def _trajectory_metrics(samples: list[Any]) -> dict[str, int | float]:
    records = list(_iter_unique_clawgym_records(samples))
    metrics: dict[str, int | float] = {"trajectory/sessions": len(records)}

    for section, metric_prefix in (
        ("trajectory", "trajectory"),
        ("emitted", "session_emitted"),
    ):
        for source_key, output_key in (
            ("samples", "samples"),
            ("trainable_samples", "trainable_samples"),
            ("response_tokens", "raw_response_tokens"),
            ("trainable_tokens", "trainable_tokens"),
        ):
            values = [
                value
                for record in records
                if (value := _record_number(record, section, source_key)) is not None
            ]
            if not values:
                continue
            metrics[f"{metric_prefix}/{output_key}/sum"] = sum(values)
            metrics |= _prefixed_stats(f"{metric_prefix}/{output_key}/", values)

    request_counts = [
        value
        for record in records
        if (value := _record_number(record, "requests", "count")) is not None
    ]
    if request_counts:
        metrics["requests/count/sum"] = sum(request_counts)
        metrics |= _prefixed_stats("requests/count/", request_counts)

    for source_key, output_prefix in (
        ("prompt_build_modes", "prompt_build"),
        ("prompt_build_reasons", "prompt_build_reason"),
    ):
        counts: dict[str, int] = {}
        for record in records:
            requests = record.get("requests")
            raw_counts = requests.get(source_key) if isinstance(requests, dict) else None
            if not isinstance(raw_counts, dict):
                continue
            for raw_key, raw_count in raw_counts.items():
                try:
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                if count > 0:
                    key = _metric_key(raw_key)
                    counts[key] = counts.get(key, 0) + count
        for key, count in sorted(counts.items()):
            metrics[f"requests/{output_prefix}/{key}/count"] = count

    raw_prefix_violations = [
        value
        for record in records
        if (value := _record_number(record, "requests", "raw_prefix_violations")) is not None
    ]
    if raw_prefix_violations:
        metrics["requests/raw_prefix_violations/count"] = sum(raw_prefix_violations)
        metrics["requests/raw_prefix_violations/coverage_sessions"] = len(raw_prefix_violations)

    for value_key, output_prefix in (
        ("prompt_len_values", "prompt_len"),
        ("response_len_values", "response_len"),
        ("total_len_values", "total_len"),
    ):
        values: list[float] = []
        for record in records:
            values.extend(_record_number_list(record, "requests", value_key))
        if values:
            metrics[f"requests/{output_prefix}/count"] = len(values)
            metrics |= _prefixed_stats(f"requests/{output_prefix}/", values)

    for source_key, output_key in (
        ("prompt_tokens", "prompt_tokens"),
        ("response_tokens", "response_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        values = [
            value
            for record in records
            if (value := _record_number(record, "requests", source_key)) is not None
        ]
        if values:
            metrics[f"requests/{output_key}/sum"] = sum(values)
            metrics |= _prefixed_stats(f"requests/{output_key}/", values)

    turn_counts = [_record_turn_count(record) for record in records]
    turn_counts = [value for value in turn_counts if value is not None]
    if turn_counts:
        metrics["trajectory/turns/sum"] = sum(turn_counts)
        metrics |= _prefixed_stats("trajectory/turns/", turn_counts)

    turn_durations = [_record_turn_duration_sum(record) for record in records]
    turn_durations = [value for value in turn_durations if value is not None]
    if turn_durations:
        metrics["trajectory/turn_duration_s/sum"] = sum(turn_durations)
        metrics |= _prefixed_stats("trajectory/turn_duration_s/", turn_durations)

    nonzero_returncodes = [
        count
        for record in records
        if (count := _record_turn_nonzero_returncode_count(record)) is not None
    ]
    if nonzero_returncodes:
        metrics["trajectory/turn_nonzero_returncode/count"] = sum(nonzero_returncodes)
        metrics["trajectory/turn_returncode_coverage_sessions"] = len(nonzero_returncodes)
    metrics |= _trajectory_prepare_metrics(records)
    metrics |= _split_reason_metrics(records)
    # ResearchClaw records are already deduplicated by session above, so judge
    # calls, required outputs, and solver activity are counted once even when
    # one trajectory emits multiple trainable token-prefix samples.
    metrics |= researchclaw_metrics.compute_researchclaw_sample_metrics(records)
    return metrics


def _trajectory_prepare_metrics(records: list[dict[str, Any]]) -> dict[str, int | float]:
    """Aggregate one trajectory-preparation summary per deduplicated session."""
    actions = ("initial", "clean", "realign", "token_fork", "message_branch", "rewrite_merge")
    totals = {action: 0 for action in actions}
    active_sessions = {action: 0 for action in actions}
    discarded_trainable_tokens = 0
    summaries = 0
    for record in records:
        summary = record.get("trajectory_prepare")
        if not isinstance(summary, dict):
            continue
        summaries += 1
        counts = summary.get("counts")
        if isinstance(counts, dict):
            for action in actions:
                try:
                    count = max(0, int(counts.get(action, 0) or 0))
                except (TypeError, ValueError):
                    count = 0
                totals[action] += count
                active_sessions[action] += int(count > 0)
        try:
            discarded_trainable_tokens += max(
                0, int(summary.get("realign_discarded_trainable_tokens", 0) or 0)
            )
        except (TypeError, ValueError):
            pass
    if not summaries:
        return {}
    metrics: dict[str, int | float] = {"trajectory_prepare/sessions": summaries}
    for action in actions:
        metrics[f"trajectory_prepare/{action}/count"] = totals[action]
        metrics[f"trajectory_prepare/{action}/sessions"] = active_sessions[action]
    metrics["trajectory_prepare/realign_discarded_trainable_tokens/sum"] = discarded_trainable_tokens
    return metrics


def _split_reason_metrics(records: list[dict[str, Any]]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    counts: dict[str, int] = {}
    covered = 0
    multi_sample_sessions = 0
    for record in records:
        summary = record.get("split_reasons")
        if not isinstance(summary, dict):
            continue
        covered += 1
        try:
            is_multi = int(summary.get("multi_sample", 0) or 0) > 0
        except (TypeError, ValueError):
            is_multi = False
        if is_multi:
            multi_sample_sessions += 1
        reasons = summary.get("reasons")
        if not isinstance(reasons, dict):
            continue
        for reason, raw_count in reasons.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                key = _metric_key(reason)
                counts[key] = counts.get(key, 0) + count
    if covered:
        metrics["split_reasons/coverage_sessions"] = covered
        metrics["split_reasons/multi_sample_sessions"] = multi_sample_sessions
        metrics["split_reasons/multi_sample_fraction"] = multi_sample_sessions / covered
    for reason, count in sorted(counts.items()):
        metrics[f"split_reasons/{reason}/count"] = count
    return metrics


def _iter_unique_clawgym_records(samples: list[Any]):
    polar_records: dict[str, dict[str, Any]] = {}
    polar_trace_ids: dict[str, set[int]] = {}
    seen: set[str] = set()
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        record = metadata.get("clawgym_nanobot")
        if isinstance(record, dict):
            session_id = record.get("session_id") or getattr(sample, "session_id", None)
            key = f"session:{session_id}" if session_id else f"sample:{id(sample)}"
            if key in seen:
                continue
            seen.add(key)
            yield record
            continue

        polar = metadata.get("polar")
        if not isinstance(polar, dict):
            continue
        session_id = polar.get("session_id") or getattr(sample, "session_id", None)
        if not session_id:
            continue
        session_key = str(session_id)
        normalized = polar_records.setdefault(
            session_key,
            _new_polar_clawgym_record(sample, polar, session_key),
        )
        trace_index = _nonnegative_int(polar.get("trace_index", 0))
        trace_ids = polar_trace_ids.setdefault(session_key, set())
        if trace_index in trace_ids:
            continue
        trace_ids.add(trace_index)
        trajectory = normalized["trajectory"]
        response_tokens = _response_tokens(sample)
        trainable_tokens = _trainable_tokens(sample)
        trajectory["samples"] += 1
        trajectory["trainable_samples"] += int(trainable_tokens > 0)
        trajectory["response_tokens"] += response_tokens
        trajectory["trainable_tokens"] += trainable_tokens

    yield from polar_records.values()


def _new_polar_clawgym_record(
    sample: Any,
    polar: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    trajectory_metadata = polar.get("trajectory_metadata")
    if not isinstance(trajectory_metadata, dict):
        trajectory_metadata = {}
    evaluation = trajectory_metadata.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    timing = polar.get("timing")
    if not isinstance(timing, dict):
        timing = {}

    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        reward = reward.get("score", next(iter(reward.values()), None))
    status = str(getattr(polar.get("session_status"), "value", polar.get("session_status") or ""))
    error = polar.get("result_error") or polar.get("trajectory_error")
    duration_parts = [
        _finite_number(timing.get(key))
        for key in (
            "register_to_init_queue_ms",
            "init_ms",
            "run_ms",
            "postrun_ms",
        )
    ]
    record: dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "reason": error,
        "error": error,
        "prompt_index": getattr(sample, "group_index", None),
        "reward": reward,
        "trajectory": {
            "samples": 0,
            "trainable_samples": 0,
            "response_tokens": 0,
            "trainable_tokens": 0,
        },
    }
    if all(value is not None for value in duration_parts):
        record["duration_s"] = (
            sum(value for value in duration_parts if value is not None) / 1000.0
        )
    record_count = _finite_number(trajectory_metadata.get("record_count"))
    trace_count = _finite_number(trajectory_metadata.get("trace_count"))
    if record_count is not None:
        record["step_count"] = record_count
        record["requests"] = {"count": record_count}
    if trace_count is not None:
        is_multi = int(trace_count > 1)
        record["split_reasons"] = {
            "multi_sample": is_multi,
            "reasons": {"multiple_trace_chains": is_multi} if is_multi else {},
        }
    code_score = _finite_number(evaluation.get("code_verifier_score"))
    user_agent_metadata = evaluation.get("user_agent")
    if isinstance(user_agent_metadata, dict):
        # Copy before adding compatibility fallbacks: trajectory metadata is
        # shared by split traces and must remain immutable during aggregation.
        user_agent_metadata = dict(user_agent_metadata)
    else:
        user_agent_metadata = {}
    user_agent_score = _finite_number(evaluation.get("user_agent_score"))
    if user_agent_score is None:
        user_agent_score = _finite_number(user_agent_metadata.get("score"))
    elif _finite_number(user_agent_metadata.get("score")) is None:
        user_agent_metadata["score"] = user_agent_score
    if code_score is None:
        code_score = _finite_number(user_agent_metadata.get("verifier_current_score"))
    if code_score is not None:
        record["code_verifier_score"] = code_score
    if user_agent_score is not None:
        record["user_agent_score"] = user_agent_score
    if user_agent_metadata:
        record["user_agent"] = user_agent_metadata
    for key in (
        "raw_reward", "raw_gt_reward", "raw_self_reward", "self_reward_repeat_count"
    ):
        value = _finite_number(evaluation.get(key))
        if value is not None:
            record[key] = value
    for key in ("self_reward_valid", "self_reward_trainable"):
        value = evaluation.get(key)
        if isinstance(value, bool):
            record[key] = value
    group_summary = trajectory_metadata.get("self_reward_group")
    if isinstance(group_summary, dict):
        record["self_reward_group"] = group_summary
    feedback = evaluation.get("user_agent_feedback")
    if not isinstance(feedback, str):
        feedback = user_agent_metadata.get("feedback")
    if isinstance(feedback, str) and feedback.strip():
        record["user_agent_feedback"] = feedback.strip()
    verifier_duration = _finite_number(evaluation.get("code_verifier_duration_seconds"))
    if verifier_duration is None:
        verifier_duration = _finite_number(
            user_agent_metadata.get("user_agent_verifier_duration_sec")
        )
    if verifier_duration is not None:
        record["code_verifier_duration_s"] = verifier_duration
    user_agent_duration = _finite_number(evaluation.get("user_agent_duration_seconds"))
    if user_agent_duration is not None:
        record["user_agent_duration_s"] = user_agent_duration
    verifier_returncode = evaluation.get("code_verifier_returncode")
    if verifier_returncode is None:
        verifier_returncode = user_agent_metadata.get("verifier_current_returncode")
    try:
        if verifier_returncode is not None:
            record["code_verifier_returncode"] = int(verifier_returncode)
    except (TypeError, ValueError):
        pass
    verifier_ran = evaluation.get("code_verifier_ran_by_agent")
    if verifier_ran is None:
        verifier_ran = user_agent_metadata.get("verifier_ran_by_agent")
    if isinstance(verifier_ran, bool):
        record["code_verifier_ran_by_agent"] = verifier_ran
    tool_usage = trajectory_metadata.get("tool_usage")
    if isinstance(tool_usage, dict):
        record["tool_usage"] = dict(tool_usage)
    return record


def _add_self_reward_group_metrics(
    metrics: dict[str, int | float],
    records: list[dict[str, Any]],
    *,
    target: int,
) -> None:
    """Aggregate one diagnostic row per self-reward group.

    Undefined correlations remain absent and are paired with explicit
    count/coverage metrics; they are never converted to zero.
    """
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        summary = record.get("self_reward_group")
        if not isinstance(summary, dict):
            continue
        group_id = summary.get("group_id")
        if not group_id:
            continue
        groups.setdefault(str(group_id), summary)
    if not groups:
        return

    rows = list(groups.values())
    group_count = len(rows)
    metrics["rollout/self_reward_group_count"] = group_count

    field_names = {
        "coverage": "coverage",
        "spearman": "spearman",
        "kendall_tau_b": "kendall_tau_b",
        "pairwise_accuracy": "pairwise_accuracy",
        "top1_agreement": "top1_agreement",
        "top1_regret": "top1_regret",
        "top1_repeat_agreement": "top1_repeat_agreement",
        "mean_repeat_std": "mean_repeat_std",
        "mean_repeat_range": "mean_repeat_range",
        "planned_candidates": "planned_candidates",
        "comparable_candidates": "comparable_candidates",
        "planned_repeats": "planned_repeats",
    }
    for source_key, metric_key in field_names.items():
        values = [
            value
            for row in rows
            if isinstance((packet_metrics := row.get("metrics")), dict)
            and (value := _finite_number(packet_metrics.get(source_key))) is not None
        ]
        if not values:
            continue
        prefix = f"rollout/self_reward_{metric_key}"
        metrics[prefix] = sum(values) / len(values)
        metrics[f"{prefix}_count"] = len(values)
        metrics[f"{prefix}_group_coverage"] = len(values) / group_count
        add_distribution(metrics, f"{prefix}_distribution", values)

    repeat_correlations: list[float] = []
    for row in rows:
        packet_metrics = row.get("metrics")
        if not isinstance(packet_metrics, dict):
            continue
        values = packet_metrics.get("repeat_to_consensus_spearman")
        if not isinstance(values, list):
            continue
        repeat_correlations.extend(
            value for raw in values if (value := _finite_number(raw)) is not None
        )
    if repeat_correlations:
        metrics["rollout/self_reward_repeat_to_consensus_spearman"] = (
            sum(repeat_correlations) / len(repeat_correlations)
        )
        add_distribution(
            metrics,
            "rollout/self_reward_repeat_to_consensus_spearman_distribution",
            repeat_correlations,
        )

    # Candidate-level accounting shows whether the median is based on all
    # requested repeats or only a surviving subset.
    accepted = _record_values(records, "self_reward_repeat_count")
    planned_total = 0.0
    for row in rows:
        packet_metrics = row.get("metrics")
        if not isinstance(packet_metrics, dict):
            continue
        planned_candidates = _finite_number(packet_metrics.get("planned_candidates"))
        planned_repeats = _finite_number(packet_metrics.get("planned_repeats"))
        if planned_candidates is not None and planned_repeats is not None:
            planned_total += planned_candidates * planned_repeats
    metrics["rollout/self_reward_accepted_repeat_count"] = sum(accepted)
    if planned_total > 0:
        metrics["rollout/self_reward_planned_repeat_count"] = planned_total
        metrics["rollout/self_reward_repeat_acceptance_fraction"] = sum(accepted) / planned_total

    reasons: dict[str, int] = {}
    for record in records:
        if not isinstance(record.get("self_reward_valid"), bool):
            continue
        key = _metric_key(record.get("reward_source") or record.get("self_reward_reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    for reason, count in sorted(reasons.items()):
        metrics[f"rollout/self_reward_reason/{reason}/count"] = count
    if target:
        metrics["rollout/self_reward_diagnostic_trajectory_coverage"] = (
            sum(1 for record in records if isinstance(record.get("self_reward_valid"), bool)) / target
        )


def _add_tool_usage_metrics(
    metrics: dict[str, int | float],
    records: list[dict[str, Any]],
) -> None:
    """Aggregate solver-agent tool calls from structured Polar metadata."""
    row_pairs = [
        (record, usage)
        for record in records
        if isinstance((usage := record.get("tool_usage")), dict)
    ]
    if not row_pairs:
        return
    rows = [usage for _record, usage in row_pairs]

    def numbers(key: str) -> list[float]:
        return [
            value
            for usage in rows
            if (value := _finite_number(usage.get(key))) is not None
        ]

    calls = numbers("tool_calls")
    call_turns = numbers("tool_call_turns")
    assistant_messages = numbers("assistant_messages")
    parallel_turns = numbers("parallel_tool_call_turns")
    unique_tools = numbers("unique_tools")
    named_calls = numbers("named_tool_calls")
    unnamed_calls = numbers("unnamed_tool_calls")
    finish_reason_calls = numbers("finish_reason_tool_calls")
    max_parallel = numbers("max_parallel_tool_calls")
    available_tools_mean = numbers("available_tools_mean")
    available_tools_min = numbers("available_tools_min")
    available_tools_max = numbers("available_tools_max")
    used = sum(
        1
        for usage in rows
        if usage.get("used_tools") is True
        or (_finite_number(usage.get("tool_calls")) or 0.0) > 0.0
    )
    covered = len(rows)
    total_records = len(records)
    metrics["openclaw/tool_usage_metadata_count"] = covered
    metrics["openclaw/tool_usage_metadata_coverage"] = (
        covered / total_records if total_records else 0.0
    )
    metrics["openclaw/tool_using_trajectories"] = used
    metrics["openclaw/tool_use_rate"] = used / covered
    metrics["openclaw/tool_use_rate_with_missing_as_zero"] = (
        used / total_records if total_records else 0.0
    )
    metrics["rollout/tool_use_rate"] = metrics["openclaw/tool_use_rate"]

    if calls:
        total_calls = sum(calls)
        metrics["openclaw/tool_calls_total"] = total_calls
        metrics["openclaw/tool_calls_per_trajectory"] = total_calls / covered
        metrics["openclaw/tool_calls_per_tool_using_trajectory"] = (
            total_calls / used if used else 0.0
        )
        metrics["rollout/tool_calls_total"] = total_calls
        metrics["rollout/tool_calls_per_trajectory"] = total_calls / covered
        add_distribution(metrics, "openclaw/tool_calls_per_trajectory_distribution", calls)
    if call_turns:
        total_call_turns = sum(call_turns)
        metrics["openclaw/tool_call_turns_total"] = total_call_turns
        metrics["openclaw/tool_call_turns_per_trajectory"] = total_call_turns / covered
        add_distribution(metrics, "openclaw/tool_call_turns_per_trajectory_distribution", call_turns)
    else:
        total_call_turns = 0.0
    if assistant_messages:
        total_assistant_messages = sum(assistant_messages)
        metrics["openclaw/assistant_messages_total"] = total_assistant_messages
        metrics["openclaw/assistant_messages_per_trajectory"] = (
            total_assistant_messages / covered
        )
        metrics["openclaw/tool_call_message_rate"] = (
            total_call_turns / total_assistant_messages
            if total_assistant_messages
            else 0.0
        )
    if parallel_turns:
        total_parallel_turns = sum(parallel_turns)
        metrics["openclaw/parallel_tool_call_turns_total"] = total_parallel_turns
        metrics["openclaw/parallel_tool_call_turn_fraction"] = (
            total_parallel_turns / total_call_turns if total_call_turns else 0.0
        )
    if unique_tools:
        metrics["openclaw/unique_tools_per_trajectory_mean"] = sum(unique_tools) / len(
            unique_tools
        )
        metrics["rollout/unique_tools_per_trajectory"] = metrics[
            "openclaw/unique_tools_per_trajectory_mean"
        ]
        add_distribution(metrics, "openclaw/unique_tools_per_trajectory", unique_tools)
    if named_calls or unnamed_calls:
        named_total = sum(named_calls)
        unnamed_total = sum(unnamed_calls)
        known_total = named_total + unnamed_total
        metrics["openclaw/named_tool_calls_total"] = named_total
        metrics["openclaw/unnamed_tool_calls_total"] = unnamed_total
        metrics["openclaw/named_tool_call_fraction"] = (
            named_total / known_total if known_total else 0.0
        )
    if finish_reason_calls:
        metrics["openclaw/finish_reason_tool_calls_total"] = sum(finish_reason_calls)
    if max_parallel:
        metrics["openclaw/max_parallel_tool_calls"] = max(max_parallel)
    if available_tools_mean:
        metrics["openclaw/tool_availability_metadata_coverage"] = (
            len(available_tools_mean) / covered
        )
        metrics["openclaw/available_tools_per_trajectory_mean"] = sum(
            available_tools_mean
        ) / len(available_tools_mean)
        metrics["openclaw/available_tools_min"] = min(
            available_tools_min or available_tools_mean
        )
        metrics["openclaw/available_tools_max"] = max(
            available_tools_max or available_tools_mean
        )
        add_distribution(
            metrics,
            "openclaw/available_tools_per_trajectory",
            available_tools_mean,
        )

    calls_by_name: dict[str, float] = {}
    trajectories_by_name: dict[str, int] = {}
    outcome_values_by_name: dict[str, dict[str, list[float]]] = {}
    for record, usage in row_pairs:
        raw = usage.get("calls_by_name")
        if not isinstance(raw, dict):
            continue
        names_in_trajectory: set[str] = set()
        for name, count in raw.items():
            value = _finite_number(count)
            if value is None or value <= 0:
                continue
            key = _metric_key(name)
            calls_by_name[key] = calls_by_name.get(key, 0.0) + value
            names_in_trajectory.add(key)
        for name in names_in_trajectory:
            trajectories_by_name[name] = trajectories_by_name.get(name, 0) + 1
            outcomes = outcome_values_by_name.setdefault(
                name,
                {"reward": [], "code_verifier": [], "user_agent": []},
            )
            reward = _finite_number(record.get("reward"))
            code = _finite_number(record.get("code_verifier_score"))
            ua = _finite_number(record.get("user_agent_score"))
            if ua is None and isinstance(record.get("user_agent"), dict):
                ua = _finite_number(record["user_agent"].get("score"))
            if reward is not None:
                outcomes["reward"].append(reward)
            if code is not None:
                outcomes["code_verifier"].append(code)
            if ua is not None:
                outcomes["user_agent"].append(ua)
    named_total = sum(calls_by_name.values())
    metrics["openclaw/tool_types_observed"] = len(calls_by_name)
    for name, count in sorted(calls_by_name.items()):
        metrics[f"openclaw/tool_calls_by_name/{name}/count"] = count
        metrics[f"openclaw/tool_calls_by_name/{name}/fraction"] = (
            count / named_total if named_total else 0.0
        )
        trajectory_count = trajectories_by_name.get(name, 0)
        metrics[f"openclaw/tool_calls_by_name/{name}/trajectory_count"] = (
            trajectory_count
        )
        metrics[f"openclaw/tool_calls_by_name/{name}/trajectory_fraction"] = (
            trajectory_count / covered
        )
        for outcome_name, values in outcome_values_by_name.get(name, {}).items():
            if values:
                metrics[
                    f"openclaw/tool_calls_by_name/{name}/{outcome_name}_mean"
                ] = sum(values) / len(values)

    _add_tool_outcome_metrics(metrics, row_pairs)


def _add_tool_outcome_metrics(
    metrics: dict[str, int | float],
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    """Relate solver tool intensity to reward and both evaluator scores."""
    for outcome_name in ("reward", "code_verifier", "user_agent"):
        paired: list[tuple[float, float]] = []
        for record, usage in rows:
            calls = _finite_number(usage.get("tool_calls"))
            if calls is None:
                continue
            if outcome_name == "reward":
                outcome = _finite_number(record.get("reward"))
            elif outcome_name == "code_verifier":
                outcome = _finite_number(record.get("code_verifier_score"))
            else:
                outcome = _finite_number(record.get("user_agent_score"))
                if outcome is None and isinstance(record.get("user_agent"), dict):
                    outcome = _finite_number(record["user_agent"].get("score"))
            if outcome is not None:
                paired.append((calls, outcome))
        if not paired:
            continue

        used_outcomes = [outcome for calls, outcome in paired if calls > 0]
        unused_outcomes = [outcome for calls, outcome in paired if calls <= 0]
        metrics[f"openclaw/tool_outcome/{outcome_name}/pair_count"] = len(paired)
        metrics[f"openclaw/tool_outcome/{outcome_name}/pair_coverage"] = (
            len(paired) / len(rows)
        )
        if used_outcomes:
            metrics[f"openclaw/tool_outcome/{outcome_name}/tool_used_mean"] = (
                sum(used_outcomes) / len(used_outcomes)
            )
            metrics[f"openclaw/tool_outcome/{outcome_name}/tool_used_pass_rate"] = (
                sum(value >= 0.5 for value in used_outcomes) / len(used_outcomes)
            )
        if unused_outcomes:
            metrics[f"openclaw/tool_outcome/{outcome_name}/no_tool_mean"] = (
                sum(unused_outcomes) / len(unused_outcomes)
            )
            metrics[f"openclaw/tool_outcome/{outcome_name}/no_tool_pass_rate"] = (
                sum(value >= 0.5 for value in unused_outcomes) / len(unused_outcomes)
            )
        call_values = [calls for calls, _outcome in paired]
        outcome_values = [outcome for _calls, outcome in paired]
        pearson_value = pearson(call_values, outcome_values)
        if pearson_value is not None:
            metrics[f"openclaw/tool_outcome/{outcome_name}/calls_pearson"] = pearson_value
        spearman = user_agent.spearman(call_values, outcome_values)
        if spearman is not None:
            metrics[f"openclaw/tool_outcome/{outcome_name}/calls_spearman"] = spearman


def _record_number(record: dict[str, Any], section: str, key: str) -> float | None:
    values = record.get(section)
    if not isinstance(values, dict):
        return None
    return _finite_number(values.get(key))


def _record_number_list(record: dict[str, Any], section: str, key: str) -> list[float]:
    values = record.get(section)
    if not isinstance(values, dict):
        return []
    raw = values.get(key)
    if not isinstance(raw, list):
        return []
    parsed: list[float] = []
    for value in raw:
        number = _finite_number(value)
        if number is not None:
            parsed.append(number)
    return parsed


def _record_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    run = record.get("run")
    if not isinstance(run, dict):
        return []
    turns = run.get("turns")
    if not isinstance(turns, list):
        return []
    return [turn for turn in turns if isinstance(turn, dict)]


def _record_turn_count(record: dict[str, Any]) -> float | None:
    turns = _record_turns(record)
    if turns:
        return float(len(turns))
    return _finite_number(record.get("step_count"))


def _record_step_count(record: dict[str, Any]) -> float | None:
    value = _finite_number(record.get("step_count"))
    if value is not None:
        return value
    turns = _record_turns(record)
    return float(len(turns)) if turns else None


def _record_turn_duration_sum(record: dict[str, Any]) -> float | None:
    values = [_finite_number(turn.get("duration_s")) for turn in _record_turns(record)]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _record_turn_nonzero_returncode_count(record: dict[str, Any]) -> int | None:
    turns = _record_turns(record)
    if not turns:
        return None
    count = 0
    for turn in turns:
        try:
            returncode = int(turn.get("returncode", 0) or 0)
        except (TypeError, ValueError):
            returncode = 0
        if returncode != 0:
            count += 1
    return count


def _response_tokens(sample: Any) -> int:
    return _nonnegative_int(getattr(sample, "response_length", 0))


def _total_tokens(sample: Any) -> int:
    tokens = getattr(sample, "tokens", None)
    try:
        return _nonnegative_int(len(tokens))
    except TypeError:
        return _response_tokens(sample)


def _trainable_tokens(sample: Any) -> int:
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is not None:
        try:
            return _nonnegative_int(sum(int(value) for value in loss_mask))
        except TypeError:
            return 0
    effective = getattr(sample, "effective_response_length", None)
    if isinstance(effective, (int, float)) and not isinstance(effective, bool):
        return _nonnegative_int(effective)
    return _response_tokens(sample)


def _nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def _metric_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    key = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in key.split("_") if part) or "unknown"


def _retain_latest_train_nanobot_logs(rollout_id: int) -> None:
    """Keep only this rollout's direct-runtime training session logs.

    ResearchClaw eval sessions use the ``eval-task-`` prefix and are retained
    independently by the eval snapshot path, so this rotation deliberately
    touches only ``task-<rollout>-*`` directories.
    """
    if os.environ.get("RESEARCHCLAW_KEEP_TRAIN_NANOBOT_LOGS", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    root_raw = os.environ.get("NANOBOT_LOG_DIR", "").strip()
    if not root_raw:
        return
    root = Path(root_raw).resolve()
    # Polar rollouts can persist only the collected snapshots on the trainer
    # node, leaving no local direct-runtime log directory to rotate.
    if not root.exists():
        return
    if root == Path(root.anchor) or len(root.parts) < 3 or not root.is_dir():
        raise ValueError(f"Refusing unsafe Nanobot log rotation root: {root}")

    current = int(rollout_id)
    session_pattern = re.compile(r"^task-(\d+)-")
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            continue
        match = session_pattern.match(candidate.name)
        if match is None or int(match.group(1)) == current:
            continue
        shutil.rmtree(candidate)

    marker = root / "LATEST_TRAIN_ROLLOUT_ID"
    temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(f"{current}\n", encoding="utf-8")
    os.replace(temporary, marker)


def _prefixed_stats(prefix: str, values: list[int] | list[float]) -> dict[str, int | float]:
    if not values:
        return {
            f"{prefix}mean": 0.0,
            f"{prefix}median": 0.0,
            f"{prefix}max": 0,
            f"{prefix}min": 0,
        }
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2
    return {
        f"{prefix}mean": sum(ordered) / count,
        f"{prefix}median": median,
        f"{prefix}max": ordered[-1],
        f"{prefix}min": ordered[0],
    }


def _slime_log_helpers():
    rollout = importlib.import_module("slime.ray.rollout")
    metric_utils = importlib.import_module("slime.utils.metric_utils")
    logging_utils = importlib.import_module("slime.utils.logging_utils")
    return rollout, metric_utils, logging_utils


add_clawgym_rollout_metrics = log_clawgym_rollout
log_clawgym_eval_metrics = log_clawgym_eval_rollout
