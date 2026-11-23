from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any


def _sample_has_trainable_tokens(sample: Any) -> bool:
    """Return whether a rollout piece can contribute policy-gradient tokens.

    ClawGym emits a fully masked placeholder when a rollout times out.  Those
    placeholders keep group cardinality observable, but they must not change
    the GRPO baseline because they cannot contribute to the actor loss.

    ``None`` (or a missing ``loss_mask`` attribute) retains the historical
    meaning of an unmasked response, which keeps ordinary Slime samples
    backward compatible.
    """
    if bool(getattr(sample, "remove_sample", False)):
        return False

    missing = object()
    loss_mask = getattr(sample, "loss_mask", missing)
    if loss_mask is missing:
        return True
    if loss_mask is None:
        response_length = getattr(sample, "response_length", None)
        return True if response_length is None else int(response_length) > 0

    try:
        total = loss_mask.sum()
    except AttributeError:
        return any(float(value) > 0.0 for value in loss_mask)
    if hasattr(total, "item"):
        total = total.item()
    return float(total) > 0.0


def grpo_normalize_by_group_index(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if getattr(args, "advantage_estimator", "grpo") in {"policy_gradient", "ppo"}:
        return raw_rewards, raw_rewards.copy()

    # Normalize over trajectories, not flattened trace pieces.  A trajectory
    # can fan out into multiple pieces, so it must count only once.  Fully
    # masked/removed placeholders are retained in the returned arrays but are
    # excluded from the reward population and receive zero advantage.
    groups: dict[int, dict[Any, list[tuple[int, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pos, sample in enumerate(samples):
        group_key = getattr(sample, "group_index", None)
        if group_key is None:
            group_key = pos
        trajectory_key = getattr(sample, "rollout_id", None)
        if trajectory_key is None:
            trajectory_key = getattr(sample, "index", None)
        if trajectory_key is None:
            trajectory_key = pos
        groups[int(group_key)][trajectory_key].append(
            (pos, raw_rewards[pos], _sample_has_trainable_tokens(sample))
        )

    normalized = [0.0] * len(samples)
    use_std = getattr(args, "grpo_std_normalization", True)
    for trajectories in groups.values():
        trajectory_rewards: dict[Any, float] = {}
        for trajectory_key, indexed_rewards in trajectories.items():
            eligible_rewards = [
                reward
                for _, reward, is_eligible in indexed_rewards
                if is_eligible
            ]
            if eligible_rewards:
                first_reward = eligible_rewards[0]
                if any(reward != first_reward for reward in eligible_rewards[1:]):
                    raise ValueError(
                        "pieces from the same trajectory must carry an identical reward: "
                        f"trajectory={trajectory_key!r}, rewards={eligible_rewards!r}"
                    )
                trajectory_rewards[trajectory_key] = first_reward

        if not trajectory_rewards:
            continue

        values = list(trajectory_rewards.values())
        mean = sum(values) / len(values)
        denom = 1.0
        if use_std:
            centered_values = [value - mean for value in values]
            # Match Slime's native torch.std() semantics for the unsplit case:
            # sample variance (N-1), with a finite singleton convention.
            variance = (
                sum(value * value for value in centered_values) / (len(centered_values) - 1)
                if len(centered_values) > 1
                else 0.0
            )
            denom = sqrt(variance) + 1e-6
        for trajectory_key, indexed_rewards in trajectories.items():
            if trajectory_key not in trajectory_rewards:
                continue
            trajectory_reward = trajectory_rewards[trajectory_key]
            trajectory_advantage = float((trajectory_reward - mean) / denom)
            for pos, _reward, is_eligible in indexed_rewards:
                if is_eligible:
                    normalized[pos] = trajectory_advantage
    return raw_rewards, normalized


def policy_gradient_batch_mean_by_group_index(
    args: Any,
    samples: list[Any],
) -> tuple[list[float], list[float]]:
    """Subtract one trajectory-weighted batch reward mean for policy gradient.

    A single agent trajectory can be split into multiple trainable samples.
    Computing the baseline over the flattened samples would therefore give
    trajectories with more trainable pieces extra weight.  ``group_index`` is
    shared by all pieces from the same rollout group, so compute one reward per
    group and broadcast ``reward - batch_group_mean`` back to every piece.
    """
    estimator = getattr(args, "advantage_estimator", None)
    if estimator != "policy_gradient":
        raise ValueError(
            "policy_gradient_batch_mean_by_group_index requires "
            f"advantage_estimator='policy_gradient', got {estimator!r}"
        )

    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not samples:
        return raw_rewards, []

    grouped_trajectories: dict[int, dict[Any, list[tuple[int, float, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pos, (sample, reward) in enumerate(zip(samples, raw_rewards, strict=True)):
        group_key = getattr(sample, "group_index", None)
        group_key = pos if group_key is None else int(group_key)
        trajectory_key = getattr(sample, "rollout_id", None)
        if trajectory_key is None:
            trajectory_key = getattr(sample, "index", None)
        if trajectory_key is None:
            trajectory_key = pos
        grouped_trajectories[group_key][trajectory_key].append(
            (pos, reward, _sample_has_trainable_tokens(sample))
        )

    eligible_positions: set[int] = set()
    group_reward_means: dict[int, float] = {}
    for group_key, trajectories in grouped_trajectories.items():
        trajectory_rewards: list[float] = []
        for trajectory_key, rows in trajectories.items():
            eligible = [(pos, reward) for pos, reward, is_eligible in rows if is_eligible]
            if not eligible:
                continue
            first_reward = eligible[0][1]
            if any(reward != first_reward for _pos, reward in eligible[1:]):
                raise ValueError(
                    "pieces from the same trajectory must carry an identical reward: "
                    f"trajectory={trajectory_key!r}"
                )
            trajectory_rewards.append(first_reward)
            eligible_positions.update(pos for pos, _reward in eligible)
        if trajectory_rewards:
            group_reward_means[group_key] = sum(trajectory_rewards) / len(trajectory_rewards)
    if not group_reward_means:
        return raw_rewards, [0.0] * len(samples)
    batch_reward_mean = sum(group_reward_means.values()) / len(group_reward_means)
    centered_rewards = [
        float(reward - batch_reward_mean) if pos in eligible_positions else 0.0
        for pos, reward in enumerate(raw_rewards)
    ]
    return raw_rewards, centered_rewards
