from __future__ import annotations

from typing import Any


async def reward_func(args: Any, sample_or_samples: Any, **kwargs: Any) -> Any:
    del kwargs
    if isinstance(sample_or_samples, list):
        return [_reward_value(args, sample) for sample in sample_or_samples]
    return _reward_value(args, sample_or_samples)


def _reward_value(args: Any, sample: Any) -> float | dict[str, float]:
    reward = getattr(sample, "reward", None)
    if reward is None:
        return {"score": 0.0}
    if isinstance(reward, dict):
        key = getattr(args, "reward_key", None)
        if key:
            return {key: float(reward.get(key, 0.0))}
        if "score" in reward:
            return {"score": float(reward["score"])}
        return {"score": float(next(iter(reward.values()), 0.0))}
    return {"score": float(reward)}
