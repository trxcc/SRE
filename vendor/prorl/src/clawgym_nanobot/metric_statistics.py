"""Small dependency-free helpers for scalar metric summaries."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence


def percentile(ordered: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank value from an already sorted sequence."""
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"percentile fraction must be in [0, 1], got {fraction}")
    return float(ordered[round((len(ordered) - 1) * fraction)])


def add_distribution(
    metrics: MutableMapping[str, int | float],
    prefix: str,
    values: Sequence[float],
) -> None:
    """Add count, moments, extrema, and stable quantiles under ``prefix``."""
    if not values:
        return
    ordered = sorted(float(value) for value in values)
    mean = sum(ordered) / len(ordered)
    metrics[f"{prefix}/count"] = len(ordered)
    metrics[f"{prefix}/mean"] = mean
    metrics[f"{prefix}/std"] = (
        sum((value - mean) ** 2 for value in ordered) / len(ordered)
    ) ** 0.5
    metrics[f"{prefix}/min"] = ordered[0]
    metrics[f"{prefix}/p10"] = percentile(ordered, 0.10)
    metrics[f"{prefix}/p50"] = percentile(ordered, 0.50)
    metrics[f"{prefix}/p90"] = percentile(ordered, 0.90)
    metrics[f"{prefix}/max"] = ordered[-1]


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Pearson correlation, or ``None`` for insufficient/constant data."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = (
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    ) ** 0.5
    if denominator == 0.0:
        return None
    return (
        sum(
            left_value * right_value
            for left_value, right_value in zip(left_centered, right_centered, strict=True)
        )
        / denominator
    )
