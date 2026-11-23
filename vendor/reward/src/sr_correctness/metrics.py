from __future__ import annotations

import itertools
import math
import statistics
from typing import Any


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    ml, mr = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - ml) * (b - mr) for a, b in zip(left, right))
    dl = sum((a - ml) ** 2 for a in left)
    dr = sum((b - mr) ** 2 for b in right)
    return None if dl == 0 or dr == 0 else numerator / math.sqrt(dl * dr)


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(_ranks(left), _ranks(right))


def kendall_tau_b(left: list[float], right: list[float]) -> float | None:
    concordant = discordant = tie_left = tie_right = 0
    for i, j in itertools.combinations(range(len(left)), 2):
        dl = (left[i] > left[j]) - (left[i] < left[j])
        dr = (right[i] > right[j]) - (right[i] < right[j])
        if dl == 0 and dr == 0:
            continue
        if dl == 0:
            tie_left += 1
        elif dr == 0:
            tie_right += 1
        elif dl == dr:
            concordant += 1
        else:
            discordant += 1
    denominator = math.sqrt((concordant + discordant + tie_left) * (concordant + discordant + tie_right))
    return None if denominator == 0 else (concordant - discordant) / denominator


def pairwise_accuracy(gt: list[float], pred: list[float]) -> float | None:
    values = []
    for i, j in itertools.combinations(range(len(gt)), 2):
        dg = (gt[i] > gt[j]) - (gt[i] < gt[j])
        if dg == 0:
            continue
        dp = (pred[i] > pred[j]) - (pred[i] < pred[j])
        values.append(1.0 if dg == dp else 0.5 if dp == 0 else 0.0)
    return statistics.fmean(values) if values else None


def trimmed_mean_five(values: list[float]) -> float:
    if len(values) != 5:
        raise ValueError(f"expected exactly five scores, got {len(values)}")
    return statistics.fmean(sorted(values)[1:4])


def aggregate_self_scores(
    values: list[float], *, planned_repeats: int = 3
) -> tuple[float | None, str]:
    """Aggregate successful sessions while retaining any partial evidence."""
    clean = [float(value) for value in values]
    if not clean:
        return None, "all_failed"
    if planned_repeats == 3:
        return statistics.median(clean), "median_3" if len(clean) == 3 else "median_available"
    if planned_repeats == 5 and len(clean) == 5:
        return trimmed_mean_five(clean), "trimmed_mean_5"
    if planned_repeats != 5:
        raise ValueError(f"unsupported planned repeat count: {planned_repeats}")
    return statistics.fmean(clean), "mean_available"


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row for row in rows
        if row.get("gt_reward") is not None and len(row.get("self_scores") or []) >= 1
    ]
    ids = [int(row["candidate_id"]) for row in usable]
    gt = [float(row["gt_reward"]) for row in usable]
    pred = [
        aggregate_self_scores(
            [float(x) for x in row["self_scores"]],
            planned_repeats=int(row.get("planned_repeats") or 3),
        )[0]
        for row in usable
    ]
    pred = [float(value) for value in pred if value is not None]
    result: dict[str, Any] = {
        "candidate_ids": ids,
        "planned_candidates": 7,
        "comparable_candidates": len(usable),
        "coverage": len(usable) / 7.0,
        "spearman": spearman(gt, pred) if len(usable) >= 2 else None,
        "kendall_tau_b": kendall_tau_b(gt, pred) if len(usable) >= 2 else None,
        "pairwise_accuracy": pairwise_accuracy(gt, pred) if len(usable) >= 2 else None,
    }
    if usable:
        gt_best = max(gt)
        pred_best = max(range(len(pred)), key=pred.__getitem__)
        result["top1_agreement"] = float(gt[pred_best] == gt_best)
        result["top1_regret"] = gt_best - gt[pred_best]
        result["mean_repeat_std"] = statistics.fmean(statistics.pstdev(row["self_scores"]) for row in usable)
        result["mean_repeat_range"] = statistics.fmean(max(row["self_scores"]) - min(row["self_scores"]) for row in usable)
        planned_repeats = int(usable[0].get("planned_repeats") or 3)
        result["planned_repeats"] = planned_repeats
        if all(
            int(row.get("planned_repeats") or 3) == planned_repeats
            and len(row["self_scores"]) == planned_repeats
            for row in usable
        ):
            winners = [
                max(range(len(usable)), key=lambda i: usable[i]["self_scores"][repeat])
                for repeat in range(planned_repeats)
            ]
            consensus_winner = max(range(len(pred)), key=pred.__getitem__)
            result["top1_repeat_agreement"] = (
                sum(index == consensus_winner for index in winners) / planned_repeats
            )
            result["repeat_to_consensus_spearman"] = [
                spearman([row["self_scores"][repeat] for row in usable], pred)
                for repeat in range(planned_repeats)
            ]
        else:
            result["top1_repeat_agreement"] = None
            result["repeat_to_consensus_spearman"] = []
    return result
