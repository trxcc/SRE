from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .common import read_json


BINARY_SCORES = (0.0, 1.0)


class ContractError(ValueError):
    pass


def validate_rubric(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("rubric must be a JSON object")
    if value.get("schema_version") != "self_reward_rubric_v3":
        raise ContractError("schema_version must be self_reward_rubric_v3")
    criteria = value.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ContractError("criteria must be a non-empty list")
    seen: set[str] = set()
    total_weight = 0.0
    for index, item in enumerate(criteria, 1):
        if not isinstance(item, dict):
            raise ContractError(f"criterion {index} must be an object")
        required = {"id", "criterion", "expected", "target_paths", "reason", "weight"}
        missing = required.difference(item)
        if missing:
            raise ContractError(f"criterion {index} is missing keys {sorted(missing)}")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ContractError(f"criterion {index}.id must be a non-empty string")
        if item_id in seen:
            raise ContractError(f"criterion id {item_id!r} is duplicated")
        seen.add(item_id)
        for key in ("criterion", "expected", "reason"):
            text = item.get(key)
            if not isinstance(text, str) or not text.strip():
                raise ContractError(f"{item_id}.{key} must be a non-empty string")
        paths = item.get("target_paths")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ContractError(f"{item_id}.target_paths must be a list of strings")
        weight = item.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ContractError(f"{item_id}.weight must be a finite non-negative number")
        total_weight += float(weight)
    if total_weight <= 0.0:
        raise ContractError("rubric must have positive total weight")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
        raise ContractError("limitations must be a list of strings")
    return value


def normalize_equal_weight_rubric(value: Any) -> tuple[dict[str, Any], list[str]]:
    """Repair only omitted constant weights; never infer semantic content."""
    from copy import deepcopy
    repaired = deepcopy(value)
    changes = []
    if isinstance(repaired, dict) and isinstance(repaired.get("criteria"), list):
        for index, criterion in enumerate(repaired["criteria"]):
            if isinstance(criterion, dict) and "weight" not in criterion:
                criterion["weight"] = 1.0
                changes.append(f"criteria[{index}].weight")
    return validate_rubric(repaired), changes


def load_rubric(path: Path) -> dict[str, Any]:
    return validate_rubric(read_json(path))


def weighted_score(rubric: dict[str, Any], scores: dict[str, Any], *, binary: bool) -> float:
    total = 0.0
    total_weight = 0.0
    expected = {item["id"] for item in rubric["criteria"]}
    if set(scores) != expected:
        raise ContractError(f"score keys differ: expected {sorted(expected)}, got {sorted(scores)}")
    for item in rubric["criteria"]:
        raw = scores[item["id"]]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ContractError(f"{item['id']} score is not numeric")
        value = float(raw)
        if binary:
            if not any(abs(value - allowed) < 1e-9 for allowed in BINARY_SCORES):
                raise ContractError(f"{item['id']} score {value} is not one of {BINARY_SCORES}")
        elif not 0.0 <= value <= 1.0:
            raise ContractError(f"{item['id']} score {value} is outside [0, 1]")
        weight = float(item["weight"])
        total += weight * value
        total_weight += weight
    return total / total_weight


def canonicalize_native_score(dataset: str, native: dict[str, Any], rubric: dict[str, Any], slot_id: str) -> dict[str, Any]:
    if dataset == "wsclaw":
        raw = native.get("rubric_scores")
        if not isinstance(raw, dict):
            raise ContractError("WSClaw result missing rubric_scores")
        scores = {item["id"]: raw.get(str(i)) for i, item in enumerate(rubric["criteria"], 1)}
        notes = str(native.get("feedback") or "")
    elif dataset == "researchclaw":
        raw = native.get("scores")
        if not isinstance(raw, dict):
            raise ContractError("ResearchClaw result missing scores")
        scores = {item["id"]: raw.get(f"criterion_{i}") for i, item in enumerate(rubric["criteria"], 1)}
        notes = str(native.get("notes") or "")
    else:
        raise ContractError(f"unsupported dataset: {dataset}")
    score = weighted_score(rubric, scores, binary=dataset == "researchclaw")
    return {
        "schema_version": "self_reward_score_v1",
        "slot_id": slot_id,
        "criterion_scores": {key: float(value) for key, value in scores.items()},
        "weighted_score": score,
        "notes": notes,
    }
