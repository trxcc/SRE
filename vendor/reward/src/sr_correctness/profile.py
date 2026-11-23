from __future__ import annotations

import os
from typing import Any


COMMON = {
    "ROLLOUT_TEMPERATURE": "0.7",
    "ROLLOUT_TOP_P": "1.0",
    "ROLLOUT_TOP_K": "-1",
    "NANOBOT_MAX_TOOL_ITERATIONS": "200",
    "CLAWGYM_NANOBOT_REASONING_CONTEXT_POLICY": "preserve",
    "NANOBOT_STRIP_REASONING_CONTEXT": "0",
    "NANOBOT_ENABLE_THINKING": "1",
    "NANOBOT_DISABLE_TOOL_LOOP_FINALIZATION": "0",
    "CLAWGYM_NANOBOT_TRAIN_SAMPLE_MODE": "token_prefix",
    "USE_ROLLOUT_LOGPROBS": "0",
    "CLAWGYM_TASK_PROMPT_MODE": "original",
    "CLAWGYM_POLAR_PROMPT_INPUT_MODE": "raw",
    "OPENCLAW_TRAIN_MAX_TURNS": "1",
}


def expected_profile(dataset: str, model: str) -> dict[str, str]:
    values = dict(COMMON)
    if dataset == "wsclaw":
        values.update(
            {
                "OPENCLAW_CONTEXT_WINDOW": "65536",
                "OPENCLAW_MAX_TOKENS": "8192",
                "CLAWGYM_ALLOW_MULTI_TOOL_CALLS": "0",
                "NANOBOT_MICROCOMPACT_ENABLED": "1",
                "NANOBOT_TASK_TIMEOUT": "1800" if model == "qwen3.5-9b" else "1500",
            }
        )
    elif dataset == "researchclaw":
        values.update(
            {
                "OPENCLAW_CONTEXT_WINDOW": "73728",
                "OPENCLAW_MAX_TOKENS": "16384",
                "CLAWGYM_ALLOW_MULTI_TOOL_CALLS": "1",
                "NANOBOT_MICROCOMPACT_ENABLED": "0",
                "NANOBOT_TASK_TIMEOUT": "1800" if model == "qwen3.5-9b" else "1200",
                "OPENCLAW_TRAIN_ORDER": "sequential",
            }
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return values


def validate_profile(dataset: str, model: str, sampling: dict[str, Any]) -> dict[str, Any]:
    expected = expected_profile(dataset, model)
    mismatches = {
        name: {"expected": value, "actual": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    expected_sampling = {"temperature": 0.7, "top_p": 1.0, "top_k": -1}
    for name, value in expected_sampling.items():
        if float(sampling.get(name)) != float(value):
            mismatches[f"sampling.{name}"] = {"expected": value, "actual": sampling.get(name)}
    max_expected = 8192 if dataset == "wsclaw" else 16384
    if int(sampling.get("max_new_tokens")) != max_expected:
        mismatches["sampling.max_new_tokens"] = {
            "expected": max_expected,
            "actual": sampling.get("max_new_tokens"),
        }
    if mismatches:
        raise ValueError(f"rollout profile differs from successful training profile: {mismatches}")
    return {"dataset": dataset, "model": model, "validated": expected, "sampling": sampling}
