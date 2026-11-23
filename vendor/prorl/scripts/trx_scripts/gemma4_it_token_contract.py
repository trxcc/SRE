#!/usr/bin/env python3
"""Validate the Gemma 4 IT chat/termination contract without loading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


EXPECTED_TOKEN_IDS = {
    "<eos>": 1,
    "<|tool_response>": 50,
    "<|turn>": 105,
    "<turn|>": 106,
}


def _token_ids(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def validate(model_path: Path, expected_stop_token_ids: list[int]) -> dict[str, object]:
    generation_config = json.loads((model_path / "generation_config.json").read_text())
    configured = generation_config.get("eos_token_id")
    configured = configured if isinstance(configured, list) else [configured]
    if configured != expected_stop_token_ids:
        raise ValueError(
            "Gemma 4 IT generation_config EOS mismatch: "
            f"configured={configured}, expected={expected_stop_token_ids}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actual_token_ids = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in EXPECTED_TOKEN_IDS
    }
    if actual_token_ids != EXPECTED_TOKEN_IDS:
        raise ValueError(
            f"Gemma 4 IT special-token mismatch: actual={actual_token_ids}, "
            f"expected={EXPECTED_TOKEN_IDS}"
        )

    closed_ids = _token_ids(
        tokenizer,
        [
            {"role": "user", "content": "contract probe"},
            {"role": "assistant", "content": "contract response"},
        ],
        add_generation_prompt=False,
    )
    open_ids = _token_ids(
        tokenizer,
        [{"role": "user", "content": "contract probe"}],
        add_generation_prompt=True,
    )

    turn_start = EXPECTED_TOKEN_IDS["<|turn>"]
    turn_end = EXPECTED_TOKEN_IDS["<turn|>"]
    if not closed_ids or closed_ids[0] != int(tokenizer.bos_token_id):
        raise ValueError("Gemma 4 IT closed chat is missing BOS")
    last_turn_end = len(closed_ids) - 1 - closed_ids[::-1].index(turn_end) if turn_end in closed_ids else -1
    if last_turn_end < 0 or tokenizer.decode(closed_ids[last_turn_end + 1 :]).strip():
        raise ValueError(f"Gemma 4 IT assistant turn does not end in <turn|>: tail={closed_ids[-8:]}")
    if closed_ids.count(turn_start) != closed_ids.count(turn_end):
        raise ValueError("Gemma 4 IT closed chat has unbalanced turn tags")
    if open_ids.count(turn_start) != open_ids.count(turn_end) + 1:
        raise ValueError("Gemma 4 IT generation prompt must leave exactly one model turn open")
    if open_ids[-1] == turn_end:
        raise ValueError("Gemma 4 IT generation prompt was unexpectedly closed")

    thinking_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "thinking contract probe"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not thinking_prompt.endswith("<|channel>thought\n"):
        raise ValueError(
            "Gemma 4 E IT thinking prompt must enter the thought channel; "
            f"tail={thinking_prompt[-80:]!r}"
        )

    historical_reasoning = "GEMMA4_TOOL_REASONING_CONTRACT_MARKER"
    tool_history_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Inspect the workspace."},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": historical_reasoning,
                "tool_calls": [
                    {
                        "id": "contract_call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "contract.txt"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "contract_call_1",
                "name": "read_file",
                "content": "contract result",
            },
            {"role": "user", "content": "Now summarize."},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if historical_reasoning not in tool_history_prompt:
        raise ValueError(
            "Gemma 4 E IT chat template dropped tool-call reasoning after a "
            "later user turn"
        )

    template_path = model_path / "chat_template.jinja"
    template = template_path.read_text()
    for marker in ("<|turn>model", "<turn|>", "continue_same_model_turn", "continues_into_next"):
        if marker not in template:
            raise ValueError(f"Gemma 4 IT canonical chat template marker missing: {marker}")

    return {
        "model_path": str(model_path),
        "stop_token_ids": configured,
        "token_ids": actual_token_ids,
        "closed_turn_starts": closed_ids.count(turn_start),
        "closed_turn_ends": closed_ids.count(turn_end),
        "generation_turn_starts": open_ids.count(turn_start),
        "generation_turn_ends": open_ids.count(turn_end),
        "thinking_generation_prefix": True,
        "tool_reasoning_history_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--expected-stop-token-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    summary = validate(args.model_path, args.expected_stop_token_ids)
    print("GEMMA4_IT_TOKEN_CONTRACT_OK " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
