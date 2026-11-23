"""Expose exact SGLang token IDs to Polar's OpenAI-compatible gateway.

SGLang computes exact prompt/output token IDs for chat completions but its
OpenAI response schema only emits token text and logprobs. Polar needs the IDs
sampled by the server so stale rollouts can train against their behavior-policy
logprobs without retokenization drift. This opt-in patch adds those IDs to the
non-streaming response while leaving the standard fields unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _output_token_ids(ret_item: dict[str, Any]) -> list[int]:
    meta = ret_item.get("meta_info")
    if not isinstance(meta, dict):
        return []
    output_logprobs = meta.get("output_token_logprobs")
    if not isinstance(output_logprobs, list):
        return []

    token_ids: list[int] = []
    for item in output_logprobs:
        token_id = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            token_id = item[1]
        elif isinstance(item, dict):
            token_id = item.get("token_id")
        if token_id is None:
            return []
        try:
            token_ids.append(int(token_id))
        except (TypeError, ValueError):
            return []
    return token_ids


def enrich_training_response(
    payload: dict[str, Any],
    *,
    prompt_ids: list[int],
    generation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach exact token IDs and align each logprob entry with its token ID."""
    enriched = deepcopy(payload)
    choices = enriched.get("choices")
    if not isinstance(choices, list):
        return enriched

    for choice, ret_item in zip(choices, generation_results):
        if not isinstance(choice, dict) or not isinstance(ret_item, dict):
            continue
        response_ids = _output_token_ids(ret_item)
        if not response_ids:
            continue
        choice["input_token_ids"] = list(prompt_ids)
        choice["token_ids"] = response_ids

        logprobs = choice.get("logprobs")
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if isinstance(content, list) and len(content) == len(response_ids):
            for entry, token_id in zip(content, response_ids):
                if isinstance(entry, dict):
                    entry["token_id"] = token_id
    return enriched


def install_patch() -> None:
    """Patch SGLang's non-streaming chat response builder once per process."""
    from fastapi.responses import ORJSONResponse
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    if getattr(OpenAIServingChat, "_polar_training_tokens_patch", False):
        return

    original = OpenAIServingChat._build_chat_response

    def _build_chat_response(self, request, ret, created):
        response = original(self, request, ret, created)
        if isinstance(response, ORJSONResponse):
            return response

        is_multimodal = self.tokenizer_manager.model_config.is_multimodal
        processed = self._process_messages(request, is_multimodal)
        prompt_ids = processed.prompt_ids
        if not isinstance(prompt_ids, list) or not all(
            isinstance(token_id, int) for token_id in prompt_ids
        ):
            return response

        payload = response.model_dump(mode="json")
        payload = enrich_training_response(
            payload,
            prompt_ids=prompt_ids,
            generation_results=ret,
        )
        return ORJSONResponse(content=payload)

    OpenAIServingChat._build_chat_response = _build_chat_response
    OpenAIServingChat._polar_training_tokens_patch = True


__all__ = ["enrich_training_response", "install_patch"]
