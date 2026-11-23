"""Adapters from Slime rollout batches to NeMo-RL AutoModel batches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def attach_old_policy_logprobs(
    batch: Any,
    previous_logprobs: torch.Tensor,
    *,
    has_rollout_log_probs: bool,
) -> None:
    """Attach actor-old logprobs without discarding rollout diagnostics.

    REINFORCE++ uses actor-recomputed logprobs as the detached old policy when
    ``--use-rollout-logprobs`` is disabled.  SGLang's generation logprobs are
    still valuable for the train/rollout mismatch diagnostic and must remain
    intact when the rollout supplied them.  Only backfill that field when the
    rollout did not provide logprobs at all.
    """

    batch["prev_logprobs"] = previous_logprobs
    if not has_rollout_log_probs:
        batch["generation_logprobs"] = previous_logprobs.clone()


def _response_start(total_length: int, response_length: int) -> int:
    start = total_length - response_length
    if start < 0:
        raise ValueError(
            f"response_length ({response_length}) exceeds total_length ({total_length})"
        )
    return start


def build_automodel_batch(
    rollout_data: dict[str, Any],
    *,
    pad_token_id: int,
    max_sequence_length: int,
    sequence_length_multiple: int = 1,
) -> dict[str, torch.Tensor]:
    """Right-pad one Slime DP shard and preserve response-token alignment.

    Slime stores loss masks and rollout log-probabilities only for generated
    tokens. NeMo-RL uses full-sequence tensors whose value at position ``i``
    corresponds to input token ``i``. This function places every response
    field back at the exact suffix occupied by the original generated tokens;
    it never decodes or reapplies a chat template.
    """

    tokens = rollout_data["tokens"]
    total_lengths = [int(x) for x in rollout_data["total_lengths"]]
    response_lengths = [int(x) for x in rollout_data["response_lengths"]]
    loss_masks = rollout_data["loss_masks"]
    rewards = rollout_data["rewards"]
    truncated = rollout_data.get("truncated", [0] * len(tokens))
    rollout_mask_sums = rollout_data.get(
        "rollout_mask_sums",
        [float(torch.as_tensor(mask).sum().item()) for mask in loss_masks],
    )

    if not tokens:
        raise ValueError("AutoModel received an empty Slime DP shard")
    if not (
        len(tokens)
        == len(total_lengths)
        == len(response_lengths)
        == len(loss_masks)
        == len(rewards)
        == len(truncated)
        == len(rollout_mask_sums)
    ):
        raise ValueError("Slime rollout fields have inconsistent sample counts")

    max_length = max(total_lengths)
    if max_length > max_sequence_length:
        raise ValueError(
            f"trajectory length {max_length} exceeds AutoModel max_total_sequence_length "
            f"{max_sequence_length}"
        )
    if sequence_length_multiple < 1:
        raise ValueError("sequence_length_multiple must be positive")
    max_length = (
        (max_length + sequence_length_multiple - 1)
        // sequence_length_multiple
        * sequence_length_multiple
    )
    if max_length > max_sequence_length:
        raise ValueError(
            f"trajectory padding to multiple {sequence_length_multiple} requires "
            f"length {max_length}, exceeding max_total_sequence_length "
            f"{max_sequence_length}"
        )

    batch_size = len(tokens)
    input_ids = torch.full(
        (batch_size, max_length), int(pad_token_id), dtype=torch.long
    )
    token_mask = torch.zeros((batch_size, max_length), dtype=torch.float32)
    advantages = torch.zeros_like(token_mask)
    generation_logprobs = torch.zeros_like(token_mask)
    reference_policy_logprobs = torch.zeros_like(token_mask)

    rollout_logprobs = rollout_data.get("rollout_log_probs")
    if rollout_logprobs is not None and len(rollout_logprobs) != batch_size:
        raise ValueError("rollout_log_probs has an inconsistent sample count")

    for i, (token_row, total_length, response_length, loss_mask, reward) in enumerate(
        zip(
            tokens,
            total_lengths,
            response_lengths,
            loss_masks,
            rewards,
            strict=True,
        )
    ):
        token_row = torch.as_tensor(token_row, dtype=torch.long).reshape(-1)
        if token_row.numel() != total_length:
            raise ValueError(
                f"sample {i}: token count {token_row.numel()} != total_length {total_length}"
            )
        loss_mask = torch.as_tensor(loss_mask, dtype=torch.float32).reshape(-1)
        if loss_mask.numel() != response_length:
            raise ValueError(
                f"sample {i}: loss mask length {loss_mask.numel()} != response_length "
                f"{response_length}"
            )

        response_start = _response_start(total_length, response_length)
        response_slice = slice(response_start, total_length)
        input_ids[i, :total_length] = token_row
        token_mask[i, response_slice] = loss_mask
        advantages[i, response_slice] = float(reward)

        if rollout_logprobs is not None:
            row_logprobs = torch.as_tensor(
                rollout_logprobs[i], dtype=torch.float32
            ).reshape(-1)
            if row_logprobs.numel() != response_length:
                raise ValueError(
                    f"sample {i}: rollout logprob length {row_logprobs.numel()} != "
                    f"response_length {response_length}"
                )
            generation_logprobs[i, response_slice] = row_logprobs

    sample_mask = torch.ones(batch_size, dtype=torch.float32)
    if "dp_padding_mask" in rollout_data:
        padding_mask = torch.as_tensor(
            rollout_data["dp_padding_mask"], dtype=torch.bool
        )
        if padding_mask.numel() != batch_size:
            raise ValueError("dp_padding_mask has an inconsistent sample count")
        sample_mask.masked_fill_(padding_mask, 0.0)

    return {
        "input_ids": input_ids,
        "input_lengths": torch.tensor(total_lengths, dtype=torch.long),
        "advantages": advantages,
        "prev_logprobs": generation_logprobs.clone(),
        "generation_logprobs": generation_logprobs,
        "reference_policy_logprobs": reference_policy_logprobs,
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        # Preserve the fields needed by Slime's native actor-rollout logger.
        # They remain ordinary per-sample tensors through NeMo's global-batch
        # reconstruction and DP sharder, so the NeMo backend can invoke the
        # exact same metric reducer as the non-NeMo Megatron path.
        "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
        "rollout_mask_sums": torch.tensor(
            rollout_mask_sums, dtype=torch.float32
        ),
        "truncated": torch.tensor(truncated, dtype=torch.float32),
    }


def pad_local_batch(
    batch: dict[str, torch.Tensor],
    *,
    target_batch_size: int,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad uneven Slime DP shards with zero-loss rows for FSDP2 global batches."""

    current = int(batch["input_ids"].shape[0])
    if current > target_batch_size:
        raise ValueError(f"cannot pad batch {current} down to {target_batch_size}")
    if current == target_batch_size:
        return batch

    count = target_batch_size - current
    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        shape = (count, *value.shape[1:])
        fill_value = int(pad_token_id) if key == "input_ids" else 0
        padding = torch.full(shape, fill_value, dtype=value.dtype)
        result[key] = torch.cat((value, padding), dim=0)

    # A one-token dummy row avoids zero-length tokenizer/model edge cases. Its
    # sample mask and token mask remain zero, so it contributes no loss.
    result["input_lengths"][current:] = 1
    return result


def concatenate_automodel_batches(
    batches: list[dict[str, torch.Tensor]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Concatenate Slime DP shards into NeMo's logical global batch."""

    if not batches:
        raise ValueError("cannot concatenate an empty AutoModel batch list")
    keys = tuple(batches[0])
    if any(tuple(batch) != keys for batch in batches[1:]):
        raise ValueError("AutoModel DP shards have inconsistent fields")

    max_sequence_length = max(int(batch["input_ids"].shape[1]) for batch in batches)
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        padded = []
        for batch in batches:
            value = batch[key]
            if value.ndim >= 2 and value.shape[1] < max_sequence_length:
                fill_value = int(pad_token_id) if key == "input_ids" else 0
                value = F.pad(
                    value,
                    (0, max_sequence_length - value.shape[1]),
                    value=fill_value,
                )
            padded.append(value)
        result[key] = torch.cat(padded, dim=0)
    return result
