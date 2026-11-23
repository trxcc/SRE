"""Thin metric glue between NeMo-RL's loss API and Slime logging.

NeMo-RL returns each data-parallel rank's microbatch contributions.  Slime's
native Megatron backend sums those contributions across DP before logging.  The
helpers here preserve that convention and reuse Slime's own PPO/KL formulas;
they do not implement a second training loss.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.rl_metric_utils import add_rl_training_metrics


SLIME_TRAIN_METRIC_KEYS = (
    "loss",
    "pg_loss",
    "entropy_loss",
    "pg_clipfrac",
    "ppo_kl",
    "ppo_kl_abs",
    "policy_ratio",
    "policy_ratio_abs_deviation",
    "policy_ratio_low_clipfrac",
    "policy_ratio_high_clipfrac",
    "train_rollout_logprob_abs_diff",
    "behavior_kl",
    "kl_loss",
)


def build_native_rollout_log_data(
    batch: Mapping[str, Any],
    *,
    raw_reward: list[Any] | None,
    global_batch_sizes: list[int],
    has_rollout_log_probs: bool,
    include_actor_log_probs: bool,
    include_reference_log_probs: bool,
) -> dict[str, Any]:
    """Rebuild Slime's response-space rollout batch from a NeMo batch.

    NeMo consumes full-sequence tensors, while Slime's native rollout logger
    consumes one response-length tensor per emitted training sample. Keeping
    this conversion explicit lets the NeMo path call the same advantage and
    metric reducers as native Megatron instead of maintaining a second set of
    approximately equivalent metric formulas.
    """

    total_lengths = [int(value) for value in batch["input_lengths"]]
    response_lengths = [int(value) for value in batch["response_lengths"]]
    if len(total_lengths) != len(response_lengths):
        raise ValueError("NeMo rollout log lengths have inconsistent counts")

    def response_rows(key: str) -> list[torch.Tensor]:
        values = batch[key]
        rows = []
        for row, total_length, response_length in zip(
            values, total_lengths, response_lengths, strict=True
        ):
            start = total_length - response_length
            if start < 0:
                raise ValueError(
                    f"response_length ({response_length}) exceeds "
                    f"total_length ({total_length})"
                )
            rows.append(row[start:total_length].detach().float())
        return rows

    rewards = [float(value) for value in batch["rewards"]]
    result: dict[str, Any] = {
        "response_lengths": response_lengths,
        "rewards": rewards,
        "raw_reward": (
            [float(value) for value in raw_reward]
            if raw_reward is not None
            else list(rewards)
        ),
        "truncated": [float(value) for value in batch["truncated"]],
        "loss_masks": response_rows("token_mask"),
        "total_lengths": total_lengths,
        "rollout_mask_sums": [
            value.detach().float() for value in batch["rollout_mask_sums"]
        ],
        "global_batch_sizes": [int(value) for value in global_batch_sizes],
        "dp_padding_mask": [
            not bool(value) for value in batch["sample_mask"]
        ],
    }
    if has_rollout_log_probs:
        result["rollout_log_probs"] = response_rows("generation_logprobs")
    if include_actor_log_probs:
        result["log_probs"] = response_rows("prev_logprobs")
    if include_reference_log_probs:
        result["ref_log_probs"] = response_rows(
            "reference_policy_logprobs"
        )
    return result


def _masked_global_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    global_valid_tokens: torch.Tensor | float,
) -> torch.Tensor:
    return (values * mask).sum() / (
        torch.as_tensor(
            global_valid_tokens,
            dtype=values.dtype,
            device=values.device,
        )
        + 1e-8
    )


@torch.no_grad()
def compute_slime_report_metrics(
    *,
    args: Namespace,
    current_logprobs: torch.Tensor,
    data: Mapping[str, Any],
    global_valid_tokens: torch.Tensor,
    approximate_entropy: float,
) -> dict[str, float]:
    """Compute Slime-native diagnostics from NeMo's sampled log-probabilities.

    Every returned scalar is one rank/microbatch's contribution divided by the
    global token count.  ``reduce_nemo_dp_metrics`` performs the missing DP sum.
    """

    token_mask = data["token_mask"][:, 1:].to(current_logprobs.dtype)
    sample_mask = data["sample_mask"].to(current_logprobs.dtype).unsqueeze(-1)
    mask = token_mask * sample_mask
    old_logprobs = data["prev_logprobs"][:, 1:].to(current_logprobs.dtype)
    rollout_logprobs = data["generation_logprobs"][:, 1:].to(
        current_logprobs.dtype
    )
    advantages = data["advantages"][:, 1:].to(current_logprobs.dtype)

    ppo_kl_values = old_logprobs - current_logprobs
    policy_ratio_values = torch.exp(-ppo_kl_values)
    pg_loss_values, pg_clipfrac_values = compute_policy_loss(
        ppo_kl_values,
        advantages,
        float(args.eps_clip),
        float(args.eps_clip_high or args.eps_clip),
        getattr(args, "eps_clip_c", None),
    )

    def mean(values: torch.Tensor) -> float:
        return float(
            _masked_global_mean(values, mask, global_valid_tokens).item()
        )

    metrics = {
        "pg_loss": mean(pg_loss_values),
        "entropy_loss": float(approximate_entropy),
        "pg_clipfrac": mean(pg_clipfrac_values),
        "ppo_kl": mean(ppo_kl_values),
        "ppo_kl_abs": mean(ppo_kl_values.abs()),
        "policy_ratio": mean(policy_ratio_values),
        "policy_ratio_abs_deviation": mean(
            (policy_ratio_values - 1.0).abs()
        ),
        "policy_ratio_low_clipfrac": mean(
            (policy_ratio_values < 1.0 - float(args.eps_clip)).float()
        ),
        "policy_ratio_high_clipfrac": mean(
            (
                policy_ratio_values
                > 1.0 + float(args.eps_clip_high or args.eps_clip)
            ).float()
        ),
    }

    logprobs_to_compare = (
        current_logprobs if args.use_rollout_logprobs else old_logprobs
    )
    metrics["train_rollout_logprob_abs_diff"] = mean(
        (logprobs_to_compare - rollout_logprobs).abs()
    )
    metrics["behavior_kl"] = mean(
        compute_approx_kl(
            current_logprobs,
            rollout_logprobs,
            kl_loss_type=args.kl_loss_type,
        )
    )

    if args.use_kl_loss:
        if "reference_policy_logprobs" not in data:
            raise RuntimeError(
                "--use-kl-loss requires reference_policy_logprobs even when "
                "--kl-loss-coef is zero"
            )
        reference_logprobs = data["reference_policy_logprobs"][:, 1:].to(
            current_logprobs.dtype
        )
        importance_ratio = None
        if getattr(args, "use_unbiased_kl", False):
            importance_ratio = torch.exp(current_logprobs - old_logprobs)
        metrics["kl_loss"] = mean(
            compute_approx_kl(
                current_logprobs,
                reference_logprobs,
                kl_loss_type=args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
        )

    return metrics


class SlimeAlignedClippedPGLoss:
    """Delegate the loss to NeMo and add detached Slime-compatible metrics."""

    def __init__(self, base_loss: Any, args: Namespace):
        self.base_loss = base_loss
        self.args = args
        self.input_type = base_loss.input_type
        self.loss_type = base_loss.loss_type
        self.use_fused_linear_logprobs = getattr(
            base_loss, "use_fused_linear_logprobs", False
        )
        self.metric_normalizations = dict(base_loss.metric_normalizations)
        gradient_normalizer = self.metric_normalizations["loss"]
        token_normalizer = self.metric_normalizations["approx_entropy"]
        for name in SLIME_TRAIN_METRIC_KEYS:
            self.metric_normalizations.setdefault(
                name,
                gradient_normalizer if name in {"loss", "pg_loss"} else token_normalizer,
            )

    def __call__(
        self,
        next_token_logprobs: torch.Tensor,
        data: Mapping[str, Any],
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        loss, metrics = self.base_loss(
            next_token_logprobs,
            data,
            global_valid_seqs,
            global_valid_toks,
        )
        metrics.update(
            compute_slime_report_metrics(
                args=self.args,
                current_logprobs=next_token_logprobs,
                data=data,
                global_valid_tokens=global_valid_toks,
                approximate_entropy=metrics["approx_entropy"],
            )
        )
        return loss, metrics


def reduce_nemo_dp_metrics(
    all_mb_metrics: Mapping[str, list[Any]],
    *,
    data_parallel_group: dist.ProcessGroup,
    device: torch.device | str | int,
) -> dict[str, float]:
    """Sum NeMo microbatch contributions across the full DP group."""

    keys = [key for key in SLIME_TRAIN_METRIC_KEYS if key in all_mb_metrics]
    local_values = [
        sum(
            float(value.item() if torch.is_tensor(value) else value)
            for value in all_mb_metrics[key]
        )
        for key in keys
    ]
    values = torch.tensor(local_values, dtype=torch.float64, device=device)
    if values.numel():
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=data_parallel_group)
    return dict(zip(keys, values.cpu().tolist(), strict=True))


def build_slime_train_log(
    reduced_metrics: Mapping[str, float],
    *,
    args: Namespace,
    grad_norm: float,
    learning_rate: float,
    global_batch_size: int,
    step: int,
) -> dict[str, float | int]:
    """Build the same primary W&B namespace as Slime's Megatron backend."""

    logged: dict[str, float | int] = {
        f"train/{key}": float(value) for key, value in reduced_metrics.items()
    }
    logged["train/grad_norm"] = float(grad_norm)
    logged["train/lr-pg_0"] = float(learning_rate)
    logged["train/global_batch_size"] = int(global_batch_size)
    add_rl_training_metrics(logged, args)
    logged["train/step"] = int(step)
    return logged
