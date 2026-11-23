from argparse import Namespace


def add_rl_training_metrics(log_dict: dict[str, float], args: Namespace, role_tag: str = "") -> None:
    """Add unambiguous RL diagnostics while preserving legacy metric names."""
    prefix = f"train/{role_tag}"

    aliases = {
        "entropy": "entropy_loss",
        "old_policy_kl": "ppo_kl",
        "old_policy_kl_abs": "ppo_kl_abs",
        "importance_ratio": "policy_ratio",
        "importance_ratio_abs_deviation": "policy_ratio_abs_deviation",
        "reference_kl": "kl_loss",
        "rollout_train_logprob_abs_diff": "train_rollout_logprob_abs_diff",
    }
    for alias, source in aliases.items():
        source_key = f"{prefix}{source}"
        if source_key in log_dict:
            log_dict[f"{prefix}{alias}"] = log_dict[source_key]

    entropy_key = f"{prefix}entropy"
    if entropy_key in log_dict:
        entropy_coef = float(getattr(args, "entropy_coef", 0.0))
        log_dict[f"{prefix}entropy_coef"] = entropy_coef
        log_dict[f"{prefix}entropy_loss_contribution"] = -entropy_coef * log_dict[entropy_key]

    reference_kl_key = f"{prefix}reference_kl"
    if reference_kl_key in log_dict:
        kl_loss_coef = float(getattr(args, "kl_loss_coef", 0.0))
        log_dict[f"{prefix}reference_kl_coef"] = kl_loss_coef
        log_dict[f"{prefix}reference_kl_loss_contribution"] = kl_loss_coef * log_dict[reference_kl_key]
