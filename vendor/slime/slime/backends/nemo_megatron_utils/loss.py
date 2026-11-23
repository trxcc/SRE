"""NeMo adapter configuration for reference KL in the native training loss.

This candidate is not deployed. Reward-side KL remains unsupported here;
Reinforce++'s existing reward normalization is left unchanged.
"""

import math


def build_loss(args):
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn

    if args.kl_coef != 0:
        raise ValueError("NeMo adapter requires reward KL coef=0; use --use-kl-loss --kl-loss-coef for reference regularization")
    coefficient = float(args.kl_loss_coef)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Reference KL loss coefficient must be finite and nonnegative")
    if coefficient and not args.use_kl_loss:
        raise ValueError("A nonzero KL loss coefficient requires --use-kl-loss")
    if coefficient and not getattr(args, "ref_load", None):
        raise ValueError("Reference KL requires an explicit --ref-load checkpoint")
    if args.advantage_estimator not in {"grpo", "gspo", "cispo", "policy_gradient"}:
        raise NotImplementedError(f"Unsupported advantage estimator: {args.advantage_estimator}")

    use_tis = bool(args.use_tis)
    cfg = ClippedPGLossConfig(
        ratio_clip_min=float(args.eps_clip),
        ratio_clip_max=float(args.eps_clip_high or args.eps_clip),
        ratio_clip_c=args.eps_clip_c,
        reference_policy_kl_penalty=coefficient if args.use_kl_loss else 0.0,
        reference_policy_kl_type="k3" if args.kl_loss_type == "low_var_kl" else args.kl_loss_type,
        use_kl_in_reward=False,
        use_importance_sampling_correction=use_tis,
        truncated_importance_sampling_type="tis" if use_tis else None,
        truncated_importance_sampling_ratio=float(args.tis_clip) if use_tis else None,
        truncated_importance_sampling_ratio_min=float(args.tis_clip_low) if use_tis else None,
        token_level_loss=True,
        force_on_policy_ratio=False,
        use_cispo=args.advantage_estimator == "cispo",
        sequence_level_importance_ratios=args.advantage_estimator == "gspo",
    )
    loss = ClippedPGLossFn(cfg)
    if loss.reference_policy_kl_penalty != coefficient:
        raise RuntimeError("Native reference KL coefficient differs from the requested coefficient")
    return loss
