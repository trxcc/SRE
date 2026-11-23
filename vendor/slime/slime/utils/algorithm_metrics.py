from __future__ import annotations


def derive_algorithm_metrics(reduced: dict[str, float]) -> dict[str, float]:
    """Turn globally reduced token moments into user-facing RL metrics."""
    out = {key: value for key, value in reduced.items() if not key.startswith("__moments/")}
    moment_names = {key.split("/")[1] for key in reduced if key.startswith("__moments/") and len(key.split("/")) >= 3}
    for name in sorted(moment_names):
        prefix = f"__moments/{name}/"
        mean = reduced.get(f"{prefix}mean")
        second = reduced.get(f"{prefix}second")
        finite_fraction = reduced.get(f"{prefix}finite_fraction")
        output_prefix = f"algorithm/{name}"
        if finite_fraction is not None:
            out[f"{output_prefix}/finite_fraction"] = finite_fraction
        if mean is None or second is None:
            continue
        variance = max(0.0, second - mean * mean)
        out[f"{output_prefix}/mean"] = mean
        out[f"{output_prefix}/std"] = variance**0.5

    return_std = out.get("algorithm/return/std")
    residual_std = out.get("algorithm/value_residual/std")
    if return_std is not None and residual_std is not None:
        return_variance = return_std * return_std
        residual_variance = residual_std * residual_std
        return_finite = out.get("algorithm/return/finite_fraction", 1.0)
        residual_finite = out.get("algorithm/value_residual/finite_fraction", 1.0)
        valid = return_variance > 1e-12 and return_finite == 1.0 and residual_finite == 1.0
        out["critic/explained_variance_valid"] = float(valid)
        if valid:
            out["critic/explained_variance"] = 1.0 - residual_variance / return_variance
        out["critic/return_variance"] = return_variance
        out["critic/residual_variance"] = residual_variance

    tis_mean = out.get("algorithm/tis_importance_weight/mean")
    tis_std = out.get("algorithm/tis_importance_weight/std")
    if tis_mean is not None and tis_std is not None:
        tis_finite = out.get("algorithm/tis_importance_weight/finite_fraction", 1.0)
        second = tis_std * tis_std + tis_mean * tis_mean
        valid = tis_finite == 1.0 and second > 0
        out["algorithm/tis_effective_sample_size_valid"] = float(valid)
        if valid:
            out["algorithm/tis_effective_sample_size_fraction"] = tis_mean * tis_mean / second
    return out
