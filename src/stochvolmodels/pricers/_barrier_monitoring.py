"""Within-step any-name lower-barrier probabilities under a frozen log diffusion."""

import numpy as np
from numba import njit


@njit
def _bridge(levels, cumulative, indices, barriers, eligible):
    result = np.zeros((levels.shape[0], 3))
    for path in range(len(levels)):
        surv_lo, surv_mid, surv_hi = 1.0, 1.0, 1.0
        for t in range(len(eligible)):
            max_hit, sum_hit, independent = 0.0, 0.0, 1.0
            for j, idx in enumerate(indices):
                if not eligible[t, j]:
                    continue
                left = levels[path, t, idx]
                right = levels[path, t + 1, idx]
                var = cumulative[path, t + 1, idx] - cumulative[path, t, idx]
                if min(left, right) <= barriers[j]:
                    max_hit, sum_hit, independent = 1.0, 1.0, 0.0
                    break
                if var > 0.0:
                    exponent = -2.0 * np.log(left / barriers[j]) * np.log(right / barriers[j]) / var
                    p = np.exp(exponent)
                    max_hit = max(max_hit, p)
                    sum_hit += p
                    independent *= -np.expm1(exponent)
            surv_lo *= max(0.0, 1.0 - sum_hit)
            surv_mid *= independent
            surv_hi *= 1.0 - max_hit
            if surv_hi == 0.0:
                break
        result[path, 0] = 1.0 - surv_hi
        result[path, 1] = 1.0 - surv_mid
        result[path, 2] = 1.0 - surv_lo
    return result


def bridge_inputs(paths):
    """Validate the scheme and numeric state once per path block."""
    if (
        paths.sampling_measure != "Q_MMA"
        or paths.target_measure != "Q_MMA"
        or paths.log_likelihood_ratios is not None
    ):
        raise ValueError("ordinary money-market diffusion paths required")
    if paths.scheme == "lsv_log_euler":
        if not paths.provenance.get("frozen_log_diffusion", False):
            raise ValueError("LSV bridge requires an explicitly verified frozen diffusion")
        if any(k != 1 for k in paths.provenance["interval_step_counts"]):
            raise ValueError("observation grid hides internal diffusion steps")
        if np.any(paths.states["jump_count"] != 0):
            raise ValueError("jump paths require a different monitor")
    elif paths.scheme != "basket_frozen_log_euler":
        raise ValueError("unsupported bridge scheme")
    if paths.state_units.get("quadratic_variance") != "integrated variance":
        raise ValueError("expected quadratic_variance in integrated variance units")
    var = paths.states["quadratic_variance"]
    if var.ndim == 2 and len(paths.asset_ids) == 1:
        var = var[:, :, None]
    if (
        var.shape != paths.assets.shape
        or not np.isfinite(var).all()
        or np.any(var < 0)
        or np.any(np.diff(var, axis=1) < 0)
        or not np.isfinite(paths.assets).all()
        or np.any(paths.assets <= 0)
    ):
        raise ValueError("invalid diffusion levels or cumulative variance")
    return paths.assets, var


def barrier_hit_weights(levels, cumulative, indices, barriers, eligible):
    """Conditional lower, independent and upper hit probabilities per path."""
    indices = np.asarray(indices, dtype=int)
    barriers = np.asarray(barriers, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if (
        levels.ndim != 3
        or cumulative.shape != levels.shape
        or indices.ndim != 1
        or barriers.shape != indices.shape
        or eligible.ndim != 2
        or eligible.shape[1] != len(indices)
        or len(eligible) >= levels.shape[1]
        or np.any(indices < 0)
        or np.any(indices >= levels.shape[2])
        or not np.all(np.isfinite(barriers) & (barriers > 0))
    ):
        raise ValueError("barriers and eligibility must match the selected assets and intervals")
    return _bridge(levels, cumulative, indices, barriers, eligible)
