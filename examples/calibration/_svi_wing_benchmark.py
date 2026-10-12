"""Opt-in finite-domain raw-SVI constraints; repository-only research code.

The density factor is Gatheral/Jacquier (2013), equation (2.1). The call slope
comes from differentiating their Black formula (section 2.2) in strike space.
Sampled constraints are not a global no-arbitrage or extrapolation certificate.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from _svi_benchmark import RawSvi, _scaled_svi_residual, durrleman, raw_svi


DENSITY_BUFFER = 1e-5
DENSITY_TOLERANCE = 1e-8
SLOPE_TOLERANCE = 1e-10
OPTIMIZATION_POINTS = 801
VALIDATION_POINTS = 4001
PRICE_SPREAD_FLOOR = 0.01
DOMAIN_ROUNDOFF = 1e-12


def svi_prediction_domain(k, bounds):
    """Describe requested log-moneyness without clipping or extending the domain."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1:
        raise ValueError("SVI prediction coordinates must be one dimensional")
    low, high = bounds
    outside = ~np.isfinite(k) | (k < low - DOMAIN_ROUNDOFF) | (k > high + DOMAIN_ROUNDOFF)
    return dict(k_domain=list(bounds), requested_k=k.tolist(),
                supported=bool(len(k) and not np.any(outside)),
                outside_quotes=int(outside.sum()), outside_indices=np.flatnonzero(outside).tolist(),
                boundary_roundoff=DOMAIN_ROUNDOFF)


def _sorted_price_quotes(quotes, order):
    """Validate original training premiums and reorder every quote field together."""
    arrays = {name: np.asarray(quotes.get(name)) for name in ("bid", "ask", "optiontypes")}
    scalars = [quotes.get(name) for name in ("forward", "discount")]
    if (any(not np.isscalar(x) or not np.isfinite(x) or x <= 0 for x in scalars)
            or any(a.shape != order.shape for a in arrays.values())
            or not np.all(np.isfinite(arrays["bid"]))
            or not np.all(np.isfinite(arrays["ask"]))
            or np.any(arrays["bid"] < 0) or np.any(arrays["ask"] < arrays["bid"])
            or not np.all(np.isin(arrays["optiontypes"], ["C", "P"]))):
        raise ValueError("Invalid training price quotes: require finite two-sided premiums and F/D")
    return dict(forward=float(scalars[0]), discount=float(scalars[1]),
                **{name: array[order].copy() for name, array in arrays.items()})


def _spread_price_objective(p, z, target, k, ttm, vol, quotes):
    """Mean squared half-spread price error and its Black-vega chain-rule gradient."""
    iv = vol * (_scaled_svi_residual(p, z, target) + target)
    root = iv * np.sqrt(ttm)
    d1, d2 = -k / root + root / 2, -k / root - root / 2
    forward, discount = quotes["forward"], quotes["discount"]
    strikes = forward * np.exp(k)
    calls = discount * (forward * ndtr(d1) - strikes * ndtr(d2))
    puts = discount * (strikes * ndtr(-d2) - forward * ndtr(-d1))
    prices = np.where(quotes["optiontypes"] == "C", calls, puts)
    half = np.maximum((quotes["ask"] - quotes["bid"]) / 2, PRICE_SPREAD_FLOOR)
    residual = (prices - (quotes["ask"] + quotes["bid"]) / 2) / half
    vega = discount * forward * np.exp(-d1*d1 / 2) / np.sqrt(2*np.pi) * np.sqrt(ttm)
    jac = _scaled_svi_residual(p, z, target, jacobian=True) * (vol * vega / half)[:, None]
    return float(np.mean(residual**2)), 2 * jac.T @ residual / len(k)


def svi_wing_values(k, params):
    """Return unclipped density factor and call slope divided by discount.

    With k=log(K/F), dC/dK / D = -N(d2) + n(d2) w'(k)/(2 sqrt(w)).
    Valid call slopes lie in [-1, 0]; g >= 0 is a separate sampled condition.
    """
    w, first, second = raw_svi(k, params)
    root = np.sqrt(w)
    d2 = -np.asarray(k) / root - root / 2
    slope = -ndtr(d2) + np.exp(-d2**2 / 2) / np.sqrt(2 * np.pi) * first / (2 * root)
    return durrleman(k, w, first, second), slope


def check_svi_wings(k, params):
    """Recompute finite-domain evidence without repairing any model values."""
    k = np.asarray(k, dtype=float)
    w = raw_svi(k, params)[0]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        g, slope = svi_wing_values(k, params)
    finite = bool(np.all(np.isfinite(np.r_[w, g, slope])) and np.all(w > 0))
    left, right = params.b * (1 - params.rho), params.b * (1 + params.rho)
    passed = bool(finite and np.min(g) >= -DENSITY_TOLERANCE
                  and np.min(slope) >= -1 - SLOPE_TOLERANCE
                  and np.max(slope) <= SLOPE_TOLERANCE
                  and 0 <= left < 2 and 0 <= right < 2)
    return dict(passed=passed, finite=finite, points=len(k),
                k_domain=[float(np.min(k)), float(np.max(k))],
                min_density_factor=float(np.min(g)),
                min_call_slope=float(np.min(slope)), max_call_slope=float(np.max(slope)),
                left_tail_slope=float(left), right_tail_slope=float(right),
                density_tolerance=DENSITY_TOLERANCE, slope_tolerance=SLOPE_TOLERANCE,
                scope="sampled strike domain; no calendar or global certificate")


def fit_raw_svi_constrained(k, iv, ttm, *, price_quotes=None, iv_seed=False):
    """Fit IV SSE or half-spread price MSE under finite-grid strike constraints.

    Parameters use the same positive-minimum and training-relative bounds as
    the bounded research candidate. Three independent SLSQP starts receive no
    held-out data. Unsuccessful or dense-check-infeasible solutions are rejected
    before objective selection. Optional price_quotes contains training bid, ask,
    optiontypes, forward and discount; IVs then supply scaling and starts only.
    With iv_seed=True, a constrained training-IV fit supplies a fourth price start;
    its cost and evidence are retained. It must converge again on the price loss.
    Exhaustion is a failed fit, not a flat fallback. This is a repository-only helper.
    """
    if iv_seed and price_quotes is None:
        raise ValueError("The training-IV seed requires price quotes")
    k, iv = np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    if (k.ndim != 1 or iv.shape != k.shape or len(k) < 5
            or not np.all(np.isfinite(k)) or len(np.unique(k)) < 5
            or not np.all(np.isfinite(iv) & (iv > 0))
            or not np.isscalar(ttm) or not np.isfinite(ttm) or ttm <= 0):
        raise ValueError("Raw SVI needs five distinct finite strikes, positive IVs and finite T")
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    if price_quotes is not None:
        price_quotes = _sorted_price_quotes(price_quotes, order)
    center, scale = (k[0] + k[-1]) / 2, (k[-1] - k[0]) / 2
    vol = float(np.mean(iv))
    variance = ttm * vol**2
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Nonfinite or underflowed training variance")
    z, target = (k - center) / scale, iv / vol
    lower = np.array([1e-10 / variance, 0, -0.999,
                      max(-2, (-1.5 - center) / scale), max(0.05, 1e-4 / scale)])
    upper = np.array([5 / variance, scale / variance, 0.999,
                      min(2, (1.5 - center) / scale), min(2, 3 / scale)])
    if np.any(lower >= upper) or not np.all(np.isfinite(np.r_[lower, upper])):
        raise ValueError("Training domain does not intersect the research parameter bounds")
    grid = np.unique(np.r_[k, center + scale * np.linspace(-2, 2, OPTIMIZATION_POINTS)])

    def unpack(p):
        minimum, b, rho, m, sigma = p
        return RawSvi(float(variance * (minimum - b * sigma * np.sqrt(1 - rho**2))),
                      float(b * variance / scale), float(rho), float(center + scale * m),
                      float(scale * sigma))

    def objective(p):
        if price_quotes is not None:
            value, gradient = _spread_price_objective(p, z, target, k, ttm, vol, price_quotes)
            return value / price_scale, gradient / price_scale
        residual = _scaled_svi_residual(p, z, target)
        jac = _scaled_svi_residual(p, z, target, jacobian=True)
        return float(residual @ residual), 2 * jac.T @ residual

    def constraints(p):
        g, slope = svi_wing_values(grid, unpack(p))
        return np.r_[g - DENSITY_BUFFER, 1 + slope, -slope]

    def validate(params):
        # A separate, denser grid and shape-local points can reject a result
        # missed by the optimizer grid; rejection never triggers a hidden refit.
        local = params.m + params.sigma * np.linspace(-8, 8, 161)
        local = local[(local >= grid[0]) & (local <= grid[-1])]
        dense = np.unique(np.r_[grid, local, np.linspace(grid[0], grid[-1], VALIDATION_POINTS)])
        return check_svi_wings(dense, params)

    # A common training-only multiplier preserves the minimizer while avoiding
    # huge gradients from index-point premiums divided by very narrow spreads.
    price_scale = 1.0
    if price_quotes is not None:
        flat = np.clip([1, 0, 0, 0, 1], lower, upper)
        price_scale = max(1.0, _spread_price_objective(
            flat, z, target, k, ttm, vol, price_quotes)[0])
        if not np.isfinite(price_scale):
            raise ValueError("Nonfinite training price objective scale")
    geometry = dict(k_center=float(center), k_scale=float(scale), iv_scale=vol,
                    scaled_lower=lower.tolist(), scaled_upper=upper.tolist(),
                    scaled_parameter_order=["minimum", "b", "rho", "m", "sigma"],
                    optimization_grid=grid.tolist(), density_buffer=DENSITY_BUFFER,
                    maxiter=1000, ftol=1e-12,
                    prediction_policy="reject outside training-derived constraint domain",
                    objective=("uniform_iv_sse" if price_quotes is None
                               else "mean_squared_half_spread_price_error"))
    if price_quotes is not None:
        geometry["price_spread_floor"] = PRICE_SPREAD_FLOOR
        geometry["optimizer_objective_scale"] = price_scale
    if price_quotes is None and np.all(iv == iv[0]):
        if not 1e-10 <= variance <= 5:
            raise ValueError("Flat variance is outside the research bounds")
        fit = unpack([1, 0, 0, np.clip(0, lower[3], upper[3]),
                      np.clip(1, lower[4], upper[4])])
        check = validate(fit)
        if not check["passed"]:
            raise RuntimeError(f"Flat limit failed wing checks: {check}")
        return fit, dict(method="constrained_exact_flat_limit", training_sse=0.0,
                         starts=[], nfev_total=0, wing_check=check,
                         jacobian_note="flat rho/m/sigma are unidentifiable", **geometry)

    starts = [(rho, np.clip([0.8 * np.min(target**2), 0.1, rho, 0, 1], lower, upper))
              for rho in (-0.6, 0.0, 0.6)]
    seed_nfev = 0
    if iv_seed:
        seed_params, seed_evidence = fit_raw_svi_constrained(k, iv, ttm)
        if seed_evidence["starts"]:
            seed = np.array(seed_evidence["starts"][seed_evidence["selected_start"]]
                            ["scaled_parameters"])
        else:
            seed = np.clip([1, 0, 0, 0, 1], lower, upper)
        starts.append((float(seed[2]), seed))
        seed_nfev = seed_evidence["nfev_total"]
        geometry["iv_seed"] = dict(parameters=vars(seed_params), optimizer=seed_evidence)
        geometry["seed_nfev_total"] = seed_nfev

    attempts, accepted = [], []
    for index, (rho, start) in enumerate(starts):
        result = minimize(objective, start, method="SLSQP", jac=True,
                          bounds=list(zip(lower, upper)),
                          constraints={"type": "ineq", "fun": constraints},
                          options={"maxiter": 1000, "ftol": 1e-12})
        params = unpack(result.x)
        check = validate(params)
        valid = bool(result.success and np.isfinite(result.fun)
                     and np.all(np.isfinite(result.x)) and check["passed"]
                     and np.all(result.x >= lower) and np.all(result.x <= upper))
        sse = float(np.sum((np.sqrt(raw_svi(k, params)[0] / ttm) - iv)**2))
        score = (sse if price_quotes is None else
                 _spread_price_objective(result.x, z, target, k, ttm, vol, price_quotes)[0])
        valid = bool(valid and np.isfinite(score))
        attempts.append(dict(start_rho=rho, success=bool(result.success), accepted=valid,
                             status=int(result.status), message=str(result.message),
                             nfev=int(result.nfev), nit=int(result.nit), training_sse=sse,
                             training_objective=score,
                             scaled_parameters=result.x.tolist(), wing_check=check))
        if iv_seed:
            attempts[-1].update(start_kind="fixed" if index < 3 else "training_iv_seed",
                                initial_scaled_parameters=start.tolist())
        if valid:
            accepted.append((score, params, len(attempts) - 1))
    if not accepted:
        raise RuntimeError(f"All constrained SVI starts failed or were infeasible: {attempts}")
    score, fit, selected = min(accepted, key=lambda item: item[0])
    price_nfev = sum(item["nfev"] for item in attempts)
    if iv_seed:
        geometry["price_nfev_total"] = price_nfev
    method = "scaled_slsqp_finite_wing_constraints" + ("_iv_seed" if iv_seed else "")
    return fit, dict(method=method,
                     training_sse=attempts[selected]["training_sse"], training_objective=score,
                     starts=attempts, selected_start=selected,
                     nfev_total=price_nfev + seed_nfev,
                     wing_check=validate(fit), **geometry)
