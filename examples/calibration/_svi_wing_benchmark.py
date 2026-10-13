"""Opt-in finite-domain raw-SVI constraints; repository-only research code.

The density factor is Gatheral/Jacquier (2013), equation (2.1). The call slope
comes from differentiating their Black formula (section 2.2) in strike space.
Sampled constraints are not a global no-arbitrage or extrapolation certificate.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import erfcx, ndtr

from _svi_benchmark import RawSvi, _scaled_svi_residual, durrleman, raw_svi


DENSITY_BUFFER = 1e-5
DENSITY_TOLERANCE = 1e-8
SLOPE_TOLERANCE = 1e-10
OPTIMIZATION_POINTS = 801
VALIDATION_POINTS = 4001
PRICE_SPREAD_FLOOR = 0.01
DOMAIN_ROUNDOFF = 1e-12
IV_ANCHOR_SCALE = 0.005
IV_ANCHOR_POINTS = 81
INITIALIZATION_MAX_NFEV = 500
MAX_GRID_REFINEMENTS = 2


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


def _iv_anchor_objective(p, z, reference_iv, vol):
    """Fixed-scale IV departure from a training-only reference, with gradient."""
    zero = np.zeros_like(z)
    residual = (vol * _scaled_svi_residual(p, z, zero) - reference_iv) / IV_ANCHOR_SCALE
    jac = vol / IV_ANCHOR_SCALE * _scaled_svi_residual(p, z, zero, jacobian=True)
    return float(np.mean(residual**2)), 2 * jac.T @ residual / len(z)


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


def _normal_mills(x):
    """Return n(x)/N(-x) and its derivative without subtracting tail probabilities."""
    x = np.asarray(x)
    value = np.empty_like(x, dtype=float)
    positive = x >= 0
    value[positive] = np.sqrt(2/np.pi)/erfcx(x[positive]/np.sqrt(2))
    value[~positive] = np.exp(-x[~positive]**2/2)/np.sqrt(2*np.pi)/ndtr(-x[~positive])
    return value, value*(value-x)


def _scaled_wing_constraints(p, k, center, scale, variance, *, jacobian=False):
    """Evaluate equivalent normalized slope margins and their analytic Jacobian.

    Apply the chain rule to the positive-minimum raw variance, its first two
    strike derivatives and g(k). Dividing 1+slope by N(-d2), and -slope by
    N(d2), preserves their feasible set; normal Mills ratios keep these positive
    probability rescalings finite when raw tail probabilities round to zero.
    """
    minimum, b, rho, m, sigma = p
    k = np.asarray(k)
    x = (k-center)/scale-m
    h, r = np.hypot(x, sigma), np.sqrt(1-rho*rho)
    params = RawSvi(variance*(minimum-b*sigma*r), b*variance/scale,
                    rho, center+scale*m, scale*sigma)
    w, first, second = raw_svi(k, params)
    root = np.sqrt(w)
    d2, a = -k/root-root/2, first/(2*root)
    left, left_derivative = _normal_mills(d2)
    right, right_derivative = _normal_mills(-d2)
    if not jacobian:
        return np.r_[durrleman(k, w, first, second)-DENSITY_BUFFER,
                     1+left*a, 1-right*a]
    zero = np.zeros_like(x)
    dw = variance*np.column_stack((
        np.ones_like(x), rho*x+h-sigma*r, b*(x+sigma*rho/r),
        -b*(rho+x/h), b*(sigma/h-r)))
    df = variance/scale*np.column_stack((
        zero, rho+x/h, np.full_like(x, b), -b*sigma*sigma/h**3, -b*x*sigma/h**3))
    ds = variance/scale**2*np.column_stack((
        zero, sigma*sigma/h**3, zero, 3*b*sigma*sigma*x/h**5,
        b*(2*sigma/h**3-3*sigma**3/h**5)))
    wcol, fcol = w[:, None], first[:, None]
    factor = 1-k*first/(2*w)
    dg = (-factor*k)[:, None]*(df/wcol-fcol*dw/wcol**2)
    dg += -fcol*df/2*(1/wcol+.25)+fcol**2*dw/(4*wcol**2)+ds/2
    dd2 = (k/(2*w*root)-1/(4*root))[:, None]*dw
    da = df/(2*root[:, None])-fcol*dw/(4*(w*root)[:, None])
    lower_jac = (left_derivative*a)[:, None]*dd2+left[:, None]*da
    upper_jac = (right_derivative*a)[:, None]*dd2-right[:, None]*da
    return np.vstack((dg, lower_jac, upper_jac))


def _scaled_wing_jacobian(p, k, center, scale, variance):
    """Return the Jacobian of the optimizer's equivalent wing constraints."""
    return _scaled_wing_constraints(p, k, center, scale, variance, jacobian=True)


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


def _dense_wing_grid(grid, params):
    """Combine current constraint, uniform validation and shape-local points."""
    local = params.m+params.sigma*np.linspace(-8, 8, 161)
    local = local[(local >= grid[0]) & (local <= grid[-1])]
    return np.unique(np.r_[grid, local, np.linspace(grid[0], grid[-1], VALIDATION_POINTS)])


def _refinement_points(dense, params):
    """Retain dense violations inside the fixed domain for a fresh constrained solve."""
    g, slope = svi_wing_values(dense, params)
    return dense[(g < DENSITY_BUFFER) | (slope < -1-SLOPE_TOLERANCE)
                 | (slope > SLOPE_TOLERANCE)]


def _narrow_initializations(starts, z, target, lower, upper):
    """Prepare bounded training-only IV starts; never accept them as final fits."""
    records, seeds = [], []
    for index, (rho, start) in enumerate(starts):
        record = dict(start_rho=rho, initial_scaled_parameters=start.tolist(),
                      accepted_as_start=False, nfev=0, residual_calls=0, jacobian_calls=0)

        def residual(p):
            record['residual_calls'] += 1
            return _scaled_svi_residual(p, z, target)

        def jac(p):
            record['jacobian_calls'] += 1
            return _scaled_svi_residual(p, z, target, jacobian=True)

        try:
            result = least_squares(residual, start.copy(), bounds=(lower, upper), jac=jac,
                                   max_nfev=INITIALIZATION_MAX_NFEV,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
            point = np.asarray(result.x)
            finite = bool(np.all(np.isfinite(point)))
            inside = bool(point.shape == start.shape and finite
                          and np.all(point >= lower) and np.all(point <= upper))
            score = (float(np.sum(_scaled_svi_residual(point, z, target)**2))
                     if inside else np.nan)
            valid = bool((result.success or result.status == 0) and inside and np.isfinite(score))
            record.update(success=bool(result.success), status=int(result.status),
                          message=str(result.message), nfev=int(result.nfev),
                          normalized_iv_sse=score if np.isfinite(score) else None,
                          scaled_parameters=[float(x) if np.isfinite(x) else None for x in point],
                          accepted_as_start=valid, budget_exhausted=bool(result.status == 0),
                          finite=finite, in_bounds=inside)
            if valid:
                seeds.append((index, point.copy()))
        except Exception as error:
            record.update(error=f'{type(error).__name__}: {error}',
                          nfev=record['residual_calls'], nfev_source='counted_before_error')
        records.append(record)
    return seeds, records


def fit_raw_svi_constrained(k, iv, ttm, *, price_quotes=None, iv_seed=False, iv_anchor=False):
    """Fit IV SSE or half-spread price MSE under finite-grid strike constraints.

    Parameters use the same positive-minimum and training-relative bounds as
    the bounded research candidate. Three independent SLSQP starts receive no
    held-out data. One-sided IV support adds bounded training-only IV initializers.
    Dense violations trigger at most two explicit grid refinements per start.
    Unsuccessful or finally dense-check-infeasible solutions are rejected
    before objective selection. Optional price_quotes contains training bid, ask,
    optiontypes, forward and discount; IVs then supply scaling and starts.
    With iv_seed=True, a constrained training-IV fit supplies a fourth price start;
    its cost and evidence are retained. It must converge again on the price loss.
    With iv_anchor=True, require that seed and add mean squared IV departure from
    its curve on 81 uniform training-domain points, divided by fixed 0.005 squared.
    Optimization and accepted-start selection then use the combined loss.
    Exhaustion is a failed fit, not a flat fallback. This is a repository-only helper.
    """
    if iv_anchor and (not iv_seed or price_quotes is None):
        raise ValueError("The IV anchor requires an IV seed and price quotes")
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
            if iv_anchor:
                penalty, derivative = _iv_anchor_objective(p, anchor_z, reference_iv, vol)
                value, gradient = value + penalty, gradient + derivative
            return value / price_scale, gradient / price_scale
        residual = _scaled_svi_residual(p, z, target)
        jac = _scaled_svi_residual(p, z, target, jacobian=True)
        return float(residual @ residual), 2 * jac.T @ residual

    def constraints(p):
        return _scaled_wing_constraints(p, solve_grid, center, scale, variance)

    def constraint_jac(p):
        return _scaled_wing_jacobian(p, solve_grid, center, scale, variance)

    def validate(params, points=None):
        return check_svi_wings(_dense_wing_grid(grid if points is None else points, params), params)

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
                    maxiter=1000, ftol=1e-12, constraint_jacobian="analytic",
                    refinement_limit=MAX_GRID_REFINEMENTS,
                    constraint_scaling="call-slope margins / positive Black tail probability",
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
    kinds = [('fixed', None)]*3
    initialization = []
    if price_quotes is None and (k[0] > 0 or k[-1] < 0):
        seeds, initialization = _narrow_initializations(starts, z, target, lower, upper)
        for index, point in seeds:
            starts.append((float(point[2]), point))
            kinds.append(('training_iv_initialization', index))
    initialization_nfev = sum(r['nfev'] for r in initialization)
    geometry.update(initialization=initialization, initialization_nfev_total=initialization_nfev,
                    initialization_policy='bounded IV least squares for strictly one-sided support',
                    initialization_max_nfev=INITIALIZATION_MAX_NFEV)
    seed_nfev = 0
    if iv_seed:
        seed_params, seed_evidence = fit_raw_svi_constrained(k, iv, ttm)
        if seed_evidence["starts"]:
            seed = np.array(seed_evidence["starts"][seed_evidence["selected_start"]]
                            ["scaled_parameters"])
        else:
            seed = np.clip([1, 0, 0, 0, 1], lower, upper)
        starts.append((float(seed[2]), seed))
        kinds.append(('training_iv_seed', None))
        seed_nfev = seed_evidence["nfev_total"]
        geometry["iv_seed"] = dict(parameters=vars(seed_params), optimizer=seed_evidence)
        geometry["seed_nfev_total"] = seed_nfev

    if iv_anchor:
        anchor_grid = np.linspace(grid[0], grid[-1], IV_ANCHOR_POINTS)
        anchor_z = (anchor_grid - center) / scale
        reference_iv = np.sqrt(raw_svi(anchor_grid, seed_params)[0] / ttm)
        geometry["objective"] = "half_spread_price_mse_plus_iv_anchor_mse"
        geometry["iv_anchor"] = dict(scale=IV_ANCHOR_SCALE, grid=anchor_grid.tolist(),
                                     reference_iv=reference_iv.tolist(), weight=1.0,
                                     rule="fixed before evaluation; training-IV reference only")

    attempts, accepted = [], []
    for index, (rho, start) in enumerate(starts):
        solve_grid, point, rounds = grid.copy(), start.copy(), []
        for refinement in range(MAX_GRID_REFINEMENTS+1):
            result = minimize(objective, point.copy(), method="SLSQP", jac=True,
                              bounds=list(zip(lower, upper)),
                              constraints={"type": "ineq", "fun": constraints,
                                           "jac": constraint_jac},
                              options={"maxiter": 1000, "ftol": 1e-12})
            params = unpack(result.x)
            dense = _dense_wing_grid(solve_grid, params)
            check = check_svi_wings(dense, params)
            eligible = bool(result.success and np.isfinite(result.fun)
                            and np.all(np.isfinite(result.x))
                            and np.all(result.x >= lower) and np.all(result.x <= upper))
            sse = float(np.sum((np.sqrt(raw_svi(k, params)[0] / ttm) - iv)**2))
            score = (sse if price_quotes is None else
                     _spread_price_objective(result.x, z, target, k, ttm, vol, price_quotes)[0])
            entry = dict(success=bool(result.success), accepted=False, status=int(result.status),
                         message=str(result.message), nfev=int(result.nfev), nit=int(result.nit),
                         solver_objective=float(result.fun) if np.isfinite(result.fun) else None,
                         training_sse=sse, training_objective=score,
                         initial_scaled_parameters=point.tolist(),
                         scaled_parameters=result.x.tolist(), wing_check=check,
                         optimization_grid=solve_grid.tolist(), added_points=[])
            if iv_anchor:
                anchor_score = _iv_anchor_objective(result.x, anchor_z, reference_iv, vol)[0]
                entry.update(price_objective=score, anchor_objective=anchor_score)
                score += anchor_score
                entry['training_objective'] = score
            valid = bool(eligible and np.isfinite(score) and check['passed'])
            entry['accepted'] = valid
            rounds.append(entry)
            if valid or not eligible or not np.isfinite(score) or not check['finite']:
                entry['stop_reason'] = 'accepted' if valid else 'solver_or_numerical_failure'
                break
            if refinement == MAX_GRID_REFINEMENTS:
                entry['stop_reason'] = 'refinement_limit'
                break
            additions = np.setdiff1d(_refinement_points(dense, params), solve_grid)
            if not len(additions):
                entry['stop_reason'] = 'no_new_points'
                break
            entry.update(added_points=additions.tolist(), stop_reason='refine_grid')
            solve_grid = np.unique(np.r_[solve_grid, additions])
            point = result.x.copy()
        attempt = dict(rounds[-1], start_rho=rho, start_kind=kinds[index][0],
                       initialization_index=kinds[index][1], rounds=rounds,
                       initial_scaled_parameters=start.tolist(),
                       nfev=sum(r['nfev'] for r in rounds), nit=sum(r['nit'] for r in rounds))
        if index == 0 and initialization:
            attempt['initialization'] = initialization
        attempts.append(attempt)
        if valid:
            accepted.append((score, params, len(attempts) - 1))
    if not accepted:
        raise RuntimeError(f"All constrained SVI starts failed or were infeasible: {attempts}")
    score, fit, selected = min(accepted, key=lambda item: item[0])
    price_nfev = sum(item["nfev"] for item in attempts)
    if iv_seed:
        geometry["price_nfev_total"] = price_nfev
    method = "scaled_slsqp_finite_wing_constraints" + ("_iv_seed" if iv_seed else "")
    if iv_anchor:
        method += "_iv_anchor"
        geometry.update(price_objective=attempts[selected]["price_objective"],
                        anchor_objective=attempts[selected]["anchor_objective"])
    return fit, dict(method=method,
                     training_sse=attempts[selected]["training_sse"], training_objective=score,
                     starts=attempts, selected_start=selected,
                     nfev_total=price_nfev + seed_nfev + initialization_nfev,
                     wing_check=validate(fit, np.array(attempts[selected]['optimization_grid'])),
                     **geometry)
