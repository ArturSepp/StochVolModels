"""Opt-in shared-rho raw SVI; repository-only research code.

This is a joint calibration restriction on raw SVI, not SSVI. Strike checks use
the existing finite-domain helper; no maturity or calendar guarantee is added.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from _svi_benchmark import RawSvi, _scaled_svi_residual, raw_svi
from _svi_wing_benchmark import (
    DENSITY_BUFFER, OPTIMIZATION_POINTS, VALIDATION_POINTS, check_svi_wings, svi_wing_values,
)


@dataclass
class _Slice:
    k: np.ndarray
    iv: np.ndarray
    ttm: float
    center: float
    scale: float
    vol: float
    variance: float
    lower: np.ndarray
    upper: np.ndarray
    grid: np.ndarray


def _prepare_slices(ks, ivs, ttms):
    """Copy, sort and validate training-only geometry, preserving slice identity."""
    ttms = np.asarray(ttms, dtype=float)
    if (ttms.ndim != 1 or not len(ttms) or len(ks) != len(ttms) or len(ivs) != len(ttms)
            or not np.all(np.isfinite(ttms) & (ttms > 0)) or np.any(np.diff(ttms) <= 0)):
        raise ValueError("Shared-rho SVI requires aligned slices and increasing positive tenors")
    slices = []
    for k, iv, ttm in zip(ks, ivs, ttms):
        k, iv = np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
        if (k.ndim != 1 or iv.shape != k.shape or len(k) < 5
                or not np.all(np.isfinite(k)) or len(np.unique(k)) < 5
                or not np.all(np.isfinite(iv) & (iv > 0))):
            raise ValueError("Each shared-rho slice needs five distinct finite strikes and IVs > 0")
        order = np.argsort(k)
        k, iv = k[order], iv[order]
        center, scale = (k[0] + k[-1]) / 2, (k[-1] - k[0]) / 2
        vol = float(np.mean(iv))
        variance = ttm * vol**2
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Nonfinite or underflowed training variance")
        lower = np.array([1e-10 / variance, 0, max(-2, (-1.5-center)/scale),
                          max(0.05, 1e-4/scale)])
        upper = np.array([5 / variance, scale / variance, min(2, (1.5-center)/scale),
                          min(2, 3/scale)])
        if np.any(lower >= upper) or not np.all(np.isfinite(np.r_[lower, upper])):
            raise ValueError("Training domain does not intersect the research parameter bounds")
        grid = np.unique(np.r_[k, center + scale*np.linspace(-2, 2, OPTIMIZATION_POINTS)])
        slices.append(_Slice(k, iv, float(ttm), float(center), float(scale), vol,
                             float(variance), lower, upper, grid))
    return slices


def _local_parameters(p, index):
    minimum, b, m, sigma = p[1+4*index:5+4*index]
    return np.array([minimum, b, p[0], m, sigma])


def _unpack(p, slices):
    params = []
    for i, s in enumerate(slices):
        minimum, b, rho, m, sigma = _local_parameters(p, i)
        params.append(RawSvi(float(s.variance*(minimum-b*sigma*np.sqrt(1-rho**2))),
                             float(b*s.variance/s.scale), float(rho),
                             float(s.center+s.scale*m), float(s.scale*sigma)))
    return params


def _joint_residual(p, slices, common_vol, jacobian=False):
    """Uniform physical IV residuals divided by one common optimizer scale."""
    rows = []
    for i, s in enumerate(slices):
        value = s.vol/common_vol * _scaled_svi_residual(
            _local_parameters(p, i), (s.k-s.center)/s.scale, s.iv/s.vol, jacobian=jacobian,
        )
        if jacobian:
            block = np.zeros((len(s.k), len(p)))
            block[:, 0] = value[:, 2]
            block[:, 1+4*i:5+4*i] = value[:, [0, 1, 3, 4]]
            value = block
        rows.append(value)
    return np.concatenate(rows)


def _validate(params, slices):
    checks = []
    for p, s in zip(params, slices):
        local = p.m + p.sigma*np.linspace(-8, 8, 161)
        local = local[(local >= s.grid[0]) & (local <= s.grid[-1])]
        grid = np.unique(np.r_[s.grid, local,
                               np.linspace(s.grid[0], s.grid[-1], VALIDATION_POINTS)])
        checks.append(check_svi_wings(grid, p))
    return checks


def fit_raw_svi_shared_rho(ks, ivs, ttms):
    """Jointly fit training IVs with shared rho and four free parameters per expiry.

    Parameters
    ----------
    ks, ivs : sequence of one-dimensional arrays
        Log-strike/forward and decimal IV observations, at least five distinct
        strikes per expiry. Only training quotes may be supplied.
    ttms : array-like
        Strictly increasing positive maturities in years.

    Returns
    -------
    params : list of RawSvi
        One raw smile per maturity, with identical rho.
    evidence : dict
        All three starts, physical IV SSE, joint bounds, per-slice geometry and
        dense strike checks. Exact flat slices together bypass optimization.

    Notes
    -----
    Bounds and finite-grid constraints match the independent constrained fitter.
    A single common loss scale preserves uniform quote weighting across expiries.
    Only successful, finite, in-bounds, dense-feasible starts enter selection.
    Exhaustion raises an error; no fallback, interpolation or repair is applied.
    """
    slices = _prepare_slices(ks, ivs, ttms)
    common = float(np.mean(np.concatenate([s.iv for s in slices])))
    lower = np.r_[-0.999, np.concatenate([s.lower for s in slices])]
    upper = np.r_[0.999, np.concatenate([s.upper for s in slices])]
    geometry = dict(
        objective="uniform_iv_sse", optimizer_iv_scale=common,
        scaled_lower=lower.tolist(), scaled_upper=upper.tolist(),
        scaled_parameter_order=["rho"] + [f"{name}_{i}" for i in range(len(slices))
                                          for name in ("minimum", "b", "m", "sigma")],
        maxiter=1000, ftol=1e-12, density_buffer=DENSITY_BUFFER,
        prediction_policy="fitted tenors and training-derived strike domains only",
        slices=[dict(ttm=s.ttm, k_center=s.center, k_scale=s.scale, iv_scale=s.vol,
                     optimization_grid=s.grid.tolist()) for s in slices],
    )

    def objective(p):
        residual = _joint_residual(p, slices, common)
        jac = _joint_residual(p, slices, common, jacobian=True)
        return float(residual@residual), 2*jac.T@residual

    def constraints(p):
        values = []
        for params, s in zip(_unpack(p, slices), slices):
            g, slope = svi_wing_values(s.grid, params)
            values.append(np.r_[g-DENSITY_BUFFER, 1+slope, -slope])
        return np.concatenate(values)

    if all(np.all(s.iv == s.iv[0]) for s in slices):
        if any(not 1e-10 <= s.variance <= 5 for s in slices):
            raise ValueError("Flat variance is outside the research bounds")
        p = np.r_[0.0, np.concatenate([np.clip([1, 0, 0, 1], s.lower, s.upper)
                                       for s in slices])]
        params = _unpack(p, slices)
        checks = _validate(params, slices)
        if not all(c["passed"] for c in checks):
            raise RuntimeError(f"Flat shared-rho limit failed wing checks: {checks}")
        return params, dict(method="shared_rho_exact_flat_limit", training_sse=0.0,
                            starts=[], nfev_total=0, shared_rho=0.0, wing_checks=checks,
                            jacobian_note="flat rho/m/sigma are unidentifiable", **geometry)

    attempts, accepted = [], []
    for index, rho in enumerate((-0.6, 0.0, 0.6)):
        start = np.r_[rho, np.concatenate([np.clip(
            [0.8*np.min((s.iv/s.vol)**2), 0.1, 0, 1], s.lower, s.upper) for s in slices])]
        result = minimize(objective, start, method="SLSQP", jac=True,
                          bounds=list(zip(lower, upper)),
                          constraints={"type": "ineq", "fun": constraints},
                          options={"maxiter": 1000, "ftol": 1e-12})
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            params = _unpack(result.x, slices)
            checks = _validate(params, slices)
            sse = float(sum(np.sum((np.sqrt(raw_svi(s.k, p)[0]/s.ttm)-s.iv)**2)
                            for p, s in zip(params, slices)))
        valid = bool(result.success and np.isfinite(result.fun) and np.isfinite(sse)
                     and np.all(np.isfinite(result.x)) and np.all(result.x >= lower)
                     and np.all(result.x <= upper) and all(c["passed"] for c in checks))
        attempts.append(dict(start_rho=rho, initial_parameters=start.tolist(),
                             scaled_parameters=result.x.tolist(), success=bool(result.success),
                             accepted=valid, status=int(result.status), message=str(result.message),
                             nfev=int(result.nfev), nit=int(result.nit), training_sse=sse,
                             training_objective=float(result.fun), wing_checks=checks))
        if valid:
            accepted.append((sse, index, params, checks))
    if not accepted:
        raise RuntimeError(f"No shared-rho start passed convergence and wing checks: {attempts}")
    sse, selected, params, checks = min(accepted, key=lambda item: item[0])
    return params, dict(method="shared_rho_constrained_iv", training_sse=sse,
                        selected_start=selected, starts=attempts,
                        nfev_total=sum(a["nfev"] for a in attempts),
                        shared_rho=params[0].rho, wing_checks=checks, **geometry)
