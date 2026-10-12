"""Finite-domain checks for standard European Black volatility surfaces.

Calendar interpretation follows Gatheral and Jacquier, arXiv:1204.0646,
Section 2.1, under the stated common-underlying and proportional-carry assumptions.
Finite strike grids do not certify global absence of arbitrage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import vanilla_option_pricers as bsm

from stochvolmodels.fitters.diagnostics import DiagnosticReport


def _grid(values, name: str, *, positive=False, minimum=1) -> np.ndarray:
    """Require finite ordered coordinates without silently sorting model outputs."""
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or len(array) < minimum
        or not np.all(np.isfinite(array))
        or np.any(np.diff(array) <= 0)
        or (positive and np.any(array <= 0))
    ):
        raise ValueError(f"{name} must be a finite strictly increasing grid")
    return array


def _positive(value: float, name: str) -> None:
    """Validate positive scalar conventions, including discounts above one."""
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _check(name, margins, coordinates, tolerance, unit, *, reason="") -> dict:
    """Check nonnegative margins; incomplete grids cannot silently pass."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    margins = np.asarray(margins, dtype=float)
    finite = np.isfinite(margins)
    valid = int(finite.sum())
    minimum = float(np.min(margins[finite])) if valid else np.nan
    index = int(np.flatnonzero(finite)[np.argmin(margins[finite])]) if valid else None
    status = (
        "fail"
        if valid and minimum < -tolerance
        else "not_checked"
        if not valid or not finite.all()
        else "pass"
    )
    return dict(
        check_id=name,
        status=status,
        severity="error" if status == "fail" else "info",
        value=minimum,
        unit=unit,
        tolerance=tolerance,
        valid_count=valid,
        invalid_count=int(len(margins) - valid),
        location=float(coordinates[index]) if index is not None else np.nan,
        reason=reason or ("incomplete_or_empty_grid" if status == "not_checked" else ""),
    )


def check_call_slice(
    strikes: Sequence[float],
    call_prices: Sequence[float],
    *,
    forward: float,
    discount: float,
    price_unit: str,
    tolerance: float = 1e-10,
    density: Sequence[float] | None = None,
    density_tolerance: float = 1e-10,
    payoff: str = "standard_european",
) -> DiagnosticReport:
    """Check bounds, vertical spreads and convexity on an irregular strike grid.

    Parameters
    ----------
    strikes, call_prices : sequence of float
        Increasing positive strikes and aligned discounted call prices in the
        forward's currency per underlying unit. Never pass a mixed call/put curve.
    forward, discount : float
        Positive forward and discount; discounts greater than one are supported.
    price_unit : str
        Explicit common unit of call prices and forward.
    tolerance : float
        Nonnegative tolerance on prices normalized by discount*forward, and on
        slopes/slope increments normalized by discount.
    density : sequence of float, optional
        Independent analytic density with respect to strike, at every strike.
        Without it the check uses interior second divided differences / discount.
    density_tolerance : float
        Nonnegative density tolerance in inverse strike units.
    payoff : str
        Only ``standard_european`` has this interpretation.

    Returns
    -------
    DiagnosticReport
        ``checks``, ``grid`` and ``density`` tables. Negative mass is integrated
        only over adjacent finite density points; no renormalization is applied.
    """
    strikes = _grid(strikes, "strikes", positive=True)
    prices = np.asarray(call_prices, dtype=float)
    if prices.shape != strikes.shape:
        raise ValueError("call_prices must align with strikes")
    _positive(forward, "forward")
    _positive(discount, "discount")
    if not isinstance(price_unit, str) or not price_unit.strip():
        raise ValueError("price_unit must be explicit")
    if payoff != "standard_european":
        check = _check("payoff", [], [], tolerance, "dimensionless")
        check.update(status="not_applicable", reason="unsupported_payoff")
        return DiagnosticReport.from_tables({"checks": pd.DataFrame([check])}, {"payoff": payoff})
    safe = np.where(np.isfinite(prices), prices, np.nan)
    bounds = np.minimum(
        safe - discount * np.maximum(forward - strikes, 0), discount * forward - safe
    ) / (discount * forward)
    slopes = np.diff(safe) / np.diff(strikes)
    vertical = np.minimum(-slopes, slopes + discount) / discount
    convex = np.diff(slopes) / discount
    checks = [
        _check("call_bounds", bounds, strikes, tolerance, "price/(discount*forward)"),
        _check("vertical_spread", vertical, strikes[:-1], tolerance, "slope/discount"),
        _check("convexity", convex, strikes[1:-1], tolerance, "slope_increment/discount"),
    ]
    numerical_density = 2 * np.diff(slopes) / (strikes[2:] - strikes[:-2]) / discount
    if density is None:
        density_values, density_strikes = numerical_density, strikes[1:-1]
    else:
        density_values = np.asarray(density, dtype=float)
        if density_values.shape != strikes.shape:
            raise ValueError("density must align with strikes and use strike measure")
        density_strikes = strikes
    checks.append(
        _check(
            "density_nonnegative", density_values, density_strikes, density_tolerance, "1/strike"
        )
    )
    adjacent = np.isfinite(density_values[:-1]) & np.isfinite(density_values[1:])
    widths = np.diff(density_strikes)[adjacent]
    left, right = density_values[:-1][adjacent], density_values[1:][adjacent]
    mass = float(np.sum(widths * (left + right) / 2)) if adjacent.any() else np.nan
    negative_mass = (
        float(np.sum(widths * (np.maximum(-left, 0) + np.maximum(-right, 0)) / 2))
        if adjacent.any()
        else np.nan
    )
    density_frame = pd.DataFrame({"strike": density_strikes, "density": density_values})
    return DiagnosticReport.from_tables(
        {
            "checks": pd.DataFrame(checks),
            "grid": pd.DataFrame({"strike": strikes, "call_price": prices}),
            "density": density_frame,
        },
        {
            "kind": "call_slice",
            "payoff": payoff,
            "price_unit": price_unit,
            "forward": forward,
            "discount": discount,
            "domain": [strikes[0], strikes[-1]],
            "density_method": "supplied_strike_density"
            if density is not None
            else "divided_differences",
            "density_measure": "dK",
            "density_mass_on_grid": mass,
            "negative_density_mass_on_grid": negative_mass,
            "density_integrated_intervals": int(adjacent.sum()),
            "global_certificate": False,
        },
    )


def check_calendar(
    ttms: Sequence[float],
    log_moneyness: Sequence[float],
    total_variance: np.ndarray,
    *,
    underlying_id: str | None = None,
    carry_model: str = "unspecified",
    payoff: str = "standard_european",
    tolerance: float = 1e-10,
) -> DiagnosticReport:
    """Compare total variance at common forward log-moneyness across maturities.

    ``total_variance`` has shape (n_maturities, n_log_moneyness). Calendar
    interpretation requires one declared underlying_id and carry_model
    ``proportional``. Use ``different_underlyings`` for separate futures.
    Unspecified assumptions are not checked; unsupported ones are not applicable.
    Non-finite or nonpositive variance cannot pass. A grid pass is not global.
    """
    ttms = _grid(ttms, "ttms", positive=True)
    k = _grid(log_moneyness, "log_moneyness")
    variance = np.asarray(total_variance, dtype=float)
    if variance.shape != (len(ttms), len(k)):
        raise ValueError("total_variance must have shape (len(ttms), len(log_moneyness))")
    valid = np.isfinite(variance) & (variance > 0)
    checks = []
    for i in range(len(ttms) - 1):
        gap = np.where(valid[i] & valid[i + 1], variance[i + 1] - variance[i], np.nan)
        check = _check("calendar", gap, k, tolerance, "total_variance")
        check.update(ttm=ttms[i], next_ttm=ttms[i + 1])
        if payoff != "standard_european" or carry_model not in ("proportional", "unspecified"):
            check.update(
                status="not_applicable", reason="unsupported_payoff_or_carry", severity="info"
            )
        elif not underlying_id or carry_model == "unspecified":
            check.update(
                status="not_checked", reason="undeclared_underlying_or_carry", severity="info"
            )
        checks.append(check)
    if not checks:
        check = _check(
            "calendar", [], [], tolerance, "total_variance", reason="fewer_than_two_maturities"
        )
        check.update(ttm=ttms[0], next_ttm=np.nan)
        checks.append(check)
    grid = pd.DataFrame(
        {
            "ttm": np.repeat(ttms, len(k)),
            "log_moneyness": np.tile(k, len(ttms)),
            "total_variance": variance.ravel(),
            "valid": valid.ravel(),
        }
    )
    return DiagnosticReport.from_tables(
        {"checks": pd.DataFrame(checks), "grid": grid},
        {
            "kind": "calendar",
            "underlying_id": underlying_id,
            "carry_model": carry_model,
            "payoff": payoff,
            "domain": [k[0], k[-1]],
            "global_certificate": False,
        },
    )


@dataclass(frozen=True)
class SurfaceEvaluator:
    """Declared surface callbacks and their supported domain.

    iv(strikes, ttm) returns decimal Black IV; forward(ttm) and discount(ttm)
    return positive scalars. Domains refer to maturity and forward log-moneyness.
    Use interpolation='none' for evaluators defined only at calibration maturities.
    No interpolation or conversion is supplied implicitly by the diagnostics.
    """

    iv: Callable[[np.ndarray, float], np.ndarray]
    forward: Callable[[float], float]
    discount: Callable[[float], float]
    ttm_domain: tuple[float, float]
    k_domain: tuple[float, float]
    underlying_id: str
    interpolation: str
    carry_model: str = "unspecified"
    payoff: str = "standard_european"
    extrapolation: str = "none"


def diagnose_surface(
    evaluator: SurfaceEvaluator,
    ttms: Sequence[float],
    log_moneyness: Sequence[float],
    *,
    extra_ttms: Sequence[float] = (),
    allow_extrapolation: bool = False,
    price_unit: str,
    tolerance: float = 1e-10,
    density_tolerance: float = 1e-10,
    metadata: Mapping | None = None,
) -> DiagnosticReport:
    """Evaluate slice and calendar checks, optionally between fitted expiries.

    Every evaluation uses strike = forward(ttm)*exp(log_moneyness). Declared
    extrapolation requires explicit opt-in. Reported domain labels distinguish
    extrapolated points from points inside the evaluator's supported domain.
    Optional metadata records source timestamps, calibration identity and other
    JSON-compatible provenance. Density tolerance is in inverse strike units.
    """
    base = _grid(ttms, "ttms", positive=True)
    k = _grid(log_moneyness, "log_moneyness", minimum=3)
    if not evaluator.interpolation or not evaluator.extrapolation:
        raise ValueError("interpolation and extrapolation policies must be explicit")
    extras = np.asarray(extra_ttms, dtype=float)
    if extras.size:
        extras = _grid(extras, "extra_ttms", positive=True)
        if evaluator.interpolation == "none":
            raise ValueError("extra maturities require an explicit interpolation policy")
    times = np.unique(np.concatenate([base, extras]))
    for domain, name in ((evaluator.ttm_domain, "ttm_domain"), (evaluator.k_domain, "k_domain")):
        if len(domain) != 2 or not np.all(np.isfinite(domain)) or domain[0] > domain[1]:
            raise ValueError(f"invalid {name}")
    if evaluator.ttm_domain[0] <= 0:
        raise ValueError("ttm_domain must be positive")
    outside = (
        (times[:, None] < evaluator.ttm_domain[0])
        | (times[:, None] > evaluator.ttm_domain[1])
        | (k[None, :] < evaluator.k_domain[0])
        | (k[None, :] > evaluator.k_domain[1])
    )
    if outside.any() and (not allow_extrapolation or evaluator.extrapolation == "none"):
        raise ValueError("requested grid requires explicitly supported extrapolation")
    if evaluator.payoff != "standard_european":
        check = _check("surface", [], [], tolerance, "dimensionless")
        check.update(status="not_applicable", reason="unsupported_payoff")
        return DiagnosticReport.from_tables(
            {"checks": pd.DataFrame([check])}, {"kind": "surface", "payoff": evaluator.payoff}
        )
    checks, grids, variances, densities, summaries = [], [], [], [], []
    for i, ttm in enumerate(times):
        forward, discount = evaluator.forward(float(ttm)), evaluator.discount(float(ttm))
        _positive(forward, "forward")
        _positive(discount, "discount")
        strikes = forward * np.exp(k)
        iv = np.asarray(evaluator.iv(strikes.copy(), float(ttm)), dtype=float)
        if iv.shape != k.shape:
            raise ValueError("iv evaluator must return one value per strike")
        valid = np.isfinite(iv) & (iv > 0)
        variance = np.full(len(k), np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            variance[valid] = ttm * iv[valid] ** 2
        valid &= np.isfinite(variance)
        variances.append(variance)
        prices = np.full(len(k), np.nan)
        for j in np.flatnonzero(valid):
            prices[j] = bsm.compute_bsm_vanilla_price(
                forward=forward,
                strike=strikes[j],
                ttm=ttm,
                vol=iv[j],
                optiontype="C",
                discfactor=discount,
            )
        report = check_call_slice(
            strikes,
            prices,
            forward=forward,
            discount=discount,
            price_unit=price_unit,
            tolerance=tolerance,
            density_tolerance=density_tolerance,
        )
        frame = report.table("checks")
        frame["ttm"] = ttm
        frame["next_ttm"] = np.nan
        checks.append(frame)
        density_frame = report.table("density")
        density_frame["ttm"] = ttm
        densities.append(density_frame)
        summaries.append(
            dict(
                ttm=ttm,
                density_mass_on_grid=report.metadata["density_mass_on_grid"],
                negative_density_mass_on_grid=report.metadata["negative_density_mass_on_grid"],
                density_integrated_intervals=report.metadata["density_integrated_intervals"],
            )
        )
        grids.append(
            pd.DataFrame(
                {
                    "ttm": ttm,
                    "log_moneyness": k,
                    "strike": strikes,
                    "forward": forward,
                    "discount": discount,
                    "iv": iv,
                    "total_variance": variance,
                    "call_price": prices,
                    "extrapolated": outside[i],
                }
            )
        )
    calendar = check_calendar(
        times,
        k,
        np.array(variances),
        underlying_id=evaluator.underlying_id,
        carry_model=evaluator.carry_model,
        payoff=evaluator.payoff,
        tolerance=tolerance,
    )
    checks.append(calendar.table("checks"))
    interpolation = _check(
        "intermediate_maturities",
        [],
        [],
        tolerance,
        "maturities",
        reason="no_intermediate_maturities_requested",
    )
    intermediate = [t for t in extras if base[0] < t < base[-1] and t not in base]
    if intermediate:
        interpolation.update(
            status="pass",
            reason="sampled_only",
            value=len(intermediate),
            valid_count=len(intermediate),
            location=float(intermediate[0]),
        )
    checks.append(pd.DataFrame([interpolation]))
    return DiagnosticReport.from_tables(
        {
            "checks": pd.concat(checks, ignore_index=True),
            "grid": pd.concat(grids, ignore_index=True),
            "density": pd.concat(densities, ignore_index=True),
            "expiries": pd.DataFrame(summaries),
        },
        {
            "kind": "surface",
            "underlying_id": evaluator.underlying_id,
            "carry_model": evaluator.carry_model,
            "payoff": evaluator.payoff,
            "price_unit": price_unit,
            "iv_unit": "decimal",
            "ttm_unit": "years",
            "interpolation": evaluator.interpolation,
            "extrapolation": evaluator.extrapolation,
            "calibration_ttms": base.tolist(),
            "ttm_domain": evaluator.ttm_domain,
            "k_domain": evaluator.k_domain,
            "global_certificate": False,
            "provenance": dict(metadata or {}),
        },
    )


def compare_surface_reports(
    previous: DiagnosticReport,
    current: DiagnosticReport,
    *,
    previous_fit: DiagnosticReport | None = None,
    current_fit: DiagnosticReport | None = None,
) -> DiagnosticReport:
    """Compare IV shape only on identical maturity and log-moneyness coordinates.

    Callers must first evaluate both snapshots at the same time-to-expiry grid.
    This is a constant-tenor comparison, not attribution of changes to market
    moves versus calibration or quote churn. No regridding or time filling occurs.
    Optional paired fit reports add coverage/count and fit-error changes at matching
    tenors. Their units, spread floors and weighting labels must agree. Differences
    summarize each observed quote universe; they do not attribute quote churn.
    """
    for key in ("kind", "underlying_id", "carry_model", "payoff", "price_unit"):
        if previous.metadata.get(key) != current.metadata.get(key):
            raise ValueError(f"incompatible report convention: {key}")
    if current.metadata.get("kind") != "surface" or not current.metadata.get("underlying_id"):
        raise ValueError("comparison requires surface reports with declared underlying identity")
    keys = ["ttm", "log_moneyness"]
    before = previous.table("grid").sort_values(keys).reset_index(drop=True)
    after = current.table("grid").sort_values(keys).reset_index(drop=True)
    if not before[keys].equals(after[keys]):
        raise ValueError("evaluate both reports on an identical constant-tenor grid first")
    # Strict JSON represents unavailable numbers as null. An entirely absent
    # numeric column consequently restores as object dtype rather than float.
    for frame in (before, after):
        frame[["iv", "total_variance"]] = frame[["iv", "total_variance"]].astype(float)
    grid = after[keys].copy()
    grid["previous_iv"] = before.iv
    grid["current_iv"] = after.iv
    valid = (
        np.isfinite(before.iv)
        & (before.iv > 0)
        & np.isfinite(after.iv)
        & (after.iv > 0)
        & np.isfinite(before.total_variance)
        & (before.total_variance > 0)
        & np.isfinite(after.total_variance)
        & (after.total_variance > 0)
    )
    grid["iv_change"] = (after.iv - before.iv).where(valid)
    grid["total_variance_change"] = (after.total_variance - before.total_variance).where(valid)
    grid["status"] = np.where(valid, "pass", "not_checked")
    rows = []
    for ttm, group in grid.groupby("ttm"):
        k = group.log_moneyness.to_numpy(dtype=float)
        supported = len(k) >= 3 and k[0] <= 0 <= k[-1] and group.status.eq("pass").all()
        row = dict(
            ttm=ttm,
            status="pass" if supported else "not_checked",
            reason="" if supported else "missing_atm_support_or_invalid_values",
        )
        for field in ("previous_iv", "current_iv"):
            values = group[field].to_numpy(dtype=float)
            slope = np.gradient(values, k, edge_order=2) if supported else None
            curvature = np.gradient(slope, k, edge_order=2) if supported else None
            for name, array in (("atm", values), ("skew", slope), ("curvature", curvature)):
                row[f"{field}_{name}"] = float(np.interp(0, k, array)) if supported else np.nan
        for name in ("atm", "skew", "curvature"):
            row[f"{name}_change"] = row[f"current_iv_{name}"] - row[f"previous_iv_{name}"]
        rows.append(row)
    tables = {"grid": grid, "expiries": pd.DataFrame(rows)}
    if (previous_fit is None) != (current_fit is None):
        raise ValueError("supply both previous_fit and current_fit")
    if previous_fit is not None:
        tables["fit_changes"] = _compare_fit_summaries(
            previous_fit, current_fit, grid.ttm.unique(), current.metadata["price_unit"]
        )
    return DiagnosticReport.from_tables(
        tables,
        {
            "kind": "surface_comparison",
            "underlying_id": current.metadata["underlying_id"],
            "coordinate_policy": "identical_time_to_expiry_and_forward_log_moneyness",
            "shape_method": "finite_differences_then_linear_interpolation_at_k_zero",
            "previous": previous.metadata,
            "current": current.metadata,
            "fit_comparison": "each observed quote universe; no churn attribution"
            if previous_fit is not None
            else "not_checked",
        },
    )


def _compare_fit_summaries(previous, current, tenors, price_unit):
    """Align already-computed fit metrics by explicit common time to expiry."""
    for report in (previous, current):
        if report.metadata.get("kind") != "fit" or report.metadata["price_unit"] != price_unit:
            raise ValueError("fit comparison requires fit reports in the surface price units")
    for key in ("spread_floor", "weighting", "coverage_denominator"):
        if previous.metadata[key] != current.metadata[key]:
            raise ValueError(f"incompatible fit convention: {key}")
    frames = []
    for report in (previous, current):
        mapping = report.table("quotes")[["expiry_id", "ttm"]].drop_duplicates()
        if mapping.expiry_id.duplicated().any() or mapping.ttm.duplicated().any():
            raise ValueError("fit comparison requires one expiry per tenor")
        frame = report.table("expiries").merge(mapping, on="expiry_id", validate="one_to_one")
        frames.append(frame.set_index("ttm").sort_index())
    before, after = frames
    if not before.index.equals(after.index) or not before.index.isin(tenors).all():
        raise ValueError("fit reports must use matching constant tenors on the surface grid")
    result = pd.DataFrame(index=before.index)
    fields = (
        "n_quotes",
        "n_iv",
        "n_price",
        "iv_rmse",
        "price_rmse",
        "iv_weighted_rmse",
        "price_weighted_rmse",
        "price_inside_bid_ask_fraction",
    )
    for field in fields:
        left, right = before[field].astype(float), after[field].astype(float)
        result[f"previous_{field}"] = left
        result[f"current_{field}"] = right
        result[f"{field}_change"] = right - left
        result[f"{field}_status"] = np.where(
            np.isfinite(left) & np.isfinite(right), "pass", "not_checked"
        )
    return result.reset_index()
