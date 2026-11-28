"""Fit a time-homogeneous local-volatility smile to five six-month equity quotes.

Inputs use native forwards, spot moneyness and annualized decimal Black volatilities.
The forward-density PDE handles discounting and vanilla payoff projection.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
import vanilla_option_pricers as vop

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.surface_checks import check_call_slice
from stochvolmodels.models.lsv.local_vol import LocalVolSurface
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer, LocalVolPdeExpiry


@dataclass(frozen=True)
class SparseLocalVolFit:
    """Calibrated law and repricing errors in basis points of absolute Black IV."""

    surface: LocalVolSurface
    time_grid: np.ndarray
    residual_iv_bp: np.ndarray
    refined_iv_bp: np.ndarray
    condition_number: float
    active_bounds: np.ndarray
    density_diagnostics: LocalVolPdeExpiry
    status: str
    grid_size: int
    log_width: float
    spot: float
    native_forward: float
    native_discount: float
    spatial_extrapolation: str = "flat_endpoint_volatility"
    time_extrapolation: str = "constant_six_month_row"


def fit_sparse_local_vol(
    option_chain: OptionChain, *, spot: float, grid_size: int = 401, log_width: float = 2.0
) -> SparseLocalVolFit:
    """Fit the five 80/90/100/110/120 percent spot strikes from one six-month chain.

    Use seven calendar days per full propagation step on ACT/365 and one final stub.
    The fitted row extends flat through time and clamps volatility at its spatial endpoints.
    """
    if len(option_chain.ttms) != 1 or not np.isfinite(spot) or spot <= 0:
        raise ValueError("require a single six-month expiry and positive spot")
    t = float(option_chain.ttms[0])
    if not 0.45 <= t <= 0.55:
        raise ValueError("expected a six-month tenor in years")
    forward = option_chain.forwards[0]
    strikes = np.asarray(option_chain.strikes_ttms[0])
    if strikes.shape != (5,) or not np.allclose(
        strikes / spot, [0.8, 0.9, 1, 1.1, 1.2], atol=1e-8, rtol=0
    ):
        raise ValueError("expected ordered strikes at 80/90/100/110/120 percent of spot")
    mids = option_chain.get_mid_vols()
    if (
        mids is None
        or np.shape(mids[0]) != (5,)
        or not np.all(np.isfinite(mids[0]) & (np.asarray(mids[0]) > 0))
    ):
        raise ValueError("five finite positive decimal Black IVs are required")
    if grid_size < 31 or grid_size % 2 != 1 or not np.isfinite(log_width) or log_width <= 0:
        raise ValueError("use an odd solver grid >= 31 with positive log_width")
    k = strikes / forward
    if np.max(np.abs(np.log(k))) >= log_width:
        raise ValueError("strikes must lie inside the PDE domain")
    iv = np.asarray(mids[0])
    kinds = np.where(k < 1, "P", "C")
    targets = vop.compute_bsm_vanilla_slice_prices(t, 1.0, k, iv, kinds)
    calls = targets + np.where(kinds == "P", 1.0 - k, 0.0)
    checks = check_call_slice(k, calls, forward=1.0, discount=1.0, price_unit="unit forward").table(
        "checks"
    )
    failures = checks.loc[checks["status"] == "fail", ["check_id", "location"]]
    if not failures.empty:
        raise ValueError(f"input call arbitrage: {failures.to_dict('records')}")
    vegas = vop.compute_bsm_slice_vegas(t, 1.0, k, iv)
    if not np.all(np.isfinite(vegas) & (vegas > 1e-12)):
        raise ValueError("insufficient Black vega at a fitted strike")
    chain = OptionChain(
        ttms=np.array([t]),
        forwards=np.ones(1),
        discfactors=np.ones(1),
        strikes_ttms=[k],
        optiontypes_ttms=[kinds],
        ids=np.array(["6M"]),
    )
    space = Grid1D(np.log(k))
    mesh = np.r_[np.arange(0.0, t, 7 / 365), t]
    coarse = LocalVolPdePricer(grid=Grid1D.uniform(-log_width, log_width, grid_size))

    def surface(nodes: np.ndarray) -> LocalVolSurface:
        return LocalVolSurface(np.array([t]), space, nodes[None, :])

    def residual(nodes: np.ndarray) -> np.ndarray:
        return (coarse.price_chain(chain, surface(nodes), time_grid=mesh)[0] - targets) / vegas

    fit = least_squares(
        residual, np.clip(iv, 0.01, 2.0), bounds=(0.01, 2.0), gtol=1e-11, ftol=1e-11, xtol=1e-11
    )
    law = surface(fit.x)
    base = coarse.price_chain_detailed(chain, law, time_grid=mesh)
    fine = LocalVolPdePricer(grid=Grid1D.uniform(-log_width, log_width, 2 * grid_size - 1))
    detail = fine.price_chain_detailed(chain, law, time_grid=np.r_[np.arange(0.0, t, 7 / 1460), t])
    errors = 1e4 * (base.implied_vols[0] - iv)
    refined = 1e4 * (detail.implied_vols[0] - iv)
    condition = float(np.linalg.cond(fit.jac))
    status = "ok"
    if not fit.success or not np.all(np.isfinite(errors)) or np.max(np.abs(errors)) > 1:
        status = "fit_failed"
    elif np.any(fit.active_mask) or condition > 1e8:
        status = "ill_conditioned"
    elif not np.all(np.isfinite(refined)) or np.max(np.abs(refined)) > 5:
        status = "refinement_failed"
    for diagnostic in (mesh, errors, refined, fit.active_mask):
        diagnostic.setflags(write=False)
    return SparseLocalVolFit(
        surface=law,
        time_grid=mesh,
        residual_iv_bp=errors,
        refined_iv_bp=refined,
        condition_number=condition,
        active_bounds=fit.active_mask,
        density_diagnostics=detail.expiries[0],
        status=status,
        grid_size=grid_size,
        log_width=log_width,
        spot=spot,
        native_forward=float(forward),
        native_discount=float(option_chain.discfactors[0]),
    )
