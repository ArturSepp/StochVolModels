"""Parity-defined calls and the normalized forward for flat-backbone Markov LogSV.

The Lewis contour prices 1 - E[min(S_T/F,K/F)]. The real endpoint phi=-1
gives survival to T of the auxiliary volatility killed at the upper grid edge.
Volatilities are annual decimals, maturities are years and forwards are one
in the calibration. These entry points belong only to the bound experiment.
"""

import numpy as np

from stochvolmodels import LogSvParams, infer_bsm_implied_vol
from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.fourier import (
    calls_from_transform,
    gaussian_grid,
)
from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.logvol_pde import (
    PdeGrid,
    _evolve,
)


def _transform(params, maturity, phi, grid):
    levels = (params.sigma0, params.theta)
    rates = (params.kappa1, params.kappa2, params.volvol)
    if not np.all(np.isfinite((*levels, *rates, params.beta, maturity))) or maturity < 0:
        raise ValueError("Require finite parameters and nonnegative maturity.")
    if min(levels) <= 0 or min(rates) < 0:
        raise ValueError("Require positive levels and nonnegative rates and orthogonal loading.")
    if params.H != 0.5 or params.vol_backbone is not None:
        raise NotImplementedError("The study uses H=1/2 and a flat volatility backbone.")
    if maturity == 0:
        return np.ones(len(phi), dtype=complex)
    center = np.log(params.sigma0)
    y = np.linspace(
        center - grid.logvol_halfwidth, center + grid.logvol_halfwidth, grid.space_nodes
    )
    u = _evolve(
        phi,
        y,
        params.sigma0,
        params.theta,
        params.kappa1,
        params.kappa2,
        params.beta,
        params.volvol,
        maturity,
        grid.time_steps,
    )
    if not np.all(np.isfinite(u)):
        raise FloatingPointError("Nonfinite finite-difference transform.")
    return u


def parity_call_prices(
    params: LogSvParams, maturity: float, strikes: np.ndarray, grid: PdeGrid
) -> np.ndarray:
    """Return undiscounted parity calls on strikes divided by the market forward."""
    phi, weights = gaussian_grid(grid.quadrature_order, grid.cutoff)
    u = _transform(params, maturity, phi, grid)
    return calls_from_transform(u, phi, weights, maturity, params.sigma0, strikes)


def atm_parity_gap(params: LogSvParams, maturity: float, defect: float, grid: PdeGrid) -> float:
    """Return ATM parity-call IV less expectation-call IV, in percentage points."""
    parity = float(parity_call_prices(params, maturity, np.array([1.0]), grid)[0])
    vols = [
        infer_bsm_implied_vol(
            forward=1.0,
            ttm=maturity,
            strike=1.0,
            given_price=price,
            discfactor=1.0,
            optiontype="C",
        )
        for price in (parity, parity - defect)
    ]
    if not np.all(np.isfinite(vols)):
        raise FloatingPointError("The at-the-money parity gap has no implied volatility.")
    return 100.0 * (vols[0] - vols[1])


def forward_ratio(
    params: LogSvParams,
    maturity: float,
    halfwidth: float = 10.0,
    space_nodes: int = 1335,
    time_steps: int = 512,
) -> float:
    """Compute E[S_T]/S_0 using the killed auxiliary diffusion at phi=-1."""
    grid = PdeGrid(space_nodes=space_nodes, time_steps=time_steps, logvol_halfwidth=halfwidth)
    u = _transform(params, maturity, np.array([-1.0 + 0.0j]), grid)
    return float(u[0].real)


def evaluate_iv(chain, params, grid):
    """Convert parity calls and expectation puts to Black IVs with the archived discounts."""
    prices = []
    for t, forward, discount, strikes, types in zip(
        chain.ttms,
        chain.forwards,
        chain.discfactors,
        chain.strikes_ttms,
        chain.optiontypes_ttms,
        strict=True,
    ):
        if not (np.isfinite(t + forward + discount) and min(t, forward, discount) > 0):
            raise ValueError("Require positive finite maturity, forward and discount.")
        if not np.all(np.isin(types, ["C", "P"])):
            raise NotImplementedError("Only plain calls and puts are supported.")
        k = np.asarray(strikes) / forward
        calls = parity_call_prices(params, float(t), k, grid)
        is_put = np.asarray(types) == "P"
        normalized = calls - np.where(is_put, 1.0 - k, 0.0)
        intrinsic = np.maximum(np.where(is_put, k - 1.0, 1.0 - k), 0.0)
        upper = np.where(is_put, k, 1.0)
        if (
            not np.all(np.isfinite(normalized))
            or np.any(normalized < intrinsic - 1e-9)
            or np.any(normalized > upper + 1e-9)
        ):
            raise FloatingPointError("PDE quote violates a vanilla price bound.")
        prices.append(discount * forward * normalized)
    iv = np.concatenate(chain.compute_model_ivols_from_chain_data(prices))
    if not np.all(np.isfinite(iv) & (iv > 0)):
        raise FloatingPointError("A retained quote has no positive implied volatility.")
    return iv
