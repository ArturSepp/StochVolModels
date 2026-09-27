"""Canonical script of docs/factor_hjm_stochastic_volatility.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. A three-factor Nelson-Siegel HJM model with a log-normal stochastic
volatility driver prices swaptions by the first-order affine expansion of Sepp and Rakhmonov (2025),
and a Monte Carlo simulation of the full dynamics checks the prices. The parameters are the base
scenario of Table 2 of the article. ``stochvolmodels.pricers.factor_hjm`` is an experimental
research surface. Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

from stochvolmodels import ExpansionOrder
from stochvolmodels.pricers.factor_hjm.factor_hjm_pricer import calc_mc_vols
from stochvolmodels.pricers.factor_hjm.rate_factor_basis import NelsonSiegel
from stochvolmodels.pricers.factor_hjm.rate_logsv_params import (MultiFactRateLogSvParams,
                                                                 TermStructure)
from stochvolmodels.pricers.factor_hjm.rate_logsv_pricer import logsv_chain_de_pricer
from stochvolmodels.utils.rate_core import generate_ttms_grid, get_default_swap_term_structure

TENORS = np.array([2.0, 5.0, 10.0])  # benchmark swap tenors, also the Nelson-Siegel key terms
EXPIRY = 2.0
# historical correlation of the 2y, 5y and 10y yields used in the article
CORRELATION = np.array([[1.0, 0.99, 0.97], [0.99, 1.0, 0.98], [0.97, 0.98, 1.0]])


def model_params(beta_scale: float = 1.0) -> MultiFactRateLogSvParams:
    """Base scenario of RDR Table 2, with the volatility betas multiplied by beta_scale."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 5.0])  # piecewise-constant parameters up to 5y
    return MultiFactRateLogSvParams(
        sigma0=1.0, theta=1.0, kappa1=0.25, kappa2=0.5,
        beta=TermStructure.create_multi_fact_from_vec(times, beta_scale * np.full(3, 0.2)),
        volvol=TermStructure.create_from_scalar(times, 0.2),
        A=np.array([0.01, 0.01, 0.01]),  # normal volatilities of the 2y, 5y and 10y yields
        R=CORRELATION, basis=NelsonSiegel(meanrev=0.55, key_terms=TENORS),
        ccy="USD", vol_interpolation="BY_YIELD")


def forward_swap_rates(params: MultiFactRateLogSvParams, expiry: float = EXPIRY) -> np.ndarray:
    """Forward swap rates of the initial curve, with the factors at zero, one per tenor."""
    x0 = np.zeros((1, params.basis.get_nb_factors()))
    y0 = np.zeros((1, params.basis.get_nb_aux_factors()))
    return np.array([np.ravel(params.basis.swap_rate(
        t=expiry, ts_sw=get_default_swap_term_structure(expiry=expiry, tenor=tenor), x=x0, y=y0,
        ccy=params.ccy)[0])[0] for tenor in TENORS])


def strike_grid(params: MultiFactRateLogSvParams, expiry: float = EXPIRY) -> list:
    """Seven strikes from 150 bp below to 150 bp above the forward, for each tenor."""
    return [f + np.linspace(-0.015, 0.015, 7) for f in forward_swap_rates(params, expiry)]


def expansion_smiles(params: MultiFactRateLogSvParams, expiry: float = EXPIRY) -> list:
    """Normal implied volatilities of payer swaptions by the first-order affine expansion."""
    ttms = np.array([expiry])
    strikes = strike_grid(params, expiry)
    _, vols = logsv_chain_de_pricer(
        params=params, t_grid=generate_ttms_grid(ttms), ttms=ttms,
        forwards=[np.array([f]) for f in forward_swap_rates(params, expiry)],
        strikes_ttms=[[k] for k in strikes], optiontypes_ttms=[np.repeat("C", strikes[0].size)],
        expansion_order=ExpansionOrder.FIRST)
    return [np.asarray(vol[0]) for vol in vols]


def monte_carlo_smiles(params: MultiFactRateLogSvParams, expiry: float = EXPIRY,
                       nb_path: int = 20000) -> tuple:
    """Monte Carlo normal volatilities with their 95% bounds, simulated without drift freezing."""
    strikes = strike_grid(params, expiry)
    _, mid, up, down = calc_mc_vols(
        basis_type="NELSON-SIEGEL", params=params, ttm=expiry, tenors=TENORS,
        forwards=[np.array([f]) for f in forward_swap_rates(params, expiry)],
        strikes_ttms=[[k] for k in strikes], optiontypes=np.repeat("C", strikes[0].size),
        is_annuity_measure=False, nb_path=nb_path)
    return [np.asarray(v) for v in mid], [np.asarray(v) for v in up], [np.asarray(v) for v in down]


def annuity_kappa2(params: MultiFactRateLogSvParams, expiry: float = EXPIRY) -> np.ndarray:
    """Smallest quadratic mean reversion kappa2 - beta2(t) under each annuity measure, Eq. (33)."""
    t_grid = generate_ttms_grid(np.array([expiry]))
    return np.array([np.min(params.transform_QA_params(expiry=expiry, tenor=tenor,
                                                       t_grid=t_grid)[3]) for tenor in TENORS])


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    FACTOR_VOLS = 1
    FORWARDS = 2
    ANNUITY_MEASURE = 3
    SKEWS = 4
    MONTE_CARLO = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.FACTOR_VOLS:
        params = model_params()
        basis_at_key_terms = params.basis.get_matrix_B()
        factor_vols = params.C[0]
        # C = B^-1 diag(a) chol(R) reproduces the covariance of the benchmark yields, Eq. (130)
        yield_cov = basis_at_key_terms @ factor_vols @ factor_vols.T @ basis_at_key_terms.T
        np.testing.assert_allclose(yield_cov, 1e-4 * CORRELATION, atol=1e-14)

    elif local == Locals.FORWARDS:
        # the "USD" curve of the module is a flat 4.3% continuously compounded zero curve
        np.testing.assert_allclose(forward_swap_rates(model_params()), np.exp(0.043) - 1.0,
                                   rtol=1e-12)

    elif local == Locals.ANNUITY_MEASURE:
        # the measure change moves kappa2 by beta2(t); condition (33) holds with a wide margin
        kappa2 = {scale: annuity_kappa2(model_params(scale)) for scale in (-4.0, -2.0, 1.0, 4.0)}
        print(kappa2)
        assert all(np.all(v > 0.4) for v in kappa2.values())
        np.testing.assert_allclose(kappa2[-4.0], [0.4704, 0.4551, 0.4288], atol=1e-4)
        assert all(model_params(-2.0).check_QA_kappa2(expiry=EXPIRY, tenor=t) for t in TENORS)

    elif local == Locals.SKEWS:
        smiles = {scale: expansion_smiles(model_params(scale)) for scale in (-2.0, 0.0, 1.0)}
        print({s: [np.round(v * 1e4, 2) for v in vols] for s, vols in smiles.items()})
        # without a volatility beta the smile is symmetric; its sign sets the direction of the skew
        np.testing.assert_allclose(smiles[0.0][0], smiles[0.0][0][::-1], atol=1e-7)
        assert all(np.all(np.diff(v) > 0) for v in smiles[1.0])
        assert all(np.all(np.diff(v) < 0) for v in smiles[-2.0])
        np.testing.assert_allclose(1e4 * smiles[1.0][0][[0, 3, -1]], [96.94, 105.58, 116.48],
                                   atol=0.01)
        np.testing.assert_allclose(1e4 * smiles[-2.0][0][[0, 3, -1]], [126.62, 104.77, 90.40],
                                   atol=0.01)

    elif local == Locals.MONTE_CARLO:
        inside, gaps = {}, {}
        for scale in (1.0, -2.0):
            expansion = expansion_smiles(model_params(scale))
            mid, up, down = monte_carlo_smiles(model_params(scale))
            inside[scale] = sum(int(np.sum((e >= d) & (e <= u)))
                                for e, u, d in zip(expansion, up, down))
            gaps[scale] = max(float(np.max(np.abs(e - m))) for e, m in zip(expansion, mid))
        print(inside, gaps)
        # 21 options per scenario; with the betas doubled and reversed, the expansion drifts
        # above the simulation at low strikes
        assert inside == {1.0: 21, -2.0: 18}
        np.testing.assert_allclose(1e4 * np.array([gaps[1.0], gaps[-2.0]]), [2.58, 3.75], atol=0.01)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
