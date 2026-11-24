"""Variance-swap calibration contracts in volatility units at every expiry."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from stochvolmodels import LogSvParams, get_btc_test_chain_data
from stochvolmodels.pricers.logsv import vol_moments_ode as moments


def _integrated_loaded_variance(params, loadings):
    """Integrate instantaneous second moments, independently of the integrated-moment formula."""
    def second_moment(t):
        centered = moments.compute_analytic_vol_moments(params, t=t, n_terms=4)
        return centered[1] + 2 * params.theta * centered[0] + params.theta**2

    ends = np.asarray(loadings.index, dtype=float)
    intervals = [
        eta**2 * quad(second_moment, start, end, epsabs=1e-12, epsrel=1e-12)[0]
        for start, end, eta in zip(np.r_[0.0, ends[:-1]], ends, loadings)
    ]
    return np.cumsum(intervals)


def test_backbone_recovers_known_interval_volatilities():
    """A deterministic volatility state gives an exact, closed-form independent reference."""
    params = LogSvParams(sigma0=0.8, theta=0.8, kappa1=2.0, kappa2=2.0, beta=0, volvol=0)
    ttms = np.array([0.02, 0.06, 0.25, 0.75])
    expected = np.array([1.3, 0.75, 1.15, 0.85])
    integrated = params.sigma0**2 * np.cumsum(expected**2 * np.diff(np.r_[0.0, ttms]))
    targets = pd.Series(np.sqrt(integrated / ttms), index=ttms)
    original = targets.copy()

    actual = moments.fit_model_vol_backbone_to_varswaps(params, targets)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_integrated_loaded_variance(params, actual), integrated, atol=1e-12)
    pd.testing.assert_series_equal(targets, original)


@pytest.mark.parametrize("beta,volvol", [(0.0, 0.921487415907961), (0.24, 1.14)])
def test_btc_backbone_matches_every_expiry(beta, volvol):
    """Each quoted maturity reprices after numerical integration of the instantaneous moments."""
    chain = get_btc_test_chain_data()
    params = LogSvParams(sigma0=0.85, theta=0.85, kappa1=2.21, kappa2=2.18,
                         beta=beta, volvol=volvol)
    targets = chain.get_slice_varswap_strikes(floor_with_atm_vols=True)
    actual = moments.fit_model_vol_backbone_to_varswaps(params, targets)
    reproduced = np.sqrt(_integrated_loaded_variance(params, actual) / chain.ttms)
    np.testing.assert_allclose(reproduced, targets, rtol=2e-10, atol=1e-12)


@pytest.mark.repository_only
@pytest.mark.parametrize("scale", [0.75, 1.5])
def test_paper_atm_floor_uses_volatility_units(monkeypatch, scale):
    """Volatilities above one expose the extra erroneous ATM-squared floor in the driver."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "papers/forward_var").is_dir():
        pytest.skip("requires the paper checkout")
    monkeypatch.syspath_prepend(str(root))
    from papers.forward_var.calibrate_forward_var import varswap_fit

    original = get_btc_test_chain_data()
    chain = replace(original, bid_ivs=tuple(scale * v for v in original.bid_ivs),
                    ask_ivs=tuple(scale * v for v in original.ask_ivs))
    raw = chain.get_slice_varswap_strikes(floor_with_atm_vols=False)
    expected = np.maximum(raw, chain.get_chain_atm_vols())
    params = varswap_fit(chain)
    reproduced = np.sqrt(_integrated_loaded_variance(params, params.vol_backbone) / chain.ttms)
    np.testing.assert_allclose(reproduced, expected, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("times,vols", [
    ([], []), ([0.0], [0.8]), ([-0.1], [0.8]), ([0.2, 0.1], [0.8, 0.8]),
    ([0.1, 0.1], [0.8, 0.8]), ([np.nan], [0.8]), ([np.inf], [0.8]),
    ([0.1], [np.nan]), ([0.1], [np.inf]), ([0.1], [0.0]), ([0.1], [-0.8]),
    ([0.1, 0.2], [1.0, 0.5]),
])
def test_backbone_rejects_unfittable_market_targets(times, vols):
    """Invalid maturities and nonpositive forward variance must not silently return unit loading."""
    params = LogSvParams(sigma0=0.8, theta=0.8, kappa1=2.0, kappa2=2.0)
    with pytest.raises(ValueError):
        moments.fit_model_vol_backbone_to_varswaps(params, pd.Series(vols, index=times))


@pytest.mark.parametrize("qvar", [np.nan, -1.0, 0.0])
def test_backbone_rejects_invalid_model_moments(monkeypatch, qvar):
    """The calibration objective recognizes invalid model trials as numerical failures."""
    monkeypatch.setattr(moments, "compute_analytic_qvar", lambda **kwargs: qvar)
    with pytest.raises(FloatingPointError):
        moments.fit_model_vol_backbone_to_varswaps(LogSvParams(), pd.Series([0.8], index=[0.1]))
