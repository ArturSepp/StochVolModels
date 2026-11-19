"""Independent numerical checks for the repository-only SPX feasibility study."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

STUDY = Path(__file__).resolve().parents[3] / "papers/risk_premia_gmm/replication/spx_feasibility.py"
if not STUDY.exists():
    pytest.skip("Repository-only SPX study is absent", allow_module_level=True)
spec = importlib.util.spec_from_file_location("spx_feasibility", STUDY)
spx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spx)
pytestmark = pytest.mark.repository_only


def parameters(n):
    """Asymmetric mixture with distinct drifts and variances."""
    return np.r_[np.linspace(-.4, .6, n-1), np.linspace(-.3, .2, n-1),
                 np.log(np.linspace(.15, .55, n))]


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("ttm", [30/365, .25, 1.])
def test_constraints_jacobian_and_core_price(n, ttm):
    """Compare analytical derivatives with differences and prices with core BSM mixture."""
    from stochvolmodels.data.option_chain import OptionSlice
    from stochvolmodels.pricers.gmm_pricer import GmmParams, GmmTerminalModel

    theta = parameters(n)
    strikes = np.array([.8, .95, 1., 1.05, 1.2])
    signs = np.array([-1., -1., 1., 1., 1.])
    prices, jac = spx.mixture_prices(theta, n, ttm, strikes, signs, True)
    step = 1e-5
    numerical = np.column_stack([
        (spx.mixture_prices(theta + np.eye(len(theta))[i]*step, n, ttm, strikes, signs)
         - spx.mixture_prices(theta - np.eye(len(theta))[i]*step, n, ttm, strikes, signs))/(2*step)
        for i in range(len(theta))
    ])
    np.testing.assert_allclose(jac, numerical, atol=2e-10, rtol=2e-6)
    w, m, v = spx.unpack(theta, n, ttm)
    np.testing.assert_allclose(w.sum(), 1., atol=1e-14)
    np.testing.assert_allclose(np.sum(w*np.exp(m+.5*v)), 1., atol=1e-14)
    params = GmmParams(w, m/ttm, np.sqrt(v/ttm), ttm)
    option_slice = OptionSlice(ttm, 100., 100*strikes, np.where(signs>0, "C", "P"),
                               "synthetic", discfactor=np.exp(-.04*ttm))
    core = GmmTerminalModel(params).price_european(option_slice)
    np.testing.assert_allclose(core, prices*100*np.exp(-.04*ttm), rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("tilt", [0., 1., 3., 4.])
def test_erp_against_quadrature_and_nonzero_rate(tilt):
    """Integrate the tilted density independently; detect missing rate/annualization factors."""
    theta, n, ttm, rate = parameters(3), 3, .75, .065
    w, m, v = spx.unpack(theta, n, ttm)
    def density(x):
        return sum(wi*norm.pdf(x, mi, np.sqrt(vi)) for wi, mi, vi in zip(w, m, v))
    denominator = quad(lambda x: np.exp(tilt*x)*density(x), -12, 12, epsabs=1e-11)[0]
    numerator = quad(lambda x: np.exp((tilt+1)*x)*density(x), -12, 12, epsabs=1e-11)[0]
    expected = np.exp(rate*ttm)*(numerator/denominator-1)/ttm
    np.testing.assert_allclose(spx.erp(theta, n, ttm, rate, tilt), expected, atol=1e-11)


@pytest.mark.parametrize("ttm", [30/365, .5, 1.])
def test_single_gaussian_closed_form(ttm):
    """Coincident mixture states must give the lognormal ERP exactly."""
    vol, rate, tilt = .23, .05, 3.
    theta = np.r_[0., 0., np.log([vol, vol])]
    expected = np.exp(rate*ttm)*np.expm1(tilt*vol*vol*ttm)/ttm
    np.testing.assert_allclose(spx.erp(theta, 2, ttm, rate, tilt), expected, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_synthetic_smile_fit_and_erp(n):
    """Recover identifiable synthetic option values and ERP with the true seed included."""
    theta, ttm = parameters(n), .5
    strikes = spx.MONEYNESS
    signs = np.where(strikes>=1., 1., -1.)
    prices = spx.mixture_prices(theta, n, ttm, strikes, signs)
    vols = spx.implied_vols(prices, strikes, ttm, signs)
    task = {"date": "2020-01-01", "tenor": "6m", "ttm": ttm, "components": n,
            "rate": .03, "dividend": .03, "vols": vols, "starts": 2,
            "extra_start": theta.tolist(), "seed": 17}
    record, rows = spx.fit_slice(task)
    assert record["accepted"]
    assert record["iv_rmse_bp"] < 1e-5
    assert record["martingale_error"] < 1e-13
    np.testing.assert_allclose(record["erp_lambda3"], spx.erp(theta, n, ttm, .03, 3.), atol=1e-8)
    assert len(rows) == 3


def test_dates_use_available_observations():
    """Do not invent month-end dates or backfill missing surfaces."""
    df = pd.DataFrame({"x": [1., 2., 3.]}, index=pd.to_datetime(["2024-01-30", "2024-02-28", "2024-03-12"]))
    assert spx.selected_dates(df).equals(df.index)


def test_tail_probabilities_against_physical_density():
    """Validate tails by independent integration of the tilted mixture."""
    theta, n, ttm, tilt = parameters(2), 2, .5, 3.
    w, m, v = spx.unpack(theta, n, ttm)
    normalizer = np.exp(spx.log_mgf(theta, n, ttm, tilt))
    def physical(x):
        return np.exp(tilt*x)/normalizer*sum(wi*norm.pdf(x, mi, np.sqrt(vi))
                                           for wi, mi, vi in zip(w, m, v))
    tails = spx.tail_diagnostics(theta, n, ttm, tilt, .8, 1.2)
    np.testing.assert_allclose(tails["p_below_grid"], quad(physical, -10, np.log(.8))[0], atol=1e-10)
    np.testing.assert_allclose(tails["p_above_grid"], quad(physical, np.log(1.2), 10)[0], atol=1e-10)
