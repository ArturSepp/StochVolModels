"""Independent checks of the raw conditional estimator and stage-0 boundaries."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels.utils.mc_payoffs import compute_conditional_mc_payoff


@pytest.mark.parametrize("kind", ["C", "P", "IC", "IP"])
def test_conditional_prices_match_independent_gaussian_quadrature(kind):
    """Integrate actual payoffs, splitting at the exercise threshold (no Black formula)."""
    forward, strike, discount = 1.3, 1.1, 0.97
    ratios, variances = np.array([-0.2, 0.1, 0.4]), np.array([0.04, 0.12, 0.3])
    references = []
    for ratio, variance in zip(ratios, variances):
        mean = np.log(forward) + ratio - variance / 2.0
        threshold = (np.log(strike) - mean) / np.sqrt(variance)

        def integrand(z):
            spot = np.exp(mean + np.sqrt(variance) * z)
            payoff = max(spot - strike, 0) if kind in ("C", "IC") else max(strike - spot, 0)
            if kind.startswith("I"):
                payoff /= spot
            return payoff * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)

        references.append(quad(integrand, -12, threshold, epsabs=1e-12)[0]
                          + quad(integrand, threshold, 12, epsabs=1e-12)[0])
    values = discount * np.array(references)
    prices, errors = compute_conditional_mc_payoff(
        ratios, variances, forward, np.array([strike]), np.array([kind]), discount
    )
    np.testing.assert_allclose(prices, [values.mean()], rtol=2e-11, atol=1e-12)
    np.testing.assert_allclose(errors, [values.std(ddof=1) / np.sqrt(3)], rtol=2e-11)


def test_raw_forward_and_error_are_not_normalized():
    """Three deterministic conditional spots give a hand-computed mean and standard error."""
    ratios = np.log(np.array([1.0, 2.0, 4.0]))
    prices, errors = compute_conditional_mc_payoff(
        ratios, np.zeros(3), 1.0, np.array([1.0, 1.0]), np.array(["C", "P"]), 0.9
    )
    np.testing.assert_allclose(prices, [1.2, 0.0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(errors, [0.9 * np.sqrt(7) / 3, 0.0], rtol=1e-14)


@pytest.mark.parametrize("variance", [np.array([-0.1, 0.2]), np.array([np.nan, 0.2])])
def test_conditional_estimator_rejects_invalid_variance(variance):
    with pytest.raises(ValueError):
        compute_conditional_mc_payoff(
            np.zeros(2), variance, 1.0, np.array([1.0]), np.array(["C"])
        )


@pytest.mark.parametrize("backbone", [1.2, np.nan])
def test_rough_model_rejects_unimplemented_backbone(backbone):
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([0.25]), ids=np.array(["3m"]), strikes=np.array([1.0])
    )
    params = LogSvParams(vol_backbone=pd.Series([backbone], index=[0.25]))
    with pytest.raises(NotImplementedError, match="backbone"):
        LogSVPricer().model_mc_price_chain(chain, params, use_rough_mc=True, seed=1)


def test_rough_model_rejects_unimplemented_inverse_measure():
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([0.25]), ids=np.array(["3m"]), strikes=np.array([1.0])
    )
    with pytest.raises(NotImplementedError, match="money-market"):
        LogSVPricer().model_mc_price_chain(
            chain, LogSvParams(), is_spot_measure=False, use_rough_mc=True, seed=1
        )


def test_high_level_split_step_default_grid_prices_constant_volatility():
    """Public routing and its default grid recover a deterministic-volatility Black price."""
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([0.25]), ids=np.array(["3m"]), strikes=np.array([1.0])
    )
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0,
                         weights=np.array([1.0]), nodes=np.array([0.0]))
    prices, errors = LogSVPricer().model_mc_price_chain(
        chain, params, nb_path=4, use_rough_mc=True, rough_scheme="split_step", seed=1
    )
    np.testing.assert_allclose(prices[0], [2 * ndtr(0.05) - 1], rtol=1e-12)
    np.testing.assert_allclose(errors[0], [0.0], atol=1e-14)
