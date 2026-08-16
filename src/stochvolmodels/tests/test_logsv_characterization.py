import numpy as np
import pytest
from numba.typed import List
from scipy.stats import norm
from stochvolmodels.pricers.logsv import affine_expansion as afe
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv_pricer import (
    LogSVPricer,
    get_randoms_for_chain_valuation,
    logsv_mc_chain_pricer_fixed_randoms,
    simulate_logsv_x_vol_terminal,
)


def _paired_chain() -> OptionChain:
    strikes = np.array([0.9, 1.0, 1.1])
    return OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.concatenate((strikes, strikes)),
        optiontypes=np.array(["C", "C", "C", "P", "P", "P"]),
        discfactor=0.98,
        id="3m",
    )


def _representative_params() -> LogSvParams:
    return LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=-0.3,
        volvol=0.4,
    )


def _reference_bsm_prices(chain: OptionChain, vol: float) -> np.ndarray:
    strikes = np.asarray(chain.strikes_ttms[0])
    optiontypes = np.asarray(chain.optiontypes_ttms[0])
    forward = chain.forwards[0]
    total_vol = vol * np.sqrt(chain.ttms[0])
    d1 = np.log(forward / strikes) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    undiscounted = np.where(
        optiontypes == "C",
        forward * norm.cdf(d1) - strikes * norm.cdf(d2),
        strikes * norm.cdf(-d2) - forward * norm.cdf(-d1),
    )
    return chain.discfactors[0] * undiscounted


def test_logsv_constant_volatility_limit_matches_independent_bsm_reference() -> None:
    chain = _paired_chain()
    params = LogSvParams(
        sigma0=0.2,
        theta=0.2,
        kappa1=4.0,
        kappa2=20.0,
        beta=0.0,
        volvol=1.0e-3,
    )

    prices = np.asarray(LogSVPricer().price_chain(chain, params)[0])
    expected = _reference_bsm_prices(chain, vol=0.2)
    calls, puts = prices[:3], prices[3:]
    strikes = np.array([0.9, 1.0, 1.1])

    np.testing.assert_allclose(prices, expected, rtol=0.0, atol=1.2e-5)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)
    assert np.all(np.diff(calls) <= 0.0)
    assert np.all(np.diff(puts) >= 0.0)
    np.testing.assert_allclose(
        calls - puts,
        chain.discfactors[0] * (chain.forwards[0] - strikes),
        rtol=0.0,
        atol=2.0e-8,
    )


@pytest.mark.parametrize(
    ("is_spot_measure", "roots"),
    [(True, [0.0, -1.0]), (False, [0.0, 1.0])],
)
def test_logsv_mgf_normalization_roots(
    is_spot_measure: bool, roots: list[float]
) -> None:
    params = _representative_params()
    phi = np.asarray(roots, dtype=np.complex128)
    zeros = np.zeros_like(phi)

    coefficients, log_mgf = afe.compute_logsv_a_mgf_grid(
        ttm=0.25,
        phi_grid=phi,
        psi_grid=zeros,
        theta_grid=zeros,
        is_spot_measure=is_spot_measure,
        **params.to_dict(),
    )

    assert np.all(np.isfinite(coefficients))
    np.testing.assert_allclose(log_mgf, 0.0, rtol=0.0, atol=1.0e-14)


def test_logsv_analytic_prices_and_martingale_match_fixed_random_mc() -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.98,
        id="3m",
    )
    params = _representative_params()
    nb_path = 40_000
    nb_steps = 91
    dt = chain.ttms[0] / nb_steps
    rng = np.random.default_rng(123)
    w0s = List([rng.standard_normal((nb_steps, nb_path))])
    w1s = List([rng.standard_normal((nb_steps, nb_path))])
    dts = List([dt])

    analytic = np.asarray(LogSVPricer().price_chain(chain, params)[0])
    mc_prices, mc_errors = logsv_mc_chain_pricer_fixed_randoms(
        ttms=chain.ttms,
        forwards=chain.forwards,
        discfactors=chain.discfactors,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        W0s=w0s,
        W1s=w1s,
        dts=dts,
        v0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        vol_backbone_etas=params.get_vol_backbone_etas(chain.ttms),
    )
    mc_prices = np.asarray(mc_prices[0])
    mc_errors = np.asarray(mc_errors[0])

    assert np.all(np.isfinite(mc_prices))
    assert np.all(mc_errors > 0.0)
    assert np.all(np.abs(analytic - mc_prices) <= 4.0 * mc_errors)

    x, sigma, qvar = simulate_logsv_x_vol_terminal(
        ttm=chain.ttms[0],
        x0=np.zeros(nb_path),
        sigma0=np.full(nb_path, params.sigma0),
        qvar0=np.zeros(nb_path),
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        nb_path=nb_path,
        W0=w0s[0],
        W1=w1s[0],
        dt=dt,
    )
    terminal_spot = np.exp(x)
    spot_error = np.std(terminal_spot, ddof=1) / np.sqrt(nb_path)

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(sigma))
    assert np.all(np.isfinite(qvar))
    assert np.all(sigma > 0.0)
    assert np.all(qvar >= 0.0)
    assert abs(np.mean(terminal_spot) - 1.0) <= 4.0 * spot_error


def test_logsv_martingale_and_fourth_moment_constraints_hold() -> None:
    params = _representative_params()
    eigenvalues = np.linalg.eigvals(params.get_vol_moments_lambda(n_terms=4))

    assert params.kappa2 - params.beta > 0.0
    assert params.kappa2 - 2.0 * params.beta > 0.0
    assert params.kappa - 1.5 * params.vartheta2 > 0.0
    assert np.all(np.isfinite(eigenvalues))
    assert np.all(np.real(eigenvalues) < 0.0)


def test_logsv_fixed_random_generator_replays_the_same_seed() -> None:
    kwargs = dict(ttms=np.array([0.25]), nb_path=8, nb_steps_per_year=12, seed=7)
    first_w0, first_w1, first_dt = get_randoms_for_chain_valuation(**kwargs)
    second_w0, second_w1, second_dt = get_randoms_for_chain_valuation(**kwargs)

    np.testing.assert_array_equal(first_w0[0], second_w0[0])
    np.testing.assert_array_equal(first_w1[0], second_w1[0])
    np.testing.assert_array_equal(first_dt, second_dt)


def test_logsv_fixed_random_generator_does_not_mutate_global_rng() -> None:
    np.random.seed(91)
    expected = np.random.random()
    np.random.seed(91)

    get_randoms_for_chain_valuation(
        ttms=np.array([0.25]), nb_path=8, nb_steps_per_year=12, seed=7
    )

    np.testing.assert_allclose(np.random.random(), expected, rtol=0.0, atol=0.0)
