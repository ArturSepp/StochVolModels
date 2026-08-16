import numpy as np
from scipy.stats import norm

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.heston_pricer import (
    HestonParams,
    HestonPricer,
    compute_heston_mgf_grid,
)
from stochvolmodels.utils.funcs import set_seed


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


def test_heston_constant_variance_limit_matches_independent_bsm_reference() -> None:
    chain = _paired_chain()
    params = HestonParams(
        v0=0.04,
        theta=0.04,
        kappa=3.0,
        rho=-0.4,
        volvol=1.0e-4,
    )

    prices = np.asarray(HestonPricer().price_chain(chain, params)[0])
    expected = _reference_bsm_prices(chain, vol=0.2)
    calls, puts = prices[:3], prices[3:]
    strikes = np.array([0.9, 1.0, 1.1])

    np.testing.assert_allclose(prices, expected, rtol=0.0, atol=1.0e-6)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)
    assert np.all(np.diff(calls) <= 0.0)
    assert np.all(np.diff(puts) >= 0.0)
    assert np.all(calls <= chain.discfactors[0] * chain.forwards[0])
    assert np.all(puts <= chain.discfactors[0] * strikes)
    np.testing.assert_allclose(
        calls - puts,
        chain.discfactors[0] * (chain.forwards[0] - strikes),
        rtol=0.0,
        atol=2.0e-8,
    )


def test_heston_mgf_normalization_and_semigroup() -> None:
    kwargs = dict(v0=0.04, theta=0.05, kappa=2.0, volvol=0.4, rho=-0.5)
    roots = np.array([0.0 + 0.0j, -1.0 + 0.0j])
    zeros = np.zeros_like(roots)

    root_values, _, _ = compute_heston_mgf_grid(
        ttm=0.5, phi_grid=roots, psi_grid=zeros, **kwargs
    )
    np.testing.assert_allclose(root_values, 0.0, rtol=0.0, atol=1.0e-14)

    phi = np.array([-0.5 + 0.0j, -0.5 + 0.7j, -0.5 + 1.4j])
    psi = np.zeros_like(phi)
    direct, _, _ = compute_heston_mgf_grid(
        ttm=0.5, phi_grid=phi, psi_grid=psi, **kwargs
    )
    _, a_t0, b_t0 = compute_heston_mgf_grid(
        ttm=0.2, phi_grid=phi, psi_grid=psi, **kwargs
    )
    chained, _, _ = compute_heston_mgf_grid(
        ttm=0.3,
        phi_grid=phi,
        psi_grid=psi,
        a_t0=a_t0,
        b_t0=b_t0,
        **kwargs,
    )

    assert np.all(np.isfinite(direct))
    np.testing.assert_allclose(chained, direct, rtol=0.0, atol=2.0e-13)


def test_heston_analytic_prices_lie_inside_seeded_mc_confidence_intervals() -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.98,
        id="3m",
    )
    params = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    assert 2.0 * params.kappa * params.theta - params.volvol**2 > 0.0

    analytic = np.asarray(HestonPricer().price_chain(chain, params)[0])
    set_seed(123)
    mc_prices, mc_errors = HestonPricer().model_mc_price_chain(
        chain, params, nb_path=40_000
    )
    mc_prices = np.asarray(mc_prices[0])
    mc_errors = np.asarray(mc_errors[0])

    assert np.all(np.isfinite(mc_prices))
    assert np.all(np.isfinite(mc_errors))
    assert np.all(mc_errors > 0.0)
    assert np.all(np.abs(analytic - mc_prices) <= 4.0 * mc_errors)
