import numpy as np
import pytest
from scipy.stats import norm

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.heston_pricer import (
    HestonParams,
    HestonPricer,
    compute_heston_mgf_grid,
    heston_chain_pricer,
    simulate_heston_x_vol_terminal,
)
from stochvolmodels.utils.config import VariableType
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


def test_heston_chain_slice_and_vanilla_interfaces_are_consistent() -> None:
    """All public pricing entry points must delegate to the same Heston result."""
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.98,
        id="3m",
    )
    params = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    pricer = HestonPricer()

    chain_prices = np.asarray(pricer.price_chain(chain, params)[0])
    python_prices = np.asarray(
        heston_chain_pricer.py_func(
            v0=params.v0,
            theta=params.theta,
            kappa=params.kappa,
            volvol=params.volvol,
            rho=params.rho,
            ttms=chain.ttms,
            forwards=chain.forwards,
            strikes_ttms=chain.strikes_ttms,
            optiontypes_ttms=chain.optiontypes_ttms,
            discfactors=chain.discfactors,
        )[0]
    )
    slice_prices, slice_ivols = pricer.price_slice(
        params=params,
        ttm=chain.ttms[0],
        forward=chain.forwards[0],
        strikes=np.asarray(chain.strikes_ttms[0]),
        optiontypes=np.asarray(chain.optiontypes_ttms[0]),
        discfactor=chain.discfactors[0],
    )

    np.testing.assert_allclose(slice_prices, chain_prices, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(python_prices, chain_prices, rtol=0.0, atol=1.0e-13)
    assert np.all(np.isfinite(slice_ivols))
    for index, (strike, optiontype) in enumerate(
        zip(chain.strikes_ttms[0], chain.optiontypes_ttms[0])
    ):
        vanilla_price, vanilla_ivol = pricer.price_vanilla(
            params=params,
            ttm=chain.ttms[0],
            forward=chain.forwards[0],
            strike=strike,
            optiontype=optiontype,
            discfactor=chain.discfactors[0],
        )
        np.testing.assert_allclose(vanilla_price, chain_prices[index], rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(vanilla_ivol, slice_ivols[index], rtol=0.0, atol=1.0e-12)


def test_heston_python_chain_dispatch_prices_quadratic_variation() -> None:
    """The QVAR branch must use its transform grid and recover constant variance."""
    strikes = np.array([0.02, 0.03, 0.05, 0.06])
    params = HestonParams(v0=0.04, theta=0.04, kappa=2.0, rho=-0.5, volvol=1.0e-4)
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=0.04,
        strikes=strikes,
        optiontypes=np.array(["C", "C", "C", "C"]),
        id="qvar",
    )

    prices = np.asarray(
        heston_chain_pricer.py_func(
            v0=params.v0,
            theta=params.theta,
            kappa=params.kappa,
            volvol=params.volvol,
            rho=params.rho,
            ttms=chain.ttms,
            forwards=chain.forwards,
            strikes_ttms=chain.strikes_ttms,
            optiontypes_ttms=chain.optiontypes_ttms,
            discfactors=chain.discfactors,
            variable_type=VariableType.Q_VAR,
        )[0]
    )

    expected = np.maximum(params.v0 - strikes, 0.0)
    np.testing.assert_allclose(prices, expected, rtol=0.0, atol=2.1e-5)
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)
    assert np.all(np.diff(prices) <= 0.0)


def test_heston_python_chain_dispatch_rejects_sigma() -> None:
    params = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=0.04,
        strikes=np.array([0.03, 0.04, 0.05]),
        optiontypes=np.array(["P", "C", "C"]),
        id="qvar",
    )

    with pytest.raises(NotImplementedError, match="variable_type"):
        heston_chain_pricer.py_func(
            v0=params.v0,
            theta=params.theta,
            kappa=params.kappa,
            volvol=params.volvol,
            rho=params.rho,
            ttms=chain.ttms,
            forwards=chain.forwards,
            strikes_ttms=chain.strikes_ttms,
            optiontypes_ttms=chain.optiontypes_ttms,
            discfactors=chain.discfactors,
            variable_type=VariableType.SIGMA,
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
    python_direct, _, _ = compute_heston_mgf_grid.py_func(
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
    np.testing.assert_allclose(python_direct, direct, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(chained, direct, rtol=0.0, atol=2.0e-13)


def test_heston_qvar_transform_mean_matches_variance_process_expectation() -> None:
    """The Laplace-transform derivative must equal expected integrated variance."""
    params = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    ttm = 0.25
    epsilon = 1.0e-6
    log_mgf, _, _ = compute_heston_mgf_grid(
        v0=params.v0,
        theta=params.theta,
        kappa=params.kappa,
        volvol=params.volvol,
        rho=params.rho,
        ttm=ttm,
        phi_grid=np.array([0.0 + 0.0j]),
        psi_grid=np.array([epsilon + 0.0j]),
    )
    transformed_mean = -log_mgf[0].real / (epsilon * ttm)
    expected_mean = params.theta + (params.v0 - params.theta) * (
        1.0 - np.exp(-params.kappa * ttm)
    ) / (params.kappa * ttm)

    np.testing.assert_allclose(transformed_mean, expected_mean, rtol=0.0, atol=1.0e-8)


def test_heston_python_simulator_preserves_state_invariants() -> None:
    path_count = 256
    x, variance, qvar = simulate_heston_x_vol_terminal.py_func(
        ttm=0.02,
        x0=np.array([0.0]),
        var0=np.array([0.04]),
        qvar0=np.array([0.0]),
        theta=0.05,
        kappa=2.0,
        rho=-0.5,
        volvol=0.4,
        nb_path=path_count,
        nb_steps_per_year=360,
    )

    assert x.shape == variance.shape == qvar.shape == (path_count,)
    assert np.all(np.isfinite(x))
    assert np.all(variance >= 1.0e-4)
    assert np.all(qvar >= 0.0)


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
