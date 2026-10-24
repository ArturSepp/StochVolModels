import warnings

import numpy as np
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import (
    get_randoms_for_rough_vol_chain_valuation,
    rough_logsv_mc_chain_pricer_split_step,
)
from stochvolmodels.pricers.rough_logsv.kernel_fit import MomentKey, markov_moment_targets
from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
from stochvolmodels.utils.mc_payoffs import compute_conditional_mc_payoff, compute_mc_vars_payoff


def _lognormal_payoffs(spots: np.ndarray, strike: float, type_: str) -> np.ndarray:
    if type_ == "C":
        return np.maximum(spots - strike, 0.0)
    if type_ == "P":
        return np.maximum(strike - spots, 0.0)
    if type_ == "IC":
        return np.maximum(spots - strike, 0.0) / spots
    return np.maximum(strike - spots, 0.0) / spots


@pytest.mark.parametrize("type_", ["C", "P", "IC", "IP"])
def test_conditional_payoff_matches_lognormal_monte_carlo(type_: str) -> None:
    n_path, total_var = 400000, 0.09
    strikes = np.array([0.8, 1.0, 1.3])
    prices, errors = compute_conditional_mc_payoff(
        np.zeros(n_path), np.full(n_path, total_var), 1.0, strikes, np.array([type_] * 3)
    )
    rng = np.random.default_rng(0)
    spots = np.exp(-0.5 * total_var + np.sqrt(total_var) * rng.standard_normal(n_path))
    for price, error, strike in zip(prices, errors, strikes):
        payoffs = _lognormal_payoffs(spots, strike, type_)
        assert error < 1e-12  # deterministic variance: the conditional estimator has no noise
        assert abs(price - payoffs.mean()) < 4.0 * payoffs.std() / np.sqrt(n_path)


def test_conditional_payoff_satisfies_put_call_parity() -> None:
    rng = np.random.default_rng(1)
    n_path, forward, discfactor = 5000, 1.3, 0.97
    log_forward_ratio = 0.3 * rng.standard_normal(n_path)
    total_var = 0.05 * rng.random(n_path)
    strikes = np.array([0.9, 1.3, 1.6, 0.9, 1.3, 1.6])
    types = np.array(["C", "C", "C", "P", "P", "P"])
    prices, _ = compute_conditional_mc_payoff(
        log_forward_ratio, total_var, forward, strikes, types, discfactor
    )
    np.testing.assert_allclose(
        prices[:3] - prices[3:],
        discfactor * (forward * np.mean(np.exp(log_forward_ratio)) - strikes[:3]),
        rtol=0.0, atol=1e-12,
    )


def test_payoff_standard_error_does_not_depend_on_path_layout() -> None:
    rng = np.random.default_rng(2)
    log_returns = 0.2 * rng.standard_normal(20000) - 0.02
    strikes, types = np.array([1.0]), np.array(["C"])
    _, flat_error = compute_mc_vars_payoff(
        log_returns, log_returns, log_returns, 1.0, 1.0, strikes, types
    )
    row = log_returns[None, :]
    _, row_error = compute_mc_vars_payoff(row, row, row, 1.0, 1.0, strikes, types)
    np.testing.assert_allclose(row_error, flat_error, rtol=1e-12)
    assert flat_error[0] < 0.01


def _btc_like_kernel(nb_path: int, ttm: float = 0.43, steps_per_year: int = 360):
    params = LogSvParams(
        sigma0=0.377, theta=0.347, kappa1=1.29, kappa2=1.93, beta=2.45, volvol=1.81, H=0.1
    )
    params.approximate_kernel(T=ttm)
    n_steps = int(np.ceil(steps_per_year * ttm))
    rng = np.random.default_rng(10)
    z0 = rng.standard_normal((n_steps, nb_path))
    z1 = rng.standard_normal((n_steps, nb_path))
    nodes = np.repeat(params.nodes[:, None], nb_path, axis=1)
    weights = np.repeat(params.weights[:, None], nb_path, axis=1)
    factors = np.full_like(nodes, params.sigma0 / params.weights.sum())
    grid = np.linspace(0.0, ttm, n_steps + 1)
    return params, nodes, weights, factors, grid, z0, z1


def test_split_step_fallback_replaces_the_floor() -> None:
    nb_path = 2000
    params, nodes, weights, factors, grid, z0, z1 = _btc_like_kernel(nb_path)
    paths = simulate_split_step(
        nodes,
        weights,
        factors,
        params.theta,
        params.kappa1,
        params.kappa2,
        0.0,
        factors.copy(),
        params.rho,
        params.vartheta,
        grid,
        nb_path,
        z0,
        z1,
    )
    assert paths.n_fallback > 0
    assert paths.n_floor == 0
    assert np.all(np.isfinite(paths.log_spot))
    assert np.all(params.weights @ paths.factors > 0.0)
    # the price step uses J from the diffusion sub-step, so J_T and I_T rebuild the log-price mean
    assert paths.jump.shape == paths.qvar.shape == paths.log_spot.shape == (nb_path,)


@pytest.mark.slow
def test_split_step_reproduces_exact_moments_of_its_kernel() -> None:
    sigma0, theta, kappa1, vartheta, rho, ttm = 0.5, 0.4, 2.0, 0.72, -0.7, 0.25
    params = LogSvParams(
        sigma0=sigma0,
        theta=theta,
        kappa1=kappa1,
        kappa2=0.0,
        beta=rho * vartheta,
        volvol=vartheta * np.sqrt(1.0 - rho * rho),
        H=0.3,
    )
    params.approximate_kernel(T=1.0)
    nb_path, n_steps = 40000, 63
    rng = np.random.default_rng(4)
    z0 = rng.standard_normal((n_steps, nb_path))
    z1 = rng.standard_normal((n_steps, nb_path))
    nodes = np.repeat(params.nodes[:, None], nb_path, axis=1)
    weights = np.repeat(params.weights[:, None], nb_path, axis=1)
    factors = np.full_like(nodes, sigma0 / params.weights.sum())
    paths = simulate_split_step(
        nodes,
        weights,
        factors,
        theta,
        kappa1,
        0.0,
        0.0,
        factors.copy(),
        rho,
        vartheta,
        np.linspace(0.0, ttm, n_steps + 1),
        nb_path,
        z0,
        z1,
    )
    exact = markov_moment_targets(
        params.nodes, params.weights, sigma0, theta, kappa1, vartheta, maturities=(ttm,)
    )
    sigma_t = params.weights @ paths.factors
    mean_i, se_i = paths.qvar.mean(), paths.qvar.std() / np.sqrt(nb_path)
    assert abs(sigma_t.mean() - exact[MomentKey.MEAN][0]) < 4.0 * sigma_t.std() / np.sqrt(nb_path)
    assert abs(mean_i - ttm * exact[MomentKey.LEVEL][0] ** 2) < 4.0 * se_i
    # cov(log S_T + I_T / 2, sigma_T) = rho vartheta Lambda(T)
    martingale = paths.log_spot + 0.5 * paths.qvar
    cross = (martingale - martingale.mean()) * (sigma_t - sigma_t.mean())
    leverage = cross.mean() / (rho * vartheta)
    leverage_se = cross.std() / np.sqrt(nb_path) / abs(rho * vartheta)
    assert abs(leverage - exact[MomentKey.LEVERAGE][0]) < 4.0 * leverage_se
    assert paths.n_floor == 0


def _moderate_chain_prices(is_conditional_mc: bool, steps_per_year: int = 64):
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([0.25]), ids=np.array(["3m"]), strikes=np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    )
    params = LogSvParams(
        sigma0=0.3, theta=0.3, kappa1=2.0, kappa2=2.0, beta=-0.3, volvol=0.3, H=0.3
    )
    params.approximate_kernel(T=0.25)
    z0, z1, grids = get_randoms_for_rough_vol_chain_valuation(
        chain.ttms, nb_path=8000, nb_steps_per_year=steps_per_year, seed=3
    )
    return rough_logsv_mc_chain_pricer_split_step(
        ttms=chain.ttms,
        forwards=chain.forwards,
        discfactors=chain.discfactors,
        strikes_ttms=chain.strikes_ttms,
        optiontypes_ttms=chain.optiontypes_ttms,
        Z0=z0,
        Z1=z1,
        sigma0=params.sigma0,
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        orthog_vol=params.volvol,
        weights=params.weights,
        nodes=params.nodes,
        timegrids=grids,
        is_conditional_mc=is_conditional_mc,
    )


def test_conditional_and_plain_chain_prices_agree() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the step rule holds, so no warning is expected
        conditional, conditional_errors = _moderate_chain_prices(True)
        plain, plain_errors = _moderate_chain_prices(False)
    combined = np.hypot(conditional_errors[0], plain_errors[0])
    assert np.all(np.abs(conditional[0] - plain[0]) < 4.0 * combined)
    assert np.all(conditional_errors[0] < plain_errors[0])
    assert np.all(plain_errors[0] < 0.01)


def test_pricer_warns_when_step_rule_fails() -> None:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _moderate_chain_prices(True, steps_per_year=4)
    messages = [str(item.message) for item in record if item.category is UserWarning]
    assert any("exceeds" in message for message in messages)
