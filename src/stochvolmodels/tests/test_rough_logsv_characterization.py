import numpy as np
import pytest
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.rough_logsv.rough_kernel import (
    Gaussian_interval,
    Gaussian_parameters,
    fractional_kernel,
    fractional_kernel_approximation,
    kernel_rheston,
)

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv_pricer import (
    get_randoms_for_rough_vol_chain_valuation,
    rough_logsv_mc_chain_pricer_fixed_randoms,
)


@pytest.mark.parametrize(("hurst", "expected_terms"), [(0.5, 1), (0.45, 2), (0.1, 3)])
def test_rough_kernel_approximation_is_deterministic_and_positive(
    hurst: float, expected_terms: int
) -> None:
    first = LogSvParams(H=hurst)
    second = LogSvParams(H=hurst)

    first.approximate_kernel(T=0.05)
    second.approximate_kernel(T=0.05)

    assert first.nodes.shape == first.weights.shape == (expected_terms,)
    assert np.all(np.isfinite(first.nodes))
    assert np.all(np.isfinite(first.weights))
    assert np.all(first.nodes >= 0.0)
    assert np.all(first.weights > 0.0)
    np.testing.assert_array_equal(first.nodes, second.nodes)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_three_factor_rough_kernel_tracks_fractional_reference() -> None:
    params = LogSvParams(H=0.1)
    params.approximate_kernel(T=0.05)
    times = np.array([0.001, 0.005, 0.01, 0.025, 0.05])

    exact = fractional_kernel(params.H, times)
    approximation = fractional_kernel_approximation(
        params.H, times, params.nodes, params.weights
    )
    relative_error = np.abs(approximation - exact) / exact

    assert np.all(np.isfinite(approximation))
    assert np.all(approximation > 0.0)
    assert np.max(relative_error) < 0.25
    assert np.max(relative_error[1:]) < 0.12


def test_rough_random_grid_replays_seed_and_has_aligned_shapes() -> None:
    kwargs = dict(
        ttms=np.array([0.05, 0.1]),
        nb_path=16,
        nb_steps_per_year=100,
        seed=123,
    )
    np.random.seed(91)
    expected_global_draw = np.random.random()
    np.random.seed(91)
    first_z0, first_z1, first_grids = get_randoms_for_rough_vol_chain_valuation(**kwargs)
    actual_global_draw = np.random.random()
    second_z0, second_z1, second_grids = get_randoms_for_rough_vol_chain_valuation(**kwargs)

    assert first_z0.shape == first_z1.shape == (11, 16)
    np.testing.assert_array_equal(first_z0, second_z0)
    np.testing.assert_array_equal(first_z1, second_z1)
    assert len(first_grids) == len(second_grids) == 2
    for first, second in zip(first_grids, second_grids):
        np.testing.assert_array_equal(first, second)
        assert first[0] == 0.0
    np.testing.assert_allclose(
        [grid[-1] for grid in first_grids], kwargs["ttms"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(actual_global_draw, expected_global_draw, rtol=0.0, atol=0.0)


@pytest.mark.slow
def test_rough_fixed_random_pricer_is_finite_and_deterministic() -> None:
    chain = OptionChain.slice_to_chain(
        ttm=0.05,
        forward=1.0,
        strikes=np.array([0.95, 1.0, 1.05]),
        optiontypes=np.array(["P", "C", "C"]),
        id="rough",
    )
    params = LogSvParams(
        sigma0=0.2,
        theta=0.2,
        kappa1=2.0,
        kappa2=8.0,
        beta=-0.2,
        volvol=0.3,
        H=0.1,
    )
    params.approximate_kernel(T=chain.ttms[-1])
    z0, z1, grids = get_randoms_for_rough_vol_chain_valuation(
        chain.ttms, nb_path=128, nb_steps_per_year=100, seed=123
    )

    def price():
        return rough_logsv_mc_chain_pricer_fixed_randoms(
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
        )

    first_prices, first_errors = price()
    second_prices, second_errors = price()

    assert np.asarray(first_prices[0]).shape == (3,)
    assert np.all(np.isfinite(first_prices[0]))
    assert np.all(np.isfinite(first_errors[0]))
    assert np.all(np.asarray(first_errors[0]) >= 0.0)
    np.testing.assert_array_equal(first_prices[0], second_prices[0])
    np.testing.assert_array_equal(first_errors[0], second_errors[0])


def test_unknown_rough_quadrature_mode_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="not been implemented"):
        Gaussian_parameters(H=0.1, N=8, T=1.0, mode="unsupported")


def test_optional_rough_quadrature_reports_missing_dependencies() -> None:
    with pytest.raises(ImportError, match="orthopy.*quadpy"):
        Gaussian_interval(H=0.1, m=2, a=0.1, b=1.0)


def test_experimental_rough_heston_kernel_fails_precisely() -> None:
    kernel = kernel_rheston(H=0.1, lam=1.5, zeta=0.3)

    with pytest.raises(NotImplementedError, match="Mittag-Leffler"):
        kernel._k(0.5)


def test_logsv_params_vartheta_and_rho_follow_beta_and_epsilon() -> None:
    params = LogSvParams(beta=-0.3, volvol=0.4)

    assert params.vartheta == pytest.approx(0.5, rel=1e-15)
    assert params.rho == pytest.approx(-0.6, rel=1e-15)
    assert params.vartheta ** 2 == pytest.approx(params.vartheta2, rel=1e-15)


def _linear_drift_lift_moments(nodes: np.ndarray, weights: np.ndarray, sigma0: float, theta: float,
                               kappa1: float, vartheta: float, ttm: float):
    """exact E[sigma_T], E[sigma_T^2], E[I_T] of the sum-of-exponentials model with kappa2 = 0.

    The factors U_i = sigma^(i) - sigma0 / sum(w) solve a linear SDE with diffusion vartheta sigma,
    sigma = sigma0 + w'U, so their first and second moments solve a closed linear ODE system.
    """
    from scipy.integrate import solve_ivp

    n = nodes.size
    ones = np.ones(n)
    b_mat = np.diag(nodes) + kappa1 * np.outer(ones, weights)
    a_vec = kappa1 * (theta - sigma0) * ones

    def rhs(_t, y):
        mu = y[:n]
        p_mat = y[n:n + n * n].reshape(n, n)
        e_sigma2 = sigma0 ** 2 + 2.0 * sigma0 * weights @ mu + weights @ p_mat @ weights
        d_p = (np.outer(a_vec, mu) + np.outer(mu, a_vec) - b_mat @ p_mat - p_mat @ b_mat.T
               + vartheta ** 2 * e_sigma2 * np.outer(ones, ones))
        return np.concatenate([a_vec - b_mat @ mu, d_p.ravel(), [e_sigma2]])

    sol = solve_ivp(rhs, (0.0, ttm), np.zeros(n + n * n + 1), method="Radau",
                    rtol=1e-10, atol=1e-12)
    mu = sol.y[:n, -1]
    p_mat = sol.y[n:n + n * n, -1].reshape(n, n)
    e_sigma2 = sigma0 ** 2 + 2.0 * sigma0 * weights @ mu + weights @ p_mat @ weights
    return sigma0 + weights @ mu, e_sigma2, sol.y[-1, -1]


@pytest.mark.slow
def test_rough_split_scheme_matches_exact_linear_drift_moments() -> None:
    """split-step Monte Carlo against the exact moments of the working paper (Sec. 3, Tab. 4a)."""
    from stochvolmodels.pricers.rough_logsv.split_simulation import log_spot_full_combined
    from stochvolmodels.utils.funcs import set_time_grid

    params = LogSvParams(sigma0=0.5, theta=0.4, kappa1=2.0, kappa2=0.0,
                         beta=-0.4, volvol=0.6, H=0.3)
    ttm, nb_path = 1.0, 40000
    params.approximate_kernel(T=ttm)
    nodes, weights = params.nodes, params.weights
    nb_steps, _dt, grid = set_time_grid(ttm, 360)
    rng = np.random.default_rng(3)
    z0 = rng.standard_normal((nb_steps, nb_path))
    z1 = rng.standard_normal((nb_steps, nb_path))
    sigma0_factors = np.repeat(np.full(nodes.size, params.sigma0 / weights.sum())[:, None],
                               nb_path, axis=1)

    log_s, factors, qvar = log_spot_full_combined(
        np.repeat(nodes[:, None], nb_path, axis=1), np.repeat(weights[:, None], nb_path, axis=1),
        sigma0_factors, params.theta, params.kappa1, params.kappa2, 0.0, sigma0_factors.copy(),
        params.rho, params.vartheta, grid, nb_path, z0, z1)
    sigma_t = weights @ factors
    spot = np.exp(log_s[0])
    exact_mean, exact_second, exact_qvar = _linear_drift_lift_moments(
        nodes, weights, params.sigma0, params.theta, params.kappa1, params.vartheta, ttm)

    def z_score(sample: np.ndarray, exact: float) -> float:
        return abs(sample.mean() - exact) / (sample.std() / np.sqrt(sample.size))

    assert z_score(sigma_t, exact_mean) < 4.0
    assert z_score(sigma_t ** 2, exact_second) < 4.0
    assert z_score(qvar[0], exact_qvar) < 4.0
    assert z_score(spot, 1.0) < 4.0
