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
