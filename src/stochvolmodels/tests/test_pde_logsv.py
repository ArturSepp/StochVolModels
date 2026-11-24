"""
Lognormal stochastic-volatility coefficient and pricing tests.

These tests pin the current SVM LogSV drift, backbone, spot/inverse-measure, and covariance
conventions. Limiting cases are checked before PDE moments and vanilla prices are compared with
independent moment ODE, transform, and Monte Carlo references.
"""

import numpy as np
import pandas as pd
import pytest
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundarySet2D
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_vol_moments

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pde_solvers.adi import ADIConfig, ADIScheme
from stochvolmodels.pricers.density_pricer import (
    CoordinateSystem,
    DensityPricer,
    DensityState2D,
)
from stochvolmodels.models.logsv.coefficients import LogSvCoefficientProvider
from stochvolmodels.pde_solvers.solver_2d import PDE2DADISolver
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import (
    LogSVPricer,
    simulate_logsv_x_vol_terminal,
)


def _natural_boundaries() -> BoundarySet2D:
    return BoundarySet2D(x=BoundaryPair1D.natural(), y=BoundaryPair1D.natural())


def test_logsv_coefficients_match_current_model_dynamics_and_psd_identity() -> None:
    params = LogSvParams(
        sigma0=0.2,
        theta=0.24,
        kappa1=1.7,
        kappa2=4.5,
        beta=-0.35,
        volvol=0.6,
    )
    grid = TensorGrid2D(
        Grid1D.uniform(-0.3, 0.3, 5),
        Grid1D(np.array([0.0, 0.1, 0.25, 0.5])),
    )
    provider = LogSvCoefficientProvider(params)
    coefficients = provider.evaluate(0.2, grid)
    sigma = grid.y_mesh
    sigma2 = sigma * sigma

    assert provider.params is params
    assert not provider.is_time_dependent
    np.testing.assert_allclose(coefficients.variance_x, 0.5 * sigma2)
    np.testing.assert_allclose(coefficients.drift_x, -0.5 * sigma2)
    np.testing.assert_allclose(
        coefficients.variance_y, 0.5 * params.vartheta2 * sigma2
    )
    np.testing.assert_allclose(
        coefficients.drift_y,
        (params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma),
    )
    np.testing.assert_allclose(coefficients.covariance_xy, params.beta * sigma2)
    np.testing.assert_array_equal(coefficients.discount, np.zeros(grid.shape))
    np.testing.assert_allclose(
        coefficients.covariance_psd_margin,
        params.volvol**2 * sigma**4,
        rtol=2.0e-15,
        atol=2.0e-18,
    )
    assert np.min(coefficients.covariance_psd_margin) >= 0.0


def test_logsv_inverse_measure_and_backbone_follow_current_simulator_convention() -> None:
    params = LogSvParams(
        sigma0=0.2,
        theta=0.2,
        kappa1=2.0,
        kappa2=5.0,
        beta=0.3,
        volvol=0.4,
        vol_backbone=pd.Series([0.8, 1.1], index=[0.1, 0.3]),
    )
    grid = TensorGrid2D(
        Grid1D.uniform(-0.2, 0.2, 3), Grid1D.uniform(0.1, 0.4, 4)
    )
    provider = LogSvCoefficientProvider(params, is_spot_measure=False)
    coefficients = provider.evaluate(0.2, grid)
    scale = 1.1
    sigma = grid.y_mesh
    sigma2 = sigma * sigma
    base_vol_drift = (params.kappa1 + params.kappa2 * sigma) * (
        params.theta - sigma
    )

    assert provider.is_time_dependent
    np.testing.assert_allclose(coefficients.variance_x, 0.5 * scale**2 * sigma2)
    np.testing.assert_allclose(coefficients.drift_x, 0.5 * scale**2 * sigma2)
    np.testing.assert_allclose(
        coefficients.drift_y, base_vol_drift + params.beta * scale * sigma2
    )
    np.testing.assert_allclose(
        coefficients.covariance_xy, params.beta * scale * sigma2
    )


@pytest.mark.parametrize(
    ("kappa1", "kappa2"),
    [(1.5, 0.0), (1.5, 3.25), (1.5, None)],
    ids=("linear", "quadratic", "pure_quadratic"),
)
def test_logsv_linear_quadratic_and_pure_quadratic_drift_limits(
    kappa1: float, kappa2: float | None
) -> None:
    params = LogSvParams(
        sigma0=0.3,
        theta=0.25,
        kappa1=kappa1,
        kappa2=kappa2,
        beta=0.0,
        volvol=0.2,
    )
    grid = TensorGrid2D(
        Grid1D.uniform(-0.1, 0.1, 3), Grid1D.uniform(0.1, 0.5, 5)
    )
    coefficients = LogSvCoefficientProvider(params).evaluate(0.0, grid)
    expected_kappa2 = kappa1 / params.theta if kappa2 is None else kappa2

    assert params.kappa2 == expected_kappa2
    np.testing.assert_allclose(
        coefficients.drift_y,
        (kappa1 + expected_kappa2 * grid.y_mesh) * (params.theta - grid.y_mesh),
    )


def test_logsv_adapter_rejects_rough_and_invalid_volatility_coordinates() -> None:
    rough = LogSvParams(H=0.4)
    with pytest.raises(ValueError, match="H=0.5"):
        LogSvCoefficientProvider(rough)

    params = LogSvParams()
    invalid_grid = TensorGrid2D(
        Grid1D.uniform(-0.2, 0.2, 3), Grid1D.uniform(-0.1, 0.4, 5)
    )
    with pytest.raises(ValueError, match="volatility nodes must be nonnegative"):
        LogSvCoefficientProvider(params).evaluate(0.1, invalid_grid)


def test_logsv_linear_drift_volatility_moments_match_existing_moment_ode() -> None:
    params = LogSvParams(
        sigma0=0.32,
        theta=0.24,
        kappa1=1.8,
        kappa2=0.0,
        beta=0.0,
        volvol=0.28,
    )
    maturity = 0.3
    grid = TensorGrid2D(
        Grid1D.uniform(-1.5, 1.5, 7),
        Grid1D.concentrated(0.0, 1.1, 161, center=params.sigma0, scale=0.045),
    )
    solution = PDE2DADISolver(
        ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS)
    ).solve(
        grid,
        LogSvCoefficientProvider(params),
        _natural_boundaries(),
        PDEDirection.FORWARD,
        np.linspace(0.0, maturity, 181),
        grid.project_delta(0.0, params.sigma0),
    )
    marginal = grid.marginalize_x(solution.values)
    mass = grid.y.integrate(marginal)
    pde_mean = grid.y.integrate(grid.y.nodes * marginal)
    pde_second = grid.y.integrate(grid.y.nodes**2 * marginal)
    centered = compute_analytic_vol_moments(params=params, t=maturity, n_terms=4)
    expected_mean = params.theta + centered[0]
    expected_second = params.theta2 + 2.0 * params.theta * centered[0] + centered[1]

    np.testing.assert_allclose(mass, 1.0, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(pde_mean, expected_mean, rtol=0.0, atol=3.0e-4)
    np.testing.assert_allclose(pde_second, expected_second, rtol=0.0, atol=3.0e-4)


@pytest.mark.slow
def test_no_leverage_logsv_pde_calls_match_transform_and_independent_mc() -> None:
    params = LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=0.0,
        volvol=0.35,
    )
    maturity = 0.25
    strikes = np.array([0.9, 1.0, 1.1])
    discount = 0.98
    grid = TensorGrid2D(
        Grid1D.concentrated(-0.8, 0.65, 141, center=-0.01, scale=0.08),
        Grid1D.concentrated(0.025, 0.60, 101, center=params.sigma0, scale=0.035),
    )
    solution = PDE2DADISolver(
        ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS)
    ).solve(
        grid,
        LogSvCoefficientProvider(params),
        _natural_boundaries(),
        PDEDirection.FORWARD,
        np.linspace(0.0, maturity, 161),
        grid.project_delta(0.0, params.sigma0),
    )
    density = DensityState2D(
        grid=grid,
        values=solution.values,
        time=maturity,
        x_coordinate=CoordinateSystem.LOG_FORWARD,
    )
    pde_result = DensityPricer(
        density.marginal_x_state(), forward=1.0, discount_factor=discount
    ).price_slice(strikes)

    chain = OptionChain.slice_to_chain(
        ttm=maturity,
        forward=1.0,
        strikes=strikes,
        optiontypes=np.full(strikes.size, "C"),
        discfactor=discount,
        id="3m",
    )
    transform_calls = np.asarray(LogSVPricer().price_chain(chain, params)[0])

    paths = 50_000
    steps = 100
    dt = maturity / steps
    rng = np.random.default_rng(20260818)
    x_terminal, _, _ = simulate_logsv_x_vol_terminal(
        ttm=maturity,
        x0=np.zeros(paths),
        sigma0=np.full(paths, params.sigma0),
        qvar0=np.zeros(paths),
        theta=params.theta,
        kappa1=params.kappa1,
        kappa2=params.kappa2,
        beta=params.beta,
        volvol=params.volvol,
        nb_path=paths,
        W0=rng.standard_normal((steps, paths)),
        W1=rng.standard_normal((steps, paths)),
        dt=dt,
    )
    terminal_spot = np.exp(x_terminal)
    payoffs = np.maximum(terminal_spot[:, None] - strikes[None, :], 0.0)
    mc_calls = discount * np.mean(payoffs, axis=0)
    mc_errors = discount * np.std(payoffs, axis=0, ddof=1) / np.sqrt(paths)
    spot_error = np.std(terminal_spot, ddof=1) / np.sqrt(paths)

    assert abs(pde_result.mass - 1.0) < 3.0e-12
    assert abs(pde_result.forward_moment - 1.0) < 4.0e-4
    assert abs(np.mean(terminal_spot) - 1.0) <= 4.0 * spot_error
    np.testing.assert_allclose(
        pde_result.calls, transform_calls, rtol=0.0, atol=5.0e-4
    )
    assert np.all(np.abs(pde_result.calls - mc_calls) <= 4.0 * mc_errors + 5.0e-4)
