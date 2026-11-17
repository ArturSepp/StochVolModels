"""
Tests for ADI schemes and the generic two-dimensional solver.

The cases pin weighted adjointness, constant/mass preservation, tensor-product limits, Gaussian
convergence, exact save times, scheme dispatch, temporal convergence, and high-correlation
anisotropy for Douglas, Craig--Sneyd, and Hundsdorfer--Verwer schemes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundarySet2D
from stochvolmodels.pde_solvers.coefficients import (
    PDE1DCoefficients,
    PDE2DCoefficients,
    StaticCoefficientProvider1D,
    StaticCoefficientProvider2D,
)
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import PDE1DSolver, TimeStepConfig

from stochvolmodels.pde_solvers.adi import (
    ADIConfig,
    ADIScheme,
    craig_sneyd_step,
    douglas_step,
    hundsdorfer_verwer_step,
)
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D
from stochvolmodels.pde_solvers.solver_2d import PDE2DADISolver

ADIStep = Callable[
    [SplitOperator2D, np.ndarray, PDEDirection, float, float], np.ndarray
]
_HUNDSDORFER_VERWER_THETA = 0.5 + np.sqrt(3.0) / 6.0
_CORRECTOR_CASES = (
    pytest.param(
        ADIScheme.CRAIG_SNEYD,
        craig_sneyd_step,
        0.5,
        id="craig_sneyd",
    ),
    pytest.param(
        ADIScheme.HUNDSDORFER_VERWER,
        hundsdorfer_verwer_step,
        _HUNDSDORFER_VERWER_THETA,
        id="hundsdorfer_verwer",
    ),
)


def _boundaries() -> BoundarySet2D:
    return BoundarySet2D(x=BoundaryPair1D.natural(), y=BoundaryPair1D.natural())


def _constant_provider(
    grid: TensorGrid2D,
    sigma_x: float,
    sigma_y: float,
    drift_x: float,
    drift_y: float,
    correlation: float,
) -> StaticCoefficientProvider2D:
    return StaticCoefficientProvider2D(
        PDE2DCoefficients.constant(
            grid.shape,
            variance_x=0.5 * sigma_x**2,
            drift_x=drift_x,
            variance_y=0.5 * sigma_y**2,
            drift_y=drift_y,
            covariance_xy=correlation * sigma_x * sigma_y,
            discount=0.0,
        )
    )


def _normal_density_2d(
    grid: TensorGrid2D,
    maturity: float,
    sigma_x: float,
    sigma_y: float,
    drift_x: float,
    drift_y: float,
    correlation: float,
) -> np.ndarray:
    mean_x = drift_x * maturity
    mean_y = drift_y * maturity
    standardized_x = (grid.x_mesh - mean_x) / (sigma_x * np.sqrt(maturity))
    standardized_y = (grid.y_mesh - mean_y) / (sigma_y * np.sqrt(maturity))
    denominator = 2.0 * np.pi * sigma_x * sigma_y * maturity * np.sqrt(
        1.0 - correlation**2
    )
    exponent = -(
        standardized_x**2
        - 2.0 * correlation * standardized_x * standardized_y
        + standardized_y**2
    ) / (2.0 * (1.0 - correlation**2))
    return np.exp(exponent) / denominator


def _solve_correlated_density(
    size_x: int,
    size_y: int,
    steps: int,
    correlation: float,
) -> tuple[TensorGrid2D, np.ndarray, object]:
    maturity = 0.7
    sigma_x = 0.25
    sigma_y = 0.32
    drift_x = 0.04
    drift_y = -0.025
    grid = TensorGrid2D(
        Grid1D.uniform(-1.35, 1.35, size_x),
        Grid1D.uniform(-1.55, 1.45, size_y),
    )
    solver = PDE2DADISolver(ADIConfig(theta=0.5))
    solution = solver.solve(
        grid,
        _constant_provider(
            grid, sigma_x, sigma_y, drift_x, drift_y, correlation
        ),
        _boundaries(),
        PDEDirection.FORWARD,
        np.linspace(0.0, maturity, steps + 1),
        grid.project_delta(0.0, 0.0),
    )
    analytic = _normal_density_2d(
        grid, maturity, sigma_x, sigma_y, drift_x, drift_y, correlation
    )
    return grid, analytic, solution


def test_forward_douglas_step_is_weighted_adjoint_of_backward_step() -> None:
    grid = TensorGrid2D(
        Grid1D.concentrated(-1.5, 1.5, 17, center=0.1, scale=0.35),
        Grid1D.concentrated(-1.2, 1.2, 15, center=-0.1, scale=0.3),
    )
    provider = _constant_provider(grid, 0.3, 0.25, 0.04, -0.02, 0.45)
    operator = SplitOperator2D.build(grid, provider.coefficients, _boundaries())
    rng = np.random.default_rng(4401)
    value = rng.normal(size=grid.shape)
    density = rng.uniform(0.1, 1.0, size=grid.shape)

    backward = douglas_step(operator, value, PDEDirection.BACKWARD, 0.01, 0.5)
    forward = douglas_step(operator, density, PDEDirection.FORWARD, 0.01, 0.5)

    assert grid.integrate(density * backward) == pytest.approx(
        grid.integrate(forward * value), abs=3.0e-12
    )


def test_douglas_preserves_backward_constant_and_forward_mass() -> None:
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 23), Grid1D.uniform(-1.2, 1.2, 21))
    provider = _constant_provider(grid, 0.3, 0.24, 0.03, -0.02, -0.4)
    operator = SplitOperator2D.build(grid, provider.coefficients, _boundaries())
    density = grid.project_delta(0.0, 0.0)

    backward = douglas_step(
        operator, np.ones(grid.shape), PDEDirection.BACKWARD, 0.005, 0.5
    )
    forward = douglas_step(operator, density, PDEDirection.FORWARD, 0.005, 0.5)

    np.testing.assert_allclose(backward, 1.0, rtol=0.0, atol=3.0e-13)
    assert grid.integrate(forward) == pytest.approx(1.0, abs=3.0e-13)


def test_zero_correlation_solution_agrees_with_tensor_product_of_1d_solutions() -> None:
    maturity = 0.6
    sigma_x, sigma_y = 0.23, 0.31
    drift_x, drift_y = 0.025, -0.035
    grid = TensorGrid2D(Grid1D.uniform(-1.3, 1.3, 61), Grid1D.uniform(-1.5, 1.5, 57))
    times = np.linspace(0.0, maturity, 161)
    solver_2d = PDE2DADISolver(ADIConfig(theta=0.5))
    solution_2d = solver_2d.solve(
        grid,
        _constant_provider(grid, sigma_x, sigma_y, drift_x, drift_y, 0.0),
        _boundaries(),
        PDEDirection.FORWARD,
        times,
        grid.project_delta(0.0, 0.0),
    )

    solver_1d = PDE1DSolver(TimeStepConfig(theta=0.5))
    coefficient_x = StaticCoefficientProvider1D(
        PDE1DCoefficients.constant(grid.x.size, 0.5 * sigma_x**2, drift_x, 0.0)
    )
    coefficient_y = StaticCoefficientProvider1D(
        PDE1DCoefficients.constant(grid.y.size, 0.5 * sigma_y**2, drift_y, 0.0)
    )
    solution_x = solver_1d.solve(
        grid.x,
        coefficient_x,
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        times,
        grid.x.project_delta(0.0),
    )
    solution_y = solver_1d.solve(
        grid.y,
        coefficient_y,
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        times,
        grid.y.project_delta(0.0),
    )
    tensor_reference = np.multiply.outer(solution_x.values, solution_y.values)

    assert grid.integrate(np.abs(solution_2d.values - tensor_reference)) < 2.5e-3


def test_correlated_gaussian_density_and_moments_converge_under_refinement() -> None:
    correlation = 0.45
    coarse_grid, coarse_analytic, coarse = _solve_correlated_density(31, 29, 80, correlation)
    fine_grid, fine_analytic, fine = _solve_correlated_density(55, 51, 160, correlation)
    coarse_error = coarse_grid.integrate(np.abs(coarse.values - coarse_analytic))
    fine_error = fine_grid.integrate(np.abs(fine.values - fine_analytic))
    maturity = 0.7
    expected_mean_x = 0.04 * maturity
    expected_mean_y = -0.025 * maturity
    expected_covariance = correlation * 0.25 * 0.32 * maturity

    assert fine_error < coarse_error
    assert fine_error < 0.035
    assert fine_grid.integrate(fine.values) == pytest.approx(1.0, abs=3.0e-11)
    final = fine.diagnostics[-1]
    assert final.mean_x == pytest.approx(expected_mean_x, abs=1.0e-3)
    assert final.mean_y == pytest.approx(expected_mean_y, abs=1.0e-3)
    assert final.covariance_xy == pytest.approx(expected_covariance, abs=1.5e-3)
    assert final.negative_mass < 2.0e-5
    assert abs(final.mass_error) < 3.0e-11
    assert final.minimum_covariance_psd_margin > 0.0


def test_full_solver_preserves_backward_constant_and_discrete_duality() -> None:
    grid = TensorGrid2D(Grid1D.uniform(-1.2, 1.2, 31), Grid1D.uniform(-1.1, 1.1, 29))
    provider = _constant_provider(grid, 0.26, 0.21, 0.03, -0.015, -0.35)
    times = np.linspace(0.0, 0.4, 61)
    solver = PDE2DADISolver(ADIConfig(theta=0.5))
    payoff = np.maximum(grid.x_mesh + 0.3 * grid.y_mesh, 0.0)
    density = grid.project_delta(0.0, 0.0)

    constant = solver.solve(
        grid,
        provider,
        _boundaries(),
        PDEDirection.BACKWARD,
        times,
        np.ones(grid.shape),
    )
    backward = solver.solve(
        grid, provider, _boundaries(), PDEDirection.BACKWARD, times, payoff
    )
    forward = solver.solve(
        grid, provider, _boundaries(), PDEDirection.FORWARD, times, density
    )

    np.testing.assert_allclose(constant.values, 1.0, rtol=0.0, atol=3.0e-12)
    assert grid.integrate(density * backward.values) == pytest.approx(
        grid.integrate(forward.values * payoff), abs=8.0e-12
    )


@dataclass
class TimeDependent2DProvider:
    """Provide time-dependent coefficients for ADI convergence tests."""
    calls: list[float] = field(default_factory=list)
    is_time_dependent: bool = True

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        self.calls.append(float(time))
        return PDE2DCoefficients.constant(
            grid.shape,
            variance_x=0.02 + 0.005 * time,
            drift_x=0.0,
            variance_y=0.03 + 0.004 * time,
            drift_y=0.0,
            covariance_xy=0.0,
            discount=0.0,
        )


def test_2d_solver_uses_exact_times_and_saves_immutable_states() -> None:
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 15), Grid1D.uniform(-1.0, 1.0, 13))
    times = np.array([0.0, 0.1, 0.4, 1.0])
    provider = TimeDependent2DProvider()
    solver = PDE2DADISolver(ADIConfig(theta=0.5))
    solution = solver.solve(
        grid,
        provider,
        _boundaries(),
        PDEDirection.FORWARD,
        times,
        grid.project_delta(0.0, 0.0),
        save_times=(0.0, 0.4, 1.0),
    )

    np.testing.assert_allclose(provider.calls, [0.05, 0.25, 0.7], rtol=0.0, atol=1.0e-15)
    assert [state.time for state in solution.saved_states] == [0.0, 0.4, 1.0]
    assert all(not state.values.flags.writeable for state in solution.saved_states)
    assert not solution.values.flags.writeable
    assert not solution.time_grid.flags.writeable
    assert solution.config.scheme is ADIScheme.DOUGLAS


@pytest.mark.parametrize("direction", list(PDEDirection))
def test_graded_time_dependent_directional_solve_matches_dense_steps(direction):
    grid = TensorGrid2D(Grid1D.concentrated(-1., 1., 13, 0., .2), Grid1D.uniform(-1., 1., 7))
    times = np.array([0., .01, .03, .05, .08, .11])
    state = np.random.default_rng(31).normal(size=grid.shape)
    theta = .6

    class Provider:
        def evaluate(self, t, grid):
            return PDE2DCoefficients(
                .2 + (1. + grid.y_mesh**2) * t, .03 * grid.x_mesh,
                np.zeros(grid.shape), np.zeros(grid.shape), np.zeros(grid.shape),
                np.zeros(grid.shape),
            )

    provider = Provider()
    actual = PDE2DADISolver(ADIConfig(theta=theta)).solve(
        grid, provider, _boundaries(), direction, times, state,
    )
    # With only x dynamics, each Douglas step is the ordinary 1D theta update.
    # Reassemble at each midpoint and solve densely, including repeated step lengths.
    pairs = list(zip(times[:-1], times[1:]))
    if direction is PDEDirection.BACKWARD:
        pairs.reverse()
    for left, right in pairs:
        coeff = provider.evaluate((left + right) / 2., grid)
        op = SplitOperator2D.build(grid, coeff, _boundaries())
        h = right - left
        for j, line in enumerate(op.x_lines):
            generator = line.generator(direction).to_dense()
            rhs = state[:, j] + (1. - theta) * h * (generator @ state[:, j])
            state[:, j] = np.linalg.solve(np.eye(grid.x.size) - theta * h * generator, rhs)
    np.testing.assert_allclose(actual.values, state, atol=2e-14, rtol=0)


def test_adi_config_and_forward_boundary_contract_are_validated() -> None:
    with pytest.raises(ValueError, match="theta"):
        ADIConfig(theta=0.0)

    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 9), Grid1D.uniform(-1.0, 1.0, 9))
    provider = _constant_provider(grid, 0.2, 0.2, 0.0, 0.0, 0.0)
    boundaries = BoundarySet2D(
        x=BoundaryPair1D.dirichlet(0.0, 0.0), y=BoundaryPair1D.natural()
    )
    with pytest.raises(ValueError, match="natural boundaries"):
        PDE2DADISolver().solve(
            grid,
            provider,
            boundaries,
            PDEDirection.FORWARD,
            np.array([0.0, 0.1]),
            grid.project_delta(0.0, 0.0),
        )


@pytest.mark.parametrize("scheme,step_function,theta", _CORRECTOR_CASES)
def test_craig_sneyd_and_hundsdorfer_steps_are_exact_weighted_adjoints(
    scheme: ADIScheme, step_function: ADIStep, theta: float
) -> None:
    del scheme
    grid = TensorGrid2D(
        Grid1D.concentrated(-1.4, 1.4, 19, center=0.1, scale=0.32),
        Grid1D.concentrated(-1.2, 1.2, 17, center=-0.1, scale=0.28),
    )
    provider = _constant_provider(grid, 0.31, 0.23, 0.035, -0.018, 0.72)
    operator = SplitOperator2D.build(grid, provider.coefficients, _boundaries())
    rng = np.random.default_rng(8675309)
    value = rng.normal(size=grid.shape)
    density = rng.uniform(0.05, 1.0, size=grid.shape)

    backward = step_function(operator, value, PDEDirection.BACKWARD, 0.012, theta)
    forward = step_function(operator, density, PDEDirection.FORWARD, 0.012, theta)

    assert grid.integrate(density * backward) == pytest.approx(
        grid.integrate(forward * value), abs=5.0e-12
    )
    constant = step_function(
        operator, np.ones(grid.shape), PDEDirection.BACKWARD, 0.012, theta
    )
    propagated = step_function(
        operator, grid.project_delta(0.0, 0.0), PDEDirection.FORWARD, 0.012, theta
    )
    np.testing.assert_allclose(constant, 1.0, rtol=0.0, atol=5.0e-13)
    assert grid.integrate(propagated) == pytest.approx(1.0, abs=5.0e-13)


def test_craig_sneyd_zero_covariance_corrector_is_initialized() -> None:
    grid = TensorGrid2D(
        Grid1D.concentrated(-1.2, 1.2, 17, center=0.0, scale=0.3),
        Grid1D.concentrated(-1.1, 1.1, 15, center=0.1, scale=0.27),
    )
    provider = _constant_provider(grid, 0.28, 0.21, 0.03, -0.02, 0.0)
    operator = SplitOperator2D.build(grid, provider.coefficients, _boundaries())
    rng = np.random.default_rng(1729)

    for direction, values in (
        (PDEDirection.BACKWARD, rng.normal(size=grid.shape)),
        (PDEDirection.FORWARD, rng.uniform(0.1, 1.0, size=grid.shape)),
    ):
        douglas = douglas_step(operator, values, direction, 0.01, 0.5)
        craig_sneyd = craig_sneyd_step(operator, values, direction, 0.01, 0.5)
        np.testing.assert_allclose(craig_sneyd, douglas, rtol=0.0, atol=3.0e-13)


@pytest.mark.parametrize("scheme,step_function,theta", _CORRECTOR_CASES)
def test_craig_sneyd_and_hundsdorfer_solver_dispatch_preserves_duality(
    scheme: ADIScheme, step_function: ADIStep, theta: float
) -> None:
    del step_function
    grid = TensorGrid2D(Grid1D.uniform(-1.2, 1.2, 25), Grid1D.uniform(-1.1, 1.1, 23))
    provider = _constant_provider(grid, 0.27, 0.22, 0.025, -0.012, -0.6)
    times = np.linspace(0.0, 0.3, 31)
    payoff = np.maximum(grid.x_mesh + 0.25 * grid.y_mesh, 0.0)
    density = grid.project_delta(0.0, 0.0)
    solver = PDE2DADISolver(ADIConfig(theta=theta, scheme=scheme))

    backward = solver.solve(
        grid, provider, _boundaries(), PDEDirection.BACKWARD, times, payoff
    )
    forward = solver.solve(
        grid, provider, _boundaries(), PDEDirection.FORWARD, times, density
    )

    assert backward.config.scheme is scheme
    assert forward.config.scheme is scheme
    assert grid.integrate(density * backward.values) == pytest.approx(
        grid.integrate(forward.values * payoff), abs=1.2e-11
    )


def _propagate_corrector_steps(
    operator: SplitOperator2D,
    values: np.ndarray,
    step_function: ADIStep,
    theta: float,
    maturity: float,
    steps: int,
) -> np.ndarray:
    state = np.array(values, copy=True)
    step_size = maturity / steps
    for _ in range(steps):
        state = step_function(
            operator, state, PDEDirection.FORWARD, step_size, theta
        )
    return state


@pytest.mark.parametrize("scheme,step_function,theta", _CORRECTOR_CASES)
def test_craig_sneyd_and_hundsdorfer_temporal_convergence_is_measured(
    scheme: ADIScheme, step_function: ADIStep, theta: float
) -> None:
    del scheme
    maturity = 0.35
    grid = TensorGrid2D(Grid1D.uniform(-1.3, 1.3, 31), Grid1D.uniform(-1.1, 1.1, 29))
    provider = _constant_provider(grid, 0.34, 0.19, 0.025, -0.015, 0.8)
    operator = SplitOperator2D.build(grid, provider.coefficients, _boundaries())
    initial = np.exp(
        -0.5 * (grid.x_mesh / 0.24) ** 2 - 0.5 * (grid.y_mesh / 0.2) ** 2
    )
    initial /= grid.integrate(initial)
    reference = _propagate_corrector_steps(
        operator, initial, step_function, theta, maturity, 256
    )
    errors = [
        grid.integrate(
            np.abs(
                _propagate_corrector_steps(
                    operator, initial, step_function, theta, maturity, steps
                )
                - reference
            )
        )
        for steps in (4, 8, 16)
    ]

    assert errors[2] < errors[1] < errors[0]
    assert errors[0] / errors[1] > 2.5
    assert errors[1] / errors[2] > 2.5


def test_craig_sneyd_and_hundsdorfer_high_correlation_anisotropy_matrix() -> None:
    maturity = 0.45
    steps = 90
    initial_sigma = 0.18
    results: list[tuple[float, float, float]] = []
    for size_x, size_y, sigma_x, sigma_y in (
        (39, 25, 0.38, 0.16),
        (25, 39, 0.16, 0.38),
    ):
        grid = TensorGrid2D(
            Grid1D.uniform(-1.5, 1.5, size_x),
            Grid1D.uniform(-1.5, 1.5, size_y),
        )
        for correlation in (-0.9, 0.9):
            provider = _constant_provider(
                grid, sigma_x, sigma_y, 0.02, -0.015, correlation
            )
            terminal_variance_x = initial_sigma**2 + sigma_x**2 * maturity
            terminal_variance_y = initial_sigma**2 + sigma_y**2 * maturity
            terminal_covariance = correlation * sigma_x * sigma_y * maturity
            determinant = (
                terminal_variance_x * terminal_variance_y - terminal_covariance**2
            )
            centered_x = grid.x_mesh - 0.02 * maturity
            centered_y = grid.y_mesh + 0.015 * maturity
            exponent = -0.5 * (
                terminal_variance_y * centered_x**2
                - 2.0 * terminal_covariance * centered_x * centered_y
                + terminal_variance_x * centered_y**2
            ) / determinant
            analytic = np.exp(exponent) / (2.0 * np.pi * np.sqrt(determinant))
            times = np.linspace(0.0, maturity, steps + 1)
            initial = np.exp(
                -0.5 * (grid.x_mesh / initial_sigma) ** 2
                - 0.5 * (grid.y_mesh / initial_sigma) ** 2
            )
            initial /= grid.integrate(initial)
            douglas = PDE2DADISolver(ADIConfig(theta=0.5)).solve(
                grid,
                provider,
                _boundaries(),
                PDEDirection.FORWARD,
                times,
                initial,
            )
            douglas_error = grid.integrate(np.abs(douglas.values - analytic))
            for scheme, _, theta in (
                (ADIScheme.CRAIG_SNEYD, craig_sneyd_step, 0.5),
                (
                    ADIScheme.HUNDSDORFER_VERWER,
                    hundsdorfer_verwer_step,
                    _HUNDSDORFER_VERWER_THETA,
                ),
            ):
                corrected = PDE2DADISolver(
                    ADIConfig(theta=theta, scheme=scheme)
                ).solve(
                    grid,
                    provider,
                    _boundaries(),
                    PDEDirection.FORWARD,
                    times,
                    initial,
                )
                corrected_error = grid.integrate(np.abs(corrected.values - analytic))
                assert np.all(np.isfinite(corrected.values))
                assert grid.integrate(corrected.values) == pytest.approx(1.0, abs=5.0e-11)
                assert corrected.diagnostics[-1].negative_mass < 1.0e-3
                assert corrected_error < 0.16
                assert corrected_error <= douglas_error + 0.02
                results.append(
                    (
                        corrected_error,
                        corrected.diagnostics[-1].negative_mass,
                        douglas_error,
                    )
                )

    assert len(results) == 8


@pytest.mark.parametrize("scheme", list(ADIScheme))
def test_extrapolation_preserves_affine_plane_on_uneven_axes(scheme: ADIScheme) -> None:
    grid = TensorGrid2D(
        Grid1D(np.array([0.0, 0.1, 0.3, 0.6, 1.0])),
        Grid1D(np.array([-1.0, -0.85, -0.4, -0.1, 0.7])),
    )
    boundaries = BoundarySet2D(
        BoundaryPair1D.linear_extrapolation(), BoundaryPair1D.linear_extrapolation()
    )
    provider = StaticCoefficientProvider2D(
        PDE2DCoefficients.constant(
            grid.shape,
            0.15,
            0.0,
            0.1,
            0.0,
            0.05,
            0.0,
        )
    )
    plane = 1.0 + 3.0 * grid.x_mesh - 2.0 * grid.y_mesh
    result = PDE2DADISolver(ADIConfig(scheme=scheme)).solve(
        grid,
        provider,
        boundaries,
        PDEDirection.BACKWARD,
        [0.0, 0.15, 0.3],
        plane,
        save_times=(0.0, 0.15, 0.3),
    )
    for state in result.saved_states:
        np.testing.assert_allclose(state.values, plane, atol=2.0e-13, rtol=0.0)
