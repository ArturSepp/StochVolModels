"""
Tests for generic one-dimensional forward/backward PDE solving.

The cases verify exact coefficient times, forward/backward time direction, Black and log-density
convergence, discrete payoff-density duality, boundary enforcement, immutable saved states,
density policies, and invalid-grid contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import BoundaryCondition1D, BoundaryPair1D, BoundarySide
from stochvolmodels.pde_solvers.diagnostics import boundary_residuals
from stochvolmodels.pde_solvers.coefficients import (
    PDE1DCoefficients,
    StaticCoefficientProvider1D,
)
from stochvolmodels.pde_solvers.grids import Grid1D
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import (
    CoefficientEvaluation,
    DensityPolicy,
    PDE1DSolver,
    TimeScheme,
    TimeStepConfig,
)


@dataclass
class TimeDependentDiscountProvider:
    """Test provider whose call sequence exposes incorrect accumulated times."""

    calls: list[float] = field(default_factory=list)
    is_time_dependent: bool = True

    def evaluate(self, time: float, grid: Grid1D) -> PDE1DCoefficients:
        self.calls.append(float(time))
        return PDE1DCoefficients.constant(
            grid.size,
            variance=0.0,
            drift=0.0,
            discount=0.2 + float(time),
        )


def _constant_log_forward_provider(
    grid: Grid1D, volatility: float
) -> StaticCoefficientProvider1D:
    half_variance = 0.5 * volatility**2
    return StaticCoefficientProvider1D(
        PDE1DCoefficients.constant(
            grid.size,
            variance=half_variance,
            drift=-half_variance,
            discount=0.0,
        )
    )


def _solve_bsm_call(size: int, steps: int) -> float:
    maturity = 1.0
    volatility = 0.3
    strike = 1.1
    grid = Grid1D.uniform(-2.0, 2.0, size)
    payoff = np.maximum(np.exp(grid.nodes) - strike, 0.0)
    solver = PDE1DSolver(TimeStepConfig(theta=1.0))
    solution = solver.solve(
        grid=grid,
        coefficient_provider=_constant_log_forward_provider(grid, volatility),
        boundaries=BoundaryPair1D.natural(),
        direction=PDEDirection.BACKWARD,
        time_grid=np.linspace(0.0, maturity, steps + 1),
        values=payoff,
    )
    return float(solution.values[grid.size // 2])


def _solve_log_density(size: int, steps: int) -> tuple[Grid1D, np.ndarray, object]:
    maturity = 0.8
    volatility = 0.28
    grid = Grid1D.uniform(-1.8, 1.4, size)
    solver = PDE1DSolver(TimeStepConfig(theta=1.0))
    solution = solver.solve(
        grid=grid,
        coefficient_provider=_constant_log_forward_provider(grid, volatility),
        boundaries=BoundaryPair1D.natural(),
        direction=PDEDirection.FORWARD,
        time_grid=np.linspace(0.0, maturity, steps + 1),
        values=grid.project_delta(0.0),
    )
    standard_deviation = volatility * np.sqrt(maturity)
    mean = -0.5 * volatility**2 * maturity
    analytic = np.exp(-0.5 * ((grid.nodes - mean) / standard_deviation) ** 2)
    analytic /= np.sqrt(2.0 * np.pi) * standard_deviation
    return grid, analytic, solution


def test_time_dependent_coefficients_use_exact_declared_midpoints() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 9)
    time_grid = np.array([0.0, 0.1, 0.4, 1.0])
    provider = TimeDependentDiscountProvider()
    terminal = np.ones(grid.size)
    solver = PDE1DSolver(
        TimeStepConfig(theta=1.0, coefficient_evaluation=CoefficientEvaluation.MIDPOINT)
    )

    solution = solver.solve(
        grid=grid,
        coefficient_provider=provider,
        boundaries=BoundaryPair1D.natural(),
        direction=PDEDirection.BACKWARD,
        time_grid=time_grid,
        values=terminal,
    )

    np.testing.assert_allclose(provider.calls, [0.7, 0.25, 0.05], rtol=0.0, atol=1.0e-15)
    expected = 1.0
    for lower, upper in reversed(list(zip(time_grid[:-1], time_grid[1:]))):
        midpoint = 0.5 * (lower + upper)
        expected /= 1.0 + (upper - lower) * (0.2 + midpoint)
    np.testing.assert_allclose(solution.values, expected, rtol=0.0, atol=2.0e-15)
    assert solution.time == 0.0
    assert [item.coefficient_time for item in solution.diagnostics] == pytest.approx(
        [0.7, 0.25, 0.05]
    )


def test_forward_time_dependent_calls_advance_without_triangular_time_bug() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 9)
    time_grid = np.array([0.0, 0.1, 0.4, 1.0])
    provider = TimeDependentDiscountProvider()
    solver = PDE1DSolver(TimeStepConfig(theta=1.0))
    initial = grid.project_delta(0.0)

    solution = solver.solve(
        grid=grid,
        coefficient_provider=provider,
        boundaries=BoundaryPair1D.natural(),
        direction=PDEDirection.FORWARD,
        time_grid=time_grid,
        values=initial,
    )

    np.testing.assert_allclose(provider.calls, [0.05, 0.25, 0.7], rtol=0.0, atol=1.0e-15)
    expected_mass = 1.0
    for lower, upper in zip(time_grid[:-1], time_grid[1:]):
        midpoint = 0.5 * (lower + upper)
        expected_mass /= 1.0 + (upper - lower) * (0.2 + midpoint)
    assert grid.integrate(solution.values) == pytest.approx(expected_mass, abs=3.0e-15)
    assert max(abs(item.mass_error) for item in solution.diagnostics) < 3.0e-15


def test_backward_black_scholes_price_converges_to_existing_svm_formula() -> None:
    from vanilla_option_pricers import compute_bsm_vanilla_price

    expected = compute_bsm_vanilla_price(
        forward=1.0,
        strike=1.1,
        ttm=1.0,
        vol=0.3,
        optiontype="C",
        discfactor=1.0,
    )
    coarse = _solve_bsm_call(size=101, steps=100)
    fine = _solve_bsm_call(size=201, steps=200)
    coarse_error = abs(coarse - expected)
    fine_error = abs(fine - expected)

    assert fine_error < coarse_error
    assert fine_error < 2.5e-4


def test_forward_log_density_converges_and_exposes_mass_moment_boundary_diagnostics() -> None:
    coarse_grid, coarse_analytic, coarse = _solve_log_density(size=101, steps=120)
    fine_grid, fine_analytic, fine = _solve_log_density(size=201, steps=240)
    coarse_error = coarse_grid.integrate(np.abs(coarse.values - coarse_analytic))
    fine_error = fine_grid.integrate(np.abs(fine.values - fine_analytic))
    expected_mean = -0.5 * 0.28**2 * 0.8

    assert fine_error < coarse_error
    assert fine_error < 0.012
    assert fine_grid.integrate(fine.values) == pytest.approx(1.0, abs=2.0e-12)
    assert fine_grid.integrate(fine_grid.nodes * fine.values) == pytest.approx(
        expected_mean, abs=4.0e-4
    )
    assert fine_grid.integrate(np.exp(fine_grid.nodes) * fine.values) == pytest.approx(
        1.0, abs=4.0e-4
    )
    assert len(fine.diagnostics) == 240
    assert max(item.negative_mass for item in fine.diagnostics) < 1.0e-14
    assert max(abs(item.mass_error) for item in fine.diagnostics) < 2.0e-12
    assert abs(fine.diagnostics[-1].first_moment - expected_mean) < 4.0e-4
    assert abs(fine.diagnostics[-1].left_boundary_flux) < 1.0e-8
    assert abs(fine.diagnostics[-1].right_boundary_flux) < 1.0e-6


def test_static_forward_backward_solutions_preserve_discrete_payoff_density_duality() -> None:
    grid = Grid1D.concentrated(-2.0, 2.0, 101, center=0.0, scale=0.45)
    provider = _constant_log_forward_provider(grid, volatility=0.32)
    boundaries = BoundaryPair1D.natural()
    times = np.linspace(0.0, 0.7, 81)
    payoff = np.maximum(np.exp(grid.nodes) - 1.05, 0.0)
    density = grid.project_delta(0.0)
    solver = PDE1DSolver(TimeStepConfig(theta=0.6))

    backward = solver.solve(
        grid, provider, boundaries, PDEDirection.BACKWARD, times, payoff
    )
    forward = solver.solve(
        grid, provider, boundaries, PDEDirection.FORWARD, times, density
    )

    initial_price = grid.integrate(density * backward.values)
    terminal_price = grid.integrate(forward.values * payoff)
    assert initial_price == pytest.approx(terminal_price, abs=3.0e-13)


def test_backward_dirichlet_and_linear_boundaries_are_enforced_without_aliasing() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 5)
    provider = StaticCoefficientProvider1D(
        PDE1DCoefficients.constant(grid.size, 0.0, 0.0, 0.0)
    )
    terminal = np.array([9.0, 1.0, 2.0, 4.0, 8.0])
    solver = PDE1DSolver(TimeStepConfig(theta=1.0))

    dirichlet = solver.solve(
        grid,
        provider,
        BoundaryPair1D.dirichlet(0.0, 10.0),
        PDEDirection.BACKWARD,
        np.array([0.0, 1.0]),
        terminal,
    )
    extrapolated = solver.solve(
        grid,
        provider,
        BoundaryPair1D.linear_extrapolation(),
        PDEDirection.BACKWARD,
        np.array([0.0, 1.0]),
        terminal,
    )

    np.testing.assert_allclose(dirichlet.values, [0.0, 1.0, 2.0, 4.0, 10.0])
    np.testing.assert_allclose(extrapolated.values, [0.0, 1.0, 2.0, 4.0, 6.0])
    assert dirichlet.diagnostics[-1].left_boundary_residual == pytest.approx(0.0)
    assert dirichlet.diagnostics[-1].right_boundary_residual == pytest.approx(0.0)
    assert extrapolated.diagnostics[-1].left_boundary_residual == pytest.approx(0.0)
    assert extrapolated.diagnostics[-1].right_boundary_residual == pytest.approx(0.0)
    np.testing.assert_array_equal(terminal, [9.0, 1.0, 2.0, 4.0, 8.0])


def test_saved_states_and_density_policy_are_explicit_and_immutable() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 21)
    provider = _constant_log_forward_provider(grid, volatility=0.2)
    times = np.array([0.0, 0.1, 0.3, 0.6])
    solver = PDE1DSolver(
        TimeStepConfig(theta=1.0),
        density_policy=DensityPolicy(
            negativity_tolerance=1.0e-12,
            mass_error_tolerance=1.0e-12,
            renormalize=True,
        ),
    )
    solution = solver.solve(
        grid,
        provider,
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        times,
        grid.project_delta(0.0),
        save_times=(0.0, 0.3, 0.6),
    )

    assert [state.time for state in solution.saved_states] == [0.0, 0.3, 0.6]
    assert solution.config.scheme is TimeScheme.THETA
    assert all(not state.values.flags.writeable for state in solution.saved_states)
    assert not solution.values.flags.writeable
    assert not solution.time_grid.flags.writeable
    assert all(item.renormalized for item in solution.diagnostics)
    assert all(item.raw_mass is not None for item in solution.diagnostics)


def test_solver_validates_time_grid_and_forward_boundary_contract() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 9)
    provider = _constant_log_forward_provider(grid, volatility=0.2)
    solver = PDE1DSolver()

    with pytest.raises(ValueError, match="strictly increasing"):
        solver.solve(
            grid,
            provider,
            BoundaryPair1D.natural(),
            PDEDirection.BACKWARD,
            np.array([0.0, 0.5, 0.4]),
            np.ones(grid.size),
        )
    with pytest.raises(ValueError, match="natural boundaries"):
        solver.solve(
            grid,
            provider,
            BoundaryPair1D.dirichlet(0.0, 0.0),
            PDEDirection.FORWARD,
            np.array([0.0, 0.5]),
            grid.project_delta(0.0),
        )


def test_nonuniform_extrapolation_preserves_linear_payoff_and_explicit_relation() -> None:
    grid = Grid1D(np.array([0.0, 0.1, 0.3, 0.6, 1.0]))
    provider = StaticCoefficientProvider1D(PDE1DCoefficients.constant(grid.size, 0.1, 0.0, 0.0))
    times = np.array([0.0, 0.1, 0.3])
    payoff = 2.0 * grid.nodes - 1.0
    geometric = BoundaryPair1D.linear_extrapolation()
    np.testing.assert_allclose(boundary_residuals(payoff, geometric, grid=grid), 0.0, atol=2.0e-14)
    solution = PDE1DSolver().solve(
        grid,
        provider,
        geometric,
        PDEDirection.BACKWARD,
        times,
        payoff,
        save_times=tuple(times),
    )
    for state in solution.saved_states:
        np.testing.assert_allclose(state.values, payoff, atol=2.0e-14, rtol=0.0)
    for step in solution.diagnostics:
        np.testing.assert_allclose(
            [step.left_boundary_residual, step.right_boundary_residual], 0.0, atol=2.0e-14
        )

    # These explicit weights are intentionally different from geometric extrapolation.
    boundaries = BoundaryPair1D(
        BoundaryCondition1D.linear_relation(BoundarySide.LEFT, 0.25, 2.0, -1.0),
        BoundaryCondition1D.linear_relation(BoundarySide.RIGHT, 0.0, 2.0, -1.0),
    )
    result = PDE1DSolver().solve(grid, provider, boundaries, PDEDirection.BACKWARD, times, payoff)
    v = result.values
    np.testing.assert_allclose([v[0], v[-1]], [0.25 + 2.0 * v[1] - v[2], 2.0 * v[-2] - v[-3]])
