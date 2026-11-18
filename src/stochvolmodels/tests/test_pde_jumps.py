"""
Numerical tests for weighted-adjoint deterministic and compound-Poisson jumps.

The cases verify weighted-adjoint translations, mass and moment diagnostics, analytic compound-
Poisson mixtures, lognormal jump-diffusion martingales, Strang splitting, tensor translations,
and explicit-jump stability limits.
"""

from __future__ import annotations


import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D
from stochvolmodels.pde_solvers.coefficients import (
    PDE1DCoefficients,
    StaticCoefficientProvider1D,
)
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import Operator1D, PDEDirection
from stochvolmodels.pde_solvers.solver_1d import PDE1DSolver, TimeStepConfig

from stochvolmodels.pricers.density_pricer import (
    CoordinateSystem,
    DensityPricer,
    DensityState1D,
)
from stochvolmodels.pde_solvers.jumps import (
    JumpCoordinate,
    TranslationJump,
    log_forward_jump_compensator,
    strang_split_step,
)


@pytest.mark.parametrize(
    "grid,coordinate,jump_size",
    [
        (
            Grid1D.concentrated(-2.0, 2.0, 53, center=0.1, scale=0.4),
            JumpCoordinate.ADDITIVE,
            0.27,
        ),
        (
            Grid1D.concentrated(0.2, 2.2, 51, center=1.0, scale=0.3),
            JumpCoordinate.LEVEL_MULTIPLICATIVE,
            -0.18,
        ),
    ],
)
def test_translation_forward_is_weighted_adjoint_of_backward_interpolation(
    grid: Grid1D, coordinate: JumpCoordinate, jump_size: float
) -> None:
    jump = TranslationJump(grid, intensity=1.3, jump_size=jump_size, coordinate=coordinate)
    rng = np.random.default_rng(123456)
    value = rng.normal(size=grid.size)
    density = rng.uniform(0.05, 1.0, size=grid.size)

    backward = jump.translate(value, PDEDirection.BACKWARD)
    forward = jump.translate(density, PDEDirection.FORWARD)

    assert grid.integrate(density * backward) == pytest.approx(
        grid.integrate(forward * value), abs=3.0e-14
    )
    np.testing.assert_allclose(
        jump.translate(np.ones(grid.size), PDEDirection.BACKWARD),
        1.0,
        rtol=0.0,
        atol=0.0,
    )
    assert grid.integrate(forward) == pytest.approx(grid.integrate(density), abs=3.0e-14)


def test_translation_reports_additive_multiplicative_and_boundary_moment_effects() -> None:
    additive_grid = Grid1D.uniform(-2.0, 2.0, 101)
    additive = TranslationJump(
        additive_grid, intensity=0.8, jump_size=0.36, coordinate=JumpCoordinate.ADDITIVE
    )
    additive_result = additive.translation_step(
        additive_grid.project_delta(0.0), PDEDirection.FORWARD
    )
    assert additive_result.diagnostics.mass_after == pytest.approx(1.0, abs=2.0e-15)
    assert additive_result.diagnostics.first_moment_after == pytest.approx(0.36, abs=2.0e-15)

    level_grid = Grid1D.uniform(0.1, 3.0, 291)
    multiplicative = TranslationJump(
        level_grid,
        intensity=0.8,
        jump_size=np.log(1.2),
        coordinate=JumpCoordinate.LEVEL_MULTIPLICATIVE,
    )
    level_result = multiplicative.translation_step(
        level_grid.project_delta(1.0), PDEDirection.FORWARD
    )
    assert level_result.diagnostics.first_moment_after == pytest.approx(1.2, abs=3.0e-15)

    boundary_jump = TranslationJump(
        additive_grid, intensity=1.0, jump_size=0.4, coordinate=JumpCoordinate.ADDITIVE
    )
    boundary_result = boundary_jump.translation_step(
        additive_grid.project_delta(1.9), PDEDirection.FORWARD
    )
    diagnostics = boundary_result.diagnostics
    assert diagnostics.mass_after == pytest.approx(1.0, abs=2.0e-15)
    assert diagnostics.first_moment_after == pytest.approx(2.0, abs=2.0e-15)
    assert diagnostics.upper_boundary_mass_change > 0.99
    assert diagnostics.lower_boundary_mass_change == pytest.approx(0.0)


def test_compound_poisson_step_preserves_mass_and_full_discrete_duality() -> None:
    grid = Grid1D.concentrated(-2.5, 2.5, 101, center=0.0, scale=0.45)
    jump = TranslationJump(
        grid, intensity=1.4, jump_size=-0.22, coordinate=JumpCoordinate.ADDITIVE
    )
    rng = np.random.default_rng(8080)
    value = rng.normal(size=grid.size)
    density = rng.uniform(0.1, 1.0, size=grid.size)
    density /= grid.integrate(density)
    backward = jump.poisson_step(value, PDEDirection.BACKWARD, 0.7)
    forward = jump.poisson_step(density, PDEDirection.FORWARD, 0.7)

    assert grid.integrate(density * backward.values) == pytest.approx(
        grid.integrate(forward.values * value), abs=5.0e-14
    )
    assert forward.diagnostics.mass_after == pytest.approx(1.0, abs=5.0e-14)
    assert forward.diagnostics.retained_probability > 1.0 - 1.0e-13
    assert forward.diagnostics.poisson_terms >= 10
    assert forward.diagnostics.truncated_probability < 1.0e-13


@pytest.mark.parametrize("jump_size", [-0.35, 0.35])
def test_normal_compound_poisson_density_matches_analytic_mixture(
    jump_size: float,
) -> None:
    grid = Grid1D.uniform(-5.0, 5.0, 1001)
    maturity = 0.8
    intensity = 0.9
    volatility = 0.3
    drift = 0.05
    base_mean = drift * maturity
    standard_deviation = volatility * np.sqrt(maturity)
    base_density = np.exp(
        -0.5 * ((grid.nodes - base_mean) / standard_deviation) ** 2
    ) / (np.sqrt(2.0 * np.pi) * standard_deviation)
    jump = TranslationJump(
        grid, intensity, jump_size, JumpCoordinate.ADDITIVE
    )
    numerical = jump.poisson_step(
        base_density, PDEDirection.FORWARD, maturity
    ).values
    weights = _poisson_weights(intensity * maturity, 40)
    analytic = np.zeros(grid.size)
    for count, weight in enumerate(weights):
        mean = base_mean + count * jump_size
        analytic += weight * np.exp(
            -0.5 * ((grid.nodes - mean) / standard_deviation) ** 2
        ) / (np.sqrt(2.0 * np.pi) * standard_deviation)

    assert grid.integrate(np.abs(numerical - analytic)) < 3.0e-12
    assert grid.integrate(numerical) == pytest.approx(1.0, abs=3.0e-13)
    assert grid.integrate(grid.nodes * numerical) == pytest.approx(
        base_mean + intensity * maturity * jump_size, abs=3.0e-12
    )


_MATLAB_JUMP_CASES = (
    pytest.param(
        -0.3,
        np.array(
            [
                0.32800483759663074,
                0.2189382232283433,
                0.13519344049764542,
                0.07663163539705778,
                0.039606416867651095,
            ]
        ),
        id="negative_jump",
    ),
    pytest.param(
        0.3,
        np.array(
            [
                0.3213472480394169,
                0.2208329969012801,
                0.1501305438583566,
                0.10231691598832408,
                0.07035856542028998,
            ]
        ),
        id="positive_jump",
    ),
)


@pytest.mark.parametrize("jump_size,expected_calls", _MATLAB_JUMP_CASES)
def test_lognormal_jump_diffusion_matches_matlab_poisson_mixture_and_martingale(
    jump_size: float, expected_calls: np.ndarray
) -> None:
    maturity = 1.0
    volatility = 0.2
    intensity = 1.0
    grid = Grid1D.uniform(-5.0, 5.0, 1001)
    compensator = log_forward_jump_compensator(intensity, jump_size)
    half_variance = 0.5 * volatility**2
    provider = StaticCoefficientProvider1D(
        PDE1DCoefficients.constant(
            grid.size,
            variance=half_variance,
            drift=-half_variance + compensator,
            discount=0.0,
        )
    )
    diffusion = PDE1DSolver(TimeStepConfig(theta=1.0)).solve(
        grid,
        provider,
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        np.linspace(0.0, maturity, 401),
        grid.project_delta(0.0),
    )
    jump = TranslationJump(
        grid, intensity, jump_size, JumpCoordinate.ADDITIVE
    )
    jumped = jump.poisson_step(
        diffusion.values, PDEDirection.FORWARD, maturity
    ).values
    analytic = _analytic_log_jump_density(
        grid.nodes,
        maturity,
        volatility,
        intensity,
        jump_size,
        compensator,
    )
    state = DensityState1D(
        grid, jumped, maturity, CoordinateSystem.LOG_FORWARD
    )
    strikes = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    calls = DensityPricer(state, forward=1.0).price_slice(strikes).calls

    assert grid.integrate(np.abs(jumped - analytic)) < 0.012
    np.testing.assert_allclose(calls, expected_calls, atol=5.0e-4)
    assert grid.integrate(jumped) == pytest.approx(1.0, abs=5.0e-12)
    assert grid.integrate(np.exp(grid.nodes) * jumped) == pytest.approx(
        1.0, abs=5.0e-4
    )


def test_strang_jump_diffusion_step_and_tensor_translation_preserve_duality() -> None:
    grid = Grid1D.uniform(-2.0, 2.0, 61)
    coefficients = PDE1DCoefficients.constant(grid.size, 0.04, -0.02, 0.0)
    operator = Operator1D.build(grid, coefficients, BoundaryPair1D.natural())
    jump = TranslationJump(
        grid, intensity=0.7, jump_size=0.24, coordinate=JumpCoordinate.ADDITIVE
    )
    theta = 0.6

    def diffusion_step(
        values: np.ndarray, direction: PDEDirection, step_size: float
    ) -> np.ndarray:
        generator = operator.generator(direction)
        rhs = values + (1.0 - theta) * step_size * generator.apply(values)
        return operator.implicit_matrix(direction, step_size, theta).solve(rhs)

    rng = np.random.default_rng(4242)
    value = rng.normal(size=grid.size)
    density = rng.uniform(0.1, 1.0, size=grid.size)
    backward = strang_split_step(
        value, PDEDirection.BACKWARD, 0.04, jump, diffusion_step
    )
    forward = strang_split_step(
        density, PDEDirection.FORWARD, 0.04, jump, diffusion_step
    )
    assert grid.integrate(density * backward) == pytest.approx(
        grid.integrate(forward * value), abs=8.0e-14
    )

    y_grid = Grid1D.concentrated(-1.0, 1.0, 17, center=0.0, scale=0.3)
    tensor_grid = TensorGrid2D(grid, y_grid)
    value_2d = rng.normal(size=tensor_grid.shape)
    density_2d = rng.uniform(0.1, 1.0, size=tensor_grid.shape)
    backward_2d = jump.translate_tensor(
        value_2d, tensor_grid, axis=0, direction=PDEDirection.BACKWARD
    )
    forward_2d = jump.translate_tensor(
        density_2d, tensor_grid, axis=0, direction=PDEDirection.FORWARD
    )
    assert tensor_grid.integrate(density_2d * backward_2d) == pytest.approx(
        tensor_grid.integrate(forward_2d * value_2d), abs=8.0e-13
    )
    backward_poisson_2d = jump.poisson_step_tensor(
        value_2d,
        tensor_grid,
        axis=0,
        direction=PDEDirection.BACKWARD,
        step_size=0.3,
    )
    forward_poisson_2d = jump.poisson_step_tensor(
        density_2d,
        tensor_grid,
        axis=0,
        direction=PDEDirection.FORWARD,
        step_size=0.3,
    )
    assert tensor_grid.integrate(density_2d * backward_poisson_2d) == pytest.approx(
        tensor_grid.integrate(forward_poisson_2d * value_2d), abs=8.0e-13
    )


def test_explicit_jump_step_enforces_positivity_stability_limit() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 21)
    jump = TranslationJump(
        grid, intensity=5.0, jump_size=0.1, coordinate=JumpCoordinate.ADDITIVE
    )
    with pytest.raises(ValueError, match="intensity.*step_size"):
        jump.explicit_step(grid.project_delta(0.0), PDEDirection.FORWARD, 0.21)


def _poisson_weights(mean: float, max_count: int) -> np.ndarray:
    weights = np.empty(max_count + 1)
    weights[0] = np.exp(-mean)
    for count in range(max_count):
        weights[count + 1] = weights[count] * mean / (count + 1)
    weights /= np.sum(weights)
    return weights


def _analytic_log_jump_density(
    nodes: np.ndarray,
    maturity: float,
    volatility: float,
    intensity: float,
    jump_size: float,
    compensator: float,
) -> np.ndarray:
    standard_deviation = volatility * np.sqrt(maturity)
    weights = _poisson_weights(intensity * maturity, 40)
    density = np.zeros_like(nodes)
    for count, weight in enumerate(weights):
        mean = (-0.5 * volatility**2 + compensator) * maturity + count * jump_size
        density += weight * np.exp(
            -0.5 * ((nodes - mean) / standard_deviation) ** 2
        ) / (np.sqrt(2.0 * np.pi) * standard_deviation)
    return density


def test_poisson_term_budget_and_extreme_compensator() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 7)
    jump = TranslationJump(grid, intensity=12.0, jump_size=0.0)
    initial = grid.project_delta(0.0)
    advanced = jump.poisson_step(initial, PDEDirection.FORWARD, 1.0, tail_tolerance=1e-18)
    assert 0.0 < advanced.diagnostics.truncated_probability <= 1e-18
    np.testing.assert_allclose(advanced.values, initial, atol=1e-14, rtol=0)
    with pytest.raises(RuntimeError, match="max_terms"):
        jump.poisson_step(initial, PDEDirection.FORWARD, 1.0, max_terms=3)
    assert log_forward_jump_compensator(0.0, 1000.0) == 0.0
    with pytest.raises(ValueError, match="compensator must be finite"):
        log_forward_jump_compensator(1.0, 1000.0)
