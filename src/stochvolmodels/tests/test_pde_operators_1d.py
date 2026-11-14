"""
Tests for generic one-dimensional PDE operators.

The cases pin nonuniform polynomial exactness, weighted adjointness, the historical conservative
uniform stencil, monotonicity fallbacks, discount/mass identities, boundary contracts, and
implicit theta matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundaryTarget
from stochvolmodels.pde_solvers.coefficients import PDE1DCoefficients
from stochvolmodels.pde_solvers.grids import Grid1D
from stochvolmodels.pde_solvers.operators_1d import (
    DriftDiscretization,
    Operator1D,
    PDEDirection,
)


def test_centered_nonuniform_stencil_is_exact_for_quadratic() -> None:
    grid = Grid1D(np.array([-1.4, -0.7, -0.2, 0.1, 0.55, 1.4]))
    variance = 0.7 + 0.1 * grid.nodes**2
    drift = 0.2 - 0.05 * grid.nodes
    discount = 0.03 + 0.01 * grid.nodes**2
    coefficients = PDE1DCoefficients(variance, drift, discount)
    operator = Operator1D.build(
        grid,
        coefficients,
        BoundaryPair1D.natural(),
        drift_discretization=DriftDiscretization.CENTERED,
    )
    values = grid.nodes**2
    expected = 2.0 * variance + 2.0 * drift * grid.nodes - discount * values

    np.testing.assert_allclose(
        operator.apply(values, PDEDirection.BACKWARD)[1:-1],
        expected[1:-1],
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    "grid",
    [
        Grid1D.uniform(-2.0, 2.0, 41),
        Grid1D.concentrated(-2.0, 2.0, 41, center=0.15, scale=0.35),
    ],
)
def test_weighted_forward_backward_adjoint_identity(grid: Grid1D) -> None:
    rng = np.random.default_rng(1731)
    coefficients = PDE1DCoefficients(
        variance=0.08 + 0.03 * np.cos(grid.nodes) ** 2,
        drift=0.12 * np.sin(1.7 * grid.nodes),
        discount=0.01 + 0.02 * (grid.nodes / 2.0) ** 2,
    )
    operator = Operator1D.build(grid, coefficients, BoundaryPair1D.natural())
    backward_values = rng.normal(size=grid.size)
    density = rng.uniform(0.1, 1.0, size=grid.size)

    backward_pairing = grid.integrate(
        density * operator.apply(backward_values, PDEDirection.BACKWARD)
    )
    forward_pairing = grid.integrate(
        backward_values * operator.apply(density, PDEDirection.FORWARD)
    )

    assert backward_pairing == pytest.approx(forward_pairing, abs=2.0e-14)
    assert operator.weighted_adjoint_residual(backward_values, density) < 2.0e-14


def test_forward_operator_matches_legacy_conservative_stencil_on_uniform_interior() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 13)
    variance = 0.2 + 0.03 * (1.0 + grid.nodes) ** 2
    drift = 0.015 * np.cos(grid.nodes)
    discount = 0.01 + 0.005 * grid.nodes**2
    coefficients = PDE1DCoefficients(variance, drift, discount)
    operator = Operator1D.build(
        grid,
        coefficients,
        BoundaryPair1D.natural(),
        drift_discretization=DriftDiscretization.CENTERED,
    )
    dense_forward = operator.forward.to_dense()
    spacing = grid.spacings[0]

    for index in range(2, grid.size - 2):
        legacy_lower = variance[index - 1] / spacing**2 + drift[index - 1] / (
            2.0 * spacing
        )
        legacy_diagonal = -2.0 * variance[index] / spacing**2 - discount[index]
        legacy_upper = variance[index + 1] / spacing**2 - drift[index + 1] / (
            2.0 * spacing
        )
        np.testing.assert_allclose(
            dense_forward[index, index - 1 : index + 2],
            [legacy_lower, legacy_diagonal, legacy_upper],
            rtol=0.0,
            atol=5.0e-14,
        )


def test_monotonicity_fallback_switches_to_expected_one_sided_stencils() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 7)
    coefficients = PDE1DCoefficients(
        variance=np.full(grid.size, 1.0e-6),
        drift=np.array([1.0, 1.0, 0.7, 0.0, -0.7, -1.0, -1.0]),
        discount=np.zeros(grid.size),
    )
    fallback = Operator1D.build(grid, coefficients, BoundaryPair1D.natural())
    centered = Operator1D.build(
        grid,
        coefficients,
        BoundaryPair1D.natural(),
        drift_discretization=DriftDiscretization.CENTERED,
    )

    np.testing.assert_array_equal(fallback.drift_stencil[1:-1], [1, 1, 0, -1, -1])
    np.testing.assert_array_equal(fallback.one_sided_mask, fallback.drift_stencil != 0)
    assert fallback.backward.has_nonnegative_off_diagonal
    assert fallback.forward.has_nonnegative_off_diagonal
    assert not centered.backward.has_nonnegative_off_diagonal
    assert not fallback.drift_stencil.flags.writeable


def test_constant_preservation_and_discounted_forward_mass_rate() -> None:
    grid = Grid1D.concentrated(-2.0, 2.0, 35, center=0.0, scale=0.4)
    discount = 0.02 + 0.01 * np.cos(grid.nodes) ** 2
    coefficients = PDE1DCoefficients(
        variance=0.1 + 0.02 * grid.nodes**2,
        drift=-0.05 * grid.nodes,
        discount=discount,
    )
    operator = Operator1D.build(grid, coefficients, BoundaryPair1D.natural())
    density = np.exp(-grid.nodes**2)

    np.testing.assert_allclose(
        operator.apply(np.ones(grid.size), PDEDirection.BACKWARD),
        -discount,
        rtol=0.0,
        atol=5.0e-14,
    )
    mass_rate = grid.integrate(operator.apply(density, PDEDirection.FORWARD))
    expected_rate = -grid.integrate(discount * density)
    assert mass_rate == pytest.approx(expected_rate, abs=5.0e-14)


def test_operator_rejects_forward_boundary_as_backward_source() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 7)
    coefficients = PDE1DCoefficients.constant(grid.size, 0.1, 0.0, 0.0)
    boundaries = BoundaryPair1D.natural(target=BoundaryTarget.FORWARD_ADJOINT)

    with pytest.raises(ValueError, match="backward-value"):
        Operator1D.build(grid, coefficients, boundaries)


def test_implicit_matrix_has_expected_theta_form() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 7)
    coefficients = PDE1DCoefficients.constant(grid.size, 0.1, -0.02, 0.03)
    operator = Operator1D.build(grid, coefficients, BoundaryPair1D.natural())
    step_size = 0.07
    theta = 0.6

    matrix = operator.implicit_matrix(PDEDirection.BACKWARD, step_size, theta).to_dense()
    expected = np.eye(grid.size) - theta * step_size * operator.backward.to_dense()
    np.testing.assert_allclose(matrix, expected, rtol=0.0, atol=1.0e-15)
