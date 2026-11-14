"""
Tests for PDE numerical foundations.

The cases pin quadrature geometry, concentrated-node construction, interpolation and delta
projection, tensor marginals, boundary typing, coefficient validation, and tridiagonal
factor/solve behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import (
    BoundaryCondition1D,
    BoundaryKind,
    BoundaryPair1D,
    BoundarySet2D,
    BoundarySide,
    BoundaryTarget,
)
from stochvolmodels.pde_solvers.coefficients import (
    PDE1DCoefficients,
    PDE2DCoefficients,
    StaticCoefficientProvider1D,
    StaticCoefficientProvider2D,
)
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.tridiagonal import TridiagonalOperator


def test_uniform_grid_geometry_and_quadrature() -> None:
    grid = Grid1D.uniform(lower=-2.0, upper=3.0, size=6)

    np.testing.assert_allclose(grid.nodes, np.linspace(-2.0, 3.0, 6))
    np.testing.assert_allclose(grid.cell_edges, [-2.0, -1.5, -0.5, 0.5, 1.5, 2.5, 3.0])
    np.testing.assert_allclose(grid.quadrature_weights, [0.5, 1.0, 1.0, 1.0, 1.0, 0.5])
    assert grid.size == 6
    assert grid.bounds == (-2.0, 3.0)
    assert grid.integrate(np.ones(grid.size)) == pytest.approx(5.0)
    assert grid.integrate(grid.nodes) == pytest.approx(2.5)
    assert grid.nodes.flags.c_contiguous
    assert not grid.nodes.flags.writeable


@pytest.mark.parametrize(
    "nodes, message",
    [
        ([0.0, 1.0], "at least three"),
        ([0.0, 1.0, 1.0], "strictly increasing"),
        ([0.0, 2.0, 1.0], "strictly increasing"),
        ([0.0, np.nan, 1.0], "finite"),
    ],
)
def test_grid_rejects_invalid_nodes(nodes: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Grid1D(np.asarray(nodes))


def test_concentrated_grid_has_exact_bounds_and_tighter_central_spacing() -> None:
    grid = Grid1D.concentrated(lower=-5.0, upper=5.0, size=21, center=0.0, scale=0.4)

    assert grid.nodes[0] == pytest.approx(-5.0)
    assert grid.nodes[-1] == pytest.approx(5.0)
    assert np.all(np.diff(grid.nodes) > 0.0)
    assert np.min(np.diff(grid.nodes)) < np.max(np.diff(grid.nodes))
    assert np.diff(grid.nodes)[9] < np.diff(grid.nodes)[0]


def test_delta_projection_preserves_mass_and_first_moment() -> None:
    grid = Grid1D(np.array([-2.0, -0.7, 0.1, 1.4, 3.0]))

    at_node = grid.project_delta(0.1)
    between_nodes = grid.project_delta(0.6)
    for x0, density in ((0.1, at_node), (0.6, between_nodes)):
        assert grid.integrate(density) == pytest.approx(1.0, abs=1.0e-15)
        assert grid.integrate(grid.nodes * density) == pytest.approx(x0, abs=1.0e-15)
        assert np.count_nonzero(density) <= 2

    with pytest.raises(ValueError, match="inside the grid"):
        grid.project_delta(3.1)


def test_grid_location_and_interpolation_weights() -> None:
    grid = Grid1D(np.array([-1.0, -0.2, 0.5, 2.0]))

    assert grid.locate(-1.0) == (0, 0, 1.0, 0.0)
    assert grid.locate(2.0) == (3, 3, 1.0, 0.0)
    left, right, weight_left, weight_right = grid.locate(0.1)
    assert (left, right) == (1, 2)
    assert weight_left + weight_right == pytest.approx(1.0)
    assert weight_left * grid.nodes[left] + weight_right * grid.nodes[right] == pytest.approx(
        0.1
    )


def test_tensor_grid_shapes_integration_marginals_and_delta() -> None:
    x = Grid1D.uniform(-1.0, 2.0, 5)
    y = Grid1D.uniform(0.0, 4.0, 6)
    grid = TensorGrid2D(x=x, y=y)
    values = 2.0 + grid.x_mesh - 0.5 * grid.y_mesh

    assert grid.shape == (5, 6)
    assert grid.x_mesh.shape == grid.shape
    assert grid.y_mesh.shape == grid.shape
    assert grid.quadrature_weights.shape == grid.shape
    assert grid.integrate(np.ones(grid.shape)) == pytest.approx(12.0)
    assert grid.integrate(values) == pytest.approx(18.0)
    np.testing.assert_allclose(grid.marginalize_x(np.ones(grid.shape)), 3.0 * np.ones(y.size))
    np.testing.assert_allclose(grid.marginalize_y(np.ones(grid.shape)), 4.0 * np.ones(x.size))

    delta = grid.project_delta(x0=0.25, y0=1.2)
    assert grid.integrate(delta) == pytest.approx(1.0, abs=1.0e-15)
    assert grid.integrate(grid.x_mesh * delta) == pytest.approx(0.25, abs=1.0e-15)
    assert grid.integrate(grid.y_mesh * delta) == pytest.approx(1.2, abs=1.0e-15)

    with pytest.raises(ValueError, match="expected shape"):
        grid.validate_shape(np.zeros((y.size, x.size)))


def test_boundary_conditions_are_typed_immutable_and_asymmetric() -> None:
    left = BoundaryCondition1D.linear_extrapolation(
        BoundarySide.LEFT, target=BoundaryTarget.BACKWARD_VALUE
    )
    right = BoundaryCondition1D.linear_extrapolation(
        BoundarySide.RIGHT, target=BoundaryTarget.FORWARD_ADJOINT
    )
    pair = BoundaryPair1D(left=left, right=right)
    boundaries = BoundarySet2D(x=pair, y=BoundaryPair1D.natural())

    assert left.kind is BoundaryKind.LINEAR_EXTRAPOLATION
    assert left.target is BoundaryTarget.BACKWARD_VALUE
    assert left.relation is None
    assert right.target is BoundaryTarget.FORWARD_ADJOINT
    assert right.relation is None
    assert boundaries.x.left is not boundaries.x.right
    assert boundaries.y.left is not boundaries.y.right
    assert BoundaryPair1D.dirichlet(left_value=1.0, right_value=2.0).right.value == 2.0

    with pytest.raises(ValueError, match="finite"):
        BoundaryCondition1D.dirichlet(np.inf)
    with pytest.raises(ValueError, match="only valid"):
        BoundaryCondition1D(kind=BoundaryKind.NATURAL, value=1.0)
    with pytest.raises((AttributeError, TypeError)):
        pair.left = right  # type: ignore[misc]


def test_1d_coefficients_validate_shape_values_and_provider_grid() -> None:
    coefficients = PDE1DCoefficients.constant(
        size=5, variance=0.2, drift=-0.1, discount=0.03
    )
    provider = StaticCoefficientProvider1D(coefficients)
    grid = Grid1D.uniform(-1.0, 1.0, 5)

    assert provider.evaluate(0.4, grid) is coefficients
    np.testing.assert_allclose(coefficients.variance, 0.2)
    assert not coefficients.variance.flags.writeable
    with pytest.raises(ValueError, match="grid size"):
        provider.evaluate(0.4, Grid1D.uniform(-1.0, 1.0, 6))
    with pytest.raises(ValueError, match="same shape"):
        PDE1DCoefficients(np.ones(4), np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="non-negative"):
        PDE1DCoefficients.constant(5, variance=-0.1, drift=0.0, discount=0.0)


def test_2d_coefficients_validate_covariance_psd_and_provider_grid() -> None:
    shape = (4, 5)
    coefficients = PDE2DCoefficients.constant(
        shape=shape,
        variance_x=0.5,
        drift_x=-0.5,
        variance_y=0.18,
        drift_y=0.0,
        covariance_xy=0.48,
        discount=0.0,
    )
    provider = StaticCoefficientProvider2D(coefficients)
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 4), Grid1D.uniform(0.1, 1.0, 5))

    assert provider.evaluate(0.2, grid) is coefficients
    assert np.min(coefficients.covariance_psd_margin) == pytest.approx(0.1296)
    with pytest.raises(ValueError, match="grid shape"):
        provider.evaluate(
            0.2,
            TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 5), Grid1D.uniform(0.1, 1.0, 5)),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        PDE2DCoefficients.constant(
            shape,
            variance_x=0.5,
            drift_x=0.0,
            variance_y=0.18,
            drift_y=0.0,
            covariance_xy=0.61,
            discount=0.0,
        )


def test_tridiagonal_apply_solve_and_factor_reuse_match_dense_numpy() -> None:
    rng = np.random.default_rng(8122026)
    for size in (2, 3, 8, 17):
        lower = rng.normal(scale=0.2, size=size - 1)
        upper = rng.normal(scale=0.2, size=size - 1)
        diagonal = np.full(size, 1.0)
        diagonal[:-1] += np.abs(upper)
        diagonal[1:] += np.abs(lower)
        operator = TridiagonalOperator(lower=lower, diagonal=diagonal, upper=upper)
        dense = operator.to_dense()
        rhs = rng.normal(size=size)
        rhs_matrix = rng.normal(size=(size, 4))

        np.testing.assert_allclose(operator.apply(rhs), dense @ rhs, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            operator.apply(rhs_matrix), dense @ rhs_matrix, rtol=0.0, atol=1.0e-15
        )
        np.testing.assert_allclose(
            operator.solve(rhs), np.linalg.solve(dense, rhs), rtol=2.0e-14, atol=2.0e-14
        )
        factorization = operator.factor()
        np.testing.assert_allclose(
            factorization.solve(rhs_matrix),
            np.linalg.solve(dense, rhs_matrix),
            rtol=2.0e-14,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(operator.row_sums, dense.sum(axis=1))
        assert operator.is_weakly_diagonally_dominant
        assert not operator.diagonal.flags.writeable


def test_tridiagonal_rejects_bad_shapes_and_zero_pivot() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        TridiagonalOperator(np.ones(2), np.ones((3, 1)), np.ones(2))
    with pytest.raises(ValueError, match="length n - 1"):
        TridiagonalOperator(np.ones(3), np.ones(3), np.ones(2))

    operator = TridiagonalOperator(
        lower=np.array([1.0, 1.0]),
        diagonal=np.array([0.0, 1.0, 1.0]),
        upper=np.array([1.0, 1.0]),
    )
    with pytest.raises(np.linalg.LinAlgError, match="pivot"):
        operator.factor()



@pytest.mark.parametrize(
    ("a", "b", "cov"),
    [
        (0.0, 0.0, 1.0e-7),
        (0.0, 0.5, -1.0e-200),
        (0.5, 0.0, 1.0e-300),
        (1.0e-150, 1.0e-150, 2.01e-150),
        (1.0e-250, 1.0e-250, -2.01e-250),
    ],
)
def test_covariance_rejects_indefinite_diffusion_at_small_scales(a, b, cov) -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        PDE2DCoefficients.constant((3, 3), a, 0.0, b, 0.0, cov, 0.0)


@pytest.mark.parametrize(
    ("a", "b", "cov"),
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (1.0e-250, 1.0e-250, -2.0e-250),
        (1.0e-200, 1.0e200, 2.0),
        (0.125, 0.5, -0.5),
    ],
)
def test_covariance_accepts_deterministic_axes_and_perfect_correlation(a, b, cov) -> None:
    coefficients = PDE2DCoefficients.constant((3, 3), a, 0.0, b, 0.0, cov, 0.0)
    np.testing.assert_array_equal(coefficients.covariance_xy, cov)
