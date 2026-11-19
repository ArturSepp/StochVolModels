"""
Tests for generic split two-dimensional PDE operators.

The cases verify bivariate polynomial exactness, full and mixed adjoints, zero-covariance
initialization, discount/mass identities, directional line solves, and observable axis
monotonicity switches.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import (
    BoundaryCondition1D, BoundaryPair1D, BoundarySet2D, BoundarySide,
)
from stochvolmodels.pde_solvers.coefficients import PDE2DCoefficients
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import DriftDiscretization, PDEDirection

from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D


def _boundaries() -> BoundarySet2D:
    return BoundarySet2D(x=BoundaryPair1D.natural(), y=BoundaryPair1D.natural())


def _smooth_coefficients(grid: TensorGrid2D, covariance: float = 0.03) -> PDE2DCoefficients:
    return PDE2DCoefficients(
        variance_x=0.18 + 0.02 * np.cos(grid.x_mesh) ** 2,
        drift_x=0.03 + 0.01 * grid.y_mesh,
        variance_y=0.12 + 0.015 * np.sin(grid.y_mesh) ** 2,
        drift_y=-0.02 + 0.008 * grid.x_mesh,
        covariance_xy=np.full(grid.shape, covariance),
        discount=0.01 + 0.004 * (grid.x_mesh**2 + grid.y_mesh**2),
    )


def test_split_backward_operator_is_exact_for_bivariate_quadratic() -> None:
    grid = TensorGrid2D(
        Grid1D(np.array([-1.2, -0.5, -0.1, 0.3, 0.9, 1.6])),
        Grid1D(np.array([-1.4, -0.6, -0.15, 0.2, 0.75, 1.5])),
    )
    coefficients = _smooth_coefficients(grid)
    operator = SplitOperator2D.build(
        grid,
        coefficients,
        _boundaries(),
        drift_discretization=DriftDiscretization.CENTERED,
    )
    values = grid.x_mesh**2 + grid.x_mesh * grid.y_mesh + 2.0 * grid.y_mesh**2
    expected = (
        2.0 * coefficients.variance_x
        + coefficients.drift_x * (2.0 * grid.x_mesh + grid.y_mesh)
        + 4.0 * coefficients.variance_y
        + coefficients.drift_y * (grid.x_mesh + 4.0 * grid.y_mesh)
        + coefficients.covariance_xy
        - coefficients.discount * values
    )
    a0 = operator.apply_a0(values, PDEDirection.BACKWARD)
    a1 = operator.apply_a1(values, PDEDirection.BACKWARD)
    a2 = operator.apply_a2(values, PDEDirection.BACKWARD)
    full = operator.apply_full(values, PDEDirection.BACKWARD)

    np.testing.assert_allclose(full, a0 + a1 + a2, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(full[1:-1, 1:-1], expected[1:-1, 1:-1], atol=2.0e-13)
    np.testing.assert_allclose(
        a0[1:-1, 1:-1], coefficients.covariance_xy[1:-1, 1:-1], atol=2.0e-14
    )


@pytest.mark.parametrize(
    "grid",
    [
        TensorGrid2D(Grid1D.uniform(-2.0, 2.0, 21), Grid1D.uniform(-1.5, 1.5, 19)),
        TensorGrid2D(
            Grid1D.concentrated(-2.0, 2.0, 21, center=0.1, scale=0.4),
            Grid1D.concentrated(-1.5, 1.5, 19, center=-0.2, scale=0.35),
        ),
    ],
)
def test_full_and_mixed_weighted_adjoint_identities(grid: TensorGrid2D) -> None:
    rng = np.random.default_rng(314159)
    operator = SplitOperator2D.build(grid, _smooth_coefficients(grid), _boundaries())
    backward_values = rng.normal(size=grid.shape)
    density = rng.uniform(0.05, 1.0, size=grid.shape)

    for apply in (operator.apply_a0, operator.apply_full):
        left = grid.integrate(density * apply(backward_values, PDEDirection.BACKWARD))
        right = grid.integrate(backward_values * apply(density, PDEDirection.FORWARD))
        assert left == pytest.approx(right, abs=2.0e-12)
    assert operator.weighted_adjoint_residual(backward_values, density) < 2.0e-12


def test_zero_covariance_mixed_path_is_initialized_to_exact_zero() -> None:
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 9), Grid1D.uniform(-1.2, 1.2, 11))
    coefficients = PDE2DCoefficients.constant(
        grid.shape,
        variance_x=0.2,
        drift_x=0.01,
        variance_y=0.1,
        drift_y=-0.02,
        covariance_xy=0.0,
        discount=0.0,
    )
    operator = SplitOperator2D.build(grid, coefficients, _boundaries())
    values = np.arange(np.prod(grid.shape), dtype=np.float64).reshape(grid.shape)

    np.testing.assert_array_equal(operator.apply_a0(values, PDEDirection.BACKWARD), 0.0)
    np.testing.assert_array_equal(operator.apply_a0(values, PDEDirection.FORWARD), 0.0)


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("small", [False, True])
def test_mixed_derivative_matches_dense_action_with_variable_covariance(layout, small):
    """Check nonuniform center weights, endpoint transfers and both adjoint directions."""
    grid = TensorGrid2D(
        Grid1D.uniform(-1., 1., 3) if small else
        Grid1D(np.array([-1.5, -.8, -.2, .15, .6, 1.8, 2.5])),
        Grid1D(np.array([-1.3, -.5, .1, .4, 1.2])),
    )
    coeff = replace(_smooth_coefficients(grid),
                    covariance_xy=.08 * np.tanh(grid.x_mesh + 2 * grid.y_mesh))
    op = SplitOperator2D.build(grid, coeff, _boundaries())
    rng = np.random.default_rng(20261119)
    point = grid.project_delta(grid.x.nodes[1], grid.y.nodes[1])
    edge = np.zeros(grid.shape)
    edge[[0, -1], :] = rng.normal(size=(2, grid.y.size))
    edge[:, [0, -1]] = rng.normal(size=(grid.x.size, 2))
    for state in (rng.normal(size=grid.shape), point, edge):
        if layout == "fortran":
            state = np.asfortranarray(state)
        elif layout == "strided":
            padded = np.zeros((2 * grid.x.size, 2 * grid.y.size))
            padded[::2, ::2] = state
            state = padded[::2, ::2]
        original = state.copy()
        state.setflags(write=False)
        for direction in PDEDirection:
            if direction is PDEDirection.BACKWARD:
                expected = coeff.covariance_xy * (op.derivative_x @ state @ op.derivative_y.T)
            else:
                weighted = grid.quadrature_weights * coeff.covariance_xy * state
                expected = (op.derivative_x.T @ weighted @ op.derivative_y)
                expected /= grid.quadrature_weights
            actual = op.apply_a0(state, direction)
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
        np.testing.assert_array_equal(state, original)


def test_nonuniform_mixed_derivative_retains_center_contribution():
    """The four-corner secant would not differentiate x**2*y exactly on this grid."""
    grid = TensorGrid2D(Grid1D(np.array([-.8, -.2, .15, .6, 1.5])),
                        Grid1D(np.array([-.9, -.4, .1, .8])))
    op = SplitOperator2D.build(grid, _smooth_coefficients(grid), _boundaries())
    values = grid.x_mesh**2 * grid.y_mesh
    expected = 2 * grid.x_mesh * op.coefficients.covariance_xy
    np.testing.assert_allclose(op.apply_a0(values, PDEDirection.BACKWARD)[1:-1, 1:-1],
                               expected[1:-1, 1:-1], rtol=0., atol=2e-15)


def test_supplied_mixed_derivative_matrices_keep_their_dense_action():
    """Custom matrices passed to the public dataclass need not have three bands."""
    grid = TensorGrid2D(Grid1D.uniform(-1., 1., 5), Grid1D.uniform(-1., 1., 4))
    op = SplitOperator2D.build(grid, _smooth_coefficients(grid), _boundaries())
    dx, dy = op.derivative_x.copy(), op.derivative_y.copy()
    dx[1, -1], dy[1, -1] = .3, -.2
    op = replace(op, derivative_x=dx, derivative_y=dy)
    state = np.arange(np.prod(grid.shape)).reshape(grid.shape) / 10.
    np.testing.assert_allclose(op.apply_a0(state, PDEDirection.BACKWARD),
                               op.coefficients.covariance_xy * (dx @ state @ dy.T))
    weighted = grid.quadrature_weights * op.coefficients.covariance_xy * state
    np.testing.assert_allclose(op.apply_a0(state, PDEDirection.FORWARD),
                               (dx.T @ weighted @ dy) / grid.quadrature_weights)


def test_backward_constant_and_forward_mass_rate_include_full_discount_once() -> None:
    grid = TensorGrid2D(
        Grid1D.concentrated(-1.5, 1.5, 17, center=0.0, scale=0.35),
        Grid1D.concentrated(-1.2, 1.2, 15, center=0.1, scale=0.3),
    )
    coefficients = _smooth_coefficients(grid)
    operator = SplitOperator2D.build(grid, coefficients, _boundaries())
    density = np.exp(-grid.x_mesh**2 - 1.3 * grid.y_mesh**2)

    np.testing.assert_allclose(
        operator.apply_full(np.ones(grid.shape), PDEDirection.BACKWARD),
        -coefficients.discount,
        rtol=0.0,
        atol=2.0e-12,
    )
    mass_rate = grid.integrate(operator.apply_full(density, PDEDirection.FORWARD))
    expected_rate = -grid.integrate(coefficients.discount * density)
    assert mass_rate == pytest.approx(expected_rate, abs=2.0e-12)


def test_directional_implicit_line_solves_match_dense_references() -> None:
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 8), Grid1D.uniform(-1.2, 1.2, 7))
    operator = SplitOperator2D.build(grid, _smooth_coefficients(grid), _boundaries())
    rng = np.random.default_rng(901)
    rhs = rng.normal(size=grid.shape)
    step_size = 0.03
    theta = 0.6

    solved_x = operator.solve_a1(rhs, PDEDirection.BACKWARD, step_size, theta)
    for y_index, line in enumerate(operator.x_lines):
        matrix = line.implicit_matrix(PDEDirection.BACKWARD, step_size, theta).to_dense()
        np.testing.assert_allclose(
            solved_x[:, y_index], np.linalg.solve(matrix, rhs[:, y_index]), atol=3.0e-14
        )

    solved_y = operator.solve_a2(rhs, PDEDirection.BACKWARD, step_size, theta)
    for x_index, line in enumerate(operator.y_lines):
        matrix = line.implicit_matrix(PDEDirection.BACKWARD, step_size, theta).to_dense()
        np.testing.assert_allclose(
            solved_y[x_index, :], np.linalg.solve(matrix, rhs[x_index, :]), atol=3.0e-14
        )


@pytest.mark.parametrize("direction", list(PDEDirection))
def test_factor_reuse_tracks_exact_scale_and_operator(monkeypatch, direction):
    from stochvolmodels.pde_solvers.operators_2d import _LineBatch

    calls = []
    factor = _LineBatch._factor

    def count(batch, scale):
        calls.append(scale)
        return factor(batch, scale)

    monkeypatch.setattr(_LineBatch, "_factor", count)
    grid = TensorGrid2D(Grid1D.uniform(-1., 1., 8), Grid1D.uniform(-1., 1., 6))
    coeff = _smooth_coefficients(grid)
    op = SplitOperator2D.build(grid, coeff, _boundaries())
    rhs = np.random.default_rng(17).normal(size=grid.shape)
    h = 1 / 32
    close = np.nextafter(h, 1.)
    for step, theta in [(h, 1.), (2 * h, .5), (close, 1.), (h, 1.)]:
        for solve, lines, axis in [(op.solve_a1, op.x_lines, 0), (op.solve_a2, op.y_lines, 1)]:
            result = solve(rhs, direction, step, theta)
            expected = [
                np.linalg.solve(line.implicit_matrix(direction, step, theta).to_dense(), values)
                for line, values in zip(lines, np.moveaxis(rhs, axis, -1), strict=True)
            ]
            np.testing.assert_allclose(np.moveaxis(result, axis, -1), expected, atol=3e-14, rtol=0)
    assert calls == [h, h, close, close]

    # A change of direction or a new coefficient bucket must create new factors.
    opposite = PDEDirection.FORWARD if direction is PDEDirection.BACKWARD else PDEDirection.BACKWARD
    op.solve_a1(rhs, opposite, h, 1.)
    changed = SplitOperator2D.build(grid, replace(coeff, variance_x=2. * coeff.variance_x),
                                    _boundaries())
    changed_values = changed.solve_a1(rhs, direction, h, 1.)
    assert not np.allclose(changed_values, op.solve_a1(rhs, direction, h, 1.))
    assert calls[4:] == [h, h]


@pytest.mark.parametrize("n", [3, 8])
def test_prescribed_line_factors_match_full_boundary_equations(n):
    grid = TensorGrid2D(Grid1D.uniform(-1., 1., n), Grid1D.uniform(-0.5, 1.5, 5))
    if n == 3:
        left, right = (0.7, 0., 0.), (-0.2, 0., 0.)
        bounds = BoundaryPair1D.dirichlet(left[0], right[0])
    else:
        left, right = (0.3, 1.4, -0.4), (-0.2, 1.6, -0.6)
        bounds = BoundaryPair1D(
            BoundaryCondition1D.linear_relation(BoundarySide.LEFT, *left),
            BoundaryCondition1D.linear_relation(BoundarySide.RIGHT, *right),
        )
    op = SplitOperator2D.build(grid, _smooth_coefficients(grid),
                               BoundarySet2D(bounds, BoundaryPair1D.natural()))
    rng = np.random.default_rng(18)
    for _ in range(2):
        rhs = rng.normal(size=grid.shape)
        result = op.solve_a1(rhs, PDEDirection.BACKWARD, .031, .7)
        for j, line in enumerate(op.x_lines):
            matrix = line.implicit_matrix(PDEDirection.BACKWARD, .031, .7).to_dense()
            matrix[[0, -1]] = 0.
            matrix[0, :3] = [1., -left[1], -left[2]]
            matrix[-1, -3:] = [-right[2], -right[1], 1.]
            column = rhs[:, j].copy()
            column[[0, -1]] = [left[0], right[0]]
            np.testing.assert_allclose(result[:, j], np.linalg.solve(matrix, column),
                                       atol=1e-14, rtol=0)
    with pytest.raises(ValueError, match="natural boundaries"):
        op.solve_a1(rhs, PDEDirection.FORWARD, .031, .7)


def test_one_singular_line_rejects_batch_and_other_scales_still_solve():
    grid = TensorGrid2D(Grid1D.uniform(-1., 1., 7), Grid1D.uniform(-1., 1., 5))
    coeff = PDE2DCoefficients.constant(grid.shape, 0., 0., 0., 0., 0., 0.)
    discount = np.zeros(grid.shape)
    discount[:, 1] = -2.
    op = SplitOperator2D.build(grid, replace(coeff, discount=discount), _boundaries())
    rhs = np.ones(grid.shape)
    for _ in range(2):
        with pytest.raises(np.linalg.LinAlgError, match="pivot"):
            op.solve_a1(rhs, PDEDirection.FORWARD, 1., 1.)
        np.testing.assert_array_equal(op.solve_a1(rhs, PDEDirection.FORWARD, .5, 1.),
                                      rhs / (1. + .25 * discount))


def test_axis_monotonicity_switches_are_observable() -> None:
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 9), Grid1D.uniform(-1.0, 1.0, 7))
    coefficients = PDE2DCoefficients.constant(
        grid.shape,
        variance_x=1.0e-7,
        drift_x=1.0,
        variance_y=1.0e-7,
        drift_y=-1.0,
        covariance_xy=0.0,
        discount=0.0,
    )
    operator = SplitOperator2D.build(grid, coefficients, _boundaries())

    assert operator.one_sided_x_nodes == (grid.x.size - 2) * grid.y.size
    assert operator.one_sided_y_nodes == (grid.y.size - 2) * grid.x.size
    assert all(line.backward.has_nonnegative_off_diagonal for line in operator.x_lines)
    assert all(line.backward.has_nonnegative_off_diagonal for line in operator.y_lines)
