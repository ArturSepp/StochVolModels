"""Independent cell-quadrature, tensor, and grid-refinement controls for jump marks."""

import numpy as np
import pytest
from scipy.integrate import quad_vec

from stochvolmodels.pde_solvers import (
    ExponentialJumpKernel,
    Grid1D,
    PDEDirection,
    TensorGrid2D,
    apply_common_exponential_jump,
)


def quadrature_matrix(grid, rate, side, exterior):
    """Integrate the nodal interpolant's hats directly along each jump ray."""
    x = grid.nodes
    eye = np.eye(x.size)
    matrix = []
    for source in x:
        lower, upper = (x[0], source) if side == "down" else (source, x[-1])

        def integrand(target):
            j = np.clip(np.searchsorted(x, target), 1, x.size - 1)
            fraction = (target - x[j - 1]) / (x[j] - x[j - 1])
            hats = (1 - fraction) * eye[j - 1] + fraction * eye[j]
            return rate * np.exp(-rate * abs(target - source)) * hats

        row = np.zeros(x.size)
        if lower < upper:
            row, _ = quad_vec(integrand, lower, upper, points=x[(lower < x) & (x < upper)])
        if exterior == "clamp":
            row[0 if side == "down" else -1] += np.exp(-rate * (upper - lower))
        matrix.append(row)
    return np.array(matrix)


@pytest.mark.parametrize(
    "grid", [Grid1D.uniform(-1, 2, 9), Grid1D.concentrated(-1, 2, 11, center=0.1, scale=0.18)]
)
@pytest.mark.parametrize("side", ["up", "down"])
@pytest.mark.parametrize("exterior", ["clamp", "zero"])
def test_kernel_against_integrated_hats(grid, side, exterior):
    kernel = ExponentialJumpKernel(grid, 2.7, side, exterior)
    matrix = quadrature_matrix(grid, 2.7, side, exterior)
    actual = kernel.apply(np.eye(grid.size), PDEDirection.BACKWARD)
    np.testing.assert_allclose(actual, matrix, rtol=0, atol=1e-13)
    assert np.min(actual) >= 0
    row_totals = 1.0 if exterior == "clamp" else 1.0 - kernel.exterior_probability
    np.testing.assert_allclose(actual.sum(axis=1), row_totals, rtol=0, atol=1e-14)

    rng = np.random.default_rng(1742)
    density, payoff = rng.random(grid.size), rng.normal(size=grid.size)
    w = grid.quadrature_weights
    forward = kernel.apply(density, PDEDirection.FORWARD)
    np.testing.assert_allclose(forward, (matrix.T @ (w * density)) / w, rtol=0, atol=2e-13)
    lhs = grid.integrate(forward * payoff)
    rhs = grid.integrate(density * kernel.apply(payoff, PDEDirection.BACKWARD))
    assert lhs == pytest.approx(rhs, abs=1e-13)
    loss = kernel.exterior_mass(density) if exterior == "zero" else 0
    assert grid.integrate(density - forward) == pytest.approx(loss, abs=2e-14)


@pytest.mark.parametrize("direction", list(PDEDirection))
def test_common_event_with_independent_marks(direction):
    x = Grid1D.concentrated(-0.7, 0.8, 11, center=0.15, scale=0.2)
    y = Grid1D.uniform(-1.2, 0.6, 7)
    grid = TensorGrid2D(x, y)
    kx, ky = ExponentialJumpKernel(x, 3.1), ExponentialJumpKernel(y, 4.3, "up", "zero")
    a = quadrature_matrix(x, 3.1, "down", "clamp")
    b = quadrature_matrix(y, 4.3, "up", "zero")
    u = np.sin(grid.x_mesh + 2 * grid.y_mesh) + grid.x_mesh * grid.y_mesh
    if direction is PDEDirection.FORWARD:
        w = grid.quadrature_weights
        reference = (a.T @ (w * u) @ b) / w
    else:
        reference = a @ u @ b.T
    combined = apply_common_exponential_jump(u, kx, ky, direction)
    np.testing.assert_allclose(combined, reference, atol=3e-13, rtol=0)


@pytest.mark.parametrize("side", ["down", "up"])
def test_second_order_refinement(side):
    errors = []
    for n in (33, 65, 129, 257):
        grid = Grid1D.concentrated(-1, 1, n, center=0.2, scale=0.4)
        x = grid.nodes if side == "down" else -grid.nodes
        edge = min(x)
        rate, p = 5.0, 0.7
        # E[exp(p * max(x-Y, edge))] for Y exponential.
        exact = (rate * np.exp(p * x) + p * np.exp(p * edge - rate * (x - edge))) / (rate + p)
        approx = ExponentialJumpKernel(grid, rate, side).apply(np.exp(p * x), PDEDirection.BACKWARD)
        errors.append(np.max(np.abs(approx - exact)))
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 3.75) & (ratios < 4.25))


def test_extreme_width_times_rate():
    grid = Grid1D(np.array([0.0, 1e-18, 1.0, 1e5]))
    kernel = ExponentialJumpKernel(grid, 1.0, exterior="zero")
    matrix = kernel.apply(np.eye(4), PDEDirection.BACKWARD)
    np.testing.assert_allclose(matrix[1, :2], [5e-19, 5e-19], atol=0, rtol=1e-14)
    np.testing.assert_allclose(matrix[-1, -2:], [1 / 99999, 1 - 1 / 99999], rtol=1e-14)
    np.testing.assert_array_equal(matrix[-1, :2], 0)


def test_invalid_kernel_inputs():
    grid = Grid1D.uniform(-1, 1, 7)
    for rate in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="rate"):
            ExponentialJumpKernel(grid, rate)
    with pytest.raises(ValueError, match="exterior"):
        ExponentialJumpKernel(grid, 1, exterior="wrap")
    k = ExponentialJumpKernel(grid, 1)
    with pytest.raises(ValueError, match="grid"):
        k.apply(np.zeros(5), PDEDirection.FORWARD)
    with pytest.raises(ValueError, match="axis"):
        k.apply(np.zeros(7), PDEDirection.BACKWARD, axis=1)
    with pytest.raises(TypeError, match="direction"):
        k.apply(np.zeros(7), "backward")
    with pytest.raises(ValueError, match="finite"):
        k.exterior_mass(np.full(7, np.nan))
