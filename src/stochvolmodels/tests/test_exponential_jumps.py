"""Independent cell-quadrature, tensor, and grid-refinement controls for jump marks."""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm

from stochvolmodels.pde_solvers import (
    ADIConfig,
    ADIScheme,
    BoundaryPair1D,
    BoundarySet2D,
    DensityPolicy,
    ExponentialJumpKernel,
    ExponentialJumpProcess,
    Grid1D,
    PDE1DCoefficients,
    PDE1DSolver,
    PDE2DADISolver,
    PDE2DCoefficients,
    PDEDirection,
    StaticCoefficientProvider1D,
    StaticCoefficientProvider2D,
    TensorGrid2D,
    TimeStepConfig,
    apply_common_exponential_jump,
    strang_split_step,
)
from stochvolmodels.pde_solvers.operators_1d import Operator1D
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D


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


def compound_case(dim, rates):
    """Build independent backward jump matrices by integrating the hat functions."""
    x = Grid1D(np.array([-1.0, -0.42, -0.1, 0.24, 0.55, 1.1]))
    kx = ExponentialJumpKernel(x, 2.4, "down", "zero")
    k = quadrature_matrix(x, kx.rate, kx.side, kx.exterior)
    lx, ly, common = rates
    if dim == 1:
        process = ExponentialJumpProcess(x, x_kernel=kx, x_intensity=lx)
        return x, process, lx * (k - np.eye(x.size))
    y = Grid1D(np.array([-0.8, -0.16, 0.37, 0.95]))
    grid = TensorGrid2D(x, y)
    ky = ExponentialJumpKernel(y, 1.8, "up", "zero")
    h = quadrature_matrix(y, ky.rate, ky.side, ky.exterior)
    process = ExponentialJumpProcess(
        grid, x_kernel=kx, y_kernel=ky, x_intensity=lx, y_intensity=ly, common_intensity=common
    )
    ident = np.eye(x.size * y.size)
    generator = (
        lx * (np.kron(k, np.eye(y.size)) - ident)
        + ly * (np.kron(np.eye(x.size), h) - ident)
        + common * (np.kron(k, h) - ident)
    )
    return grid, process, generator


@pytest.mark.parametrize(
    "dim, rates",
    [
        (1, (1.4, 0, 0)),
        (2, (0, 0, 0)),
        (2, (0, 1.7, 0)),
        (2, (1.4, 0, 0)),
        (2, (0, 0, 0.8)),
        (2, (1.4, 1.7, 0.8)),
    ],
)
def test_compound_poisson_dense_and_mass(dim, rates):
    grid, process, generator = compound_case(dim, rates)
    w = grid.quadrature_weights
    rng = np.random.default_rng(1709)
    density, payoff = rng.random(w.shape), rng.normal(size=w.shape)
    dt = 0.83
    dense = expm(dt * generator)
    back = process.poisson_step(payoff, PDEDirection.BACKWARD, dt)
    fwd = process.poisson_step(density, PDEDirection.FORWARD, dt)
    np.testing.assert_allclose(back.values.ravel(), dense @ payoff.ravel(), rtol=0, atol=8e-14)
    np.testing.assert_allclose(
        (w * fwd.values).ravel(), dense.T @ (w * density).ravel(), rtol=0, atol=8e-14
    )
    assert back.killed_mass is None
    assert fwd.killed_mass == pytest.approx(grid.integrate(density - fwd.values), abs=3e-14)
    assert grid.integrate(payoff * fwd.values) == pytest.approx(
        grid.integrate(density * back.values), abs=3e-14
    )
    assert fwd.truncated_probability <= 1e-14
    split = strang_split_step(payoff, PDEDirection.BACKWARD, dt, process, lambda v, *_: v)
    np.testing.assert_allclose(split.ravel(), dense @ payoff.ravel(), rtol=0, atol=1e-13)


@pytest.mark.parametrize("dim,scheme", [(1, None), *((2, scheme) for scheme in ADIScheme)])
def test_compound_diffusion_refinement_and_duality(dim, scheme):
    grid, process, generator = compound_case(dim, (0.9, 0.32, 1.1))
    w = grid.quadrature_weights
    policy = DensityPolicy(fail_on_violation=True)
    natural = BoundaryPair1D.natural()
    if dim == 1:
        solver = PDE1DSolver(TimeStepConfig(theta=0.5), policy)
        coeff = PDE1DCoefficients(
            np.full(grid.size, 0.048), 0.015 * grid.nodes, 0.03 + 0.02 * grid.nodes
        )
        boundaries = natural
        provider = StaticCoefficientProvider1D(coeff)
        operator = Operator1D.build(grid, coeff, boundaries)
        diffuse = operator.backward.apply
    else:
        solver = PDE2DADISolver(ADIConfig(scheme=scheme), policy)
        coeff = PDE2DCoefficients.constant(grid.shape, 0.048, 0.019, 0.07, -0.025, 0.035, 0.03)
        boundaries = BoundarySet2D(natural, natural)
        provider = StaticCoefficientProvider2D(coeff)
        operator = SplitOperator2D.build(grid, coeff, boundaries)

        def diffuse(v):
            return operator.apply_full(v, PDEDirection.BACKWARD)

    local = np.column_stack([diffuse(v.reshape(w.shape)).ravel() for v in np.eye(w.size)])
    maturity = 0.5
    exact = expm(maturity * (generator + local))
    payoff = np.cos(2.8 * grid.nodes) if dim == 1 else np.cos(2.8 * grid.x_mesh + 1.1 * grid.y_mesh)
    density = np.linspace(0.8, 1.4, w.size).reshape(w.shape)
    errors = []
    for n in (4, 8, 16):
        times = maturity * np.linspace(0, 1, n + 1) ** 1.2
        kwargs = dict(jump=process, save_times=(times[n // 2],))
        back = solver.solve(
            grid, provider, boundaries, PDEDirection.BACKWARD, times, payoff, **kwargs
        )
        fwd = solver.solve(
            grid, provider, boundaries, PDEDirection.FORWARD, times, density, **kwargs
        )
        errors.append(np.max(abs(back.values.ravel() - exact @ payoff.ravel())))
        assert grid.integrate(density * back.values) == pytest.approx(
            grid.integrate(payoff * fwd.values), rel=0, abs=8e-13
        )
        assert grid.integrate(density * back.values) == pytest.approx(
            grid.integrate(back.saved_states[0].values * fwd.saved_states[0].values),
            rel=0,
            abs=8e-13,
        )
        assert all(d.jump_killed_mass > 0 and abs(d.mass_error) < 2e-13 for d in fwd.diagnostics)
        assert all(d.jump_truncated_probability <= 2e-14 for d in fwd.diagnostics)
    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all(ratios > (1.7 if scheme is ADIScheme.DOUGLAS else 3.4)), (scheme, errors)


def test_compound_killing_is_separate_from_poisson_truncation():
    grid, process, _ = compound_case(1, (2, 0, 0))
    density = grid.project_delta(grid.nodes[0])
    # Every nonzero count kills this initial state. Retain counts 0, 1 and 2.
    result = process.poisson_step(density, PDEDirection.FORWARD, 0.5, tail_tolerance=0.09)
    assert result.poisson_terms == 3
    assert result.truncated_probability == pytest.approx(1 - 2.5 * np.exp(-1))
    assert result.killed_mass == pytest.approx(0.6)
    assert grid.integrate(result.values) == pytest.approx(0.4)
    with pytest.raises(RuntimeError, match="max_terms"):
        process.poisson_step(density, PDEDirection.FORWARD, 0.5, max_terms=2)
    unchanged = process.poisson_step(density, PDEDirection.FORWARD, 0)
    np.testing.assert_array_equal(unchanged.values, density)
    assert unchanged.killed_mass == unchanged.truncated_probability == 0
    x = Grid1D.uniform(0, 1, 5)
    tensor = TensorGrid2D(x, grid)
    only_y = ExponentialJumpProcess(tensor, y_kernel=ExponentialJumpKernel(grid, 3), y_intensity=2)
    fwd = only_y.poisson_step(np.ones(tensor.shape), PDEDirection.FORWARD, 0.5)
    assert fwd.killed_mass == 0
    assert tensor.integrate(fwd.values) == pytest.approx(tensor.integrate(np.ones(tensor.shape)))
    with pytest.raises(ValueError, match="matching grid"):
        process.validate_grid(Grid1D.uniform(-2, 2, grid.size))
    with pytest.raises(ValueError, match="kernel"):
        ExponentialJumpProcess(tensor, common_intensity=1)
    with pytest.raises(ValueError, match="tensor"):
        ExponentialJumpProcess(x, y_intensity=1)
    with pytest.raises(ValueError, match="nonnegative"):
        ExponentialJumpProcess(x, x_intensity=-0.1)


@pytest.mark.parametrize("dim", [1, 2])
def test_solver_keeps_physical_loss_with_renormalization(dim):
    grid, process, generator = compound_case(dim, (1.5, 0.2, 0.8))
    policy = DensityPolicy(renormalize=True, fail_on_violation=True)
    natural = BoundaryPair1D.natural()
    if dim == 1:
        solver = PDE1DSolver(density_policy=policy)
        provider = StaticCoefficientProvider1D(PDE1DCoefficients.constant(grid.size, 0, 0, 0))
        boundaries = natural
    else:
        solver = PDE2DADISolver(density_policy=policy)
        provider = StaticCoefficientProvider2D(
            PDE2DCoefficients.constant(grid.shape, 0, 0, 0, 0, 0, 0)
        )
        boundaries = BoundarySet2D(natural, natural)
    w = grid.quadrature_weights
    density = np.ones(w.shape) / w.sum()
    ans = solver.solve(
        grid, provider, boundaries, PDEDirection.FORWARD, [0, 0.31, 0.6], density, jump=process
    )
    want = expm(0.6 * generator).T @ (w * density).ravel()
    np.testing.assert_allclose((w * ans.values).ravel(), want, rtol=0, atol=2e-14)
    assert sum(d.jump_killed_mass for d in ans.diagnostics) == pytest.approx(
        1 - want.sum(), abs=2e-14
    )
    if dim == 2:
        with pytest.raises(ValueError, match="only one"):
            solver.solve(
                grid,
                provider,
                boundaries,
                PDEDirection.FORWARD,
                [0, 0.6],
                density,
                jump=process,
                forward_x_jump=object(),
            )
