"""Independent dense and semigroup checks for the repository-only unsplit reference."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm, solve

from stochvolmodels.pde_solvers import (
    BoundaryCondition1D, BoundaryPair1D, BoundarySet2D, Grid1D,
    PDE2DCoefficients, PDEDirection, TensorGrid2D,
)
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D

pytestmark = pytest.mark.repository_only


@pytest.fixture
def reference(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root / "validation").is_dir():
        pytest.skip("unsplit reference is repository-only")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_unsplit
    return pde_unsplit


def _operator(covariance=.09):
    grid = TensorGrid2D(Grid1D(np.array([-1., -.6, -.1, .2, .9, 1.4])),
                        Grid1D(np.array([-.8, -.35, .05, .3, 1.1])))
    x, y = grid.x_mesh, grid.y_mesh
    coeff = PDE2DCoefficients(.16 + .03*x**2, .1 + .06*y,
                              .12 + .01*y**2, -.08 + .02*x,
                              covariance * np.tanh(x + 2*y), .02 + .01*(x**2 + y**2))
    boundary = BoundaryPair1D.natural()
    return SplitOperator2D.build(grid, coeff, BoundarySet2D(boundary, boundary))


def _dense_action(operator, direction):
    """Probe the existing matrix-free action, independently of sparse assembly."""
    n = int(np.prod(operator.grid.shape))
    basis = np.eye(n)
    return np.column_stack([operator.apply_full(v.reshape(operator.grid.shape), direction).ravel()
                            for v in basis])


@pytest.mark.parametrize("covariance", [-.09, 0., .09])
def test_sparse_assembly_matches_all_dense_columns_and_boundary_rows(reference, covariance):
    operator = _operator(covariance)
    full = reference.Unsplit2DReference.from_split(operator)
    for direction in PDEDirection:
        np.testing.assert_allclose(full.generator(direction).toarray(),
                                   _dense_action(operator, direction), rtol=2e-14, atol=2e-14)
    w = operator.grid.quadrature_weights.ravel()
    expected = full.backward.toarray().T * w[None, :] / w[:, None]
    np.testing.assert_allclose(full.forward.toarray(), expected, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(full.backward @ np.ones(w.size),
                               -operator.coefficients.discount.ravel(), atol=1e-14)
    assert full.statistics["constant_discount_residual"] < 1e-14
    # Boundary transfers are real entries, not discarded by interior-only assembly.
    assert np.count_nonzero(full.forward.toarray()[0]) > 1


@pytest.mark.parametrize("method,theta", [("be", 1.), ("cn", .5)])
@pytest.mark.parametrize("direction", list(PDEDirection))
def test_sparse_step_matches_dense_solve_with_variable_discount(
    reference, method, theta, direction
):
    operator = _operator()
    full = reference.Unsplit2DReference.from_split(operator)
    a = _dense_action(operator, direction)
    initial = np.asfortranarray(np.random.default_rng(56).normal(size=operator.grid.shape))
    initial.setflags(write=False)
    dt = .037
    matrix = np.eye(a.shape[0]) - theta * dt * a
    rhs = (np.eye(a.shape[0]) + (1-theta) * dt * a) @ initial.ravel()
    expected = solve(matrix, rhs)
    actual = full.step(initial, direction, dt, method)
    np.testing.assert_allclose(actual.values.ravel(), expected, rtol=3e-14, atol=3e-14)
    assert actual.residual_relative < 1e-15
    np.testing.assert_allclose(matrix @ actual.values.ravel(), rhs, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("method", ["be", "cn", "expm"])
def test_complete_time_varying_map_retains_weighted_adjoint(reference, method):
    operator = _operator()
    second = SplitOperator2D.build(operator.grid, replace(
        operator.coefficients, drift_x=operator.coefficients.drift_x + .15), operator.boundaries)
    first = reference.Unsplit2DReference.from_split(operator)
    last = reference.Unsplit2DReference.from_split(second)
    grid = operator.grid
    density = grid.project_delta(.1, .1)
    payoff = np.maximum(np.exp(grid.x_mesh) - 1., 0.) * (1 + .1*grid.y_mesh)
    forward = last.step(first.step(density, PDEDirection.FORWARD, .04, method).values,
                        PDEDirection.FORWARD, .07, method).values
    backward = first.step(last.step(payoff, PDEDirection.BACKWARD, .07, method).values,
                          PDEDirection.BACKWARD, .04, method).values
    np.testing.assert_allclose(grid.integrate(forward * payoff),
                               grid.integrate(density * backward), rtol=2e-13, atol=2e-14)
    # Variable discount is retained; a raw Euclidean transpose would fail on these weights.
    assert grid.integrate(forward) < 1.


@pytest.mark.parametrize("rho", [0., -.98, .98])
def test_time_refinement_against_dense_exponential_retains_spatial_negativity(reference, rho,
                                                                            record_property):
    grid = TensorGrid2D(Grid1D.uniform(-1., 1., 13), Grid1D.uniform(-1., 1., 13))
    boundary = BoundaryPair1D.natural()
    split = SplitOperator2D.build(
        grid, PDE2DCoefficients.constant(grid.shape, .5, 0., .5, 0., rho, 0.),
        BoundarySet2D(boundary, boundary))
    full = reference.Unsplit2DReference.from_split(split)
    initial = grid.project_delta(0., 0.)
    duration = .002
    a = _dense_action(split, PDEDirection.FORWARD)
    exact = (expm(duration*a) @ initial.ravel()).reshape(grid.shape)
    sparse_exact = full.step(initial, PDEDirection.FORWARD, duration, "expm")
    np.testing.assert_allclose(sparse_exact.values, exact, atol=5e-13, rtol=3e-13)
    assert sparse_exact.residual_relative is None
    diag = reference.density_diagnostics(grid, exact)
    assert abs(diag["mass"] - 1.) < 1e-14
    assert full.statistics["negative_off_diagonals"] == (242 if rho else 0)
    if rho:
        assert .03 < diag["negative_mass"] < .031
    else:
        assert diag["negative_mass"] < 1e-15
    payoff = np.maximum(np.exp(grid.x_mesh)-1., 0.)
    evidence = {"rho": rho, "operator": full.statistics, "exponential": diag, "methods": {}}
    for method in ("be", "cn"):
        observations = []
        for count in (4, 8, 16, 32, 64):
            state = initial
            peak = residual = 0.
            for _ in range(count):
                step = full.step(state, PDEDirection.FORWARD, duration/count, method)
                state = step.values
                peak = max(peak, -grid.integrate(np.minimum(state, 0.)))
                residual = max(residual, step.residual_relative)
            l1 = grid.integrate(np.abs(state-exact))
            observations.append({"steps": count, "weighted_l1_error": l1,
                                 "price_error": abs(grid.integrate((state-exact)*payoff)),
                                 "peak_negative_mass": peak, "max_residual": residual,
                                 **reference.density_diagnostics(grid, state)})
            assert abs(grid.integrate(state)-1.) < 2e-14
            assert residual < 1e-14
        errors = [o["weighted_l1_error"] for o in observations]
        assert all(a > b for a, b in zip(errors[:-1], errors[1:]))
        order_ratio = errors[-2] / errors[-1]
        assert (1.9 < order_ratio < 2.1) if method == "be" else (3.8 < order_ratio < 4.2)
        if rho:
            assert observations[-1]["negative_mass"] > .03
        evidence["methods"][method] = observations
    record_property("unsplit_semigroup_evidence", json.dumps(evidence))


def test_limits_and_unsupported_boundaries_fail_before_expensive_work(reference, monkeypatch):
    operator = _operator(0.)
    with monkeypatch.context() as mp:
        mp.setattr(reference.sparse, "kron",
                   lambda *a, **kw: pytest.fail("assembled oversized matrix"))
        with pytest.raises(ValueError, match="max_nodes"):
            reference.Unsplit2DReference.from_split(operator, reference.UnsplitConfig(max_nodes=9))
    full = reference.Unsplit2DReference.from_split(
        operator, reference.UnsplitConfig(max_expm_nodes=9))
    monkeypatch.setattr(reference, "expm_multiply",
                        lambda *a, **kw: pytest.fail("oversized exponential"))
    with pytest.raises(ValueError, match="max_expm_nodes"):
        full.step(np.ones(operator.grid.shape), PDEDirection.FORWARD, .01, "expm")
    natural = BoundaryCondition1D.natural()
    boundaries = BoundarySet2D(BoundaryPair1D(BoundaryCondition1D.dirichlet(1.), natural),
                               BoundaryPair1D.natural())
    with pytest.raises(ValueError, match="natural boundaries"):
        reference.Unsplit2DReference.from_split(replace(operator, boundaries=boundaries))


def test_bad_linear_solution_is_rejected_by_residual(reference):
    operator = _operator()
    full = reference.Unsplit2DReference.from_split(operator)
    values = np.ones(operator.grid.shape)
    full.step(values, PDEDirection.FORWARD, .01)
    matrix, norm, _ = full._factors[(PDEDirection.FORWARD, .01)]
    full._factors[(PDEDirection.FORWARD, .01)] = (
        matrix, norm, SimpleNamespace(solve=lambda rhs: np.zeros_like(rhs)))
    with pytest.raises(RuntimeError, match="linear residual"):
        full.step(values, PDEDirection.FORWARD, .01)


@pytest.mark.parametrize("changes", [{"max_nodes": 8}, {"max_nodes": True},
    {"max_expm_nodes": 1.5}, {"residual_tolerance": 0.}, {"residual_tolerance": np.nan},
    {"methods": ()}, {"methods": ("be", "be")}, {"methods": ("adi",)}])
def test_invalid_reference_policies(reference, changes):
    with pytest.raises(ValueError):
        reference.UnsplitConfig(**changes)


def test_raw_negative_boundary_mass_is_observed_without_repair(reference):
    grid = _operator().grid
    state = grid.project_delta(.1, .1).copy()
    state[0, 0] = -.02 / grid.quadrature_weights[0, 0]
    state[2, 2] += .02 / grid.quadrature_weights[2, 2]
    before = state.copy()
    observed = reference.density_diagnostics(grid, state)
    assert observed["mass"] == pytest.approx(1.)
    assert observed["negative_mass"] == pytest.approx(.02)
    assert observed["boundary_mass"]["absolute_total"] == pytest.approx(.02)
    np.testing.assert_array_equal(state, before)
