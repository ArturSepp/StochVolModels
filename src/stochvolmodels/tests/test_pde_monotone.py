"""Independent moments, positivity, duality and convergence of the spatial prototype."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm, solve

from stochvolmodels.pde_solvers import (
    BoundaryPair1D, BoundarySet2D, Grid1D, PDE2DCoefficients, PDEDirection, TensorGrid2D,
)
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D

pytestmark = pytest.mark.repository_only


@pytest.fixture
def backend(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root / "validation").is_dir():
        pytest.skip("monotone prototype is repository-only")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_monotone
    return pde_monotone


def operator(rho, n=13, uneven=False, drift_x=0., drift_y=0.):
    x = np.linspace(-2., 2., n)
    y = np.linspace(-1.7, 1.7, n)
    if uneven:
        x = x + .04*x*x
        y = y + .025*y*y
    grid = TensorGrid2D(Grid1D(x), Grid1D(y))
    coeff = PDE2DCoefficients.constant(grid.shape, .5, drift_x, .32, drift_y, .8*rho, 0.)
    pair = BoundaryPair1D.natural()
    return SplitOperator2D.build(grid, coeff, BoundarySet2D(pair, pair))


@pytest.mark.parametrize("rho", [-.98, 0., .98])
@pytest.mark.parametrize("uneven", [False, True])
def test_positive_rates_independent_polynomial_moments_and_adjoint(backend, rho, uneven):
    op = operator(rho, uneven=uneven)
    ref = backend.Monotone2DReference.from_split(
        op, spatial=backend.MonotoneConfig(fit_log_forward=False))
    a, f = ref.backward.toarray(), ref.forward.toarray()
    coo = ref.backward.tocoo()
    assert np.min(coo.data[coo.row != coo.col], initial=0.) >= 0.
    np.testing.assert_allclose(a.sum(axis=1), 0., atol=3e-13)
    w = op.grid.quadrature_weights.ravel()
    np.testing.assert_allclose(f, a.T*w[None, :]/w[:, None], atol=3e-14, rtol=3e-14)
    x, y = op.grid.x_mesh.ravel(), op.grid.y_mesh.ravel()
    center = 6*13+6
    np.testing.assert_allclose((a @ x)[center], 0., atol=3e-13)
    np.testing.assert_allclose((a @ y)[center], 0., atol=3e-13)
    np.testing.assert_allclose((a @ (x*y))[center], .8*rho, atol=3e-13)
    # A bounded ray search may expose variance inflation. It must never hide it.
    for coordinates, expected, key in ((x, 1., "variance_x"), (y, .64, "variance_y")):
        observed = (a @ ((coordinates-coordinates[center])**2))[center]
        np.testing.assert_allclose(observed-expected,
                                   ref.moment_errors[key].ravel()[center], atol=3e-13)
        assert observed >= expected-3e-13


@pytest.mark.parametrize("rho", [-.98, .98])
def test_exact_semigroup_stays_positive_and_recovers_gaussian_covariance(backend, rho):
    op = operator(rho, n=17)
    ref = backend.Monotone2DReference.from_split(
        op, backend.UnsplitConfig(max_expm_nodes=400),
        spatial=backend.MonotoneConfig(fit_log_forward=False))
    p = op.grid.project_delta(0., 0.)
    t = .002
    actual = ref.step(p, PDEDirection.FORWARD, t, "expm").values
    assert actual.min() >= -1e-14
    assert op.grid.integrate(actual) == pytest.approx(1., abs=2e-14)
    assert op.grid.integrate(op.grid.x_mesh*op.grid.y_mesh*actual) == pytest.approx(
        .8*rho*t, abs=3e-7)
    assert op.grid.integrate(op.grid.x_mesh**2*actual) == pytest.approx(t, abs=5e-7)
    assert op.grid.integrate(op.grid.y_mesh**2*actual) == pytest.approx(.64*t, abs=5e-7)
    # Independent full dense matrix exponential; no time-integration approximation.
    expected = expm(t*ref.forward.toarray()) @ p.ravel()
    np.testing.assert_allclose(actual.ravel(), expected, atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("direction", list(PDEDirection))
def test_be_matches_dense_solve_and_positive_resolvent(backend, direction):
    ref = backend.Monotone2DReference.from_split(
        operator(-.8, n=7), spatial=backend.MonotoneConfig(fit_log_forward=False))
    a = ref.generator(direction).toarray()
    matrix = np.eye(a.shape[0]) - .17*a
    inverse = solve(matrix, np.eye(a.shape[0]))
    assert inverse.min() >= -2e-14
    initial = ref.grid.project_delta(.1, -.2)
    observed = ref.step(initial, direction, .17, "be")
    np.testing.assert_allclose(observed.values.ravel(), inverse @ initial.ravel(),
                               atol=3e-13, rtol=3e-13)
    assert observed.values.min() >= -1e-14
    assert observed.residual_relative < 1e-14
    with pytest.raises(ValueError, match="CN positivity"):
        ref.step(initial, direction, 100., "cn")


@pytest.mark.parametrize("rho", [-.98, .98])
def test_exponential_fit_preserves_forward_without_density_repair(backend, rho):
    op = operator(rho, uneven=True, drift_x=-.5, drift_y=.1)
    ref = backend.Monotone2DReference.from_split(op)
    payoff = np.exp(ref.grid.x_mesh)
    np.testing.assert_allclose(ref.backward @ payoff.ravel(), 0., atol=8e-13)
    p = ref.grid.project_delta(.1, -.2)
    before = ref.grid.integrate(p*payoff)
    for _ in range(8):
        p = ref.step(p, PDEDirection.FORWARD, .05).values
    assert p.min() >= -2e-14
    assert ref.grid.integrate(p) == pytest.approx(1., abs=2e-13)
    assert ref.grid.integrate(p*payoff) == pytest.approx(before, abs=2e-13)
    # Off-node delta projection preserves x, not exp(x); propagation must preserve its input.
    assert before != pytest.approx(np.exp(.1), abs=1e-6)


def test_boundary_closure_and_inflation_are_explicit(backend):
    op = operator(-.999, uneven=True)
    ref = backend.Monotone2DReference.from_split(
        op, spatial=backend.MonotoneConfig(radii=(1,), fit_log_forward=False))
    assert ref.statistics["diffusion_inflation_nodes"] > 0
    with pytest.raises(ValueError, match="inflation"):
        backend.Monotone2DReference.from_split(
            op, spatial=backend.MonotoneConfig(radii=(1,), fit_log_forward=False,
                                               reject_inflation=True))
    x, y = op.grid.x_mesh.ravel(), op.grid.y_mesh.ravel()
    corner = 0
    assert ref.backward.getrow(corner).nnz == 0
    # Tangential variance retained at x edge; cross covariance removed explicitly.
    edge = 6
    np.testing.assert_allclose((ref.backward @ ((x-x[edge])*(y-y[edge])))[edge], 0., atol=1e-13)
    assert ref.moment_errors["covariance_xy"].ravel()[edge] == pytest.approx(.8*.999)
    assert ref.statistics["boundary_closure"] == "absorbing normal, evolving tangent"


@pytest.mark.parametrize("fitted", [False, True])
def test_smooth_generator_refines_with_both_covariance_signs(backend, fitted):
    for rho in (-.8, .8):
        errors = []
        for n in (17, 33, 65):
            bx = -.5 if fitted else 0.
            op = operator(rho, n=n, uneven=True, drift_x=bx, drift_y=.1)
            ref = backend.Monotone2DReference.from_split(op, spatial=backend.MonotoneConfig(
                fit_log_forward=fitted))
            x, y = op.grid.x_mesh, op.grid.y_mesh
            u = np.exp(.2*x+.3*y)
            target = (.5*.2**2 + .32*.3**2 + .8*rho*.2*.3 + bx*.2 + .1*.3)*u
            observed = (ref.backward @ u.ravel()).reshape(op.grid.shape)
            region = (abs(x) < .4) & (abs(y) < .4)
            errors.append(np.max(abs(observed[region]-target[region])))
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < errors[0]/5


@pytest.mark.parametrize("kwargs", [{"radii": ()}, {"radii": (2, 1)}, {"radii": (64,)},
                                     {"radii": (True,)}, {"fit_log_forward": "yes"}])
def test_invalid_settings_are_rejected(backend, kwargs):
    with pytest.raises((ValueError, TypeError)):
        backend.MonotoneConfig(**kwargs)


def test_scope_and_size_rejected_before_assembly(backend):
    op = operator(.5)
    with pytest.raises(ValueError, match="max_nodes"):
        backend.Monotone2DReference.from_split(op, backend.UnsplitConfig(max_nodes=9))
    with pytest.raises(ValueError, match="drift_x"):
        backend.Monotone2DReference.from_split(op)
    discount = replace(op.coefficients, discount=-np.ones(op.grid.shape))
    op = SplitOperator2D.build(op.grid, discount, op.boundaries)
    with pytest.raises(ValueError, match="discount"):
        backend.Monotone2DReference.from_split(op)


def test_black_vanillas_converge_under_positive_spatial_refinement(backend):
    from scipy.special import ndtr
    from stochvolmodels.data.option_chain import OptionChain
    from stochvolmodels.models.lsv import LSVModel
    from stochvolmodels.models.logsv.params import LogSvParams
    from validation.pde_qualification import QualificationCase, Variant, _price_time_method

    chain = OptionChain(ttms=np.array([.2]), forwards=np.array([93.]),
                        discfactors=np.array([.98]), ids=None,
                        strikes_ttms=(np.array([87., 93., 99.]),),
                        optiontypes_ttms=(np.array(["C", "C", "C"]),))
    model = LSVModel.from_params(LogSvParams(sigma0=.2, theta=.2, beta=0., volvol=0.))
    k = chain.strikes_ttms[0]
    sd = .2*np.sqrt(.2)
    d1 = np.log(93./k)/sd + .5*sd
    exact = .98*(93.*ndtr(d1) - k*ndtr(d1-sd))
    errors = []
    for n in (31, 61, 121):
        grid = TensorGrid2D(Grid1D.uniform(-.8, .8, n), Grid1D([.1, .2, .3]))
        case = QualificationCase("black", chain, model, grid, (1600,))
        result = _price_time_method(case, Variant("test", grid, case.steps, False), "be",
                                    backend.UnsplitConfig(),
                                    operator_factory=backend.Monotone2DReference.from_split)
        assert result["diagnostics"]["peak_negative_mass"] <= 1e-14
        assert result["diagnostics"]["max_forward_error"] < 2e-12
        errors.append(np.max(abs(result["prices"][0]-exact)))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < errors[0]/6


def test_runner_preserves_frozen_inputs_and_exposes_missing_evidence(
    backend, tmp_path, monkeypatch
):
    import json
    from validation import pde_positivity as runner
    from validation.pde_startup import one_week_case

    case = one_week_case()
    case = replace(case, grid=TensorGrid2D(Grid1D.uniform(-.25, .25, 13),
                                          Grid1D.uniform(.05, .65, 9)), steps=(8,))
    coarser = runner.coarsen(case)
    assert coarser.x.bounds == case.grid.x.bounds and coarser.y.bounds == case.grid.y.bounds
    for axis, old, point in ((coarser.x, case.grid.x, 0.), (coarser.y, case.grid.y, .2)):
        a, b, _, _ = old.locate(point)
        assert old.nodes[a] in axis.nodes and old.nodes[b] in axis.nodes
    before = runner.input_snapshot(case)
    result = runner.investigate(case, variants=("base", "space_coarse", "time_2"),
                                step_factor=1, repeats=1)
    assert result["complete"] and result["fixed_inputs"]
    assert not result["production_qualified"]
    assert result["comparisons"]["positive_domain_last"] is None
    assert runner.input_snapshot(case) == before
    assert any("spatial" in s for r in result["runs"] if r["backend"] == "positive"
               for s in r["diagnostics"]["step_moments"])
    runner.write_report(tmp_path, {"cases": [result], "metadata": {}})
    saved = json.loads((tmp_path/"positivity.json").read_text())
    assert saved["cases"][0]["input_sha256"] == result["input_sha256_after"]
    original = runner.measure

    def missing(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[2] == "centered":
            value.update(status="error", error="deliberate missing spatial reference")
        return value

    monkeypatch.setattr(runner, "measure", missing)
    failed = runner.investigate(case, variants=("base",), step_factor=1)
    assert not failed["complete"] and not failed["production_qualified"]
    assert failed["status"] == "error"


def test_factory_cannot_silently_leave_old_adi_dynamics(backend):
    from validation.pde_qualification import Variant, _price_time_method
    from validation.pde_startup import one_week_case, SCHEMES

    case = one_week_case()
    with pytest.raises(ValueError, match="cannot be combined"):
        _price_time_method(case, Variant("base", case.grid, case.steps), "be",
                           backend.UnsplitConfig(), adi=SCHEMES["douglas_1"],
                           operator_factory=backend.Monotone2DReference.from_split)
