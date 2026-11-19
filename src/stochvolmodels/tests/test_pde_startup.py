"""Independent startup composition, semigroup and fixed-model qualification checks."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm, solve

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import LSVModel, LeverageSurface, FixedLogForwardJump
from stochvolmodels.pde_solvers import (
    BoundaryPair1D, BoundarySet2D, Grid1D, PDEDirection, TensorGrid2D,
)
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer

pytestmark = pytest.mark.repository_only


@pytest.fixture
def study(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root / "validation").is_dir():
        pytest.skip("startup qualification is repository-only")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_startup
    return pde_startup


@pytest.fixture
def case(study):
    grid = TensorGrid2D(Grid1D([-.5, -.2, -.07, 0., .09, .25, .55]),
                        Grid1D([.03, .13, .2, .31, .55]))
    model = LSVModel.from_params(LogSvParams(sigma0=.2, theta=.2, kappa1=2., kappa2=2.,
                                            beta=-.5, volvol=.8))
    chain = OptionChain(ttms=np.array([.04]), forwards=np.array([96.]), ids=None,
                        discfactors=np.array([.99]), strikes_ttms=(np.array([92., 96., 101.]),),
                        optiontypes_ttms=(np.array(["P", "C", "C"]),))
    return study.QualificationCase("btc_adverse", chain, model, grid, (4,))


def dense_generator(case, time):
    """Probe backward columns and form the quadrature adjoint independently of assembly."""
    pair = BoundaryPair1D.natural()
    op = SplitOperator2D.build(case.grid, case.model.continuous_coefficients(time, case.grid),
                               BoundarySet2D(pair, pair))
    basis = np.eye(np.prod(case.grid.shape))
    a = np.column_stack([op.apply_full(v.reshape(case.grid.shape), PDEDirection.BACKWARD)
                         .ravel() for v in basis])
    w = case.grid.quadrature_weights.ravel()
    return a.T * w[None, :] / w[:, None]


@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_startup_is_two_full_dense_half_steps_only_at_zero(study, case, monkeypatch,
                                                          positive, quadratic):
    from validation import pde_qualification as runner
    params = case.model.dynamics.params
    model = LSVModel.from_params(LogSvParams(
        sigma0=params.sigma0, theta=params.theta, kappa1=params.kappa1, kappa2=params.kappa2,
        beta=.5 if positive else -.5, volvol=params.volvol))
    model = replace(model, backbone_times=(.013, .04), backbone_scales=(.8, 1.2))
    case = replace(case, model=model)
    states = []
    original = runner.density_diagnostics

    def observe(grid, values):
        states.append(values.copy())
        return original(grid, values)

    monkeypatch.setattr(runner, "density_diagnostics", observe)
    result = runner._price_time_method(
        case, study.Variant("test", case.grid, (2,), quadratic), "cn", study.UnsplitConfig(),
        adi=study.SCHEMES["cs_half"], startup=True)
    dt = .013 / (4 if quadratic else 2) / 2
    matrix = np.eye(np.prod(case.grid.shape)) - dt * dense_generator(case, .013)
    expected = case.grid.project_delta(0., .2).ravel()
    for actual in states[:2]:
        expected = solve(matrix, expected)
        np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-13, atol=2e-13)
    steps = result["diagnostics"]["step_moments"]
    assert len(steps) == 5  # two sub-buckets, two nominal steps each, one replacement
    assert [s["integrator"] for s in steps] == ["be_startup"] * 2 + ["craig_sneyd"] * 3
    np.testing.assert_allclose([s["time_end"] for s in steps],
                               [dt, 2*dt, .013, .0265, .04], rtol=0, atol=1e-17)
    assert steps[0]["time_start"] == 0.
    assert all(a["time_end"] == b["time_start"] for a, b in zip(steps, steps[1:]))
    assert [s["coefficient_time"] for s in steps] == [.013]*3 + [.04]*2
    assert result["max_linear_residual"] < 1e-14


@pytest.mark.parametrize("scheme", ["douglas_1", "cs_half", "hv_half", "hv_stable"])
@pytest.mark.parametrize("quadratic", [False, True])
def test_undamped_path_matches_existing_pricer_across_knots(study, case, scheme, quadratic):
    chain = OptionChain(ttms=np.array([.019, .04]), forwards=np.array([96., 103.]), ids=None,
                        discfactors=np.array([.99, .98]),
                        strikes_ttms=(np.array([92., 96., 101.]), np.array([97., 103.])),
                        optiontypes_ttms=(np.array(["P", "C", "C"]), np.array(["P", "C"])))
    surface = LeverageSurface(np.array([.013, .04]), case.grid.x,
                              np.array([[1., .9, .95, 1., 1.05, 1.1, 1.15], [1.1]*7]))
    model = LSVModel.from_params(case.model.dynamics.params, surface)
    case = replace(case, chain=chain, model=model, steps=(3, 5))
    result = study.measure(case, scheme=scheme, factor=1, quadratic=quadratic, startup=False,
                           config=study.UnsplitConfig(), repeats=1)
    expected = LogSvPdePricer(grid=case.grid, steps_per_interval=case.steps,
                              quadratic_start=quadratic, adi=study.SCHEMES[scheme])
    expected = expected.price_chain_detailed(case.chain, case.model)
    assert result["status"] == "ok"
    for a, b in zip(result["prices"], expected.prices, strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(result["diagnostics"]["step_moments"], expected.steps, strict=True):
        for name in ("time_end", "mass", "negative_mass", "mean_x", "variance_x", "covariance_xy"):
            assert a[name] == pytest.approx(getattr(b, name), rel=2e-13, abs=2e-13)


@pytest.mark.parametrize("scheme,order", [("douglas_1", 1), ("cs_half", 2),
                                          ("hv_half", 2), ("hv_stable", 2)])
@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("beta", [-.5, .5])
def test_time_order_against_independent_dense_exponential(study, case, monkeypatch,
                                                         scheme, order, startup, beta):
    from validation import pde_qualification as runner
    params = LogSvParams(sigma0=.2, theta=.2, kappa1=2., kappa2=2., beta=beta, volvol=.8)
    case = replace(case, model=LSVModel.from_params(params))
    exact = expm(.04 * dense_generator(case, .04)) @ case.grid.project_delta(0., .2).ravel()
    states = []
    diagnostic = runner.density_diagnostics

    def observe(grid, values):
        states.append(values.copy())
        return diagnostic(grid, values)

    monkeypatch.setattr(runner, "density_diagnostics", observe)
    errors = []
    for count in (20, 40):
        runner._price_time_method(case, study.Variant("test", case.grid, (count,), False),
                                  "cn", study.UnsplitConfig(), adi=study.SCHEMES[scheme],
                                  startup=startup)
        errors.append(np.linalg.norm(states[-1].ravel() - exact, np.inf))
    assert 0.85 * 2**order < errors[0] / errors[1] < 1.15 * 2**order


@pytest.mark.parametrize("jump", [False, True])
def test_snapshot_roundtrip_retains_exact_model_and_rejects_corruption(study, case, jump):
    surface = LeverageSurface(np.array([.02, .04]), case.grid.x, np.ones((2, 7)))
    model = LSVModel.from_params(case.model.dynamics.params, surface,
                                jump=FixedLogForwardJump(.4, -.1) if jump else None)
    model = replace(model, backbone_times=(.015, .04), backbone_scales=(.9, 1.1))
    case = replace(case, model=model)
    source = study.input_snapshot(case)
    saved = {"name": case.name, "input": source, "model_sha256": study._hash(source["model"]),
             "policy": {"negative_mass_limit": case.negative_mass_limit}}
    actual = study.restore_case(json.loads(json.dumps(saved)))
    assert study.input_snapshot(actual) == source
    saved["model_sha256"] = "corrupted"
    with pytest.raises(ValueError, match="hash"):
        study.restore_case(saved)


def test_reference_failure_cannot_qualify_prices(study, case, monkeypatch):
    original = study.measure

    def fail(*args, **kwargs):
        if kwargs["scheme"] == "cn":
            return {"status": "error", "error": "deliberate reference failure"}
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "measure", fail)
    result = study.qualify_startup(case, repeats=1, factors=(1,), reference_factors=(2, 4))
    assert result["status"] == "error" and not result["reference_resolved"]
    assert not any(r["time_qualified"] or r["accepted"] for r in result["runs"])


def test_reference_gap_and_model_mutation_cannot_qualify(study, case, monkeypatch):
    first = study.qualify_startup(case, repeats=1, factors=(1,), reference_factors=(2, 4),
                                  reference_bp=0.)
    assert first["complete"] and not first["reference_resolved"]
    assert not any(r["accepted"] for r in first["runs"])
    original = study.input_snapshot
    calls = 0

    def changed(value):
        nonlocal calls
        snapshot = original(value)
        calls += 1
        if calls > 1:
            snapshot["model"]["params"]["beta"] += .01
        return snapshot

    monkeypatch.setattr(study, "input_snapshot", changed)
    result = study.qualify_startup(case, repeats=1, factors=(1,), reference_factors=(2, 4),
                                   accuracy_bp=100., reference_bp=100.)
    assert not result["fixed_inputs"] and result["status"] == "error"
    assert not any(r["accepted"] or r["time_qualified"] for r in result["runs"])


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"factors": (2, 1)},
                                     {"reference_factors": (4, 8)}, {"reference_bp": np.nan},
                                     {"accuracy_bp": -.1}, {"factors": (True,)}])
def test_invalid_study_policies(study, case, kwargs):
    with pytest.raises(ValueError):
        study.qualify_startup(case, **kwargs)


def test_unsupported_jump_and_size_are_visible(study, case):
    jump = replace(case, model=replace(case.model, jump=FixedLogForwardJump(.4, -.1)))
    for value, config in ((jump, study.UnsplitConfig()), (case, study.UnsplitConfig(max_nodes=9))):
        result = study.qualify_startup(value, config=config)
        assert result["status"] == "skipped" and not result["complete"]
        assert result["reason"] and not result["reference_resolved"]


def test_cli_restores_without_refitting_and_keeps_full_evidence(study, case, tmp_path, monkeypatch):
    def no_fit(*args, **kwargs):
        pytest.fail("snapshot replay must never recalibrate")

    monkeypatch.setattr(study, "load_case", no_fit)
    monkeypatch.setattr(study, "SCHEMES", {"douglas_1": study.SCHEMES["douglas_1"]})
    snapshot = study.input_snapshot(case)
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"cases": [{"name": case.name, "input": snapshot,
                      "model_sha256": study._hash(snapshot["model"]),
                      "policy": {"negative_mass_limit": case.negative_mass_limit}}]}))
    output = tmp_path / "result"
    args = ["--cases", case.name, "--frozen-report", str(frozen), "--output", str(output),
            "--repeats", "1", "--factors", "1", "--reference-factors", "2", "4"]
    assert study.main(args) == 0
    report = json.loads((output / "startup.json").read_text())
    saved = report["cases"][0]
    assert saved["input_sha256"] == saved["input_sha256_after"] == study._hash(snapshot)
    assert saved["runs"][0]["diagnostics"]["step_moments"]
    assert report["metadata"]["frozen_report_sha256"]
    assert "no spatial accuracy" in (output / "startup.md").read_text()
    with pytest.raises(SystemExit):
        study.main(args)
