"""Contracts for the repository-only PDE qualification experiment."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

pytestmark = pytest.mark.repository_only


@pytest.fixture
def runner(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root / "validation").is_dir():
        pytest.skip("qualification runner is repository-only")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_qualification
    return pde_qualification


@pytest.fixture
def black_case(runner):
    chain = OptionChain(
        ttms=np.array([.1, .3]), forwards=np.array([91., 123.]),
        discfactors=np.array([.99, .96]), ids=None,
        strikes_ttms=(np.array([87., 91., 96.]), np.array([120., 123.])),
        optiontypes_ttms=(np.array(["P", "C", "C"]), np.array(["C", "P"])),
    )
    prices = []
    for t, f, d, k, kinds in zip(chain.ttms, chain.forwards, chain.discfactors,
                                chain.strikes_ttms, chain.optiontypes_ttms):
        sd = .2 * np.sqrt(t)
        d1 = np.log(f / k) / sd + sd / 2
        calls = f * ndtr(d1) - k * ndtr(d1 - sd)
        prices.append(d * np.where(kinds == "C", calls, calls - f + k))
    return runner.QualificationCase(
        name="black_control", chain=chain,
        model=LSVModel.from_params(LogSvParams(sigma0=.2, theta=.2, beta=0., volvol=0.)),
        grid=TensorGrid2D(Grid1D.uniform(-.6, .6, 81), Grid1D.uniform(.1, .3, 5)),
        steps=(16, 24), benchmark_prices=tuple(prices),
        benchmark_errors=tuple(np.zeros_like(p) for p in prices),
        benchmark_description="exact Black", negative_mass_limit=1e-8,
    )


def test_refinement_axes_preserve_the_fixed_problem(runner, black_case):
    variants = {v.name: v for v in runner.make_variants(black_case)}
    base = variants["base"]
    for name in ("domain_2", "domain_4"):
        v = variants[name]
        assert v.steps == base.steps and v.quadratic_start
        for old, new in ((base.grid.x, v.grid.x), (base.grid.y, v.grid.y)):
            assert np.isin(old.nodes, new.nodes).all()
        assert v.grid.x.bounds[0] < base.grid.x.bounds[0]
        assert v.grid.y.bounds[0] == 0.0
        assert v.grid.y.spacings.min() > 1e-6  # No duplicate endpoint from ceil roundoff.
    for factor in (2, 3):
        v = variants[f"space_{factor}"]
        assert v.steps == base.steps
        for old, new in ((base.grid.x, v.grid.x), (base.grid.y, v.grid.y)):
            assert old.bounds == new.bounds
            assert new.size == factor * (old.size - 1) + 1
            np.testing.assert_array_equal(new.nodes[::factor], old.nodes)
    for factor in (2, 4):
        v = variants[f"time_{factor}"]
        assert v.grid is base.grid and v.steps == tuple(factor * n for n in base.steps)
    assert not variants["archive"].quadratic_start


def test_quote_errors_use_each_expiry_discounted_forward_and_keep_invalids(runner, black_case):
    ref = black_case.benchmark_prices
    scales = black_case.chain.discfactors * black_case.chain.forwards
    actual = tuple(p + s * 1e-4 * np.arange(1, p.size + 1)
                   for p, s in zip(ref, scales))
    error = runner.price_errors(black_case.chain, actual, ref)
    np.testing.assert_allclose(error["signed_bp"], [1., 2., 3., 1., 2.], atol=1e-12)
    assert error["max_abs_bp"] == pytest.approx(3.)
    assert error["rms_bp"] == pytest.approx(np.sqrt(19 / 5))
    bad = (np.array([np.nan, 1., 2.]), ref[1])
    assert runner.price_errors(black_case.chain, bad, ref)["max_abs_bp"] is None
    with pytest.raises(ValueError, match="shape"):
        runner.price_errors(black_case.chain, (ref[0][:-1], ref[1]), ref)


def test_black_refinement_and_serialized_evidence(runner, black_case, tmp_path):
    result = runner.qualify_case(black_case, repeats=1)
    runs = {r["name"]: r for r in result["runs"]}
    fine = runs["reference_check"]
    base_error = runs["base"]["benchmark_error"]["max_abs_bp"]
    fine_error = fine["benchmark_error"]["max_abs_bp"]
    assert fine_error < base_error / 3 and fine_error < 1.
    assert fine["invalid_iv_count"] == 0
    assert fine["diagnostics"]["max_mass_error"] < 1e-11
    assert len(fine["timings_seconds"]) == 1
    assert result["model_sha256"] == result["model_sha256_after"]
    runner.write_report(tmp_path, {"cases": [result], "metadata": {"test": True}})
    import json
    saved = json.loads((tmp_path / "qualification.json").read_text())
    assert saved["cases"][0]["runs"][-1]["benchmark_error"] == fine["benchmark_error"]
    assert "not an error bound" in (tmp_path / "qualification.md").read_text()


def test_failed_solve_is_visible_and_cannot_qualify(runner, black_case, monkeypatch):
    original = runner.LogSvPdePricer.price_chain_detailed

    def fail_refined(pricer, *args, **kwargs):
        if pricer.grid.x.size > 100:
            raise ValueError("deliberate refined-grid failure")
        return original(pricer, *args, **kwargs)

    monkeypatch.setattr(runner.LogSvPdePricer, "price_chain_detailed", fail_refined)
    result = runner.qualify_case(black_case, repeats=1)
    assert result["status"] == "error"
    assert not result["qualified"]
    assert any("deliberate" in r.get("error", "") for r in result["runs"])


@pytest.mark.parametrize("repeats,tolerance,boundary", [(0, 1., .01), (1, -1., .01),
                                                       (1, np.nan, .01), (1, 1., -1.)])
def test_invalid_policies_rejected_before_solving(runner, black_case, repeats, tolerance, boundary):
    with pytest.raises(ValueError):
        runner.qualify_case(black_case, repeats=repeats, accuracy_bp=tolerance,
                            boundary_limit=boundary)


def test_frozen_stress_replay_and_density_rejection(runner):
    case = runner.load_case("extreme_negative_correlation")
    variants = runner.make_variants(case)
    archive = runner.measure_variant(case, variants[0], repeats=1)
    assert archive["archive_error"]["max_abs_bp"] < 1e-7
    assert archive["diagnostics"]["max_expiry_negative_mass"] > .04
    assert case.archive_expected_failures == ("pde_negative_mass",)
    assert runner.input_snapshot(case)["model"]["params"]["beta"] == -.58
    limited = replace(case, negative_mass_limit=0.)
    assert not runner.density_checks(archive, limited, .01)["negative_mass"]


@pytest.mark.parametrize("status,require,exit_code", [("complete", False, 0),
                                                     ("complete", True, 2), ("error", False, 1)])
def test_cli_checkpoint_exit_codes_and_no_overwrite(runner, monkeypatch, tmp_path,
                                                   status, require, exit_code):
    import json

    monkeypatch.setattr(runner, "qualify_case", lambda case, **kwargs: {
        "name": case.name, "status": status, "qualified": False,
        "checks": {"independent_benchmark": False}, "runs": [],
    })
    args = ["--cases", "high_vol_of_vol", "--output", str(tmp_path)]
    if require:
        args.append("--require-qualified")
    assert runner.main(args) == exit_code
    saved = json.loads((tmp_path / "qualification.json").read_text())
    assert saved["cases"][0]["status"] == status
    with pytest.raises(SystemExit) as exc:
        runner.main(args)
    assert exc.value.code == 2
    assert json.loads((tmp_path / "qualification.json").read_text()) == saved


@pytest.fixture
def tiny_case(black_case):
    return replace(black_case,
                   grid=TensorGrid2D(Grid1D.uniform(-.6, .6, 13), Grid1D.uniform(.1, .3, 5)),
                   steps=(4, 6))


def test_unsplit_cli_prices_time_refinement_and_serialized_diagnostics(runner, tiny_case,
                                                                     monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(runner, "load_case", lambda name: replace(tiny_case, name=name))
    assert runner.main(["--cases", "high_vol_of_vol", "--repeats", "1", "--output", str(tmp_path),
                        "--unsplit-reference", "be", "cn", "expm"]) == 0
    case = json.loads((tmp_path / "qualification.json").read_text())["cases"][0]
    reference = case["unsplit_reference"]
    assert reference["complete"] and len(reference["runs"]) == 7
    assert case["checks"]["unsplit_reference_available"]
    assert case["model_sha256"] == case["model_sha256_after"]
    rows = {(r["method"], r["variant"]): r for r in reference["runs"]}
    adi_rows = {r["name"]: r for r in case["runs"]}
    for r in reference["runs"]:
        assert r["status"] == "ok" and r["actual_steps"] == adi_rows[r["variant"]]["actual_steps"]
        assert r["grid_sha256"] == adi_rows[r["variant"]]["grid_sha256"]
        assert r["diagnostics"]["max_mass_error"] < 1e-12
        assert r["diagnostics"]["step_moments"][-1]["time_end"] == tiny_case.chain.ttms[-1]
        if r["method"] != "expm":
            assert r["max_linear_residual"] < 1e-14
        else:
            assert r["max_linear_residual"] is None
    for method in ("be", "cn"):
        errors = [rows[(method, v)]["vs_exponential"]["max_abs_bp"]
                  for v in ("base", "time_2", "time_4")]
        assert errors[2] < errors[1] < errors[0]
    # With volatility fixed at its equilibrium node, Douglas theta=1 reduces to full BE.
    assert rows[("be", "base")]["vs_adi"]["max_abs_bp"] < 1e-9
    report = (tmp_path / "qualification.md").read_text(encoding="utf-8")
    assert "Unsplit time reference" in report and "not new price-acceptance thresholds" in report
    assert "be_time_last" in report and "Max scaled residual" in report


def test_unsplit_right_ended_buckets_against_independent_dense_exponentials(runner, tiny_case):
    from scipy.linalg import expm
    from stochvolmodels.pde_solvers import BoundaryPair1D, BoundarySet2D, PDEDirection
    from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D

    model = replace(tiny_case.model, backbone_times=(.05, .18), backbone_scales=(.7, 1.2))
    case = replace(tiny_case, model=model)
    variant = runner.Variant("base", case.grid, case.steps)
    result = runner.measure_unsplit_variant(case, variant, "expm",
                                            runner.UnsplitConfig(methods=("expm",)), repeats=1)
    assert result["status"] == "ok"
    assert [s["coefficient_time"] for s in result["operators"]] == [.05, .1, .18, .3]
    assert result["diagnostics"]["step_moments"][0]["time_end"] == .05 / 4**2
    grid = case.grid
    state = grid.project_delta(0., .2)
    n = int(np.prod(grid.shape))
    basis = np.eye(n)
    boundary = BoundaryPair1D.natural()
    start, expiry = 0., 0
    for stop in (.05, .1, .18, .3):
        op = SplitOperator2D.build(grid, model.continuous_coefficients(stop, grid),
                                   BoundarySet2D(boundary, boundary))
        dense = np.column_stack([op.apply_full(e.reshape(grid.shape), PDEDirection.FORWARD).ravel()
                                 for e in basis])
        state = (expm((stop-start)*dense) @ state.ravel()).reshape(grid.shape)
        start = stop
        if stop in case.chain.ttms:
            f, d = case.chain.forwards[expiry], case.chain.discfactors[expiry]
            prices = []
            for k, kind in zip(case.chain.strikes_ttms[expiry],
                               case.chain.optiontypes_ttms[expiry]):
                sign = 1 if kind == "C" else -1
                payoff = np.maximum(sign*(f*np.exp(grid.x_mesh)-k), 0.)
                prices.append(d*grid.integrate(state*payoff))
            np.testing.assert_allclose(result["prices"][expiry], prices, atol=2e-12, rtol=2e-13)
            expiry += 1


def test_unsplit_limits_and_jumps_are_visible_skips(runner, tiny_case, monkeypatch):
    monkeypatch.setattr(runner, "_price_unsplit",
                        lambda *a: pytest.fail("unsupported solve attempted"))
    variant = runner.Variant("base", tiny_case.grid, tiny_case.steps)
    for config, method in [(runner.UnsplitConfig(max_nodes=9), "be"),
                           (runner.UnsplitConfig(max_expm_nodes=9), "expm")]:
        skipped = runner.measure_unsplit_variant(tiny_case, variant, method, config, repeats=1)
        assert skipped["status"] == "skipped" and "limit" in skipped["reason"]
        assert "prices" not in skipped
    jump = runner.load_case("compensated_down_jumps")
    skipped = runner.measure_unsplit_variant(jump, runner.Variant("base", jump.grid, jump.steps),
                                             "be", runner.UnsplitConfig(), repeats=1)
    assert skipped["status"] == "skipped" and "jump" in skipped["reason"]


def test_requested_reference_failures_cannot_qualify(runner, tiny_case, monkeypatch):
    def fail(*args):
        raise RuntimeError("deliberate unsplit solve failure")
    monkeypatch.setattr(runner, "_price_unsplit", fail)
    result = runner.qualify_case(tiny_case, repeats=1, unsplit_config=runner.UnsplitConfig(("be",)))
    assert result["status"] == "error" and not result["qualified"]
    assert not result["checks"]["unsplit_reference_available"]
    assert all("deliberate" in r["error"] for r in result["unsplit_reference"]["runs"])


def test_requested_oversized_reference_is_incomplete_not_successful(runner, tiny_case):
    result = runner.qualify_case(tiny_case, repeats=1,
                                 unsplit_config=runner.UnsplitConfig(("expm",), max_expm_nodes=9))
    assert result["status"] == "complete" and not result["qualified"]
    assert not result["checks"]["unsplit_reference_available"]
    assert result["unsplit_reference"]["comparisons"] == {}
    assert result["unsplit_reference"]["runs"][0]["status"] == "skipped"
