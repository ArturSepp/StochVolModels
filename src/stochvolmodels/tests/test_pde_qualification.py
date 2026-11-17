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
