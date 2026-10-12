"""Independent formula, leakage, failure-accounting and research-runner contracts."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def benchmark():
    """The benchmark is deliberately absent from installed wheels."""
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/compare_smile_fitters.py").is_file()), None)
    if root is None:
        pytest.skip("research benchmark is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield (importlib.import_module("compare_smile_fitters"),
               importlib.import_module("_svi_benchmark"))


def test_svi_derivatives_and_ssvi_equivalent_raw_form(benchmark):
    """Separate parameter mappings and finite differences verify both formulae."""
    _, svi = benchmark
    theta, rho, eta = 0.04, -0.6, 0.7
    phi = eta / np.sqrt(theta)
    equivalent = svi.RawSvi(theta * (1 - rho**2) / 2, theta * phi / 2,
                            rho, -rho / phi, np.sqrt(1 - rho**2) / phi)
    k = np.linspace(-0.6, 0.6, 31)
    for actual, expected in zip(svi.ssvi(k, theta, rho, eta), svi.raw_svi(k, equivalent)):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-15)
    at_zero = svi.ssvi(np.array([0.0]), theta, rho, eta)
    np.testing.assert_allclose([x[0] for x in at_zero],
                               [theta, theta * rho * phi, theta * phi**2 * (1 - rho**2) / 2])
    p = svi.RawSvi(0.03, 0.08, -0.4, 0.03, 0.2)
    w, first, second = svi.raw_svi(k, p)
    for step in (1e-4, 3e-5):
        plus, minus = svi.raw_svi(k + step, p)[0], svi.raw_svi(k - step, p)[0]
        np.testing.assert_allclose(first, (plus - minus) / (2 * step), rtol=2e-6, atol=2e-9)
        np.testing.assert_allclose(second, (plus - 2 * w + minus) / step**2,
                                   rtol=1e-5, atol=6e-8)


def test_durrleman_matches_independent_call_price_curvature(benchmark):
    """Density in strike space agrees with independently differentiated Black calls."""
    _, svi = benchmark
    p = svi.RawSvi(0.03, 0.06, -0.3, 0.02, 0.25)
    strikes = np.array([80.0, 95.0, 100.0, 110.0, 125.0])
    k = np.log(strikes / 100)
    w, first, second = svi.raw_svi(k, p)
    density = svi.durrleman(k, w, first, second) * norm.pdf(-k / np.sqrt(w) - np.sqrt(w) / 2)
    density /= strikes * np.sqrt(w)

    def independent_call(strike):
        x = np.log(strike / 100) - p.m
        variance = p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2))
        root = np.sqrt(variance)
        d1 = np.log(100 / strike) / root + root / 2
        return 100 * norm.cdf(d1) - strike * norm.cdf(d1 - root)

    for h in (0.03, 0.01):
        numerical = (independent_call(strikes + h) - 2 * independent_call(strikes)
                     + independent_call(strikes - h)) / h**2
        np.testing.assert_allclose(density, numerical, rtol=2e-6, atol=3e-9)


def test_published_negative_density_is_not_clipped(benchmark):
    """Gatheral/Jacquier example 3.1 is a positive-variance butterfly counterexample."""
    _, svi = benchmark
    p = svi.RawSvi(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    k = np.linspace(-2, 2, 801)
    w, first, second = svi.raw_svi(k, p)
    assert np.min(w) > 0
    assert np.min(svi.durrleman(k, w, first, second)) < -0.01


def test_constrained_ssvi_interiors_and_long_tenor_refusal(benchmark):
    """Check the whole fitted theta range and dense strike density independently."""
    _, svi = benchmark
    times = np.array([0.1, 0.4, 1.0])
    ks = [np.linspace(-0.2, 0.2, 11)] * 3
    vols = [np.full(11, v) for v in (0.30, 0.27, 0.25)]
    fit, _ = svi.fit_ssvi(ks, vols, times)
    assert np.all(fit.margins() > 0)
    k = np.linspace(-5, 5, 301)
    values = np.array([fit.variance(k, t) for t in np.linspace(0.1, 1, 23)])
    assert np.min(np.diff(values, axis=0)) >= -1e-12
    for theta in np.linspace(fit.theta[0], fit.theta[-1], 23):
        w, first, second = svi.ssvi(k, theta, fit.rho, fit.eta)
        assert np.min(svi.durrleman(k, w, first, second)) > -1e-12
    with pytest.raises(ValueError, match="outside"):
        fit.variance(k, 2.0)


@pytest.mark.parametrize("fold", ["interior", "wings"])
def test_holdouts_do_not_enter_training_or_initial_values(benchmark, fold):
    """Changing held-out quotes cannot affect a fit made from the training snapshot."""
    bench, _ = benchmark
    sample = bench.make_sample("skewed")
    mask = sample.training_mask(fold)
    before = sample.chain(mask)
    params_before = bench.fit_model("ssvi", before).parameters
    for values in sample.ivs:
        values[~mask] += 0.2
    after = sample.chain(mask)
    params_after = bench.fit_model("ssvi", after).parameters
    for key in params_before:
        np.testing.assert_array_equal(params_before[key], params_after[key])
    for a, b in zip(before.bid_ivs, after.bid_ivs):
        np.testing.assert_array_equal(a, b)
    assert np.count_nonzero(mask & ~mask) == 0
    assert np.count_nonzero(mask) + np.count_nonzero(~mask) == len(sample.z)


@pytest.mark.parametrize("model", ["approx_logsv", "ssvi"])
def test_final_reports_reprice_known_flat_target(benchmark, model):
    """The full job recomputes base and perturbed diagnostics on the same holdouts."""
    bench, _ = benchmark
    row, evidence = bench.run_job(model, bench.make_sample("flat"), "interior", repeats=1)
    assert row["status"] == "ok", row.get("error")
    assert row["n_test"] == 18
    # The native approximate fitter has a positive curvature floor; its flat fit
    # need not be exact. Assert the predeclared economic tolerance and coverage.
    assert row["iv_rmse"] < bench.CRITERIA["iv_rmse"]
    assert row["coverage"] == 1
    assert row["surface_failures"] == 0
    assert "perturbed" in evidence
    assert np.all(np.array(evidence["timings"]) > 0)


def test_raw_svi_recovers_nondegenerate_formula(benchmark):
    """A fit agrees with independent direct values, without assuming flat identifiability."""
    _, svi = benchmark
    k = np.linspace(-0.4, 0.4, 21)
    target = np.sqrt(0.03 + 0.08 * (-0.4 * (k - 0.03)
                                  + np.sqrt((k - 0.03)**2 + 0.2**2)))
    fitted, evidence = svi.fit_raw_svi(k, target, 1.0)
    np.testing.assert_allclose(np.sqrt(svi.raw_svi(k, fitted)[0]), target, atol=2e-6)
    assert evidence["successful_starts"] > 0


def test_nonfinite_predictions_and_fit_failures_remain_visible(benchmark, monkeypatch):
    """No success-only aggregate or finite subset may conceal a failed fit."""
    bench, _ = benchmark
    sample = bench.make_sample("flat")
    chain = sample.chain()
    prices = [np.ones_like(k) for k in chain.strikes_ttms]
    ivs = [np.full_like(k, 0.28) for k in chain.strikes_ttms]
    ivs[1][3] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        bench._validate_predictions(chain, prices, ivs)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberately failed optimization")

    monkeypatch.setattr(bench, "fit_model", fail)
    row, _ = bench.run_job("raw_svi", sample, "wings", repeats=1)
    assert row["status"] == "failed"
    assert not row["screen_pass"]
    assert "deliberately failed" in row["error"]
    summary = bench.summarize([row])[0]
    assert summary["attempted"] == 1 and summary["failed"] == 1
    assert summary["screen_passed"] == 0
    assert summary["mean_coverage_successful_only"] is None


def test_research_screen_rejects_perturbed_surface_violation(benchmark):
    """Even a good base fit cannot hide invalid behavior after quote perturbation."""
    bench, _ = benchmark
    row = dict(model="raw_svi", status="ok", p95_spread_error=0.1, coverage=1,
               iv_rmse=0.001, p95_perturbation=0.1, surface_failures=0,
               perturbed_surface_failures=0, warm_seconds=0.1)
    assert bench.passes_screen(row)
    row["perturbed_surface_failures"] = 1
    assert not bench.passes_screen(row)


def test_deadline_checks_after_a_single_long_objective(benchmark, monkeypatch):
    """An evaluation that starts within budget cannot return unnoticed after an overrun."""
    bench, _ = benchmark

    class Native:
        def compute_model_ivols_for_chain(self):
            return [np.array([0.3])]

    class Limited(bench._CalibrationDeadline, Native):
        pass

    times = iter([0.0, 1.0, 76.0])
    monkeypatch.setattr(bench, "perf_counter", lambda: next(times))
    limited = Limited()
    with pytest.raises(TimeoutError, match="75s"):
        limited.compute_model_ivols_for_chain()
