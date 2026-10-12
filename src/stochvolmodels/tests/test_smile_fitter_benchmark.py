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
@pytest.mark.parametrize("model", ["ssvi", "raw_svi", "raw_svi_constrained", "raw_svi_price"])
def test_holdouts_do_not_enter_training_or_initial_values(benchmark, fold, model):
    """Changing held-out quotes cannot affect a fit made from the training snapshot."""
    bench, _ = benchmark
    sample = bench.make_sample("skewed")
    mask = sample.training_mask(fold)
    before = sample.chain(mask)
    params_before = bench.fit_model(model, before).parameters
    for values in sample.ivs:
        values[~mask] += 0.2
    after = sample.chain(mask)
    params_after = bench.fit_model(model, after).parameters
    assert bench._json_safe(params_before) == bench._json_safe(params_after)
    for a, b in zip(before.bid_ivs, after.bid_ivs):
        np.testing.assert_array_equal(a, b)
    assert np.count_nonzero(mask & ~mask) == 0
    assert np.count_nonzero(mask) + np.count_nonzero(~mask) == len(sample.z)


@pytest.mark.parametrize("model", ["approx_logsv", "ssvi", "raw_svi"])
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


@pytest.mark.parametrize("rho,b", [(-0.8, 0.4), (0.7, 0.001), (0.0, 0.0)])
def test_scaled_optimizer_jacobian_against_independent_raw_formula(benchmark, rho, b):
    """Parameter derivatives agree with finite differences in the original raw form."""
    _, svi = benchmark
    z = np.linspace(-1.5, 1.5, 19)
    p = np.array([0.6, b, rho, 0.13, 0.3])
    target = np.full(len(z), 1.1)

    def independent(p):
        v, slope, corr, shift, width = p
        a = v - slope * width * np.sqrt(1 - corr**2)
        x = z - shift
        return np.sqrt(a + slope * (corr * x + np.sqrt(x*x + width*width))) - target

    analytic = svi._scaled_svi_residual(p, z, target, jacobian=True)
    steps = np.eye(5) * 1e-6
    numerical = np.column_stack([(independent(p + h) - independent(p - h)) / 2e-6
                                 for h in steps])
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=2e-10)
    np.testing.assert_allclose(svi._scaled_svi_residual(p, z, target), independent(p),
                               rtol=1e-13, atol=1e-15)


def test_exact_flat_limit_and_training_order_invariance(benchmark):
    """Flat shape is explicitly unidentified; noisy estimates are reproducible by quote set."""
    bench, svi = benchmark
    k = np.linspace(-0.2, 0.2, 11)
    for ttm in (0.01, 1.0, 3.0):
        params, evidence = svi.fit_raw_svi(k, np.full(len(k), 0.28), ttm)
        assert params.b == 0 and params.a == pytest.approx(ttm * 0.28**2)
        assert evidence["method"] == "exact_flat_limit"
        assert evidence["jacobian_condition"] is None
        np.testing.assert_allclose(svi.raw_svi(np.array([-5, 0, 5]), params)[0], ttm * 0.28**2)
    sample = bench.make_sample("flat")
    mask = sample.training_mask("interior")
    chain = sample.chain(mask, perturb=True)
    values = (chain.bid_ivs[0] + chain.ask_ivs[0]) / 2
    k = sample.ks[0][mask]
    before, evidence = svi.fit_raw_svi(k, values, sample.ttms[0])
    after, _ = svi.fit_raw_svi(k[::-1], values[::-1], sample.ttms[0])
    assert before == after
    assert evidence["successful_starts"] > 0
    assert evidence["nfev_total"] >= evidence["nfev_selected"]
    assert 0 < evidence["jacobian_rank"] <= 5
    # Independently recomputed IV SSE is in original decimal-IV units.
    x = k - before.m
    pred = np.sqrt((before.a + before.b * (before.rho*x
                                          + np.sqrt(x*x + before.sigma**2))) / sample.ttms[0])
    assert evidence["training_sse"] == pytest.approx(np.sum((pred - values)**2), abs=1e-15)


@pytest.mark.parametrize("bad", ["strike", "duplicate", "shape", "iv", "time_nan", "time_inf"])
def test_raw_svi_rejects_unusable_training_data(benchmark, bad):
    """Bad calibration domains fail explicitly rather than passing finite subsets."""
    _, svi = benchmark
    k, iv, ttm = np.linspace(-0.2, 0.2, 7), np.full(7, 0.3), 0.4
    if bad == "strike":
        k[2] = np.nan
    elif bad == "duplicate":
        k[:] = 0
    elif bad == "shape":
        iv = iv[:-1]
    elif bad == "iv":
        iv[2] = -0.1
    else:
        ttm = np.nan if bad == "time_nan" else np.inf
    with pytest.raises(ValueError, match="distinct finite"):
        svi.fit_raw_svi(k, iv, ttm)


def test_stability_quotes_and_shocks_are_frozen_and_training_only(benchmark):
    """No new fitter is evaluated here; verify the predeclared experiment's inputs."""
    bench, _ = benchmark
    sample = bench.make_stability_sample("flat", 301)
    assert len(sample.z) == 25
    np.testing.assert_array_equal(sample.ttms, [0.08, 0.3, 0.8])
    expected = np.random.default_rng(301)
    for t, k, iv in zip(sample.ttms, sample.ks, sample.ivs):
        np.testing.assert_array_equal(k, sample.z * 0.30 * np.sqrt(t))
        np.testing.assert_array_equal(iv, 0.24 + expected.normal(0, 0.0005, 25))
    shocks = np.random.default_rng(10301)
    mask = sample.training_mask("wings")
    train = sample.chain(mask, perturb=True)
    for i, iv in enumerate(sample.ivs):
        expected_iv = (iv + shocks.normal(0, 0.0005, 25))[mask]
        np.testing.assert_allclose(train.bid_ivs[i], expected_iv - 0.005, rtol=0, atol=0)
    for actual, original in zip(sample.chain().bid_ivs, sample.ivs):
        np.testing.assert_array_equal(actual, original - 0.005)


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


def test_wing_derivatives_against_independent_black_prices(benchmark):
    """Differentiate prices in strike space, independent of the analytic constraints."""
    _, svi = benchmark
    wing = importlib.import_module("_svi_wing_benchmark")
    p = svi.RawSvi(0.03, 0.06, -0.3, 0.02, 0.25)
    forward, discount = 3580.0, 1.013
    strikes = forward * np.exp(np.linspace(-0.3, 0.3, 7))

    def call(strike):
        x = np.log(strike / forward) - p.m
        w = p.a + p.b * (p.rho * x + np.sqrt(x*x + p.sigma**2))
        d1 = np.log(forward / strike) / np.sqrt(w) + np.sqrt(w) / 2
        return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d1 - np.sqrt(w)))

    k = np.log(strikes / forward)
    g, slope = wing.svi_wing_values(k, p)
    w = p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m)**2 + p.sigma**2))
    density = g * norm.pdf(-k / np.sqrt(w) - np.sqrt(w) / 2) / (strikes * np.sqrt(w))
    for h in (0.30, 0.15):
        np.testing.assert_allclose(slope, (call(strikes + h) - call(strikes - h))
                                   / (2 * h * discount), rtol=3e-7, atol=1e-8)
        numerical = (call(strikes + h) - 2 * call(strikes) + call(strikes - h)) / h**2
        np.testing.assert_allclose(density, numerical / discount, rtol=3e-6, atol=2e-10)


def test_positive_sampled_density_does_not_hide_positive_call_slope(benchmark):
    """A synthetic raw-SVI counterexample has positive g but increasing call prices."""
    _, svi = benchmark
    wing = importlib.import_module("_svi_wing_benchmark")
    # Hand-constructed parameters, no vendor quotes or calibration output.
    p = svi.RawSvi(-0.005, 0.2, 0.8, 0.05, 0.05)
    check = wing.check_svi_wings(np.linspace(0.055, 0.065, 21), p)
    assert check["min_density_factor"] > 0
    assert check["max_call_slope"] > 0.1
    assert not check["passed"]


def test_constrained_wings_resolve_observed_flat_density_defect(benchmark):
    """Previously observed development case: independent finite call slopes stay convex."""
    bench, _ = benchmark
    sample = bench.make_stability_sample("flat", 304)
    mask = sample.training_mask("wings")
    fit = bench.fit_model("raw_svi_constrained", sample.chain(mask))
    assert all(e["wing_check"]["passed"] for e in fit.optimizer)
    for i, t in enumerate(sample.ttms):
        k = np.linspace(sample.ks[i][0], sample.ks[i][-1], 1001)
        chain = bench._quote_chain(np.array([t]), [k], [np.full(len(k), 0.24)])
        p = fit.parameters[i]
        w = p["a"] + p["b"] * (p["rho"] * (k - p["m"])
                                  + np.sqrt((k - p["m"])**2 + p["sigma"]**2))
        d1 = -k / np.sqrt(w) + np.sqrt(w) / 2
        prices = chain.discfactors[0] * (chain.forwards[0] * norm.cdf(d1)
                                         - chain.strikes_ttms[0] * norm.cdf(d1 - np.sqrt(w)))
        slopes = np.diff(prices) / np.diff(chain.strikes_ttms[0]) / chain.discfactors[0]
        assert np.min(slopes) >= -1 - 1e-9
        assert np.max(slopes) <= 1e-9
        assert np.min(np.diff(slopes)) >= -1e-9


@pytest.mark.parametrize("failure", ["solver", "dense_check"])
def test_constrained_svi_rejects_unacceptable_starts(benchmark, monkeypatch, failure):
    """Optimizer success cannot bypass the final independent grid gate."""
    from types import SimpleNamespace

    wing = importlib.import_module("_svi_wing_benchmark")
    calls = []

    def fake_solver(objective, start, **kwargs):
        calls.append(start)
        x = np.array([1.0, 0.01, 0.0, 0.0, 1.0])
        return SimpleNamespace(x=x, fun=objective(x)[0], success=failure != "solver",
                               status=0, message="injected solver result", nfev=1, nit=1)

    monkeypatch.setattr(wing, "minimize", fake_solver)
    if failure == "dense_check":
        original = wing.check_svi_wings

        def bad_dense(k, p):
            assert len(k) >= wing.VALIDATION_POINTS
            return dict(original(k, p), passed=False, min_density_factor=-0.01)

        monkeypatch.setattr(wing, "check_svi_wings", bad_dense)
    with pytest.raises(RuntimeError, match="All constrained SVI starts failed"):
        wing.fit_raw_svi_constrained(np.linspace(-0.2, 0.2, 11), np.linspace(0.20, 0.22, 11), 0.5)
    assert len(calls) == 3


def test_constrained_flat_limit_and_quote_order(benchmark):
    bench, svi = benchmark
    wing = importlib.import_module("_svi_wing_benchmark")
    k = np.linspace(-0.2, 0.2, 11)
    p, evidence = wing.fit_raw_svi_constrained(k, np.full(11, 0.28), 0.3)
    np.testing.assert_allclose(svi.raw_svi(np.array([-5.0, 0, 5.0]), p)[0], 0.3 * 0.28**2)
    assert p.b == 0 and evidence["starts"] == [] and evidence["wing_check"]["passed"]
    sample = bench.make_stability_sample("flat", 304)
    mask = sample.training_mask("wings")
    k, iv, t = sample.ks[1][mask], sample.ivs[1][mask], sample.ttms[1]
    before, evidence = wing.fit_raw_svi_constrained(k, iv, t)
    after, _ = wing.fit_raw_svi_constrained(k[::-1], iv[::-1], t)
    assert before == after
    assert evidence["training_sse"] == pytest.approx(
        np.sum((np.sqrt(svi.raw_svi(k, before)[0] / t) - iv)**2), abs=1e-15)
    assert "raw_svi_constrained" not in bench.MODELS


def test_wing_study_quotes_match_frozen_protocol(benchmark):
    """Only input generation is checked; no fresh-case fit is used for development."""
    bench, _ = benchmark
    sample = bench.make_wing_sample("flat", 401)
    assert len(sample.z) == 29
    np.testing.assert_array_equal(sample.ttms, [0.06, 0.25, 0.65])
    expected = np.random.default_rng(401)
    shocks = np.random.default_rng(10401)
    mask = sample.training_mask("wings")
    train = sample.chain(mask, perturb=True)
    for i, (t, k, iv) in enumerate(zip(sample.ttms, sample.ks, sample.ivs)):
        np.testing.assert_array_equal(k, sample.z * 0.30 * np.sqrt(t))
        np.testing.assert_array_equal(iv, 0.22 + expected.normal(0, 0.0005, 29))
        np.testing.assert_array_equal(train.bid_ivs[i],
                                      (iv + shocks.normal(0, 0.0005, 29))[mask] - 0.005)


def test_price_objective_and_gradient_against_independent_vip(benchmark):
    """Different parameter mapping and VIP prices verify all five gradient columns."""
    import vanilla_option_pricers as vip

    wing = importlib.import_module("_svi_wing_benchmark")
    k = np.linspace(-0.16, 0.18, 9)
    z, target = (k - 0.01) / 0.17, np.linspace(0.9, 1.1, len(k))
    t, vol, forward, discount = 0.3, 0.28, 3456.0, 1.014
    cp = np.where(k < 0, "P", "C")
    strikes = forward * np.exp(k)
    midpoint = np.array([vip.compute_bsm_vanilla_price(forward, s, t, 0.29, c, discount)
                         for s, c in zip(strikes, cp)])
    half = np.array([0.0, .02, .1, .2, .4, .3, .15, .08, .04])
    quotes = dict(forward=forward, discount=discount, optiontypes=cp,
                  bid=midpoint-half, ask=midpoint+half)

    def independent(p):
        minimum, b, rho, m, sigma = p
        x = z-m
        variance = minimum + b*(rho*x + np.sqrt(x*x+sigma*sigma)
                                - sigma*np.sqrt(1-rho*rho))
        ivs = vol*np.sqrt(variance)
        prices = np.array([vip.compute_bsm_vanilla_price(forward, s, t, v, c, discount)
                           for s, v, c in zip(strikes, ivs, cp)])
        return np.mean(((prices-midpoint)/np.maximum(half, .01))**2)

    p = np.array([.85, .22, -.35, .1, .6])
    value, grad = wing._spread_price_objective(p, z, target, k, t, vol, quotes)
    assert value == pytest.approx(independent(p), rel=2e-11)
    for step in (1e-5, 3e-6):
        expected = [(independent(p+step*e)-independent(p-step*e))/(2*step)
                    for e in np.eye(5)]
        np.testing.assert_allclose(grad, expected, rtol=2e-6, atol=2e-5)


def test_price_objective_selects_price_best_start_without_flat_iv_shortcut(benchmark, monkeypatch):
    """Price midpoints can disagree with midpoint IV; start selection must use prices."""
    from types import SimpleNamespace

    bench, svi = benchmark
    wing = importlib.import_module("_svi_wing_benchmark")
    k, iv, t = np.linspace(-.2, .2, 11), np.full(11, .28), .3
    cp = np.where(k < 0, "P", "C")
    midpoint = bench.black_prices(100, 100*np.exp(k), t, np.full(11, .308), .98, cp)
    quotes = dict(forward=100., discount=.98, optiontypes=cp,
                  bid=midpoint-.001, ask=midpoint+.001)
    calls = []

    def solver(objective, start, **kwargs):
        minimum = 1.21 if len(calls) % 3 == 1 else 1.
        calls.append(minimum)
        p = np.array([minimum, 0., 0., 0., 1.])
        return SimpleNamespace(x=p, fun=123., success=True, status=0,
                               message="controlled price/IV disagreement", nfev=1, nit=1)

    monkeypatch.setattr(wing, "minimize", solver)
    p, evidence = wing.fit_raw_svi_constrained(k, iv, t, price_quotes=quotes)
    assert len(calls) == 3 and evidence["selected_start"] == 1
    assert evidence["objective"] == "mean_squared_half_spread_price_error"
    assert evidence["training_objective"] < 1e-15
    assert evidence["training_sse"] > .001
    np.testing.assert_allclose(np.sqrt(svi.raw_svi(k, p)[0]/t), .308)
    reversed_quotes = {name: value[::-1] if isinstance(value, np.ndarray) else value
                       for name, value in quotes.items()}
    reordered, _ = wing.fit_raw_svi_constrained(k[::-1], iv[::-1], t,
                                               price_quotes=reversed_quotes)
    assert reordered == p
    for name, value in [("bid", np.full(11, np.nan)), ("ask", midpoint-.1),
                        ("forward", -100.), ("discount", np.inf),
                        ("optiontypes", np.full(11, "X"))]:
        with pytest.raises(ValueError, match="price quotes"):
            wing.fit_raw_svi_constrained(k, iv, t, price_quotes=dict(quotes, **{name: value}))


def test_constrained_prediction_domain_is_enforced_and_reported(benchmark):
    bench, _ = benchmark
    sample = bench.make_sample("flat")
    fitted = bench.fit_model("raw_svi_constrained", sample.chain())
    low, high = fitted.optimizer[0]["wing_check"]["k_domain"]
    t = sample.ttms[0]
    edge = bench._quote_chain(np.array([t]), [np.array([low, 0., high])],
                             [np.full(3, .28)])
    evidence = {}
    bench.enforce_prediction_domain(fitted, edge, evidence)
    assert evidence["domain"]["supported"]
    assert evidence["domain"]["outside_quotes"] == 0
    fitted.predict(edge)
    beyond = bench._quote_chain(np.array([t]), [np.array([low-.01, 0., high+.01])],
                               [np.full(3, .28)])
    with pytest.raises(ValueError, match="outside SVI prediction domain"):
        fitted.predict(beyond)
    with pytest.raises(ValueError, match="outside SVI prediction domain"):
        bench.enforce_prediction_domain(fitted, beyond, evidence)
    assert evidence["domain"]["outside_quotes"] == 2
    assert evidence["domain"]["slices"][0]["outside_indices"] == [0, 2]
    assert not evidence["domain"]["supported"]
    unseen = bench._quote_chain(np.array([t/2]), [np.array([0.])], [np.array([.28])])
    with pytest.raises(ValueError, match="maturity interpolation"):
        fitted.predict(unseen)
