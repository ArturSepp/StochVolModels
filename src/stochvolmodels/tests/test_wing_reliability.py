"""Independent sensitivity and failure-accounting contracts for research diagnostics."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import erf


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def reliability():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/compare_smile_fitters.py").is_file()), None)
    if root is None:
        pytest.skip("wing reliability is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("diagnose_wing_reliability")


def call_price(k, iv, t, forward, discount):
    root = iv*np.sqrt(t)
    d1 = -k/root+root/2
    return discount*forward*((1+erf(d1/np.sqrt(2)))/2
                             - np.exp(k)*(1+erf((d1-root)/np.sqrt(2)))/2)


@pytest.fixture
def training(reliability):
    bench = importlib.import_module("compare_smile_fitters")
    chain = bench._quote_chain(np.array([0.2, 0.6]),
                              [np.linspace(-.2, .2, 9), np.linspace(-.1, .3, 11)],
                              [np.full(9, .25), np.full(11, .30)])
    chain.forwards[:] = [3500., 3600.]
    chain.discfactors[:] = [1.01, .94]
    chain.strikes_ttms[0] = 3500.*np.exp(np.linspace(-.2, .2, 9))
    chain.strikes_ttms[1] = 3600.*np.exp(np.linspace(-.1, .3, 11))
    return chain


def fake_fit(chain, offset):
    domains = []
    for t, f, strike in zip(chain.ttms, chain.forwards, chain.strikes_ttms):
        k = np.log(strike/f)
        center, half = (k.min()+k.max())/2, np.ptp(k)/2
        domains.append((float(t), center-2*half, center+2*half))

    def predict(target):
        ivs, prices = [], []
        for i, (t, f, d, strike) in enumerate(zip(target.ttms, target.forwards,
                                                target.discfactors, target.strikes_ttms)):
            k = np.log(strike/f)
            iv = np.full(len(k), [.25, .30][i]) + offset*(1+k)
            ivs.append(iv)
            prices.append(call_price(k, iv, t, f, d))
        return prices, ivs

    return SimpleNamespace(domains=tuple(domains), predict=predict,
                           parameters={"offset": offset}, optimizer={"success": True})


def test_dispersion_uses_base_rms_and_fixed_training_grid(reliability, training, monkeypatch):
    offsets = iter([0., .001, .002, .003, .004])
    monkeypatch.setattr(reliability, "fit_model", lambda model, c: fake_fit(c, next(offsets)))
    out = reliability.diagnose_wing_reliability(training, draws=4, seed=91, shock_std=.0005)
    assert out["status"] == "complete" and out["successful"] == out["requested"] == 4
    assert out["failed"] == 0
    for i, item in enumerate(out["slices"]):
        grid = item["grid"]
        k = np.asarray(grid["k"])
        delta = np.array([.001, .002, .003, .004])[:, None]*(1+k)
        rms = np.sqrt(np.mean(delta**2, axis=0))
        np.testing.assert_allclose(grid["rms_iv_change"], rms, atol=1e-15)
        assert not np.allclose(rms, np.std(delta, axis=0))
        expected = np.quantile([.25, .30][i]+delta, [.05, .5, .95], axis=0)
        for field, values in zip(["iv_q05", "iv_q50", "iv_q95"], expected):
            np.testing.assert_allclose(grid[field], values, atol=1e-15)
        np.testing.assert_allclose(grid["max_abs_iv_change"], np.max(abs(delta), axis=0))
        f, d, t = training.forwards[i], training.discfactors[i], training.ttms[i]
        p0 = call_price(k, np.full(len(k), [.25, .30][i]), t, f, d)
        pr = call_price(k, [.25, .30][i]+delta, t, f, d)
        np.testing.assert_allclose(grid["rms_normalized_call_change"],
                                   np.sqrt(np.mean(((pr-p0)/(f*d))**2, axis=0)), atol=1e-15)
        assert len(k) == 81
        assert item["regions"]["combined_wings"]["max_rms_amplification"] == pytest.approx(
            max(rms)/.0005)
        assert item["regions"]["left_extrapolation"]["points"] > 0
        assert item["regions"]["right_extrapolation"]["points"] > 0
    assert out["metadata"]["warning_threshold"] is None


def test_shocks_copy_inputs_keep_spreads_and_reprice_at_original_conventions(
        reliability, training, monkeypatch):
    seen = []
    original = [x.copy() for x in training.bid_ivs]

    def record(model, c):
        seen.append(c)
        return fake_fit(c, 0.)

    monkeypatch.setattr(reliability, "fit_model", record)
    out = reliability.diagnose_wing_reliability(training, draws=3, seed=88)
    rng = np.random.default_rng(88)
    for attempt, changed in zip(out["attempts"], seen[1:]):
        for i, (bid, ask) in enumerate(zip(training.bid_ivs, training.ask_ivs)):
            shock = rng.normal(0, .0005, len(bid))
            np.testing.assert_array_equal(attempt["shocks"][i], shock)
            np.testing.assert_allclose(changed.bid_ivs[i], bid+shock, rtol=0, atol=0)
            np.testing.assert_allclose(changed.ask_ivs[i]-changed.bid_ivs[i], ask-bid)
            np.testing.assert_array_equal(changed.forwards, training.forwards)
            k = np.log(changed.strikes_ttms[i]/changed.forwards[i])
            for side in ["bid", "ask"]:
                call = call_price(k, getattr(changed, f"{side}_ivs")[i], changed.ttms[i],
                                  changed.forwards[i], changed.discfactors[i])
                price = np.where(changed.optiontypes_ttms[i] == "C", call,
                                 call+changed.discfactors[i]*(changed.strikes_ttms[i]
                                                            - changed.forwards[i]))
                np.testing.assert_allclose(getattr(changed, f"{side}_prices")[i], price,
                                           rtol=1e-10, atol=1e-10)
    for before, after in zip(original, training.bid_ivs):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("all_fail", [False, True])
def test_failed_draws_never_produce_complete_scores(reliability, training, monkeypatch, all_fail):
    calls = []

    def fail(model, c):
        calls.append(True)
        if len(calls) > 1 and (all_fail or len(calls) == 3):
            raise RuntimeError("injected optimizer failure")
        return fake_fit(c, .001*(len(calls)-1))

    monkeypatch.setattr(reliability, "fit_model", fail)
    out = reliability.diagnose_wing_reliability(training, draws=3, seed=92)
    assert len(calls) == 4 and out["status"] == "incomplete"
    assert out["failed"] == (3 if all_fail else 1) and len(out["attempts"]) == 3
    assert any(a.get("error") == "RuntimeError: injected optimizer failure"
               for a in out["attempts"])
    for item in out["slices"]:
        assert item["regions"]["combined_wings"]["max_rms_amplification"] is None
        if all_fail:
            assert item["grid"]["rms_iv_change"] is None


@pytest.mark.parametrize("bad", ["draws", "seed", "shock", "grid", "model"])
def test_invalid_configuration_rejected(reliability, training, bad):
    kwargs = {"draws": 2}
    kwargs.update({"draws": {"draws": 1}, "seed": {"seed": -1},
                   "shock": {"shock_std": 0}, "grid": {"grid_points": 8},
                   "model": {"model": "ssvi"}}[bad])
    with pytest.raises(ValueError):
        reliability.diagnose_wing_reliability(training, **kwargs)


def test_invalid_shocked_ivs_are_retained_without_clipping(reliability, training, monkeypatch):
    monkeypatch.setattr(reliability, "fit_model", lambda model, c: fake_fit(c, 0.))
    out = reliability.diagnose_wing_reliability(training, draws=3, seed=7, shock_std=10.)
    assert out["failed"] == 3 and out["successful"] == 0
    assert all(a["stage"] == "perturb" and "nonpositive" in a["error"] for a in out["attempts"])
    assert all(len(a["shocks"]) == len(training.ttms) for a in out["attempts"])


@pytest.mark.parametrize("defect", ["base", "domain", "prediction"])
def test_failure_stage_and_fixed_domain_are_preserved(reliability, training, monkeypatch, defect):
    count = 0

    def bad(model, c):
        nonlocal count
        count += 1
        if defect == "base":
            raise RuntimeError("bad base")
        fit = fake_fit(c, 0.)
        if count > 1:
            if defect == "domain":
                fit.domains = tuple((t, lo+.01, hi-.01) for t, lo, hi in fit.domains)
            else:
                fit.predict = lambda target: ([np.full(81, np.nan)]*2, [np.full(81, .25)]*2)
        return fit

    monkeypatch.setattr(reliability, "fit_model", bad)
    out = reliability.diagnose_wing_reliability(training, draws=2)
    if defect == "base":
        assert out["status"] == "failed" and out["stage"] == "base_fit"
        assert len(out["attempts"]) == out["successful"] == out["failed"] == 0
    else:
        assert out["status"] == "incomplete" and out["failed"] == 2
        assert {a["stage"] for a in out["attempts"]} == {
            "prediction_domain" if defect == "domain" else "prediction"}
        if defect == "domain":
            assert all(a["domain"]["outside_quotes"] == 4 for a in out["attempts"])


def test_seed_repeats_and_training_snapshot_are_exact(reliability, training, monkeypatch):
    monkeypatch.setattr(reliability, "fit_model", lambda model, c: fake_fit(c, 0.))
    a = reliability.diagnose_wing_reliability(training, draws=2, seed=814)
    b = reliability.diagnose_wing_reliability(training, draws=2, seed=814)
    for before, after in zip(a["attempts"], b["attempts"]):
        for x, y in zip(before["shocks"], after["shocks"]):
            np.testing.assert_array_equal(x, y)
    training.bid_ivs[0][0] = .15
    assert a["training"]["bid_ivs"][0][0] == .245


def test_complete_evidence_is_strict_json_serializable(reliability, training, monkeypatch):
    import json

    monkeypatch.setattr(reliability, "fit_model", lambda model, c: fake_fit(c, 0.))
    report = reliability.diagnose_wing_reliability(training, draws=2)
    restored = json.loads(json.dumps(reliability._json_safe(report), allow_nan=False))
    assert restored["training"]["bid_ivs"][0] == training.bid_ivs[0].tolist()
