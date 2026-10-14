"""Historical runner contracts without redistributing any observed market data."""

import gzip
from hashlib import sha256
import importlib
import json
from pathlib import Path

import numpy as np
from numba.typed import List
import pandas as pd
import pytest
import vanilla_option_pricers as bsm

from stochvolmodels import OptionChain


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def runner():
    """Load the advanced repository example; absent from installed wheels."""
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/validate_historical_surfaces.py").is_file()), None)
    if root is None:
        pytest.skip("historical validation runner is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("validate_historical_surfaces")


@pytest.fixture
def chain():
    """Independent VIP quotes with realistic index scale and a negative-rate expiry."""
    times, forwards, discounts = np.array([0.08, 0.3]), [3456.0, 3520.0], [1.012, 0.965]
    k = np.linspace(-0.18, 0.18, 21)
    strikes = [f * np.exp(k) for f in forwards]
    cps = [np.where(k < 0, "P", "C") for _ in times]
    ivs = [0.24 - 0.18 * k + 0.05 * k**2 for _ in times]
    prices = [np.array([bsm.compute_bsm_vanilla_price(f, s, t, v, cp, d)
                       for s, v, cp in zip(ss, vs, cc)])
              for t, f, d, ss, vs, cc in zip(times, forwards, discounts, strikes, ivs, cps)]
    return OptionChain(
        ttms=times, forwards=np.asarray(forwards), discfactors=np.asarray(discounts),
        ids=np.array(["short", "long"]), strikes_ttms=List(strikes), optiontypes_ttms=List(cps),
        bid_ivs=List([v - 0.002 for v in ivs]), ask_ivs=List([v + 0.002 for v in ivs]),
        bid_prices=List([p - 0.01 for p in prices]), ask_prices=List([p + 0.01 for p in prices]),
    )


def test_historical_prices_use_observed_forward_and_discount(runner, chain):
    """Independent VIP prices catch accidentally reusing the synthetic F=100 fixture."""
    row, evidence = runner.run_job("approx_logsv", chain, "interior")
    assert row["status"] == "completed", row
    table = evidence["fit"]["tables"]["quotes"]
    quotes = pd.DataFrame(table["data"], columns=table["columns"])
    _, masks = runner.fold_masks(chain, "interior")
    target = runner.subset(chain, masks)
    expected = []
    start = 0
    for i, t in enumerate(target.ttms):
        ivs = quotes.model_iv.iloc[start:start + len(target.strikes_ttms[i])]
        expected.extend(bsm.compute_bsm_vanilla_price(
            target.forwards[i], strike, t, vol, cp, target.discfactors[i],
        ) for strike, cp, vol in zip(target.strikes_ttms[i], target.optiontypes_ttms[i], ivs))
        start += len(ivs)
    np.testing.assert_allclose(quotes.model_price, expected, rtol=1e-11, atol=1e-10)
    # Check error aggregation independently of the diagnostics implementation.
    expected_scaled = abs(np.asarray(expected) - (quotes.bid_price + quotes.ask_price) / 2)
    expected_scaled /= np.maximum((quotes.ask_price - quotes.bid_price) / 2, runner.PRICE_FLOOR)
    assert row["p95_spread_error"] == pytest.approx(np.percentile(expected_scaled, 95))
    assert row["n_test"] == len(quotes) == 14
    assert row["surface_unchecked"] == 1  # undeclared carry cannot silently pass
    grid = runner.grid_chain(chain, 1, np.array([-0.1, 0, 0.1]))
    np.testing.assert_allclose(grid.forwards, [3520])
    np.testing.assert_allclose(grid.discfactors, [0.965])


@pytest.mark.parametrize("fold", ["interior", "wings"])
@pytest.mark.parametrize("model", ["approx_logsv", "raw_svi_price", "raw_svi_price_seeded",
                                 "raw_svi_price_regularized", "raw_svi_shared_rho"])
def test_held_out_quotes_never_enter_calibration(runner, chain, fold, model):
    train, held = runner.fold_masks(chain, fold)
    assert all(not (a & b).any() and (a | b).all() for a, b in zip(train, held))
    original, evidence = runner.run_job(model, chain, fold)
    poisoned = runner.subset(chain, [np.ones(len(k), dtype=bool) for k in chain.strikes_ttms])
    for i, mask in enumerate(held):
        poisoned.bid_ivs[i][mask] += 0.2
        poisoned.ask_ivs[i][mask] += 0.2
        poisoned.bid_prices[i][mask] += 3
        poisoned.ask_prices[i][mask] += 3
    changed, changed_evidence = runner.run_job(model, poisoned, fold)
    assert original["status"] == changed["status"] == "completed"
    assert evidence["parameters"] == changed_evidence["parameters"]
    assert changed["iv_rmse"] > original["iv_rmse"] + 0.1
    tables = [item["fit"]["tables"]["quotes"] for item in (evidence, changed_evidence)]
    before, after = [pd.DataFrame(item["data"], columns=item["columns"]) for item in tables]
    np.testing.assert_allclose(before.model_price, after.model_price, rtol=0, atol=1e-10)
    expected = np.sqrt(np.mean((before.model_price - (before.bid_price+before.ask_price)/2 - 3)**2))
    assert changed["price_rmse"] == pytest.approx(expected, abs=1e-10)
    for role in ("train", "evaluation"):
        mappings = [r for r in evidence["quote_mapping"] if r["role"] == role]
        assert len({q for r in mappings for q in r["original_quote_ids"]}) == sum(
            m.sum() for m in (train if role == "train" else held))


def test_all_failure_stages_remain_in_denominator(runner, chain, monkeypatch, tmp_path):
    sparse = runner.subset(chain, [np.arange(len(k)) < 4 for k in chain.strikes_ttms])
    row, _ = runner.run_job("approx_logsv", sparse, "wings")
    assert row["status"] == "failed" and "insufficient" in row["error"]

    def fail(*args, **kwargs):
        raise RuntimeError("diagnostic failed after finite repricing")

    monkeypatch.setattr(runner, "surface_evidence", fail)
    failed, _ = runner.run_job("approx_logsv", chain, "interior")
    assert failed["status"] == "failed" and "diagnostic failed" in failed["error"]
    successful = dict(failed, status="completed", coverage=0.75)
    runner.write_summary(tmp_path, [failed, successful], [])
    summary = json.loads((tmp_path / "summary.json").read_text())["models"][0]
    assert (summary["attempted"], summary["completed"], summary["failed"]) == (2, 1, 1)
    assert summary["mean_coverage_completed_only"] == 0.75
    assert len(pd.read_csv(tmp_path / "jobs.csv")) == 2


def test_invalid_predictions_fail_instead_of_finite_subset_metrics(runner, chain, monkeypatch):
    class BadFit:
        parameters, optimizer = {}, "test"

        def predict(self, target):
            return [np.full(len(k), np.nan) for k in target.strikes_ttms], target.get_mid_vols()

    monkeypatch.setattr(runner, "fit_model", lambda *args: BadFit())
    row, _ = runner.run_job("approx_logsv", chain, "in_sample")
    assert row["status"] == "failed" and "Invalid model" in row["error"]


def test_explicit_eod_dst_and_json(runner, tmp_path):
    assert runner.request_time("2020-03-06").tz_convert("UTC").hour == 21
    assert runner.request_time("2020-03-16").tz_convert("UTC").hour == 20
    with pytest.raises(ValueError, match="calendar dates"):
        runner.request_time("2020-03-16 15:00")
    path = tmp_path / "evidence.json.gz"
    digest = runner.save_json(path, {"invalid": np.nan, "counts": np.array([1, 2])})
    assert len(digest) == 64
    assert json.loads(gzip.decompress(path.read_bytes())) == {"invalid": None, "counts": [1, 2]}


def test_optional_oca_exact_time_missing_targets_and_reconciliation(runner):
    oca = pytest.importorskip("option_chain_analytics")
    pytest.importorskip("option_chain_analytics.diagnostics",
                        reason="requires OCA >=5.3 diagnostics")
    data = oca.generate_simulated_options_data()
    time = data.get_timeindex()[0]
    missing, evidence = runner.reconstruct_snapshot(data, time - pd.Timedelta(seconds=1))
    assert missing is None and evidence["error"] == "no_exact_snapshot"
    missing, evidence = runner.reconstruct_snapshot(data, time + pd.Timedelta(seconds=1))
    assert missing is None and evidence["error"] == "no_exact_snapshot"
    chain, evidence = runner.reconstruct_snapshot(data, time, days=(1, 2, 7, 13, 30, 9999))
    assert chain is not None
    statuses = [row["status"] for row in evidence["targets"]]
    assert "unavailable" in statuses and "duplicate" in statuses
    assert evidence["quality"]["metadata"]["source_time"] == time.isoformat()
    table = evidence["provenance"]["tables"]["source_selection"]
    selection = pd.DataFrame(table["data"], columns=table["columns"])
    assert len(selection) == data.chain_ts.exchange_time.eq(time).sum()
    assert selection.svm_selected.sum() == sum(len(k) for k in chain.strikes_ttms)


def test_optional_missing_or_rejected_cache_preserves_every_job(runner, tmp_path, monkeypatch):
    pytest.importorskip("option_chain_analytics")
    pytest.importorskip("pyarrow")
    from option_chain_analytics.data import cboe

    source = Path(runner.__file__)
    hashes = {name: sha256(source.with_name(name).read_bytes()).hexdigest()
              for name in (source.name, "compare_smile_fitters.py", "_svi_benchmark.py",
                           "_svi_wing_benchmark.py")}
    monkeypatch.setattr(runner, "source_identity", lambda: {"runner_sha256": hashes})
    calls = []

    def stale(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("stale fingerprint")

    monkeypatch.setattr(cboe, "load_local_cboe_options_data", stale)
    for name in ("missing", "stale"):
        if name == "stale":
            (tmp_path / "spx_options_oca.parquet").touch()
        output = runner.run_validation(tmp_path, tmp_path / name,
                                       dates=["2020-03-16"], models=["approx_logsv"])
        jobs = pd.read_csv(output / "jobs.csv")
        assert len(jobs) == 3 and jobs.status.eq("failed").all()
        assert jobs.error.str.contains("cache required" if name == "missing" else "stale").all()
        for filename, digest in hashes.items():
            assert sha256((output / "run_sources" / filename).read_bytes()).hexdigest() == digest
    assert len(calls) == 1 and calls[0]["is_use_cache"] is True
    with pytest.raises(ValueError, match="unique calendar dates"):
        runner.run_validation(tmp_path, tmp_path / "duplicate",
                              dates=["2020-03-16", "2020-3-16"])
    assert not (tmp_path / "duplicate").exists()


def test_training_shocks_preserve_inputs_spreads_and_black_units(runner, chain):
    original = [v.copy() for v in chain.bid_ivs]
    changed, shocks = runner.perturb_training(chain, 927)
    again, other_shocks = runner.perturb_training(chain, 927)
    for i, shock in enumerate(shocks):
        np.testing.assert_array_equal(chain.bid_ivs[i], original[i])
        np.testing.assert_array_equal(shock, other_shocks[i])
        np.testing.assert_array_equal(changed.bid_ivs[i], again.bid_ivs[i])
        np.testing.assert_allclose(changed.bid_ivs[i] - chain.bid_ivs[i], shock, atol=1e-16)
        np.testing.assert_allclose(changed.ask_ivs[i] - changed.bid_ivs[i],
                                   chain.ask_ivs[i] - chain.bid_ivs[i], atol=1e-16)
        for vols, prices in ((changed.bid_ivs[i], changed.bid_prices[i]),
                             (changed.ask_ivs[i], changed.ask_prices[i])):
            expected = [bsm.compute_bsm_vanilla_price(chain.forwards[i], k, chain.ttms[i],
                                                     v, cp, chain.discfactors[i])
                        for k, v, cp in zip(chain.strikes_ttms[i], vols,
                                           chain.optiontypes_ttms[i])]
            np.testing.assert_allclose(prices, expected, rtol=1e-11, atol=1e-10)
    changed.bid_ivs[0][:] = 1e-10
    with pytest.raises(ValueError, match="perturbed training IV"):
        runner.perturb_training(changed, 927)


def test_robustness_uses_original_targets_and_independent_movement(runner, chain):
    row, evidence = runner.run_job("approx_logsv", chain, "interior", perturbation_seed=927)
    assert row["status"] == "completed", row
    extra = evidence["robustness"]
    assert len(extra["repeats"]) == 2 and extra["stage"] == "completed"
    assert row["warm_seconds"] == np.median([r["seconds"] for r in extra["repeats"]])
    tables = [evidence["fit"]["tables"]["quotes"], extra["fit"]["tables"]["quotes"]]
    base, perturbed = [pd.DataFrame(t["data"], columns=t["columns"]) for t in tables]
    for field in ("strike", "forward", "discount", "bid_price", "ask_price", "bid_iv", "ask_iv"):
        np.testing.assert_array_equal(base[field], perturbed[field])
    movement = abs(perturbed.model_price - base.model_price) / np.maximum(
        (base.ask_price - base.bid_price) / 2, 0.01)
    np.testing.assert_allclose(extra["price_changes_half_spreads"], movement)
    assert row["p95_perturbation"] == pytest.approx(np.percentile(movement, 95))
    assert len(evidence["buckets"]) == len(base)
    assert row["perturbed_surface_unchecked"] == 1
    _, held = runner.fold_masks(chain, "interior")
    poisoned = runner.subset(chain, [np.ones(len(k), dtype=bool) for k in chain.strikes_ttms])
    for i, mask in enumerate(held):
        poisoned.bid_ivs[i][mask] += 0.20
        poisoned.ask_ivs[i][mask] += 0.20
        poisoned.bid_prices[i][mask] += 100
        poisoned.ask_prices[i][mask] += 100
    changed, other = runner.run_job("approx_logsv", poisoned, "interior", perturbation_seed=927)
    assert changed["status"] == "completed"
    assert other["parameters"] == evidence["parameters"]
    assert other["robustness"]["parameters"] == extra["parameters"]
    assert changed["iv_rmse"] > row["iv_rmse"] + 0.1 and not changed["screen_pass"]


@pytest.mark.parametrize("failure_call,stage", [(2, "warm_repeat_1"), (4, "perturbed_fit")])
def test_robustness_fit_failure_keeps_partial_evidence(runner, chain, monkeypatch,
                                                     failure_call, stage, tmp_path):
    real_fit, calls = runner.fit_model, []

    def failing_fit(*args):
        calls.append(args)
        if len(calls) == failure_call:
            raise RuntimeError("deliberate repeat/refit failure")
        return real_fit(*args)

    monkeypatch.setattr(runner, "fit_model", failing_fit)
    row, evidence = runner.run_job("approx_logsv", chain, "wings", perturbation_seed=927)
    assert row["status"] == "failed" and not row["screen_pass"]
    assert "deliberate repeat/refit failure" in row["error"]
    assert evidence["robustness"]["stage"] == stage and "fit" in evidence
    runner.write_summary(tmp_path, [row], [])
    result = json.loads((tmp_path / "summary.json").read_text())["models"][0]
    assert result["attempted"] == result["failed"] == 1 and result["screen_passed"] == 0


def test_quote_buckets_use_actual_tenor_and_market_forward(runner):
    quotes = pd.DataFrame(dict(ttm=np.array([5, 7, 14, 30, 60, 90, 120])/365,
                               forward=[2000.0] * 7,
                               strike=2000 * np.exp([-0.2, -0.05, 0, 0.01, 0.05, 0.2, 0.3])))
    result = runner.quote_buckets(quotes)
    assert result.maturity_bucket.tolist() == ["0-7d", "0-7d", "7-30d", "7-30d",
                                               "30-90d", "30-90d", "90d+"]
    assert result.moneyness_bucket.tolist() == ["left_far", "left", "atm", "atm",
                                                "right", "right_far", "right_far"]


def test_changed_warm_repeat_is_rejected(runner, chain, monkeypatch):
    real_fit, calls = runner.fit_model, []

    def changed_fit(*args):
        fitted = real_fit(*args)
        calls.append(fitted)
        if len(calls) == 2:
            predict = fitted.predict

            def shifted(target):
                prices, ivs = predict(target)
                return [p + 1 for p in prices], ivs

            fitted.predict = shifted
        return fitted

    monkeypatch.setattr(runner, "fit_model", changed_fit)
    row, evidence = runner.run_job("approx_logsv", chain, "interior", perturbation_seed=927)
    assert row["status"] == "failed" and not row["screen_pass"]
    assert "fresh repeated fit changed predictions" in row["error"]
    assert evidence["robustness"]["stage"] == "warm_repeat_1"


@pytest.mark.parametrize("reliability_draws", [0, 2])
def test_domain_refusal_retains_parameters_and_all_quote_denominators(runner, chain,
                                                                    reliability_draws):
    """An unsupported held-out strike must not disappear into a finite-subset score."""
    for strikes in chain.strikes_ttms:
        strikes[-1] *= np.exp(1.)
    row, evidence = runner.run_job("raw_svi_constrained", chain, "wings",
                                    perturbation_seed=123, reliability_draws=reliability_draws)
    assert row["status"] == "failed" and not row["screen_pass"]
    assert row["stage"] == "base_domain"
    assert row["domain_outside_quotes"] == 2
    assert "outside SVI prediction domain" in row["error"]
    assert len(evidence["parameters"]) == len(chain.ttms)
    assert evidence["domain"]["outside_quotes"] == 2
    assert len(evidence["domain"]["slices"]) == len(chain.ttms)
    assert row["n_test"] == 16
    assert "fit" not in evidence and "coverage" not in row
    if reliability_draws:
        assert row["reliability_status"] == evidence["reliability"]["status"] == "complete"
        assert evidence["reliability"]["successful"] == 2
    else:
        assert "reliability" not in evidence


def test_price_objective_handles_index_scale_and_narrow_spreads(runner, chain):
    """A realizable SVI curve priced by VIP catches nominally successful poor fits."""
    for i, (t, forward, strikes, discount) in enumerate(zip(
            chain.ttms, chain.forwards, chain.strikes_ttms, chain.discfactors)):
        k = np.log(strikes / forward)
        vol = np.sqrt(.045 + .05 * (-.4*k + np.sqrt(k*k + .14**2)))
        prices = np.array([bsm.compute_bsm_vanilla_price(forward, s, t, v, cp, discount)
                           for s, v, cp in zip(strikes, vol, chain.optiontypes_ttms[i])])
        chain.bid_ivs[i][:], chain.ask_ivs[i][:] = vol-.002, vol+.002
        chain.bid_prices[i][:], chain.ask_prices[i][:] = prices-.01, prices+.01
    row, evidence = runner.run_job("raw_svi_price", chain, "in_sample")
    assert row["status"] == "completed", row.get("error")
    assert row["price_rmse"] < .0002
    assert all(e["optimizer_objective_scale"] > 1 for e in evidence["optimizer"])
    for item in evidence["optimizer"]:
        scores = [s["training_objective"] for s in item["starts"] if s["accepted"]]
        assert item["training_objective"] == min(scores)


def test_training_reliability_excludes_held_out_quotes(runner, chain):
    row, before = runner.run_job("raw_svi_constrained", chain, "wings", reliability_draws=2,
                                 reliability_seed=791)
    train, held = runner.fold_masks(chain, "wings")
    changed = runner.subset(chain, [np.ones(len(k), dtype=bool) for k in chain.strikes_ttms])
    for i, mask in enumerate(held):
        changed.bid_ivs[i][mask] += .2
        changed.ask_ivs[i][mask] += .2
        changed.bid_prices[i][mask] += 3
        changed.ask_prices[i][mask] += 3
    other, after = runner.run_job("raw_svi_constrained", changed, "wings", reliability_draws=2,
                                  reliability_seed=791)
    assert row["status"] == other["status"] == "completed"
    assert row["reliability_status"] == other["reliability_status"] == "complete"
    assert other["iv_rmse"] > row["iv_rmse"] + .1
    for evidence in (before, after):
        evidence["reliability"].pop("base_seconds")
        for attempt in evidence["reliability"]["attempts"]:
            attempt.pop("seconds")
    assert runner._json_safe(before["reliability"]) == runner._json_safe(after["reliability"])
    assert [len(k) for k in before["reliability"]["training"]["strikes_ttms"]] == [
        int(mask.sum()) for mask in train]


def test_incomplete_reliability_does_not_change_fit_status_or_screen(runner, chain, monkeypatch):
    base, _ = runner.run_job("raw_svi_constrained", chain, "wings")
    monkeypatch.setattr(runner, "diagnose_wing_reliability", lambda *a, **k: dict(
        status="incomplete", successful=1, failed=1, slices=[]))
    row, detail = runner.run_job("raw_svi_constrained", chain, "wings", reliability_draws=2)
    assert row["status"] == base["status"] == "completed"
    assert row["reliability_status"] == detail["reliability"]["status"] == "incomplete"
    assert row["reliability_max_amplification"] is None
    assert row["iv_rmse"] == base["iv_rmse"] and row["coverage"] == base["coverage"]
