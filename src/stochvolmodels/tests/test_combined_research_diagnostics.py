"""Independent consumer arithmetic, provenance and abstention checks."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochvolmodels import OptionChain
from stochvolmodels.fitters.diagnostics import DiagnosticReport


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def combined():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/combined_research_diagnostics.py").is_file()), None)
    if root is None:
        pytest.skip("combined report is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("combined_research_diagnostics")


@pytest.fixture
def chain():
    return OptionChain(ttms=np.array([.1]), forwards=np.array([100.]),
        discfactors=np.array([.95]), ids=np.array(["T0"]),
        strikes_ttms=[100*np.exp(np.linspace(-.2, .2, 5))],
        optiontypes_ttms=[np.array(["P", "P", "C", "C", "C"])],
        bid_ivs=[np.full(5, .245)], ask_ivs=[np.full(5, .255)],
        bid_prices=[np.ones(5)], ask_prices=[np.full(5, 3.)])


def sidecars(chain):
    """Build a serialized synthetic fixture with explicitly matched parity legs."""
    rows, checks, pairs, ids = [], [], [], []
    config = dict(price_unit="USD per unit", payoff="standard_european", settlement="cash")
    metadata = dict(config=config, requested_time="2026-10-13T08:00:00+00:00",
                    source_time="2026-10-13T08:00:00+00:00")
    for i, (strike, cp) in enumerate(zip(chain.strikes_ttms[0], chain.optiontypes_ttms[0])):
        common = dict(strike=float(strike), expiry="2026-11-18T20:00:00+00:00",
                      exchange_time=metadata["source_time"], mat_id="T0", underlying_index="TEST")
        legs = []
        identity = f"selected:{i}"
        ids.append(identity)
        for side in (cp, "C" if cp == "P" else "P"):
            source_id = identity if side == cp else f"counterpart:{i}"
            shift = (chain.discfactors[0]*(chain.forwards[0]-strike)
                     * (1 if cp == "P" else -1)) if side != cp else 0.
            rows.append(dict(**common, source_row_id=source_id, optiontype=side,
                ttm=float(chain.ttms[0]), forward_price=float(chain.forwards[0]),
                discount=float(chain.discfactors[0]), bid_price=chain.bid_prices[0][i]+shift,
                ask_price=chain.ask_prices[0][i]+shift, bid_iv=chain.bid_ivs[0][i],
                ask_iv=chain.ask_ivs[0][i]))
            for name in ("strike", "forward_price", "discount", "ttm", "optiontype",
                         "two_sided_price", "nonnegative_price", "uncrossed_price", "iv",
                         "exchange_time", "expiry", "price_bounds", "quote_age"):
                checks.append(dict(source_row_id=source_id, check=name, status="pass", reason=""))
            legs.append(source_id)
        pairs.append(dict(**common, source_row_ids=legs, status="pass", reason=""))
    quality = DiagnosticReport.from_tables(dict(quotes=pd.DataFrame(rows),
        checks=pd.DataFrame(checks), summary=pd.DataFrame([dict(n_rows=len(rows),
        snapshot_age_status="pass")])), dict(kind="quote_quality", **metadata)).to_dict()
    parity = DiagnosticReport.from_tables(dict(pairs=pd.DataFrame(pairs)),
        dict(kind="parity", **metadata)).to_dict()
    return quality, parity, [ids]


def call(combined, chain, **kwargs):
    defaults = dict(model_prices=[np.full(len(chain.strikes_ttms[0]), 2.)],
                    model_ivols=[np.full(len(chain.strikes_ttms[0]), .25)],
                    evaluation_k=[np.linspace(-.4, .4, 9)],
                    price_unit="USD per unit", spread_floor=.01)
    defaults.update(kwargs)
    return combined.combine_research_diagnostics(chain, **defaults)


def statuses(report):
    return report.table("facets").set_index("facet").status.to_dict()


def test_residuals_match_hand_arithmetic_and_keep_separate_facets(combined, chain):
    q, p, ids = sidecars(chain)
    report = call(combined, chain, model_prices=[[1., 4., 2., 5., 2.]],
                  model_ivols=[[.25, .256, .244, .25, .25]], quality=q, parity=p, quote_ids=ids)
    summary = report.table("summary").iloc[0]
    assert summary.price_rmse == pytest.approx((14/5)**.5)
    assert summary.iv_rmse == pytest.approx((.000072/5)**.5)
    assert summary.price_interval_inside_fraction == summary.iv_interval_inside_fraction == .6
    assert summary.price_interval_outside == summary.iv_interval_outside == 2
    assert summary.price_interval_checked == summary.iv_interval_checked == 5
    assert summary.price_worst_quote_id == "selected:3"
    assert statuses(report) == dict(quote_quality="pass", parity="pass", observed_fit="flag",
                                   strike_support="extrapolated", wing_sensitivity="unavailable")
    support = report.table("support")
    assert support.region.eq("left_extrapolation").sum() == 2
    assert support.region.eq("right_extrapolation").sum() == 2
    assert support.accuracy_status.eq("unvalidated_extrapolation").sum() == 4
    assert report.metadata["overall_accuracy_status"] is None
    assert not report.metadata["accuracy_certificate"] and not report.metadata["automatic_warning"]


def test_locked_intervals_accept_ties_and_only_declared_roundoff(combined, chain):
    chain.bid_prices[0][:] = chain.ask_prices[0][:] = 2.
    chain.bid_ivs[0][:] = chain.ask_ivs[0][:] = .25
    report = call(combined, chain, model_prices=[[2.+5e-11]*5], model_ivols=[[.25+5e-13]*5])
    assert statuses(report)["observed_fit"] == "clear"
    assert call(combined, chain).table("summary").n_locked_prices.iloc[0] == 5
    out = call(combined, chain, model_prices=[[2.+2e-10]*5], model_ivols=[[.25+2e-12]*5])
    assert statuses(out)["observed_fit"] == "flag"


def test_incomplete_residuals_preserve_exclusions_and_known_outside_counts(combined, chain):
    chain.bid_prices[0][0] = np.nan
    chain.ask_ivs[0][1] = np.nan
    out = call(combined, chain, model_prices=[[2., 4., np.inf, 2., 2.]])
    summary = out.table("summary").iloc[0]
    assert summary.price_interval_excluded == 2 and summary.iv_interval_excluded == 1
    assert summary.price_interval_outside == 1
    assert statuses(out)["observed_fit"] == "unavailable"
    json.dumps(out.to_dict(), allow_nan=False)


@pytest.mark.parametrize("defect", ["price", "iv", "strike", "cp", "forward", "units",
                                   "future", "naive", "row_id", "parity_time", "parity_ids"])
def test_mismatched_sidecars_cannot_grant_eligibility(combined, chain, defect):
    q, p, ids = sidecars(chain)
    frame = DiagnosticReport.from_dict(q).table("quotes")
    if defect in ("price", "iv", "strike", "cp", "forward", "row_id"):
        name = dict(price="bid_price", iv="ask_iv", strike="strike", cp="optiontype",
                    forward="forward_price", row_id="source_row_id")[defect]
        frame.loc[0, name] = "X" if defect == "cp" else "unknown" if defect == "row_id" else 999.
        q["tables"]["quotes"] = dict(columns=list(frame.columns),
                                      data=frame.to_numpy(dtype=object).tolist())
    elif defect == "units":
        q["metadata"]["config"]["price_unit"] = "index points"
    elif defect in ("future", "naive"):
        q["metadata"]["source_time"] = ("2026-10-13T08:01:00+00:00" if defect == "future"
                                             else "2026-10-13T08:00:00")
    elif defect == "parity_time":
        p["metadata"]["requested_time"] = "2026-10-13T08:01:00+00:00"
    else:
        rows = DiagnosticReport.from_dict(p).table("pairs")
        rows.at[0, "source_row_ids"] = ["absent", "selected:0"]
        p["tables"]["pairs"] = dict(columns=list(rows.columns),
                                    data=rows.to_numpy(dtype=object).tolist())
    with pytest.raises(ValueError):
        call(combined, chain, quality=q, parity=p, quote_ids=ids)


@pytest.mark.parametrize("defect", ["missing_quality", "missing_parity", "unchecked_age",
                                   "stale", "parity_fail", "source_unknown"])
def test_missing_unchecked_and_failed_evidence_stays_visible(combined, chain, defect):
    q, p, ids = sidecars(chain)
    expected = "unavailable"
    facet = "quote_quality"
    if defect == "missing_quality":
        q = None
    elif defect == "missing_parity":
        p, facet = None, "parity"
    elif defect == "source_unknown":
        q["metadata"]["source_time"] = p["metadata"]["source_time"] = None
    elif defect in ("unchecked_age", "stale"):
        checks = DiagnosticReport.from_dict(q).table("checks")
        checks.loc[checks.check.eq("quote_age"), "status"] = (
            "fail" if defect == "stale" else "not_checked")
        checks.loc[checks.check.eq("quote_age"), "reason"] = "quote_age"
        q["tables"]["checks"] = dict(columns=list(checks.columns),
                                      data=checks.to_numpy(dtype=object).tolist())
        expected = "fail" if defect == "stale" else "unavailable"
    else:
        pairs = DiagnosticReport.from_dict(p).table("pairs")
        pairs.loc[0, ["status", "reason"]] = ["fail", "outside_parity_interval"]
        p["tables"]["pairs"] = dict(columns=list(pairs.columns),
                                    data=pairs.to_numpy(dtype=object).tolist())
        facet, expected = "parity", "fail"
    out = call(combined, chain, quality=q, parity=p, quote_ids=ids)
    assert statuses(out)[facet] == expected
    assert statuses(out)["wing_sensitivity"] == "unavailable"


def test_immutable_roundtrip_does_not_retain_models_or_sidecars(combined, chain):
    q, p, ids = sidecars(chain)
    model = np.full(5, .25)
    report = call(combined, chain, model_ivols=[model], quality=q, parity=p, quote_ids=ids)
    before = report.to_dict()
    model[:] = .9
    q["metadata"]["config"]["price_unit"] = "changed"
    chain.bid_prices[0][:] = 99
    table = report.table("quotes")
    table.loc[0, "model_iv"] = .9
    assert report.to_dict() == before
    assert DiagnosticReport.from_dict(json.loads(json.dumps(before, allow_nan=False))) == report


@pytest.fixture
def sensitivity_fixture(combined):
    bench = importlib.import_module("compare_smile_fitters")
    svi = importlib.import_module("_svi_benchmark")
    wing = importlib.import_module("_wing_research_policy")
    params = dict(a=.0016, b=.0072, rho=-.4, m=.01, sigma=.08)
    k = np.linspace(-.2, .2, 31)
    base = np.sqrt(svi.raw_svi(k, svi.RawSvi(**params))[0]/.1)
    chain = bench._quote_chain(np.array([.1]), [k], [base])
    snapshot = {n: [np.asarray(a).copy() for a in getattr(chain, n)]
                if n.endswith("_ttms") or n in ("bid_ivs", "ask_ivs", "bid_prices", "ask_prices")
                else getattr(chain, n).copy()
                for n in ("ttms", "forwards", "discfactors", "ids", "strikes_ttms",
                          "optiontypes_ttms", "bid_ivs", "ask_ivs", "bid_prices", "ask_prices")}
    grid = np.linspace(-.4, .4, 81)
    curve = np.sqrt(svi.raw_svi(grid, svi.RawSvi(**params))[0]/.1)
    reliability = dict(training=snapshot, status="complete", successful=128, failed=0,
        requested=128, metadata=dict(model="raw_svi_constrained", grid_points=81, shock_std=.001),
        base=dict(parameters=[params]), slices=[dict(grid=dict(k=grid, base_iv=curve))],
        attempts=[dict(index=i, status="completed", ivs=[curve+.0001]) for i in range(128)])
    policy = dict(format=wing.FORMAT, policy_id="fixture", scope=deepcopy(wing.SCOPE),
                  cutoffs={"0.2": 5., "0.4": 3.}, numerical_sha256=wing.numerical_hashes(),
                  calibration_sha256="a"*64)
    q, p, ids = sidecars(chain)
    prices = bench.black_prices(chain.forwards[0], chain.strikes_ttms[0], .1, base,
                                 chain.discfactors[0], chain.optiontypes_ttms[0])
    return chain, dict(model_prices=[prices], model_ivols=[base], evaluation_k=[grid],
        quality=q, parity=p, quote_ids=ids, reliability=reliability, policy=policy,
        noise_assumption="known_iid_gaussian_iv")


@pytest.mark.parametrize("defect", [None, "failure", "snapshot", "model", "grid",
                                   "base_curve", "quality"])
def test_sensitivity_requires_same_inputs_model_grid_and_quality(
        combined, sensitivity_fixture, defect):
    chain, args = sensitivity_fixture
    if defect == "failure":
        args["reliability"]["attempts"][4]["status"] = "failed"
    elif defect == "snapshot":
        args["reliability"]["training"]["bid_prices"][0][15] *= .99
    elif defect == "model":
        args["model_ivols"][0][0] += .001
    elif defect == "grid":
        args["evaluation_k"] = [np.linspace(-.8, .8, 81)]
    elif defect == "base_curve":
        args["reliability"]["slices"][0]["grid"]["base_iv"][-1] += .001
    elif defect == "quality":
        args["parity"] = None
    out = call(combined, chain, **args)
    assert statuses(out)["wing_sensitivity"] == ("clear" if defect is None else "unavailable")
    assert statuses(out)["strike_support"] == "extrapolated"
    assert out.metadata["overall_accuracy_status"] is None


@pytest.mark.parametrize("conventions", [[], ["--price-unit", "coin per unit"],
                                        ["--spread-floor", "1e-8"]])
def test_saved_cli_requires_explicit_price_conventions(combined, monkeypatch, capsys,
                                                      conventions):
    import sys
    monkeypatch.setattr(sys, "argv", ["combined_research_diagnostics.py",
                        "--saved-reliability", "missing.json", *conventions])
    with pytest.raises(SystemExit) as error:
        combined.run_local()
    assert error.value.code == 2
    assert "Saved replay requires --price-unit and --spread-floor" in capsys.readouterr().err


def test_saved_cli_preserves_declared_price_conventions(
        combined, sensitivity_fixture, tmp_path, monkeypatch):
    import sys
    _, args = sensitivity_fixture
    saved = tmp_path / "reliability.json"
    saved.write_text(json.dumps(args["reliability"], default=lambda a: a.tolist()),
                     encoding="utf-8")
    output = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["combined_research_diagnostics.py",
                        "--saved-reliability", str(saved), "--price-unit", "coin per unit",
                        "--spread-floor", "1e-8", "--output", str(output)])
    combined.run_local()
    report = DiagnosticReport.from_dict(json.loads((output / "report.json").read_text())["report"])
    assert report.metadata["price_unit"] == "coin per unit"
    assert report.metadata["spread_floor"] == 1e-8
    assert statuses(report)["quote_quality"] == "unavailable"
    assert statuses(report)["wing_sensitivity"] == "unavailable"
