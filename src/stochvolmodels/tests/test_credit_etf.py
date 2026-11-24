"""Information-boundary and failed-conversion tests for the credit ETF adapter."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import credit_etf_analysis as analysis
from papers.risk_premia_gmm.replication import credit_etf_study as study
from papers.risk_premia_gmm.replication import etf_atm_fit as fit


def test_expiry_selection_is_bounded_deterministic_and_missing():
    date = pd.Timestamp("2024-01-02")
    expiries = [date + pd.Timedelta(days=d) for d in (400, 370, 360, 200, 500)]
    assert study.select_expiry(expiries, date) == date + pd.Timedelta(days=360)
    assert study.select_expiry(expiries[::-1], date) == date + pd.Timedelta(days=360)
    assert study.select_expiry([date + pd.Timedelta(days=200)], date) is None
    assert study.select_expiry(expiries, date, 90, 299) == date + pd.Timedelta(days=200)


def test_same_session_spot_never_uses_previous_or_following_day():
    data = pd.DataFrame(
        {"exchange_time": ["2024-01-30T22:20:00Z", "2024-02-01T22:20:00Z"], "close": [100.0, 110.0]}
    )
    with pytest.raises(ValueError, match="same-session"):
        study.same_session_spot(data, "2024-01-31")
    today = pd.DataFrame({"exchange_time": ["2024-02-01T00:15:00Z"], "close": [102.0]})
    assert study.same_session_spot(today, "2024-01-31")[0] == 102.0
    with pytest.raises(ValueError, match="duplicate"):
        study.same_session_spot(pd.concat([today, today]), "2024-01-31")


def test_future_distribution_records_cannot_change_prior_forecast():
    past = pd.DataFrame(
        {
            "declared_date": ["2023-12-29"],
            "ex_date": ["2024-01-02"],
            "payable_date": ["2024-01-05"],
            "dividend_amount": [0.4],
        }
    )
    future = pd.DataFrame(
        {
            "declared_date": ["2024-02-01"],
            "ex_date": ["2024-02-02"],
            "payable_date": ["2024-02-08"],
            "dividend_amount": [50.0],
        }
    )
    a = study.pilot.distribution_schedule(past, "2024-01-31", "2024-07-31", 0.04)
    b = study.pilot.distribution_schedule(
        pd.concat([past, future]), "2024-01-31", "2024-07-31", 0.04
    )
    assert a == b
    assert all(x["cash"] == 0.4 for x in a)


def test_freeze_rejects_changed_inputs(tmp_path):
    path = tmp_path / "design.json"
    study.freeze(path, {"spread": 1.0, "hash": "a"})
    study.freeze(path, {"spread": 1.0, "hash": "a"})
    with pytest.raises(ValueError, match="Changed frozen design"):
        study.freeze(path, {"spread": 1.0, "hash": "b"})
    assert json.loads(path.read_text())["hash"] == "a"


def test_failed_central_conversion_is_not_dropped(monkeypatch, tmp_path):
    strikes = [90.0, 94.0, 97.0, 99.0, 101.0, 103.0, 106.0, 110.0]
    quotes = [
        {
            "contract": str(k),
            "strike": k,
            "optiontype": "P" if k < 100 else "C",
            "bid_price": 1.0,
            "ask_price": 2.0,
            "bid_size": 1.0,
            "ask_size": 1.0,
        }
        for k in strikes
    ]
    meta = {
        "asset": "LQD",
        "date": "2024-01-31",
        "expiry": "2025-01-31",
        "ttm": 366 / 365,
        "spot": 100.0,
        "forward": 100.0,
        "discount": 0.96,
        "rate": 0.04,
    }

    def conversion(spot, strike, *args, **kwargs):
        if strike == 99.0:
            raise ValueError("Synthetic central conversion failure")
        return {
            "exercise_premium": 0.1,
            "european_mid": 1.4,
            "american_fit_error": 0.0,
            "american_iv": 0.2,
        }

    monkeypatch.setattr(study.pilot.exercise, "conversion", conversion)
    result = study.convert_date({"key": "test", "meta": meta, "quotes": quotes, "schedule": []})
    assert len(result["quotes"]) == 8
    assert result["task"]["k"][3] == 0.99
    assert np.isnan(result["task"]["iv"][3])
    with pytest.raises(ValueError, match="Invalid interval"):
        fit.AtmProblem(result["task"], 2, 0.5, 2.0)
    (tmp_path / "full_tasks.jsonl").write_text(json.dumps(result["task"]) + "\n")
    pd.DataFrame(result["quotes"]).to_csv(tmp_path / "conversion_inputs.csv", index=False)
    assert analysis.conversion_audit(tmp_path) == 8
    corrupted = copy.deepcopy(result["task"])
    corrupted["ask"][0] += 0.01
    (tmp_path / "full_tasks.jsonl").write_text(json.dumps(corrupted) + "\n")
    with pytest.raises(AssertionError):
        analysis.conversion_audit(tmp_path)
