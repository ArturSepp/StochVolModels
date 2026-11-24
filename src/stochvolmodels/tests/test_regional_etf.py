"""Causal seasonal equity-ETF cash and expiry-selection regression checks."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import regional_etf_study as study


def cash(ex, amount, kind="Income", declared=None, pay=None):
    ex = pd.Timestamp(ex)
    return dict(
        ex_date=ex,
        declared_date=pd.Timestamp(declared) if declared else ex - pd.Timedelta(days=3),
        payable_date=pd.Timestamp(pay) if pay else ex + pd.Timedelta(days=4),
        dividend_amount=amount,
        dividend_type=kind,
    )


def schedule(rows, date="2024-01-31", expiry="2025-01-31"):
    return study.distribution_schedule("FEZ", pd.DataFrame(rows), date, expiry, 0.04)


def test_seasonal_cash_preserves_quarterly_amounts_and_discounting():
    rows = [
        cash(f"2023-{month}-20", value)
        for month, value in [(3, 0.1), (6, 1.2), (9, 0.2), (12, 0.4)]
    ]
    result = schedule(rows)
    assert len(result) == 4
    np.testing.assert_allclose([r["cash"] for r in result], [0.1, 1.2, 0.2, 0.4])
    for r in result:
        lag = (pd.Timestamp(r["pay_date"]) - pd.Timestamp(r["ex_date"])).days
        assert r["drop"] == pytest.approx(r["cash"] * np.exp(-0.04 * lag / 365))
        assert pd.Timestamp(r["amount_information_date"]) < pd.Timestamp("2024-01-31")


def test_future_declarations_never_change_calendar_or_cash():
    base = [cash("2023-06-20", 1), cash("2023-12-20", 0.2)]
    future = cash("2024-06-21", 99, declared="2024-06-19")
    assert schedule(base) == schedule(base + [future])


def test_announced_replaces_forecast_and_paid_event_is_not_duplicated():
    rows = [
        cash("2023-06-20", 1),
        cash("2023-12-20", 0.2),
        cash("2024-06-18", 0.7, declared="2024-06-10"),
    ]
    before = schedule(rows, "2024-06-12", "2024-07-31")
    assert (
        len(before) == 1
        and before[0]["cash"] == 0.7
        and before[0]["kind"] == "prior_declared"
    )
    assert schedule(rows, "2024-06-19", "2024-07-31") == []
    assert schedule(rows, "2024-06-12", "2024-06-18") == []


def test_splits_and_cancelled_cash_are_excluded_and_special_cash_not_repeated():
    rows = [
        cash("2023-06-20", 1),
        cash("2023-12-20", 0.2),
        cash("2023-12-20", 2, "Special Cash"),
        cash("2023-06-20", 4, "Stock Split"),
        cash("2023-09-20", 5, "Cancelled"),
    ]
    assert [r["cash"] for r in schedule(rows)] == [1, 0.2]
    rows += [
        cash("2024-06-18", 0.5, declared="2024-06-10"),
        cash("2024-06-18", 0.3, "Special Cash", declared="2024-06-10"),
    ]
    result = schedule(rows, "2024-06-12", "2024-07-31")
    assert len(result) == 1 and result[0]["cash"] == pytest.approx(0.8)


def test_latest_quarter_and_omitted_cash_replace_older_amount():
    rows = [
        cash("2022-12-20", 3),
        cash("2023-06-20", 0.5),
        cash("2023-12-20", 0, "Omitted"),
    ]
    assert [r["cash"] for r in schedule(rows)] == [0.5, 0]
    with pytest.raises(ValueError, match="Negative"):
        schedule([cash("2023-06-20", -0.5)])


def test_horizon_beyond_year_contains_next_season_without_duplicate_dates():
    rows = [cash("2023-06-20", 0.5), cash("2023-12-20", 0.2)]
    result = schedule(rows, "2024-05-31", "2025-06-30")
    assert [r["cash"] for r in result] == [0.5, 0.2, 0.5]
    assert len({r["ex_date"] for r in result}) == 3


def test_omitted_payment_without_pay_date_is_zero_but_positive_missing_date_fails():
    omitted = cash("2023-12-20", -2.4e-14, "Omitted")
    omitted["payable_date"] = pd.NaT
    result = schedule([cash("2023-06-20", .5), omitted])
    assert [r["cash"] for r in result] == [.5, 0]
    assert result[-1]["drop"] == 0
    invalid = cash("2023-12-20", 1)
    invalid["payable_date"] = pd.NaT
    with pytest.raises(ValueError, match="Missing cash event dates"):
        schedule([invalid])


def test_expiry_selection_is_calendar_only_and_fallback_is_longest():
    dates = pd.to_datetime(["2024-04-19", "2024-09-20", "2025-01-17", "2025-02-21"])
    assert study.select_expiry(dates, "2024-01-31") == pd.Timestamp("2025-01-17")
    assert study.select_expiry(dates[:2], "2024-01-31", 90, 299) == pd.Timestamp(
        "2024-09-20"
    )
