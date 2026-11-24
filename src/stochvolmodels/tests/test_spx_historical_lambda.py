"""Independent moment, timing and calendar checks for historical EWMA lambda."""

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite import hermgauss

mod = pytest.importorskip("papers.risk_premia_gmm.replication.spx_historical_lambda")
pytestmark = pytest.mark.repository_only


def test_binomial_root_with_cash_and_unequal_weights():
    """Solve the two-state equation algebraically, including its sign and cash leg."""
    gross = np.array([0.9, 1.15])
    weights = np.array([0.4, 0.6])
    cash = 1.01
    expected = np.log(weights[1] * (gross[1] - cash) / (weights[0] * (cash - gross[0])))
    expected /= np.log(gross[1] / gross[0])
    root, error, derivative = mod.fit_lambda(np.log(gross), np.log([cash, cash]), weights)
    assert root == pytest.approx(expected, abs=1e-10)
    assert abs(error) < 1e-12
    assert derivative < 0


@pytest.mark.parametrize("mean", [0.005, -0.005])
def test_normal_log_returns_against_closed_form(mean):
    """Gaussian quadrature reproduces the exact positive or negative CRRA root."""
    nodes, weights = hermgauss(48)
    variance, cash = 0.0025, 0.001
    x = mean + np.sqrt(2 * variance) * nodes
    root, _, _ = mod.fit_lambda(x, np.full(len(x), cash), weights / np.sqrt(np.pi))
    assert root == pytest.approx((mean - cash) / variance + 0.5, abs=1e-10)


def test_weights_match_explicit_recursive_ewma():
    """Independent recursive accumulation verifies newest-observation orientation."""
    x, span = np.array([1.0, -2.0, 8.0, 4.0]), 5
    numerator, denominator = 0.0, 0.0
    for value in x:
        numerator = (2 / 3) * numerator + value
        denominator = (2 / 3) * denominator + 1
    w = mod.ewma_weights(len(x), span)
    assert w @ x == pytest.approx(numerator / denominator)
    assert w[-1] > w[0]


def test_cash_uses_previous_published_rate_and_calendar_days():
    """Friday-to-Monday cash has three days, and Friday's unavailable rate is excluded."""
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    levels = pd.Series([100, 101, 99], index=dates)
    rates = pd.Series(
        [3.6, 36.0, 72.0], index=pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
    )
    daily, _ = mod.build_returns(levels, rates)
    assert np.expm1(daily.log_cash.iloc[0]) == pytest.approx(0.036 * 3 / 360)
    assert daily.rate_date.iloc[0] == pd.Timestamp("2024-01-04")
    assert np.expm1(daily.log_cash.iloc[1]) == pytest.approx(0.36 / 360)


def test_incomplete_month_is_excluded_and_cash_compounds():
    """A partial final month cannot become a monthly historical observation."""
    dates = pd.bdate_range("2023-12-01", "2024-02-14")
    levels = pd.Series(100 * np.exp(np.arange(len(dates)) * 0.001), index=dates)
    rates = pd.Series(3.6, index=pd.bdate_range("2023-11-27", "2024-02-14"))
    daily, monthly = mod.build_returns(levels, rates)
    assert monthly.index.tolist() == [pd.Timestamp("2024-01-31")]
    assert np.exp(monthly.log_cash.iloc[0]) == pytest.approx(
        np.exp(daily.loc["2024-01", "log_cash"]).prod()
    )


def test_estimation_is_prefix_invariant_and_excludes_signal_day():
    """Changing signal-day and future returns cannot change a historical estimate."""
    dates = pd.bdate_range("2020-01-01", periods=270)
    x = np.resize([0.02, -0.015, 0.005], len(dates))
    frame = pd.DataFrame({"log_gross": x, "log_cash": 0.0001}, index=dates)
    signal = dates[260]
    expected = mod.estimate_paths(frame, [signal], "daily", years=(1,))
    changed = frame.copy()
    changed.loc[signal:, "log_gross"] = -0.5
    actual = mod.estimate_paths(changed, [signal], "daily", years=(1,))
    truncated = mod.estimate_paths(frame.loc[: dates[259]], [signal], "daily", years=(1,))
    assert actual["lambda"].iloc[0] == expected["lambda"].iloc[0]
    assert truncated["lambda"].iloc[0] == expected["lambda"].iloc[0]
    assert actual.last_return_date.iloc[0] == dates[259]


def test_missing_root_is_not_reported_as_boundary():
    """A bound cannot silently act as an estimator when the moment has no root."""
    with pytest.raises(ValueError, match="two-sided"):
        mod.fit_lambda([0.01, 0.02], [0, 0], [0.5, 0.5])
    with pytest.raises(ValueError, match="bracket"):
        mod.fit_lambda([-0.01, 0.02], [0, 0], [0.01, 0.99], bound=1)
