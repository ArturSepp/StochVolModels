"""Independent tests of the shared monthly historical-tilt policy."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import cross_asset_historical_lambda as model

pytestmark = pytest.mark.repository_only


def monthly_sample():
    dates = pd.date_range("2000-01-31", periods=280, freq="ME")
    x = np.resize([0.045, -0.03, 0.018, -0.012], len(dates))
    x[-80:] -= 0.009
    return pd.DataFrame({"log_gross": x, "log_cash": 0.001}, index=dates)


def test_monthly_spans_equal_existing_equity_estimator_on_every_date():
    obs = monthly_sample()
    dates = pd.date_range("2010-01-15", "2023-03-15", freq="17D")
    got = model.monthly_paths(obs, dates)
    expected = model.history.estimate_paths(obs, dates, "monthly", years=(10, 20))
    paired = got.merge(expected, on=["date", "span_years"], suffixes=("", "_equity"))
    assert len(paired) == 2 * len(dates)
    np.testing.assert_allclose(paired["lambda"], paired.lambda_equity, atol=1e-12)
    assert set(got.span_observations) == {120, 240}
    assert got.groupby("span_years")["lambda"].nunique().min() > 20
    model.verify_paths(obs, got)


def test_monthly_future_and_signal_month_never_affect_past_root():
    obs = monthly_sample()
    date = obs.index[200]
    original = model.monthly_paths(obs, [date])
    obs.loc[date:, "log_gross"] = 0.8
    changed = model.monthly_paths(obs, [date])
    np.testing.assert_array_equal(original["lambda"], changed["lambda"])
    assert original.last_return_date.eq(date - pd.offsets.MonthEnd()).all()


def test_no_fixed_fallback_during_warmup_or_failed_root():
    obs = monthly_sample()
    obs["log_gross"] = 0.01
    got = model.monthly_paths(obs, [obs.index[10], obs.index[150]])
    assert len(got) == 4 and got["lambda"].isna().all()
    assert got.status.str.contains("insufficient|two-sided").all()


@pytest.mark.parametrize("day_count", [360, 365])
def test_local_cash_daycount_and_strict_publication_lag(day_count):
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])
    levels = pd.Series([100.0, 101.0, 99.0], index=dates)
    rates = pd.Series(
        [3.6, 36.0, 72.0], index=pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
    )
    daily, _ = model.monthly_returns(levels, rates, day_count)
    assert np.expm1(daily.log_cash.iloc[0]) == pytest.approx(0.036 * 3 / day_count)
    assert (daily.rate_date < daily.start_date).all()


def test_japan_closure_exception_does_not_relax_other_gaps():
    levels = pd.Series(
        [100.0, 101.0, 99.0], index=pd.to_datetime(["2019-04-26", "2019-05-07", "2019-05-08"])
    )
    rates = pd.Series(
        [0.1, 0.2, 0.3], index=pd.to_datetime(["2019-04-25", "2019-04-26", "2019-05-07"])
    )
    with pytest.raises(ValueError):
        model.monthly_returns(levels, rates, 365)
    daily, _ = model.monthly_returns(levels, rates, 365, ["2019-05-07"])
    assert np.expm1(daily.log_cash.iloc[0]) == pytest.approx(0.001 * 11 / 365)
    rates.index = rates.index + pd.DateOffset(years=1)
    levels.index = levels.index + pd.DateOffset(years=1)
    with pytest.raises(ValueError):
        model.monthly_returns(levels, rates, 365, ["2019-05-07"])


def fit_sample():
    theta = json.dumps([0.0, 0.0, np.log(0.2), np.log(0.2)])
    return pd.DataFrame(
        [
            {
                "fit_id": str(i),
                "asset": "GLD",
                "date": pd.Timestamp("2024-01-31"),
                "components": 2,
                "accepted": i < 2,
                "theta": theta,
                "ttm": t,
                "scale": 1.04,
                "moment_low": -4.0,
                "moment_high": 5.0,
                "prior_fixed3": np.nan,
            }
            for i, t in enumerate([0.25, 1.2, 0.25])
        ]
    )


def test_one_unclipped_lambda_per_asset_date_across_models_and_tenors():
    paths = pd.DataFrame(
        [
            {
                "asset": "GLD",
                "date": pd.Timestamp("2024-01-31"),
                "span_years": span,
                "lambda": lam,
                "status": "estimated",
            }
            for span, lam in [(10, -5.0), (20, 12.0)]
        ]
    )
    got, checks = model.apply_paths(fit_sample(), paths)
    assert got.groupby("span_years")["lambda"].nunique().eq(1).all()
    assert set(got["lambda"]) == {-5.0, 12.0}
    assert got.loc[~got.accepted, "premium_annual"].isna().all()
    assert got.outside_moment_range.all()
    p = got[got.accepted]
    expected = 1.04 * np.expm1(p["lambda"] * 0.2**2 * p.ttm) / p.ttm
    np.testing.assert_allclose(p.premium_annual, expected, atol=1e-11)
    assert checks["finite_premia"] == 4


def test_migration_verifier_rejects_wrong_frozen_cash_mapping():
    fits = fit_sample().iloc[:1].copy()
    fits["prior_fixed3"] = 0.5
    paths = pd.DataFrame(
        [
            {
                "asset": "GLD",
                "date": pd.Timestamp("2024-01-31"),
                "span_years": span,
                "lambda": 1.0,
                "status": "estimated",
            }
            for span in (10, 20)
        ]
    )
    with pytest.raises(AssertionError, match="migration"):
        model.apply_paths(fits, paths)


def test_independent_root_check_rejects_constant_three():
    obs = monthly_sample()
    paths = model.monthly_paths(obs, [obs.index[-1]])
    paths["lambda"] = 3.0
    with pytest.raises(AssertionError, match="Euler"):
        model.verify_paths(obs, paths)


def test_mixed_discount_and_rate_rows_preserve_every_accepted_fit(tmp_path):
    """Legacy SPX monthly two-component rows omit D but retain the exact rate."""
    path = tmp_path / "mixed.csv"
    data = fit_sample().iloc[:2].copy()
    data["rate"] = [0.04, 0.03]
    data["discount"] = [np.nan, 0.95]
    data.to_csv(path, index=False)
    result = model.normalize_fits(
        {
            "id": "mixed",
            "asset": "SPX",
            "path": str(path),
            "sha256": model.sha256(path),
            "mapping": "reinvested_equity",
        }
    )
    np.testing.assert_allclose(result.scale, [np.exp(0.04 * 0.25), 1 / 0.95], atol=1e-12)
