"""Independent checks of nearest-expiry choice, delayed labels, HAC, and purged forecasts."""

import numpy as np
import pandas as pd
import pytest

daily = pytest.importorskip("papers.risk_premia_gmm.replication.spx_oca_daily")
study = pytest.importorskip("papers.risk_premia_gmm.replication.spx_erp_predictability")
pytestmark = pytest.mark.repository_only


def test_nearest_365_is_not_always_longest():
    """Select before quote filtering and retain a closer expiry even if its quote is bad."""
    frame = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2027-01-01", "2027-02-10"]),
            "ttm": [360 / 365, 400 / 365],
            "bid_price": [0.0, 10.0],
        }
    )
    selected = daily.select_expiry(frame)
    assert selected.expiry.iloc[0] == frame.expiry.iloc[0]
    assert selected.bid_price.iloc[0] == 0.0


def test_nearest_expiry_tie_uses_earlier_date():
    """A deterministic tie break cannot depend on the input row order."""
    frame = pd.DataFrame(
        {"expiry": pd.to_datetime(["2027-04-01", "2026-10-01"]), "ttm": [1.25, 0.75]}
    )
    assert daily.select_expiry(frame).expiry.iloc[0] == pd.Timestamp("2026-10-01")


def target_fixture():
    """Give a Friday signal with a Monday entry and an observed Friday expiry."""
    returns = pd.DataFrame(
        {"spx_total_return": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]},
        index=pd.bdate_range("2025-01-03", periods=8),
    )
    fits = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-03")],
            "case": ["gmm3"],
            "expiry": ["2025-01-10 21:00:00+00:00"],
            "rate": [0.03],
            "erp_lambda3": [0.08],
        }
    )
    return fits, returns


def test_next_session_entry_and_exact_total_return_label():
    """Known compounded levels give a hand-calculated simple annualized excess return."""
    fits, returns = target_fixture()
    actual = study.build_targets(fits, returns).query("horizon == 'listed_expiry'").iloc[0]
    assert actual.entry_date == pd.Timestamp("2025-01-06")
    assert actual.end_date == pd.Timestamp("2025-01-10")
    years = 4 / 365
    expected = (105 / 101 - 1) / years
    assert actual.annual_total_return == pytest.approx(expected)
    assert actual.annual_excess_return == pytest.approx(expected - np.expm1(0.03 * years) / years)


def test_future_endpoints_remain_censored():
    """A current index value must never substitute for a future 365-day outcome."""
    fits, returns = target_fixture()
    row = study.build_targets(fits, returns).query("horizon == 'fixed365'").iloc[0]
    assert not row.matured and pd.isna(row.end_date)
    assert np.isnan(row.annual_total_return)
    assert row.planned_end == pd.Timestamp("2026-01-06")


def test_hac_matches_statsmodels_reference():
    """Equal-spacing HAC covariance agrees with the independent library implementation."""
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(72)
    x = np.cumsum(rng.normal(0, 0.01, 800))
    error = rng.normal(0, 0.02, 800)
    error[1:] += 0.4 * error[:-1]
    y = 0.03 + 0.7 * x + error
    reference = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 80, "use_correction": True}, use_t=False
    )
    actual = study.hac_regression(x, y, np.arange(len(x)), 80)
    assert actual["beta"] == pytest.approx(reference.params[1], rel=1e-10)
    assert actual["se_hac"] == pytest.approx(reference.bse[1], rel=1e-10)
    assert actual["p_hac_normal"] == pytest.approx(reference.pvalues[1], abs=1e-12)


def test_hac_respects_missing_session_gaps():
    """Pairwise covariance weights use actual session distances, not compressed row numbers."""
    rng = np.random.default_rng(901)
    x, y = rng.normal(size=(2, 30))
    positions = np.r_[np.arange(15), np.arange(50, 65)]
    design = np.c_[np.ones(30), x]
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    score = design * residual[:, None]
    lag = 8
    weight = np.maximum(0.0, 1 - abs(positions[:, None] - positions) / (lag + 1))
    meat = score.T @ weight @ score
    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ meat @ bread * 30 / 28
    actual = study.hac_regression(x, y, positions, lag)
    assert actual["se_hac"] == pytest.approx(np.sqrt(covariance[1, 1]), rel=1e-10)


def forecast_fixture():
    """Create labels that mature after their signal dates, including future training traps."""
    dates = pd.bdate_range("2020-01-02", periods=80)
    x = np.sin(np.arange(80) / 7) * 0.04 + 0.08
    return pd.DataFrame(
        {
            "date": dates,
            "end_date": dates + pd.Timedelta(days=20),
            "entry_date": dates + pd.Timedelta(days=1),
            "matured": True,
            "erp_lambda3": x,
            "annual_total_return": 0.02 + 1.3 * x + np.cos(np.arange(80)) * 0.01,
            "annual_excess_return": 1.3 * x + np.cos(np.arange(80)) * 0.01,
            "annual_cash_return": 0.02,
        }
    )


def test_future_label_changes_cannot_change_current_forecast():
    """Changing labels unavailable at the forecast date cannot alter its fitted coefficients."""
    data = forecast_fixture()
    original = study.expanding_forecasts(data, "annual_excess_return", min_train=10)
    date = original.date.iloc[5]
    changed = data.copy()
    changed.loc[changed.end_date >= date, "annual_excess_return"] += 10.0
    altered = study.expanding_forecasts(changed, "annual_excess_return", min_train=10)
    a = original[original.date == date].iloc[0]
    b = altered[altered.date == date].iloc[0]
    assert a.linear_forecast == pytest.approx(b.linear_forecast, abs=1e-12)
    assert a.mean_forecast == pytest.approx(b.mean_forecast, abs=1e-12)
    assert a.latest_training_end < date


def test_expanding_forecast_matches_explicit_purged_least_squares():
    """Cumulative sufficient statistics match a separately reconstructed eligible regression."""
    data = forecast_fixture()
    predictions = study.expanding_forecasts(data, "annual_excess_return", min_train=10)
    for _, row in predictions.iloc[::8].iterrows():
        train = data[(data.end_date < row.date) & (data.date < row.date)]
        beta = np.linalg.lstsq(
            np.c_[np.ones(len(train)), train.erp_lambda3], train.annual_excess_return, rcond=None
        )[0]
        assert row.linear_forecast == pytest.approx(beta @ [1.0, row.erp_lambda3], abs=1e-10)
        assert row.training_count == len(train)


def test_nonoverlap_diagnostic_has_disjoint_holding_periods():
    """No two retained outcomes may share a holding interval."""
    selected = study.nonoverlap_sample(forecast_fixture())
    assert len(selected) >= 3
    assert (selected.entry_date.iloc[1:].to_numpy() > selected.end_date.iloc[:-1].to_numpy()).all()


def test_bootstrap_preserves_slope_under_a_constant_return_shift():
    """Seeded blocks and the intercept preserve an exact linear relation."""
    x = np.linspace(0.01, 0.2, 300)
    first = study.block_slope_interval(x, 2 * x + 0.03, np.arange(300), 60, replications=49)
    second = study.block_slope_interval(x, 2 * x + 0.5, np.arange(300), 60, replications=49)
    assert first["beta_ci_low"] == pytest.approx(2.0)
    assert first["beta_ci_high"] == pytest.approx(second["beta_ci_high"])


def test_oos_block_interval_exact_forecast_identities():
    """Perfect forecasts score one and forecasts equal to the benchmark score zero."""
    data = forecast_fixture()
    data["calendar_position"] = np.arange(len(data))
    data["mean_forecast"] = 0.0
    data["linear_forecast"] = data.annual_excess_return
    perfect = study.block_oos_interval(data, "annual_excess_return", "linear", 20, 49)
    assert perfect["oos_r2_ci_low"] == pytest.approx(1.0)
    data["linear_forecast"] = data.mean_forecast
    equal = study.block_oos_interval(data, "annual_excess_return", "linear", 20, 49)
    assert equal["oos_r2_ci_low"] == pytest.approx(0.0)
    assert equal["oos_r2_ci_high"] == pytest.approx(0.0)
