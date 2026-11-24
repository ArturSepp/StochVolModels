"""Independent contracts for the prespecified mean-reverting historical filter."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import mean_reverting_lambda as model

pytestmark = pytest.mark.repository_only


@pytest.mark.parametrize("span", [10, 20])
@pytest.mark.parametrize("half_life", [2, 5, 10])
def test_constant_signal_has_analytic_transient_and_shrinkage_limit(span, half_life):
    raw = np.repeat(-4.0, 1500)
    got = model.filter_roots(raw, span, half_life)
    rho, gain = 2 ** (-1 / (12 * half_life)), 2 / (12 * span + 1)
    a = rho * (1 - gain)
    n = np.arange(1, len(raw) + 1)
    expected = 3 + gain * (-4 - 3) * (1 - a**n) / (1 - a)
    np.testing.assert_allclose(got["lambda"], expected, atol=2e-12)
    model.independent_filter_check(raw, got, span, half_life)


@pytest.mark.parametrize("half_life", [2, 5, 10])
def test_prediction_only_halves_distance_at_requested_years(half_life):
    got = model.filter_roots(np.full(12 * half_life, np.nan), 10, half_life, initial=7.0)
    assert got.internal_state.iloc[-1] == pytest.approx(5.0, abs=1e-12)
    assert got["lambda"].isna().all()  # No fixed/prior fallback is published.


def test_signed_signal_is_not_clipped_and_failed_month_is_not_skipped():
    raw = np.r_[np.repeat(-50.0, 60), np.nan, 4.0]
    got = model.filter_roots(raw, 10, 5)
    assert got["lambda"].min() < 0
    rho, gain = 2 ** (-1 / 60), 2 / 121
    expected_prior = 3 + rho * (got.internal_state.iloc[-3] - 3)
    assert got.internal_state.iloc[-2] == pytest.approx(expected_prior)
    final_prior = 3 + rho * (expected_prior - 3)
    assert got["lambda"].iloc[-1] == pytest.approx(final_prior + gain * (4 - final_prior))
    model.independent_filter_check(raw, got, 10, 5)


def sample():
    dates = pd.date_range("2000-01-31", periods=100, freq="ME")
    x = np.resize([0.045, -0.03, 0.018, -0.012], len(dates))
    return pd.DataFrame({"log_gross": x, "log_cash": 0.001}, index=dates)


def test_future_returns_and_quote_frequency_do_not_change_filter():
    monthly = sample()
    date = monthly.index[70]
    states, _ = model.monthly_states(monthly)
    dates = [date, date + pd.Timedelta(days=1), date + pd.Timedelta(days=12)]
    got = model.signal_paths(states, dates)
    obs = monthly.copy()
    obs.loc[date:, "log_gross"] = 0.4
    revised, _ = model.monthly_states(obs)
    before = model.signal_paths(revised, [date])
    np.testing.assert_allclose(got[got.date.eq(date)]["lambda"], before["lambda"])
    assert got[got.date.eq(date)].last_return_date.eq(monthly.index[69]).all()
    assert got[got.date.gt(date)].groupby(["span_years", "policy"])["lambda"].nunique().eq(1).all()
    separate = model.signal_paths(states, [dates[-1]])
    np.testing.assert_allclose(got[got.date.eq(dates[-1])]["lambda"], separate["lambda"])
    expected = model.legacy.monthly_paths(monthly, dates)
    benchmark = got[got.policy.eq("ewma")].merge(
        expected, on=["date", "span_years"], suffixes=("", "_ref")
    )
    np.testing.assert_allclose(benchmark["lambda"], benchmark.lambda_ref, atol=1e-12)


def test_short_history_stale_history_and_failed_roots_stay_missing():
    obs = sample()
    obs["log_gross"] = 0.01
    states, _ = model.monthly_states(obs)
    got = model.signal_paths(
        states, [obs.index[10], obs.index[70], obs.index[-1] + pd.Timedelta(100, "D")]
    )
    assert got["lambda"].isna().all()
    assert {"insufficient_history", "stale_history"}.issubset(set(got.status))


def test_benchmark_is_exact_original_euler_estimator():
    obs = sample()
    dates = obs.index[30::13]
    states, _ = model.monthly_states(obs)
    got = model.signal_paths(states, dates)
    expected = model.legacy.monthly_paths(obs, dates)
    q = got[got.policy.eq("ewma")].merge(expected, on=["date", "span_years"], suffixes=("", "_ref"))
    np.testing.assert_allclose(q["lambda"], q.lambda_ref, atol=1e-12)


def test_independent_reference_detects_wrong_recursion():
    raw = np.linspace(-8, 12, 50)
    got = model.filter_roots(raw, 10, 2)
    got.loc[20, "internal_state"] += 0.01
    with pytest.raises(AssertionError, match="affine"):
        model.independent_filter_check(raw, got, 10, 2)


def test_each_model_uses_same_lambda_and_preserves_failures():
    from stochvolmodels.tests.test_cross_asset_historical_lambda import fit_sample

    fits = fit_sample()
    fits.loc[1, "components"] = 3
    fits.loc[1, "theta"] = json.dumps([0.0, 0.0, 0.0, 0.0] + [np.log(0.2)] * 3)
    paths = pd.DataFrame(
        [
            {
                "asset": "GLD",
                "date": pd.Timestamp("2024-01-31"),
                "span_years": span,
                "policy": policy,
                "lambda": lam,
                "status": "estimated",
            }
            for policy in model.POLICIES
            for span, lam in [(10, -5.0), (20, 12.0)]
        ]
    )
    got, checks = model.apply_policies(fits, paths)
    assert len(got) == 24
    assert got.loc[~got.accepted, "premium_annual"].isna().all()
    q = got[got.accepted]
    np.testing.assert_allclose(
        q.premium_annual, 1.04 * np.expm1(q["lambda"] * 0.04 * q.ttm) / q.ttm, atol=1e-11
    )
    assert all(v["finite_premia"] == 4 for v in checks.values())
