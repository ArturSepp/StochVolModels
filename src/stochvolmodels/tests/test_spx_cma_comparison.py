"""Regression checks for quarterly economic/option premium comparisons."""

import numpy as np
import pandas as pd
import pytest

cma = pytest.importorskip("papers.risk_premia_gmm.replication.spx_cma_comparison")
pytestmark = pytest.mark.repository_only


def option_rows(dates):
    """Make identical-date model rows for alignment checks."""
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(d), "case": case, "erp_lambda3": 0.01 * (i + 1)}
            for i, d in enumerate(dates)
            for case in cma.CASES
        ]
    )


def test_quarter_alignment_never_uses_future_or_stale_options():
    """A nearer future observation must lose to a prior quote; stale matches stay absent."""
    quarters = pd.DatetimeIndex(["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"])
    economic = pd.DataFrame({"cma_us": [0.03] * 4}, index=quarters).rename_axis("quarter")
    daily = option_rows(["2020-03-27", "2020-04-01", "2020-06-19", "2020-09-30", "2020-10-01"])
    out = cma.align_quarters(economic, daily)
    assert len(out) == 3  # December is beyond actual option history.
    assert out.loc[quarters[0], "option_date"] == pd.Timestamp("2020-03-27")
    assert out.loc[quarters[0], "gmm2"] == 0.01
    assert not out.loc[quarters[1], "matched"]
    assert out.loc[quarters[2], "option_age_days"] == 0


def test_alignment_keeps_the_same_option_date_for_every_model():
    """A partial newer fit cannot change one model's comparison date."""
    economic = pd.DataFrame({"cma_us": [0.03]}, index=pd.to_datetime(["2020-03-31"]))
    economic.index.name = "quarter"
    daily = option_rows(["2020-03-30", "2020-03-31"])
    daily = daily[~((daily.date == pd.Timestamp("2020-03-31")) & (daily.case == "gmm3"))]
    out = cma.align_quarters(economic, daily)
    assert out.iloc[0].option_date == pd.Timestamp("2020-03-30")
    np.testing.assert_allclose(out[list(cma.CASES)].to_numpy(float), 0.01)


def test_rank_correlation_uses_average_ties_and_is_scale_invariant():
    """Tied observations use mid-ranks, and percent/decimal display does not change results."""
    x, y = [1, 1, 3, 4], [1, 2, 2, 4]
    expected = np.corrcoef([1.5, 1.5, 3, 4], [1, 2.5, 2.5, 4])[0, 1]
    original = cma.correlations(x, y)
    scaled = cma.correlations(100 * np.array(x), 100 * np.array(y))
    assert original["spearman"] == pytest.approx(expected)
    assert scaled == pytest.approx(original)


def test_block_intervals_preserve_a_perfect_negative_relationship():
    """Dependent resampling must resample predictor and response as a pair."""
    x = np.arange(20.0)
    intervals = cma.block_intervals(x, 2 - 3 * x, block=4, draws=99)
    np.testing.assert_allclose(list(intervals.values()), -1, atol=1e-12)


def test_valuation_comparison_excludes_equilibrium():
    """The requested valuation target must never be replaced by the blended premium."""
    dates = pd.date_range("2015-03-31", periods=20, freq="QE")
    panel = pd.DataFrame(
        {"valuation_leg": np.arange(20.0), "cma_us": -np.arange(20.0)}, index=dates
    )
    for case in cma.CASES:
        panel[case] = np.arange(20.0)
    stats, _, _ = cma.summarize(panel, 19, economic="valuation_leg")
    assert set(stats.economic) == {"valuation_leg"}
    level = stats[stats["sample"] == "quarterly_levels"]
    np.testing.assert_allclose(level[["pearson", "spearman"]], 1.0)
