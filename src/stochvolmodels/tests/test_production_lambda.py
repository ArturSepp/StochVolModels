"""Independent checks of the adopted cross-asset estimator and selection contract."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import production_lambda as model

pytestmark = pytest.mark.repository_only


def roots():
    dates = pd.date_range("2000-01-01", periods=72, freq="MS")
    parts = []
    for asset, sign in [("SPX", 1), ("TLT", -1)]:
        for span in (10, 20):
            parts.append(
                pd.DataFrame(
                    {
                        "asset": asset,
                        "date": dates,
                        "last_return_date": dates - pd.Timedelta(days=1),
                        "span_years": span,
                        "policy": "ewma",
                        "status": "estimated",
                        "lambda": sign * (5 + 2 * np.sin(np.arange(72) / 7)),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def test_adopted_specification_is_fixed_and_exact():
    assert model.PRODUCTION.span_years == 10
    assert model.PRODUCTION.half_life_years == 2
    assert model.PRODUCTION.anchor == 3
    assert model.PRODUCTION.gain == 2 / 121
    assert model.PRODUCTION.rho == 2 ** (-1 / 24)
    with pytest.raises(FrozenInstanceError):
        model.PRODUCTION.span_years = 20


@pytest.mark.parametrize("span", [10, 20])
@pytest.mark.parametrize("half_life", [1, 2, 5, 10])
def test_each_asset_matches_independent_geometric_weights(span, half_life):
    raw = roots()
    spec = model.LambdaSpec(span, half_life)
    got = model.filtered_states(raw, spec)
    for asset, group in got.groupby("asset"):
        original = raw[raw.asset.eq(asset) & raw.span_years.eq(span)]["lambda"].to_numpy()
        decay = (1 - spec.gain) * spec.rho
        expected = (
            3
            + spec.gain
            * np.convolve(original - 3, decay ** np.arange(len(original)))[: len(original)]
        )
        np.testing.assert_allclose(group["lambda"], expected, atol=1e-13)
        assert group.is_production.eq(span == 10 and half_life == 2).all()


def test_future_roots_do_not_change_past_estimates():
    raw = roots()
    got = model.filtered_states(raw)
    cutoff = pd.Timestamp("2003-01-01")
    changed = raw.copy()
    changed.loc[changed.date > cutoff, "lambda"] = 90
    other = model.filtered_states(changed)
    pd.testing.assert_frame_equal(got[got.date.le(cutoff)], other[other.date.le(cutoff)])


def test_failed_root_is_not_published_as_anchor_and_negative_values_are_retained():
    raw = roots()
    raw.loc[raw.date.eq(pd.Timestamp("2001-01-01")), "lambda"] = np.nan
    got = model.filtered_states(raw)
    assert got.loc[got.date.eq(pd.Timestamp("2001-01-01")), "lambda"].isna().all()
    assert np.isfinite(got.internal_state).all()
    assert got.loc[got.asset.eq("TLT"), "lambda"].min() < 0


@pytest.mark.parametrize("defect", ["duplicate", "gap", "current_return", "availability"])
def test_invalid_historical_chronology_is_rejected(defect):
    raw = roots()
    if defect == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    elif defect == "gap":
        raw = raw.drop(index=1)
    elif defect == "current_return":
        raw.loc[0, "last_return_date"] = raw.loc[0, "date"]
    else:
        raw.loc[0, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        model.filtered_states(raw)


def test_production_selection_does_not_filter_failed_quotes_or_pick_other_span():
    rows = pd.DataFrame(
        {
            "span_years": [10, 10, 20, 10],
            "policy": ["mr_h2", "mr_h2", "mr_h2", "mr_h5"],
            "accepted": [True, False, True, True],
            "premium_annual": [0.1, np.nan, 0.2, 0.3],
        }
    )
    chosen = model.select_production(rows)
    assert chosen.index.tolist() == [0, 1]
    assert chosen.premium_annual.isna().sum() == 1
    with pytest.raises(ValueError, match="no alternative"):
        model.select_production(rows[rows.span_years.eq(20)])


def comparison_panel():
    dates = pd.to_datetime(["2020-01-15", "2020-01-31", "2020-02-15", "2020-02-28", "2020-03-31"])
    rows = []
    for n in (2, 3):
        for span in (10, 20):
            for policy in ("ewma", "mr_h1", "mr_h2", "mr_h5", "mr_h10"):
                for j, date in enumerate(dates):
                    value = 0.05 + j * 0.01
                    if n == 3 and span == 20 and policy == "mr_h10" and j == 3:
                        value = np.nan
                    rows.append(
                        {
                            "asset": "SPX",
                            "components": n,
                            "span_years": span,
                            "policy": policy,
                            "date": date,
                            "premium_annual": value,
                            "lambda": 3.0,
                            "outside_moment_range": False,
                            "moment_restriction_available": True,
                            "full_span_available": True,
                        }
                    )
    return pd.DataFrame(rows)


def test_common_comparison_requires_both_models_and_every_setting():
    from papers.risk_premia_gmm.replication import production_lambda_analysis as analysis

    p = comparison_panel()
    dates = analysis.common_dates(p, "premium_annual")
    assert len(dates) == 4
    assert pd.Timestamp("2020-02-28") not in dates


def test_month_end_failure_does_not_become_earlier_quote_or_bridge_months(tmp_path):
    from papers.risk_premia_gmm.replication import production_lambda_analysis as analysis

    result = analysis.premium_summaries(comparison_panel(), tmp_path)
    assert result.adjacent_month_pairs.eq(0).all()
    assert result.median_abs_month_move_pp.isna().all()
    assert result.common_dates.eq(4).all()
