"""Independent convention, valuation-unblending and maturity tests for regional ERP."""

from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

regional = pytest.importorskip("papers.risk_premia_gmm.replication.regional_equities")
validation = pytest.importorskip("papers.risk_premia_gmm.replication.regional_validation")
pytestmark = pytest.mark.repository_only


def test_spot_strikes_are_normalized_by_observed_forward():
    """The 100% spot strike is 100/105 of forward, independently of the cash proxy."""
    task = regional.normalized_inputs(100.0, 105.0, np.full(9, 0.2), 5.0, days=365)
    assert task["k"][4] == pytest.approx(100 / 105)
    assert task["k"][0] == pytest.approx(80 / 105)
    assert task["k"][-1] == pytest.approx(120 / 105)
    assert task["discount"] == pytest.approx(1 / 1.05)
    assert np.exp(task["rate"]) == pytest.approx(1.05)
    assert task["ttm"] == 1.0


def test_negative_local_ois_and_360_day_units():
    """Minus 50bp implies a discount above one, with ACT/365 year fraction."""
    task = regional.normalized_inputs(100.0, 98.0, np.full(9, 0.2), -0.5)
    assert task["ttm"] == pytest.approx(360 / 365)
    assert task["discount"] == pytest.approx(1 / (1 - 0.005 * 360 / 365))
    assert task["discount"] > 1 and task["rate"] < 0


@pytest.mark.parametrize(
    "spot,forward,vols,rate,message",
    [
        (100.0, 100.0, np.full(8, 0.2), 2.0, "Missing"),
        (100.0, 100.0, np.r_[np.nan, np.full(8, 0.2)], 2.0, "Missing"),
        (100.0, 100.0, np.full(9, 0.2), np.nan, "Missing"),
        (0.0, 100.0, np.full(9, 0.2), 2.0, "Invalid"),
        (100.0, 100.0, np.r_[0.0, np.full(8, 0.2)], 2.0, "Invalid"),
        (100.0, 100.0, np.full(9, 0.2), 100.0, "Invalid"),
        (100.0, 130.0, np.full(9, 0.2), 2.0, "bracket"),
    ],
)
def test_invalid_or_incomplete_inputs_are_rejected(spot, forward, vols, rate, message):
    """No fill or unit fallback should turn unusable inputs into calibration tasks."""
    with pytest.raises(ValueError, match=message):
        regional.normalized_inputs(spot, forward, vols, rate)


def test_no_arbitrage_rejects_obvious_call_price_spike():
    """An ATM volatility spike makes adjacent call prices rise with strike."""
    vols = np.full(9, 0.2)
    vols[4] = 1.0
    with pytest.raises(ValueError, match="monotonicity/convexity"):
        regional.normalized_inputs(100.0, 100.0, vols, 2.0)


def test_valuation_uses_raw_regions_and_removes_global_anchor(monkeypatch):
    """Hand-specified blends recover valuation only, including a regional add-on."""
    dates = pd.to_datetime(["2026-03-31", "2026-06-30"])
    sheets = {
        "equity_cma_excess": pd.DataFrame(
            {
                "Europe": [0.065, 0.065],
                "UK": [0.075, 0.075],
                "Japan": [0.030, 0.030],
                "Equity EU/Asia": [999.0, 999.0],
            },
            index=dates,
        ),
        "factor_vols": pd.DataFrame({"Equity": [0.2, 0.2], "Europe": [99.0, 99.0]}, index=dates),
        "config_snapshot": pd.DataFrame(
            {"value": [0.5, 0.4, 0.01]},
            index=[
                "blend_weight_current",
                "sharpe_ratio.Equity",
                "regional_equity_pcaey_add_on.UK",
            ],
        ),
    }
    monkeypatch.setattr(pd, "ExcelFile", lambda _: nullcontext("fixture"))
    monkeypatch.setattr(pd, "read_excel", lambda book, sheet_name, index_col: sheets[sheet_name])
    recovered = regional.valuation_components("fixture").groupby("asset").first()
    assert recovered.loc["sx5e", "valuation"] == pytest.approx(0.05)
    assert recovered.loc["ukx", "valuation"] == pytest.approx(0.06)
    assert recovered.loc["nky", "valuation"] == pytest.approx(-0.02)
    assert not recovered.loc["sx5e", "index_match"]
    assert recovered.loc["ukx", "index_match"] and recovered.loc["nky", "index_match"]


def target_fixture():
    """A Friday signal, Monday entry and holiday-delayed 360-day exit."""
    index = pd.bdate_range("2025-01-03", "2026-01-08").difference(pd.to_datetime(["2026-01-01"]))
    values = np.full(len(index), 100.0)
    values[index.get_loc("2025-01-03")] = 80.0
    values[index.get_loc("2025-01-06")] = 101.0
    values[index.get_loc("2026-01-02")] = 111.1
    returns = pd.DataFrame(
        {"total_return_check": values, "generic_total_return": values * 2}, index=index
    )
    fits = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-03")],
            "case": ["gmm2"],
            "rate": [-0.01],
            "erp_lambda3": [0.05],
        }
    )
    return fits, returns


def test_delayed_entry_360_day_exit_and_cash_are_hand_calculated():
    """The return excludes the signal-day jump and uses the actual 361-day holding period."""
    fits, returns = target_fixture()
    row = validation.build_targets(fits, returns).iloc[0]
    assert row.entry_date == pd.Timestamp("2025-01-06")
    assert row.planned_end == pd.Timestamp("2026-01-01")
    assert row.end_date == pd.Timestamp("2026-01-02")
    assert row.target_years == pytest.approx(361 / 365)
    expected = 0.1 / (361 / 365)
    assert row.annual_total_return == pytest.approx(expected)
    assert row.annual_excess_return == pytest.approx(
        expected - np.expm1(-0.01 * 361 / 365) / (361 / 365)
    )


def test_immature_target_cannot_use_last_available_close():
    """Incomplete outcomes remain missing even if the latest level is available."""
    fits, returns = target_fixture()
    row = validation.build_targets(fits, returns.loc[:"2025-12-31"]).iloc[0]
    assert not row.matured and not row.observed_outcome
    assert pd.isna(row.end_date) and np.isnan(row.annual_total_return)


def test_generic_sensitivity_keeps_primary_calendar_and_missing_endpoint():
    """A missing generic index level must not shift either endpoint or get filled."""
    fits, returns = target_fixture()
    returns.loc["2026-01-02", "generic_total_return"] = np.nan
    row = validation.build_targets(fits, returns, "generic_total_return").iloc[0]
    assert row.end_date == pd.Timestamp("2026-01-02")
    assert row.matured and not row.observed_outcome and np.isnan(row.annual_total_return)


def test_holiday_gap_is_retained_but_missing_month_raises():
    """The ten-day Japanese holiday is valid; an unexplained month-long gap is not."""
    fits, returns = target_fixture()
    fits.loc[0, "date"] = pd.Timestamp("2025-01-03")
    returns = returns.loc["2025-01-13":]
    assert validation.build_targets(fits, returns).iloc[0].entry_date == pd.Timestamp("2025-01-13")
    with pytest.raises(ValueError, match="gap"):
        validation.build_targets(fits, returns.loc["2025-02-03":])


def test_heldout_volatility_cannot_set_training_cap(monkeypatch):
    """A high withheld IV cannot widen the fitted model's component-volatility limit."""
    seen = []

    def capture(task):
        seen.append(task)
        return {"accepted": False}, []

    monkeypatch.setattr(validation.base, "fit_slice", capture)
    task = {
        "asset": "ukx",
        "date": "2025-01-03",
        "case": "gmm3_maxiv12",
        "starts": 8,
        "rate": 0.02,
        "ttm": 360 / 365,
        "vols": [0.3, 0.9, 0.25, 0.24, 0.8, 0.22, 0.21, 0.7, 0.2],
        "k": np.linspace(0.8, 1.2, 9).tolist(),
    }
    validation.heldout_task(task)
    assert seen[0]["vol_cap"] == pytest.approx(0.36)
    assert len(seen[0]["vols"]) == 6
    assert max(seen[0]["vols"]) == 0.3


def test_quarterly_changes_do_not_bridge_missing_quarters():
    """A jump across a missing quarter is not a one-quarter change."""
    panel = pd.DataFrame(
        {
            "quarter": pd.to_datetime(["2025-03-31", "2025-06-30", "2025-12-31", "2026-03-31"]),
            "valuation": [1.0, 2.0, 99.0, 100.0],
            "gmm2": [3.0, 4.0, 50.0, 52.0],
        }
    )
    actual = regional.quarterly_changes(panel, "gmm2")
    np.testing.assert_allclose(actual.to_numpy(), [[1.0, 1.0], [1.0, 2.0]])
