"""Independent arithmetic and missing-data contracts for fit diagnostics."""

import json

import numpy as np
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.diagnostics import DiagnosticReport, diagnose_fit


def chain():
    """Two uneven expiry buckets with exactly calculable price and vol residuals."""
    return OptionChain(
        ttms=np.array([0.25, 0.5]),
        forwards=np.array([100.0, 103.0]),
        strikes_ttms=[np.array([90.0, 100.0, 110.0]), np.array([103.0])],
        optiontypes_ttms=[np.array(["P", "C", "C"]), np.array(["C"])],
        ids=np.array(["near", "far"]),
        discfactors=np.array([1.01, 0.97]),
        bid_prices=[np.array([0.0, 2.0, 4.0]), np.array([6.0])],
        ask_prices=[np.array([2.0, 4.0, 4.0]), np.array([8.0])],
        bid_ivs=[np.array([0.19, 0.19, 0.19]), np.array([0.19])],
        ask_ivs=[np.array([0.21, 0.21, 0.21]), np.array([0.21])],
    )


def test_report_matches_hand_arithmetic_and_pooled_weights():
    report = diagnose_fit(
        chain(),
        model_prices=[[1.0, 5.0, 3.0], [7.0]],
        model_ivols=[[0.21, 0.18, 0.20], [0.24]],
        weights=[[1.0, 2.0, 0.0], [7.0]],
        weight_label="fixture",
        price_unit="USD per unit",
        spread_floor=0.5,
    )
    summary = report.table("summary").iloc[0]
    assert summary.price_rmse == pytest.approx(np.sqrt(5 / 4))
    assert summary.iv_rmse == pytest.approx(np.sqrt(0.0021 / 4))
    assert summary.iv_weighted_rmse == pytest.approx(np.sqrt(0.0121 / 10))
    assert summary.price_inside_bid_ask_fraction == 0.5
    assert summary.price_weight_sum == 10
    assert summary.n_locked_prices == 1
    assert summary.price_worst_quote_id == "0:1"
    np.testing.assert_allclose(report.table("quotes").spread_scaled_error, [0, 2, -2, 0])
    np.testing.assert_allclose(report.table("quotes").outside_spread, [0, 1, 1, 0])


def test_snapshot_and_json_round_trip_are_independent_of_mutable_inputs():
    prices = [np.array([1.0, 3.0, 4.0]), np.array([7.0])]
    metadata = {"selection": {"time": "2024-01-01T00:00:00+00:00"}}
    original = diagnose_fit(
        chain(), model_prices=prices, price_unit="USD", spread_floor=0.01, metadata=metadata
    )
    prices[0][0] = 20
    metadata["selection"]["time"] = "changed"
    changed = diagnose_fit(chain(), model_prices=prices, price_unit="USD", spread_floor=0.01)
    assert original.table("summary").price_rmse.iloc[0] == 0
    assert changed.table("summary").price_rmse.iloc[0] > 0
    frame = original.table("quotes")
    frame.loc[0, "model_price"] = -1
    assert original.table("quotes").model_price.iloc[0] == 1
    restored = DiagnosticReport.from_dict(
        json.loads(json.dumps(original.to_dict(), allow_nan=False))
    )
    assert restored.to_dict() == original.to_dict()
    assert original.metadata["provenance"]["selection"]["time"].startswith("2024")


def test_missing_crossed_quotes_and_invalid_models_are_not_counted_as_fits():
    quotes = chain()
    # Deliberately damaged external observations: do not weaken OptionChain validation.
    quotes.bid_prices[0][:] = [np.nan, 5.0, -1.0]
    quotes.ask_ivs[0][:] = np.nan
    report = diagnose_fit(
        quotes,
        model_prices=[[1.0, 3.0, 4.0], [np.inf]],
        model_ivols=[[0.2, 0.2, 0.2], [0.2]],
        price_unit="USD",
        spread_floor=0.1,
    )
    summary = report.table("summary").iloc[0]
    assert summary.n_price == 0
    assert summary.price_status == "not_checked"
    assert summary.price_inside_bid_ask_fraction is None
    assert summary.n_iv == 1
    assert summary.n_one_sided_prices == 1
    assert summary.n_crossed_prices == 1
    assert report.table("quotes").price_reason.tolist() == [
        "missing_or_nonfinite_quote",
        "crossed_quote",
        "invalid_quote",
        "invalid_or_missing_model",
    ]


def test_prices_survive_missing_ivs_and_zero_weights_are_explicit():
    quotes = chain()
    quotes.bid_ivs = quotes.ask_ivs = None
    result = diagnose_fit(
        quotes,
        model_prices=[[1, 3, 4], [7]],
        weights=[[0, 0, 0], [0]],
        weight_label="zero",
        price_unit="USD",
        spread_floor=0.01,
    )
    summary = result.table("summary").iloc[0]
    assert summary.price_rmse == 0
    assert summary.price_weighted_status == "not_checked"
    assert summary.iv_status == "not_checked"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spread_floor": 0},
        {"price_unit": ""},
        {"model_prices": [[1], [2]]},
        {"weights": [[1, 1, 1], [1]]},
        {"weights": [[1, -1, 1], [1]], "weight_label": "bad"},
        {"quote_ids": [["x", "x", "y"], ["z"]]},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    inputs = dict(model_prices=[[1, 3, 4], [7]], price_unit="USD", spread_floor=0.01)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        diagnose_fit(chain(), **inputs)


def test_order_invariance_with_stable_quote_ids():
    base = chain()
    kwargs = dict(
        model_prices=[[1, 5, 3], [7]],
        model_ivols=[[0.21, 0.18, 0.2], [0.24]],
        price_unit="USD",
        spread_floor=0.1,
        quote_ids=[["a", "b", "c"], ["d"]],
    )
    expected = diagnose_fit(base, **kwargs).table("summary")
    for name in (
        "strikes_ttms",
        "optiontypes_ttms",
        "bid_prices",
        "ask_prices",
        "bid_ivs",
        "ask_ivs",
    ):
        getattr(base, name)[0] = getattr(base, name)[0][::-1]
    for name in ("model_prices", "model_ivols", "quote_ids"):
        kwargs[name][0] = kwargs[name][0][::-1]
    actual = diagnose_fit(base, **kwargs).table("summary")
    assert actual.equals(expected)
