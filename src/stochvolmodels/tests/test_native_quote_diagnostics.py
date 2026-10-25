"""Independent native-unit arithmetic and fail-closed quote-scale contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.adapters.native_quotes import (
    diagnose_native_quotes,
    normalized_vanilla_price_bounds,
)
from stochvolmodels.fitters.diagnostics import DiagnosticReport, diagnose_fit


def _native_inputs() -> dict[str, object]:
    return dict(
        bid_prices=np.array([0.001, 0.006, 0.01]),
        ask_prices=np.array([0.002, 0.006, 0.014]),
        tick_sizes=np.array([0.0001, 0.0005, 0.0001]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
        price_unit="coin per unit",
        quote_ids=["spread", "locked", "wide"],
    )


def _chain() -> OptionChain:
    return OptionChain(
        ttms=np.array([0.5]), forwards=np.array([100.0]), ids=np.array(["native"]),
        strikes_ttms=[np.array([90.0, 100.0, 110.0])],
        optiontypes_ttms=[np.array(["P", "C", "C"])],
        bid_prices=[np.array([0.001, 0.006, 0.01])],
        ask_prices=[np.array([0.002, 0.006, 0.014])],
    )


def test_native_scales_and_open_shock_intervals_match_hand_arithmetic() -> None:
    report = diagnose_native_quotes(**_native_inputs())
    quotes = report.table("quotes")
    np.testing.assert_allclose(quotes.spread_scale, [0.0005, 0.00025, 0.002])
    np.testing.assert_allclose(quotes.spread_floor, [0.00005, 0.00025, 0.00005])
    np.testing.assert_allclose(quotes.lower_coefficient, [-2.0, -24.0, -5.0])
    np.testing.assert_allclose(quotes.upper_coefficient, [1996.0, 3976.0, 493.0])
    assert quotes.admissibility_status.tolist() == ["pass"] * 3
    assert report.metadata["coefficient_interval"] == "strictly open"
    assert report.metadata["probability_model"] is None
    assert report.metadata["automatic_policy"] is None


def test_effective_native_scales_feed_fit_diagnostics_without_unit_conversion() -> None:
    native = diagnose_native_quotes(**_native_inputs()).table("quotes")
    fitted = diagnose_fit(
        _chain(), model_prices=[[0.002, 0.0065, 0.01]], price_unit="coin per unit",
        spread_floor=[native.spread_scale.to_numpy()], quote_ids=[native.quote_id.tolist()],
    )
    quotes = fitted.table("quotes")
    np.testing.assert_allclose(quotes.spread_scaled_error, [1.0, 2.0, -1.0])
    np.testing.assert_allclose(quotes.spread_scale, native.spread_scale)
    assert quotes.quote_id.tolist() == native.quote_id.tolist()


def test_price_and_tick_rescaling_preserves_coefficients_and_fit_errors() -> None:
    inputs = _native_inputs()
    original = diagnose_native_quotes(**inputs).table("quotes")
    factor = 70000.0
    for key in ("bid_prices", "ask_prices", "tick_sizes", "lower_bounds", "upper_bounds"):
        inputs[key] = np.asarray(inputs[key]) * factor
    inputs["price_unit"] = "USD per unit"
    converted = diagnose_native_quotes(**inputs).table("quotes")
    for key in ("lower_coefficient", "upper_coefficient"):
        np.testing.assert_allclose(converted[key], original[key])
    np.testing.assert_allclose(converted.spread_scale, factor * original.spread_scale)
    chain = _chain()
    chain.bid_prices[0] *= factor
    chain.ask_prices[0] *= factor
    fit = diagnose_fit(chain, model_prices=[np.array([0.002, 0.0065, 0.01]) * factor],
                       price_unit="USD per unit", spread_floor=[converted.spread_scale.to_numpy()])
    np.testing.assert_allclose(fit.table("quotes").spread_scaled_error, [1.0, 2.0, -1.0])


def test_missing_ticks_abstain_without_hiding_known_price_bound_failures() -> None:
    inputs = _native_inputs()
    inputs.update(bid_prices=[0.001, 0.0, 0.01], ask_prices=[0.002, 0.001, 1.0],
                  tick_sizes=[np.nan, np.nan, 0.0001])
    quotes = diagnose_native_quotes(**inputs).table("quotes")
    assert quotes.scale_status.tolist() == ["unavailable", "unavailable", "pass"]
    assert quotes.admissibility_status.tolist() == ["unavailable", "fail", "fail"]
    assert quotes.admissibility_reason.iloc[1] == "outside_open_price_bounds"
    assert quotes.admissibility_reason.iloc[2] == "outside_open_price_bounds"
    assert quotes.lower_coefficient.isna().all()
    assert quotes.upper_coefficient.isna().all()
    fit = diagnose_fit(_chain(), model_prices=[[0.002, 0.0065, 0.01]], price_unit="coin per unit",
                       spread_floor=[[np.nan, 0.00025, 0.00005]])
    rows = fit.table("quotes")
    assert rows.price_reason.tolist() == ["valid"] * 3
    assert rows.spread_scaled_error.isna().tolist() == [True, False, False]
    assert rows.scale_reason.iloc[0] == "missing_spread_floor"
    assert rows.price_residual.iloc[0] == pytest.approx(0.0005)


@pytest.mark.parametrize("field,value,reason", [
    ("bid_prices", [0.003, 0.006, 0.01], "crossed_quote"),
    ("bid_prices", [-0.001, 0.006, 0.01], "invalid_quote"),
    ("tick_sizes", [0.0, 0.0005, 0.0001], "nonpositive_tick"),
    ("tick_sizes", [-0.1, 0.0005, 0.0001], "nonpositive_tick"),
])
def test_invalid_native_rows_are_retained_with_reasons(
    field: str, value: list[float], reason: str,
) -> None:
    inputs = _native_inputs()
    inputs[field] = value
    report = diagnose_native_quotes(**inputs)
    rows = report.table("quotes")
    assert len(rows) == 3
    assert rows.scale_reason.iloc[0] == reason
    assert rows.admissibility_status.iloc[0] == "fail"
    assert rows.spread_scale.isna().iloc[0]


def test_normalized_bounds_are_explicit_and_match_intrinsic_and_maximum_payoffs() -> None:
    low, high = normalized_vanilla_price_bounds(
        100.0, np.array([120.0, 80.0, 120.0, 80.0]), np.array(["C", "C", "P", "P"]),
    )
    np.testing.assert_allclose(low, [0.0, 0.2, 0.2, 0.0])
    np.testing.assert_allclose(high, [1.0, 1.0, 1.2, 0.8])
    for forward, strikes, sides in [
        (0.0, [100.0], ["C"]), (100.0, [np.nan], ["C"]),
        (100.0, [100.0], ["X"]), (100.0, [100.0, 110.0], ["C"]),
    ]:
        with pytest.raises(ValueError):
            normalized_vanilla_price_bounds(forward, np.asarray(strikes), np.asarray(sides))


def test_native_report_is_immutable_and_round_trips_nonfinite_evidence() -> None:
    inputs = _native_inputs()
    inputs["tick_sizes"][0] = np.nan
    report = diagnose_native_quotes(**inputs)
    inputs["bid_prices"][0] = 0.8
    table = report.table("quotes")
    table.loc[0, "bid_price"] = 0.9
    assert report.table("quotes").bid_price.iloc[0] == 0.001
    restored = DiagnosticReport.from_dict(json.loads(json.dumps(report.to_dict(), allow_nan=False)))
    assert restored.to_dict() == report.to_dict()


@pytest.mark.parametrize("change", [
    {"price_unit": ""}, {"tick_sizes": [0.001]},
    {"quote_ids": ["x", "x", "z"]}, {"quote_ids": ["x"]},
])
def test_native_alignment_and_explicit_units_are_required(change: dict[str, object]) -> None:
    inputs = _native_inputs()
    inputs.update(change)
    with pytest.raises(ValueError):
        diagnose_native_quotes(**inputs)


@pytest.mark.parametrize("floor", [None, [[0.0, 0.1, 0.1]], [[np.inf, 0.1, 0.1]], [[0.1, 0.1]]])
def test_vector_fit_floors_reject_invalid_configuration(floor: object) -> None:
    with pytest.raises(ValueError):
        diagnose_fit(_chain(), model_prices=[[0.002, 0.0065, 0.01]], price_unit="coin per unit",
                     spread_floor=floor)


def test_missing_quotes_and_invalid_bounds_never_produce_admissible_intervals() -> None:
    inputs = _native_inputs()
    inputs.update(bid_prices=[np.nan, 0.006, 0.01], lower_bounds=[0.0, 1.0, np.nan])
    rows = diagnose_native_quotes(**inputs).table("quotes")
    assert rows.admissibility_status.tolist() == ["unavailable"] * 3
    assert rows.admissibility_reason.tolist() == [
        "missing_or_nonfinite_quote", "invalid_or_missing_price_bounds",
        "invalid_or_missing_price_bounds",
    ]
    assert rows.lower_coefficient.isna().all()


def test_empty_native_inventory_and_unrepresentable_scales_remain_explicit() -> None:
    empty = diagnose_native_quotes(bid_prices=[], ask_prices=[], tick_sizes=[],
                                   lower_bounds=[], upper_bounds=[], price_unit="coin per unit")
    assert empty.table("quotes").empty
    assert empty.table("summary").n_quotes.iloc[0] == 0
    inputs = _native_inputs()
    inputs.update(bid_prices=[0.001] * 3, ask_prices=[0.001] * 3,
                  tick_sizes=[np.nextafter(0.0, 1.0), 1e-320, 0.0001])
    rows = diagnose_native_quotes(**inputs).table("quotes")
    assert rows.admissibility_status.tolist() == ["unavailable", "unavailable", "pass"]
    assert rows.admissibility_reason.tolist()[:2] == [
        "unrepresentable_scale", "unrepresentable_coefficient_interval",
    ]


def test_constant_vector_floor_matches_scalar_fit_results() -> None:
    kwargs = dict(model_prices=[[0.002, 0.0065, 0.01]], price_unit="coin per unit")
    scalar = diagnose_fit(_chain(), spread_floor=0.001, **kwargs)
    scalar_array = diagnose_fit(_chain(), spread_floor=np.array(0.001), **kwargs)
    vector = diagnose_fit(_chain(), spread_floor=[[0.001] * 3], **kwargs)
    for name in ("quotes", "expiries", "summary"):
        assert scalar.table(name).equals(scalar_array.table(name))
        assert scalar.table(name).equals(vector.table(name))
