import numpy as np
import pandas as pd
import pytest
import vanilla_option_pricers as bsm
from numba.typed import List

from stochvolmodels.data.option_chain import FutOptionChain, OptionChain, OptionSlice, SwOptionChain
from stochvolmodels.data import sample_option_chains as samples


def _two_slice_chain() -> OptionChain:
    return OptionChain(
        ttms=np.array([0.5, 1.0]),
        forwards=np.array([1.0, 1.1]),
        strikes_ttms=List(
            [np.array([0.9, 1.1]), np.array([1.0, 1.2])]
        ),
        optiontypes_ttms=List(
            [np.array(["P", "C"]), np.array(["P", "C"])]
        ),
        ids=np.array(["6m", "1y"]),
        discount_rates=np.array([0.02, 0.03]),
        bid_ivs=List(
            [np.array([0.18, 0.38]), np.array([0.24, 0.34])]
        ),
        ask_ivs=List(
            [np.array([0.22, 0.42]), np.array([0.26, 0.36])]
        ),
    )


def test_option_slice_and_chain_convert_discount_conventions() -> None:
    option_slice = OptionSlice(
        ttm=2.0,
        forward=1.0,
        strikes=np.array([0.9, 1.1]),
        optiontypes=np.array(["P", "C"]),
        id="2y",
        discfactor=np.exp(-0.04),
    )
    np.testing.assert_allclose(option_slice.discount_rate, 0.02, atol=1.0e-14)

    chain = _two_slice_chain()
    np.testing.assert_allclose(
        chain.discfactors,
        np.exp(-chain.discount_rates * chain.ttms),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_option_chain_dimensions_mid_vols_and_atm_interpolation() -> None:
    chain = _two_slice_chain()

    assert len(chain.ttms) == len(chain.forwards) == len(chain.ids) == 2
    assert [len(strikes) for strikes in chain.strikes_ttms] == [2, 2]
    assert [len(types) for types in chain.optiontypes_ttms] == [2, 2]
    mid_vols = chain.get_mid_vols()
    np.testing.assert_allclose(mid_vols[0], np.array([0.2, 0.4]), atol=1.0e-14)
    np.testing.assert_allclose(mid_vols[1], np.array([0.25, 0.35]), atol=1.0e-14)
    np.testing.assert_allclose(chain.get_chain_atm_vols(), np.array([0.3, 0.3]), atol=1.0e-14)


def test_option_chain_skew_interpolates_put_and_call_wings_separately() -> None:
    chain = OptionChain(
        ttms=np.array([0.25]),
        forwards=np.array([100.0]),
        strikes_ttms=List([np.array([80.0, 90.0, 100.0, 110.0, 120.0])]),
        optiontypes_ttms=List([np.array(["P", "P", "C", "C", "C"])]),
        ids=np.array(["3m"]),
        bid_ivs=List([np.array([0.34, 0.29, 0.24, 0.21, 0.19])]),
        ask_ivs=List([np.array([0.36, 0.31, 0.26, 0.23, 0.21])]),
    )
    deltas = chain.get_chain_deltas()[0]
    mid_vols = chain.get_mid_vols()[0]
    put_mask = chain.optiontypes_ttms[0] == "P"
    call_mask = chain.optiontypes_ttms[0] == "C"
    put_order = np.argsort(deltas[put_mask])
    call_order = np.argsort(deltas[call_mask])
    put_vol = np.interp(
        -0.25,
        deltas[put_mask][put_order],
        mid_vols[put_mask][put_order],
    )
    call_vol = np.interp(
        0.25,
        deltas[call_mask][call_order],
        mid_vols[call_mask][call_order],
    )
    expected = (put_vol - call_vol) / chain.get_chain_atm_vols()[0]

    np.testing.assert_allclose(chain.get_chain_skews(delta=0.25), [expected], atol=1.0e-14)


def test_option_chain_get_slice_preserves_aligned_data() -> None:
    chain = _two_slice_chain()
    option_slice = chain.get_slice("1y")

    assert option_slice.id == "1y"
    assert option_slice.ttm == 1.0
    assert option_slice.forward == 1.1
    np.testing.assert_array_equal(option_slice.strikes, np.array([1.0, 1.2]))
    np.testing.assert_array_equal(option_slice.optiontypes, np.array(["P", "C"]))
    np.testing.assert_allclose(option_slice.bid_ivs, np.array([0.24, 0.34]))
    np.testing.assert_allclose(option_slice.ask_ivs, np.array([0.26, 0.36]))
    np.testing.assert_allclose(option_slice.discfactor, np.exp(-0.03))
    np.testing.assert_allclose(option_slice.discount_rate, 0.03)


def test_uniform_chain_broadcasts_default_forward_to_requested_maturities() -> None:
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([0.25]),
        ids=np.array(["3m"]),
        strikes=np.array([0.9, 1.0, 1.1]),
    )

    np.testing.assert_array_equal(chain.forwards, np.array([1.0]))
    assert len(chain.strikes_ttms) == len(chain.optiontypes_ttms) == 1
    np.testing.assert_array_equal(chain.optiontypes_ttms[0], np.array(["P", "C", "C"]))


def test_option_slice_rejects_misaligned_strikes_and_types() -> None:
    with pytest.raises(ValueError, match="strikes.*optiontypes|optiontypes.*strikes"):
        OptionSlice(
            ttm=1.0,
            forward=1.0,
            strikes=np.array([0.9, 1.0, 1.1]),
            optiontypes=np.array(["P", "C"]),
            id="bad",
        )


def test_option_chain_rejects_misaligned_slice_dimensions() -> None:
    with pytest.raises(ValueError, match="ttms.*forwards|forwards.*ttms"):
        OptionChain(
            ttms=np.array([0.5, 1.0]),
            forwards=np.array([1.0]),
            strikes_ttms=List([np.array([0.9, 1.1]), np.array([0.9, 1.1])]),
            optiontypes_ttms=List(
                [np.array(["P", "C"]), np.array(["P", "C"])]
            ),
            ids=np.array(["6m", "1y"]),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ttm", 0.0, "ttm.*positive"),
        ("forward", np.nan, "forward.*finite"),
        ("strikes", np.array([0.0, 1.0]), "strikes.*positive"),
        ("optiontypes", np.array(["P", "BAD"]), "optiontypes"),
    ],
)
def test_option_slice_rejects_invalid_public_inputs(
    field: str, value, message: str
) -> None:
    kwargs = dict(
        ttm=1.0,
        forward=1.0,
        strikes=np.array([0.9, 1.1]),
        optiontypes=np.array(["P", "C"]),
        id="invalid",
    )
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        OptionSlice(**kwargs)


def test_option_chain_rejects_unsorted_maturities_and_crossed_quotes() -> None:
    with pytest.raises(ValueError, match="ttms.*increasing"):
        OptionChain(
            ttms=np.array([1.0, 0.5]),
            forwards=np.array([1.0, 1.0]),
            strikes_ttms=List([np.array([0.9]), np.array([0.9])]),
            optiontypes_ttms=List([np.array(["P"]), np.array(["P"])]),
            ids=np.array(["1y", "6m"]),
        )

    with pytest.raises(ValueError, match="bid_ivs.*ask_ivs"):
        OptionChain(
            ttms=np.array([0.5]),
            forwards=np.array([1.0]),
            strikes_ttms=List([np.array([0.9, 1.1])]),
            optiontypes_ttms=List([np.array(["P", "C"])]),
            ids=np.array(["6m"]),
            bid_ivs=List([np.array([0.3, 0.2])]),
            ask_ivs=List([np.array([0.2, 0.3])]),
        )


def test_option_chain_price_ivol_round_trip_and_calibration_view() -> None:
    """Chain calibration inputs preserve ragged slices and invert delegated BSM prices."""
    chain = _two_slice_chain()
    x, target = chain.get_chain_data_as_xy()
    assert x[0] is chain.ttms
    assert x[3] is chain.strikes_ttms
    assert len(target) == 2

    prices = List(
        bsm.compute_bsm_vanilla_slice_prices(
            ttm=ttm,
            forward=forward,
            strikes=strikes,
            vols=vols,
            optiontypes=optiontypes,
            discfactor=discfactor,
        )
        for ttm, forward, strikes, vols, optiontypes, discfactor in zip(
            chain.ttms,
            chain.forwards,
            chain.strikes_ttms,
            target,
            chain.optiontypes_ttms,
            chain.discfactors,
        )
    )
    recovered = chain.compute_model_ivols_from_chain_data(prices)
    for actual, expected in zip(recovered, target):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-10)


def test_option_chain_normalization_uniform_grid_and_subsets_preserve_conventions() -> None:
    """Display transforms and maturity subsets retain all aligned market conventions."""
    chain = _two_slice_chain()
    normalized = OptionChain.to_forward_normalised_strikes(chain)
    np.testing.assert_allclose(normalized.forwards, 1.0)
    np.testing.assert_allclose(normalized.forwards0, chain.forwards)
    for strikes, original, forward in zip(
        normalized.strikes_ttms, chain.strikes_ttms, chain.forwards
    ):
        np.testing.assert_allclose(strikes, original / forward)

    uniform = OptionChain.to_uniform_strikes(chain, num_strikes=5)
    assert uniform.bid_ivs is None and uniform.ask_ivs is None
    for strikes, optiontypes, forward in zip(
        uniform.strikes_ttms, uniform.optiontypes_ttms, uniform.forwards
    ):
        assert len(strikes) == 5
        np.testing.assert_array_equal(optiontypes, np.where(strikes >= forward, "C", "P"))

    one = OptionChain.get_slices_as_chain(chain, ids=["1y"])
    two = OptionChain.get_slices_as_chain(chain, ids=["6m", "1y"])
    assert list(one.ids) == ["1y"]
    np.testing.assert_allclose(one.ttms, [1.0])
    assert list(two.ids) == chain.ids.tolist()
    np.testing.assert_allclose(two.discfactors, chain.discfactors)


def test_varswap_replication_floor_is_explicit() -> None:
    """The sparse-grid variance strike floor never falls below the chain ATM volatility."""
    chain = _two_slice_chain()
    raw = chain.get_slice_varswap_strikes(floor_with_atm_vols=False)
    floored = chain.get_slice_varswap_strikes(floor_with_atm_vols=True)
    assert np.all(np.isfinite(raw)) and np.all(raw > 0.0)
    assert np.all(floored.to_numpy() >= chain.get_chain_atm_vols())


@pytest.mark.parametrize(
    "constructor",
    [
        samples.get_oca_simulated_chain_data,
        samples.get_btc_test_chain_data,
        samples.get_vix_test_chain_data,
        samples.get_gld_test_chain_data_6m,
        samples.get_gld_test_chain_data,
        samples.get_sqqq_test_chain_data,
        samples.get_spy_test_chain_data,
        samples.get_qv_options_test_chain_data,
    ],
)
def test_all_bundled_sample_chains_are_deterministic_and_aligned(constructor) -> None:
    """Every bundled illustration is provider-free, deterministic, and convention-valid."""
    first = constructor()
    second = constructor()
    assert len(first.ttms) == len(first.forwards) == len(first.ids)
    assert np.all(np.diff(first.ttms) > 0.0)
    assert np.all(first.discfactors > 0.0)
    np.testing.assert_array_equal(first.ttms, second.ttms)
    np.testing.assert_array_equal(first.forwards, second.forwards)
    for left, right, optiontypes in zip(
        first.strikes_ttms, second.strikes_ttms, first.optiontypes_ttms
    ):
        np.testing.assert_array_equal(left, right)
        assert len(left) == len(optiontypes)
        assert set(optiontypes) <= {"C", "P", "IC", "IP"}


def _swaption_chain() -> SwOptionChain:
    ttms = np.array([0.5, 1.0])
    tenors = np.array([2.0, 5.0])
    forwards = [np.array([0.02, 0.025]), np.array([0.03, 0.035])]
    strikes = [
        [forward + np.linspace(-0.01, 0.01, 5) for forward in tenor_forwards]
        for tenor_forwards in forwards
    ]
    bid = [
        [0.2 + 0.01 * np.arange(5) for _ in ttms]
        for _ in tenors
    ]
    ask = [
        [vols + 0.02 for vols in tenor_vols]
        for tenor_vols in bid
    ]
    return SwOptionChain(
        ccy="USD",
        ttms=ttms,
        tenors=tenors,
        ttms_ids=np.array(["6m", "1y"]),
        tenors_ids=["2y", "5y"],
        forwards=forwards,
        strikes_ttms=strikes,
        bid_ivs=bid,
        ask_ivs=ask,
        ticker="synthetic",
    )


def test_swaption_chain_reductions_atm_vegas_and_delta_remaps() -> None:
    chain = _swaption_chain()
    mid = chain.get_mid_vols()
    atm = chain.get_chain_atm_vols()
    vegas = chain.get_chain_vegas(is_unit_ttm_vega=True)

    assert len(mid) == len(atm) == len(vegas) == 2
    for tenor_mid, tenor_atm, tenor_vegas in zip(mid, atm, vegas):
        assert len(tenor_mid) == len(tenor_atm) == len(tenor_vegas) == 2
        np.testing.assert_allclose(tenor_atm, np.array([0.23, 0.23]))
        assert all(np.all(np.asarray(slice_vega) > 0.0) for slice_vega in tenor_vegas)

    reduced_strikes = chain.reduce_strikes(nb_otms=1)
    reduced_ttms = chain.reduce_ttms(["1y"])
    reduced_tenors = chain.reduce_tenors(["5y"])
    sliced = SwOptionChain.get_slices_as_chain(chain, ids=["6m"])
    assert reduced_strikes.strikes_ttms[0][0].size == 3
    np.testing.assert_array_equal(reduced_ttms.ttms, np.array([1.0]))
    assert reduced_tenors.tenors_ids == ["5y"]
    np.testing.assert_array_equal(sliced.ttms, np.array([0.5]))

    with pytest.raises(ValueError, match="number of strikes"):
        chain.reduce_strikes(nb_otms=3)
    with pytest.raises(ValueError, match="Expiries"):
        chain.reduce_ttms(["2y"])
    with pytest.raises(ValueError, match="Tenors"):
        chain.reduce_tenors(["10y"])

    indexed = SwOptionChain.remap_to_inc_delta(pd.Series([0.2, 0.3], index=[-0.25, 0.25]))
    np.testing.assert_allclose(indexed.index.to_numpy(), np.array([0.25, -0.25]))
    np.testing.assert_allclose(
        SwOptionChain.remap_to_pc_delta(np.array([-0.75, -0.25])),
        np.array([-0.25, 0.25]),
    )


def _futures_chain(with_open_interest: bool = True) -> FutOptionChain:
    strikes = List([np.linspace(90.0, 110.0, 5), np.linspace(95.0, 115.0, 5)])
    calls = List([np.linspace(0.25, 0.21, 5), np.linspace(0.26, 0.22, 5)])
    puts = List([np.linspace(0.27, 0.23, 5), np.linspace(0.28, 0.24, 5)])
    call_oi = List([np.array([10, 20, 30, 40, 50]), np.array([15, 25, 35, 45, 55])])
    put_oi = List([np.array([5, 10, 15, 20, 25]), np.array([6, 11, 16, 21, 26])])
    return FutOptionChain(
        ccy="USD",
        ttms=np.array([0.25, 0.5]),
        forwards=np.array([100.0, 105.0]),
        strikes_ttms=strikes,
        ttms_ids=np.array(["3m", "6m"]),
        ivs_call_ttms=calls,
        ivs_put_ttms=puts,
        call_oi=call_oi if with_open_interest else None,
        put_oi=put_oi if with_open_interest else None,
        ticker="synthetic",
    )


def test_futures_chain_open_interest_filter_and_maturity_reduction() -> None:
    chain = _futures_chain(with_open_interest=True)
    filtered = chain.filter_by_oi(max_strikes=2, include_atm=False)
    assert all(strikes.size == 2 for strikes in filtered.strikes_ttms)
    assert len(filtered.get_mid_vols()) == 2
    with pytest.raises(ValueError, match="atm strike"):
        chain.filter_by_oi(max_strikes=1, include_atm=True)

    no_oi = _futures_chain(with_open_interest=False)
    reduced = no_oi.reduce_ttms(["6m"])
    np.testing.assert_array_equal(reduced.ttms, np.array([0.5]))
    np.testing.assert_array_equal(reduced.forwards, np.array([105.0]))
    with pytest.raises(NotImplementedError, match="open interest"):
        no_oi.filter_by_oi(max_strikes=2, include_atm=False)
    with pytest.raises(ValueError, match="Expiries"):
        no_oi.reduce_ttms(["1y"])
