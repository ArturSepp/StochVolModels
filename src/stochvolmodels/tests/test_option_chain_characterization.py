import numpy as np
import pytest
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain, OptionSlice


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
