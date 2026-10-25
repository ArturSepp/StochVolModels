"""Price and volatility convention round trips through the market-data containers."""

from __future__ import annotations

import numpy as np
import pytest
import vanilla_option_pricers as vanilla

from stochvolmodels.data.option_chain import FutOptionChain, OptionChain, SwOptionChain


def _normal_cube() -> SwOptionChain:
    """Build unequal tenor/expiry axes, including negative rates, in absolute normal vol."""
    ttms = np.array([0.25, 0.5, 1.0])
    forwards = [np.array([-0.005, 0.0, 0.01]), np.array([0.02, 0.025, 0.03])]
    strikes = [[forward + np.array([-0.005, 0.0, 0.005]) for forward in row]
               for row in forwards]
    vols = [[np.full(3, 0.008 + 0.001 * tenor + 0.0005 * expiry)
             for expiry in range(3)] for tenor in range(2)]
    return SwOptionChain(
        ccy="USD", ttms=ttms, tenors=np.array([2.0, 5.0]),
        ttms_ids=np.array(["3m", "6m", "1y"]), tenors_ids=["2y", "5y"],
        forwards=forwards, strikes_ttms=strikes, bid_ivs=vols, ask_ivs=vols,
    )


def _normal_cube_prices(chain: SwOptionChain) -> list[list[np.ndarray]]:
    """Price calls in units of their annuity, matching the rates pricer convention."""
    return [[vanilla.compute_normal_slice_prices(
        ttm=ttm, forward=chain.forwards[i][j], strikes=chain.strikes_ttms[i][j],
        vols=chain.bid_ivs[i][j], optiontypes=chain.optiontypes_ttms[j], discfactor=1.0,
    ) for j, ttm in enumerate(chain.ttms)] for i in range(len(chain.tenors))]


def test_swaption_prices_recover_normal_vols_by_tenor_then_expiry() -> None:
    chain = _normal_cube()
    recovered = chain.compute_model_ivols_from_chain_data(_normal_cube_prices(chain))
    assert len(recovered) == 2
    for actual_tenor, expected_tenor in zip(recovered, chain.bid_ivs):
        assert len(actual_tenor) == 3
        for actual, expected in zip(actual_tenor, expected_tenor):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)


@pytest.mark.parametrize("axis", ["tenor", "expiry", "strike"])
def test_swaption_inversion_rejects_misaligned_price_axes(axis: str) -> None:
    chain = _normal_cube()
    prices = _normal_cube_prices(chain)
    if axis == "tenor":
        prices.pop()
    elif axis == "expiry":
        prices[0].pop()
    else:
        prices[0][0] = prices[0][0][:-1]
    with pytest.raises(ValueError, match="model_prices"):
        chain.compute_model_ivols_from_chain_data(prices)


def test_forward_normalization_preserves_price_quotes_and_black_vols() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    optiontypes = np.array(["P", "C", "C"])
    bid_vols, ask_vols = np.full(3, 0.19), np.full(3, 0.21)
    price_kwargs = dict(ttm=0.5, forward=100.0, strikes=strikes,
                        optiontypes=optiontypes, discfactor=0.98)
    bid = vanilla.compute_bsm_vanilla_slice_prices(vols=bid_vols, **price_kwargs)
    ask = vanilla.compute_bsm_vanilla_slice_prices(vols=ask_vols, **price_kwargs)
    chain = OptionChain(
        ttms=np.array([0.5]), forwards=np.array([100.0]), strikes_ttms=(strikes,),
        optiontypes_ttms=(optiontypes,), ids=np.array(["6m"]), discfactors=np.array([0.98]),
        bid_ivs=[bid_vols], ask_ivs=[ask_vols], bid_prices=[bid], ask_prices=[ask],
    )
    normalized = OptionChain.to_forward_normalised_strikes(chain)
    assert normalized.bid_prices is not None
    assert normalized.ask_prices is not None
    np.testing.assert_allclose(normalized.bid_prices[0], bid / 100.0)
    np.testing.assert_allclose(normalized.ask_prices[0], ask / 100.0)
    recovered = normalized.compute_model_ivols_from_chain_data(normalized.bid_prices)
    np.testing.assert_allclose(recovered[0], bid_vols, rtol=0.0, atol=2e-10)
    np.testing.assert_array_equal(chain.bid_prices[0], bid)
    np.testing.assert_array_equal(chain.ask_prices[0], ask)


def test_futures_vegas_preserve_all_expiries_with_ragged_strike_slices() -> None:
    """A one-strike first expiry must not truncate later expiry/strike arrays."""
    ttms = np.array([0.25, 0.5, 1.0])
    forwards = np.array([0.02, 0.025, 0.03])
    strikes = tuple(forward + 0.002 * np.arange(index + 1)
                    for index, forward in enumerate(forwards))
    vols = tuple(np.full_like(slice_strikes, 0.01) for slice_strikes in strikes)
    chain = FutOptionChain(
        ccy="USD", ttms=ttms, forwards=forwards, strikes_ttms=strikes,
        ttms_ids=np.array(["3m", "6m", "1y"]), ivs_call_ttms=vols, ivs_put_ttms=vols,
    )
    vegas = chain.get_chain_vegas()
    assert len(vegas) == len(ttms)
    for index, actual in enumerate(vegas):
        d = (forwards[index] - strikes[index]) / (vols[index] * np.sqrt(ttms[index]))
        expected = np.exp(-0.5 * d**2) * np.sqrt(ttms[index] / (2.0 * np.pi))
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
