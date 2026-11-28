"""Hand-settled monthly cash, inclusive barriers, and immutable post-close state."""

from dataclasses import replace

import numpy as np
import pytest

from stochvolmodels.products.accumulator import (
    AccumulatorDirection,
    AccumulatorTerms,
    AccumulatorState,
)
from stochvolmodels.pricers.accumulator import value_accumulator


@pytest.mark.parametrize(
    "direction,closes,barrier",
    [
        (AccumulatorDirection.ACCUMULATOR, [90.0, 105.0, 75.0], 105.0),
        (AccumulatorDirection.DECUMULATOR, [110.0, 95.0, 125.0], 95.0),
    ],
)
def test_earned_cash_survives_touch(direction, closes, barrier):
    terms = AccumulatorTerms(
        asset_id="X",
        currency="USD",
        notional=300,
        original_spot=100,
        direction=direction,
        barrier=barrier,
        fixing_times=[0.01, 0.02, 0.03],
        payment_indices=[0, 0, 1],
        payment_times=[0.025, 0.04],
    )
    st = AccumulatorState.at_issue(terms, "issue")
    assert terms.fixing_cash(100, 1) == 0
    stopped, paid = st.advance(terms, elapsed=0.02, asof="stop", closes=closes[:2], strike_ratio=1)
    assert stopped.stopped and paid == 0
    np.testing.assert_array_equal(stopped.outstanding_payments(terms), [0])
    args = dict(strike_ratio=1.0, discount_currency="USD")
    mark = value_accumulator(terms, stopped, None, discount_factors=[0.99], **args)
    assert mark.value == pytest.approx(-19.8)
    assert mark.status == "fixed" and mark.future_fixings_pv == 0
    replay, cash = stopped.advance(terms, elapsed=0.02, asof="stop", closes=[], strike_ratio=1)
    assert cash == 0 and replay.fixed_unpaid[0] == stopped.fixed_unpaid[0]
    final, paid = stopped.advance(terms, elapsed=0.04, asof="end", closes=[np.nan], strike_ratio=1)
    assert final.history_complete and paid == pytest.approx(-20)
    assert value_accumulator(terms, final, None, discount_factors=[], **args).status == "settled"
    first, _ = st.advance(terms, elapsed=0.01, asof="first", closes=[barrier], strike_ratio=1)
    assert first.stopped and not first.fixed_unpaid.any()
    missing, _ = st.advance(
        terms, elapsed=0.02, asof="partial", closes=[np.nan, barrier], strike_ratio=1
    )
    assert missing.stopped and not missing.history_complete
    assert (
        value_accumulator(terms, missing, None, discount_factors=[], **args).status
        == "unresolved_history"
    )
    with pytest.raises(ValueError, match="live stochastic"):
        value_accumulator(terms, st, None, discount_factors=[1, 1], **args)
    with pytest.raises(ValueError, match="precedes"):
        replace(terms, payment_times=[0.012, 0.03])
    with pytest.raises(ValueError, match="map"):
        replace(terms, payment_indices=[0, 1, 0])
    with pytest.raises(ValueError, match="distinct"):
        replace(terms, fixing_times=[0.01, 0.01 + 1e-14, 0.03])
    with pytest.raises(ValueError):
        terms.fixing_times[0] = 0
    same_close = replace(terms, payment_times=[0.02 - 1e-14, 0.04])
    assert same_close.payment_times[0] == same_close.fixing_times[1]

    # Three surviving closes, two settlements, last observed level exactly at strike.
    mid, cash = st.advance(
        terms, elapsed=0.025, asof="month1", closes=[closes[0], 100], strike_ratio=1
    )
    assert cash == pytest.approx(-20) and not mid.stopped
    final, cash = mid.advance(terms, elapsed=0.04, asof="month2", closes=[100], strike_ratio=1)
    assert cash == 0
    assert value_accumulator(terms, final, None, discount_factors=[], **args).value == 0
