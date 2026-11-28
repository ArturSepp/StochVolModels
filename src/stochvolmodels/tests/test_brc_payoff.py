"""Contract cashflows verified by hand on four named-asset paths."""

from dataclasses import replace

import numpy as np
import pytest

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.products.barrier_reverse_convertible import BrcPayoff, BrcTerms, Monitoring
from stochvolmodels.valuation import value_paths


def test_four_cashflows_and_group_error():
    terms = BrcTerms(
        asset_ids=("A", "B"),
        fixings=[100, 200],
        barriers=[70, 140],
        fixing_time=1.0,
        payment_time=1.01,
        currency="USD",
        face=100.0,
    )
    levels = np.array(
        [
            [[200, 100], [190, 70], [160, 105]],
            [[200, 100], [180, 95], [140, 90]],
            [[200, 100], [120, 80], [208, 105]],
            [[200, 100], [182, 85], [180, 80]],
        ],
        float,
    )
    weights = np.maximum.accumulate(np.any(levels <= [140, 70], axis=2), axis=1).astype(float)
    paths = ModelPaths(
        observation_times=np.array([0.0, 0.5, 1.0]),
        assets=levels,
        asset_ids=("B", "A"),
        sampling_measure="Q_MMA",
        target_measure="Q_MMA",
        numeraire="money_market_account",
        scheme="stipulated",
        states={"knock_in_weight": weights},
        state_units={"knock_in_weight": "probability"},
    )
    payoff = BrcPayoff(terms, 0.08)
    np.testing.assert_allclose(payoff(paths), [88, 78, 108, 108])
    result = value_paths(
        paths=paths,
        payoff=payoff,
        discount_factor=0.96,
        independent_group_ids=np.array([0, 1, 0, 1]),
    )
    assert result.value == pytest.approx(91.68)
    assert result.standard_error == pytest.approx(2.4)
    np.testing.assert_allclose(
        BrcPayoff(replace(terms, monitoring=Monitoring.MATURITY), 0.08)(paths), [108, 78, 108, 108]
    )
    with pytest.raises(ValueError, match="knock_in_weight"):
        payoff(replace(paths, states={}, state_units={}))
    with pytest.raises(ValueError):
        replace(terms, payment_time=0.9)
