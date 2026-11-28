"""Independent pathwise checks of calibrated-skew PDE quotes, midlife and settled marks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import brentq

from stochvolmodels.fitters.sparse_local_vol import fit_sparse_local_vol
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pricers.accumulator import quote_accumulator_strike, value_accumulator
from stochvolmodels.products.accumulator import AccumulatorDirection, AccumulatorState
from stochvolmodels.tests.test_accumulator_pde import contract, flat_market
from stochvolmodels.tests.test_sparse_local_vol import chain
from stochvolmodels.valuation import reduce_discounted_mc


@pytest.fixture(scope="module")
def fitted_surface():
    fit = fit_sparse_local_vol(chain([0.25, 0.235, 0.22, 0.212, 0.204]), spot=1.0)
    assert fit.status == "ok"
    return fit.surface


def path_estimator(terms, state, market, *, seed, refinement):
    first = state.next_fixing(terms)
    times = terms.fixing_times[first:] - state.elapsed
    cfg = LSVMonteCarloConfig(paths=65536, seed=seed, antithetic=True)
    blocks = market.iter_path_blocks(
        observation_times=np.r_[0.0, times],
        config=cfg,
        groups_per_block=2000,
        refinement=refinement,
    )
    levels, masks, ids = [], [], []
    for paths in blocks:
        loc = np.searchsorted(paths.observation_times, times)
        np.testing.assert_array_equal(paths.observation_times[loc], times)
        spots = paths.assets[:, loc, 0]
        levels.append(spots)
        alive = np.logical_and.accumulate(~terms.knocked_out(spots), axis=1)
        masks.append(alive)
        ids.append(paths.provenance["independent_group_ids"])
    levels, masks, ids = np.concatenate(levels), np.concatenate(masks), np.concatenate(ids)
    discounts = np.exp(-0.03 * (terms.payment_times - state.elapsed))
    weights = masks * discounts[terms.payment_indices[first:]]
    fixed = float(discounts @ state.fixed_unpaid)
    stream = f"accumulator-seed-{seed}"

    def estimate(k):
        amounts = fixed + np.sum(weights * terms.fixing_cash(levels, k), axis=1)
        return reduce_discounted_mc(amounts, ids, group_size=2, stream_id=stream, keep_groups=True)

    def par():
        k = brentq(lambda x: estimate(x).value, 0.5, 1.6)
        adverse = levels < k * terms.original_spot
        if terms.direction is AccumulatorDirection.DECUMULATOR:
            adverse = levels > k * terms.original_spot
        slope = (terms.notional / len(terms.fixing_times)) * np.mean(
            np.sum(weights * (1.0 + adverse), axis=1)
        )
        return k, estimate(k).standard_error / slope

    return estimate, par


@pytest.mark.slow
@pytest.mark.parametrize(
    "direction,carry",
    [(AccumulatorDirection.ACCUMULATOR, 0.03), (AccumulatorDirection.DECUMULATOR, -0.02)],
)
def test_calibrated_quanto_mc(direction, carry, fitted_surface):
    market = replace(flat_market(carry, foreign=True), surfaces=(fitted_surface,))
    terms = contract(direction, count=63, monthly=True)
    issue = AccumulatorState.at_issue(terms, market.asof)
    args = dict(discount_factors=np.exp(-0.03 * terms.payment_times), discount_currency="USD")
    quote = quote_accumulator_strike(terms, market, **args)
    assert quote.status == "ok"
    k = quote.strike_ratio
    base, _ = path_estimator(terms, issue, market, seed=419, refinement=0)
    coarse = base(k)
    del base, _
    fine, par = path_estimator(terms, issue, market, seed=419, refinement=1)
    fine_at_k = fine(k)
    kmc, kse = par()
    del fine, par
    delta = fine_at_k.paired_difference(coarse)
    assert kse <= 10e-4
    assert abs(k - kmc) <= 3 * kse + 0.5e-4
    assert abs(delta.value) <= 3 * delta.standard_error + terms.notional * 0.25e-4
    independent, _ = path_estimator(terms, issue, market, seed=827, refinement=1)
    check = independent(k)
    del independent, _
    assert check.standard_error <= terms.notional * 5e-4
    assert abs(check.value) <= 3 * check.standard_error + terms.notional * 0.25e-4

    closes = np.linspace(100, 98 if direction is AccumulatorDirection.ACCUMULATOR else 102, 10)
    mid, paid = issue.advance(
        terms, asof="midlife", elapsed=terms.fixing_times[9], closes=closes, strike_ratio=k
    )
    assert paid == 0 and mid.fixed_unpaid[0] != 0
    updated = replace(
        market, asof=mid.asof, forward_levels=market.forward_levels * closes[-1] / 100
    )
    discounts = np.exp(-0.03 * (terms.payment_times[mid.outstanding_payments(terms)] - mid.elapsed))
    mark = value_accumulator(
        terms, mid, updated, strike_ratio=k, discount_factors=discounts, discount_currency="USD"
    )
    assert mark.status == "ok"
    estimator, _ = path_estimator(terms, mid, updated, seed=871, refinement=1)
    mc = estimator(k)
    assert mc.standard_error <= terms.notional * 5e-4
    assert abs(mark.value - mc.value) <= 3 * mc.standard_error + terms.notional * 0.25e-4
    stop, paid = mid.advance(
        terms,
        asof="stopped",
        elapsed=terms.fixing_times[10],
        closes=[terms.barrier],
        strike_ratio=k,
    )
    fixed = value_accumulator(
        terms, stop, None, strike_ratio=k, discount_factors=[0.995], discount_currency="USD"
    )
    assert paid == 0 and fixed.status == "fixed"
    assert fixed.value == pytest.approx(0.995 * mid.fixed_unpaid[0])
    print(
        f"{direction.value} PDE k={k:.7f} MC k={kmc:.7f} "
        f"strike gap/SE (bp)={(k - kmc) * 1e4:.3f}/{kse * 1e4:.3f}; "
        f"issue MC PV/SE (bp)={check.value * 100:.3f}/{check.standard_error * 100:.3f}; "
        f"paired refined-base (bp)={delta.value * 100:.3f}/{delta.standard_error * 100:.3f}; "
        f"midlife PDE-MC/SE (bp)={(mark.value - mc.value) * 100:.3f}/{mc.standard_error * 100:.3f}"
    )
