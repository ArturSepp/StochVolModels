"""Independent single-name reference, coupled monitoring and contract state tests."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig, LSVMonteCarloEngine
from stochvolmodels.pricers._barrier_monitoring import barrier_hit_weights, bridge_inputs
from stochvolmodels.pricers.barrier_reverse_convertible import (
    BrcConfig,
    quote_brc_coupon,
    value_brc,
)
from stochvolmodels.products.barrier_reverse_convertible import (
    BrcTerms,
    BrcState,
    TouchState,
    Monitoring,
)
from stochvolmodels.tests.test_correlated_local_vol import market
from stochvolmodels.tests.test_lsv_mc import _constant


def note(m, barrier=0.7):
    return BrcTerms(
        asset_ids=m.asset_ids,
        fixings=m.spots,
        barriers=m.spots * barrier,
        fixing_time=1.0,
        payment_time=1.0,
        currency="USD",
        face=100.0,
    )


def quote(t, m, cfg):
    return quote_brc_coupon(t, m, discount_factor=0.96, discount_currency="USD", config=cfg)


def reference_loss():
    # Integrate the reflected drifted Brownian density for surviving / crossed paths.
    b, sigma = np.log(0.7), 0.25
    mu = -(sigma**2) / 2

    def integrand(x):
        density = (
            norm.pdf(x, mu, sigma)
            if x <= b
            else np.exp(2 * mu * b / sigma**2) * norm.pdf(x, 2 * b + mu, sigma)
        )
        return -np.expm1(x) * density

    return quad(integrand, -np.inf, b)[0] + quad(integrand, b, 0.0)[0]


def test_daily_bridge_against_reflection_and_paired_half_days():
    model = market()
    terms = replace(note(model), asset_ids=("A",), fixings=[100.0], barriers=[70.0])
    cfg = BrcConfig(
        mc=LSVMonteCarloConfig(paths=32768, steps_per_year=252, seed=219),
        groups_per_block=512,
        keep_groups=True,
    )
    coarse = quote(terms, model, cfg)
    fine = quote(terms, model, replace(cfg, initial_refinement=1))
    ref = reference_loss()
    assert ref == pytest.approx(0.054982, abs=1e-6)
    assert abs(coarse.valuation.expected_loss - ref) < 4 * coarse.standard_error
    diff = coarse.valuation.loss_groups.paired_difference(fine.valuation.loss_groups)
    assert 1e4 * (abs(diff.value) + 1.96 * diff.standard_error) < 5
    maturity = quote(
        replace(terms, monitoring=Monitoring.MATURITY),
        model,
        replace(cfg, mc=replace(cfg.mc, paths=4096)),
    )
    assert maturity.coupon < coarse.coupon - 0.01
    assert coarse.coupon_bounds == pytest.approx((coarse.coupon, coarse.coupon))


def test_native_lsv_scalar_bridge_and_eligibility():
    cfg = LSVMonteCarloConfig(paths=8192, steps_per_year=16, seed=31)
    engine = LSVMonteCarloEngine(LSVModel.from_params(_constant(sigma0=0.25, theta=0.25)), cfg)
    paths = engine.simulate_paths(observation_times=np.linspace(0.0, 1.0, 17), spot0=100.0)
    with pytest.raises(ValueError, match="verified"):
        bridge_inputs(paths)
    paths = replace(paths, provenance={**paths.provenance, "frozen_log_diffusion": True})
    levels, var = bridge_inputs(paths)
    hit = barrier_hit_weights(levels, var, [0], [70.0], np.ones((16, 1), bool))
    loss = hit[:, 1] * np.maximum(1 - levels[:, -1, 0] / 100, 0)
    pairs = loss.reshape(2, -1).mean(axis=0)
    assert abs(loss.mean() - reference_loss()) < 4 * pairs.std(ddof=1) / np.sqrt(len(pairs))
    with pytest.raises(ValueError, match="units"):
        bridge_inputs(
            replace(paths, state_units={**paths.state_units, "quadratic_variance": "variance rate"})
        )
    with pytest.raises(ValueError, match="scheme"):
        bridge_inputs(replace(paths, scheme="jump"))
    fixed = np.array([[[100.0], [70.0]], [[100.0], [80.0]]])
    np.testing.assert_array_equal(
        barrier_hit_weights(fixed, np.zeros_like(fixed), [0], [70.0], [[True]]),
        [[1, 1, 1], [0, 0, 0]],
    )
    assert not barrier_hit_weights(fixed, np.zeros_like(fixed), [0], [70.0], [[False]]).any()


def test_touch_cutoff_payment_and_currency():
    m = market()
    t = note(m)

    def mark(terms, state, **kw):
        return value_brc(
            terms,
            state,
            m,
            coupon=0.08,
            discount_factor=0.96,
            discount_currency=kw.get("currency", "USD"),
        )

    st = BrcState(asof=m.asof)
    assert mark(t, st).status == "unresolved_history"
    with pytest.raises(ValueError, match="currency"):
        mark(t, st, currency="EUR")
    fixed = replace(t, fixing_time=0.0, payment_time=0.1)
    known = replace(st, final_levels=[90.0, 204.0], touch=TouchState.HIT)

    def without_model(terms, state):
        return value_brc(
            terms, state, None, coupon=0.08, discount_factor=0.96, discount_currency="USD"
        )

    assert without_model(fixed, known) == mark(fixed, known)
    assert (
        without_model(fixed, replace(known, touch=TouchState.UNRESOLVED)).status
        == "unresolved_history"
    )
    paid = without_model(replace(fixed, payment_time=0.0), replace(known, paid=True))
    assert paid.status == "paid" and paid.value == 0.0 and paid.asof == known.asof
    with pytest.raises(ValueError, match="live stochastic"):
        without_model(t, replace(st, touch=TouchState.CLEAR))
    assert mark(fixed, known).value == pytest.approx(0.96 * 98)
    assert mark(fixed, replace(known, final_levels=[101.0, 202.0])).value == pytest.approx(
        0.96 * 108
    )
    assert mark(
        fixed, replace(known, final_levels=[70.0, 108.0], touch=TouchState.UNRESOLVED)
    ).value == pytest.approx(0.96 * 62.0)
    assert mark(replace(fixed, payment_time=0.0), replace(known, paid=True)).value == 0.0
    with pytest.raises(ValueError, match="as-of"):
        mark(t, replace(st, asof="yesterday"))
