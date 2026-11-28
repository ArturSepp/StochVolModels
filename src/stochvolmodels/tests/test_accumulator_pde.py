"""Accumulator PDE tests against Gaussian-transition quadrature and vanilla expectations."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr

from stochvolmodels.models.basket import CorrelatedLocalVolModel
from stochvolmodels.models.lsv.local_vol import LocalVolSurface
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.accumulator import (
    AccumulatorPdeConfig,
    quote_accumulator_strike,
    value_accumulator,
)
from stochvolmodels.products.accumulator import (
    AccumulatorDirection,
    AccumulatorState,
    AccumulatorTerms,
)


def flat_market(carry, spot=100.0, foreign=False):
    surface = LocalVolSurface(
        np.array([0.5]), Grid1D(np.array([-1.0, 0.0, 1.0])), np.full((1, 3), 0.20)
    )
    return CorrelatedLocalVolModel(
        asof="today",
        asset_ids=("X",),
        currencies=("EUR" if foreign else "USD",),
        settlement_currency="USD",
        surfaces=(surface,),
        curve_times=[0, 1],
        forward_levels=[[spot], [spot * np.exp(carry)]],
        correlation=[[1]],
        fx_vols=[0.14 if foreign else 0],
        equity_fx_correlations=[0.45 if foreign else 0],
    )


def contract(direction, count=252, monthly=False):
    t = np.arange(1, count + 1) / 252
    idx = np.arange(count) // (21 if monthly else 1)
    payments = t[np.r_[idx[:-1] != idx[1:], True]]
    return AccumulatorTerms(
        asset_id="X",
        currency="USD",
        direction=direction,
        original_spot=100,
        barrier=105 if direction is AccumulatorDirection.ACCUMULATOR else 95,
        notional=100,
        fixing_times=t,
        payment_times=payments,
        payment_indices=idx,
    )


# Independent integration of daily log-spot transitions on a 6,401-node
# barrier-aligned surviving half-line, with trapezoidal endpoint weights.
# Halving spacing from 3,201 nodes moved all reference strikes by <0.009 bp.
REFERENCE = (
    (0.00, 0.8837736, 0.8837683, 1.1300013, 1.1300078),
    (0.03, 0.8916759, 0.8916707, 1.1412353, 1.1412421),
    (-0.02, 0.8783664, 0.8783610, 1.1228492, 1.1228556),
)


@pytest.mark.slow
@pytest.mark.parametrize("row", REFERENCE)
@pytest.mark.parametrize("monthly", (False, True))
@pytest.mark.parametrize("direction", list(AccumulatorDirection))
def test_daily_gaussian_barrier_reference(row, monthly, direction):
    market = flat_market(row[0])
    terms = contract(direction, monthly=monthly)
    k = row[1 + int(monthly) + 2 * (direction is AccumulatorDirection.DECUMULATOR)]
    args = dict(discount_currency="USD", discount_factors=np.exp(-0.03 * terms.payment_times))
    quote = quote_accumulator_strike(terms, market, **args)
    mark = value_accumulator(
        terms, AccumulatorState.at_issue(terms, market.asof), market, strike_ratio=k, **args
    )
    assert quote.status == mark.status == "ok"
    assert abs(mark.value) * 1e4 / terms.notional <= 0.25
    assert abs(quote.strike_ratio - k) * 1e4 <= 0.5
    assert abs(quote.par_residual) * 1e4 / terms.notional <= 0.1
    assert max(np.diff(quote.diagnostics.survival)) <= 1e-9
    print(
        f"{direction.value} carry={row[0]} monthly={monthly}: "
        f"strike_bp={(quote.strike_ratio - k) * 1e4:.4f}, pv_bp={mark.value * 100:.4f}"
    )


@pytest.mark.parametrize("direction", list(AccumulatorDirection))
def test_no_barrier_vanilla_decomposition(direction):
    market = flat_market(-0.02, foreign=True)
    upper = direction is AccumulatorDirection.ACCUMULATOR
    terms = replace(contract(direction, count=15), barrier=1e5 if upper else 0.001)
    t = terms.fixing_times
    f = np.exp((-0.02 - 0.2 * 0.14 * 0.45) * t)
    k = 1.005
    sd = 0.2 * np.sqrt(t)
    d1 = np.log(f / k) / sd + sd / 2
    call = f * ndtr(d1) - k * ndtr(d1 - sd)
    put = call - f + k
    df = np.exp(-0.03 * t)
    expected = terms.notional * np.mean(df * ((f - k - put) if upper else (k - f - call)))
    value = value_accumulator(
        terms,
        AccumulatorState.at_issue(terms, market.asof),
        market,
        strike_ratio=k,
        discount_factors=df,
        discount_currency="USD",
    )
    assert value.status == "ok"
    assert value.value == pytest.approx(expected, abs=terms.notional * 0.25e-4)


def test_resolution_failure_and_inputs():
    market = flat_market(0)
    terms = contract(AccumulatorDirection.ACCUMULATOR, count=3)
    args = dict(discount_factors=np.ones(3), discount_currency="USD")
    bad = AccumulatorPdeConfig(nodes=31, half_width=0.008, substeps=1, max_refinements=1)
    quote = quote_accumulator_strike(terms, market, config=bad, **args)
    assert quote.status == "uncertified" and np.isnan(quote.strike_ratio)
    assert quote.diagnostics.peak_boundary_mass > bad.boundary_tolerance
    overflow = quote_accumulator_strike(
        terms, market, config=AccumulatorPdeConfig(nodes=31, half_width=800), **args
    )
    assert overflow.status == "uncertified" and overflow.diagnostics.failure_reason
    with pytest.raises(ValueError, match="discount"):
        quote_accumulator_strike(terms, market, discount_factors=[1], discount_currency="USD")
    with pytest.raises(ValueError, match="original spot"):
        quote_accumulator_strike(replace(terms, original_spot=99), market, **args)


def test_tail_expansion_uses_the_marginal_volatility():
    market = flat_market(0)
    terms = contract(AccumulatorDirection.ACCUMULATOR, count=21, monthly=True)
    config = AccumulatorPdeConfig(nodes=241, half_width=0.11)
    q = quote_accumulator_strike(
        terms, market, discount_factors=[1.0], discount_currency="USD", config=config
    )
    assert q.status == "ok"
    assert q.diagnostics.half_width > 0.3
    assert q.diagnostics.peak_boundary_mass <= config.boundary_tolerance


def test_irregular_closes_and_marginal_correlation():
    from stochvolmodels.models.lsv.local_vol import LocalVolCoefficientProvider1D
    from stochvolmodels.tests.test_correlated_local_vol import market as basket

    model = basket()
    terms = replace(
        contract(AccumulatorDirection.ACCUMULATOR, count=3),
        asset_id="A",
        fixing_times=[1 / 365, 4 / 365, 6 / 365],
        payment_indices=[0, 0, 1],
        payment_times=[5 / 365, 8 / 365],
    )
    args = dict(discount_factors=[0.999, 0.998], discount_currency="USD")
    control = quote_accumulator_strike(terms, model, **args)
    changed = replace(model, correlation=[[1, -0.2], [-0.2, 1]])
    quote = quote_accumulator_strike(terms, changed, **args)
    assert control.status == quote.status == "ok"
    assert quote.strike_ratio == control.strike_ratio
    grid = Grid1D.uniform(-0.4, 0.4, 31)
    for i, asset in enumerate(model.asset_ids):
        native = LocalVolCoefficientProvider1D(model.surfaces[i]).evaluate(0.1, grid)
        actual = model.marginal_pde_coefficient_provider(asset).evaluate(0.1, grid)
        np.testing.assert_array_equal(actual.variance, native.variance)
        np.testing.assert_allclose(
            actual.drift,
            native.drift
            - np.sqrt(2 * native.variance) * model.fx_vols[i] * model.equity_fx_correlations[i],
        )


@pytest.mark.slow
def test_one_implicit_step_fails_barrier_accuracy(monkeypatch):
    # Negative control: replace the two implicit half-steps by one Euler step.
    from stochvolmodels.pde_solvers import PDE1DSolver
    from stochvolmodels.pricers.accumulator import _surviving_law

    solve = PDE1DSolver.solve

    def undamped(self, grid, provider, boundaries, direction, mesh, density):
        return solve(self, grid, provider, boundaries, direction, [mesh[0], mesh[-1]], density)

    monkeypatch.setattr(PDE1DSolver, "solve", undamped)
    terms, market = contract(AccumulatorDirection.ACCUMULATOR), flat_market(0)
    state = AccumulatorState.at_issue(terms, market.asof)
    pv, *_ = _surviving_law(terms, state, market, np.exp(-0.03 * terms.payment_times), 1601, 1.2, 1)
    assert 4.8 < pv(REFERENCE[0][1]) * 1e4 / terms.notional < 5.3
