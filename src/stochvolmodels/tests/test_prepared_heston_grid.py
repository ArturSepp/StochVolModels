"""Prepared Heston pricing, certification, and incremental transform evaluations."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import ndtr

import stochvolmodels as svm
from stochvolmodels.models.heston import HestonModel
from stochvolmodels.tests.test_fourier_reference_cases import (
    HESTON_STRESS,
    _heston_cf,
    _heston_probability_calls,
)
from stochvolmodels.utils._fourier_quadrature import QuadratureError


def chain(strikes, *, t=10.0, forward=100.0, discount=1.0, types=None):
    strikes = np.asarray(strikes, dtype=float)
    codes = np.full(len(strikes), "C") if types is None else np.asarray(types)
    return svm.OptionChain.slice_to_chain(t, forward, strikes, codes, discount)


def test_ten_year_prices_and_incremental_refinement(monkeypatch):
    pricer = svm.HestonPricer()
    wing, dense = chain([140.0]), chain(np.linspace(50.0, 150.0, 1001))
    fresh = pricer.price_chain(dense, HESTON_STRESS, fourier={})[0]
    reference = _heston_probability_calls(
        10.0, HESTON_STRESS, np.array([50.0, 100.0, 140.0, 150.0])
    )
    seen = set()
    original = HestonModel._affine_coefficients

    def track(self, *, ttm, phi_grid):
        assert seen.isdisjoint(phi_grid)
        seen.update(phi_grid)
        return original(self, ttm=ttm, phi_grid=phi_grid)

    monkeypatch.setattr(HestonModel, "_affine_coefficients", track)
    params = replace(HESTON_STRESS)
    grid = pricer.prepare_fourier(wing, params)
    assert isinstance(grid, svm.PreparedHestonGrid)
    before = len(seen)
    params.rho = 0.1
    np.testing.assert_allclose(grid.price_chain(wing)[0], reference[2], rtol=0.0, atol=1e-8)
    with pytest.raises(QuadratureError, match="prepared grid"):
        grid.price_chain(dense)
    assert len(seen) == before
    upgraded = grid.refined_for(dense)
    assert upgraded.diagnostics.evaluations == len(seen) > before
    assert upgraded.refined_for(dense) is upgraded
    monkeypatch.setattr(
        HestonModel, "_affine_coefficients", lambda *a, **kw: pytest.fail("warm transform")
    )
    quotes = upgraded.price_chain(dense)[0]
    np.testing.assert_allclose(quotes, fresh, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(quotes[[0, 500, 900, 1000]], reference, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(upgraded.price_chain(wing)[0], reference[2], atol=1e-6, rtol=0.0)
    with pytest.raises(QuadratureError):
        grid.price_chain(dense)


def test_new_quotes_at_prepared_expiries_and_aligned_budgets(monkeypatch):
    original = svm.OptionChain(
        ttms=np.array([0.5, 10.0]),
        forwards=np.array([98.0, 100.0]),
        discfactors=np.array([0.99, 1.0]),
        strikes_ttms=(np.array([93.0, 102.0]), np.array([50.0, 100.0, 150.0])),
        optiontypes_ttms=(np.array(["P", "C"]), np.array(["C", "C", "P"])),
        ids=None,
    )
    tolerances = [np.full(2, 1e-7), np.full(3, 1e-7)]
    pricer = svm.HestonPricer()
    grid = pricer.prepare_fourier(original, HESTON_STRESS, fourier={"price_atol": tolerances})
    changed = chain([110.0, 98.0, 110.0], forward=105.0, discount=0.97, types=["P", "C", "C"])
    tight = {"price_atol": 1e-6}
    expected = pricer.price_chain(changed, HESTON_STRESS, fourier=tight)[0]
    base = pricer.price_chain(original, HESTON_STRESS, fourier={"price_atol": tolerances})
    tolerances[0][:] = np.nan
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **kw: pytest.fail("warm transform")
        )
        for got, target in zip(grid.price_chain(original), base, strict=True):
            np.testing.assert_allclose(got, target, atol=1e-7, rtol=0.0)
        with pytest.raises(ValueError, match="strike-aligned"):
            grid.price_chain(changed)
        np.testing.assert_allclose(
            grid.price_chain(changed, fourier=tight)[0], expected, atol=1e-6, rtol=0.0
        )
        with pytest.raises(ValueError, match="not prepared"):
            grid.price_chain(chain([100.0], t=1.0))
        with pytest.raises(ValueError, match="query settings"):
            grid.price_chain(changed, fourier={"max_nodes": 512})
        with pytest.raises(ValueError, match="option type"):
            grid.price_chain(chain([100.0], types=["IC"]), fourier=tight)


def test_refinement_respects_work_limits_and_preserves_seed():
    pricer = svm.HestonPricer()
    wing = chain([140.0])
    grid = pricer.prepare_fourier(wing, HESTON_STRESS, fourier={"max_nodes": 180})
    original = grid.price_chain(wing)[0]
    with pytest.raises(QuadratureError, match="budget"):
        grid.refined_for(chain(np.linspace(50.0, 150.0, 1001)))
    np.testing.assert_array_equal(grid.price_chain(wing)[0], original)
    with pytest.raises(ValueError, match="controlled"):
        pricer.prepare_fourier(wing, HESTON_STRESS, fourier="legacy")
    with pytest.raises(ValueError, match="spot measure"):
        pricer.prepare_fourier(wing, HESTON_STRESS, is_spot_measure=False)


def test_same_quotes_with_a_tighter_budget_need_certification(monkeypatch):
    pricer = svm.HestonPricer()
    wing = chain([140.0])
    # A relative tolerance can dominate the absolute tolerance on the seed.
    loose = {"price_atol": 1e-10, "price_rtol": 0.1, "time_values": np.array([1.0])}
    strict = {"price_rtol": 0.0}
    grid = pricer.prepare_fourier(wing, HESTON_STRESS, fourier=loose)
    reference = pricer.price_chain(wing, HESTON_STRESS, fourier={"price_atol": 1e-10})
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **kw: pytest.fail("warm transform")
        )
        with pytest.raises(QuadratureError, match="prepared grid"):
            grid.price_chain(wing, fourier=strict)
    refined = grid.refined_for(wing, fourier=strict)
    assert refined.diagnostics.evaluations > grid.diagnostics.evaluations
    np.testing.assert_allclose(
        refined.price_chain(wing, fourier=strict)[0], reference[0], atol=1e-10, rtol=0.0
    )
    assert refined.refined_for(wing, fourier=strict) is refined


def test_forward_greeks_one_hour_black_limit_and_separate_budgets(monkeypatch):
    sigma, discount, t = 0.24, 0.98, 1 / (365 * 24)
    quotes = chain([100.0, 100.0, 100.3, 99.7], t=t, discount=discount, types=["C", "P", "C", "P"])
    params = replace(HESTON_STRESS, v0=sigma**2, theta=sigma**2, volvol=0.0)
    grid = svm.HestonPricer().prepare_fourier(quotes, params, fourier={"price_atol": 1.0})
    strict = dict(delta_atol=1e-9, gamma_atol=1e-9)
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **k: pytest.fail("warm transform")
        )
        for tolerances in (
            dict(delta_atol=1e-9, gamma_atol=10.0),
            dict(delta_atol=10.0, gamma_atol=1e-9),
        ):
            with pytest.raises(QuadratureError, match="prepared grid"):
                grid.forward_greeks(quotes, **tolerances)
    risk = grid.refined_for(quotes, **strict)
    assert risk.diagnostics.evaluations > grid.diagnostics.evaluations
    assert risk.refined_for(quotes, **strict) is risk
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **k: pytest.fail("warm transform")
        )
        deltas, gammas = risk.forward_greeks(quotes, **strict)
    std = sigma * np.sqrt(t)
    d1 = np.log(100.0 / quotes.strikes_ttms[0]) / std + std / 2
    expected_delta = discount * (ndtr(d1) - (quotes.optiontypes_ttms[0] == "P"))
    expected_gamma = discount * np.exp(-(d1**2) / 2) / (100.0 * std * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(deltas[0], expected_delta, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(gammas[0], expected_gamma, rtol=0.0, atol=1e-9)
    assert deltas[0][0] - deltas[0][1] == pytest.approx(discount)
    assert gammas[0][0] == gammas[0][1]

    def price(shift):
        moved = replace(quotes, forwards=quotes.forwards + shift)
        return svm.HestonPricer().price_chain(moved, params, fourier={"price_atol": 1e-11})[0]

    centre = price(0.0)
    for h in (0.001, 0.0005):
        up, down = price(h), price(-h)
        np.testing.assert_allclose(deltas[0], (up - down) / (2 * h), atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(
            gammas[0], ((up - centre) + (down - centre)) / h**2, atol=4e-6, rtol=0.0
        )
    for name in strict:
        for bad in (True, 0.0, -1.0, np.inf, np.nan):
            with pytest.raises(ValueError, match=name):
                risk.forward_greeks(quotes, **{name: bad})


def test_forward_greeks_match_fresh_bumps_on_changed_quotes(monkeypatch):
    pricer = svm.HestonPricer()
    seed = svm.OptionChain(
        ttms=np.array([1.0, 10.0]),
        forwards=np.array([100.0, 100.0]),
        discfactors=np.array([0.96, 1.0]),
        strikes_ttms=(np.array([85.0, 100.0, 120.0]), np.linspace(50.0, 150.0, 31)),
        optiontypes_ttms=(np.array(["P", "C", "C"]), np.full(31, "C")),
        ids=None,
    )
    quotes = svm.OptionChain(
        ttms=seed.ttms,
        forwards=np.array([98.0, 103.0]),
        discfactors=np.array([0.95, 0.97]),
        strikes_ttms=(np.array([95.0, 100.0]), np.array([140.0, 100.0, 140.0])),
        optiontypes_ttms=(np.array(["P", "C"]), np.array(["C", "C", "P"])),
        ids=None,
    )
    tolerances = dict(delta_atol=1e-9, gamma_atol=1e-10)
    grid = pricer.prepare_fourier(seed, HESTON_STRESS).refined_for(seed, **tolerances)
    # First query on new contracts checks the fixed grid. The explicit certificate
    # exercises the fast projection with the same ragged rows and repeated strikes.
    first = grid.forward_greeks(quotes, **tolerances)
    certified = grid.refined_for(quotes, **tolerances)
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **k: pytest.fail("warm transform")
        )
        delta, gamma = (
            np.concatenate(rows) for rows in certified.forward_greeks(quotes, **tolerances)
        )
    np.testing.assert_allclose(delta, np.concatenate(first[0]), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(gamma, np.concatenate(first[1]), rtol=0.0, atol=1e-10)

    def prices(shift):
        bumped = replace(quotes, forwards=quotes.forwards + shift)
        return np.concatenate(
            pricer.price_chain(bumped, HESTON_STRESS, fourier={"price_atol": 1e-11})
        )

    centre = prices(0.0)
    for h in (0.01, 0.005):
        up, down = prices(h), prices(-h)
        np.testing.assert_allclose(delta, (up - down) / (2 * h), atol=6e-7, rtol=0.0)
        np.testing.assert_allclose(
            gamma, ((up - centre) + (down - centre)) / h**2, atol=2e-7, rtol=0.0
        )


def _probability_greeks(strikes):
    """Independent ten-year share-measure tail probability and log density at each strike."""
    log_fk = np.log(100.0 / strikes)

    def integrand(u):
        tilted = np.exp(1j * u * log_fk) * _heston_cf(u - 1j, 10.0, HESTON_STRESS)
        return np.array([tilted.imag / u, tilted.real / 100.0]) / np.pi

    values, estimate = quad_vec(integrand, 0.0, np.inf, epsabs=1e-11, epsrel=1e-12)
    assert estimate < 1e-10
    return 0.5 + values[0], values[1]


def test_ten_year_forward_greeks_against_share_measure_inversion():
    strikes = np.linspace(50.0, 150.0, 1001)
    quotes = chain(strikes)
    settings = dict(delta_atol=1e-9, gamma_atol=1e-10)
    risk = svm.HestonPricer().prepare_fourier(quotes, HESTON_STRESS).refined_for(quotes, **settings)
    delta, gamma = risk.forward_greeks(quotes, **settings)
    ref_delta, ref_gamma = _probability_greeks(strikes)
    np.testing.assert_allclose(delta[0], ref_delta, atol=settings["delta_atol"], rtol=0.0)
    np.testing.assert_allclose(gamma[0], ref_gamma, atol=settings["gamma_atol"], rtol=0.0)


@pytest.mark.parametrize("kappa, state", [(0.0, 0.09), (2.0, 0.09), (2.0, 0.0)])
def test_variance_scenarios_match_deterministic_black(kappa, state, monkeypatch):
    t, f, discount = 0.4, 102.0, 0.97
    params = replace(HESTON_STRESS, theta=0.04, kappa=kappa, volvol=0.0)
    quotes = chain([108.0, 95.0, 108.0], t=t, forward=f, discount=discount, types=["C", "P", "P"])
    pricer = svm.HestonPricer()
    seed = pricer.prepare_fourier(quotes, params)
    original = seed.price_chain(quotes)[0].copy()
    risk = seed.refined_for(
        quotes, v0=state, sensitivity_atol=1e-8, delta_atol=1e-8, gamma_atol=1e-8
    )
    duration = t if kappa == 0.0 else -np.expm1(-kappa * t) / kappa
    std = np.sqrt(state * duration + params.theta * (t - duration))
    d1 = np.log(f / quotes.strikes_ttms[0]) / std + 0.5 * std
    side = np.where(quotes.optiontypes_ttms[0] == "C", 1.0, -1.0)
    expected = (
        discount * side * (f * ndtr(side * d1) - quotes.strikes_ttms[0] * ndtr(side * (d1 - std)))
    )
    density = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **kw: pytest.fail("new coefficients")
        )
        np.testing.assert_allclose(
            risk.price_chain(quotes, v0=state)[0], expected, atol=1e-6, rtol=0.0
        )
        sensitivity = risk.v0_sensitivity(quotes, v0=state, atol=1e-8)[0]
        np.testing.assert_allclose(
            sensitivity, discount * f * density * duration / (2 * std), atol=1e-8, rtol=0.0
        )
        assert sensitivity[0] == sensitivity[2]
        delta, gamma = risk.forward_greeks(quotes, v0=state, delta_atol=1e-8, gamma_atol=1e-8)
        np.testing.assert_allclose(
            delta[0],
            discount * (ndtr(d1) - (quotes.optiontypes_ttms[0] == "P")),
            atol=1e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(gamma[0], discount * density / (f * std), atol=1e-8, rtol=0.0)
        np.testing.assert_array_equal(seed.price_chain(quotes)[0], original)
        np.testing.assert_array_equal(seed.price_chain(quotes, v0=0.04)[0], original)


def test_v0_certification_is_state_specific_and_matches_fresh_central_bumps(monkeypatch):
    quotes = chain([140.0, 85.0, 140.0], types=["P", "C", "C"], discount=0.97)
    params = replace(HESTON_STRESS)
    pricer = svm.HestonPricer()
    seed = pricer.prepare_fourier(quotes, params, fourier={"price_atol": 1.0})
    params.v0 = 0.9
    state, atol = 0.01, 1e-8
    with pytest.raises(QuadratureError, match="prepared grid"):
        seed.v0_sensitivity(quotes, v0=state, atol=atol)
    risk = seed.refined_for(quotes, v0=state, fourier={"price_atol": 1e-6}, sensitivity_atol=atol)
    assert risk.refined_for(quotes, v0=state, sensitivity_atol=atol) is risk
    assert risk.refined_for(quotes, v0=0.09, sensitivity_atol=atol) is not risk

    def fresh(v):
        return pricer.price_chain(
            quotes, replace(HESTON_STRESS, v0=v), fourier={"price_atol": 1e-11}
        )[0]

    derivative = risk.v0_sensitivity(quotes, v0=state, atol=atol)[0]
    for h in (1e-4, 5e-5):
        np.testing.assert_allclose(
            derivative, (fresh(state + h) - fresh(state - h)) / (2 * h), rtol=0.0, atol=1e-5
        )
    np.testing.assert_allclose(
        risk.price_chain(quotes, v0=state)[0], fresh(state), rtol=0.0, atol=1e-6
    )
    # A nearby, uncertified state can pass the full checks on an already adequate grid.
    nearby, h = state + 0.001, 5e-5
    expected_price = fresh(nearby)
    expected_risk = (fresh(nearby + h) - fresh(nearby - h)) / (2 * h)
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "_affine_coefficients", lambda *a, **kw: pytest.fail("new coefficients")
        )
        np.testing.assert_allclose(
            risk.price_chain(quotes, v0=nearby, fourier={"price_atol": 1e-6})[0],
            expected_price,
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            risk.v0_sensitivity(quotes, v0=nearby, atol=1e-6)[0],
            expected_risk,
            rtol=0.0,
            atol=1e-6,
        )
    with pytest.raises(QuadratureError):
        seed.v0_sensitivity(quotes, v0=state, atol=atol)
    for bad in (True, -1.0, np.nan, np.inf, [0.04]):
        with pytest.raises(ValueError, match="v0"):
            risk.price_chain(quotes, v0=bad)
    with pytest.raises(ValueError, match="atol"):
        risk.v0_sensitivity(quotes, atol=0.0)


def test_v0_at_two_expiries_against_independent_probability_inversion():
    """Richardson-extrapolated bumps of a separate P1/P2 inversion check the variance loading."""
    state = 0.09
    strikes = (np.array([98.0, 100.0, 102.0]), np.array([60.0, 100.0, 140.0]))
    maturities = np.array([0.1, 10.0])
    quotes = svm.OptionChain(
        ttms=maturities,
        forwards=np.full(2, 100.0),
        discfactors=np.array([1.0, 0.97]),
        strikes_ttms=strikes,
        optiontypes_ttms=(np.full(3, "C"), np.full(3, "C")),
        ids=None,
    )
    risk = (
        svm.HestonPricer()
        .prepare_fourier(quotes, HESTON_STRESS)
        .refined_for(quotes, v0=state, fourier={"price_atol": 1e-8}, sensitivity_atol=1e-8)
    )
    prices = risk.price_chain(quotes, v0=state, fourier={"price_atol": 1e-8})
    derivatives = risk.v0_sensitivity(quotes, v0=state, atol=1e-8)
    for t, k, discount, price, derivative in zip(
        maturities, strikes, quotes.discfactors, prices, derivatives, strict=True
    ):

        def independent(v):
            return discount * _heston_probability_calls(t, replace(HESTON_STRESS, v0=v), k)

        np.testing.assert_allclose(price, independent(state), atol=1e-8, rtol=0.0)
        d1, d2 = [(independent(state + h) - independent(state - h)) / (2 * h) for h in (1e-4, 5e-5)]
        np.testing.assert_allclose(derivative, (4 * d2 - d1) / 3, atol=1e-8, rtol=0.0)
