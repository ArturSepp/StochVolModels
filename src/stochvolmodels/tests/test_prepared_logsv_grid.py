"""Prepared-grid ownership, quote certification and incremental coefficient work."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import affine_expansion as afe
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels.tests.test_fourier_reference_cases import (
    LOGSV_ALT,
    LOGSV_BASE,
    _legendre_reference_grid,
    _logsv_calls,
)
from stochvolmodels.tests.test_logsv_fourier_quadrature import _chain
from stochvolmodels.utils._fourier_quadrature import QuadratureError


def test_forward_greeks_short_atm_black_limit_and_refinement(monkeypatch):
    sigma = .24
    params = replace(LOGSV_BASE, sigma0=sigma, theta=sigma, kappa1=0., kappa2=0., beta=0., volvol=0.)
    chain = OptionChain.slice_to_chain(1 / (365 * 24), 100., np.array([100., 100., 100.3, 99.7]),
                                       np.array(["C", "P", "C", "P"]), .98)
    grid = LogSVPricer().prepare_fourier(chain, params, fourier={"price_atol": 1.})
    strict = dict(delta_atol=1e-9, gamma_atol=1e-9)
    with monkeypatch.context() as patch:
        patch.setattr(afe, "compute_logsv_a_mgf_grid", lambda **kw: pytest.fail("query solved ODEs"))
        for tolerances in (dict(delta_atol=1e-9, gamma_atol=10.), dict(delta_atol=10., gamma_atol=1e-9)):
            with pytest.raises(QuadratureError, match="prepared grid"):
                grid.forward_greeks(chain, **tolerances)

    refined = grid.refined_for(chain, **strict)
    assert refined.diagnostics.evaluations > grid.diagnostics.evaluations
    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", lambda **kw: pytest.fail("query solved ODEs"))
    assert refined.refined_for(chain, **strict) is refined
    delta, gamma = refined.forward_greeks(chain, **strict)
    std = sigma * np.sqrt(chain.ttms[0])
    d1 = np.log(100. / chain.strikes_ttms[0]) / std + .5 * std
    black_delta = .98 * (ndtr(d1) - (chain.optiontypes_ttms[0] == "P"))
    black_gamma = .98 * np.exp(-.5 * d1**2) / (np.sqrt(2 * np.pi) * 100. * std)
    np.testing.assert_allclose(delta[0], black_delta, rtol=0, atol=1e-9)
    np.testing.assert_allclose(gamma[0], black_gamma, rtol=0, atol=1e-9)
    assert delta[0][0] - delta[0][1] == pytest.approx(.98)
    assert gamma[0][0] == gamma[0][1]
    # Changing the contract's unit of account preserves delta and scales gamma inversely.
    for scale in (1e-170, 1e170):
        scaled = replace(chain, forwards=scale * chain.forwards,
                         strikes_ttms=(scale * chain.strikes_ttms[0],))
        scaled_delta, scaled_gamma = refined.forward_greeks(
            scaled, delta_atol=1e-9, gamma_atol=1e-9 / scale)
        np.testing.assert_allclose(scaled_delta[0], black_delta, rtol=0, atol=1e-9)
        np.testing.assert_allclose(scale * scaled_gamma[0], black_gamma, rtol=0, atol=1e-9)
    centre = refined.price_chain(chain)[0]
    for h in (.001, .0005):
        up = refined.price_chain(replace(chain, forwards=chain.forwards + h))[0]
        down = refined.price_chain(replace(chain, forwards=chain.forwards - h))[0]
        np.testing.assert_allclose(delta[0], (up - down) / (2*h), rtol=0, atol=1e-6)
        np.testing.assert_allclose(gamma[0], (up - 2*centre + down) / h**2, rtol=0, atol=3e-6)
    for name in strict:
        with pytest.raises(ValueError, match=name):
            refined.forward_greeks(chain, **{name: 0.})


@pytest.mark.parametrize("spot", [True, False], ids=["money-market", "inverse"])
def test_forward_greeks_against_central_forward_bumps(monkeypatch, spot):
    # Hold strikes, dynamics and discount fixed. Reprice the bumped forwards independently.
    quotes = OptionChain(
        ttms=np.array([1., 30.]) / 365, forwards=np.array([100., 103.]),
        strikes_ttms=(np.array([100., 100., 102.]), np.array([97., 110.])),
        optiontypes_ttms=(np.array(["C", "P", "C"]), np.array(["P", "C"])) if spot else
                         (np.array(["IC", "IP", "IC"]), np.array(["IP", "IC"])),
        discfactors=np.array([.99, .95]), ids=None,
    )
    pricer = LogSVPricer()
    params = replace(LOGSV_BASE, sigma0=.37)
    tight_price = {"price_atol": 1e-10}
    seed = _chain((1, 30), np.array([100.]))
    grid = pricer.prepare_fourier(seed, LOGSV_BASE, is_spot_measure=spot)
    strict = dict(sigma0=params.sigma0, delta_atol=1e-8, gamma_atol=1e-8)
    grid = grid.refined_for(quotes, **strict)
    with monkeypatch.context() as patch:
        patch.setattr(afe, "compute_logsv_a_mgf_grid", lambda **kw: pytest.fail("query solved ODEs"))
        deltas, gammas = grid.forward_greeks(quotes, **strict)

    def price(shift):
        bumped = replace(quotes, forwards=quotes.forwards + shift)
        return np.concatenate(pricer.price_chain(bumped, params, fourier=tight_price, is_spot_measure=spot))

    centre = price(0.)
    for h in (.005, .0025):
        up, down = price(h), price(-h)
        np.testing.assert_allclose(np.concatenate(deltas), (up - down) / (2*h), rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.concatenate(gammas), (up - 2*centre + down) / h**2,
                                   rtol=0, atol=5e-7)


@pytest.mark.parametrize("order,spot", [(afe.ExpansionOrder.FIRST, False),
                                       (afe.ExpansionOrder.SECOND, True)])
def test_state_prices_and_sensitivity_use_saved_coefficients(monkeypatch, order, spot):
    curve = pd.Series([.9, 1.1], index=np.array([12., 80.]) / 365)
    params = replace(LOGSV_BASE, vol_backbone=curve)
    chain = _chain((20, 60))
    quotes = OptionChain.slice_to_chain(60 / 365, 102., np.array([108., 97., 108.]),
                                         np.array(["C", "P", "P"] if spot else ["IC", "IP", "IP"]), .96)
    pricer = LogSVPricer()
    opts = dict(is_spot_measure=spot, expansion_order=order)
    strict = {"price_atol": 1e-8}
    grid = pricer.prepare_fourier(chain, params, fourier=strict, **opts)
    sigma = params.theta  # polynomial derivative must also work at the expansion centre
    state_grid = grid.refined_for(quotes, sigma0=sigma, sensitivity_atol=1e-7)
    h = 1e-5

    def fresh(s):
        return pricer.price_chain(quotes, replace(params, sigma0=s), fourier=strict, **opts)[0]

    reference = fresh(sigma)
    difference = (fresh(sigma+h) - fresh(sigma-h)) / (2*h)
    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", lambda **kw: pytest.fail("new solve"))
    np.testing.assert_allclose(state_grid.price_chain(quotes, sigma0=sigma)[0], reference,
                               rtol=0, atol=1e-7)
    risk = state_grid.sigma0_sensitivity(quotes, sigma0=sigma, atol=1e-7)[0]
    np.testing.assert_allclose(risk, difference, rtol=0, atol=2e-6)
    assert risk[0] == risk[2]  # call-put parity at fixed forward and discount
    np.testing.assert_array_equal(state_grid.price_chain(chain)[0], state_grid.price_chain(
        chain, sigma0=params.sigma0)[0])


def test_subset_measure_backbone_and_scalar_snapshot(monkeypatch):
    curve = pd.Series([0.8, 1.3, 1.0], index=np.array([12.0, 40.0, 80.0]) / 365)
    params = replace(LOGSV_BASE, vol_backbone=curve)
    seed = _chain((20, 60))
    quotes = OptionChain.slice_to_chain(
        60 / 365, 103.0, np.array([110.0, 95.0, 110.0]), np.array(["IC", "IP", "IP"]), 0.97
    )
    pricer = LogSVPricer()
    grid = pricer.prepare_fourier(seed, params, is_spot_measure=False)
    expected = pricer.price_chain(quotes, params, is_spot_measure=False, fourier={})[0]
    params.sigma0 = 0.9
    curve.iloc[0] = 2.0
    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", lambda **kw: pytest.fail("new solve"))
    with ThreadPoolExecutor(2) as pool:
        for prices in pool.map(grid.price_chain, (quotes, quotes)):
            np.testing.assert_allclose(prices[0], expected, rtol=0, atol=1e-6)
            assert prices[0][0] - prices[0][2] == pytest.approx(0.97 * (103.0 - 110.0))
    refined = grid.refined_for(quotes)
    np.testing.assert_array_equal(refined.price_chain(quotes)[0], grid.price_chain(quotes)[0])
    with pytest.raises(ValueError, match="not prepared"):
        grid.price_chain(_chain((59,)))


def test_vector_budgets_follow_original_quotes_only():
    seed = _chain((20, 60))
    seed.strikes_ttms = (seed.strikes_ttms[0], np.array([95.0, 100.0]))
    seed.optiontypes_ttms = (seed.optiontypes_ttms[0], np.array(["P", "C"]))
    atols = [np.full(3, 2e-6), np.full(2, 1e-6)]
    grid = LogSVPricer().prepare_fourier(seed, LOGSV_BASE, fourier={"price_atol": atols})
    original = grid.price_chain(seed)
    atols[0][:] = 1e-30
    assert np.array_equal(grid.price_chain(seed)[0], original[0])
    changed = _chain((20,), np.array([97.0, 103.0]))
    with pytest.raises(ValueError, match="strike-aligned price_atol"):
        grid.price_chain(changed)
    grid.price_chain(changed, fourier={"price_atol": 1e-6})
    with pytest.raises(ValueError, match="time value"):
        grid.price_chain(changed, fourier={"price_atol": 1e-6, "price_rtol": 0.01})


def test_refine_for_a_volatility_state_and_its_derivative(monkeypatch):
    # With constant volatility the model is Black, including its derivative in sigma0.
    params = replace(LOGSV_BASE, kappa1=0., kappa2=0., beta=0., volvol=0.)
    days = 7
    t = days / 365
    quotes = _chain((days,), np.array([97., 100., 103.]))
    sigma = .12
    solver = afe.compute_logsv_a_mgf_grid
    seen = set()

    def counted(**kwargs):
        nodes = kwargs["phi_grid"].imag
        assert seen.isdisjoint(nodes)
        seen.update(nodes)
        return solver(**kwargs)

    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", counted)
    grid = LogSVPricer().prepare_fourier(quotes, params, fourier={"price_atol": .1})
    original = grid.price_chain(quotes)[0].copy()
    before = len(seen)
    neighbour = grid.refined_for(quotes, sigma0=params.sigma0 + .001, fourier={"price_atol": 1.})
    assert neighbour._coefficients is grid._coefficients
    with pytest.raises(QuadratureError):
        grid.sigma0_sensitivity(quotes, atol=1e-12)
    tight = {"price_atol": 1e-8}
    with pytest.raises(QuadratureError):
        grid.price_chain(quotes, sigma0=sigma, fourier=tight)
    with pytest.raises(QuadratureError):
        grid.sigma0_sensitivity(quotes, sigma0=sigma, atol=1e-8)
    assert len(seen) == before
    refined = grid.refined_for(quotes, sigma0=sigma, fourier=tight, sensitivity_atol=1e-8)
    assert len(seen) > before
    assert refined.refined_for(quotes, sigma0=sigma, fourier=tight, sensitivity_atol=1e-8) is refined
    after = len(seen)
    prices = refined.price_chain(quotes, sigma0=sigma, fourier=tight)[0]
    vegas = refined.sigma0_sensitivity(quotes, sigma0=sigma, atol=1e-8)[0]
    strikes = quotes.strikes_ttms[0]
    d1 = np.log(100. / strikes) / (sigma * np.sqrt(t)) + .5 * sigma * np.sqrt(t)
    expected = 100. * ndtr(d1) - strikes * ndtr(d1 - sigma * np.sqrt(t))
    np.testing.assert_allclose(prices, expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(vegas, 100. * np.sqrt(t / (2 * np.pi)) * np.exp(-.5 * d1**2),
                               rtol=0, atol=1e-8)
    np.testing.assert_array_equal(grid.price_chain(quotes)[0], original)
    assert len(seen) == after
    assert not refined._coefficients.flags.writeable
    assert refined._coefficients.shape == (1, refined._plan.nodes.size, 5)
    for bad in [True, -.1, np.inf, np.nan, [sigma]]:
        with pytest.raises(ValueError, match="sigma0"):
            grid.price_chain(quotes, sigma0=bad)
    with pytest.raises(ValueError, match="atol"):
        grid.sigma0_sensitivity(quotes, atol=0.)


@pytest.mark.slow
def test_tail_and_panel_refinement_use_only_new_frequencies(monkeypatch):
    solver = afe.compute_logsv_a_mgf_grid
    sampled = set()

    def counted(**kwargs):
        nodes = kwargs["phi_grid"].imag.tolist()
        assert sampled.isdisjoint(nodes)
        sampled.update(nodes)
        return solver(**kwargs)

    seed = _chain((1,))
    request = _chain((1,), np.array([87.0, 95.0, 105.0, 113.0]))
    reference = _logsv_calls(
        LOGSV_ALT, (1,), _legendre_reference_grid(4096), strikes=request.strikes_ttms[0]
    )[0]
    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", counted)
    grid = LogSVPricer().prepare_fourier(seed, LOGSV_ALT, fourier={"price_atol": 1.0})
    before = len(sampled)
    original = grid.price_chain(seed)[0].copy()
    strict = {"price_atol": 1e-6}
    unverified = grid._plan.price_vanillas(
        grid._quotes[0].evaluate(grid._coefficients[0], grid._plan.nodes),
        forward=100.0,
        strikes=request.strikes_ttms[0],
        optiontypes=request.optiontypes_ttms[0],
    )
    assert np.max(np.abs(unverified - reference)) > 1e-6
    with pytest.raises(QuadratureError, match="prepared grid") as miss:
        grid.price_chain(request, fourier=strict)
    assert miss.value.diagnostics.strike in request.strikes_ttms[0]
    assert len(sampled) == before
    upgraded = grid.refined_for(request, fourier=strict)
    assert upgraded.diagnostics.evaluations == len(sampled) > before
    assert upgraded.diagnostics.refinements > grid.diagnostics.refinements
    assert upgraded.diagnostics.cutoff > grid.diagnostics.cutoff
    np.testing.assert_allclose(
        upgraded.price_chain(request, fourier=strict)[0], reference, rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(grid.price_chain(seed)[0], original)


def test_failed_refinement_preserves_handle():
    seed = _chain((1,))
    limits = {"price_atol": 1.0, "max_cutoff": 256}
    grid = LogSVPricer().prepare_fourier(seed, LOGSV_ALT, fourier=limits)
    original = grid.price_chain(seed)[0]
    with pytest.raises(QuadratureError, match="budget"):
        grid.refined_for(seed, fourier={"price_atol": 1e-8})
    np.testing.assert_array_equal(grid.price_chain(seed)[0], original)


def test_refinement_uses_snapshot_after_solver_failure(monkeypatch):
    days = (20, 60)
    chain = _chain(days)
    curve = pd.Series([.8, 1.25, .95], index=np.array([12., 40., 75.]) / 365)
    params = replace(LOGSV_BASE, vol_backbone=curve)
    pricer = LogSVPricer()
    grid = pricer.prepare_fourier(chain, params, fourier={"price_atol": 1.})
    original = np.asarray(grid.price_chain(chain))
    strict = {"price_atol": 1e-8}
    reference = np.asarray(pricer.price_chain(chain, params, fourier=strict))
    params.sigma0 = .7
    curve.iloc[1] = 2.
    solver = afe.compute_logsv_a_mgf_grid

    def fail(**kwargs):
        if kwargs["vol_backbone_eta"] == .95:
            raise FloatingPointError("coefficient solver failure")
        return solver(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(afe, "compute_logsv_a_mgf_grid", fail)
        with pytest.raises(QuadratureError, match="coefficient solver failure") as error:
            grid.refined_for(chain, fourier=strict)
        assert error.value.diagnostics.maturity == days[1] / 365
    np.testing.assert_array_equal(np.asarray(grid.price_chain(chain)), original)
    upgraded = grid.refined_for(chain, fourier=strict)
    np.testing.assert_allclose(np.asarray(upgraded.price_chain(chain, fourier=strict)), reference,
                               rtol=0, atol=2e-8)
