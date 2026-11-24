"""Frozen LogSV chain grids, offline premium references and coefficient-solve counts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import affine_expansion as afe
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels.tests.test_fourier_reference_cases import (
    FORWARD, LOGSV_ALT, LOGSV_BASE, LOGSV_STRIKES, PANELS_TO_128, PREMIUM_ATOL,
    _legendre_reference_grid, _logsv_calls,
)
from stochvolmodels.utils import _fourier_quadrature as fq
from stochvolmodels.utils.config import VariableType


def _chain(days, strikes=LOGSV_STRIKES):
    return OptionChain(
        ttms=np.asarray(days) / 365, forwards=np.full(len(days), FORWARD),
        discfactors=np.ones(len(days)), strikes_ttms=tuple(strikes for _ in days),
        optiontypes_ttms=tuple(np.full(len(strikes), "C") for _ in days), ids=None,
    )


def test_one_day_tail_exceeds_135_node_seed():
    with pytest.raises(fq.QuadratureError, match="cutoff budget") as failure:
        LogSVPricer().price_chain(_chain((1, 7)), LOGSV_ALT, fourier={"max_cutoff": 128})
    diag = failure.value.diagnostics
    assert diag.maturity == 1 / 365 and diag.cutoff == 128
    assert diag.evaluations >= 135 and diag.tail_estimate > PREMIUM_ATOL


@pytest.mark.slow
def test_held_out_quotes_against_wide_inversion(monkeypatch):
    """Count actual coefficient solves, including preflight, on new one- and seven-day calls."""
    days = (1, 7)
    strikes = np.array([87., 95., 105., 113.])
    reference = _logsv_calls(LOGSV_ALT, days, _legendre_reference_grid(4096), strikes)
    truncated = _logsv_calls(LOGSV_ALT, days, _legendre_reference_grid(64), strikes)
    assert np.max(np.abs(reference - truncated)) > PREMIUM_ATOL
    solver = afe.solve_ode_for_a
    frequencies = []

    def count(**kwargs):
        frequencies.append(kwargs["phi"])
        return solver(**kwargs)

    monkeypatch.setattr(afe, "solve_ode_for_a", count)
    pricer = LogSVPricer()
    grid = pricer.prepare_fourier(_chain(days), LOGSV_ALT)
    prepared = len(frequencies)
    assert prepared == 2 * len(set(frequencies))
    cached = grid.price_chain(_chain(days, strikes))
    assert len(frequencies) == prepared
    frequencies.clear()
    fresh = pricer.price_chain(_chain(days, strikes), LOGSV_ALT, fourier={})
    assert len(frequencies) == 2 * len(set(frequencies)) > 0
    np.testing.assert_allclose(np.asarray(cached), reference, rtol=0, atol=PREMIUM_ATOL)
    np.testing.assert_allclose(np.asarray(fresh), reference, rtol=0, atol=PREMIUM_ATOL)
    print(f"prepare={prepared}, query=0, fresh={len(frequencies)} coefficient solves")


@pytest.mark.slow
@pytest.mark.parametrize("params,days,cutoff", [
    (LOGSV_BASE, (30, 180), 512), (LOGSV_ALT, (30, 180), 1024), (LOGSV_ALT, (1, 7), 4096),
])
def test_chain_against_independent_legendre(monkeypatch, params, days, cutoff):
    evaluate = fq.FourierPlan.price_vanillas

    def check_range(plan, *args, **kwargs):
        assert plan.edges[-1] < cutoff
        return evaluate(plan, *args, **kwargs)

    monkeypatch.setattr(fq.FourierPlan, "price_vanillas", check_range)
    prices = LogSVPricer().price_chain(_chain(days), params, fourier={})
    reference = _logsv_calls(params, days, _legendre_reference_grid(cutoff))
    np.testing.assert_allclose(np.asarray(prices), reference, rtol=0, atol=PREMIUM_ATOL)


def test_frozen_nodes_shared_across_expiries_and_strikes(monkeypatch):
    solver = afe.compute_logsv_a_mgf_grid
    ode = afe.solve_ode_for_a
    evaluate = fq.FourierPlan.price_vanillas
    batches, plans = [], []
    solve_count = 0

    def count_ode(**kwargs):
        nonlocal solve_count
        solve_count += 1
        return ode(**kwargs)

    def record_solve(**kwargs):
        batches.append((kwargs["ttm"], kwargs["phi_grid"].copy()))
        return solver(**kwargs)

    def record_price(plan, values, **quotes):
        plans.append(plan)
        return evaluate(plan, values, **quotes)

    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", record_solve)
    monkeypatch.setattr(afe, "solve_ode_for_a", count_ode)
    monkeypatch.setattr(fq.FourierPlan, "price_vanillas", record_price)
    chain = _chain((1, 7))
    pricer = LogSVPricer()
    params = replace(LOGSV_ALT)
    original = pricer.price_chain(chain, params, fourier={})
    assert len(plans) == 2 and plans[0] is plans[1]
    assert plans[0].edges[-1] > 128
    seed = fq.FourierPlan(PANELS_TO_128)
    np.testing.assert_array_equal(batches[0][1].imag, seed.nodes)
    first = np.concatenate([phi for dt, phi in batches if dt == chain.ttms[0]])
    second = np.concatenate([phi for dt, phi in batches if dt == np.diff(chain.ttms)[0]])
    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == first.size and np.isin(plans[0].nodes, first.imag).all()
    assert solve_count == 2 * first.size
    batches.clear()
    solve_count = 0
    interior = _chain((1, 7), np.array([120., 95., 100., 105., 80.]))
    expanded = pricer.price_chain(interior, params, fourier={})
    np.testing.assert_array_equal(np.concatenate([phi for _, phi in batches[::2]]), first)
    assert solve_count == 2 * first.size
    np.testing.assert_allclose(np.asarray(expanded)[:, [4, 2, 0]], original, atol=1e-12, rtol=0)
    batches.clear()
    params.sigma0 = LOGSV_BASE.sigma0
    changed = pricer.price_chain(chain, params, fourier={})
    assert batches and abs(changed[0][1] - original[0][1]) > 1e-3


def test_calendar_backbone_chain_matches_separate_expiries():
    params = replace(LOGSV_BASE, vol_backbone=pd.Series([0.8, 1.25, 0.95],
                                                      index=np.array([12, 40, 75]) / 365))
    chain = OptionChain(
        ttms=np.array([20, 60]) / 365, forwards=np.array([103., 98.]),
        discfactors=np.array([0.99, 0.96]), strikes_ttms=(LOGSV_STRIKES, np.array([109., 91.])),
        optiontypes_ttms=(np.array(["C", "P", "C"]), np.array(["P", "C"])), ids=None,
    )
    pricer = LogSVPricer()
    atols = [np.full(k.shape, PREMIUM_ATOL) for k in chain.strikes_ttms]
    prices = pricer.price_chain(chain, params, fourier={"price_atol": atols})
    for i, ttm in enumerate(chain.ttms):
        single, _ = pricer.price_slice(params, ttm, chain.forwards[i], chain.strikes_ttms[i],
                                       chain.optiontypes_ttms[i], chain.discfactors[i], fourier={})
        np.testing.assert_allclose(prices[i], single, atol=2 * PREMIUM_ATOL, rtol=0)


def test_inverse_contour_against_weighted_reference():
    u, weights = _legendre_reference_grid(512)
    phi = 0.5 + 1j * u
    ttm, forward, discount = 30 / 365, 96.0, 0.973
    strikes = np.array([105., 85., 105., 85., 105., 85.])
    codes = np.array(["IC", "IP", "IP", "IC", "C", "P"])
    _, transform = afe.compute_logsv_a_mgf_grid(
        ttm=ttm, phi_grid=phi, psi_grid=np.zeros_like(phi), theta_grid=np.zeros_like(phi),
        is_spot_measure=False, **LOGSV_ALT.to_dict(),
    )
    x = np.log(forward / strikes)
    capped = (np.exp(transform - x[:, None] * phi).real / (0.25 + u**2)) @ weights / np.pi
    expected = discount * (np.where(np.isin(codes, ["C", "IC"]), forward, strikes)
                           - forward * capped)
    chain = OptionChain.slice_to_chain(ttm, forward, strikes, codes, discount)
    prices = LogSVPricer().price_chain(chain, LOGSV_ALT, is_spot_measure=False, fourier={})[0]
    np.testing.assert_allclose(prices, expected, atol=PREMIUM_ATOL, rtol=0)
    grid = LogSVPricer().prepare_fourier(chain, LOGSV_ALT, is_spot_measure=False)
    np.testing.assert_allclose(grid.price_chain(chain)[0], expected, rtol=0, atol=PREMIUM_ATOL)
    np.testing.assert_allclose(prices[[0, 3]] - prices[[2, 1]],
                               discount * (forward - strikes[[0, 1]]), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(prices[[4, 5]], prices[[0, 1]])


@pytest.mark.parametrize("kwargs", [
    {"max_phi": 135}, {"vol_scaler": 0.1}, {"is_simpson": False},
    {"variable_type": VariableType.Q_VAR}, {"historical_backbone_order": True},
])
def test_controlled_rejects_incompatible_legacy_arguments(kwargs):
    with pytest.raises(ValueError):
        LogSVPricer().price_chain(_chain((30,)), LOGSV_BASE, fourier={}, **kwargs)


@pytest.mark.parametrize(
    "reason", ["coefficient ODE failed", "nonfinite log transform", "overflow"],
)
def test_late_expiry_transform_failure_identifies_affected_quotes(monkeypatch, reason):
    original = afe.compute_logsv_a_mgf_grid
    chain = _chain((1, 7))

    def fail_on_continuation(**kwargs):
        if kwargs["ttm"] > chain.ttms[0]:
            if reason == "coefficient ODE failed":
                raise FloatingPointError(reason)
            value = np.nan if reason == "nonfinite log transform" else 900.0
            return kwargs["a_t0"], np.full_like(kwargs["phi_grid"], value)
        return original(**kwargs)

    monkeypatch.setattr(afe, "compute_logsv_a_mgf_grid", fail_on_continuation)
    with pytest.raises(fq.QuadratureError, match=reason) as failure:
        LogSVPricer().price_chain(chain, LOGSV_ALT, fourier={})
    assert failure.value.diagnostics.maturity == chain.ttms[1]
    assert isinstance(failure.value.__cause__, FloatingPointError)


def test_explicit_legacy_replays_custom_grid_and_helpers_forward_settings():
    pricer = LogSVPricer()
    chain = _chain((1, 7))
    historical = {"max_phi": 151, "vol_scaler": 0.05}
    default = pricer.price_chain(chain, LOGSV_ALT, **historical)
    replay = pricer.price_chain(chain, LOGSV_ALT, fourier="legacy", **historical)
    np.testing.assert_array_equal(replay, default)
    limited = {"max_cutoff": 128}
    for method in (pricer.compute_chain_prices_with_vols, pricer.compute_model_ivols_for_chain):
        with pytest.raises(fq.QuadratureError, match="cutoff budget"):
            method(chain, LOGSV_ALT, fourier=limited)
