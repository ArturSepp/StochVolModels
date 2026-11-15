"""Continuous Dupire inversion, conversion to log-forward coordinates and vanilla repricing."""

from pathlib import Path

import numpy as np
import pytest
from numba.typed import List
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import CallSurface, DupireRecoverer, DupireRepairPolicy
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer


def _black_fixture():
    times = 0.20 + 0.80 * np.linspace(0.0, 1.0, 35) ** 1.2
    strikes = 100.0 * np.exp(np.linspace(-0.20, 0.22, 65))
    forwards = 100.0 * np.exp(0.075 * times)
    discounts = np.exp(-0.032 * times)
    f = forwards[:, None]
    sd = 0.23 * np.sqrt(times[:, None])
    d1 = np.log(f / strikes) / sd + 0.5 * sd
    calls = discounts[:, None] * (f * ndtr(d1) - strikes * ndtr(d1 - sd))
    return CallSurface(times, strikes, calls, forwards, discounts)


def _chain(surface):
    # Request both legs for every input node, even when one of them is deep ITM.
    strikes = np.tile(surface.strikes, 2)
    types = np.repeat(["C", "P"], len(surface.strikes))
    return OptionChain(
        ttms=surface.times,
        ids=None,
        forwards=surface.forwards,
        discfactors=surface.discount_factors,
        strikes_ttms=List([strikes.copy() for _ in surface.times]),
        optiontypes_ttms=List([types.copy() for _ in surface.times]),
    )


def test_nonuniform_black_recovery_and_discounted_repricing(record_property):
    quotes = _black_fixture()
    fit = DupireRecoverer().continuous(quotes)
    assert fit.valid_mask.all()
    vol_error = np.max(np.abs(fit.local_volatility - 0.23))
    assert vol_error < 2.0e-4
    chain = _chain(quotes)
    grid = Grid1D.concentrated(-1.5, 1.5, 501, 0.0, 0.19)
    prices = np.asarray(
        list(
            LocalVolPdePricer(grid=grid, steps_per_interval=128).price_chain(
                chain, fit.to_local_vol_surface()
            )
        )
    )
    puts = quotes.calls + quotes.discount_factors[:, None] * (
        quotes.strikes - quotes.forwards[:, None]
    )
    truth = np.hstack([quotes.calls, puts])
    err = np.abs(prices - truth) / (quotes.discount_factors * quotes.forwards)[:, None]
    assert err.max() < 8e-5
    record_property("black_max_vol_error", float(vol_error))
    record_property("black_max_discounted_forward_price_error", float(err.max()))


def test_forward_specific_regridding_and_flat_wings():
    quotes = _black_fixture()
    # Multiply all calls by a time-dependent factor to generate a spatially varying local law.
    calls = quotes.calls * np.exp(0.015 * quotes.times[:, None])
    result = DupireRecoverer().continuous(
        CallSurface(quotes.times, quotes.strikes, calls, quotes.forwards, quotes.discount_factors)
    )
    axis = Grid1D(np.array([-0.40, -0.06, 0.09, 0.35]))
    surface = result.to_local_vol_surface(axis)
    assert np.ptp(result.local_volatility[-1]) > 0.005
    for i, forward in enumerate(quotes.forwards):
        expected = np.interp(
            axis.nodes, np.log(quotes.strikes / forward), result.local_volatility[i]
        )
        np.testing.assert_allclose(surface.values[i], expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        result.to_local_vol_surface().grid.nodes, np.log(quotes.strikes / quotes.forwards[0])
    )


def test_raw_invalid_nodes_explicit_repair_and_input_ownership():
    quotes = _black_fixture()
    calls = quotes.calls.copy()
    calls[7, 20] += 2.0
    source = CallSurface(
        quotes.times, quotes.strikes, calls, quotes.forwards, quotes.discount_factors
    )
    calls[:] = 0.0
    raw = DupireRecoverer().continuous(source)
    assert not raw.valid_mask[7, 20]
    assert not raw.repaired_mask.any()
    assert not raw.interior_mask[[0, -1]].any() and not raw.interior_mask[:, [0, -1]].any()
    with pytest.raises(ValueError, match="explicit repair"):
        raw.to_local_vol_surface()
    for floor in (0.0, 0.03):
        policy = DupireRepairPolicy(floor, 0.05)
        result = DupireRecoverer().continuous(source, repair_policy=policy)
        np.testing.assert_equal(result.raw_local_variance, raw.raw_local_variance)
        np.testing.assert_array_equal(result.valid_mask, raw.valid_mask)
        assert (result.local_variance[~raw.valid_mask] == floor).all()
        expected = (
            (~raw.valid_mask) | (raw.raw_local_variance < floor) | (raw.raw_local_variance > 0.05)
        )
        np.testing.assert_array_equal(result.repaired_mask, expected)
        assert not result.local_variance.flags.writeable and not source.calls.flags.writeable
        if floor == 0.0:
            with pytest.raises(ValueError, match="strictly positive"):
                result.to_local_vol_surface()
        else:
            assert np.isfinite(result.to_local_vol_surface().values).all()


def _require_archive():
    if not (Path(__file__).resolve().parents[1] / "pde_solvers" / "dupire.py").is_file():
        pytest.skip("requires the archived Dupire and density implementations")


def _compare_inversions(source, policy):
    from stochvolmodels.pde_solvers import dupire as archive

    quotes = CallSurface(
        source.times, source.strikes, source.calls, source.forwards, source.discount_factors
    )
    fit = DupireRecoverer().continuous(quotes, repair_policy=policy)
    old_policy = (
        None
        if policy is None
        else archive.DupireRepairPolicy(policy.variance_floor, policy.variance_cap)
    )
    reference = archive.DupireRecoverer().continuous(source, repair_policy=old_policy)
    for field in (
        "call_time_derivative",
        "call_strike_derivative",
        "call_strike_second_derivative",
        "numerator",
        "denominator",
        "raw_local_variance",
        "local_variance",
        "local_volatility",
        "valid_mask",
        "repaired_mask",
        "interior_mask",
    ):
        np.testing.assert_array_equal(getattr(fit, field), getattr(reference, field), err_msg=field)
    return fit, reference


def _compare_archive_prices(fit, old_vols, old_axis, grid, steps):
    from stochvolmodels.pde_solvers import BoundaryPair1D, PDE1DSolver, PDEDirection, TimeStepConfig
    from stochvolmodels.pde_solvers import local_vol as old
    from stochvolmodels.pde_solvers.density import DensityPricer, DensityState1D

    source = fit.call_surface
    chain = _chain(source)
    archived = old.LocalVolSurface(
        source.times, Grid1D(old_axis), old_vols, old.CoordinateSystem.LOG_FORWARD
    )
    times = np.r_[
        0.0,
        np.concatenate(
            [
                np.linspace(a, b, steps + 1)[1:]
                for a, b in zip(np.r_[0.0, source.times[:-1]], source.times)
            ]
        ),
    ]
    states = PDE1DSolver(TimeStepConfig(theta=1.0)).solve(
        grid,
        old.LocalVolCoefficientProvider1D(archived),
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        times,
        grid.project_delta(0.0),
        save_times=tuple(source.times),
    )
    expected = []
    for i, snapshot in enumerate(states.saved_states):
        law = DensityState1D(grid, snapshot.values, snapshot.time, archived.coordinate)
        prices = DensityPricer(law, source.forwards[i], source.discount_factors[i]).price_slice(
            source.strikes
        )
        expected.append(np.r_[prices.calls, prices.puts])
    calculated = LocalVolPdePricer(grid=grid, steps_per_interval=steps).price_chain(
        chain, fit.to_local_vol_surface()
    )
    actual = np.array(list(calculated))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
    max_error = float(np.max(np.abs(actual - expected)))
    input_residual = np.abs(actual[:, : len(source.strikes)] - source.calls)
    return max_error, input_residual


@pytest.mark.repository_only
def test_archived_skew_recovery_refinement_and_repricing(record_property):
    _require_archive()
    from stochvolmodels.pde_solvers.tests.test_pde_dupire import _pde_call_surface

    var_errors = []
    for space, steps in ((201, 200), (601, 1200)):
        source, truth = _pde_call_surface(space, steps)
        raw, _ = _compare_inversions(source, None)
        interior = np.ix_(
            (source.times >= 0.12) & (source.times <= 0.45),
            (source.strikes >= 0.78) & (source.strikes <= 1.25),
        )
        expected = np.vstack([truth.variance(t, np.log(source.strikes)) for t in source.times])
        assert raw.valid_mask[interior].all()
        var_errors.append(float(np.max(np.abs(raw.local_variance[interior] - expected[interior]))))
    assert var_errors[1] < var_errors[0] and var_errors[1] < 2.5e-3
    fit, reference = _compare_inversions(source, DupireRepairPolicy())
    price_error, residual = _compare_archive_prices(
        fit,
        reference.local_volatility,
        np.log(source.strikes),
        Grid1D.concentrated(-1.45, 1.45, 601, 0.0, 0.12),
        120,
    )
    assert residual.max() < 0.001
    for key, val in {
        "coarse_variance_error": var_errors[0],
        "fine_variance_error": var_errors[1],
        "fine_vol_error": float(
            np.max(np.abs(raw.local_volatility[interior] - np.sqrt(expected[interior])))
        ),
        "fine_repaired_nodes": int(fit.repaired_mask.sum()),
        "fine_nodes": int(source.calls.size),
        "fine_repaired_interior": int(fit.repaired_mask[interior].sum()),
        "fine_archive_price_error": price_error,
        "fine_max_call_roundtrip": float(residual.max()),
    }.items():
        record_property(key, val)


@pytest.mark.repository_only
def test_frozen_btc_dupire_and_vanilla_prices(record_property):
    _require_archive()
    from stochvolmodels.pde_solvers import dupire as old
    from stochvolmodels.pde_solvers.integration import LSVOptionChainAdapter
    from stochvolmodels.pde_solvers.market_pipeline import (
        btc_20211021_reference_config,
        load_frozen_btc_20211021_snapshot,
    )

    snapshot = load_frozen_btc_20211021_snapshot()
    config = btc_20211021_reference_config()
    inputs = LSVOptionChainAdapter(
        snapshot.to_option_chain(), config.quote_selection
    ).build_calibration_input(grid_size=config.calibration_grid_size)
    target = inputs.target
    source = old.CallSurface(
        target.times, target.strikes, target.target_calls, target.forward, target.discount_factor
    )
    policy = config.dupire_repair_policy
    fit, reference = _compare_inversions(
        source, DupireRepairPolicy(policy.variance_floor, policy.variance_cap)
    )
    np.testing.assert_allclose(
        fit.to_local_vol_surface().values, reference.local_volatility, rtol=0, atol=1e-14
    )
    assert not np.any(fit.repaired_mask & fit.interior_mask)
    price_error, residual = _compare_archive_prices(
        fit,
        reference.local_volatility,
        inputs.log_moneyness,
        Grid1D.concentrated(-3.0, 3.0, 501, 0.0, 0.22),
        128,
    )
    record_property("btc_nodes", int(source.calls.size))
    record_property("btc_repaired_nodes", int(fit.repaired_mask.sum()))
    record_property("btc_archive_price_error", price_error)
    record_property("btc_max_call_roundtrip", float(residual.max()))
