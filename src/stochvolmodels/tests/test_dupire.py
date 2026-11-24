"""Continuous Dupire inversion, conversion to log-forward coordinates and vanilla repricing."""


import numpy as np
import pytest
from numba.typed import List
from scipy.special import ndtr
from scipy.stats import poisson

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import CallSurface, DupireRecoverer, DupireRepairPolicy
from stochvolmodels.models.lsv import FixedLogForwardJump, LSVModel, LeverageSurface
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer


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


@pytest.mark.parametrize("n", [5, 21])
@pytest.mark.parametrize("initial,slope", [(0.2, 0.4), (0.6, -0.4)])
def test_bucket_variance_reprices_sloped_term_structures(n, initial, slope):
    times = np.linspace(0.25, 0.75, n)
    strikes = np.linspace(60.0, 160.0, 81)
    total = initial**2 * times + initial * slope * times**2 + slope**2 * times**3 / 3
    sd = np.sqrt(total[:, None])
    d1 = np.log(100.0 / strikes) / sd + 0.5 * sd
    calls = 100.0 * ndtr(d1) - strikes * ndtr(d1 - sd)
    result = DupireRecoverer().continuous(CallSurface(times, strikes, calls, forwards=100.0))
    surface = result.to_local_vol_surface()
    quoted = [0, n // 2, n - 1]
    chain = OptionChain(
        ttms=times[quoted], forwards=np.full(3, 100.0), ids=None,
        strikes_ttms=(np.array([95.0, 100.0, 105.0]),) * 3,
        optiontypes_ttms=(np.array(["P", "C", "C"]),) * 3,
    )
    grid = Grid1D.concentrated(-3.0, 3.0, 501, 0.0, 0.22)
    prices = LocalVolPdePricer(grid=grid, steps_per_interval=128).price_chain(chain, surface)
    ivs = np.vstack(chain.compute_model_ivols_from_chain_data(prices))
    expected = np.sqrt(total[quoted] / times[quoted])
    assert np.max(np.abs(ivs - expected[:, None])) < 0.001
    bucket_variance = np.diff(np.r_[0.0, total]) / np.diff(np.r_[0.0, times])
    np.testing.assert_allclose(surface.values**2, np.broadcast_to(bucket_variance[:, None],
                                                                  surface.values.shape),
                               atol=1e-12, rtol=0)


def test_historical_forward_specific_regridding_and_flat_wings():
    quotes = _black_fixture()
    # Multiply all calls by a time-dependent factor to generate a spatially varying local law.
    calls = quotes.calls * np.exp(0.015 * quotes.times[:, None])
    result = DupireRecoverer().continuous(
        CallSurface(quotes.times, quotes.strikes, calls, quotes.forwards, quotes.discount_factors)
    )
    axis = Grid1D(np.array([-0.40, -0.06, 0.09, 0.35]))
    surface = result.to_local_vol_surface(axis, historical_endpoints=True)
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
            effective = result.to_local_vol_surface().values**2
            assert np.all(effective >= floor - 1e-14) and np.all(effective <= 0.05 + 1e-14)
            np.testing.assert_equal(result.raw_local_variance, raw.raw_local_variance)


def _jump_quotes(jump):
    """Discounted Poisson mixture of Black calls with nonconstant forward and discount."""
    t = np.linspace(0.04, 0.9, 87)
    k = np.linspace(55.0, 175.0, 241)
    f, d = 100 * np.exp(0.02 * t), np.exp(-0.04 * t)
    sd = 0.20 * np.sqrt(t[:, None])
    calls = np.zeros((len(t), len(k)))
    for n in range(20):
        forward = (f * np.exp(t * jump.compensator + n * jump.jump_size))[:, None]
        d1 = np.log(forward / k) / sd + sd / 2
        calls += poisson.pmf(n, jump.intensity * t)[:, None] * (
            forward * ndtr(d1) - k * ndtr(d1 - sd)
        )
    return CallSurface(t, k, d[:, None] * calls, f, d)


@pytest.mark.parametrize("size", [-0.16, 0.16])
def test_jump_adjusted_diffusion_recovery_and_vanilla_roundtrip(size, record_property):
    jump = FixedLogForwardJump(0.7, size)
    quotes = _jump_quotes(jump)
    fit = DupireRecoverer().jump_adjusted(quotes, jump, repair_policy=DupireRepairPolicy(0.01, 1))
    a = fit.jump_adjustment
    shifted = quotes.strikes * np.exp(-size)
    outside = (shifted < quotes.strikes[0]) | (shifted > quotes.strikes[-1])
    np.testing.assert_array_equal(a.extrapolated_mask, np.tile(outside, (len(quotes.times), 1)))
    assert not a.values.flags.writeable and not a.extrapolated_mask.flags.writeable
    inner = np.ix_(
        (quotes.times >= 0.12) & (quotes.times <= 0.82),
        (quotes.strikes >= 78) & (quotes.strikes <= 130),
    )
    assert np.all(fit.valid_mask[inner]) and not np.any(a.extrapolated_mask[inner])
    error = np.max(abs(fit.raw_local_variance[inner] - 0.04))
    assert error < 0.0015
    assert error < 0.2 * np.max(
        abs(DupireRecoverer().continuous(quotes).raw_local_variance[inner] - 0.04)
    )
    local = fit.to_local_vol_surface(Grid1D.uniform(-0.22, 0.22, 31))
    model = LSVModel.from_params(
        LogSvParams(sigma0=0.2, theta=0.2, kappa1=0, kappa2=0, beta=0, volvol=0),
        LeverageSurface(local.times, local.grid, local.values / 0.2, leverage_cap=5),
        jump=jump,
    )
    rows, cols = [0, 28, 58], [70, 90, 110]
    chain = OptionChain(
        ttms=quotes.times[rows],
        forwards=quotes.forwards[rows],
        discfactors=quotes.discount_factors[rows],
        ids=np.array(["first", "short", "long"]),
        strikes_ttms=(quotes.strikes[cols],) * 3,
        optiontypes_ttms=(np.full(3, "C"),) * 3,
    )
    grid = TensorGrid2D(
        Grid1D.concentrated(-1.5, 1.5, 201, 0.0, 0.15), Grid1D([0.10, 0.15, 0.20, 0.25, 0.30])
    )
    prices = LogSvPdePricer(grid=grid, steps_per_interval=16).price_chain(chain, model)
    gap = np.max(
        abs(np.vstack(prices) - quotes.calls[np.ix_(rows, cols)])
        / (chain.discfactors * chain.forwards)[:, None]
    )
    assert gap < 4e-4
    record_property("max_diffusion_variance_error", float(error))
    record_property("max_price_error_per_discounted_forward", float(gap))


def test_both_lsv_calibrators_receive_the_effective_bucket_variance():
    from stochvolmodels.fitters.lsv_fixed_point import LSVFixedPointCalibrator
    from stochvolmodels.fitters.lsv_projection import SequentialLSVProjector

    times = np.linspace(0.25, 0.75, 5)
    forwards, discounts = 100 * np.exp(0.06 * times), np.exp(-0.04 * times)
    strikes = np.linspace(80.0, 130.0, 51)
    total = 0.04 * times + 0.08 * times**2 + 0.16 * times**3 / 3
    sd = np.sqrt(total[:, None])
    d1 = np.log(forwards[:, None] / strikes) / sd + sd / 2
    calls = discounts[:, None] * (forwards[:, None] * ndtr(d1) - strikes * ndtr(d1 - sd))
    fit = DupireRecoverer().continuous(CallSurface(times, strikes, calls, forwards, discounts))
    grid = TensorGrid2D(Grid1D.uniform(-1.2, 1.2, 81), Grid1D.uniform(0.1, 0.3, 5))
    target = fit.to_local_vol_surface(grid.x)
    params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=0.0, kappa2=0.0, beta=0.0, volvol=0.0)
    expected = np.sqrt(np.diff(np.r_[0.0, total]) / np.diff(np.r_[0.0, times]))
    for fitted in (SequentialLSVProjector(grid).project(params, target, 12),
                   LSVFixedPointCalibrator(grid).calibrate(params, target, 12)):
        leverage = fitted.model.leverage_surface
        np.testing.assert_array_equal(leverage.times, times)
        np.testing.assert_allclose(leverage.values * params.sigma0,
                                   np.broadcast_to(expected[:, None], leverage.values.shape),
                                   atol=1e-12, rtol=0)


@pytest.mark.parametrize("theta,historical_rms", [(1.0, 58.288), (None, 57.906)])
def test_packaged_btc_bucket_prefit_reprices_all_common_calls(theta, historical_rms):
    from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
    from stochvolmodels.fitters.lsv_market import LSVOptionChainAdapter

    market = LSVOptionChainAdapter(get_btc_test_chain_data()).build_calibration_input(grid_size=11)
    fit = market.recover_dupire(repair_policy=DupireRepairPolicy(0.04, 4.0))
    source = fit.call_surface
    chain = _chain(source)
    pricer = LocalVolPdePricer(grid=Grid1D.concentrated(-3.0, 3.0, 501, 0.0, 0.22),
                               steps_per_interval=128, theta=theta)
    rms = []
    for historical in (True, False):
        local = fit.to_local_vol_surface(historical_endpoints=historical)
        calls = np.vstack(pricer.price_chain(chain, local))[:, :source.strikes.size]
        gap = 1e4 * (calls - source.calls) / (source.discount_factors * source.forwards)[:, None]
        assert gap.shape == (4, 11)
        rms.append(np.sqrt(np.mean(gap**2)))
    assert rms[0] == pytest.approx(historical_rms, abs=0.02)
    assert rms[1] < 35.0 and np.max(abs(gap)) < 140.0


def test_inactive_jump_matches_continuous_inversion():
    surface = _black_fixture()
    plain = DupireRecoverer().continuous(surface)
    assert plain.jump_adjustment is None
    for jump in (FixedLogForwardJump(0, -1000), FixedLogForwardJump(5, 0)):
        fit = DupireRecoverer().jump_adjusted(surface, jump)
        assert not fit.jump_adjustment.extrapolated_mask.any()
        assert not fit.jump_adjustment.values.any()
        np.testing.assert_array_equal(fit.raw_local_variance, plain.raw_local_variance)
        np.testing.assert_array_equal(fit.to_local_vol_surface().values,
                                      plain.to_local_vol_surface().values)


@pytest.mark.repository_only
@pytest.mark.pde_reference
def test_btc_market_jump_adjustment_matches_archive():
    from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
    from stochvolmodels.fitters.lsv_market import LSVOptionChainAdapter
    from validation.pde_reference import dupire as archive

    market = LSVOptionChainAdapter(get_btc_test_chain_data()).build_calibration_input(grid_size=11)
    policy = DupireRepairPolicy(0.04, 4.0)
    for size in (-0.18, 0.16):
        fit = market.recover_dupire(jump=FixedLogForwardJump(2, size), repair_policy=policy)
        surface = archive.CallSurface(
            market.target.times, market.target.strikes, market.target.target_calls
        )
        term = archive.deterministic_log_jump_term(surface, 2, size)
        old = archive.DupireRecoverer().jump_adjusted(
            surface, term, repair_policy=archive.DupireRepairPolicy(0.04, 4.0)
        )
        np.testing.assert_allclose(fit.jump_adjustment.values, term.values, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(fit.jump_adjustment.extrapolated_mask, term.extrapolated_mask)
        for field in ("raw_local_variance", "local_variance", "local_volatility"):
            np.testing.assert_allclose(
                getattr(fit, field), getattr(old, field), atol=1e-12, rtol=1e-12
            )
        for field in ("valid_mask", "repaired_mask", "interior_mask"):
            np.testing.assert_array_equal(getattr(fit, field), getattr(old, field))


def _compare_inversions(source, policy):
    from validation.pde_reference import dupire as archive

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
    from validation.pde_reference import local_vol as old
    from stochvolmodels.pricers.density_pricer import DensityPricer, DensityState1D

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
    calculated = LocalVolPdePricer(grid=grid, steps_per_interval=steps, theta=1.0).price_chain(
        chain, fit.to_local_vol_surface(historical_endpoints=True)
    )
    actual = np.array(list(calculated))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
    max_error = float(np.max(np.abs(actual - expected)))
    input_residual = np.abs(actual[:, : len(source.strikes)] - source.calls)
    return max_error, input_residual


@pytest.mark.repository_only
@pytest.mark.pde_reference
def test_archived_skew_recovery_refinement_and_repricing(record_property):
    from validation.pde_reference.cases import _pde_call_surface

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
@pytest.mark.pde_reference
def test_frozen_btc_dupire_and_vanilla_prices(record_property):
    from validation.pde_reference import dupire as old
    from validation.pde_reference.integration import LSVOptionChainAdapter
    from validation.pde_reference.market_pipeline import (
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
        fit.to_local_vol_surface(historical_endpoints=True).values,
        reference.local_volatility, rtol=0, atol=1e-14
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


def test_immutable_arrays_preserve_dupire_missing_values():
    from stochvolmodels.utils._arrays import readonly_float_array

    source = np.array([[1, np.nan], [2, 3]], dtype=np.float32).T
    result = readonly_float_array(source, ndim=2, name="quotes", require_finite=False)
    source[0, 0] = 9
    assert result[0, 0] == 1 and np.isnan(result[1, 0])
    assert result.dtype == np.float64 and result.flags.c_contiguous and not result.flags.writeable
    with pytest.raises(ValueError, match="quotes must be a finite 2-dimensional array"):
        readonly_float_array(source, ndim=2, name="quotes")
    with pytest.raises(ValueError, match="quotes must be a 1-dimensional array"):
        readonly_float_array(source, ndim=1, name="quotes", require_finite=False)
