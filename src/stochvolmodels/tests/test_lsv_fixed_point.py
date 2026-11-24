"""LSV fixed-point convergence, retries and repricing through the public vanilla pricer."""

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from stochvolmodels.fitters.lsv_fixed_point import (
    LSVConvergenceError,
    LSVFixedPointCalibrator,
    LSVFixedPointConfig,
)
from stochvolmodels.models.lsv.local_vol import LocalVolSurface
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.fitters.lsv_projection import LowDensityPolicy
from stochvolmodels.tests.test_lsv_projection import _archive_prices, _quotes


def _deterministic():
    x = Grid1D.concentrated(-0.9, 0.9, 151, 0.0, 0.12)
    grid = TensorGrid2D(x, Grid1D([0.1, 0.15, 0.2, 0.25, 0.3]))
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    times = np.array([0.1, 0.2])
    target = LocalVolSurface(times, x, np.repeat([[0.20], [0.26]], x.size, axis=1))
    return grid, params, target


@pytest.mark.parametrize("beta", [-0.5, 0.5])
def test_correlated_default_fits_and_strict_density_limits(beta):
    from stochvolmodels.pricers.logsv_pde_pricer import _default_grid
    from stochvolmodels.fitters.lsv_projection import SequentialLSVProjector

    params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=2.0, beta=beta, volvol=0.8)
    chain = _quotes(np.array([0.1, 0.25, 0.5]))
    grid = _default_grid(params, chain)
    target = LocalVolSurface(chain.ttms, grid.x, np.full((3, grid.x.size), 0.25))
    projector = SequentialLSVProjector(grid)
    fitter = LSVFixedPointCalibrator(grid)
    reference = LocalVolPdePricer().price_chain(chain, target)
    for engine, method in ((projector, "project"), (fitter, "calibrate")):
        run = getattr(engine, method)(params, target)
        assert run.peak_negative_mass > engine.config.negative_mass_tolerance
        replay = LogSvPdePricer(grid=grid, steps_per_interval=64).price_chain_detailed(
            chain, run.model
        )
        for price, ref, disc, forward, expiry in zip(
            replay.prices, reference, chain.discfactors, chain.forwards, replay.expiries,
            strict=True,
        ):
            np.testing.assert_allclose(price, ref, atol=0.0015 * disc * forward, rtol=0)
            assert expiry.negative_mass < engine.config.negative_mass_tolerance
            assert abs(expiry.mass - 1) < engine.config.mass_tolerance
        if method == "calibrate":
            assert all(row.iterations[-1].converged for row in run.slices)
        strict = replace(engine, config=replace(engine.config, negative_mass_tolerance=1e-10))
        with pytest.raises(ValueError, match="T=0.1.*negative mass"):
            getattr(strict, method)(params, target, 64)


def test_converged_candidate_density_and_black_repricing():
    grid, params, target = _deterministic()
    # The second row's bounded proposal is 1.3. The 1.15 retry is within the
    # chosen absolute tolerance, so its own propagated law must carry forward.
    config = LSVFixedPointConfig(damping=0.5, absolute_tolerance=0.16, relative_tolerance=0)
    fit = LSVFixedPointCalibrator(grid, config).calibrate(params, target, [60, 60])
    assert [len(s.iterations) for s in fit.slices] == [1, 2]
    np.testing.assert_allclose(fit.model.leverage_surface.values[1], 1.15, atol=1e-12)
    row = fit.slices[-1]
    assert row.start_density is fit.slices[0].accepted.terminal_density
    assert row.iterations[0].maximum_absolute_residual == pytest.approx(0.30)
    assert row.iterations[1].maximum_absolute_residual == pytest.approx(0.15)
    np.testing.assert_allclose(row.accepted.leverage_update.accepted_leverage, 1.225, atol=1e-12)
    chain = _quotes(target.times)
    prices = LogSvPdePricer(grid=grid, steps_per_interval=60).price_chain(chain, fit.model)
    for i, (t, f, d, k, cp) in enumerate(
        zip(
            chain.ttms,
            chain.forwards,
            chain.discfactors,
            chain.strikes_ttms,
            chain.optiontypes_ttms,
        )
    ):
        sd = np.sqrt(0.20**2 * min(0.10, t) + 0.23**2 * max(0.0, t - 0.10))
        d1 = np.log(f / k) / sd + 0.5 * sd
        calls = d * (f * ndtr(d1) - k * ndtr(d1 - sd))
        np.testing.assert_allclose(
            (prices[i] - np.where(cp == "C", calls, calls + d * (k - f))) / (d * f),
            0.0,
            atol=1.1e-4,
        )
        marginal = (
            grid.marginalize_y(fit.slices[i].accepted.terminal_density.values)
            * grid.x.quadrature_weights
        )
        signs = np.where(cp == "C", 1.0, -1.0)
        expected = d * (marginal @ np.maximum((f * np.exp(grid.x.nodes[:, None]) - k) * signs, 0.0))
        np.testing.assert_allclose(prices[i], expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("method", ["project", "calibrate"])
def test_internal_midpoint_rows_and_independent_quote_repricing(method):
    from stochvolmodels.fitters.lsv_projection import LSVProjectionConfig, SequentialLSVProjector

    grid, params, target = _deterministic()
    params = replace(params, vol_backbone=pd.Series([1.1, 0.95], index=[0.06, 0.2]))
    if method == "project":
        engine = SequentialLSVProjector(grid, LSVProjectionConfig(collocation="midpoint"))
    else:
        engine = LSVFixedPointCalibrator(grid, LSVFixedPointConfig(collocation="midpoint"))
    quotes = _quotes(target.times)
    fit = getattr(engine, method)(params, target, [12, 16], leverage_times=[0.035, 0.16],
                                  quote_chain=quotes)
    np.testing.assert_allclose(fit.model.leverage_surface.times, [0.035, 0.06, 0.1, 0.16, 0.2])
    slices = [getattr(s, "accepted", s) for s in fit.slices]
    np.testing.assert_allclose([s.collocation_time for s in slices],
                               [0.0175, 0.0475, 0.08, 0.13, 0.18])
    np.testing.assert_allclose([s.terminal_density.time for s in slices],
                               fit.model.leverage_surface.times)
    replay = LogSvPdePricer(grid=grid, steps_per_interval=[12, 16]).price_chain(quotes, fit.model)
    assert len(fit.repricing) == 2
    for i, row in enumerate(fit.repricing):
        np.testing.assert_array_equal(row.strikes, quotes.strikes_ttms[i])
        np.testing.assert_array_equal(row.option_types, quotes.optiontypes_ttms[i])
        np.testing.assert_allclose(row.model_prices, replay[i], atol=1e-12, rtol=0)
        np.testing.assert_allclose(row.price_errors, row.model_prices - row.target_prices)
        expected_iv = 0.2 if i == 0 else np.sqrt((0.2**2 + 0.26**2) / 2)
        np.testing.assert_allclose(row.target_ivols, expected_iv)
        np.testing.assert_allclose(row.iv_errors, row.model_ivols - expected_iv, equal_nan=True)
        assert row.max_abs_price_error == pytest.approx(np.max(abs(row.price_errors)))


@pytest.mark.parametrize("dates", [[0.0], [0.3], [0.15, 0.1], [np.nan]])
def test_invalid_internal_leverage_dates(dates):
    grid, params, target = _deterministic()
    with pytest.raises(ValueError, match="leverage_times"):
        LSVFixedPointCalibrator(grid).calibrate(params, target, leverage_times=dates)


def test_flat_target_preserves_fractional_premiums_on_integer_strikes():
    from stochvolmodels.data.option_chain import OptionChain

    grid, params, target = _deterministic()
    chain = OptionChain.slice_to_chain(0.1, 100., np.array([100]), np.array(["C"]), 0.98)
    fit = LSVFixedPointCalibrator(grid).calibrate(params, target, 12, quote_chain=chain)
    (row,) = fit.repricing
    exact = 0.98 * 100 * (2 * ndtr(0.2 * np.sqrt(0.1) / 2) - 1)
    assert row.target_prices[0] == pytest.approx(exact, abs=1e-12)
    assert row.target_ivols[0] == pytest.approx(0.2)


def test_sparse_midpoint_quote_uses_target_step_counts_and_skew_oracle():
    from stochvolmodels.data.option_chain import OptionChain
    from stochvolmodels.fitters.lsv_projection import LSVProjectionConfig, SequentialLSVProjector

    grid, params, original = _deterministic()
    target = LocalVolSurface(original.times, grid.x, original.values * (1 + 0.1 * grid.x.nodes))
    full = _quotes(target.times)
    last = OptionChain.slice_to_chain(full.ttms[-1], full.forwards[-1], full.strikes_ttms[-1],
                                      full.optiontypes_ttms[-1], full.discfactors[-1])
    fit = SequentialLSVProjector(grid, LSVProjectionConfig(collocation="midpoint")).project(
        params, target, [3, 20], quote_chain=last,
    )
    (row,) = fit.repricing
    replay = LogSvPdePricer(grid=grid, steps_per_interval=[3, 20]).price_chain(full, fit.model)
    reference = LocalVolPdePricer().price_chain(last, target)
    np.testing.assert_allclose(row.model_prices, replay[-1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(row.target_prices, reference[0], atol=1e-12, rtol=0)
    assert row.density_mass == pytest.approx(1., abs=1e-10)
    assert row.negative_mass < 1e-4


@pytest.mark.parametrize("method", ["project", "calibrate"])
def test_refining_midpoint_rows_reduces_flat_smile_error(method):
    from stochvolmodels.data.option_chain import OptionChain
    from stochvolmodels.pricers.logsv_pde_pricer import _default_grid
    from stochvolmodels.fitters.lsv_projection import LSVProjectionConfig, SequentialLSVProjector

    params = LogSvParams(sigma0=.2, theta=.2, kappa1=2., kappa2=2., beta=-.5, volvol=.8)
    times = np.array([.1, .25, .5])
    strikes = np.array([.85, .95, 1., 1.05, 1.15])
    quotes = OptionChain(
        ttms=times, forwards=np.ones(3), discfactors=np.ones(3), ids=None,
        strikes_ttms=(strikes,) * 3, optiontypes_ttms=(np.where(strikes < 1, "P", "C"),) * 3,
    )
    grid = _default_grid(params, quotes)
    target = LocalVolSurface(times, grid.x, np.full((3, grid.x.size), .25))
    if method == "project":
        fitter = SequentialLSVProjector(grid, LSVProjectionConfig(collocation="midpoint"))
    else:
        fitter = LSVFixedPointCalibrator(grid, LSVFixedPointConfig(collocation="midpoint"))
    maxima = []
    for dates in (None, np.arange(1, 11) / 20):
        result = getattr(fitter, method)(params, target, leverage_times=dates, quote_chain=quotes)
        assert max(row.negative_mass for row in result.repricing) < 1e-4
        maxima.append(max(row.max_abs_iv_error for row in result.repricing))
    assert maxima[1] < 0.6 * maxima[0]
    assert maxima[1] < .0015


def test_zero_damping_exhausts_budget_only_on_unconverged_interval():
    grid, params, target = _deterministic()
    cfg = LSVFixedPointConfig(
        damping=0.0, maximum_iterations=3, absolute_tolerance=1e-10, relative_tolerance=0.0
    )
    with pytest.raises(LSVConvergenceError, match="after 3 iterations") as caught:
        LSVFixedPointCalibrator(grid, cfg).calibrate(params, target, 16)
    failure = caught.value
    assert failure.time_start == 0.1 and failure.time_end == 0.2
    assert not failure.start_density.values.flags.writeable
    for trial in failure.iterations:
        assert not trial.converged and trial.maximum_relative_residual == pytest.approx(0.3 / 1.3)
        np.testing.assert_array_equal(
            trial.projection.terminal_density.values,
            failure.iterations[0].projection.terminal_density.values,
        )
        assert not trial.projection.propagation_leverage.flags.writeable
    for changes in (
        dict(maximum_iterations=0),
        dict(maximum_iterations=True),
        dict(relative_tolerance=np.nan),
        dict(absolute_tolerance=-1),
    ):
        with pytest.raises(ValueError):
            replace(cfg, **changes)


def test_bounded_fixed_point_and_interior_backbone_knot():
    grid, params, target = _deterministic()
    params = replace(params, vol_backbone=pd.Series([0.5, 1.0, 0.8], index=[0.06, 0.1, 0.2]))
    config = LSVFixedPointConfig(leverage_cap=1.4, maximum_iterations=3)
    fit = LSVFixedPointCalibrator(grid, config).calibrate(params, target, [8, 12])
    assert [len(s.iterations) for s in fit.slices] == [1, 2]
    assert len(fit.slices[0].accepted.diagnostics) == 16
    assert fit.slices[1].accepted.leverage_update.capped_mask.all()
    np.testing.assert_allclose(fit.model.leverage_surface.values[1], 1.4, atol=1e-12)
    chain = _quotes(target.times)
    local = LocalVolSurface(
        [0.06, 0.1, 0.2],
        grid.x,
        np.repeat([[0.10], [0.20], [0.2 * 0.8 * 1.4]], grid.x.size, axis=1),
    )
    expected = LocalVolPdePricer(grid=grid.x, steps_per_interval=32, theta=1.0).price_chain(
        chain, local
    )
    # The independent 1D price above uses equal time steps in every bucket.
    actual = LogSvPdePricer(grid=grid, steps_per_interval=32, quadratic_start=False).price_chain(
        chain, fit.model
    )
    np.testing.assert_allclose(list(actual), list(expected), rtol=0.0, atol=2e-11)


def _config(archived):
    settings = {
        f.name: getattr(archived, f.name) for f in fields(LSVFixedPointConfig)
        if f.name not in ("quadratic_start", "collocation")
    }
    settings["low_density_policy"] = LowDensityPolicy(archived.low_density_policy.value)
    return LSVFixedPointConfig(**settings, quadratic_start=False)


def _compare_history(fit, ref):
    np.testing.assert_allclose(
        fit.model.leverage_surface.values, ref.leverage_surface.values, rtol=0, atol=2e-11
    )
    assert len(fit.slices) == len(ref.slices)
    residual_error = 0.0
    for row, historic in zip(fit.slices, ref.slices):
        assert len(row.iterations) == len(historic.iterations)
        for trial, previous in zip(row.iterations, historic.iterations):
            assert trial.converged == previous.converged
            residual_error = max(
                residual_error,
                abs(trial.maximum_absolute_residual - previous.maximum_absolute_residual),
                abs(trial.maximum_relative_residual - previous.maximum_relative_residual),
            )
            proj = trial.projection
            np.testing.assert_allclose(
                proj.propagation_leverage, previous.candidate_leverage, rtol=0, atol=2e-11
            )
            np.testing.assert_allclose(
                proj.terminal_density.values,
                previous.terminal_density.values,
                rtol=2e-10,
                atol=2e-10,
            )
            for name in (
                "capped_mask",
                "extrapolated_mask",
                "effective_second_moment",
                "bounded_leverage",
                "accepted_leverage",
            ):
                np.testing.assert_allclose(
                    getattr(proj.leverage_update, name),
                    getattr(previous.leverage_update, name),
                    rtol=0,
                    atol=2e-11,
                )
        assert row.accepted is row.iterations[-1].projection
    assert residual_error < 2e-11
    return residual_error


def _price_errors(actual, reference, chain):
    return [
        float(np.max(abs(p - q)) / (f * d))
        for p, q, f, d in zip(actual, reference, chain.forwards, chain.discfactors)
    ]


def _record(record_property, fit, ref, residual_error, archive_errors, target_errors):
    for key, val in dict(
        iterations=[len(r.iterations) for r in fit.slices],
        fitted_leverage_max_error=float(
            np.max(abs(fit.model.leverage_surface.values - ref.leverage_surface.values))
        ),
        residual_history_max_error=float(residual_error),
        vanilla_archive_errors=archive_errors,
        vanilla_target_errors=target_errors,
        peak_negative_mass=fit.peak_negative_mass,
        max_trial_end_negative_mass=max(
            i.projection.conditional_moments.negative_density_mass
            for r in fit.slices
            for i in r.iterations
        ),
        max_accepted_absolute_residual=max(
            r.iterations[-1].maximum_absolute_residual for r in fit.slices
        ),
    ).items():
        record_property(key, val)


@pytest.mark.repository_only
@pytest.mark.slow
@pytest.mark.parametrize("stress", [False, True], ids=["time_refinement", "correlated_skew"])
@pytest.mark.pde_reference
def test_archived_convergence_surface_and_repricing(stress, record_property):
    from validation.pde_reference import lsv as old
    from validation.pde_reference.cases import (
        _on_the_fly_case,
        _on_the_fly_config,
    )

    times = np.array([0.1, 0.2, 0.3]) if stress else np.array([0.075, 0.15, 0.225, 0.30])
    grid, old_target, params, adi = _on_the_fly_case(
        times, beta=-0.20 if stress else 0.0, volvol=0.28 if stress else 0.18
    )
    cfg = (
        _on_the_fly_config(
            damping=0.6, maximum_iterations=18, relative_tolerance=2e-5, absolute_tolerance=2e-7
        )
        if stress
        else _on_the_fly_config(damping=0.7, maximum_iterations=10)
    )
    target = LocalVolSurface(times, grid.x, old_target.values)
    fitter = LSVFixedPointCalibrator(grid, _config(cfg), adi)
    archive = old.LSVOnTheFlyCalibrator(params, grid, old_target, cfg, adi)
    if not stress:
        coarse = fitter.calibrate(params, target, 16)
        old_coarse = archive.calibrate(16)
        _compare_history(coarse, old_coarse)
    fit = fitter.calibrate(params, target, 32)
    ref = archive.calibrate(32)
    residual_error = _compare_history(fit, ref)
    if not stress:
        change = np.max(
            abs(fit.model.leverage_surface.values - coarse.model.leverage_surface.values)
        )
        assert change < 4e-3
        record_property("time_refinement_max_leverage_change", float(change))
    chain = _quotes(times)
    steps = 40
    pricer = LogSvPdePricer(grid=grid, steps_per_interval=steps, adi=adi, quadratic_start=False)
    prices = pricer.price_chain(chain, fit.model)
    errors = _price_errors(
        prices, _archive_prices(chain, params, ref.leverage_surface, grid, steps, adi), chain
    )
    assert max(errors) < 2e-12
    target_prices = LocalVolPdePricer(grid=grid.x, steps_per_interval=steps, theta=1.0).price_chain(
        chain, target
    )
    target_errors = _price_errors(prices, target_prices, chain)
    if stress:
        lagged = old.SequentialLSVProjector(params, grid, old_target, cfg, adi).project(32)
        baseline = _price_errors(
            _archive_prices(chain, params, lagged.leverage_surface, grid, steps, adi),
            target_prices,
            chain,
        )
        assert target_errors[-1] < 1e-4 and target_errors[-1] < 0.75 * baseline[-1]
        record_property("lagged_target_last", baseline[-1])
    _record(record_property, fit, ref, residual_error, errors, target_errors)


@pytest.mark.repository_only
@pytest.mark.pde_reference
def test_high_correlation_rejects_unusable_trial_density():
    from validation.pde_reference import lsv as old
    from validation.pde_reference.cases import (
        _on_the_fly_case,
        _on_the_fly_config,
    )

    g, t, p, adi = _on_the_fly_case(np.array([0.15]), beta=-0.42, volvol=0.48)
    cfg = _on_the_fly_config(
        damping=0.5,
        maximum_iterations=2,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-12,
        fail_on_density_violation=False,
    )
    with pytest.raises(old.LSVConvergenceError) as historic:
        old.LSVOnTheFlyCalibrator(p, g, t, cfg, adi).calibrate(20)
    assert historic.value.iterations[0].terminal_density.negative_mass > 0.03
    with pytest.raises(ValueError, match="trial 1 at T=0.15.*negative mass"):
        LSVFixedPointCalibrator(g, _config(cfg), adi).calibrate(
            p, LocalVolSurface(t.times, g.x, t.values), 20
        )


@pytest.mark.repository_only
@pytest.mark.slow
@pytest.mark.pde_reference
def test_frozen_btc_fixed_point_history_and_49_vanillas(record_property):
    from validation.pde_reference import dupire as old_dupire, local_vol as old_local, lsv as old
    from validation.pde_reference.integration import LSVOptionChainAdapter
    from validation.pde_reference.market_pipeline import (
        btc_20211021_reference_config,
        load_frozen_btc_20211021_snapshot,
    )

    c = btc_20211021_reference_config()
    chain = load_frozen_btc_20211021_snapshot().to_option_chain()
    inputs = LSVOptionChainAdapter(chain, c.quote_selection).build_calibration_input(
        grid_size=c.calibration_grid_size
    )
    t = inputs.target
    vol = old_dupire.DupireRecoverer().continuous(
        old_dupire.CallSurface(t.times, t.strikes, t.target_calls, t.forward, t.discount_factor),
        repair_policy=c.dupire_repair_policy,
    )
    x = Grid1D(inputs.log_moneyness)
    old_target = old_local.LocalVolSurface(
        t.times, x, vol.local_volatility, old_local.CoordinateSystem.LOG_FORWARD
    )
    target = LocalVolSurface(t.times, x, vol.local_volatility)
    market = c.calibration_config
    cfg = old.LSVProjectionConfig(
        density_floor=1e-10,
        leverage_floor=market.leverage_floor,
        leverage_cap=market.leverage_cap,
        mass_tolerance=market.mass_tolerance,
        negative_mass_tolerance=market.negative_mass_tolerance,
        damping=0.7,
        maximum_iterations=20,
        relative_tolerance=2e-4,
        absolute_tolerance=2e-6,
    )
    g = c.calibration_grid
    ref = old.LSVOnTheFlyCalibrator(c.params, g, old_target, cfg, c.adi_config).calibrate(
        c.calibration_steps_per_interval
    )
    fit = LSVFixedPointCalibrator(g, _config(cfg), c.adi_config).calibrate(
        c.params, target, c.calibration_steps_per_interval
    )
    residual = _compare_history(fit, ref)
    g = c.repricing_grid
    expected = _archive_prices(chain, c.params, ref.leverage_surface, g, 96, c.adi_config)
    actual = LogSvPdePricer(
        grid=g, steps_per_interval=96, adi=c.adi_config, quadratic_start=False
    ).price_chain(chain, fit.model)
    errors = _price_errors(actual, expected, chain)
    assert sum(map(len, actual)) == 49 and max(errors) < 2e-12
    target_prices = LocalVolPdePricer(grid=g.x, steps_per_interval=96, theta=1.0).price_chain(
        chain, target
    )
    _record(
        record_property, fit, ref, residual, errors, _price_errors(actual, target_prices, chain)
    )
