"""Conditional variance, lagged leverage and vanilla pricing against analytic and archived laws."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numba.typed import List

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.local_vol import LocalVolSurface
from stochvolmodels.models.lsv import LSVModel, LeverageExtrapolation, LeverageSurface
from stochvolmodels.pde_solvers import ADIConfig, CoefficientEvaluation, Grid1D, TensorGrid2D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.lsv_projection import (
    LSVProjectionConfig,
    LowDensityPolicy,
    SequentialLSVProjector,
    project_conditional_variance,
    recover_leverage_slice,
)


def _quotes(times):
    f = 100 * np.exp(0.035 * times)
    ratios = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
    strikes = List([level * np.tile(ratios, 2) for level in f])
    types = List([np.array(list("CCCCCPPPPP")) for _ in f])
    return OptionChain(
        ttms=times,
        ids=None,
        forwards=f,
        discfactors=np.exp(-0.02 * times),
        strikes_ttms=strikes,
        optiontypes_ttms=types,
    )


def test_quadratic_start_at_zero_only_in_pricing_and_projection():
    times = np.array([0.125, 0.25])
    chain = _quotes(times)
    grid = TensorGrid2D(Grid1D.uniform(-0.8, 0.8, 45), Grid1D.uniform(0.0, 0.4, 9))
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    pricer = LogSvPdePricer(grid=grid, steps_per_interval=[4, 2])
    plain = pricer.price_chain_detailed(chain, params)
    target = LocalVolSurface(times, grid.x, np.full((2, grid.x.size), 0.2))
    fit = SequentialLSVProjector(grid).project(params, target, [4, 2])
    expected = [1 / 128, 4 / 128, 9 / 128, 0.125, 0.1875, 0.25]
    np.testing.assert_array_equal([d.time_end for d in plain.steps], expected)
    np.testing.assert_array_equal(
        [d.time_end for row in fit.slices for d in row.diagnostics], expected
    )
    # With a backbone knot before the first expiry, grade up to the first actual bucket end.
    params.vol_backbone = pd.Series([1.0, 1.25], index=[0.0625, 0.25])
    with_knot = SequentialLSVProjector(grid).project(params, target, [4, 2])
    replay = pricer.price_chain_detailed(chain, with_knot.model)
    ends = [1 / 256, 4 / 256, 9 / 256, 0.0625, 5 / 64, 6 / 64, 7 / 64, 0.125, 0.1875, 0.25]
    np.testing.assert_array_equal([d.time_end for d in replay.steps], ends)
    np.testing.assert_array_equal(
        [d.time_end for row in with_knot.slices for d in row.diagnostics], ends
    )


def test_projection_quadrature_repair_damping_and_density_gates():
    grid = TensorGrid2D(Grid1D([-1, -0.8, -0.6, 0.3, 1]), Grid1D([0.10, 0.23, 0.62]))
    # Unit row masses at indices 1 and 3, carried by distinct volatility nodes.
    units = np.zeros(grid.shape)
    units[1, 0], units[3, 2] = 0.35, 0.65
    density = units / grid.quadrature_weights
    cfg = LSVProjectionConfig(leverage_floor=0.2, leverage_cap=1.5, damping=0.25)
    m = project_conditional_variance(grid, density, cfg)
    np.testing.assert_array_equal(m.invalid_moment_mask, [1, 0, 1, 0, 1])
    np.testing.assert_allclose(m.conditional_second_moment[[1, 3]], [0.1**2, 0.62**2])
    np.testing.assert_allclose(m.marginal * grid.x.quadrature_weights, units.sum(axis=1))
    assert m.total_mass == pytest.approx(1.0) and m.negative_density_mass == 0
    assert not m.marginal.flags.writeable
    u = recover_leverage_slice([0.35] * 5, m, np.ones(5), cfg, volatility_scale=0.8)
    moments = np.array([0.01, 0.01, (0.01 + 0.62**2) / 2, 0.62**2, 0.62**2])
    np.testing.assert_allclose(u.effective_second_moment, moments)
    expected = 0.35 / (0.8 * np.sqrt(moments))
    np.testing.assert_array_equal(u.capped_mask, expected > 1.5)
    np.testing.assert_allclose(u.accepted_leverage, 0.75 + 0.25 * np.clip(expected, 0.2, 1.5))
    assert np.isnan(u.raw_leverage[u.extrapolated_mask]).all()
    with pytest.raises(ValueError, match="conditional moment"):
        recover_leverage_slice(
            np.ones(5), m, np.ones(5), replace(cfg, low_density_policy=LowDensityPolicy.RAISE)
        )
    with pytest.raises(ValueError, match="mass"):
        project_conditional_variance(grid, 0.99 * density)
    units[1, 0] += 0.03
    units[2, 1] = -0.03
    with pytest.raises(ValueError, match="negative mass"):
        project_conditional_variance(grid, units / grid.quadrature_weights)
    relaxed = replace(cfg, negative_mass_tolerance=0.04)
    signed = project_conditional_variance(grid, units / grid.quadrature_weights, relaxed)
    with pytest.raises(ValueError, match="negative mass"):
        recover_leverage_slice(np.ones(5), signed, np.ones(5), cfg)


def test_lsv_generator_time_buckets_and_snapshot():
    x = Grid1D([-0.5, 0.0, 0.7])
    grid = TensorGrid2D(x, Grid1D([0.1, 0.2, 0.4]))
    params = LogSvParams(
        beta=-0.3, volvol=0.4, vol_backbone=pd.Series([0.8, 1.3], index=[0.1, 0.3])
    )
    s = LeverageSurface([0.2, 0.4], x, [[0.4, 0.75, 1.2], [1.3, 0.9, 0.65]])
    model = LSVModel.from_params(params, s)
    params.vol_backbone.iloc[0] = 12.0
    params.beta = 2.0
    np.testing.assert_array_equal(model.time_knots, [0.1, 0.2, 0.3, 0.4])
    for t, eta, row in [(0.1, 0.8, 0), (0.2, 1.3, 0), (0.20001, 1.3, 1), (0.5, 1.3, 1)]:
        c = model.evaluate(t, grid)
        a = eta * s.values[row, :, None]
        v = grid.y_mesh**2
        np.testing.assert_allclose(c.variance_x, 0.5 * a * a * v)
        np.testing.assert_allclose(c.drift_x, -c.variance_x)
        np.testing.assert_allclose(c.covariance_xy, -0.3 * a * v)
        np.testing.assert_allclose(c.variance_y, 0.5 * (0.3**2 + 0.4**2) * v)
        np.testing.assert_allclose(
            c.drift_y, (params.kappa1 + params.kappa2 * grid.y_mesh) * (params.theta - grid.y_mesh)
        )
    strict = replace(s, extrapolation=LeverageExtrapolation.RAISE)
    with pytest.raises(ValueError, match="outside"):
        strict.evaluate(0.01, [0.9])
    with pytest.raises(ValueError, match="kappa2"):
        LSVModel.from_params(params, s).evaluate(0.01, grid)
    with pytest.raises(ValueError, match="increasing"):
        LSVModel.from_params(
            replace(params, vol_backbone=pd.Series([0.8, 1.1], index=[0.2, 0.1])), s
        )


def test_interior_backbone_and_leverage_knots_against_independent_1d_law():
    x = Grid1D.concentrated(-1.25, 1.25, 101, 0.0, 0.12)
    grid = TensorGrid2D(x, Grid1D([0.1, 0.15, 0.2, 0.25, 0.30]))
    params = LogSvParams(
        sigma0=0.2,
        theta=0.2,
        beta=0.0,
        volvol=0.0,
        vol_backbone=pd.Series([0.6, 1.2], index=[0.07, 0.26]),
    )
    surface = LeverageSurface(
        [0.12, 0.29], x, np.array([0.8, 1.35])[:, None] * np.ones((2, x.size))
    )
    model = LSVModel.from_params(params, surface)
    ts = np.array([0.07, 0.12, 0.26, 0.29])
    vols = np.array([0.6 * 0.8, 1.2 * 0.8, 1.2 * 1.35, 1.2 * 1.35]) * 0.2
    law = LocalVolSurface(ts, x, vols[:, None] * np.ones((len(ts), x.size)))
    chain = _quotes(np.array([0.1, 0.31]))
    ref = LocalVolPdePricer(grid=x, steps_per_interval=32, theta=1.0).price_chain(chain, law)
    adi = ADIConfig(theta=1.0, coefficient_evaluation=CoefficientEvaluation.STEP_START)
    # Match the uniform time grid of the independent 1D discretization.
    result = LogSvPdePricer(
        grid=grid, steps_per_interval=32, adi=adi, quadratic_start=False
    ).price_chain_detailed(chain, model)
    np.testing.assert_allclose(list(result.prices), list(ref), rtol=0.0, atol=2e-11)
    assert len(result.steps) == 32 * 6
    assert [s.step_index for s in result.steps] == list(range(len(result.steps)))
    assert max(abs(e.normalized_forward - 1.0) for e in result.expiries) < 3e-6
    # Recover with a backbone change strictly inside the first target interval.
    target = LocalVolSurface([0.12, 0.29], x, np.full((2, x.size), 0.18))
    fit = SequentialLSVProjector(grid, adi=adi).project(params, target, 32)
    np.testing.assert_allclose(fit.model.leverage_surface.values, 0.18 / (1.2 * 0.2), atol=1e-12)
    assert len(fit.slices[0].diagnostics) == 64


def _require_archive():
    if not (Path(__file__).resolve().parents[1] / "pde_solvers" / "lsv.py").is_file():
        pytest.skip("requires the archived LSV projection and density pricer")


def _archive_prices(chain, params, surface, grid, steps, adi):
    from stochvolmodels.pde_solvers import (
        BoundaryPair1D,
        BoundarySet2D,
        PDE2DADISolver,
        PDEDirection,
    )
    from stochvolmodels.pde_solvers import lsv as old
    from stochvolmodels.pde_solvers.density import DensityPricer, DensityState2D, CoordinateSystem

    times = chain.ttms
    edges = np.r_[
        0.0,
        np.concatenate(
            [np.linspace(a, b, steps + 1)[1:] for a, b in zip(np.r_[0.0, times[:-1]], times)]
        ),
    ]
    natural = BoundaryPair1D.natural()
    law = PDE2DADISolver(adi).solve(
        grid,
        old.LSVCoefficientProvider(params, surface),
        BoundarySet2D(natural, natural),
        PDEDirection.FORWARD,
        edges,
        grid.project_delta(0.0, params.sigma0),
        save_times=tuple(times),
    )
    expected = []
    for i, s in enumerate(law.saved_states):
        marginal = DensityState2D(
            grid, s.values, s.time, CoordinateSystem.LOG_FORWARD
        ).marginal_x_state()
        expected.append(
            DensityPricer(marginal, chain.forwards[i], chain.discfactors[i]).price_vanillas(
                chain.strikes_ttms[i], chain.optiontypes_ttms[i]
            )
        )
    return expected


@pytest.mark.repository_only
@pytest.mark.parametrize("stochastic", [False, True], ids=["deterministic", "stochastic"])
def test_archived_sequential_leverage_and_vanilla_replay(stochastic, record_property):
    _require_archive()
    from stochvolmodels.pde_solvers import local_vol as old_local
    from stochvolmodels.pde_solvers import lsv as old
    from stochvolmodels.pde_solvers.density import CoordinateSystem

    x = Grid1D.concentrated(-0.75, 0.75, 101, 0.0, 0.08)
    if stochastic:
        y = Grid1D.concentrated(0.035, 0.55, 61, 0.2, 0.025)
        times = np.linspace(0.025, 0.3, 12)
        vol = 0.2 * (1 - 0.12 * np.tanh(2 * x.nodes))
        params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=4.0, kappa2=10.0, beta=0.0, volvol=0.23)
        settings = dict(
            density_floor=1e-10,
            leverage_floor=0.1,
            leverage_cap=2.0,
            mass_tolerance=2e-9,
            negative_mass_tolerance=2e-7,
        )
        steps = 10
    else:
        y = Grid1D([0.12, 0.16, 0.2, 0.24, 0.28])
        times = np.array([0.1, 0.2, 0.3])
        vol = 0.2 * (1 - 0.18 * np.tanh(2 * x.nodes) + 0.04 * x.nodes**2)
        params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=5.0, beta=0.0, volvol=0.0)
        settings = dict(mass_tolerance=2e-11, negative_mass_tolerance=2e-10)
        steps = 40
    grid = TensorGrid2D(x, y)
    vals = np.tile(vol, (len(times), 1))
    target = LocalVolSurface(times, x, vals)
    adi = ADIConfig(theta=1.0)
    old_target = old_local.LocalVolSurface(times, x, vals, CoordinateSystem.LOG_FORWARD)
    ref = old.SequentialLSVProjector(
        params, grid, old_target, old.LSVProjectionConfig(**settings), adi
    ).project(steps)
    config = LSVProjectionConfig(**settings, quadratic_start=False)
    new = SequentialLSVProjector(grid, config, adi).project(params, target, steps)
    surf = new.model.leverage_surface
    err_leverage = np.max(np.abs(surf.values - ref.leverage_surface.values))
    assert err_leverage < 2e-11
    if not stochastic:
        np.testing.assert_allclose(surf.values, vals / params.sigma0, rtol=0, atol=3e-12)
    err_moment = 0.0
    for row, original in zip(new.slices, ref.slices):
        m, expected = row.conditional_moments, original.conditional_moments
        for name in (
            "marginal",
            "second_moment_numerator",
            "conditional_second_moment",
            "invalid_moment_mask",
            "low_density_mask",
        ):
            np.testing.assert_allclose(
                getattr(m, name), getattr(expected, name), rtol=2e-11, atol=1e-12
            )
        err_moment = max(
            err_moment,
            np.nanmax(abs(m.conditional_second_moment - expected.conditional_second_moment)),
        )
        np.testing.assert_allclose(
            row.terminal_density.values, original.terminal_density.values, rtol=2e-10, atol=1e-11
        )
        assert abs(m.total_mass - 1) <= settings["mass_tolerance"]
        assert m.negative_density_mass <= settings["negative_mass_tolerance"]
        for name in ("capped_mask", "extrapolated_mask"):
            np.testing.assert_array_equal(
                getattr(row.leverage_update, name), getattr(original.leverage_update, name)
            )
    chain = _quotes(times)
    replay = LogSvPdePricer(grid=grid, steps_per_interval=steps, adi=adi, quadratic_start=False)
    prices = np.array(list(replay.price_chain(chain, new.model)))
    expected = _archive_prices(chain, params, ref.leverage_surface, grid, steps, adi)
    numeraire = (chain.forwards * chain.discfactors)[:, None]
    price_error = np.max(np.abs(prices - expected) / numeraire)
    assert price_error < 2e-12
    local_pricer = LocalVolPdePricer(grid=x, steps_per_interval=steps, theta=1.0)
    local_prices = np.array(list(local_pricer.price_chain(chain, target)))
    residual = np.abs(prices - local_prices) / numeraire
    if stochastic:
        assert max(residual[-1]) < 1.5e-5
    else:
        assert residual.max() < 2e-10
    for key, value in dict(
        maturities=len(times),
        nodes=surf.values.size,
        vanillas=prices.size,
        leverage_error=err_leverage,
        conditional_variance_error=err_moment,
        normalized_price_error=price_error,
        local_vol_target_max=residual.max(),
        local_vol_target_last=residual[-1].max(),
        peak_negative_mass=new.peak_negative_mass,
        projection_max_negative_mass=max(
            r.conditional_moments.negative_density_mass for r in new.slices
        ),
    ).items():
        record_property(key, float(value))


@pytest.mark.repository_only
def test_frozen_btc_projection_dates_and_ragged_vanillas(record_property):
    _require_archive()
    from stochvolmodels.fitters.dupire import CallSurface, DupireRecoverer, DupireRepairPolicy
    from stochvolmodels.pde_solvers import dupire as old_dupire
    from stochvolmodels.pde_solvers import local_vol as old_local
    from stochvolmodels.pde_solvers import lsv as old
    from stochvolmodels.pde_solvers.integration import LSVOptionChainAdapter
    from stochvolmodels.pde_solvers.market_pipeline import (
        btc_20211021_reference_config,
        load_frozen_btc_20211021_snapshot,
    )

    reference = btc_20211021_reference_config()
    chain = load_frozen_btc_20211021_snapshot().to_option_chain()
    inputs = LSVOptionChainAdapter(chain, reference.quote_selection).build_calibration_input(
        grid_size=reference.calibration_grid_size
    )
    t = inputs.target
    quotes = (t.times, t.strikes, t.target_calls, t.forward, t.discount_factor)
    policy = reference.dupire_repair_policy
    target = (
        DupireRecoverer()
        .continuous(
            CallSurface(*quotes),
            repair_policy=DupireRepairPolicy(policy.variance_floor, policy.variance_cap),
        )
        .to_local_vol_surface(historical_endpoints=True)
    )
    archived = old_dupire.DupireRecoverer().continuous(
        old_dupire.CallSurface(*quotes), repair_policy=policy
    )
    old_target = old_local.LocalVolSurface(
        t.times,
        Grid1D(inputs.log_moneyness),
        archived.local_volatility,
        old_local.CoordinateSystem.LOG_FORWARD,
    )
    config = reference.calibration_config
    settings = dict(
        density_floor=1e-10,
        leverage_floor=config.leverage_floor,
        leverage_cap=config.leverage_cap,
        mass_tolerance=config.mass_tolerance,
        negative_mass_tolerance=config.negative_mass_tolerance,
    )
    grid, adi, params = reference.calibration_grid, reference.adi_config, reference.params
    counts = reference.calibration_steps_per_interval
    orig = old.SequentialLSVProjector(
        params, grid, old_target, old.LSVProjectionConfig(**settings), adi
    ).project(counts)
    proj = SequentialLSVProjector(grid, LSVProjectionConfig(**settings, quadratic_start=False), adi)
    fit = proj.project(params, target, counts)
    np.testing.assert_allclose(
        fit.model.leverage_surface.values, orig.leverage_surface.values, rtol=0, atol=2e-11
    )
    for row, old_row in zip(fit.slices, orig.slices):
        m = row.conditional_moments
        np.testing.assert_allclose(
            m.conditional_second_moment,
            old_row.conditional_moments.conditional_second_moment,
            rtol=0,
            atol=1e-11,
        )
        np.testing.assert_array_equal(
            m.invalid_moment_mask, old_row.conditional_moments.invalid_moment_mask
        )
        assert abs(m.total_mass - 1) <= config.mass_tolerance
        assert m.negative_density_mass <= config.negative_mass_tolerance
    assert fit.peak_negative_mass > config.negative_mass_tolerance
    strict = replace(proj, config=replace(proj.config, negative_mass_tolerance=1e-10))
    with pytest.raises(ValueError, match="projection at T=.*negative mass"):
        strict.project(params, target, counts)
    g = reference.repricing_grid
    expected = _archive_prices(chain, params, orig.leverage_surface, g, 96, adi)
    replay = LogSvPdePricer(grid=g, steps_per_interval=96, adi=adi, quadratic_start=False)
    actual = replay.price_chain(chain, fit.model)
    error = max(
        np.max(abs(p - q)) / (f * d)
        for p, q, f, d in zip(actual, expected, chain.forwards, chain.discfactors)
    )
    assert error < 2e-12
    record_property("btc_discounted_forward_error", float(error))
    record_property("btc_peak_negative_mass", fit.peak_negative_mass)
