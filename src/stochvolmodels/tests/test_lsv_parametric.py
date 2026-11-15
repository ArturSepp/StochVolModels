"""Parametric leverage calibration, density acceptance and the frozen BTC replay."""

from dataclasses import fields, replace
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import DupireRepairPolicy
from stochvolmodels.fitters.lsv_market import LSVOptionChainAdapter
from stochvolmodels.fitters.lsv_parametric import (
    CalibrationWeighting,
    ImpliedVolCalibrationTarget,
    LSVCalibrationError,
    LSVParametricCalibrationConfig,
    LSVParametricCalibrator,
    calibration_weights,
)
from stochvolmodels.models.local_vol import (
    LocalVolFamily,
    LocalVolSurface,
    ParametricLocalVolFormula,
)
from stochvolmodels.models.lsv import LSVModel, LeverageSurface
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer


def _flat_case():
    x = Grid1D.concentrated(-0.9, 0.9, 101, 0.0, 0.12)
    grid = TensorGrid2D(x, Grid1D([0.1, 0.15, 0.2, 0.25, 0.3]))
    p = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    vols = np.array([0.2, np.sqrt((0.2**2 + 0.26**2) / 2)])
    target = ImpliedVolCalibrationTarget(
        [0.1, 0.2], [94.0, 100.0, 106.0], np.repeat(vols[:, None], 3, axis=1), 100.0, 0.97
    )
    local = LocalVolSurface(target.times, x, np.repeat([[0.2], [0.26]], x.size, axis=1))
    return grid, p, target, local


def _calls(target):
    return OptionChain(
        ttms=target.times,
        forwards=np.full(len(target.times), target.forward),
        discfactors=np.full(len(target.times), target.discount_factor),
        strikes_ttms=tuple(target.strikes for t in target.times),
        optiontypes_ttms=tuple(np.full(len(target.strikes), "C") for t in target.times),
        ids=np.array([str(t) for t in target.times]),
    )


def test_flat_term_structure_and_discounted_repricing():
    grid, p, t, local = _flat_case()
    cfg = LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT)
    fit = LSVParametricCalibrator(grid, cfg).calibrate(
        p, t, local, [40, 60], initial_parameters=[[0.8], [0.9]]
    )
    assert fit.parameter_names == ("atm",)
    np.testing.assert_allclose(fit.parameter_values[:, 0], [1.0, 1.3], atol=0.009)
    assert fit.slices[1].start_density is fit.slices[0].final_evaluation.terminal_density
    prices = LogSvPdePricer(grid=grid, steps_per_interval=[40, 60]).price_chain(
        _calls(t), fit.model
    )
    for i, row in enumerate(fit.slices):
        ev = row.final_evaluation
        assert row.pde_replays > row.optimizer_evaluations_count
        assert ev.parameters[0] == fit.model.leverage_surface.values[i, 0]
        assert not ev.parameters.flags.writeable
        assert np.max(abs(ev.weighted_residuals)) < 3.5e-4
        np.testing.assert_allclose(prices[i], ev.model_calls, atol=5e-13, rtol=0)
        np.testing.assert_allclose(prices[i], t.target_calls[i], atol=0.007, rtol=0)


@pytest.mark.parametrize(
    ("family", "coefficients"),
    [
        (LocalVolFamily.SKEW, (1.7, 0.7)),
        (LocalVolFamily.SMILE, (1.1, 2.0)),
        (LocalVolFamily.PRODUCT, (1.7, 0.7, 2.0)),
        (LocalVolFamily.WING, (1.1, -0.1, 0.6, 0.15, 0.2)),
    ],
)
def test_family_replay_through_backbone_knot(family, coefficients):
    grid, params, _, _ = _flat_case()
    params = replace(params, vol_backbone=pd.Series([0.85, 1.1], index=[0.12, 0.25]))
    vols = ParametricLocalVolFormula(family, coefficients).evaluate(grid.x.nodes)
    planted = LeverageSurface([0.25], grid.x, vols[None, :])
    target = ImpliedVolCalibrationTarget([0.25], np.linspace(0.88, 1.12, 7), np.full((1, 7), 0.21))
    chain = _calls(target)
    prices = LogSvPdePricer(grid=grid, steps_per_interval=20).price_chain_detailed(
        chain, LSVModel.from_params(params, planted)
    )
    target = replace(target, implied_volatilities=np.vstack(prices.implied_vols))
    local = LocalVolSurface(target.times, grid.x, planted.values * (1.1 * params.sigma0))
    cfg = LSVParametricCalibrationConfig(family=family)
    fit = LSVParametricCalibrator(grid, cfg).calibrate(
        params, target, local, 20, initial_parameters=[coefficients]
    )
    np.testing.assert_allclose(fit.parameter_values[0], coefficients, atol=2e-7, rtol=0)
    np.testing.assert_allclose(fit.model.leverage_surface.values, planted.values, atol=2e-7)
    np.testing.assert_allclose(
        fit.slices[0].final_evaluation.model_calls, prices.prices[0], atol=1e-10
    )


def test_invalid_black_inversion_identifies_expiry_and_strike():
    grid, params, _, _ = _flat_case()
    target = ImpliedVolCalibrationTarget([0.2], [1.0, 4.0], [[0.2, 0.2]])
    local = LocalVolSurface([0.2], grid.x, np.full((1, grid.x.size), 0.2))
    cfg = LSVParametricCalibrationConfig(
        family=LocalVolFamily.FLAT, weighting=CalibrationWeighting.UNIFORM
    )
    with pytest.raises(LSVCalibrationError, match=r"T=0.2.*IV inversion failed at strikes .*4"):
        LSVParametricCalibrator(grid, cfg).calibrate(params, target, local)


@pytest.mark.parametrize("weighting", list(CalibrationWeighting))
def test_normalized_weights(weighting):
    w = calibration_weights(weighting, 0.3, 1.0, [0.9, 1.0, 1.1], [0.3, 0.3, 0.3])
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0) and not w.flags.writeable
    if weighting is CalibrationWeighting.LEGACY_ATM:
        assert w[1] > 0.97


def test_fit_rejects_invalid_starts_and_failed_optimizers():
    grid, p, t, local = _flat_case()
    cfg = LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT)
    fitter = LSVParametricCalibrator(grid, cfg)
    for initial in ([[1.0]], [[-1.0], [1.0]], [[1.0], [np.nan]]):
        with pytest.raises(ValueError, match="initial_parameters"):
            fitter.calibrate(p, t, local, initial_parameters=initial)
    with pytest.raises(LSVCalibrationError, match="T=0.1.*optimizer failed"):
        replace(fitter, config=replace(cfg, maximum_evaluations=1)).calibrate(
            p, t, local, initial_parameters=[[0.75], [1.0]]
        )
    with pytest.raises(ValueError, match="positive integers|shape"):
        fitter.calibrate(p, t, local, [2, 0])


@pytest.mark.slow
@pytest.mark.repository_only
@pytest.mark.parametrize("adverse", [False, True], ids=["baseline", "adverse"])
def test_frozen_btc_parameters_surface_and_vanillas(adverse, record_property):
    if find_spec("stochvolmodels.pde_solvers.market_pipeline") is None:
        pytest.skip("requires the archived BTC market pipeline in a source checkout")
    from stochvolmodels.pde_solvers.density import DensityPricer
    from stochvolmodels.pde_solvers.integration import LSVOptionChainPricer
    from stochvolmodels.pde_solvers.lsv import LeverageSurface as ArchivedLeverage
    from stochvolmodels.pde_solvers.lsv_mc import LSVMonteCarloPricer
    from stochvolmodels.pde_solvers.market_pipeline import (
        LSVMarketPipeline,
        btc_20211021_reference_config,
        load_frozen_btc_20211021_snapshot,
    )
    from stochvolmodels.pde_solvers.stress import difficult_btc_initial_parameters

    c = btc_20211021_reference_config()
    if adverse:
        c = replace(
            c,
            calibration_initial_parameters=difficult_btc_initial_parameters(),
            calibration_config=replace(c.calibration_config, maximum_evaluations=24),
        )
    ref = LSVMarketPipeline(c).run(load_frozen_btc_20211021_snapshot())
    chain = ref.snapshot.to_option_chain()
    market = LSVOptionChainAdapter(chain).build_calibration_input(grid_size=c.calibration_grid_size)
    target = market.target
    dupire = market.recover_dupire(
        repair_policy=DupireRepairPolicy(
            c.dupire_repair_policy.variance_floor, c.dupire_repair_policy.variance_cap
        ),
    )
    local = LocalVolSurface(target.times, Grid1D(market.log_moneyness), dupire.local_volatility)
    np.testing.assert_allclose(local.values, ref.prefit_local_volatility.values, atol=1e-13, rtol=0)
    old = c.calibration_config
    settings = {f.name: getattr(old, f.name) for f in fields(LSVParametricCalibrationConfig)}
    settings.update(
        family=LocalVolFamily(old.family.value), weighting=CalibrationWeighting(old.weighting.value)
    )
    cfg = LSVParametricCalibrationConfig(**settings)
    fitter = LSVParametricCalibrator(c.calibration_grid, cfg, c.adi_config)
    args = (c.params, target, local, c.calibration_steps_per_interval)
    starts = dict(initial_parameters=c.calibration_initial_parameters)
    # The archived fit disabled candidate-density checks. Its third expiry breaches
    # 2e-5 even for the normal start. Keep that strict failure reproducible.
    with pytest.raises(LSVCalibrationError, match=f"T={target.times[2]:g}.*negative mass"):
        fitter.calibrate(*args, **starts)
    trial_limit = 1.5e-4 if adverse else 3e-5
    fit = replace(fitter, config=replace(cfg, negative_mass_tolerance=trial_limit)).calibrate(
        *args, **starts
    )
    expected = ref.calibration_result
    surface = fit.model.leverage_surface
    param_error = np.max(abs(fit.parameter_values - expected.parameter_values), axis=1)
    leverage_error = np.max(abs(surface.values - expected.leverage_surface.values), axis=1)
    prefit_error = np.max(
        [
            abs(row.prefit_parameters - r.prefit.parameters)
            for row, r in zip(fit.slices, expected.slices)
        ]
    )
    coarse_calls_error = np.max(
        [
            abs(row.final_evaluation.model_calls - r.final_evaluation.model_calls)
            for row, r in zip(fit.slices, expected.slices)
        ]
    )
    pricer = LogSvPdePricer(
        grid=c.repricing_grid, steps_per_interval=c.repricing_steps_per_interval, adi=c.adi_config
    )
    normalized_chain = _calls(target)
    fine = pricer.price_chain_detailed(normalized_chain, fit.model)
    iv = np.vstack(fine.implied_vols)
    assert np.isfinite(iv).all()
    fine_call_error = np.max(abs(np.vstack(fine.prices) - ref.pde_repricing.model_calls), axis=1)
    fine_iv_error = np.max(abs(iv - ref.pde_repricing.model_implied_volatilities), axis=1)
    original = pricer.price_chain_detailed(chain, fit.model)
    ragged = original.prices
    quote_errors = []
    quotes = []
    for i, sl in enumerate(ref.pde_repricing.slices):
        f, d = chain.forwards[i], chain.discfactors[i]
        oracle = DensityPricer(sl.density.marginal_x_state(), forward=f, discount_factor=d)
        pvs = oracle.price_vanillas(chain.strikes_ttms[i], chain.optiontypes_ttms[i])
        quote_errors.append(float(np.max(abs(ragged[i] - pvs)) / (f * d)))
        simulated = ref.option_chain_pricing.slices[i]
        mc_se = simulated.standard_errors
        assert not original.invalid_iv[i].any() and not simulated.model_price_clipped_mask.any()
        assert np.all(mc_se > 0)
        gap = np.abs(ragged[i] - simulated.model_prices)
        if not adverse:
            # Same fine PDE grid and pointwise MC check as the frozen-engine replay.
            assert np.all(gap < 5 * mc_se)
        quotes.append(
            dict(
                expiry=market.source_slices[i].slice_id,
                forward=float(f),
                discount=float(d),
                strike=chain.strikes_ttms[i].tolist(),
                option_type=chain.optiontypes_ttms[i].tolist(),
                public_pv=ragged[i].tolist(),
                archived_pv=pvs.tolist(),
                archived_mc_pv=simulated.model_prices.tolist(),
                archived_mc_se=mc_se.tolist(),
                max_mc_se=float(np.max(gap / mc_se)),
                inside_mc_95=int(np.sum(gap <= 1.95996398454 * mc_se)),
                inside_market_spread=int(
                    np.sum(
                        (chain.bid_ivs[i] <= original.implied_vols[i])
                        & (original.implied_vols[i] <= chain.ask_ivs[i])
                    )
                ),
            )
        )
    assert sum(map(len, ragged)) == 49 and iv.shape == (4, 11)
    # Run the independent historical simulator on the newly fitted leverage rows.
    mc = LSVOptionChainPricer(
        LSVMonteCarloPricer(
            c.params,
            ArchivedLeverage(
                surface.times, surface.grid, surface.values, cfg.leverage_floor, cfg.leverage_cap
            ),
            c.monte_carlo_config,
        )
    ).price(normalized_chain)
    mc_iv = np.vstack([row.model_implied_volatilities for row in mc.slices])

    bids, asks = market.bid_ivs, market.ask_ivs
    inside = int(np.count_nonzero((bids <= iv) & (iv <= asks)))
    target_rmse = np.sqrt(np.mean((iv - target.implied_volatilities) ** 2, axis=1))
    mc_rmse = np.sqrt(np.mean((iv - mc_iv) ** 2, axis=1))
    acceptance = replace(
        ref.acceptance,
        pde_target_iv_rmse=target_rmse,
        pde_mc_iv_rmse=mc_rmse,
        pde_mass_errors=[abs(row.mass - 1) for row in fine.expiries],
        pde_negative_masses=[row.negative_mass for row in fine.expiries],
        mc_forward_z_scores=[row.forward_martingale_z_score for row in mc.slices],
        market_spread_coverage=inside / iv.size,
    )
    for name, value in dict(
        snapshot=ref.snapshot.fingerprint,
        expiries=[s.slice_id for s in market.source_slices],
        original_quotes=quotes,
        parameters=fit.parameter_values.tolist(),
        archive_parameters=expected.parameter_values.tolist(),
        parameter_error=param_error.tolist(),
        leverage_error=leverage_error.tolist(),
        prefit_error=float(prefit_error),
        coarse_call_error=float(coarse_calls_error),
        fine_call_error=fine_call_error.tolist(),
        fine_iv_error=fine_iv_error.tolist(),
        quote_error_over_discounted_forward=quote_errors,
        optimizer_evaluations=[r.optimizer_evaluations_count for r in fit.slices],
        pde_replays=[r.pde_replays for r in fit.slices],
        trial_negative_mass_limit=trial_limit,
        peak_candidate_negative_mass=max(r.peak_candidate_negative_mass for r in fit.slices),
        fine_negative_mass=max(row.negative_mass for row in fine.expiries),
        inside_spread=inside,
        target_iv_rmse=target_rmse.tolist(),
        mc_iv_rmse=mc_rmse.tolist(),
        failed_checks=list(acceptance.failed_checks),
    ).items():
        record_property(name, value)
    assert acceptance.failed_checks == ref.acceptance.failed_checks
    assert acceptance.passed == (not adverse)
    assert inside == (12 if adverse else 35)
    assert all(r.peak_candidate_negative_mass <= trial_limit for r in fit.slices)
    np.testing.assert_allclose(
        [r.optimizer_evaluations_count for r in fit.slices],
        [r.optimizer_evaluations_count for r in expected.slices],
        rtol=0,
        atol=0,
    )
    assert prefit_error < 1e-8
    assert [r.pde_replays for r in fit.slices] == [
        len(r.optimizer_evaluations) + 1 for r in expected.slices
    ]
    assert max(param_error) < 1e-8 and max(leverage_error) < 1e-8
    assert coarse_calls_error < 2e-10 and max(fine_call_error) < 2e-10
    assert max(quote_errors) < 2e-10 and max(fine_iv_error) < 5e-9
