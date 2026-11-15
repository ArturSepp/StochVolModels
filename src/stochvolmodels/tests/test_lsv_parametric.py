"""Parametric leverage calibration, density acceptance and the frozen BTC replay."""

from dataclasses import fields, replace
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
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


@pytest.fixture(scope="module", params=[False, True], ids=["baseline", "adverse"])
def btc_replay(request):
    """Run the tracked market replay once per start, including in a clean checkout."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "examples/calibration").is_dir():
        pytest.skip("the frozen market replay is repository-only")
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(root))
        from examples.calibration.frozen_btc_lsv import replay_btc

        adverse = request.param
        yield adverse, replay_btc(adverse=adverse), replay_btc


@pytest.mark.slow
@pytest.mark.repository_only
def test_btc_market_replay_matches_frozen_prices_and_decisions(btc_replay):
    adverse, result, replay = btc_replay
    expected = (
        ("pde_target_iv", "pde_mc_iv", "pde_negative_mass", "market_spread") if adverse else ()
    )
    assert result.acceptance.failed_checks == expected
    assert result.acceptance.market_spread_coverage == (12 if adverse else 35) / 44
    assert sum(r.strikes.size for r in result.market.source_slices) == 49
    # Archive PVs of the nearest-to-forward original contract at each of the four expiries.
    prices = []
    for row, quote in zip(result.original, result.market.source_slices, strict=True):
        i = np.argmin(abs(quote.strikes - quote.forward))
        prices.append(
            np.array([row.pde_prices[i], row.mc_prices[i]])
            / (quote.discount_factor * quote.forward)
        )
    reference = (
        [
            [0.064348140889, 0.064539607411],
            [0.091141506573, 0.096560975329],
            [0.153435721892, 0.156106899007],
            [0.212433379975, 0.226311897749],
        ]
        if adverse
        else [
            [0.064348561098, 0.064540030038],
            [0.101714744921, 0.101201578350],
            [0.155083774151, 0.154103152815],
            [0.207226778182, 0.206519587955],
        ]
    )
    np.testing.assert_allclose(prices, reference, rtol=0, atol=1e-6)
    chain = get_btc_test_chain_data()
    with pytest.raises(LSVCalibrationError, match=f"T={chain.ttms[2]:g}.*negative mass"):
        replay(adverse=adverse, candidate_negative_mass_limit=2e-5)


@pytest.mark.slow
@pytest.mark.repository_only
def test_frozen_btc_parameters_surface_and_vanillas(btc_replay, record_property):
    adverse, result, _ = btc_replay
    if find_spec("stochvolmodels.pde_solvers.market_pipeline") is None:
        pytest.skip("requires the archived BTC market pipeline in a source checkout")
    from stochvolmodels.pde_solvers.density import DensityPricer
    from stochvolmodels.pde_solvers.market_pipeline import (
        LSVMarketPipeline as ArchivedPipeline,
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
    ref = ArchivedPipeline(c).run(load_frozen_btc_20211021_snapshot())
    chain = ref.snapshot.to_option_chain()
    trial_limit = 1.5e-4 if adverse else 3e-5
    fit, market = result.calibration, result.market
    np.testing.assert_array_equal(result.dupire.repaired_mask, ref.dupire_result.repaired_mask)
    np.testing.assert_allclose(
        result.prefit_local_volatility.values,
        ref.prefit_local_volatility.values,
        atol=1e-13,
        rtol=0,
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
    iv = np.vstack([s.pde_implied_vols for s in result.normalized])
    fine_call_error = np.max(
        abs(np.vstack([s.pde_prices for s in result.normalized]) - ref.pde_repricing.model_calls),
        axis=1,
    )
    fine_iv_error = np.max(abs(iv - ref.pde_repricing.model_implied_volatilities), axis=1)
    ragged = [s.pde_prices for s in result.original]
    mc_price_errors = []
    mc_se_errors = []
    for rows, oracle, scales in (
        (
            result.normalized,
            ref.monte_carlo_validation.normalized_pricing,
            np.ones(len(chain.ttms)),
        ),
        (result.original, ref.option_chain_pricing, chain.forwards * chain.discfactors),
    ):
        for row, old_row, scale in zip(rows, oracle.slices, scales, strict=True):
            assert not old_row.model_price_clipped_mask.any()
            mc_price_errors.append(float(np.max(abs(row.mc_prices - old_row.model_prices)) / scale))
            mc_se_errors.append(
                float(np.max(abs(row.mc_standard_errors - old_row.standard_errors)) / scale)
            )
            np.testing.assert_allclose(
                row.mc_implied_vols, old_row.model_implied_volatilities, atol=5e-9, rtol=0
            )
    assert max(mc_price_errors) < 2e-10 and max(mc_se_errors) < 2e-10
    quote_errors = []
    quotes = []
    for i, sl in enumerate(ref.pde_repricing.slices):
        f, d = chain.forwards[i], chain.discfactors[i]
        oracle = DensityPricer(sl.density.marginal_x_state(), forward=f, discount_factor=d)
        pvs = oracle.price_vanillas(chain.strikes_ttms[i], chain.optiontypes_ttms[i])
        quote_errors.append(float(np.max(abs(ragged[i] - pvs)) / (f * d)))
        simulated = ref.option_chain_pricing.slices[i]
        mc_se = simulated.standard_errors
        assert (
            np.isfinite(result.original[i].pde_implied_vols).all()
            and not simulated.model_price_clipped_mask.any()
        )
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
                public_mc_pv=result.original[i].mc_prices.tolist(),
                public_mc_se=result.original[i].mc_standard_errors.tolist(),
                archived_mc_se=mc_se.tolist(),
                max_mc_se=float(np.max(gap / mc_se)),
                inside_mc_95=int(np.sum(gap <= 1.95996398454 * mc_se)),
                inside_market_spread=int(
                    np.sum(
                        (chain.bid_ivs[i] <= result.original[i].pde_implied_vols)
                        & (result.original[i].pde_implied_vols <= chain.ask_ivs[i])
                    )
                ),
            )
        )
    assert sum(map(len, ragged)) == 49 and iv.shape == (4, 11)
    inside = int(np.sum((market.bid_ivs <= iv) & (iv <= market.ask_ivs)))
    acceptance = result.acceptance
    assert acceptance.checks == ref.acceptance.checks
    expected_failures = (
        ("pde_target_iv", "pde_mc_iv", "pde_negative_mass", "market_spread") if adverse else ()
    )
    assert acceptance.failed_checks == expected_failures
    if adverse:
        with pytest.raises(RuntimeError, match=", ".join(expected_failures)):
            result.require_accepted()
    else:
        result.require_accepted()
    metric_names = (
        "dupire_repaired_fraction",
        "pde_target_iv_rmse",
        "pde_mc_iv_rmse",
        "pde_mass_errors",
        "pde_negative_masses",
        "mc_forward_z_scores",
        "market_spread_coverage",
    )
    for key in metric_names:
        np.testing.assert_allclose(
            getattr(acceptance, key), getattr(ref.acceptance, key), rtol=0, atol=5e-9, err_msg=key
        )
    for key in [f.name for f in fields(acceptance.criteria)]:
        assert getattr(acceptance.criteria, key) == getattr(ref.acceptance.criteria, key)
    for name, value in dict(
        snapshot=ref.snapshot.fingerprint,
        expiries=[s.slice_id for s in market.source_slices],
        original_quotes=quotes,
        normalized_pde_calls=[sl.pde_prices.tolist() for sl in result.normalized],
        normalized_mc_calls=[sl.mc_prices.tolist() for sl in result.normalized],
        archived_normalized_mc_calls=[
            s.model_prices.tolist() for s in ref.monte_carlo_validation.normalized_pricing.slices
        ],
        normalized_mc_standard_errors=[sl.mc_standard_errors.tolist() for sl in result.normalized],
        archived_normalized_pde_calls=ref.pde_repricing.model_calls.tolist(),
        mc_price_errors_over_discounted_forward=mc_price_errors,
        mc_standard_error_differences_over_discounted_forward=mc_se_errors,
        metrics={key: np.asarray(getattr(acceptance, key)).tolist() for key in metric_names},
        archived_metrics={
            key: np.asarray(getattr(ref.acceptance, key)).tolist() for key in metric_names
        },
        criteria={
            f.name: getattr(acceptance.criteria, f.name) for f in fields(acceptance.criteria)
        },
        checks=dict(acceptance.checks),
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
        fine_negative_mass=max(row.negative_mass for row in result.pde_expiries),
        inside_spread=inside,
        target_iv_rmse=list(acceptance.pde_target_iv_rmse),
        mc_iv_rmse=list(acceptance.pde_mc_iv_rmse),
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
