"""LSV path, sampling-error and vanilla repricing contracts."""

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from vanilla_option_pricers import compute_bsm_vanilla_price

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import FixedLogForwardJump, LSVModel, LeverageSurface
from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig, LSVMonteCarloEngine
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


def _constant(**changes):
    return LogSvParams(
        **(dict(sigma0=0.2, theta=0.2, kappa1=0.0, kappa2=0.0, beta=0.0, volvol=0.0) | changes)
    )


def _chain(times=(0.125, 0.25)):
    return OptionChain(
        ttms=np.array(times),
        forwards=np.array([98.0, 105.0]),
        discfactors=np.array([0.99, 0.95]),
        strikes_ttms=(np.array([104.0, 96.0, 98.0]), np.array([100.0, 107.0])),
        optiontypes_ttms=(np.array(["P", "C", "C"]), np.array(["P", "C"])),
        ids=np.array(["a", "b"]),
    )


@pytest.mark.parametrize("paired", [False, True])
def test_black_prices_and_independent_sample_errors(paired):
    chain = _chain()
    model = LSVModel.from_params(_constant())
    config = LSVMonteCarloConfig(paths=8192, steps_per_year=24, seed=31, antithetic=paired)
    pricer = LSVMonteCarloPricer(config=config)
    fit = pricer.price_chain_detailed(chain, model)
    prices, errors = pricer.model_mc_price_chain(chain, model)
    for i, t in enumerate(chain.ttms):
        f, d = chain.forwards[i], chain.discfactors[i]
        terminal = f * fit.paths.assets[:, i + 1, 0]
        for j, (k, cp) in enumerate(zip(chain.strikes_ttms[i], chain.optiontypes_ttms[i])):
            samples = d * np.maximum((terminal - k) * (1 if cp == "C" else -1), 0)
            if paired:
                samples = samples.reshape(2, -1).mean(axis=0)
            se = samples.std(ddof=1) / np.sqrt(len(samples))
            np.testing.assert_allclose(prices[i][j], samples.mean(), rtol=1e-14)
            np.testing.assert_allclose(errors[i][j], se, rtol=1e-14)
            black = compute_bsm_vanilla_price(
                forward=f, strike=k, ttm=t, vol=0.2, optiontype=cp, discfactor=d
            )
            assert abs(fit.prices[i][j] - black) < 4 * se
            assert fit.expiries[i].price_upper[j] - fit.prices[i][j] == pytest.approx(
                1.95996398454 * se
            )
        assert not fit.invalid_iv[i].any()
    np.testing.assert_allclose(
        fit.paths.states["quadratic_variance"][0], np.r_[0, chain.ttms] * 0.2**2
    )


def test_path_partition_and_random_stream_match_explicit_observation_knots():
    p = _constant(vol_backbone=pd.Series([1.4, 0.9], index=[0.07, 0.5]))
    x = Grid1D([-1.0, 0.0, 1.0])
    surface = LeverageSurface([0.12, 0.5], x, [[1.3, 1.3, 1.3], [0.8, 0.8, 0.8]])
    engine = LSVMonteCarloEngine(
        LSVModel.from_params(p, surface), LSVMonteCarloConfig(paths=64, steps_per_year=10, seed=55)
    )
    paths = engine.simulate_paths(observation_times=np.array([0.0, 0.25]), spot0=103.0)
    split = engine.simulate_paths(observation_times=np.array([0.0, 0.07, 0.12, 0.25]), spot0=103.0)
    for key in paths.states:
        np.testing.assert_array_equal(paths.states[key], split.states[key][:, [0, -1]])
        assert not paths.states[key].flags.writeable
    assert paths.provenance["interval_step_counts"] == (4,)
    integrated = 0.2**2 * (
        0.07 * (1.4 * 1.3) ** 2 + 0.05 * (0.9 * 1.3) ** 2 + 0.13 * (0.9 * 0.8) ** 2
    )
    np.testing.assert_allclose(paths.states["quadratic_variance"][:, -1], integrated)
    logs = paths.states["log_zero_drift_return"][:, -1]
    np.testing.assert_allclose(logs[:32] + logs[32:], -integrated, atol=1e-14)
    np.testing.assert_allclose(paths.assets[:, -1, 0], 103.0 * np.exp(logs))


def test_compensated_grid_free_jump_matches_conditional_black_prices():
    jump = FixedLogForwardJump(1.5, -0.19)
    model = LSVModel.from_params(_constant(), jump=jump)
    cfg = LSVMonteCarloConfig(paths=40000, steps_per_year=32, seed=19)
    chain = _chain()
    result = LSVMonteCarloPricer(config=cfg).price_chain_detailed(chain, model)
    for i, t in enumerate(chain.ttms):
        f, d = chain.forwards[i], chain.discfactors[i]
        for j, (k, cp) in enumerate(zip(chain.strikes_ttms[i], chain.optiontypes_ttms[i])):
            mean = jump.intensity * t
            prob = np.exp(-mean)
            exact = 0.0
            for n in range(20):
                exact += prob * compute_bsm_vanilla_price(
                    forward=f * np.exp(-mean * np.expm1(jump.jump_size) + n * jump.jump_size),
                    strike=k,
                    ttm=t,
                    vol=0.2,
                    optiontype=cp,
                    discfactor=d,
                )
                prob *= mean / (n + 1)
            assert abs(result.prices[i][j] - exact) < 4 * result.standard_errors[i][j]
        count = result.paths.states["jump_count"][:, i + 1]
        assert abs(count.mean() - jump.intensity * t) < 4 * np.sqrt(jump.intensity * t / cfg.paths)
        np.testing.assert_allclose(result.paths.states["quadratic_variance"][:, i + 1], 0.04 * t)
        sl = result.expiries[i]
        assert abs(sl.forward_mean - f) < 4 * sl.forward_standard_error
    pde = LogSvPdePricer().price_chain(chain, model)
    for pv, mc, se in zip(pde, result.prices, result.standard_errors):
        assert np.all(abs(pv - mc) < 4 * se)


def test_raw_zero_payoff_and_failed_black_inversion():
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=100.0,
        strikes=np.array([100.0, 1.0e5]),
        optiontypes=np.array(["C", "C"]),
        discfactor=0.99,
    )
    pricer = LSVMonteCarloPricer(
        config=LSVMonteCarloConfig(128, 12, 11), confidence_level=np.nextafter(1.0, 0.0)
    )
    result = pricer.price_chain_detailed(chain, LSVModel.from_params(_constant()))
    assert result.prices[0][1] == result.standard_errors[0][1] == 0.0
    assert result.invalid_iv[0].tolist() == [False, True]
    assert np.isfinite(result.expiries[0].price_upper).all()


def test_measure_martingale_and_sample_size_contracts():
    cfg = LSVMonteCarloConfig(4, 8, 5)
    model = LSVModel.from_params(_constant(beta=0.6))
    with pytest.raises(ValueError, match="kappa2"):
        LSVMonteCarloEngine(model, cfg).simulate_paths(observation_times=np.array([0.0, 0.1]))
    with pytest.raises(ValueError, match="money-market"):
        LSVMonteCarloEngine(model, cfg).simulate_paths(
            observation_times=np.array([0.0, 0.1]), measure="Q_INVERSE"
        )
    with pytest.raises(ValueError, match="independent"):
        LSVMonteCarloPricer(config=replace(cfg, paths=2)).price_chain(_chain(), model)
    with pytest.raises(ValueError, match="even"):
        replace(cfg, paths=3)
    with pytest.raises(ValueError, match="intensity"):
        FixedLogForwardJump(-1, 0.2)
    with pytest.raises(ValueError, match="jump_size"):
        FixedLogForwardJump(1, np.nan)


@pytest.mark.repository_only
@pytest.mark.slow
def test_frozen_btc_vanilla_mc_archive_and_pde(record_property):
    if find_spec("stochvolmodels.pde_solvers.market_pipeline") is None:
        pytest.skip("requires the archived BTC pipeline in the source checkout")
    from stochvolmodels.pde_solvers.market_pipeline import (
        LSVMarketPipeline,
        btc_20211021_reference_config,
        load_frozen_btc_20211021_snapshot,
    )

    reference = btc_20211021_reference_config()
    frozen = LSVMarketPipeline(reference).run(load_frozen_btc_20211021_snapshot())
    original = frozen.snapshot.to_option_chain()
    fitted = frozen.calibration_result.leverage_surface
    # Hold the calibrated model fixed to isolate the Monte Carlo and PDE pricing engines.
    model = LSVModel.from_params(
        reference.params,
        LeverageSurface(
            fitted.times, fitted.grid, fitted.values, fitted.leverage_floor, fitted.leverage_cap
        ),
    )
    old = reference.monte_carlo_config
    config = LSVMonteCarloConfig(old.paths, old.steps_per_year, old.seed, old.antithetic)
    mc = LSVMonteCarloPricer(config=config, confidence_level=old.confidence_level)
    pde = LogSvPdePricer(
        grid=reference.repricing_grid,
        steps_per_interval=reference.repricing_steps_per_interval,
        adi=reference.adi_config,
    )
    t = frozen.calibration_input.target
    normalized = OptionChain(
        ttms=t.times,
        forwards=np.full(len(t.times), t.forward),
        discfactors=np.full(len(t.times), t.discount_factor),
        ids=original.ids,
        strikes_ttms=tuple(t.strikes for _ in t.times),
        optiontypes_ttms=tuple(np.full(len(t.strikes), "C") for _ in t.times),
    )
    for name, chain, oracle, n_quotes in [
        ("original", original, frozen.option_chain_pricing, 49),
        ("normalized", normalized, frozen.monte_carlo_validation.normalized_pricing, 44),
    ]:
        sim = mc.price_chain_detailed(chain, model)
        fine = pde.price_chain_detailed(chain, model)
        assert len(sim.expiries) == 4 and sum(map(len, sim.prices)) == n_quotes
        ref_paths = oracle.monte_carlo_result.paths
        path_errors = {}
        for key, expected in [
            ("log_zero_drift_return", ref_paths.log_forwards),
            ("sigma", ref_paths.volatilities),
            ("quadratic_variance", ref_paths.quadratic_variations),
            ("jump_count", ref_paths.jump_counts),
        ]:
            np.testing.assert_allclose(sim.paths.states[key][:, 1:].T, expected, rtol=0, atol=1e-12)
            path_errors[key] = float(np.max(np.abs(sim.paths.states[key][:, 1:].T - expected)))
        record_property(name + "_path_errors", path_errors)
        np.testing.assert_array_equal(
            sim.paths.provenance["interval_step_counts"], ref_paths.time_steps
        )
        np.testing.assert_allclose(
            sim.paths.diagnostics["diffusion_shock_correlation"],
            ref_paths.diffusion_shock_correlation,
            rtol=0,
            atol=1e-12,
        )
        rows = []
        for i, archived in enumerate(oracle.slices):
            assert not sim.invalid_iv[i].any() and not fine.invalid_iv[i].any()
            df = chain.forwards[i] * chain.discfactors[i]
            pvs, se = sim.prices[i], sim.standard_errors[i]
            sl = sim.expiries[i]
            assert np.all(se > 0)
            oracle_gap = np.max(abs(pvs - archived.model_prices)) / df
            sem_gap = np.max(abs(se - archived.standard_errors)) / df
            assert oracle_gap < 1e-12 and sem_gap < 1e-12
            np.testing.assert_allclose(
                sl.price_lower, archived.price_confidence_lower, atol=1e-12 * df, rtol=0
            )
            np.testing.assert_allclose(
                sl.price_upper, archived.price_confidence_upper, atol=1e-12 * df, rtol=0
            )
            np.testing.assert_allclose(sl.forward_mean, archived.forward_mean, rtol=1e-13)
            np.testing.assert_allclose(
                sl.forward_standard_error, archived.forward_standard_error, rtol=1e-12
            )
            np.testing.assert_allclose(
                sim.implied_vols[i], archived.model_implied_volatilities, rtol=0, atol=1e-10
            )
            gap = pvs - fine.prices[i]
            rmse = float(np.sqrt(np.mean((sim.implied_vols[i] - fine.implied_vols[i]) ** 2)))
            row = dict(
                expiry=str(chain.ids[i]),
                quotes=len(pvs),
                archive_max_pv_over_df=float(oracle_gap),
                archive_max_se_over_df=float(sem_gap),
                pde_max_pv_over_df=float(np.max(abs(gap)) / df),
                pde_max_standard_errors=float(np.max(abs(gap) / se)),
                pde_iv_rmse=rmse,
                pde_inside_95=int(
                    np.sum((sl.price_lower <= fine.prices[i]) & (fine.prices[i] <= sl.price_upper))
                ),
                forward_z=float((sl.forward_mean - chain.forwards[i]) / sl.forward_standard_error),
            )
            rows.append(row)
            if name == "normalized":
                np.testing.assert_allclose(
                    fine.prices[i], frozen.pde_repricing.model_calls[i], atol=1e-12, rtol=0
                )
                assert rmse == pytest.approx(frozen.acceptance.pde_mc_iv_rmse[i], abs=1e-10)
        record_property(name, rows)
        assert (
            max(r["pde_iv_rmse"] for r in rows) < reference.acceptance_criteria.max_pde_mc_iv_rmse
        )
        assert (
            max(abs(r["forward_z"]) for r in rows)
            < reference.acceptance_criteria.max_abs_mc_forward_z
        )
        assert max(r["pde_max_standard_errors"] for r in rows) < 5
