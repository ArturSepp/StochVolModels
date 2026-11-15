"""LSV market calibration workflow, price units and final acceptance gates."""

from dataclasses import replace

import numpy as np
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import DupireRepairPolicy
from stochvolmodels.fitters.lsv_market_pipeline import (
    LSVMarketAcceptance,
    LSVMarketAcceptanceCriteria,
    LSVMarketPipeline,
)
from stochvolmodels.fitters.lsv_parametric import (
    LSVParametricCalibrationConfig,
    LSVParametricCalibrator,
)
from stochvolmodels.models.local_vol import LocalVolFamily
from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


def test_thresholds_are_inclusive_and_each_gate_detects_one_violation():
    c = LSVMarketAcceptanceCriteria()
    limits = LSVMarketAcceptance(
        c,
        c.max_dupire_repaired_fraction,
        (0.0, c.max_pde_target_iv_rmse),
        (0.0, c.max_pde_mc_iv_rmse),
        (0.0, c.max_pde_mass_error),
        (0.0, c.max_pde_negative_mass),
        (0.0, -c.max_abs_mc_forward_z),
        c.min_market_spread_coverage,
    )
    assert limits.passed and len(limits.checks) == 7 and not limits.failed_checks
    for field, value, failed in [
        ("dupire_repaired_fraction", 0.16, "dupire_repair"),
        ("pde_target_iv_rmse", (0.021, 0.0), "pde_target_iv"),
        ("pde_mc_iv_rmse", (0.0, float("nan")), "pde_mc_iv"),
        ("pde_mass_errors", (0.0, 1.1e-8), "pde_mass"),
        ("pde_negative_masses", (2.1e-5, 0.0), "pde_negative_mass"),
        ("mc_forward_z_scores", (-4.01, 0.0), "mc_martingale"),
        ("market_spread_coverage", 0.59, "market_spread"),
    ]:
        assert replace(limits, **{field: value}).failed_checks == (failed,)
    for field, value in (
        ("min_market_spread_coverage", 1.01),
        ("max_pde_mass_error", float("inf")),
        ("max_pde_target_iv_rmse", -0.01),
    ):
        with pytest.raises(ValueError):
            replace(c, **{field: value})


def test_invalid_raw_dupire_and_outside_strikes_stop_before_fitting():
    t = np.array([0.1, 0.2, 0.3])
    k = np.linspace(0.9, 1.1, 7)
    chain = OptionChain.get_uniform_chain(t, np.array(["1", "2", "3"]), strikes=k)
    # A much lower middle-expiry ATM call breaks calendar monotonicity.
    chain.bid_ivs[1][3] = chain.ask_ivs[1][3] = 0.015
    y = Grid1D([0.1, 0.15, 0.2, 0.25, 0.3])
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 61), y)
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    pipe = LSVMarketPipeline(LSVParametricCalibrator(grid), LogSvPdePricer(grid=grid))
    with pytest.raises(ValueError, match="invalid Dupire nodes"):
        pipe.run(chain, params, grid_size=7)
    narrow = TensorGrid2D(Grid1D.uniform(-0.02, 0.02, 11), y)
    with pytest.raises(ValueError, match="common market strikes"):
        replace(pipe, pde_pricer=LogSvPdePricer(grid=narrow)).run(chain, params, grid_size=7)
    with pytest.raises(TypeError, match="criteria"):
        replace(pipe, criteria=None)


def test_market_pipeline_preserves_ragged_quote_units_and_independent_repricing():
    f, d = np.array([79.0, 104.0, 131.0]), np.array([0.99, 0.96, 0.92])
    moneyness = (
        [1.15, 0.87, 1.0, 0.95, 1.05],
        [0.85, 1.10, 0.97, 1.03],
        [1.13, 1.0, 0.94, 0.89, 1.06, 1.18],
    )
    chain = OptionChain(
        ttms=np.array([0.1, 0.2, 0.35]),
        forwards=f,
        discfactors=d,
        ids=np.array(["front", "next", "last"]),
        strikes_ttms=tuple(v * np.array(k) for v, k in zip(f, moneyness)),
        optiontypes_ttms=tuple(np.where(np.array(k) < 1, "P", "C") for k in moneyness),
        bid_ivs=tuple(np.full(len(k), 0.17) for k in moneyness),
        ask_ivs=tuple(np.full(len(k), 0.23) for k in moneyness),
    )
    y = Grid1D([0.10, 0.15, 0.20, 0.25, 0.30])
    grid = TensorGrid2D(Grid1D.concentrated(-1.0, 1.0, 101, 0.0, 0.12), y)
    fine = TensorGrid2D(Grid1D.concentrated(-1.1, 1.1, 151, 0.0, 0.12), y)
    pipeline = LSVMarketPipeline(
        LSVParametricCalibrator(grid, LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT)),
        LogSvPdePricer(grid=fine, steps_per_interval=(24, 36, 48)),
        LSVMonteCarloPricer(config=LSVMonteCarloConfig(16384, 144, 71)),
    )
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    result = pipeline.run(
        chain,
        params,
        grid_size=7,
        repair_policy=DupireRepairPolicy(0.03, 0.038),
        calibration_steps_per_interval=[18, 24, 32],
    )
    assert len(result.pde_steps) == 24 + 36 + 48
    assert np.count_nonzero(result.dupire.interior_mask) == 5
    assert result.acceptance.dupire_repaired_fraction == np.mean(
        result.dupire.repaired_mask[result.dupire.interior_mask]
    )
    assert result.acceptance.market_spread_coverage == 1.0
    target = result.market.target
    calls = OptionChain(
        ttms=chain.ttms,
        forwards=np.ones(3),
        ids=chain.ids,
        strikes_ttms=(target.strikes,) * 3,
        optiontypes_ttms=(np.full(target.strikes.size, "C"),) * 3,
    )
    for rows, quotes in ((result.normalized, calls), (result.original, chain)):
        mc = pipeline.mc_pricer.price_chain_detailed(quotes, result.calibration.model)
        pde = pipeline.pde_pricer.price_chain_detailed(quotes, result.calibration.model)
        for i, row in enumerate(rows):
            np.testing.assert_allclose(row.pde_prices, pde.prices[i], atol=1.0e-12, rtol=0)
            np.testing.assert_allclose(row.mc_prices, mc.prices[i], atol=1.0e-12, rtol=0)
            np.testing.assert_allclose(
                row.mc_standard_errors, mc.standard_errors[i], atol=1.0e-13, rtol=0
            )
            np.testing.assert_allclose(row.mc_implied_vols, mc.implied_vols[i], atol=1.0e-9, rtol=0)
            np.testing.assert_allclose(
                row.pde_implied_vols, pde.implied_vols[i], atol=1.0e-9, rtol=0
            )
            assert not row.pde_prices.flags.writeable
