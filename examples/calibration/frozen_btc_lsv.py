"""Replay the 21 October 2021 BTC LSV calibration from packaged quotes.

Run `python -m examples.calibration.frozen_btc_lsv` for the baseline and adverse
starts. Both use four maturities, 11 common strikes, and 49 original contracts.
"""

from enum import Enum

import numpy as np

from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.fitters.dupire import DupireRepairPolicy
from stochvolmodels.fitters.lsv_market_pipeline import LSVMarketPipeline, LSVMarketPipelineResult
from stochvolmodels.fitters.lsv_parametric import (
    CalibrationWeighting,
    LSVParametricCalibrationConfig,
    LSVParametricCalibrator,
)
from stochvolmodels.models.local_vol import LocalVolFamily
from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import ADIConfig, ADIScheme, Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


def replay_btc(
    *,
    adverse: bool = False,
    candidate_negative_mass_limit: float | None = None,
) -> LSVMarketPipelineResult:
    """Price the archived starts with explicit fit and independent repricing policies.

    The original fitter observed candidate density violations without enforcing them.
    The public fitter penalizes violating trials and enforces the limit on the final replay.
    The historical replay limits
    are 3e-5 (baseline) and 1.5e-4 (adverse). Final expiry acceptance uses 2e-5 for both.
    """
    trial_limit = 1.5e-4 if adverse else 3e-5
    if candidate_negative_mass_limit is not None:
        trial_limit = candidate_negative_mass_limit
    params = LogSvParams(sigma0=0.88, theta=0.90, kappa1=2.0, kappa2=3.0, beta=-0.25, volvol=0.45)
    grid = TensorGrid2D(
        Grid1D.concentrated(-2.4, 2.2, 71, 0.0, 0.16),
        Grid1D.concentrated(0.05, 2.8, 45, 0.88, 0.10),
    )
    fine = TensorGrid2D(
        Grid1D.concentrated(-2.4, 2.2, 101, 0.0, 0.13),
        Grid1D.concentrated(0.05, 2.8, 61, 0.88, 0.075),
    )
    config = LSVParametricCalibrationConfig(
        family=LocalVolFamily.PRODUCT,
        formula_weight=0.2,
        weighting=CalibrationWeighting.VEGA_NORMALIZED,
        maximum_evaluations=24 if adverse else 12,
        prefit_maximum_evaluations=100,
        optimizer_ftol=5e-5,
        optimizer_xtol=5e-5,
        optimizer_gtol=5e-5,
        leverage_floor=0.1,
        leverage_cap=3.0,
        mass_tolerance=1e-6,
        negative_mass_tolerance=trial_limit,
        quadratic_start=False,  # Match the archived candidate densities.
    )
    start = [[0.35, -2.5, 8], [3.2, 4.5, 12], [0.40, -1.5, 15], [3.5, 5.5, 18]] if adverse else None
    adi = ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS)
    pipeline = LSVMarketPipeline(
        LSVParametricCalibrator(grid, config, adi),
        LogSvPdePricer(
            grid=fine, steps_per_interval=(80, 80, 100, 160), adi=adi, quadratic_start=False
        ),
        LSVMonteCarloPricer(
            config=LSVMonteCarloConfig(paths=100_000, steps_per_year=720, seed=20260817)
        ),
    )
    return pipeline.run(
        get_btc_test_chain_data(),
        params,
        grid_size=11,
        repair_policy=DupireRepairPolicy(0.04, 4.0),
        historical_dupire_endpoints=True,
        calibration_steps_per_interval=(20, 20, 30, 50),
        initial_parameters=start,
    )


class Locals(Enum):
    """Frozen BTC replay cases."""

    BASELINE = 1
    ADVERSE = 2


def run_local(local: Locals) -> None:
    """Report the actual decision, spread coverage and price differences for each expiry."""
    adverse = local == Locals.ADVERSE
    result = replay_btc(adverse=adverse)
    failed = result.acceptance.failed_checks
    print(f"{local.name}: {', '.join(failed) if failed else 'all seven checks passed'}")
    print(f"Common-strike bid/ask coverage: {result.acceptance.market_spread_coverage:.2%}")
    for source, prices in zip(result.market.source_slices, result.original, strict=True):
        gap = np.abs(prices.pde_prices - prices.mc_prices) / (
            source.discount_factor * source.forward
        )
        print(f"{source.slice_id:>5}  {gap.size:2} quotes  max |PDE-MC| / (DF) {gap.max():.6g}")
    expected = (
        ("pde_target_iv", "pde_mc_iv", "pde_negative_mass", "market_spread") if adverse else ()
    )
    if failed != expected:
        raise RuntimeError(f"{local.name} expected {expected}, observed {failed}")


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
