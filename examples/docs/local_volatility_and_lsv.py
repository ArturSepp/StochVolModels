"""Canonical offline cases for docs/local_volatility_and_lsv.md.

Requires the 2.18.8 development source. Run every case with this script's normal
entry point, including against an installed core wheel. Seeded Monte Carlo
uncertainty is reported separately from PDE and time-discretization errors.
"""
from enum import Enum

import numpy as np
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import CallSurface, DupireRecoverer, DupireRepairPolicy
from stochvolmodels.fitters.lsv_market_pipeline import LSVMarketPipeline
from stochvolmodels.fitters.lsv_parametric import (
    ImpliedVolCalibrationTarget, LSVParametricCalibrationConfig, LSVParametricCalibrator,
)
from stochvolmodels.models.local_vol import LocalVolFamily, LocalVolSurface
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


def bucket_refinement():
    """Recover effective buckets from rising and falling deterministic Black surfaces."""
    times = np.linspace(0.25, 0.75, 5)
    strikes = np.linspace(60.0, 140.0, 81)
    rows = []
    for name, intercept, slope in (("Rising", 0.2, 0.4), ("Falling", 0.6, -0.4)):
        variance = intercept**2 * times + intercept * slope * times**2 + slope**2 * times**3 / 3
        sd = np.sqrt(variance[:, None])
        d1 = np.log(100.0 / strikes) / sd + sd / 2
        calls = 100.0 * ndtr(d1) - strikes * ndtr(d1 - sd)
        recovered = DupireRecoverer().continuous(CallSurface(times, strikes, calls, forwards=100.0))
        surface = recovered.to_local_vol_surface()
        expected_buckets = np.diff(np.r_[0.0, variance]) / np.diff(np.r_[0.0, times])
        np.testing.assert_allclose(surface.values**2,
                                   np.broadcast_to(expected_buckets[:, None], surface.values.shape),
                                   atol=1e-11)
        for nodes, steps in ((251, 64), (501, 128)):
            pricer = LocalVolPdePricer(
                grid=Grid1D.concentrated(-3.0, 3.0, nodes, 0.0, 0.12),
                steps_per_interval=steps)
            for maturity in (0.25, 0.5, 0.75):
                price, iv = pricer.price_vanilla(surface, maturity, 100.0, 100.0, "C")
                index = np.flatnonzero(np.isclose(times, maturity))[0]
                exact_iv = np.sqrt(variance[index] / maturity)
                exact_price = 100.0 * (2 * ndtr(np.sqrt(variance[index]) / 2) - 1)
                assert abs(iv - exact_iv) * 10000 < 15
                rows.append(dict(curve=name, nodes=nodes, steps=steps, maturity=maturity,
                                 price=price, exact_price=float(exact_price),
                                 iv_error_bp=float((iv - exact_iv) * 10000)))
    np.testing.assert_allclose([r['iv_error_bp'] for r in rows],
        [-0.453, -0.307, -0.272, -0.113, -0.0768, -0.0679,
         -0.479, -0.357, -0.294, -0.120, -0.0891, -0.0735], atol=0.002, rtol=0)
    return rows


def parametric_example():
    """Fit a flat leverage bucket to an independent 24% Black smile."""
    x = Grid1D.concentrated(-0.9, 0.9, 81, 0.0, 0.12)
    grid = TensorGrid2D(x, Grid1D([0.1, 0.15, 0.2, 0.25, 0.3]))
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    target = ImpliedVolCalibrationTarget([0.2], [0.95, 1.0, 1.05], [[0.24] * 3])
    local = LocalVolSurface(target.times, x, np.full((1, x.size), 0.24))
    config = LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT)
    fit = LSVParametricCalibrator(grid, config).calibrate(params, target, local, 64)
    assert abs(fit.parameter_values[0, 0] - 1.2) < 0.02
    price, iv = LogSvPdePricer().price_vanilla(fit.model, 0.2, 1.0, 1.0, "C")
    assert abs(iv - 0.24) < 0.003
    return fit


def monte_carlo_example():
    """Reprice a correlated LSV smile with independent PDE and antithetic paths."""
    params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=2.0,
                         beta=-0.5, volvol=0.7)
    model = LSVModel.from_params(params)
    chain = OptionChain.get_uniform_chain(np.array([0.25]), np.array(["3m"]),
                                          strikes=np.array([0.9, 1.0, 1.1]))
    config = LSVMonteCarloConfig(paths=32768, steps_per_year=720, seed=7, antithetic=True)
    mc = LSVMonteCarloPricer(config=config).price_chain_detailed(chain, model)
    grid = TensorGrid2D(Grid1D.concentrated(-1.5, 1.5, 241, 0.0, 0.1),
                        Grid1D.concentrated(0.0, 1.6, 121, 0.2, 0.05))
    pde = LogSvPdePricer(grid=grid, steps_per_interval=256).price_chain_detailed(chain, model)
    z = (np.asarray(mc.prices[0]) - pde.prices[0]) / mc.standard_errors[0]
    assert np.max(np.abs(z)) < 5.0
    return dict(strikes=chain.strikes_ttms[0].tolist(), pde=pde.prices[0].tolist(),
                mc=mc.prices[0].tolist(), standard_errors=mc.standard_errors[0].tolist(),
                z=z.tolist())


def market_pipeline_example():
    """Fit synthetic bid/ask quotes and inspect every independently repriced gate."""
    times = np.array([0.1, 0.2, 0.3])
    chain = OptionChain.get_uniform_chain(times, np.array(["1", "2", "3"]),
                                          strikes=np.linspace(0.9, 1.1, 7))
    y = Grid1D([0.1, 0.15, 0.2, 0.25, 0.3])
    fit_grid = TensorGrid2D(Grid1D.concentrated(-1.0, 1.0, 81, 0.0, 0.12), y)
    replay_grid = TensorGrid2D(Grid1D.concentrated(-1.0, 1.0, 121, 0.0, 0.12), y)
    fitter = LSVParametricCalibrator(
        fit_grid, LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT))
    pipeline = LSVMarketPipeline(
        fitter, LogSvPdePricer(grid=replay_grid, steps_per_interval=64),
        LSVMonteCarloPricer(config=LSVMonteCarloConfig(8192, 120, 42)))
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    result = pipeline.run(chain, params, grid_size=7, repair_policy=DupireRepairPolicy(0.01, 0.16))
    assert (len(result.original), len(result.original[0].pde_prices),
            len(result.acceptance.checks)) == (3, 7, 7)
    assert result.acceptance.failed_checks == ("market_spread",)
    assert np.all(np.isfinite(result.original[0].pde_prices))
    return result


class Locals(Enum):
    """Every case executes in the ordinary offline documentation lane."""
    BUCKETS = 1
    PARAMETRIC = 2
    MONTE_CARLO = 3
    MARKET_PIPELINE = 4


def run_local(local: Locals):
    """Run one example with assertions through the public entry points."""
    if local == Locals.BUCKETS:
        return bucket_refinement()
    if local == Locals.PARAMETRIC:
        return parametric_example()
    if local == Locals.MONTE_CARLO:
        return monte_carlo_example()
    if local == Locals.MARKET_PIPELINE:
        return market_pipeline_example()
    raise ValueError(local)


if __name__ == "__main__":
    for case in Locals:
        run_local(case)
