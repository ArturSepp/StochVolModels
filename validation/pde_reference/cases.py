"""Historical inputs for the Dupire and fixed-point compatibility cases."""

import numpy as np
from stochvolmodels.pde_solvers.adi import ADIConfig, ADIScheme
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import PDE1DSolver, TimeStepConfig
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.density_pricer import CoordinateSystem, DensityPricer, DensityState1D
from validation.pde_reference.dupire import CallSurface
from validation.pde_reference.local_vol import LocalVolCoefficientProvider1D, LocalVolSurface
from validation.pde_reference.lsv import LSVProjectionConfig


def _spatial_local_vol_surface() -> LocalVolSurface:
    grid = Grid1D.uniform(-1.25, 1.25, 161)
    volatility = 0.2 * (1.0 - 0.22 * np.tanh(1.8 * grid.nodes) + 0.08 * grid.nodes**2)
    return LocalVolSurface(
        times=np.array([1.0]),
        grid=grid,
        values=volatility[None, :],
        coordinate=CoordinateSystem.LOG_FORWARD,
    )


def _pde_call_surface(spatial_size: int, time_steps: int) -> tuple[CallSurface, LocalVolSurface]:
    local_vol = _spatial_local_vol_surface()
    grid = Grid1D.concentrated(-1.45, 1.45, spatial_size, center=0.0, scale=0.12)
    maturities = np.linspace(0.05, 0.5, 19)
    time_grid = np.linspace(0.0, 0.5, time_steps + 1)
    solution = PDE1DSolver(TimeStepConfig(theta=1.0)).solve(
        grid,
        LocalVolCoefficientProvider1D(local_vol),
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        time_grid,
        grid.project_delta(0.0),
        save_times=tuple(maturities),
    )
    strikes = np.linspace(0.62, 1.55, 141)
    calls = np.empty((maturities.size, strikes.size))
    for index, state in enumerate(solution.saved_states):
        density = DensityState1D(
            grid, state.values, state.time, CoordinateSystem.LOG_FORWARD
        )
        calls[index] = DensityPricer(density, forward=1.0).price_slice(strikes).calls
    return (
        CallSurface(
            maturities,
            strikes,
            calls,
            forwards=np.ones(maturities.size),
            discount_factors=np.ones(maturities.size),
        ),
        local_vol,
    )


def _on_the_fly_case(
    maturities: np.ndarray,
    *,
    beta: float = 0.0,
    volvol: float = 0.28,
) -> tuple[TensorGrid2D, LocalVolSurface, LogSvParams, ADIConfig]:
    x_grid = Grid1D.concentrated(-0.75, 0.75, 81, center=0.0, scale=0.08)
    sigma0 = 0.2
    grid = TensorGrid2D(
        x_grid,
        Grid1D.concentrated(0.035, 0.58, 51, center=sigma0, scale=0.025),
    )
    local_volatility = 0.2 * (
        1.0 - 0.16 * np.tanh(2.2 * x_grid.nodes) + 0.05 * x_grid.nodes**2
    )
    target = LocalVolSurface(
        maturities,
        x_grid,
        np.tile(local_volatility, (maturities.size, 1)),
        CoordinateSystem.LOG_FORWARD,
    )
    params = LogSvParams(
        sigma0=sigma0,
        theta=sigma0,
        kappa1=3.5,
        kappa2=9.0,
        beta=beta,
        volvol=volvol,
    )
    return grid, target, params, ADIConfig(theta=1.0, scheme=ADIScheme.DOUGLAS)


def _on_the_fly_config(
    *,
    damping: float,
    maximum_iterations: int,
    relative_tolerance: float = 2.0e-4,
    absolute_tolerance: float = 2.0e-6,
    fail_on_density_violation: bool = True,
) -> LSVProjectionConfig:
    return LSVProjectionConfig(
        density_floor=1.0e-10,
        leverage_floor=0.1,
        leverage_cap=2.0,
        damping=damping,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        mass_tolerance=2.0e-8,
        negative_mass_tolerance=2.0e-5,
        fail_on_density_violation=fail_on_density_violation,
    )
