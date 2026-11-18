"""
Numerical tests for density states and vanilla pricing projections.

The cases compare density-projected vanillas with Black prices and parity, verify level versus
log coordinates, reconcile two-dimensional marginals with direct projection, and enforce
immutable state/result contracts.
"""

from __future__ import annotations

import numpy as np
import pytest
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D
from stochvolmodels.pde_solvers.coefficients import (
    PDE1DCoefficients,
    StaticCoefficientProvider1D,
)
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import PDE1DSolver, TimeStepConfig
from vanilla_option_pricers import compute_bsm_vanilla_price

from stochvolmodels.pricers.density_pricer import (
    CoordinateSystem,
    DensityPricer,
    DensityState1D,
    DensityState2D,
)


def _log_forward_density_state() -> tuple[DensityState1D, float]:
    maturity = 0.75
    volatility = 0.28
    grid = Grid1D.uniform(-2.2, 1.8, 401)
    half_variance = 0.5 * volatility**2
    provider = StaticCoefficientProvider1D(
        PDE1DCoefficients.constant(
            grid.size,
            variance=half_variance,
            drift=-half_variance,
            discount=0.0,
        )
    )
    solution = PDE1DSolver(TimeStepConfig(theta=1.0)).solve(
        grid,
        provider,
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        np.linspace(0.0, maturity, 301),
        grid.project_delta(0.0),
    )
    return (
        DensityState1D(
            grid=grid,
            values=solution.values,
            time=maturity,
            coordinate=CoordinateSystem.LOG_FORWARD,
        ),
        volatility,
    )


def test_no_jump_pde_density_prices_match_bsm_parity_and_implied_volatility() -> None:
    state, volatility = _log_forward_density_state()
    forward = 1.0
    discount_factor = 0.97
    strikes = np.array([0.75, 0.9, 1.0, 1.1, 1.3])
    pricer = DensityPricer(
        state=state, forward=forward, discount_factor=discount_factor
    )
    result = pricer.price_slice(strikes)
    expected_calls = np.array(
        [
            compute_bsm_vanilla_price(
                forward,
                strike,
                state.time,
                volatility,
                "C",
                discount_factor,
            )
            for strike in strikes
        ]
    )
    expected_puts = np.array(
        [
            compute_bsm_vanilla_price(
                forward,
                strike,
                state.time,
                volatility,
                "P",
                discount_factor,
            )
            for strike in strikes
        ]
    )

    np.testing.assert_allclose(result.calls, expected_calls, atol=3.5e-4)
    np.testing.assert_allclose(result.puts, expected_puts, atol=3.5e-4)
    assert np.max(np.abs(result.parity_residuals)) < 2.0e-15
    assert result.mass == pytest.approx(1.0, abs=3.0e-12)
    assert result.forward_moment == pytest.approx(forward, abs=4.0e-4)

    option_types = np.array(["C", "P", "C", "P", "C"])
    prices = pricer.price_vanillas(strikes, option_types)
    expected = np.where(option_types == "C", result.calls, result.puts)
    np.testing.assert_allclose(prices, expected, rtol=0.0, atol=0.0)
    implied_vols = pricer.implied_volatilities(
        strikes, option_types, ttm=state.time
    )
    np.testing.assert_allclose(implied_vols, volatility, atol=1.1e-3)


def test_level_coordinate_pricing_uses_level_nodes_directly() -> None:
    grid = Grid1D.uniform(0.2, 2.0, 501)
    sigma = 0.18
    density = np.exp(-0.5 * ((grid.nodes - 1.0) / sigma) ** 2)
    density /= grid.integrate(density)
    state = DensityState1D(grid, density, 0.5, CoordinateSystem.LEVEL)
    pricer = DensityPricer(state, forward=grid.integrate(grid.nodes * density))
    strikes = np.array([0.8, 1.0, 1.2])
    result = pricer.price_slice(strikes)

    assert np.all(result.calls >= 0.0)
    assert np.all(result.puts >= 0.0)
    assert np.max(np.abs(result.parity_residuals)) < 2.0e-15
    assert result.forward_moment == pytest.approx(
        grid.integrate(grid.nodes * density), abs=2.0e-15
    )


def test_2d_density_marginal_prices_equal_direct_1d_projection() -> None:
    state_x, _ = _log_forward_density_state()
    y_grid = Grid1D.concentrated(-1.0, 1.0, 51, center=0.0, scale=0.25)
    y_density = np.exp(-0.5 * (y_grid.nodes / 0.2) ** 2)
    y_density /= y_grid.integrate(y_density)
    tensor_grid = TensorGrid2D(state_x.grid, y_grid)
    values = np.multiply.outer(state_x.values, y_density)
    state_2d = DensityState2D(
        grid=tensor_grid,
        values=values,
        time=state_x.time,
        x_coordinate=CoordinateSystem.LOG_FORWARD,
    )
    marginal = state_2d.marginal_x_state()
    strikes = np.array([0.85, 1.0, 1.2])

    np.testing.assert_allclose(marginal.values, state_x.values, atol=3.0e-15)
    assert state_2d.mass == pytest.approx(1.0, abs=3.0e-12)
    direct = DensityPricer(state_x, forward=1.0).price_slice(strikes)
    projected = DensityPricer(marginal, forward=1.0).price_slice(strikes)
    np.testing.assert_allclose(projected.calls, direct.calls, atol=3.0e-15)
    np.testing.assert_allclose(projected.puts, direct.puts, atol=3.0e-15)


def test_density_states_and_pricing_results_are_immutable_and_validated() -> None:
    grid = Grid1D.uniform(-1.0, 1.0, 11)
    values = grid.project_delta(0.0)
    state = DensityState1D(grid, values, 0.25, CoordinateSystem.LOG_FORWARD)
    result = DensityPricer(state, forward=1.0).price_slice(np.array([0.9, 1.0, 1.1]))

    assert not state.values.flags.writeable
    assert not result.strikes.flags.writeable
    assert not result.calls.flags.writeable
    assert not result.puts.flags.writeable
    with pytest.raises(ValueError, match="positive"):
        DensityPricer(state, forward=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        DensityPricer(state, forward=1.0).price_slice(np.array([0.0, 1.0]))
