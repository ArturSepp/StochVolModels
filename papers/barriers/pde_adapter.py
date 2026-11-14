"""Conditional barrier Green densities and survival on the replication's log-price grid."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numba import njit

from stochvolmodels.pde_solvers import (
    BoundaryPair1D,
    DriftDiscretization,
    Grid1D,
    PDE1DCoefficients,
    PDEDirection,
)
from stochvolmodels.pde_solvers.operators_1d import Operator1D
from .pde import tridag_mult, tridag_solve


@dataclass
class BarrierGreen:
    """Signed killed density and cumulative exit probabilities, one column per path."""

    grid: Grid1D
    density: np.ndarray
    surviving_mass: np.ndarray
    lower_exit: np.ndarray
    upper_exit: np.ndarray

    def call_prices(self, log_strikes, discount=1.0):
        """Apply the replication's discounted payoff convolution to the killed density."""
        from .mc_volterra import convolute_green_with_payoff

        return convolute_green_with_payoff(self.grid.nodes, log_strikes, self.density, discount)


def _padded_generator(operator, direction):
    """Convert the package's n-1 off-diagonals to the paper Numba kernels' row layout."""
    g = operator.generator(direction)
    return np.stack((np.r_[0.0, g.lower], g.diagonal, np.r_[g.upper, 0.0]))


@njit
def _propagate(initial, drift_op, variance_op, advection, variance, dt, substeps, reverse, first):
    """Crank–Nicolson on identical grids, with a separate conditional law in each column."""
    n_times, n_paths = advection.shape
    output = np.empty((len(initial), n_paths))
    for path in range(n_paths):
        state = initial.copy()
        for k in range(n_times - 1 - first):
            right = n_times - 1 - k if reverse else k + first + 1
            a, b, c = (0.5 * dt) * (
                advection[right, path] * drift_op + variance[right, path] * variance_op
            )
            implicit_diag, explicit_diag = 1.0 - b, 1.0 + b
            lower, upper = -a, -c
            for _ in range(substeps):
                rhs = tridag_mult(a, explicit_diag, c, state)
                state = tridag_solve(lower, implicit_diag, upper, rhs)
        output[:, path] = state
    return output


class BarrierPde1D:
    """Paper adapter using the package's centered generator and weighted forward adjoint.

    Supply right-endpoint advection and instantaneous variance at equally spaced dates,
    including a row at time zero. Each column is a conditional path. The replication
    parameter mt adds mt intermediate nodes (mt + 1 steps per original interval).
    Green laws start at the node nearest x0. All outputs are undiscounted.
    """

    def __init__(self, x_grid):
        self.grid = Grid1D(x_grid)
        if not np.allclose(self.grid.spacings, self.grid.spacings[0], rtol=1e-10, atol=0.0):
            raise ValueError("the replication requires a uniform log-price grid")
        # The two spatial operators are linear in the interval drift and variance.
        # Natural backward rows hold endpoint values. Their weighted adjoints collect exits.
        operators = [
            Operator1D.build(
                self.grid,
                PDE1DCoefficients.constant(self.grid.size, half_var, drift, 0.0),
                BoundaryPair1D.natural(),
                drift_discretization=DriftDiscretization.CENTERED,
            )
            for half_var, drift in ((0.0, 1.0), (0.5, 0.0))
        ]
        self._generators = {
            direction: tuple(_padded_generator(op, direction) for op in operators)
            for direction in (PDEDirection.FORWARD, PDEDirection.BACKWARD)
        }

    def _solve(self, initial, maturity, mt, advection, variance, direction, first):
        if not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be finite and positive")
        if isinstance(mt, bool) or not isinstance(mt, Integral) or mt < 0:
            raise ValueError("mt must be a nonnegative integer")
        drift, var = (np.asarray(a, dtype=float) for a in (advection, variance))
        if (
            drift.shape != var.shape
            or drift.ndim not in (1, 2)
            or drift.shape[0] < 2
            or drift.size == 0
        ):
            raise ValueError("coefficients must have matching (dates,) or (dates, paths) shape")
        if not np.isfinite(drift).all() or not np.isfinite(var).all() or np.any(var < 0):
            raise ValueError("coefficients must be finite and variance nonnegative")
        drift = drift.reshape(len(drift), -1)
        var = var.reshape(len(var), -1)
        dt = maturity / (len(drift) - 1) / (mt + 1)
        return _propagate(
            initial,
            *self._generators[direction],
            drift,
            var,
            dt,
            mt + 1,
            direction is PDEDirection.BACKWARD,
            first,
        )

    def green(self, maturity, mt, advection, variance, *, x0=0.0):
        """Return the surviving interior law and both exits, retaining their original mass."""
        if not self.grid.nodes[0] <= x0 <= self.grid.nodes[-1]:
            raise ValueError("x0 must be inside the log-price domain")
        nearest = np.argmin(np.abs(self.grid.nodes - x0))
        initial = self.grid.project_delta(self.grid.nodes[nearest])
        density = self._solve(initial, maturity, mt, advection, variance, PDEDirection.FORWARD, 0)
        weights = self.grid.quadrature_weights
        lower, upper = weights[0] * density[0], weights[-1] * density[-1]
        density[[0, -1]] = 0.0
        return BarrierGreen(self.grid, density, weights @ density, lower, upper)

    def survival(self, maturity, mt, advection, variance, *, partition="full"):
        """Backward lower-barrier survival. The upper truncation value is one.

        'full' steps to time zero. 'historical' stops at the first original time
        node, exactly as compute_pde_survival_prob does in the replication.
        """
        if partition not in ("full", "historical"):
            raise ValueError("partition must be 'full' or 'historical'")
        terminal = np.ones(self.grid.size)
        terminal[0] = 0.0
        return self._solve(
            terminal,
            maturity,
            mt,
            advection,
            variance,
            PDEDirection.BACKWARD,
            int(partition == "historical"),
        )
