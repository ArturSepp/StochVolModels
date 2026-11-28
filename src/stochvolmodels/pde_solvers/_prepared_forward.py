"""Prepared forward theta steps for a static one-dimensional generator.

The spatial operator and its implicit matrices come from Operator1D. Reuse each LU
only on the same generator, theta and exact floating-point step size. The caller
applies observation operators between segments.
"""

import numpy as np
from numba import njit

from stochvolmodels.pde_solvers.operators_1d import Operator1D, PDEDirection


@njit(cache=True)
def _factor(lower, diagonal, upper, tolerance):
    pivots = diagonal.copy()
    multipliers = np.empty_like(lower)
    for j in range(len(lower)):
        if abs(pivots[j]) <= tolerance:
            raise np.linalg.LinAlgError("near-zero pivot in prepared forward step")
        multipliers[j] = lower[j] / pivots[j]
        pivots[j+1] -= multipliers[j] * upper[j]
    if abs(pivots[-1]) <= tolerance:
        raise np.linalg.LinAlgError("near-zero pivot in prepared forward step")
    return multipliers, pivots


@njit(cache=True)
def _step(x, rhs, lower, diagonal, upper, explicit, mult, pivots, implicit_upper, weights):
    n = len(x)
    for j in range(n):
        v = diagonal[j] * x[j]
        if j+1 < n:
            v += upper[j] * x[j+1]
        if j:
            v += lower[j-1] * x[j-1]
        rhs[j] = x[j] + explicit * v
    for j in range(1,n):
        rhs[j] -= mult[j-1] * rhs[j-1]
    x[-1] = rhs[-1] / pivots[-1]
    negative = min(x[-1],0.0) * weights[-1]
    for j in range(n-2,-1,-1):
        x[j] = (rhs[j] - implicit_upper[j]*x[j+1]) / pivots[j]
        negative += min(x[j],0.0) * weights[j]
    return -negative


class PreparedForwardSteps:
    """Forward density on one fixed grid with no mass renormalization."""

    def __init__(self, grid, provider, boundaries):
        self.operator = Operator1D.build(grid, provider.evaluate(0.0, grid), boundaries)
        self.weights = grid.quadrature_weights
        self.factors = {}
        self.rhs = np.empty(grid.size)

    def advance(self, times, theta, density):
        generator = self.operator.forward
        peak_negative = 0.0
        for dt in np.diff(times):
            key = theta, float(dt)
            factors = self.factors.get(key)
            if factors is None:
                matrix = self.operator.implicit_matrix(PDEDirection.FORWARD, dt, theta)
                scale = max(1.0, np.max(np.abs(matrix.diagonal)), np.max(np.abs(matrix.lower)),
                            np.max(np.abs(matrix.upper)))
                mult,pivots = _factor(matrix.lower,matrix.diagonal,matrix.upper,
                                    64*np.finfo(float).eps*scale*matrix.size)
                factors = self.factors[key] = mult,pivots,matrix.upper
            negative = _step(density,self.rhs,generator.lower,generator.diagonal,generator.upper,
                             (1.0-theta)*dt,*factors,self.weights)
            peak_negative = max(peak_negative, negative)
        if not np.isfinite(density).all():
            raise FloatingPointError("non-finite prepared forward density")
        return peak_negative
