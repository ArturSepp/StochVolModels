"""Conservative forward translations along the first axis of a tensor density grid.

A Poisson count moves mass by its total additive displacement. Linear interpolation
deposits quadrature mass onto the adjacent nodes, clamping outside the grid. The
resulting sparse matrix acts on pointwise densities and preserves every y marginal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.stats import poisson

from stochvolmodels.pde_solvers.grids import Grid1D


class FixedXJump(Protocol):
    """Grid-independent constant event rate and fixed displacement in the x coordinate."""

    @property
    def intensity(self) -> float:
        """Poisson events per unit time."""
        ...

    @property
    def jump_size(self) -> float:
        """Additive x displacement per event."""
        ...


@dataclass(frozen=True, slots=True)
class _ForwardTranslation:
    """Prepared transition and source weights for measuring out-of-grid mass."""

    matrix: csr_matrix
    clamped_weights: NDArray[np.float64]

    def clamped_mass(self, values: NDArray[np.float64], y: Grid1D) -> float:
        """Positive incoming mass whose translated location is outside the x domain."""
        return float(self.clamped_weights @ (np.maximum(values, 0.0) @ y.quadrature_weights))


def _forward_translation(
    grid: Grid1D, intensity: float, size: float, dt: float
) -> _ForwardTranslation:
    """Build the quadrature-weighted forward adjoint of the Poisson interpolation map."""
    mean = intensity * dt
    # A bounded term budget also prevents accidental allocation for an unresolved time step.
    if not np.isfinite(mean) or mean >= 10_000:
        raise ValueError("Poisson step exceeds the count budget; refine the time partition")
    last = max(1, int(poisson.isf(1e-14, mean)))
    if last >= 10_000:
        raise ValueError("Poisson step exceeds the count budget; refine the time partition")
    probabilities = poisson.pmf(np.arange(last + 1), mean)
    probabilities /= probabilities.sum()
    x, weights = grid.nodes, grid.quadrature_weights
    source = np.arange(grid.size)
    rows = [source]
    cols = [source]
    entries = [np.full(grid.size, probabilities[0])]
    clamped = np.zeros(grid.size)
    for n, probability in enumerate(probabilities[1:], 1):
        target = x + n * size
        clamped += probability * ((target < x[0]) | (target > x[-1]))
        target = np.clip(target, x[0], x[-1])
        right = np.clip(np.searchsorted(x, target), 1, grid.size - 1)
        left = right - 1
        fraction = (target - x[left]) / (x[right] - x[left])
        rows.extend((left, right))
        cols.extend((source, source))
        mass = probability * weights
        entries.extend((mass * (1.0 - fraction) / weights[left], mass * fraction / weights[right]))
    transition = csr_matrix(
        (np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))),
        shape=(grid.size, grid.size),
    )
    transition.eliminate_zeros()
    return _ForwardTranslation(transition, clamped * weights)
