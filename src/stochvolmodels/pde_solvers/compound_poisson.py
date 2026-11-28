"""Compound-Poisson propagation of exponential marks on additive 1D and tensor grids."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exponential_jumps import ExponentialJumpKernel
from .grids import Grid1D, TensorGrid2D
from .jumps import _poisson_weights, _validated_step_size
from .operators_1d import PDEDirection


@dataclass(frozen=True, slots=True, eq=False)
class ExponentialJumpStep:
    """One jump step, its omitted upper count tail and physical forward overshoot mass."""

    values: NDArray[np.float64]
    poisson_terms: int
    truncated_probability: float
    killed_mass: float | None


@dataclass(frozen=True, slots=True, eq=False)
class ExponentialJumpProcess:
    """Constant-rate independent Poisson clocks for individual and common exponential marks.

    Each active axis requires a prepared kernel on its own grid. A common event draws
    independent marks from both kernels. Intensities are exclusive arrival rates per
    unit time. A coordinate's total arrival rate includes its common clock.
    """

    grid: Grid1D | TensorGrid2D
    _: KW_ONLY
    x_kernel: ExponentialJumpKernel | None = None
    x_intensity: float = 0.0
    y_kernel: ExponentialJumpKernel | None = None
    y_intensity: float = 0.0
    common_intensity: float = 0.0
    total_intensity: float = field(init=False)
    _probabilities: tuple[float, float, float] = field(init=False, repr=False)
    _exit_weights: NDArray[np.float64] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Prepare event probabilities and the quadrature functional for killed overshoots."""
        if not isinstance(self.grid, (Grid1D, TensorGrid2D)):
            raise TypeError("grid must be a Grid1D or TensorGrid2D")
        for key in ("x_intensity", "y_intensity", "common_intensity"):
            rate = float(getattr(self, key))
            if not np.isfinite(rate) or rate < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            object.__setattr__(self, key, rate)
        total = self.x_intensity + self.y_intensity + self.common_intensity
        if not np.isfinite(total):
            raise ValueError("total event intensity must be finite")
        tensor = isinstance(self.grid, TensorGrid2D)
        if not tensor and (self.y_kernel is not None or self.y_intensity or self.common_intensity):
            raise ValueError("y and common events require a tensor grid")
        x = self.grid.x if tensor else self.grid
        self._check_axis(x, self.x_kernel, self.x_intensity + self.common_intensity)
        if tensor:
            self._check_axis(self.grid.y, self.y_kernel, self.y_intensity + self.common_intensity)
        px, py, pc = (
            rate / total if total else 0.0
            for rate in (self.x_intensity, self.y_intensity, self.common_intensity)
        )
        object.__setattr__(self, "total_intensity", total)
        object.__setattr__(self, "_probabilities", (px, py, pc))
        # One common arrival can exit on either axis. Count a double exit only once.
        ex = ey = 0.0
        if (px or pc) and self.x_kernel.exterior == "zero":
            ex = self.x_kernel.exterior_probability
            if tensor:
                ex = ex[:, None]
        if (py or pc) and self.y_kernel.exterior == "zero":
            ey = self.y_kernel.exterior_probability[None, :]
        weights = None
        if np.ndim(ex) or np.ndim(ey):
            weights = self.grid.quadrature_weights * (px * ex + py * ey + pc * (ex + (1 - ex) * ey))
            weights.setflags(write=False)
        object.__setattr__(self, "_exit_weights", weights)

    @staticmethod
    def _check_axis(grid, kernel, intensity):
        """Validate the active kernel and the geometry used by its quadrature adjoint."""
        if kernel is None:
            if intensity:
                raise ValueError("an active clock requires a kernel on each affected axis")
        elif not isinstance(kernel, ExponentialJumpKernel):
            raise TypeError("each kernel must be an ExponentialJumpKernel")
        elif not np.array_equal(grid.nodes, kernel.grid.nodes):
            raise ValueError("kernel grid nodes differ from process grid nodes")

    def validate_grid(self, grid: Grid1D | TensorGrid2D) -> None:
        """Reject a solver grid with a different dimension or different nodes."""
        if grid is self.grid:
            return
        if isinstance(grid, Grid1D) and isinstance(self.grid, Grid1D):
            if np.array_equal(grid.nodes, self.grid.nodes):
                return
        if isinstance(grid, TensorGrid2D) and isinstance(self.grid, TensorGrid2D):
            if np.array_equal(grid.x.nodes, self.grid.x.nodes) and np.array_equal(
                grid.y.nodes, self.grid.y.nodes
            ):
                return
        raise ValueError("solver and jump process must use matching grid nodes")

    def _event(self, values, direction):
        """Apply the intensity-weighted event mixture in at most two 1D sweeps."""
        px, py, pc = self._probabilities
        if pc:
            x = self.x_kernel.apply(values, direction)
            return px * x + self.y_kernel.apply(py * values + pc * x, direction, axis=1)
        if px:
            x = self.x_kernel.apply(values, direction)
            if not py:
                return x
            return px * x + py * self.y_kernel.apply(values, direction, axis=1)
        return self.y_kernel.apply(values, direction, axis=1)

    def poisson_step(
        self,
        values: ArrayLike,
        direction: PDEDirection,
        step_size: float,
        *,
        tail_tolerance: float = 1e-14,
        max_terms: int = 10_000,
    ) -> ExponentialJumpStep:
        """Sum the event-kernel powers with normalized retained Poisson probabilities.

        Forward killed mass sums the analytic one-event overshoot functional along the
        surviving states. It uses the same retained count distribution as the propagated
        density. For signed input densities this diagnostic is also signed.
        """
        if not isinstance(direction, PDEDirection):
            raise TypeError("direction must be a PDEDirection")
        term = self.grid.validate_shape(values)
        if not np.all(np.isfinite(term)):
            raise ValueError("values must be finite")
        dt = _validated_step_size(step_size)
        weights, tail = _poisson_weights(self.total_intensity * dt, tail_tolerance, max_terms)
        result = weights[0] * term
        lost_at_count = killed = 0.0
        forward = direction is PDEDirection.FORWARD
        for weight in weights[1:]:
            if forward and self._exit_weights is not None:
                lost_at_count += np.sum(self._exit_weights * term)
                killed += weight * lost_at_count
            term = self._event(term, direction)
            result += weight * term
        result.setflags(write=False)
        return ExponentialJumpStep(result, len(weights), tail, float(killed) if forward else None)
