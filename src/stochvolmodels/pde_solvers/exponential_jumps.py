"""Single-event exponential jump expectations on nonuniform additive grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from .grids import Grid1D
from .operators_1d import PDEDirection


@njit
def _backward(values, decay, far, near, clamp):
    result = np.zeros_like(values)
    if clamp:
        result[0] = values[0]
    for i in range(1, values.shape[0]):
        for j in range(values.shape[1]):
            result[i, j] = (
                decay[i - 1] * result[i - 1, j]
                + far[i - 1] * values[i - 1, j]
                + near[i - 1] * values[i, j]
            )
    return result


@njit
def _transpose(mass, decay, far, near, clamp):
    # Reverse accumulation through the backward recurrence, including its initial value.
    result = np.zeros_like(mass)
    carry = mass[-1].copy()
    for i in range(mass.shape[0] - 1, 0, -1):
        for j in range(mass.shape[1]):
            result[i, j] += near[i - 1] * carry[j]
            result[i - 1, j] += far[i - 1] * carry[j]
            carry[j] = mass[i - 1, j] + decay[i - 1] * carry[j]
    if clamp:
        result[0] += carry
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ExponentialJumpKernel:
    """Prepared expectation for one signed exponential mark in an additive coordinate.

    ``rate`` is the inverse mean mark magnitude, in inverse coordinate units.
    ``side="down"`` evaluates the backward payoff at x minus the mark magnitude.
    ``exterior="clamp"`` extends the endpoint value and ``"zero"`` kills the overshoot.
    The forward operation acts on pointwise densities under the grid quadrature.
    Poisson arrival intensity and calendar time are supplied by the caller.
    """

    grid: Grid1D
    rate: float
    side: Literal["down", "up"] = "down"
    exterior: Literal["clamp", "zero"] = "clamp"
    exterior_probability: NDArray[np.float64] = field(init=False, repr=False)
    _decay: NDArray[np.float64] = field(init=False, repr=False)
    _far: NDArray[np.float64] = field(init=False, repr=False)
    _near: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        rate = float(self.rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        if self.side not in ("down", "up") or self.exterior not in ("clamp", "zero"):
            raise ValueError("expected side down/up and exterior clamp/zero")
        step = 1 if self.side == "down" else -1
        nodes = self.grid.nodes[::step]
        with np.errstate(over="ignore", under="ignore"):
            z = rate * self.grid.spacings[::step]
            decay = np.exp(-z)
            tail = np.exp(-rate * np.abs(nodes - nodes[0]))[::step]
        far = np.empty_like(z)
        near = np.empty_like(z)
        small = z < 1e-3
        t = z[small]
        # Linear-basis integrals. Direct subtraction loses both weights for tiny cells.
        far[small] = t * (0.5 + t * (-1 / 3 + t * (1 / 8 - t / 30)))
        near[small] = t * (0.5 + t * (-1 / 6 + t * (1 / 24 - t / 120)))
        mean_decay = -np.expm1(-z[~small]) / z[~small]
        far[~small] = mean_decay - decay[~small]
        near[~small] = 1.0 - mean_decay
        object.__setattr__(self, "rate", rate)
        for name, values in (
            ("_decay", decay),
            ("_far", far),
            ("_near", near),
            ("exterior_probability", tail),
        ):
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def apply(
        self, values: ArrayLike, direction: PDEDirection, *, axis: int = 0
    ) -> NDArray[np.float64]:
        """Apply a backward expectation or its weighted forward adjoint along one axis."""
        if not isinstance(direction, PDEDirection):
            raise TypeError("direction must be a PDEDirection")
        data = np.asarray(values, dtype=np.float64)
        if data.ndim not in (1, 2) or axis not in range(data.ndim):
            raise ValueError("expected a vector or matrix and a valid axis")
        if data.shape[axis] != self.grid.size or not np.all(np.isfinite(data)):
            raise ValueError("values must be finite and match the kernel grid on the jump axis")
        columns = np.moveaxis(data, axis, 0).reshape(self.grid.size, -1)
        step = 1 if self.side == "down" else -1
        oriented = np.ascontiguousarray(columns[::step])
        clamp = self.exterior == "clamp"
        if direction is PDEDirection.BACKWARD:
            result = _backward(oriented, self._decay, self._far, self._near, clamp)
        else:
            weights = self.grid.quadrature_weights[::step, None]
            result = _transpose(oriented * weights, self._decay, self._far, self._near, clamp)
            result /= weights
        result = result[::step]
        if data.ndim == 1:
            return result[:, 0]
        return np.moveaxis(result, 0, axis)

    def exterior_mass(self, density: ArrayLike) -> float:
        """Mass that jumps beyond the domain in one event, killed or clamped by policy."""
        values = self.grid.validate_shape(density)
        if not np.all(np.isfinite(values)):
            raise ValueError("density must be finite")
        return self.grid.integrate(values * self.exterior_probability)


def apply_common_exponential_jump(
    values: ArrayLike,
    x_kernel: ExponentialJumpKernel,
    y_kernel: ExponentialJumpKernel,
    direction: PDEDirection,
) -> NDArray[np.float64]:
    """Apply one common event with independent marks and axis-specific exterior policies.

    The tensor uses (x, y) order. Each sweep moves one coordinate even when the other
    coordinate reaches a clamped face. Applying the two sweeps costs O(nx * ny).
    """
    if np.shape(values) != (x_kernel.grid.size, y_kernel.grid.size):
        raise ValueError("expected a tensor matching the x and y kernel grids")
    return y_kernel.apply(x_kernel.apply(values, direction, axis=0), direction, axis=1)
