"""Bounded, repository-only unsplit reference for the existing 2D spatial operator.

Backward Euler and Crank--Nicolson use sparse direct solves. The exponential action
is restricted to tiny grids. All three preserve the spatial discretization,
including its natural endpoint rows and possible negative off-diagonal entries.
They are time-discretization references, not independent continuous-PDE solutions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from numbers import Integral

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply, splu

from stochvolmodels.pde_solvers import BoundaryKind, PDEDirection, TensorGrid2D
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D


@dataclass(frozen=True)
class UnsplitConfig:
    """Explicit methods, problem-size bounds and normalized linear residual limit."""

    methods: tuple[str, ...] = ("be", "cn")
    max_nodes: int = 10_000
    max_expm_nodes: int = 256
    residual_tolerance: float = 1e-11

    def __post_init__(self):
        """Reject invalid policies before assembling or factorizing a matrix."""
        if (not isinstance(self.methods, tuple) or not self.methods
                or any(m not in ("be", "cn", "expm") for m in self.methods)
                or len(set(self.methods)) != len(self.methods)):
            raise ValueError("methods must be a nonempty tuple of distinct be, cn or expm")
        for name in ("max_nodes", "max_expm_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 9:
                raise ValueError(f"{name} must be an integer of at least nine")
        if not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be finite and positive")


@dataclass(frozen=True)
class UnsplitStep:
    """Unrepaired state and absolute/scaled infinity-norm linear solve residuals."""

    values: np.ndarray
    residual_absolute: float | None
    residual_relative: float | None


@dataclass
class Unsplit2DReference:
    """Full sparse generator in C-order, including every natural boundary node.

    Construct with ``from_split``. The forward generator is W^-1 A.T W. A theta
    step uses (I-theta*h*A)^-1 (I+(1-theta)*h*A); its forward map is also the
    weighted adjoint because the two factors are functions of the same frozen A.
    """

    grid: TensorGrid2D
    backward: sparse.csc_matrix
    forward: sparse.csc_matrix
    config: UnsplitConfig
    statistics: dict
    _factors: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_split(cls, operator: SplitOperator2D, config: UnsplitConfig | None = None):
        """Assemble sparse line bands and the exact tensor mixed stencil, without probing."""
        config = UnsplitConfig() if config is None else config
        if not isinstance(operator, SplitOperator2D) or not isinstance(config, UnsplitConfig):
            raise TypeError("expected SplitOperator2D and UnsplitConfig")
        grid = operator.grid
        nx, ny = grid.shape
        n = nx * ny
        if n > config.max_nodes:
            raise ValueError(f"grid has {n} nodes, exceeding max_nodes={config.max_nodes}")
        if any(c.kind is not BoundaryKind.NATURAL for pair in
               (operator.boundaries.x, operator.boundaries.y) for c in (pair.left, pair.right)):
            raise ValueError("unsplit reference supports natural boundaries only")
        rows, columns, entries = [], [], []

        def append_lines(lines, indices):
            for line, index in zip(lines, indices, strict=True):
                a = line.backward
                rows.extend((index, index[1:], index[:-1]))
                columns.extend((index, index[:-1], index[1:]))
                entries.extend((a.diagonal, a.lower, a.upper))

        append_lines(operator.x_lines, (np.arange(nx) * ny + j for j in range(ny)))
        append_lines(operator.y_lines, (i * ny + np.arange(ny) for i in range(nx)))
        axial = sparse.coo_matrix((np.concatenate(entries),
                                   (np.concatenate(rows), np.concatenate(columns))), shape=(n, n))
        dx, dy = sparse.csr_matrix(operator.derivative_x), sparse.csr_matrix(operator.derivative_y)
        if dx.nnz * dy.nnz > 9 * n:
            raise ValueError("mixed derivative exceeds the bounded local-stencil sparsity")
        mixed = sparse.diags(operator.coefficients.covariance_xy.ravel()) @ sparse.kron(
            dx, dy, format="csc")
        backward = (axial + mixed).tocsc()
        backward.sum_duplicates()
        backward.eliminate_zeros()
        w = grid.quadrature_weights.ravel()
        forward = (sparse.diags(1 / w) @ backward.T @ sparse.diags(w)).tocsc()
        if not np.isfinite(backward.data).all() or not np.isfinite(forward.data).all():
            raise ValueError("assembled generator must be finite")
        coo = backward.tocoo()
        off = coo.data[coo.row != coo.col]
        stats = {
            "nodes": n, "nonzeros": backward.nnz,
            "negative_off_diagonals": int(np.count_nonzero(off < 0)),
            "minimum_off_diagonal": float(np.min(off)) if off.size else 0.,
            "constant_discount_residual": float(np.max(np.abs(
                backward @ np.ones(n) + operator.coefficients.discount.ravel()))),
        }
        return cls(grid, backward, forward, config, stats)

    def generator(self, direction: PDEDirection):
        """Return the generator for a validated propagation direction."""
        if not isinstance(direction, PDEDirection):
            raise TypeError("direction must be a PDEDirection")
        return self.backward if direction is PDEDirection.BACKWARD else self.forward

    def step(self, values, direction: PDEDirection, step_size: float, method: str = "be"):
        """Apply a full implicit/CN/exponential step and expose failed solves, without repair."""
        a = self.generator(direction)
        dt = float(step_size)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("step_size must be finite and positive")
        if method not in ("be", "cn", "expm"):
            raise ValueError("method must be be, cn or expm")
        state = self.grid.validate_shape(values).ravel()
        if not np.isfinite(state).all():
            raise ValueError("values must be finite")
        if method == "expm":
            if state.size > self.config.max_expm_nodes:
                raise ValueError("matrix exponential exceeds max_expm_nodes")
            answer = expm_multiply(dt * a, state, traceA=dt * float(a.diagonal().sum()))
            absolute = relative = None  # No linear system is solved by expm_multiply.
        else:
            theta = 1. if method == "be" else .5
            key = (direction, theta * dt)
            if key not in self._factors:
                matrix = sparse.eye(state.size, format="csc") - theta * dt * a
                norm = float(np.max(np.asarray(abs(matrix).sum(axis=1))))
                lu = splu(matrix)
                # Exact step sizes only, with bounded fill-in storage on graded meshes.
                if len(self._factors) >= 4:
                    self._factors.clear()
                self._factors[key] = matrix, norm, lu
            matrix, norm, lu = self._factors[key]
            rhs = state + ((1 - theta) * dt) * (a @ state)
            answer = lu.solve(rhs)
            absolute = float(np.max(np.abs(matrix @ answer - rhs)))
            scale = norm * np.max(np.abs(answer)) + np.max(np.abs(rhs))
            if not np.isfinite(scale):
                raise RuntimeError("nonfinite unsplit residual scaling")
            relative = absolute / max(float(scale), np.finfo(float).tiny)
            if not np.isfinite(relative) or relative > self.config.residual_tolerance:
                raise RuntimeError(f"unsplit linear residual {relative:g} exceeds tolerance")
        if not np.isfinite(answer).all():
            raise RuntimeError("unsplit step produced a nonfinite state")
        return UnsplitStep(answer.reshape(self.grid.shape), absolute, relative)


def density_diagnostics(grid: TensorGrid2D, values) -> dict:
    """Observe signed mass, raw moments, covariance and boundary mass without clipping."""
    values = grid.validate_shape(values)
    mass = grid.integrate(values)
    if not np.isfinite(values).all() or not np.isfinite(mass):
        raise ValueError("density must be finite")
    mx = my = vx = vy = cov = None
    if abs(mass) > np.finfo(float).tiny:
        mx = grid.integrate(grid.x_mesh * values) / mass
        my = grid.integrate(grid.y_mesh * values) / mass
        vx = grid.integrate((grid.x_mesh - mx)**2 * values) / mass
        vy = grid.integrate((grid.y_mesh - my)**2 * values) / mass
        cov = grid.integrate((grid.x_mesh - mx) * (grid.y_mesh - my) * values) / mass
    return {"mass": mass, "negative_mass": -grid.integrate(np.minimum(values, 0.)),
            "mean_x": mx, "mean_y": my, "variance_x": vx, "variance_y": vy,
            "covariance_xy": cov, "minimum_value": float(values.min()),
            "maximum_value": float(values.max()),
            "boundary_mass": asdict(grid.boundary_mass(values))}
