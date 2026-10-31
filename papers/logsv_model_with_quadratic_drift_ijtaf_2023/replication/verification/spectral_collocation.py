"""Experimental full-state Chebyshev transform solver, not a production pricer.

The state is y=log(sigma), and the operator is the same untruncated backward
generator as logvol_pde.py. Chebyshev differentiation replaces the finite-
difference spatial stencil; dense matrix exponentials remove time stepping.
No polynomial truncation of log(transform), coefficient tensors, or approximate
MGF are used. The finite-domain boundary values remain approximations and require
independent domain/node/Fourier convergence checks. A finite returned value is not
an accuracy certificate. In particular, spectral collocation is not monotone.

Only the MMA Lewis contour, flat price backbone and H=1/2 are supported. The
separately returned endpoint identities are normalization conventions, not tests
of the discretized PDE. This experimental module does not change public APIs.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .fourier import calls_from_transform, gaussian_grid
from .logvol_pde import _validate


@dataclass(frozen=True)
class SpectralGrid:
    """State resolution; an explicit half-width overrides asymmetric bounds.

    The asymmetric default was checked on the documented BTC/SPX cases. It is
    not a universal grid for arbitrary calibration candidates.
    """

    space_nodes: int = 97
    logvol_halfwidth: float | None = None
    quadrature_order: int = 32
    cutoff: float = 800.0
    lower_width: float = 3.0
    upper_width: float = 6.0
    mapping: str = "centered_quadratic"

    def __post_init__(self):
        if self.space_nodes < 9 or self.space_nodes % 2 != 1:
            raise ValueError("Use an odd Chebyshev node count >= 9.")
        if self.logvol_halfwidth is not None and (
                not np.isfinite(self.logvol_halfwidth) or self.logvol_halfwidth <= 0):
            raise ValueError("Use a finite positive log-volatility half-width.")
        for width in [self.lower_width, self.upper_width]:
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Asymmetric widths must be finite and positive.")
        if self.mapping not in ["affine", "centered_quadratic"]:
            raise ValueError("Use affine or centered_quadratic mapping.")


def differentiation_matrices(nodes: int, halfwidth: float):
    """Ascending Lobatto nodes and barycentric first/second derivative matrices."""
    x = -np.cos(np.pi*np.arange(nodes)/(nodes-1))
    weights = (-1.0)**np.arange(nodes)
    weights[[0, -1]] *= 0.5
    differences = x[:, None]-x[None, :]
    np.fill_diagonal(differences, 1.0)
    derivative = weights[None, :]/weights[:, None]/differences
    np.fill_diagonal(derivative, 0.0)
    np.fill_diagonal(derivative, -derivative.sum(axis=1))
    derivative /= halfwidth
    return halfwidth*x, derivative, derivative@derivative


def logvol_transform(params, maturity: float, phi, grid: SpectralGrid = SpectralGrid()):
    """E[exp(-phi X_T)] by a spatial collocation and exact matrix propagation.

    Boundary values are 1 at the remote low-volatility endpoint and 0 at the high
    endpoint. An augmented constant state handles their inhomogeneous forcing.
    Interior contour transforms must be finite and satisfy the martingale-based
    modulus bound up to numerical tolerance; this is only a failure guard.
    """
    _validate(params, maturity)
    phi = np.asarray(phi, dtype=complex)
    if phi.ndim != 1 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a finite one-dimensional array.")
    endpoint = (phi == 0) | (phi == -1)
    if not np.all(endpoint | np.isclose(phi.real, -0.5, atol=1e-14, rtol=0.0)):
        raise NotImplementedError("Only the MMA Lewis contour and exact endpoints are supported.")
    result = np.ones(len(phi), dtype=complex)
    if maturity == 0 or np.all(endpoint):
        return result
    lower = grid.lower_width if grid.logvol_halfwidth is None else grid.logvol_halfwidth
    upper = grid.upper_width if grid.logvol_halfwidth is None else grid.logvol_halfwidth
    if grid.mapping == "affine":
        offsets, dy, dyy = differentiation_matrices(grid.space_nodes, 0.5*(lower+upper))
        offsets += 0.5*(upper-lower)
    else:
        # y(x)=b*x+c*x^2 keeps y(0)=0 exactly while reaching asymmetric endpoints.
        # Chain-rule differentiation avoids a posteriori interpolation at sigma0.
        x, dx, dxx = differentiation_matrices(grid.space_nodes, 1.0)
        b, c = 0.5*(upper+lower), 0.5*(upper-lower)
        jacobian = b+2*c*x
        if np.min(jacobian) <= 0:
            raise ValueError("Centered quadratic mapping requires max(widths) < 3*min(widths).")
        offsets = b*x+c*x*x
        dy = dx/jacobian[:, None]
        dyy = dxx/(jacobian[:, None]**2)-2*c*dx/(jacobian[:, None]**3)
    sigma = params.sigma0*np.exp(offsets)
    v2 = params.beta**2+params.volvol**2
    drift = (params.kappa1*params.theta/sigma+params.kappa2*params.theta
             -params.kappa1-params.kappa2*sigma-0.5*v2)
    physical = 0.5*v2*dyy+drift[:, None]*dy
    # Barycentric interpolation at log(sigma0). Symmetric odd grids hit a node exactly.
    evaluation = np.zeros(grid.space_nodes)
    at_zero = np.flatnonzero(abs(offsets) < 1e-14)
    if len(at_zero):
        evaluation[at_zero[0]] = 1.0
    else:
        evaluation = (-1.0)**np.arange(grid.space_nodes)/(-offsets)
        evaluation[[0, -1]] *= 0.5
        evaluation /= evaluation.sum()
    dimension = grid.space_nodes-1  # interior nodes plus the constant forcing state
    initial = np.ones(dimension, dtype=complex)
    for q in np.flatnonzero(~endpoint):
        z = phi[q]
        operator = physical-z*params.beta*sigma[:, None]*dy
        operator[np.diag_indices(grid.space_nodes)] += 0.5*z*(z+1)*sigma**2
        augmented = np.zeros((dimension, dimension), dtype=complex)
        augmented[:-1, :-1] = operator[1:-1, 1:-1]
        augmented[:-1, -1] = operator[1:-1, 0]  # low boundary is 1; high boundary 0
        propagated = expm(maturity*augmented)@initial
        result[q] = evaluation[0]+evaluation[1:-1]@propagated[:-1]
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Nonfinite collocation transform; reject this configuration.")
    if np.max(np.abs(result), initial=0.0) > 1.0+1e-7:
        raise FloatingPointError("Collocation transform violates the Lewis-contour modulus bound.")
    return result


def call_prices(params, maturity: float, strikes, grid: SpectralGrid = SpectralGrid()):
    """Forward-normalized calls without clipping, using the common Lewis inversion."""
    phi, weights = gaussian_grid(grid.quadrature_order, grid.cutoff)
    transform = logvol_transform(params, maturity, phi, grid)
    return calls_from_transform(transform, phi, weights, maturity, params.sigma0, strikes)
