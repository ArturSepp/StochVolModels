"""Independent finite-difference transform PDE for the paper's LogSV dynamics.

Repository-only numerical reference for the MMA measure, flat price backbone,
and H=1/2. For y=log(sigma), v2=beta**2+volvol**2 and
a(y)=kappa1*theta*exp(-y)+kappa2*theta-kappa1-kappa2*exp(y)-v2/2,
the reduced version of published Eq. (4.6) is

    u_tau = v2*u_yy/2 + (a(y)-phi*beta*exp(y))*u_y
            + phi*(phi+1)*exp(2*y)*u/2,  u(0,y)=1.

No finite polynomial expansion or M/L/H coefficient tensors are used here.
Approximate remote boundaries (one at low volatility, zero at high volatility)
are used only on the Lewis contour. Exact transform endpoints phi=0,-1 are
handled separately under the enforced martingale condition. Grid/domain checks
remain necessary. Artificial diffusion stabilizes the physical-drift stencil;
this is not a proof of unconditional stability for the complex tilted operator.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from stochvolmodels import LogSvParams
from .fourier import calls_from_transform, gaussian_grid


@dataclass(frozen=True)
class PdeGrid:
    """Finite domain, time grid and inversion resolution for an audit run."""

    space_nodes: int = 801
    time_steps: int = 512
    logvol_halfwidth: float = 6.
    quadrature_order: int = 48
    cutoff: float = 800.

    def __post_init__(self):
        if self.space_nodes < 5 or self.space_nodes % 2 != 1 or self.time_steps < 1:
            raise ValueError("Require an odd spatial node count >= 5 and positive time steps.")
        if not np.isfinite(self.logvol_halfwidth) or self.logvol_halfwidth <= 0:
            raise ValueError("Require a positive finite log-volatility half-width.")


def _validate(params, maturity):
    values = [params.sigma0, params.theta, params.kappa1, params.kappa2,
              params.beta, params.volvol, maturity]
    if not np.all(np.isfinite(values)) or maturity < 0:
        raise ValueError("Finite parameters and nonnegative maturity are required.")
    if min(params.sigma0, params.theta) <= 0 or min(params.kappa1, params.kappa2,
                                                  params.volvol) < 0:
        raise ValueError("Positive volatility levels and nonnegative drift/orthogonal vol required.")
    if params.kappa2 < params.beta:
        raise ValueError("Normalized vanilla pricing requires kappa2 >= beta (MMA martingale).")
    if params.H != .5 or params.vol_backbone is not None:
        raise NotImplementedError("The reference PDE supports H=1/2 and no volatility backbone.")


@njit(cache=False)
def _evolve(phi, y, sigma0, theta, k1, k2, beta, eps, maturity, steps):
    n = len(y)
    dy = y[1]-y[0]
    v2 = beta*beta+eps*eps
    sig = np.exp(y)
    drift = k1*theta/sig+k2*theta-k1-k2*sig-.5*v2
    output = np.empty(len(phi), np.complex128)
    ix = np.searchsorted(y, np.log(sigma0))-1
    frac = (np.log(sigma0)-y[ix])/dy
    for q in range(len(phi)):
        z = phi[q]
        lo = np.empty(n, np.complex128)
        diag = np.empty(n, np.complex128)
        hi = np.empty(n, np.complex128)
        for j in range(n):
            diffusion = max(.5*v2, .5*dy*abs(drift[j]))
            mu = drift[j]-z*beta*sig[j]
            lo[j] = diffusion/(dy*dy)-mu/(2*dy)
            hi[j] = diffusion/(dy*dy)+mu/(2*dy)
            diag[j] = -2*diffusion/(dy*dy)+.5*z*(z+1)*sig[j]*sig[j]
        u = np.ones(n, dtype=np.complex128)
        u[-1] = 0
        # Two implicit half steps smooth the initial/remote-boundary mismatch.
        for stage in range(2):
            dt = maturity/steps*(.5 if stage == 0 else 1.)
            alpha = 1. if stage == 0 else .5
            iterations = 2 if stage == 0 else steps-1
            b = np.empty(n-2, np.complex128)
            c = np.empty(n-2, np.complex128)
            a = np.empty(n-2, np.complex128)
            for i in range(n-2):
                j = i+1
                a[i] = -alpha*dt*lo[j]
                b[i] = 1-alpha*dt*diag[j]
                c[i] = -alpha*dt*hi[j]
            denom = np.empty(n-2, np.complex128)
            cprime = np.empty(n-2, np.complex128)
            denom[0] = b[0]
            cprime[0] = c[0]/denom[0]
            for i in range(1, n-2):
                denom[i] = b[i]-a[i]*cprime[i-1]
                cprime[i] = c[i]/denom[i]
            for _ in range(iterations):
                r = np.empty(n-2, np.complex128)
                for i in range(n-2):
                    j = i+1
                    r[i] = u[j]+(1-alpha)*dt*(lo[j]*u[j-1]+diag[j]*u[j]+hi[j]*u[j+1])
                r[0] -= a[0]
                r[0] /= denom[0]
                for i in range(1, n-2):
                    r[i] = (r[i]-a[i]*r[i-1])/denom[i]
                u[-2] = r[-1]
                for i in range(n-4, -1, -1):
                    u[i+1] = r[i]-cprime[i]*u[i+2]
        output[q] = (1-frac)*u[ix]+frac*u[ix+1]
    return output


def logvol_transform(params: LogSvParams, maturity: float, phi: np.ndarray,
                     grid: PdeGrid = PdeGrid()) -> np.ndarray:
    """Compute the MMA return transform on the valuation contour.

    Endpoints zero and minus one return the exact probability/martingale
    identities, rather than applying contour-specific boundary approximations.
    Other contours and joint QV/terminal-volatility transforms are unsupported.
    """
    _validate(params, maturity)
    phi = np.asarray(phi, dtype=np.complex128)
    if phi.ndim != 1 or not np.all(np.isfinite(phi)):
        raise ValueError("phi must be a finite one-dimensional vector.")
    endpoint = (phi == 0) | (phi == -1)
    if not np.all(endpoint | np.isclose(phi.real, -.5, atol=1e-14, rtol=0)):
        raise NotImplementedError("Only the Lewis contour and exact endpoints are supported.")
    result = np.ones(len(phi), complex)
    if maturity and np.any(~endpoint):
        y = np.linspace(np.log(params.sigma0)-grid.logvol_halfwidth,
                        np.log(params.sigma0)+grid.logvol_halfwidth, grid.space_nodes)
        result[~endpoint] = _evolve(phi[~endpoint], y, params.sigma0, params.theta,
                                   params.kappa1, params.kappa2, params.beta, params.volvol,
                                   maturity, grid.time_steps)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("The finite-difference transform became nonfinite.")
    return result


def call_prices(params: LogSvParams, maturity: float, strikes: np.ndarray,
                grid: PdeGrid = PdeGrid()) -> np.ndarray:
    """Price forward-normalized undiscounted calls using the grid transform."""
    phi, weights = gaussian_grid(grid.quadrature_order, grid.cutoff)
    transform = logvol_transform(params, maturity, phi, grid)
    return calls_from_transform(transform, phi, weights, maturity, params.sigma0, strikes)
