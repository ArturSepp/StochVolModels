"""Gaussian Lewis integration shared by the paper's numerical reference checks.

This repository-only helper does not change the library's historical quadrature.
Prices are undiscounted and forward-normalized. The transform convention is
E[exp(-phi X)] with X(0)=0 and phi=-1/2+i*u.
"""
import numpy as np
from scipy.special import ndtr, roots_legendre


def gaussian_grid(order=48, cutoff=800.0):
    """Return a composite Gauss-Legendre grid on the Lewis contour."""
    if order < 8 or not np.isfinite(cutoff) or cutoff <= 256:
        raise ValueError("Require order >= 8 and cutoff > 256.")
    edges = [0., 1., 4., 16., 64., 256., float(cutoff)]
    z, w = roots_legendre(order)
    u = np.concatenate([.5*(a+b)+.5*(b-a)*z for a, b in zip(edges[:-1], edges[1:])])
    weights = np.concatenate([.5*(b-a)*w for a, b in zip(edges[:-1], edges[1:])])
    return -.5+1j*u, weights


def calls_from_transform(transform, phi, weights, maturity, sigma0, strikes):
    """Invert a transform using a Black control variate, without clipping prices."""
    strikes = np.asarray(strikes, dtype=float)
    transform = np.asarray(transform, dtype=complex)
    if not np.all(np.isfinite(transform)):
        raise FloatingPointError("Nonfinite transform cannot be integrated.")
    if strikes.ndim != 1 or not np.all(np.isfinite(strikes)) or np.any(strikes <= 0):
        raise ValueError("Strikes must be a finite positive vector.")
    if maturity == 0:
        return np.maximum(1.-strikes, 0.)
    sd = sigma0*np.sqrt(maturity)
    d1 = -np.log(strikes)/sd+.5*sd
    black = ndtr(d1)-strikes*ndtr(d1-sd)
    correction = transform-np.exp(.5*sd*sd*phi*(phi+1))
    integral = np.real(np.exp(np.log(strikes)[:, None]*phi)*correction) @ (
        weights/(phi.imag**2+.25)/np.pi)
    return black-strikes*integral
