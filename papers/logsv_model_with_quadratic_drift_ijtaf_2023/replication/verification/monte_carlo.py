"""Paper validation through the existing constant-kernel split-step simulator.

Scrambled Sobol Brownian bridges are used only for deterministic replay of the
reference experiment. Standard errors are estimated across independent scrambles,
not with the kernel's IID-path formula.
"""
import warnings

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from stochvolmodels.pricers.logsv_pricer import rough_logsv_mc_chain_pricer_split_step
from .logvol_pde import _validate


def bridge_normals(power=15, steps=512, seed=291026):
    """Return unit normal increments of a scrambled Sobol Brownian bridge."""
    if steps < 1 or steps & (steps-1):
        raise ValueError("Bridge step count must be a positive power of two.")
    uniforms = qmc.Sobol(d=steps, scramble=True, bits=64, seed=seed).random_base2(power)
    z = ndtri(np.clip(uniforms, 2.**-53, 1.-2.**-53)).T
    w = np.zeros((steps+1, z.shape[1]))
    w[-1] = z[0]
    queue = [(0, steps)]
    col = 1
    while queue:
        left, right = queue.pop(0)
        mid = (left+right)//2
        if mid == left:
            continue
        w[mid] = ((right-mid)*w[left]+(mid-left)*w[right])/(right-left)
        w[mid] += np.sqrt((mid-left)*(right-mid)/(steps*(right-left)))*z[col]
        col += 1
        queue.extend([(left, mid), (mid, right)])
    return np.ascontiguousarray(np.diff(w, axis=0)*np.sqrt(steps))


def call_prices(params, maturity, strikes, normals):
    """Use the installed conditional split-step kernel, with K(t)=1 exactly.

    Reject floored paths and preserve other simulator warnings, including the
    empirical time-step resolution diagnostic.
    """
    _validate(params, maturity)
    strikes = np.asarray(strikes, float)
    if maturity == 0:
        return np.maximum(1-strikes, 0.)
    steps = normals.shape[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        prices, _ = rough_logsv_mc_chain_pricer_split_step(
            ttms=np.array([maturity]), forwards=np.ones(1), discfactors=np.ones(1),
            strikes_ttms=[strikes], optiontypes_ttms=[np.full(len(strikes), 'C')],
            Z0=normals, Z1=np.zeros_like(normals), sigma0=params.sigma0, theta=params.theta,
            kappa1=params.kappa1, kappa2=params.kappa2, beta=params.beta,
            orthog_vol=params.volvol, weights=np.ones(1), nodes=np.zeros(1),
            timegrids=[np.linspace(0, maturity, steps+1)])
    for warning in caught:
        if 'floored' not in str(warning.message).lower():
            warnings.warn(warning.message, warning.category, stacklevel=2)
    if any('floored' in str(w.message).lower() for w in caught):
        raise FloatingPointError("Split-step kernel reported floored paths.")
    return np.asarray(prices[0])
