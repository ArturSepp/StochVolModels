"""Finite-volume and path checks for the fitted stationary Gamma boundary.

Use the October experiment's detailed-balance discretisation, Brownian coupling
and positive splitting step, with stationary starts drawn from the exact Gamma law.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp
from scipy.stats import gamma

from ..empirical_20261029.synthetic.synthetic_observation_acf import (
    CHECK_LAGS, DPY, SEED, sample_acfs, split_step,
    stationary_distribution as interior_distribution,
)


def stationary_distribution(p):
    if p["A"] > 0 and p["C"] > 0:
        return interior_distribution(p)
    shape = 2*p["B"]/p["v"]**2 - 1
    if p["A"] != 0 or p["C"] <= 0 or shape <= 0:
        raise ValueError("The synthetic study requires an interior GIG or a proper Gamma boundary.")
    return gamma(a=shape, scale=p["v"]**2/(2*p["C"]))


def generator(p, n=1200, tail=1e-9):
    """Conservative detailed-balance finite-volume generator in log volatility.

    All eigenmodes of the symmetric tridiagonal matrix are used: this is not a
    truncated transition-density likelihood. Endpoints reflect. Domain/grid
    sensitivity is assessed separately.
    """
    dist = stationary_distribution(p)
    y = np.linspace(*np.log(dist.ppf([tail, 1-tail])), n)
    dy = y[1]-y[0]
    def log_density(z):
        return (2*p['B']/p['v']**2-1)*z-2*p['A']/p['v']**2*np.exp(-z)-2*p['C']/p['v']**2*np.exp(z)
    lp = log_density(y)
    lm = log_density(0.5*(y[:-1]+y[1:]))
    pi = np.exp(lp-logsumexp(lp))
    right = 0.5*p['v']**2/dy**2*np.exp(lm-lp[:-1])
    left = 0.5*p['v']**2/dy**2*np.exp(lm-lp[1:])
    diag = np.r_[right, 0.0]+np.r_[0.0, left]
    off = -np.sqrt(right*left)
    lam, u = eigh_tridiagonal(diag, off)
    assert abs(lam[0]) < 1e-7, lam[0]
    lam[0] = 0.0
    root_pi = np.sqrt(pi)
    ground_alignment = abs(u[:, 0] @ root_pi)
    assert ground_alignment > 1-1e-8
    return dict(y=y, s=np.exp(y), pi=pi, lam=lam, u=u, root_pi=root_pi,
                n=n, tail=tail, ground_alignment=float(ground_alignment))


def simulate(p, years, n_paths, substeps, seed):
    rng = np.random.default_rng(seed)
    s = stationary_distribution(p).rvs(size=n_paths, random_state=rng)
    paths = np.empty((years*DPY, n_paths))
    dt = 1/(DPY*substeps)
    for j in range(paths.shape[0]):
        paths[j] = s
        for _ in range(substeps):
            s = split_step(s, dt, rng.standard_normal(n_paths), p)
    assert np.isfinite(paths).all() and np.all(paths > 0)
    return paths


def paired_step_check(p, years=10, n_paths=400):
    """Compare four and eight substeps with exactly paired Brownian increments."""
    rng = np.random.default_rng(SEED+3000)
    initial = stationary_distribution(p).rvs(size=n_paths,random_state=rng)
    coarse, fine = initial.copy(), initial.copy()
    pc, pf = np.empty((years*DPY,n_paths)), np.empty((years*DPY,n_paths))
    for day in range(years*DPY):
        pc[day], pf[day] = coarse, fine
        z = rng.standard_normal((8,n_paths))
        for j in range(8):
            fine = split_step(fine,1/(DPY*8),z[j],p)
        for j in range(4):
            coarse = split_step(coarse,1/(DPY*4),(z[2*j]+z[2*j+1])/np.sqrt(2),p)
    delta_acf = sample_acfs(pf)-sample_acfs(pc)
    dm = (pf-pc).mean(axis=0)
    dm2 = (pf**2-pc**2).mean(axis=0)
    return dict(seed=SEED+3000, years=years,n_paths=n_paths,coarse_substeps=4,fine_substeps=8,
                mean_difference=dm.mean(),mean_difference_se=dm.std(ddof=1)/np.sqrt(n_paths),
                second_moment_difference=dm2.mean(),second_moment_difference_se=dm2.std(ddof=1)/np.sqrt(n_paths),
                acf_lags=CHECK_LAGS,mean_acf_difference=delta_acf.mean(axis=1)[list(CHECK_LAGS)],
                acf_difference_se=delta_acf.std(axis=1,ddof=1)[list(CHECK_LAGS)]/np.sqrt(n_paths),
                max_absolute_mean_acf_difference=np.max(abs(delta_acf.mean(axis=1))))
