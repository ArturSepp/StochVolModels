"""Coupled log-Euler observations for the continuous, unit-loading two-year fixture."""

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os

from numba import njit
import numpy as np


@njit(nogil=True)
def _step(x, log_s, sigma, dw, dz, dt, params):
    _, theta, k1, k2, beta, volvol = params
    x += -.5 * sigma**2 * dt + sigma * dw
    log_s += ((k1 * theta / sigma - k1 + k2 * (theta - sigma)
               - .5 * (beta**2 + volvol**2)) * dt + beta * dw + volvol * dz)
    return x, log_s, np.exp(log_s)


@njit(nogil=True)
def _advance(state, brownian, dt, params):
    """Advance [coarse/fine, antithetic sign, x/log-sigma/sigma, pair] in place."""
    for t in range(brownian.shape[0]):
        for side in range(2):
            sign = 1. if side == 0 else -1.
            for i in range(brownian.shape[-1]):
                w1, z1 = sign*brownian[t, 0, 0, i], sign*brownian[t, 0, 1, i]
                w2, z2 = sign*brownian[t, 1, 0, i], sign*brownian[t, 1, 1, i]
                x, log_s, s = _step(state[1, side, 0, i], state[1, side, 1, i],
                                    state[1, side, 2, i], w1, z1, dt, params)
                x, log_s, s = _step(x, log_s, s, w2, z2, dt, params)
                state[1, side, 0, i], state[1, side, 1, i], state[1, side, 2, i] = x, log_s, s
                x, log_s, s = _step(state[0, side, 0, i], state[0, side, 1, i],
                                    state[0, side, 2, i], w1+w2, z1+z2, 2*dt, params)
                state[0, side, 0, i], state[0, side, 1, i], state[0, side, 2, i] = x, log_s, s


def sample(model, seed, pairs, times, rate):
    if (model.jump is not None or model.leverage_surface is not None or model.time_knots.size
            or model.return_loading(times[-1], 0.) != 1.):
        raise ValueError("coupled sampler requires continuous dynamics and unit return loading")
    p = model.dynamics.params
    params = (p.sigma0, p.theta, p.kappa1, p.kappa2, p.beta, p.volvol)
    rng = np.random.Generator(np.random.PCG64(seed))
    dt = 1./(2*rate)
    counts = np.diff([0., *times])*rate
    if np.any(counts <= 0) or not np.array_equal(counts, counts.astype(int)):
        raise ValueError("each observation must end on an exact coarse step")
    state = np.zeros((2, 2, 3, pairs))
    state[:, :, 1] = np.log(p.sigma0)
    state[:, :, 2] = p.sigma0
    paths = np.empty((2, len(times), 2, 2, pairs))
    for j, count in enumerate(counts.astype(int)):
        for start in range(0, count, 16):
            shocks = rng.standard_normal((min(16, count-start), 2, 2, pairs))
            shocks *= np.sqrt(dt)
            _advance(state, shocks, dt, params)
        paths[:, j, 0] = state[:, :, 0]
        paths[:, j, 1] = state[:, :, 2]
    if not np.isfinite(paths).all() or np.any(paths[:, :, 1] <= 0):
        raise FloatingPointError("nonfinite states or nonpositive volatility in coupled replay")
    return paths


def _statistics(paths, strikes):
    """Keep covariance across time resolutions, treating each antithetic pair as one draw."""
    xs, variances = paths[:, :, 0], paths[:, :, 1]**2
    bins = np.searchsorted(np.log(strikes), xs, side="right")
    means, scatters = [], []

    def record(columns):
        values = np.stack(columns, axis=1)
        mean = values.mean(axis=0)
        centered = values - mean
        means.append(mean)
        scatters.append(centered.T @ centered)

    for time in range(xs.shape[1]):
        for index in range(len(strikes)+1):
            hit = bins[:, time] == index
            den = hit.mean(axis=1)
            num = (variances[:, time]*hit).mean(axis=1)
            record((den[0], num[0], den[1], num[1]))
    forward = np.exp(xs[:, -1])
    for strike in strikes:
        raw = np.maximum(forward-strike, 0.)
        control = raw-(forward-1.)
        record([v.mean(axis=0) for v in (raw[0], control[0], raw[1], control[1])])
    return np.array(means), np.array(scatters)


def coupled_statistics(model, times, strikes, sampling, workers):
    coarse, fine = sampling["rates"]
    batches, size, chunk = (sampling[k] for k in ("batches", "pairs_per_batch", "chunk_pairs"))
    if (fine != 2*coarse or coarse <= 0 or min(batches, size, chunk, workers) <= 0
            or size % chunk or batches*size < 2):
        raise ValueError("invalid coupled sampling design")

    def run(index):
        seed = np.random.SeedSequence(sampling["master_seed"], spawn_key=index)
        return _statistics(sample(model, seed, chunk, times, coarse), strikes)

    # The merge order is fixed even when chunks are evaluated concurrently.
    indexes = product(range(batches), range(size//chunk))
    with ThreadPoolExecutor(max_workers=min(workers, os.cpu_count() or 1)) as pool:
        records = list(pool.map(run, indexes))
    means, scatters = map(np.stack, zip(*records, strict=True))
    mean = means.mean(axis=0)
    between = means-mean
    scatter = scatters.sum(axis=0) + chunk*np.einsum("cij,cik->ijk", between, between)
    pairs = batches*size
    return mean, scatter/(pairs*(pairs-1))


def ratio_statistics(mean, covariance):
    """Delta-method uncertainty of the two ratios and their paired difference."""
    probability = mean[::2]
    if np.any(probability <= 0):
        raise ValueError("empty Monte Carlo bin")
    estimates = mean[1::2]/probability
    gradients = np.array([[-estimates[0], 1., 0., 0.], [0., 0., -estimates[1], 1.]])
    gradients /= probability[:, None]
    contrast = gradients[1]-gradients[0]
    errors = np.sqrt(np.diag(gradients@covariance@gradients.T))
    return estimates, errors, np.sqrt(contrast@covariance@contrast)
