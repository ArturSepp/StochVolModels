"""Full-state weak splitting for finite exponential rough-logSV lifts.

The factor state is never reset to current volatility. Linear memory and
Stratonovich correction are propagated exactly in a symmetric eigenbasis.
Research implementation; no calibration qualification is implied.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.special import log_ndtr, roots_hermitenorm
from numba import njit


@dataclass(frozen=True)
class Model:
    sigma0: float
    a: float
    c: float
    d: float
    beta: float
    epsilon: float

    @property
    def nu(self):
        return np.hypot(self.beta, self.epsilon)


def model(row, linear=False):
    return Model(
        row.sigma0,
        row.kappa1 * row.theta,
        -row.kappa1 if linear else row.kappa2 * row.theta - row.kappa1,
        0.0 if linear else row.kappa2,
        row.beta,
        row.epsilon,
    )


def prepare(p, nodes, weights, dt, tilt=0.5, correction=True):
    """Build exact linear flows and a bounded-error Gram factor for integral sigma²."""
    x = np.asarray(nodes)
    w = np.asarray(weights)
    root = np.sqrt(w)
    k0 = w.sum()
    assert np.all(x >= 0) and np.all(w > 0) and p.nu > 0
    ell = p.c - (0.5 * k0 * p.nu**2 if correction else 0.0)
    matrix = np.diag(x) - ell * np.outer(root, root)
    lam, Q = eigh(matrix)
    r = Q.T @ root
    assert lam.min() > 0 and abs(r @ r / k0 - 1) < 1e-12
    qstar = (p.a + ell * p.sigma0) * r / lam
    sigma_bar = p.sigma0 + r @ qstar
    h = dt / 2
    decay = np.exp(-lam * h)
    phi = -np.expm1(-lam * h) / lam
    rates = np.r_[0.0, lam]
    pair = rates[:, None] + rates[None, :]
    gram = np.divide(-np.expm1(-pair * h), pair, out=np.full_like(pair, h), where=pair > 0)
    vals, vecs = eigh(gram)
    keep = vals > max(vals.max() * 1e-16, 0.0)
    gram_factor = np.sqrt(vals[keep])[:, None] * vecs[:, keep].T
    gram_residual = float(np.linalg.norm(gram - gram_factor.T @ gram_factor, ord=2))
    # The exact quadratic reward in augmented [1,q] coordinates, used by moment checks.
    affine = np.zeros((len(r) + 1, len(r) + 1))
    affine[0, 0] = sigma_bar
    affine[1:, 0] = -r * qstar
    affine[1:, 1:] = np.diag(r)
    reward = affine.T @ gram @ affine
    return dict(
        s0=float(p.sigma0),
        nu=float(p.nu),
        rho=float(p.beta / p.nu),
        k0=float(k0),
        d=float(p.d),
        dt=float(dt),
        r=r,
        lam=lam,
        qstar=qstar,
        sigma_bar=float(sigma_bar),
        decay=decay,
        phi=phi,
        gram_factor=gram_factor,
        gram_residual=gram_residual,
        reward=reward,
        Q=Q,
        nodes=x,
        weights=w,
        linear_matrix=matrix,
        ell=ell,
        tilt=tilt,
    )


@njit(cache=False)
def linear_flow(q, r, qstar, sbar, decay, phi, factor, residual, h, k0, nu):
    features = np.empty(len(q) + 1)
    features[0] = sbar
    first = sbar * h
    for i in range(len(q)):
        features[i + 1] = r[i] * (q[i] - qstar[i])
        first += features[i + 1] * phi[i]
    integral = 0.0
    for i in range(factor.shape[0]):
        v = 0.0
        for j in range(len(features)):
            v += factor[i, j] * features[j]
        integral += v * v
    norm = 0.0
    for v in features:
        norm += v * v
    for i in range(len(q)):
        q[i] = qstar[i] + decay[i] * (q[i] - qstar[i])
    return integral, -0.5 * k0 * nu * first, residual * norm


@njit(cache=False)
def simulate(
    normal, stops, s0, nu, rho, tilt, k0, d, dt, r, qstar, sbar, decay, phi, factor, residual
):
    n, paths = normal.shape
    nf = len(r)
    output = np.empty((4, len(stops), paths))
    min_sigma = np.inf
    for path in range(paths):
        q = np.zeros(nf)
        integral = 0.0
        jump = 0.0
        allowance = 0.0
        target = 0
        for step in range(n):
            held = s0 + np.dot(r, q)
            if held <= 0 or not np.isfinite(held):
                raise ValueError("Invalid adapted price state")
            integral += held * held * dt
            jump += held * np.sqrt(dt) * normal[step, path]
            shifted = normal[step, path] + tilt * rho * held * np.sqrt(dt)
            for f in range(nf):
                q[f] = qstar[f] + decay[f] * (q[f] - qstar[f])
            sigma = s0 + np.dot(r, q)
            if sigma <= 0 or not np.isfinite(sigma):
                raise ValueError("Nonpositive linear-flow state")
            next_sigma = sigma / (1 + k0 * d * dt * sigma / 2)
            for f in range(nf):
                q[f] += r[f] * (next_sigma - sigma) / k0
            sigma = next_sigma
            # Combine noise and its following quadratic flow before updating any factor.
            log_noise = np.log(sigma) + k0 * nu * np.sqrt(dt) * shifted
            coefficient = k0 * d * dt / 2
            if coefficient > 0:
                log_product = np.log(coefficient) + log_noise
                if log_product > 0:
                    next_sigma = np.exp(-np.log(coefficient) - np.log1p(np.exp(-log_product)))
                else:
                    next_sigma = np.exp(log_noise - np.log1p(np.exp(log_product)))
            else:
                next_sigma = np.exp(log_noise)
            for f in range(nf):
                q[f] += r[f] * (next_sigma - sigma) / k0
            for f in range(nf):
                q[f] = qstar[f] + decay[f] * (q[f] - qstar[f])
            sigma = s0 + np.dot(r, q)
            min_sigma = min(min_sigma, sigma, next_sigma)
            if sigma <= 0 or not np.isfinite(sigma + integral + jump):
                raise ValueError("Invalid output state")
            if step + 1 == stops[target]:
                output[0, target, path] = integral
                output[1, target, path] = jump
                output[2, target, path] = sigma
                output[3, target, path] = allowance
                target += 1
    return output, min_sigma


def paths(prep, normal, days, per_day):
    z = np.ascontiguousarray(normal, dtype=float)
    return simulate(
        z,
        np.asarray(days, dtype=np.int64) * per_day,
        prep["s0"],
        prep["nu"],
        prep["rho"],
        prep["tilt"],
        prep["k0"],
        prep["d"],
        prep["dt"],
        prep["r"],
        prep["qstar"],
        prep["sigma_bar"],
        prep["decay"],
        prep["phi"],
        prep["gram_factor"],
        prep["gram_residual"],
    )


def half_payoffs(integral, jump, rho, k, is_call):
    """Bounded complement under tilt 1/2, without empirical normalization.

    M(.5-iu)=E_half exp[-I/8-(1-rho²)u² I/2-iu rho J].
    The scalar Gaussian integral is analytic; no Fourier truncation is needed.
    """
    v = (1 - rho * rho) * integral
    root = np.sqrt(v)
    z = rho * jump - np.log(k)
    first = -0.5 * z + v / 8 + log_ndtr((z - v / 2) / root)
    second = 0.5 * z + v / 8 + log_ndtr((-z - v / 2) / root)
    complement = np.sqrt(k) * np.exp(-integral / 8 + np.logaddexp(first, second))
    assert np.isfinite(complement).all() and np.all(complement <= np.sqrt(k) + 1e-12)
    return np.where(is_call, 1.0, k) - complement


@njit(cache=False)
def covariance_a(C, D, f, reward):
    n = len(D)
    out = np.empty_like(C)
    value = 0.0
    for i in range(n):
        for j in range(n):
            value += reward[i, j] * C[i, j]
            out[i, j] = (
                D[i] * D[j] * C[i, j] + D[i] * C[i, 0] * f[j] + f[i] * C[0, j] * D[j] + f[i] * f[j]
            )
    return out, value


@njit(cache=False)
def linear_moments(stops, s0, nu, k0, dt, r, qstar, decay, reward):
    """Exact moments of this discrete split, no stochastic estimator."""
    n = len(r) + 1
    C = np.zeros((n, n))
    C[0, 0] = 1
    D = np.empty(n)
    f = np.empty(n)
    s = np.empty(n)
    v = np.empty(n)
    D[0] = 1.0
    f[0] = 0.0
    s[0] = s0
    v[0] = 0.0
    for i in range(1, n):
        D[i] = decay[i - 1]
        f[i] = (1 - decay[i - 1]) * qstar[i - 1]
        s[i] = r[i - 1]
        v[i] = r[i - 1] / k0
    m1 = np.expm1(0.5 * (k0 * nu) ** 2 * dt)
    m2 = np.expm1(2 * (k0 * nu) ** 2 * dt) - 2 * m1
    result = np.empty((3, len(stops)))
    integ = 0.0
    target = 0
    for step in range(stops[-1]):
        integ += dt * np.dot(s, C @ s)
        C, ii = covariance_a(C, D, f, reward)
        cs = C @ s
        scalar = np.dot(s, cs)
        for i in range(n):
            for j in range(n):
                C[i, j] += m1 * (v[i] * cs[j] + cs[i] * v[j]) + m2 * scalar * v[i] * v[j]
        C, ii = covariance_a(C, D, f, reward)
        if step + 1 == stops[target]:
            result[0, target] = np.dot(s, C[0])
            result[1, target] = np.dot(s, C @ s)
            result[2, target] = integ
            target += 1
    return result


def exact_discrete(prep, days, per_day):
    assert prep["tilt"] == 0 and prep["d"] == 0
    return linear_moments(
        np.asarray(days, dtype=np.int64) * per_day,
        prep["s0"],
        prep["nu"],
        prep["k0"],
        prep["dt"],
        prep["r"],
        prep["qstar"],
        prep["decay"],
        prep["reward"],
    )


def scalar_maps(p, subdivisions, halfwidth, per_day, degree=9):
    """Same split in the constant-kernel limit, integrated by Gaussian quadrature."""
    dt = 1 / (365 * per_day)
    h = dt / 2
    nu = p.nu
    lam = -(p.c - 0.5 * nu * nu)
    eq = p.a / lam
    decay = np.exp(-lam * h)
    phi = -np.expm1(-lam * h) / lam
    phi2 = -np.expm1(-2 * lam * h) / (2 * lam)
    y = np.arange(-int(halfwidth * subdivisions), int(halfwidth * subdivisions) + 1) / subdivisions
    sigma = p.sigma0 * np.exp(y)
    z, w = roots_hermitenorm(degree)
    w = w / np.sqrt(2 * np.pi)

    def A(s):
        R = s - eq
        return (
            eq + R * decay,
            eq * eq * h + 2 * eq * R * phi + R * R * phi2,
            -0.5 * nu * (eq * h + R * phi),
        )

    s1, I1, J1 = A(sigma[:, None])
    s2 = s1 / (1 + p.d * h * s1)
    shifted = z[None, :] + 0.5 * (p.beta / nu) * sigma[:, None] * np.sqrt(dt)
    s3 = s2 * np.exp(nu * np.sqrt(dt) * shifted)
    s4 = s3 / (1 + p.d * h * s3)
    s5, I2, J2 = A(s4)
    I = sigma[:, None] ** 2 * dt + np.zeros_like(s5)
    J = sigma[:, None] * np.sqrt(dt) * z[None, :]
    coordinate = (np.log(s5 / p.sigma0) + halfwidth) * subdivisions
    base = np.floor(coordinate).astype(np.int64) - 2
    indices = base[:, :, None] + np.arange(6)
    interpolation = np.ones(indices.shape)
    for j in range(6):
        for k in range(6):
            if j != k:
                interpolation[:, :, j] *= (coordinate - base - k) / (j - k)
    return sigma, w, indices, interpolation, I, J, s5


@njit(cache=False)
def scalar_frequency(u, rho, sigma, w, indices, interp, I, J, end_sigma, days, per_day):
    """Backward six-point interpolation, with explicit frozen-volatility exterior data."""
    n = len(sigma)
    v = np.ones(n, np.complex128)
    out = np.empty(len(days), np.complex128)
    weight = np.exp(-(0.125 + 0.5 * (1 - rho * rho) * u * u) * I - 1j * u * rho * J)
    center = n // 2
    target = 0
    dt = 1 / (365 * per_day)
    for step in range(days[-1] * per_day):
        new = np.zeros(n, np.complex128)
        for i in range(n):
            for k in range(len(w)):
                if indices[i, k, 0] < 0 or indices[i, k, -1] >= n:
                    value = np.exp(-(0.125 + 0.5 * u * u) * end_sigma[i, k] ** 2 * step * dt) + 0j
                else:
                    value = 0j
                    for j in range(6):
                        value += interp[i, k, j] * v[indices[i, k, j]]
                new[i] += w[k] * weight[i, k] * value
        v = new
        if step + 1 == days[target] * per_day:
            out[target] = v[center]
            target += 1
    return out


def scalar_transform(p, u, days, subdivisions=80, halfwidth=6, per_day=8, degree=9, progress=None):
    maps = scalar_maps(p, subdivisions, halfwidth, per_day, degree)
    out = np.empty((len(days), len(u)), complex)
    for j, frequency in enumerate(u):
        out[:, j] = scalar_frequency(
            frequency, p.beta / p.nu, *maps, np.asarray(days, dtype=np.int64), per_day
        )
        if progress is not None and (j % 16 == 15 or j == len(u) - 1):
            progress()
    assert np.isfinite(out).all()
    return out
