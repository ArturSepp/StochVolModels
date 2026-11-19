"""Research-only Fourier/backward-PDE reference for a scalar volatility diffusion.

d sigma = (a+c*sigma-d*sigma**2) dt + beta*sigma dW0 + epsilon*sigma dW1.
d log F = -sigma**2/2 dt + sigma dW0. This is H=1/2, not a rough pricer.
The full polynomial drift is retained; no affine expansion is used.
"""

from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.special import roots_legendre, ndtr
from scipy.integrate import simpson


@dataclass(frozen=True)
class Model:
    sigma0: float
    a: float
    c: float
    d: float
    beta: float
    epsilon: float

    @property
    def nu2(self):
        return self.beta**2 + self.epsilon**2


def from_row(p):
    return Model(
        float(p.sigma0),
        float(p.kappa1 * p.theta),
        float(p.kappa2 * p.theta - p.kappa1),
        float(p.kappa2),
        float(p.beta),
        float(p.epsilon),
    )


def grid(model, subdivisions, halfwidth):
    n = int(round(2 * halfwidth * subdivisions))
    assert n % 2 == 0
    y = np.log(model.sigma0) + np.arange(-n // 2, n // 2 + 1) / subdivisions
    return y, np.exp(y), n // 2 - 1


def fitted_rates(drift, diffusion, dy):
    """Positive real drift rates; second-order consistent as dy -> 0."""
    if diffusion == 0:
        return np.maximum(-drift, 0) / dy, np.maximum(drift, 0) / dy
    peclet = drift * dy / diffusion

    def bernoulli(x):
        out = np.empty_like(x)
        small = np.abs(x) < 1e-5
        out[small] = 1 - x[small] / 2 + x[small] ** 2 / 12 - x[small] ** 4 / 720
        mid = (~small) & (x < 700) & (x > -700)
        out[mid] = x[mid] / np.expm1(x[mid])
        out[x >= 700] = 0.0
        out[x <= -700] = -x[x <= -700]
        return out

    return diffusion / dy**2 * bernoulli(peclet), diffusion / dy**2 * bernoulli(-peclet)


def operator(model, z, psi, subdivisions, halfwidth, formulation="direct", beta_sign=1):
    y, sigma, center = grid(model, subdivisions, halfwidth)
    h = 1 / subdivisions
    nu2 = model.nu2
    v = sigma[1:-1]
    drift = model.a / v + model.c - model.d * v - 0.5 * nu2
    q = 0.5 * z * (z - 1) - psi
    if formulation == "direct":
        lower, upper = fitted_rates(drift + z.real * beta_sign * model.beta * v, 0.5 * nu2, h)
        lower = lower.astype(complex) - 1j * z.imag * beta_sign * model.beta * v / (2 * h)
        upper = upper.astype(complex) + 1j * z.imag * beta_sign * model.beta * v / (2 * h)
        potential = q * v * v
        initial = np.ones(len(v), complex)
        factors = np.ones(3, complex)
    else:
        assert formulation == "gauge" and nu2 > 0
        lower, upper = fitted_rates(drift, 0.5 * nu2, h)
        lower = lower.astype(complex)
        upper = upper.astype(complex)
        loading = z * model.beta / nu2
        b = model.a + model.c * v - model.d * v * v
        potential = (0.5 * (z * z * (1 - model.beta**2 / nu2) - z) - psi) * v * v - loading * b
        initial = np.exp(loading * v)
        factors = np.exp(loading * np.array([sigma[0], sigma[-1], model.sigma0]))
    diagonal = -(lower + upper) + potential
    return lower, diagonal, upper, initial, factors, q, sigma, center


@njit(cache=False)
def factorize(lower, diag, upper, dt, theta):
    n = len(diag)
    inv = np.empty(n, np.complex128)
    cp = np.empty(n, np.complex128)
    inv[0] = 1 / (1 - theta * dt * diag[0])
    cp[0] = -theta * dt * upper[0] * inv[0]
    for j in range(1, n):
        inv[j] = 1 / (1 - theta * dt * diag[j] + theta * dt * lower[j] * cp[j - 1])
        cp[j] = -theta * dt * upper[j] * inv[j]
    return inv, cp


@njit(cache=False)
def advance(values, lower, diag, upper, inv, cp, dt, theta, old_boundary, new_boundary):
    n = len(values)
    rhs = np.empty(n, np.complex128)
    for j in range(n):
        rhs[j] = (1 + (1 - theta) * dt * diag[j]) * values[j]
        if j > 0:
            rhs[j] += (1 - theta) * dt * lower[j] * values[j - 1]
        if j < n - 1:
            rhs[j] += (1 - theta) * dt * upper[j] * values[j + 1]
    rhs[0] += dt * lower[0] * ((1 - theta) * old_boundary[0] + theta * new_boundary[0])
    rhs[-1] += dt * upper[-1] * ((1 - theta) * old_boundary[1] + theta * new_boundary[1])
    rhs[0] *= inv[0]
    for j in range(1, n):
        rhs[j] = (rhs[j] + theta * dt * lower[j] * rhs[j - 1]) * inv[j]
    for j in range(n - 2, -1, -1):
        rhs[j] -= cp[j] * rhs[j + 1]
    return rhs


@njit(cache=False)
def evolve(
    lower, diag, upper, initial, factors, q, edge_sigma, center, days, per_day, killed=False
):
    dt = 1 / (365 * per_day)
    value = initial.copy()
    inv, cp = factorize(lower, diag, upper, dt, 0.5)
    inv_be, cp_be = factorize(lower, diag, upper, dt / 2, 1.0)
    outputs = np.empty(len(days), np.complex128)
    target = 0
    for step in range(int(days[-1]) * per_day):
        if step == 0:
            for half in range(2):
                old = np.zeros(2, np.complex128)
                new = np.zeros(2, np.complex128)
                if not killed:
                    for j in range(2):
                        old[j] = factors[j] * np.exp(q * edge_sigma[j] ** 2 * half * dt / 2)
                        new[j] = factors[j] * np.exp(q * edge_sigma[j] ** 2 * (half + 1) * dt / 2)
                value = advance(value, lower, diag, upper, inv_be, cp_be, dt / 2, 1.0, old, new)
        else:
            old = np.zeros(2, np.complex128)
            new = np.zeros(2, np.complex128)
            if not killed:
                for j in range(2):
                    old[j] = factors[j] * np.exp(q * edge_sigma[j] ** 2 * step * dt)
                    new[j] = factors[j] * np.exp(q * edge_sigma[j] ** 2 * (step + 1) * dt)
            value = advance(value, lower, diag, upper, inv, cp, dt, 0.5, old, new)
        if step + 1 == int(days[target]) * per_day:
            outputs[target] = value[center] / factors[2]
            target += 1
    return outputs


def transforms(
    model,
    z,
    days,
    subdivisions=160,
    halfwidth=6,
    per_day=8,
    formulation="direct",
    psi=0j,
    killed=False,
    progress=None,
):
    z = np.atleast_1d(z).astype(complex)
    days = np.asarray(days, dtype=np.int64)
    assert (np.diff(days) > 0).all() and days[0] > 0
    result = np.empty((len(days), len(z)), complex)
    for i, zz in enumerate(z):
        lower, diag, upper, initial, factors, q, sigma, center = operator(
            model, zz, psi, subdivisions, halfwidth, formulation
        )
        result[:, i] = evolve(
            lower, diag, upper, initial, factors, q, sigma[[0, -1]], center, days, per_day, killed
        )
        if progress is not None and (i % 32 == 31 or i == len(z) - 1):
            progress(i + 1, len(z))
    assert np.isfinite(result).all(), "Nonfinite transform"
    return result


def contour(degree, cutoff, method="gauss"):
    if method == "gauss":
        x, w = roots_legendre(degree)
        return (x + 1) * cutoff / 2, w * cutoff / 2
    u = np.linspace(0, cutoff, degree + 1)
    return u, None


def calls(u, weights, m, k):
    k = np.atleast_1d(k)
    integrand = np.real(np.exp(1j * np.outer(np.log(k), u)) * m) / (u * u + 0.25)
    integral = integrand @ weights if weights is not None else simpson(integrand, x=u, axis=1)
    return 1 - np.sqrt(k) / np.pi * integral


def black(k, variance, forward=1.0):
    k = np.asarray(k)
    root = np.sqrt(variance)
    d1 = (np.log(forward / k) + variance / 2) / root
    return forward * ndtr(d1) - k * ndtr(d1 - root)
