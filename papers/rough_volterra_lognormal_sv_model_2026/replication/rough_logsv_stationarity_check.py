"""
Second-moment stationarity criterion for the log-normal Volterra SV model with linear drift (kappa2 = 0).

Model (draft eq. (1)): sigma_t = sigma_0 + int_0^t K(t-s)[kappa1 (theta - sigma_s) ds + vartheta sigma_s dW_s],
K(t) = t^{alpha-1}/Gamma(alpha), alpha = H + 1/2.
Draft Proposition 3.2: v(t) = E[sigma_t^2] solves v = m^2 + vartheta^2 G^2 * v with
G(t) = t^{alpha-1} E_{alpha,alpha}(-kappa1 t^alpha), Laplace transform 1/(z^alpha + kappa1).

Claim checked here (renewal theory for the positive kernel h = vartheta^2 G^2):
    sup_t v(t) < infinity  <=>  ||h||_{L1} = vartheta^2 ||G||_{L2(R+)}^2 < 1,
    and then v(t) -> theta^2 / (1 - vartheta^2 ||G||_2^2).
||G||_2^2 is computed by Parseval,
    ||G||_2^2 = (1/pi) int_0^inf dw / (w^{2 alpha} + 2 kappa1 cos(pi alpha/2) w^alpha + kappa1^2)
             = kappa1^{-(2 alpha - 1)/alpha} C(alpha),  C(1) = 1/2.
Units: time in years, kappa1 in year^(-H-1/2), vartheta in year^(-H).
Entry points: g2_norm, solve_second_moment, and the cases of run_local_test (paper exhibits of
Section 3, Proposition on second-moment stationarity).
Not in this module: anything with kappa2 > 0.
"""
# packages
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import digamma, gamma, gammaln, logsumexp, rgamma
from functools import lru_cache
from enum import Enum
from typing import Tuple



def c_alpha(alpha: float) -> float:
    """Closed resolvent-norm constant; the endpoint value is its continuous limit."""
    if not 0.5 < alpha <= 1.0:
        raise ValueError("alpha must lie in (1/2, 1]")
    # sin(pi/alpha) = -sin(pi*(1-alpha)/alpha), avoiding cancellation at one.
    delta = 1.0 - alpha
    return 0.5 if delta == 0.0 else float(
        np.tan(np.pi * delta / 2.0) / (alpha * np.sin(np.pi * delta / alpha)))


def g2_norm(H: float, kappa1: float) -> float:
    """Integral of G squared from the closed expression in paper eq:gamma."""
    if kappa1 <= 0.0:
        raise ValueError("kappa1 must be positive")
    alpha = H + 0.5
    return c_alpha(alpha) * kappa1 ** (-2.0 * H / alpha)


def g2_norm_spectral(H: float, kappa1: float) -> float:
    """Independent Parseval quadrature used to validate the closed expression."""
    if not 0.0 < H <= 0.5 or kappa1 <= 0.0:
        raise ValueError("require H in (0, 1/2] and positive kappa1")
    alpha = H + 0.5
    value = quad(lambda u: 1.0 / (u ** (2 * alpha)
                 + 2 * np.cos(np.pi * alpha / 2) * u ** alpha + 1),
                 0, np.inf, epsabs=1e-11, epsrel=1e-11, limit=1000)[0]
    return value / np.pi * kappa1 ** (-2 * H / alpha)


def _ml_aa_mp(alpha: float, x: float) -> float:
    """Optional high-precision reference, requiring mpmath only when called."""
    import mpmath as mp
    with mp.workdps(int(40 + 0.6 * float(x) ** (1.0 / alpha))):
        a, z = mp.mpf(alpha), mp.mpf(x)
        total = mp.mpf(0)
        for k in range(20000):
            term = (-z) ** k / mp.gamma(a * k + a)
            total += term
            if k > 30 and abs(term) < mp.mpf('1e-28') * max(1, abs(total)):
                return float(total)
    raise RuntimeError("High-precision reference did not converge")


def _ml_negative_spectral(alpha: float, x: float, beta: float) -> float:
    """Positive spectral integral for E_(alpha,beta)(-x), beta=1 or alpha."""
    if x == 0:
        return float(rgamma(beta))
    if alpha == 1:
        return float(np.exp(-x))
    if not 0 < alpha < 1 or beta not in (1.0, alpha) or x < 0:
        raise ValueError("spectral formula requires 0<alpha<1, x>=0, beta=1 or alpha")
    sine, cosine = np.sin(np.pi * alpha), np.cos(np.pi * alpha)
    def integrand(u):
        ua = u ** alpha
        denominator = (ua + x * cosine) ** 2 + (x * sine) ** 2
        numerator = ua if beta == alpha else x * u ** (alpha - 1)
        return np.exp(-u) * numerator / denominator
    peak = x ** (1 / alpha)
    value = quad(integrand, 0, peak, epsabs=2e-12, epsrel=2e-12, limit=300)[0]
    value += quad(integrand, peak, np.inf, epsabs=2e-12, epsrel=2e-12, limit=300)[0]
    return float(sine * value / np.pi)


@lru_cache(maxsize=32)
def _ml_log_chebyshev(alpha: float, beta: float, upper: float):
    """Checked interpolation of the positive spectral representation on [3, upper]."""
    for degree in (32, 64, 128, 256):
        nodes = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)
        args = 3.0 + (upper - 3.0) * (nodes + 1) / 2
        values = np.array([_ml_negative_spectral(alpha, x, beta) for x in args])
        coeff = np.polynomial.chebyshev.chebfit(nodes, np.log(values), degree - 1)
        probes = np.linspace(-1, 1, 2 * degree + 1)
        exact = np.array([_ml_negative_spectral(alpha, 3 + (upper - 3) * (z + 1) / 2,
                                               beta) for z in probes])
        estimate = np.exp(np.polynomial.chebyshev.chebval(probes, coeff))
        if np.max(np.abs(estimate / exact - 1)) < 2e-10:
            return coeff
    raise ArithmeticError("Mittag-Leffler spectral interpolation failed its check")


def mittag_leffler_negative_vec(alpha: float, x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """E_(alpha,beta)(-x), without cancellation outside the small-argument series."""
    x = np.asarray(x, dtype=float)
    if not 0.5 < alpha <= 1 or beta not in (1.0, alpha) or np.any(x < 0):
        raise ValueError("require alpha in (1/2,1], beta=1 or alpha, and x>=0")
    if alpha == 1:
        return np.exp(-x)
    out = np.empty_like(x)
    small = x <= 3.0
    coeff = rgamma(alpha * np.arange(180) + beta)
    out[small] = np.polynomial.polynomial.polyval(-x[small], coeff)
    if np.any(~small):
        upper = max(4.0, float(np.ceil(np.max(x[~small]))))
        coeff = _ml_log_chebyshev(alpha, beta, upper)
        z = 2 * (x[~small] - 3) / (upper - 3) - 1
        out[~small] = np.exp(np.polynomial.chebyshev.chebval(z, coeff))
    if np.any(out <= 0):
        raise ArithmeticError("Non-positive value of a completely monotone function")
    return out


def mittag_leffler_aa_vec(alpha: float, x: np.ndarray) -> np.ndarray:
    """E_(alpha,alpha)(-x); retained entry point for moment-kernel evaluation."""
    return mittag_leffler_negative_vec(alpha, x, beta=alpha)


def log_mittag_leffler_positive(z: float, alpha: float, beta: float = 1.0) -> float:
    """Log E_(alpha,beta)(z), z>=0, by an adaptive log-concave series window.

    Both omitted tails are bounded by geometric series using successive term
    ratios. This avoids truncating before the peak (about k=89,300 in Table 2).
    """
    if z < 0 or alpha <= 0 or beta <= 0:
        raise ValueError("require z>=0, alpha>0 and beta>0")
    if z == 0:
        return float(-gammaln(beta))
    logz = np.log(z)
    if alpha == 1 and beta == 1:
        return float(z)
    rate = np.exp(logz / alpha)
    hi = max(32.0, 2 * rate / alpha + 2 * beta)
    derivative = lambda k: logz - alpha * digamma(alpha * k + beta)
    center = 0 if derivative(0) <= 0 else int(brentq(derivative, 0, hi))
    width = max(32, int(12 * np.sqrt(rate + 1) / alpha))
    def logterm(k):
        return np.asarray(k) * logz - gammaln(alpha * np.asarray(k) + beta)
    for _ in range(20):
        lo, hi = max(0, center - width), center + width
        if hi - lo > 2000000:
            raise ArithmeticError("Series window is too large; use an asymptotic with an error bound")
        total = float(logsumexp(logterm(np.arange(lo, hi + 1))))
        upper_ratio = float(np.exp(logterm(hi + 2) - logterm(hi + 1)))
        upper = float(logterm(hi + 1) - np.log1p(-upper_ratio))
        lower = -np.inf
        if lo > 0:
            ratio = 0.0 if lo == 1 else float(np.exp(logterm(lo - 2) - logterm(lo - 1)))
            lower = float(logterm(lo - 1) - np.log1p(-ratio))
        if np.logaddexp(lower, upper) < total + np.log(1e-13):
            return total
        width *= 2
    raise ArithmeticError("Positive Mittag-Leffler series did not converge")


def cumulative_g2(H: float, kappa1: float, t_grid: np.ndarray, n_sub: int = 24) -> np.ndarray:
    """Integral of G squared, using Gauss-Legendre cells after y=t^(2H).

    Removing the singular factor before quadrature is essential in the first
    cell. This is numerical quadrature, not an assertion of exact cell masses.
    """
    if not 0 < H <= 0.5 or kappa1 < 0 or n_sub < 2:
        raise ValueError("invalid H, kappa1 or quadrature order")
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid[0] != 0 or np.any(np.diff(t_grid) <= 0):
        raise ValueError("grid must start at zero and increase strictly")
    alpha, power = H + 0.5, 2 * H
    y_grid = t_grid ** power
    nodes, weights = np.polynomial.legendre.leggauss(n_sub)
    left, right = y_grid[:-1, None], y_grid[1:, None]
    y = (right + left) / 2 + (right - left) * nodes / 2
    ml = mittag_leffler_aa_vec(alpha, kappa1 * y ** (alpha / power))
    cell = (right[:, 0] - left[:, 0]) / 2 * ((ml * ml) @ weights) / power
    return np.r_[0.0, np.cumsum(cell)]


def solve_second_moment(H: float, kappa1: float, vartheta: float, theta: float, sigma0: float,
                        T: float = 1.0, n_steps: int = 2000,
                        log_domain: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Product integration of eq:cond_second; optionally return log(v) to avoid overflow.

    Cell masses use 24-point singularity-removed quadrature. Adjacent endpoint
    averages approximate v within each cell. Grid refinement is still required.
    """
    if T <= 0 or n_steps < 1 or min(theta, sigma0) <= 0 or vartheta < 0:
        raise ValueError("require positive T, steps, theta, sigma0 and nonnegative vartheta")
    t = np.linspace(0, T, n_steps + 1)
    cells = vartheta ** 2 * np.diff(cumulative_g2(H, kappa1, t))
    denominator = 1 - cells[0] / 2
    if denominator <= 0:
        raise ValueError("grid too coarse: implicit first-cell denominator is nonpositive")
    mean = (np.full_like(t, theta) if sigma0 == theta else
            theta + (sigma0 - theta) * mittag_leffler_negative_vec(
                H + 0.5, kappa1 * t ** (H + 0.5)))
    if log_domain:
        logs = np.empty(n_steps + 1)
        logs[0] = 2 * np.log(sigma0)
        logcells = np.full_like(cells, -np.inf)
        np.log(cells, out=logcells, where=cells > 0)
        for n in range(1, n_steps + 1):
            known1 = logsumexp(logcells[n - 1::-1] + logs[:n])
            known2 = logsumexp(logcells[n - 1:0:-1] + logs[1:n])
            numerator = np.logaddexp(2 * np.log(mean[n]),
                                     np.logaddexp(known1, known2) - np.log(2))
            logs[n] = numerator - np.log(denominator)
        return t, logs
    values = np.empty(n_steps + 1)
    values[0] = sigma0 ** 2
    with np.errstate(over='ignore', invalid='ignore'):
        for n in range(1, n_steps + 1):
            known = (np.dot(cells[n - 1::-1], values[:n])
                     + np.dot(cells[n - 1:0:-1], values[1:n])) / 2
            values[n] = (mean[n] ** 2 + known) / denominator
    if not np.all(np.isfinite(values)):
        raise OverflowError("second moment exceeds floating-point range; set log_domain=True")
    return t, values



def solve_second_moment_linear(H: float, kappa1: float, vartheta: float,
                               theta: float, sigma0: float, T: float = 1.0,
                               n_steps: int = 2000, quadrature_order: int = 24,
                               log_domain: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Singularity-weighted piecewise-linear product integration.

    Unlike the legacy cell-average method, the interpolation weights integrate
    the two linear basis functions against G(s)^2. This substantially improves
    refinement for small H. Grid and quadrature refinement remain necessary.
    """
    if not 0 < H <= 0.5 or kappa1 < 0 or vartheta < 0 or min(theta, sigma0) <= 0:
        raise ValueError("invalid model parameters")
    if T <= 0 or n_steps < 1 or quadrature_order < 2:
        raise ValueError("invalid horizon, grid size or quadrature order")
    t = np.linspace(0, T, n_steps + 1)
    dt, power, alpha = T / n_steps, 2 * H, H + .5
    y = t ** power
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    ys = (y[1:, None] + y[:-1, None]) / 2 + (y[1:, None] - y[:-1, None]) * nodes / 2
    times = ys ** (1 / power)
    ml = mittag_leffler_aa_vec(alpha, kappa1 * times ** alpha)
    density_weights = (y[1:, None] - y[:-1, None]) / (2 * power) * ml ** 2 * weights
    left = vartheta ** 2 * np.sum(density_weights * (t[1:, None] - times) / dt, axis=1)
    right = vartheta ** 2 * np.sum(density_weights * (times - t[:-1, None]) / dt, axis=1)
    denominator = 1 - left[0]
    if denominator <= 0:
        raise ValueError("grid too coarse: weighted implicit coefficient is at least one")
    mean = (np.full_like(t, theta) if sigma0 == theta else
            theta + (sigma0 - theta) * mittag_leffler_negative_vec(alpha, kappa1 * t ** alpha))
    if log_domain:
        values = np.empty(n_steps + 1)
        values[0] = 2 * np.log(sigma0)
        logleft, logright = np.full_like(left, -np.inf), np.full_like(right, -np.inf)
        np.log(left, out=logleft, where=left > 0)
        np.log(right, out=logright, where=right > 0)
        for n in range(1, n_steps + 1):
            a = logsumexp(logleft[n - 1:0:-1] + values[1:n])
            b = logsumexp(logright[n - 1::-1] + values[:n])
            values[n] = np.logaddexp(2 * np.log(mean[n]), np.logaddexp(a, b)) - np.log(denominator)
        return t, values
    values = np.empty(n_steps + 1)
    values[0] = sigma0 ** 2
    for n in range(1, n_steps + 1):
        known = left[n - 1:0:-1] @ values[1:n] + right[n - 1::-1] @ values[:n]
        values[n] = (mean[n] ** 2 + known) / denominator
    if not np.all(np.isfinite(values)):
        raise OverflowError("use log_domain=True for this parameter set")
    return t, values


class LocalTests(Enum):
    ML_EVALUATOR_CHECK = 1
    CRITICAL_VARTHETA_TABLE = 2
    TABLE2_CHECK = 3
    LONG_HORIZON_CHECK = 4
    PARAMETER_SETS = 5
    DAILY_INNOVATION_TABLE = 6
    GROWTH_TABLE_LOG_DOMAIN = 7
    HIGH_GROWTH_REFINEMENT = 8


def run_local_test(local_test: LocalTests) -> None:
    if local_test == LocalTests.ML_EVALUATOR_CHECK:
        print("Mittag-Leffler evaluator versus independent spectral integrals")
        for alpha in [0.6, 0.7, 0.8, 0.9]:
            xs = np.array([0.0, 0.5, 3.0, 3.01, 4.5, 11.9, 12.0, 15.0, 19.0])
            ref = np.array([_ml_negative_spectral(alpha, xi, alpha) for xi in xs])
            got = mittag_leffler_aa_vec(alpha, xs)
            error = np.max(np.abs(got / ref - 1))
            print(f"  alpha={alpha}: max relative error={error:.3e}")
            if error > 2e-8:
                raise AssertionError("Mittag-Leffler evaluator discrepancy")

    elif local_test == LocalTests.CRITICAL_VARTHETA_TABLE:
        print("C(alpha) and scaling check")
        for H in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
            alpha = H + 0.5
            C = c_alpha(alpha)
            chk = g2_norm_spectral(H, 3.0) * 3.0 ** ((2 * alpha - 1) / alpha)
            print(f"  H={H:4.2f}  C(alpha)={C:.5f}  scaling check at kappa1=3: {chk:.5f}")
        print("\nCritical vartheta_c = ||G||_2^{-1} (second moment bounded iff vartheta < vartheta_c)")
        kappas = [1.0, 2.0, 3.0, 5.0, 10.0]
        print("  H \\ kappa1 " + "".join(f"{k:>9.1f}" for k in kappas))
        for H in [0.5, 0.4, 0.3, 0.2, 0.1]:
            row = [1.0 / np.sqrt(g2_norm(H, k)) for k in kappas]
            print(f"  {H:4.2f}       " + "".join(f"{r:9.3f}" for r in row))

    elif local_test == LocalTests.TABLE2_CHECK:
        print("Draft Table 2, kappa1 = 3 panel: gamma, limit theta^2/(1-gamma) and v(1)/theta^2 from the Volterra solve")
        draft = {(0.5, 0.5): 1.043, (0.5, 1.0): 1.199, (0.5, 1.85): 2.227,
                 (0.3, 0.5): 1.067, (0.3, 1.0): 1.336, (0.3, 1.85): 6.219,
                 (0.2, 0.5): 1.110, (0.2, 1.0): 1.661, (0.2, 1.85): np.nan,
                 (0.1, 0.5): 1.320, (0.1, 1.0): np.nan, (0.1, 1.85): np.nan}
        for H in [0.5, 0.3, 0.2, 0.1]:
            for vt in [0.5, 1.0, 1.85]:
                if H == 0.1 and vt == 1.85:
                    print("  H=0.1 vartheta=1.85: use GROWTH_TABLE_LOG_DOMAIN")
                    continue
                gamma_value = vt ** 2 * g2_norm(H, 3.0)
                lim = 1.0 / (1.0 - gamma_value) if gamma_value < 1 else np.inf
                t, v2 = solve_second_moment(H, 3.0, vt, 1.0, 1.0, T=1.0, n_steps=2000)
                t, v4 = solve_second_moment(H, 3.0, vt, 1.0, 1.0, T=1.0, n_steps=4000)
                _, v8 = solve_second_moment(H, 3.0, vt, 1.0, 1.0, T=1.0, n_steps=8000)
                print(f"  H={H:4.2f} vartheta={vt:4.2f}  gamma={gamma_value:8.4f}  limit={lim:10.4g}"
                      f"  v(1): n=2000 {v2[-1]:10.4g}  n=4000 {v4[-1]:10.8g}  n=8000 {v8[-1]:10.8g}  draft {draft[(H, vt)]}")

    elif local_test == LocalTests.LONG_HORIZON_CHECK:
        print("Long-horizon check of the limit, H=0.3, kappa1=3, vartheta=1.85 (bounded case): v(T)/theta^2")
        for T in [1.0, 3.0, 6.0]:
            t, v = solve_second_moment(0.3, 3.0, 1.85, 1.0, 1.0, T=T, n_steps=int(2000 * T))
            print(f"  T={T:5.1f}  v(T)={v[-1]:.4f}   limit={1.0 / (1.0 - 1.85 ** 2 * g2_norm(0.3, 3.0)):.4f}")

    elif local_test == LocalTests.PARAMETER_SETS:
        print("Parameter sets of the draft")
        for name, H, k, vt in [("Bitcoin May 2025, kappa1=3", 0.2, 3.0, 1.85),
                               ("kernel-fit moderate set", 0.1, 2.0, 0.72),
                               ("kernel-fit moderate set", 0.2, 2.0, 0.72),
                               ("kernel-fit moderate set", 0.3, 2.0, 0.72),
                               ("regression set beta=2.45 eps=1.81, kappa1=3 assumed", 0.1, 3.0, np.hypot(2.45, 1.81))]:
            gamma_value = vt ** 2 * g2_norm(H, k)
            verdict = 'bounded, v_inf/theta^2=%.3f' % (1 / (1 - gamma_value)) if gamma_value < 1 else 'explosive'
            print(f"  {name:55s} H={H:4.2f} kappa1={k:4.1f} vartheta={vt:5.3f}  gamma={gamma_value:9.4f}  -> {verdict}")

    elif local_test == LocalTests.DAILY_INNOVATION_TABLE:
        delta = 1.0 / 252
        print("Critical one-day innovation proxy (columns kappa1=1,2,3,5,10)")
        for H in [0.5, 0.4, 0.3, 0.2, 0.1]:
            scale = delta ** H / (np.sqrt(2 * H) * gamma(H + 0.5))
            row = [scale / np.sqrt(g2_norm(H, k)) for k in [1, 2, 3, 5, 10]]
            print(f"H={H:.1f}: " + " ".join(f"{v:.6f}" for v in row)
                  + f"; proxy at vartheta=1.85: {1.85 * scale:.6f}")

    elif local_test == LocalTests.GROWTH_TABLE_LOG_DOMAIN:
        for H, vt in [(0.2, 1.85), (0.1, 1.85)]:
            previous = None
            for n in [4000, 8000, 16000]:
                _, logv = solve_second_moment(H, 3.0, vt, 1.0, 1.0, n_steps=n,
                                             log_domain=True)
                value = logv[-1] / np.log(10)
                diff = np.nan if previous is None else value - previous
                print(f"H={H:.1f} vartheta={vt}: steps={n}, log10 ratio={value:.9f}, "
                      f"refinement difference={diff:.9f}")
                previous = value

    elif local_test == LocalTests.HIGH_GROWTH_REFINEMENT:
        previous = None
        for n in [16000, 32000, 64000]:
            _, values = solve_second_moment(0.2, 3.0, 1.85, 1.0, 1.0, n_steps=n)
            value = values[-1]
            difference = np.nan if previous is None else value / previous - 1
            print(f"H=0.2 vartheta=1.85: steps={n}, ratio={value:.9f}, "
                  f"relative refinement={difference:.9g}")
            previous = value

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    for local_test in LocalTests:
        run_local_test(local_test=local_test)
        print()
