"""
Moment-matched Markov kernels for the log-normal Volterra SV model.

Model, with linear drift (kappa2 = 0) for the targets:
    sigma_t = sigma0 + int_0^t K(t - s) [kappa1 (theta - sigma_s) ds + vartheta sigma_s dW_s],
    K(t) = t^(H - 1/2) / Gamma(H + 1/2),    I_T = int_0^T sigma_t^2 dt.
Markov lift: K^N(t) = sum_i w_i exp(-x_i t), so K^N(0) = w_bar = sum_i w_i.

Targets at maturity T, the quantities the pricer depends on:
    mean       m(T) = E[sigma_T]
    level      L(T) = sqrt(E[I_T] / T), the variance-swap volatility, which sets the level of the smile
    leverage   Lambda(T) = int_0^T G(T - s) v(s) ds, with cov(log S_T + I_T / 2, sigma_T) = rho vartheta Lambda(T)
where G is the resolvent of kappa1 K divided by kappa1 (G = K for kappa1 = 0) and v = E[sigma^2] solves
v = m^2 + vartheta^2 G^2 * v.

Fractional targets: product integration on a uniform grid over [0, T] with exact cell masses of K and K^2,
on n, 2n and 4n cells, then Richardson extrapolation with the observed order.
Markov targets: exact, by the matrix exponential of the linear ODE for E[u], E[u u'], E[I] and the leverage state,
where u_i = sigma^(i) - sigma0 / w_bar are the factor deviations.
Fit: (x, w) minimising the squared relative errors of m, L and Lambda over a parameter set and maturities, by
differential evolution on (log x, log w) and a local polish, subject to w_bar <= w_bar_max and x_N <= x_max.

Step rule (empirical, LocalTests.MC_CHECK): the split-step diffusion sub-step is a GBM of sigma^N with volatility
w_bar vartheta, and the scheme reproduces the Markov targets within Monte Carlo error when
w_bar vartheta sqrt(h) <= 0.4. A fitted kernel with a larger w_bar therefore needs a smaller step.

Units: maturities and steps in years, vols as annualised decimals, level errors in bp of vol, leverage errors in %.
Run with OMP_NUM_THREADS=1: the fit calls expm on 17 x 17 matrices about 10^6 times, and threaded BLAS slowed it
by two orders of magnitude when two runs shared two cores.
Entry points: fractional_moment_targets, markov_moment_targets, fit_kernel_to_moments, max_time_step, mc_moments.
Not here: targets for kappa2 > 0 (no closed moment equations) and the scheme itself, which lives in
stochvolmodels.pricers.rough_logsv.split_simulation.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma
# stochvolmodels
from stochvolmodels.models.rough_logsv.rough_kernel import european_rule
from stochvolmodels.pricers.rough_logsv.split_simulation import log_spot_full_combined

ModelParams = Tuple[float, float, float, float]   # (sigma0, theta, kappa1, vartheta)

MATURITIES = (1.0 / 52.0, 1.0 / 12.0, 0.25, 0.5, 1.0)
FIT_PARAMS = ((0.5, 0.5, 1.0, 0.3), (0.5, 0.5, 1.0, 0.6), (0.5, 0.5, 4.0, 0.4),
              (0.5, 0.5, 4.0, 0.8), (0.5, 0.5, 8.0, 1.0), (0.6, 0.4, 2.0, 0.5))
TEST_PARAMS = ((0.5, 0.4, 2.0, 0.72), (0.3, 0.3, 0.5, 0.5), (0.8, 0.8, 6.0, 0.9), (0.6, 0.9, 3.0, 0.6))
STEP_RULE_CONSTANT = 0.4    # admissible w_bar vartheta sqrt(h)


class MomentKey(str, Enum):
    MEAN = 'mean'
    LEVEL = 'level'
    LEVERAGE = 'leverage'


def _fractional_moments_on_grid(hurst: float,
                                sigma0: float,
                                theta: float,
                                kappa1: float,
                                vartheta: float,
                                maturity: float,
                                n_grid: int
                                ) -> Dict[MomentKey, float]:
    """mean, level and leverage at the maturity by product integration on n_grid uniform cells."""
    alpha = hurst + 0.5
    dt = maturity / n_grid
    t = np.arange(n_grid + 1) * dt
    k = np.arange(n_grid)
    c1 = ((k + 1) ** alpha - k ** alpha) * dt ** alpha / gamma(alpha + 1.0)    # cell masses of K
    c2 = ((k + 1) ** (2.0 * hurst) - k ** (2.0 * hurst)) * dt ** (2.0 * hurst) / (2.0 * hurst * gamma(alpha) ** 2)
    # F = int_0^t R solves F = kappa1 int K - kappa1 K * F; then g1 = cell masses of G
    if kappa1 > 0.0:
        cum_k = t ** alpha / gamma(alpha + 1.0)
        f = np.zeros(n_grid + 1)
        for i in range(1, n_grid + 1):
            lag = i - 1 - np.arange(i - 1)
            conv = np.dot(c1[lag], 0.5 * (f[:i - 1] + f[1:i])) if i > 1 else 0.0
            f[i] = kappa1 * (cum_k[i] - conv - 0.5 * c1[0] * f[i - 1]) / (1.0 + 0.5 * kappa1 * c1[0])
        g1 = np.diff(f) / kappa1
    else:
        f = np.zeros(n_grid + 1)
        g1 = c1.copy()
    m = theta + (sigma0 - theta) * (1.0 - f)
    g2 = c2 * (g1 / c1) ** 2    # cell masses of G^2, with G / K frozen within a cell
    v = np.zeros(n_grid + 1)
    v[0] = sigma0 ** 2
    for i in range(1, n_grid + 1):
        lag = i - 1 - np.arange(i - 1)
        conv = np.dot(g2[lag], 0.5 * (v[:i - 1] + v[1:i])) if i > 1 else 0.0
        v[i] = (m[i] ** 2 + vartheta ** 2 * (conv + 0.5 * g2[0] * v[i - 1])) / (1.0 - 0.5 * vartheta ** 2 * g2[0])
    v_bar = 0.5 * (v[:-1] + v[1:])
    return {MomentKey.MEAN: m[-1],
            MomentKey.LEVEL: np.sqrt(np.sum(v_bar) * dt / maturity),
            MomentKey.LEVERAGE: np.dot(g1[::-1], v_bar)}


def fractional_moment_targets(hurst: float,
                              sigma0: float,
                              theta: float,
                              kappa1: float,
                              vartheta: float,
                              maturities: Tuple[float, ...] = MATURITIES,
                              n_grid: int = 4000    # cells of the coarsest grid
                              ) -> Dict[MomentKey, np.ndarray]:
    """exact targets for the fractional kernel, extrapolated from n_grid, 2 n_grid and 4 n_grid cells."""
    if not 0.0 < hurst <= 0.5:
        raise ValueError(f"hurst must be in (0, 0.5], given {hurst}")
    if np.any(np.asarray(maturities) <= 0.0):
        raise ValueError(f"maturities must be positive, given {maturities}")
    out = {key: np.zeros(len(maturities)) for key in MomentKey}
    for j, maturity in enumerate(maturities):
        runs = [_fractional_moments_on_grid(hurst, sigma0, theta, kappa1, vartheta, maturity, n_grid * 2 ** p)
                for p in range(3)]
        for key in MomentKey:
            e0, e1, e2 = (run[key] for run in runs)
            d1, d2 = e1 - e0, e2 - e1
            if abs(d2) < 1e-14 or d1 * d2 <= 0.0 or abs(d2) >= abs(d1):
                out[key][j] = e2    # no monotone convergence to extrapolate
            else:
                out[key][j] = e2 + d2 / (d1 / d2 - 1.0)
    return out


def markov_moment_targets(nodes: np.ndarray,
                          weights: np.ndarray,
                          sigma0: float,
                          theta: float,
                          kappa1: float,
                          vartheta: float,
                          maturities: Tuple[float, ...] = MATURITIES
                          ) -> Dict[MomentKey, np.ndarray]:
    """exact targets for K^N = sum_i w_i exp(-x_i t) by the matrix exponential of the moment ODE."""
    x = np.asarray(nodes, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape != w.shape:
        raise ValueError(f"nodes and weights differ in shape: {x.shape} vs {w.shape}")
    n = x.size
    one = np.ones(n)
    a_mat = np.diag(x) + kappa1 * np.outer(one, w)
    a_vec = kappa1 * (theta - sigma0) * one
    # state z = [mu (n), vec P (n^2, row-major), I, y (n), 1]
    i_mu, i_p = slice(0, n), slice(n, n + n * n)
    i_i = n + n * n
    i_y = slice(i_i + 1, i_i + 1 + n)
    i_c = i_i + 1 + n
    dim = i_c + 1
    e2 = np.zeros(dim)    # E[(sigma^N)^2] = sigma0^2 + 2 sigma0 w'mu + w'Pw as a row over z
    e2[i_mu] = 2.0 * sigma0 * w
    e2[i_p] = np.outer(w, w).ravel()
    e2[i_c] = sigma0 ** 2
    gen = np.zeros((dim, dim))
    gen[i_mu, i_mu] = -a_mat                       # mu' = a - A mu
    gen[i_mu, i_c] = a_vec
    eye = np.eye(n)
    gen[i_p, i_p] = -(np.kron(a_mat, eye) + np.kron(eye, a_mat))   # P' = a mu' + mu a' - A P - P A' + vartheta^2 e2 1 1'
    for r in range(n):
        for c in range(n):
            row = n + r * n + c
            gen[row, c] += a_vec[r]
            gen[row, r] += a_vec[c]
            gen[row, :] += vartheta ** 2 * e2
    gen[i_i, :] = e2                               # I' = e2
    gen[i_y, i_y] = -a_mat                         # y' = -A y + 1 e2, Lambda = w'y
    gen[i_y, :] += np.outer(one, e2)
    z0 = np.zeros(dim)
    z0[i_c] = 1.0
    out = {key: np.zeros(len(maturities)) for key in MomentKey}
    for j, maturity in enumerate(maturities):
        z = expm(gen * maturity) @ z0
        out[MomentKey.MEAN][j] = sigma0 + w @ z[i_mu]
        out[MomentKey.LEVEL][j] = np.sqrt(z[i_i] / maturity)
        out[MomentKey.LEVERAGE][j] = w @ z[i_y]
    return out


def fit_kernel_to_moments(hurst: float,
                          n_nodes: int = 3,
                          fit_params: Tuple[ModelParams, ...] = FIT_PARAMS,
                          maturities: Tuple[float, ...] = MATURITIES,
                          w_bar_max: Optional[float] = None,    # cap on K^N(0), sets the admissible step
                          x_max: float = 360.0,                 # cap on the largest node, 1/years
                          targets: Optional[List[Dict[MomentKey, np.ndarray]]] = None,
                          seed: int = 1
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """nodes and weights of K^N fitted to the mean, level and leverage of the fractional model."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be positive, given {n_nodes}")
    if w_bar_max is not None and w_bar_max <= 0.0:
        raise ValueError(f"w_bar_max must be positive, given {w_bar_max}")
    if targets is None:
        targets = [fractional_moment_targets(hurst, *p, maturities=maturities) for p in fit_params]

    def loss(z: np.ndarray) -> float:
        x, w = np.exp(z[:n_nodes]), np.exp(z[n_nodes:])
        total = 0.0
        if w_bar_max is not None:
            total += 1e3 * max(w.sum() / w_bar_max - 1.0, 0.0) ** 2
        for p, target in zip(fit_params, targets):
            got = markov_moment_targets(x, w, *p, maturities=maturities)
            for key in MomentKey:
                total += np.sum((got[key] / target[key] - 1.0) ** 2)
        return total if np.isfinite(total) else 1e6

    t_max = float(np.max(maturities))
    lo_x, hi_x = np.log(1e-3 / t_max), np.log(x_max)
    bounds = [(lo_x, hi_x)] * n_nodes + [(np.log(1e-3), np.log(3e2))] * n_nodes
    # seed the population with the European rule, scaled down to w_bar_max if needed
    x_e, w_e = map(np.asarray, european_rule(hurst, n_nodes, t_max))
    if w_bar_max is not None:
        w_e = w_e * min(1.0, w_bar_max / w_e.sum())
    z_e = np.r_[np.log(np.clip(x_e, np.exp(lo_x), x_max)), np.log(np.clip(w_e, 1e-3, 3e2))]
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds).T
    population = np.vstack([z_e, lo + (hi - lo) * rng.random((30 * n_nodes - 1, 2 * n_nodes))])
    best = differential_evolution(loss, bounds, seed=seed, maxiter=400, tol=1e-10, polish=False, init=population)
    for method in ('L-BFGS-B', 'Nelder-Mead'):
        local = minimize(loss, best.x, method=method, bounds=bounds, options=dict(maxiter=4000))
        if np.isfinite(local.fun) and local.fun < best.fun:
            best = local
    x, w = np.exp(best.x[:n_nodes]), np.exp(best.x[n_nodes:])
    order = np.argsort(x)
    return x[order], w[order]


def max_time_step(w_bar: float,
                  vartheta: float,
                  constant: float = STEP_RULE_CONSTANT
                  ) -> float:
    """largest step in years with w_bar vartheta sqrt(h) <= constant."""
    if w_bar <= 0.0 or vartheta <= 0.0:
        raise ValueError(f"w_bar and vartheta must be positive, given {w_bar}, {vartheta}")
    return (constant / (w_bar * vartheta)) ** 2


def kernel_mass_errors(nodes: np.ndarray,
                       weights: np.ndarray,
                       hurst: float,
                       horizon: float = 1.0
                       ) -> Tuple[float, float]:
    """relative errors (%) of int_0^horizon K^N and int_0^horizon (K^N)^2 against the fractional kernel."""
    x, w = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    alpha = hurst + 0.5
    mass1 = np.sum(w * np.where(x > 0.0, -np.expm1(-x * horizon) / np.where(x > 0.0, x, 1.0), horizon))
    s = x[:, None] + x[None, :]
    mass2 = np.sum(np.outer(w, w) * np.where(s > 0.0, -np.expm1(-s * horizon) / np.where(s > 0.0, s, 1.0), horizon))
    exact1 = horizon ** alpha / gamma(alpha + 1.0)
    exact2 = horizon ** (2.0 * hurst) / (2.0 * hurst * gamma(alpha) ** 2)
    return 100.0 * (mass1 / exact1 - 1.0), 100.0 * (mass2 / exact2 - 1.0)


def kernel_errors(nodes: np.ndarray,
                  weights: np.ndarray,
                  params: Tuple[ModelParams, ...],
                  targets: List[Dict[MomentKey, np.ndarray]],
                  maturities: Tuple[float, ...] = MATURITIES
                  ) -> pd.DataFrame:
    """level error (bp of vol), leverage and mean errors of K^N against the targets, one row per (params, maturity)."""
    rows = []
    for p, target in zip(params, targets):
        got = markov_moment_targets(nodes, weights, *p, maturities=maturities)
        for j, maturity in enumerate(maturities):
            rows.append(dict(params=str(p), maturity=maturity,
                             level_bp=1e4 * (got[MomentKey.LEVEL][j] - target[MomentKey.LEVEL][j]),
                             leverage_pct=100.0 * (got[MomentKey.LEVERAGE][j] / target[MomentKey.LEVERAGE][j] - 1.0),
                             mean_bp=1e4 * (got[MomentKey.MEAN][j] - target[MomentKey.MEAN][j])))
    return pd.DataFrame(rows)


def mc_moments(nodes: np.ndarray,
               weights: np.ndarray,
               sigma0: float,
               theta: float,
               kappa1: float,
               vartheta: float,
               rho: float,
               maturity: float,
               n_steps: int,
               n_paths: int = 100000,
               n_batches: int = 5,
               seed: int = 11
               ) -> Dict[str, float]:
    """level, leverage and E[S_T] with standard errors from the stochvolmodels split-step scheme, kappa2 = 0."""
    if n_paths % n_batches != 0:
        raise ValueError(f"n_paths={n_paths} must be a multiple of n_batches={n_batches}")
    x, w = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    nb = n_paths // n_batches
    node_mat = np.repeat(x[:, None], nb, axis=1)
    weight_mat = np.repeat(w[:, None], nb, axis=1)
    sigma0_factors = np.full_like(node_mat, sigma0 / w.sum())
    timegrid = np.linspace(0.0, maturity, n_steps + 1)
    rng = np.random.default_rng(seed)
    log_s, qvar, sigma_t = [], [], []
    for _ in range(n_batches):
        z0 = rng.standard_normal((n_steps, nb))
        z1 = rng.standard_normal((n_steps, nb))
        # positional call: the order is the same on main and on the rough-logsv-notation branch, the names differ
        ls, factors, qv = log_spot_full_combined(node_mat, weight_mat, sigma0_factors, theta, kappa1, 0.0, 0.0,
                                                 sigma0_factors.copy(), rho, vartheta, timegrid, nb, z0, z1)
        log_s.append(ls[0])
        qvar.append(qv[0])
        sigma_t.append(np.sum(weight_mat * factors, axis=0))
    log_s, qvar, sigma_t = np.concatenate(log_s), np.concatenate(qvar), np.concatenate(sigma_t)
    level = np.sqrt(np.mean(qvar) / maturity)
    martingale = log_s + 0.5 * qvar
    cross = (martingale - martingale.mean()) * (sigma_t - sigma_t.mean())
    spot = np.exp(log_s)
    return dict(level=level, level_se=np.std(qvar / maturity) / np.sqrt(n_paths) / (2.0 * level),
                leverage=np.mean(cross) / (rho * vartheta),
                leverage_se=np.std(cross) / np.sqrt(n_paths) / abs(rho * vartheta),
                spot_mean=spot.mean(), spot_mean_se=spot.std() / np.sqrt(n_paths))


class LocalTests(Enum):
    KERNEL_FIT_TABLE = 1
    MC_CHECK = 2
    LEVEL_BY_MATURITY = 3


def run_local_test(local_test: LocalTests):

    hursts = (0.1, 0.2, 0.3)
    pd.set_option('display.width', 200)

    def fitted_kernels(hurst: float) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], list, list]:
        fit_targets = [fractional_moment_targets(hurst, *p) for p in FIT_PARAMS]
        test_targets = [fractional_moment_targets(hurst, *p) for p in TEST_PARAMS]
        x_e, w_e = map(np.asarray, european_rule(hurst, 3, 1.0))
        kernels = {'european': (x_e, w_e),
                   'fit, same w_bar': fit_kernel_to_moments(hurst, w_bar_max=w_e.sum(), targets=fit_targets),
                   'fit, free w_bar': fit_kernel_to_moments(hurst, targets=fit_targets)}
        return kernels, fit_targets, test_targets

    if local_test == LocalTests.KERNEL_FIT_TABLE:
        rows = []
        for hurst in hursts:
            kernels, fit_targets, test_targets = fitted_kernels(hurst)
            for name, (x, w) in kernels.items():
                err_fit = kernel_errors(x, w, FIT_PARAMS, fit_targets)
                err_test = kernel_errors(x, w, TEST_PARAMS, test_targets)
                mass1_pct, mass2_pct = kernel_mass_errors(x, w, hurst)
                rows.append(dict(H=hurst, kernel=name, nodes=np.round(x, 3), weights=np.round(w, 3), w_bar=w.sum(),
                                 steps_per_year=1.0 / max_time_step(w.sum(), 1.0),
                                 int_k_pct=mass1_pct, int_k2_pct=mass2_pct,
                                 fit_level_bp=err_fit['level_bp'].abs().max(),
                                 fit_leverage_pct=err_fit['leverage_pct'].abs().max(),
                                 test_level_bp=err_test['level_bp'].abs().max(),
                                 test_leverage_pct=err_test['leverage_pct'].abs().max(),
                                 test_mean_bp=err_test['mean_bp'].abs().max()))
                print(f"H={hurst} {name}: nodes {np.round(x, 3)} weights {np.round(w, 3)}", flush=True)
        print(pd.DataFrame(rows).drop(columns=['nodes', 'weights']).round(1).to_string(index=False))

    elif local_test == LocalTests.LEVEL_BY_MATURITY:
        # term structure of the kernel error of the quadrature rule on the test sets
        for hurst in hursts:
            x_e, w_e = map(np.asarray, european_rule(hurst, 3, 1.0))
            test_targets = [fractional_moment_targets(hurst, *p) for p in TEST_PARAMS]
            errors = kernel_errors(x_e, w_e, TEST_PARAMS, test_targets)
            print(f"H={hurst}, quadrature rule")
            print(errors.pivot(index='params', columns='maturity', values='level_bp').round(0).to_string())
            print(errors.pivot(index='params', columns='maturity', values='leverage_pct').round(1).to_string())

    elif local_test == LocalTests.MC_CHECK:
        sigma0, theta, kappa1, vartheta = TEST_PARAMS[0]
        rho, maturity = -0.7, 0.25
        for hurst in (0.1, 0.2):
            kernels, _, test_targets = fitted_kernels(hurst)
            exact = fractional_moment_targets(hurst, *TEST_PARAMS[0], maturities=(maturity,))
            for name, (x, w) in kernels.items():
                markov = markov_moment_targets(x, w, *TEST_PARAMS[0], maturities=(maturity,))
                n_rule = int(np.ceil(maturity / max_time_step(w.sum(), vartheta)))
                for n_steps in sorted({63, n_rule}):
                    mc = mc_moments(x, w, sigma0, theta, kappa1, vartheta, rho, maturity, n_steps=n_steps)
                    print(f"H={hurst} {name:16s} w_bar={w.sum():5.2f} steps/yr={n_steps / maturity:5.0f} "
                          f"w_bar*vartheta*sqrt(h)={w.sum() * vartheta * np.sqrt(maturity / n_steps):.2f} | "
                          f"level MC-Markov {1e4 * (mc['level'] - markov[MomentKey.LEVEL][0]):+5.0f}±{1e4 * mc['level_se']:.0f}bp, "
                          f"Markov-exact {1e4 * (markov[MomentKey.LEVEL][0] - exact[MomentKey.LEVEL][0]):+6.0f}bp | "
                          f"leverage MC/Markov-1 {100 * (mc['leverage'] / markov[MomentKey.LEVERAGE][0] - 1):+5.1f}"
                          f"±{100 * mc['leverage_se'] / markov[MomentKey.LEVERAGE][0]:.1f}%, "
                          f"Markov/exact-1 {100 * (markov[MomentKey.LEVERAGE][0] / exact[MomentKey.LEVERAGE][0] - 1):+6.1f}% | "
                          f"E[S]-1 {1e4 * (mc['spot_mean'] - 1):+4.0f}±{1e4 * mc['spot_mean_se']:.0f}bp", flush=True)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.KERNEL_FIT_TABLE)
