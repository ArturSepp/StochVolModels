"""
Sum-of-exponentials kernels fitted to the moments of the rough log-normal Volterra model.

Model with linear drift (kappa2 = 0) and total volatility of volatility vartheta:

    sigma_t = sigma0 + int_0^t K(t - s) [kappa1 (theta - sigma_s) ds + vartheta sigma_s dW_s],
    K(t) = t^(H - 1/2) / Gamma(H + 1/2),    I_T = int_0^T sigma_t^2 dt.

The Markovian approximation is K^N(t) = sum_i weights_i exp(-nodes_i t), so that
K^N(0) = w_bar = sum_i weights_i. Prices depend on the kernel through three moment functionals
at each maturity T:

    mean      m(T) = E[sigma_T]
    level     L(T) = sqrt(E[I_T] / T), the variance-swap volatility
    leverage  Lambda(T) = int_0^T G(T - s) v(s) ds, with cov(log S_T + I_T / 2, sigma_T)
              = rho vartheta Lambda(T),

where G = R / kappa1 with R the resolvent of kappa1 K (G = K for kappa1 = 0), and v = E[sigma^2]
solves the linear Volterra equation v = m^2 + vartheta^2 G^2 * v.

``fractional_moment_targets`` evaluates the three functionals for the fractional kernel by product
integration with exact cell masses of K and K^2 on n, 2n and 4n cells, followed by Richardson
extrapolation. ``markov_moment_targets`` evaluates them exactly for K^N by the matrix exponential
of the linear ODE for the first and second moments of the factors. ``fit_kernel_to_moments``
chooses (nodes, weights) to minimise the squared relative errors over a set of parameters and
maturities, by differential evolution in (log nodes, log weights) and a local polish. An optional
w_bar penalty is followed by an achieved-cap check; an infeasible fit raises rather than returning
a kernel above the cap. ``moment_fitted_kernel`` caches accepted fits for
``LogSvParams.approximate_kernel``.

The European rule of Bayer and Breneis (2023) matches int K but underestimates int K^2, on which
the second moment depends: by 16.9% over one year for H = 0.1 and N = 3. The fit removes most of
the resulting error but raises w_bar. The split-step scheme of ``split_simulation`` moves sigma^N
as a geometric Brownian motion with volatility w_bar vartheta over each step and reproduces the
moments of its own kernel only when w_bar vartheta sqrt(h) <= 0.4 (an empirical bound, see
``max_time_step`` and ``max_weight_sum``).

Reference: A. Sepp and P. Rakhmonov (2026), Log-normal Volterra stochastic volatility model,
working paper, subsection "Kernel fitted to the moments".

Conventions: time in years, nodes in 1/years, volatilities annualised. A fit evaluates the loss
about 3.6e4 times, each with N_params x N_maturities matrix exponentials of dimension N^2 + 2N + 2;
it takes about a minute with single-threaded BLAS, and threaded BLAS can be far slower when
several processes share the cores (set OMP_NUM_THREADS=1).
Not here: moment targets for kappa2 > 0 (no closed moment equations) and the simulation
(``split_simulation``).
"""
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize
from scipy.special import gamma

from stochvolmodels.pricers.rough_logsv.rough_kernel import european_rule

MATURITY_FRACTIONS = (1.0 / 52.0, 1.0 / 12.0, 0.25, 0.5, 1.0)
"""fit maturities as fractions of the kernel horizon T (one week to one year for T = 1)."""

DEFAULT_FIT_PARAMS = ((0.5, 0.5, 1.0, 0.3), (0.5, 0.5, 1.0, 0.6), (0.5, 0.5, 4.0, 0.4),
                      (0.5, 0.5, 4.0, 0.8), (0.5, 0.5, 8.0, 1.0), (0.6, 0.4, 2.0, 0.5))
"""fit parameter sets (sigma0, theta, kappa1, vartheta); with kappa2 = 0 the moments are invariant
to a joint scaling of sigma0 and theta, so the sets cover theta / sigma0 in [0.67, 1], kappa1 in
[1, 8] and vartheta in [0.3, 1]."""

STEP_RULE_CONSTANT = 0.4
"""admissible w_bar vartheta sqrt(h) of the split-step scheme (empirical)."""

ParamSet = Tuple[float, float, float, float]    # (sigma0, theta, kappa1, vartheta)


class MomentKey(str, Enum):
    """moment functionals of the model that the kernel fit matches."""
    MEAN = 'mean'
    LEVEL = 'level'
    LEVERAGE = 'leverage'


class KernelRule(str, Enum):
    """rule for the Markovian approximation of the fractional kernel."""
    EUROPEAN = 'european'
    MOMENT_FIT = 'moment_fit'


def _validate_maturities(maturities) -> np.ndarray:
    """Require a nonempty finite positive maturity vector."""
    values = np.asarray(maturities, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError('maturities must be a nonempty 1-d vector of finite positive values')
    return values


def _validate_model_params(sigma0, theta, kappa1, vartheta) -> None:
    """Validate the nonnegative linear-drift moment model."""
    values = np.asarray([sigma0, theta, kappa1, vartheta], dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError('sigma0, theta, kappa1 and vartheta must be finite nonnegative scalars')


def _validate_kernel(nodes, weights) -> Tuple[np.ndarray, np.ndarray]:
    """Require a nonempty positive-weight exponential kernel."""
    x, w = np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)
    if x.shape != w.shape or x.ndim != 1:
        raise ValueError(f'nodes and weights must be 1-d of equal shape: {x.shape} vs {w.shape}')
    if (x.size == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(w))
            or np.any(x < 0.0) or np.any(w <= 0.0)):
        raise ValueError('kernel requires finite nonnegative nodes and finite positive weights')
    return x, w


def _validate_positive(value, name: str) -> None:
    """Reject nonfinite and nonpositive scalar controls."""
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be finite and positive, given {value}')


def _validate_fit_domain(hurst, n_nodes, maturities, w_bar_max, x_max) -> np.ndarray:
    """Check the search domain before constructing targets or starting an optimizer."""
    if not 0.0 < hurst < 0.5:
        raise ValueError(f'hurst must be in (0, 0.5), given {hurst}')
    if isinstance(n_nodes, (bool, np.bool_)) or not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError(f'n_nodes must be a positive integer, given {n_nodes}')
    times = _validate_maturities(maturities)
    _validate_positive(x_max, 'x_max')
    if x_max < 1e-3 / np.max(times):
        raise ValueError('x_max is below the smallest search node 1e-3 / horizon')
    if w_bar_max is not None:
        _validate_positive(w_bar_max, 'w_bar_max')
        if w_bar_max < n_nodes * 1e-3:
            raise ValueError('w_bar_max is infeasible for the minimum search weight 1e-3 per node')
    return times


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
    c2 = (((k + 1) ** (2.0 * hurst) - k ** (2.0 * hurst)) * dt ** (2.0 * hurst)
          / (2.0 * hurst * gamma(alpha) ** 2))                                  # cell masses of K^2
    # F = int_0^t R solves F = kappa1 int K - kappa1 K * F, and g1 are the cell masses of G
    f = np.zeros(n_grid + 1)
    if kappa1 > 0.0:
        cum_k = t ** alpha / gamma(alpha + 1.0)
        denominator = 1.0 + 0.5 * kappa1 * c1[0]
        for i in range(1, n_grid + 1):
            lag = i - 1 - np.arange(i - 1)
            conv = np.dot(c1[lag], 0.5 * (f[:i - 1] + f[1:i])) if i > 1 else 0.0
            f[i] = kappa1 * (cum_k[i] - conv - 0.5 * c1[0] * f[i - 1]) / denominator
        g1 = np.diff(f) / kappa1
    else:
        g1 = c1.copy()
    m = theta + (sigma0 - theta) * (1.0 - f)
    g2 = c2 * (g1 / c1) ** 2    # cell masses of G^2, with G / K frozen within a cell
    variance_denominator = 1.0 - 0.5 * vartheta ** 2 * g2[0]
    if not np.isfinite(variance_denominator) or variance_denominator <= 0.0:
        raise ValueError('n_grid is too small for a positive variance solve; increase n_grid')
    v = np.zeros(n_grid + 1)
    v[0] = sigma0 ** 2
    for i in range(1, n_grid + 1):
        lag = i - 1 - np.arange(i - 1)
        conv = np.dot(g2[lag], 0.5 * (v[:i - 1] + v[1:i])) if i > 1 else 0.0
        v[i] = ((m[i] ** 2 + vartheta ** 2 * (conv + 0.5 * g2[0] * v[i - 1]))
                / variance_denominator)
    v_bar = 0.5 * (v[:-1] + v[1:])
    return {MomentKey.MEAN: m[-1],
            MomentKey.LEVEL: np.sqrt(np.sum(v_bar) * dt / maturity),
            MomentKey.LEVERAGE: np.dot(g1[::-1], v_bar)}


def fractional_moment_targets(hurst: float,
                              sigma0: float,
                              theta: float,
                              kappa1: float,
                              vartheta: float,
                              maturities: Tuple[float, ...] = MATURITY_FRACTIONS,
                              n_grid: int = 4000
                              ) -> Dict[MomentKey, np.ndarray]:
    """
    mean, level and leverage of the model with the fractional kernel and kappa2 = 0.

    Parameters
    ----------
    hurst : float
        Hurst exponent H in (0, 0.5].
    sigma0, theta, kappa1, vartheta : float
        Initial volatility, mean level, linear mean reversion and total volatility of volatility.
    maturities : tuple of float
        Maturities in years.
    n_grid : int, default 4000
        Cells of the coarsest grid. The finer grids have 2 n_grid and 4 n_grid cells, and the
        result is the Richardson extrapolation with the observed order of convergence.

    Returns
    -------
    Dict[MomentKey, np.ndarray]
        One array per functional, aligned with ``maturities``.

    Notes
    -----
    These are numerical targets, not certified error bounds. If the three grids do not show
    decreasing monotone differences, the finest-grid value is returned without extrapolation.
    Research use should repeat the calculation at larger ``n_grid`` and report the change.
    """
    if not 0.0 < hurst <= 0.5:
        raise ValueError(f"hurst must be in (0, 0.5], given {hurst}")
    maturities = _validate_maturities(maturities)
    _validate_model_params(sigma0, theta, kappa1, vartheta)
    if isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer)) or n_grid < 1:
        raise ValueError(f'n_grid must be a positive integer, given {n_grid}')
    out = {key: np.zeros(len(maturities)) for key in MomentKey}
    for j, maturity in enumerate(maturities):
        runs = [_fractional_moments_on_grid(hurst, sigma0, theta, kappa1, vartheta, maturity,
                                            n_grid * 2 ** p) for p in range(3)]
        for key in MomentKey:
            e0, e1, e2 = (run[key] for run in runs)
            d1, d2 = e1 - e0, e2 - e1
            if abs(d2) < 1e-14 or d1 * d2 <= 0.0 or abs(d2) >= abs(d1):
                out[key][j] = e2    # no monotone convergence to extrapolate
            else:
                out[key][j] = e2 + d2 / (d1 / d2 - 1.0)
    if any(not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in out.values()):
        raise FloatingPointError('fractional moment targets are nonfinite or negative; refine n_grid or parameters')
    return out


def markov_moment_targets(nodes: np.ndarray,
                          weights: np.ndarray,
                          sigma0: float,
                          theta: float,
                          kappa1: float,
                          vartheta: float,
                          maturities: Tuple[float, ...] = MATURITY_FRACTIONS
                          ) -> Dict[MomentKey, np.ndarray]:
    """
    exact mean, level and leverage of the model with K^N and kappa2 = 0.

    With factor deviations u_i = sigma^(i) - sigma0 / w_bar, A = diag(nodes) + kappa1 1 w' and
    a = kappa1 (theta - sigma0) 1, the moments mu = E[u] and P = E[u u'] solve
    mu' = a - A mu and P' = a mu' + mu a' - A P - P A' + vartheta^2 e2 1 1', with
    e2 = E[(sigma^N)^2] = sigma0^2 + 2 sigma0 w'mu + w'Pw. Then E[I]' = e2, and the leverage
    state y' = -A y + 1 e2 gives Lambda = w'y. The system is linear in (mu, P, E[I], y, 1) and is
    solved by the matrix exponential.

    Parameters
    ----------
    nodes, weights : np.ndarray
        Nodes x_i >= 0 and weights w_i > 0 of K^N.
    sigma0, theta, kappa1, vartheta : float
        Initial volatility, mean level, linear mean reversion and total volatility of volatility.
    maturities : tuple of float
        Maturities in years.

    Returns
    -------
    Dict[MomentKey, np.ndarray]
        One array per functional, aligned with ``maturities``.
    """
    x, w = _validate_kernel(nodes, weights)
    maturities = _validate_maturities(maturities)
    _validate_model_params(sigma0, theta, kappa1, vartheta)
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
    e2 = np.zeros(dim)    # E[(sigma^N)^2] as a row over z
    e2[i_mu] = 2.0 * sigma0 * w
    e2[i_p] = np.outer(w, w).ravel()
    e2[i_c] = sigma0 ** 2
    gen = np.zeros((dim, dim))
    gen[i_mu, i_mu] = -a_mat
    gen[i_mu, i_c] = a_vec
    eye = np.eye(n)
    gen[i_p, i_p] = -(np.kron(a_mat, eye) + np.kron(eye, a_mat))
    for r in range(n):
        for c in range(n):
            row = n + r * n + c
            gen[row, c] += a_vec[r]
            gen[row, r] += a_vec[c]
            gen[row, :] += vartheta ** 2 * e2
    gen[i_i, :] = e2
    gen[i_y, i_y] = -a_mat
    gen[i_y, :] += np.outer(one, e2)
    z0 = np.zeros(dim)
    z0[i_c] = 1.0
    out = {key: np.zeros(len(maturities)) for key in MomentKey}
    for j, maturity in enumerate(maturities):
        z = expm(gen * maturity) @ z0
        out[MomentKey.MEAN][j] = sigma0 + w @ z[i_mu]
        out[MomentKey.LEVEL][j] = np.sqrt(z[i_i] / maturity)
        out[MomentKey.LEVERAGE][j] = w @ z[i_y]
    if any(not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in out.values()):
        raise FloatingPointError('Markov moment targets are nonfinite or negative for this kernel and horizon')
    return out


def fit_kernel_to_moments(hurst: float,
                          n_nodes: int = 3,
                          maturities: Tuple[float, ...] = MATURITY_FRACTIONS,
                          fit_params: Tuple[ParamSet, ...] = DEFAULT_FIT_PARAMS,
                          w_bar_max: Optional[float] = None,
                          x_max: float = 360.0,
                          targets: Optional[List[Dict[MomentKey, np.ndarray]]] = None,
                          seed: int = 1
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    nodes and weights of K^N fitted to the mean, level and leverage of the fractional model.

    The loss is the sum over ``fit_params`` and ``maturities`` of the squared relative errors of
    the three functionals of ``markov_moment_targets`` against ``fractional_moment_targets``.
    Differential evolution in (log nodes, log weights), with the European rule in the initial
    population, is followed by L-BFGS-B and Nelder-Mead polishing.

    Parameters
    ----------
    hurst : float
        Hurst exponent H in (0, 0.5).
    n_nodes : int, default 3
        Number of exponentials N.
    maturities : tuple of float
        Fit maturities in years; the longest one is the kernel horizon.
    fit_params : tuple of (sigma0, theta, kappa1, vartheta)
        Parameter sets of the fit.
    w_bar_max : float, optional
        Requested cap on w_bar = K^N(0), which sets the admissible step of the split-step scheme
        (see ``max_weight_sum``). The optimizer uses a penalty, not a constrained solver; a
        result exceeding the cap beyond floating-point tolerance raises ValueError. No cap
        by default. A rejected result requires another fit design, not silent rescaling.
    x_max : float, default 360
        Cap on the largest node, in 1/years.
    targets : list of dict, optional
        Precomputed ``fractional_moment_targets`` for ``fit_params``.
    seed : int, default 1
        Seed of the differential evolution.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Nodes in increasing order and the matching weights.

    Notes
    -----
    The two-array return does not retain convergence metadata or certify the approximation
    error. An optimizer that did not report convergence produces a RuntimeWarning; inspect
    achieved moment errors independently before accepting a research kernel.
    """
    maturities = _validate_fit_domain(hurst, n_nodes, maturities, w_bar_max, x_max)
    if len(fit_params) == 0:
        raise ValueError('fit_params must contain at least one parameter set')
    for params in fit_params:
        if len(params) != 4:
            raise ValueError('each fit_params entry must contain sigma0, theta, kappa1 and vartheta')
        _validate_model_params(*params)
    if targets is None:
        targets = [fractional_moment_targets(hurst, *p, maturities=maturities) for p in fit_params]
    if len(targets) != len(fit_params):
        raise ValueError(f"targets has {len(targets)} entries for {len(fit_params)} parameter sets")
    for target in targets:
        for key in MomentKey:
            if key not in target:
                raise ValueError(f'target is missing {key.value}')
            values = np.asarray(target[key], dtype=float)
            if (values.shape != maturities.shape or not np.all(np.isfinite(values))
                    or np.any(values <= 0.0)):
                raise ValueError(f'target {key.value} must be finite, positive and aligned with maturities')

    def loss(z: np.ndarray) -> float:
        x, w = np.exp(z[:n_nodes]), np.exp(z[n_nodes:])
        total = 0.0
        if w_bar_max is not None:
            total += 1e3 * max(w.sum() / w_bar_max - 1.0, 0.0) ** 2
        # the search visits kernels whose second moment explodes; they get the penalty 1e6
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            for p, target in zip(fit_params, targets):
                try:
                    got = markov_moment_targets(x, w, *p, maturities=maturities)
                except (FloatingPointError, OverflowError):
                    return 1e6
                for key in MomentKey:
                    total += np.sum((got[key] / target[key] - 1.0) ** 2)
        return total if np.isfinite(total) else 1e6

    horizon = float(np.max(maturities))
    lo_x, hi_x = np.log(1e-3 / horizon), np.log(x_max)
    bounds = [(lo_x, hi_x)] * n_nodes + [(np.log(1e-3), np.log(3e2))] * n_nodes
    x_e, w_e = map(np.asarray, european_rule(hurst, n_nodes, horizon))
    if w_bar_max is not None:
        w_e = w_e * min(1.0, w_bar_max / w_e.sum())
    z_e = np.r_[np.log(np.clip(x_e, np.exp(lo_x), x_max)), np.log(np.clip(w_e, 1e-3, 3e2))]
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds).T
    population = np.vstack([z_e, lo + (hi - lo) * rng.random((30 * n_nodes - 1, 2 * n_nodes))])
    best = differential_evolution(loss, bounds, seed=seed, maxiter=400, tol=1e-10, polish=False,
                                  init=population)
    for method in ('L-BFGS-B', 'Nelder-Mead'):
        local = minimize(loss, best.x, method=method, bounds=bounds, options=dict(maxiter=4000))
        if np.isfinite(local.fun) and local.fun < best.fun:
            best = local
    x, w = np.exp(best.x[:n_nodes]), np.exp(best.x[n_nodes:])
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or not np.isfinite(best.fun):
        raise RuntimeError('kernel fit returned nonfinite parameters or loss')
    if loss(best.x) >= 1e6:
        raise RuntimeError('kernel fit did not find a finite moment solution below the failure penalty')
    if w_bar_max is not None and w.sum() > w_bar_max * (1.0 + 1e-12):
        raise ValueError(f'kernel fit exceeds requested weight cap: sum(weights)={w.sum():.12g}, '
                         f'w_bar_max={w_bar_max:.12g}; the penalty optimizer is infeasible')
    if not best.success:
        warnings.warn(f'kernel fit optimizer did not report convergence: {best.message}',
                      RuntimeWarning, stacklevel=2)
    order = np.argsort(x)
    return x[order], w[order]


@lru_cache(maxsize=32)
def _cached_fit(hurst: float,
                n_nodes: int,
                horizon: float,
                w_bar_max: Optional[float],
                x_max: float
                ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """memoised fit for the default parameter sets and maturities scaled to the horizon."""
    maturities = tuple(horizon * f for f in MATURITY_FRACTIONS)
    x, w = fit_kernel_to_moments(hurst, n_nodes=n_nodes, maturities=maturities,
                                 w_bar_max=w_bar_max, x_max=x_max)
    return tuple(x), tuple(w)


def moment_fitted_kernel(hurst: float,
                         n_nodes: int,
                         horizon: float,
                         w_bar_max: Optional[float] = None,
                         x_max: float = 360.0
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    moment-fitted nodes and weights over [0, horizon], memoised per argument set.

    The fit maturities are ``MATURITY_FRACTIONS`` times the horizon and the parameter sets are
    ``DEFAULT_FIT_PARAMS``. The first call for an argument set costs about a minute.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Copies of the nodes and weights.
    """
    _validate_positive(horizon, 'horizon')
    _validate_fit_domain(hurst, n_nodes, (horizon,), w_bar_max, x_max)
    x, w = _cached_fit(float(hurst), int(n_nodes), float(horizon),
                       None if w_bar_max is None else float(w_bar_max), float(x_max))
    return np.array(x), np.array(w)


def kernel_mass_errors(nodes: np.ndarray,
                       weights: np.ndarray,
                       hurst: float,
                       horizon: float = 1.0
                       ) -> Tuple[float, float]:
    """
    relative errors in percent of int_0^horizon K^N and int_0^horizon (K^N)^2.

    Returns
    -------
    Tuple[float, float]
        Errors of the masses of K and K^2 against the fractional kernel.
    """
    x, w = _validate_kernel(nodes, weights)
    if not 0.0 < hurst <= 0.5:
        raise ValueError(f'hurst must be in (0, 0.5], given {hurst}')
    _validate_positive(horizon, 'horizon')
    alpha = hurst + 0.5
    safe_x = np.where(x > 0.0, x, 1.0)
    mass1 = np.sum(w * np.where(x > 0.0, -np.expm1(-x * horizon) / safe_x, horizon))
    s = x[:, None] + x[None, :]
    safe_s = np.where(s > 0.0, s, 1.0)
    mass2 = np.sum(np.outer(w, w) * np.where(s > 0.0, -np.expm1(-s * horizon) / safe_s, horizon))
    exact1 = horizon ** alpha / gamma(alpha + 1.0)
    exact2 = horizon ** (2.0 * hurst) / (2.0 * hurst * gamma(alpha) ** 2)
    return 100.0 * (mass1 / exact1 - 1.0), 100.0 * (mass2 / exact2 - 1.0)


def max_time_step(w_bar: float,
                  vartheta: float,
                  constant: float = STEP_RULE_CONSTANT
                  ) -> float:
    """
    largest split-step time step in years with w_bar vartheta sqrt(h) <= constant.

    The diffusion sub-step moves sigma^N as a geometric Brownian motion with volatility
    w_bar vartheta over the whole step, while the fast factors revert within the step. For
    larger steps the scheme compounds noise that the factor model damps and is biased.
    """
    _validate_positive(w_bar, 'w_bar')
    _validate_positive(vartheta, 'vartheta')
    _validate_positive(constant, 'constant')
    return (constant / (w_bar * vartheta)) ** 2


def max_weight_sum(vartheta: float,
                   time_step: float,
                   constant: float = STEP_RULE_CONSTANT
                   ) -> float:
    """
    largest w_bar = K^N(0) with w_bar vartheta sqrt(h) <= constant, the inverse of max_time_step.

    Pass the result as ``w_bar_max`` to fit a kernel for a given step budget.
    """
    _validate_positive(vartheta, 'vartheta')
    _validate_positive(time_step, 'time_step')
    _validate_positive(constant, 'constant')
    return constant / (vartheta * np.sqrt(time_step))
