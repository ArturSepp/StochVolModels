"""
Exhibits for the working paper "Rough Log-normal Volterra Stochastic Volatility Model with Quadratic Drift".

Model under Q (zero rates), kernel K(t) = t^(H-1/2) / Gamma(H+1/2) or its Markovian approximation
K^N(t) = sum_i w_i exp(-x_i t):
    dS_t / S_t = sigma_t dW0_t,
    sigma_t = sigma0 + int_0^t K(t-s) [(kappa1 + kappa2 sigma_s)(theta - sigma_s) ds + beta sigma_s dW0_s + eps sigma_s dW1_s],
    vartheta^2 = beta^2 + eps^2, rho = beta / vartheta, I_t = int_0^t sigma_s^2 ds (xi0 = 1).
Exact results for kappa2 = 0 checked here (paper Section 3):
    m(t) = theta + (sigma0 - theta)(1 - int_0^t R),  R = kappa1 K - kappa1 K*R,  G = R / kappa1,
    v(t) = m(t)^2 + vartheta^2 int_0^t G(t-s)^2 v(s) ds,
    kappa1 = 0:  v(t) = sigma0^2 E_{2H}(vartheta^2 Gamma(2H) t^(2H) / Gamma(H+1/2)^2).
Units: time in years, volatilities annualised. Monte Carlo cases use the split-step scheme of the rough
chain pricer, stochvolmodels.pricers.rough_logsv.split_simulation.simulate_split_step (J from the
diffusion sub-step, exponential-implicit fallback), which exists from commit 422c340 of the branch
rough-logsv-kernel-fit on. SCHEME_COMPARISON also runs the 2.4.2 scheme log_spot_full_combined.
Expectations of S_T are estimated conditionally on the volatility path, E[S_T | path] = exp(rho J_T -
rho^2 I_T / 2), and the leverage functional as cov(J_T, sigma_T) / vartheta, which removes the orthogonal
price noise from both estimates.
Entry point: run_local_test(local_test) with the LocalTests cases; each case prints the numbers of one
table or statement of the paper. The kernel-fit cases (subsection "Kernel fitted to the moments") import
stochvolmodels.models.rough_logsv.kernel_fit, which exists from the branch rough-logsv-kernel-fit on;
run them with OMP_NUM_THREADS=1, since each fit calls expm about 10^6 times.
Not in scope: kappa2 > 0 moments (no closed system), calibration, figures.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple
from scipy.integrate import solve_ivp
from scipy.special import gamma
from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

# Load the companion numerics for both direct execution and runpy.run_path.
_moment_spec = spec_from_file_location("rough_logsv_paper_moments",
    Path(__file__).with_name("rough_logsv_stationarity_check.py"))
_moments = module_from_spec(_moment_spec)
_moment_spec.loader.exec_module(_moments)
# stochvolmodels
import stochvolmodels as sv
from stochvolmodels import LogSvParams
from stochvolmodels.pricers.rough_logsv.split_simulation import log_spot_full_combined, drift_diffus_strand_f64
from stochvolmodels.utils.funcs import set_time_grid


def mittag_leffler(z: float,
                   alpha: float,
                   beta: float = 1.0,
                   dps: int = 60
                   ) -> float:
    """Mittag-Leffler value using stable SciPy numerics for the paper's cases.

    Positive arguments use an adaptive log-sum-exp. Negative arguments with
    beta=1 or alpha use a small-argument series and a positive spectral integral.
    The historical dps argument is used only by the optional general mpmath fallback.
    """
    if z >= 0:
        value = _moments.log_mittag_leffler_positive(z, alpha, beta)
        return float(np.exp(value)) if value < np.log(np.finfo(float).max) else np.inf
    if 0.5 < alpha <= 1 and beta in (1.0, alpha):
        return float(_moments.mittag_leffler_negative_vec(alpha, np.array([-z]), beta)[0])
    import mpmath as mp
    with mp.workdps(dps):
        a, b, zz = mp.mpf(alpha), mp.mpf(beta), mp.mpf(z)
        total = mp.mpf(0)
        for k in range(20000):
            term = zz ** k / mp.gamma(a * k + b)
            total += term
            if k > 30 and abs(term) < mp.mpf(10) ** (-dps + 10) * max(1, abs(total)):
                return float(total)
    raise ArithmeticError("Mittag-Leffler series did not converge")


def second_moment_growth_kappa1_zero(hurst: float,
                                     vartheta: float,
                                     ttm: float = 1.0
                                     ) -> float:
    """Finite-horizon ratio from paper eq:ml_second (may exceed float range)."""
    q = 2.0 * hurst
    a = vartheta ** 2 * gamma(q) / gamma(hurst + 0.5) ** 2
    return mittag_leffler(a * ttm ** q, alpha=q)


def second_moment_fractional(hurst: float,
                             vartheta: float,
                             kappa1: float,
                             sigma0: float,
                             theta: float,
                             ttm: float = 1.0,
                             nb_steps: int = 4000
                             ) -> float:
    """Product integration with 24-point quadrature after removing the G^2 singularity.

    The companion solve_second_moment supports log_domain=True when values
    exceed floating-point range. Refine the time grid before quoting digits.
    """
    return float(_moments.solve_second_moment(hurst, kappa1, vartheta, theta, sigma0,
                                              T=ttm, n_steps=nb_steps)[1][-1])


def markov_lift_moments(nodes: np.ndarray,
                        weights: np.ndarray,
                        sigma0: float,
                        theta: float,
                        kappa1: float,
                        vartheta: float,
                        ttm: float = 1.0,
                        u0: np.ndarray = None  # factor state U(0); None means zero
                        ) -> Tuple[float, float, float]:
    """
    exact E[sigma_T], E[sigma_T^2] and E[I_T] for K(t) = sum_i w_i exp(-x_i t) and kappa2 = 0.

    U_i solves dU_i = (-x_i U_i + kappa1 (theta - sigma)) dt + vartheta sigma dW with sigma = sigma0 + w'U;
    the first and second moments of U solve a closed linear system of ordinary differential equations.
    """
    if nodes.shape != weights.shape:
        raise ValueError(f"nodes and weights must have the same shape, got {nodes.shape} and {weights.shape}")
    n = nodes.size
    ones = np.ones(n)
    b_mat = np.diag(nodes) + kappa1 * np.outer(ones, weights)
    a_vec = kappa1 * (theta - sigma0) * ones
    u_init = np.zeros(n) if u0 is None else u0

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        mu = y[:n]
        p_mat = y[n:n + n * n].reshape(n, n)
        e_sigma2 = sigma0 ** 2 + 2.0 * sigma0 * weights @ mu + weights @ p_mat @ weights
        d_p = (np.outer(a_vec, mu) + np.outer(mu, a_vec) - b_mat @ p_mat - p_mat @ b_mat.T
               + vartheta ** 2 * e_sigma2 * np.outer(ones, ones))
        return np.concatenate([a_vec - b_mat @ mu, d_p.ravel(), [e_sigma2]])

    y0 = np.concatenate([u_init, np.outer(u_init, u_init).ravel(), [0.0]])
    sol = solve_ivp(rhs, (0.0, ttm), y0, method='Radau', rtol=1e-11, atol=1e-13)
    mu = sol.y[:n, -1]
    p_mat = sol.y[n:n + n * n, -1].reshape(n, n)
    e_sigma2 = sigma0 ** 2 + 2.0 * sigma0 * weights @ mu + weights @ p_mat @ weights
    return sigma0 + weights @ mu, e_sigma2, sol.y[-1, -1]


def resolvent_moments_on_grid(nodes: np.ndarray,
                              weights: np.ndarray,
                              psi: np.ndarray,  # psi(t) = Theta_t - theta on the grid
                              theta: float,
                              kappa1: float,
                              vartheta: float,
                              dt: float
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """m = theta + psi - R*psi and v = m^2 + vartheta^2 G^2 * v on a uniform grid by the trapezoidal rule."""
    nb = psi.size - 1
    t = np.arange(nb + 1) * dt
    k_t = np.exp(-np.outer(t, nodes)) @ weights
    r = np.zeros(nb + 1)
    r[0] = kappa1 * k_t[0]
    for i in range(1, nb + 1):
        conv = dt * (0.5 * k_t[i] * r[0] + np.dot(k_t[i - 1:0:-1], r[1:i]))
        r[i] = (kappa1 * k_t[i] - kappa1 * conv) / (1.0 + 0.5 * kappa1 * dt * k_t[0])
    m = np.empty(nb + 1)
    m[0] = theta + psi[0]
    for i in range(1, nb + 1):
        vals = r[i::-1] * psi[:i + 1]
        m[i] = theta + psi[i] - dt * (vals.sum() - 0.5 * (vals[0] + vals[-1]))
    g2 = (r / kappa1) ** 2
    v = np.zeros(nb + 1)
    v[0] = m[0] ** 2
    for i in range(1, nb + 1):
        conv = dt * (0.5 * g2[i] * v[0] + np.dot(g2[i - 1:0:-1], v[1:i]))
        v[i] = (m[i] ** 2 + vartheta ** 2 * conv) / (1.0 - 0.5 * vartheta ** 2 * dt * g2[0])
    return m, v


def frozen_curve_expected_qvar(nodes: np.ndarray,
                               weights: np.ndarray,
                               sigma0: float,
                               theta: float,
                               kappa1: float,
                               vartheta: float,
                               ttm: float = 1.0,
                               nb_steps: int = 4000
                               ) -> float:
    """E[I_T] from the frozen-curve system (paper eq. g_system) with kappa2 = 0, xi0 = 1, K(s-t) direction."""
    dt = ttm / nb_steps
    t = np.arange(nb_steps + 1) * dt
    k_lag = np.exp(-np.outer(t, nodes)) @ weights
    k2_lag = k_lag ** 2
    g2, g1, g0 = np.zeros(nb_steps + 1), np.zeros(nb_steps + 1), np.zeros(nb_steps + 1)
    for i in range(nb_steps, -1, -1):
        j = np.arange(i + 1, nb_steps + 1)
        w = np.full(j.size, dt)
        if j.size:
            w[-1] = 0.5 * dt
        c_k = np.dot(w, g2[j] * k_lag[j - i])
        c_k2 = np.dot(w, g2[j] * k2_lag[j - i])
        g2[i] = ((-1.0 - 2.0 * kappa1 * c_k + vartheta ** 2 * c_k2)
                 / (1.0 + kappa1 * dt * k_lag[0] - 0.5 * vartheta ** 2 * dt * k2_lag[0]))
        c_k2_full = c_k2 + 0.5 * dt * g2[i] * k2_lag[0]
        c1_k = np.dot(w, g1[j] * k_lag[j - i])
        g1[i] = (-2.0 * theta - kappa1 * c1_k + 2.0 * theta * vartheta ** 2 * c_k2_full) / (1.0 + 0.5 * kappa1 * dt * k_lag[0])
        g0[i] = -theta ** 2 + theta ** 2 * vartheta ** 2 * c_k2_full
    y0 = sigma0 - theta
    return -np.trapezoid(g0 + g1 * y0 + g2 * y0 ** 2, t)


def simulate_split_scheme(hurst: float,
                          sigma0: float,
                          theta: float,
                          kappa1: float,
                          kappa2: float,
                          beta: float,
                          eps: float,
                          ttm: float = 1.0,
                          nb_path: int = 200000,
                          nb_steps_per_year: int = 360,
                          seed: int = 3
                          ) -> dict:
    """terminal sigma^N, I_T, J_T and the conditional and plain S_T from simulate_split_step (code kernel rule)."""
    from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
    params = LogSvParams(sigma0=sigma0, theta=theta, kappa1=kappa1, kappa2=kappa2, beta=beta, volvol=eps, H=hurst)
    params.approximate_kernel(T=ttm)
    nodes, weights = params.nodes, params.weights
    vartheta = np.hypot(beta, eps)
    rho = beta / vartheta
    nb_steps, _dt, grid = set_time_grid(ttm, nb_steps_per_year)
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal((nb_steps, nb_path))
    z1 = rng.standard_normal((nb_steps, nb_path))
    sigma0_factors = np.repeat(np.full(nodes.size, sigma0 / np.sum(weights))[:, None], nb_path, axis=1)
    paths = simulate_split_step(np.repeat(nodes[:, None], nb_path, axis=1), np.repeat(weights[:, None], nb_path, axis=1),
                                sigma0_factors, theta, kappa1, kappa2, 0.0, sigma0_factors.copy(), rho, vartheta, grid,
                                nb_path, z0, z1)
    return dict(sigma=weights @ paths.factors, qvar=paths.qvar, jump=paths.jump,
                spot_cond=np.exp(rho * paths.jump - 0.5 * rho ** 2 * paths.qvar), spot=np.exp(paths.log_spot),
                nodes=nodes, weights=weights, n_fallback=paths.n_fallback, n_floor=paths.n_floor)


KERNEL_TEST_PARAMS = ((0.5, 0.4, 2.0, 0.72), (0.3, 0.3, 0.5, 0.5), (0.8, 0.8, 6.0, 0.9), (0.6, 0.9, 3.0, 0.6))
"""out-of-sample parameter sets (sigma0, theta, kappa1, vartheta) of the kernel-fit table."""


def fitted_kernels(hurst: float) -> Tuple[dict, list, list]:
    """quadrature rule and the two moment fits (w_bar fixed at the rule, w_bar free) with their targets."""
    from stochvolmodels.models.rough_logsv.kernel_fit import (DEFAULT_FIT_PARAMS, fit_kernel_to_moments,
                                                               fractional_moment_targets)
    from stochvolmodels.models.rough_logsv.rough_kernel import european_rule
    fit_targets = [fractional_moment_targets(hurst, *p) for p in DEFAULT_FIT_PARAMS]
    test_targets = [fractional_moment_targets(hurst, *p) for p in KERNEL_TEST_PARAMS]
    x_e, w_e = map(np.asarray, european_rule(hurst, 3, 1.0))
    kernels = {'quadrature rule': (x_e, w_e),
               'fit, w_bar fixed': fit_kernel_to_moments(hurst, w_bar_max=w_e.sum(), targets=fit_targets),
               'fit, w_bar free': fit_kernel_to_moments(hurst, targets=fit_targets)}
    return kernels, fit_targets, test_targets


def kernel_errors(nodes: np.ndarray,
                  weights: np.ndarray,
                  params: tuple,
                  targets: list
                  ) -> pd.DataFrame:
    """level error (bp of vol), leverage error (%) and mean error (bp) of K^N, one row per (params, maturity)."""
    from stochvolmodels.models.rough_logsv.kernel_fit import MATURITY_FRACTIONS, MomentKey, markov_moment_targets
    rows = []
    for p, target in zip(params, targets):
        got = markov_moment_targets(nodes, weights, *p)
        for j, maturity in enumerate(MATURITY_FRACTIONS):
            rows.append(dict(params=str(p), maturity=maturity,
                             level_bp=1e4 * (got[MomentKey.LEVEL][j] - target[MomentKey.LEVEL][j]),
                             leverage_pct=100.0 * (got[MomentKey.LEVERAGE][j] / target[MomentKey.LEVERAGE][j] - 1.0),
                             mean_bp=1e4 * (got[MomentKey.MEAN][j] - target[MomentKey.MEAN][j])))
    return pd.DataFrame(rows)


def kernel_mc_moments(nodes: np.ndarray,
                      weights: np.ndarray,
                      sigma0: float,
                      theta: float,
                      kappa1: float,
                      vartheta: float,
                      rho: float,
                      ttm: float,
                      n_steps: int,
                      n_paths: int = 100000,
                      n_batches: int = 5,
                      seed: int = 11
                      ) -> dict:
    """variance-swap volatility, leverage cov(J_T, sigma_T) / vartheta and E[S_T | path], kappa2 = 0."""
    from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
    nb = n_paths // n_batches
    node_mat = np.repeat(nodes[:, None], nb, axis=1)
    weight_mat = np.repeat(weights[:, None], nb, axis=1)
    sigma0_factors = np.full_like(node_mat, sigma0 / weights.sum())
    grid = np.linspace(0.0, ttm, n_steps + 1)
    rng = np.random.default_rng(seed)
    jump, qvar, sigma_t = [], [], []
    for _ in range(n_batches):
        z0 = rng.standard_normal((n_steps, nb))
        z1 = rng.standard_normal((n_steps, nb))
        paths = simulate_split_step(node_mat, weight_mat, sigma0_factors, theta, kappa1, 0.0, 0.0,
                                    sigma0_factors.copy(), rho, vartheta, grid, nb, z0, z1)
        jump.append(paths.jump)
        qvar.append(paths.qvar)
        sigma_t.append(np.sum(weight_mat * paths.factors, axis=0))
    jump, qvar, sigma_t = np.concatenate(jump), np.concatenate(qvar), np.concatenate(sigma_t)
    level = np.sqrt(np.mean(qvar) / ttm)
    cross = (jump - jump.mean()) * (sigma_t - sigma_t.mean())
    spot = np.exp(rho * jump - 0.5 * rho ** 2 * qvar)
    return dict(level=level, level_se=np.std(qvar / ttm) / np.sqrt(n_paths) / (2.0 * level),
                leverage=np.mean(cross) / vartheta,
                leverage_se=np.std(cross) / np.sqrt(n_paths) / vartheta,
                spot_mean=spot.mean(), spot_mean_se=spot.std() / np.sqrt(n_paths))


class LocalTests(Enum):
    GROWTH_TABLE_KAPPA1_ZERO = 1
    GROWTH_TABLE_WITH_MEAN_REVERSION = 2
    LINEAR_DRIFT_IDENTITY_CHECK = 3
    FRACTIONAL_MEAN_CHECK = 4
    DRAFT_KERNEL_DIRECTION = 5
    FROZEN_CURVE_ACCURACY = 6
    VIX_QUADRATIC_FORM_CHECK = 7
    SPLIT_SCHEME_VALIDATION = 8
    MARTINGALE_SWEEP = 9
    REGRESSION_SET_DIAGNOSTICS = 10
    KERNEL_FIT_TABLE = 11
    KERNEL_ERROR_BY_MATURITY = 12
    KERNEL_FIT_MC_CHECK = 13
    SCHEME_COMPARISON = 14
    CONDITIONAL_MC_VARIANCE = 15
    GROWTH_TABLE_LOG_DOMAIN = 16


def run_local_test(local_test: LocalTests) -> None:
    """print the numbers of one exhibit of the working paper."""
    four_nodes = np.array([0.3, 3.0, 30.0, 300.0])
    four_weights = np.array([0.6, 0.8, 1.0, 1.2])

    if local_test == LocalTests.GROWTH_TABLE_KAPPA1_ZERO:
        print("Positive Mittag-Leffler series with adaptive tail bounds")
        for h in [0.5, 0.3, 0.2, 0.1]:
            for vt in [0.5, 1.0, 1.85]:
                q = 2 * h
                a = vt ** 2 * gamma(q) / gamma(h + 0.5) ** 2
                log10 = _moments.log_mittag_leffler_positive(a, q) / np.log(10)
                asymptotic = (a ** (1 / q) - np.log(q)) / np.log(10)
                value = f"{10 ** log10:.9g}" if log10 < 300 else f"10^({log10:.9f})"
                print(f"H={h:.1f} vartheta={vt:.2f}: ratio={value}, log10={log10:.9f}, "
                      f"asymptotic log10={asymptotic:.9f}")

    elif local_test == LocalTests.GROWTH_TABLE_WITH_MEAN_REVERSION:
        _moments.run_local_test(_moments.LocalTests.TABLE2_CHECK)

    elif local_test == LocalTests.GROWTH_TABLE_LOG_DOMAIN:
        _moments.run_local_test(_moments.LocalTests.GROWTH_TABLE_LOG_DOMAIN)

    elif local_test == LocalTests.LINEAR_DRIFT_IDENTITY_CHECK:
        params = dict(sigma0=0.6, theta=0.4, kappa1=2.0, vartheta=0.9)
        m_ode, v_ode, _ = markov_lift_moments(four_nodes, four_weights, **params)
        nb = 4000
        m_vol, v_vol = resolvent_moments_on_grid(four_nodes, four_weights, psi=np.full(nb + 1, 0.6 - 0.4),
                                                 theta=0.4, kappa1=2.0, vartheta=0.9, dt=1.0 / nb)
        print(f"E[sigma_1]:   moment ODE {m_ode:.6f}  Volterra {m_vol[-1]:.6f}")
        print(f"E[sigma_1^2]: moment ODE {v_ode:.6f}  Volterra {v_vol[-1]:.6f}")

    elif local_test == LocalTests.FRACTIONAL_MEAN_CHECK:
        hurst, kappa1, sigma0, theta, nb = 0.2, 3.0, 0.8, 0.5, 4000
        alpha = hurst + 0.5
        t = np.linspace(0.0, 1.0, nb + 1)
        m = np.zeros(nb + 1)
        m[0] = sigma0
        for i in range(1, nb + 1):
            j = np.arange(i)
            cells = ((t[i] - t[j]) ** alpha - (t[i] - t[j + 1]) ** alpha) / gamma(alpha + 1.0)
            m[i] = sigma0 + kappa1 * np.dot(cells, theta - m[j])
        exact = theta + (sigma0 - theta) * mittag_leffler(-kappa1, alpha=alpha)
        print(f"E[sigma_1]: Volterra grid {m[-1]:.6f}  Mittag-Leffler {exact:.6f}")

    elif local_test == LocalTests.DRAFT_KERNEL_DIRECTION:
        for hurst, vt in [(0.375, 0.75), (0.2, 1.0), (0.1, 1.0)]:
            alpha, beta = hurst + 0.5, 2.0 * hurst
            c2 = 1.0 / gamma(alpha) ** 2
            u = np.linspace(0.0, 1.0, 2001)
            exact = np.trapezoid([mittag_leffler(vt ** 2 * c2 * gamma(beta) * x ** beta, alpha=beta) for x in u], u)
            draft = np.trapezoid(np.exp(vt ** 2 * c2 * u ** beta / beta), u)
            print(f"H={hurst} vartheta={vt}: E[I_1] exact {exact:.6g}  2024 closed-form example {draft:.6g}")

    elif local_test == LocalTests.FROZEN_CURVE_ACCURACY:
        cases = [(np.array([0.0]), np.array([1.0]), 2.0, 0.9, 0.6),
                 (np.array([2.0]), np.array([1.5]), 2.0, 0.9, 0.6),
                 (four_nodes, four_weights, 0.0, 0.9, 0.6),
                 (four_nodes, four_weights, 2.0, 0.9, 0.2),
                 (four_nodes, four_weights, 2.0, 0.9, 0.4),
                 (four_nodes, four_weights, 2.0, 0.9, 0.6),
                 (four_nodes, four_weights, 5.0, 1.5, 0.6)]
        for nodes, weights, kappa1, vt, sigma0 in cases:
            exact = markov_lift_moments(nodes, weights, sigma0, 0.4, kappa1, vt)[2]
            frozen = frozen_curve_expected_qvar(nodes, weights, sigma0, 0.4, kappa1, vt)
            print(f"N={nodes.size} kappa1={kappa1} vartheta={vt} sigma0={sigma0}: exact {exact:.5f} "
                  f"frozen {frozen:.5f} rel.err {(frozen - exact) / exact:+.3%}")

    elif local_test == LocalTests.VIX_QUADRATIC_FORM_CHECK:
        sigma0, theta, kappa1, vt, delta = 0.6, 0.4, 2.0, 0.9, 30.0 / 365.0
        u = np.array([0.05, -0.1, 0.2, -0.05])
        vix2_ode = markov_lift_moments(four_nodes, four_weights, sigma0, theta, kappa1, vt, ttm=delta, u0=u)[2] / delta
        nb = 6000
        dt = delta / nb
        t = np.arange(nb + 1) * dt
        psi = sigma0 + np.exp(-np.outer(t, four_nodes)) @ (four_weights * u) - theta
        _m, v = resolvent_moments_on_grid(four_nodes, four_weights, psi=psi, theta=theta, kappa1=kappa1, vartheta=vt, dt=dt)
        print(f"VIX^2 (30d): moment ODE {vix2_ode:.8f}  Proposition 3.4 {np.trapezoid(v, t) / delta:.8f}")

    elif local_test == LocalTests.SPLIT_SCHEME_VALIDATION:
        se = lambda x: x.std() / np.sqrt(x.size)
        for label, hurst, beta, eps in [("(a)", 0.3, -0.4, 0.6), ("(b)", 0.1, -0.4, 0.6), ("(c)", 0.1, 0.6, 0.8)]:
            r = simulate_split_scheme(hurst, 0.5, 0.4, 2.0, 0.0, beta, eps)
            m, v, e_i = markov_lift_moments(r['nodes'], r['weights'], 0.5, 0.4, 2.0, np.hypot(beta, eps))
            sig, qvar, spot_c, spot = r['sigma'], r['qvar'], r['spot_cond'], r['spot']
            print(f"{label} H={hurst}: E[sigma] {sig.mean():.5f} ({se(sig):.5f}) exact {m:.5f} | "
                  f"E[sigma^2] {np.mean(sig ** 2):.5f} ({se(sig ** 2):.5f}) exact {v:.5f} | "
                  f"E[I] {qvar.mean():.5f} ({se(qvar):.5f}) exact {e_i:.5f} | "
                  f"E[S] conditional {spot_c.mean():.5f} ({se(spot_c):.5f}), plain {spot.mean():.5f} ({se(spot):.5f}) | "
                  f"fallback {r['n_fallback']} floors {r['n_floor']}")

    elif local_test == LocalTests.MARTINGALE_SWEEP:
        se = lambda x: x.std() / np.sqrt(x.size)
        for kappa2 in [0.0, 0.3, 0.6, 1.2]:
            r = simulate_split_scheme(0.1, 0.5, 0.4, 2.0, kappa2, 0.6, 0.8)
            print(f"kappa2={kappa2}: E[S_1] conditional {r['spot_cond'].mean():.4f} ({se(r['spot_cond']):.4f}), "
                  f"plain {r['spot'].mean():.4f} ({se(r['spot']):.4f}) | fallback {r['n_fallback']} floors {r['n_floor']}")

    elif local_test == LocalTests.REGRESSION_SET_DIAGNOSTICS:
        chain = sv.get_btc_test_chain_data()
        z0, _z1, grids = sv.get_randoms_for_rough_vol_chain_valuation(ttms=chain.ttms, nb_path=10000,
                                                                      nb_steps_per_year=360, seed=10)
        for kappa2 in [1.93, 0.0]:
            params = LogSvParams(sigma0=0.377, theta=0.347, kappa1=1.29, kappa2=kappa2, beta=2.45, volvol=1.81)
            params.H = 0.1
            params.approximate_kernel(T=chain.ttms[-1])
            nodes, weights = params.nodes, params.weights
            nb_path = z0.shape[1]
            vartheta = np.hypot(params.beta, params.volvol)
            factors0 = np.repeat(np.full(nodes.size, params.sigma0 / weights.sum())[:, None], nb_path, axis=1)
            w_mat = np.repeat(weights[:, None], nb_path, axis=1)
            x_mat = np.repeat(nodes[:, None], nb_path, axis=1)
            grid = grids[-1]
            h = grid[1] - grid[0]
            vol = factors0.copy()
            floored = np.zeros(nb_path, dtype=bool)
            with np.errstate(all='ignore'):
                for k in range(grid.size - 1):
                    vol = drift_diffus_strand_f64(x_mat, factors0, params.theta, params.kappa1, params.kappa2,
                                                  vartheta, vol, w_mat, h, nb_path, z0[k])
                    sig = (w_mat * vol).sum(axis=0)
                    bad = np.isnan(sig) | (sig <= 0.0)
                    floored |= bad
                    vol[:, bad] = 1e-6
            sig = (w_mat * vol).sum(axis=0)
            print(f"kappa2={kappa2}: w_bar={weights.sum():.2f}, w_bar*vartheta*sqrt(h)={weights.sum() * vartheta * np.sqrt(h):.2f}")
            print(f"   2.4.2 scheme: floored paths {floored.mean():.2%}, sample mean sigma_T {sig.mean():.3f}, "
                  f"median {np.median(sig):.3f}")
            from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
            paths = simulate_split_step(x_mat, w_mat, factors0, params.theta, params.kappa1, params.kappa2, 0.0,
                                        factors0.copy(), params.beta / vartheta, vartheta, grid, nb_path,
                                        z0[:grid.size - 1], _z1[:grid.size - 1])
            sig_new = weights @ paths.factors
            print(f"   pricer scheme: fallback path-steps {paths.n_fallback}, floored path-steps {paths.n_floor}, "
                  f"sample mean sigma_T {sig_new.mean():.3f}, median {np.median(sig_new):.3f}")
            if kappa2 == 0.0:
                m, v, _ = markov_lift_moments(nodes, weights, params.sigma0, params.theta, params.kappa1, vartheta,
                                              ttm=chain.ttms[-1])
                print(f"   exact E[sigma_T] {m:.3f}, E[sigma_T^2] {v:.3g}")


    elif local_test == LocalTests.KERNEL_FIT_TABLE:
        from stochvolmodels.models.rough_logsv.kernel_fit import DEFAULT_FIT_PARAMS, kernel_mass_errors, max_time_step
        rows = []
        for hurst in (0.1, 0.2, 0.3):
            kernels, fit_targets, test_targets = fitted_kernels(hurst)
            for name, (x, w) in kernels.items():
                err_fit = kernel_errors(x, w, DEFAULT_FIT_PARAMS, fit_targets)
                err_test = kernel_errors(x, w, KERNEL_TEST_PARAMS, test_targets)
                mass_k, mass_k2 = kernel_mass_errors(x, w, hurst)
                rows.append(dict(H=hurst, kernel=name, x_N=x[-1], w_bar=w.sum(),
                                 steps_per_year=1.0 / max_time_step(w.sum(), 1.0),
                                 mass_k_pct=mass_k, mass_k2_pct=mass_k2,
                                 fit_level_bp=err_fit['level_bp'].abs().max(),
                                 fit_leverage_pct=err_fit['leverage_pct'].abs().max(),
                                 test_level_bp=err_test['level_bp'].abs().max(),
                                 test_leverage_pct=err_test['leverage_pct'].abs().max(),
                                 test_mean_bp=err_test['mean_bp'].abs().max()))
                print(f"H={hurst} {name}: nodes {np.round(x, 3)} weights {np.round(w, 3)}", flush=True)
        print(pd.DataFrame(rows).round(1).to_string(index=False))

    elif local_test == LocalTests.KERNEL_ERROR_BY_MATURITY:
        from stochvolmodels.models.rough_logsv.kernel_fit import fractional_moment_targets
        from stochvolmodels.models.rough_logsv.rough_kernel import european_rule
        for hurst in (0.1, 0.2, 0.3):
            x_e, w_e = map(np.asarray, european_rule(hurst, 3, 1.0))
            test_targets = [fractional_moment_targets(hurst, *p) for p in KERNEL_TEST_PARAMS]
            errors = kernel_errors(x_e, w_e, KERNEL_TEST_PARAMS, test_targets)
            print(f"H={hurst}, quadrature rule: level error (bp) and leverage error (%) by maturity")
            print(errors.pivot(index='params', columns='maturity', values='level_bp').round(0).to_string())
            print(errors.pivot(index='params', columns='maturity', values='leverage_pct').round(1).to_string())

    elif local_test == LocalTests.KERNEL_FIT_MC_CHECK:
        from stochvolmodels.models.rough_logsv.kernel_fit import (MomentKey, fractional_moment_targets,
                                                                   markov_moment_targets, max_time_step)
        sigma0, theta, kappa1, vartheta = KERNEL_TEST_PARAMS[0]
        rho, ttm = -0.7, 0.25
        for hurst in (0.1, 0.2):
            kernels, _, _ = fitted_kernels(hurst)
            exact = fractional_moment_targets(hurst, *KERNEL_TEST_PARAMS[0], maturities=(ttm,))
            for name, (x, w) in kernels.items():
                markov = markov_moment_targets(x, w, *KERNEL_TEST_PARAMS[0], maturities=(ttm,))
                n_rule = int(np.ceil(ttm / max_time_step(w.sum(), vartheta)))
                for n_steps in sorted({63, n_rule}):
                    mc = kernel_mc_moments(x, w, sigma0, theta, kappa1, vartheta, rho, ttm, n_steps=n_steps)
                    print(f"H={hurst} {name:16s} w_bar={w.sum():5.2f} steps/yr={n_steps / ttm:5.0f} "
                          f"w_bar*vartheta*sqrt(h)={w.sum() * vartheta * np.sqrt(ttm / n_steps):.2f} | "
                          f"level MC-Markov {1e4 * (mc['level'] - markov[MomentKey.LEVEL][0]):+5.0f}"
                          f"+-{1e4 * mc['level_se']:.0f}bp, "
                          f"Markov-exact {1e4 * (markov[MomentKey.LEVEL][0] - exact[MomentKey.LEVEL][0]):+6.0f}bp | "
                          f"leverage MC/Markov-1 {100 * (mc['leverage'] / markov[MomentKey.LEVERAGE][0] - 1):+5.1f}"
                          f"+-{100 * mc['leverage_se'] / markov[MomentKey.LEVERAGE][0]:.1f}%, "
                          f"Markov/exact-1 "
                          f"{100 * (markov[MomentKey.LEVERAGE][0] / exact[MomentKey.LEVERAGE][0] - 1):+6.1f}% | "
                          f"E[S]-1 {1e4 * (mc['spot_mean'] - 1):+4.0f}+-{1e4 * mc['spot_mean_se']:.0f}bp", flush=True)


    elif local_test == LocalTests.SCHEME_COMPARISON:
        from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
        from scipy.optimize import brentq
        from scipy.stats import norm

        def implied_vols(spots: np.ndarray, strikes: np.ndarray, ttm: float) -> np.ndarray:
            spots = spots / spots.mean()
            out = []
            for k in strikes:
                price = np.mean(np.maximum(spots - k, 0.0)) if k >= 1.0 else np.mean(np.maximum(k - spots, 0.0)) + 1.0 - k
                call = lambda vol: norm.cdf(-np.log(k) / (vol * np.sqrt(ttm)) + 0.5 * vol * np.sqrt(ttm)) \
                    - k * norm.cdf(-np.log(k) / (vol * np.sqrt(ttm)) - 0.5 * vol * np.sqrt(ttm)) - price
                out.append(brentq(call, 1e-4, 10.0))
            return np.array(out)

        cases = {'moderate (rho<0)': (0.4, 0.4, 2.0, 2.0, -0.4, 0.6, np.array([0.8, 0.9, 1.0, 1.1, 1.25])),
                 'crypto-like (rho>0)': (0.8, 0.8, 3.0, 3.0, 0.6, 1.2, np.array([0.7, 0.85, 1.0, 1.15, 1.4]))}
        ttm, nb, n_fine = 0.25, 50000, 2880
        rng = np.random.default_rng(11)
        z0f = rng.standard_normal((int(ttm * n_fine), nb))
        z1f = rng.standard_normal((int(ttm * n_fine), nb))
        aggregate = lambda z, m: z.reshape(z.shape[0] // m, m, z.shape[1]).sum(axis=1) / np.sqrt(m)
        for name, (sigma0, theta, kappa1, kappa2, beta, eps, strikes) in cases.items():
            params = LogSvParams(sigma0=sigma0, theta=theta, kappa1=kappa1, kappa2=kappa2, beta=beta, volvol=eps, H=0.1)
            params.approximate_kernel(T=ttm)
            vartheta, rho = np.hypot(beta, eps), beta / np.hypot(beta, eps)
            x_mat = np.repeat(params.nodes[:, None], nb, axis=1)
            w_mat = np.repeat(params.weights[:, None], nb, axis=1)
            f0 = np.full_like(x_mat, sigma0 / params.weights.sum())

            def run(z0, z1, legacy):
                grid = np.linspace(0.0, ttm, z0.shape[0] + 1)
                if legacy:
                    ls, vol, _qv = log_spot_full_combined(x_mat, w_mat, f0, theta, kappa1, kappa2, 0.0, f0.copy(), rho,
                                                          vartheta, grid, nb, z0, z1)
                    return np.exp(ls[0]), 0, 0
                paths = simulate_split_step(x_mat, w_mat, f0, theta, kappa1, kappa2, 0.0, f0.copy(), rho, vartheta, grid,
                                            nb, z0, z1)
                return np.exp(paths.log_spot), paths.n_floor, paths.n_fallback

            ref_spots, _, _ = run(z0f, z1f, legacy=False)
            iv_ref = implied_vols(ref_spots, strikes, ttm)
            print(f"== {name}: reference ({n_fine} steps/yr) implied vols {np.round(100 * iv_ref, 2)}")
            for steps in (180, 360, 720):
                m = n_fine // steps
                z0c, z1c = aggregate(z0f, m), aggregate(z1f, m)
                ratio = params.weights.sum() * vartheta * np.sqrt(1.0 / steps)
                row = []
                for legacy in (True, False):
                    spots, n_floor, n_fallback = run(z0c, z1c, legacy)
                    err = np.max(np.abs(implied_vols(spots, strikes, ttm) - iv_ref)) * 1e4
                    if legacy:
                        row.append(f"2.4.2: max IV error {err:5.0f} bp, E[S] {spots.mean():.4f}")
                    else:
                        row.append(f"pricer: max IV error {err:5.0f} bp, E[S] {spots.mean():.4f}, fallback path-steps "
                                   f"{n_fallback}, floored path-steps {n_floor}")
                print(f"   {steps} steps/yr, w_bar*vartheta*sqrt(h)={ratio:.2f} | " + " | ".join(row))

    elif local_test == LocalTests.CONDITIONAL_MC_VARIANCE:
        from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
        from scipy.stats import norm
        strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        sigma0, theta, kappa1, vartheta = 0.5, 0.4, 2.0, 0.72
        for hurst in (0.1, 0.2):
            params = LogSvParams(H=hurst)
            params.approximate_kernel(T=1.0)
            x, w = params.nodes, params.weights
            for rho in (-0.7, 0.3):
                for ttm in (1.0 / 12.0, 0.5):
                    n, nb = int(round(ttm * 252)), 100000
                    rng = np.random.default_rng(5)
                    z0 = rng.standard_normal((n, nb))
                    z1 = rng.standard_normal((n, nb))
                    x_mat = np.repeat(x[:, None], nb, axis=1)
                    w_mat = np.repeat(w[:, None], nb, axis=1)
                    f0 = np.full_like(x_mat, sigma0 / w.sum())
                    paths = simulate_split_step(x_mat, w_mat, f0, theta, kappa1, 0.0, 0.0, f0.copy(), rho, vartheta,
                                                np.linspace(0.0, ttm, n + 1), nb, z0, z1)
                    spots = np.exp(paths.log_spot)
                    fwd = np.exp(rho * paths.jump - 0.5 * rho ** 2 * paths.qvar)
                    sd = np.sqrt((1.0 - rho ** 2) * paths.qvar)
                    ratios = []
                    for k in strikes:
                        d1 = np.log(fwd / k) / sd + 0.5 * sd
                        call = fwd * norm.cdf(d1) - k * norm.cdf(d1 - sd)
                        plain = np.maximum(spots - k, 0.0) if k >= 1.0 else np.maximum(k - spots, 0.0)
                        cond = call if k >= 1.0 else call - (fwd - k)
                        ratios.append(plain.var() / cond.var())
                    print(f"H={hurst} rho={rho:+.1f} T={ttm:.3f}: variance ratio plain / conditional by strike "
                          f"{np.round(ratios, 1)}")


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SPLIT_SCHEME_VALIDATION)
