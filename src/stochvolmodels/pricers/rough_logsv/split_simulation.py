"""
Split-step Monte Carlo scheme for the rough log-normal Volterra stochastic volatility model.

Model under Q with zero rates (Sepp and Rakhmonov, rough log-normal Volterra working paper,
October 2026):

    dS_t / S_t = sigma_t dW0_t,
    sigma_t = sigma0 + int_0^t K(t-s) [(kappa1 + kappa2 sigma_s)(theta - sigma_s) ds
                                       + vartheta sigma_s dW^sigma_s],

with vartheta^2 = beta^2 + epsilon^2, rho = beta / vartheta = corr(dW0, dW^sigma) and the
Markovian kernel K^N(t) = sum_i w_i exp(-x_i t) (nodes x_i, weights w_i). With factor targets
sigma0_factors_i = sigma0 / sum_i w_i the volatility is sigma^N_t = sum_i w_i sigma^(i)_t, where

    d sigma^(i)_t = -x_i (sigma^(i)_t - sigma0_factors_i) dt
                    + (kappa1 + kappa2 sigma^N_t)(theta - sigma^N_t) dt
                    + vartheta sigma^N_t dW^sigma_t.

One step of size h is the Strang splitting D(h/2) o S(h) o D(h/2):
    D  drift ODE, one classical Runge-Kutta 4 step (drift_ode_solve2), full quadratic drift;
    S  diffusion, exact: sigma^N is a geometric Brownian motion with volatility
       vartheta * sum_i w_i, and every factor receives the same increment (diffus_sde_solve_f64).
The price step recovers J = int sigma^N dW^sigma from the volatility increment with the
trapezoidal rule and sets, with Vbar = 0.5 (sigma^N_t^2 + sigma^N_{t+h}^2),

    log S_{t+h} = log S_t - 0.5 Vbar h + rho J + sqrt(1 - rho^2) sqrt(Vbar h) Z1,
    qvar_{t+h} = qvar_t + Vbar h.

Paths whose sigma^N is non-positive or NaN after the volatility step have all factors reset to
1e-6, because the Runge-Kutta drift step does not preserve positivity when N > 1.
Array conventions: factor arrays have shape (N, nb_path); nodes and weights are broadcast to
(N, nb_path); Z0 (drives W^sigma) and Z1 (orthogonal price noise) have shape (nb_steps, nb_path).
Entry point: log_spot_full_combined. Option payoffs and chain pricing belong in
stochvolmodels.pricers.logsv_pricer.
"""
import numpy as np
from numba import njit

from stochvolmodels.pricers.rough_logsv.expm import batch_expA, batch_invA


@njit(cache=False, fastmath=True)
def drift_ode_solve(nodes: np.ndarray, sigma0_factors: np.ndarray, theta: float, kappa1: float, kappa2: float,
                    z0: np.ndarray, weights: np.ndarray, h: float):
    """
    drift map D_h by one explicit midpoint (Runge-Kutta 2) step. Not used by the default scheme.

    Parameters
    ----------
    nodes : exponents x_i, array of size (n, nb_path)
    sigma0_factors : factor targets sigma0_factors_i, array of size (n, nb_path)
    theta : long-run level, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    z0 : factor values at the start of the step, array of size (n, nb_path)
    weights : kernel weights w_i, array of size (n, nb_path)
    h : step size

    Returns
    -------
    Array of size (n, nb_path)
    """
    assert nodes.shape == sigma0_factors.shape == z0.shape == weights.shape
    n = z0.shape[0]
    z0w = np.sum(weights * z0, axis=0)
    g0 = (kappa1 + kappa2*z0w)*(theta-z0w)
    k1 = -nodes * (z0 - sigma0_factors)
    for j in range(n):
        k1[j] += g0
    k1 *= 0.5 * h

    zmid = z0 + k1
    zmidw = np.sum(weights * zmid, axis=0)
    gmid = (kappa1 + kappa2 * zmidw) * (theta - zmidw)
    k2 = -nodes * (zmid - sigma0_factors)
    for j in range(n):
        k2[j] += gmid
    k2 *= h
    Dzh = z0 + k2

    return Dzh


@njit(cache=False, fastmath=True)
def drift_ode_solve2(nodes: np.ndarray, sigma0_factors: np.ndarray, theta: float, kappa1: float, kappa2: float,
                     z0: np.ndarray, weights: np.ndarray, h: float):
    """
    drift map D_h by one classical Runge-Kutta 4 step; the default drift step of the scheme.

    Solves d xi_i / dt = -x_i (xi_i - sigma0_factors_i) + (kappa1 + kappa2 xi^N)(theta - xi^N)
    with xi^N = sum_i w_i xi_i.

    Parameters
    ----------
    nodes : exponents x_i, array of size (n, nb_path)
    sigma0_factors : factor targets sigma0_factors_i, array of size (n, nb_path)
    theta : long-run level, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    z0 : factor values at the start of the step, array of size (n, nb_path)
    weights : kernel weights w_i, array of size (n, nb_path)
    h : step size

    Returns
    -------
    Array of size (n, nb_path)
    """
    assert nodes.shape == sigma0_factors.shape == z0.shape == weights.shape
    n, nb_path = weights.shape

    # --- k1 ---
    z0w = np.sum(weights * z0, axis=0)
    c1 = (kappa1 + kappa2 * z0w) * (theta - z0w)
    s1 = -nodes * (z0 - sigma0_factors) + c1

    # --- k2 ---
    z_tmp = z0 + 0.5 * h * s1
    z1w = np.sum(weights * z_tmp, axis=0)
    c2 = (kappa1 + kappa2 * z1w) * (theta - z1w)
    s2 = -nodes * (z_tmp - sigma0_factors) + c2

    # --- k3 ---
    z_tmp = z0 + 0.5 * h * s2
    z2w = np.sum(weights * z_tmp, axis=0)
    c3 = (kappa1 + kappa2 * z2w) * (theta - z2w)
    s3 = -nodes * (z_tmp - sigma0_factors) + c3

    # --- k4 ---
    z_tmp = z0 + h * s3
    z3w = np.sum(weights * z_tmp, axis=0)
    c4 = (kappa1 + kappa2 * z3w) * (theta - z3w)
    s4 = -nodes * (z_tmp - sigma0_factors) + c4

    zh = z0 + (h / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)

    return zh


@njit(cache=False, fastmath=True)
def drift_ode_solve2_f32(nodes: np.ndarray, sigma0_factors: np.ndarray, theta: float, kappa1: float, kappa2: float,
                         z0: np.ndarray, weights: np.ndarray, h: float):
    """
    float32-friendly variant of drift_ode_solve2 (avoids float64 literals promoting intermediates).
    """
    assert nodes.shape == sigma0_factors.shape == z0.shape == weights.shape
    n, nb_path = weights.shape

    half = np.float32(0.5)
    two = np.float32(2.0)
    six = np.float32(6.0)

    z0w = np.sum(weights * z0, axis=0)
    c1 = (kappa1 + kappa2 * z0w) * (theta - z0w)
    s1 = -nodes * (z0 - sigma0_factors) + c1

    z_tmp = z0 + (half * h) * s1
    z1w = np.sum(weights * z_tmp, axis=0)
    c2 = (kappa1 + kappa2 * z1w) * (theta - z1w)
    s2 = -nodes * (z_tmp - sigma0_factors) + c2

    z_tmp = z0 + (half * h) * s2
    z2w = np.sum(weights * z_tmp, axis=0)
    c3 = (kappa1 + kappa2 * z2w) * (theta - z2w)
    s3 = -nodes * (z_tmp - sigma0_factors) + c3

    z_tmp = z0 + h * s3
    z3w = np.sum(weights * z_tmp, axis=0)
    c4 = (kappa1 + kappa2 * z3w) * (theta - z3w)
    s4 = -nodes * (z_tmp - sigma0_factors) + c4

    zh = z0 + (h / six) * (s1 + two * s2 + two * s3 + s4)

    return zh


@njit(cache=False, fastmath=True)
def drift_ode_solve3(nodes: np.ndarray, sigma0_factors: np.ndarray, theta: float, kappa1: float, kappa2: float,
                     z0: np.ndarray, weights: np.ndarray, h: float):
    """
    drift map D_h in closed form, with kappa1 + kappa2 xi^N frozen at the start of the step.

    xi_h = exp(A h) xi_0 + A^{-1} (exp(A h) - I) b with A = -(kappa1 + kappa2 xi^N_0) 1 w' - diag(x)
    and b = (kappa1 + kappa2 xi^N_0) theta 1 + diag(x) sigma0_factors. Not used by the default
    scheme; freezing drops the damping of the quadratic term within the step.

    Parameters
    ----------
    nodes : exponents x_i, array of size (n, nb_path)
    sigma0_factors : factor targets sigma0_factors_i, array of size (n, nb_path)
    theta : long-run level, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    z0 : factor values at the start of the step, array of size (n, nb_path)
    weights : kernel weights w_i, array of size (n, nb_path)
    h : step size

    Returns
    -------
    Array of size (n, nb_path)
    """
    assert nodes.shape == sigma0_factors.shape == z0.shape == weights.shape
    n, nb_path = weights.shape
    z0w = np.sum(weights * z0, axis=0)
    kappa = kappa1 + kappa2 * z0w

    b_ = np.zeros((nb_path, n))
    for k in range(n):
        b_[:, k] = kappa * theta + nodes[k] * sigma0_factors[k]

    eAh = batch_expA(kappa, nodes.T * h, weights.T * h)
    I = np.identity(n)
    invA = batch_invA(kappa, nodes.T, weights.T)
    tmp2 = np.zeros((nb_path, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                tmp2[:,i,j] += invA[:,i,k] * (eAh[:, k, j] - I[k,j])

    Dzh_v1 = np.zeros((nb_path, n))
    for i in range(n):
        for j in range(n):
            Dzh_v1[:, i] += eAh[:, i, j] * z0[j, :] + tmp2[:, i, j] * b_[:, j]

    return Dzh_v1.T


@njit(cache=False, fastmath=True)
def diffus_sde_solve_f64(y0: np.ndarray, weights: np.ndarray, vartheta: float, h: float, nb_path: int,
                         z_rand: np.ndarray):
    """
    diffusion map S_h, exact: d eta_i = vartheta eta^N dW^sigma for every factor i.

    eta^N = sum_i w_i eta_i is a geometric Brownian motion with volatility vartheta * sum_i w_i,
    and every factor receives the increment (eta^N_{t+h} - eta^N_t) / sum_i w_i.

    Parameters
    ----------
    y0 : factor values at the start of the step, array of size (n, nb_path)
    weights : kernel weights w_i, array of size (n, nb_path)
    vartheta : total volatility-of-volatility sqrt(beta^2 + epsilon^2), scalar
    h : step size
    nb_path : number of paths
    z_rand : iid standard normals driving W^sigma, shape (nb_path, )
    """
    assert y0.shape == weights.shape and y0.shape[-1] == nb_path
    assert z_rand.shape == (nb_path,)
    weight_sum = np.sum(weights, axis=0)
    vartheta_ = vartheta * weight_sum

    yw = np.sum(weights * y0, axis=0)

    dW = z_rand * np.sqrt(h)
    Yh = yw * np.exp(-0.5 * vartheta_ ** 2 * h + vartheta_ * dW)

    Q = 1.0 / weight_sum * (Yh - yw)
    Yh_vec = y0.copy()
    for i in range(Yh_vec.shape[0]):
        Yh_vec[i] += Q

    return Yh_vec


@njit(cache=False, fastmath=True)
def drift_diffus_strand_f64(nodes: np.ndarray, sigma0_factors: np.ndarray, theta: float, kappa1: float, kappa2: float,
                            vartheta: float, factors: np.ndarray, weights: np.ndarray, h: float,
                            nb_path: int, z_rand: np.ndarray):
    """
    one volatility step by Strang splitting D(h/2) o S(h) o D(h/2).

    Parameters
    ----------
    nodes : exponents x_i, array of size (n, nb_path)
    sigma0_factors : factor targets sigma0_factors_i = sigma0 / sum_i w_i,
        array of size (n, nb_path)
    theta : long-run level, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    vartheta : total volatility-of-volatility sqrt(beta^2 + epsilon^2), scalar
    factors : factor values sigma^(i)_t, array of size (n, nb_path)
    weights : kernel weights w_i, array of size (n, nb_path)
    h : time step size
    nb_path : number of paths for MC simulation
    z_rand : iid standard normals driving W^sigma, shape (nb_path, )

    Returns
    -------
    Factor values sigma^(i)_{t+h}, array of size (n, nb_path)
    """
    D_inn = drift_ode_solve2(nodes, sigma0_factors, theta, kappa1, kappa2, factors, weights, 0.5 * h)
    S_inn = diffus_sde_solve_f64(D_inn, weights, vartheta, h, nb_path, z_rand)
    sol = drift_ode_solve2(nodes, sigma0_factors, theta, kappa1, kappa2, S_inn, weights, 0.5 * h)

    return sol


@njit(cache=False, fastmath=True)
def log_spot_full_solve2_f64(nodes: np.ndarray, weights: np.ndarray,
                             sigma0_factors: np.ndarray, qvar0: np.ndarray,
                             theta: float, kappa1: float, kappa2: float, log_s: np.ndarray, factors: np.ndarray,
                             qvar: np.ndarray, rho: float, vartheta: float, h: float, nb_path: int,
                             z0: np.ndarray, z1: np.ndarray):
    """
    one step of the volatility factors, the log-price and the integrated variance qvar.

    term1 is the trapezoidal estimate J of int_t^{t+h} sigma^N dW^sigma recovered from the
    volatility increment; term2 is Vbar h with Vbar = 0.5 (sigma^N_t^2 + sigma^N_{t+h}^2).
    """
    assert nodes.shape == weights.shape and weights.ndim == 2
    assert factors.shape == weights.shape and factors.shape[-1] == nb_path
    assert qvar.shape == (1, nb_path)
    assert log_s.shape == (1, nb_path)
    assert sigma0_factors.shape == weights.shape and sigma0_factors.shape[-1] == nb_path
    assert qvar0.shape == (1, nb_path)
    assert z0.shape == (nb_path,) and z1.shape == (nb_path,)

    vol_h = drift_diffus_strand_f64(nodes, sigma0_factors, theta, kappa1, kappa2, vartheta, factors, weights, h,
                                    nb_path, z0)
    w_vol_h = np.sum(weights * vol_h, axis=0)
    idx_bad = np.nonzero(np.logical_or(np.isnan(w_vol_h), w_vol_h <= 0.0))[0]
    vol_h[:, idx_bad] = 1e-6

    wlam = weights * nodes
    vw = np.sum(weights * factors, axis=0)
    volw_h = np.sum(weights * vol_h, axis=0)
    w_inv = 1.0 / np.sum(weights, axis=0)

    c1 = 0.5
    c2 = 0.5
    rho_comp = np.sqrt(1.0 - rho * rho)

    sq_vw = np.square(vw)
    sq_vhw = np.square(volw_h)

    w_lam_vol = np.sum(wlam * factors, axis=0)
    w_lam_vol_h = np.sum(wlam * vol_h, axis=0)
    w_lam_v0 = np.sum(wlam * sigma0_factors, axis=0)

    term1 = 1.0 / vartheta * (((volw_h - vw) / h + c1 * w_lam_vol + c2 * w_lam_vol_h - w_lam_v0) * w_inv
                              - kappa1 * theta + (kappa1 - kappa2 * theta) * (c1 * vw + c2 * volw_h)
                              + kappa2 * (c1 * sq_vw + c2 * sq_vhw)) * h

    term2 = c1 * h * sq_vw + c2 * h * sq_vhw
    log_spot_h = log_s - 0.5 * term2 + rho * term1 + rho_comp * np.sqrt(term2) * z1

    qvar_h = qvar + 0.5 * h * (vw * vw + volw_h * volw_h)

    return vol_h, qvar_h, log_spot_h


@njit(cache=False, fastmath=True)
def log_spot_full_combined_f64(nodes: np.ndarray, weights: np.ndarray,
                               sigma0_factors: np.ndarray,
                               theta: float, kappa1: float, kappa2: float, log_s0: float, factors_init: np.ndarray,
                               rho: float, vartheta: float, timegrid: np.ndarray, nb_path: int,
                               Z0: np.ndarray, Z1: np.ndarray):
    """float64 loop over a uniform grid; returns log S_T, the factors at T and I_T."""
    h = timegrid[1] - timegrid[0]

    qvar0 = np.zeros((1, nb_path))

    vol_h = factors_init.copy()
    qvar_h = np.zeros((1, nb_path))
    log_spot_h = np.ones((1, nb_path)) * log_s0

    for idx, _ in enumerate(timegrid[:-1]):
        vol_h, qvar_h, log_spot_h = log_spot_full_solve2_f64(nodes, weights, sigma0_factors, qvar0, theta, kappa1,
                                                             kappa2, log_spot_h, vol_h, qvar_h, rho, vartheta, h,
                                                             nb_path, Z0[idx], Z1[idx])

    return log_spot_h, vol_h, qvar_h


@njit(cache=False, fastmath=True)
def diffus_sde_solve_f32(y0: np.ndarray, weights: np.ndarray, weight_sum: np.ndarray,
                         vartheta_weight_sum: np.ndarray, h: float, sqrt_h: float, nb_path: int,
                         z_rand: np.ndarray):
    """float32 variant of diffus_sde_solve_f64 with precomputed weight sums."""
    assert y0.shape == weights.shape and y0.shape[-1] == nb_path
    assert z_rand.shape == (nb_path,)
    yw = np.sum(weights * y0, axis=0)

    half = np.float32(0.5)
    one = np.float32(1.0)
    dW = z_rand * sqrt_h
    Yh = yw * np.exp(-half * vartheta_weight_sum * vartheta_weight_sum * h + vartheta_weight_sum * dW)

    Q = one / weight_sum * (Yh - yw)
    Yh_vec = y0.copy()
    for i in range(Yh_vec.shape[0]):
        Yh_vec[i] += Q

    return Yh_vec


@njit(cache=False, fastmath=True)
def drift_diffus_strand_f32(nodes: np.ndarray, sigma0_factors: np.ndarray, theta: float, kappa1: float, kappa2: float,
                            vartheta_weight_sum: np.ndarray, factors: np.ndarray, weights: np.ndarray, h: float,
                            sqrt_h: float, weight_sum: np.ndarray, nb_path: int, z_rand: np.ndarray):
    """float32 variant of drift_diffus_strand_f64."""
    half = np.float32(0.5)
    D_inn = drift_ode_solve2_f32(nodes, sigma0_factors, theta, kappa1, kappa2, factors, weights, half * h)
    S_inn = diffus_sde_solve_f32(D_inn, weights, weight_sum, vartheta_weight_sum, h, sqrt_h, nb_path, z_rand)
    sol = drift_ode_solve2_f32(nodes, sigma0_factors, theta, kappa1, kappa2, S_inn, weights, half * h)

    return sol


@njit(cache=False, fastmath=True)
def log_spot_full_solve2_f32(nodes: np.ndarray, weights: np.ndarray, wlam: np.ndarray, w_inv: np.ndarray,
                             w_lam_v0: np.ndarray, weight_sum: np.ndarray, sigma0_factors: np.ndarray,
                             qvar0: np.ndarray, theta: float, kappa1: float, kappa2: float, log_s: np.ndarray,
                             factors: np.ndarray, qvar: np.ndarray, rho: float, rho_comp: float,
                             inv_vartheta: float, vartheta_weight_sum: np.ndarray,
                             h: float, sqrt_h: float, nb_path: int, z0: np.ndarray, z1: np.ndarray):
    """float32 variant of log_spot_full_solve2_f64 with precomputed constants."""
    assert nodes.shape == weights.shape and weights.ndim == 2
    assert factors.shape == weights.shape and factors.shape[-1] == nb_path
    assert qvar.shape == (1, nb_path)
    assert log_s.shape == (1, nb_path)
    assert sigma0_factors.shape == weights.shape and sigma0_factors.shape[-1] == nb_path
    assert qvar0.shape == (1, nb_path)
    assert z0.shape == (nb_path,) and z1.shape == (nb_path,)

    half = np.float32(0.5)
    eps = np.float32(1e-6)

    vol_h = drift_diffus_strand_f32(nodes, sigma0_factors, theta, kappa1, kappa2, vartheta_weight_sum, factors,
                                    weights, h, sqrt_h, weight_sum, nb_path, z0)
    volw_h = np.sum(weights * vol_h, axis=0)
    idx_bad = np.nonzero(np.logical_or(np.isnan(volw_h), volw_h <= 0.0))[0]
    vol_h[:, idx_bad] = eps

    vw = np.sum(weights * factors, axis=0)

    c1 = half
    c2 = half

    sq_vw = np.square(vw)
    sq_vhw = np.square(volw_h)

    w_lam_vol = np.sum(wlam * factors, axis=0)
    w_lam_vol_h = np.sum(wlam * vol_h, axis=0)

    term1 = inv_vartheta * (((volw_h - vw) / h + c1 * w_lam_vol + c2 * w_lam_vol_h - w_lam_v0) * w_inv
                            - kappa1 * theta + (kappa1 - kappa2 * theta) * (c1 * vw + c2 * volw_h)
                            + kappa2 * (c1 * sq_vw + c2 * sq_vhw)) * h

    term2 = c1 * h * sq_vw + c2 * h * sq_vhw
    log_spot_h = log_s - half * term2 + rho * term1 + rho_comp * np.sqrt(term2) * z1

    qvar_h = qvar + half * h * (vw * vw + volw_h * volw_h)

    return vol_h, qvar_h, log_spot_h


@njit(cache=False, fastmath=True)
def log_spot_full_combined_f32(nodes: np.ndarray, weights: np.ndarray,
                               sigma0_factors: np.ndarray,
                               theta: float, kappa1: float, kappa2: float, log_s0: float, factors_init: np.ndarray,
                               rho: float, vartheta: float, timegrid: np.ndarray, nb_path: int,
                               Z0: np.ndarray, Z1: np.ndarray):
    """float32 loop over a uniform grid; returns log S_T, the factors at T and I_T."""
    one = np.float32(1.0)
    h = timegrid[1] - timegrid[0]
    sqrt_h = np.sqrt(h)
    wlam = weights * nodes
    weight_sum = np.sum(weights, axis=0)
    w_inv = one / weight_sum
    w_lam_v0 = np.sum(wlam * sigma0_factors, axis=0)
    rho_comp = np.sqrt(one - rho * rho)
    inv_vartheta = one / vartheta
    vartheta_weight_sum = vartheta * weight_sum

    qvar0 = np.zeros_like(sigma0_factors[:1, :])

    vol_h = factors_init.copy()
    qvar_h = np.zeros_like(qvar0)
    log_spot_h = np.empty_like(qvar0)
    log_spot_h[:] = log_s0

    for idx, _ in enumerate(timegrid[:-1]):
        vol_h, qvar_h, log_spot_h = log_spot_full_solve2_f32(nodes, weights, wlam, w_inv, w_lam_v0, weight_sum,
                                                             sigma0_factors, qvar0, theta, kappa1, kappa2, log_spot_h,
                                                             vol_h, qvar_h, rho, rho_comp, inv_vartheta,
                                                             vartheta_weight_sum, h, sqrt_h, nb_path, Z0[idx], Z1[idx])

    return log_spot_h, vol_h, qvar_h


def log_spot_full_combined(nodes: np.ndarray, weights: np.ndarray,
                           sigma0_factors: np.ndarray,
                           theta: float, kappa1: float, kappa2: float, log_s0: float, factors_init: np.ndarray,
                           rho: float, vartheta: float, timegrid: np.ndarray, nb_path: int,
                           Z0: np.ndarray, Z1: np.ndarray):
    """
    simulate log S_T, the factors at T and the integrated variance I_T on a uniform time grid.

    Parameters
    ----------
    nodes, weights : kernel nodes x_i and weights w_i, arrays of size (n, nb_path)
    sigma0_factors : factor targets sigma0_factors_i = sigma0 / sum_i w_i,
        array of size (n, nb_path)
    theta, kappa1, kappa2 : drift parameters of (kappa1 + kappa2 sigma)(theta - sigma)
    log_s0 : initial log-price
    factors_init : factor values at time 0, array of size (n, nb_path), equal to sigma0_factors
    rho : correlation beta / vartheta between W0 and W^sigma
    vartheta : total volatility-of-volatility sqrt(beta^2 + epsilon^2)
    timegrid : uniform grid starting at 0, size nb_steps + 1
    nb_path : number of paths
    Z0, Z1 : standard normals for W^sigma and the orthogonal price noise,
        shape (>= nb_steps, nb_path)

    Dispatches to the float64 kernel by default and to the float32 kernel only when all array
    inputs are float32.
    """
    if (nodes.dtype == np.float32 and weights.dtype == np.float32 and sigma0_factors.dtype == np.float32 and
            timegrid.dtype == np.float32 and Z0.dtype == np.float32 and Z1.dtype == np.float32):
        return log_spot_full_combined_f32(nodes, weights, sigma0_factors, theta, kappa1, kappa2, log_s0, factors_init,
                                          rho, vartheta, timegrid, nb_path, Z0, Z1)
    return log_spot_full_combined_f64(nodes, weights, sigma0_factors, theta, kappa1, kappa2, log_s0, factors_init,
                                      rho, vartheta, timegrid, nb_path, Z0, Z1)
