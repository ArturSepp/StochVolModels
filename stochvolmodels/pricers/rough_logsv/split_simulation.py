import numpy as np
from enum import Enum

from typing import Tuple
from scipy.linalg import expm
from numba import njit
from numba import njit, objmode

from stochvolmodels.pricers.rough_logsv.expm import batch_expA, batch_invA


# from stochvolmodels.utils.config import VariableType
# from RoughKernel import european_rule
# from stochvolmodels.pricers.logsv_pricer import LogSvParams


@njit(cache=False, fastmath=True)
def drift_ode_solve(nodes: np.ndarray, v0: np.ndarray, theta: float, kappa1: float, kappa2: float,
                    z0: np.ndarray, weight: np.ndarray, h: float):
    """

    Parameters
    ----------
    nodes : (fixed argument) exponents x_i, array of size (n,)
    v0 : (fixed argument) array of size (n, nb_path)
    theta : long-run level, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    z0 : initial values, array of size (n, nb_path)
    weight : wieghts, array of size (n, )
    h : step size

    Returns
    -------
    Array of size (n, nb_path)
    """
    assert nodes.shape == v0.shape == z0.shape == weight.shape
    n = z0.shape[0]
    z0w = np.sum(weight * z0, axis=0)
    g0 = (kappa1 + kappa2*z0w)*(theta-z0w)
    k1 = -nodes * (z0 - v0)
    for j in range(n):
        k1[j] += g0
    k1 *= 0.5 * h

    zmid = z0 + k1
    zmidw = np.sum(weight * zmid, axis=0)
    gmid = (kappa1 + kappa2 * zmidw) * (theta - zmidw)
    k2 = -nodes * (zmid - v0)
    for j in range(n):
        k2[j] += gmid
    k2 *= h
    Dzh = z0 + k2

    # s1 = -nodes[:, None] * (z0 - v0) * h + (kappa1 + kappa2 * z0w) * (theta - z0w) * h
    # s2 = -nodes[:, None] * (z0 + 0.5 * s1 - v0) * h + (kappa1 + kappa2 * (z0w + 0.5 * s1)) * (theta - (z0w + 0.5 * s1)) * h
    # Dzh = z0 + s2



    # z1 =  -nodes * (z0 - v0) * h + (kappa1 + kappa2 * z0w) * (theta - z0w) * h
    # z2 = -nodes * (z0 + 0.5 * z1 - v0) * h + (kappa1 + kappa2 * (z0w + 0.5 * z1)) * (theta - (z0w + 0.5 * z1)) * h
    # Dzh = z0 + z2

    # lamda_vec = np.repeat(lamda, n)
    # theta_vec = np.repeat(theta, n)[:, None]
    # diag_x = np.diag(nodes)
    # I = np.identity(n)
    # A = -np.outer(lamda_vec, weight) - diag_x  # matrix of size (n, n)
    # b = theta_vec + diag_x @ v0  # vector of size (n, nb_path)
    # # with objmode(eAh='float64[:, ::1]'):
    # eAh = expm(A * h)
    # Dzh = eAh @ z0 + (np.linalg.inv(A) @ (eAh - I)) @ b  # vector of size (n, nb_path)

    return Dzh


@njit(cache=False, fastmath=True)
def drift_ode_solve3(nodes: np.ndarray, v0: np.ndarray, theta: float, kappa1: float, kappa2: float,
                    z0: np.ndarray, weight: np.ndarray, h: float):
    """

    Parameters
    ----------
    nodes : (fixed argument) exponents x_i, array of size (n,)
    v0 : (fixed argument) array of size (n, nb_path)
    theta : long-run level, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    z0 : initial values, array of size (n, nb_path)
    weight : wieghts, array of size (n, )
    h : step size

    Returns
    -------
    Array of size (n, nb_path)
    """
    assert nodes.shape == v0.shape == z0.shape == weight.shape
    n, nb_path = weight.shape
    z0w = np.sum(weight * z0, axis=0)
    kappa = kappa1 + kappa2 * z0w

    b_ = np.zeros((nb_path, n))
    for k in range(n):
        b_[:, k] = kappa * theta + nodes[k] * v0[k]

    eAh = batch_expA(kappa, nodes.T * h, weight.T * h)
    I = np.identity(n)
    invA = batch_invA(kappa, nodes.T, weight.T)
    tmp2 = np.zeros((nb_path, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                tmp2[:,i,j] += invA[:,i,k] * (eAh[:, k, j] - I[k,j])

    Dzh_v1 = np.zeros((nb_path, n))
    for i in range(n):
        for j in range(n):
            Dzh_v1[:, i] += eAh[:, i, j] * z0[j, :] + tmp2[:, i, j] * b_[:, j]
    # for p in range(nb_path):
    #     Dzh_v1[p] = eAh[p] @ z0[..., p] + tmp2[p] @ b_[p]

    # def ff2():
    #     lamda = kappa1 + kappa2 * z0w
    #     coeff0_vec = np.repeat(kappa1 * theta, n)
    #     Dzh_v2 = np.zeros_like(z0)
    #     for p in range(nb_path):
    #         diag_x = np.diag(nodes[:, p])
    #         lamda_vec = np.repeat(lamda[p], n)
    #         A = -np.outer(lamda_vec, weight[:, p]) - diag_x  # matrix of size (n,n)
    #         b = coeff0_vec + diag_x @ v0[:, p]
    #         eAh_v2 = expm(A * h)
    #         Dzh_v2[:, p] = eAh_v2 @ z0[:, p] + (np.linalg.inv(A) @ (eAh_v2 - I)) @ b
    #
    #     return Dzh_v2
    #
    # Dzh_v2 = ff2()
    # diff = np.linalg.norm(Dzh_v2-Dzh_v1.T)
    # assert diff < 1e-12

    return Dzh_v1.T


@njit(cache=False, fastmath=True)
def diffus_sde_solve(y0: np.ndarray, weight: np.ndarray, volvol: float, h: float, nb_path: int,
                     z_rand: np.ndarray):
    assert y0.shape == weight.shape  and y0.shape[-1] == nb_path
    assert z_rand.shape == (nb_path,)
    weight_sum = np.sum(weight, axis=0)
    volvol_ = volvol * weight_sum

    yw = np.sum(weight * y0, axis=0)

    dW = z_rand * np.sqrt(h)
    Yh = yw * np.exp(-0.5 * volvol_ ** 2 * h + volvol_ * dW)

    Q = 1.0 / weight_sum * (Yh - yw)
    Yh_vec = y0.copy()
    for i in range(Yh_vec.shape[0]):
        Yh_vec[i] += Q

    return Yh_vec

@njit(cache=False, fastmath=True)
def drift_diffus_strand(nodes: np.ndarray, v0: np.ndarray, theta: float, kappa1: float, kappa2: float,
                        volvol: float, v_init: np.ndarray, weight: np.ndarray, h: float,
                        nb_path: int, z_rand: np.ndarray):
    """

    Parameters
    ----------
    nodes : exponents x_i, array of size (n,)
    v0 : (fixed argument) array of size (n, nb_path)
    theta : long-run level, scaled by lambda, scalar
    kappa1 : linear mean-reversion speed, scalar
    kappa2 : quadratic mean-reversion speed, scalar
    volvol :  total volatility. in other words, vartheta. scalar
    v_init :  initial values, array of size (n, nb_path)
    weight :  (fixed argument) wieghts, array of size (n, )
    h : time step size
    nb_path : number of paths for MC simulation
    z_rand : iid standard normals, shape (nb_path, )

    Returns
    -------

    """
    D_inn = drift_ode_solve3(nodes, v0, theta, kappa1, kappa2, v_init, weight, 0.5 * h)
    S_inn = diffus_sde_solve(D_inn, weight, volvol, h, nb_path, z_rand)
    sol = drift_ode_solve3(nodes, v0, theta, kappa1, kappa2, S_inn, weight, 0.5 * h)

    return sol


@njit(cache=False, fastmath=True)
def log_spot_full_solve2(nodes: np.ndarray, weight: np.ndarray,
                         v0: np.ndarray, y0: np.ndarray,
                         theta: float, kappa1: float, kappa2: float, log_s: np.ndarray, v: np.ndarray, y: np.ndarray,
                         rho: float, volvol: float, h: float, nb_path: int,
                         z0: np.ndarray, z1: np.ndarray):
    # raise ValueError
    assert nodes.shape == weight.shape and weight.ndim == 2
    assert v.shape == weight.shape and v.shape[-1] == nb_path
    assert y.shape == (1, nb_path)
    assert log_s.shape == (1, nb_path)
    assert v0.shape == weight.shape and v0.shape[-1] == nb_path
    assert y0.shape == (1, nb_path)
    assert z0.shape == (nb_path,) and z1.shape == (nb_path,)

    vol_h = drift_diffus_strand(nodes, v0, theta, kappa1, kappa2, volvol, v, weight, h, nb_path, z0)

    wlam = weight * nodes
    vw = np.sum(weight * v, axis=0)
    volw_h = np.sum(weight * vol_h, axis=0)
    w_inv = 1.0 / np.sum(weight, axis=0)

    c1 = 0.5
    c2 = 0.5
    rho_comp = np.sqrt(1.0 - rho * rho)

    sq_vw = np.square(vw)
    sq_vhw = np.square(volw_h)

    w_lam_vol = np.sum(wlam * v, axis=0)
    w_lam_vol_h = np.sum(wlam * vol_h, axis=0)
    w_lam_v0 = np.sum(wlam * v0, axis=0)

    term1 = 1.0 / volvol * (((volw_h - vw) / h + c1 * w_lam_vol + c2 * w_lam_vol_h - w_lam_v0) * w_inv
                            - kappa1 * theta + (kappa1 - kappa2 * theta) * (c1 * vw + c2 * volw_h)
                            + kappa2 * (c1 * sq_vw + c2 * sq_vhw)) * h

    term2 = c1 * h * sq_vw + c2 * h * sq_vhw
    log_spot_h = log_s - 0.5 * term2 + rho * term1 + rho_comp * np.sqrt(term2) * z1

    y_h = y + 0.5 * h * (vw * vw + volw_h * volw_h)  # don't need it but keep for plotting

    return vol_h, y_h, log_spot_h

@njit(cache=False, fastmath=True)
def log_spot_full_combined(nodes: np.ndarray, weight: np.ndarray,
                           v0: np.ndarray,
                           theta: float, kappa1: float, kappa2: float, log_s0: float, v_init: np.ndarray,
                           rho: float, volvol: float, timegrid: np.ndarray, nb_path: int,
                           Z0: np.ndarray, Z1: np.ndarray):
    h = timegrid[1] - timegrid[0]
    # assert np.all(np.isclose(np.diff(timegrid)[1:], h)) and timegrid[0] == 0.0
    # assert Z0.shape == (timegrid.size - 1, nb_path) and Z1.shape == (timegrid.size - 1, nb_path)

    y0 = np.zeros((1, nb_path))

    vol_h = v_init.copy()
    y_h = np.zeros((1, nb_path))
    log_spot_h = np.ones((1, nb_path)) * log_s0

    # a, b, c = solve_coeffs(nodes, weight, v0[:, 0], theta, lamda, rho, volvol)

    for idx, _ in enumerate(timegrid[:-1]):
        vol_h, y_h, log_spot_h = log_spot_full_solve2(nodes, weight, v0, y0, theta, kappa1, kappa2,
                                                     log_spot_h, vol_h, y_h, rho, volvol, h, nb_path,
                                                      Z0[idx], Z1[idx])

    return log_spot_h, vol_h, y_h
