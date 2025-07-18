import numpy as np
from enum import Enum

from typing import Tuple
from scipy.linalg import expm
from numba import njit
from numba import njit, objmode

# from stochvolmodels.examples.run_pricing_options_on_qvar import nb_path


# from stochvolmodels.utils.config import VariableType
# from RoughKernel import european_rule
# from stochvolmodels.pricers.logsv_pricer import LogSvParams

@njit(cache=False, fastmath=True)
def drift_ode_solve(nodes: np.ndarray, v0: np.ndarray, theta: float, lamda: float,
                    z0: np.ndarray, weight: np.ndarray, h: float):
    """

    Parameters
    ----------
    nodes : (fixed argument) exponents x_i, array of size (n,)
    v0 : (fixed argument) array of size (n, nb_path)
    theta : long-run level, scaled by lambda, scalar
    lamda : mean-reversion speed, scalar
    z0 : initial values, array of size (n, nb_path)
    weight : wieghts, array of size (n, )
    h : step size

    Returns
    -------
    Array of size (n, nb_path)
    """
    assert nodes.shape == v0.shape[:-1] == z0.shape[:-1] == weight.shape
    n = weight.size
    lamda_vec = np.repeat(lamda, n)
    theta_vec = np.repeat(theta, n)[:, None]
    diag_x = np.diag(nodes)
    I = np.identity(n)
    A = -np.outer(lamda_vec, weight) - diag_x  # matrix of size (n, n)
    b = theta_vec + diag_x @ v0  # vector of size (n, nb_path)
    with objmode(eAh='float64[:, ::1]'):
        eAh = expm(A * h)
    Dzh = eAh @ z0 + (np.linalg.inv(A) @ (eAh - I)) @ b  # vector of size (n, nb_path)

    return Dzh

@njit(cache=False, fastmath=True)
def diffus_sde_solve(y0: np.ndarray, weight: np.ndarray, volvol: float, h: float, nb_path: int,
                     z_rand: np.ndarray):
    assert y0.shape[:-1] == weight.shape and y0.shape[-1] == nb_path
    assert z_rand.shape == (nb_path,)
    n = weight.size
    weight_sum = np.sum(weight)
    volvol_ = volvol * weight_sum

    yw = np.dot(weight, y0)

    dW = z_rand * np.sqrt(h)
    Yh = yw * np.exp(-0.5 * volvol_ ** 2 * h + volvol_ * dW)

    Q = 1.0 / weight_sum * (Yh - yw)
    Yh_vec = y0.copy()
    for i in range(Yh_vec.shape[0]):
        Yh_vec[i] += Q

    return Yh_vec

@njit(cache=False, fastmath=True)
def drift_diffus_strand(nodes: np.ndarray, v0: np.ndarray, theta: float, lamda: float,
                        volvol: float, v_init: np.ndarray, weight: np.ndarray, h: float,
                        nb_path: int, z_rand: np.ndarray):
    """

    Parameters
    ----------
    nodes : exponents x_i, array of size (n,)
    v0 : (fixed argument) array of size (n, nb_path)
    theta : long-run level, scaled by lambda, scalar
    lamda : mean-reversion speed, scalar
    volvol :  total volatility. in other words, vartheta. scalar
    v_init :  initial values, array of size (n, nb_path)
    weight :  (fixed argument) wieghts, array of size (n, )
    h : time step size
    nb_path : number of paths for MC simulation
    z_rand : iid standard normals, shape (nb_path, )

    Returns
    -------

    """
    D_inn = drift_ode_solve(nodes, v0, theta, lamda, v_init, weight, 0.5 * h)
    S_inn = diffus_sde_solve(D_inn, weight, volvol, h, nb_path, z_rand)
    sol = drift_ode_solve(nodes, v0, theta, lamda, S_inn, weight, 0.5 * h)

    return sol


@njit(cache=False, fastmath=True)
def log_spot_full_solve2(nodes: np.ndarray, weight: np.ndarray,
                         v0: np.ndarray, y0: np.ndarray,
                         theta: float, lamda: float, log_s: np.ndarray, v: np.ndarray, y: np.ndarray,
                         rho: float, volvol: float, h: float, nb_path: int,
                         z0: np.ndarray, z1: np.ndarray):
    # raise ValueError
    assert nodes.shape == weight.shape and weight.ndim == 1
    assert v.shape[:-1] == weight.shape and v.shape[-1] == nb_path
    assert y.shape == (1, nb_path)
    assert log_s.shape == (1, nb_path)
    assert v0.shape[:-1] == weight.shape and v0.shape[-1] == nb_path
    assert y0.shape == (1, nb_path)
    assert z0.shape == (nb_path,) and z1.shape == (nb_path,)

    vol_h = drift_diffus_strand(nodes, v0, theta, lamda, volvol, v, weight, h, nb_path, z0)

    ww = weight[None, :]
    wlam = (weight * nodes)[None, :]
    vw = ww @ v
    volw_h = ww @ vol_h
    w_inv = 1.0 / np.sum(weight)

    c1 = 0.5
    c2 = 0.5
    rho_comp = np.sqrt(1.0 - rho * rho)

    term1 = 1.0 / volvol * ((volw_h - vw + c1 * h * wlam @ v + c2 * h * wlam @ vol_h - wlam @ v0 * h) * w_inv
                            - theta * h + lamda * c1 * h * vw + lamda * c2 * h * volw_h)
    term2 = c1 * h * np.square(vw) + c2 * h * np.square(volw_h)
    log_spot_h = log_s - 0.5 * term2 + rho * term1 + rho_comp * np.sqrt(term2) * z1

    y_h = y + 0.5 * h * (vw * vw + volw_h * volw_h)  # don't need it but keep for plotting

    return vol_h, y_h, log_spot_h

@njit(cache=False, fastmath=True)
def log_spot_full_combined(nodes: np.ndarray, weight: np.ndarray,
                           v0: np.ndarray,
                           theta: float, lamda: float, log_s0: float, v_init: np.ndarray,
                           rho: float, volvol: float, timegrid: np.ndarray, nb_path: int,
                           Z0: np.ndarray, Z1: np.ndarray):
    h = timegrid[1] - timegrid[0]
    assert np.all(np.isclose(np.diff(timegrid)[1:], h)) and timegrid[0] == 0.0
    assert Z0.shape == (timegrid.size - 1, nb_path) and Z1.shape == (timegrid.size - 1, nb_path)

    y0 = np.zeros((1, nb_path))

    vol_h = v_init.copy()
    y_h = np.zeros((1, nb_path))
    log_spot_h = np.ones((1, nb_path)) * log_s0

    # a, b, c = solve_coeffs(nodes, weight, v0[:, 0], theta, lamda, rho, volvol)

    for idx, _ in enumerate(timegrid[:-1]):
        vol_h, y_h, log_spot_h = log_spot_full_solve2(nodes, weight, v0, y0, theta, lamda,
                                                     log_spot_h, vol_h, y_h, rho, volvol, h, nb_path,
                                                      Z0[idx], Z1[idx])

    return log_spot_h, vol_h, y_h
