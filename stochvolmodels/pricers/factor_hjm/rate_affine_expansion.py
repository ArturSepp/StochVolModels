import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict, Union, Any
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder, get_expansion_n
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import splrep, splev


class UnderlyingType(Enum):
    SWAP = 1
    FUTURES = 2


def compute_logsv_a_mgf_grid(ttm: float,
                             phi_grid: np.ndarray,
                             sigma0: float,
                             q: float,
                             times: np.ndarray,
                             a0: np.ndarray,
                             a1: np.ndarray,
                             kappa0: np.ndarray,
                             kappa1: np.ndarray,
                             kappa2: np.ndarray,
                             beta: np.ndarray,
                             volvol: np.ndarray,
                             b: np.ndarray = None,
                             expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                             underlying_type: UnderlyingType = UnderlyingType.SWAP,
                             a_t0: Optional[np.ndarray] = None,
                             is_stiff_solver: bool = False,
                             **kwargs
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    affine expansion of MGF in rates beta SV model
    """
    if a_t0 is None:
        n_terms = get_expansion_n(expansion_order=expansion_order)
        a_t0 = np.zeros((phi_grid.shape[0], n_terms), dtype=np.complex128)

    if b is None:
        b = np.zeros_like(times)

    a_t1 = solve_a_ode_grid(phi_grid=phi_grid, ttm=ttm, q=q, times=times, a0=a0, a1=a1, kappa0=kappa0,
                            kappa1=kappa1, kappa2=kappa2, beta=beta, volvol=volvol, b=b,
                            a_t0=a_t0, expansion_order=expansion_order, is_stiff_solver=is_stiff_solver,
                            underlying_type=underlying_type)
    y = sigma0 - q
    if expansion_order == ExpansionOrder.FIRST:
        ys = np.array([1.0, y, y * y])  # params.v0 - params.theta
    elif expansion_order == ExpansionOrder.SECOND:
        y2 = y * y
        ys = np.array([1.0, y, y2, y2 * y, y2 * y2])
    else:
        raise NotImplementedError
    ys = ys + 1j * 0.0  # simple way to convert to complex array
    log_mgf = a_t1 @ ys
    return a_t1, log_mgf


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
def solve_a_ode_grid(phi_grid: np.ndarray,
                     ttm: float,
                     q: float,
                     times: np.ndarray,
                     a0: np.ndarray,
                     a1: np.ndarray,
                     kappa0: np.ndarray,
                     kappa1: np.ndarray,
                     kappa2: np.ndarray,
                     beta: np.ndarray,
                     volvol: np.ndarray,
                     b: np.ndarray = None,
                     a_t0: Optional[np.ndarray] = None,
                     is_stiff_solver: bool = False,
                     expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                     underlying_type: UnderlyingType = UnderlyingType.SWAP,
                     ) -> np.ndarray:
    """
    solve ode for range phi
    """

    def fun(phi: np.complex128, a_t0: np.ndarray):
        oderesult = solve_ode_for_a(ttm=ttm, q=q, times=times, a0=a0, a1=a1, kappa0=kappa0, kappa1=kappa1,
                                    kappa2=kappa2, beta=beta, volvol=volvol, b=b,
                                    phi=phi, expansion_order=expansion_order,
                                    is_stiff_solver=is_stiff_solver, underlying_type=underlying_type,
                                    a_t0=a_t0)
        return oderesult

    # pre-allocate array for solutions
    a_t1 = np.zeros((phi_grid.shape[0], get_expansion_n(expansion_order)), dtype=np.complex128)
    for idx, phi in enumerate(phi_grid):
        a_t1[idx, :] = fun(phi, a_t0[idx, :]).y[:, -1]
        # print(f"Time for MGF calculation at {phi}: {toc - tic:0.4f} seconds")
    return a_t1


def solve_ode_for_a(ttm: float,
                    q: float,
                    times: np.ndarray,
                    a0: np.ndarray,
                    a1: np.ndarray,
                    kappa0: np.ndarray,
                    kappa1: np.ndarray,
                    kappa2: np.ndarray,
                    beta: np.ndarray,
                    volvol: np.ndarray,
                    phi: np.complex128,
                    b: np.ndarray = None,
                    a_t0: Optional[np.ndarray] = None,
                    expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                    is_stiff_solver: bool = False,
                    underlying_type: UnderlyingType = UnderlyingType.SWAP,
                    dense_output: bool = False
                    ) -> OdeResult:
    """
    solve ode for given scalar phi
    """
    a0 = [splrep(times, a0[:,j], s=0) for j in range(a0.shape[1])]
    a1 = splrep(times, a1)
    kappa0 = splrep(times, kappa0)
    kappa1 = splrep(times, kappa1)
    kappa2 = splrep(times, kappa2)
    beta = [splrep(times, beta[:,j]) for j in range(beta.shape[1])]
    volvol = splrep(times, volvol)
    b = splrep(times, b)
    if is_stiff_solver:
        ode_sol = solve_ivp(fun=func_rhs, t_span=(0.0, ttm), y0=a_t0,
                            args=(ttm, q, times, a0, a1, kappa0, kappa1, kappa2, beta, volvol, b, phi, underlying_type, expansion_order),
                            method='BDF',
                            jac=func_rhs_jac,
                            dense_output=dense_output)
    else:
        ode_sol = solve_ivp(fun=func_rhs, t_span=(0.0, ttm), y0=a_t0,
                            args=(ttm, q, times, a0, a1, kappa0, kappa1, kappa2, beta, volvol, b, phi, underlying_type, expansion_order),
                            dense_output=dense_output)
    return ode_sol


# AFFINE EXPANSION FOR SWAPTIONS and FUTURES OPTIONS
# @njit(cache=False, fastmath=True) # TODO: uncomment
def func_rhs(tau: float,  # for ode solver compatibility
             A0: np.ndarray,
             ttm: float,
             q: float,
             times: np.ndarray,
             a0: Any,
             a1: Any,
             kappa0: Any,
             kappa1: Any,
             kappa2: Any,
             beta: Any,
             volvol: Any,
             b: np.ndarray,
             phi: np.complex128,
             underlying_type: UnderlyingType,
             expansion_order: ExpansionOrder) -> np.ndarray:
    """
    returns rhs of the ODEs
    """
    n = A0.shape[0]
    quadratic = np.zeros(n, dtype=np.complex128)
    # a_i = pw_const(ts=times, vs=a, t=ttm - tau)
    # delta_i = pw_const(ts=times, vs=delta, t=ttm - tau)
    # kappa0_i = pw_const(ts=times, vs=kappa0, t=ttm - tau)
    # kappa1_i = pw_const(ts=times, vs=kappa1, t=ttm - tau)
    # kappa2_i = pw_const(ts=times, vs=kappa2, t=ttm - tau)
    # beta_i = pw_const(ts=times, vs=beta, t=ttm - tau)
    # volvol_i = pw_const(ts=times, vs=volvol, t=ttm - tau)
    a0_i = np.array([float(splev(ttm-tau, a0[j])) for j in range(len(a0))])
    a1_i = float(splev(ttm-tau, a1))
    b_i = float(splev(ttm-tau, b))
    kappa0_i = float(splev(ttm-tau, kappa0))
    kappa1_i = float(splev(ttm-tau, kappa1))
    kappa2_i = float(splev(ttm-tau, kappa2))
    beta_i = np.array([float(splev(ttm-tau, beta[j])) for j in range(len(beta))])
    volvol_i = float(splev(ttm-tau, volvol))

    M, L, H = func_a_ode_quadratic_terms(q=q, a0=a0_i, a1=a1_i, kappa0=kappa0_i, kappa1=kappa1_i, kappa2=kappa2_i,
                                         beta=beta_i, volvol=volvol_i, phi=phi, b=b_i,
                                         underlying_type=underlying_type, expansion_order=expansion_order)

    for n_ in np.arange(n):
        quadratic[n_] = A0.T @ M[n_] @ A0
    rhs = quadratic + L @ A0 + H
    return rhs


# @njit(cache=False, fastmath=True)  # TODO: uncomment
def func_rhs_jac(tau: float,  # for ode solver compatibility
                 A0: np.ndarray,
                 ttm: float,
                 q: float,
                 times: np.ndarray,
                 a0: np.ndarray,
                 a1: np.ndarray,
                 kappa0: np.ndarray,
                 kappa1: np.ndarray,
                 kappa2: np.ndarray,
                 beta: np.ndarray,
                 volvol: np.ndarray,
                 b: np.ndarray,
                 phi: np.complex128,
                 underlying_type: UnderlyingType,
                 expansion_order: ExpansionOrder) -> np.ndarray:
    """
    returns rhs jacobian evaluation using matrices for the quadratic form A_t = A.T@M@A + L@A + H
    """
    n = A0.shape[0]
    quadratic = np.zeros((n, n), dtype=np.complex128)
    # a_i = pw_const(ts=times, vs=a, t=ttm - tau)
    # delta_i = pw_const(ts=times, vs=delta, t=ttm - tau)
    # kappa0_i = pw_const(ts=times, vs=kappa0, t=ttm - tau)
    # kappa1_i = pw_const(ts=times, vs=kappa1, t=ttm - tau)
    # kappa2_i = pw_const(ts=times, vs=kappa2, t=ttm - tau)
    # beta_i = pw_const(ts=times, vs=beta, t=ttm - tau)
    # volvol_i = pw_const(ts=times, vs=volvol, t=ttm - tau)
    a0_i = np.array([float(splev(ttm - tau, a0[j])) for j in range(len(a0))])
    a1_i = float(splev(ttm - tau, a1))
    b_i = float(splev(ttm - tau, b))
    kappa0_i = float(splev(ttm - tau, kappa0))
    kappa1_i = float(splev(ttm - tau, kappa1))
    kappa2_i = float(splev(ttm - tau, kappa2))
    beta_i = np.array([float(splev(ttm - tau, beta[j])) for j in range(len(beta))])
    volvol_i = float(splev(ttm - tau, volvol))

    M, L, H = func_a_ode_quadratic_terms(q=q, a0=a0_i, a1=a1_i, kappa0=kappa0_i, kappa1=kappa1_i, kappa2=kappa2_i,
                                         beta=beta_i, volvol=volvol_i, phi=phi, b=b_i,
                                         underlying_type=underlying_type, expansion_order=expansion_order)
    for n_ in np.arange(n):
        quadratic[n_, :] = 2.0 * M[n_] @ A0
    rhs = quadratic + A0
    return rhs

# @njit(cache=False, fastmath=True) # TODO: uncomment
def func_a_ode_quadratic_terms(q: float,
                               a0: Union[float, np.ndarray],
                               a1: float,
                               kappa0: float,
                               kappa1: float,
                               kappa2: float,
                               beta: Union[float, np.ndarray],
                               volvol: float,
                               b: float,
                               phi: np.complex128,
                               underlying_type: UnderlyingType,
                               expansion_order: ExpansionOrder
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Matrices for the quadratic form A_t = A.T@M@A + L@A + H
    """
    q2 = q * q
    # if isinstance(beta, float):
    #     vartheta2 = beta * beta + volvol * volvol
    # elif isinstance(beta, np.ndarray):
    #     vartheta2 = np.dot(beta, beta) + volvol * volvol
    # else:
    #     raise NotImplementedError(f"beta can be either scalar or vector")
    vartheta2 = np.dot(beta, beta) + volvol * volvol
    qv = q * vartheta2
    qv2 = q2 * vartheta2

    # if underlying_type == UnderlyingType.FUTURES and expansion_order == ExpansionOrder.SECOND:
    #     raise NotImplementedError(f"Second order expansion for ED options is not supported yet")

    # if isinstance(a0, float):
    #     a_prod_beta = a0 * beta
    #     a_prod_a = a0 * a0
    if underlying_type == UnderlyingType.FUTURES:
        a_prod_beta = np.dot(a0, beta) + a1 * volvol
        # a_prod_beta = np.dot(a, beta)
        a_prod_a = np.dot(a0, a0) + a1*a1
    elif underlying_type == UnderlyingType.SWAP:
        a_prod_beta = np.dot(a0, beta)
        a_prod_a = np.dot(a0, a0)
    else:
        raise NotImplementedError(f"beta can be either scalar or vector")

    # fill Ms: M should be of same type as L and H for numba, even though they are real
    # utilize that M is symmetric
    # although tuple is more intuitive for M, because of numba we need to make it a tensor
    n = get_expansion_n(expansion_order=expansion_order)
    M = np.zeros((n, n, n), dtype=np.complex128)
    M[0, 1, 1] = 0.5 * qv2

    M[1, 1, 1] = qv
    M[1, 1, 2] = M[1, 2, 1] = qv2

    M[2, 1, 1] = 0.5 * vartheta2
    M[2, 2, 2] = 2.0 * qv2
    M[2, 2, 1] = M[2, 1, 2] = 2.0 * qv

    # if expansion_order == ExpansionOrder.SECOND:
    #     raise ValueError("Second-order expansion not supported. TODO")
    if expansion_order == ExpansionOrder.SECOND:
        M[2, 1, 3] = M[2, 3, 1] = 1.5*qv2

        M[3, 2, 2] = 4.0*qv
        M[3, 1, 2] = M[3, 2, 1] = vartheta2
        M[3, 1, 3] = M[3, 3, 1] = 3.0*qv
        M[3, 1, 4] = M[3, 4, 1] = 2.0 * qv2
        M[3, 2, 3] = M[3, 3, 2] = 3.0 * qv2

        M[4, 2, 2] = 2.0 * vartheta2
        M[4, 3, 3] = 4.5 * qv2
        M[4, 1, 3] = M[4, 3, 1] = 1.5 * vartheta2
        M[4, 1, 4] = M[4, 4, 1] = 4.0 * qv
        M[4, 2, 3] = M[4, 3, 2] = 6.0 * qv
        M[4, 2, 4] = M[4, 4, 2] = 4.0 * qv2

    # fills Ls
    L = np.zeros((n, n), dtype=np.complex128)
    L[0, 1] = kappa0 - q2 * a_prod_beta * phi
    L[0, 2] = qv2
    L[1, 1] = -kappa1 - 2.0 * q * a_prod_beta * phi
    L[1, 2] = 2.0 * (kappa0 + qv - q2 * a_prod_beta * phi)
    L[2, 1] = -kappa2 - a_prod_beta * phi
    L[2, 2] = vartheta2 - 2.0 * kappa1 - 4.0 * q * a_prod_beta * phi

    # if expansion_order == ExpansionOrder.SECOND:
    #     raise ValueError("Second-order expansion not supported. TODO")
    if expansion_order == ExpansionOrder.SECOND:
        L[1, 3] = 3.0*qv2
        L[2, 3] = 3.0 * (kappa0 - q2 * a_prod_beta * phi + 2.0 * qv)
        L[2, 4] = 6.0 * qv2
        L[3, 2] = -2.0 * (kappa2 + a_prod_beta * phi)
        L[3, 3] = 3.0 * (vartheta2 - kappa1 - 2.0 * q * a_prod_beta * phi)
        L[3, 4] = 4.0 * (3.0 * qv + kappa0 - q2 * a_prod_beta * phi)
        L[4, 3] = -3.0 * (kappa2 + a_prod_beta * phi)
        L[4, 4] = 2.0 * (3.0 * vartheta2 - 2.0 * kappa1 - 4.0 * q * a_prod_beta * phi)

    # fill Hs
    H = np.zeros(n, dtype=np.complex128)

    H[0] = 0.5 * q2 * phi * (2.0 * b + a_prod_a * phi)
    H[1] = q * phi * (2.0 * b + a_prod_a * phi)
    H[2] = 0.5 * phi * (2.0 * b + a_prod_a * phi)

    return M, L, H
