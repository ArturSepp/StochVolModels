"""
numba analytics for affine expansion
"""

import numpy as np
import numpy.linalg as la
from numba import njit
from enum import Enum
from typing import Tuple, Optional
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from ...pricers.core.config import VariableType


class ExpansionOrder(Enum):
    FIRST = 1
    SECOND = 2


@njit(cache=False, fastmath=True)
def get_expansion_n(expansion_order: ExpansionOrder = ExpansionOrder.FIRST) -> int:
    if expansion_order == ExpansionOrder.FIRST:
        n = 3
    else:
        n = 5
    return n


@njit(cache=False, fastmath=True)
def func_a_ode_quadratic_terms(theta: float,
                               kappa1: float,
                               kappa2: float,
                               beta: float,
                               volvol: float,
                               phi: np.complex128,
                               psi: np.complex128,
                               is_spot_measure: bool = True,
                               expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Matrices for the quadratic form A_t = A.T@M@A + L@A + H
    """
    theta2 = theta * theta
    vartheta2 = beta * beta + volvol * volvol
    qv = theta * vartheta2
    qv2 = theta2 * vartheta2
    if is_spot_measure:
        lamda = 0
        kappa_p = kappa1 + kappa2 * theta
        kappa2_p = kappa2
    else:
        lamda = beta*theta2
        kappa_p = kappa1 + kappa2 * theta - 2*beta*theta
        kappa2_p = kappa2-beta

    # fill Ms: M should be of same type as L and H for numba, eventhough they are real
    # utilize that M is symmetric
    # although tuple is more intuitive for M, because of numba we need to make it a tensor
    n = get_expansion_n(expansion_order=expansion_order)
    M = np.zeros((n, n, n), dtype=np.complex128)
    M[0, 1, 1] = 0.5 * qv2

    M[1, 1, 1] = qv
    M[1, 1, 2] = M[1, 2, 1] = qv2

    M[2, 1, 1], M[2, 2, 2] = 0.5 * vartheta2, 2.0 * qv2
    M[2, 2, 1] = M[2, 1, 2] = 2.0 * qv

    if expansion_order == ExpansionOrder.SECOND:
        M[2, 1, 3] = M[2, 3, 1] = 1.5*qv2
        M[3, 2, 2] = 4.0*qv
        M[3, 1, 2] = M[3, 2, 1] = vartheta2
        M[3, 1, 3] = M[3, 3, 1] = 3.0*qv
        M[3, 1, 4] = M[3, 4, 1] = 2.0 * qv2
        M[3, 2, 3] = M[3, 3, 2] = 3.0 * qv2

        M[4, 2, 2], M[4, 3, 3] = 2.0 * vartheta2, 4.5*qv2
        M[4, 1, 3] = M[4, 3, 1] = 1.5 * vartheta2
        M[4, 1, 4] = M[4, 4, 1] = 4.0 * qv
        M[4, 2, 3] = M[4, 3, 2] = 6.0 * qv
        M[4, 2, 4] = M[4, 4, 2] = 4.0 * qv2

    # fills Ls
    L = np.zeros((n, n), dtype=np.complex128)
    L[0, 1], L[0, 2] = lamda - theta2 * beta * phi, qv2
    L[1, 1], L[1, 2] = -kappa_p - 2.0 * theta * beta * phi, 2.0 * (lamda + qv - theta2 * beta * phi)
    L[2, 1], L[2, 2] = -kappa2_p - beta * phi, vartheta2 - 2.0 * kappa_p - 4.0 * theta * beta * phi

    if expansion_order == ExpansionOrder.SECOND:
        L[1, 3] = 3.0*qv2
        L[2, 3], L[2, 4] = 3.0 * (2.0 * qv - theta2 * beta * phi), 6.0 * qv2
        L[3, 2], L[3, 3], L[3, 4] = -2.0 * (kappa2_p + beta * phi), 3.0 * (vartheta2 - kappa_p - 2.0 * theta * beta * phi), 4.0 * (3.0 * qv - theta2 * beta * phi)
        L[4, 3], L[4, 4] = -3.0 * (kappa2_p + beta * phi), 2.0 * (vartheta2 - 2.0 * kappa_p - 4.0 * theta * beta * phi)

    # fill Hs
    H = np.zeros(n, dtype=np.complex128)
    if is_spot_measure:
        rhs = (phi * (phi + 1.0) - 2.0 * psi)
    else:
        rhs = (phi * (phi - 1.0) - 2.0 * psi)
    H[0], H[1], H[2] = 0.5 * theta2 * rhs, theta * rhs, 0.5 * rhs

    return M, L, H


@njit(cache=False, fastmath=True)
def func_rhs(t: float,   # for ode solver compatibility
             A0: np.ndarray,
             M: Tuple[np.ndarray],
             L: np.ndarray,
             H: np.ndarray
             ) -> np.ndarray:
    """
    returns rhs evaluation using matrices for the quadratic form A_t = A.T@M@A + L@A + H
    """
    n = A0.shape[0]
    quadratic = np.zeros(n, dtype=np.complex128)
    for n_ in np.arange(n):
        quadratic[n_] = A0.T @ M[n_] @ A0
    rhs = quadratic + L @ A0 + H
    return rhs


@njit(cache=False, fastmath=True)
def func_rhs_jac(t: float,   # for ode solver compatibility
                 A0: np.ndarray,
                 M: Tuple[np.ndarray],
                 L: np.ndarray,
                 H: np.ndarray
                 ) -> np.ndarray:
    """
    returns rhs jacobian evaluation using matrices for the quadratic form A_t = A.T@M@A + L@A + H
    """
    n = A0.shape[0]
    quadratic = np.zeros((n, n), dtype=np.complex128)
    for n_ in np.arange(n):
        quadratic[n_, :] = 2.0 * M[n_] @ A0
    rhs = quadratic + A0
    return rhs


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
def solve_ode_for_a(ttm: float,
                    theta: float,
                    kappa1: float,
                    kappa2: float,
                    beta: float,
                    volvol: float,
                    phi: np.complex128,
                    psi: np.complex128,
                    is_spot_measure: bool = True,
                    a_t0: Optional[np.ndarray] = None,
                    expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                    is_stiff_solver: bool = False,
                    dense_output: bool = False
                    ) -> OdeResult:
    """
    solve ode for given phi
    next: numba implementation to compute in range of phi
    """
    M, L, H = func_a_ode_quadratic_terms(theta=theta,
                                         kappa1=kappa1,
                                         kappa2=kappa2,
                                         beta=beta,
                                         volvol=volvol,
                                         phi=phi,
                                         psi=psi,
                                         expansion_order=expansion_order,
                                         is_spot_measure=is_spot_measure)

    if a_t0 is None:
        a_t0 = np.zeros_like(H, dtype=np.complex128)

    if is_stiff_solver:
        ode_sol = solve_ivp(fun=func_rhs, t_span=(0.0, ttm), y0=a_t0, args=(M, L, H),
                            method='BDF',
                            jac=func_rhs_jac,
                            dense_output=dense_output)
    else:
        ode_sol = solve_ivp(fun=func_rhs, t_span=(0.0, ttm), y0=a_t0, args=(M, L, H),
                            dense_output=dense_output)

    return ode_sol


@njit(cache=False, fastmath=True)
def solve_analytic_ode_for_a(ttm: float,
                             theta: float,
                             kappa1: float,
                             kappa2: float,
                             beta: float,
                             volvol: float,
                             phi: np.complex128,
                             psi: np.complex128,
                             is_spot_measure: bool,
                             a_t0: Optional[np.ndarray] = None,
                             expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                             year_days: int = 260
                             ) -> np.ndarray:
    """
    solve ode for given phi
    """
    M, L, H = func_a_ode_quadratic_terms(theta=theta,
                                         kappa1=kappa1,
                                         kappa2=kappa2,
                                         beta=beta,
                                         volvol=volvol,
                                         phi=phi,
                                         psi=psi,
                                         expansion_order=expansion_order,
                                         is_spot_measure=is_spot_measure)

    nb_steps = int(np.ceil(year_days * ttm))  # daily steps using 260 in year
    dt = ttm / nb_steps

    if a_t0 is None:
        a_t0 = np.zeros_like(H, dtype=np.complex128)
    quadratic = np.zeros_like(a_t0, dtype=np.complex128)

    w, v = la.eig(L)
    v_inv = la.inv(v)
    v_lambda = v @ np.diag(np.exp(w * dt)) @ v_inv

    reciprocal = np.reciprocal(w)
    reciprocal[0] = 0.0
    m_rhs = v @ np.diag(reciprocal*(np.exp(w * dt) - np.ones_like(H))) @ v_inv

    # fixed point
    nfp = 10
    n = H.shape[0]
    for t in np.arange(0, nb_steps):
        A_fp0 = a_t0
        for _ in np.arange(nfp):
            # for idx, m in enumerate(M):
            for n_ in np.arange(n):
                quadratic[n_] = A_fp0.T @ M[n_] @ A_fp0
            # rhs = m_rhs @ (H+quadratic)
            rhs = m_rhs @ H + quadratic * dt
            rhs[0] = (H[0] + quadratic[0])*dt
            A_fp0 = v_lambda @ a_t0 + rhs
        a_t0 = A_fp0

    return a_t0


@njit(cache=False, fastmath=True)
def solve_analytic_ode_for_a0(t_span: Tuple[float, float],
                              theta: float,
                              kappa1: float,
                              kappa2: float,
                              beta: float,
                              volvol: float,
                              phi: np.complex128,
                              psi: np.complex128,
                              expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                              ) -> np.ndarray:
    """
    solve ode for given phi - too slow
    """
    M, L, H = func_a_ode_quadratic_terms(theta=theta,
                                         kappa1=kappa1,
                                         kappa2=kappa2,
                                         beta=beta,
                                         volvol=volvol,
                                         phi=phi,
                                         psi=psi,
                                         expansion_order=expansion_order)

    nt = 100
    dt = (t_span[1]-t_span[0]) / nt

    A0 = np.zeros_like(H, dtype=np.complex128)
    quadratic = np.zeros_like(A0, dtype=np.complex128)

    w, v = la.eig(L)
    v_inv = la.inv(v)
    v_lambda = v @ np.diag(np.exp(w * dt)) @ v_inv

    reciprocal = np.reciprocal(w)
    reciprocal[0] = 0.0
    m_rhs = v @ np.diag(reciprocal*(np.exp(w * dt) - np.ones_like(H))) @ v_inv

    # fixed point
    nfp = 2
    n = H.shape[0]
    Lq = np.zeros((n,n), dtype=np.complex128)

    for t in np.arange(0, nt):

        A_fp0 = A0
        for _ in np.arange(nfp):
            for n_ in np.arange(n):
                Lq[n_] = A_fp0.T @ M[n_]

            Lp = L + Lq
            w, v = la.eig(Lp)
            v_inv = la.inv(v)
            v_lambda = v @ np.diag(np.exp(w * dt)) @ v_inv

            reciprocal = np.reciprocal(w)
            reciprocal[0] = 0.0
            m_rhs = v @ np.diag(reciprocal * (np.exp(w * dt) - np.ones_like(H))) @ v_inv

            rhs = m_rhs @ H
            rhs[0] = H[0]*dt
            A_fp0 = v_lambda @ A0 + rhs

        A0 = A_fp0

    return A0


@njit(cache=False, fastmath=True)
def solve_analytic_ode_grid_phi(phi_grid: np.ndarray,
                                psi_grid: np.ndarray,
                                ttm: float,
                                theta: float,
                                kappa1: float,
                                kappa2: float,
                                beta: float,
                                volvol: float,
                                is_spot_measure: bool = True,
                                a_t0: Optional[np.ndarray] = None,
                                expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                                ) -> np.ndarray:
    """
    solve ode for range phi
    next: numba implementation to compute in range of phi
    """
    if a_t0 is None:
        a_t0 = np.zeros((phi_grid.shape[0], get_expansion_n(expansion_order)), dtype=np.complex128)

    f = lambda phi, psi, a0: solve_analytic_ode_for_a(ttm=ttm,
                                                      theta=theta,
                                                      kappa1=kappa1,
                                                      kappa2=kappa2,
                                                      beta=beta,
                                                      volvol=volvol,
                                                      phi=phi,
                                                      psi=psi,
                                                      a_t0=a0,
                                                      expansion_order=expansion_order,
                                                      is_spot_measure=is_spot_measure)

    for idx, (phi, psi) in enumerate(zip(phi_grid, psi_grid)):
        a_t0[idx, :] = f(phi, psi, a_t0[idx, :])

    return a_t0


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
def solve_a_ode_grid(phi_grid: np.ndarray,
                     psi_grid: np.ndarray,
                     ttm: float,
                     theta: float,
                     kappa1: float,
                     kappa2: float,
                     beta: float,
                     volvol: float,
                     is_spot_measure: bool = True,
                     a_t0: Optional[np.ndarray] = None,
                     is_stiff_solver: bool = False,
                     expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                     ) -> np.ndarray:
    """
    solve ode for range phi
    next: numba implementation to compute in range of phi
    """
    if a_t0 is None:
        a_t0 = np.zeros((phi_grid.shape[0], get_expansion_n(expansion_order)), dtype=np.complex128)

    f = lambda phi, psi, a0_: solve_ode_for_a(ttm=ttm,
                                              theta=theta,
                                              kappa1=kappa1,
                                              kappa2=kappa2,
                                              beta=beta,
                                              volvol=volvol,
                                              phi=phi,
                                              psi=psi,
                                              a_t0=a0_,
                                              is_stiff_solver=is_stiff_solver,
                                              dense_output=False,
                                              expansion_order=expansion_order,
                                              is_spot_measure=is_spot_measure)

    a_t1 = np.zeros((phi_grid.shape[0], get_expansion_n(expansion_order)), dtype=np.complex128)
    for idx, (phi, psi) in enumerate(zip(phi_grid, psi_grid)):
        a_t1[idx, :] = f(phi, psi, a_t0[idx, :]).y[:, -1]

    return a_t1


@njit(cache=False, fastmath=True)
def get_init_conditions_a(phi_grid: np.ndarray,
                          psi_grid: np.ndarray,
                          theta_grid: np.ndarray,
                          n_terms: int,
                          variable_type: VariableType = VariableType.LOG_RETURN
                          ) -> np.ndarray:
    """
    compute grid for a(0)
    """
    if variable_type == VariableType.LOG_RETURN:
        a_t0 = np.zeros((phi_grid.shape[0], n_terms), dtype=np.complex128)
    elif variable_type == VariableType.Q_VAR:
        a_t0 = np.zeros((psi_grid.shape[0], n_terms), dtype=np.complex128)
    elif variable_type == VariableType.SIGMA:
        a_t0 = np.zeros((theta_grid.shape[0], n_terms), dtype=np.complex128)
        a_t0[:, 1] = -theta_grid
    else:
        raise NotImplementedError
    return a_t0


def compute_logsv_a_mgf_grid(ttm: float,
                             phi_grid: np.ndarray,
                             psi_grid: np.ndarray,
                             theta_grid: np.ndarray,
                             sigma0: float,
                             theta: float,
                             kappa1: float,
                             kappa2: float,
                             beta: float,
                             volvol: float,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                             a_t0: Optional[np.ndarray] = None,
                             is_stiff_solver: bool = False,
                             is_analytic: bool = False,
                             is_spot_measure: bool = True,
                             **kwargs
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute
     1. solution a_t1 for ode A given a_t0
     2. log mgf function: we save an exponent calulation when pricing options
    mmg in x or QV as function of phi
    ode_solution is computed per grid of phi
    to do: numba implementation: need numba ode solvers
    """

    if a_t0 is None:
        a_t0 = get_init_conditions_a(phi_grid=phi_grid,
                                     psi_grid=psi_grid,
                                     theta_grid=theta_grid,
                                     n_terms=get_expansion_n(expansion_order=expansion_order),
                                     variable_type=variable_type)

    if is_analytic:
        a_t1 = solve_analytic_ode_grid_phi(phi_grid=phi_grid,
                                           psi_grid=psi_grid,
                                           ttm=ttm,
                                           theta=theta,
                                           kappa1=kappa1,
                                           kappa2=kappa2,
                                           beta=beta,
                                           volvol=volvol,
                                           a_t0=a_t0,
                                           expansion_order=expansion_order,
                                           is_spot_measure=is_spot_measure)
    else:
        a_t1 = solve_a_ode_grid(phi_grid=phi_grid,
                                psi_grid=psi_grid,
                                ttm=ttm,
                                theta=theta,
                                kappa1=kappa1,
                                kappa2=kappa2,
                                beta=beta,
                                volvol=volvol,
                                a_t0=a_t0,
                                is_stiff_solver=is_stiff_solver,
                                expansion_order=expansion_order,
                                is_spot_measure=is_spot_measure)

    y = sigma0 - theta
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
