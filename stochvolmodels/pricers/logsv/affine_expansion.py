"""
Affine expansion of the moment generating function of the log-normal SV model.

Section 4 of Sepp and Rakhmonov (2024). The model of Eq. (3.12) is not affine, so
the MGF of Eq. (4.5) admits no exponential-affine solution with finitely many
coefficients. The expansion truncates the infinite ansatz of Eq. (4.13) at order
m, leaving a leading term

    E^[m](tau, Y; Phi, Psi, Theta; p) = exp{ sum_k A^(k)(tau) Y^k },

with k = 0, 1, 2 at first order in Eq. (4.16) and k = 0, ..., 4 at second order in
Eq. (4.24), where Y = sigma - theta is the mean-adjusted volatility of Eq. (3.32).
The coefficient vector A(tau) solves the quadratic ODE system

    A^(k)_tau = A' M^(k) A + (L^(k)(p))' A + H^(k)(p),

given in Eq. (4.17) for the first order and Eq. (4.25) for the second, with the
initial condition A(0) = (0, -Theta, 0, ...).

The binary parameter p selects the measure: p = 1 is the money-market account
measure Q with dynamics of Eq. (3.34), p = -1 the inverse measure Q~ with dynamics
of Eq. (3.36). It enters only through L^(k)(p) and H^(k)(p); the quadratic
matrices M^(k) do not depend on it (Remark 4.1).

Reference
---------
A. Sepp and P. Rakhmonov (2024), Log-normal Stochastic Volatility Model with
Quadratic Drift, International Journal of Theoretical and Applied Finance 26(8),
2450003. Equation numbers throughout this module refer to that article.
"""

import numpy as np
import numpy.linalg as la
from numba import njit
from enum import Enum
from typing import Tuple, Optional
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from stochvolmodels.utils.config import VariableType


class ExpansionOrder(Enum):
    """
    truncation order of the affine expansion of the MGF.

    FIRST is the leading term E^[1] of Eq. (4.16), carrying A^(0), A^(1), A^(2).
    SECOND is E^[2] of Eq. (4.24), carrying A^(0) through A^(4); it is the order
    used for option valuation in Sec. 6 and is the only one Propositions 4.4-4.6
    show to reproduce the variances of the state variables.
    """
    ZERO = 0
    FIRST = 1
    SECOND = 2


@njit(cache=False, fastmath=True)
def get_expansion_n(expansion_order: ExpansionOrder = ExpansionOrder.FIRST) -> int:
    """number of coefficients A^(k): 3 for the first order, Eq. (4.16), else 5, Eq. (4.24)."""
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
                               expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                               vol_backbone_eta: float = 1.0
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    assemble the matrices M^(k), L^(k)(p) and H^(k)(p) of the coefficient ODEs.

    Builds the right-hand side of Eq. (4.14),

        A^(k)_tau = A' M^(k) A + (L^(k)(p))' A + H^(k)(p),

    with entries given by Eq. (4.17) at first order and Eq. (4.25) at second.

    Parameters
    ----------
    theta, kappa1, kappa2, beta, volvol : float
        Model parameters of Eq. (3.12). Only the combination
        vartheta^2 = beta^2 + volvol^2 of Eq. (3.13) enters M^(k); beta enters
        L^(k) separately through the measure-dependent terms of Eq. (4.3).
    phi : np.complex128
        Transform variable Phi conjugate to the log-price X_tau in Eq. (4.5).
        Theorem 4.2 gives existence for Re(Phi) in (-1, 0) when p = 1 and in
        (0, 1) when p = -1; option valuation uses Re(Phi) = -1/2 in Eq. (5.4) and
        Re(Phi) = 1/2 in Eq. (5.13).
    psi : np.complex128
        Transform variable Psi conjugate to the quadratic variance I_tau. Theorem
        4.3 gives existence for Re(Psi) < 0 when kappa2 > vartheta sqrt(-2 Re Psi).
    is_spot_measure : bool, default True
        True selects p = 1, the MMA measure Q, so kappa_2^(p) = kappa2 and
        lambda^(p) = 0. False selects p = -1, the inverse measure Q~, so
        kappa_2^(p) = kappa2 - beta and lambda^(p) = beta theta^2, per Eq. (4.3).
    expansion_order : ExpansionOrder, default ExpansionOrder.FIRST
        Truncation order, which fixes the dimension n at 3 or 5.
    vol_backbone_eta : float, default 1.0
        Multiplicative scaling of the mean volatility theta at this maturity. Not
        part of the article, which fixes a single theta; 1.0 reproduces Eqs.
        (4.17) and (4.25) exactly.

    Returns
    -------
    M : np.ndarray, shape (n, n, n), complex
        Quadratic forms M^(k), stacked on the leading axis and symmetric in the
        trailing two. Independent of p (Remark 4.1).
    L : np.ndarray, shape (n, n), complex
        Linear terms L^(k)(p).
    H : np.ndarray, shape (n,), complex
        Free terms H^(k)(p), proportional to Phi^2 + p Phi - 2 Psi.
    """
    theta2 = theta * theta
    vartheta2 = beta * beta + volvol * volvol
    qv = theta * vartheta2
    qv2 = theta2 * vartheta2
    vol_backbone_eta2 = vol_backbone_eta * vol_backbone_eta
    if is_spot_measure:
        lamda = 0
        kappa2_p = kappa2
        kappa_p = kappa1 + kappa2 * theta
    else:
        lamda = beta*theta2*vol_backbone_eta
        kappa2_p = kappa2-beta*vol_backbone_eta
        kappa_p = kappa1 + kappa2 * theta - 2*beta*theta*vol_backbone_eta

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
    L[0, 1], L[0, 2] = lamda - theta2 * beta * vol_backbone_eta * phi, qv2
    L[1, 1], L[1, 2] = -kappa_p - 2.0 * theta * beta * vol_backbone_eta * phi, 2.0 * (lamda + qv - theta2 * beta * vol_backbone_eta * phi)
    L[2, 1], L[2, 2] = -kappa2_p - beta * vol_backbone_eta * phi, vartheta2 - 2.0 * kappa_p - 4.0 * theta * beta * vol_backbone_eta * phi

    if expansion_order == ExpansionOrder.SECOND:
        L[1, 3] = 3.0*qv2
        L[2, 3], L[2, 4] = 3.0 * (2.0 * qv - theta2 * beta * vol_backbone_eta * phi), 6.0 * qv2
        L[3, 2], L[3, 3], L[3, 4] = -2.0 * (kappa2_p + beta * vol_backbone_eta * phi), 3.0 * (vartheta2 - kappa_p - 2.0 * theta * beta * vol_backbone_eta * phi), 4.0 * (3.0 * qv - theta2 * beta * vol_backbone_eta * phi)
        L[4, 3], L[4, 4] = -3.0 * (kappa2_p + beta * vol_backbone_eta * phi), 2.0 * (vartheta2 - 2.0 * kappa_p - 4.0 * theta * beta * vol_backbone_eta * phi)

    # fill Hs
    H = np.zeros(n, dtype=np.complex128)
    if is_spot_measure:
        rhs = (phi * (phi + 1.0) - 2.0 * psi)
    else:
        rhs = (phi * (phi - 1.0) - 2.0 * psi)
    H[0], H[1], H[2] = 0.5*theta2 * vol_backbone_eta2 * rhs, theta * vol_backbone_eta2 * rhs, 0.5*vol_backbone_eta2 * rhs

    return M, L, H


@njit(cache=False, fastmath=True)
def func_rhs(t: float,   # for ode solver compatibility
             A0: np.ndarray,
             M: Tuple[np.ndarray],
             L: np.ndarray,
             H: np.ndarray
             ) -> np.ndarray:
    """
    right-hand side of the coefficient ODE system, Eq. (4.14).

    Signature ordered for ``scipy.integrate.solve_ivp``; ``t`` is unused because
    the system is autonomous in tau.
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
    Jacobian of :func:`func_rhs` with respect to A, for the stiff BDF solver.

    Signature ordered for ``scipy.integrate.solve_ivp``; ``t`` is unused.
    """
    n = A0.shape[0]
    quadratic = np.zeros((n, n), dtype=np.complex128)
    for n_ in np.arange(n):
        quadratic[n_, :] = 2.0 * M[n_] @ A0
    rhs = quadratic + L
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
                    dense_output: bool = False,
                    vol_backbone_eta: float = 1.0
                    ) -> OdeResult:
    """
    integrate the coefficient ODEs of Eq. (4.14) numerically for a single Phi.

    Parameters
    ----------
    ttm : float
        Time to maturity tau in years.
    theta, kappa1, kappa2, beta, volvol : float
        Model parameters of Eq. (3.12).
    phi, psi : np.complex128
        Transform variables Phi and Psi of Eq. (4.5).
    is_spot_measure : bool, default True
        True for the MMA measure (p = 1), False for the inverse measure (p = -1).
    a_t0 : Optional[np.ndarray], default None
        Initial condition A(0). None gives the zero vector, which is
        A(0) = (0, -Theta, 0, ...) of Eq. (4.17) with Theta = 0.
    expansion_order : ExpansionOrder, default ExpansionOrder.FIRST
        Truncation order.
    is_stiff_solver : bool, default False
        Use the BDF method with an analytic Jacobian rather than the default RK45.
    dense_output : bool, default False
        Request a continuous solution from the solver.
    vol_backbone_eta : float, default 1.0
        Maturity scaling of theta; 1.0 reproduces the article.

    Returns
    -------
    OdeResult
        Solver result; the coefficients at tau are ``ode_sol.y[:, -1]``.

    Notes
    -----
    Theorem 4.7 gives conditions under which A(tau) stays continuous on
    [0, tau_0); the solution of a quadratic system can otherwise blow up in
    finite time, and no check for that is performed here.
    """
    M, L, H = func_a_ode_quadratic_terms(theta=theta,
                                         kappa1=kappa1,
                                         kappa2=kappa2,
                                         beta=beta,
                                         volvol=volvol,
                                         phi=phi,
                                         psi=psi,
                                         expansion_order=expansion_order,
                                         is_spot_measure=is_spot_measure,
                                         vol_backbone_eta=vol_backbone_eta)

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
    integrate the coefficient ODEs semi-analytically for a single Phi.

    Steps on a daily grid, at each step advancing the linear part exactly by
    eigendecomposition of L and treating the quadratic part A' M^(k) A by fixed
    point iteration. This is the cheaper alternative to :func:`solve_ode_for_a`
    referred to in Sec. 6.1, where the cost of the ODE step is
    O(P N_max) in Eq. (6.1).

    Parameters
    ----------
    year_days : int, default 260
        Business days per year setting the step count ceil(year_days tau).

    Returns
    -------
    np.ndarray, shape (n,), complex
        Coefficient vector A(tau).

    Notes
    -----
    Does not accept ``vol_backbone_eta``, so it is fixed at the article's flat
    theta. The fixed point runs a fixed 10 iterations with no convergence test.
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
    integrate the coefficient ODEs over a time span with a dense solver. Superseded.

    Retained for reference; :func:`solve_analytic_ode_for_a` is the faster path.
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
    """apply :func:`solve_analytic_ode_for_a` across the transform grid."""
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
                     expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                     vol_backbone_eta: float = 1.0
                     ) -> np.ndarray:
    """apply :func:`solve_ode_for_a` across the transform grid, returning A(tau) per point."""
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
                                              is_spot_measure=is_spot_measure,
                                              vol_backbone_eta=vol_backbone_eta)

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
    initial condition A(0) of the coefficient ODEs, over the transform grid.

    Equations (4.17) and (4.25) set A(0) = (0, -Theta, 0, ...), where Theta is the
    transform variable conjugate to the mean-adjusted volatility Y_tau. For the
    log-return and quadratic variance the payoff transform enters through Phi and
    Psi and Theta is zero, so A(0) vanishes; only VariableType.SIGMA populates the
    second component with -Theta.

    Returns
    -------
    np.ndarray, shape (n_grid, n_terms), complex
        Initial coefficients, one row per transform grid point.

    Raises
    ------
    NotImplementedError
        If variable_type is not one of the three state variables of Eq. (3.34).
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
                             vol_backbone_eta: float = 1.0,
                             **kwargs
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    log of the affine expansion of the MGF, over a grid of transform variables.

    Solves the coefficient ODEs of Eq. (4.14) over the grid and contracts the
    result against the powers of the initial mean-adjusted volatility
    Y = sigma0 - theta to give

        log E^[m](tau, Y) = sum_k A^(k)(tau) Y^k,

    the exponent of Eq. (4.16) at first order and Eq. (4.24) at second. The
    exponential is deliberately not taken: the option valuation formulas of
    Eqs. (5.4), (5.13) and (5.20) consume E^[m] directly, and returning the log
    saves one exponentiation per grid point.

    Parameters
    ----------
    ttm : float
        Time to maturity tau in years.
    phi_grid, psi_grid, theta_grid : np.ndarray, complex
        Transform grids for Phi, Psi and Theta of Eq. (4.5). Sec. 6.1 uses about
        500 points for Phi.
    sigma0, theta, kappa1, kappa2, beta, volvol : float
        Model parameters of Eq. (3.12).
    variable_type : VariableType, default VariableType.LOG_RETURN
        State variable whose transform drives the initial condition.
    expansion_order : ExpansionOrder, default ExpansionOrder.SECOND
        Truncation order. SECOND is the order used throughout Sec. 6.
    a_t0 : Optional[np.ndarray], default None
        Initial coefficients; None calls :func:`get_init_conditions_a`.
    is_stiff_solver : bool, default False
        Use the BDF solver in the numerical path.
    is_analytic : bool, default False
        Take the semi-analytic fixed-point path rather than ``solve_ivp``.
    is_spot_measure : bool, default True
        True for the MMA measure (p = 1), False for the inverse measure (p = -1).
    vol_backbone_eta : float, default 1.0
        Maturity scaling of theta. Ignored on the analytic path.

    Returns
    -------
    a_t1 : np.ndarray, shape (n_grid, n_terms), complex
        Coefficients A(tau) on the grid.
    log_mgf : np.ndarray, shape (n_grid,), complex
        log E^[m](tau, Y) on the grid.

    Raises
    ------
    NotImplementedError
        If expansion_order is ExpansionOrder.ZERO.
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
                                is_spot_measure=is_spot_measure,
                                vol_backbone_eta=vol_backbone_eta)

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
