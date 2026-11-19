"""
Moments of the volatility process and the expected quadratic variance.

Closed-form truncation solution of Proposition 3.3 in Sepp and Rakhmonov (2024).
The mean-adjusted volatility Y_t = sigma_t - theta of Eq. (3.32) has moments
m_bar^(n)(tau) = E[Y_tau^n] obeying the recursion of Eq. (3.50), which closes
into the finite linear system of Eq. (3.48) once truncated at order k*. That
system is solved by matrix exponentiation in Eq. (3.49), and integrated over
[0, tau] in Eq. (3.54) to give the expected quadratic variance of Eq. (3.53).

Manual scenarios are available in
``stochvolmodels.pricers.logsv.run_local.vol_moments_ode_run``.

Reference
---------
A. Sepp and P. Rakhmonov (2023), Log-normal Stochastic Volatility Model with
Quadratic Drift, International Journal of Theoretical and Applied Finance 26(8),
2450003. Equation numbers throughout this module refer to that article.
"""

# packages
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla

# project
from stochvolmodels.models.logsv.params import LogSvParams


def compute_analytic_vol_moments(
    params: LogSvParams, t: float = 1.0, n_terms: int = 4, is_qvar: bool = False
) -> np.ndarray:
    """
    solve the truncated moment system for the mean-adjusted volatility.

    Integrates ``d_tau M = Lambda M + C`` of Eq. (3.48) in closed form. With
    ``is_qvar=False`` this returns the moment vector M^(1,k*)(tau) of Eq. (3.49),

        M(tau) = expm(Lambda tau) M(0) + Lambda^-1 (expm(Lambda tau) - I) C.

    With ``is_qvar=True`` it returns the time-integrated moments M_hat^(1,k*)(tau)
    of Eq. (3.54), which :func:`compute_analytic_qvar` consumes.

    Parameters
    ----------
    params : LogSvParams
        Model parameters. Only theta, kappa1, kappa2, beta, volvol and sigma0 enter.
    t : float, default 1.0
        Horizon tau in years. At t = 0 the initial vector M(0) is returned unchanged.
    n_terms : int, default 4
        Truncation order k* of Eq. (3.51).
    is_qvar : bool, default False
        Return time-integrated moments of Eq. (3.54) rather than the moments
        themselves.

    Returns
    -------
    np.ndarray, shape (n_terms,)
        Moments m_bar^(1) ... m_bar^(k*), or their integrals over [0, tau].

    Notes
    -----
    The initial vector is M(0) = (Y_0, Y_0^2, ..., Y_0^k*) with Y_0 = sigma0 - theta,
    and the free term is C = (0, c(2) theta^2, 0, ..., -k* kappa2 Y_0^(k*+1)), both
    as given in Eq. (3.48). Accuracy degrades with k*: Fig. 2 shows k* = 4 matching
    Monte Carlo on the first two moments and k* = 8 on the first four.
    """

    lambda_m = params.get_vol_moments_lambda(n_terms=n_terms)

    y = params.sigma0 - params.theta
    y0 = np.zeros(n_terms)
    for n in range(0, n_terms):
        y0[n] = np.power(y, n + 1)

    if np.isclose(np.abs(t), 0.0):
        return y0

    rhs = np.zeros(n_terms)
    rhs[1] = params.vartheta2 * params.theta2

    # closure of Eq. (3.51): the (k*+1)th moment is frozen at its initial value
    rhs[-1] = -n_terms * params.kappa2 * np.power(y, n_terms + 1)

    i_m = la.inv(lambda_m)
    is_expm = True
    if is_expm:
        e_m = sla.expm(lambda_m * t)
        m_rhs = i_m @ (e_m - np.eye(n_terms))
    else:
        w, v = la.eig(lambda_m)
        v_inv = la.inv(v)
        e_m = np.real(v @ np.diag(np.exp(w * t)) @ v_inv)
        m_rhs = np.real(v @ np.diag(np.reciprocal(w) * (np.exp(w * t) - np.ones(n_terms))) @ v_inv)
        # m_rhs = i_m @ (e_m - np.eye(n_terms))

    if is_qvar:
        sol1 = m_rhs @ y0
        intm2 = i_m @ (m_rhs - t * np.eye(n_terms))
        sol2 = intm2 @ rhs
    else:
        sol1 = e_m @ y0
        sol2 = m_rhs @ rhs

    sol = sol1 + sol2
    return sol


def compute_analytic_qvar(params: LogSvParams, ttm: float = 1.0, n_terms: int = 4) -> float:
    """
    annualized expected quadratic variance E[(1/tau) int_0^tau sigma_t^2 dt].

    Corollary 3.4, Eq. (3.53),

        I_hat_tau = (1/tau) (m_hat^(2)(tau) + 2 theta m_hat^(1)(tau)) + theta^2
                    + O(k*),

    where m_hat^(1) and m_hat^(2) are the first two integrated moments returned by
    :func:`compute_analytic_vol_moments` with ``is_qvar=True``, and O(k*) is the
    truncation error. This is the model fair value of a continuously monitored
    variance swap, usable for calibration.

    Parameters
    ----------
    params : LogSvParams
        Model parameters.
    ttm : float, default 1.0
        Time to maturity tau in years. At tau = 0 the value collapses to sigma0^2.
    n_terms : int, default 4
        Truncation order k*.

    Returns
    -------
    float
        Annualized expected quadratic variance.
    """
    if np.isclose(ttm, 0.0):
        qvar = np.square(params.sigma0)
    else:
        int_moments = compute_analytic_vol_moments(
            params=params, t=ttm, n_terms=n_terms, is_qvar=True
        )
        qvar = (int_moments[1] + 2.0 * params.theta * int_moments[0]) / ttm + params.theta2
    return qvar


def compute_vol_moments_t(
    params: LogSvParams, ttm: np.ndarray, n_terms: int = 4, is_print: bool = False
) -> np.ndarray:
    """evaluate :func:`compute_analytic_vol_moments` over an array of maturities."""
    moments = np.zeros((len(ttm), n_terms))
    for idx, t_ in enumerate(ttm):
        moments_ = compute_analytic_vol_moments(t=t_, params=params, n_terms=n_terms)
        if is_print:
            print(f"t={t_}: {moments_}")
        moments[idx, :] = moments_
    return moments


def compute_expected_vol_t(
    params: LogSvParams,
    t: np.ndarray,
    n_terms: int = 4,
) -> np.ndarray:
    """
    expected volatility E[sigma_tau] = E[Y_tau] + theta over an array of maturities.
    """
    ev = np.zeros(len(t))
    for idx, t_ in enumerate(t):
        moments = compute_analytic_vol_moments(t=t_, params=params, n_terms=n_terms)
        ev[idx] = moments[0] + params.theta
    return ev


def compute_sqrt_qvar_t(params: LogSvParams, t: np.ndarray, n_terms: int = 4) -> np.ndarray:
    """square root of the expected quadratic variance of Eq. (3.53), the model var-swap rate."""
    ev = np.zeros(len(t))
    for idx, t_ in enumerate(t):
        ev[idx] = np.sqrt(compute_analytic_qvar(ttm=t_, params=params, n_terms=n_terms))
    return ev


def get_vol_moments_lambda(params: LogSvParams, n_terms: int = 4) -> np.ndarray:
    """
    assemble the truncated moment generator Lambda^(1,k*) of Eq. (3.48).

    The mean-adjusted volatility Y_t = sigma_t - theta of Eq. (3.32) has moments
    m_bar^(n)(tau) = E[Y_tau^n] solving the recursion of Eq. (3.50). Truncating at
    order k* under the closure of Eq. (3.51) gives the finite linear system
    ``d_tau M = Lambda M + C`` of Eq. (3.48), whose generator this returns:

        row 1   (-kappa, -kappa2, 0, ...)
        row n   (c(n) theta^2, 2 c(n) theta, c(n) - n kappa, -n kappa2, ...)

    with c(n) = 0.5 vartheta^2 n (n - 1) from Eq. (3.46).

    Parameters
    ----------
    n_terms : int, default 4
        Truncation order k*. Fig. 2 compares k* = 4 and k* = 8 against Monte Carlo;
        k* = 4 matches the first two moments, k* = 8 the first four.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_terms, n_terms)``.
        Lower-Hessenberg generator Lambda^(1,k*).
    """

    kappa2 = params.kappa2
    kappa = params.kappa
    vartheta2 = params.vartheta2
    theta = params.theta
    theta2 = params.theta2

    def c(n: int) -> float:
        """Return the diffusion contribution to the order-``n`` moment."""
        return 0.5 * vartheta2 * n * (n - 1.0)

    lambda_m = np.zeros((n_terms, n_terms))
    lambda_m[0, 0] = -kappa
    lambda_m[0, 1] = -kappa2
    lambda_m[1, 0] = 2.0 * c(2) * theta
    lambda_m[1, 1] = c(2) - 2.0 * kappa
    lambda_m[1, 2] = -2.0 * kappa2

    for n_ in np.arange(2, n_terms):
        n = n_ + 1  # n_ is array counter, n is formula counter
        c_n = c(n)
        lambda_m[n_, n_ - 2] = c_n * theta2
        lambda_m[n_, n_ - 1] = 2.0 * c_n * theta
        lambda_m[n_, n_] = c_n - n * kappa
        if n_ + 1 < n_terms:
            lambda_m[n_, n_ + 1] = -n * kappa2

    return lambda_m


def assert_vol_moments_stability(params: LogSvParams, n_terms: int = 4) -> None:
    """
    print whether all eigenvalues of Lambda^(1,k*) have negative real part.

    The regularity condition under Eq. (3.50) that makes the truncation of
    Eq. (3.51) admissible. Despite the name this asserts nothing; it prints.
    """
    lambda_m = params.get_vol_moments_lambda(n_terms=n_terms)
    w, v = la.eig(lambda_m)
    cond = np.all(np.real(w) < 0.0)
    print(f"vol moments stable = {cond}")


def print_vol_moments_stability(params: LogSvParams, n_terms: int = 4) -> None:
    """
    print the per-moment diagonal conditions c(n) - n kappa and the spectrum of Lambda.
    """

    def c(n: int) -> float:
        """Return the diffusion contribution to the order-``n`` moment."""
        return 0.5 * params.vartheta2 * n * (n - 1.0)

    cond_m2 = c(2) - 2.0 * params.kappa
    print(f"con2:\n{cond_m2}")
    cond_m3 = c(3) - 3.0 * params.kappa
    print(f"con3:\n{cond_m3}")
    cond_m4 = c(4) - 4.0 * params.kappa
    print(f"cond4:\n{cond_m4}")

    lambda_m = params.get_vol_moments_lambda(n_terms=n_terms)
    print(f"lambda_m:\n{lambda_m}")

    w, v = la.eig(lambda_m)
    print(f"eigenvalues w:\n{w}")
    print(f"vol moments stable = {np.all(np.real(w) < 0.0)}")
