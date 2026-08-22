
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import norm
from typing import Optional
from enum import Enum

from .utils import h, ncdf


@njit
def solve_volterra(upsilon_k: np.ndarray,
                   barrier_k: np.ndarray,
                   x0: float,
                   barrier_dt_k: Optional[np.ndarray] = None
                   ) -> np.ndarray:
    """
    numerical solution of eq 29
    compute functions on upsilon_k grid K(upsilon, upsilon') in eq 26 using functions in eq 20, 21
    """
    nt = upsilon_k.shape[0]

    # compute time derivative
    if barrier_dt_k is None:
        barrier_dt_k = np.zeros_like(barrier_k)
        for n, nu in enumerate(upsilon_k):
            if n > 0:
                barrier_dt_k[n] = (barrier_k[n] - barrier_k[n - 1]) / (upsilon_k[n] - upsilon_k[n - 1])
            else:
                barrier_dt_k[0] = (barrier_k[1] - barrier_k[0]) / (upsilon_k[1] - upsilon_k[0])

    #  1. fill theta and psi
    ker1_theta_2d = np.ones((nt, nt))
    ker2_psi_2d = np.zeros((nt, nt))
    delta_2d = np.zeros((nt, nt))
    pi_k_l = np.zeros((nt, nt))
    for k, nu in enumerate(upsilon_k):
        for l, nu_ in enumerate(upsilon_k[:k]):
            ker1_theta_2d[k, l] = -(barrier_k[k]-barrier_k[l])/(nu-nu_)
            ker2_psi_2d[k, l] = np.exp(-0.5 * (nu - nu_) * np.square(ker1_theta_2d[k, l]))
            delta_2d[k, l] = nu - nu_
            pi_k_l[k, l] = delta_2d[l, l - 1] / (np.sqrt(delta_2d[k, l-1]) + np.sqrt(delta_2d[k, l]))
        ker1_theta_2d[k, k] = -barrier_dt_k[k]
        ker2_psi_2d[k, k] = 1.0

    kernel_k_l = ker1_theta_2d * ker2_psi_2d / np.sqrt(2.0*np.pi)

    # 2. fill f
    f_k = h(upsilon_k, barrier_k + x0)
    f_k[0] = 0.0  # numerically, h(0, y) is nan
    nu_k = np.zeros_like(upsilon_k)  # solution
    nu_k[0] = f_k[0]  # nb: here we are setting nu_k to zero
    nu_k[1] = f_k[1] / (1.0+kernel_k_l[1, 1]*np.sqrt(upsilon_k[1]-upsilon_k[0]))
    delta_2d = np.sqrt(delta_2d)  # to sqrt
    for k in np.arange(2, upsilon_k.shape[0]):
        aux = 0.0
        for l in np.arange(1, k):
            aux = aux + pi_k_l[k, l]*(kernel_k_l[k, l] * nu_k[l]+kernel_k_l[k, l-1] * nu_k[l-1])
        nu_k[k] = (f_k[k] - delta_2d[k, k-1] * kernel_k_l[k, k - 1] * nu_k[k - 1] - aux) / (1.0 + delta_2d[k, k-1] * kernel_k_l[k, k])  # eq 31

    return nu_k


#@njit
def compute_f(upsilon_k: np.ndarray,
              barrier_k: np.ndarray,
              phi_k: np.ndarray,
              y: np.ndarray,
              psi0: float,  # = log(B/S0)
              is_new: bool = True
              ) -> np.ndarray:
    """
    implementation of eq 23
    """
    f = np.zeros(y.shape[0])
    zs = y - psi0
    if is_new:
        # final point will be u = up, initial point u = 0 is dropped
        # integration defined for  up to
        du = upsilon_k[1:] - upsilon_k[:-1]
        delta = upsilon_k[-1] - upsilon_k[:-1]
        delta_inv = np.reciprocal(delta)
        delta_inv_sqrt = np.sqrt(delta_inv)
        barrier_k_dt = -(barrier_k[1:] - barrier_k[:-1])/(upsilon_k[1:]-upsilon_k[:-1])
        theta_k = -(barrier_k[-1] - barrier_k[:-1])/delta
        psi_k = np.append(np.exp(-0.5*delta[:-1]*np.square(theta_k[:-1])), 1.0)
        for idx, z in enumerate(zs):  # compute integral for grid of y
            exp_zn = np.exp(z * barrier_k_dt[-1])
            term1 = 2.0 * exp_zn * phi_k[-1] * ncdf(-z / np.sqrt(upsilon_k[-1]))
            i1 = (z * np.exp(-0.5 * np.square(z) * delta_inv) ) * (np.exp(-z * theta_k)*psi_k*phi_k[:-1]-exp_zn * phi_k[-1])*delta_inv#negative
            i2 = theta_k*np.exp(-0.5*np.square(z-barrier_k[-1]+barrier_k[:-1])*delta_inv) * phi_k[:-1]
            f[idx] = term1 + np.nansum(((i1+i2)*delta_inv_sqrt*du))/np.sqrt(2.0*np.pi)  # trapezoid
    else:

        du = np.append(upsilon_k[1:] - upsilon_k[:-1], 0.0)
        delta = upsilon_k[-1] - upsilon_k
        scaler = np.reciprocal(np.sqrt(2.0 * np.pi * np.power(delta, 3)))
        for idx, z in enumerate(zs):  # compute integral for grid of y
            intg = (z - barrier_k[-1] + barrier_k)*np.exp(-0.5*np.square(z - barrier_k[-1] + barrier_k)/delta)*scaler
            f[idx] = np.nansum(intg * phi_k * du)  # trapezoid
    return f


#@njit
def compute_green(upsilon_k: np.ndarray,
                  barrier_k: np.ndarray,
                  x_grid: np.ndarray,
                  psi0: float, # = log(B/S0)
                  x0: float = 0.0,  # = log(S/S0)
                  n_t: float = 0.0,  # deterministic drift part = M_t
                  is_terminal: bool = False
                  ) -> np.ndarray:

    nu_k = solve_volterra(upsilon_k=upsilon_k,
                          barrier_k=-psi0 + barrier_k,
                          x0=x0,
                          barrier_dt_k=None)

    f = compute_f(upsilon_k=upsilon_k,
                  barrier_k=barrier_k,
                  phi_k=nu_k,
                  psi0=psi0,
                  y=x_grid)

    if is_terminal:
        ht = h(p=upsilon_k[-1] * np.ones_like(x_grid), q=x_grid - n_t)
    else:
        ht = h(p=upsilon_k * np.ones_like(x_grid), q=x_grid - n_t)
    green = ht-f
    # need to display
    #green = np.where(np.greater(x_grid, -barrier_k[-1]), green, np.nan)

    return green


def compute_analytic(upsilon_k: np.ndarray, x0: float, omega: float):
    this = np.sqrt(np.reciprocal(upsilon_k, where=np.greater(upsilon_k, 0.0)))
    analytic = h(p=upsilon_k, q=x0 + omega * upsilon_k) + omega * np.exp(-2.0 * omega * x0) * norm.cdf(-(x0 - omega * upsilon_k) * this)
    return analytic


def compute_analytic_f(upsilon_k: float, y: np.ndarray, x0: float, omega: float) -> np.ndarray:
    f = np.zeros(y.shape[0])
    for idx, y_ in enumerate(y):  # compute integral for grid of y
        f[idx] = np.exp(-2.0*omega*x0-0.5*np.square(y_+x0)/upsilon_k)/np.sqrt(2.0*np.pi*upsilon_k)
    return f


class LocalTests(Enum):
    PLOT_SOLUTION = 1
    PLOT_SOLUTION_F = 2
    SOLVE_BARRIER = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    upsilon = 1.0  #  constant time to maturity T
    n = 260
    x0 = 1.0
    upsilon_k = np.linspace(0.0, upsilon, n)
    omega = 0.5
    barrier_k = 1.0 - np.exp(-omega * upsilon_k)  # barrier function
    barrier_dt_k = omega * np.exp(-omega * upsilon_k)  # barrier function
    barrier_k = omega * upsilon_k  # barrier function
    barrier_dt_k = omega * np.ones_like(upsilon_k)

    nu_k = solve_volterra(upsilon_k=upsilon_k,
                          barrier_k=barrier_k,
                          x0=x0,
                          barrier_dt_k=barrier_dt_k)

    if local_test == LocalTests.PLOT_SOLUTION:
        numeric = pd.Series(nu_k, index=upsilon_k, name='numeric')
        analytic = compute_analytic(upsilon_k=upsilon_k, x0=x0, omega=omega)
        analytic = pd.Series(analytic, index=upsilon_k, name='analytic')
        data = pd.concat([numeric, analytic], axis=1)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            sns.lineplot(data=data, dashes=False, ax=axs[0])
            axs[0].set_xlabel('t')
            axs[0].set_ylabel('f')
            sns.lineplot(data=np.subtract(numeric, analytic), dashes=False, ax=axs[1])
            axs[1].set_xlabel('t')
            axs[1].set_ylabel('f')

    elif local_test == LocalTests.PLOT_SOLUTION_F:

        y = np.linspace(0.0, 5.0, 1000)
        f = compute_f(upsilon_k=upsilon_k,
                      barrier_k=barrier_k,
                      phi_k=nu_k,
                      y=y)
        numeric = pd.Series(f, index=y, name='numeric')
        analytic = compute_analytic_f(upsilon_k=upsilon_k[-1], y=y, x0=x0, omega=omega)
        analytic = pd.Series(analytic, index=y, name='analytic')
        data = pd.concat([numeric, analytic], axis=1)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            sns.lineplot(data=data, dashes=False, ax=axs[0])
            axs[0].set_xlabel('t')
            axs[0].set_ylabel('f')
            sns.lineplot(data=np.subtract(numeric, analytic), dashes=False, ax=axs[1])
            axs[1].set_xlabel('t')
            axs[1].set_ylabel('f')

    elif local_test == LocalTests.SOLVE_BARRIER:
        y = np.linspace(0.0, 5.0, 1000)
        f = compute_f(upsilon_k=upsilon_k,
                      barrier_k=barrier_k,
                      phi_k=nu_k,
                      y=y)

        f = np.where(y>barrier_k[-1], f, np.nan)
        bar = pd.Series(f, index=y, name='Volterra Sol Barrier')

        green = h(p=upsilon * np.ones_like(y), q=y - x0)
        green = np.where(y>barrier_k[-1], green, np.nan)
        green = pd.Series(green, index=y, name='Green 0')

        barrier_sol = np.where(y>barrier_k[-1], green-f, np.nan)
        barrier_sol = pd.Series(barrier_sol, index=y, name='Green - Barrier')

        data = pd.concat([bar, green, barrier_sol], axis=1)
        print(data)
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sns.lineplot(data=data, dashes=False, ax=ax)
            ax.set_xlabel('y')
            ax.set_ylabel('pdf')

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.PLOT_SOLUTION_F

    run_local_test(local_test=local_test)
