import time
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import linalg as la

from pricers.logsv.affine_expansion import ExpansionOrder, solve_analytic_ode_grid_phi, solve_a_ode_grid, \
    func_a_ode_quadratic_terms, func_rhs, func_rhs_jac
from pricers.logsv.logsv_params import LogSvParams


def compute_pdf_x(ttm: float,
                  params: LogSvParams,
                  x_grid: np.ndarray,
                  is_stiff_solver: bool = False,
                  is_analytic: bool = True,
                  expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                  ) -> pd.Series:
    """
    mmg in x as function of phi
    """
    PHI = - 0.0 + 1j * np.linspace(0, 50, 4000)
    dp = np.imag(PHI[1]-PHI[0])
    if is_analytic:
        ode_sol = solve_analytic_ode_grid_phi(phi_grid=PHI,
                                              psi_grid=np.zeros_like(PHI, dtype=np.complex128),
                                              ttm=ttm,
                                              theta=params.theta,
                                              kappa1=params.kappa1,
                                              kappa2=params.kappa2,
                                              beta=params.beta,
                                              volvol=params.volvol,
                                              expansion_order=expansion_order)
    else:
        ode_sol = solve_a_ode_grid(phi_grid=PHI,
                                   psi_grid=np.zeros_like(PHI, dtype=np.complex128),
                                   ttm=ttm,
                                   theta=params.theta,
                                   kappa1=params.kappa1,
                                   kappa2=params.kappa2,
                                   beta=params.beta,
                                   volvol=params.volvol,
                                   is_stiff_solver=is_stiff_solver,
                                   expansion_order=expansion_order)
    y = params.sigma0 - params.theta
    if expansion_order == ExpansionOrder.FIRST:
        ys = np.array([1.0, y, y*y])
    else:
        y2 = y*y
        ys = np.array([1.0, y, y2, y2*y, y2*y2])

    exp_part = ode_sol @ ys

    dx = x_grid[1] - x_grid[0]
    pdf_x = np.zeros_like(x_grid)
    for idx, x in enumerate(x_grid):
        mgf = np.real(np.exp(x*PHI+exp_part))
        pdf_x[idx] = 0.5*mgf[0] + np.nansum(mgf[1:])

    pdf_x = (dx*dp/np.pi)*pdf_x
    pdf_x = pd.Series(pdf_x, index=x_grid)
    return pdf_x


def plot_ode_sol_in_t(ttm: float,
                      ode_sol: Any
                      ) -> None:

    t = np.linspace(0.0, ttm, 300)
    z = ode_sol.sol(t)

    parts = {'real': pd.DataFrame(data=np.real(z).T, index=t, columns=['A0', 'A1', 'A2']),
             'imag': pd.DataFrame(data=np.imag(z).T, index=t, columns=['A0', 'A1', 'A2'])}

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)
        for (key, data), ax in zip(parts.items(), axs):
            sns.lineplot(data=data, ax=ax)
            ax.set_title(key)


def plot_ode_sol_in_phi(PHI: np.ndarray,
                        ode_sol: Any
                        ) -> None:

    k = np.imag(PHI)

    parts = {'real': pd.DataFrame(data=np.real(ode_sol), index=k, columns=['A0', 'A1', 'A2']),
             'imag': pd.DataFrame(data=np.imag(ode_sol), index=k, columns=['A0', 'A1', 'A2'])}

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), tight_layout=True)

        for (key, data), ax in zip(parts.items(), axs):
            sns.lineplot(data=data, ax=ax)
            ax.set_title(key)


class UnitTests(Enum):
    CHECK_A1 = 1
    CHECK_RHS = 2
    CHECK_ODE_GRID_PHI = 3
    CHECK_X_PDF = 4


def run_unit_test(unit_test: UnitTests):

    params = LogSvParams(sigma0=0.8306,
                         theta=1.008,
                         kappa1=5.4684,
                         kappa2=5.4248,
                         beta=0.1628,
                         volvol=2.6951)

    params.print_vol_moments_stability()

    if unit_test == UnitTests.CHECK_A1:

        M, L, H = func_a_ode_quadratic_terms(theta=params.theta,
                                             kappa1=params.kappa1,
                                             kappa2=params.kappa2,
                                             beta=params.beta,
                                             volvol=params.volvol,
                                             phi=1.0 + 0.0j,
                                             psi=0.0 + 0.0j)

        print(M)
        for m in M:
            w, v = la.eig(m)
            print(w)

        print(L)
        w, v = la.eig(L)
        print(w)

        print(H)

    elif unit_test == UnitTests.CHECK_RHS:
        M, L, H = func_a_ode_quadratic_terms(theta=params.theta,
                                             kappa1=params.kappa1,
                                             kappa2=params.kappa2,
                                             beta=params.beta,
                                             volvol=params.volvol,
                                             phi=1.0 + 0.0j,
                                             psi=0.0 + 0.0j)

        n = 3
        A0 = np.ones(n, dtype=np.complex128)
        rhs = func_rhs(t=1.0, A0=A0, M=M, L=L, H=H)
        print(rhs)

        jac = func_rhs_jac(t=1.0, A0=A0, M=M, L=L, H=H)
        print(jac)

    elif unit_test == UnitTests.CHECK_ODE_GRID_PHI:
        ttm = 0.1
        PHI = 0.5 + 1j*np.linspace(0, 20, 1000)
        tic = time.perf_counter()
        ode_sol = solve_a_ode_grid(phi_grid=PHI,
                                   psi_grid=np.zeros_like(PHI),
                                   ttm=ttm,
                                   theta=params.theta,
                                   kappa1=params.kappa1,
                                   kappa2=params.kappa2,
                                   beta=params.beta,
                                   volvol=params.volvol,
                                   is_stiff_solver=True)
        toc = time.perf_counter()
        print(f"{toc - tic} secs to run ODE solver")
        plot_ode_sol_in_phi(PHI=PHI, ode_sol=ode_sol)

        tic = time.perf_counter()
        ode_sol = solve_analytic_ode_grid_phi(phi_grid=PHI,
                                              psi_grid=np.zeros_like(PHI),
                                              ttm=ttm,
                                              theta=params.theta,
                                              kappa1=params.kappa1,
                                              kappa2=params.kappa2,
                                              beta=params.beta,
                                              volvol=params.volvol)
        toc = time.perf_counter()
        print(f"{toc - tic} secs to run Analytic solver")

        plot_ode_sol_in_phi(PHI=PHI, ode_sol=ode_sol)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CHECK_ODE_GRID_PHI

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)