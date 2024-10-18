"""
figures 8 and 9
plot ODE system solutions in time
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg as sla
from numpy import linalg as la
from matplotlib import pyplot as plt
from enum import Enum
from scipy.integrate._ivp.ivp import OdeResult
from typing import Tuple, List

import stochvolmodels.utils.plots as plot
from stochvolmodels.pricers.logsv.affine_expansion import solve_ode_for_a, ExpansionOrder, func_a_ode_quadratic_terms, get_expansion_n
from stochvolmodels import LogSvParams


def plot_ode_sol_in_t(params: LogSvParams,
                      ttm: float,
                      ode_sol: OdeResult,
                      expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                      title: str = None,
                      headers: List[str] = None,
                      axs: List[plt.Subplot] = None
                      ) -> None:
    """
    plot real and imag parts of given solution
    """
    t = np.linspace(0.0, ttm, 300)
    z = ode_sol.sol(t)

    if expansion_order == ExpansionOrder.FIRST:
        y = params.sigma0 - params.theta
        ys = np.array([1.0, y, y * y])
        mgf = np.exp(z.T @ ys)
        data = {r'$\Re[A(\tau)]$': pd.DataFrame(data=np.real(z).T, index=t, columns=[r'$A^{(0)}$', r'$A^{(1)}$', r'$A^{(2)}$']),
                r'$\Im[A(\tau)]$': pd.DataFrame(data=np.imag(z).T, index=t, columns=[r'$A^{(0)}$', r'$A^{(1)}$', r'$A^{(2)}$']),
                r'$E^{[1]}(\tau)$': pd.concat([pd.Series(np.real(mgf), index=t, name=r'$\Re[E^{[1]}]$'),
                                         pd.Series(np.imag(mgf), index=t, name=r'$\Im[E^{[1]}]$')],
                                        axis=1)}

    elif expansion_order == ExpansionOrder.SECOND:
        y = params.sigma0 - params.theta
        ys = np.array([1.0, y, y * y, y ** 3, y ** 4])
        mgf = np.exp(z.T @ ys)
        data = {r'$\Re[A(\tau)]$': pd.DataFrame(data=np.real(z).T, index=t, columns=[r'$A^{(0)}$', r'$A^{(1)}$', r'$A^{(2)}$', r'$A^{(3)}$', r'$A^{(4)}$']),
                r'$\Im[A(\tau)]$': pd.DataFrame(data=np.imag(z).T, index=t, columns=[r'$A^{(0)}$', r'$A^{(1)}$', r'$A^{(2)}$', r'$A^{(3)}$', r'$A^{(4)}$']),
                r'$E^{[2]}(\tau)$': pd.concat([pd.Series(np.real(mgf), index=t, name=r'$\Re[E^{[2]}]$'),
                                         pd.Series(np.imag(mgf), index=t, name=r'$\Im[E^{[2]}]$')],
                                        axis=1)}

    else:
        raise TypeError(f"not implemented")

    if axs is None:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(3, 1, figsize=(8, 8), tight_layout=True)
            fig.suptitle(title, color='darkblue')
    for idx, (key, data) in enumerate(data.items()):
        sns.lineplot(data=data, ax=axs[idx])
        if headers is not None: # and idx == 0:
            title_ = f"{headers[idx]} {key}, {title}"
        else:
            title_ = key
        axs[idx].set_title(title_, color='darkblue')
        axs[idx].set(xlabel=r"$\tau$")


def plot_ode_solutions(params: LogSvParams,
                       ttm: float = 1,
                       is_stiff_solver: bool = True,
                       expansion_order: ExpansionOrder = ExpansionOrder.FIRST,
                       is_spot_measure: bool = True
                       ) -> plt.Figure:
    """
    solve and plot real and imag parts of solutions in ttm as function of phi
    """
    if is_spot_measure:
        real_part = -0.5
    else:
        real_part = 0.5

    is_1d = True
    if is_1d:
        phis = np.array([real_part + 2.0j], dtype=np.complex128)
        headers = [['(A)', '(B)', '(C)']]
    else:
        phis = np.array([real_part + 1.0j, real_part + 10.0j], dtype=np.complex128)
        headers = [['(A1)', '(A2)', '(A3)'], ['(B1)', '(B2)', '(B3)']]

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(len(phis), 3, figsize=(18, 6), tight_layout=True)

    for idx, phi in enumerate(phis):
        tic = time.perf_counter()
        ode_sol = solve_ode_for_a(ttm=ttm,
                                  theta=params.theta,
                                  kappa1=params.kappa1,
                                  kappa2=params.kappa2,
                                  beta=params.beta,
                                  volvol=params.volvol,
                                  phi=phi,
                                  psi=0.0 + 0.0j,
                                  dense_output=True,
                                  expansion_order=expansion_order,
                                  is_stiff_solver=is_stiff_solver,
                                  is_spot_measure=is_spot_measure)
        toc = time.perf_counter()
        print(f"{toc - tic} secs to run solver")
        title = f"$\Phi$={np.real(phi):0.2f}+{np.imag(phi):0.2f}i"
        plot_ode_sol_in_t(params=params, ttm=ttm, ode_sol=ode_sol, expansion_order=expansion_order, title=title,
                          axs=axs[idx, :] if len(phis) > 1 else axs,
                          headers=headers[idx])
    plot.set_subplot_border(fig=fig, n_ax_rows=len(phis), n_ax_col=3)
    return fig


def solve_approximate_solutions(ttm: float,
                                theta: float,
                                kappa1: float,
                                kappa2: float,
                                beta: float,
                                volvol: float,
                                phi: np.complex128,
                                psi: np.complex128,
                                expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                                ) -> np.ndarray:

    M, L, H = func_a_ode_quadratic_terms(theta=theta,
                                         kappa1=kappa1,
                                         kappa2=kappa2,
                                         beta=beta,
                                         volvol=volvol,
                                         phi=phi,
                                         psi=psi,
                                         expansion_order=expansion_order)
    L = np.transpose(L)
    term = (sla.expm(ttm*L) - la.pinv(L))*H
    return term


def plot_approximate_solutions(params: LogSvParams,
                               phi: np.complex128,
                               ttm: float = 1,
                               is_stiff_solver: bool = True,
                               expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                               ) -> None:
    """
    solve and plot real and imag parts of solutions in ttm as function of phi
    """

    ttms = np.linspace(0.0, ttm, 100)
    apr_sol = np.ndarray((len(ttms), get_expansion_n(expansion_order)))
    for t, ttm in enumerate(ttms):
        apr_sol[t] = solve_approximate_solutions(ttm=ttm,
                                                 theta=params.theta,
                                                 kappa1=params.kappa1,
                                                 kappa2=params.kappa2,
                                                 beta=params.beta,
                                                 volvol=params.volvol,
                                                 phi=phi,
                                                 psi=0.0 + 0.0j,
                                                 expansion_order=expansion_order)[0]

    ode_sol = solve_ode_for_a(ttm=ttm,
                              theta=params.theta,
                              kappa1=params.kappa1,
                              kappa2=params.kappa2,
                              beta=params.beta,
                              volvol=params.volvol,
                              phi=phi,
                              psi=0.0 + 0.0j,
                              dense_output=True,
                              expansion_order=expansion_order,
                              is_stiff_solver=is_stiff_solver)
    z = ode_sol.sol(ttms)

    if expansion_order == ExpansionOrder.FIRST:
        columns = ['A0', 'A1', 'A2']
    elif expansion_order == ExpansionOrder.SECOND:
        columns = ['A0', 'A1', 'A2', 'A3', 'A4']
    else:
        raise TypeError(f"not implemented")

    apr_real = pd.DataFrame(data=np.real(apr_sol), index=ttms, columns=columns)
    apr_imag = pd.DataFrame(data=np.imag(apr_sol), index=ttms, columns=columns)
    sol_real = pd.DataFrame(data=np.real(z).T, index=ttms, columns=columns)
    sol_imag = pd.DataFrame(data=np.imag(z).T, index=ttms, columns=columns)

    real_data = {}
    imag_data = {}
    for column in columns:
        real_data[f"{column}: real"] = pd.concat([sol_real[column].rename('Solution'), apr_real[column].rename('Appr')], axis=1)
        imag_data[f"{column}: imag"] = pd.concat([sol_imag[column].rename('Solution'), apr_imag[column].rename('Appr')], axis=1)

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(get_expansion_n(expansion_order), 2, figsize=(18, 6), tight_layout=True)
        title = f"phi={np.real(phi):0.2f}+{np.imag(phi):0.2f}i"
        fig.suptitle(title, color='darkblue')
        for idx, ((key1, data1), (key2, data2)) in enumerate(zip(real_data.items(), imag_data.items())):
            sns.lineplot(data=data1, ax=axs[idx, 0])
            axs[idx, 0].set_title(key1)
            sns.lineplot(data=data2, ax=axs[idx, 1])
            axs[idx, 1].set_title(key2)


def detect_ode_explosion(params: LogSvParams,
                         ttm: float = 1,
                         is_stiff_solver: bool = True,
                         expansion_order: ExpansionOrder = ExpansionOrder.FIRST
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect an explosion is solution of ODE and check against having eigen value with positive real part in matrix
    of linear terms. Solution with value >=1e16 is considered an explosion
    """
    phis = np.array([-0.5 + 0.0j, -0.5 + 1.0j, -0.5 + 10.0j, -0.5 + 100.0j], dtype=np.complex128)
    t = np.linspace(0.0, ttm, 300)
    is_explosion = np.full(phis.size, False)
    pos_eigenvals = np.full(phis.size, False)

    for j_phi, phi in enumerate(phis):
        tic = time.perf_counter()
        ode_sol = solve_ode_for_a(ttm=ttm,
                                  theta=params.theta,
                                  kappa1=params.kappa1,
                                  kappa2=params.kappa2,
                                  beta=params.beta,
                                  volvol=params.volvol,
                                  phi=phi,
                                  psi=0.0 + 0.0j,
                                  dense_output=True,
                                  expansion_order=expansion_order,
                                  is_stiff_solver=is_stiff_solver)
        toc = time.perf_counter()
        print(f"{toc - tic} secs to run solver")
        # Calculate eigenvalues of linear term
        _, L, _ = func_a_ode_quadratic_terms(theta=params.theta,
                                             kappa1=params.kappa1,
                                             kappa2=params.kappa2,
                                             beta=params.beta,
                                             volvol=params.volvol,
                                             phi=phi,
                                             psi=0.0 + 0.0j,
                                             expansion_order=expansion_order,
                                             is_spot_measure=True)
        eigenvals = la.eigvals(L)
        if np.max(np.real(eigenvals)) > 0.0:
            pos_eigenvals[j_phi] = True
        z = ode_sol.sol(t)
        for zk in z:
            if np.max(zk) >= 1.0e16:
                is_explosion[j_phi] = True
                break
    return is_explosion, pos_eigenvals


class UnitTests(Enum):
    FIRST_ORDER = 1
    SECOND_ORDER = 2
    APPROXIMATION = 3


def run_unit_test(unit_test: UnitTests):

    # params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=1.2, volvol=3.0)
    # params = LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=1.0, volvol=1.0)
    params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)

    params.assert_vol_moments_stability()
    params.print_vol_moments_stability()

    ttm = 1.0
    is_spot_measure = True

    is_save = False
    if unit_test == UnitTests.FIRST_ORDER:
        fig = plot_ode_solutions(params=params, ttm=ttm, expansion_order=ExpansionOrder.FIRST,
                                 is_spot_measure=is_spot_measure)
        if is_save:
            plot.save_fig(fig=fig, local_path='../../../docs/figures//', file_name="first_order_fig")

    elif unit_test == UnitTests.SECOND_ORDER:
        fig = plot_ode_solutions(params=params, ttm=ttm, expansion_order=ExpansionOrder.SECOND,
                                 is_spot_measure=is_spot_measure)
        if is_save:
            plot.save_fig(fig=fig, local_path='../../../docs/figures//', file_name="second_order_fig")

    elif unit_test == UnitTests.APPROXIMATION:
        plot_approximate_solutions(phi=-0.5 + 1j,
                                   params=params, ttm=ttm, expansion_order=ExpansionOrder.SECOND)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SECOND_ORDER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
