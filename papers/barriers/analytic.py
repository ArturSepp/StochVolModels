import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from enum import Enum
from typing import Tuple, Any, Optional


# qis
import qis.plots.lineplot as pli


@njit
def compute_green_psi_grid(psi: np.ndarray,
                         ttm: float,
                         v0: float,
                         kappa: float,
                         theta: float,
                         volvol: float
                         ) -> np.ndarray:
    volvol2 = volvol*volvol
    psi2 = np.square(psi) + 0.25
    zeta = np.sqrt(kappa*kappa+volvol2*psi2)
    exp_zeta = np.exp(-ttm*zeta)
    psi_p = -kappa + zeta
    psi_m = kappa + zeta
    b = -psi2*((1.0-exp_zeta)/(psi_m+psi_p*exp_zeta))
    a = -(kappa*theta/volvol2)*(ttm*psi_p + 2.0*np.log((psi_m+psi_p*exp_zeta)/(2.0*zeta)))
    return np.exp(a+b*v0)


@njit
def compute_green_x_barrier0(x_grid: np.ndarray,
                             ttm: float,
                             x0: float,
                             v0: float,
                             theta: float,
                             kappa: float,
                             volvol: float
                             ) -> np.ndarray:

    psi = np.linspace(0.0, 40, 4000)
    d_psi = psi[1] - psi[0]
    green_psi = compute_green_psi_grid(psi=psi, ttm=ttm, v0=v0, kappa=kappa, theta=theta, volvol=volvol)
    green_x = np.zeros_like(x_grid)

    for n, x in enumerate(x_grid):
        f_psi = (np.cos(psi*(x-x0))-np.cos(psi*(x+x0)))*green_psi
        green_x[n] = 0.5*f_psi[0] + np.nansum(f_psi[1:])
    # green_x = green_x * np.exp(-0.5*x_grid)*(d_psi/np.pi)
    green_x = green_x * (d_psi / np.pi)

    return green_x


def plot_green_x(ttm: float = 1.0,
                 x0: float = 0.5,
                 v0: float = 0.04,
                 theta: float = 0.04,
                 kappa: float = 4.0,
                 volvol: float = 0.5
                 ):
    x_grid = np.linspace(0.0, 2.0, 500)
    green_x = compute_green_x_barrier0(x_grid=x_grid, ttm=ttm, x0=x0, v0=v0, theta=theta, kappa=kappa, volvol=volvol)
    data = pd.Series(green_x, index=x_grid, name='analytic')
    print(data)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(14, 12), tight_layout=True)
        pli.plot_line(df=data,
                      xvar_format='{:0.2f}',
                      yvar_format='{:0.2f}',
                      ax=ax)


class LocalTests(Enum):
    ANALYTIC = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.ANALYTIC:
        plot_green_x()

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.ANALYTIC
    run_local_test(local_test=local_test)
