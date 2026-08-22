import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit
from typing import Union, Tuple
from enum import Enum


@njit
def erfcc(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Complementary error function. can be vectorized
    """
    z = np.abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+ t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+t*.17087277)))))))))
    fcc = np.where(np.greater(x, 0.0), r, 2.0-r)
    return fcc


@njit
def ncdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1. - 0.5*erfcc(x/(np.sqrt(2.0)))


@njit
def npdf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.exp(-0.5*np.square(x))/np.sqrt(2.0*np.pi)


@njit
def h0(t: float, x0: float, x_grid: np.ndarray) -> np.ndarray:
    """
    zero order term
    """
    return (np.exp(-np.square(x0-x_grid)/(2.0*t)) - np.exp(-np.square(x0+x_grid)/(2.0*t)))/np.sqrt(2.0*np.pi*t)


@njit
def h1(t_grid: np.ndarray,
       x_grid: np.ndarray,
       drift_t: np.ndarray,
       x0: float
       ) -> np.ndarray:
    """
    1st order correction
    """
    te = t_grid[-1]
    dt = np.append(t_grid[1:]-t_grid[:-1], 0.0)
    sol = np.zeros_like(x_grid)
    for idx, xx in enumerate(x_grid):
        vol_scale = np.sqrt((te-t_grid)*te*t_grid)
        int1 = np.nansum(drift_t*(2.0*ncdf(x0*t_grid+xx*(te-t_grid)/vol_scale)-1.0)*dt)
        term1 = ((x0-xx)/te)*int1 * np.exp(-np.square(x0-xx)/(2.0*te))
        int2 = np.nansum(drift_t*(2.0*ncdf(x0*t_grid-xx*(te-t_grid)/vol_scale)-1.0)*dt)
        term2 = ((x0+xx)/te)*int2 * np.exp(-np.square(x0+xx)/(2.0*te))
        sol[idx] = (term1 - term2)/np.sqrt(2.0*np.pi*te)
    return sol


@njit
def compute_barrier_1st_expansion(t_grid: np.ndarray,
                                  x_grid: np.ndarray,
                                  drift_t: np.ndarray,
                                  x0: float
                                  ) -> np.ndarray:
    """
    1st order solution of G_t - mu(t)G_x + 0.5*G_xx=0
    """
    sol0 = h0(t=t_grid[-1], x_grid=x_grid, x0=x0)
    sol1 = h1(t_grid=t_grid, x_grid=x_grid, drift_t=drift_t, x0=x0)
    return sol0 + sol1


@njit
def compute_barrier_1st_path_solutions(it: np.ndarray,
                                       vt: np.ndarray,
                                       mt: np.ndarray,
                                       x_grid: np.ndarray,
                                       t_grid: np.ndarray,
                                       x0: float
                                       ) -> np.ndarray:
    """
    it intergrated var
    vt var
    1st order solution of G_t - mu(t)G_x + 0.5*G_xx=0
    """
    nb_path = it.shape[1]
    green_ttms = np.zeros((x_grid.shape[0], nb_path))
    for path in np.arange(nb_path):
        path_mt = -mt[:, path]
        path_drift_t = np.append(0.0, (path_mt[1:] - path_mt[:-1]) / (t_grid[1:] - t_grid[:-1])) / vt[:, path]
        green_ttms[:, path] = compute_barrier_1st_expansion(t_grid=it[:, path],
                                                            x_grid=x_grid,
                                                            drift_t=path_drift_t,
                                                            x0=x0)
    avg_green = np.zeros(x_grid.shape[0])
    for nn in np.arange(x_grid.shape[0]):
        avg_green[nn] = np.mean(green_ttms[nn, :])
    return avg_green


class LocalTests(Enum):
    SOLUTION = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.SOLUTION:
        te = 1.0
        n = 1000
        x0 = 1.0
        t_grid = np.linspace(0.0, te, n)
        drift_t = -0.1 * t_grid

        x_grid = np.linspace(0.0, 6.0, 200)

        H0 = pd.Series(h0(t=te, x_grid=x_grid, x0=x0), index=x_grid, name='H0')

        H1 = h1(t_grid=t_grid, x_grid=x_grid, drift_t=drift_t, x0=x0)
        H1 = pd.Series(H1, index=x_grid, name='H1')

        data = pd.concat([H0, H1], axis=1)
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sns.lineplot(data=data, dashes=False, ax=ax)
            ax.set_xlabel('t')
            ax.set_ylabel('pdf')

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.SOLUTION
    run_local_test(local_test=local_test)
