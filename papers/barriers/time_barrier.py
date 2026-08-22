"""
BM with time dependent barrier(t) starting at x0=0
to compare Volterra with MC
"""
# packages
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
from . import pde
import qis.plots.lineplot as pli
import qis.plots.utils as put
from qis.utils.np_ops import compute_histogram_data

# internal
from . import volterra as volt
from .mc_engine import YEAR_DAYS, simulate_bm_barrier_drift_paths


def plot_time_barrier(t: float,
                      v0: float,  #  =vol^2
                      x0: float = 0.0,  # =log(S/S0)
                      nb_x_grid: int = 500,
                      nb_mc_path: int = 10000,
                      steps_per_day: int = 1,
                      seed_id: int = 3,
                      mc_color: str = 'lightblue'
                      ) -> None:
    # get all on same grid
    t_grid = np.linspace(0, t, int(np.ceil(steps_per_day*YEAR_DAYS * t)))
    barrier_t = -0.1- 0.1*t_grid - 0.2*np.sqrt(t_grid)  # defines barrier level  =log(B/S0)
    x_grid = np.linspace(barrier_t[-1], 4.0*np.sqrt(v0*t), nb_x_grid)
    dx = x_grid[1]-x_grid[0]

    # pde 1
    tic = time.perf_counter()
    g0 = pde.set_one_to_nearest(a=x_grid, x0=-barrier_t[-1], dx=x_grid[1]-x_grid[0])
    drift_dt = np.append(0.0, (barrier_t[1:] - barrier_t[:-1]))
    path_drift_t = drift_dt / np.append(1.0, t_grid[1:] - t_grid[:-1])
    pde_sol_2s = pde.solve_forward_pde1d(g0=g0,
                                         t_grid=t_grid,
                                         x_grid=x_grid,
                                         var_t=v0*np.ones_like(t_grid),
                                         drift_t=path_drift_t,
                                         diff_type=pde.DiffType.SYMMETRIC)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute symmetric PDE")
    pde_sol_2s = pd.Series(dx*pde_sol_2s, index=x_grid, name='Pde Symmetric Solution')

    # volterra
    tic = time.perf_counter()
    volterra_sol = volt.compute_green(upsilon_k=v0*t_grid,
                                      barrier_k=-barrier_t,
                                      x_grid=x_grid,
                                      x0=x0,
                                      n_t=0.0,#barrier_t[-1],
                                      is_terminal=True)

    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra_sol = pd.Series(dx*volterra_sol, index=x_grid, name='Volterra Solution')

    # apply mc
    tic = time.perf_counter()
    x_terminal = simulate_bm_barrier_drift_paths(t_grid=t_grid,
                                                 barrier_t=barrier_t,
                                                 x0=x0,
                                                 v0=v0,
                                                 nb_path=nb_mc_path,
                                                 seed_id=seed_id)
    mc_hist_data = compute_histogram_data(a=x_terminal, x_grid=x_grid, name='MC Histogram')
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC")

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), tight_layout=True)

    data = pd.concat([mc_hist_data, pde_sol_2s, volterra_sol], axis=1)
    print(f"sum={data.sum(axis=0)}")

    prob_survival_mc = np.nansum(np.where(np.isfinite(x_terminal), 1.0, 0.0)) / nb_mc_path
    title_lable = f"Prob survival: " \
                  f"MC={prob_survival_mc:0.4f}, " \
                  f"Pde ={np.sum(pde_sol_2s):0.4f}, " \
                  f"Volterra={np.sum(volterra_sol):0.4f}"

    kwargs = {'legend_loc': 'upper center'}
    colors = put.get_n_colors(n=3, first_color_fixed=True)
    colors[0] = mc_color
    pli.plot_line(df=data,
                  title=title_lable,
                  yvar_format='{:6.1E}',
                  xvar_format='{:.2f}',
                  colors=colors,
                  fontsize=14,
                  ax=axs[0],
                  **kwargs)

    axs[0].fill_between(mc_hist_data.index, np.zeros_like(mc_hist_data.to_numpy()), mc_hist_data.to_numpy(),
                        facecolor=mc_color, step='mid', alpha=0.8, lw=1.0)

    diff_data2 = np.subtract(pde_sol_2s, pde_sol_2s).rename('PDE 2s-PDE 2s')
    diff_data3 = np.subtract(volterra_sol, pde_sol_2s).rename('Volterra-PDE 2s')
    diff_data = pd.concat([diff_data2, diff_data3], axis=1)
    diff_data.iloc[:20, :] = np.nan
    pli.plot_line(df=diff_data,
                  title=f"Differences relative to expansion",
                  yvar_format='{:6.0E}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[1],
                  **kwargs)


class LocalTests(Enum):
    PLOT_TIME_BARRIER = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    t = 1.0

    if local_test == LocalTests.PLOT_TIME_BARRIER:
        plot_time_barrier(t=t,
                          x0=0.0,
                          v0=0.2**2,
                          nb_x_grid=500,
                          nb_mc_path=400000,
                          seed_id=3)

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.PLOT_TIME_BARRIER

    run_local_test(local_test=local_test)
