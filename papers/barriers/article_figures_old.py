
# packages
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from enum import Enum

import qis.plots.utils as put
import qis.file_utils as fu

# qis
from stochvolmodels import local_path as lp

from . import pde
import qis.plots.lineplot as pli
import qis.plots.table as ptb
from qis.utils.np_ops import compute_histogram_data

# internal
from . import mc_engine as mce
from . import mc_volterra as mcv
from . import volterra as volt


FIG_SIZE = (12, 8)

@njit
def compute_heston_pde_barrier(t: float,
                               x0: float,
                               v0: float,
                               theta: float,
                               kappa: float,
                               rho: float,
                               volvol: float,
                               x_grid: np.ndarray,
                               nb_path: int = 10000,
                               diff_type: pde.DiffType = pde.DiffType.SYMMETRIC,
                               vt: np.ndarray = None,
                               it: np.ndarray = None,
                               mt: np.ndarray = None,
                               log_strikes: np.ndarray = None
                               ) -> np.ndarray:

    if vt is None:
        vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                             v0=v0,
                                                             theta=theta,
                                                             kappa=kappa,
                                                             rho=rho,
                                                             volvol=volvol,
                                                             nb_path=nb_path)

    t_grid = np.linspace(0, t, int(np.ceil(mce.YEAR_DAYS * t)))
    rho_1 = 1.0 - rho*rho
    green_ttms = np.zeros((x_grid.shape[0], nb_path))
    g0 = pde.set_one_to_nearest(a=x_grid, x0=x0, dx=x_grid[1]-x_grid[0])
    for path in np.arange(nb_path):
        path_mt = mt[:, path]
        path_drift_t = np.append(0.0, (path_mt[1:] - path_mt[:-1]) / (t_grid[1:] - t_grid[:-1]))
        green_ttms[:, path] = pde.solve_forward_pde1d(g0=g0,
                                                      t_grid=t_grid,
                                                      x_grid=x_grid,
                                                      var_t=rho_1*vt[:, path],
                                                      drift_t=path_drift_t,
                                                      diff_type=diff_type)

    if log_strikes is None:  # compute green
        avg_values = np.zeros(x_grid.shape[0])
        for nn in np.arange(x_grid.shape[0]):
            avg_values[nn] = np.mean(green_ttms[nn, :])
    else:  # compute calls
        avg_values = mcv.convolute_green_with_payoff(x_grid=x_grid,
                                                     log_strikes=log_strikes,
                                                     green_ttms=green_ttms)
    return avg_values


def plot_heston_barrier_green(t: float,
                              psi0: float,
                              x0: float,
                              v0: float,
                              theta: float,
                              kappa: float,
                              rho: float,
                              volvol: float,
                              nb_x_grid: int = 500,
                              nb_mc_pde_path: int = 10000,
                              nb_mc_path: int = 10000,
                              steps_per_day: int = 1,
                              seed_id: int = 3,
                              mc_color: str = 'lightblue',
                              is_title: bool = False
                              ) -> plt.Figure:
    # get all on same grid
    x_grid = np.linspace(psi0, x0 + 0.8, nb_x_grid)
    dx = x_grid[1]-x_grid[0]

    vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         nb_path=nb_mc_pde_path,
                                                         seed_id=seed_id)

    # pde
    tic = time.perf_counter()
    pde_sol_2s = compute_heston_pde_barrier(t=t,
                                            x0=x0,
                                            v0=v0,
                                            theta=theta,
                                            kappa=kappa,
                                            rho=rho,
                                            volvol=volvol,
                                            nb_path=nb_mc_pde_path,
                                            x_grid=x_grid,
                                            diff_type=pde.DiffType.SYMMETRIC,
                                            vt=vt,
                                            it=it,
                                            mt=mt)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute PDE")
    pde_sol_2s = pd.Series(dx*pde_sol_2s, index=x_grid, name='Pde')

    # volterra
    tic = time.perf_counter()
    volterra = mcv.compute_heston_volterra_barrier(t=t,
                                                   x0=x0,
                                                   psi0=psi0,
                                                   v0=v0,
                                                   theta=theta,
                                                   kappa=kappa,
                                                   rho=rho,
                                                   volvol=volvol,
                                                   nb_path=nb_mc_pde_path,
                                                   x_grid=x_grid,
                                                   vt=vt,
                                                   it=it,
                                                   mt=mt
                                                   )
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra_sol = pd.Series(dx*volterra, index=x_grid, name='Volterra')

    # apply mc
    tic = time.perf_counter()
    x_terminal = mce.simulate_x_barrier_paths(t=t,
                                              x0=x0,
                                              psi0=psi0,
                                              v0=v0,
                                              theta=theta,
                                              kappa=kappa,
                                              rho=rho,
                                              volvol=volvol,
                                              nb_path=nb_mc_path,
                                              steps_per_day=steps_per_day,
                                              is_barrier=True,
                                              seed_id=seed_id)
    mc_hist_data = compute_histogram_data(a=x_terminal, x_grid=x_grid, name='MC Histogram')
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC")

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

    data = pd.concat([mc_hist_data, volterra_sol, pde_sol_2s], axis=1)
    print(f"sum={data.sum(axis=0)}")

    prob_survival_mc = np.nansum(np.where(np.isfinite(x_terminal), 1.0, 0.0)) / nb_mc_path
    if is_title:
        title_lable = f"Prob survival: " \
                      f"MC={prob_survival_mc:0.4f}, " \
                      f"Pde ={np.sum(pde_sol_2s):0.4f}, " \
                      f"Volterra={np.sum(volterra_sol):0.4f}"
    else:
        title_lable = None

    kwargs = {'legend_loc': 'upper center'}
    colors = put.get_n_colors(n=3, first_color_fixed=True)
    colors[0] = mc_color
    pli.plot_line(df=data,
                  title=title_lable,
                  yvar_format='{:6.1E}',
                  xvar_format='{:.2f}',
                  colors=colors,
                  fontsize=14,
                  ax=ax,
                  **kwargs)

    ax.fill_between(mc_hist_data.index, np.zeros_like(mc_hist_data.to_numpy()), mc_hist_data.to_numpy(),
                        facecolor=mc_color, step='mid', alpha=0.8, lw=1.0)

    return fig


def plot_heston_barrier_call(t: float,
                             psi0: float,
                             log_strikes: np.ndarray,
                             x0: float,
                             v0: float,
                             theta: float,
                             kappa: float,
                             rho: float,
                             volvol: float,
                             nb_x_grid: int = 500,
                             nb_mc_pde_path: int = 10000,
                             nb_mc_path: int = 10000,
                             steps_per_day: int = 10,
                             seed_id: int = 3,
                             mc_color: str = 'lightblue'
                             ) -> (plt.Figure, plt.Figure):
    # get all on same grid
    strikes = np.exp(log_strikes)
    x_grid = np.linspace(psi0, x0 + 0.8, nb_x_grid)

    vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         nb_path=nb_mc_pde_path,
                                                         seed_id=seed_id)

    # pde
    tic = time.perf_counter()
    pde_sol_2s = compute_heston_pde_barrier(t=t,
                                            x0=x0,
                                            v0=v0,
                                            theta=theta,
                                            kappa=kappa,
                                            rho=rho,
                                            volvol=volvol,
                                            nb_path=nb_mc_pde_path,
                                            x_grid=x_grid,
                                            diff_type=pde.DiffType.SYMMETRIC,
                                            vt=vt,
                                            it=it,
                                            mt=mt,
                                            log_strikes=log_strikes)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute PDE")
    pde_sol_2s = pd.Series(pde_sol_2s, index=strikes, name='Pde')

    # volterra
    tic = time.perf_counter()
    volterra = mcv.compute_heston_volterra_barrier(t=t,
                                                   x0=x0,
                                                   psi0=psi0,
                                                   v0=v0,
                                                   theta=theta,
                                                   kappa=kappa,
                                                   rho=rho,
                                                   volvol=volvol,
                                                   nb_path=nb_mc_pde_path,
                                                   x_grid=x_grid,
                                                   vt=vt,
                                                   it=it,
                                                   mt=mt,
                                                   log_strikes=log_strikes)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra_sol = pd.Series(volterra, index=strikes, name='Volterra')

    # apply mc
    tic = time.perf_counter()
    x_terminal = mce.simulate_x_barrier_paths(t=t,
                                              x0=x0,
                                              psi0=psi0,
                                              v0=v0,
                                              theta=theta,
                                              kappa=kappa,
                                              rho=rho,
                                              volvol=volvol,
                                              nb_path=nb_mc_path,
                                              steps_per_day=steps_per_day,
                                              is_barrier=True,
                                              seed_id=seed_id)
    mc_prices = mce.compute_barrier_calls(xt=x_terminal, log_strikes=log_strikes)
    mc_prices = pd.Series(mc_prices, index=strikes, name='MC')
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC")

    data = pd.concat([mc_prices, volterra_sol, pde_sol_2s], axis=1)
    print(data)

    kwargs = {'legend_loc': 'upper center'}
    colors = put.get_n_colors(n=3, first_color_fixed=True)
    colors[0] = mc_color
    with sns.axes_style("darkgrid"):
        fig1, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
        pli.plot_line(df=data,
                      title='barrier call',
                      yvar_format='{:0.0%}',
                      xvar_format='{:.2f}',
                      colors=colors,
                      ylabel='Premium',
                      xlabel='Strike',
                      fontsize=14,
                      ax=ax,
                      **kwargs)

    # fig2, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
    fig2 = ptb.plot_df_table(df=data,
                             # add_index_as_column=False,
                             var_format='{:0.4f}',
                             index_column_name='Strike',
                             special_columns_colors=[(0, 'steelblue')],
                             # ax=axs[1],
                             **kwargs)

    return fig1, fig2


def plot_drift_vol(t: float,
                   x0: float,
                   v0: float,
                   theta: float,
                   kappa: float,
                   rho: float,
                   volvol: float,
                   psi0: float = -0.3,
                   nb_path: int = 10,
                   path_id: int = 2,
                   nb_x_grid: int = 500
                   ) -> (plt.Figure, plt.Figure, plt.Figure):

    vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         nb_path=nb_path)

    rho_1 = (1.0 - rho*rho)
    t_grid = np.linspace(0, t, int(np.ceil(mce.YEAR_DAYS * t)))
    path_mt = mt[:, path_id]
    mu_t = np.append(0.0, (path_mt[1:] - path_mt[:-1]) / (t_grid[1:] - t_grid[:-1]))
    var_t = np.sqrt((1.0 - rho*rho)*vt[:, path_id])
    upsilon_t = rho_1*it[:, path_id]
    lambda_t = mu_t / var_t

    # 1
    datas = {'(A) Advection': pd.Series(mu_t, index=t_grid),
             '(B) Diffusion': pd.Series(var_t, index=t_grid)}
    with sns.axes_style("darkgrid"):
        fig1, axs = plt.subplots(2, 1, figsize=FIG_SIZE, tight_layout=True)
        for idx, (title, data) in enumerate(datas.items()):
            pli.plot_line(df=data,
                          title=title,
                          legend_loc=None,
                          yvar_format='{:0.2f}',
                          xvar_format='{:.2f}',
                          xlabel='time',
                          fontsize=14,
                          ax=axs[idx])

    # 2
    datas = {r'(A) $\Upsilon(t)$': pd.Series(upsilon_t, index=t_grid),
             r'(B) $\nu(\Upsilon)$': pd.Series(lambda_t, index=upsilon_t)}
    with sns.axes_style("darkgrid"):
        fig2, axs = plt.subplots(2, 1, figsize=FIG_SIZE, tight_layout=True)
        for idx, (title, data) in enumerate(datas.items()):
            pli.plot_line(df=data,
                          title=title,
                          legend_loc=None,
                          yvar_format='{:0.2f}',
                          xvar_format='{:.2f}',
                          xlabel='time' if idx==0 else r'$\Upsilon(t)$',
                          fontsize=14,
                          ax=axs[idx])

    # 3
    x_grid = np.linspace(psi0, x0 + 0.8, nb_x_grid)
    green_volterra = volt.compute_green(upsilon_k=rho_1 * it[:, path_id],
                                        barrier_k=mt[:, path_id],
                                        x_grid=x_grid,
                                        x0=x0,
                                        psi0=psi0,
                                        n_t=mt[-1, path_id],
                                        is_terminal=True)
    green_volterra = pd.Series(green_volterra, index=x_grid, name='Volterra')
    path_mt = mt[:, path_id]
    path_drift_t = np.append(0.0, (path_mt[1:] - path_mt[:-1]) / (t_grid[1:] - t_grid[:-1]))
    green_pde = pde.solve_forward_pde1d(g0=pde.set_one_to_nearest(a=x_grid, x0=x0, dx=x_grid[1] - x_grid[0]),
                                        t_grid=t_grid,
                                        x_grid=x_grid,
                                        var_t=rho_1 * vt[:, path_id],
                                        drift_t=path_drift_t)
    green_pde = pd.Series(green_pde, index=x_grid, name='PDE')
    data = pd.concat([green_volterra, green_pde], axis=1)

    with sns.axes_style("darkgrid"):
        fig3, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
        pli.plot_line(df=data,
                      title=None,
                      # legend_loc=None,
                      yvar_format='{:0.2f}',
                      xvar_format='{:.2f}',
                      xlabel='x',
                      fontsize=14,
                      # y_limits=(0.0, None),
                      ax=ax)

    return fig1, fig2, fig3


class LocalTests(Enum):
    PLOT_DRIFT_VAR = 1
    PLOT_HESTON_BARRIER_GREEN = 2
    PLOT_HESTON_BARRIER_CALL = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    local_path = lp.get_output_path()
    t = 1.0
    x0 = 0.0
    v0 = 0.25
    theta = 0.2
    kappa = 1.0
    rho = -0.3
    volvol = 0.4
    psi0 = -1.0  # log(B/S)

    if local_test == LocalTests.PLOT_DRIFT_VAR:
        fig1, fig2, fig3 = plot_drift_vol(t=t,
                                          x0=x0,
                                          psi0=psi0,
                                          v0=v0,
                                          theta=theta,
                                          kappa=kappa,
                                          rho=rho,
                                          volvol=volvol,
                                          path_id=1)

        fu.save_fig(fig=fig1, local_path=local_path, file_name='AdvectionDiffusion')
        fu.save_fig(fig=fig2, local_path=local_path, file_name='UpsilonLambda')
        fu.save_fig(fig=fig3, local_path=local_path, file_name='PathSolution')

    elif local_test == LocalTests.PLOT_HESTON_BARRIER_GREEN:
        fig = plot_heston_barrier_green(t=t,
                                        psi0=psi0,
                                        x0=x0,
                                        v0=v0,
                                        theta=theta,
                                        kappa=kappa,
                                        rho=rho,
                                        volvol=volvol,
                                        nb_x_grid=500,
                                        nb_mc_pde_path=2000,  #2000,
                                        nb_mc_path=50000,
                                        seed_id=3,
                                        is_title=False)

        fu.save_fig(fig=fig, local_path=local_path, file_name='GreenSolution')

    elif local_test == LocalTests.PLOT_HESTON_BARRIER_CALL:

        strikes = np.linspace(0.9, 1.2, 7)
        log_strikes = np.log(strikes)
        print(log_strikes)
        psi0 = log_strikes[0]
        fig1, fig2 = plot_heston_barrier_call(t=t,
                                              x0=0.0,
                                              psi0=psi0,  # log(B/S)
                                              log_strikes=log_strikes,
                                              v0=v0,
                                              theta=theta,
                                              kappa=kappa,
                                              rho=rho,
                                              volvol=volvol,
                                              nb_x_grid=500,
                                              nb_mc_pde_path=5000,  #2000,
                                              nb_mc_path=100000,
                                              seed_id=3)

        fu.save_fig(fig=fig1, local_path=local_path, file_name='BarrierCall')
        fu.save_fig(fig=fig2, local_path=local_path, file_name='BarrierCallTable')

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.PLOT_DRIFT_VAR

    run_local_test(local_test=local_test)
