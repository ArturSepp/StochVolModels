
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
from qis.plots.bars import plot_bars
from qis.utils.np_ops import compute_histogram_data

# internal
from . import mc_engine as mce
from . import mc_volterra as mcv

# new
from . import green_solutions as vgr

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
                               rfrate: float = 0.0,
                               nb_path: int = 10000,
                               mt_: int = 6,
                               vt: np.ndarray = None,
                               it: np.ndarray = None,
                               mt: np.ndarray = None,
                               log_strikes: np.ndarray = np.array([0.0]),
                               function_type=mcv.FunctionType.GREEN
                               ) -> np.ndarray:

    if vt is None:
        vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                             v0=v0,
                                                             theta=theta,
                                                             kappa=kappa,
                                                             rho=rho,
                                                             volvol=volvol,
                                                             rfrate=rfrate,
                                                             nb_path=nb_path)

    t_grid = np.linspace(0, t, int(np.ceil(mce.YEAR_DAYS * t)))
    rho_1 = 1.0 - rho*rho
    green_ttms = np.zeros((x_grid.shape[0], nb_path))
    g0 = pde.set_one_to_nearest(a=x_grid, x0=x0, dx=x_grid[1]-x_grid[0])
    for path in np.arange(nb_path):
        """
        path_mt = mt[:, path]
        path_drift_t = np.append(0.0, (path_mt[1:] - path_mt[:-1]) / (t_grid[1:] - t_grid[:-1]))
        green_ttms[:, path] = pde.solve_forward_pde1d(g0=g0,
                                                      t_grid=t_grid,
                                                      x_grid=x_grid,
                                                      var_t=rho_1*vt[:, path],
                                                      drift_t=path_drift_t,
                                                      diff_type=diff_type)
        """
        if function_type in [mcv.FunctionType.GREEN, mcv.FunctionType.CALL]:
            green_ttms[:, path] = vgr.compute_pde_green(maturity=t,
                                                        mt=mt_,
                                                        gridx=x_grid,
                                                        advection=mt[:, path],
                                                        diffusion=rho_1 * vt[:, path])
        else:
            green_ttms[:, path] = vgr.compute_pde_survival_prob(maturity=t,
                                                                mt=mt_,
                                                                gridx=x_grid,
                                                                advection=mt[:, path],
                                                                diffusion=rho_1 * vt[:, path])

    if function_type == mcv.FunctionType.CALL:
        avg_values = mcv.convolute_green_with_payoff(x_grid=x_grid,
                                                     log_strikes=log_strikes,
                                                     green_ttms=green_ttms,
                                                     discfactor=np.exp(-rfrate*t))
    else:
        avg_values = np.zeros(x_grid.shape[0])
        for nn in np.arange(x_grid.shape[0]):
            avg_values[nn] = np.mean(green_ttms[nn, :])

        if function_type == mcv.FunctionType.SURVIVAL_PROB:
            avg_values = np.exp(-rfrate*t)*avg_values

    return avg_values


def plot_heston_barrier_green(t: float,
                              psi0: float,
                              x0: float,
                              v0: float,
                              xmax: float,
                              theta: float,
                              kappa: float,
                              rho: float,
                              volvol: float,
                              nb_x_grid: int = 500,
                              nb_mc_pde_path: int = 10000,
                              nb_mc_path: int = 10000,
                              steps_per_day: int = 6,
                              seed_id: int = 3,
                              mc_color: str = 'lightblue',
                              is_title: bool = False
                              ) -> (plt.Figure, plt.Figure):
    # get all on same grid
    x_grid = np.linspace(psi0, xmax, nb_x_grid)
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
                                            mt_=steps_per_day,
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
                                                   mt_=steps_per_day,
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
        fig1, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

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

    my_indices = [x%int(nb_x_grid/15) == 0 for x in np.arange(nb_x_grid)]
    my_indices[0] = False
    fig2, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
    ptb.plot_df_table(df=data.iloc[my_indices, :],
                      # add_index_as_column=False,
                      var_format='{:0.4f}',
                      index_column_name='x',
                      special_columns_colors=[(0, 'steelblue')],
                      ax=ax,
                      **kwargs)

    return fig1, fig2


def plot_heston_barrier_survival(t: float,
                                 psi0: float,
                                 xmax: float,
                                 x0: float,
                                 v0: float,
                                 theta: float,
                                 kappa: float,
                                 rho: float,
                                 volvol: float,
                                 rfrate: float = 0.0,
                                 nb_x_grid: int = 500,
                                 nb_mc_pde_path: int = 10000,
                                 nb_mc_path: int = 10000,
                                 steps_per_day: int = 6,
                                 seed_id: int = 3,
                                 mc_color: str = 'lightblue'
                                 ) -> (plt.Figure, plt.Figure):
    # get all on same grid
    x_grid = np.linspace(psi0, xmax, nb_x_grid)

    vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         rfrate=rfrate,
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
                                            rfrate=rfrate,
                                            nb_path=nb_mc_pde_path,
                                            x_grid=x_grid,
                                            mt_=steps_per_day,
                                            vt=vt,
                                            it=it,
                                            mt=mt,
                                            function_type=mcv.FunctionType.SURVIVAL_PROB)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute PDE")
    pde_sol_2s = pd.Series(pde_sol_2s, index=x_grid, name='Pde')

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
                                                   mt_=steps_per_day,
                                                   rfrate=rfrate,
                                                   vt=vt,
                                                   it=it,
                                                   mt=mt,
                                                   function_type=mcv.FunctionType.SURVIVAL_PROB)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra_sol = pd.Series(volterra, index=x_grid, name='Volterra')


    # simulate mc and apply barrier
    tic = time.perf_counter()
    mc_price = mce.simulate_x_survival(t=t,
                                       x_grid=x_grid,
                                       x0=x0,
                                       psi0=psi0,
                                       v0=v0,
                                       theta=theta,
                                       kappa=kappa,
                                       rho=rho,
                                       volvol=volvol,
                                       rfrate=rfrate,
                                       nb_path=nb_mc_path,
                                       steps_per_day=8*steps_per_day,
                                       seed_id=seed_id)

    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC ")
    mc_price = pd.Series(mc_price, index=x_grid, name='MC')

    data = pd.concat([mc_price, volterra_sol, pde_sol_2s], axis=1)

    kwargs = {'legend_loc': 'upper left'}
    colors = put.get_n_colors(n=len(data.columns), first_color_fixed=True)
    colors[0] = mc_color
    with sns.axes_style("darkgrid"):
        fig1, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
        pli.plot_line(df=data,
                      title=None,
                      yvar_format='{:0.0%}',
                      xvar_format='{:.2f}',
                      colors=colors,
                      ylabel='Premium',
                      xlabel='x_0',
                      fontsize=14,
                      ax=ax,
                      **kwargs)

    my_indices = [x%int(nb_x_grid/15) == 0 for x in np.arange(nb_x_grid)]
    my_indices[0] = False
    fig2, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
    ptb.plot_df_table(df=data.iloc[my_indices, :],
                             # add_index_as_column=False,
                             var_format='{:0.4f}',
                             index_column_name='x',
                             special_columns_colors=[(0, 'steelblue')],
                             ax=ax,
                             **kwargs)

    return fig1, fig2


def plot_heston_barrier_call(t: float,
                             psi0: float,
                             xmax: float,
                             log_strikes: np.ndarray,
                             x0: float,
                             v0: float,
                             theta: float,
                             kappa: float,
                             rho: float,
                             volvol: float,
                             rfrate: float = 0.0,
                             nb_x_grid: int = 500,
                             nb_mc_pde_path: int = 10000,
                             nb_mc_path: int = 10000,
                             steps_per_day: int = 6,
                             seed_id: int = 3,
                             mc_color: str = 'lightblue'
                             ) -> (plt.Figure, plt.Figure):
    # get all on same grid
    strikes = np.exp(log_strikes)
    x_grid = np.linspace(psi0, xmax, nb_x_grid)

    vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         rfrate=rfrate,
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
                                            rfrate=rfrate,
                                            nb_path=nb_mc_pde_path,
                                            x_grid=x_grid,
                                            mt_=steps_per_day,
                                            vt=vt,
                                            it=it,
                                            mt=mt,
                                            log_strikes=log_strikes,
                                            function_type=mcv.FunctionType.CALL)
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
                                                   mt_=steps_per_day,
                                                   rfrate=rfrate,
                                                   vt=vt,
                                                   it=it,
                                                   mt=mt,
                                                   log_strikes=log_strikes,
                                                   function_type=mcv.FunctionType.CALL)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra_sol = pd.Series(volterra, index=strikes, name='Volterra')

    # apply mc for barrier
    tic = time.perf_counter()
    x_terminal = mce.simulate_x_barrier_paths(t=t,
                                              x0=x0,
                                              psi0=psi0,
                                              v0=v0,
                                              theta=theta,
                                              kappa=kappa,
                                              rho=rho,
                                              volvol=volvol,
                                              rfrate=rfrate,
                                              nb_path=nb_mc_path,
                                              steps_per_day=8*steps_per_day,
                                              is_barrier=True,
                                              seed_id=seed_id)
    mc_prices = mce.compute_barrier_calls(xt=x_terminal,
                                          log_strikes=log_strikes,
                                          discfactor=np.exp(-rfrate*t))
    mc_prices = pd.Series(mc_prices, index=strikes, name='MC')
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC with barrier")

    """
    tic = time.perf_counter()
    x_terminal = mce.simulate_x_barrier_paths(t=t,
                                              x0=x0,
                                              psi0=psi0,
                                              v0=v0,
                                              theta=theta,
                                              kappa=kappa,
                                              rho=rho,
                                              volvol=volvol,
                                              rfrate=rfrate,
                                              nb_path=nb_mc_path,
                                              steps_per_day=8*steps_per_day,
                                              is_barrier=False,
                                              seed_id=seed_id)
    call_prices = mce.compute_barrier_calls(xt=x_terminal,
                                            log_strikes=log_strikes,
                                            discfactor=np.exp(-rfrate*t))
    call_prices = pd.Series(call_prices, index=strikes, name='Vanilla')
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC no barrier")
    """

    data = pd.concat([mc_prices, volterra_sol, pde_sol_2s], axis=1)
    print(data)

    kwargs = {'legend_loc': 'upper center'}
    colors = put.get_n_colors(n=len(data.columns), first_color_fixed=True)
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

    fig2, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
    ptb.plot_df_table(df=data,
                      # add_index_as_column=False,
                      var_format='{:0.4f}',
                      index_column_name='Strike',
                      special_columns_colors=[(0, 'steelblue')],
                      ax=ax,
                      **kwargs)

    return fig1, fig2


def plot_drift_vol(t: float,
                   x0: float,
                   v0: float,
                   xmax: float,
                   theta: float,
                   kappa: float,
                   rho: float,
                   volvol: float,
                   psi0: float = -0.3,
                   nb_path: int = 10,
                   path_id: int = 2,
                   nb_x_grid: int = 500,
                   mt_: int = 6,
                   is_barplot: bool = False
                   ) -> (plt.Figure, plt.Figure, plt.Figure):

    vt, it, mt = mce.simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         nb_path=nb_path)

    t_grid = np.linspace(0, t, int(np.ceil(mce.YEAR_DAYS * t)))
    rho_1 = (1.0 - rho*rho)
    upsilon_t = rho_1*it[:, path_id]
    lambda_t = mt[:, path_id] / (rho_1 * vt[:, path_id])

    kwargs = {'xvar_major_ticks': np.linspace(0, t, 10), 'yvar_format': '{:0.2f}', 'xvar_format': '{:.1f}'}
    # 1
    datas = {'(A) Advection': pd.Series(mt[:, path_id], index=t_grid),
             '(B) Diffusion': pd.Series(rho_1 * vt[:, path_id], index=t_grid)}
    with sns.axes_style("darkgrid"):
        fig1, axs = plt.subplots(2, 1, figsize=FIG_SIZE, tight_layout=True)
        for idx, (title, data) in enumerate(datas.items()):
            if is_barplot:
                plot_bars(df=data,
                          title=title,
                          legend_loc=None,
                          skip_y_axis=True,
                          xlabel='time',
                          fontsize=14,
                          ax=axs[idx],
                          **kwargs)
            else:
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
            if is_barplot:
                plot_bars(df=data,
                          title=title,
                          legend_loc=None,
                          skip_y_axis=True,
                          xlabel='time',
                          fontsize=14,
                          ax=axs[idx],
                          **kwargs)
            else:
                pli.plot_line(df=data,
                              title=title,
                              legend_loc=None,
                              yvar_format='{:0.2f}',
                              xvar_format='{:.2f}',
                              xlabel='time' if idx==0 else r'$\Upsilon(t)$',
                              fontsize=14,
                              ax=axs[idx])

    # 3
    gridx = np.linspace(psi0, xmax, nb_x_grid)
    green_volterra = vgr.compute_volterra_green(maturity=t,
                                                mt=mt_,
                                                xi=psi0,
                                                gridx=gridx,
                                                advection=mt[:, path_id],
                                                diffusion=rho_1 * vt[:, path_id])
    green_volterra = pd.Series(green_volterra, index=gridx, name='Volterra')
    green_pde = vgr.compute_pde_green(maturity=t,
                                      mt=mt_,
                                      gridx=gridx,
                                      advection=mt[:, path_id],
                                      diffusion=rho_1 * vt[:, path_id])

    green_pde = pd.Series(green_pde, index=gridx, name='PDE')
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
                      y_limits=(0.0, None),
                      ax=ax)

    return fig1, fig2, fig3


class LocalTests(Enum):
    PLOT_DRIFT_VAR = 1
    PLOT_HESTON_BARRIER_GREEN = 2
    PLOT_HESTON_SURVIVAL = 3
    PLOT_HESTON_BARRIER_CALL = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    local_path = lp.get_output_path()
    t = 1.0
    x0 = 0.0
    v0 = 0.25
    theta = 0.20
    kappa = 1.0
    rho = -0.3
    volvol = 0.4
    psi0 = -0.5 # log(B/S)
    xmax = 2.0
    rfrate = 0.03

    if local_test == LocalTests.PLOT_DRIFT_VAR:
        fig1, fig2, fig3 = plot_drift_vol(t=t,
                                          x0=x0,
                                          v0=v0,
                                          xmax=xmax,
                                          psi0=psi0,
                                          theta=theta,
                                          kappa=kappa,
                                          rho=rho,
                                          volvol=volvol,
                                          path_id=3)

        fu.save_fig(fig=fig1, local_path=local_path, file_name='Fig2')
        fu.save_fig(fig=fig2, local_path=local_path, file_name='Fig3')
        fu.save_fig(fig=fig3, local_path=local_path, file_name='Fig4')

    elif local_test == LocalTests.PLOT_HESTON_BARRIER_GREEN:
        fig1, fig2 = plot_heston_barrier_green(t=t,
                                               psi0=psi0,  # log(B/S)
                                               x0=x0,
                                               v0=v0,
                                               xmax=xmax,
                                               theta=theta,
                                               kappa=kappa,
                                               rho=rho,
                                               volvol=volvol,
                                               nb_x_grid=200,
                                               nb_mc_pde_path=5000,  #2000,
                                               nb_mc_path=400000,
                                               seed_id=2,
                                               is_title=False)

        fu.save_fig(fig=fig1, local_path=local_path, file_name='Fig9a')
        fu.save_fig(fig=fig2, local_path=local_path, file_name='Fig9b')

    elif local_test == LocalTests.PLOT_HESTON_SURVIVAL:

        psi0 = -0.5
        fig1, fig2 = plot_heston_barrier_survival(t=t,
                                                  x0=0.0,
                                                  psi0=psi0,  # log(B/S)
                                                  xmax=xmax,
                                                  v0=v0,
                                                  theta=theta,
                                                  kappa=kappa,
                                                  rho=rho,
                                                  volvol=volvol,
                                                  steps_per_day=6,
                                                  rfrate=rfrate,
                                                  nb_x_grid=400,
                                                  nb_mc_pde_path=6000,  #2000,
                                                  nb_mc_path=200000,
                                                  seed_id=3)

        fu.save_fig(fig=fig1, local_path=local_path, file_name='Fig10a')
        fu.save_fig(fig=fig2, local_path=local_path, file_name='Fig10b')

    elif local_test == LocalTests.PLOT_HESTON_BARRIER_CALL:

        strikes = np.linspace(0.9, 1.5, 13)
        log_strikes = np.log(strikes)
        print(log_strikes)
        psi0 = log_strikes[0]
        fig1, fig2 = plot_heston_barrier_call(t=t,
                                              x0=0.0,
                                              psi0=psi0,  # log(B/S)
                                              xmax=xmax,
                                              log_strikes=log_strikes,
                                              v0=v0,
                                              theta=theta,
                                              kappa=kappa,
                                              rho=rho,
                                              volvol=volvol,
                                              steps_per_day=6,
                                              rfrate=rfrate,
                                              nb_x_grid=400,
                                              nb_mc_pde_path=2000,  #2000,
                                              nb_mc_path=200000,
                                              seed_id=3)

        fu.save_fig(fig=fig1, local_path=local_path, file_name='Fig11')
        #fu.save_fig(fig=fig2, local_path=local_path, file_name='Fig11b')

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.PLOT_HESTON_BARRIER_CALL

    run_local_test(local_test=local_test)
