
# packages
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy import stats
from enum import Enum

import qis.plots.lineplot as pli
from . import analytic as ana
from . import volterra as volt
from .mc_engine import YEAR_DAYS, simulate_heston_decomposition_paths, simulate_x_barrier_paths

from . import green_solutions as vgr


class FunctionType(Enum):
    GREEN = 1
    SURVIVAL_PROB = 2
    CALL = 3


@njit
def convolute_green_with_payoff(x_grid: np.ndarray,
                                log_strikes: np.ndarray,
                                green_ttms: np.ndarray,  # [x_grid, nb_bath]
                                discfactor: float = 1.0
                                ) -> np.ndarray:
    """
    compute barrier call convolution
    """
    nb_path = green_ttms.shape[1]
    spot = np.exp(x_grid)
    strikes = np.exp(log_strikes)
    dx = x_grid[1] - x_grid[0]
    avg_values = np.zeros(log_strikes.shape[0])
    for idx, strike in enumerate(strikes):
        payoff = np.maximum(spot - strike, 0.0)
        option_prices = np.zeros(nb_path)
        for path in np.arange(nb_path):
            option_prices[path] = dx * np.sum(payoff * green_ttms[:, path])
        avg_values[idx] = discfactor*np.mean(option_prices)
    return avg_values


@njit
def compute_heston_volterra_barrier(t: float,
                                    psi0: float,   # = log (B/S)
                                    x0: float,
                                    v0: float,
                                    theta: float,
                                    kappa: float,
                                    rho: float,
                                    volvol: float,
                                    x_grid: np.ndarray,
                                    rfrate: float = 0.0,
                                    mt_: int = 6,
                                    nb_path: int = 10000,
                                    vt: np.ndarray = None,
                                    it: np.ndarray = None,
                                    mt: np.ndarray = None,
                                    log_strikes: np.ndarray = np.array([0.0]),
                                    function_type: FunctionType = FunctionType.GREEN
                                    ) -> np.ndarray:

    if vt is None:
        vt, it, mt = simulate_heston_decomposition_paths(t=t,
                                                         v0=v0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         rfrate=rfrate,
                                                         nb_path=nb_path)
    # apply analytic
    rho_1 = 1.0 - rho*rho
    green_ttms = np.zeros((x_grid.shape[0], nb_path))

    for path in np.arange(nb_path):

        if function_type in [FunctionType.GREEN, FunctionType.CALL]:
            this = vgr.compute_volterra_green(maturity=t,
                                              mt=mt_,
                                              xi=psi0,
                                              gridx=x_grid,
                                              advection=mt[:, path],
                                              diffusion=rho_1 * vt[:, path])

        else:
            this = vgr.compute_volterra_survival_prob(maturity=t,
                                                      mt=mt_,
                                                      xi=psi0,
                                                      gridx=x_grid,
                                                      advection=mt[:, path],
                                                      diffusion=rho_1 * vt[:, path])
        # need to fix it
        if np.any(np.abs(this)>1e3):
            this = np.full_like(x_grid, fill_value=np.nan)
        green_ttms[:, path] = this

    if function_type == FunctionType.CALL:
        avg_values = convolute_green_with_payoff(x_grid=x_grid,
                                                 log_strikes=log_strikes,
                                                 green_ttms=green_ttms,
                                                 discfactor=np.exp(-rfrate*t))
    else:
        avg_values = np.zeros(x_grid.shape[0])
        for nn in np.arange(x_grid.shape[0]):
            avg_values[nn] = np.nanmean(green_ttms[nn, :])

        if function_type == FunctionType.SURVIVAL_PROB:
            avg_values = np.exp(-rfrate*t)*avg_values

    return avg_values


def plot_heston_zero_corr(t: float,
                          x0: float,
                          sigma0: float,
                          theta: float,
                          kappa: float,
                          volvol: float,
                          nb_path: int = 10000,
                          steps_per_day: int = 1
                          ) -> None:

    # get all on same grid
    x_grid = np.linspace(0.0, 1.5, 500)

    # analytic
    tic = time.perf_counter()
    green_x = ana.compute_green_x_barrier0(x_grid=x_grid, ttm=t, x0=x0, v0=sigma0*sigma0, theta=theta, kappa=kappa, volvol=volvol)
    analytic = pd.Series(green_x, index=x_grid, name='analytic green')
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute analytic")

    # volterra
    tic = time.perf_counter()
    volterra = compute_heston_volterra_barrier(t=t,
                                               x0=x0,
                                               v0=sigma0,
                                               theta=theta,
                                               kappa=kappa,
                                               rho=0.0,
                                               volvol=volvol,
                                               nb_path=10000,
                                               x_grid=x_grid)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra = pd.Series(volterra, index=x_grid, name='volterra')

    # apply mc
    tic = time.perf_counter()
    x_terminal = simulate_x_barrier_paths(t=t,
                                          x0=x0,
                                          v0=sigma0,
                                          theta=theta,
                                          kappa=kappa,
                                          rho=0.0,
                                          volvol=volvol,
                                          nb_path=nb_path,
                                          steps_per_day=steps_per_day)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC")

    # mc kernel
    prob_survival = np.nansum(np.where(x_terminal > 0.0, 1, 0.0)) / nb_path
    kernel = stats.gaussian_kde(x_terminal[np.isfinite(x_terminal)])
    y = prob_survival*kernel(x_grid)
    mc_kernel = pd.Series(y, index=x_grid, name='mc kernel')

    dy = x_grid[1]-x_grid[0]
    data = pd.concat([analytic, volterra, mc_kernel], axis=1)
    print(f"sum={data.sum(axis=0)}")

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), tight_layout=True)

    axs[0].hist(x=np.where(np.isfinite(x_terminal), x_terminal,0.0),
                bins=100,
                color='lightblue',
                density=True)

    pli.plot_line(df=data,
                  title=f"Analytic vs MC solution, Prob survival = {prob_survival}",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[0])
    axs[0].set_ylim(0.0, 1.1*np.max(volterra))

    diff_data = pd.concat([np.subtract(volterra, analytic).rename('volterra-analytic'),
                           np.subtract(volterra, mc_kernel).rename('volterra-mc_kernel')],
                          axis=1)

    pli.plot_line(df=diff_data,
                  title=f"Volterra differences",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[1])


def plot_heston_barrier(t: float,
                        x0: float,
                        sigma0: float,
                        theta: float,
                        kappa: float,
                        rho: float,
                        volvol: float,
                        nb_path: int = 10000,
                        steps_per_day: int = 10
                        ) -> None:
    # get all on same grid
    x_grid = np.linspace(0.0, 1.5, 250)
    # volterra
    tic = time.perf_counter()
    volterra = compute_heston_volterra_barrier(t=t,
                                               x0=x0,
                                               v0=sigma0,
                                               theta=theta,
                                               kappa=kappa,
                                               rho=rho,
                                               volvol=volvol,
                                               nb_path=10000,
                                               x_grid=x_grid)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute Volterra")
    volterra = pd.Series(volterra, index=x_grid, name='volterra')

    # apply mc
    tic = time.perf_counter()
    x_terminal = simulate_x_barrier_paths(t=t,
                                          x0=x0,
                                          v0=sigma0,
                                          theta=theta,
                                          kappa=kappa,
                                          rho=rho,
                                          volvol=volvol,
                                          nb_path=nb_path,
                                          steps_per_day=steps_per_day)
    toc = time.perf_counter()
    print(f"{toc - tic:0.2f} secs to compute MC")

    # mc kernel
    prob_survival = np.nansum(np.where(x_terminal > 0.0, 1, 0.0)) / nb_path
    kernel = stats.gaussian_kde(x_terminal[np.isfinite(x_terminal)])
    y = prob_survival*kernel(x_grid)
    mc_kernel = pd.Series(y, index=x_grid, name='mc kernel')

    dy = x_grid[1]-x_grid[0]
    data = pd.concat([volterra, mc_kernel], axis=1)
    print(f"sum={data.sum(axis=0)}")

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), tight_layout=True)

    axs[0].hist(x=np.where(np.isfinite(x_terminal), x_terminal,0.0),
                bins=100,
                density=True)

    pli.plot_line(df=data,
                  title=f"Volterra vs MC solution, Prob survival = {prob_survival}",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[0])
    axs[0].set_ylim(0.0, 1.1*np.max(volterra))

    diff_data = np.subtract(volterra, mc_kernel).rename('volterra-mc_kernel')
    pli.plot_line(df=diff_data,
                  title=f"Volterra differences",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[1])


def plot_heston_barrier_path(t: float,
                             x0: float,
                             sigma0: float,
                             theta: float,
                             kappa: float,
                             rho: float,
                             volvol: float,
                             nb_path: int = 1,
                             steps_per_day: int = 10
                             ) -> None:
    # get all on same grid
    x_grid = np.linspace(0.0, 1.5, 500)

    vt, it, mt = simulate_heston_decomposition_paths(t=t,
                                                     v0=sigma0,
                                                     theta=theta,
                                                     kappa=kappa,
                                                     rho=rho,
                                                     volvol=volvol,
                                                     nb_path=nb_path)

    # apply analytic
    rho_1 = 1.0 - rho*rho
    nb_steps = int(np.ceil(YEAR_DAYS * t))  # daily steps
    ttms = np.linspace(0, t, nb_steps+1)[1:]
    green_ttms = np.zeros((x_grid.shape[0], nb_path))

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), tight_layout=True)

    data = pd.concat([pd.Series(it[:, 0], index=ttms, name='I_t'),
                      pd.Series(mt[:, 0], index=ttms, name='M_t')],
                     axis=1)
    pli.plot_line(df=data,
                  title=f"Simulated paths of I_t, M_t",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[0])


    # Volterra
    #q2 = rho_1 * it[:, path] / ttms
    #n_t = np.interp(x=ttms, xp=q2, fp=ttms)
    upsilon_k = rho_1 * it[:, 0]
    #upsilon_k = it[:, path]
    ##n_upsilon = np.interp(x=ttms[-1]*upsilon_k/upsilon_k[-1], xp=ttms, fp=mt[:, path])
    n_upsilon = np.interp(x=upsilon_k, xp=ttms, fp=mt[:, 0])

    nu_k = volt.solve_volterra(upsilon_k=upsilon_k,
                               barrier_k=-mt[:, 0],
                               x0=x0,
                               barrier_dt_k=None)

    nu = pd.Series(nu_k, index=ttms, name='nu_k')
    pli.plot_line(df=nu,
                  title=f"Solution of Volterra nu(t)",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[1])

    f = volt.compute_f(upsilon_k=upsilon_k,
                       barrier_k=-n_upsilon,
                       phi_k=nu_k,
                       y=x_grid)
    green = volt.compute_green(upsilon_k=upsilon_k,
                               barrier_k=-n_upsilon,
                               x_grid=x_grid,
                               x0=x0,
                               is_terminal=True)
    data = pd.concat([pd.Series(f, index=x_grid, name='f'),
                      pd.Series(green, index=x_grid, name='F and Green functions')], axis=1)
    pli.plot_line(df=data,
                  title=f"Green",
                  yvar_format='{:.2f}',
                  xvar_format='{:.2f}',
                  fontsize=14,
                  ax=axs[2])


class LocalTests(Enum):
    PLOT_ANALYTIC_ZERO_CORR = 1
    PLOT_HESTON_BARRIER = 2
    PLOT_HESTON_BARRIER_PATH = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    t = 1.0

    if local_test == LocalTests.PLOT_ANALYTIC_ZERO_CORR:
        sigma0 = 0.2
        plot_heston_zero_corr(t=t,
                              x0=0.25,
                              sigma0=sigma0,
                              theta=sigma0 * sigma0,
                              kappa=4.0,
                              volvol=0.5,
                              steps_per_day=1,
                              nb_path=100000)

    elif local_test == LocalTests.PLOT_HESTON_BARRIER:
        sigma0 = 0.2
        plot_heston_barrier(t=t,
                            x0=0.25,
                            sigma0=sigma0,
                            theta=sigma0*sigma0,
                            kappa=4.0,
                            rho=-0.5,
                            volvol=0.25,
                            nb_path=500)

    elif local_test == LocalTests.PLOT_HESTON_BARRIER_PATH:
        sigma0 = 0.2
        plot_heston_barrier_path(t=t,
                                 x0=0.25,
                                 sigma0=sigma0,
                                 theta=sigma0*sigma0,
                                 kappa=4.0,
                                 rho=0.0,
                                 volvol=0.5,
                                 nb_path=1000)

    # maximize figure on screen
    print('done')
    mng = plt.get_current_fig_manager()
    if hasattr(mng.window, "state"):
        mng.window.state('zoomed')

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.PLOT_HESTON_BARRIER

    run_local_test(local_test=local_test)
