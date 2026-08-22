
# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from typing import Tuple
from enum import Enum

# qis
import qis.plots.lineplot as pli
from qis.utils.np_ops import np_min
from qis.plots import histogram as hist

YEAR_DAYS = 52
SEED_ID = 3


@njit
def simulate_x_barrier_paths(t: float,
                             x0:  float,
                             psi0: float,  # = log(B/S0)
                             v0: float,
                             theta: float,
                             kappa: float,
                             rho: float,
                             volvol: float,
                             rfrate: float = 0.0,
                             nb_path: int = 100000,
                             steps_per_day: int = 6,
                             is_barrier: bool = True,
                             seed_id: int = SEED_ID
                             ) -> np.ndarray:
    np.random.seed(seed_id)
    nb_steps = int(np.ceil(steps_per_day*YEAR_DAYS * t))  # daily steps
    dt = t / nb_steps
    sdt = np.sqrt(dt)
    rho_1 = np.sqrt(1.0-rho*rho)
    xt = np.zeros((nb_steps, nb_path))
    x0 = x0*np.ones(nb_path)
    v0_ = v0 * np.ones(nb_path)
    for t in range(nb_steps):
        b = sdt * np.random.normal(0, 1, size=(nb_path))
        w = sdt * np.random.normal(0, 1, size=(nb_path))
        sigma0_ = np.sqrt(v0_)
        x0 = x0 + rfrate*dt - 0.5 * v0_ * dt + sigma0_ * (rho * b + rho_1 * w)
        v0_ = np.maximum(v0_ + kappa*(theta-v0_) * dt + volvol * sigma0_ * b, 1e-6)
        xt[t, :] = x0
    if is_barrier:
        path_min = np_min(xt, axis=0)[0]
        terminal = xt[-1, :]
        xt = np.where(np.greater(path_min, psi0), terminal, np.nan)
    else:
        xt = xt[-1, :]
    return xt


@njit
def simulate_x_survival(t: float,
                        x0:  float,
                        psi0: float,  # = log(B/S0)
                        x_grid: np.ndarray,
                        v0: float,
                        theta: float,
                        kappa: float,
                        rho: float,
                        volvol: float,
                        rfrate: float = 0.0,
                        nb_path: int = 100000,
                        steps_per_day: int = 6,
                        seed_id: int = SEED_ID
                        ) -> np.ndarray:
    np.random.seed(seed_id)
    nb_steps = int(np.ceil(steps_per_day*YEAR_DAYS * t))  # daily steps
    dt = t / nb_steps
    sdt = np.sqrt(dt)
    rho_1 = np.sqrt(1.0-rho*rho)
    xt = np.zeros((nb_steps, nb_path))
    x0 = x0*np.ones(nb_path)
    v0_ = v0 * np.ones(nb_path)
    for t_ in range(nb_steps):
        b = sdt * np.random.normal(0, 1, size=(nb_path))
        w = sdt * np.random.normal(0, 1, size=(nb_path))
        sigma0_ = np.sqrt(v0_)
        x0 = x0 + rfrate*dt - 0.5 * v0_ * dt + sigma0_ * (rho * b + rho_1 * w)
        v0_ = np.maximum(v0_ + kappa*(theta-v0_) * dt + volvol * sigma0_ * b, 1e-6)
        xt[t_, :] = x0

    path_min = np_min(xt, axis=0)[0]
    price = np.zeros_like(x_grid)
    for idx, xi0 in enumerate(x_grid):
        price[idx] = np.mean(np.where(np.greater(path_min+xi0, psi0), 1.0, 0.0))
    price = price*np.exp(-t*rfrate)
    return price


@njit
def compute_barrier_calls(xt: np.ndarray,
                          log_strikes: np.ndarray,
                          discfactor: float = 1.0
                          ) -> np.ndarray:
    """
    compute barrier call convolution
    """
    spot = np.exp(xt)
    strikes = np.exp(log_strikes)
    avg_values = np.zeros(log_strikes.shape[0])
    for idx, strike in enumerate(strikes):
        payoff = np.where(np.isnan(xt), 0.0, np.maximum(spot - strike, 0.0))
        avg_values[idx] = discfactor*np.mean(payoff)
    return avg_values


@njit
def simulate_bm_barrier_paths(t_grid: np.ndarray,
                              barrier_t: np.ndarray,
                              x0:  float,
                              v0: float,
                              nb_path: int = 100000,
                              seed_id: int = SEED_ID
                              ) -> np.ndarray:
    """
    deterministic dbarrier
    """
    np.random.seed(seed_id)
    vol = np.sqrt(v0)
    xt = np.zeros((t_grid.shape[0]-1, nb_path))
    x0 = x0*np.ones(nb_path)
    sdts = vol*np.sqrt(t_grid[1:]-t_grid[:-1])
    paths_survived_ind = np.full(nb_path, True)
    for t, sdt in enumerate(sdts):
        x0 = x0 + sdt * np.random.normal(0, 1, size=(nb_path))
        paths_survived_ind = np.logical_and(paths_survived_ind, np.greater(x0, barrier_t[t+1]))
        xt[t, :] = np.where(paths_survived_ind, x0, np.nan)

    return xt[-1, :]


@njit
def simulate_bm_barrier_drift_paths(t_grid: np.ndarray,
                                    barrier_t: np.ndarray,
                                    x0:  float,
                                    v0: float,
                                    nb_path: int = 100000,
                                    seed_id: int = SEED_ID
                                    ) -> np.ndarray:
    """
    deterministic drift with barrier at zero
    """
    np.random.seed(seed_id)
    vol = np.sqrt(v0)
    xt = np.zeros((t_grid.shape[0]-1, nb_path))
    x0 = x0*np.ones(nb_path)
    sdts = vol*np.sqrt(t_grid[1:]-t_grid[:-1])
    paths_survived_ind = np.full(nb_path, True)
    for t, sdt in enumerate(sdts):
        x0 = x0 + sdt * np.random.normal(0, 1, size=(nb_path))
        paths_survived_ind = np.logical_and(paths_survived_ind, np.greater(x0, barrier_t[t]))
        xt[t, :] = np.where(paths_survived_ind, x0, np.nan)

    return xt[-1, :]


@njit
def simulate_heston_decomposition_paths(t: float,
                                        v0: float,
                                        theta: float,
                                        kappa: float,
                                        rho: float,
                                        volvol: float,
                                        rfrate: float = 0.0,
                                        nb_path: int = 100000,
                                        seed_id: int = SEED_ID
                                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed_id)
    nb_steps = int(np.ceil(YEAR_DAYS * t))  # daily steps
    ttms = np.linspace(0, t, nb_steps)
    dt = ttms[1]
    sdt = np.sqrt(dt)

    vt = np.zeros((nb_steps, nb_path))
    it = np.zeros((nb_steps, nb_path))
    mt = np.zeros((nb_steps, nb_path))

    # initial values
    drift0 = rfrate - rho * kappa * theta / volvol if volvol > 0.0 else 0.0
    drift1 = (rho*kappa/volvol - 0.5 if volvol > 0.0 else - 0.5) / 2.0
    beta = rho / volvol / dt if volvol > 0.0 else 0.0

    v00 = v0 * np.ones(nb_path)
    vt[0, :] = v00

    v0_ = v00
    # 1. fill v and it and yt
    for t, ttm in enumerate(ttms):
        if t > 0:
            b = sdt * np.random.normal(0, 1, size=(nb_path))
            dv = kappa * (theta - v0_) * dt + volvol * np.sqrt(v0_) * b
            it[t, :] = it[t-1, :] + (v0_+0.5*dv)*dt
            v1 = np.maximum(v0_ + dv, 1e-6)
            vt[t, :] = v1
            mt[t, :] = drift0 + drift1 * (v1+v0_) + beta * (v1-v0_)
            v0_ = v1

    return vt, it, mt


def plot_heston_decomposition_pdfs(t: float,
                                   x0: float,
                                   sigma0: float,
                                   theta: float,
                                   kappa: float,
                                   rho: float,
                                   volvol: float,
                                   nb_path: int = 100000,
                                   ) -> None:

    vt, it, mt = simulate_heston_decomposition_paths(t=t,
                                                     v0=sigma0,
                                                     theta=theta,
                                                     kappa=kappa,
                                                     rho=rho,
                                                     volvol=volvol,
                                                     nb_path=nb_path)

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)

    vars = {'i_t': it, 'm_t': mt}
    n_cut = 50
    for idx, (key, var) in enumerate(vars.items()):
        data_t = pd.DataFrame(var[:, :n_cut], columns=[f"path {x+1}" for x in range(n_cut)])
        pli.plot_line(df=data_t,
                      title=f"{n_cut} Paths of {key}",
                      legend_loc=None,
                      yvar_format='{:.2f}',
                      ax=axs[idx][0])

        hist.plot_histogram(df=pd.Series(var[-1, :], name=key),
                            pdf_type=hist.PdfType.HISTOGRAM,
                            desc_table_type=hist.DescTableType.EXTENSIVE,
                            title=f"Terminal distribution of {key}",
                            ax=axs[idx][1])


@njit
def simulate_heston_iv_jv(t: float,
                          v0: float,
                          theta: float,
                          kappa: float,
                          rho: float,
                          volvol: float,
                          nb_path: int = 100000,
                          year_days: float = 360.0
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    nb_steps = int(np.ceil(year_days * t))  # daily steps
    ttms = np.linspace(0, t, nb_steps)
    dt = ttms[1]
    sdt = np.sqrt(dt)

    vt = v0 * np.ones(nb_path)
    iv = np.zeros(nb_path)
    jv = np.zeros(nb_path)

    # 1. fill v and it and yt
    for t, ttm in enumerate(ttms):
        b = sdt * np.random.normal(0, 1, size=(nb_path))
        db = np.sqrt(vt) * b
        dv = kappa * (theta - vt) * dt + volvol * db
        jv = jv + db
        iv = iv + (vt+0.5*dv)*dt
        vt = np.maximum(vt + dv, 1e-6)

    return vt, iv, jv


class LocalTests(Enum):
    PLOT_HESTON_DECOMPOSITION = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    t = 1.0

    if local_test == LocalTests.PLOT_HESTON_DECOMPOSITION:
        plot_heston_decomposition_pdfs(t=t,
                                       x0=0.5,
                                       sigma0=0.2,
                                       theta=0.2,
                                       kappa=4.0,
                                       rho=-0.5,
                                       volvol=0.5,
                                       nb_path=10000)

    plt.show()


if __name__ == '__main__':

    local_test = LocalTests.PLOT_HESTON_DECOMPOSITION

    run_local_test(local_test=local_test)
