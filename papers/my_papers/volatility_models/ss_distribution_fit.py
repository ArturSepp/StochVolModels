import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special as sps
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Optional
from enum import Enum
import qis as qis

from my_papers.volatility_models.load_data import fetch_ohlc_vol
from stochvolmodels import LogSvParams, HestonParams


def lognormal_sv_ss_pdf(sigma: np.ndarray,
                        params: LogSvParams
                        ) -> np.ndarray:
    nu = 2.0*(params.kappa2*params.theta-params.kappa1)/params.vartheta2 - 1.0
    q = 2.0*params.kappa1*params.theta/params.vartheta2
    b = 2.0*params.kappa2/params.vartheta2
    if params.kappa1 >= 1e-6:
        if params.kappa2 >= 1e-6:
            c = np.power(b/q, nu/2.0) / (2.0*sps.kv(nu, 2.0*np.sqrt(q*b)))
        else:
            c = np.power(q, -nu) / sps.gamma(-nu)
    else:
        raise NotImplementedError(f"kappa1 = 0 is not implemented")
    g = c*np.power(sigma, nu-1.0)*np.exp(-q*np.reciprocal(sigma)-b*sigma)
    return g


def lognormal_sv_ss_log_pdf(log_sigma: np.ndarray,
                            params: LogSvParams
                            ) -> np.ndarray:
    sigma = np.exp(log_sigma)
    return sigma*lognormal_sv_ss_pdf(sigma=sigma, params=params)


def heston_ss_pdf(var: np.ndarray,
                  params: HestonParams
                  ) -> np.ndarray:
    alpha = np.square(params.volvol) / (2.0*params.kappa)
    beta = params.theta / alpha
    c = np.power(alpha, beta) * sps.gamma(beta)
    g = np.power(var, beta-1.0)*np.exp(-var/alpha) / c
    return g


def heston_ss_log_vol_pdf(log_sigma: np.ndarray,
                          params: HestonParams
                          ) -> np.ndarray:
    var = np.exp(2.0*log_sigma)
    return 2.0*var*heston_ss_pdf(var=var, params=params)


def compute_vol_histogram(vol: pd.Series, bins: int = 100) -> pd.Series:
    # Get histogram of original data
    y, x = np.histogram(vol.to_numpy(), bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    hist = pd.Series(y / np.sum(y), index=x)
    return hist


def estimate_beta_sv():
    pass


def plot_estimated_beta_sv(vol: pd.Series,
                           bins: int = 50,
                           is_log: bool = True
                           ):
    hist = compute_vol_histogram(vol=vol, bins=bins)
    params = LogSvParams(theta=0.17, kappa1=2.0, kappa2=2.0, beta=-1.0, volvol=2.0)
    sigma = hist.index.to_numpy()
    dv = sigma[1] - sigma[0]
    analytic = lognormal_sv_ss_pdf(sigma=sigma, params=params)*dv
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)

        if is_log:
            analytic = pd.Series(analytic, index=hist.index)
            qis.plot_line(df=analytic, ax=ax)
            qis.plot_line(hist, colors=['red'],
                          markers=["*"],
                          linewidth=0.0,
                          ax=ax)
            ax.set_yscale('log')
        else:
            hist.plot.bar(stacked=False, color='green', edgecolor='none', linewidth=0.0, width=1.0, ax=ax)
            current_ticks = ax.get_xticks()  # sigma' is mapped into ticks
            analytic = pd.Series(analytic, index=current_ticks)
            qis.plot_line(analytic, ax=ax)
            qis.reset_xticks(ax=ax, data=hist.index.to_numpy())


def plot_estimated_beta_log_sv(vol: pd.Series,
                               params: LogSvParams = LogSvParams(theta=0.14, kappa1=2.0, kappa2=2.0, beta=-1.0, volvol=1.0),
                               bins: int = 50,
                               is_log: bool = True
                               ):
    log_sigma = np.log(vol).replace([np.inf, -np.inf], np.nan).dropna()
    hist = compute_vol_histogram(vol=log_sigma, bins=bins)
    dv = hist.index[1] - hist.index[0]
    analytic = lognormal_sv_ss_log_pdf(log_sigma=hist.index.to_numpy(), params=params)*dv
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)

        if is_log:
            analytic = pd.Series(analytic, index=hist.index)
            qis.plot_line(df=analytic, ax=ax)
            qis.plot_line(hist, colors=['red'],
                          markers=["*"],
                          linewidth=0.0,
                          ax=ax)
            ax.set_yscale('log')
        else:
            # hist.plot.bar(stacked=False, color='green', edgecolor='none', linewidth=0.0, width=1.0, ax=ax)
            analytic = pd.Series(analytic, index=hist.index, name='analytic')
            distr = pd.concat([hist, analytic], axis=1)
            qis.plot_line(distr, ax=ax)
            # qis.reset_xticks(ax=ax, data=hist.index.to_numpy())


def fit_distribution_log_sv(vol: pd.Series,
                            bins: int = 50
                            ) -> LogSvParams:
    log_sigma = np.log(vol).replace([np.inf, -np.inf], np.nan).dropna()
    hist = compute_vol_histogram(vol=log_sigma, bins=bins)
    dv = hist.index[1] - hist.index[0]

    def unpack_pars(pars: np.ndarray) -> LogSvParams:
        theta, kappa1, volvol = pars[0], pars[1], pars[2]
        params = LogSvParams(sigma0=theta, theta=theta, kappa1=kappa1, kappa2=kappa1, beta=0.0, volvol=volvol)
        return params

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        params = unpack_pars(pars=pars)
        analytic = lognormal_sv_ss_log_pdf(log_sigma=hist.index.to_numpy(), params=params) * dv
        sse = np.nansum(np.square(hist.to_numpy() - analytic))
        return sse

    options = {'disp': True, 'ftol': 1e-8}
    p0 = np.abs([0.15, 4.0, 1.0])
    bounds = ((0.05, 1.0), (0.5, 10), (0.1, 5.0))
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

    fit_params = unpack_pars(pars=res.x)
    return fit_params


def fit_distribution_log_sv_fixed_kappa(vol: pd.Series,
                                        kappa1: float,
                                        kappa2: float,
                                        bins: int = 50
                                        ) -> LogSvParams:
    """
    given kappa_1 and kappa_2 fit theta and volvol
    """
    log_sigma = np.log(vol).replace([np.inf, -np.inf], np.nan).dropna()
    hist = compute_vol_histogram(vol=log_sigma, bins=bins)
    dv = hist.index[1] - hist.index[0]

    def unpack_pars(pars: np.ndarray) -> LogSvParams:
        theta, volvol = pars[0], pars[1]
        params = LogSvParams(sigma0=theta, theta=theta, kappa1=kappa1, kappa2=kappa2, beta=0.0, volvol=volvol)
        return params

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        params = unpack_pars(pars=pars)
        analytic = lognormal_sv_ss_log_pdf(log_sigma=hist.index.to_numpy(), params=params) * dv
        sse = np.nansum(np.square(hist.to_numpy() - analytic))
        return sse

    options = {'disp': False, 'ftol': 1e-8}
    p0 = np.array([0.3, 1.0])
    bounds = ((0.05, 1.0), (0.25, 5.0))
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

    fit_params = unpack_pars(pars=res.x)
    return fit_params


def fit_distribution_heston(vol: pd.Series,
                            bins: int = 50
                            ) -> HestonParams:
    log_sigma = np.log(vol).replace([np.inf, -np.inf], np.nan).dropna()
    hist = compute_vol_histogram(vol=log_sigma, bins=bins)
    dv = hist.index[1] - hist.index[0]

    def unpack_pars(pars: np.ndarray) -> HestonParams:
        theta, kappa, volvol = pars[0], pars[1], pars[2]
        params = HestonParams(v0=theta, theta=theta, kappa=kappa, rho=0.0, volvol=volvol)
        return params

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        params = unpack_pars(pars=pars)
        analytic = heston_ss_log_vol_pdf(log_sigma=hist.index.to_numpy(), params=params) * dv
        sse = np.nansum(np.square(hist.to_numpy() - analytic))
        return sse

    options = {'disp': True, 'ftol': 1e-8}
    p0 = np.abs([0.04, 4.0, 1.0])
    bounds = ((0.001, 0.5), (0.5, 10), (0.1, 5.0))
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

    fit_params = unpack_pars(pars=res.x)
    return fit_params


def plot_estimated_svs(vol: pd.Series,
                       logsv_params: LogSvParams = LogSvParams(),
                       heston_params: Optional[HestonParams] = HestonParams(),
                       bins: int = 100,
                       ax: plt.Subplot = None,
                       **kwargs
                       ) -> None:
    log_sigma = np.log(vol).replace([np.inf, -np.inf], np.nan).dropna()
    hist = compute_vol_histogram(vol=log_sigma, bins=bins)
    hist_range = hist.index.to_numpy()

    dv = hist_range[1] - hist_range[0]
    analytic1 = pd.Series(lognormal_sv_ss_log_pdf(log_sigma=hist_range, params=logsv_params)*dv, index=hist_range, name='LogNormal')
    norm_mean, norm_std = np.mean(log_sigma), np.std(log_sigma)
    analytic3 = pd.Series(norm.pdf(x=hist_range, loc=norm_mean, scale=norm_std)*dv, index=hist_range, name='Normal PDF')

    if heston_params is not None:
        analytic2 = pd.Series(heston_ss_log_vol_pdf(log_sigma=hist_range, params=heston_params)*dv, index=hist_range, name='Heston')
        df = pd.concat([analytic1, analytic2, analytic3], axis=1)
    else:
        df = pd.concat([analytic1, analytic3], axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)

    qis.plot_line(hist, colors=['red'],
                  markers=["*"],
                  linewidth=0.0,
                  ax=ax,
                  **kwargs)

    colors = qis.get_n_colors(n=len(df.columns))
    qis.plot_line(df=df,
                  colors=colors,
                  y_limits=(1e-10, None),
                  xlabel='Log volatility',
                  xvar_format='{:.1f}',
                  ax=ax,
                  **kwargs)

    qis.set_legend(ax=ax,
                   markers=["*"]+["_"]*len(df.columns),
                   labels=['Empirical']+list(df.columns),
                   handlelength=0,
                   colors=['red']+colors,
                   legend_loc='lower center',
                   framealpha=0.90,
                   **kwargs)

    ax.set_yscale('log')


def produce_joint_report(vol: pd.Series, bins: int = 100):
    logsv_params = fit_distribution_log_sv(vol=vol, bins=bins)
    print(logsv_params)
    heston_params = fit_distribution_heston(vol=vol, bins=bins)
    print(heston_params)
    plot_estimated_svs(vol=vol, logsv_params=logsv_params, heston_params=heston_params, bins=bins)


class UnitTests(Enum):
    FETCH_VOL = 1
    PLOT_ESTIMATED_VOL = 2
    PLOT_ESTIMATED_LOG_VOL = 3
    FIT_LOGSV = 4
    FIT_HESTON = 5
    JOINT = 6


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.FETCH_VOL:
        vol, returns = fetch_ohlc_vol(ticker='SPY')
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
            qis.plot_time_series(df=vol, ax=ax)
        hist = compute_vol_histogram(vol=vol)
        qis.plot_bars(hist)

    elif unit_test == UnitTests.PLOT_ESTIMATED_VOL:
        vol, returns = fetch_ohlc_vol(ticker='SPY')
        plot_estimated_beta_sv(vol=vol)

    elif unit_test == UnitTests.PLOT_ESTIMATED_LOG_VOL:
        vol, returns = fetch_ohlc_vol(ticker='SPY')
        plot_estimated_beta_log_sv(vol=vol)

    elif unit_test == UnitTests.FIT_LOGSV:
        vol, returns = fetch_ohlc_vol(ticker='SPY')
        fit_params = fit_distribution_log_sv(vol=vol)
        print(fit_params)
        plot_estimated_beta_log_sv(vol=vol, params=fit_params, is_log=True)
        plot_estimated_beta_log_sv(vol=vol, params=fit_params, is_log=False)

    elif unit_test == UnitTests.FIT_HESTON:
        vol, returns = fetch_ohlc_vol(ticker='SPY')
        fit_params = fit_distribution_heston(vol=vol)
        print(fit_params)

    elif unit_test == UnitTests.JOINT:
        vol, returns = fetch_ohlc_vol(ticker='MOVE')
        produce_joint_report(vol=vol)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.JOINT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

