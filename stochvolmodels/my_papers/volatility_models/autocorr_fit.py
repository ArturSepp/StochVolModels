import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis
from typing import Tuple
from scipy.optimize import minimize
from enum import Enum

import stochvolmodels.my_papers.volatility_models.ss_distribution_fit as ssd
from stochvolmodels.my_papers.volatility_models.load_data import fetch_ohlc_vol
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels import LogSvParams
from stochvolmodels.utils.funcs import set_seed


def compute_autocorr_power(alpha: float = 0.1, c: float = 1.0, num_lags: int = 20) -> np.ndarray:
    dts = np.arange(0, num_lags) / 260.0
    pf = 1.0 - c*np.power(dts, 2.0*alpha+1)
    return pf


def fit_autocorr_power(vol: pd.Series, num_lags: int = 60) -> Tuple[float, float]:

    # estimate empirical acf
    empirical = qis.compute_path_autocorr(a=vol.to_numpy(), num_lags=num_lags)

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        alpha, c = pars[0], pars[1]
        model_acfs = compute_autocorr_power(alpha=alpha, c=c, num_lags=num_lags)
        sse = np.nansum(np.square(model_acfs - empirical))
        return sse

    options = {'disp': True, 'ftol': 1e-8}
    p0 = np.array([0.1, 0.99])
    bounds = ((-0.5, 0.5), (0.01, 1.5))
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)
    fit_alpha, fit_c = res.x[0], res.x[1]
    return fit_alpha, fit_c


def simulate_autocorr(params: LogSvParams,
                      brownians: np.ndarray = None,
                      nb_path: int = 1000,
                      nb_steps: int = 260,
                      num_lags: int = 20,
                      ttm: float = 10.0
                      ) -> np.ndarray:
    """
    use mc paths to compute paths of autocorrelation
    """
    logsv_pricer = LogSVPricer()
    if brownians is None:
        brownians = np.sqrt(1.0 / 260) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    sigma_t, grid_t = logsv_pricer.simulate_vol_paths(params=params,
                                                      nb_path=nb_path,
                                                      nb_steps=nb_steps,
                                                      ttm=ttm,
                                                      brownians=brownians)
    acfs = qis.compute_path_autocorr(a=sigma_t, num_lags=num_lags)
    return acfs


def get_brownians(nb_steps: int, nb_path: int) -> np.ndarray:
    try:
        brownians = get_brownians.brownians
    except AttributeError:  # read static data and save for next call
        dt = 1.0 / 260.0
        brownians = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
        get_brownians.brownians = brownians
    return brownians


def fit_autocorr_logsv(vol: pd.Series,
                       nb_path: int = 1000,
                       num_lags: int = 60,
                       ttm: float = 10.0
                       ) -> LogSvParams:
    """
    fit autocorrelation of log sv model using MC simulations
    """
    # fix brownians
    nb_steps = int(260*ttm)
    brownians = get_brownians(nb_steps=nb_steps, nb_path=nb_path)

    # estimate empirical acf
    empirical = qis.compute_path_autocorr(a=vol.to_numpy(), num_lags=num_lags)

    def unpack_pars(pars: np.ndarray) -> LogSvParams:
        kappa1, kappa2 = pars[0], pars[1]
        params = ssd.fit_distribution_log_sv_fixed_kappa(vol=vol, kappa1=kappa1, kappa2=kappa2)
        return params

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        params = unpack_pars(pars=pars)
        model_acfs = simulate_autocorr(params=params,
                                       brownians=brownians,
                                       nb_path=nb_path,
                                       nb_steps=nb_steps,
                                       num_lags=num_lags,
                                       ttm=ttm)
        model_acfs = np.mean(model_acfs, axis=1)
        sse = np.nansum(np.square(model_acfs - empirical))
        return sse

    options = {'disp': True, 'ftol': 1e-8}
    p0 = np.array([2.0, 2.0])
    bounds = ((0.2, 10), (0.2, 10))
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

    fit_params = unpack_pars(pars=res.x)
    return fit_params


def autocorr_fit_report_logsv(vol: pd.Series,
                              params: LogSvParams,
                              nb_path: int = 5000,
                              num_lags: int = 90,
                              ttm: float = 10.0,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:

    # estimate empirical acf
    index = range(0, num_lags)
    empirical = qis.compute_path_autocorr(a=vol.to_numpy(), num_lags=num_lags)
    empirical = pd.Series(empirical, index=index, name='Empirical')

    # simulated
    nb_steps = int(260*ttm)
    brownians = get_brownians(nb_steps=nb_steps, nb_path=nb_path)
    model_acfs = simulate_autocorr(params=params,
                                   brownians=brownians,
                                   nb_path=nb_path,
                                   nb_steps=nb_steps,
                                   num_lags=num_lags,
                                   ttm=ttm)
    model_acf = np.mean(model_acfs, axis=1)
    model_acf = pd.Series(model_acf, index=index, name='Log SV')

    # power
    alpha, c = fit_autocorr_power(vol=vol, num_lags=num_lags)
    pf = compute_autocorr_power(alpha=alpha, c=c, num_lags=num_lags)
    pf_power = pd.Series(pf, index=index, name='Rough ' + r'$\alpha$' + f"={alpha:0.2f}")

    # plot acfc
    acfs_df = pd.concat([empirical, model_acf, pf_power], axis=1)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)

    qis.plot_line(acfs_df,
                  linestyles=['dotted', 'solid', 'dashed'],
                  ax=ax,
                  legend_loc='upper center',
                  xlabel='Lag',
                  **kwargs)


class UnitTests(Enum):
    EMPIRICAL_AUTOCORR = 1
    AUTOCORR_POWER = 2
    FIT_AUTOCORR_LOGSV = 3
    FIT_AUTOCORR_HESTON = 4
    FIT_REPORT = 5


def run_unit_test(unit_test: UnitTests):

    set_seed(3)
    np.random.seed(3)

    vix_log_params = LogSvParams(sigma0=0.19928505844247962, theta=0.19928505844247962, kappa1=1.2878835150774184,
                                 kappa2=1.9267876555824357, beta=0.0, volvol=0.7210463316739526)

    move_log_params = LogSvParams(sigma0=0.9109917133860931, theta=0.9109917133860931,
                                  kappa1=0.1, kappa2=0.41131244621275886, beta=0.0, volvol=0.3564212939473691)

    ovx_log_params = LogSvParams(sigma0=0.3852514800317871, theta=0.3852514800317871, kappa1=2.7774564907918027,
                                 kappa2=2.2351296851221107, beta=0.0, volvol=0.8344408577025486)

    btc_log_params = LogSvParams(sigma0=0.7118361434192538, theta=0.7118361434192538,
                                 kappa1=2.214702576955766, kappa2=2.18028273418397, beta=0.0, volvol=0.921487415907961)

    eth_log_params = LogSvParams(sigma0=0.8657438901704476, theta=0.8657438901704476, kappa1=1.955809653686808,
                                 kappa2=1.978367101612294, beta=0.0, volvol=0.8484117641903834)

    nb_path: int = 10000
    num_lags: int = 120
    ttm: float = 10.0

    if unit_test == UnitTests.EMPIRICAL_AUTOCORR:
        vol, returns = fetch_ohlc_vol(ticker='VIX')
        empirical = qis.compute_path_autocorr(a=vol.to_numpy(), num_lags=num_lags)
        print(empirical)
        qis.plot_line(df=pd.Series(empirical))

        dvols = vol.diff(1)
        vol_returns = qis.compute_path_corr(a1=dvols.to_numpy(), a2=returns.to_numpy(), num_lags=num_lags)[1:]
        print(vol_returns)
        qis.plot_line(df=pd.Series(vol_returns))

    elif unit_test == UnitTests.AUTOCORR_POWER:
        vol, returns = fetch_ohlc_vol(ticker='OVX')
        alpha, c = fit_autocorr_power(vol=vol)
        print(f"alpha={alpha}, c={c}")
        pf = compute_autocorr_power(alpha=alpha, num_lags=60)
        print(pf)

    elif unit_test == UnitTests.FIT_AUTOCORR_LOGSV:
        vol, returns = fetch_ohlc_vol(ticker='ETH')
        fit_params = fit_autocorr_logsv(vol=vol, nb_path=nb_path, num_lags=num_lags, ttm=ttm)
        print(f"fit_params={fit_params}")
        qis.plot_time_series(df=vol)
        autocorr_fit_report_logsv(params=fit_params, vol=vol, nb_path=nb_path, num_lags=num_lags, ttm=ttm)
        ssd.plot_estimated_svs(vol=vol, logsv_params=fit_params, heston_params=None, bins=50)

    elif unit_test == UnitTests.FIT_REPORT:
        vol, returns = fetch_ohlc_vol(ticker='VIX')
        autocorr_fit_report_logsv(params=vix_log_params,
                                  vol=vol,
                                  nb_path=nb_path,
                                  num_lags=120,
                                  ttm=ttm)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FIT_AUTOCORR_LOGSV

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)