import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import minimize
from typing import Tuple
from enum import Enum
import qis
from qis import TimePeriod

# analytics
from papers.jump_risk_premia_clustered_jumps import hawkes_estimator as he
from stochvolmodels.data.fetch_option_chain import load_price_data

SECONDS_PER_YEAR = 365*24*60*60  # days, hours, minute, seconds


def get_funding_rate(ticker: str = 'BTC',
                     time_period: TimePeriod = None,
                     freq: str = 'D'
                     ) -> Tuple[pd.Series, pd.Series]:
    funding_rate = load_price_data(ticker=ticker, data='funding_rate')
    perp = load_price_data(ticker=ticker, data='perp')
    perp_funding_annual = 365.0*funding_rate.resample('8H').last().resample(freq).sum()#.rolling(7).sum()
    perp_funding_annual = perp_funding_annual.dropna()
    perp = perp.reindex(index=perp_funding_annual.index, method='ffill')
    if time_period is not None:
        perp_funding_annual = time_period.locate(perp_funding_annual)
        perp = time_period.locate(perp)
    return perp_funding_annual, perp


@njit
def log_ml_ou(r: np.ndarray,
              x: np.ndarray,
              dts: np.ndarray,
              theta: float,
              kappa: float,
              vol: float,
              beta: float
              ) -> Tuple[float, np.ndarray]:
    """
    ML of ou
    """
    # 1st. fill-in lambda paths
    exp_dts = np.exp(-kappa*dts)
    s_dts = np.sqrt(dts)
    log_lik = 0.0
    drs = r[1:] - r[:-1]
    predictor_t = np.zeros_like(r)
    for idx, (r0, dr, x_, dt, exp_dt, s_dt) in enumerate(zip(r[:-1], drs, x[1:], dts, exp_dts, s_dts)):
        predictor = kappa*(theta-r0)*dt + beta*x_
        predictor_t[idx+1] = r0 + predictor
        log_lik += 0.5*np.square((dr-predictor)/(s_dt*vol))
    log_lik += np.nansum(np.log(s_dts * vol))
    return log_lik, predictor_t


@njit
def log_ml_ou_hawkes(r: np.ndarray,
                     x: np.ndarray,
                     lambda_p: np.ndarray,
                     lambda_m: np.ndarray,
                     dts: np.ndarray,
                     theta: float,
                     kappa: float,
                     vol: float,
                     beta: float,
                     beta_up: float,
                     beta_down: float,
                     is_change_lambdas: bool = True
                     ) -> Tuple[float, np.ndarray]:
    """
    ML of joint ou and Hawkes process
    """
    exp_dts = np.exp(-kappa*dts)
    s_dts = np.sqrt(dts)
    log_lik = 0.0
    drs = r[1:] - r[:-1]
    if is_change_lambdas:
        lambda_p = lambda_p[1:] - lambda_p[:-1]
        lambda_m = lambda_m[1:] - lambda_m[:-1]
    else:
        lambda_p = lambda_p[:-1]
        lambda_m = lambda_m[:-1]
    predictor_t = np.zeros_like(r)
    for idx, (r0, dr, x_, lambda_p_, lambda_m_, dt, exp_dt, s_dt) in enumerate(zip(r[:-1], drs, x[1:], lambda_p, lambda_m, dts, exp_dts, s_dts)):
        predictor = kappa*(theta-r0)*dt + beta*x_ + beta_up * lambda_p_ + beta_down * lambda_m_
        # predictor = kappa*(theta-r0)*dt + beta*x_ + 0.1 * lambda_p_ - 0.1 * lambda_m_
        predictor_t[idx+1] = r0 + predictor
        log_lik += 0.5*np.square((dr-predictor)/(s_dt*vol))
    log_lik += np.nansum(np.log(s_dts[1:] * vol))
    return log_lik, predictor_t


def fit_rate_ou(ticker: str = 'BTC', time_period: TimePeriod = None):
    funding_rate, perp = get_funding_rate(ticker=ticker, time_period=time_period)
    funding_rate = funding_rate.dropna()
    perp = perp.reindex(index=funding_rate.index, method='ffill')
    x = np.log(perp).diff().to_numpy()
    r = funding_rate.to_numpy() / 365.0
    dts = (funding_rate.index[1:] - funding_rate.index[:-1]).total_seconds().to_numpy() / SECONDS_PER_YEAR

    def unpack_pars(pars: np.ndarray) -> tuple[float, ...]:
        theta, kappa, vol, beta = pars[0], pars[1], pars[2], pars[3]
        return theta, kappa, vol, beta

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        theta, kappa, vol, beta = unpack_pars(pars)
        ml, _ = log_ml_ou(r=r,
                          x=x,
                          dts=dts,
                          theta=theta,
                          kappa=kappa,
                          vol=vol,
                          beta=beta)
        return ml

    p0 = np.array([0.0, 5.0, 0.5, 0.0])
    print(f"p0={p0}")
    bounds = ((-0.01, 0.01), (0.01, 1000.0), (0.0001, 5.0), (-10.0, 10.0) )
    print(bounds)
    options = {'disp': True, 'ftol': 1e-12}
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)
    pars = res.x
    theta, kappa, vol, beta = unpack_pars(pars)
    print(f"theta={theta:,.4f}, kappa={kappa:,.2f}, vol={vol:,.4f}, beta={beta:,.4f}")
    return theta, kappa, vol, beta


def fit_rate_ou_hawkes(ticker: str = 'BTC', time_period: TimePeriod = None):
    funding_rate, perp = get_funding_rate(ticker=ticker, time_period=time_period)

    lambda_p, lambda_m = he.infer_lambdas_hawkes_jd_joint(price=perp)
    lambdas = pd.concat([lambda_p, lambda_m], axis=1)
    lambda_p_np, lambda_m_np = lambda_p.to_numpy() / 365.0, lambda_m.to_numpy() / 365.0
    x = np.log(perp).diff().to_numpy()
    r = funding_rate.to_numpy() / 365.0
    dts = (funding_rate.index[1:]- funding_rate.index[:-1]).total_seconds().to_numpy() / SECONDS_PER_YEAR

    def unpack_pars(pars: np.ndarray) -> tuple[float, ...]:
        theta, kappa, vol, beta, beta_up, beta_down = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5]
        return theta, kappa, vol, beta, beta_up, beta_down

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        theta, kappa, vol, beta, beta_up, beta_down = unpack_pars(pars)
        ml, prediction = log_ml_ou_hawkes(r=r,
                                          x=x,
                                          lambda_p=lambda_p_np,
                                          lambda_m=lambda_m_np,
                                          dts=dts,
                                          theta=theta,
                                          kappa=kappa,
                                          vol=vol,
                                          beta=beta,
                                          beta_up=beta_up,
                                          beta_down=beta_down)
        return ml

    p0 = np.array([0.0, 5.0, 0.05, 0.0, -0.0001, 0.0001])
    print(f"p0={p0}")
    bounds = ((-1.0, 1.0), (0.01, 1000.0), (0.0001, 1.0), (-1.0, 1.0), (-100.0, 100.0), (-100.0, 100.0) )
    print(bounds)
    options = {'disp': True, 'ftol': 1e-10, 'maxiter': 200}
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, constraints=None,
                   tol=1e-10, options=options)
    pars = res.x
    theta, kappa, vol, beta, beta_up, beta_down = unpack_pars(pars)
    ml, prediction = log_ml_ou_hawkes(r=r,
                                      x=x,
                                      lambda_p=lambda_p_np,
                                      lambda_m=lambda_m_np,
                                      dts=dts,
                                      theta=theta,
                                      kappa=kappa,
                                      vol=vol,
                                      beta=beta,
                                      beta_up=beta_up,
                                      beta_down=beta_down)
    print(f"theta={theta:,.2f}, kappa={kappa:,.2f}, vol={vol:,.4f}, beta={beta:,.4f}, beta_up={beta_up:,.4f}, beta_down={beta_down:,.4f}")
    return pars, prediction, lambdas


def plot_rate_ou_hawkes(ticker: str = 'BTC', time_period: TimePeriod = None):

    funding_rate, perp = get_funding_rate(ticker=ticker, time_period=time_period)
    x = np.log(perp).diff().to_numpy()
    r = funding_rate.to_numpy() / 365.0
    dts = (funding_rate.index[1:] - funding_rate.index[:-1]).total_seconds().to_numpy() / SECONDS_PER_YEAR

    theta, kappa, vol, beta = fit_rate_ou(ticker=ticker, time_period=time_period)
    ml, prediction = log_ml_ou(r=r,
                               x=x,
                               dts=dts,
                               theta=theta,
                               kappa=kappa,
                               vol=vol,
                               beta=beta)
    prediction_ou = pd.Series(prediction, index=funding_rate.index, name='Prediction OU')

    pars, prediction, lambdas = fit_rate_ou_hawkes(ticker=ticker, time_period=time_period)
    prediction_hk = pd.Series(prediction, index=funding_rate.index, name='Prediction Hawkes')
    df = pd.concat([funding_rate.rename('Funding rate') / 365.0, prediction_ou, prediction_hk], axis=1)

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
        qis.set_suptitle(fig, f"{ticker}")
        qis.plot_time_series(df=lambdas, var_format='{:,.2f}',
                             legend_stats=qis.LegendStats.AVG_MEDIAN_STD_NONNAN_LAST,
                             title='lambdas',
                             ax=axs[0])
        qis.plot_time_series(df=df, var_format='{:,.2%}',
                             legend_stats=qis.LegendStats.AVG_MEDIAN_STD_NONNAN_LAST,
                             title='funding rates and predictions',
                             ax=axs[1])

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), tight_layout=True)
        qis.set_suptitle(fig, f"{ticker}")
        d_df = df.diff(1)
        qis.plot_scatter(df=d_df,
                         x_column='Funding rate',
                         xlabel='Change in funding rate',
                         ylabel='1-day predicted change',
                         add_universe_model_label=False,
                         ax=axs[0])

        r_p = pd.concat([lambdas.iloc[:, 0], funding_rate], axis=1).diff()
        r_m = pd.concat([lambdas.iloc[:, 1], funding_rate], axis=1).diff()
        qis.plot_scatter(df=r_p,
                         ylabel='1-day Change in funding rate',
                         xlabel='1-day change in lambda_p',
                         ax=axs[1])
        qis.plot_scatter(df=r_m,
                         ylabel='1-day Change in funding rate',
                         xlabel='1-day change in lambda_m',
                         ax=axs[2])


class LocalTests(Enum):
    SPOT_DATA = 1
    FIT_OU = 2
    FIT_OU_HAWKES = 3
    PLOT = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'ETH'
    time_period = TimePeriod(start='30Apr2019', end='19Jan2023')

    if local_test == LocalTests.SPOT_DATA:
        rate, perp = get_funding_rate(time_period=time_period)
        qis.plot_time_series(df=rate, legend_stats=qis.LegendStats.AVG_MEDIAN_STD_NONNAN_LAST, var_format='{:,.2%}')
        print(rate)

    elif local_test == LocalTests.FIT_OU:
        fit_rate_ou(time_period=time_period)

    elif local_test == LocalTests.FIT_OU_HAWKES:
        fit_rate_ou_hawkes(time_period=time_period)

    elif local_test == LocalTests.PLOT:
        plot_rate_ou_hawkes(ticker=ticker, time_period=time_period)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.FIT_OU)
