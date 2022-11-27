"""
estimation of hawkes model
"""
# built in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize
from numba import njit
from typing import Tuple, Optional
from enum import Enum


# analytics
from stochvolmodels.pricers.hawkes_jd_pricer import HawkesJDParams


DAYS_PER_YEAR = 365
HOURS_PER_YEAR = 365 * 24
SECONDS_PER_YEAR = 365 * 24 * 60 * 60  # minute, seconds


class Frequency(Enum):
    HOURLY = 1
    DAILY = 2


def get_af(frequency: Frequency = Frequency.HOURLY) -> float:
    if frequency == Frequency.HOURLY:
        return HOURS_PER_YEAR
    else:
        return DAYS_PER_YEAR


def load_data(ticker: str = 'BTC',
              frequency: Frequency = Frequency.HOURLY
              ) -> pd.Series:
    file_path = f"../resources/{ticker}.csv"
    df = pd.read_csv(filepath_or_buffer=file_path, index_col=0, parse_dates=True)
    if frequency == Frequency.DAILY:
        df = df.resample('D').last()
    return df['close'].rename(ticker)


def to_log_returns(price: pd.Series) -> pd.Series:
    returns = np.log(price).diff(1)
    return returns.iloc[1:]


def index_to_dt(time_index: pd.DatetimeIndex) -> np.ndarray:
    dts = (time_index[1:] - time_index[:-1]).total_seconds() / SECONDS_PER_YEAR
    return np.array(dts)


def compute_moments(a: np.ndarray):
    avg = HOURS_PER_YEAR*np.nanmean(a)
    std = np.sqrt(HOURS_PER_YEAR)*np.nanstd(a)
    skew_ = skew(a, nan_policy='omit')
    kurt = kurtosis(a, nan_policy='omit')
    print(f"avg={avg:0.4f}, std={std:0.4f}, skew={skew_:0.4f}, kurt={kurt:0.4f}")


@njit
def clip_returns(returns_np: np.ndarray,
                 shift_p: Optional[float],
                 shift_m: Optional[float]
                 ) -> np.ndarray:
    if shift_m is not None and shift_p is not None:
        return np.where(np.logical_and(np.greater(returns_np, shift_m), np.less(returns_np, shift_p)), returns_np, np.nan)
    elif shift_m is not None:
        return np.where(np.greater(returns_np, shift_m), returns_np, np.nan)
    elif shift_p is not None:
        return np.where(np.less(returns_np, shift_p), returns_np, np.nan)
    else:
        return returns_np


def estimate_hawkes_jd(price: pd.Series,
                       is_constant_jump: bool = False,
                       af: float = HOURS_PER_YEAR
                       ) -> HawkesJDParams:

    returns = to_log_returns(price=price)
    returns_np = returns.to_numpy()

    # estimate thresholds
    shift_p, shift_m = estimate_thresholds(returns=returns_np)

    theta_p, kappa_p, beta_p = estimate_lambda_params(returns=returns, shift=shift_p, is_constant_jump=is_constant_jump, af=af)
    theta_m, kappa_m, beta_m = estimate_lambda_params(returns=returns, shift=shift_m, is_constant_jump=is_constant_jump, af=af)

    mid_returns = clip_returns(returns_np=returns_np, shift_p=shift_p, shift_m=shift_m)

    model_params = HawkesJDParams(mu=np.nanmean(mid_returns) * af,
                                  sigma=np.nanstd(mid_returns)*np.sqrt(af),
                                  shift_p=shift_p,
                                  mean_p=np.nanmean(returns_np[np.greater(returns_np, shift_p)]),
                                  shift_m=shift_m,
                                  mean_m=np.nanmean(returns_np[np.less(returns_np, shift_m)]),
                                  theta_p=theta_p,
                                  kappa_p=kappa_p,
                                  beta_p=beta_p,
                                  theta_m=theta_m,
                                  kappa_m=kappa_m,
                                  beta_m=beta_m,
                                  is_constant_jump=is_constant_jump)
    return model_params


def forecast_hawkes_jd_vol(price: pd.Series,
                           model_params: HawkesJDParams,
                           mid_vol_span: float = 7,
                           af: float = DAYS_PER_YEAR
                           ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    compute vol forecast
    """
    returns = to_log_returns(price=price)
    returns_np = returns.to_numpy()

    lambda_p = filter_jump_lambda(returns=returns, shift=model_params.shift_p,
                                  theta=model_params.theta_p, kappa=model_params.kappa_p, beta=model_params.beta_p,
                                  is_constant_jump=model_params.is_constant_jump).rename('lambda_p')

    lambda_m = filter_jump_lambda(returns=returns, shift=model_params.shift_m,
                                  theta=model_params.theta_m, kappa=model_params.kappa_m, beta=model_params.beta_m,
                                  is_constant_jump=model_params.is_constant_jump).rename('lambda_m')

    # quadratic var jump
    jump_var = model_params.jumps_var_p*lambda_p.to_numpy() + model_params.jumps_var_m*lambda_m.to_numpy()

    # vols
    mid_returns = pd.Series(clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m),
                            index=returns.index, name='sigma').fillna(0.0)
    sigma = np.sqrt(af)*mid_returns.ewm(span=mid_vol_span).std()
    vol_hawks = pd.Series(np.sqrt(sigma.to_numpy()*sigma.to_numpy()+jump_var), index=returns.index, name='Hawkes Vol')
    model_data = pd.concat([lambda_p, lambda_m, sigma], axis=1)
    return vol_hawks, model_data


def filter_jump_lambda(returns: pd.Series,
                       shift: float,
                       theta: float = 50.0,
                       kappa: float = 10.0,
                       beta: float = 1.0,
                       is_constant_jump: bool = False
                       ) -> pd.Series:
    """
    jump path is classified when returns<shift_m or returns>shift_p
    """
    if shift > 0.0:
        jumps_path = returns[np.greater(returns, shift)]
    else:
        jumps_path = returns[np.less(returns, shift)]

    dts = index_to_dt(returns.index)
    jumps = jumps_path.reindex(index=returns.index).fillna(0.0)
    lambdas = filter_jump_lambda_np(dts=dts, jumps=jumps.to_numpy(),
                                    shift=shift, theta=theta, kappa=kappa, beta=beta,
                                    is_constant_jump=is_constant_jump)
    return pd.Series(lambdas, index=returns.index, name='jumps lambda')


@njit
def filter_jump_lambda_np(dts: np.ndarray,
                          jumps: np.ndarray,
                          shift: float,
                          theta: float = 50.0,
                          kappa: float = 10.0,
                          beta: float = 1.0,
                          is_constant_jump: bool = True
                          ) -> np.ndarray:
    lambdas = np.zeros(dts.shape[0]+1)
    lambdas[0], lambda0 = theta, theta
    exp_dt = np.exp(-kappa*dts)
    for idx, (dt, jump) in enumerate(zip(dts, jumps)):
        lambda0 = theta + exp_dt[idx]*(lambda0-theta)
        if np.abs(jump) > 0.0:
            if is_constant_jump:
                lambda0 += beta
            else:
                lambda0 += beta*np.abs(jump / shift)
        lambdas[idx+1] = lambda0
    return lambdas


@njit
def log_ml_jump_times(dts: np.ndarray,
                      jumps: np.ndarray,
                      shift: float,
                      theta: float = 50.0,
                      kappa: float = 10.0,
                      beta: float = 1.0,
                      is_constant_jump: bool = True
                      ) -> float:
    exp_dts = np.exp(-kappa * dts)
    exp_dt1s = (1.0-exp_dts)/kappa
    log_lik = 0.0
    lambda0, lambda00 = theta, theta
    for exp_dt, exp_dt1, dt, jump in zip(exp_dts, exp_dt1s, dts, jumps):
        lambda0 = theta + exp_dt*(lambda00-theta)
        log_lik += np.log(lambda0*np.exp(-theta*dt - (lambda00-theta)*exp_dt1))
        if is_constant_jump:
            lambda0 += beta
        else:
            lambda0 += beta*np.abs(jump / shift)
        lambda00 = lambda0
    return -log_lik


def estimate_lambda_params(returns: pd.Series,
                           shift: float,
                           is_constant_jump: bool = True,
                           af: float = DAYS_PER_YEAR
                           ) -> Tuple[float, float, float]:
    """
    jump path is classified when returns<shift_m or returns>shift_p
    """
    if shift > 0.0:
        jumps_path = returns[np.greater(returns, shift)]
    else:
        jumps_path = returns[np.less(returns, shift)]

    theta0 = af*len(jumps_path.index) / len(returns.index)
    jump_dts = index_to_dt(jumps_path.index)
    jumps = jumps_path[1:].to_numpy()

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        theta, kappa, beta = pars[0], pars[1], pars[2]
        ml = log_ml_jump_times(dts=jump_dts, jumps=jumps,
                               shift=shift, theta=theta, kappa=kappa, beta=beta,
                               is_constant_jump=is_constant_jump)
        return ml
    p0 = np.array([theta0, 100.0, 0.1*theta0])
    print(f"p0={p0}")
    bounds = ((0.01, theta0), (0.1, 500.0), (0.01*theta0, theta0))
    options = {'disp': True, 'ftol': 1e-12}
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)
    pars = res.x
    print(f"fitted={pars}")
    theta, kappa, beta = pars[0], pars[1], pars[2]
    return theta, kappa, beta


def estimate_thresholds(returns: np.ndarray) -> Tuple[float, float]:
    n = 16
    qqs = np.linspace(0.01, 0.16, n)
    shift_ms = [np.quantile(returns, q) for q in qqs]
    shift_ps = [np.quantile(returns, 1-q) for q in qqs]

    func = np.zeros((n, n))
    for n1, shift_m in enumerate(shift_ms):
        for n2, shift_p in enumerate(shift_ps):
            a = np.where(np.logical_and(np.greater(returns, shift_m), np.less(returns, shift_p)), returns, np.nan)
            a = a[np.isnan(a) == False]
            func[n1, n2] = np.abs(skew(a)) + np.abs(kurtosis(a))
            # func[n1, n2] = np.abs(skew(a))
    ind = np.unravel_index(np.argmin(func, axis=None), func.shape)
    shift_m, shift_p = shift_ms[ind[0]], shift_ps[ind[1]]
    print(f"ind={ind}, shift_m={shift_m}, shift_p={shift_p}")
    return shift_p, shift_m


def illustrate_hawkes_jd(price: pd.Series,
                         model_params: HawkesJDParams,
                         af: float = HOURS_PER_YEAR,
                         mid_vol_span: int = 7
                         ):
    """
    illustrate paths of hawkes jd
    """
    returns = to_log_returns(price=price)
    returns_np = returns.to_numpy()
    mid_returns = pd.Series(clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m),
                            index=returns.index)

    lambda_p = filter_jump_lambda(returns=returns, shift=model_params.shift_p,
                                  theta=model_params.theta_p, kappa=model_params.kappa_p, beta=model_params.beta_p,
                                  is_constant_jump=model_params.is_constant_jump).rename('Positive Jumps')

    lambda_m = filter_jump_lambda(returns=returns, shift=model_params.shift_m,
                                  theta=model_params.theta_m, kappa=model_params.kappa_m, beta=model_params.beta_m,
                                  is_constant_jump=model_params.is_constant_jump).rename('Negatve Jumps')

    # quadratic var jump
    jump_var = model_params.jumps_var_p*lambda_p.to_numpy() + model_params.jumps_var_m*lambda_m.to_numpy()

    # vols
    ewma_vol = np.sqrt(af) * returns.ewm(span=mid_vol_span).std().rename('Ewma-94 Vol')
    mid_returns_ewma = np.sqrt(af) * mid_returns.ewm(span=mid_vol_span).std()
    expanding_vols = np.sqrt(af) * returns.expanding().std().rename('expanding vol')
    vol_hawks = pd.Series(np.sqrt(np.square(mid_returns_ewma.to_numpy())+jump_var), index=returns.index, name='Hawkes Vol')
    vols = pd.concat([vol_hawks, ewma_vol,
                      # mid_returns_ewma,
                      expanding_vols], axis=1)

    # returns
    returns_p = returns[np.greater(returns_np, model_params.shift_p)].rename('Positive jump returns')
    returns_m = returns[np.less(returns_np, model_params.shift_m)].rename('Negative jump returns')
    mid = returns[np.isnan(clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m))==False].rename('Normal returns')

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=(16, 12), tight_layout=True)
        colors = ['darkgreen', 'darkorange', 'red']
        markers = ["o", "v",  "D"]

        joint = pd.concat([returns_p, mid, returns_m], axis=1)
        sns.lineplot(data=joint, palette=colors, dashes=False, markers=markers, linewidth=0,
                     ax=axs[0])
        axs[0].set(title=f"{price.name} daily returns")

        lambdas = pd.concat([lambda_p, lambda_m], axis=1)
        sns.lineplot(data=lambdas, palette=['darkgreen', 'red'], dashes=False,
                     ax=axs[1])
        axs[1].set(title=f"Estimated jump intensities (annualized)")

        sns.lineplot(data=vols, palette=colors, dashes=False,
                     ax=axs[2])
        axs[2].set(title=f"Inferred volatilities (annualized)",)


class UnitTests(Enum):
    ESTIMATE_MODEL= 1
    COMPARE = 2


def run_unit_test(unit_test: UnitTests):

    ticker = 'BTC'
    frequency = Frequency.DAILY

    price = load_data(ticker=ticker, frequency=frequency)

    if unit_test == UnitTests.ESTIMATE_MODEL:
        model_params = estimate_hawkes_jd(price=price, is_constant_jump=False, af=get_af(frequency))
        model_params.print()
        illustrate_hawkes_jd(price=price, model_params=model_params, af=get_af(frequency))

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ESTIMATE_MODEL

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
