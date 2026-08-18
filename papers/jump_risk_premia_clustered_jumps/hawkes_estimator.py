"""
create data object with options time series data
"""
# built in
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.optimize import minimize
from numba import njit
from typing import Tuple, Optional
from enum import Enum
import qis as qis

# analytics
from papers import local_path as lp
from stochvolmodels.pricers.hawkes_jd_pricer import HawkesJDParams

DAYS_PER_YEAR = 365
HOURS_PER_YEAR = 365 * 24
SECONDS_PER_YEAR = 365 * 24 * 60 * 60  # minute, seconds


def index_to_dt(time_index: pd.DatetimeIndex, af: float = DAYS_PER_YEAR) -> np.ndarray:
    dts = (time_index[1:] - time_index[:-1]).total_seconds() / SECONDS_PER_YEAR
    return np.array(dts)


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


def estimate_thresholds(returns: np.ndarray,
                        n: int = 16,
                        low_quantile: float = 0.01,
                        up_quantile: float = 0.16
                        ) -> Tuple[float, float]:
    """
    run a grid search on quantiles of distribution to find lower and upper quantiles
    so that the skeweness and kurtosis of mid returns is smallest
    """
    qqs = np.linspace(low_quantile, up_quantile, n)
    shift_ms = [np.nanquantile(returns, q) for q in qqs]
    shift_ps = [np.nanquantile(returns, 1-q) for q in qqs]

    func = np.zeros((n, n))
    for n1, shift_m in enumerate(shift_ms):
        for n2, shift_p in enumerate(shift_ps):
            a = np.where(np.logical_and(np.greater(returns, shift_m), np.less(returns, shift_p)), returns, np.nan)
            a = a[np.isnan(a) == False]
            func[n1, n2] = np.abs(skew(a)) + np.abs(kurtosis(a))
            # func[n1, n2] = np.abs(skew(a))
            # func[n1, n2] = np.abs(kurtosis(a))
    ind = np.unravel_index(np.argmin(func, axis=None), func.shape)
    shift_m, shift_p = shift_ms[ind[0]], shift_ps[ind[1]]
    print(f"quantile for mid returns, ind={ind}, qs=({qqs[ind[0]]:0.2f}, {qqs[ind[1]]:0.2f}),"
          f" shift_m={shift_m:0.4f}, shift_p={shift_p:0.4f}")
    return shift_p, shift_m


def infer_jump_times(price: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=False)
    returns_np = returns.to_numpy()
    # estimate thresholds
    shift_p, shift_m = estimate_thresholds(returns=returns_np)
    jumps_cond = np.logical_or(np.greater(returns, shift_p), np.less(returns, shift_m))
    jumps_path = np.where(jumps_cond, returns, np.nan)
    return jumps_path, returns_np


def estimate_hawkes_jd_joint(price: pd.Series,
                             low_quantile: float = 0.01,
                             up_quantile: float = 0.16,
                             af: float = DAYS_PER_YEAR,
                             is_print: bool = True
                             ) -> HawkesJDParams:
    """
    estimation of hawkes process using independent intensity jumps
    """
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=True)
    returns_np = returns.to_numpy()

    # estimate thresholds
    shift_p, shift_m = estimate_thresholds(returns=returns_np, low_quantile=low_quantile, up_quantile=up_quantile)
    mean_p = np.nanmean(returns_np[np.greater(returns_np, shift_p)]) - shift_p
    mean_m = np.nanmean(returns_np[np.less(returns_np, shift_m)]) - shift_m
    theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m = estimate_lambda_params_joint(returns=returns,
                                                                                                          shift_p=shift_p,
                                                                                                          shift_m=shift_m,
                                                                                                          mean_p=mean_p,
                                                                                                          mean_m=mean_m,
                                                                                                          af=af,
                                                                                                          is_print=is_print)

    mid_returns = clip_returns(returns_np=returns_np, shift_p=shift_p, shift_m=shift_m)

    model_params = HawkesJDParams(mu=np.nanmean(mid_returns)*af,
                                  sigma=np.nanstd(mid_returns)*np.sqrt(af),
                                  shift_p=shift_p,
                                  mean_p=mean_p,
                                  shift_m=shift_m,
                                  mean_m=mean_m,
                                  lambda_p=theta_p,
                                  theta_p=theta_p,
                                  kappa_p=kappa_p,
                                  beta1_p=beta1_p,
                                  beta2_p=beta2_p,
                                  lambda_m=theta_m,
                                  theta_m=theta_m,
                                  kappa_m=kappa_m,
                                  beta1_m=beta1_m,
                                  beta2_m=beta2_m)
    return model_params


def estimate_hawkes_jd_independent(price: pd.Series,
                                   af: float = HOURS_PER_YEAR
                                   ) -> HawkesJDParams:
    """
    estimation of hawkes process using independent intensity jumps
    """
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=True)
    returns_np = returns.to_numpy()

    # estimate thresholds
    shift_p, shift_m = estimate_thresholds(returns=returns_np)
    mean_p = np.nanmean(returns_np[np.greater(returns_np, shift_p)]) - shift_p
    mean_m = np.nanmean(returns_np[np.less(returns_np, shift_m)]) - shift_m

    theta_p, kappa_p, beta_p = estimate_lambda_params_independent(returns=returns, shift=shift_p, af=af)
    theta_m, kappa_m, beta_m = estimate_lambda_params_independent(returns=returns, shift=shift_m, af=af)

    mid_returns = clip_returns(returns_np=returns_np, shift_p=shift_p, shift_m=shift_m)

    model_params = HawkesJDParams(mu=np.nanmean(mid_returns) * af,
                                  sigma=np.nanstd(mid_returns)*np.sqrt(af),
                                  shift_p=shift_p,
                                  mean_p=mean_p,
                                  shift_m=shift_m,
                                  mean_m=mean_m,
                                  theta_p=theta_p,
                                  kappa_p=kappa_p,
                                  beta1_p=beta_p,
                                  beta2_p=0.0,
                                  theta_m=theta_m,
                                  kappa_m=kappa_m,
                                  beta1_m=0.0,
                                  beta2_m=beta_m)
    return model_params


def estimate_lambda_params_joint(returns: pd.Series,
                                 shift_p: float,
                                 shift_m: float,
                                 mean_p: float,
                                 mean_m: float,
                                 af: float = DAYS_PER_YEAR,
                                 is_print: bool = True
                                 ) -> Tuple[float, ...]:
    """
    jump path is classified when returns<shift_m or returns>shift_p
    """
    jumps_cond = np.logical_or(np.greater(returns, shift_p), np.less(returns, shift_m))
    jumps_path = returns[jumps_cond]
    jump_dts = index_to_dt(jumps_path.index, af=af)
    jumps = jumps_path[1:].to_numpy()

    def unpack_pars(pars: np.ndarray) -> Tuple[float, ...]:
        theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m \
            = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7]
        return theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m

    def kappa_p(pars: np.ndarray) -> float:
        theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m = unpack_pars(pars)
        return 0.90*kappa_p - beta1_p*(shift_p+mean_p) - beta2_p*(shift_m+mean_m)

    def kappa_m(pars: np.ndarray) -> float:
        theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m = unpack_pars(pars)
        return 0.90*kappa_m - beta1_m*(shift_p+mean_p) - beta2_m*(shift_m+mean_m)

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m = unpack_pars(pars)
        ml = log_ml_jump_times_joint(dts=jump_dts,
                                     jumps=jumps,
                                     shift_p=shift_p,
                                     shift_m=shift_m,
                                     theta_p=theta_p,
                                     theta_m=theta_m,
                                     kappa_p=kappa_p,
                                     kappa_m=kappa_m,
                                     beta1_p=beta1_p,
                                     beta2_p=beta2_p,
                                     beta1_m=beta1_m,
                                     beta2_m=beta2_m)
        return ml

    jumps_path_p, jumps_path_m = jumps_path[jumps_path > 0.0], jumps_path[jumps_path < 0.0]
    theta_p0 = af * len(jumps_path_p.index) / len(returns.index)
    theta_m0 = af * len(jumps_path_m.index) / len(returns.index)

    p0 = np.array([0.5*theta_p0, 0.5*theta_m0,
                   50.0, 50.0,  # kappas
                   theta_p0, -theta_p0,
                   theta_m0, -theta_m0])

    bounds = ((0.01, theta_p0), (0.01, theta_m0),
              (0.1, 1000.0), (0.1, 1000.0),  # kappas
              (0.01, 10000.0), (-10000.0, -0.01),  # beta1, beta2
              (0.01, 10000.0), (-10000.0, -0.01))  # beta1, beta2

    cons = [{'type': 'ineq', 'fun': kappa_p}, {'type': 'ineq', 'fun': kappa_m}]
    options = {'disp': True, 'ftol': 1e-14, 'maxiter': 200}
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, constraints=cons, options=options)
    pars = res.x
    if is_print:
        print(f"p0={p0}")
        print(f"bounds=\n{bounds}")
        print(f"fitted={pars}")
    theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m = unpack_pars(pars)
    return theta_p, theta_m, kappa_p, kappa_m, beta1_p, beta2_p, beta1_m, beta2_m


def estimate_lambda_params_independent(returns: pd.Series,
                                       shift: float,
                                       af: float = DAYS_PER_YEAR
                                       ) -> Tuple[float, float, float]:
    """
    jump path is classified when returns<shift_m or returns>shift_p
    """
    if shift > 0.0:
        jumps_path = returns[np.greater(returns, shift)]
    else:
        jumps_path = returns[np.less(returns, shift)]

    theta0 = af * len(jumps_path.index) / len(returns.index)
    jump_dts = index_to_dt(jumps_path.index)
    jumps = jumps_path[1:].to_numpy()

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        theta, kappa, beta = pars[0], pars[1], pars[2]
        ml = log_ml_jump_times(dts=jump_dts, jumps=jumps,
                               shift=shift, theta=theta, kappa=kappa, beta=beta)
        return ml

    if shift > 0.0:
        p0 = np.array([0.5*theta0, 100.0, theta0])
        bounds = ((0.01, theta0), (0.1, 1000.0), (0.01, 1000.0))
    else:
        p0 = np.array([0.5*theta0, 100.0, - theta0])
        bounds = ((0.01, theta0), (0.1, 1000.0), (-1000.0, -0.1))

    print(f"p0={p0}")
    options = {'disp': True, 'ftol': 1e-12}
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)
    pars = res.x
    print(f"fitted={pars}")
    theta, kappa, beta = pars[0], pars[1], pars[2]
    return theta, kappa, beta


def forecast_hawkes_jd_vol(price: pd.Series,
                           model_params: HawkesJDParams,
                           mid_vol_span: float = 7,
                           af: float = DAYS_PER_YEAR
                           ) -> Tuple[pd.Series, pd.DataFrame]:
    """
    compute vol forecast
    """
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=True)
    returns_np = returns.to_numpy()

    lambda_p = filter_jump_lambda(returns=returns, shift=model_params.shift_p,
                                  theta=model_params.theta_p, kappa=model_params.kappa_p, beta=model_params.beta1_p,
                                  ).rename('lambda_p')

    lambda_m = filter_jump_lambda(returns=returns, shift=model_params.shift_m,
                                  theta=model_params.theta_m, kappa=model_params.kappa_m, beta=model_params.beta1_m,
                                  ).rename('lambda_m')

    # quadratic var jump
    jump_var = model_params.jumps_var_p*lambda_p.to_numpy() + model_params.jumps_var_m*lambda_m.to_numpy()

    # vols
    mid_returns = pd.Series(clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m),
                            index=returns.index, name='sigma').fillna(0.0)
    sigma = qis.compute_ewm_vol(data=mid_returns, ewm_lambda=1.0 - 2.0 / (mid_vol_span + 1.0), af=af, mean_adj_type=qis.MeanAdjType.NONE)
    vol_hawks = pd.Series(np.sqrt(sigma.to_numpy()*sigma.to_numpy()+jump_var), index=returns.index, name='Hawkes Vol')
    model_data = pd.concat([lambda_p, lambda_m, sigma], axis=1)
    return vol_hawks, model_data


@njit
def log_ml_jump_times_joint(dts: np.ndarray,
                            jumps: np.ndarray,
                            shift_p: float,
                            shift_m: float,
                            theta_p: float = 50.0,
                            theta_m: float = 50.0,
                            kappa_m: float = 10.0,
                            kappa_p: float = 10.0,
                            beta1_p: float = 1.0,
                            beta2_p: float = -1.0,
                            beta1_m: float = 1.0,
                            beta2_m: float = -1.0
                            ) -> float:
    """
    ML of joint Hawkes process
    """
    # 1st. fill-in lambda paths
    exp_dts_p, exp_dts_m = np.exp(-kappa_p * dts), np.exp(-kappa_m * dts)
    lambda_p0, lambda_m0 = theta_p, theta_m  # initial condition
    lambda_p_prejump, lambda_m_prejump = np.zeros_like(dts), np.zeros_like(dts)
    lambda_p_postjump, lambda_m_postjump = np.zeros_like(dts), np.zeros_like(dts)
    for idx, jump in enumerate(jumps):
        lambda_p0 = theta_p + exp_dts_p[idx] * (lambda_p0-theta_p)
        lambda_m0 = theta_m + exp_dts_m[idx] * (lambda_m0-theta_m)
        lambda_p_prejump[idx] = lambda_p0  # pre-jump
        lambda_m_prejump[idx] = lambda_m0  # pre-jump
        if jump > 0.0:
            lambda_p0 += beta1_p * jump
            lambda_m0 += beta1_m * jump
        else:
            lambda_p0 += beta2_p * jump
            lambda_m0 += beta2_m * jump
        lambda_p_postjump[idx] = lambda_p0
        lambda_m_postjump[idx] = lambda_m0

    # 2. now compute ml
    log_lik = 0.0
    for idx, (dt, jump) in enumerate(zip(dts, jumps)):
        if idx > 0:
            i_p = theta_p * dt + (lambda_p_postjump[idx-1] - theta_p) * ((1.0 - np.exp(-kappa_p * dt)) / kappa_p)
            i_m = theta_m * dt + (lambda_m_postjump[idx-1] - theta_m) * ((1.0 - np.exp(-kappa_m * dt)) / kappa_m)
            if jump > 0.0:
                ff = lambda_p_prejump[idx]  # / (lambda_ps[last_jump_idx_p]+lambda_ms[last_jump_idx_m])
            else:
                ff = lambda_m_prejump[idx]  # / (lambda_ps[last_jump_idx_p] + lambda_ms[last_jump_idx_m])
            log_lik += (np.log(ff) - i_p - i_m)
    return -log_lik


@njit
def log_ml_jump_times(dts: np.ndarray,
                      jumps: np.ndarray,
                      shift: float,
                      theta: float = 50.0,
                      kappa: float = 10.0,
                      beta: float = 1.0
                      ) -> float:
    """
    ML of single Hawkes process
    """
    exp_dts = np.exp(-kappa * dts)
    exp_dt1s = (1.0-exp_dts)/kappa
    log_lik = 0.0
    lambda0, lambda00 = theta, theta
    for exp_dt, exp_dt1, dt, jump in zip(exp_dts, exp_dt1s, dts, jumps):
        lambda0 = theta + exp_dt*(lambda00-theta)
        log_lik += np.log(lambda0*np.exp(-theta*dt - (lambda00-theta)*exp_dt1))
        lambda0 += beta*jump
        lambda00 = lambda0
    return -log_lik


def filter_jump_lambda(returns: pd.Series,
                       shift: float,
                       theta: float = 50.0,
                       kappa: float = 10.0,
                       beta: float = 1.0
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
                                    )
    return pd.Series(lambdas, index=returns.index, name='jumps lambda')


@njit
def filter_jump_lambda_np(dts: np.ndarray,
                          jumps: np.ndarray,
                          shift: float,
                          theta: float = 50.0,
                          kappa: float = 10.0,
                          beta: float = 1.0
                          ) -> np.ndarray:
    lambdas = np.zeros(dts.shape[0]+1)
    lambdas[0], lambda0 = theta, theta
    exp_dt = np.exp(-kappa*dts)
    for idx, (dt, jump) in enumerate(zip(dts, jumps)):
        lambda0 = theta + exp_dt[idx]*(lambda0-theta)
        if np.abs(jump) > 0.0:
            lambda0 += beta*jump
        lambdas[idx+1] = lambda0
    return lambdas


def illustrate_hawkes_jd_independent(price: pd.Series, model_params: HawkesJDParams, af: float = HOURS_PER_YEAR):
    """
    illustrate paths of hawkes jd
    """
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=True)
    returns_np = returns.to_numpy()
    mid_returns = clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m)

    lambda_p = filter_jump_lambda(returns=returns, shift=model_params.shift_p,
                                  theta=model_params.theta_p, kappa=model_params.kappa_p, beta=model_params.beta1_p,
                                  ).rename('Positive Jumps')

    lambda_m = filter_jump_lambda(returns=returns, shift=model_params.shift_m,
                                  theta=model_params.theta_m, kappa=model_params.kappa_m, beta=model_params.beta2_m,
                                  ).rename('Negatve Jumps')

    # quadratic var jump
    jump_var = model_params.jumps_var_p*lambda_p.to_numpy() + model_params.jumps_var_m*lambda_m.to_numpy()

    # vols
    ewm_vol = np.sqrt(af) * qis.compute_ewm_vol(data=returns, ewm_lambda=0.94, annualize=False).rename('Ewma-94 Vol')
    mid_returns_vol = np.sqrt(af) * qis.compute_ewm_vol(data=mid_returns, ewm_lambda=0.94, annualize=False)
    expanding_vols = np.sqrt(af) * returns.expanding().std().rename('expanding vol')
    mid_returns_ewm = pd.Series(mid_returns_vol, index=returns.index, name='mid_returns_ewm')
    vol_hawks = pd.Series(np.sqrt(mid_returns_vol*mid_returns_vol+jump_var), index=returns.index, name='Hawkes Vol')
    vols = pd.concat([vol_hawks, ewm_vol,
                      # mid_returns_ewm,
                      expanding_vols], axis=1)

    # returns
    returns_p = returns[np.greater(returns_np, model_params.shift_p)].rename('Positive jump returns')
    returns_m = returns[np.less(returns_np, model_params.shift_m)].rename('Negative jump returns')
    mid = returns[np.isnan(clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m))==False].rename('Normal returns')

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=(16, 12), tight_layout=True)

        joint = pd.concat([returns_p, mid, returns_m], axis=1)
        qis.plot_time_series(df=joint,
                             colors=['darkgreen', 'darkorange', 'red'],
                             var_format='{:,.2%}',
                             legend_stats=qis.LegendStats.AVG_STD_SKEW_KURT,
                             markers=qis.get_n_markers(n=len(joint.columns)),
                             title=f"{price.name} daily returns",
                             markersize=4,
                             linewidth=0,
                             ax=axs[0])
        lambdas = pd.concat([lambda_p, lambda_m], axis=1)
        qis.plot_time_series(df=lambdas,
                             colors=['darkgreen', 'red'],
                             var_format='{:,.2f}',
                             legend_stats=qis.LegendStats.AVG_STD_SKEW_KURT,
                             title=f"Estimated jump intensities (annualized)",
                             ax=axs[1])
        qis.plot_time_series(df=vols,
                             colors=['darkgreen', 'darkorange', 'red'],
                             var_format='{:,.0%}',
                             legend_stats=qis.LegendStats.AVG_STD_SKEW_KURT,
                             title=f"Inferred volatilities (annualized)",
                             ax=axs[2])


def filter_jump_lambda_joint(price: pd.Series,
                             model_params: HawkesJDParams,
                             ) -> Tuple[pd.Series, pd.Series]:
    """
    jump path is classified when returns<shift_m or returns>shift_p
    """
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=True)
    dts = index_to_dt(price.index)
    jumps_cond = np.logical_or(np.greater(returns, model_params.shift_p), np.less(returns, model_params.shift_m))
    jumps_path = returns[jumps_cond]
    jumps = jumps_path.reindex(index=returns.index).fillna(0.0)
    lambda_ps, lambda_ms = filter_jump_lambda_joint_np(dts=dts,
                                                       jumps=jumps.to_numpy(),
                                                       shift_p=model_params.shift_p,
                                                       shift_m=model_params.shift_m,
                                                       theta_p=model_params.theta_p,
                                                       theta_m=model_params.theta_m,
                                                       kappa_p=model_params.kappa_p,
                                                       kappa_m=model_params.kappa_m,
                                                       beta1_p=model_params.beta1_p,
                                                       beta2_p=model_params.beta2_p,
                                                       beta1_m=model_params.beta1_m,
                                                       beta2_m=model_params.beta2_m)
    lambda_p = pd.Series(lambda_ps, index=returns.index, name='lambda_p')
    lambda_m = pd.Series(lambda_ms, index=returns.index, name='lambda_m')
    return lambda_p, lambda_m


@njit
def filter_jump_lambda_joint_np(dts: np.ndarray,
                                jumps: np.ndarray,
                                shift_p: float,
                                shift_m: float,
                                theta_p: float = 50.0,
                                theta_m: float = 50.0,
                                kappa_m: float = 10.0,
                                kappa_p: float = 10.0,
                                beta1_p: float = 1.0,
                                beta2_p: float = -1.0,
                                beta1_m: float = 1.0,
                                beta2_m: float = -1.0
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    infer jump intensities
    """
    exp_dts_p, exp_dts_m = np.exp(-kappa_p * dts), np.exp(-kappa_m * dts)
    lambda_p0, lambda_m0 = theta_p, theta_m  # initial condition
    lambda_ps, lambda_ms = np.zeros_like(jumps), np.zeros_like(jumps)
    for idx, jump in enumerate(jumps):
        lambda_p0 = theta_p + exp_dts_p[idx]*(lambda_p0-theta_p)
        lambda_m0 = theta_m + exp_dts_m[idx]*(lambda_m0-theta_m)
        if np.abs(jump) > 0.0:
            if jump > 0.0:
                lambda_p0 += beta1_p * jump
                lambda_m0 += beta1_m * jump
            else:
                lambda_p0 += beta2_p * jump
                lambda_m0 += beta2_m * jump
        lambda_ps[idx] = lambda_p0  # post-jump
        lambda_ms[idx] = lambda_m0  # post-jump

    return lambda_ps, lambda_ms


def illustrate_hawkes_jd_joint(price: pd.Series,
                               model_params: HawkesJDParams,
                               af: float = HOURS_PER_YEAR
                               ) -> plt.Figure:
    """
    illustrate paths of hawkes jd
    """
    returns = qis.to_returns(prices=price, is_log_returns=True, drop_first=True)
    returns_np = returns.to_numpy()
    mid_returns = clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m)
    lambda_p, lambda_m = filter_jump_lambda_joint(price=price, model_params=model_params)

    # quadratic var jump
    jump_var = model_params.jumps_var_p*lambda_p.to_numpy() + model_params.jumps_var_m*lambda_m.to_numpy()

    # vols
    ewm_vol = np.sqrt(af) * qis.compute_ewm_vol(data=returns, ewm_lambda=0.94, annualize=False).rename('Ewma-94 Vol')
    mid_returns_vol = np.sqrt(af) * qis.compute_ewm_vol(data=mid_returns, ewm_lambda=0.94, annualize=False)
    expanding_vols = np.sqrt(af) * returns.expanding().std().rename('expanding vol')
    mid_returns_ewm = pd.Series(mid_returns_vol, index=returns.index, name='mid_returns_ewm')
    vol_hawks = pd.Series(np.sqrt(mid_returns_vol*mid_returns_vol+jump_var), index=returns.index, name='Hawkes Vol')
    vols = pd.concat([vol_hawks, ewm_vol,
                      # mid_returns_ewm,
                      expanding_vols], axis=1)

    # returns
    returns_p = returns[np.greater(returns_np, model_params.shift_p)].rename('Positive jump returns')
    returns_m = returns[np.less(returns_np, model_params.shift_m)].rename('Negative jump returns')
    mid = returns[np.isnan(clip_returns(returns_np=returns_np, shift_p=model_params.shift_p, shift_m=model_params.shift_m))==False].rename('Normal returns')

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=(16, 12), tight_layout=True)
        joint = pd.concat([returns_p, mid, returns_m], axis=1)
        qis.plot_time_series(df=joint,
                             colors=['darkgreen', 'darkorange', 'red'],
                             var_format='{:,.2%}',
                             legend_stats=qis.LegendStats.AVG_STD_SKEW_KURT,
                             markers=qis.get_n_markers(n=len(joint.columns)),
                             title=f"{price.name} daily returns",
                             markersize=4,
                             linewidth=0,
                             bbox_to_anchor=(0.01, 1.25),
                             ax=axs[0])
        lambdas = pd.concat([lambda_p.rename('Positive jump intensity'),
                             lambda_m.rename('Negative jump intensity')], axis=1)
        qis.plot_time_series(df=lambdas,
                             colors=['darkgreen', 'red'],
                             var_format='{:,.2f}',
                             legend_stats=qis.LegendStats.AVG_STD_SKEW_KURT,
                             title=f"Estimated jump intensities (annualized)",
                             ax=axs[1])
        qis.plot_time_series(df=vols,
                             colors=['darkgreen', 'darkorange', 'red'],
                             var_format='{:,.0%}',
                             legend_stats=qis.LegendStats.AVG_STD_SKEW_KURT,
                             title=f"Inferred volatilities (annualized)",
                             ax=axs[2])

        fig1, axs = plt.subplots(2, 1, figsize=(16, 8), tight_layout=True)
        qis.plot_time_series(df=joint,
                             colors=['darkgreen', 'darkorange', 'red'],
                             var_format='{:,.1%}',
                             legend_stats=qis.LegendStats.NONE,
                             markers=qis.get_n_markers(n=len(joint.columns)),
                             title=f"(A) Daily returns",
                             markersize=4,
                             linewidth=0,
                             framealpha=0.9,
                             ax=axs[0])
        lambdas = pd.concat([lambda_p.rename('Positive jump intensity'),
                             lambda_m.rename('Negative jump intensity')], axis=1)
        qis.plot_time_series(df=lambdas,
                             colors=['darkgreen', 'red'],
                             var_format='{:,.1f}',
                             legend_stats=qis.LegendStats.NONE,
                             framealpha=0.9,
                             title=f"(B) Estimated jump intensities (annualized)",
                             ax=axs[1])

    return fig1


def infer_lambdas_hawkes_jd_joint(price: pd.Series, af: float = 365.0):
    """
    illustrate paths of hawkes jd
    """
    model_params = estimate_hawkes_jd_joint(price=price, af=af)
    lambda_p, lambda_m = filter_jump_lambda_joint(price=price, model_params=model_params)
    return lambda_p, lambda_m


class LocalTests(Enum):
    ESTIMATE_JOINT_MODEL = 1
    ESTIMATE_INDEPENDENT_MODEL = 2
    COMPARE = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'
    start_end_date0 = qis.TimePeriod(None, pd.Timestamp('2021-09-03'))
    time_period = None #da.TimePeriod(None, pd.Timestamp('2022-12-02'))

    freq, af = 'D', 365.0
    from option_chain_analytics.ts_loaders import ts_data_loader_wrapper
    from option_chain_analytics import OptionsDataDFs
    from stochvolmodels.data.fetch_option_chain import load_price_data

    resource_path = str(Path(lp.get_resource_path()).joinpath('tardis'))
    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker, local_path=resource_path))
    price = load_price_data(options_data_dfs=options_data_dfs, time_period=time_period, freq='D')
    print(price)

    if local_test == LocalTests.ESTIMATE_JOINT_MODEL:
        model_params = estimate_hawkes_jd_joint(price=price,  af=af)
        model_params.print()
        fig = illustrate_hawkes_jd_joint(price=price, model_params=model_params, af=af)
        qis.save_fig(
            fig=fig,
            file_name='estimated_intensities',
            local_path=lp.get_output_path(),
        )

    if local_test == LocalTests.ESTIMATE_INDEPENDENT_MODEL:
        model_params = estimate_hawkes_jd_independent(price=price, af=af)
        model_params.print()
        illustrate_hawkes_jd_independent(price=price, model_params=model_params, af=af)

    elif local_test == LocalTests.COMPARE:
        model_params0 = estimate_hawkes_jd_independent(price=price[: '2021'], af=af)
        model_params0.print()

        model_params = estimate_hawkes_jd_independent(price=price, af=af)
        model_params.print()

        price1 = price['2021':]
        vols_insample = forecast_hawkes_jd_vol(price=price1, model_params=model_params).rename('In-Sample')
        vols_outsample = forecast_hawkes_jd_vol(price=price1, model_params=model_params0).rename('Out-Sample')
        vols = pd.concat([vols_insample, vols_outsample], axis=1)
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 1, figsize=(16, 12), tight_layout=True)
            qis.plot_time_series(df=vols,
                                 var_format='{:,.0%}',
                                 legend_stats=qis.LegendStats.AVG_STD,
                                 ax=axs)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ESTIMATE_JOINT_MODEL)
