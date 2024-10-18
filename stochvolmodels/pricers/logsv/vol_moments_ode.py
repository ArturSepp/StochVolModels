"""
analytics for vol and QV moments computation
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as la
from scipy import linalg as sla
from enum import Enum
# project
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.utils.funcs import set_seed


VOLVOL = 1.75

DRIFT_PARAMS = {'$(\kappa_{1}=4, \kappa_{2}=0)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=4)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=8)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}


def compute_analytic_vol_moments(params: LogSvParams,
                                 t: float = 1.0,
                                 n_terms: int = 4,
                                 is_qvar: bool = False
                                 ) -> np.ndarray:

    lambda_m = params.get_vol_moments_lambda(n_terms=n_terms)

    y = params.sigma0 - params.theta
    y0 = np.zeros(n_terms)
    for n in range(0, n_terms):
        y0[n] = np.power(y, n+1)

    if np.isclose(np.abs(t), 0.0):
        return y0

    rhs = np.zeros(n_terms)
    rhs[1] = params.vartheta2*params.theta2

    if is_qvar:  # need flat boundary condition
        rhs[-1] = -n_terms*params.kappa2*np.power(y, n_terms+1)
    else:
        rhs[-1] = -n_terms*params.kappa2*np.power(y, n_terms+1)

    i_m = la.inv(lambda_m)
    is_expm = True
    if is_expm:
        e_m = sla.expm(lambda_m*t)
        m_rhs = i_m @ (e_m - np.eye(n_terms))
    else:
        w, v = la.eig(lambda_m)
        v_inv = la.inv(v)
        e_m = np.real(v @ np.diag(np.exp(w*t)) @ v_inv)
        m_rhs = np.real(v @ np.diag(np.reciprocal(w)*(np.exp(w*t)-np.ones(n_terms))) @ v_inv)
        # m_rhs = i_m @ (e_m - np.eye(n_terms))

    if is_qvar:
        sol1 = m_rhs @ y0
        intm2 = i_m @ (m_rhs-t*np.eye(n_terms))
        sol2 = intm2 @ rhs
    else:
        sol1 = e_m @ y0
        sol2 = m_rhs @ rhs

    sol = sol1 + sol2
    return sol


def compute_analytic_qvar(params: LogSvParams,
                          ttm: float = 1.0,
                          n_terms: int = 4
                          ) -> float:
    """
    compute expected value [ (1/T) int^T_0 sigma^2_t dt]
    """
    if np.isclose(ttm, 0.0):
        qvar = np.square(params.sigma0)
    else:
        int_moments = compute_analytic_vol_moments(params=params, t=ttm, n_terms=n_terms, is_qvar=True)
        qvar = (int_moments[1] + 2.0*params.theta*int_moments[0]) / ttm + params.theta2
    return qvar


def compute_vol_moments_t(params: LogSvParams,
                          ttm: np.ndarray,
                          n_terms: int = 4,
                          is_print: bool = False
                          ) -> np.ndarray:
    moments = np.zeros((len(ttm), n_terms))
    for idx, t_ in enumerate(ttm):
        moments_ = compute_analytic_vol_moments(t=t_, params=params, n_terms=n_terms)
        if is_print:
            print(f"t={t_}: {moments_}")
        moments[idx, :] = moments_
    return moments


def compute_expected_vol_t(params: LogSvParams,
                           t: np.ndarray,
                           n_terms: int = 4,
                           ) -> np.ndarray:
    ev = np.zeros(len(t))
    for idx, t_ in enumerate(t):
        moments = compute_analytic_vol_moments(t=t_, params=params, n_terms=n_terms)
        ev[idx] = moments[0] + params.theta
    return ev


def compute_sqrt_qvar_t(params: LogSvParams, t: np.ndarray, n_terms: int = 4) -> np.ndarray:
    ev = np.zeros(len(t))
    for idx, t_ in enumerate(t):
        ev[idx] = np.sqrt(compute_analytic_qvar(ttm=t_, params=params, n_terms=n_terms))
    return ev


def fit_model_vol_backbone_to_varswaps(log_sv_params: LogSvParams,
                                       varswap_strikes: pd.Series,
                                       n_terms: int = 4,
                                       verbose: bool = False
                                       ) -> pd.Series:
    """
    fit model eta so that model reproduces quadratic var
    """
    ttms = varswap_strikes.index.to_numpy()
    market_qvar_dt = ttms * np.square(varswap_strikes.to_numpy())
    # compute model qvars
    model_forwards = np.array([compute_analytic_qvar(params=log_sv_params, ttm=ttm, n_terms=n_terms) for ttm in ttms])
    model_qvar_dt = model_forwards*ttms
    model_eta = np.ones_like(ttms)
    for idx, ttm in enumerate(ttms):
        if idx == 0:
            model_eta[idx] = market_qvar_dt[idx] / model_qvar_dt[idx]
        else:
            model_eta[idx] = (market_qvar_dt[idx]-market_qvar_dt[idx-1]) / (model_qvar_dt[idx]-model_qvar_dt[idx-1])
    # model_eta = np.where(model_eta > 0.0, np.sqrt(model_eta), 1.0)
    model_eta = np.where(model_eta > 0.0, model_eta, 1.0)
    # adhoc adjustemnt for now
    model_eta = np.where(ttms < 0.06, np.sqrt(model_eta), model_eta)

    model_eta = pd.Series(model_eta, index=ttms)
    if verbose:
        varswap_strikes = np.sqrt(varswap_strikes.to_frame('vars_swap strikes'))
        varswap_strikes['market_qvar_dt'] = market_qvar_dt
        varswap_strikes['model_qvar_dt'] = model_qvar_dt
        varswap_strikes['model_eta'] = model_eta
        print(f"vars_swaps\n{varswap_strikes}")
    return model_eta


class UnitTests(Enum):
    VOL_MOMENTS = 1
    EXPECTED_VOL = 2
    EXPECTED_QVAR = 3
    VOL_BACKBONE = 4


def run_unit_test(unit_test: UnitTests):

    from stochvolmodels.pricers.logsv_pricer import LogSVPricer
    logsv_pricer = LogSVPricer()

    n_terms = 4
    nb_path = 200000
    ttm = 1.0
    params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=1.75)
    params.assert_vol_moments_stability(n_terms=n_terms)
    set_seed(8)  # 8
    sigma_t, grid_t = logsv_pricer.simulate_vol_paths(ttm=ttm, params=params, nb_path=nb_path)

    if unit_test == UnitTests.VOL_MOMENTS:

        mcs = []
        for n in np.arange(n_terms):
            if n > 0:
                m_n = np.power(sigma_t-params.theta, n+1)
            else:
                m_n = sigma_t - params.theta
            mc_mean, mc_std = np.mean(m_n, axis=1), np.std(sigma_t, axis=1) / np.sqrt(nb_path)
            mc = pd.Series(mc_mean, index=grid_t, name=f"MC m{n+1}")
            mc_m = pd.Series(mc_mean-1.96*mc_std, index=grid_t, name='MC-cd')
            mc_p = pd.Series(mc_mean+1.96*mc_std, index=grid_t, name='MC+cd')
            mcs.append(mc)
        analytic_vol_moments = compute_vol_moments_t(params=params, ttm=grid_t, n_terms=n_terms)
        analytic_vol_moments = pd.DataFrame(analytic_vol_moments, index=grid_t, columns=[f"m{n+1}" for n in range(n_terms)])
        mcs = pd.concat(mcs, axis=1)

        df = pd.concat([analytic_vol_moments, mcs], axis=1)
        print(df)
        df.plot()

    elif unit_test == UnitTests.EXPECTED_VOL:

        mc_mean, mc_std = np.mean(sigma_t, axis=1), np.std(sigma_t, axis=1) / np.sqrt(nb_path)
        mc = pd.Series(mc_mean, index=grid_t, name='MC')
        mc_m = pd.Series(mc_mean-1.96*mc_std, index=grid_t, name='MC-cd')
        mc_p = pd.Series(mc_mean+1.96*mc_std, index=grid_t, name='MC+cd')

        analytic_vol_moments = compute_expected_vol_t(params=params, t=grid_t, n_terms=n_terms)
        analytic_vol_moments = pd.Series(analytic_vol_moments, index=grid_t, name='Analytic')

        df = pd.concat([analytic_vol_moments, mc, mc_m, mc_p], axis=1)
        print(df)
        df.plot()

    elif unit_test == UnitTests.EXPECTED_QVAR:

        q_var = pd.DataFrame(np.square(sigma_t)).expanding(axis=0).mean().to_numpy()
        mc_mean = np.sqrt(np.mean(q_var, axis=1))
        mc_std = np.std(q_var, axis=1) / np.sqrt(nb_path)
        mc = pd.Series(mc_mean, index=grid_t, name='MC')
        mc_m = pd.Series(mc_mean-1.96*mc_std, index=grid_t, name='MC-cd')
        mc_p = pd.Series(mc_mean+1.96*mc_std, index=grid_t, name='MC+cd')

        analytic_vol_moments = compute_sqrt_qvar_t(params=params, t=grid_t, n_terms=n_terms)
        analytic_vol_moments = pd.Series(analytic_vol_moments, index=grid_t, name='Analytic')

        df = pd.concat([analytic_vol_moments, mc, mc_m, mc_p], axis=1)
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
            sns.lineplot(data=analytic_vol_moments, dashes=False, ax=ax)
            ax.errorbar(x=df.index[::5], y=mc_mean[::5], yerr=mc_std[::5], fmt='o', color='green', capsize=8)

    elif unit_test == UnitTests.VOL_BACKBONE:
        fit_model_vol_backbone_to_varswaps(log_sv_params=params,
                                           varswap_strikes=pd.Series([1.0, 1.0], index=[1.0 / 12., 2 / 12.0]),
                                           verbose=True)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.VOL_BACKBONE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
