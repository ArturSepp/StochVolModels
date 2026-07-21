"""
Moments of the volatility process and the expected quadratic variance.

Closed-form truncation solution of Proposition 3.3 in Sepp and Rakhmonov (2024).
The mean-adjusted volatility Y_t = sigma_t - theta of Eq. (3.32) has moments
m_bar^(n)(tau) = E[Y_tau^n] obeying the recursion of Eq. (3.50), which closes
into the finite linear system of Eq. (3.48) once truncated at order k*. That
system is solved by matrix exponentiation in Eq. (3.49), and integrated over
[0, tau] in Eq. (3.54) to give the expected quadratic variance of Eq. (3.53).

Reference
---------
A. Sepp and P. Rakhmonov (2024), Log-normal Stochastic Volatility Model with
Quadratic Drift, International Journal of Theoretical and Applied Finance 26(8),
2450003. Equation numbers throughout this module refer to that article.
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

DRIFT_PARAMS = {'$(kappa_{1}=4, kappa_{2}=0)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                '$(kappa_{1}=4, kappa_{2}=4)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
                '$(kappa_{1}=4, kappa_{2}=8)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}


def compute_analytic_vol_moments(params: LogSvParams,
                                 t: float = 1.0,
                                 n_terms: int = 4,
                                 is_qvar: bool = False
                                 ) -> np.ndarray:
    """
    solve the truncated moment system for the mean-adjusted volatility.

    Integrates ``d_tau M = Lambda M + C`` of Eq. (3.48) in closed form. With
    ``is_qvar=False`` this returns the moment vector M^(1,k*)(tau) of Eq. (3.49),

        M(tau) = expm(Lambda tau) M(0) + Lambda^-1 (expm(Lambda tau) - I) C.

    With ``is_qvar=True`` it returns the time-integrated moments M_hat^(1,k*)(tau)
    of Eq. (3.54), which :func:`compute_analytic_qvar` consumes.

    Parameters
    ----------
    params : LogSvParams
        Model parameters. Only theta, kappa1, kappa2, beta, volvol and sigma0 enter.
    t : float, default 1.0
        Horizon tau in years. At t = 0 the initial vector M(0) is returned unchanged.
    n_terms : int, default 4
        Truncation order k* of Eq. (3.51).
    is_qvar : bool, default False
        Return time-integrated moments of Eq. (3.54) rather than the moments
        themselves.

    Returns
    -------
    np.ndarray, shape (n_terms,)
        Moments m_bar^(1) ... m_bar^(k*), or their integrals over [0, tau].

    Notes
    -----
    The initial vector is M(0) = (Y_0, Y_0^2, ..., Y_0^k*) with Y_0 = sigma0 - theta,
    and the free term is C = (0, c(2) theta^2, 0, ..., -k* kappa2 Y_0^(k*+1)), both
    as given in Eq. (3.48). Accuracy degrades with k*: Fig. 2 shows k* = 4 matching
    Monte Carlo on the first two moments and k* = 8 on the first four.
    """

    lambda_m = params.get_vol_moments_lambda(n_terms=n_terms)

    y = params.sigma0 - params.theta
    y0 = np.zeros(n_terms)
    for n in range(0, n_terms):
        y0[n] = np.power(y, n+1)

    if np.isclose(np.abs(t), 0.0):
        return y0

    rhs = np.zeros(n_terms)
    rhs[1] = params.vartheta2*params.theta2

    # closure of Eq. (3.51): the (k*+1)th moment is frozen at its initial value
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
    annualized expected quadratic variance E[(1/tau) int_0^tau sigma_t^2 dt].

    Corollary 3.4, Eq. (3.53),

        I_hat_tau = (1/tau) (m_hat^(2)(tau) + 2 theta m_hat^(1)(tau)) + theta^2
                    + O(k*),

    where m_hat^(1) and m_hat^(2) are the first two integrated moments returned by
    :func:`compute_analytic_vol_moments` with ``is_qvar=True``, and O(k*) is the
    truncation error. This is the model fair value of a continuously monitored
    variance swap, usable for calibration.

    Parameters
    ----------
    params : LogSvParams
        Model parameters.
    ttm : float, default 1.0
        Time to maturity tau in years. At tau = 0 the value collapses to sigma0^2.
    n_terms : int, default 4
        Truncation order k*.

    Returns
    -------
    float
        Annualized expected quadratic variance.
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
    """evaluate :func:`compute_analytic_vol_moments` over an array of maturities."""
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
    """
    expected volatility E[sigma_tau] = E[Y_tau] + theta over an array of maturities.
    """
    ev = np.zeros(len(t))
    for idx, t_ in enumerate(t):
        moments = compute_analytic_vol_moments(t=t_, params=params, n_terms=n_terms)
        ev[idx] = moments[0] + params.theta
    return ev


def compute_sqrt_qvar_t(params: LogSvParams, t: np.ndarray, n_terms: int = 4) -> np.ndarray:
    """square root of the expected quadratic variance of Eq. (3.53), the model var-swap rate."""
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


class LocalTests(Enum):
    VOL_MOMENTS = 1
    EXPECTED_VOL = 2
    EXPECTED_QVAR = 3
    VOL_BACKBONE = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from stochvolmodels.pricers.logsv_pricer import LogSVPricer
    logsv_pricer = LogSVPricer()

    n_terms = 4
    nb_path = 200000
    ttm = 1.0
    params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=1.75)
    params.assert_vol_moments_stability(n_terms=n_terms)
    set_seed(8)  # 8
    sigma_t, grid_t = logsv_pricer.simulate_vol_paths(ttm=ttm, params=params, nb_path=nb_path)

    if local_test == LocalTests.VOL_MOMENTS:

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

    elif local_test == LocalTests.EXPECTED_VOL:

        mc_mean, mc_std = np.mean(sigma_t, axis=1), np.std(sigma_t, axis=1) / np.sqrt(nb_path)
        mc = pd.Series(mc_mean, index=grid_t, name='MC')
        mc_m = pd.Series(mc_mean-1.96*mc_std, index=grid_t, name='MC-cd')
        mc_p = pd.Series(mc_mean+1.96*mc_std, index=grid_t, name='MC+cd')

        analytic_vol_moments = compute_expected_vol_t(params=params, t=grid_t, n_terms=n_terms)
        analytic_vol_moments = pd.Series(analytic_vol_moments, index=grid_t, name='Analytic')

        df = pd.concat([analytic_vol_moments, mc, mc_m, mc_p], axis=1)
        print(df)
        df.plot()

    elif local_test == LocalTests.EXPECTED_QVAR:

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

    elif local_test == LocalTests.VOL_BACKBONE:
        fit_model_vol_backbone_to_varswaps(log_sv_params=params,
                                           varswap_strikes=pd.Series([1.0, 1.0], index=[1.0 / 12., 2 / 12.0]),
                                           verbose=True)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.VOL_BACKBONE)
