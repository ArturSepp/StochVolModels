# built in
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

# stochvolmodels
import stochvolmodels as sv
import stochvolmodels.pricers.hawkes_jd_pricer as hjp
from stochvolmodels import OptionChain

sv.set_seed(24)


def implied_vol_simulations():

    # set option slices
    ttm = 1.0 /12.0
    strikes_ttm = np.linspace(0.5, 1.5, 20)
    optiontypes_ttm = np.where(strikes_ttm <= 1.0, 'P', 'C')

    # generate paths
    params = hjp.HawkesJDParams()
    pricer = hjp.HawkesJDPricer()
    x0, lambda_p0, lambda_m0 = pricer.simulate_terminal_values(params=params,
                                                                       ttm=ttm,
                                                                       nb_path=100000)

    underlying_t = np.exp(x0)
    forward = np.mean(underlying_t)
    print(f"forward={forward}")

    def compute_implied_vols(gamma: float = 3.0):

        risk_factor = np.exp(gamma*x0)
        adjuster = 1.0/np.mean(risk_factor)
        print(f"adjuster(gamma={gamma})={adjuster}")

        adjusted_forward = np.mean(risk_factor*np.exp(x0))*adjuster
        print(f"adjusted_forward(gamma={gamma})={adjusted_forward}")

        option_prices = np.zeros_like(strikes_ttm)
        option_prices_gamma = np.zeros_like(strikes_ttm)

        for idx, (strike, type_) in enumerate(zip(strikes_ttm, optiontypes_ttm)):
            if type_ == 'C':
                payoff = np.where(np.greater(underlying_t, strike), underlying_t - strike, 0.0)
            elif type_ == 'P':
                payoff = np.where(np.less(underlying_t, strike), strike - underlying_t, 0.0)
            else:
                payoff = np.zeros_like(underlying_t)
            option_prices[idx] = np.nanmean(payoff)
            option_prices_gamma[idx] = np.nanmean(risk_factor*payoff)*adjuster

        model_ivols = sv.infer_bsm_ivols_from_slice_prices(ttm=ttm,
                                                            forward=forward,
                                                            strikes=strikes_ttm,
                                                            discfactor=1.0,
                                                            optiontypes=optiontypes_ttm,
                                                            model_prices=option_prices)
        model_ivols_gamma = sv.infer_bsm_ivols_from_slice_prices(ttm=ttm,
                                                                  forward=adjusted_forward,
                                                                  strikes=strikes_ttm,
                                                                  discfactor=1.0,
                                                                  optiontypes=optiontypes_ttm,
                                                                  model_prices=option_prices_gamma)

        model_ivols = pd.Series(model_ivols, index=strikes_ttm, name='historic measure')
        model_ivols_gamma = pd.Series(model_ivols_gamma, index=strikes_ttm, name='pricing kernel')
        model_vols = pd.concat([model_ivols, model_ivols_gamma], axis=1)
        return model_vols

    model_vols_p = compute_implied_vols(gamma=1.0)
    model_vols_m = compute_implied_vols(gamma=-1.0)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(11, 6), tight_layout=True)

    sv.model_vols_ts(model_vols=model_vols_p, title=f"gamma={1.0}", ax=axs[0])
    sv.model_vols_ts(model_vols=model_vols_m, title=f"gamma={-1.0}", ax=axs[1])


def compute_forward_curve():
    params = hjp.HawkesJDParams()
    pricer = hjp.HawkesJDPricer()

    ttms = np.linspace(0.01, 0.5, 12)

    def compute_forwards(gamma: float = 3.0) -> pd.Series:
        normalizer_ttms, forward_ttms = hjp.hawkesjd_forwards_under_risk_kernel(model_params=params,
                                                                                risk_premia_gamma=gamma,
                                                                                ttms=ttms,
                                                                                forwards=np.array([1.0]))
        forward_ttms = pd.Series(forward_ttms, index=ttms, name=f"gamma={gamma}")
        return forward_ttms

    forward_ttms_p = compute_forwards(gamma=1.0)
    forward_ttms_m = compute_forwards(gamma=-1.0)
    df = pd.concat([forward_ttms_p, forward_ttms_m], axis=1)
    print(df)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(1, 1, figsize=(11, 6), tight_layout=True)
        sv.model_vols_ts(model_vols=df, title=f"forwards",  xlabel='tau', ax=axs)


def compute_implied_vols():
    ttm = 1.0 / 12.0
    forward = 1.0
    strikes = forward*np.linspace(0.5, 1.5, 20)
    # optiontypes_ttm = np.where(strikes <= forward, 'P', 'C')

    # generate paths
    params = hjp.HawkesJDParams()
    params.risk_premia_gamma = 0.0001
    pricer = hjp.HawkesJDPricer()

    option_chain = OptionChain.get_uniform_chain(ttms=np.array([ttm]),
                                                 ids=np.array(['1m']),
                                                 forwards=np.array([forward]),
                                                 strikes=strikes)
    option_chain.print()

    prices, vols = pricer.compute_chain_prices_with_vols(params=params, option_chain=option_chain)
    print(prices)
    print(vols)
    pricer.plot_model_ivols(params=params, option_chain=option_chain)


def plot_implied_vols():
    ttm = 1.0 / 12.0
    strikes = np.linspace(0.5, 1.5, 20)
    optiontypes_ttm = np.where(strikes <= 1.0, 'P', 'C')

    # generate paths
    params = hjp.HawkesJDParams()
    pricer = hjp.HawkesJDPricer()
    option_chain = OptionChain.get_uniform_chain(ttms=np.array([ttm]),
                                                 ids=np.array(['1m']),
                                                 forwards=np.array([1.0]),
                                                 strikes=strikes)
    # pricer.plot_model_ivols(params=params, option_chain=option_chain)

    risk_premia_gammas = [-2.0, - 1.0, None, 1.0, 2.0]
    model_ivols = {}
    for risk_premia_gamma in risk_premia_gammas:
        params.risk_premia_gamma = risk_premia_gamma
        model = f"gamma={risk_premia_gamma:0.2f}" if risk_premia_gamma is not None else f"gamma={0.0}"
        model_ivol = pricer.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
        model_ivols[model] = pd.Series(model_ivol[0], index=strikes)
    model_ivols = pd.DataFrame.from_dict(model_ivols, orient='columns')
    print(model_ivols)
    sv.model_vols_ts(model_vols=model_ivols)


def plot_btc_implied_vols():
    option_chain = sv.get_btc_test_chain_data()
    option_chain = OptionChain.to_forward_normalised_strikes(obj=option_chain)
    option_chain.print()
    #eslice = OptionChain.get_slices_as_chain(option_chain=option_chain, ids='1m')
    #eslice.print()

    params = hjp.HawkesJDParams()
    params.sigma = 0.60
    params.risk_premia_gamma = 4.45
    params.lambda_p = 15.0
    params.lambda_m = 30.0

    pricer = hjp.HawkesJDPricer()
    pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params,
                                       xvar_format='{:0,.2f}')


def calibrate_risk_premia():
    option_chain = sv.get_btc_test_chain_data()
    option_chain = OptionChain.to_forward_normalised_strikes(obj=option_chain)
    #option_chain = OptionChain.get_slices_as_chain(option_chain=option_chain, ids=['2w', '1m'])
    #option_chain = OptionChain.get_slices_as_chain(option_chain=option_chain, ids=['2w'])
    option_chain = OptionChain.get_slices_as_chain(option_chain=option_chain, ids=['3m'])

    option_chain.print()

    pricer = hjp.HawkesJDPricer()
    params = hjp.HawkesJDParams()
    params.print()
    params.lambda_p = 15.0
    params.lambda_m = 30.0
    params.risk_premia_gamma = 4.0

    fitted_params = pricer.calibrate_risk_premia_gamma_to_chain(option_chain=option_chain,
                                                                params0=params,
                                                                is_vega_weighted=False)
    fitted_params.print()
    pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=fitted_params,
                                       xvar_format='{:0,.2f}')


class LocalTests(Enum):
    IMPLIED_VOL_SIMULATIONS = 1
    FORWARD_CURVE = 2
    COMPUTE_IMPLIED_VOLS = 3
    PLOT_IMPLIED_VOL = 4
    PLOT_BTC_IMPLIED_VOL = 5
    CALIBRATE_RISK_PREMIA = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.IMPLIED_VOL_SIMULATIONS:
        implied_vol_simulations()

    elif local_test == LocalTests.FORWARD_CURVE:
        compute_forward_curve()

    elif local_test == LocalTests.COMPUTE_IMPLIED_VOLS:
        compute_implied_vols()

    elif local_test == LocalTests.PLOT_IMPLIED_VOL:
        plot_implied_vols()

    elif local_test == LocalTests.PLOT_BTC_IMPLIED_VOL:
        plot_btc_implied_vols()

    elif local_test == LocalTests.CALIBRATE_RISK_PREMIA:
        calibrate_risk_premia()
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CALIBRATE_RISK_PREMIA)
