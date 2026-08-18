"""
implementation of chain calibrator for risk-premia gamma given a set of p-params
"""

# packages
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

# analytics
from stochvolmodels.data.option_chain import OptionChain
import stochvolmodels.pricers.hawkes_jd_pricer as hjp
from stochvolmodels.data.fetch_option_chain import load_option_chain


def calibrate_params(option_chain: OptionChain) -> None:
    option_chain = OptionChain.to_forward_normalised_strikes(obj=option_chain)

    option_chain.print()

    pricer = hjp.HawkesJDPricer()
    params = hjp.HawkesJDParams()
    params.print()

    fitted_params = pricer.calibrate_model_params_to_chain(option_chain=option_chain,
                                                                params0=params,
                                                                is_vega_weighted=True)
    fitted_params.print()
    pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=fitted_params,
                                       xvar_format='{:0,.2f}')

def calibrate_risk_premia(option_chain: OptionChain) -> None:
    option_chain = OptionChain.to_forward_normalised_strikes(obj=option_chain)

    option_chain.print()

    pricer = hjp.HawkesJDPricer()
    params = hjp.HawkesJDParams()
    params.print()
    params.lambda_p = 50.0
    params.lambda_m = 5.0
    params.risk_premia_gamma = 0.0

    fitted_params = pricer.calibrate_risk_premia_gamma_to_chain(option_chain=option_chain,
                                                                params0=params,
                                                                is_vega_weighted=False)
    fitted_params.print()
    pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=fitted_params,
                                       xvar_format='{:0,.2f}')



class LocalTests(Enum):
    CALIBRATE_PARAMS = 1
    CALIBRATE_RISK_PREMIA = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'  # BTC, ETH
    value_time = pd.Timestamp('2021-10-21 08:00:00+00:00')  # risk_premia_gamma = 4.800793909603246
    # value_time = pd.Timestamp('2023-02-06 08:00:00+00:00')  # 0.004790984073729326
    # value_time = pd.Timestamp('2022-11-15 08:00:00+00:00')  # -2.1054741773198336

    if local_test == LocalTests.CALIBRATE_PARAMS:
        option_chain = load_option_chain(ticker='BTC',
                                         value_time=value_time,
                                         days_map={'1m': 30},
                                         delta_bounds=(-0.2, 0.2))
        option_chain.print()
        calibrate_params(option_chain=option_chain)

    elif local_test == LocalTests.CALIBRATE_RISK_PREMIA:
        option_chain = load_option_chain(ticker='BTC',
                                         value_time=value_time,
                                         days_map={'1m': 30},
                                         delta_bounds=(-0.2, 0.2))
        option_chain.print()
        calibrate_risk_premia(option_chain=option_chain)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CALIBRATE_PARAMS)
