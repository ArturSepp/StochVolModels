import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from enum import Enum


from stochvolmodels import (compute_bsm_vanilla_grid_deltas,
                            compute_bsm_forward_grid_prices,
                            compute_bsm_vanilla_price,
                            compute_bsm_vanilla_delta)


def compare_net_deltas(ttm: float,
                       forward: float,
                       vol: float,
                       strike_level: float = 1.0,
                       optiontype: str = 'C',
                       ax: plt.Subplot = None,
                       **kwargs
                       ):

    spot_drid = np.linspace(0.7*forward, 1.3*forward, 1000)
    strike = strike_level * forward

    option_prices = compute_bsm_forward_grid_prices(ttm=ttm, forwards=spot_drid, strike=strike, vol=vol, optiontype=optiontype)
    option_deltas = compute_bsm_vanilla_grid_deltas(ttm=ttm, forwards=spot_drid, strike=strike, vol=vol, optiontype=optiontype)
    option_net_delta = option_deltas - option_prices / spot_drid
    option_deltas = pd.Series(option_deltas, index=spot_drid, name='Black Delta')
    option_net_delta = pd.Series(option_net_delta, index=spot_drid, name='Net Delta')
    deltas = pd.concat([option_deltas, option_net_delta], axis=1)

    qis.plot_line(df=deltas,
                  xvar_format='{:,.0f}',
                  ylabel='BTC price',
                  ax=ax,
                  **kwargs)


def compare_pnl(ttm: float,
                forward: float,
                vol: float,
                strike_level: float = 1.0,
                optiontype: str = 'C',
                is_btc_pnl: bool = True,
                ax: plt.Subplot = None,
                **kwargs
                ):

    spot_drid = np.linspace(0.7*forward, 1.3*forward, 10000)
    returns_grid = spot_drid / forward - 1.0
    strike = strike_level*forward

    option_price0 = compute_bsm_vanilla_price(ttm=ttm, forward=forward, strike=strike, vol=vol, optiontype=optiontype)
    option_delta0 = compute_bsm_vanilla_delta(ttm=ttm, forward=forward, strike=strike, vol=vol, optiontype=optiontype)
    option_net_delta0 = option_delta0 - option_price0 / forward

    # price return
    inverse_price_return = (spot_drid-forward) / spot_drid
    dt = 1.0 / 365.0
    option_prices = compute_bsm_forward_grid_prices(ttm=ttm-dt, forwards=spot_drid, strike=strike, vol=vol, optiontype=optiontype)
    option_pnl_btc = (option_price0/forward - option_prices/spot_drid)

    # black p&l
    pnl_btc = option_pnl_btc + option_delta0 * inverse_price_return
    if not is_btc_pnl:
        pnl_btc = pnl_btc * spot_drid
    pnl_btc_positive = spot_drid[pnl_btc >= 0.0]
    lower_be = pnl_btc_positive[0] / forward - 1.0
    upper_be = pnl_btc_positive[-1] / forward - 1.0
    pnl_btc = pd.Series(pnl_btc, index=returns_grid, name=f"Black Delta: breakevens=({lower_be:0.2%}, {upper_be:0.2%})")

    # net p&l
    pnl_btc_net_delta = option_pnl_btc + option_net_delta0 * inverse_price_return
    if not is_btc_pnl:
        pnl_btc_net_delta = pnl_btc_net_delta * spot_drid
    pnl_btc_net_delta_positive = spot_drid[pnl_btc_net_delta >= 0.0]
    lower_be = pnl_btc_net_delta_positive[0] / forward - 1.0
    upper_be = pnl_btc_net_delta_positive[-1] / forward - 1.0
    pnl_btc_net_delta = pd.Series(pnl_btc_net_delta, index=returns_grid, name=f"Net Delta   : breakevens=({lower_be:0.2%}, {upper_be:0.2%})")

    pnls = pd.concat([pnl_btc, pnl_btc_net_delta], axis=1)

    if is_btc_pnl:
        ylabel = 'BTC P&L'
        yvar_format = '{:,.2f}'
    else:
        ylabel = 'USD P&L'
        yvar_format = '{:,.0f}'

    qis.plot_line(df=pnls,
                  xvar_format='{:,.1%}',
                  yvar_format=yvar_format,
                  ylabel=ylabel,
                  xlabel='BTC % change',
                  ax=ax,
                  **kwargs)


class UnitTests(Enum):
    DELTA_COMP = 1
    PNL_COMP = 2


def run_unit_test(unit_test: UnitTests):

    LOCAL_PATH = "C://Users//artur//OneDrive//My Papers//Working Papers//Crypto Options. Zurich. Oct 2022//FinalFigures//"

    kwargs = dict(fontsize=14, framealpha=0.9)

    ttm = 7.0 / 365.0

    if unit_test == UnitTests.DELTA_COMP:

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
            compare_net_deltas(ttm=ttm, forward=50000, vol=0.6, optiontype='C',
                               strike_level=1.0,
                               title='(A) Call Delta for K=100%*S_0',
                               ax=axs[0],
                               **kwargs)
            compare_net_deltas(ttm=ttm, forward=50000, vol=0.6, optiontype='P',
                               strike_level=1.0,
                               title='(B) Put Delta for K=100%*S_0',
                               ax=axs[1],
                               **kwargs)

        is_save = True
        if is_save:
            qis.save_fig(fig, file_name='delta_comp', local_path=LOCAL_PATH)
            qis.save_fig(fig, file_name='delta_comp',  file_type=qis.FileTypes.EPS, dpi=1200, local_path=LOCAL_PATH)

    elif unit_test == UnitTests.PNL_COMP:

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
            compare_pnl(ttm=ttm, forward=50000, vol=0.6, optiontype='C',
                        title='(A) ATM Call K=100%*F_0',
                        y_limits=(-0.3, 0.05),
                        ax=axs[0],
                        **kwargs)
            compare_pnl(ttm=ttm, forward=50000, vol=0.6, optiontype='C',
                        title='(B) ITM Call K=90%*F_0',
                        strike_level=0.9,
                        is_btc_pnl=True,
                        y_limits=(-0.3, 0.05),
                        ax=axs[1],
                        **kwargs)

        is_save = True
        if is_save:
            qis.save_fig(fig, file_name='pnl_comp', local_path=LOCAL_PATH)
            qis.save_fig(fig, file_name='pnl_comp',  file_type=qis.FileTypes.EPS, dpi=1200, local_path=LOCAL_PATH)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.DELTA_COMP

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
