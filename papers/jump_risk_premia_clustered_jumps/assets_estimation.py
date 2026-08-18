"""
use data for several assets to illustrate the estimation of Hawkes model
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from enum import Enum
import qis
import yfinance as yf

from papers import local_path as lp
from papers.jump_risk_premia_clustered_jumps import hawkes_estimator as he

# need a tuple for yf ticker, name, and af factor
ASSETS_FOR_ESTIMATION = {'BTC-USD': ('BTC', 365),
                         'ETH-USD': ('ETH', 365),
                         'DOGE-USD': ('DOGE', 365),
                         'USDT-USD': ('USDT', 365),
                         'USDC-USD': ('USDC', 365),
                         'BUSD-USD': ('BUSD', 365),
                         'SPY': ('S&P500', 260),
                         'QQQ': ('Nasdaq', 260),
                         'TLT': ('20y UST', 260),
                         'SHY': ('1y UST', 260),
                         'GLD': ('GOLD', 260),
                         'USO': ('WTI', 260),
                         'EURUSD=X': ('EUR', 260),
                         'JPY=X': ('JPY', 260),
                         'GBPUSD=X': ('GBP', 260)}


def run_estimation_report():

    figs = {}
    for ticker, asset in ASSETS_FOR_ESTIMATION.items():
        price = yf.download(tickers=[ticker], start="2018-06-01", end=None, ignore_tz=True, progress=False)['Adj Close']
        print(f"############# asset = {asset[0]} ##############")
        model_params = he.estimate_hawkes_jd_joint(price=price,  af=asset[1], is_print=False)
        model_params.print()
        fig = he.illustrate_hawkes_jd_joint(price=price, model_params=model_params, af=asset[1])
        fig.suptitle(f"{asset[0]}")
        figs[ticker] = fig
    qis.save_figs_to_pdf(
        figs=figs,
        file_name='hawkes_estimation_report',
        local_path=lp.get_output_path(),
    )


if __name__ == '__main__':
    run_estimation_report()
