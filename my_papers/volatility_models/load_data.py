"""
fetch vol data either using historical ohlc vol or VIX and the likes
"""
import numpy as np
import pandas as pd
import qis
import yfinance as yf
from typing import Optional, Tuple
from qis import OhlcEstimatorType


def fetch_ohlc_vol(ticker: str = 'SPY',
                   af: float = 260,
                   timeperiod: Optional[qis.TimePeriod] = qis.TimePeriod('31Dec1999', None),
                   ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.ROGERS_SATCHELL
                   ) -> Tuple[pd.Series, pd.Series]:
    if ticker in ['VIX', 'MOVE', 'OVX']:
        ohlc_data = yf.download(tickers=f"^{ticker}", start=None, end=None, ignore_tz=True)
        ohlc_data.index = ohlc_data.index.tz_localize('UTC').tz_convert('UTC')
        vol = ohlc_data['Close'] / 100.0

        if ticker == 'VIX':
            spot_ticker = '^GSPC'  # s&p 500 index
        elif ticker == 'MOVE':
            spot_ticker = '^TNX'   # 10y rate
        elif ticker == 'OVX':
            spot_ticker = 'USO'  # oil fund
        else:
            raise NotImplementedError

        prices = yf.download(tickers=spot_ticker, start=None, end=None, ignore_tz=True)['Adj Close']
        prices.index = prices.index.tz_localize('UTC').tz_convert('UTC')

        if ticker == 'MOVE':
            returns = prices.diff(1)
        else:
            # returns = np.log(prices).diff(1)
            returns = prices.pct_change()

    else:
        data = yf.download(tickers=ticker, start=None, end=None, ignore_tz=True)
        data.index = data.index.tz_localize('UTC').tz_convert('UTC')
        ohlc_data = data[['Open', 'High', 'Low', 'Close']].rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, axis=1)
        var = qis.estimate_ohlc_var(ohlc_data=ohlc_data, ohlc_estimator_type=ohlc_estimator_type)
        vol = np.sqrt(af*var)

        returns = np.log(data['Adj Close']).diff(1)

    vol = vol.replace([0.0, np.inf, -np.inf], np.nan).dropna()  # drop outliers

    if timeperiod is not None:
        vol = timeperiod.locate(vol)
        returns = timeperiod.locate(returns)

    vol = vol.rename(ticker)
    returns = returns.rename(ticker)
    return vol, returns
