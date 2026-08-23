"""
fetch vol data either using historical ohlc vol or VIX and the likes
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import qis

from stochvolmodels.estimation import OhlcEstimatorType, estimate_ohlc_var

from papers.yfinance_utils import download_yfinance_history, get_yfinance_close
from stochvolmodels import local_path as lp


def _load_crypto_atm_vol_data(ticker: str) -> pd.DataFrame:
    """Load one retained BTC/ETH ATM-volatility research series."""
    file_name = f'{ticker}_atm_vols_skew.csv'
    resource_path = Path(lp.get_resource_path())
    candidates = (
        resource_path / file_name,
        resource_path.parent / 'data' / file_name,
    )
    file_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if file_path is None:
        searched = ', '.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f'cannot find {file_name}; searched: {searched}')

    data = pd.read_csv(file_path, index_col=0)
    data.index = pd.to_datetime(data.index, utc=True)
    required_columns = {ticker, 'atm_vol'}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f'{file_path} is missing columns: {sorted(missing_columns)}')
    return data.sort_index()


def fetch_ohlc_vol(ticker: str = 'SPY',
                   af: float = 260,
                   timeperiod: Optional[qis.TimePeriod] = qis.TimePeriod('31Dec1999', None),
    ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.ROGERS_SATCHELL
                   ) -> Tuple[pd.Series, pd.Series]:
    if ticker in ['VIX', 'MOVE', 'OVX']:  # use implied indices
        ohlc_data = download_yfinance_history(ticker=f'^{ticker}')
        vol = ohlc_data['Close'] / 100.0

        if ticker == 'VIX':
            spot_ticker = '^GSPC'  # s&p 500 index
        elif ticker == 'MOVE':
            spot_ticker = '^TNX'   # 10y rate
        elif ticker == 'OVX':
            spot_ticker = 'USO'  # oil fund
        else:
            raise NotImplementedError

        prices = get_yfinance_close(download_yfinance_history(ticker=spot_ticker))

        if ticker == 'MOVE':
            returns = prices.diff(1)
        else:
            # returns = np.log(prices).diff(1)
            returns = prices.pct_change()

    elif ticker in ['BTC', 'ETH']:  # use implied atm vols from internal data
        df = _load_crypto_atm_vol_data(ticker=ticker)
        prices = df[ticker]
        vol = df['atm_vol']
        returns = prices.pct_change()

    else:  # use historical vol
        data = download_yfinance_history(ticker=ticker)
        ohlc_data = data[['Open', 'High', 'Low', 'Close']].rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, axis=1)
        var = estimate_ohlc_var(
            ohlc_data=ohlc_data,
            ohlc_estimator_type=ohlc_estimator_type,
        )
        vol = np.sqrt(af*var)

        returns = np.log(get_yfinance_close(data=data)).diff(1)

    vol = vol.replace([0.0, np.inf, -np.inf], np.nan).dropna()  # drop outliers

    if timeperiod is not None:
        vol = timeperiod.locate(vol)
        returns = timeperiod.locate(returns)

    vol = vol.rename(ticker)
    returns = returns.rename(ticker)
    return vol, returns
