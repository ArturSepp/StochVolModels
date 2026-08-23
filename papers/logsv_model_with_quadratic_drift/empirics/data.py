import numpy as np
import pandas as pd
from typing import Optional
from typing import Tuple
from enum import Enum
import qis
import qis.file_utils as fu

import stochvolmodels.estimation as ovo
from papers.yfinance_utils import download_yfinance_history


CLOSE_COL = 'close'
RETURN_COL = 'log-return'
REAL_VOL_COL = 'real vol'
ATM_VOL_COL = 'atm vol'
CHANGE_VOL_COL = 'd vol'
CHANGE_LOG_VOL_COL = 'd ln vol'
YAHOO_TICKERS = {
    'BTC': 'BTC-USD',
    'XBT': 'BTC-USD',
    'ETH': 'ETH-USD',
}


def load_ohlc_data(ticker: str = 'XBT',
                   cut_off: Optional[str] = '2019-03-27',
                   period: str = '730d',
                   interval: str = '1h',
                   ) -> pd.DataFrame:
    """Download Yahoo OHLC data with the shape expected by QIS estimators.

    Yahoo only exposes intraday history for a rolling window, so the default
    requests the provider's maximum two-year hourly history.
    """
    yahoo_ticker = YAHOO_TICKERS.get(ticker.upper(), ticker)
    prices = download_yfinance_history(
        ticker=yahoo_ticker,
        period=period,
        interval=interval,
    )
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [column for column in required_columns if column not in prices.columns]
    if missing_columns:
        raise ValueError(
            f'yfinance history for {yahoo_ticker!r} is missing OHLC columns: {missing_columns}'
        )

    ohlc_data = prices.loc[:, required_columns].rename(columns=str.lower).dropna()
    if cut_off is not None:
        cut_off_timestamp = pd.Timestamp(cut_off)
        if cut_off_timestamp.tzinfo is None:
            cut_off_timestamp = cut_off_timestamp.tz_localize('UTC')
        else:
            cut_off_timestamp = cut_off_timestamp.tz_convert('UTC')
        ohlc_data = ohlc_data.loc[cut_off_timestamp:]
    return ohlc_data


def get_ohlc_vol_data(ohlc_data: pd.DataFrame,
                      ohlc_estimator_type: ovo.OhlcEstimatorType = ovo.OhlcEstimatorType.PARKINSON,
                      is_exclude_weekends: bool = True,
                      is_filter_low: bool = False
                      ) -> pd.DataFrame:

    vol = ovo.estimate_hf_ohlc_vol(ohlc_data=ohlc_data,
                                   ohlc_estimator_type=ohlc_estimator_type,
                                   is_exclude_weekends=is_exclude_weekends)
    vol = vol.rename(REAL_VOL_COL)

    prices = ohlc_data[CLOSE_COL].reindex(index=vol.index, method='ffill')
    returns = np.log(prices).diff().rename(RETURN_COL)

    d_vol = vol.diff().rename(CHANGE_VOL_COL)
    d_log_vol = np.log(vol).diff().rename(CHANGE_LOG_VOL_COL)

    joint_data = pd.concat([returns, vol, d_vol, d_log_vol], axis=1).dropna()

    if is_filter_low:
        cond = joint_data[REAL_VOL_COL].to_numpy() > np.quantile(joint_data[REAL_VOL_COL].to_numpy(), 0.05)
        joint_data = joint_data.iloc[cond, :]
    return joint_data


def get_price_imp_real_vols(ticker: str = 'BTC',
                            ohlc_estimator_type: ovo.OhlcEstimatorType = ovo.OhlcEstimatorType.PARKINSON,
                            is_exclude_weekends: bool = True,
                            col: str = '1wk ATM Vol',   # 1wk , 1mth
                            scol: str = '1wk 25D skew',
                            is_drop_last: bool = True,
                            resource_path: Optional[str] = None,
                            ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

    if ticker == 'BTC':
        ohlc_ticker = 'XBT'
    else:
        ohlc_ticker = ticker
    ohlc_data = load_ohlc_data(ticker=ohlc_ticker)

    realized_vols = ovo.estimate_hf_ohlc_vol(ohlc_data=ohlc_data,
                                             ohlc_estimator_type=ohlc_estimator_type,
                                             is_exclude_weekends=is_exclude_weekends)

    if resource_path is None:
        resource_path = qis.get_resource_path()

    vols = fu.load_df_from_csv(file_name=f"skew_{ticker.lower()}_atm_implied_volatility",
                               local_path=resource_path,
                               folder_name='skew',
                               tz='UTC')
    prices = fu.load_df_from_csv(file_name=f"skew_{ticker.lower()}usd_spot",
                                 local_path=resource_path,
                                 folder_name='skew',
                                 tz='UTC')

    skew = fu.load_df_from_csv(file_name=f"skew_{ticker.lower()}_25d_skew",
                               local_path=resource_path,
                               folder_name='skew',
                               tz='UTC')

    if is_drop_last:  # remove incomplete day
        realized_vols = realized_vols[:-1]

    implied_vols = vols[col].reindex(index=realized_vols.index, method='ffill').multiply(0.01)
    skew = -skew[scol].reindex(index=realized_vols.index, method='ffill').multiply(0.01)
    prices = prices.reindex(index=realized_vols.index, method='ffill').iloc[:, 0]

    return prices, implied_vols, realized_vols, skew


class UnitTests(Enum):
    OHLC_VOL = 1
    JOINT_DATA = 2


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if unit_test == UnitTests.OHLC_VOL:
        ohlc = load_ohlc_data()
        vol = get_ohlc_vol_data(ohlc_data=ohlc)
        print(vol)

    elif unit_test == UnitTests.JOINT_DATA:
        price, ivols, rvols, skew = get_price_imp_real_vols()
        print(price)
        print(ivols)
        print(rvols)
        print(skew)


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.JOINT_DATA)
