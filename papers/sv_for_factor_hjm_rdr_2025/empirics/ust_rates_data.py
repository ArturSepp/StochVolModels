"""
load ust rates data
"""
import pandas as pd
import numpy as np
from enum import Enum
from pathlib import Path
import os

import qis

LOCAL_RESOURCE_PATH = f"{Path(__file__).resolve().parent}{os.sep}"
LOCAL_DATA_PATH = f"{LOCAL_RESOURCE_PATH}xml"
UST_FILE_NAME = 'ust_rates'
MOVE_NAME = 'Move volatility index'


COLUMN_MAP = {"BC_1MONTH": '1m',
              "BC_3MONTH": '3m',
              "BC_6MONTH": '6m',
              "BC_1YEAR": '1y',
              "BC_2YEAR": '2y',
              "BC_3YEAR": '3y',
              "BC_5YEAR": '5y',
              "BC_7YEAR": '7y',
              "BC_10YEAR": '10y',
              "BC_20YEAR": '20y',
              "BC_30YEAR": '30y'}


TTM_MAP = {'1m': 1.0/12.0,
           '3m': 3.0/12.0,
           '6m': 6.0/12.0,
           '1y': 1.0,
           '2y': 2.0,
           '3y': 3.0,
           '5y': 5.0,
           '7y': 7.0,
           '10y': 10.0,
           '20y': 20.0,
           '30y': 30.0}


def generate_ust_data(folder: str = f"{LOCAL_RESOURCE_PATH}xml"):
    """Refresh raw Treasury XML and the derived yield-curve CSV."""
    from .ust import available_years, read_rates, save_xml, year_now

    # save UST yield rates to local folder for selected years
    for year in available_years():
        save_xml(year, folder=folder)
    save_xml(year_now(), folder=folder, overwrite=True)
    df = read_rates(start_year=1990, end_year=2024, folder=folder)
    df = df.drop('BC_30YEARDISPLAY', axis=1).rename(COLUMN_MAP, axis=1)
    df = df.replace({0.0: np.nan})
    qis.save_df_to_csv(df=df, file_name=UST_FILE_NAME, local_path=folder)


def load_ust_rates(drop_1m: bool = True, drop_20y: bool = True) -> pd.DataFrame:
    df = qis.load_df_from_csv(file_name=UST_FILE_NAME, local_path=LOCAL_DATA_PATH)
    df = df.ffill()/100.0
    if drop_1m:
        df = df.drop(['1m'], axis=1)
    if drop_20y:
        df = df.drop(['20y'], axis=1)
    df.index = df.index.normalize()
    return df


def load_ust_3m_rate() -> pd.Series:
    df = qis.load_df_from_csv(file_name=UST_FILE_NAME, local_path=LOCAL_DATA_PATH)
    df = df.ffill()
    return df['3m'] / 100.0


def fetch_move(is_update: bool = False) -> pd.Series:
    if is_update:
        from papers.yfinance_utils import download_yfinance_history, get_yfinance_close

        history = download_yfinance_history(ticker='^MOVE', start='2003-12-31')
        move = get_yfinance_close(history).rename(MOVE_NAME).multiply(0.01)
        qis.save_df_to_csv(df=move, file_name='move', local_path=LOCAL_DATA_PATH)
    else:
        move = qis.load_df_from_csv(file_name='move', local_path=LOCAL_DATA_PATH).iloc[:, 0]
        move = move.rename(MOVE_NAME)
    return move


class UnitTests(Enum):
    GENERATE_UST_DATA = 1
    READ_UST_DATA = 2
    MOVE_INDEX = 3


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if unit_test == UnitTests.GENERATE_UST_DATA:
        generate_ust_data()

    elif unit_test == UnitTests.READ_UST_DATA:
        df = load_ust_rates()
        print(df)

    elif unit_test == UnitTests.MOVE_INDEX:
        df = fetch_move(is_update=True)
        print(df)


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.GENERATE_UST_DATA)
