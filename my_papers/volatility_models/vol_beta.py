import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import qis as qis

from my_papers.volatility_models.load_data import fetch_ohlc_vol


def estimate_vol_beta(vol: pd.Series,
                      returns: pd.Series,
                      span: int = 33
                      ) -> pd.Series:
    # dvol = np.log(vol).diff(1).rename('dvol')
    dvol = vol.diff(1).rename('dvol')
    joint = pd.concat([dvol, returns], axis=1).dropna()
    vol_beta = qis.compute_one_factor_ewm_betas(x=joint[returns.name], y=joint['dvol'].to_frame(),
                                                span=span
                                                ).iloc[:, 0]
    return vol_beta


def plot_vol_beta(vol: pd.Series, returns: pd.Series, span: int = 33):
    vol_beta = estimate_vol_beta(vol=vol, returns=returns, span=span)

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
        qis.plot_time_series(df=vol_beta,
                             ax=ax)


class UnitTests(Enum):
    VOL_BETA = 1
    PLOT_VOL_BETA = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.VOL_BETA:
        vol, returns = fetch_ohlc_vol(ticker='VIX')
        vol_beta = estimate_vol_beta(vol=vol, returns=returns)
        print(vol_beta)

    elif unit_test == UnitTests.PLOT_VOL_BETA:
        vol, returns = fetch_ohlc_vol(ticker='OVX')
        plot_vol_beta(vol=vol, returns=returns)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_VOL_BETA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
