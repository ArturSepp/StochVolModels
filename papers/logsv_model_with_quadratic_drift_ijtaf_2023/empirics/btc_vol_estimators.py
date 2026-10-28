
# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple
from scipy.optimize import minimize


# internal
from .data import get_ohlc_vol_data, load_ohlc_data
from stochvolmodels.estimation import OhlcEstimatorType, estimate_hf_ohlc_vol


DT = 1.0/260.0
DT_H = 1/(24.0*260.0)
AN_H = 24*260  # weekends are exluded


def plot_vol_estimators(ohlc_data: pd.DataFrame,
                        is_filter_low: bool = False
                        ) -> None:

    vols = []
    for ohlc_estimator_type in OhlcEstimatorType:
        joint_data = estimate_hf_ohlc_vol(ohlc_data=ohlc_data,
                                          ohlc_estimator_type=ohlc_estimator_type)
        vols.append(joint_data.rename(ohlc_estimator_type.value))
    vols = pd.concat(vols, axis=1)

    print(vols.describe())

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), tight_layout=True)

    sns.lineplot(data=vols, dashes=True, ax=axs[0])

    for col in vols.columns:
        sns.kdeplot(vols[col], fill=False, ax=axs[1])

    log_vols = np.log(vols)
    for col in vols.columns:
        sns.kdeplot(log_vols[col], fill=False, ax=axs[2])

    dlog_vols = log_vols.diff()
    for col in vols.columns:
        sns.kdeplot(dlog_vols[col], fill=False, ax=axs[3])

    print(dlog_vols.describe())


def ml_vol_fitter(params: np.ndarray, args: Tuple[np.ndarray, float]):
    # params: 0: theta, kappa0, kappa1,
    theta, kappa0, kappa1, beta, vol_vol = params

    return_vol_data, dt = args[0], args[1]
    dx, vol, dvol = return_vol_data[:, 0], return_vol_data[:, 1], return_vol_data[:, 2]

    #kappa1 = kappa0 / theta
    vol_1 = vol[:-1]
    vol_dw_0 = dx[1:]
    dvol = dvol[1:]
    pred = (kappa0*theta-(kappa0-kappa1*theta)*vol_1-kappa1*np.square(vol_1))*dt + beta*vol_dw_0
    vvol = vol_1*vol_vol*np.sqrt(dt)
    log_lh = np.sum(-0.5*np.square(dvol-pred) / vvol - np.log(vvol))
    return -log_lh


def ml_fitter_log(params: np.ndarray, args: np.ndarray):
    # params: 0: theta, kappa0, kappa1,
    theta, kappa0, kappa1, beta, vol_vol = params
    #kappa1 = kappa0 / theta

    return_vol_data, dt = args[0], args[1]
    dx, vol, dvol = return_vol_data[:, 0], return_vol_data[:, 1], return_vol_data[:, 2]

    vol_1 = vol[:-1]
    dw_0 = dx[1:] / vol_1 #+ 0.5*vol_1*dt
    dz = dvol[1:]
    var = vol_vol*vol_vol + beta*beta
    pred = (((kappa0*theta) / vol_1-(kappa0-kappa1*theta)-kappa1*vol_1-0.5*var)*dt + beta*dw_0)
    vvol = vol_vol*np.sqrt(dt)
    log_lh = np.sum(-0.5 * np.square(dz - pred) / vvol - np.log(vvol))
    return -log_lh


class UnitTests(Enum):
    INTRA_REAL_VOL = 2
    INTRA_OHLC = 5
    INTRA_OHLC_PLOT = 6
    INTRA_OHLC_ESTIMATION = 7
    BOX_PLOT_BETAS = 8
    BOX_PLOT_REVERSIONS = 9
    PLOT_PDF = 10


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if unit_test == UnitTests.INTRA_REAL_VOL:

        ohlc_data = load_ohlc_data()
        joint_data = get_ohlc_vol_data(ohlc_data=ohlc_data,
                                       ohlc_estimator_type=OhlcEstimatorType.PARKINSON,
                                       is_filter_low=False)

    elif unit_test == UnitTests.INTRA_OHLC:
        ohlc_data = load_ohlc_data()
        joint_data = estimate_hf_ohlc_vol(ohlc_data=ohlc_data, ohlc_estimator_type=OhlcEstimatorType.ROGERS_SATCHELL)
        print(joint_data)

    elif unit_test == UnitTests.INTRA_OHLC_PLOT:
        ohlc_data = load_ohlc_data()
        plot_vol_estimators(ohlc_data=ohlc_data, is_filter_low=True)

    elif unit_test == UnitTests.INTRA_OHLC_ESTIMATION:

        ohlc_data = load_ohlc_data()
        joint_data = get_ohlc_vol_data(ohlc_data,
                                       ohlc_estimator_type=OhlcEstimatorType.PARKINSON,
                                       is_filter_low=False)

        params0 = np.array([0.5, 4.0, 4.0, 0.0, 1.0])
        bounds = ((0.1, 2.0), (0.5, 100.0), (0.5, 100.0), (-1.0, 1.0), (0.1, 5.0))
        dt = DT
        args = [joint_data.to_numpy(), dt]

        mle_model = minimize(fun=ml_vol_fitter, x0=params0, args=args, method='L-BFGS-B', bounds=bounds)
        print('mle_model')
        print(mle_model)

        mle_log_model = minimize(fun=ml_fitter_log, x0=params0, args=args, method='L-BFGS-B', bounds=bounds)
        print('mle_log_model')
        print(mle_log_model)

    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.INTRA_OHLC_ESTIMATION)
