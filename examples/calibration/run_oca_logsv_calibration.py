"""Convert an OCA 5 option panel and calibrate the SVM LogSV model.

This deterministic example needs no credentials or market-data cache::

    pip install "stochvolmodels[research]"
    python examples/calibration/run_oca_logsv_calibration.py

Replace ``generate_simulated_options_data`` with any OCA ``OptionsDataDFs``
loader to use the same bridge with a normalized provider dataset. Select the case in the
``LocalTests`` call under the main guard.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd
from option_chain_analytics import generate_simulated_options_data

from stochvolmodels import (
    ConstraintsType,
    LogsvModelCalibrationType,
    LogSvParams,
    LogSVPricer,
    OptionChain,
)
from stochvolmodels.data.fetch_option_chain import load_option_chain


class LocalTests(Enum):
    """Available OCA-to-SVM illustration cases."""

    CONVERT_CHAIN = 1
    CALIBRATE_LOGSV = 2


def create_svm_chain_from_oca() -> OptionChain:
    """Create deterministic OCA data and map two maturities to an SVM chain."""
    options_data = generate_simulated_options_data(rate=0.05)
    option_chain = load_option_chain(
        options_data_dfs=options_data,
        value_time=pd.Timestamp('2024-01-05 08:00:00+00:00'),
        days_map={'1w': 7, '1m': 21},
        delta_bounds=(None, None),
    )
    if option_chain is None:
        raise RuntimeError('OCA did not produce an observation at or before value_time')
    return option_chain


def calibrate_logsv(option_chain: OptionChain) -> LogSvParams:
    """Calibrate the analytic LogSV pricer to an OCA-derived SVM chain."""
    atm_vol = float(option_chain.get_chain_atm_vols()[0])
    pricer = LogSVPricer()
    fitted = pricer.calibrate_model_params_to_chain(
        option_chain=option_chain,
        params0=LogSvParams(
            sigma0=atm_vol,
            theta=atm_vol,
            kappa1=2.0,
            kappa2=2.0,
            beta=-0.5,
            volvol=1.0,
        ),
        model_calibration_type=LogsvModelCalibrationType.PARAMS4,
        constraints_type=ConstraintsType.UNCONSTRAINT,
    )
    model_vols = pricer.compute_model_ivols_for_chain(option_chain, fitted)
    rmse = {
        str(slice_id): float(np.sqrt(np.mean((model - 0.5 * (bid + ask)) ** 2)))
        for slice_id, model, bid, ask in zip(
            option_chain.ids,
            model_vols,
            option_chain.bid_ivs,
            option_chain.ask_ivs,
        )
    }
    print(f'calibration_slice_rmse={rmse}')
    return fitted


def run_local_test(local_test: LocalTests) -> None:
    """Run the selected OCA-to-SVM integration case."""
    option_chain = create_svm_chain_from_oca()
    print(f'maturities={option_chain.ids.tolist()}')
    print(f'discount_factors={option_chain.discfactors.tolist()}')
    if local_test == LocalTests.CONVERT_CHAIN:
        option_chain.print()
    elif local_test == LocalTests.CALIBRATE_LOGSV:
        print(f'calibrated_logsv={calibrate_logsv(option_chain)}')
    else:
        raise NotImplementedError(f'unsupported local test: {local_test}')


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.CALIBRATE_LOGSV)
