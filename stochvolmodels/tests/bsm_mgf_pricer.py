"""
test option valuation using moment generating function analytics for Black-Scholes-Merton model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from enum import Enum

from stochvolmodels.pricers.core import mgf_pricer as mgfp
from stochvolmodels.pricers.core.bsm_pricer import infer_bsm_ivols_from_model_chain_prices
from stochvolmodels.pricers.core.config import VariableType


def compute_normal_mgf_grid(ttm: float,
                            vol: float,
                            is_spot_measure: bool = True
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    for normal: mgf = exp( 0.5*PHI * (PHI + 1.0)*sigma^2*T )
    add two columns to make compatible with @ for numba in slice_pricer_with_a_grid
    """
    phi_grid = mgfp.get_phi_grid(is_spot_measure=is_spot_measure)
    if is_spot_measure:  # use 1 for spot measure
        alpha = 1.0
    else:
        alpha = -1.0
    log_mgf_grid = 0.5 * phi_grid * (phi_grid + alpha) * (ttm * vol * vol)
    return log_mgf_grid, phi_grid


def compute_normal_mgf_psi_grid(ttm: float,
                                vol: float,
                                is_spot_measure: bool = True
                                ) -> Tuple[np.ndarray, np.ndarray]:
    psi_grid = mgfp.get_psi_grid(is_spot_measure=is_spot_measure)
    log_mgf_grid = -psi_grid * (ttm * vol * vol)
    return log_mgf_grid, psi_grid


def bsm_slice_pricer(ttm: float,
                     forward: float,
                     vol: float,
                     strikes: np.ndarray,
                     optiontypes: np.ndarray,
                     variable_type: VariableType = VariableType.LOG_RETURN,
                     is_spot_measure: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:

    if variable_type == VariableType.LOG_RETURN:
        log_mgf_grid, phi_grid = compute_normal_mgf_grid(ttm=ttm, vol=vol, is_spot_measure=is_spot_measure)
        bsm_prices = mgfp.slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                     phi_grid=phi_grid,
                                                     ttm=ttm,
                                                     forward=forward,
                                                     strikes=strikes,
                                                     optiontypes=optiontypes,
                                                     is_spot_measure=is_spot_measure)
        bsm_ivols = infer_bsm_ivols_from_model_chain_prices(ttms=np.array([ttm]),
                                                            forwards=np.array([forward]),
                                                            discfactors=np.array([1.0]),
                                                            strikes_ttms=(strikes,),
                                                            optiontypes_ttms=(optiontypes,),
                                                            model_prices_ttms=(bsm_prices,))
    elif variable_type == VariableType.Q_VAR:
        log_mgf_grid, psi_grid = compute_normal_mgf_psi_grid(ttm=ttm, vol=vol, is_spot_measure=is_spot_measure)
        bsm_prices = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                        psi_grid=psi_grid,
                                                        ttm=ttm,
                                                        strikes=strikes,
                                                        optiontypes=optiontypes,
                                                        is_spot_measure=is_spot_measure,
                                                        forward=forward)
        bsm_ivols = np.zeros_like(bsm_prices)
    else:
        raise ValueError(f"not implemented")

    return bsm_prices, bsm_ivols


def compare_spot_and_inverse_options():
    ttm = 1.0
    forward = 1.0
    vol = 1.0
    strikes = np.linspace(0.5, 5.0, 19)
    optiontypes = np.full(strikes.shape, 'C')
    optiontypes_inverse = np.full(strikes.shape, 'IC')

    # spot
    bsm_prices_spot, bsm_ivols = bsm_slice_pricer(ttm=ttm, forward=forward, vol=vol, strikes=strikes,
                                                  optiontypes=optiontypes,
                                                  is_spot_measure=True)
    bsm_prices_inverse, bsm_ivols = bsm_slice_pricer(ttm=ttm, forward=forward, vol=vol, strikes=strikes,
                                                     optiontypes=optiontypes_inverse,
                                                     is_spot_measure=False)

    bsm_prices_spot = pd.Series(bsm_prices_spot, index=strikes, name='spot')
    bsm_prices_inverse = pd.Series(bsm_prices_inverse, index=strikes, name='inverse')

    prices = pd.concat([bsm_prices_spot, bsm_prices_inverse], axis=1)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.0), tight_layout=True)
        sns.lineplot(data=prices, ax=ax)


def compare_spot_and_inverse_qvar_options():
    ttm = 1.0
    forward = 1.0
    vol = 1.0
    strikes = np.linspace(0.5, 5.0, 19)
    optiontypes = np.full(strikes.shape, 'C')
    optiontypes_inverse = np.full(strikes.shape, 'IC')

    # spot
    bsm_prices_spot, bsm_ivols = bsm_slice_pricer(ttm=ttm, forward=forward, vol=vol, strikes=strikes,
                                                  optiontypes=optiontypes,
                                                  variable_type=VariableType.Q_VAR,
                                                  is_spot_measure=True)
    bsm_prices_inverse, bsm_ivols = bsm_slice_pricer(ttm=ttm, forward=forward, vol=vol, strikes=strikes,
                                                     optiontypes=optiontypes_inverse,
                                                     variable_type=VariableType.Q_VAR,
                                                     is_spot_measure=False)

    bsm_prices_spot = pd.Series(bsm_prices_spot, index=strikes, name='spot')
    bsm_prices_inverse = pd.Series(bsm_prices_inverse, index=strikes, name='inverse')

    prices = pd.concat([bsm_prices_spot, bsm_prices_inverse], axis=1)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.0), tight_layout=True)
        sns.lineplot(data=prices, ax=ax)


class UnitTests(Enum):
    BSM_SLICE_PRICER = 1
    SPOT_INVERSE_COMP = 2
    SPOT_INVERSE_QVAR_COMP = 3


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.BSM_SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        vol = 1.0
        strikes = np.linspace(0.5, 5.0, 19)
        optiontypes = np.full(strikes.shape, 'C')
        bsm_prices, bsm_ivols = bsm_slice_pricer(ttm=ttm, forward=forward, vol=vol, strikes=strikes, optiontypes=optiontypes)
        print(bsm_prices)
        print(bsm_ivols)

    elif unit_test == UnitTests.SPOT_INVERSE_COMP:
        compare_spot_and_inverse_options()

    elif unit_test == UnitTests.SPOT_INVERSE_QVAR_COMP:
        compare_spot_and_inverse_qvar_options()

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BSM_SLICE_PRICER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
