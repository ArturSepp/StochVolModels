import numpy as np
from numba.typed import List
from typing import Tuple


def get_test_slices(num_ttms: int = 4,
                    cut_down: float = 4.0,
                    cut_up: float = 3.0,
                    sigma0: float = 1.0,
                    is_spot_measure: bool = True,
                    is_add_fwd=False,
                    num_strikes: int = 20) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray], Tuple[np.ndarray]]:

    if num_ttms == 2:
        ttms = np.array([3.0, 12.0]) / 12.0
        forwards = np.array([1.0, 1.0])

    elif num_ttms == 4:
        ttms = np.array([1.0, 3.0, 6.0, 12.0]) / 12.0
        forwards = np.array([1.0, 1.0, 1.0, 1.0])

    elif num_ttms == 1:
        ttms = np.array([0.5])
        forwards = np.array([1])

    else:
        raise ValueError(f"num_ttms={num_ttms} not implemented")

    strikes_log = np.linspace(-cut_down, cut_up, num_strikes)
    if is_add_fwd:
        strikes_log = np.unique(np.sort(np.append(strikes_log, 0.0)))
    strikes_ttms = List()
    optiontypes_ttms = List()
    for row_idx, (ttm, forward) in enumerate(zip(ttms, forwards)):
        strikes = forward*np.exp((sigma0*np.sqrt(ttm))*strikes_log)
        strikes_ttms.append(strikes)
        if is_spot_measure:
            optiontypes_ttms.append(np.where(np.less(strikes, forward), 'P', 'C'))
        else:
            optiontypes_ttms.append(np.where(np.less(strikes, forward), 'IP', 'IC'))

    return ttms, forwards, strikes_ttms, optiontypes_ttms
