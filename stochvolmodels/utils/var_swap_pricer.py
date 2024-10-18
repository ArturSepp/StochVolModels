import numpy as np
import pandas as pd


def compute_var_swap_strike(puts: pd.Series, calls: pd.Series, forward: float, ttm: float) -> float:
    joint_slice = pd.concat([puts.rename('puts'), calls.rename('calls')], axis=1).sort_index()
    strikes = joint_slice.index.to_numpy()
    otm = strikes < forward
    # dk = strikes[1:] - strikes[:-1]
    n = strikes.shape[0]
    dk = np.zeros(n)
    for idx in np.arange(n):
        if idx == 0:
            dk[idx] = strikes[1] - strikes[0]
        elif idx == n - 1:
            dk[idx] = strikes[idx] - strikes[idx-1]
        else:
            dk[idx] = 0.5*(strikes[idx+1] - strikes[idx-1])

    option_strip = np.where(otm, joint_slice['puts'].to_numpy(), joint_slice['calls'].to_numpy())
    var_swap_strike = 2.0 * np.nansum(dk*option_strip/np.square(strikes))
    atm_strike = strikes[otm == False][0]
    correction = np.square(forward/atm_strike - 1.0)
    var_swap_strike = (var_swap_strike-correction) / ttm
    var_swap_strike = np.sqrt(var_swap_strike)
    return var_swap_strike
