"""LogSV grids calculations delegated by ``LogSvParams``.

These model-specific computations preserve the parameter convenience API while
keeping numerical implementations separate from parameter definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from stochvolmodels.utils.config import VariableType

if TYPE_CHECKING:
    from stochvolmodels.models.logsv.params import LogSvParams


def get_x_grid(
    params: LogSvParams, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200
) -> np.ndarray:
    """
    spatial grid for the density of the log-price X_tau.

    Centred on the drift -0.5 sigma_t^2 of Eq. (3.16) with sigma_t the
    maturity-averaged volatility, and spanning ``n_stdevs + 1`` standard
    deviations either side. Used for the Fourier inversion illustrated in
    Fig. 6(a).
    """
    sigma_t = np.sqrt(ttm * 0.5 * (np.square(params.sigma0) + np.square(params.theta)))
    drift = -0.5 * sigma_t * sigma_t
    stdev = (n_stdevs + 1) * sigma_t
    return np.linspace(-stdev + drift, stdev + drift, n)


def get_sigma_grid(
    params: LogSvParams, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200
) -> np.ndarray:
    """
    spatial grid on [0, .] for the density of the volatility sigma_tau.

    Half-width scales with sqrt(vartheta^2 ttm), the diffusive spread of
    Eq. (3.19). Used for Fig. 6(c).
    """
    sigma_t = np.sqrt(0.5 * (np.square(params.sigma0) + np.square(params.theta)))
    vvol = 0.5 * np.sqrt(params.vartheta2 * ttm)
    return np.linspace(0.0, sigma_t + n_stdevs * vvol, n)


def get_qvar_grid(
    params: LogSvParams, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200
) -> np.ndarray:
    """
    spatial grid on [0, .] for the density of the quadratic variance I_tau.

    I_t is defined by dI_t = sigma_t^2 dt in Eq. (3.12). Used for Fig. 6(b).
    """
    sigma_t = np.sqrt(ttm * (np.square(params.sigma0) + np.square(params.theta)))
    vvol = np.sqrt(params.vartheta2) * ttm
    return np.linspace(0.0, sigma_t + n_stdevs * vvol, n)


def get_variable_space_grid(
    params: LogSvParams,
    variable_type: VariableType = VariableType.LOG_RETURN,
    ttm: float = 1.0,
    n_stdevs: float = 3,
    n: int = 200,
) -> np.ndarray:
    """
    dispatch to the spatial grid of the state variable selected by variable_type.

    Raises
    ------
    NotImplementedError
        If variable_type is not one of the three state variables of Eq. (3.34).
    """
    if variable_type == VariableType.LOG_RETURN:
        return params.get_x_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
    if variable_type == VariableType.SIGMA:
        return params.get_sigma_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
    elif variable_type == VariableType.Q_VAR:
        return params.get_qvar_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
    else:
        raise NotImplementedError
