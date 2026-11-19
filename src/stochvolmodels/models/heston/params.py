"""Heston params definitions.

Parameter-bound Heston transforms and paths for the provisional model interface.

The transform and Euler updates reuse the maintained Heston pricing kernels. Paths
describe a zero-drift asset under the money-market-account measure, with time in years,
instantaneous annualized variance, annualized volatility and cumulative integrated
variance recorded separately. The historical Euler variance floor is preserved at 1e-4;
this adapter does not introduce a new discretization or calibrate market quotes.
"""

from __future__ import annotations
from dataclasses import dataclass
from stochvolmodels.models.base import ModelParams


@dataclass
class HestonParams(ModelParams):
    """
    parameters of the Heston model.

    Variance follows dv = kappa (theta - v) dt + volvol sqrt(v) dW, with rho the
    return-variance correlation. The Feller condition 2 kappa theta >= volvol^2
    keeps the variance away from zero; it is exposed as a calibration constraint
    rather than enforced here.
    """

    v0: float = 0.04
    theta: float = 0.04
    kappa: float = 4.0
    rho: float = -0.5
    volvol: float = 0.4
