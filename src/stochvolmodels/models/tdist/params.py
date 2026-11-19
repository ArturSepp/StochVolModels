"""Parameters for the floored Student-t terminal asset distribution.

Volatility, drift and degrees of freedom describe the arithmetic-return shock
at a fixed maturity. Calibration and the pricer-backed terminal capability
adapter remain in pricers.tdist_pricer.
"""

from dataclasses import dataclass
from stochvolmodels.models.base import ModelParams


@dataclass
class TdistParams(ModelParams):
    """
    parameters of the Student-t model: volatility, drift and degrees of freedom nu.

    The centered arithmetic-return shock is Student-t with nu > 2 and variance
    ``vol**2 * ttm``. The terminal asset is the positive part of spot times one plus
    drift times maturity plus that shock, so the floor creates an atom at zero.
    """

    drift: float
    vol: float
    nu: float
    ttm: float  # Parameters are fixed to this maturity, which is not calibrated.
