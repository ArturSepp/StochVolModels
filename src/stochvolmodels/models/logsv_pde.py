"""Money-market LogSV generator in normalized log-forward and volatility coordinates."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.pde_solvers import PDE2DCoefficients, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams


def _mma_coefficients(params: LogSvParams, grid: TensorGrid2D, eta: float) -> PDE2DCoefficients:
    """Generator on one interval with return diffusion eta * sigma and unscaled vol dynamics."""
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("the return volatility loading must be finite and positive")
    if params.kappa2 < eta * params.beta:
        raise ValueError("Q_MMA requires kappa2 >= eta * beta on every interval")
    if grid.y.nodes[0] < 0:
        raise ValueError("volatility nodes must be nonnegative")
    sigma = grid.y_mesh
    variance = sigma**2
    half_return_variance = 0.5 * eta**2 * variance
    return PDE2DCoefficients(
        variance_x=half_return_variance,
        drift_x=-half_return_variance,
        variance_y=0.5 * params.vartheta2 * variance,
        drift_y=(params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma),
        covariance_xy=eta * params.beta * variance,
        discount=np.zeros(grid.shape),
    )


@dataclass(frozen=True, slots=True)
class LogSvPdeCoefficients:
    """Undiscounted generator of (log(S/F), sigma) under the money-market measure.

    The volatility drift and Brownian loadings follow published Eq. (3.12).
    Pure second-derivative coefficients are half variances. The coefficient of
    the mixed derivative is the full instantaneous covariance.
    """

    model: LogSvModel
    is_time_dependent: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Require the standard model snapshot and its MMA martingale condition."""
        if not isinstance(self.model, LogSvModel):
            raise TypeError("model must be a LogSvModel")
        params = self.model.params
        if params.kappa2 < params.beta:
            raise ValueError("Q_MMA requires the true-martingale condition kappa2 >= beta")

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Evaluate the time-independent diffusion on a nonnegative volatility grid."""
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be finite and nonnegative")
        return _mma_coefficients(self.model.params, grid, eta=1.0)
