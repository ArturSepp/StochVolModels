"""Money-market LogSV generator in normalized log-forward and volatility coordinates."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.pde_solvers import PDE2DCoefficients, TensorGrid2D


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
        if grid.y.nodes[0] < 0:
            raise ValueError("volatility nodes must be nonnegative")
        params = self.model.params
        sigma = grid.y_mesh
        variance = sigma**2
        return PDE2DCoefficients(
            variance_x=0.5 * variance,
            drift_x=-0.5 * variance,
            variance_y=0.5 * params.vartheta2 * variance,
            drift_y=(params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma),
            covariance_xy=params.beta * variance,
            discount=np.zeros(grid.shape),
        )
