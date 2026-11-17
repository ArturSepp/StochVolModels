"""LogSV generators in normalized log-forward and volatility coordinates."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.pde_solvers import PDE2DCoefficients, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams


def _mma_coefficients(
    params: LogSvParams, grid: TensorGrid2D, eta: float | np.ndarray
) -> PDE2DCoefficients:
    """Generator on one interval with return diffusion eta * sigma and unscaled vol dynamics."""
    if np.any(params.kappa2 < eta * params.beta):
        raise ValueError("Q_MMA requires kappa2 >= eta * beta on every interval")
    return _diffusion_coefficients(params, grid, eta, is_spot_measure=True)


def _diffusion_coefficients(
    params: LogSvParams, grid: TensorGrid2D, eta: float | np.ndarray, *, is_spot_measure: bool
) -> PDE2DCoefficients:
    """Return the local generator under the money-market or inverse measure."""
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0):
        raise ValueError("the return volatility loading must be finite and positive")
    if grid.y.nodes[0] < 0:
        raise ValueError("volatility nodes must be nonnegative")
    sigma = grid.y_mesh
    variance = sigma**2
    half_return_variance = 0.5 * eta**2 * variance
    covariance = eta * params.beta * variance
    drift_y = (params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma)
    if not is_spot_measure:
        drift_y = drift_y + covariance
    return PDE2DCoefficients(
        variance_x=half_return_variance,
        drift_x=-half_return_variance if is_spot_measure else half_return_variance,
        variance_y=0.5 * params.vartheta2 * variance,
        drift_y=drift_y,
        covariance_xy=covariance,
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


@dataclass(frozen=True, slots=True)
class LogSvCoefficientProvider:
    """Map the existing :class:`LogSvParams` dynamics to an ``(x, sigma)`` PDE.

    The pricing coordinate is normalized log-forward ``x = log(S/F)`` and the
    second coordinate is volatility ``sigma``. Under the money-market-account
    measure, the current SVM dynamics are

    ``dx = -0.5 * (eta*sigma)^2 dt + eta*sigma dW0``

    and

    ``d sigma = (kappa1+kappa2*sigma)*(theta-sigma) dt``
    ``          + beta*sigma dW0 + volvol*sigma dW1``.

    Here ``eta`` is the existing optional volatility-backbone scale. The inverse
    measure follows the convention already implemented by the current simulator:
    the x drift changes sign and ``beta*eta*sigma^2`` is added to the volatility
    drift. Only the Markovian ``H=0.5`` model is represented by this two-state PDE.
    """

    params: LogSvParams
    is_spot_measure: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LogSvCoefficientProvider`` instance."""
        if not isinstance(self.params, LogSvParams):
            raise TypeError("params must be the existing LogSvParams type")
        if not isinstance(self.is_spot_measure, bool):
            raise TypeError("is_spot_measure must be bool")
        if not np.isclose(float(self.params.H), 0.5, rtol=0.0, atol=1.0e-14):
            raise ValueError("the two-dimensional log-SV PDE requires H=0.5")
        _validate_params(self.params)

    @property
    def is_time_dependent(self) -> bool:
        """Whether the existing volatility backbone makes coefficients time dependent."""
        return self.params.vol_backbone is not None

    def volatility_scale(self, time: float) -> float:
        """Return the current model's multiplicative spot-volatility backbone scale."""
        evaluation_time = _validated_time(time)
        scale = float(self.params.get_vol_backbone_eta(tau=evaluation_time))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("volatility-backbone scales must be finite and positive")
        return scale

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Evaluate dense generic PDE coefficients on an ``(x, sigma)`` tensor grid."""
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        return _diffusion_coefficients(
            self.params, grid, self.volatility_scale(time), is_spot_measure=self.is_spot_measure
        )



def _validate_params(params: LogSvParams) -> None:
    """Validate LogSV parameters used by the coefficient provider."""
    names = ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
    values = {name: float(getattr(params, name)) for name in names}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("log-SV PDE parameters must be finite")
    if values["sigma0"] <= 0.0 or values["theta"] <= 0.0:
        raise ValueError("sigma0 and theta must be positive")
    if values["kappa1"] < 0.0 or values["kappa2"] < 0.0:
        raise ValueError("kappa1 and kappa2 must be non-negative")
    if values["volvol"] < 0.0:
        raise ValueError("volvol must be non-negative")

    backbone = params.vol_backbone
    if backbone is not None:
        maturities = np.asarray(backbone.index, dtype=np.float64)
        scales = np.asarray(backbone.to_numpy(), dtype=np.float64)
        if maturities.ndim != 1 or scales.ndim != 1 or maturities.size != scales.size:
            raise ValueError("volatility backbone must contain aligned one-dimensional data")
        if maturities.size == 0:
            raise ValueError("volatility backbone must not be empty")
        if (
            not np.all(np.isfinite(maturities))
            or np.any(maturities < 0.0)
            or np.any(np.diff(maturities) <= 0.0)
        ):
            raise ValueError("volatility-backbone maturities must be finite and increasing")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("volatility-backbone scales must be finite and positive")


def _validated_time(time: float) -> float:
    """Return a finite coefficient evaluation time."""
    value = float(time)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("evaluation time must be finite and non-negative")
    return value
