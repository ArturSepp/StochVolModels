"""Independent parameter and kernel objects for experimental rough log-normal SV."""

from dataclasses import asdict, dataclass

import numpy as np

from stochvolmodels.pricers.model_pricer import ModelParams


@dataclass
class RoughLogSvParams(ModelParams):
    """Volatility parameters in annual units, with a unit price backbone.

    epsilon is orthogonal volatility-of-volatility, not total noise.
    H labels the fractional target; an explicit finite kernel is required
    when H < 1/2. Positive nodes alone do not qualify prices or calibration.
    """

    sigma0: float = 0.4
    theta: float = 0.4
    kappa1: float = 2.0
    kappa2: float = 2.0
    beta: float = 0.2
    epsilon: float = 1.0
    H: float = 0.3

    def __post_init__(self):
        """Reject nonfinite or inadmissible physical coefficients."""
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("RoughLogSvParams must be finite")
        if self.sigma0 <= 0 or min(self.theta, self.kappa1, self.kappa2, self.epsilon) < 0:
            raise ValueError("sigma0 must be positive; levels and speeds must be nonnegative")
        if not 0 < self.H <= 0.5:
            raise ValueError("H must lie in (0, 0.5]")

    @property
    def vartheta(self):
        """Total volatility-of-volatility."""
        return float(np.hypot(self.beta, self.epsilon))

    @property
    def rho(self):
        """Correlation between price and volatility Brownian innovations."""
        return self.beta / self.vartheta if self.vartheta else 0.0

    @property
    def martingale_margin(self):
        """Sufficient unit-backbone martingale margin under the pricing measure."""
        return self.kappa2 - self.beta

    def to_dict(self):
        """Return portable scalar model coefficients."""
        return asdict(self)


@dataclass(frozen=True)
class RoughKernel:
    """Explicit positive exponential kernel, with its fractional target label.

    The nodes have inverse-year units. smoothing is provenance, not an
    accuracy certificate. Use the saved kernel to reproduce a paper result.
    """

    nodes: tuple
    weights: tuple
    H: float
    smoothing: float | None = None

    def __post_init__(self):
        """Copy arrays into immutable tuples and validate the kernel."""
        x, w = np.asarray(self.nodes, dtype=float), np.asarray(self.weights, dtype=float)
        if (
            x.ndim != 1
            or len(x) == 0
            or w.shape != x.shape
            or not np.isfinite(x).all()
            or not np.isfinite(w).all()
            or np.any(x < 0)
            or np.any(w <= 0)
            or not 0 < self.H <= 0.5
        ):
            raise ValueError("Kernel requires finite nonnegative nodes and positive weights")
        if self.H == 0.5 and (len(x) != 1 or x[0] != 0 or w[0] != 1):
            raise ValueError("H=0.5 requires the exact constant kernel [0], [1]")
        if self.smoothing is not None and (not np.isfinite(self.smoothing) or self.smoothing <= 0):
            raise ValueError("Kernel smoothing must be positive and finite when supplied")
        object.__setattr__(self, "nodes", tuple(x))
        object.__setattr__(self, "weights", tuple(w))

    @classmethod
    def classical(cls):
        """Return the exact H=1/2 kernel."""
        return cls((0.0,), (1.0,), 0.5)

    @classmethod
    def from_dict(cls, values):
        """Read a saved paper kernel without fitting or changing its coefficients."""
        return cls(values["nodes"], values["weights"], values["H"], values.get("smoothing"))
