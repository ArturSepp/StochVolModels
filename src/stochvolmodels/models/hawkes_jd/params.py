"""Hawkes Jd params definitions.

Hawkes jump-diffusion parameters and model-specific affine transform equations.

Positive and negative shifted-exponential jumps drive self- and cross-exciting
intensities. The original Riccati equations and integration settings are retained;
pricing, calibration, risk-premium conversion and Monte Carlo wrappers remain in
the concrete pricer. These equations are distinct from the LogSV expansion.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
from stochvolmodels.models.base import ModelParams


@dataclass
class HawkesJDParams(ModelParams):
    """
    parameters of 2-factor Hawkes Jump Diffusion
    annualized params, close for BTC on daily frequency
    """

    mu: float = 0.0
    sigma: float = 0.45
    # jumps
    shift_p: float = 0.06
    mean_p: float = 0.03
    shift_m: float = -0.06
    mean_m: float = -0.03
    # positive jumps intensity
    lambda_p: float = 6.55
    theta_p: float = 6.55
    kappa_p: float = 22.29
    beta1_p: float = 76.0
    beta2_p: float = -67.58
    # minus jumps intensity
    lambda_m: float = 8.50
    theta_m: float = 8.50
    kappa_m: float = 29.0
    beta1_m: float = 104.55
    beta2_m: float = -109.6
    risk_premia_gamma: float = None

    def __post_init__(self):
        """validate the jump and intensity parameters."""
        self.compensator_p = np.exp(self.shift_p) / (1.0 - self.mean_p) - 1.0
        self.compensator_m = np.exp(self.shift_m) / (1.0 - self.mean_m) - 1.0

    def to_dict(self) -> Dict[str, Any]:
        """return the parameters as a plain dict."""
        return asdict(self)

    def print(self) -> None:
        """print the parameters, for interactive inspection."""
        for k, v in self.to_dict().items():
            print(f"{k}={v}")
        print("condifions")
        print(f"jump1={self.jump1_cond:0.4f} > 0")
        print(f"jump2={self.jump2_cond:0.4f} > 0")

    @property
    def jump1_cond(self) -> float:
        """
        stationarity margin of the positive-jump intensity.

        Returns kappa_p - beta1_p E[J_p] - beta2_p E[J_m]. The intensity is stationary
        while this is positive: decay must outpace the excitation contributed by both
        jump streams. Used as a calibration constraint, not enforced on construction.
        """
        return self.kappa_p - self.beta1_p * self.exp_jump_p - self.beta2_p * self.exp_jump_m

    @property
    def jump2_cond(self) -> float:
        """stationarity margin of the negative-jump intensity, the mirror of jump1_cond."""
        return self.kappa_m - self.beta2_m * self.exp_jump_m - self.beta1_m * self.exp_jump_p

    @property
    def exp_jump_p(self) -> float:
        """expected positive jump size, shift_p + mean_p for a shifted exponential."""
        return self.shift_p + self.mean_p

    @property
    def exp_jump_m(self) -> float:
        """expected negative jump size, shift_m + mean_m."""
        return self.shift_m + self.mean_m

    @property
    def jumps_var_m(self) -> float:
        """second moment of the negative jump size, shift_m^2 + mean_m^2."""
        return np.square(self.shift_m) + np.square(self.mean_m)

    @property
    def jumps_var_p(self) -> float:
        """second moment of the positive jump size, shift_p^2 + mean_p^2."""
        return np.square(self.shift_p) + np.square(self.mean_p)
