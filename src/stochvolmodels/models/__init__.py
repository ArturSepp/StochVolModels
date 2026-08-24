"""Provisional structural capabilities for stochastic-volatility models.

Runtime checks recognize declared capability names only. Concrete implementations
remain responsible for their documented signatures, inputs, and outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from stochvolmodels.data.model_paths import ModelPaths

if TYPE_CHECKING:
    from stochvolmodels.data.option_chain import OptionSlice

__all__ = (
    "PathModel",
    "TerminalDistributionModel",
    "TerminalSmileModel",
    "TransformModel",
)


@runtime_checkable
class PathModel(Protocol):
    """Capability for dynamically consistent simulation on an observation grid."""

    def simulate_paths(self, **kwargs: object) -> ModelPaths:
        """Return a validated ``ModelPaths`` payload for a documented request."""
        ...


@runtime_checkable
class TransformModel(Protocol):
    """Capability for evaluating a model's logarithmic moment transform."""

    def log_mgf_grid(self, **kwargs: object) -> np.ndarray:
        """Return log-transform values shaped like the requested transform grid."""
        ...


@runtime_checkable
class TerminalDistributionModel(Protocol):
    """Capability for European pricing under a one-maturity terminal law."""

    @property
    def ttm(self) -> float:
        """Return the terminal law's time to maturity in years."""
        ...

    def price_european(self, option_slice: OptionSlice) -> np.ndarray:
        """Return prices shaped like ``option_slice.strikes``."""
        ...


@runtime_checkable
class TerminalSmileModel(Protocol):
    """Capability for evaluating a one-expiry implied-volatility smile."""

    @property
    def ttm(self) -> float:
        """Return the smile's time to maturity in years."""
        ...

    def implied_vols(self, option_slice: OptionSlice) -> np.ndarray:
        """Return implied volatilities shaped like ``option_slice.strikes``."""
        ...
