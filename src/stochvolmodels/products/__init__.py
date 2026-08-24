"""Structural contract for pathwise payoff definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from stochvolmodels.data.model_paths import ModelPaths

__all__ = ("Payoff",)


@runtime_checkable
class Payoff(Protocol):
    """Pathwise payoff with explicit asset, expiry, and settlement metadata."""

    @property
    def required_asset_ids(self) -> tuple[str, ...]:
        """Return the asset identifiers consumed by the payoff."""
        ...

    @property
    def expiry(self) -> float:
        """Return the payoff observation time in years."""
        ...

    @property
    def payoff_unit(self) -> str:
        """Return the unit of the unconverted payoff values."""
        ...

    @property
    def settlement_unit(self) -> str:
        """Return the unit in which the payoff settles."""
        ...

    def __call__(self, paths: ModelPaths) -> np.ndarray:
        """Return payoff values with shape ``(n_paths,)``."""
        ...
