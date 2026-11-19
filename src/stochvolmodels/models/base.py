"""Common parameter container shared by stochastic-volatility models.

This module supplies dataclass copying without importing pricing, calibration,
plotting or simulation machinery. Concrete parameterizations live beside models.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ModelParams:
    """
    abstract container for the parameters of a pricing model.

    Subclasses add the model parameters as dataclass fields and supply the
    serialisation helpers the calibration and reporting layers call.
    """
    pass

    @classmethod
    def copy(cls, obj: ModelParams) -> ModelParams:
        """return a deep copy of the parameters."""
        return cls(**asdict(obj))
