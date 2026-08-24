"""Concrete pathwise payoffs for European and variance options."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.utils.config import OptionType

__all__ = (
    "EuropeanOptionPayoff",
    "IntegratedVarianceOptionPayoff",
    "VarianceQuote",
)

FloatArray = NDArray[np.float64]
_INTEGRATED_VARIANCE_UNIT = "integrated variance"


class VarianceQuote(str, Enum):
    """Quote convention for an integrated-variance option underlying."""

    INTEGRATED = "integrated"
    ANNUALIZED = "annualized"


def _validated_label(value: object, name: str) -> str:
    """Require a non-empty string without surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    return value


def _finite_float(value: object, name: str) -> float:
    """Return a finite real scalar while rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: object, name: str) -> float:
    """Return a strictly positive finite scalar."""
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {result}")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    """Return a non-negative finite scalar."""
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _as_option_type(value: OptionType | str) -> OptionType:
    """Canonicalize a standard call or put code."""
    try:
        option_type = OptionType(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"option_type must be C or P, got {value!r}") from exc
    if option_type not in (OptionType.CALL, OptionType.PUT):
        raise ValueError(f"option_type must be C or P, got {value!r}")
    return option_type


def _as_variance_quote(value: VarianceQuote | str) -> VarianceQuote:
    """Canonicalize an explicit integrated-variance quote convention."""
    try:
        return VarianceQuote(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in VarianceQuote)
        raise ValueError(f"quote must be one of {{{allowed}}}, got {value!r}") from exc


def _require_model_paths(paths: object) -> ModelPaths:
    """Require the canonical path payload rather than an implicit duck type."""
    if not isinstance(paths, ModelPaths):
        raise TypeError("paths must be a ModelPaths instance")
    return paths


def _observation_index(paths: ModelPaths, expiry: float) -> int:
    """Return the unique observation index matching expiry exactly."""
    matches = np.flatnonzero(paths.observation_times == expiry)
    if matches.size != 1:
        raise ValueError(f"expiry {expiry} must match exactly one observation time")
    return int(matches[0])


def _validated_vector(values: object, *, name: str, n_paths: int) -> FloatArray:
    """Copy a finite real path vector into a float64 array."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have a real numeric dtype")
    if values.ndim != 1 or values.shape != (n_paths,):
        raise ValueError(f"{name} must have shape ({n_paths},)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} cannot be represented as finite float64 values")
    return result


def _vanilla_payoff(
    underlying: FloatArray,
    *,
    strike: float,
    option_type: OptionType,
) -> FloatArray:
    """Return a new read-only standard call or put payoff vector."""
    if not np.all(np.isfinite(underlying)):
        raise ValueError("option underlying must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        if option_type is OptionType.CALL:
            result = np.maximum(underlying - strike, 0.0)
        else:
            result = np.maximum(strike - underlying, 0.0)
    result = np.asarray(result, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("option payoff cannot be represented as finite float64 values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class EuropeanOptionPayoff:
    """Standard European call or put on one explicitly selected path asset.

    Parameters
    ----------
    asset_id
        Identifier on the explicit asset axis of :class:`ModelPaths`.
    expiry
        Positive observation time in years. It must occur exactly on the path grid.
    strike
        Non-negative strike in the selected asset's units.
    option_type
        Standard call (``C``) or put (``P``). Inverse option codes are rejected.
    unit
        Explicit common payoff and settlement unit.
    """

    asset_id: str
    expiry: float
    strike: float
    option_type: OptionType | str
    unit: str

    def __post_init__(self) -> None:
        """Validate and canonicalize the product definition."""
        object.__setattr__(self, "asset_id", _validated_label(self.asset_id, "asset_id"))
        object.__setattr__(self, "expiry", _positive_float(self.expiry, "expiry"))
        object.__setattr__(self, "strike", _nonnegative_float(self.strike, "strike"))
        object.__setattr__(self, "option_type", _as_option_type(self.option_type))
        object.__setattr__(self, "unit", _validated_label(self.unit, "unit"))

    @property
    def required_asset_ids(self) -> tuple[str, ...]:
        """Return the single path asset required by this payoff."""
        return (self.asset_id,)

    @property
    def payoff_unit(self) -> str:
        """Return the unit of the unconverted payoff values."""
        return self.unit

    @property
    def settlement_unit(self) -> str:
        """Return the settlement unit, identical to the payoff unit in D2."""
        return self.unit

    def _payoff_from_asset_values(
        self,
        values: np.ndarray,
        *,
        n_paths: int | None = None,
    ) -> FloatArray:
        """Evaluate finite terminal values supplied by the valuation compatibility policy."""
        if not isinstance(values, np.ndarray):
            raise TypeError("asset values must be a NumPy array")
        expected_paths = values.shape[0] if n_paths is None and values.ndim == 1 else n_paths
        if expected_paths is None:
            raise ValueError("asset values must be one-dimensional")
        underlying = _validated_vector(
            values,
            name="selected terminal asset values",
            n_paths=expected_paths,
        )
        return _vanilla_payoff(
            underlying,
            strike=self.strike,
            option_type=self.option_type,
        )

    def __call__(self, paths: ModelPaths) -> FloatArray:
        """Evaluate the payoff on the selected asset and exact expiry."""
        path_values = _require_model_paths(paths)
        asset_indices = [
            index
            for index, asset_id in enumerate(path_values.asset_ids)
            if asset_id == self.asset_id
        ]
        if len(asset_indices) != 1:
            raise ValueError(f"asset_id {self.asset_id!r} must occur exactly once in paths")
        time_index = _observation_index(path_values, self.expiry)
        selected = path_values.assets[:, time_index, asset_indices[0]]
        return self._payoff_from_asset_values(selected, n_paths=path_values.assets.shape[0])


@dataclass(frozen=True, slots=True, kw_only=True)
class IntegratedVarianceOptionPayoff:
    """Call or put on an integrated or annualized variance increment.

    Parameters
    ----------
    state_name
        Name of a scalar cumulative state whose unit is exactly ``integrated variance``.
    expiry
        Positive observation time in years. It must occur exactly on the path grid.
    strike
        Non-negative strike in the selected quote convention.
    option_type
        Standard call (``C``) or put (``P``). Inverse option codes are rejected.
    quote
        Whether the underlying is the raw integrated increment or that increment divided by
        expiry.
    unit
        Explicit common payoff and settlement unit.
    """

    state_name: str
    expiry: float
    strike: float
    option_type: OptionType | str
    quote: VarianceQuote | str
    unit: str

    def __post_init__(self) -> None:
        """Validate and canonicalize the product definition."""
        object.__setattr__(
            self,
            "state_name",
            _validated_label(self.state_name, "state_name"),
        )
        object.__setattr__(self, "expiry", _positive_float(self.expiry, "expiry"))
        object.__setattr__(self, "strike", _nonnegative_float(self.strike, "strike"))
        object.__setattr__(self, "option_type", _as_option_type(self.option_type))
        object.__setattr__(self, "quote", _as_variance_quote(self.quote))
        object.__setattr__(self, "unit", _validated_label(self.unit, "unit"))

    @property
    def required_asset_ids(self) -> tuple[str, ...]:
        """Return no asset requirements because this payoff consumes a named state."""
        return ()

    @property
    def payoff_unit(self) -> str:
        """Return the unit of the unconverted payoff values."""
        return self.unit

    @property
    def settlement_unit(self) -> str:
        """Return the settlement unit, identical to the payoff unit in D2."""
        return self.unit

    def __call__(self, paths: ModelPaths) -> FloatArray:
        """Evaluate the payoff on the selected cumulative variance state."""
        path_values = _require_model_paths(paths)
        if self.state_name not in path_values.states:
            raise ValueError(f"state {self.state_name!r} is not present in paths")

        state = path_values.states[self.state_name]
        if state.ndim != 2:
            raise ValueError(
                f"state {self.state_name!r} must be scalar with shape (n_paths, n_times)"
            )
        state_unit = path_values.state_units[self.state_name]
        if state_unit != _INTEGRATED_VARIANCE_UNIT:
            raise ValueError(
                f"state {self.state_name!r} must have unit {_INTEGRATED_VARIANCE_UNIT!r}"
            )

        time_index = _observation_index(path_values, self.expiry)
        n_paths = path_values.assets.shape[0]
        state_interval = state[:, : time_index + 1]
        if not np.all(np.isfinite(state_interval)):
            raise ValueError("integrated variance state must contain only finite values to expiry")
        if np.any(state_interval < 0.0):
            raise ValueError("integrated variance state must be non-negative to expiry")
        if np.any(np.diff(state_interval, axis=1) < 0.0):
            raise ValueError("integrated variance state must be non-decreasing to expiry")
        initial = _validated_vector(
            state[:, 0],
            name=f"states[{self.state_name!r}] at time zero",
            n_paths=n_paths,
        )
        terminal = _validated_vector(
            state[:, time_index],
            name=f"states[{self.state_name!r}] at expiry",
            n_paths=n_paths,
        )
        variance_increment = terminal - initial
        if not np.all(np.isfinite(variance_increment)):
            raise ValueError("integrated variance increments must contain only finite values")
        if np.any(variance_increment < 0.0):
            raise ValueError("integrated variance increments must be non-negative")
        if self.quote is VarianceQuote.ANNUALIZED:
            with np.errstate(over="ignore", invalid="ignore"):
                variance_increment = variance_increment / self.expiry

        return _vanilla_payoff(
            variance_increment,
            strike=self.strike,
            option_type=self.option_type,
        )
