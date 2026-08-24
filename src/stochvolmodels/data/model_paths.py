"""Canonical path payload shared by dynamic models and pathwise payoffs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


def _validate_label(name: str, value: object) -> None:
    """Require a non-empty string without surrounding whitespace."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")


def _validate_real_array(name: str, value: object) -> np.ndarray:
    """Require a real numeric NumPy array and return it without copying."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have a real numeric dtype")
    return value


def _validate_metadata(name: str, value: object) -> None:
    """Validate an opaque metadata mapping without inspecting its values."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        _validate_label(f"{name} key", key)


@dataclass(eq=False)
class ModelPaths:
    """Store one dynamic model simulation on an explicit observation grid.

    Arrays and mappings are retained by reference. The container validates their
    structure but does not copy values, normalize likelihood ratios, or derive
    importance-sampling diagnostics.

    Parameters
    ----------
    observation_times
        Strictly increasing finite times beginning at zero, with shape ``(n_times,)``.
    assets
        Simulated asset levels with shape ``(n_paths, n_times, n_assets)``. Non-finite
        or non-positive values are retained so producers can report failed paths.
    asset_ids
        Unique identifiers for the explicit asset axis.
    sampling_measure
        Measure under which paths were sampled.
    target_measure
        Measure for which downstream expectations are intended.
    numeraire
        Numeraire convention associated with the simulated paths.
    scheme
        Identity of the numerical simulation scheme.
    states
        Model-defined state arrays. Each has leading shape ``(n_paths, n_times)`` and
        may have additional non-empty axes.
    state_units
        Units for every named state, using exactly the same keys as ``states``.
    log_likelihood_ratios
        Optional raw log likelihood ratios with shape ``(n_paths,)``. Values are
        retained exactly and are not normalized. Finite values and negative infinity
        are valid; NaN and positive infinity are rejected.
    provenance
        Opaque RNG and simulation provenance.
    diagnostics
        Opaque numerical and weighting diagnostics supplied by the producer.
    """

    observation_times: np.ndarray
    assets: np.ndarray
    asset_ids: tuple[str, ...]
    sampling_measure: str
    target_measure: str
    numeraire: str
    scheme: str
    states: Mapping[str, np.ndarray] = field(default_factory=dict)
    state_units: Mapping[str, str] = field(default_factory=dict)
    log_likelihood_ratios: np.ndarray | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate path alignment and convention metadata without mutation."""
        times = _validate_real_array("observation_times", self.observation_times)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("observation_times must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(times)):
            raise ValueError("observation_times must contain only finite values")
        if times[0] != 0.0:
            raise ValueError("observation_times must begin at zero")
        if np.any(times[1:] <= times[:-1]):
            raise ValueError("observation_times must be strictly increasing")

        assets = _validate_real_array("assets", self.assets)
        if assets.ndim != 3 or any(size == 0 for size in assets.shape):
            raise ValueError("assets must have non-empty shape (n_paths, n_times, n_assets)")
        if assets.shape[1] != times.size:
            raise ValueError("the assets time axis must match observation_times")

        if not isinstance(self.asset_ids, tuple):
            raise TypeError("asset_ids must be a tuple")
        for asset_id in self.asset_ids:
            _validate_label("asset_id", asset_id)
        if len(self.asset_ids) != assets.shape[2]:
            raise ValueError("asset_ids must match the assets asset axis")
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("asset_ids must be unique")

        for name in ("sampling_measure", "target_measure", "numeraire", "scheme"):
            _validate_label(name, getattr(self, name))

        if not isinstance(self.states, Mapping):
            raise TypeError("states must be a mapping")
        for state_name, state_values in self.states.items():
            _validate_label("state name", state_name)
            state_array = _validate_real_array(f"states[{state_name!r}]", state_values)
            if state_array.ndim < 2:
                raise ValueError("state arrays must have at least path and time axes")
            if state_array.shape[:2] != assets.shape[:2]:
                raise ValueError("state path and time axes must match assets")
            if any(size == 0 for size in state_array.shape[2:]):
                raise ValueError("state trailing axes must be non-empty")

        if not isinstance(self.state_units, Mapping):
            raise TypeError("state_units must be a mapping")
        if set(self.state_units) != set(self.states):
            raise ValueError("state_units keys must exactly match states keys")
        for state_name, unit in self.state_units.items():
            _validate_label("state unit name", state_name)
            _validate_label(f"state_units[{state_name!r}]", unit)

        if self.log_likelihood_ratios is not None:
            log_ratios = _validate_real_array(
                "log_likelihood_ratios", self.log_likelihood_ratios
            )
            if log_ratios.ndim != 1 or log_ratios.size != assets.shape[0]:
                raise ValueError("log_likelihood_ratios must have shape (n_paths,)")
            if np.any(np.isnan(log_ratios)) or np.any(np.isposinf(log_ratios)):
                raise ValueError("log_likelihood_ratios cannot contain NaN or positive infinity")

        _validate_metadata("provenance", self.provenance)
        _validate_metadata("diagnostics", self.diagnostics)
