"""Tgarch model definitions.

Terminal path simulation for a discrete threshold-volatility model.

This TGARCH recursion is a distinct discrete model, not a discretization scheme for
``LogSVPricer``. Times are measured in years, rates and volatilities are annualized, and asset
returns are log returns. The terminal simulator stores only the initial and terminal observations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from stochvolmodels.data.model_paths import ModelPaths

from .params import (
    FloatArray,
    SIGMA_FLOOR,
    TgarchLimitParams,
    TgarchMeasure,
    TgarchParams,
    _as_measure,
    _positive_float,
    _positive_int,
    _validated_seed,
)

from .simulation import _DEFAULT_CHUNK_STEPS, _simulate_terminal


def _readonly(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TgarchModel:
    """Two-observation terminal adapter for the discrete TGARCH recursion."""

    params: TgarchParams

    def __post_init__(self) -> None:
        """Require a validated TGARCH parameter payload."""
        if not isinstance(self.params, TgarchParams):
            raise ValueError("params must be a TgarchParams instance")

    def simulate_paths(
        self,
        *,
        measure: TgarchMeasure | str,
        maturity: float,
        max_dt: float,
        n_paths: int,
        seed: int,
        limit_params: TgarchLimitParams | None = None,
        track_log_weights: bool = False,
        chunk_steps: int = _DEFAULT_CHUNK_STEPS,
    ) -> ModelPaths:
        """Simulate initial and terminal asset/state observations.

        ``max_dt`` is an upper bound. Equal steps use realized size
        ``maturity / ceil(maturity / max_dt)``. P-to-exact-Q log likelihood ratios are raw and
        available only for paths sampled under P; they are never self-normalized.

        Parameters
        ----------
        measure
            Physical, exact finite-step pricing, or limiting pricing law.
        maturity
            Positive terminal horizon in years.
        max_dt
            Positive upper bound for a simulation step in years.
        n_paths
            Positive number of terminal paths, including antithetic partners.
        seed
            Non-negative seed for ``Generator(PCG64(seed))``.
        limit_params
            Optional validated limit-Q drift parameters. Derived from ``params`` by default under
            Q_LIMIT; a valid payload is ignored under the other measures for legacy compatibility.
        track_log_weights
            Accumulate raw ``log(dQ_EXACT/dP)`` values on P paths.
        chunk_steps
            Positive number of time steps drawn per PCG64 block. Its value is part of the random
            ordering contract.

        Returns
        -------
        ModelPaths
            Initial and terminal spot, log-spot, volatility, measures and diagnostics.
        """
        measure_value = _as_measure(measure)
        maturity_value = _positive_float(maturity, "maturity")
        max_dt_value = _positive_float(max_dt, "max_dt")
        n_paths_value = _positive_int(n_paths, "n_paths")
        seed_value = _validated_seed(seed)
        chunk_value = _positive_int(chunk_steps, "chunk_steps")
        if not isinstance(track_log_weights, (bool, np.bool_)):
            raise ValueError("track_log_weights must be bool")
        track_weights_value = bool(track_log_weights)
        if track_weights_value and measure_value is not TgarchMeasure.P:
            raise ValueError("P-to-Q log weights can only be accumulated on P simulations")

        terminal = _simulate_terminal(
            params=self.params,
            measure=measure_value,
            maturity=maturity_value,
            max_dt=max_dt_value,
            n_paths=n_paths_value,
            seed=seed_value,
            limit_params=limit_params,
            track_log_weights=track_weights_value,
            chunk_steps=chunk_value,
        )

        observation_times = _readonly(np.array([0.0, maturity_value], dtype=np.float64))
        assets = np.empty((n_paths_value, 2, 1), dtype=np.float64)
        assets[:, 0, 0] = self.params.spot0
        assets[:, 1, 0] = terminal.spot
        log_spot = np.empty((n_paths_value, 2), dtype=np.float64)
        log_spot[:, 0] = np.log(self.params.spot0)
        log_spot[:, 1] = terminal.log_spot
        sigma = np.empty((n_paths_value, 2), dtype=np.float64)
        sigma[:, 0] = self.params.sigma0
        sigma[:, 1] = terminal.sigma

        target_measure = (
            TgarchMeasure.Q_EXACT.value
            if measure_value is TgarchMeasure.P and track_weights_value
            else measure_value.value
        )
        provenance: dict[str, Any] = {
            "requested_max_dt": max_dt_value,
            "realized_dt": terminal.dt,
            "n_steps": terminal.n_steps,
            "seed": seed_value,
            "generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "chunk_steps": chunk_value,
            "antithetic_layout": "first_half_draws_then_negatives_singleton_last",
            "numpy_version": np.__version__,
        }
        diagnostics: dict[str, Any] = {
            "sigma_floor": SIGMA_FLOOR,
            "floor_hits": terminal.floor_hits,
            "spot_overflow_count": terminal.spot_overflow_count,
            "effective_sample_size": terminal.effective_sample_size,
            "ess_fraction": terminal.ess_fraction,
            "low_ess": terminal.low_ess,
            "weight_convention": "raw_log_dQ_EXACT_dP" if track_weights_value else None,
        }
        log_weights = terminal.log_weights
        if log_weights is not None:
            _readonly(log_weights)
        return ModelPaths(
            observation_times=observation_times,
            assets=_readonly(assets),
            asset_ids=("spot",),
            sampling_measure=measure_value.value,
            target_measure=target_measure,
            numeraire="money_market_account",
            scheme="tgarch_terminal_recursion",
            states={"log_spot": _readonly(log_spot), "sigma": _readonly(sigma)},
            state_units={"log_spot": "log price", "sigma": "annualized volatility"},
            log_likelihood_ratios=log_weights,
            provenance=provenance,
            diagnostics=diagnostics,
        )
