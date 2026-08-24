"""Path adapter for the maintained continuous equity LogSV simulation.

The adapter preserves the legacy interval grid and MT19937 random ordering. It exposes the
zero-drift price, volatility and integrated variance without changing the numerical kernel in
``stochvolmodels.pricers.logsv_pricer``. Rough kernels and volatility backbones are deliberately
outside this boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from stochvolmodels.data.model_paths import ModelPaths

if TYPE_CHECKING:
    from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

__all__ = ("LogSvMeasure", "LogSvModel")

_MAX_RANDOMSTATE_SEED = 2**32 - 1
_SCHEME = "logsv_explicit_log_euler_trapezoidal_qvar_v1"
_LEGACY_SOURCE_SHA256 = "50f8b31efae2c3c714e981ced0b3b30d92cb33cb8290702e323a1433a611b40f"

FloatArray = NDArray[np.float64]


class LogSvMeasure(str, Enum):
    """Pricing measure used by the maintained continuous LogSV kernel."""

    MMA = "Q_MMA"
    INVERSE = "Q_INVERSE"


def _finite_float(value: object, name: str) -> float:
    """Return a finite built-in float while rejecting booleans and non-real payloads."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: object, name: str) -> float:
    """Return a strictly positive finite float."""
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {result}")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    """Return a non-negative finite float."""
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _positive_int(value: object, name: str) -> int:
    """Return a positive integer while rejecting booleans and integer-like floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be strictly positive, got {result}")
    return result


def _validated_seed(value: object) -> int:
    """Return a seed in the range accepted by ``numpy.random.RandomState``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    result = int(value)
    if not 0 <= result <= _MAX_RANDOMSTATE_SEED:
        raise ValueError(f"seed must be in [0, 2**32 - 1], got {result}")
    return result


def _as_measure(value: LogSvMeasure | str) -> LogSvMeasure:
    """Canonicalize a pricing-measure selector."""
    try:
        return LogSvMeasure(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in LogSvMeasure)
        raise ValueError(f"measure must be one of {{{allowed}}}, got {value!r}") from exc


def _observation_grid(value: object) -> FloatArray:
    """Validate and detach a full observation grid beginning exactly at zero."""
    if not isinstance(value, np.ndarray):
        raise ValueError("observation_times must be a NumPy array")
    if value.dtype.kind not in "iuf":
        raise ValueError("observation_times must have a real numeric dtype")
    if value.ndim != 1 or value.size < 2:
        raise ValueError("observation_times must be one-dimensional with at least two entries")
    if not np.isfinite(value).all():
        raise ValueError("observation_times must contain only finite values")

    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("observation_times cannot be represented as finite float64 values")
    if result[0] != 0.0:
        raise ValueError("observation_times must begin exactly at zero")
    if np.any(result[1:] <= result[:-1]):
        raise ValueError("observation_times must be strictly increasing")
    return result


def _readonly(array: FloatArray) -> FloatArray:
    """Mark a producer-owned array read-only and return it."""
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class LogSvModel:
    """Parameter-bound path adapter for the standard continuous LogSV dynamics.

    The mutable legacy parameter object is validated and reduced to a frozen snapshot of its six
    scalar dynamics. Accessing :attr:`params` returns a new detached ``LogSvParams`` instance.
    """

    _sigma0: float
    _theta: float
    _kappa1: float
    _kappa2: float
    _beta: float
    _volvol: float

    def __init__(self, params: LogSvParams) -> None:
        """Validate and snapshot a standard continuous ``LogSvParams`` payload."""
        from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

        if not isinstance(params, LogSvParams):
            raise ValueError("params must be a LogSvParams instance")

        hurst = _finite_float(params.H, "H")
        if hurst != 0.5:
            raise ValueError("LogSvModel supports only the standard continuous kernel with H=0.5")
        if params.vol_backbone is not None:
            raise ValueError("LogSvModel does not support vol_backbone term structures")
        if params.nodes is not None or params.weights is not None:
            raise ValueError("LogSvModel does not support rough-kernel nodes or weights")

        object.__setattr__(self, "_sigma0", _positive_float(params.sigma0, "sigma0"))
        object.__setattr__(self, "_theta", _positive_float(params.theta, "theta"))
        object.__setattr__(self, "_kappa1", _nonnegative_float(params.kappa1, "kappa1"))
        object.__setattr__(self, "_kappa2", _nonnegative_float(params.kappa2, "kappa2"))
        object.__setattr__(self, "_beta", _finite_float(params.beta, "beta"))
        object.__setattr__(self, "_volvol", _nonnegative_float(params.volvol, "volvol"))

    @property
    def params(self) -> LogSvParams:
        """Return a fresh standard ``LogSvParams`` detached from this model's snapshot."""
        from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

        return LogSvParams(
            sigma0=self._sigma0,
            theta=self._theta,
            kappa1=self._kappa1,
            kappa2=self._kappa2,
            beta=self._beta,
            volvol=self._volvol,
        )

    def simulate_paths(
        self,
        *,
        measure: LogSvMeasure | str,
        observation_times: np.ndarray,
        spot0: float,
        n_paths: int,
        steps_per_year: int,
        seed: int,
    ) -> ModelPaths:
        """Simulate the maintained LogSV kernel on a requested observation partition.

        Every interval uses the historical rule ``int(interval * steps_per_year) + 1``. The
        local ``RandomState`` helper draws the complete W0 matrix and then the complete W1 matrix
        for each interval. Consequently the observation partition is part of the numerical and
        random-ordering contract, even when the seed and terminal horizon are unchanged.

        Parameters
        ----------
        measure
            Money-market-account measure ``Q_MMA`` or spot-numeraire inverse measure
            ``Q_INVERSE``.
        observation_times
            Strictly increasing NumPy array of observation times in years, beginning at zero.
        spot0
            Positive initial level of the raw zero-drift asset.
        n_paths
            Positive number of simulated paths.
        steps_per_year
            Positive requested time-step density used separately on every observation interval.
        seed
            Integer seed in the ``RandomState`` range ``[0, 2**32 - 1]``.

        Returns
        -------
        ModelPaths
            Raw zero-drift price, log return, annualized volatility and integrated variance on
            the full observation grid. All returned arrays are read-only.
        """
        measure_value = _as_measure(measure)
        times = _observation_grid(observation_times)
        spot0_value = _positive_float(spot0, "spot0")
        n_paths_value = _positive_int(n_paths, "n_paths")
        steps_value = _positive_int(steps_per_year, "steps_per_year")
        seed_value = _validated_seed(seed)

        if measure_value is LogSvMeasure.MMA:
            if self._kappa2 < self._beta:
                raise ValueError("Q_MMA requires the true-martingale condition kappa2 >= beta")
            numeraire = "money_market_account"
            martingale_condition = "kappa2 >= beta"
            is_spot_measure = True
        else:
            if self._kappa2 < 2.0 * self._beta:
                raise ValueError(
                    "Q_INVERSE requires the true-martingale condition kappa2 >= 2 * beta"
                )
            numeraire = "spot"
            martingale_condition = "kappa2 >= 2 * beta"
            is_spot_measure = False

        # These imports stay local so importing this public adapter does not eagerly import the
        # legacy pricing/calibration stack or compile its Numba simulation kernel.
        from stochvolmodels.pricers.logsv_pricer import (
            get_randoms_for_chain_valuation,
            simulate_logsv_x_vol_terminal,
        )

        w0_blocks, w1_blocks, interval_dts_raw = get_randoms_for_chain_valuation(
            ttms=times[1:],
            nb_path=n_paths_value,
            nb_steps_per_year=steps_value,
            seed=seed_value,
        )
        interval_dts = tuple(float(value) for value in interval_dts_raw)
        interval_step_counts = tuple(int(block.shape[0]) for block in w0_blocks)

        n_times = times.size
        log_returns = np.empty((n_paths_value, n_times), dtype=np.float64)
        sigmas = np.empty((n_paths_value, n_times), dtype=np.float64)
        quadratic_variances = np.empty((n_paths_value, n_times), dtype=np.float64)
        log_returns[:, 0] = 0.0
        sigmas[:, 0] = self._sigma0
        quadratic_variances[:, 0] = 0.0

        x = np.zeros(n_paths_value, dtype=np.float64)
        sigma = np.full(n_paths_value, self._sigma0, dtype=np.float64)
        qvar = np.zeros(n_paths_value, dtype=np.float64)
        for interval_index, (w0, w1, dt) in enumerate(
            zip(w0_blocks, w1_blocks, interval_dts),
            start=1,
        ):
            interval = float(times[interval_index] - times[interval_index - 1])
            x, sigma, qvar = simulate_logsv_x_vol_terminal(
                ttm=interval,
                x0=x,
                sigma0=sigma,
                qvar0=qvar,
                theta=self._theta,
                kappa1=self._kappa1,
                kappa2=self._kappa2,
                beta=self._beta,
                volvol=self._volvol,
                vol_backbone_eta=1.0,
                is_spot_measure=is_spot_measure,
                nb_path=n_paths_value,
                nb_steps_per_year=steps_value,
                W0=w0,
                W1=w1,
                dt=dt,
            )
            log_returns[:, interval_index] = x
            sigmas[:, interval_index] = sigma
            quadratic_variances[:, interval_index] = qvar

        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            asset_levels = spot0_value * np.exp(log_returns)
        assets = np.asarray(asset_levels[:, :, np.newaxis], dtype=np.float64).copy()
        failed_path_mask = (
            np.any(~np.isfinite(asset_levels) | (asset_levels <= 0.0), axis=1)
            | np.any(~np.isfinite(log_returns), axis=1)
            | np.any(~np.isfinite(sigmas) | (sigmas <= 0.0), axis=1)
            | np.any(~np.isfinite(quadratic_variances) | (quadratic_variances < 0.0), axis=1)
        )

        provenance: dict[str, Any] = {
            "steps_per_year": steps_value,
            "interval_step_counts": interval_step_counts,
            "interval_dts": interval_dts,
            "seed": seed_value,
            "generator": "numpy.random.RandomState",
            "bit_generator": "MT19937",
            "normal_convention": "unscaled_standard_normals_kernel_scales_sqrt_dt",
            "draw_ordering": "per_interval_W0_matrix_then_W1_matrix_row_major",
            "kernel": "simulate_logsv_x_vol_terminal",
            "legacy_source_sha256": _LEGACY_SOURCE_SHA256,
            "vol_backbone_eta": 1.0,
            "numpy_version": np.__version__,
            "observation_partition_dependent": True,
        }
        diagnostics: dict[str, Any] = {
            "nonfinite_asset_count": int(np.count_nonzero(~np.isfinite(assets))),
            "nonfinite_log_return_count": int(np.count_nonzero(~np.isfinite(log_returns))),
            "nonfinite_sigma_count": int(np.count_nonzero(~np.isfinite(sigmas))),
            "nonfinite_qvar_count": int(
                np.count_nonzero(~np.isfinite(quadratic_variances))
            ),
            "asset_positive_infinity_count": int(np.count_nonzero(np.isposinf(assets))),
            "asset_underflow_count": int(np.count_nonzero(asset_levels == 0.0)),
            "nonpositive_sigma_count": int(np.count_nonzero(sigmas <= 0.0)),
            "negative_qvar_count": int(np.count_nonzero(quadratic_variances < 0.0)),
            "qvar_decrease_count": int(
                np.count_nonzero(np.diff(quadratic_variances, axis=1) < 0.0)
            ),
            "failed_path_count": int(np.count_nonzero(failed_path_mask)),
            "measure_martingale_condition": martingale_condition,
        }

        return ModelPaths(
            observation_times=_readonly(times),
            assets=_readonly(assets),
            asset_ids=("zero_drift_price",),
            sampling_measure=measure_value.value,
            target_measure=measure_value.value,
            numeraire=numeraire,
            scheme=_SCHEME,
            states={
                "log_zero_drift_return": _readonly(log_returns),
                "sigma": _readonly(sigmas),
                "quadratic_variance": _readonly(quadratic_variances),
            },
            state_units={
                "log_zero_drift_return": "log return",
                "sigma": "annualized volatility",
                "quadratic_variance": "integrated variance",
            },
            log_likelihood_ratios=None,
            provenance=provenance,
            diagnostics=diagnostics,
        )
