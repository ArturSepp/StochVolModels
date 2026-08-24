"""Discounted path valuation with explicit measure and estimator conventions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.products import Payoff
from stochvolmodels.products.payoffs import EuropeanOptionPayoff

__all__ = (
    "LegacyAdditiveForwardRecenter",
    "PathEstimator",
    "PathValuationResult",
    "RecenterMean",
    "value_paths",
    "value_paths_self_normalized",
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer]

_LOG_MAX_FLOAT = math.log(np.finfo(np.float64).max)
_LOG_MIN_SUBNORMAL = math.log(float(np.nextafter(0.0, 1.0)))


class PathEstimator(str, Enum):
    """Estimator used for a pathwise expectation."""

    MONTE_CARLO = "monte_carlo"
    RAW_LIKELIHOOD_RATIO = "raw_likelihood_ratio"
    SELF_NORMALIZED_LIKELIHOOD_RATIO = "self_normalized_likelihood_ratio"


class RecenterMean(str, Enum):
    """Sample mean used by the explicit legacy forward-recentering policy."""

    UNWEIGHTED = "unweighted"
    NORMALIZED_LIKELIHOOD_WEIGHTED = "normalized_likelihood_weighted"


def _finite_positive_float(value: object, name: str) -> float:
    """Return a finite positive float while rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _as_recenter_mean(value: object) -> RecenterMean:
    """Canonicalize a recenter-mean selection."""
    try:
        return RecenterMean(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in RecenterMean)
        raise ValueError(f"mean must be one of {{{allowed}}}") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class LegacyAdditiveForwardRecenter:
    """Opt in to the historical additive terminal-forward recentering.

    Parameters
    ----------
    target_forward
        Positive target mean for the selected terminal asset.
    mean
        Arithmetic sample mean, or a normalized-likelihood-weighted mean available only
        through :func:`value_paths_self_normalized`.
    """

    target_forward: float
    mean: RecenterMean

    def __post_init__(self) -> None:
        """Validate and canonicalize the immutable policy."""
        object.__setattr__(
            self,
            "target_forward",
            _finite_positive_float(self.target_forward, "target_forward"),
        )
        object.__setattr__(self, "mean", _as_recenter_mean(self.mean))


@dataclass(frozen=True, slots=True, kw_only=True)
class PathValuationResult:
    """Immutable pathwise value, uncertainty, and weighting diagnostics."""

    value: float
    standard_error: float
    estimator: PathEstimator
    standard_error_basis: str
    n_paths: int
    n_independent_groups: int
    group_size: int
    settlement_unit: str
    path_effective_sample_size: float
    path_ess_fraction: float
    group_effective_sample_size: float
    group_ess_fraction: float
    mean_likelihood_ratio: float | None
    log_mean_likelihood_ratio: float | None
    recenter_shift: float | None


@dataclass(frozen=True, slots=True)
class _Grouping:
    """Resolved equal-size independent-group structure."""

    inverse: IntArray
    n_groups: int
    group_size: int
    explicit: bool


@dataclass(frozen=True, slots=True)
class _Weights:
    """Shifted likelihood weights and raw-scale diagnostics.

    ``log_ratios`` stores raw ``log(d target_measure / d sampling_measure)``.
    """

    log_ratios: FloatArray
    scaled: FloatArray
    log_scale: float
    mean: float
    log_mean: float
    path_ess: float


@dataclass(frozen=True, slots=True)
class _ScaledSignedValues:
    """Signed values represented relative to a common exponential log scale."""

    scaled: FloatArray
    log_scale: float


def _validate_label(value: object, name: str) -> str:
    """Return a non-empty, trimmed label."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")
    return value


def _validate_payoff_contract(paths: ModelPaths, payoff: Payoff) -> str:
    """Validate payoff metadata shared by all concrete products."""
    if not isinstance(payoff, Payoff):
        raise TypeError("payoff must implement the Payoff protocol")

    required_asset_ids = payoff.required_asset_ids
    if not isinstance(required_asset_ids, tuple):
        raise TypeError("payoff.required_asset_ids must be a tuple")
    for asset_id in required_asset_ids:
        _validate_label(asset_id, "required asset ID")
    if len(set(required_asset_ids)) != len(required_asset_ids):
        raise ValueError("payoff.required_asset_ids must be unique")
    missing = tuple(asset_id for asset_id in required_asset_ids if asset_id not in paths.asset_ids)
    if missing:
        raise ValueError(f"payoff requires unavailable asset IDs: {missing!r}")

    expiry = _finite_positive_float(payoff.expiry, "payoff.expiry")
    if np.count_nonzero(paths.observation_times == expiry) != 1:
        raise ValueError("payoff expiry must exactly match one path observation time")

    payoff_unit = _validate_label(payoff.payoff_unit, "payoff.payoff_unit")
    settlement_unit = _validate_label(payoff.settlement_unit, "payoff.settlement_unit")
    if payoff_unit != settlement_unit:
        raise ValueError("payoff and settlement units must agree")
    return settlement_unit


def _resolve_grouping(
    independent_group_ids: np.ndarray | None,
    *,
    n_paths: int,
) -> _Grouping:
    """Validate and encode caller-declared equal-size independent groups."""
    if independent_group_ids is None:
        if n_paths < 2:
            raise ValueError("path valuation requires at least two independent paths")
        return _Grouping(
            inverse=np.arange(n_paths, dtype=np.int64),
            n_groups=n_paths,
            group_size=1,
            explicit=False,
        )

    if not isinstance(independent_group_ids, np.ndarray):
        raise TypeError("independent_group_ids must be a NumPy array")
    if independent_group_ids.ndim != 1 or independent_group_ids.shape != (n_paths,):
        raise ValueError("independent_group_ids must have shape (n_paths,)")
    if independent_group_ids.dtype.kind not in "iu":
        raise TypeError("independent_group_ids must have an integer dtype")

    _, inverse, counts = np.unique(
        independent_group_ids,
        return_inverse=True,
        return_counts=True,
    )
    if counts.size < 2:
        raise ValueError("independent_group_ids must define at least two groups")
    if np.any(counts != counts[0]):
        raise ValueError("independent groups must all have the same positive size")
    return _Grouping(
        inverse=np.asarray(inverse, dtype=np.int64),
        n_groups=int(counts.size),
        group_size=int(counts[0]),
        explicit=True,
    )


def _stable_ess(nonnegative_values: FloatArray, name: str) -> float:
    """Return Kish ESS after normalizing non-negative finite values."""
    if nonnegative_values.ndim != 1 or nonnegative_values.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(nonnegative_values)) or np.any(nonnegative_values < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    total = float(np.sum(nonnegative_values))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} cannot contain all-zero weights")
    probabilities = nonnegative_values / total
    squared_sum = float(np.dot(probabilities, probabilities))
    if not math.isfinite(squared_sum) or squared_sum <= 0.0:
        raise FloatingPointError(f"could not compute a finite {name} ESS")
    return 1.0 / squared_sum


def _resolve_weights(paths: ModelPaths) -> _Weights:
    """Shift raw ``log(d target / d sampling)`` ratios without losing raw scale."""
    log_ratios = paths.log_likelihood_ratios
    if log_ratios is None:  # pragma: no cover - guarded by estimator selection
        raise ValueError("likelihood ratios are required for this estimator")
    if not isinstance(log_ratios, np.ndarray):
        raise TypeError("log_likelihood_ratios must be a NumPy array")
    if log_ratios.ndim != 1 or log_ratios.shape != (paths.assets.shape[0],):
        raise ValueError("log_likelihood_ratios must have shape (n_paths,)")
    if log_ratios.dtype.kind not in "iuf":
        raise TypeError("log_likelihood_ratios must have a real numeric dtype")
    if np.any(np.isnan(log_ratios)) or np.any(np.isposinf(log_ratios)):
        raise ValueError("log_likelihood_ratios cannot contain NaN or positive infinity")

    log_values = np.asarray(log_ratios, dtype=np.float64)
    log_scale = float(np.max(log_values))
    if log_scale == -math.inf:
        raise ValueError("likelihood ratios cannot all be zero")
    with np.errstate(under="ignore"):
        scaled = np.exp(log_values - log_scale)
    scaled = np.asarray(scaled, dtype=np.float64)
    scaled_total = float(np.sum(scaled))
    if not math.isfinite(scaled_total) or scaled_total <= 0.0:
        raise FloatingPointError("could not normalize likelihood ratios")
    mean_scaled = scaled_total / scaled.size
    log_mean = log_scale + math.log(mean_scaled)
    try:
        mean = math.exp(log_mean)
    except OverflowError:
        mean = math.inf
    path_ess = _stable_ess(scaled, "path likelihood")
    return _Weights(
        log_ratios=log_values,
        scaled=scaled,
        log_scale=log_scale,
        mean=mean,
        log_mean=log_mean,
        path_ess=path_ess,
    )


def _stable_mean(values: FloatArray, name: str) -> float:
    """Return an overflow-resistant arithmetic mean of finite values."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    try:
        result = math.fsum(float(value) / values.size for value in values)
    except OverflowError as exc:
        raise FloatingPointError(f"{name} mean is not representable") from exc
    if not math.isfinite(result):
        raise FloatingPointError(f"{name} mean is not representable")
    return result


def _scaled_signed_weighted_values(
    values: FloatArray,
    weights: _Weights,
    name: str,
) -> _ScaledSignedValues:
    """Scale signed ``weight * value`` terms by their largest joint log magnitude."""
    if values.ndim != 1 or values.shape != weights.log_ratios.shape:
        raise ValueError(f"{name} must have shape (n_paths,)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    active = (values != 0.0) & np.isfinite(weights.log_ratios)
    scaled = np.zeros(values.shape, dtype=np.float64)
    if not np.any(active):
        return _ScaledSignedValues(scaled=scaled, log_scale=weights.log_scale)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        joint_logs = weights.log_ratios[active] + np.log(np.abs(values[active]))
    if not np.all(np.isfinite(joint_logs)):
        raise FloatingPointError(f"{name} weighted log magnitudes are not representable")
    log_scale = float(np.max(joint_logs))
    with np.errstate(under="ignore"):
        magnitudes = np.exp(joint_logs - log_scale)
    scaled[active] = np.copysign(magnitudes, values[active])
    if not np.all(np.isfinite(scaled)):
        raise FloatingPointError(f"{name} weighted contributions are not representable")
    return _ScaledSignedValues(scaled=scaled, log_scale=log_scale)


def _restore_log_scale(scaled_value: float, log_scale: float, name: str) -> float:
    """Restore one log-scaled scalar and reject float overflow or underflow."""
    if scaled_value == 0.0:
        return 0.0
    log_absolute_value = math.log(abs(scaled_value)) + log_scale
    if not math.isfinite(log_absolute_value) or log_absolute_value > _LOG_MAX_FLOAT:
        raise FloatingPointError(f"{name} is not representable")
    if log_absolute_value < _LOG_MIN_SUBNORMAL:
        raise FloatingPointError(f"{name} is not representable")
    restored = math.copysign(math.exp(log_absolute_value), scaled_value)
    if not math.isfinite(restored) or restored == 0.0:
        raise FloatingPointError(f"{name} is not representable")
    return restored


def _ratio_from_scaled_means(
    *,
    numerator: float,
    numerator_log_scale: float,
    denominator: float,
    denominator_log_scale: float,
    name: str,
) -> tuple[float, float, float]:
    """Restore a ratio whose numerator and denominator have separate log scales."""
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"{name} denominator must be positive")
    scaled_ratio = numerator / denominator
    if not math.isfinite(scaled_ratio):
        raise FloatingPointError(f"{name} scaled ratio is not representable")
    relative_log_scale = numerator_log_scale - denominator_log_scale
    value = _restore_log_scale(scaled_ratio, relative_log_scale, name)
    return value, scaled_ratio, relative_log_scale


def _weighted_ratio(values: FloatArray, weights: _Weights, name: str) -> float:
    """Return ``sum(weight * value) / sum(weight)`` with separate stable scales."""
    numerator = _scaled_signed_weighted_values(values, weights, name)
    numerator_mean = _stable_mean(numerator.scaled, f"{name} numerator")
    denominator_mean = _stable_mean(weights.scaled, f"{name} denominator")
    value, _, _ = _ratio_from_scaled_means(
        numerator=numerator_mean,
        numerator_log_scale=numerator.log_scale,
        denominator=denominator_mean,
        denominator_log_scale=weights.log_scale,
        name=name,
    )
    return value


def _select_estimator(paths: ModelPaths, *, self_normalized: bool) -> PathEstimator:
    """Enforce the declared sampling/target-measure relationship."""
    if paths.numeraire != "money_market_account":
        raise ValueError("path valuation requires the money_market_account numeraire")

    same_measure = paths.sampling_measure == paths.target_measure
    has_ratios = paths.log_likelihood_ratios is not None
    if paths.target_measure == "P":
        raise ValueError("the physical measure P is not a valid pricing target")

    if self_normalized:
        if same_measure or not has_ratios:
            raise ValueError(
                "self-normalized valuation requires distinct measures and likelihood ratios"
            )
        return PathEstimator.SELF_NORMALIZED_LIKELIHOOD_RATIO

    if same_measure:
        if has_ratios:
            raise ValueError(
                "likelihood ratios are invalid when sampling and target measures agree"
            )
        return PathEstimator.MONTE_CARLO
    if not has_ratios:
        raise ValueError("a measure change requires raw likelihood ratios")
    return PathEstimator.RAW_LIKELIHOOD_RATIO


def _evaluate_payoff(
    *,
    paths: ModelPaths,
    payoff: Payoff,
    recenter: LegacyAdditiveForwardRecenter | None,
    weights: _Weights | None,
) -> tuple[FloatArray, float | None]:
    """Evaluate raw or explicitly recentered payoff values."""
    recenter_shift: float | None = None
    if recenter is None:
        values = payoff(paths)
    else:
        if not isinstance(recenter, LegacyAdditiveForwardRecenter):
            raise TypeError("recenter must be a LegacyAdditiveForwardRecenter policy")
        if not isinstance(payoff, EuropeanOptionPayoff):
            raise TypeError("legacy recentering is supported only for EuropeanOptionPayoff")

        asset_index = paths.asset_ids.index(payoff.asset_id)
        time_matches = np.flatnonzero(paths.observation_times == payoff.expiry)
        if time_matches.size != 1:  # pragma: no cover - shared contract guards this
            raise ValueError("payoff expiry must exactly match one path observation time")
        terminal_asset = np.asarray(
            paths.assets[:, int(time_matches[0]), asset_index],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(terminal_asset)):
            raise ValueError("selected terminal asset values must be finite")

        if recenter.mean is RecenterMean.UNWEIGHTED:
            sample_mean = _stable_mean(terminal_asset, "recenter terminal asset")
        else:
            if weights is None:
                raise ValueError("weighted legacy recentering requires likelihood ratios")
            sample_mean = _weighted_ratio(
                terminal_asset,
                weights,
                "weighted recenter sample mean",
            )
        if not math.isfinite(sample_mean):
            raise FloatingPointError("could not compute a finite recenter sample mean")

        recenter_shift = recenter.target_forward - sample_mean
        with np.errstate(over="ignore", invalid="ignore"):
            adjusted_asset = terminal_asset + recenter_shift
        if not np.all(np.isfinite(adjusted_asset)) or np.any(adjusted_asset <= 0.0):
            raise ValueError("legacy-recentered terminal asset values must be finite and positive")
        values = payoff._payoff_from_asset_values(adjusted_asset, n_paths=paths.assets.shape[0])

    if not isinstance(values, np.ndarray):
        raise TypeError("payoff output must be a NumPy array")
    if values.ndim != 1 or values.shape != (paths.assets.shape[0],):
        raise ValueError("payoff output must have shape (n_paths,)")
    if values.dtype.kind not in "iuf":
        raise TypeError("payoff output must have a real numeric dtype")
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("payoff output must contain only finite values")
    return result, recenter_shift


def _group_means(values: FloatArray, grouping: _Grouping) -> FloatArray:
    """Return equal-size group means while reducing avoidable sum overflow."""
    order = np.argsort(grouping.inverse, kind="stable")
    arranged = values[order].reshape(grouping.n_groups, grouping.group_size)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.sum(arranged / grouping.group_size, axis=1)
    means = np.asarray(means, dtype=np.float64)
    if not np.all(np.isfinite(means)):
        raise FloatingPointError("independent-group contributions must be finite")
    return means


def _mean_and_standard_error(values: FloatArray) -> tuple[float, float]:
    """Return a stable mean and sample-standard-deviation standard error."""
    n_values = values.size
    if n_values < 2:  # pragma: no cover - grouping validation guards this
        raise ValueError("at least two independent groups are required")
    mean = _stable_mean(values, "estimator contributions")

    scale = max(float(np.max(np.abs(values))), abs(mean))
    if scale == 0.0:
        return mean, 0.0
    normalized_mean = mean / scale
    squared_deviations = math.fsum(
        (float(value) / scale - normalized_mean) ** 2 for value in values
    )
    standard_error = scale * math.sqrt(squared_deviations / (n_values - 1) / n_values)
    if not math.isfinite(standard_error):
        raise FloatingPointError("estimator standard error is not representable")
    return mean, standard_error


def _standard_error_basis(grouping: _Grouping, *, legacy: bool) -> str:
    """Describe the declared independent sampling units."""
    if legacy:
        return (
            "legacy_conditional_independent_groups"
            if grouping.explicit
            else "legacy_conditional_iid_paths"
        )
    return "independent_groups" if grouping.explicit else "iid_paths"


def _value_paths(
    *,
    paths: ModelPaths,
    payoff: Payoff,
    discount_factor: float,
    recenter: LegacyAdditiveForwardRecenter | None,
    independent_group_ids: np.ndarray | None,
    self_normalized: bool,
) -> PathValuationResult:
    """Implement raw and self-normalized public entry points."""
    if not isinstance(paths, ModelPaths):
        raise TypeError("paths must be a ModelPaths instance")
    settlement_unit = _validate_payoff_contract(paths, payoff)
    discount = _finite_positive_float(discount_factor, "discount_factor")
    estimator = _select_estimator(paths, self_normalized=self_normalized)
    if (
        isinstance(recenter, LegacyAdditiveForwardRecenter)
        and recenter.mean is RecenterMean.NORMALIZED_LIKELIHOOD_WEIGHTED
        and estimator is not PathEstimator.SELF_NORMALIZED_LIKELIHOOD_RATIO
    ):
        raise ValueError(
            "normalized-likelihood-weighted recentering is supported only by the "
            "self-normalized valuation entry point"
        )
    grouping = _resolve_grouping(
        independent_group_ids,
        n_paths=paths.assets.shape[0],
    )

    weights = None if estimator is PathEstimator.MONTE_CARLO else _resolve_weights(paths)
    payoff_values, recenter_shift = _evaluate_payoff(
        paths=paths,
        payoff=payoff,
        recenter=recenter,
        weights=weights,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        discounted_payoff = discount * payoff_values
    if not np.all(np.isfinite(discounted_payoff)):
        raise FloatingPointError("discounted payoff values must be representable")

    n_paths = paths.assets.shape[0]
    if estimator is PathEstimator.MONTE_CARLO:
        group_contributions = _group_means(discounted_payoff, grouping)
        value, standard_error = _mean_and_standard_error(group_contributions)
        path_ess = float(n_paths)
        group_ess = float(grouping.n_groups)
        mean_ratio = None
        log_mean_ratio = None
    else:
        if weights is None:  # pragma: no cover - estimator selection guards this
            raise RuntimeError("missing likelihood weights")
        group_weights = _group_means(weights.scaled, grouping)
        group_ess = _stable_ess(group_weights, "group likelihood")
        path_ess = weights.path_ess
        mean_ratio = weights.mean
        log_mean_ratio = weights.log_mean

        weighted_payoff = _scaled_signed_weighted_values(
            discounted_payoff,
            weights,
            "discounted payoff",
        )
        group_numerators = _group_means(weighted_payoff.scaled, grouping)

        if estimator is PathEstimator.RAW_LIKELIHOOD_RATIO:
            scaled_value, scaled_standard_error = _mean_and_standard_error(group_numerators)
            value = _restore_log_scale(
                scaled_value,
                weighted_payoff.log_scale,
                "raw likelihood-ratio value",
            )
            standard_error = _restore_log_scale(
                scaled_standard_error,
                weighted_payoff.log_scale,
                "raw likelihood-ratio standard error",
            )
        else:
            positive_groups = np.unique(grouping.inverse[np.isfinite(weights.log_ratios)]).size
            if positive_groups < 2:
                raise ValueError(
                    "self-normalized valuation requires at least two positive independent "
                    "groups under the likelihood weights"
                )
            mean_numerator, _ = _mean_and_standard_error(group_numerators)
            mean_denominator, _ = _mean_and_standard_error(group_weights)
            value, scaled_ratio, relative_log_scale = _ratio_from_scaled_means(
                numerator=mean_numerator,
                numerator_log_scale=weighted_payoff.log_scale,
                denominator=mean_denominator,
                denominator_log_scale=weights.log_scale,
                name="self-normalized estimator value",
            )
            with np.errstate(over="ignore", invalid="ignore"):
                residuals = group_numerators - scaled_ratio * group_weights
            if not np.all(np.isfinite(residuals)):
                raise FloatingPointError(
                    "self-normalized estimator residuals are not representable"
                )
            _, residual_standard_error = _mean_and_standard_error(residuals)
            scaled_standard_error = residual_standard_error / mean_denominator
            standard_error = _restore_log_scale(
                scaled_standard_error,
                relative_log_scale,
                "self-normalized estimator standard error",
            )

    return PathValuationResult(
        value=value,
        standard_error=standard_error,
        estimator=estimator,
        standard_error_basis=_standard_error_basis(
            grouping,
            legacy=recenter is not None,
        ),
        n_paths=n_paths,
        n_independent_groups=grouping.n_groups,
        group_size=grouping.group_size,
        settlement_unit=settlement_unit,
        path_effective_sample_size=path_ess,
        path_ess_fraction=path_ess / n_paths,
        group_effective_sample_size=group_ess,
        group_ess_fraction=group_ess / grouping.n_groups,
        mean_likelihood_ratio=mean_ratio,
        log_mean_likelihood_ratio=log_mean_ratio,
        recenter_shift=recenter_shift,
    )


def value_paths(
    *,
    paths: ModelPaths,
    payoff: Payoff,
    discount_factor: float = 1.0,
    recenter: LegacyAdditiveForwardRecenter | None = None,
    independent_group_ids: np.ndarray | None = None,
) -> PathValuationResult:
    """Value a payoff by ordinary MC or raw likelihood-ratio MC.

    The estimator is selected from the measures and raw log likelihood ratios declared
    by ``paths``. Each ratio is interpreted as
    ``log(d target_measure / d sampling_measure)``. Likelihood weights are never normalized
    by this entry point.

    Parameters
    ----------
    paths
        Simulated paths with explicit sampling, target-measure, and numeraire metadata.
    payoff
        Pathwise payoff returning one finite value per path.
    discount_factor
        Positive deterministic factor from payoff expiry to valuation time.
    recenter
        Optional explicit legacy additive forward-mean policy.
    independent_group_ids
        Optional equal-size grouping of dependent rows, such as antithetic pairs.

    Returns
    -------
    PathValuationResult
        Discounted value, independent-group standard error, and ESS diagnostics.
    """
    return _value_paths(
        paths=paths,
        payoff=payoff,
        discount_factor=discount_factor,
        recenter=recenter,
        independent_group_ids=independent_group_ids,
        self_normalized=False,
    )


def value_paths_self_normalized(
    *,
    paths: ModelPaths,
    payoff: Payoff,
    discount_factor: float = 1.0,
    recenter: LegacyAdditiveForwardRecenter | None = None,
    independent_group_ids: np.ndarray | None = None,
) -> PathValuationResult:
    """Value a payoff by a self-normalized likelihood-ratio estimator.

    This separately named ratio estimator requires distinct sampling and target
    measures plus raw ``log(d target_measure / d sampling_measure)`` ratios. It does not
    change the semantics of :func:`value_paths`.

    Parameters
    ----------
    paths
        Simulated paths with explicit measure-change likelihood ratios.
    payoff
        Pathwise payoff returning one finite value per path.
    discount_factor
        Positive deterministic factor from payoff expiry to valuation time.
    recenter
        Optional explicit legacy additive forward-mean policy.
    independent_group_ids
        Optional equal-size grouping of dependent rows, such as antithetic pairs.

    Returns
    -------
    PathValuationResult
        Ratio value, independent-group ratio standard error, and ESS diagnostics.
    """
    return _value_paths(
        paths=paths,
        payoff=payoff,
        discount_factor=discount_factor,
        recenter=recenter,
        independent_group_ids=independent_group_ids,
        self_normalized=True,
    )
