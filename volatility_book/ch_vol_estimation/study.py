"""Reproducible daily 1/5/21-period volatility forecast study."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from stochvolmodels.estimation import (
    TRADING_HORIZONS,
    ForecastHorizon,
    OhlcEstimatorType,
    VolatilityForecastModel,
    VolForecastComparison,
    VolForecastConfig,
    VolForecastEvaluation,
    VolForecastResult,
    build_volatility_features,
    compare_volatility_forecasts,
    estimate_ohlc_variances,
    evaluate_volatility_forecast,
    walk_forward_volatility_forecast,
)
from volatility_book.ch_vol_estimation.data import (
    load_daily_ohlc_snapshot,
    make_adjusted_ohlc,
    normalize_daily_ohlc,
)

StudyKey = tuple[str, str, VolatilityForecastModel]


@dataclass(frozen=True)
class DailyStudyConfig:
    """Point-in-time conventions shared by the initial daily book study."""

    horizons: tuple[ForecastHorizon, ...] = TRADING_HORIZONS
    feature_windows: tuple[int, int, int] = (1, 5, 21)
    annualization_factor: float = 252.0
    target_estimator: OhlcEstimatorType = OhlcEstimatorType.CLOSE_TO_CLOSE
    min_train_size: int = 504
    refit_every: int = 21
    window: int | None = None
    ewma_decay: float = 0.94
    use_adjusted_ohlc: bool = True

    def __post_init__(self) -> None:
        """Validate explicit calendar, feature, and estimation controls."""
        if not isinstance(self.horizons, tuple) or not self.horizons:
            raise TypeError("horizons must be a non-empty tuple")
        if any(not isinstance(horizon, ForecastHorizon) for horizon in self.horizons):
            raise TypeError("horizons must contain ForecastHorizon values")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if (
            not isinstance(self.feature_windows, tuple)
            or len(self.feature_windows) != 3
            or self.feature_windows[0] != 1
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.feature_windows
            )
        ):
            raise ValueError("feature_windows must be three positive integers starting with 1")
        if len(set(self.feature_windows)) != 3:
            raise ValueError("feature_windows must be unique")
        if tuple(sorted(self.feature_windows)) != self.feature_windows:
            raise ValueError("feature_windows must be strictly increasing")
        if not isinstance(self.target_estimator, OhlcEstimatorType):
            raise TypeError("target_estimator must be an OhlcEstimatorType")
        if not isinstance(self.use_adjusted_ohlc, bool):
            raise TypeError("use_adjusted_ohlc must be a bool")
        if not np.isfinite(self.annualization_factor) or self.annualization_factor <= 0.0:
            raise ValueError("annualization_factor must be finite and strictly positive")
        for name, value in (
            ("min_train_size", self.min_train_size),
            ("refit_every", self.refit_every),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be strictly positive")
        if self.window is not None:
            if isinstance(self.window, bool) or not isinstance(self.window, int):
                raise TypeError("window must be an integer or None")
            if self.window < self.min_train_size:
                raise ValueError("window must be at least min_train_size")
        VolForecastConfig(ewma_decay=self.ewma_decay)


def load_study_assets(
    manifests: Mapping[str, str | Path],
) -> dict[str, pd.DataFrame]:
    """Load checksum-verified provider snapshots keyed by canonical asset."""
    assets: dict[str, pd.DataFrame] = {}
    for asset, manifest_path in manifests.items():
        if not isinstance(asset, str) or not asset:
            raise ValueError("asset names must be non-empty strings")
        ohlc, manifest = load_daily_ohlc_snapshot(manifest_path)
        if manifest.get("canonical_ticker") != asset:
            raise ValueError(f"manifest ticker does not match requested asset {asset!r}")
        assets[asset] = ohlc
    return assets


def run_daily_forecast_study(
    assets: Mapping[str, pd.DataFrame],
    config: DailyStudyConfig = DailyStudyConfig(),
) -> dict[StudyKey, VolForecastResult]:
    """Run every initial model and horizon independently for each asset."""
    if not isinstance(config, DailyStudyConfig):
        raise TypeError("config must be a DailyStudyConfig")
    results: dict[StudyKey, VolForecastResult] = {}
    for asset, ohlc in assets.items():
        if not isinstance(asset, str) or not asset:
            raise ValueError("asset names must be non-empty strings")
        variance_panel, variance_features, volatility_features = _prepare_asset_inputs(
            ohlc,
            config,
        )
        target_name = config.target_estimator.name.lower()
        target_variance = variance_panel[target_name]
        daily_window, weekly_window, monthly_window = config.feature_windows
        if daily_window != 1:
            raise ValueError("the daily HAR feature window must equal one")
        model_inputs = (
            (
                VolatilityForecastModel.EXPANDING_MEAN,
                variance_features,
                VolForecastConfig(forecast_space="variance"),
            ),
            (
                VolatilityForecastModel.PERSISTENCE,
                variance_features,
                VolForecastConfig(
                    feature_names=(f"{target_name}_current_var",),
                    forecast_space="variance",
                ),
            ),
            (
                VolatilityForecastModel.EWMA,
                variance_features,
                VolForecastConfig(
                    feature_names=(f"{target_name}_current_var",),
                    forecast_space="variance",
                    ewma_decay=config.ewma_decay,
                ),
            ),
            (
                VolatilityForecastModel.HAR,
                variance_features,
                VolForecastConfig(
                    feature_names=(
                        f"{target_name}_current_var",
                        f"{target_name}_mean{weekly_window}_var",
                        f"{target_name}_mean{monthly_window}_var",
                    ),
                    forecast_space="variance",
                ),
            ),
            (
                VolatilityForecastModel.POOLED_OHLC_NNLS,
                volatility_features,
                VolForecastConfig(feature_names=tuple(volatility_features.columns)),
            ),
        )
        for horizon in config.horizons:
            for model, features, model_config in model_inputs:
                results[(asset, horizon.label, model)] = walk_forward_volatility_forecast(
                    features=features,
                    per_period_variance=target_variance,
                    horizon=horizon,
                    model=model,
                    config=model_config,
                    annualization_factor=config.annualization_factor,
                    min_train_size=config.min_train_size,
                    refit_every=config.refit_every,
                    window=config.window,
                )
    return results


def summarize_daily_forecast_study(
    results: Mapping[StudyKey, VolForecastResult],
    benchmark_model: VolatilityForecastModel = VolatilityForecastModel.EXPANDING_MEAN,
) -> pd.DataFrame:
    """Summarize losses and common-sample gains against one benchmark model."""
    rows: list[dict[str, object]] = []
    for (asset, horizon_label, model), result in results.items():
        benchmark_key = (asset, horizon_label, benchmark_model)
        if benchmark_key not in results:
            raise ValueError(f"missing benchmark result for {benchmark_key}")
        evaluation = _optional_evaluation(result)
        comparison = _optional_comparison(result, results[benchmark_key])
        rows.append(
            {
                "asset": asset,
                "horizon": horizon_label,
                "horizon_periods": result.horizon.periods,
                "model": model.value,
                "benchmark_model": benchmark_model.value,
                "n_obs": 0 if evaluation is None else evaluation.n_obs,
                "volatility_rmse": _metric(evaluation, "volatility_rmse"),
                "volatility_mae": _metric(evaluation, "volatility_mae"),
                "volatility_bias": _metric(evaluation, "volatility_bias"),
                "variance_mse": _metric(evaluation, "variance_mse"),
                "variance_qlike": _metric(evaluation, "variance_qlike"),
                "common_n_obs": 0 if comparison is None else comparison.n_obs,
                "volatility_rmse_gain": _metric(comparison, "volatility_rmse_gain"),
                "volatility_mae_gain": _metric(comparison, "volatility_mae_gain"),
                "absolute_volatility_bias_gain": _metric(
                    comparison,
                    "absolute_volatility_bias_gain",
                ),
                "variance_mse_gain": _metric(comparison, "variance_mse_gain"),
                "variance_qlike_gain": _metric(comparison, "variance_qlike_gain"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["asset", "horizon_periods", "model"])


def write_daily_forecast_study(
    results: Mapping[StudyKey, VolForecastResult],
    *,
    output_dir: str | Path,
) -> list[Path]:
    """Write forecasts, refit diagnostics, and one summary table; never plots."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_path = output_path / "forecast_summary.csv"
    summarize_daily_forecast_study(results).to_csv(summary_path, index=False)
    written.append(summary_path)
    for (asset, horizon_label, model), result in results.items():
        stem = "__".join((_safe_name(asset), _safe_name(horizon_label), _safe_name(model.value)))
        forecast_path = output_path / f"{stem}__forecasts.csv"
        diagnostics_path = output_path / f"{stem}__refits.csv"
        result.to_frame().to_csv(forecast_path, index_label="origin")
        result.fit_diagnostics().to_csv(diagnostics_path, index=False)
        written.extend((forecast_path, diagnostics_path))
    return written


def _prepare_asset_inputs(
    ohlc: pd.DataFrame,
    config: DailyStudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct adjusted OHLC variances and both feature-space panels."""
    normalized = normalize_daily_ohlc(ohlc, provider="canonical")
    estimation_ohlc = (
        make_adjusted_ohlc(normalized)
        if config.use_adjusted_ohlc and "adjusted_close" in normalized
        else normalized
    )
    close_column = "adjusted_close" if "adjusted_close" in normalized else "close"
    adjusted_returns = np.log(normalized[close_column]).diff()
    variance_panel = estimate_ohlc_variances(estimation_ohlc)
    common_kwargs = {
        "per_period_variances": variance_panel,
        "adjusted_returns": adjusted_returns,
        "windows": config.feature_windows,
        "lags": (1,),
        "annualization_factor": config.annualization_factor,
    }
    variance_features = build_volatility_features(
        **common_kwargs,
        feature_space="variance",
    )
    volatility_features = build_volatility_features(
        **common_kwargs,
        feature_space="volatility",
    )
    return variance_panel, variance_features, volatility_features


def _optional_evaluation(result: VolForecastResult) -> VolForecastEvaluation | None:
    """Return evaluation or ``None`` when a sample has not reached its minimum."""
    try:
        return evaluate_volatility_forecast(result)
    except ValueError as error:
        if "no common complete observations" not in str(error):
            raise
        return None


def _optional_comparison(
    result: VolForecastResult,
    benchmark: VolForecastResult,
) -> VolForecastComparison | None:
    """Return comparison or ``None`` when no common forecast sample exists."""
    try:
        return compare_volatility_forecasts(result, benchmark)
    except ValueError as error:
        if "no common complete observations" not in str(error):
            raise
        return None


def _metric(
    result: VolForecastEvaluation | VolForecastComparison | None,
    name: str,
) -> float:
    """Extract one optional metric as a floating-point value."""
    return np.nan if result is None else float(getattr(result, name))


def _safe_name(value: str) -> str:
    """Return a stable output identifier."""
    safe_value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not safe_value:
        raise ValueError("output identifiers must contain a letter or number")
    return safe_value


__all__ = [
    "DailyStudyConfig",
    "StudyKey",
    "load_study_assets",
    "run_daily_forecast_study",
    "summarize_daily_forecast_study",
    "write_daily_forecast_study",
]
