"""Compare EWMA filters driven by alternative daily OHLC variance estimates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from stochvolmodels.estimation import (
    TRADING_HORIZONS,
    ForecastHorizon,
    estimate_ohlc_variances,
    make_forward_variance_target,
)
from volatility_book.ch_vol_estimation.data import make_adjusted_ohlc, normalize_daily_ohlc
from volatility_book.ch_vol_estimation.study import load_study_assets

CHAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = CHAPTER_DIR / "resources" / "spy__bloomberg.manifest.json"
DEFAULT_OUTPUT_DIR = CHAPTER_DIR / "outputs"

EWMA_SIGNAL_ORDER = (
    "close_to_close",
    "parkinson",
    "garman_klass",
    "rogers_satchell",
    "overnight_rogers_satchell",
    "yang_zhang",
)


@dataclass(frozen=True)
class EwmaOhlcComparisonConfig:
    """Conventions for the common-target EWMA signal comparison."""

    horizons: tuple[ForecastHorizon, ...] = TRADING_HORIZONS
    annualization_factor: float = 252.0
    ewma_decay: float = 0.94
    min_history: int = 504
    yang_zhang_window: int = 21
    variance_floor: float = 1.0e-12
    use_adjusted_ohlc: bool = True

    def __post_init__(self) -> None:
        """Validate all numerical and calendar conventions."""
        if not isinstance(self.horizons, tuple) or not self.horizons:
            raise TypeError("horizons must be a non-empty tuple")
        if any(not isinstance(horizon, ForecastHorizon) for horizon in self.horizons):
            raise TypeError("horizons must contain ForecastHorizon values")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if not np.isfinite(self.annualization_factor) or self.annualization_factor <= 0.0:
            raise ValueError("annualization_factor must be finite and strictly positive")
        if not np.isfinite(self.ewma_decay) or not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must be strictly between zero and one")
        if isinstance(self.min_history, bool) or not isinstance(self.min_history, int):
            raise TypeError("min_history must be an integer")
        if self.min_history < 1:
            raise ValueError("min_history must be strictly positive")
        if isinstance(self.yang_zhang_window, bool) or not isinstance(
            self.yang_zhang_window,
            int,
        ):
            raise TypeError("yang_zhang_window must be an integer")
        if self.yang_zhang_window < 2:
            raise ValueError("yang_zhang_window must be at least two")
        if not np.isfinite(self.variance_floor) or self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be finite and strictly positive")
        if not isinstance(self.use_adjusted_ohlc, bool):
            raise TypeError("use_adjusted_ohlc must be a bool")


def compare_ewma_variance_signals(
    assets: Mapping[str, pd.DataFrame],
    config: EwmaOhlcComparisonConfig = EwmaOhlcComparisonConfig(),
) -> pd.DataFrame:
    """Evaluate alternative EWMA input signals against one close-to-close target.

    At time ``t`` each filter first incorporates the variance estimate from the
    completed OHLC bar at ``t``. The resulting annualised variance is used for
    every strictly future horizon. All signals are scored on the exact same
    origins and against future close-to-close squared log returns.
    """
    if not isinstance(config, EwmaOhlcComparisonConfig):
        raise TypeError("config must be an EwmaOhlcComparisonConfig")

    rows: list[dict[str, object]] = []
    for asset, ohlc in assets.items():
        if not isinstance(asset, str) or not asset:
            raise ValueError("asset names must be non-empty strings")
        variance_panel = _prepare_variance_panel(ohlc, config)
        missing = [name for name in EWMA_SIGNAL_ORDER if name not in variance_panel]
        if missing:
            raise ValueError(f"OHLC variance panel is missing signals: {missing}")

        annualized_signals = config.annualization_factor * variance_panel.loc[
            :, list(EWMA_SIGNAL_ORDER)
        ]
        filtered_variances = annualized_signals.ewm(
            alpha=1.0 - config.ewma_decay,
            adjust=False,
        ).mean()
        close_to_close = variance_panel["close_to_close"]

        for horizon in config.horizons:
            realized_variance = make_forward_variance_target(
                close_to_close,
                horizon,
                config.annualization_factor,
            )
            rows.extend(
                _evaluate_horizon(
                    asset=asset,
                    horizon=horizon,
                    filtered_variances=filtered_variances,
                    realized_variance=realized_variance,
                    config=config,
                )
            )

    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    baseline = summary.loc[
        summary["signal"].eq("close_to_close"),
        ["asset", "horizon", "volatility_rmse", "volatility_mae", "variance_qlike"],
    ].rename(
        columns={
            "volatility_rmse": "close_to_close_rmse",
            "volatility_mae": "close_to_close_mae",
            "variance_qlike": "close_to_close_qlike",
        }
    )
    summary = summary.merge(
        baseline,
        on=["asset", "horizon"],
        how="left",
        validate="many_to_one",
    )
    for metric in ("rmse", "mae", "qlike"):
        baseline_column = f"close_to_close_{metric}"
        model_column = "variance_qlike" if metric == "qlike" else f"volatility_{metric}"
        summary[f"{metric}_improvement_vs_close_pct"] = (
            100.0
            * (summary[baseline_column] - summary[model_column])
            / summary[baseline_column]
        )
    summary["rmse_rank"] = (
        summary.groupby(["asset", "horizon"])["volatility_rmse"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return summary.sort_values(["asset", "horizon_periods", "rmse_rank", "signal"])


def run_replication(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: EwmaOhlcComparisonConfig = EwmaOhlcComparisonConfig(),
) -> pd.DataFrame:
    """Run the fixed Bloomberg SPY comparison and write its summary table."""
    assets = load_study_assets({"SPY": manifest_path})
    summary = compare_ewma_variance_signals(assets, config=config)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / "ewma_ohlc_signal_comparison.csv", index=False)
    return summary


def _prepare_variance_panel(
    ohlc: pd.DataFrame,
    config: EwmaOhlcComparisonConfig,
) -> pd.DataFrame:
    """Return consistently adjusted per-bar variance signals."""
    normalized = normalize_daily_ohlc(ohlc, provider="canonical")
    estimation_ohlc = (
        make_adjusted_ohlc(normalized)
        if config.use_adjusted_ohlc and "adjusted_close" in normalized
        else normalized
    )
    variance_panel = estimate_ohlc_variances(estimation_ohlc)
    log_open = np.log(estimation_ohlc["open"])
    log_close = np.log(estimation_ohlc["close"])
    overnight_return = log_open - log_close.shift(1)
    open_to_close_return = log_close - log_open
    rogers_satchell = variance_panel["rogers_satchell"]

    variance_panel["overnight_rogers_satchell"] = overnight_return.pow(2.0) + rogers_satchell
    variance_panel["yang_zhang"] = _yang_zhang_variance(
        overnight_return=overnight_return,
        open_to_close_return=open_to_close_return,
        rogers_satchell=rogers_satchell,
        window=config.yang_zhang_window,
    )
    return variance_panel


def _yang_zhang_variance(
    *,
    overnight_return: pd.Series,
    open_to_close_return: pd.Series,
    rogers_satchell: pd.Series,
    window: int,
) -> pd.Series:
    """Return the trailing, drift-independent Yang-Zhang variance estimate.

    The rolling window ends at the current completed bar, so every value is
    point-in-time. The result remains in per-observation variance units.
    """
    weight = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    overnight_variance = overnight_return.rolling(window, min_periods=window).var()
    open_to_close_variance = open_to_close_return.rolling(window, min_periods=window).var()
    intraday_variance = rogers_satchell.rolling(window, min_periods=window).mean()
    variance = (
        overnight_variance
        + weight * open_to_close_variance
        + (1.0 - weight) * intraday_variance
    )
    return variance.clip(lower=0.0)


def _evaluate_horizon(
    *,
    asset: str,
    horizon: ForecastHorizon,
    filtered_variances: pd.DataFrame,
    realized_variance: pd.Series,
    config: EwmaOhlcComparisonConfig,
) -> list[dict[str, object]]:
    """Calculate every signal's losses on one exact common sample."""
    position = np.arange(len(filtered_variances))
    valid = position >= config.min_history + horizon.periods
    valid &= np.isfinite(realized_variance.to_numpy(dtype=float))
    valid &= np.isfinite(filtered_variances.to_numpy(dtype=float)).all(axis=1)
    if not valid.any():
        raise ValueError(f"{asset} {horizon.label} has no common complete observations")

    realized_var = realized_variance.to_numpy(dtype=float)[valid]
    realized_vol = np.sqrt(realized_var)
    rows: list[dict[str, object]] = []
    for signal in EWMA_SIGNAL_ORDER:
        predicted_var = filtered_variances[signal].to_numpy(dtype=float)[valid]
        predicted_vol = np.sqrt(np.maximum(predicted_var, 0.0))
        volatility_error = predicted_vol - realized_vol
        variance_error = predicted_var - realized_var
        ratio = np.maximum(realized_var, config.variance_floor) / np.maximum(
            predicted_var,
            config.variance_floor,
        )
        rows.append(
            {
                "asset": asset,
                "horizon": horizon.label,
                "horizon_periods": horizon.periods,
                "signal": signal,
                "n_obs": int(valid.sum()),
                "ewma_decay": config.ewma_decay,
                "yang_zhang_window": config.yang_zhang_window,
                "volatility_rmse": float(np.sqrt(np.mean(volatility_error**2))),
                "volatility_mae": float(np.mean(np.abs(volatility_error))),
                "volatility_bias": float(np.mean(volatility_error)),
                "variance_mse": float(np.mean(variance_error**2)),
                "variance_qlike": float(np.mean(ratio - np.log(ratio) - 1.0)),
            }
        )
    return rows


class UnitTests(Enum):
    """Available EWMA signal comparisons."""

    FIXED_SPY_SAMPLE = auto()


def run_unit_test(unit_test: UnitTests) -> None:
    """Run one comparison and print its RMSE ranking."""
    if unit_test is not UnitTests.FIXED_SPY_SAMPLE:
        raise TypeError(f"unsupported unit_test={unit_test!r}")
    summary = run_replication()
    print(
        summary.loc[
            :,
            [
                "horizon",
                "signal",
                "volatility_rmse",
                "rmse_improvement_vs_close_pct",
                "volatility_bias",
                "rmse_rank",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_unit_test(UnitTests.FIXED_SPY_SAMPLE)
