"""Replicate the volatility-estimation chapter on the fixed Bloomberg SPY sample."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import pandas as pd
from volatility_book.ch_vol_estimation.study import (
    DailyStudyConfig,
    load_study_assets,
    run_daily_forecast_study,
    summarize_daily_forecast_study,
    write_daily_forecast_study,
)

CHAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = CHAPTER_DIR / "resources" / "spy__bloomberg.manifest.json"
DEFAULT_OUTPUT_DIR = CHAPTER_DIR / "outputs"


def make_model_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank models by volatility RMSE against the expanding-mean benchmark."""
    required = {
        "asset",
        "horizon",
        "horizon_periods",
        "model",
        "n_obs",
        "volatility_rmse",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")

    baseline = summary.loc[
        summary["model"].eq("expanding_mean"),
        ["asset", "horizon", "volatility_rmse"],
    ].rename(columns={"volatility_rmse": "expanding_mean_rmse"})
    ranking = summary.merge(
        baseline,
        on=["asset", "horizon"],
        how="left",
        validate="many_to_one",
    )
    if ranking["expanding_mean_rmse"].isna().any():
        raise ValueError("every asset and horizon must have an expanding-mean benchmark")
    ranking["rmse_improvement_pct"] = (
        100.0
        * (ranking["expanding_mean_rmse"] - ranking["volatility_rmse"])
        / ranking["expanding_mean_rmse"]
    )
    ranking["rmse_rank"] = (
        ranking.groupby(["asset", "horizon"])["volatility_rmse"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    columns = [
        "asset",
        "horizon",
        "horizon_periods",
        "model",
        "n_obs",
        "volatility_rmse",
        "expanding_mean_rmse",
        "rmse_improvement_pct",
        "rmse_rank",
    ]
    return ranking.loc[:, columns].sort_values(["asset", "horizon_periods", "rmse_rank", "model"])


def run_replication(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: DailyStudyConfig = DailyStudyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the fixed-sample Bloomberg study and write forecasts plus model ranks."""
    assets = load_study_assets({"SPY": manifest_path})
    results = run_daily_forecast_study(assets, config=config)
    summary = summarize_daily_forecast_study(results)
    output_path = Path(output_dir).expanduser().resolve()
    write_daily_forecast_study(results, output_dir=output_path)
    ranking = make_model_ranking(summary)
    ranking.to_csv(output_path / "model_ranking.csv", index=False)
    return summary, ranking


class UnitTests(Enum):
    """Available chapter replications."""

    VOLATILITY_FORECASTS = auto()


def run_unit_test(unit_test: UnitTests) -> None:
    """Run one chapter replication and print its RMSE ranking."""
    if unit_test is not UnitTests.VOLATILITY_FORECASTS:
        raise TypeError(f"unsupported unit_test={unit_test!r}")
    _, ranking = run_replication()
    print(
        ranking.loc[
            :, ["horizon", "model", "volatility_rmse", "rmse_improvement_pct", "rmse_rank"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_unit_test(UnitTests.VOLATILITY_FORECASTS)
