"""Point-in-time SPX ERP prediction tests with delayed entry and overlapping outcomes.

Primary tilt is fixed at three before testing. Total returns include dividends.
HAC and block-bootstrap diagnostics are exploratory for this short annual-horizon history.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from papers.risk_premia_gmm.replication import spx_oca_longest as chain

HORIZONS = ("listed_expiry", "fixed365")
TARGETS = ("annual_total_return", "annual_excess_return")
SEED = 20261119


class Locals(Enum):
    """Available analysis operations."""

    ANALYZE = "analyze"


def local_date(value):
    """Convert an exchange timestamp to its New York calendar date."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York").tz_localize(None)
    return stamp.normalize()


def build_targets(fits, returns):
    """Match each known signal to next-session entry and uncensored future total returns.

    The listed-expiry target uses that actual expiry; the fixed horizon ends on
    the first recorded trading date on/after entry+365 calendar days. No returns
    are forward-filled or extrapolated. Rate is known from the option snapshot.
    """
    index = pd.DatetimeIndex(returns.index).normalize()
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Return index dates must be unique and increasing")
    tr = returns.spx_total_return.to_numpy(float)
    if not np.isfinite(tr).all() or (tr <= 0).any():
        raise ValueError("Total-return levels must be finite and positive")
    records = []
    for _, row in fits.iterrows():
        date = local_date(row.date)
        entry_pos = int(index.searchsorted(date, side="right"))
        if entry_pos >= len(index) or (index[entry_pos] - date).days > 7:
            continue
        entry = index[entry_pos]
        expiry = local_date(row.expiry)
        for horizon in HORIZONS:
            planned = expiry if horizon == "listed_expiry" else entry + pd.Timedelta(days=365)
            if planned <= entry:
                continue
            end_pos = int(index.searchsorted(planned, side="left"))
            matured = end_pos < len(index)
            if matured and (index[end_pos] - planned).days > 7:
                raise ValueError("Unexpected missing total-return endpoint")
            end = index[end_pos] if matured else planned
            years = (end - entry).days / 365.0
            cash = np.expm1(float(row.rate) * years) / years
            gross = tr[end_pos] / tr[entry_pos] if matured else np.nan
            total = (gross - 1) / years
            records.append(
                {
                    **row.to_dict(),
                    "date": date,
                    "horizon": horizon,
                    "entry_date": entry,
                    "planned_end": planned,
                    "end_date": index[end_pos] if matured else pd.NaT,
                    "matured": matured,
                    "target_years": years,
                    "holding_sessions": end_pos - entry_pos if matured else np.nan,
                    "entry_total_return_index": tr[entry_pos],
                    "end_total_return_index": tr[end_pos] if matured else np.nan,
                    "total_return": gross - 1,
                    "annual_total_return": total,
                    "annual_cash_return": cash,
                    "annual_excess_return": total - cash,
                    "calendar_position": int(index.searchsorted(date)),
                }
            )
    return pd.DataFrame(records)


def hac_regression(x, y, positions, lags):
    """OLS and Bartlett HAC on actual trading-session gaps, with missing scores zeroed.

    Cross-check against statsmodels on equally spaced observations. A centered
    predictor improves conditioning; returned coefficients use original units.
    """
    x, y, positions = np.asarray(x, float), np.asarray(y, float), np.asarray(positions, int)
    n = len(x)
    mu, sd = x.mean(), x.std()
    if n < 10 or sd < 1e-12:
        raise ValueError("Too few observations or constant predictor")
    design = np.c_[np.ones(n), (x - mu) / sd]
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    error = y - design @ coefficient
    bread = np.linalg.inv(design.T @ design)
    positions = positions - positions.min()
    if len(np.unique(positions)) != n:
        raise ValueError("Duplicate regression-session positions")
    score = np.zeros((positions.max() + 1, 2))
    score[positions] = design * error[:, None]
    meat = score.T @ score
    for lag in range(1, min(lags, len(score) - 1) + 1):
        gamma = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    covariance = bread @ meat @ bread * n / (n - 2)
    beta = coefficient[1] / sd
    se = np.sqrt(max(0.0, covariance[1, 1])) / sd
    return {
        "n": n,
        "alpha": float(coefficient[0] - beta * mu),
        "beta": float(beta),
        "se_hac": float(se),
        "t_hac": float(beta / se),
        "p_hac_normal": float(2 * norm.sf(abs(beta / se))),
        "hac_lags": lags,
        "r_squared": float(1 - np.sum(error**2) / np.sum((y - y.mean()) ** 2)),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
    }


def block_slope_interval(x, y, positions, block, replications=999, seed=SEED):
    """Circular trading-session block pairs bootstrap, retaining missing-date gaps."""
    positions = np.asarray(positions, int) - np.min(positions)
    size = int(positions.max() + 1)
    xx, yy = np.full(size, np.nan), np.full(size, np.nan)
    xx[positions], yy[positions] = x, y
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replications):
        starts = rng.integers(0, size, int(np.ceil(size / block)))
        ids = ((starts[:, None] + np.arange(block)) % size).ravel()[:size]
        a, b = xx[ids], yy[ids]
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        a = a - a.mean()
        if np.dot(a, a) > 1e-20:
            samples.append(float(np.dot(a, b - b.mean()) / np.dot(a, a)))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "block_sessions": block,
        "bootstrap_draws": len(samples),
        "beta_ci_low": float(lo),
        "beta_ci_high": float(hi),
    }


def expanding_forecasts(frame, target, min_train=756):
    """Estimate only on outcomes ending strictly before each signal date.

    Sorting by label maturity permits cumulative sufficient statistics without
    treating unfinished overlapping labels as available training observations.
    """
    data = frame.sort_values("date").copy()
    training = data[data.matured & np.isfinite(data[target])].sort_values("end_date")
    x, y = training.erp_lambda3.to_numpy(float), training[target].to_numpy(float)
    prefix = np.vstack([np.zeros(5), np.cumsum(np.c_[x, y, x * x, x * y, np.ones(len(x))], axis=0)])
    ends = training.end_date.to_numpy("datetime64[ns]")
    dates = data.date.to_numpy("datetime64[ns]")
    count = np.searchsorted(ends, dates, side="left")
    sums = prefix[count]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x, mean_y = sums[:, 0] / count, sums[:, 1] / count
        centered_xx = sums[:, 2] - count * mean_x**2
        slope = (sums[:, 3] - count * mean_x * mean_y) / centered_xx
    eligible = (count >= min_train) & (centered_xx > 1e-15)
    data["training_count"] = count
    data["mean_forecast"] = mean_y
    data["linear_forecast"] = mean_y + slope * (data.erp_lambda3.to_numpy() - mean_x)
    data["linear_slope"] = slope
    data["direct_forecast"] = data.erp_lambda3
    if target == "annual_total_return":
        data["direct_forecast"] += data.annual_cash_return
    data["target"] = target
    latest = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    positive = count > 0
    latest[positive] = ends[count[positive] - 1]
    data["latest_training_end"] = latest
    assert (latest[positive] < dates[positive]).all()
    return data[eligible].copy()


def oos_summary(frame, target):
    """Compare forecasts with the same causal expanding-mean benchmark."""
    scored = frame[frame.matured & np.isfinite(frame[target])]
    if scored.empty:
        return []
    y = scored[target].to_numpy()
    benchmark_error = scored.mean_forecast.to_numpy() - y
    out = []
    for method in ("direct", "linear"):
        error = scored[f"{method}_forecast"].to_numpy() - y
        out.append(
            {
                "method": method,
                "n": len(scored),
                "first": str(scored.date.min().date()),
                "last": str(scored.date.max().date()),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mean_benchmark_rmse": float(np.sqrt(np.mean(benchmark_error**2))),
                "oos_r_squared": float(1 - np.sum(error**2) / np.sum(benchmark_error**2)),
                "mae": float(np.mean(abs(error))),
                "mean_training_count": float(scored.training_count.mean()),
            }
        )
    return out


def block_oos_interval(frame, target, method, block, replications=999, seed=SEED):
    """Block-resample paired forecast losses, conditional on the historical forecasts."""
    scored = frame[frame.matured & np.isfinite(frame[target])].sort_values("date")
    positions = scored.calendar_position.to_numpy(int)
    positions = positions - positions.min()
    size = int(positions.max() + 1)
    losses = np.full((size, 2), np.nan)
    y = scored[target].to_numpy()
    losses[positions, 0] = (scored[f"{method}_forecast"].to_numpy() - y) ** 2
    losses[positions, 1] = (scored.mean_forecast.to_numpy() - y) ** 2
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replications):
        starts = rng.integers(0, size, int(np.ceil(size / block)))
        ids = ((starts[:, None] + np.arange(block)) % size).ravel()[:size]
        total = np.nansum(losses[ids], axis=0)
        if total[1] > 0:
            samples.append(float(1 - total[0] / total[1]))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "method": method,
        "block_sessions": block,
        "bootstrap_draws": len(samples),
        "oos_r2_ci_low": float(low),
        "oos_r2_ci_high": float(high),
    }


def nonoverlap_sample(frame):
    """Greedily retain forecasts with disjoint holding periods for a descriptive check."""
    selected = []
    last_end = pd.Timestamp.min
    for i, row in frame.sort_values("date").iterrows():
        if row.entry_date > last_end:
            selected.append(i)
            last_end = row.end_date
    return frame.loc[selected].copy()


def analyze(panel, output, replications, min_train):
    """Run prespecified full-sample tests, sensitivities and genuinely purged forecasts."""
    expected = len(chain.CASES)
    counts = panel.groupby(["date", "horizon"]).case.nunique()
    keys = counts[counts == expected].index
    matched = panel.set_index(["date", "horizon"]).loc[keys].reset_index()
    matched.to_csv(output / "matched_daily_panel.csv", index=False)
    regressions, bootstraps, sensitivities, nonoverlap = [], [], [], []
    forecast_frames, summaries, oos_intervals, influence = [], [], [], []
    for (horizon, case), frame in matched.groupby(["horizon", "case"]):
        mature = frame[frame.matured].sort_values("date")
        xx = mature.erp_lambda3.to_numpy()
        centered = xx - xx.mean()
        leverage = 1 / len(xx) + centered**2 / np.sum(centered**2)
        # Independent QR projection check: extreme leverage can invalidate naive inference.
        qr, _ = np.linalg.qr(np.c_[np.ones(len(xx)), centered / xx.std()])
        np.testing.assert_allclose(leverage, np.sum(qr**2, axis=1), atol=1e-12)
        peak = int(np.argmax(leverage))
        influence.append(
            {
                "horizon": horizon,
                "case": case,
                "n": len(xx),
                "maximum_leverage": float(leverage[peak]),
                "maximum_leverage_date": str(mature.iloc[peak].date.date()),
                "maximum_leverage_erp": float(xx[peak]),
                "share_centered_predictor_variation": float(leverage[peak] - 1 / len(xx)),
            }
        )
        for target in TARGETS:
            x, y = mature.erp_lambda3.to_numpy(), mature[target].to_numpy()
            positions = mature.calendar_position.to_numpy()
            common = {"horizon": horizon, "case": case, "target": target}
            for lag in (300, 600):
                regressions.append({**common, **hac_regression(x, y, positions, lag)})
                bootstraps.append(
                    {**common, **block_slope_interval(x, y, positions, lag, replications)}
                )
            for label, subset, lag in (
                ("cboe_period", mature[mature.provider == "cboe"], 300),
                ("thetadata_period", mature[mature.provider == "thetadata"], 300),
                ("maturity_300_400_days", mature[mature.ttm.between(300 / 365, 400 / 365)], 300),
                ("month_end", mature.groupby(mature.date.dt.to_period("M")).tail(1), 14),
            ):
                if len(subset) < 30:
                    continue
                pos = (
                    (subset.date.dt.year * 12 + subset.date.dt.month).to_numpy()
                    if label == "month_end"
                    else subset.calendar_position.to_numpy()
                )
                sensitivities.append(
                    {
                        **common,
                        "sample": label,
                        **hac_regression(subset.erp_lambda3, subset[target], pos, lag),
                    }
                )
            annual = nonoverlap_sample(mature)
            if len(annual) >= 3:
                xx, yy = annual.erp_lambda3.to_numpy(), annual[target].to_numpy()
                beta = np.polyfit(xx, yy, 1)[0]
                nonoverlap.append(
                    {
                        **common,
                        "n": len(annual),
                        "slope": beta,
                        "correlation": np.corrcoef(xx, yy)[0, 1],
                    }
                )
            forecasts = expanding_forecasts(frame, target, min_train)
            forecast_frames.append(forecasts)
            summaries.extend([{**common, **row} for row in oos_summary(forecasts, target)])
            for method in ("direct", "linear"):
                for block in (300, 600):
                    oos_intervals.append(
                        {
                            **common,
                            **block_oos_interval(forecasts, target, method, block, replications),
                        }
                    )
        print(f"predictability: {horizon}, {case}, {len(mature)} mature daily signals", flush=True)
    pd.DataFrame(influence).to_csv(output / "regression_influence.csv", index=False)
    pd.DataFrame(regressions).to_csv(output / "predictive_regressions.csv", index=False)
    pd.DataFrame(bootstraps).to_csv(output / "block_bootstrap_slopes.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(output / "regression_sensitivities.csv", index=False)
    pd.DataFrame(nonoverlap).to_csv(output / "nonoverlapping_diagnostics.csv", index=False)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    forecasts.to_csv(output / "expanding_forecasts.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "oos_summary.csv", index=False)
    pd.DataFrame(oos_intervals).to_csv(output / "oos_block_intervals.csv", index=False)
    return matched, forecasts


def figures(fits, panel, forecasts, quality, output):
    """Plot unsmoothed daily ERPs, subsequent total returns and forecast accuracy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "gmm2": "#777777",
        "gmm3": "#D55E00",
        "gmm3_moment2": "#0072B2",
        "gmm3_moment4": "#CC79A7",
    }
    labels = {
        "gmm2": "2 components",
        "gmm3": "3 components",
        "gmm3_moment2": "3 components, stronger cap",
        "gmm3_moment4": "3 components, weaker cap",
    }
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for case in chain.CASES:
        part = fits[fits.case == case].sort_values("date")
        # Reindex to all observed option dates so failed/filtered slices create gaps.
        line = part.set_index("date").erp_lambda3.reindex(pd.DatetimeIndex(quality.date))
        axes[0].plot(line.index, 100 * line, color=colors[case], linewidth=0.8, label=labels[case])
    axes[0].set_yscale("symlog", linthresh=20)
    axes[0].set(ylabel="Annual ERP (%)", title="Daily SPX ERP, tilt = 3; all extremes retained")
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].plot(quality.date, quality.days, color="#0072B2", linewidth=0.8)
    axes[1].axhline(365, color="black", linestyle="--", linewidth=0.8)
    axes[1].set(
        ylabel="Selected maturity (days)", title="Listed expiry nearest 365 days; no interpolation"
    )
    for ax in axes:
        ax.axvline(pd.Timestamp("2023-06-01"), color="#999999", linestyle=":", linewidth=1)
        ax.grid(alpha=0.2)
    fig.savefig(output / "daily_erp_time_series.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for ax, case in zip(axes, ("gmm2", "gmm3")):
        part = panel[
            (panel.horizon == "listed_expiry") & (panel.case == case) & panel.matured
        ].sort_values("date")
        ax.plot(
            part.date,
            part.annual_total_return * 100,
            color="#009E73",
            linewidth=0.85,
            label="Subsequent annualized total return",
        )
        ax.plot(
            part.date,
            100 * (part.erp_lambda3 + part.annual_cash_return),
            color=colors[case],
            linewidth=0.85,
            label="ERP + ex-ante cash return",
        )
        ax.set(title=labels[case], ylabel="Annual simple return (%)")
        ax.set_yscale("symlog", linthresh=30)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle("Signals versus subsequent SPX total returns: next-session entry to listed expiry")
    fig.savefig(output / "erp_and_subsequent_total_returns.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for method, style in (("direct", "--"), ("linear", "-")):
        for ax, case in zip(axes, ("gmm2", "gmm3")):
            part = forecasts[
                (forecasts.horizon == "listed_expiry")
                & (forecasts.case == case)
                & (forecasts.target == "annual_excess_return")
                & forecasts.matured
            ].sort_values("date")
            y = part.annual_excess_return
            gain = (part.mean_forecast - y) ** 2 - (part[f"{method}_forecast"] - y) ** 2
            ax.plot(
                part.date,
                gain.cumsum() * 1e4,
                linestyle=style,
                color=colors[case],
                label=f"{method} forecast",
            )
            ax.set(title=labels[case], ylabel="Cumulative squared-error advantage (pp squared)")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.legend()
            ax.grid(alpha=0.2)
    fig.suptitle("Out-of-sample accuracy versus expanding mean (overlapping forecasts; not P&L)")
    fig.savefig(output / "oos_forecast_accuracy.png", dpi=160)
    plt.close(fig)


def run_local(local: Locals = Locals.ANALYZE) -> None:
    """Analyze a completed daily calibration without fetching or changing its inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=999)
    parser.add_argument("--min-train", type=int, default=756)
    args = parser.parse_args()
    if local is not Locals.ANALYZE:
        return
    out = args.output
    fits = pd.read_csv(out / "daily_erp.csv", parse_dates=["date"])
    verified = json.loads((out / "numerical_validation.json").read_text())
    assert len(fits) == verified["accepted"]
    quality = pd.read_csv(out / "daily_selection_quality.csv", parse_dates=["date"])
    returns = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    panel = build_targets(fits, returns)
    panel.to_csv(out / "daily_forward_returns.csv", index=False)
    matched, forecasts = analyze(panel, out, args.bootstrap, args.min_train)
    figures(fits, matched, forecasts, quality, out)
    coverage = panel.groupby(["horizon", "case"]).agg(
        signals=("date", "count"),
        matured=("matured", "sum"),
        first_signal=("date", "min"),
        last_signal=("date", "max"),
        longest_holding_sessions=("holding_sessions", "max"),
    )
    coverage.to_csv(out / "prediction_coverage.csv")
    record = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": chain.sha256(__file__),
        "calibration_manifest_sha256": chain.sha256(out / "calibration_manifest.json"),
        "fits_sha256": chain.sha256(out / "baseline_fits.csv"),
        "daily_erp_sha256": chain.sha256(out / "daily_erp.csv"),
        "returns_path": str(args.returns.resolve()),
        "returns_sha256": chain.sha256(args.returns),
        "primary_tilt": 3,
        "hac_lags": [300, 600],
        "block_lengths": [300, 600],
        "bootstrap_replications": args.bootstrap,
        "seed": SEED,
        "entry": "First recorded trading-day close strictly after signal date",
        "primary_horizon": "Listed selected expiry; fixed365 is sensitivity",
        "training": "Expanding; only outcomes ending strictly before signal date",
        "minimum_matured_training_observations": args.min_train,
        "target_units": "Annualized simple total and excess returns; no CAGR",
        "cash": "Signal-date parity rate held flat over delayed holding interval",
        "inference": "Exploratory HAC-normal p values and circular block-pairs percentile CIs",
        "oos_intervals": "Block-resampled paired losses conditional on historical forecasts",
        "trimming": "None; all finite accepted estimates retained",
        "mature_return_cutoff": str(returns.index.max().date()),
    }
    (out / "prediction_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(coverage.to_string(), flush=True)


if __name__ == "__main__":
    run_local()
