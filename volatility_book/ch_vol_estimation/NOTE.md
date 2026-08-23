# Chapter: volatility estimation and forecasting

## What this replication does

This chapter uses Bloomberg daily OHLC as its default input. The included local sample contains
adjusted SPY open, high, low, and close observations from 4 January 2010 through 31 December 2025.
It is fixed rather than refreshed automatically, so repeated runs use exactly the same dates and
prices.

The code estimates daily OHLC variance measures and compares five forecasting methods for the next
1, 5, and 21 trading observations. Each model is trained only on targets that were fully known at
the forecast date. The default estimation window expands through time, starts after 504 complete
training observations, and refits every 21 observations. Volatility is annualised using 252 trading
days.

Run from the StochVolModels repository root:

```shell
python -m volatility_book.ch_vol_estimation.replicate_vol_estimation
```

The script validates the data manifest and checksum before running. It writes a summary,
per-model forecasts, refit diagnostics, and `model_ranking.csv` below the chapter's local
`outputs/` directory.

## Which model is better, in plain English?

**Pooled OHLC NNLS is the best overall model in the five-asset Bloomberg study.**

The intuition is simple. A market's open, high, low, and close contain different clues about how
much prices moved. The pooled model treats the available OHLC estimators like several volatility
thermometers. It learns non-negative weights, so it can trust the useful thermometers more without
using unstable offsetting bets between them.

Average reduction in volatility forecast RMSE relative to using one long-run historical average:

| Model | 1 day | 1 week | 1 month | Plain-English description |
|---|---:|---:|---:|---|
| Pooled OHLC NNLS | 30.4% | 41.9% | 41.0% | Learns a stable blend of several OHLC signals |
| EWMA | 23.6% | 36.1% | 34.8% | Gives recent volatility more weight |
| HAR | 20.8% | 32.3% | 29.6% | Mixes daily, weekly, and monthly volatility |
| Persistence | 3.7% | -4.4% | -16.6% | Assumes the latest reading will persist |
| Expanding mean | 0.0% | 0.0% | 0.0% | Uses the historical average benchmark |

The bundled fixed SPY sample independently gives the same main conclusion:

| Model | 1 day | 1 week | 1 month |
|---|---:|---:|---:|
| Pooled OHLC NNLS | 23.1% | 32.3% | 21.7% |
| HAR | 15.0% | 24.1% | 16.6% |
| EWMA | 12.9% | 20.9% | 8.9% |
| Expanding mean | 0.0% | 0.0% | 0.0% |
| Persistence | -4.1% | -10.2% | -32.1% |

Pooled OHLC NNLS ranks first at every horizon in both exercises. HAR ranks second in the fixed SPY
sample, while EWMA has the stronger average result across the broader five-asset universe.

Positive percentages mean lower forecast error. A negative percentage means the model was worse
than the historical-average benchmark. The result says:

1. Combining OHLC information was most accurate.
2. EWMA is the best simple model and a useful benchmark when transparency matters most.
3. HAR is sensible and robust, but it did not extract as much information as the pooled OHLC fit.
4. Persistence is too naive for weekly and monthly forecasts.

This does not mean pooled OHLC NNLS must win for every asset or every future sample. Its advantage
was measured out of sample with point-in-time refits, which makes the evidence materially stronger
than an in-sample fit, but the ranking should still be monitored through time.

## Is pooled OHLC NNLS a published model?

The ingredients are published, but the exact pooled model used here does not appear to be a named
published specification. The [OHLC-Vol repository](https://github.com/vivek-v-rao/OHLC-Vol)
documents code and empirical findings, and links to the classical Parkinson, Garman-Klass,
Rogers-Satchell, and Yang-Zhang estimators. It does not identify a paper, DOI, preprint, or formal
citation for its particular level-NNLS pooling and predictor-selection recipe.

There is substantial academic precedent for forecasting with range information. For example,
[Korkusuz, Kambouroudis, and McMillan (2023)](https://doi.org/10.1016/j.frl.2023.103992)
put Parkinson, Garman-Klass, Rogers-Satchell, and Yang-Zhang signals into rolling HAR-RV-X
forecasts. [Li and Hong (2011)](https://doi.org/10.1016/j.frl.2010.12.002) develop a
range-based autoregressive volatility model. Neither is the same as the expanding-window,
non-negative pooled regression in this chapter.

The defensible description is therefore **our pooled OHLC NNLS forecaster, built from published
OHLC estimators and established range-based forecasting ideas**. The model should not be presented
as a standard literature model, nor claimed as novel without a fuller bibliographic review.

## Does EWMA improve when its daily input is an OHLC variance estimate?

The comparison is implemented in `compare_ewma_ohlc.py`. Run:

```shell
python -m volatility_book.ch_vol_estimation.compare_ewma_ohlc
```

Every variant uses the same RiskMetrics-style decay, lambda = 0.94. The alternative observations
entering the filter are squared close-to-close return, Parkinson, Garman-Klass, Rogers-Satchell,
overnight plus Rogers-Satchell, and Yang-Zhang variance. Every forecast is evaluated against the
same strictly future close-to-close target, on the same dates, after 504 observations. There is no
in-sample rescaling of the OHLC signals. This isolates whether replacing the noisy daily squared
return helps.

Overnight plus Rogers-Satchell is the single-period decomposition
`log(open[t] / close[t-1])^2 + RS[t]`. Yang-Zhang uses a trailing 21-observation window ending at
the current completed bar; applying EWMA to it is intentionally reported as a double-smoothed
sensitivity rather than a like-for-like daily signal.

On the bundled fixed SPY sample, all three OHLC inputs improve EWMA volatility RMSE relative to
the close-to-close EWMA:

| EWMA daily input | 1 day | 1 week | 1 month |
|---|---:|---:|---:|
| Parkinson | 6.86% | 2.16% | **4.38%** |
| Garman-Klass | **6.96%** | 2.50% | 4.26% |
| Rogers-Satchell | 6.94% | **2.74%** | 4.14% |
| Squared close-to-close return | 0.00% | 0.00% | 0.00% |
| Overnight + Rogers-Satchell | -0.76% | -1.39% | -1.54% |
| Yang-Zhang, trailing 21 days | -9.85% | -18.53% | -12.54% |

The five-asset Bloomberg study gives a more qualified result. The table reports the mean RMSE
improvement across SPY, QQQ, GLD, USO, and HYG:

| EWMA daily input | 1 day | 1 week | 1 month |
|---|---:|---:|---:|
| Parkinson | **6.41%** | -2.04% | -5.39% |
| Garman-Klass | 6.34% | -1.87% | -5.49% |
| Rogers-Satchell | 5.96% | -2.10% | -6.02% |
| Squared close-to-close return | 0.00% | **0.00%** | **0.00%** |
| Overnight + Rogers-Satchell | -0.73% | -0.72% | -1.34% |
| Yang-Zhang, trailing 21 days | -5.79% | -12.90% | -12.45% |

In plain English: **use the OHLC-driven EWMA for the one-day forecast; retain close-to-close EWMA
as the safer uncalibrated weekly and monthly benchmark.** At one day, every range estimator beats
close-to-close EWMA for every asset, and Parkinson wins four of five assets. At longer horizons,
OHLC EWMA still helps SPY, QQQ, and often HYG, but it materially underforecasts GLD and USO.

This is economically plausible rather than a contradiction. Parkinson, Garman-Klass, and
Rogers-Satchell extract a cleaner intraday variance signal, but they do not include the
previous-close-to-open overnight move. Their lower noise helps the one-day forecast; their lower
level can create persistent downward bias over longer horizons.

Adding the overnight squared return removes much of that downward bias, but also restores enough
noise that overnight plus Rogers-Satchell is slightly worse than close-to-close EWMA on average:
0.73% higher RMSE at one day, 0.72% at one week, and 1.34% at one month. It improves selected cases
(notably GLD at all three horizons and SPY at one week and one month), but is not the new default.
The poor Yang-Zhang result is evidence against **EWMA on top of a 21-day Yang-Zhang estimate**, not
against Yang-Zhang itself: cascading two slow filters makes the forecast react too slowly.

## Data and interpretation notes

- The bundled chapter sample is SPY-only and demonstrates the method. The percentage table above
  comes from the broader Bloomberg study of SPY, QQQ, GLD, USO, and HYG.
- Bloomberg observations are for local author replication only and must not be redistributed.
- Bloomberg prices were requested with normal cash distributions, abnormal cash distributions,
  and capital changes adjusted. The chapter does not compare data vendors.
- Model ranking uses volatility RMSE and MAE. Persistence can occasionally forecast exactly zero
  variance, making QLIKE extremely sensitive to its numerical floor; QLIKE is therefore not used
  for the plain-English ranking until that convention receives a separate sensitivity analysis.
- `ewma_ohlc_signal_comparison.csv` is the fixed-sample result produced by the replication script.
  `ewma_ohlc_signal_comparison_full_bloomberg.csv` is the local five-asset research result; only
  the bundled SPY sample is independently reproducible from this chapter directory.
- These are volatility forecasts, not return forecasts or a trading strategy. Economic value for
  volatility risk-premia strategies must be evaluated separately with option positions and costs.
