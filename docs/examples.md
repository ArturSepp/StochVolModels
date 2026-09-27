---
myst:
  html_meta:
    description: >-
      Runnable stochvolmodels examples: documentation scripts, pricing, calibration and option-data
      workflows, and the paper replication entry points, with their data requirements and the
      pages they support.
---

# Examples and recipes

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Examples for [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This page lists every runnable script in the repository with the data it needs and the page it
supports. The scripts under
[`examples/`](https://github.com/ArturSepp/StochVolModels/tree/main/examples) are repository-only:
they are not part of the wheel. Run them from a source checkout after an editable installation,
`python -m pip install -e .`. The scripts marked "core" run offline on the core installation; the
others need the `research` extra and, where stated, a local data cache.

## Documentation scripts

Each page with Python code has one canonical script. The page shows excerpts of it, and the test
suite runs every case of it and checks the numbers the page quotes.

| Script | Page | Data |
|---|---|---|
| [`docs/option_chains_and_conventions.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/option_chains_and_conventions.py) | [Option chains, notation and conventions](option_chains_and_conventions.md) | Core |
| [`docs/logsv_model.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/logsv_model.py) | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) | Core |
| [`docs/martingale_conditions_and_skews.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/martingale_conditions_and_skews.py) | [Martingale conditions, measures and positive skews](martingale_conditions_and_skews.md) | Core |
| [`docs/volatility_distribution_and_moments.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/volatility_distribution_and_moments.py) | [Steady-state distribution, moments and expected QV](volatility_distribution_and_moments.md) | Core |
| [`docs/affine_expansion.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/affine_expansion.py) | [The moment generating function and its affine expansion](affine_expansion.md) | Core; the price-impact case takes minutes |
| [`docs/european_option_pricing.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/european_option_pricing.py) | [Fourier pricing of European options](european_option_pricing.md) | Core |
| [`docs/inverse_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/inverse_options.py) | [Inverse options and the inverse measure](inverse_options.md) | Core |
| [`docs/quadratic_variance_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/quadratic_variance_options.py) | [Expected quadratic variance, variance swaps and QV options](quadratic_variance_options.md) | Core; the QV option case takes minutes |
| [`docs/monte_carlo_simulation.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/monte_carlo_simulation.py) | [Monte Carlo simulation schemes](monte_carlo_simulation.md) | Core |
| [`docs/analytic_vs_monte_carlo.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/analytic_vs_monte_carlo.py) | [Analytic versus Monte Carlo validation](analytic_vs_monte_carlo.md) | Core, bundled Bitcoin chain |
| [`docs/calibration.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/calibration.py) | [Calibration to implied volatilities](calibration.md) | Core, bundled Bitcoin chain; the refit case takes minutes |
| [`docs/logsv_smile_fitter.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/logsv_smile_fitter.py) | [Approximate log-normal SV smile fitter](logsv_smile_fitter.md) | Core, generated chain |
| [`docs/numerical_accuracy_and_performance.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/numerical_accuracy_and_performance.py) | [Numerical accuracy and performance](numerical_accuracy_and_performance.md) | Core; prints timings |
| [`docs/app_bitcoin_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_bitcoin_options.py) | [Bitcoin options: MMA, inverse and QV valuation](app_bitcoin_options.md) | Core, bundled Bitcoin chain; the calibration case takes minutes |
| [`docs/app_positive_and_negative_skews.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_positive_and_negative_skews.py) | [One model for equity, VIX, gold and leveraged-ETF skews](app_positive_and_negative_skews.md) | Core, five bundled chains; the VIX refit takes minutes |
| [`docs/heston_model.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/heston_model.py) | [The Heston model as a benchmark](heston_model.md) | Core, bundled Bitcoin chain |
| [`docs/hawkes_jump_diffusion.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/hawkes_jump_diffusion.py) | [Jump-diffusion with clustered jumps](hawkes_jump_diffusion.md) | Core; the Monte Carlo case takes a minute |
| [`docs/terminal_distribution_models.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/terminal_distribution_models.py) | [Gaussian-mixture and Student-t smiles](terminal_distribution_models.md) | Core, bundled S&P 500 ETF chain |
| [`docs/factor_hjm_stochastic_volatility.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/factor_hjm_stochastic_volatility.py) | [Stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md) | Core, experimental module |
| [`docs/app_swaptions_and_sofr_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_swaptions_and_sofr_options.py) | [USD swaptions and SOFR futures options](app_swaptions_and_sofr_options.md) | Repository only: imports the paper modules; the fit and Monte Carlo cases take minutes |
| [`docs/app_impermanent_loss_hedging.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_impermanent_loss_hedging.py) | [Hedging impermanent loss with the LogSV MGF](app_impermanent_loss_hedging.md) | Core, model-driven |
| [`docs/app_robust_volatility_models.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_robust_volatility_models.py) | [What makes a stochastic volatility model robust](app_robust_volatility_models.md) | Core, recorded parameters, no data |
| [`getting_started/quickstart.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart.py) | [Installation and first result](getting_started.md) | Core; run on Linux, Windows and macOS in CI |

## Pricing

| Script | What it shows | Data | Related page |
|---|---|---|---|
| [`pricing/run_heston.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/run_heston.py) | Heston smiles for three correlations | Core | [Heston](heston_model.md) |
| [`pricing/run_heston_sv_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/run_heston_sv_pricer.py) | Heston pricing and calibration to the bundled Bitcoin chain | Core, compatibility names | [Heston](heston_model.md) |
| [`pricing/run_bsm_mgf_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/run_bsm_mgf_pricer.py) | Black-Scholes prices through the transform pricer, under both measures | Core, internal transform API | [Fourier pricing of European options](european_option_pricing.md) |
| [`pricing/plot_bsm_zero_dte_theta.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/plot_bsm_zero_dte_theta.py) | Time decay of zero-days-to-expiry options | Core | [Fourier pricing of European options](european_option_pricing.md) |
| [`pricing/run_pricing_options_on_qvar.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/run_pricing_options_on_qvar.py) | Options on quadratic variance under LogSV and Heston against Monte Carlo | Core, sample chain | [Analytic versus Monte Carlo](analytic_vs_monte_carlo.md) |
| [`pricing/run_qvar_analytics.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/run_qvar_analytics.py) | Quadratic-variance slice pricer against Monte Carlo | Core, internal API | [Analytic versus Monte Carlo](analytic_vs_monte_carlo.md) |
| [`pricing/run_hawkes_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricing/run_hawkes_pricer.py) | Implied volatilities of the Hawkes jump-diffusion | Core, advanced names | [API reference](api.md) |

## Calibration

| Script | What it shows | Data | Related page |
|---|---|---|---|
| [`getting_started/quick_run_lognormal_sv_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/getting_started/quick_run_lognormal_sv_pricer.py) | LogSV smile plot and calibration | Core, bundled Bitcoin chain | [Calibration](calibration.md) |
| [`calibration/run_lognormal_sv_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/run_lognormal_sv_pricer.py) | Prices, parameter sensitivities, Monte Carlo comparison, analytic and Monte Carlo calibration, rough Monte Carlo | Core, bundled Bitcoin chain; some cases take minutes | [Calibration](calibration.md) |
| [`calibration/run_logsv_smile_fitter.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/run_logsv_smile_fitter.py) | Approximate LogSV smile fit | Core, bundled generated chain | [Calibration](calibration.md) |
| [`calibration/run_oca_logsv_calibration.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/run_oca_logsv_calibration.py) | Conversion of OptionChainAnalytics data and LogSV calibration | `research` extra, generated quotes | [Calibration](calibration.md) |
| [`calibration/load_cboe_option_chain.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/load_cboe_option_chain.py) | Loading a cached SPX or VIX chain | `research` extra, local CBOE cache | [Option chains](option_chains_and_conventions.md) |
| [`calibration/run_spy_thetadata_month.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/run_spy_thetadata_month.py) | SPY smile fit and calibration over one month | `research` extra, local ThetaData cache | [Calibration](calibration.md) |

## Option data time series

| Script | What it shows | Data |
|---|---|---|
| [`options_time_series_data/plot_cboe_vol_time_series.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/options_time_series_data/plot_cboe_vol_time_series.py) | At-the-money volatility and 25-delta skew over time | `research` extra, local CBOE cache |
| [`options_time_series_data/plot_vix_1m_atm_vol.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/options_time_series_data/plot_vix_1m_atm_vol.py) | Thirty-day constant-maturity VIX at-the-money volatility | Local partitioned VIX cache |

The [examples README](https://github.com/ArturSepp/StochVolModels/blob/main/examples/README.md)
describes the four routes to an `OptionChain` (the bundled chain, OptionChainAnalytics data, and
the ThetaData and CBOE caches) and their setup. The package does not distribute provider data.

## Paper entry points

The directories under
[`papers/`](https://github.com/ArturSepp/StochVolModels/tree/main/papers) reproduce the
computations of the published papers and hold related research code. They need the `research`
extra, and several need local data. Their status and prerequisites are described in
[research papers and replication](reproducing_the_papers.md) and in the
[papers README](https://github.com/ArturSepp/StochVolModels/blob/main/papers/README.md).

| Directory | Entry point | Status |
|---|---|---|
| `logsv_model_with_quadratic_drift` | `article_figures.py` | Published paper (IJTAF 2023) |
| `sv_for_factor_hjm` | `calibration_fig_5_6_7.py`, `calibration_fig_8_9.py` | Published paper (Review of Derivatives Research 2025) |
| `volatility_models`, `il_hedging`, `inverse_options` | `article_figures.py`, `run_logsv_for_il_payoff.py`, `compare_net_delta.py` | Supporting illustrations for public papers |
| `jump_risk_premia_clustered_jumps` | `hawkes_estimator.py`, `risk_premia_calibration.py` | Development code related to a working paper |
| `risk_premia_gmm`, `t_distribution`, `forward_var`, `barriers` | see the papers README | Exploratory; no publication mapping |

## Running the examples

- Each script selects its case with a `Locals` enum and a `run_local(local=...)` dispatcher; choose
  the case at the bottom of the file. The documentation scripts run every case.
- Most examples open Matplotlib windows. Set `MPLBACKEND=Agg` in automation.
- The first call of a pricer compiles numba kernels, which takes seconds; later calls are fast.
- Examples that import `stochvolmodels.pricers` or `stochvolmodels.utils` directly use advanced or
  internal interfaces; see the [stability tiers](option_chains_and_conventions.md#stability-tiers).

## See also

- [Installation and first result](getting_started.md)
- [Option chains, notation and conventions](option_chains_and_conventions.md)
- [Research papers and replication](reproducing_the_papers.md)
