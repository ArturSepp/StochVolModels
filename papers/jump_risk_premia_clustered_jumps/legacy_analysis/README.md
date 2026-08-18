# Legacy exploratory analyses

These scripts preserve the exploratory cryptocurrency analyses that supported development of the
clustered-jump research project. They are research provenance, not a maintained replication suite
and not part of the `stochvolmodels` package or wheel.

| Script | Scope |
|---|---|
| `funding_and_volatility_figures.py` | Funding-rate, implied-volatility, skew, and realized-volatility figures. |
| `delta_analysis.py` | Black--Scholes/Bachelier delta comparisons and historical crypto-chain diagnostics. |
| `implied_vs_hawkes_volatility.py` | Hawkes realized-volatility forecasts compared with option-implied volatility. |
| `intraday_volatility_analysis.py` | Intraday implied-volatility responses to spot returns. |
| `perp_vs_futures.py` | Perpetual funding rates compared with dated-futures carry. |
| `realized_volatility_analysis.py` | Realized/implied volatility and skew time-series studies. |
| `realized_volatility_models.py` | EWMA and Hawkes realized-volatility model prototypes. |

The scripts retain imports from the former private `sigma_strats` research environment because
the corresponding data containers and loaders do not map one-for-one to the current
`option-chain-analytics` API. Their model calculations are intentionally preserved. Migrating a
script requires an explicit local-data schema decision and numerical validation; it should not be
treated as a mechanical import rename.

All new outputs must resolve through `papers.local_path`. The one previously hardcoded figure path
has been replaced accordingly. Local source data remains untracked and should be configured via
`papers/settings.yaml` as described in the parent README.
