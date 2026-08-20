# Examples

These scripts are repository-only workflows. They are intentionally outside the
`stochvolmodels` import package and are not included in wheels or source distributions.

Install the project first:

```bash
python -m pip install -e ".[dev]"
```

The core examples use packaged or generated sample data and make no network request. The
`options_time_series_data/` lane reads an explicitly installed local OCA cache. Most examples open
Matplotlib windows; use a non-interactive backend in automation. Runtime depends on the selected
`LocalTests` enum case and includes Numba compilation on first use.

## Lanes

| Directory | Purpose | Typical runtime | CI status |
|---|---|---:|---|
| `getting_started/` | Small legacy LogSV demonstration; superseded by the A04 quickstart when added | seconds to a few minutes | import-smoke only |
| `options_time_series_data/` | ATM-volatility and skew experiments using local OCA-normalized CBOE data | seconds to minutes | local-data only |
| `pricing/` | BSM, zero-DTE theta, transform, and Heston pricing demonstrations | seconds to a few minutes | import-smoke only |
| `calibration/` | Full LogSV pricing and calibration cases | minutes for optimization cases | non-gating |
| `monte_carlo/` | Quadratic-variation analytic/Monte Carlo workflows | minutes for simulation cases | non-gating |
| `advanced/` | Hawkes and rough-kernel research/contributor workflows | case-dependent | non-gating |

The core project dependencies cover the model examples. The CBOE time-series lane additionally
needs the `cboe` extra from `option-chain-analytics>=4.0.0` and local normalized SPX/VIX caches;
its directory README gives the setup. The optional `research` extra installs base OCA and is also
used by paper-replication code under `papers/`.

## Conventions

- Choose a case in each script's `LocalTests` enum and run the file directly.
- Stable user-facing examples should import the package-root public API.
- Files that import `stochvolmodels.pricers` or `stochvolmodels.utils` internals are advanced or
  contributor references, not public-API guarantees.
- Do not commit generated figures, calibration output, caches, or local paths.
- The canonical deterministic first-success command will be
  `examples/getting_started/quickstart.py` after A04; README documentation will include that file
  rather than duplicate it.

Example:

```bash
python examples/pricing/run_heston.py
```
