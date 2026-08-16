# Analytic versus Monte Carlo validation

## Problem and applicability

Use Monte Carlo as a second numerical route for selected European prices produced by the analytic
Fourier/MGF path. Agreement supports an implementation under the same model and conventions; it
does not prove that the model fits or predicts a market.

```python
analytic_prices = pricer.price_chain(option_chain=chain, params=params)
mc_prices, mc_standard_errors = pricer.model_mc_price_chain(
    option_chain=chain,
    params=params,
    nb_path=100_000,
)
```

This fragment assumes the `pricer`, `chain`, and matching parameter object created in the model
guides. Compare each analytic price with the MC estimate and its standard error, for example using
$|P_{analytic}-P_{MC}|\leq 3\,SE_{MC}$ as a diagnostic rather than a universal acceptance rule.

## Reproducibility contract

Record package, Python, NumPy, Numba, platform, path count, time-step rule, option chain, parameters,
measure, and variable type. The high-level standard Heston/LogSV MC wrappers do not currently
expose a stable seed argument; repeated estimates must therefore be interpreted statistically.
The rough-LogSV fixed-random path accepts explicit pre-generated arrays and is regression-tested,
but remains experimental.

Calibration with `CalibrationEngine.MC` or `ROUGH_MC` is different: it generates local fixed random
inputs from the supplied seed once and reuses them through optimization. This makes the objective
deterministic within that fit and avoids mutating NumPy's global RNG state.

## Interpretation and failure modes

Increase paths to reduce sampling error approximately at the usual square-root rate; refine time
steps to assess discretization bias separately. Deep out-of-the-money options can have near-zero
prices and unstable implied-vol inversion even when price agreement is adequate. Check prices,
confidence intervals, martingale diagnostics, and implied vols rather than only one metric.

Do not tune the analytic result to one MC draw, treat confidence bounds as model uncertainty, or
use this small validation workflow as a substitute for full paper replication.
