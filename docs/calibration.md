# Calibration

## Problem and applicability

Calibration minimizes model-versus-market implied-volatility residuals on an `OptionChain`.
Use it only after forwards, discount factors, option codes, expiries, and decimal volatility units
have been normalized. The optimizer cannot repair a convention mismatch.

```python
from stochvolmodels import (
    ConstraintsType,
    LogSVPricer,
    LogSvParams,
    LogsvModelCalibrationType,
)

def calibrate(market_chain):
    params0 = LogSvParams(
        sigma0=1.0, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0
    )
    return LogSVPricer().calibrate_model_params_to_chain(
        option_chain=market_chain,
        params0=params0,
        model_calibration_type=LogsvModelCalibrationType.PARAMS4,
        constraints_type=ConstraintsType.INVERSE_MARTINGALE,
    )
```

Here `market_chain` is intentionally application-provided: it must carry aligned bid/ask implied
volatilities. `PARAMS4` fits `sigma0`, `theta`, `beta`, and `volvol` while keeping `kappa1` and
`kappa2` fixed. `PARAMS5` also fits `kappa1` and derives `kappa2` through the model parameter
convention. The variance-swap mode requires meaningful per-expiry variance-swap strikes.

## Weights, engines, and reproducibility

Vega weighting is enabled by default. `is_unit_ttm_vega=True` changes the maturity normalization;
unweighted calibration is available for diagnostics, not as an automatic recommendation.
`CalibrationEngine.ANALYTIC` is the default. MC engines generate fixed random inputs once per
optimizer run, so objective comparisons are deterministic for a fixed seed.

## Convergence and validation

Optimizer failure, non-finite results, wrong vector length, or bounds violations raise
`CalibrationError`. Do not catch that exception and continue with the initial guess. Record the
chain snapshot, initial parameters, bounds, constraint type, engine, seed, and library version.
Then reprice the chain, inspect residuals by maturity/moneyness, and compare at least a small subset
with an independent numerical route.

This page does not prescribe production quote filtering, bid/ask likelihoods, regularization,
multi-start policy, or out-of-sample model governance.
