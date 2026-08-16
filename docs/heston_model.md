# Heston stochastic volatility

## Problem and applicability

Use `HestonPricer` for the standard square-root variance process, as a primary model or a benchmark
for LogSV. It provides analytic Fourier pricing, chain implied volatilities, Monte Carlo simulation,
and constrained calibration for European options.

$$
dV_t=\kappa(\theta-V_t)dt+\vartheta\sqrt{V_t}\,dW_t^{(V)},
$$

with return/variance correlation $\rho$. Parameters are `v0`, `theta`, `kappa`, `rho`, and
`volvol` ($\vartheta$). Variance parameters are in squared annualized-volatility units.

```python
from stochvolmodels import HestonParams, HestonPricer

params = HestonParams(v0=0.04, theta=0.04, kappa=4.0, rho=-0.5, volvol=0.4)
price, ivol = HestonPricer().price_vanilla(
    params=params, ttm=0.5, forward=1.0, strike=1.0, optiontype="C"
)
```

The Feller expression $2\kappa\theta-\vartheta^2$ appears as an inequality in Heston calibration.
Non-negative values keep the variance process away from zero under the classical condition, but a
calibration result still needs economic and out-of-sample review.

## Choosing Heston versus LogSV

Choose Heston when a standard square-root variance benchmark, conventional five-parameter surface,
or comparison with a broader library matters. Choose LogSV when the log-normal volatility and
quadratic-drift parameterization or its associated paper workflow is the object of study. Both
share `OptionChain` and high-level pricer methods, which makes price/volatility comparisons direct.

Failure modes include invalid chain data and explicit `CalibrationError` on unsuccessful or
out-of-bounds optimization. Barrier/path-dependent products, term-structured Heston parameters,
and PDE engines are outside the stable package workflow.
