# European option pricing

## Problem and applicability

Use the stable analytic functions for Black-Scholes-Merton or Bachelier prices, or a model pricer
for stochastic-volatility prices and implied volatilities. The package covers European vanilla
calls and puts; LogSV also supports the package's inverse-option convention. American exercise and
general path dependence are outside this workflow.

## Inputs, units, and output

- `ttm` is year fraction and must be positive for model-pricer containers.
- `forward` and `strike` use the same price units and must be positive.
- `discfactor` multiplies the forward-measure expectation; use `1.0` for an undiscounted result.
- Black-Scholes and stochastic-model volatility inputs are annualized decimal volatility, so 20%
  is `0.20`. Bachelier volatility is an annualized absolute price/rate volatility.
- Option codes are `"C"`, `"P"`, `"IC"`, and `"IP"`. Prefer `OptionType` in application code.

```python
from stochvolmodels import compute_bsm_vanilla_price

price = compute_bsm_vanilla_price(
    forward=100.0,
    strike=105.0,
    ttm=0.5,
    vol=0.20,
    optiontype="C",
    discfactor=0.98,
)
```

For a stochastic-volatility slice, `price_slice` returns aligned price and implied-volatility
arrays:

```python
import numpy as np
from stochvolmodels import HestonParams, HestonPricer

prices, ivols = HestonPricer().price_slice(
    params=HestonParams(),
    ttm=0.5,
    forward=1.0,
    strikes=np.array([0.9, 1.0, 1.1]),
    optiontypes=np.array(["P", "C", "C"]),
)
```

Interpret the returned implied volatilities under the same Black forward and discount conventions
as the input chain. Compare model prices before comparing implied volatilities when diagnosing an
inversion problem.

## Failure modes and non-goals

Unknown option codes raise instead of silently selecting a payoff. Non-finite parameters,
misaligned arrays, non-positive forwards/strikes, and impossible prices should be corrected at the
data boundary. This page does not cover calibration, market quote cleaning, day-count conversion,
or exercise/cash-settlement conventions.
