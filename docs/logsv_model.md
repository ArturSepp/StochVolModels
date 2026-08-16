# Karasinski-Sepp log-normal stochastic volatility

## Problem and applicability

Use `LogSVPricer` when volatility itself should remain log-normal and mean reversion includes the
quadratic drift used by Sepp and Rakhmonov. The model supports Fourier/MGF European-option pricing,
Monte Carlo validation, calibration, inverse options, and quadratic-variance analytics.

Under the MMA measure, the volatility dynamics are

$$
d\sigma_t=(\kappa_1+\kappa_2\sigma_t)(\theta-\sigma_t)dt
 +\beta\sigma_t dW_t^{(0)}+\varepsilon\sigma_t dW_t^{(1)}.
$$

The implementation follows Eq. (3.12) and the affine expansion in the project paper,
[Sepp and Rakhmonov (2023)](https://doi.org/10.1142/S0219024924500031).

## Parameters and public workflow

- `sigma0`: initial volatility; `theta`: long-run volatility level.
- `kappa1`, `kappa2`: linear and quadratic mean-reversion coefficients.
- `beta`: volatility loading on the return Brownian motion; its sign controls leverage direction.
- `volvol`: orthogonal volatility-of-volatility, denoted $\varepsilon$ above.

```python
from stochvolmodels import LogSVPricer, LogSvParams

params = LogSvParams(
    sigma0=1.0,
    theta=1.0,
    kappa1=5.0,
    kappa2=5.0,
    beta=0.2,
    volvol=2.0,
)
price, ivol = LogSVPricer().price_vanilla(
    params=params, ttm=0.25, forward=1.0, strike=1.0, optiontype="C"
)
```

The reference result is price `0.197331` and implied volatility `0.999577`. Small last-digit
differences across supported numerical-library versions are possible.

## Constraints and interpretation

`ConstraintsType.MMA_MARTINGALE` imposes $\kappa_2\geq\beta$;
`INVERSE_MARTINGALE` imposes $\kappa_2\geq2\beta$. Moment-four variants add the implemented fourth-
moment condition. These are calibration constraints, not automatic validation of every manually
constructed parameter set.

Rough-LogSV simulation and deep `rough_logsv` utilities are experimental. Use the standard analytic
and Monte Carlo paths first. This model is not intended for American exercise, arbitrary local-vol
surfaces, or automatic market-data convention discovery.
