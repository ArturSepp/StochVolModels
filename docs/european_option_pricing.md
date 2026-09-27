---
myst:
  html_meta:
    description: >-
      Fourier pricing of European options in stochvolmodels: the Lewis-Lipton formula for the
      capped payoff, calls and puts from the moment generating function, the integration grid and
      its accuracy against Black-Scholes, implied volatilities, and the Black-Scholes and
      Bachelier analytics.
---

# Fourier pricing of European options

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

When the moment generating function (MGF) of the log-price is known, a European option is valued
by a single Fourier integral. Following Lewis (2000) and Lipton (2001, 2002), the package values
the capped payoff $\min(S_T, K)$ and obtains calls and puts from it, as Sepp and Rakhmonov (2023,
Section 5.1) describe for the log-normal stochastic volatility (SV) model. This article derives
the formula, explains the integration grid and its accuracy, and documents the pricing interface
shared by all models, together with the Black-Scholes and Bachelier analytics used to quote
implied volatilities.

## Overview

A model pricer in stochvolmodels returns prices and implied volatilities for a single option, a
slice of strikes at one maturity, or a chain of slices. For the log-normal SV model the price is a
Fourier integral of the MGF computed by the [affine expansion](affine_expansion.md); for the Heston
model the MGF is in closed form. The integral is evaluated on a fixed grid of the transform
variable, whose size is set by the volatility and the shortest maturity. Implied volatilities are
Black-Scholes volatilities of the model prices under the chain's forwards and discount factors.

## Inputs, notation, and assumptions

| Symbol | Meaning | Code and units |
|---|---|---|
| $F$ | Forward of the underlying to the option maturity | `forward`, price units |
| $K$ | Strike | `strike`, `strikes`, same units as $F$ |
| $\tau$ | Time to maturity | `ttm`, year fraction, positive |
| $D$ | Discount factor to the maturity | `discfactor`; 1 for undiscounted values |
| $X^\ast$ | Log-moneyness $\ln(F / K)$ | |
| $E^{[m]}(\tau; \Phi)$ | Leading term of the MGF expansion of order $m$ | Eqs. (4.16) and (4.24) |
| Option code | `"C"`, `"P"`; `"IC"`, `"IP"` for [inverse options](inverse_options.md) | `OptionType` |

Volatility inputs are annualised decimals, so 20% is `0.20`; Bachelier volatility is an annualised
absolute price volatility. The forward is the expectation of the price under the valuation measure,
which requires the discounted price to be a true martingale: $\kappa_2 \ge \beta$ in the log-normal
SV model (see [martingale conditions](martingale_conditions_and_skews.md)). American exercise and
path-dependent payoffs are outside this workflow.

## Methodology

### Calls and puts from the capped payoff

With $P_T = e^{\bar{\mu}(T)} e^{X_T}$ the spot or futures price, the payoffs are, Eq. (5.2),

$$
\max(P_T - K, 0) = P_T - \min(P_T, K) , \qquad \max(K - P_T, 0) = K - \min(P_T, K) ,
$$

so both follow from the discounted value $U$ of the capped payoff $\min(P_T, K)$, Eq. (5.3). The
calls and puts on a price with forward $F$ are then, Eqs. (5.9) and (5.10),

$$
C = D F - U , \qquad P = D K - U ,
$$

and put-call parity $C - P = D (F - K)$ holds by construction.

### The capped payoff by Fourier inversion

The transform of the capped payoff, $\widehat{u}(\Phi) = -K e^{-\Phi X^\ast} / (\Phi (\Phi + 1))$,
Eq. (5.7), is finite for $-1 < \Re \Phi < 0$. Integrating it against the MGF along
$\Phi = iy - \frac{1}{2}$ gives, Proposition 5.1 and Eq. (5.4),

$$
U = \frac{D K}{\pi} \Re \int_0^\infty \frac{e^{-(iy - 1/2) X^\ast}}{y^2 + 1/4} E^{[m]}\left( \tau; \Phi = iy - \frac{1}{2} \right) dy .
$$

The contour $\Re \Phi = -\frac{1}{2}$ is the centre of the strip where the MGF exists under the
money-market-account measure (Theorem 4.2), and the weight $1 / (y^2 + 1/4)$ is bounded, so the
integrand decays with the MGF.

### The integration grid

The package evaluates the integral on $N = 1{,}000$ equally spaced points of $y \in [0, y_{max}]$,
with $y_{max} = 5.6 / s$ and the grid scale $s = \sigma_0 \sqrt{\min(\tau_{min}, 1/24)}$, where
$\tau_{min}$ is the shortest maturity of the chain. A short maturity or a low volatility widens the
grid, because the MGF decays more slowly in $y$. The weights follow Simpson's pattern
$1, 4, 2, 4, \ldots$; with an even number of points the last weight is 4 rather than 1, a historical
choice kept so that regression baselines do not move, and immaterial because the integrand has
decayed at $y_{max}$. The cost of a chain is one solve of the MGF on the grid up to the last
maturity, Eq. (6.1), and one weighted sum per option, Eq. (6.2).

### Implied volatilities

Model prices are converted to Black-Scholes implied volatilities slice by slice, with the forward and
discount factor of the chain. The inversion searches volatilities between 1% and 500% and returns
`NaN` outside that range. Far from the money, a small absolute price error is a large relative
error, so implied volatilities there are less reliable than prices.

## Worked example

Every block below is an excerpt of
[`examples/docs/european_option_pricing.py`](../examples/docs/european_option_pricing.py), which
asserts every number quoted here.

```python
def first_price() -> tuple:
    """Three-month at-the-money call of the quickstart: price and implied volatility."""
    return svm.LogSVPricer().price_vanilla(params=QUICKSTART_PARAMS, ttm=0.25, forward=1.0,
                                           strike=1.0, optiontype="C")
```

With the quickstart parameters ($\sigma_0 = \theta = 1$, $\kappa_1 = \kappa_2 = 5$, $\beta = 0.2$,
$\varepsilon = 2$) the three-month at-the-money call is worth 0.197331, an implied volatility of
99.9577%. The Fourier formula can be written out with the package's grid and slice pricer:

```python
def fourier_prices(params: svm.LogSvParams, ttm: float, forward: float, strikes: np.ndarray,
                   optiontypes: np.ndarray, discfactor: float = 1.0,
                   n_points: int = 1000) -> np.ndarray:
    """Eqs. (5.4) and (5.9) on the grid Phi = -1/2 + iy of n_points points, as in price_slice."""
    vol_scaler = set_vol_scaler(sigma0=params.sigma0, ttm=ttm)
    phi_grid = svm.get_phi_grid(max_phi=n_points, vol_scaler=vol_scaler)
    zeros = np.zeros_like(phi_grid)
    _, log_mgf = svm.compute_logsv_a_mgf_grid(ttm=ttm, phi_grid=phi_grid, psi_grid=zeros,
                                              theta_grid=zeros, **params.to_dict())
    return svm.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf, phi_grid=phi_grid,
                                                  forward=forward, strikes=strikes,
                                                  optiontypes=optiontypes, discfactor=discfactor)
```

With 1,000 points it returns the prices of `price_slice` to $10^{-14}$, and calls minus puts equal
$D(F - K)$ to $10^{-14}$ at every strike. With zero vol-of-vol ($\beta = \varepsilon = 0$) and
$\sigma_0 = \theta = 0.4$, volatility stays at 40% and the model price is the Black-Scholes price.
Over 13 strikes within three standard deviations, the largest absolute error of the Fourier price is
$1.3 \times 10^{-7}$ at one week and below $2 \times 10^{-10}$ at one month and one year. With 200
points the error is between $5 \times 10^{-3}$ and $3 \times 10^{-2}$, and with 400 points between
$5 \times 10^{-5}$ and $2 \times 10^{-3}$: the error falls by several orders of magnitude between 400
and 1,000 points. At the default grid it hardly varies across strikes and is set by the grid
spacing $\Delta y$: it is within a factor of two of $e^{-\pi / (2 \Delta y)}$, the aliasing error of
Simpson's rule near the pole of $1 / (y^2 + 1/4)$ at $y = i/2$. The one-week grid is wider, so its
spacing is larger, 0.101 against 0.069 at one month and one year, whose grids are equal because the
scale uses $\min(\tau, 1/24)$.

[![Absolute errors of Fourier prices against Black-Scholes across strikes at the default grid, and the largest error falling steeply with the number of grid points.](images/fourier_vs_bsm.png)](images/fourier_vs_bsm.png)

*Synthetic teaching exhibit. Fourier prices of the log-normal SV model with zero vol-of-vol and
$\sigma_0 = \theta = 0.4$ against Black-Scholes prices at 40% volatility. (A) Absolute error over 25
strikes within three standard deviations at the default grid of 1,000 points, floored at
$10^{-16}$ for the logarithmic axis. (B) Largest absolute error against the number of grid points.*

The Black-Scholes and Bachelier analytics price single options and invert prices to volatilities:

```python
def analytic_prices() -> dict:
    """Black-Scholes price and implied volatility, and a Bachelier price with absolute vol."""
    call = svm.compute_bsm_vanilla_price(forward=100.0, strike=105.0, ttm=0.5, vol=0.2,
                                         optiontype="C", discfactor=0.98)
    vol = svm.infer_bsm_implied_vol(forward=100.0, ttm=0.5, strike=105.0, given_price=call,
                                    discfactor=0.98, optiontype="C")
    normal = svm.compute_normal_price(forward=100.0, strike=105.0, ttm=0.5, vol=20.0,
                                      discfactor=0.98, optiontype="C")
    return {"call": call, "implied_vol": vol, "normal_call": normal}
```

The six-month call struck at 105 on a forward of 100 is worth 3.5456 at 20% volatility and a
discount factor of 0.98, and the inversion returns 20% to $10^{-12}$. With a Bachelier volatility of
20 price units, about 20% of the forward, the same call is worth 3.4211.

## Implementation in stochvolmodels

| Name | Role |
|---|---|
| `ModelPricer.price_vanilla`, `price_slice`, `price_chain` | One option, one slice, or a chain; shared by every model pricer |
| `ModelPricer.compute_chain_prices_with_vols` | Prices and Black-Scholes implied volatilities of a chain |
| `compute_bsm_vanilla_price`, `compute_bsm_vanilla_slice_prices` | Black-Scholes prices of an option or a slice |
| `compute_bsm_vanilla_delta`, `compute_bsm_vanilla_vega`, `compute_bsm_vanilla_gamma`, `compute_bsm_vanilla_theta` | Black-Scholes sensitivities |
| `compute_bsm_strike_from_delta` | Strike with a given Black-Scholes delta |
| `infer_bsm_implied_vol`, `infer_bsm_ivols_from_slice_prices` | Black-Scholes implied volatilities |
| `compute_normal_price`, `compute_normal_slice_prices` | Bachelier prices |
| `compute_normal_delta`, `compute_normal_delta_to_strike`, `compute_normal_slice_vegas` | Bachelier sensitivities and the strike of a delta |
| `infer_normal_implied_vol`, `infer_normal_ivols_from_slice_prices` | Bachelier implied volatilities |

The Black-Scholes and Bachelier functions are stable exports of stochvolmodels, implemented in the
[vanilla-option-pricers](https://pypi.org/project/vanilla-option-pricers/) dependency. The
transform-grid helpers `get_phi_grid` and `vanilla_slice_pricer_with_mgf_grid` are compatibility
exports (see [stability tiers](option_chains_and_conventions.md#stability-tiers)), and the grid
scale `set_vol_scaler` is a module function of `stochvolmodels.pricers.logsv_pricer`. For
calibration, `LogSVPricer.set_vol_scaler` fixes the grid from the chain's first at-the-money
volatility, so the grid does not move as the optimiser changes $\sigma_0$. Unknown option codes raise
`ValueError`, and the money-market-account pricer rejects the inverse codes. Run the example from a
checkout with `python examples/docs/european_option_pricing.py`.

## Interpretation and limitations

- **Quadrature is usually not the main error.** At 40% volatility the quadrature error is below
  $10^{-6}$ in price for the maturities above; the truncation of the MGF expansion is then the
  larger error, see the [affine expansion](affine_expansion.md#second-order-linear-terms). The next
  item is the exception.
- **Low volatility at short maturities.** The grid has a fixed 1,000 points over $[0, 5.6 / s]$, so
  its spacing grows as $s = \sigma_0 \sqrt{\min(\tau, 1/24)}$ falls, and with it the aliasing error.
  At 20% volatility and one week the spacing is 0.20, and the largest implied-volatility error over
  three standard deviations is 9.1 points; at 10% it is 42 points at one week and 11 points at one
  month. Passing `vol_scaler` at twice the default scale reduces the first to 0.9 points, but no
  scale removes the error, and the number of points cannot be changed through the pricers. Check
  such options against Monte Carlo.
- **The grid follows the shortest maturity.** A chain whose shortest maturity is much shorter than
  the others uses a wide grid for all of them; the cost is linear in the grid size.
- **Wings.** Implied volatilities far from the money inherit a large relative price error and are
  set to `NaN` outside 1% to 500%.
- **Scope.** European exercise only; cash settlement, day counts and quote cleaning belong to the
  data boundary described on the [conventions page](option_chains_and_conventions.md).

## See also

- [The moment generating function and its affine expansion](affine_expansion.md)
- [Inverse options and the inverse measure](inverse_options.md)
- [Expected quadratic variance, variance swaps and QV options](quadratic_variance_options.md)
- [Option chains, notation and conventions](option_chains_and_conventions.md)

## References

- Lewis, A. L. (2000). *Option Valuation under Stochastic Volatility*. Finance Press.
- Lipton, A. (2001). *Mathematical Methods for Foreign Exchange: A Financial Engineer's Approach*.
  World Scientific.
- Lipton, A. (2002). The vol smile problem. *Risk* 15(2), 61-65.
- Sepp, A. and Rakhmonov, P. (2023). Log-normal stochastic volatility model with quadratic drift.
  *International Journal of Theoretical and Applied Finance* 26(8), 2450003.
  [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
