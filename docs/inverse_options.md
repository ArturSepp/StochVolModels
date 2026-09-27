---
myst:
  html_meta:
    description: >-
      Inverse options, which are quoted and settled in the underlying coin, valued under the
      inverse measure of the log-normal stochastic volatility model: payoffs, the Fourier formula,
      equivalence with vanilla options, Monte Carlo checks under both measures, and the net delta
      for hedging in coin units, with stochvolmodels code.
---

# Inverse options and the inverse measure

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Options on Bitcoin and Ether on the Deribit exchange are inverse options: they are quoted and
settled in the underlying coin, so a call pays $\max(S_T - K, 0) / S_T$ coins. Their natural
valuation measure is the inverse measure, which takes the underlying as numeraire. This article
derives the valuation formula of Sepp and Rakhmonov (2023, Section 5.2), shows that the dollar value
of an inverse option equals that of the vanilla option when both measures are martingale measures,
checks both by Monte Carlo, and explains the net delta that Lucic and Sepp (2024) use to hedge in
coin units.

## Overview

A vanilla call on Bitcoin pays dollars; an inverse call pays the same amount converted into Bitcoin
at the terminal price. Valuing the inverse payoff under the money-market-account (MMA) measure
requires the joint law of the payoff and $1 / S_T$; valuing it under the inverse measure turns it
into a capped payoff whose transform is as simple as that of a vanilla option. The two valuations
agree when both measures are equivalent martingale measures, which in the log-normal stochastic
volatility (SV) model requires $\kappa_2 \ge 2 \beta$ (see
[martingale conditions](martingale_conditions_and_skews.md)).

For hedging, the coin value of an option moves with the price in two ways: through the dollar value
and through the conversion rate. The resulting sensitivity, the net delta, is smaller than the Black
delta for a call and more negative for a put.

## Inputs, notation, and assumptions

| Symbol | Meaning | Code and units |
|---|---|---|
| $F$ | Forward price of the coin in dollars | `forward`, for example 60,000 |
| $K$ | Strike in dollars | `strike` |
| $C$, $P$ | Dollar values of vanilla calls and puts | returned by the pricers |
| $c$, $p$ | Coin values of inverse calls and puts | $C / F$ and $P / F$ |
| $\tilde{\mathbb{Q}}$ | Inverse measure, with the coin as numeraire | `is_spot_measure=False` |
| $X^\ast$ | Log-moneyness $\ln(F / K)$ | |
| Codes | `"IC"`, `"IP"` for inverse calls and puts | `OptionType.INVERSE_CALL`, `OptionType.INVERSE_PUT` |

The dynamics under the inverse measure are those of Eq. (3.26), described in the
[martingale article](martingale_conditions_and_skews.md#dynamics-under-the-inverse-measure). The
examples below are undiscounted; a discount factor multiplies every value.

## Methodology

### Payoffs in coin units

With $P_T = e^{\bar{\mu}(T)} e^{X_T}$, the inverse payoffs are, Eq. (5.11),

$$
\frac{\max(P_T - K, 0)}{P_T} = 1 - \frac{\min(P_T, K)}{P_T} , \qquad \frac{\max(K - P_T, 0)}{P_T} = \frac{K}{P_T} - \frac{\min(P_T, K)}{P_T} ,
$$

so both follow from the inverse capped payoff $\min(P_T, K) / P_T$, Eq. (5.12).

### Valuation under the inverse measure

The transform of the inverse capped payoff is $\widehat{u}(\Phi) = -e^{-\Phi X^\ast} / (\Phi (\Phi - 1))$,
Eq. (5.14), finite for $0 < \Re \Phi < 1$, the strip where the MGF exists under the inverse measure
(Theorem 4.2). Along $\Phi = iy + \frac{1}{2}$ the value is, Proposition 5.2 and Eq. (5.13),

$$
\tilde{U} = \frac{1}{\pi} \Re \int_0^\infty \frac{e^{-(iy + 1/2) X^\ast}}{y^2 + 1/4} E^{[m]}\left( \tau; \Phi = iy + \frac{1}{2}; p = -1 \right) dy ,
$$

with $E^{[m]}$ the [affine expansion](affine_expansion.md) under the inverse measure. The coin values
of calls and puts on the futures price are then, Eqs. (5.16) and (5.18),

$$
c = 1 - \tilde{U} , \qquad p = \frac{K}{F} - \tilde{U} ,
$$

where $\tilde{E}[1 / P_T] = 1 / F$ follows from the martingale property of $1 / P_t$ under
$\tilde{\mathbb{Q}}$, Eqs. (4.33) and (5.17).

### Dollar values and the two measures

With the coin as numeraire, the dollar value of an inverse option is $F$ times its coin value, and
by the change of numeraire it equals the value of the vanilla option under the MMA measure (Theorem
2.1 of the paper):

$$
F \tilde{E}\left[ \frac{\max(P_T - K, 0)}{P_T} \right] = E\left[ \max(P_T - K, 0) \right] .
$$

The identity needs both measures to be martingale measures: $\kappa_2 \ge \beta$ for the MMA measure
and $\kappa_2 \ge 2 \beta$ for the inverse measure. The two sides are computed with different
expansions ($p = 1$ and $p = -1$), so numerically they agree to the accuracy of the expansion, not
exactly.

### The net delta

The coin value of an option is $c = C / F$. Its change for a relative move of the price is

$$
F \frac{\partial c}{\partial F} = \frac{\partial C}{\partial F} - \frac{C}{F} = \Delta - \frac{C}{F} ,
$$

the net delta of Lucic and Sepp (2024): the number of coins of exposure to a relative price move. A
trader whose profit and loss is counted in coins hedges with the net delta. It is below the Black
delta for a call, and further below zero for a put, by the coin value of the option.

## Worked example

Every block below is an excerpt of
[`examples/docs/inverse_options.py`](../examples/docs/inverse_options.py), which asserts every number
quoted here.

```python
def inverse_values(params: svm.LogSvParams = PARAMS) -> tuple:
    """Coin values of inverse options under the inverse measure, and USD values under MMA."""
    pricer = svm.LogSVPricer()
    usd_inverse, ivols_inverse = pricer.price_slice(params=params, ttm=TTM, forward=FORWARD,
                                                    strikes=STRIKES, optiontypes=INVERSE_TYPES,
                                                    is_spot_measure=False)
    vanilla_types = np.array([code[-1] for code in INVERSE_TYPES])  # "IC" -> "C", "IP" -> "P"
    usd_vanilla, ivols_vanilla = pricer.price_slice(params=params, ttm=TTM, forward=FORWARD,
                                                    strikes=STRIKES, optiontypes=vanilla_types)
    return usd_inverse / FORWARD, usd_vanilla, ivols_inverse, ivols_vanilla
```

With `LOGSV_BTC_PARAMS` ($\kappa_2 = 3.058$ above $2 \beta = 0.303$), a forward of 60,000 dollars and a
one-month maturity, the at-the-money inverse call is worth 0.1023 coins and the vanilla call 6,139.4
dollars. Over seven strikes from 45,000 to 80,000, the dollar values from the two measures agree to
a relative $3 \times 10^{-4}$, and their implied volatilities to $1.5 \times 10^{-4}$. Simulating
400,000 paths under each measure, the Monte Carlo coin values under $\tilde{\mathbb{Q}}$ and dollar
values under $\mathbb{Q}$ are within 1.5 standard errors of the Fourier values at every strike.

On the bundled Bitcoin chain the same comparison holds across four maturities, as the
[Bitcoin case study](app_bitcoin_options.md) shows:

[![Bitcoin implied volatilities from the MMA and inverse valuations against the Monte Carlo confidence interval for four maturities.](images/btc_case_measures.png)](images/btc_case_measures.png)

*Historical snapshot of 21 October 2021, bundled with the package; the exhibit of the Bitcoin case
study, with its fitted parameters. MMA and inverse implied volatilities against 95% Monte Carlo
intervals from 400,000 paths with daily steps, seed 7. The legend's "mse" is a root-mean-square
difference to the Monte Carlo implied volatility.*

The net delta follows from central differences of the dollar price:

```python
def net_delta(price, forward: float, bump: float = 1e-4) -> tuple:
    """USD delta and F times the derivative of the coin value, by central differences."""
    up, down = forward * (1.0 + bump), forward * (1.0 - bump)
    delta = (price(up) - price(down)) / (up - down)
    coin_delta = forward * (price(up) / up - price(down) / down) / (up - down)
    return delta, coin_delta, delta - price(forward) / forward
```

For the at-the-money one-month call, the model delta is 0.5440 and the net delta 0.4417; for the put,
-0.4560 and -0.5583; for the call struck at 70,000, 0.3050 and 0.2562. In each case
$F \partial c / \partial F$ equals $\Delta - C / F$ to $10^{-6}$. For comparison, the Black delta of a one-week
at-the-money call at 60% volatility is 0.5166 and its net delta 0.4834, the setting of the figure
below.

[![Black delta and net delta of one-week at-the-money calls and puts against the Bitcoin price.](images/inverse_net_delta.png)](images/inverse_net_delta.png)

*Synthetic teaching exhibit, the computation of `papers/inverse_options/compare_net_delta.py`: Black
delta and net delta $\Delta - C / F$ of one-week at-the-money options at 60% volatility with the
strike at 50,000 dollars, against the price of the coin.*

Far in the money the two deltas part: the coin value of a call tends to $1 - K / F$, so its net
delta turns back down towards $K / F$, 0.77 at 65,000 dollars; the net delta of a put tends to
$-K / F$, below $-1$.

## Implementation in stochvolmodels

| Name | Role |
|---|---|
| `LogSVPricer.price_slice`, `price_chain` with `is_spot_measure=False` | Value under the inverse measure along $\Re \Phi = 1/2$; the result is $F$ times the coin value, in the units of the forward |
| `OptionType.INVERSE_CALL`, `OptionType.INVERSE_PUT` | Codes `"IC"` and `"IP"`; the inverse-measure pricer treats them as `"C"` and `"P"` |
| `LogSVPricer.simulate_terminal_values` with `is_spot_measure=False` | Terminal log returns under the inverse measure |
| `ConstraintsType.INVERSE_MARTINGALE` | Calibration constraint $\kappa_2 \ge 2 \beta$ |

Divide by the forward to obtain coin values. The money-market-account pricer rejects the inverse
codes with `ValueError`, so a chain with inverse codes is priced with `is_spot_measure=False`. The
implied volatilities of either valuation are Black-Scholes volatilities of the dollar values.
`LOGSV_BTC_PARAMS` is a compatibility export (see
[stability tiers](option_chains_and_conventions.md#stability-tiers)). Run the example from a
checkout with `python examples/docs/inverse_options.py`.

## Interpretation and limitations

- **Equivalence is conditional.** Below $\kappa_2 = 2 \beta$ the inverse price is not a martingale
  under $\tilde{\mathbb{Q}}$, and inverse-measure values are inconsistent with the forward.
- **Numerical agreement.** The two measures use different expansions; their values agree to about
  $10^{-4}$ in implied volatility here, the accuracy of the expansion rather than an identity.
- **The net delta is a sensitivity, not a strategy.** It gives the coin exposure to a small relative
  move; discrete rebalancing, funding and margin in coins are outside this page, and Lucic and Sepp
  (2024) study them with backtests.

## See also

- [Bitcoin options: MMA, inverse and QV valuation](app_bitcoin_options.md)
- [Martingale conditions, measures and positive skews](martingale_conditions_and_skews.md)
- [Fourier pricing of European options](european_option_pricing.md)
- [Option chains, notation and conventions](option_chains_and_conventions.md#measures-mma-and-inverse)

## References

- Lucic, V. and Sepp, A. (2024). Valuation and hedging of cryptocurrency inverse options.
  *Quantitative Finance* 24(7), 851-869.
  [DOI 10.1080/14697688.2024.2364804](https://doi.org/10.1080/14697688.2024.2364804).
- Sepp, A. and Rakhmonov, P. (2023). Log-normal stochastic volatility model with quadratic drift.
  *International Journal of Theoretical and Applied Finance* 26(8), 2450003.
  [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
