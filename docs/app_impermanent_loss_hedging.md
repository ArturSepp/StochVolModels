---
myst:
  html_meta:
    description: >-
      Hedging the impermanent loss of concentrated liquidity with European options: the static
      replication by a square-root payoff, vanilla and digital options, its valuation with the
      log-normal stochastic volatility transform in stochvolmodels, a Monte Carlo check, and the
      sensitivity to vol-of-vol and the volatility beta.
---

# Hedging impermanent loss with the LogSV MGF

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

A liquidity provider in a concentrated-liquidity market maker holds a position whose value, compared
with simply holding the tokens deposited, falls whenever the price moves: the impermanent loss. Its
payoff at a horizon is a European function of the terminal price, so it can be replicated statically
by options and valued with any model that prices them. Lipton, Lucic and Sepp (2025) develop this
approach; the module that accompanies it in this repository values the replication with the
log-normal stochastic volatility (SV) model. This case study derives the replication, reproduces the
module's example, and checks it by simulation.

## Overview

For a position over a price range $[p_a, p_b]$, the loss is zero at the initial price and negative
elsewhere. Inside the range it is a smooth function of $\sqrt{p}$; outside, the position is entirely
in one token and the loss grows linearly with the price. The replicating portfolio therefore needs a
payoff in $\sqrt{p}$ over the range, a linear position, a put and a call at the range bounds, and two
digital options, all European and all priced from the moment generating function (MGF) of the
[affine expansion](affine_expansion.md).

## Study design and data

The example is that of `papers/il_hedging/run_logsv_for_il_payoff.py`: an initial and forward
price of 2,200, a range from 2,000 to 2,400, a horizon of ten days, and the module's log-normal SV
parameters ($\sigma_0 = 0.4862$, $\theta = 0.6176$, $\kappa_1 = 1.9558$, $\kappa_2 = 1.9784$,
$\beta = -0.2692$, $\varepsilon = 3.2658$). No market data are used; the parameters are given, not
fitted here. Fees earned by the position are outside the payoff.

## Configuration

With liquidity $L$, the value of the position at price $p$ is
$L (2 \sqrt{p} - \sqrt{p_a} - p / \sqrt{p_b})$ inside the range, $L p (1 / \sqrt{p_a} - 1 / \sqrt{p_b})$
below it and $L (\sqrt{p_b} - \sqrt{p_a})$ above it. Holding the tokens deposited at the initial price
$p_0$ is worth $L \left( p (1 / \sqrt{p_0} - 1 / \sqrt{p_b}) + \sqrt{p_0} - \sqrt{p_a} \right)$.
The impermanent loss is the difference; inside the range it is
$-L (\sqrt{p} - \sqrt{p_0})^2 / \sqrt{p_0}$.

```python
def impermanent_loss(p: np.ndarray, p0: float = P0, pa: float = PA, pb: float = PB) -> np.ndarray:
    """Position value minus the value of holding the initial tokens, per unit of liquidity."""
    position = np.where(p < pa, p * (1.0 / np.sqrt(pa) - 1.0 / np.sqrt(pb)),
                        np.where(p > pb, np.sqrt(pb) - np.sqrt(pa),
                                 2.0 * np.sqrt(p) - np.sqrt(pa) - p / np.sqrt(pb)))
    hold = p * (1.0 / np.sqrt(p0) - 1.0 / np.sqrt(pb)) + np.sqrt(p0) - np.sqrt(pa)
    return position - hold
```

Minus the loss, per unit of liquidity, is the payoff of

$$
-2 \sqrt{p} \mathbb{1}_{p_a \le p \le p_b} + \frac{p}{\sqrt{p_0}} + \sqrt{p_0} + \frac{(p_a - p)^+}{\sqrt{p_a}} - \frac{(p - p_b)^+}{\sqrt{p_b}} - 2 \sqrt{p_a} \mathbb{1}_{p < p_a} - 2 \sqrt{p_b} \mathbb{1}_{p > p_b} ,
$$

which can be checked on each of the three pieces of the price axis:

```python
def replication(p: np.ndarray, p0: float = P0, pa: float = PA, pb: float = PB) -> dict:
    """The static portfolio whose payoff is minus the impermanent loss, by component."""
    return {"square root in range": -2.0 * np.sqrt(p) * ((p >= pa) & (p <= pb)),
            "linear": p / np.sqrt(p0) + np.sqrt(p0),
            "put at pa": np.maximum(pa - p, 0.0) / np.sqrt(pa),
            "call at pb": -np.maximum(p - pb, 0.0) / np.sqrt(pb),
            "digital put at pa": -2.0 * np.sqrt(pa) * (p < pa),
            "digital call at pb": -2.0 * np.sqrt(pb) * (p > pb)}
```

The put, call and digital options are priced by the package's slice pricers on a transform grid
with $\Re \Phi = -0.4$ and 1,001 points; the square-root payoff has the transform
$\left( e^{(\Phi + 1/2) x_b} - e^{(\Phi + 1/2) x_a} \right) e^{-\Phi x} / (\Phi + 1/2)$ in the
log-prices $x = \ln F$, $x_a = \ln p_a$ and $x_b = \ln p_b$, and is integrated against the same MGF.
The expected loss is reported per unit of the initial value of the position.

## Results

The portfolio pays minus the impermanent loss to $10^{-12}$ at every terminal price from 1,500 to
3,000, and the loss is zero at the initial price and negative elsewhere. The expected loss over ten
days is 1.7397% of the initial value of the position, 17,397 on a position of 1,000,000, the value
the module computes. A simulation of 400,000 paths gives 1.7378% with a standard error of 0.0037%,
within one standard error.

At a fixed total vol-of-vol $\vartheta$, moving the volatility beta from -1 to 1 changes the
expected loss little: 1.726%, 1.742% and 1.734% for $\beta = -1, 0, 1$. The residual vol-of-vol
matters more: with $\varepsilon = 1$ the loss is 1.650%, with $\varepsilon = 4$ it is 1.786%. The
loss behaves like a short position in options around the range, so it grows with the curvature of
the smile, which the vol-of-vol controls; the tilt that $\beta$ adds changes it little here.

[![Impermanent loss against the terminal price with minus the payoff of the replicating portfolio, and the expected loss against the vol-of-vol for three volatility betas.](images/il_replication.png)](images/il_replication.png)

*Synthetic teaching exhibit. (A) Impermanent loss at the horizon in % of the initial position value
against the terminal price, with the range shaded and the initial price dashed; the dots are minus
the payoff of the replicating portfolio. (B) Expected loss over ten days in % of the initial
position value against the residual vol-of-vol, for three volatility betas at the module's other
parameters.*

## What the study does and does not show

- **It shows** that the impermanent loss of a concentrated-liquidity position is a European payoff
  replicated exactly by a square-root payoff, a linear position, one put, one call and two digitals,
  and that the package values it consistently with simulation.
- **It does not show** the fees earned by the position, which offset the loss, the cost of trading
  the replicating options or their availability, or any calibration to a market; the parameters are
  those of the module's example.
- **Model risk.** The value depends on the smile the model generates around the range; the
  sensitivity above shows its direction and is not a study of hedge effectiveness.

## Reproduce

Run the example from a checkout with `python examples/docs/app_impermanent_loss_hedging.py`; the
pricing module is `papers/il_hedging/run_logsv_for_il_payoff.py`, whose `logsv_il_pricer` returns the
same value on a notional of 1,000,000. The functions used are advanced and compatibility exports:
`compute_logsv_a_mgf_grid`, `get_transform_var_grid`, `vanilla_slice_pricer_with_mgf_grid`,
`digital_slice_pricer_with_mgf_grid` and `compute_integration_weights` (see
[stability tiers](option_chains_and_conventions.md#stability-tiers)).

## See also

- [The moment generating function and its affine expansion](affine_expansion.md)
- [Fourier pricing of European options](european_option_pricing.md)
- [Log-normal stochastic volatility model with quadratic drift](logsv_model.md)

## References

- Lipton, A., Lucic, V. and Sepp, A. (2025). Unified approach for hedging impermanent loss of
  liquidity provision. *Digital Finance* 7(3), 429-477.
  [DOI 10.1007/s42521-025-00144-5](https://doi.org/10.1007/s42521-025-00144-5).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
