---
myst:
  html_meta:
    description: >-
      Notation, units and conventions of stochvolmodels: the log-normal SV and Heston symbols
      mapped to code, option types, MMA and inverse measures, the OptionChain layout, discounting,
      Monte Carlo seeds and standard errors, equation numbering and the stability tiers.
---

# Option chains, notation and conventions

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This page defines, once for the whole site, the symbols, units, measures, data layout and
simulation conventions that the other pages use. They link here instead of restating them. Every
Python block below is an excerpt of
[`examples/docs/option_chains_and_conventions.py`](../examples/docs/option_chains_and_conventions.py),
which runs offline and asserts every number quoted here.

## Notation: paper and code

The log-normal stochastic volatility (SV) model of Sepp and Rakhmonov (2023) specifies the
volatility $\sigma_t$ of the underlying price under the money-market-account (MMA) measure by their
Eq. (3.12):

$$
d\sigma_t = (\kappa_1 + \kappa_2 \sigma_t)(\theta - \sigma_t) dt + \beta \sigma_t dW^{(0)}_t + \varepsilon \sigma_t dW^{(1)}_t ,
$$

where $W^{(0)}$ also drives the price and $W^{(1)}$ is an independent Brownian motion. The
parameters map to the fields of `LogSvParams`:

| Paper | Code | Meaning | Convention |
|---|---|---|---|
| $\sigma_0$ | `sigma0` | Initial volatility | Annualised decimal |
| $\theta$ | `theta` | Mean volatility | $\theta > 0$, annualised decimal |
| $\kappa_1$ | `kappa1` | Linear mean-reversion rate | $\kappa_1 \ge 0$, per year |
| $\kappa_2$ | `kappa2` | Quadratic mean-reversion rate | $\kappa_2 \ge 0$; `None` sets $\kappa_1 / \theta$ |
| $\beta$ | `beta` | Volatility beta: loading of volatility on the price shock | Its sign sets the direction of the skew |
| $\varepsilon$ | `volvol` | Volatility of residual volatility | $\varepsilon > 0$ |
| $\vartheta^2$ | derived | Total vol-of-vol $\beta^2 + \varepsilon^2$, Eq. (3.13) | Not a field |
| $\tau$ | `ttm` | Time to maturity | Years |

```python
def logsv_symbols() -> dict:
    """Map the paper's symbols to LogSvParams fields and derive the total vol-of-vol."""
    params = svm.LogSvParams(sigma0=0.8, theta=1.0, kappa1=4.0, kappa2=None,
                             beta=0.2, volvol=1.5)
    vartheta2 = params.beta ** 2 + params.volvol ** 2  # total vol-of-vol, Eq. (3.13)
    return {"kappa2": params.kappa2, "vartheta2": vartheta2}
```

Here `kappa2=None` becomes $\kappa_2 = \kappa_1 / \theta = 4$, and $\vartheta^2 = 2.29$. The paper
calls the model the log-normal beta SV model: the model of Karasinski and Sepp (2012), augmented with
the quadratic drift (Sepp and Rakhmonov, 2023, Section 3.1). The package names it the
Karasinski-Sepp model.

`HestonParams` follows Heston (1993) with the variance $v_t$,
$dv_t = \kappa (\theta - v_t) dt + \nu \sqrt{v_t} dZ_t$, where the correlation of $Z$ with the
price shock is $\rho$. Its fields are `v0`, `theta`, `kappa`, `rho` and `volvol` for $\nu$. Note that
`theta` is a volatility in `LogSvParams` and a variance in `HestonParams`.

## Units and quotation

- **Time.** `ttm` and `ttms` are year fractions. The package applies no day-count or calendar
  convention; convert dates before building a chain.
- **Prices.** Forwards and strikes share one price unit. Option values are returned in that quote
  currency and include the discount factor.
- **Discounting.** A slice carries either a discount factor $D$ or a continuously compounded rate
  $r$, with $D = e^{-r \tau}$; the other is derived. Without either, $D = 1$.
- **Volatility.** Model and Black–Scholes–Merton (BSM) implied volatilities are annualised
  decimals, so 20% is `0.20`. Bachelier (normal) volatility is an annualised absolute price or rate
  volatility.
- **Implied volatility.** Model prices are converted to BSM implied volatilities with the forward
  and discount factor of their slice, through
  [vanilla-option-pricers](https://github.com/ArturSepp/VanillaOptionPricers).

## Option types and state variables

| `OptionType` | Code | Payoff at maturity |
|---|---|---|
| `CALL` | `"C"` | $\max(S_T - K, 0)$ in the quote currency |
| `PUT` | `"P"` | $\max(K - S_T, 0)$ in the quote currency |
| `INVERSE_CALL` | `"IC"` | $\max(S_T - K, 0) / S_T$ in units of the underlying |
| `INVERSE_PUT` | `"IP"` | $\max(K - S_T, 0) / S_T$ in units of the underlying |

`VariableType` selects the state variable whose distribution is priced: `LOG_RETURN` for options
on the price, `Q_VAR` for options on the quadratic variance (QV) $I_\tau = \int_0^\tau \sigma_t^2 dt$,
and `SIGMA` for the volatility itself.

## Measures: MMA and inverse

Every price is computed under one of two valuation measures of Sepp and Rakhmonov (2023, Section
2.2). The MMA measure uses the money-market account as numeraire and is the default,
`is_spot_measure=True`. The inverse measure uses the spot price of the underlying as numeraire,
`is_spot_measure=False`. It is the natural measure for inverse options, whose premium and payoff
are quoted in units of the underlying, as on cryptocurrency exchanges. When both measures are
equivalent martingale measures, the value of an inverse option in the quote currency equals the
value of the corresponding vanilla option. The package therefore returns the same quote-currency
number for `"IC"` as for `"C"`; divide it by the spot price to express it in units of the
underlying.

By Theorem 3.7 of the paper, the model is consistent with valuation under the MMA measure when
$\kappa_2 \ge \beta$, and under the inverse measure when $\kappa_2 \ge 2\beta$.
`ConstraintsType` imposes these conditions in calibration. The parameter set of the script
satisfies both:

```python
def price_under_both_measures(params: svm.LogSvParams = PARAMS) -> dict:
    """Price one slice under the MMA and the inverse measure, and inverse codes under the latter."""
    pricer = svm.LogSVPricer()
    strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    vanilla = np.array(["P", "P", "C", "C", "C"])
    inverse = np.array(["IP", "IP", "IC", "IC", "IC"])
    mma, mma_vols = pricer.price_slice(params=params, ttm=1.0 / 12.0, forward=1.0,
                                       strikes=strikes, optiontypes=vanilla)
    inv, inv_vols = pricer.price_slice(params=params, ttm=1.0 / 12.0, forward=1.0,
                                       strikes=strikes, optiontypes=vanilla,
                                       is_spot_measure=False)
    inv_codes, _ = pricer.price_slice(params=params, ttm=1.0 / 12.0, forward=1.0,
                                      strikes=strikes, optiontypes=inverse,
                                      is_spot_measure=False)
    return {"mma": mma, "inverse": inv, "inverse_codes": inv_codes,
            "mma_vols": mma_vols, "inverse_vols": inv_vols}
```

On this one-month slice the two valuations differ by at most $1.5 \times 10^{-5}$ in price and
$1.3 \times 10^{-4}$ in implied volatility. The residual comes from the numerical approximations,
which are applied to a different transform under each measure. The inverse codes return exactly
the vanilla values. The MMA branch of the
LogSV Fourier pricer rejects `"IC"` and `"IP"` with `ValueError`, while the BSM functions price them
with the call and put formulas.

## Option chains

`OptionSlice` holds one maturity. `OptionChain` holds several strictly increasing maturities, each
with its own strike and option-type arrays. This ragged layout supports different strike grids
without padding, and it is the common input of pricing, calibration and Monte Carlo. The arrays are
`numba.typed.List` objects, because the pricing kernels are compiled.

```python
def build_chain() -> svm.OptionChain:
    """Two maturities with their own strike grids, bid/ask quotes and a continuous discount rate."""
    return svm.OptionChain(
        ttms=np.array([1.0 / 12.0, 0.25]),
        forwards=np.array([100.0, 101.0]),
        strikes_ttms=List([np.array([95.0, 100.0, 105.0]),
                           np.array([90.0, 100.0, 110.0, 120.0])]),
        optiontypes_ttms=List([np.array(["P", "C", "C"]),
                               np.array(["P", "C", "C", "C"])]),
        ids=np.array(["1m", "3m"]),
        discount_rates=np.array([0.04, 0.04]),
        bid_ivs=List([np.array([0.21, 0.19, 0.18]), np.array([0.23, 0.20, 0.19, 0.19])]),
        ask_ivs=List([np.array([0.23, 0.21, 0.20]), np.array([0.25, 0.22, 0.21, 0.21])]),
    )
```

```python
def inspect_chain(chain: svm.OptionChain) -> dict:
    """Read the derived discount factors, one slice, mid vols and forward-normalised strikes."""
    three_month = chain.get_slice("3m")
    normalised = svm.OptionChain.to_forward_normalised_strikes(chain)
    return {
        "discfactors": chain.discfactors,  # exp(-rate * ttm), derived from discount_rates
        "strikes_3m": three_month.strikes,
        "mid_vols_1m": chain.get_mid_vols()[0],
        "normalised_strikes_3m": normalised.strikes_ttms[1],  # strikes / forward
    }
```

The derived discount factors are 0.99667 and 0.99005. The one-month mid volatilities are the
averages of bid and ask, 0.22, 0.20 and 0.19. The forward-normalised chain divides strikes by the
forward of their slice and sets the forwards to one.

The alignment contract:

- `ttms`, `forwards`, `ids`, the discount arrays and every per-maturity list have one entry per
  maturity; maturities are finite, positive and strictly increasing.
- Within a slice, strikes, option types, bid and ask volatilities, and bid and ask prices have equal
  lengths.
- `OptionChain.get_uniform_chain` builds a quote-free chain on one strike grid, with puts below the
  forward and calls at or above it. This out-of-the-money convention is a convenience, not a
  restriction on pricing. Calibration needs a chain with market quotes.
- `OptionChain.compute_model_ivols_from_chain_data` maps model prices back to one implied-volatility
  array per maturity. Keep the chain's order when flattening quotes for an optimizer.

Construction raises `ValueError` for empty or non-finite grids, unordered maturities, length
mismatches, non-positive forwards, strikes or discount factors, crossed bid and ask quotes, and
unsupported option types. Validation cannot detect a spot quoted as a forward, percentages instead
of decimals, or a different settlement convention; normalise these before building the chain.

## Monte Carlo conventions

`model_mc_price_chain` returns, for each maturity, the prices and their standard errors. A 95%
confidence interval is the price plus or minus 1.96 standard errors. Set `nb_path` and `nb_steps`
(time steps per year) explicitly in reproducible work. When `nb_steps` is omitted, the LogSV
wrapper uses `int(360 * max(ttms)) + 1` steps per year, which is coarser for short chains.

The pricing simulators behind `model_mc_price_chain` for LogSV and Heston run inside numba and draw
from numba's random generator, which is separate from NumPy's.
`stochvolmodels.utils.funcs.set_seed` seeds it and makes a run repeatable:

```python
def seeded_monte_carlo(params: svm.LogSvParams = PARAMS, seed: int = 7) -> dict:
    """Run one Monte Carlo valuation twice after seeding the numba generator, and price it too."""
    chain = svm.OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(["3m"]),
                                              strikes=np.array([0.9, 1.0, 1.1]))
    pricer = svm.LogSVPricer()
    set_seed(seed)
    first, _ = pricer.model_mc_price_chain(option_chain=chain, params=params,
                                           nb_path=20000, nb_steps=360)
    set_seed(seed)
    second, standard_errors = pricer.model_mc_price_chain(option_chain=chain, params=params,
                                                          nb_path=20000, nb_steps=360)
    analytic = pricer.price_chain(option_chain=chain, params=params)
    return {"first": first[0], "second": second[0], "standard_errors": standard_errors[0],
            "analytic": analytic[0]}
```

The two runs are identical, and each Monte Carlo price lies within three standard errors of the
analytic price. Other simulators follow different rules:

- `LogSVPricer.simulate_vol_paths` and the Hawkes jump-diffusion simulator draw with NumPy's global
  generator, which `set_seed` does not reach. Pass pre-drawn increments through the `brownians`
  argument of `simulate_vol_paths`, drawn from a local `numpy.random.Generator`, or seed NumPy.
- The rough-volatility path and the Monte Carlo calibration engines take a `seed` argument: they
  draw their random numbers once from a local generator and reuse them, which leaves global random
  states untouched.

## Equation numbering and citations

Docstrings and pages cite equation numbers of the **published PDF** of each paper. The PDF of the
log-normal SV paper numbers equations by section, (3.12) for the dynamics, while its LaTeX source
numbers them sequentially. The factor HJM source predates the revision that added the auxiliary
factor, so its numbering diverges after equation (2). Never correct an equation reference against
the LaTeX source. The paper directories are described in
[research papers and replication](reproducing_the_papers.md).

## Stability tiers

| Tier | How it is reached | Contents | Contract |
|---|---|---|---|
| Stable | `stochvolmodels.__all__` | Pricers, parameters, chains, enums, BSM and Bachelier analytics, expected QV | Tested by the API contract; changes are recorded in the changelog |
| Advanced | Package root, outside `__all__` | `ExpansionOrder` and the affine-expansion functions, `HawkesJDParams` and `HawkesJDPricer`, the fixed-random Monte Carlo functions | Documented on this site; research interfaces |
| Compatibility | Package root, outside `__all__` | Historical names such as plotting helpers, sample chains, transform-grid utilities and preset parameters | Kept importable; not the recommended starting point |
| Provisional | Direct module imports | `stochvolmodels.models`, `stochvolmodels.valuation`, `stochvolmodels.products.payoffs` and the regime-switching LogSV stack | Outside `__all__` until their contracts are stabilised |
| Experimental | Deep module imports | `stochvolmodels.pricers.rough_logsv` and `stochvolmodels.pricers.factor_hjm` | Research surfaces that may evolve between minor releases |

The [API reference](api.md) groups the stable and advanced names by the page that explains them.

## Glossary

- **Affine expansion.** The approximation of the moment generating function (MGF) of the
  log-normal SV model by an exponential of a quadratic polynomial in volatility, with coefficients
  from a system of ordinary differential equations.
- **Admissibility.** The parameter region in which the discounted price, or its inverse, is a true
  martingale, so that valuation under a measure is consistent.
- **Inverse option.** An option whose payoff is paid in units of the underlying.
- **Quadratic variance (QV).** The integrated variance $I_\tau$ of the log price.
- **Variance swap.** A forward contract on realised variance; its fair strike is the expected QV
  divided by the maturity.
- **Vol backbone.** An optional term structure of multiplicative scalings of $\theta$, indexed by
  maturity, set by `LogSvParams.set_vol_backbone`.
- **Vega weighting.** Weighting implied-volatility residuals by BSM vega in calibration.

## See also

- [Installation and first result](getting_started.md)
- [Log-normal stochastic volatility model with quadratic drift](logsv_model.md)
- [Fourier pricing of European options](european_option_pricing.md)
- [Analytic versus Monte Carlo validation](analytic_vs_monte_carlo.md)
- [API reference](api.md)

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with
  applications to bond and currency options. *The Review of Financial Studies* 6(2), 327-343.
- Karasinski, P. and Sepp, A. (2012). Beta stochastic volatility model. *Risk*, October 2012.
- Sepp, A. and Rakhmonov, P. (2023). Log-normal stochastic volatility model with quadratic drift.
  *International Journal of Theoretical and Applied Finance* 26(8), 2450003.
  [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
- Sepp, A. and Rakhmonov, P. (2025). Stochastic volatility for factor Heath-Jarrow-Morton framework.
  *Review of Derivatives Research* 28, article 12.
  [DOI 10.1007/s11147-025-09217-4](https://doi.org/10.1007/s11147-025-09217-4).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
