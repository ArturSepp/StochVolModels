---
myst:
  html_meta:
    description: >-
      Case study: the log-normal stochastic volatility parameters recorded for VIX, MOVE, OVX and
      Bitcoin volatility, evaluated without market data in stochvolmodels: stationary distributions,
      autocorrelation and persistence of volatility, and the Feller condition of a matched Heston
      model, with the robustness argument of Sepp and Rakhmonov.
---

# What makes a stochastic volatility model robust

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

In a working paper, Sepp and Rakhmonov (2023) ask which stochastic volatility (SV) models can be
applied across equities, rates, commodities, currencies and cryptocurrencies. Its public abstract
proposes four principles, singles out invariance of the dynamics under a change of numeraire, and
argues that only the Heston model and the log-normal SV model with quadratic drift meet the
conditions; it calls the log-normal model robust because its numerical methods need no condition
such as Heston's Feller condition. The module that accompanies the paper records log-normal SV
parameters fitted to four volatility indices. This case study evaluates what those parameters imply,
without market data.

```{note}
The paper is a working paper whose text is not in the repository. Its claims are quoted here from
its public abstract and from the published article that cites it; the numbers are computed from
the parameters recorded in `papers/volatility_models/article_figures.py`.
```

## Overview

The log-normal SV model with quadratic drift keeps its form when the measure changes from the
statistical to the pricing one; with a linear drift a quadratic term appears (Sepp and Rakhmonov,
2023, IJTAF, Section 3.1, citing the working paper for the failure of this invariance in most
conventional models; see [martingale conditions](martingale_conditions_and_skews.md)). The
parameters of the volatility process can therefore be estimated from time series of volatility and
used for pricing. The study asks what the recorded parameters say about the stationary
distribution and the persistence of volatility of four indices, and whether a Heston model with
the same stationary moments would meet the Feller condition.

## Study design and data

**Data, as described by the code.** Daily closes of the VIX, MOVE and OVX indices from Yahoo
Finance since 31 December 1999, divided by 100 (MOVE is then a normal volatility of Treasury
yields in units of 100 bp), and at-the-money implied volatilities of Bitcoin options from a private
dataset. None of these data are distributed with the package or downloaded here.

**Fit, as described by the code.** Two nested steps. For given mean-reversion rates, $\theta$ and
the vol-of-vol $\varepsilon$ are fitted to a histogram of the logarithm of volatility through the
stationary density; the rates $\kappa_1$ and $\kappa_2$ are then fitted so that the simulated
autocorrelation of volatility matches the empirical one. The volatility beta is zero and
$\sigma_0 = \theta$. The published article uses this procedure for Bitcoin and reports
$\kappa_1 = 2.21$ and $\kappa_2 = 2.18$ (IJTAF, Section 6.2), the recorded Bitcoin values rounded.

**Exhibits.** Synthetic: computed from the recorded parameters, with no market data.

## Configuration

| Index | $\theta$ | $\kappa_1$ | $\kappa_2$ | $\varepsilon$ |
|---|---|---|---|---|
| VIX | 0.1993 | 1.2879 | 1.9268 | 0.7210 |
| MOVE | 0.9110 | 0.1000 | 0.4113 | 0.3564 |
| OVX | 0.3853 | 2.7775 | 2.2351 | 0.8344 |
| BTC | 0.7118 | 2.2147 | 2.1803 | 0.9215 |

The MOVE value of $\kappa_1$, 0.1, is the smallest the fitting code allowed when it was recorded. The
companion script
[`examples/docs/app_robust_volatility_models.py`](../examples/docs/app_robust_volatility_models.py)
asserts every number quoted below. The stationary law of volatility is the generalised inverse
Gaussian of IJTAF Eq. (3.38) (see [steady state and moments](volatility_distribution_and_moments.md)):

```python
def stationary_law(params: svm.LogSvParams):
    """Generalised inverse Gaussian law of volatility, IJTAF Eq. (3.38)."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    q = 2.0 * params.kappa1 * params.theta / vartheta2
    b = 2.0 * params.kappa2 / vartheta2
    eta = 2.0 * (params.kappa2 * params.theta - params.kappa1) / vartheta2 - 1.0
    return stats.geninvgauss(p=eta, b=2.0 * np.sqrt(q * b), scale=np.sqrt(q / b))
```

The autocorrelation is simulated with daily steps from starting values drawn from that law, so the
process is stationary from the first step:

```python
def autocorrelation(params: svm.LogSvParams, lags: np.ndarray = LAGS, nb_path: int = 50000,
                    seed: int = 7) -> np.ndarray:
    """Correlation of volatility today and after each lag, simulated from stationary starts."""
    rng = np.random.default_rng(seed)
    nb_steps, dt, _ = set_time_grid(ttm=1.0, nb_steps_per_year=260)
    start = stationary_law(params).rvs(size=nb_path, random_state=rng)
    sigma, _ = simulate_vol_paths(ttm=1.0, v0=start, theta=params.theta, kappa1=params.kappa1,
                                  kappa2=params.kappa2, beta=params.beta, volvol=params.volvol,
                                  nb_path=nb_path, nb_steps_per_year=260,
                                  brownians=np.sqrt(dt) * rng.standard_normal((nb_steps, nb_path)))
    return np.array([np.corrcoef(sigma[0], sigma[lag])[0, 1] for lag in lags])
```

## Results

**Stationary distributions.** The mean of volatility is 0.1923 for VIX, 0.8226 for MOVE, 0.3767
for OVX and 0.6809 for Bitcoin, 3.5%, 9.7%, 2.2% and 4.3% below $\theta$, with standard deviations
of 0.0773, 0.3069, 0.1179 and 0.2291. Every law is skewed to the right, with skewness of 1.49, 0.93,
1.14 and 1.09; volatility exceeds twice $\theta$ with probability 2.1%, 0.7%, 0.8% and 0.8%, and
its 99% quantiles are 0.449, 1.739, 0.745 and 1.389. On a log scale the four are close: the standard
deviation of $\ln \sigma$ is 0.374, 0.372, 0.299 and 0.324.

**Persistence.** The linearised rate of mean reversion $\kappa_1 + \kappa_2 \theta$ gives half-lives
of 108 business days for VIX, 380 for MOVE, 50 for OVX and 48 for Bitcoin. The simulated
autocorrelation at 60 and 120 days is 0.651 and 0.421 for VIX, 0.888 and 0.790 for MOVE, 0.401 and
0.163 for OVX, and 0.382 and 0.147 for Bitcoin; it decays faster than the exponential of the
linearised rate, the effect of the nonlinear drift.

**The autocorrelation is not set by the mean reversion alone.** The published article fits
$\kappa_1$ and $\kappa_2$ to the autocorrelation because it is "determined solely by mean-reversion
parameters" (IJTAF, Section 6.2). With the Bitcoin rates unchanged and the vol-of-vol halved, the
autocorrelation at 60 and 120 days rises from 0.382 and 0.147 to 0.409 and 0.173: the statement
holds approximately, not exactly.

**Feller condition of a matched Heston model.** A Heston model whose stationary variance has the
same mean and variance as $\sigma^2$ here has a Feller ratio $2 \kappa \theta / \vartheta^2$ equal
to the shape of its gamma law: 1.13 for VIX, 1.59 for MOVE, 2.10 for OVX and 1.85 for Bitcoin. All
four satisfy the condition, VIX by the smallest margin; the condition binds in the Heston fit to
Bitcoin options in the [Heston article](heston_model.md).

[![Simulated autocorrelation of volatility for four indices against the exponential of the linearised mean reversion, and their stationary densities of volatility relative to the mean.](images/robust_vol_models.png)](images/robust_vol_models.png)

*Synthetic teaching exhibit from the recorded parameters. (A) Autocorrelation of volatility
simulated from stationary starts, 50,000 paths, against the exponential of the linearised mean
reversion (dashed); (B) stationary densities of volatility divided by its mean, on a log scale.*

## What the study does and does not show

- **It shows** what the recorded parameters imply: right-skewed stationary laws with similar
  dispersion on a log scale, half-lives from about fifty business days (OVX, Bitcoin) to more than a
  year (MOVE), and an autocorrelation that depends on the vol-of-vol as well as on the mean
  reversion.
- **It does not show** the paper's four principles or its empirical evidence, whose text and data
  are not available here, nor a fit of any model to data.
- **It does not reproduce** the paper's figures. The module imports `qis`, an optional dependency
  of the `research` extra outside the core install, and its simulated autocorrelation calls
  `LogSVPricer.simulate_vol_paths` with a step count that the package now reads as steps per year;
  this study simulates with the function below that method and an explicit number of steps.
- **Robustness in practice.** The four indices would satisfy the Feller condition under a matched
  Heston model; the argument for the log-normal model concerns the dynamics and the numerical
  methods, not these four stationary laws.

## Reproduce

Run the companion script from a checkout:

```console
python examples/docs/app_robust_volatility_models.py
```

All cases take seconds. The recorded parameters are those of `MODEL_PARAMS_TABLE` in
`papers/volatility_models/article_figures.py`.

## See also

- [Steady-state distribution, moments and expected QV](volatility_distribution_and_moments.md)
- [The Heston model as a benchmark](heston_model.md)
- [Martingale conditions, measures and positive skews](martingale_conditions_and_skews.md)
- [Monte Carlo simulation schemes](monte_carlo_simulation.md)

## References

- Sepp, A. and Rakhmonov, P. (2023). What is a robust stochastic volatility model. SSRN working
  paper 4647027. [DOI 10.2139/ssrn.4647027](https://doi.org/10.2139/ssrn.4647027).
- Sepp, A. and Rakhmonov, P. (2023). Log-normal stochastic volatility model with quadratic drift.
  *International Journal of Theoretical and Applied Finance* 26(8), 2450003.
  [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
