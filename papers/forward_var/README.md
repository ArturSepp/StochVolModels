---
myst:
  html_meta:
    description: >-
      Specification of the forward-variance LogSV model, its variance-swap volatility
      bootstrap, two-parameter smile calibration, and PDE and Monte Carlo validation.
---

# Forward-variance LogSV: model specification and calibration

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This directory contains an exploratory calibration of the classical log-normal stochastic
volatility model with quadratic drift and a deterministic return-volatility loading.
The loading matches a term structure of option-implied variance-swap rates, while two
stochastic-volatility parameters fit the option smiles. This is a source-checkout workflow;
no separate published forward-variance paper is asserted. The implementation and numerical
results described here were checked on 18 November 2026.

## Overview

The model separates the level and term structure of expected return variance from the
parameters governing the shape of option smiles. A piecewise-constant loading is bootstrapped
at every quoted maturity. For each candidate pair of volatility beta and independent
volatility-of-volatility, the loading is recomputed before option prices are evaluated.

The current driver fixes the initial volatility, drift target and two mean-reversion
coefficients. It calibrates only `beta` and `volvol`. Its default mode is
`CALIBRATE_VARSWAP_PARAM_MODEL`, implemented by
[`calibrated_varswap_fit`](calibrate_forward_var.py).

The model has two Markov state variables: log-normalized asset price and volatility.
The forward-variance curve is an implication of these dynamics and the deterministic
loading, rather than an additional independently specified stochastic process.

## Inputs, notation, and assumptions

Time is measured in years. Volatilities are annualized decimals: 0.85 denotes 85%.
Option prices, forwards and strikes use the same quote currency; discount factors are
dimensionless. The example uses the bundled BTC option chain dated 21 October 2021:
49 quotes at four maturities, with discount factors equal to one.

| Symbol | Code field or input | Meaning |
|---|---|---|
| $X_t$ | normalized log-price state | Log asset price after removing deterministic carry; $X_0=0$ |
| $\sigma_t$ | volatility state | Unscaled instantaneous volatility factor |
| $\sigma_0$ | `sigma0` | Initial volatility factor |
| $\theta$ | `theta` | Zero of the volatility drift; not generally its stationary mean |
| $\kappa_1,\kappa_2$ | `kappa1`, `kappa2` | Linear and quadratic mean-reversion coefficients |
| $\beta$ | `beta` | Volatility diffusion loading on the return Brownian motion |
| $\epsilon$ | `volvol` | Volatility diffusion loading on an independent Brownian motion |
| $\eta_i$ | `vol_backbone.iloc[i-1]` | Dimensionless return-volatility loading on the interval ending at $T_i$ |
| $F_i,D_i$ | `forwards`, `discfactors` | Forward price and discount factor for maturity $T_i$ |
| $k_{ij},v_{ij}$ | strikes and mid implied vols | Strike and Black-Scholes implied volatility of quote $j$ at expiry $i$ |
| $K_i$ | `varswap_strikes` | Square root of annualized fair variance, in volatility units |

Bid and ask implied volatilities are averaged before calibration. The variance-swap targets
are derived from the same option quotes, with an at-the-money (ATM) floor; they are not a
separate observed swap-market data set. The stored labels `2w`, `1m`, `2m`, `3m` are legacy
labels. Use the actual year fractions, especially for the last maturity.

## Methodology

### Dynamics and meaning of the loading

Under the money-market-account pricing measure, let $W^0$ and $W^1$ be independent Brownian
motions. With deterministic carry removed, the implemented dynamics are

$$
dX_t=-\frac{1}{2}\eta(t)^2\sigma_t^2 dt+\eta(t)\sigma_t dW_t^0,
$$

$$
d\sigma_t=(\kappa_1+\kappa_2\sigma_t)(\theta-\sigma_t) dt
\quad +\beta\sigma_t dW_t^0+\epsilon\sigma_t dW_t^1.
$$

Here the instantaneous return volatility is $\eta(t)\sigma_t$. The loading does not multiply
the volatility-factor drift or diffusion. It multiplies return variance by its square and
return-volatility covariance linearly. The coefficients can be checked directly in the
[LogSV PDE implementation](../../src/stochvolmodels/models/logsv/coefficients.py).

Total volatility-of-volatility and instantaneous return-volatility correlation are

$$
\vartheta=\sqrt{\beta^2+\epsilon^2},
\qquad
\rho=\frac{\beta}{\sqrt{\beta^2+\epsilon^2}}.
$$

Thus `volvol` is the independent diffusion loading, not the total volatility-of-volatility.
The study uses the classical Markov model, with `H=0.5`.

For $0=T_0<T_1<\cdots<T_N$, the backbone convention is

$$
\eta(t)=\eta_i,\qquad T_{i-1}<t\le T_i.
$$

A pandas Series stores $\eta_i$ at index $T_i$, the interval's right endpoint. All reported
fits and comparisons stop at the last quoted expiry; the PDE replay requires backbone
coverage through the final observation.

For $u\ge t$, define the model's forward instantaneous variance by

$$
\xi_t(u)=\eta(u)^2 \mathbb{E}_t[\sigma_u^2].
$$

At time zero its integrated value is

$$
A^{\mathrm{model}}(T)=\int_0^T\xi_0(u) du
=\mathbb{E}\left[\int_0^T\eta(u)^2\sigma_u^2 du\right].
$$

Even when $\eta$ is constant on an interval, $\xi_0(u)$ need not be constant there: the
second moment of the volatility factor changes with time.

### Option-strip targets and the ATM floor

[`OptionChain.get_slice_varswap_strikes`](../../src/stochvolmodels/data/option_chain.py)
first converts the mid implied volatilities into undiscounted Black-Scholes option prices.
The [strip integrator](../../src/stochvolmodels/utils/var_swap_pricer.py) uses puts below
the forward and calls at or above it. Its discrete estimate is

$$
\widehat{K}_i^2=\frac{1}{T_i}
\left(2\sum_j\frac{\Delta k_{ij} Q_i(k_{ij})}{k_{ij}^2}
\quad -\left(\frac{F_i}{k_i^{*}}-1\right)^2\right).
$$

Here $Q_i$ is the undiscounted out-of-the-money option price, $\Delta k_{ij}$ is centered
strike spacing with one-sided endpoint spacing, and $k_i^{*}$ is the first available
strike at or above $F_i$. This is the code's finite-strip convention. Missing option-side
values are omitted by its numerical sum; this does not reconstruct unquoted tails.

The returned value is $\widehat{K}_i$, a volatility. The workflow applies

$$
K_i=\max\left(\widehat{K}_i,v_i^{\mathrm{ATM}}\right),
$$

where ATM volatility is interpolated from mid implied volatilities at the forward. The
floor is a safeguard against sparse-strip underestimation, not an exact completion of
the missing option wings. Both arguments of the maximum have volatility units.

### Variance-swap bootstrap

Let $A_i=T_iK_i^2$ be the market target for integrated variance, and let

$$
B_i=\int_0^{T_i}\mathbb{E}[\sigma_u^2] du,
\qquad A_0=B_0=0,
$$

be the integrated second moment of the unscaled volatility factor for the current trial
parameters. Because its dynamics do not depend on $\eta$, the loading can be solved
interval by interval:

$$
\eta_i=\sqrt{\frac{A_i-A_{i-1}}{B_i-B_{i-1}}}.
$$

The square root is required at **every** expiry. With these loadings,

$$
\sum_{j=1}^{i}\eta_j^2(B_j-B_{j-1})=A_i,
\qquad
K_i^{\mathrm{model}}=
\sqrt{\frac{1}{T_i}\sum_{j=1}^{i}\eta_j^2(B_j-B_{j-1})}=K_i.
$$

The average forward variance over interval $i$ is
$(A_i-A_{i-1})/(T_i-T_{i-1})$; it is distinct from the dimensionless loading $\eta_i^2$.
The [bootstrap implementation](../../src/stochvolmodels/pricers/logsv/vol_moments_ode.py)
rejects nonfinite or nonpositive quotes, nonpositive or unordered maturities, and nonpositive
market integrated-variance increments. Invalid model increments or loadings raise
`FloatingPointError`. Targets are not silently repaired into a monotone curve.

### Four-moment approximation

The bootstrap evaluates $B_i$ using `compute_analytic_qvar` with `n_terms=4`.
This is a finite moment approximation. Define $Y_t=\sigma_t-\theta$,
$m_n(t)=\mathbb{E}[Y_t^n]$, $m_0=1$, and

$$
\kappa=\kappa_1+\kappa_2\theta,
\qquad c_n=\frac{1}{2}n(n-1)(\beta^2+\epsilon^2).
$$

The first moment satisfies $m_1'=-\kappa m_1-\kappa_2m_2$. For $n\ge2$ the moment recursion is

$$
m_n'=(c_n-n\kappa)m_n-n\kappa_2m_{n+1}
\quad +2c_n\theta m_{n-1}+c_n\theta^2m_{n-2}.
$$

The implementation solves for $m_1,\ldots,m_4$ and freezes $m_5$ at its initial value
$(\sigma_0-\theta)^5$. Matrix exponentiation gives the moments and their time integrals,
from which

$$
\mathbb{E}[\sigma_t^2]\approx m_2(t)+2\theta m_1(t)+\theta^2
$$

determines $B_i$. Matching the targets exactly in this approximation does not establish an
exact variance-swap fit for the untruncated diffusion. The moment method and closure are
implemented in [vol_moments_ode.py](../../src/stochvolmodels/pricers/logsv/vol_moments_ode.py)
and [LogSvParams](../../src/stochvolmodels/pricers/logsv/logsv_params.py).

### Outer smile calibration

For each candidate $(\beta,\epsilon)$, the calibrator rebuilds the complete backbone,
prices all usable options with the analytic expansion, converts prices to implied
volatilities, and evaluates

$$
J(\beta,\epsilon)=\sum_{i=1}^{N}\sum_j
w_{ij}\left(v_{ij}^{\mathrm{model}}(\beta,\epsilon)-v_{ij}^{\mathrm{mid}}\right)^2,
\qquad
w_{ij}=\frac{\mathrm{Vega}_{ij}}{\sum_k\mathrm{Vega}_{ik}}.
$$

Vegas are evaluated at market mid volatilities and actual maturities
(`is_unit_ttm_vega=False`). Their normalization gives every expiry total weight one.
Variance-swap quotes are imposed through the inner bootstrap, not added as residuals to
this objective. Reported aggregate weighted root-mean-square error (RMSE), in volatility
percentage points, is $100\sqrt{J/N}$.

| Setting | Current value |
|---|---|
| Calibration mode | `LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT` |
| Fixed parameters | `sigma0=0.85`, `theta=0.85`, `kappa1=2.21`, `kappa2=2.18` |
| Free parameters | `beta`, `volvol` |
| Initial guess | `beta=0.15`, `volvol=1.5` |
| Box bounds | `beta` in [0, 3]; `volvol` in [0.2, 3] |
| Additional constraint | `ConstraintsType.INVERSE_MARTINGALE`: $\kappa_2-2\beta\ge0$ |
| Optimizer | SciPy SLSQP, `ftol=1e-8` |
| Pricing engine | `CalibrationEngine.ANALYTIC`, default second expansion order |
| Historical ordering | `historical_backbone_order=True` |

The additional constraint makes the effective upper bound on beta 1.09 for these fixed
mean-reversion parameters. No fourth-moment constraint is selected. The independent volvol
lower bound was relaxed from 1.5 to 0.2; the starting value remains 1.5.

The Fourier scale is fixed from the shortest market ATM quote throughout optimization:
`vol_scaler` is about 0.179, with 1,000 frequency nodes for this snapshot. Numerical
`FloatingPointError` trials receive a finite penalty of `1e12`. Optimizer success and
parameter bounds are validated, and the returned candidate is repriced to reject a
numerically invalid final fit. These details are in
[`LogSVPricer.calibrate_model_params_to_chain`](../../src/stochvolmodels/pricers/logsv_pricer.py).

The constraint enum imposes the **unscaled** inequality shown above; it does not by itself
check every loading-dependent measure-change condition. The PDE adapter independently
requires $\kappa_2\ge\eta_i\beta$ on every interval for its money-market-measure calculation.
The fitted parameters pass this check. Selecting an inverse-martingale constraint does
not change the calibration instruments into inverse-payoff options: this run fits ordinary
calls and puts under the money-market measure.

### Pricing conventions and independent comparisons

The historical expansion carries its coefficients forward through quoted maturities.
The current package default instead composes the transforms backward from each payoff
through calendar-time loading intervals. These compositions need not agree for changing
loadings. The reported calibration deliberately uses the historical convention; switching
to the default is a different calibration specification and requires recalibration.
[`verify_pde.py`](verify_pde.py) also reports the backward-composed expansion as a diagnostic.

The independent partial differential equation (PDE) calculation uses the generator

$$
\mathcal{L}_t f=-\frac{1}{2}\eta^2\sigma^2 f_x
\quad +(\kappa_1+\kappa_2\sigma)(\theta-\sigma)f_\sigma
\quad +\frac{1}{2}\eta^2\sigma^2 f_{xx}
\quad +\eta\beta\sigma^2 f_{x\sigma}
\quad +\frac{1}{2}(\beta^2+\epsilon^2)\sigma^2 f_{\sigma\sigma}.
$$

In particular, the mixed-derivative coefficient is the full covariance. The adapter
propagates the joint density from $(X_0,\sigma_0)=(0,\sigma_0)$ across every loading knot
and integrates the call or put payoff on the quoted strikes. A call is valued as
$D_i\mathbb{E}[(F_i\exp(X_{T_i})-k)^+]$.

The replay uses nonuniform grids and the generic alternating-direction solver with
`ADIConfig(theta=1.0)`. The numerical `theta` here is a solver parameter, not the
volatility drift target. The time-refined mesh supplies the reported PDE reference:

| Mesh | Log-price nodes | Volatility nodes | Log-price domain | Volatility ceiling | Steps per interval |
|---|---:|---:|---|---:|---:|
| Space coarse | 151 | 81 | [-6, 6] | 9 | 128 |
| Fine | 241 | 121 | [-6, 6] | 9 | 128 |
| Time refined | 241 | 121 | [-6, 6] | 9 | 256 |
| Wider domain | 261 | 141 | [-8, 8] | 14 | 128 |

Monte Carlo (MC) comparisons use 100,000 paths and seed 24 at both 156 and 360 steps/year.
A refinement uses 12 independent batches of 8,192 paths at 720 steps/year, with seeds
240 through 251. Batch standard errors come from the variation of the batch mean prices.
The existing forward recentering convention is retained. None of these MC runs supplies
the calibration objective.

## Worked example

For the historical BTC snapshot, relaxing the lower bound on `volvol` produces the
following results. Figures below are rounded; the actual calibration retains full precision.

| Metric | Lower bound 1.5 | Lower bound 0.2 |
|---|---:|---:|
| Fitted beta | 0.244 | 0.242 |
| Fitted independent volvol | 1.50 | 1.16 |
| Weighted objective | 0.00294 | 0.00227 |
| Expansion market-IV RMSE, vol points | 2.71 | 2.38 |
| PDE market-IV RMSE, vol points | 2.89 | 2.39 |
| Refined MC market-IV RMSE, vol points | 2.74 | 2.32 |
| Maximum PDE/expansion price gap, bp of discounted forward | 16.1 | 5.35 |

The optimizer converged in 10 iterations and 30 function evaluations. The objective
decreased by 22.7%, and weighted IV RMSE by 12.1%. The optimum lies inside the new bounds.
The independently priced PDE and MC fits also improve. Most of the improvement is at the
longest expiry; the first and third expiry smile RMSEs increase slightly.

The fitted backbone and target rates are:

| Actual maturity, years | Target variance-swap volatility | Loading eta |
|---:|---:|---:|
| 0.0429 | 0.882 | 1.02 |
| 0.101 | 0.909 | 1.06 |
| 0.198 | 0.946 | 1.11 |
| 0.432 | 0.969 | 1.12 |

Independent numerical quadrature of the approximate second moments reproduces all four
targets, with a maximum residual of $1.11\times10^{-16}$ in decimal volatility when using
the unrounded inputs. This verifies the bootstrap arithmetic within the four-moment model.

Both constrained and relaxed 49-quote cases passed the existing pricing tolerances.
For the relaxed fit, the largest PDE/seed-24 MC difference is 2.18 standard errors;
the largest PDE/refined-MC difference is 2.57 batch standard errors. Maximum time-step,
spatial and domain shifts are respectively 0.600, 0.691 and 0.266 bp of discounted forward.
Density mass, forward expectation, negative-mass bounds and option-price bounds also pass.

The replay commands below regenerate the current calibrated case from the
bundled BTC option chain.

## Implementation in stochvolmodels

| Component | Responsibility |
|---|---|
| [calibrate_forward_var.py](calibrate_forward_var.py) | Study settings, two-parameter fit, fixed-parameter demonstrations and plots |
| [verify_pde.py](verify_pde.py) | Original-strike price comparisons, mesh refinements, seeded MC and numerical assertions |
| [option_chain.py](../../src/stochvolmodels/data/option_chain.py) | Mid vols, vegas, ATM interpolation and variance-swap targets |
| [vol_moments_ode.py](../../src/stochvolmodels/pricers/logsv/vol_moments_ode.py) | Approximate moments, integrated variance and loading bootstrap |
| [logsv_pricer.py](../../src/stochvolmodels/pricers/logsv_pricer.py) | Calibration objective, constraints, expansion pricing and MC pricing |
| [test_logsv_varswap_backbone.py](../../src/stochvolmodels/tests/test_logsv_varswap_backbone.py) | Known-loading recovery, every-expiry fits, ATM-floor units and invalid inputs |

The study requires the `research` extra, including `qis` for its plotting workflow.
Run from a source checkout with that extra already installed. On the Windows host used
for this study, use the external interpreter below; do not create an environment in OneDrive.

```powershell
# Calibrate and display the fitted smiles; uses the default mode in the driver.
python-m papers.forward_var.calibrate_forward_var

# Recalibrate and save the complete PDE/MC comparison for the current calibrated case.
python-m papers.forward_var.verify_pde --case calibrated --output forward_variance_calibrated.json

# Exercise the bootstrap and quotation-unit regression tests.
python-m pytest src/stochvolmodels/tests/test_logsv_varswap_backbone.py -q
```

The output JSON contains parameters, loadings, prices, Monte Carlo uncertainties and mesh
diagnostics. Expansion market-IV RMSE uses the calibration's fixed market-based Fourier
scale; the standalone price comparison uses the pricer's parameter-based default scale.
PDE and MC market-IV RMSEs are obtained by converting their saved prices to implied vols.
All these RMSEs use the per-expiry market-vega weights defined above; they are not separate
fields in the public replay runner's JSON. All 20 bootstrap regression tests passed in the
18 November verification. No recalibration is needed merely to read this specification.

The other driver modes serve different purposes:

- `VARSWAP_FIT` bootstraps the loading for an older fixed parameter set.
- `COMPARE_MODEL_VOLS_TO_MC` uses fixed `beta=0.24`, `volvol=1.14` and 31 display strikes
  per expiry. It does not automatically use the newly optimized pair.
- `CALIBRATE_4PARAM_MODEL` currently displays stored parameters; its optimizer call is
  commented out. It is not a fresh four-parameter calibration.

## Interpretation and limitations

The bootstrap and option-smile optimization solve different matching problems. Every
target variance-swap rate matches within the moment approximation, while vanilla implied
volatilities generally retain residuals. The ATM floor and finite option strip affect the
targets before calibration begins.

Four-moment closure, finite transform expansion, historical interval ordering, PDE
discretization, MC discretization and MC sampling are separate approximation sources.
Machine-precision bootstrap residuals do not remove any of them. In particular, the
reported historical expansion is an approximate pricing benchmark for the stated
calendar-time diffusion, not an exact solution of its PDE.

This calibration is a local two-parameter optimum for one historical snapshot with the
other parameters fixed and beta constrained to be nonnegative. It does not establish a
global optimum, a fit for negatively skewed markets, or performance on other dates.
The general package can represent negative beta, but this study's bounds exclude it.
The model is classical Markov LogSV; rough-volatility dynamics are outside this specification.

## See also

- [Paper workflows and comparison conventions](../README.md)
- [LogSV model guide](../../docs/logsv_model.md)
- [Calibration guide](../../docs/calibration.md)
- [Analytic versus Monte Carlo pricing](../../docs/analytic_vs_monte_carlo.md)
- [LogSV article source and correction notes](../logsv_model_with_quadratic_drift_ijtaf_2023/paper/README.md)

## References

- Sepp, A. and Rakhmonov, P. (2023), *Log-normal Stochastic Volatility Model with Quadratic
  Drift*, International Journal of Theoretical and Applied Finance 26(8), 2450003.
  [Published PDF retained in this repository](../logsv_model_with_quadratic_drift_ijtaf_2023/paper/log-normal-stochastic-volatility-model-with-quadratic-drift.pdf).
  This is the foundation for the volatility dynamics and finite moment/transform methods;
  the backbone workflow and numerical study settings are implementation choices documented here.
- [stochvolmodels citation metadata](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
- [qis](https://github.com/ArturSepp/QuantInvestStrats) and its
  [citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff),
  used by the study's plotting and figure-saving workflow.
