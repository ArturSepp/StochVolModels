---
myst:
  html_meta:
    description: >-
      Rough-logSV model analytics: conditional moments, fractional-kernel stability,
      quadratic drift and martingale conditions, with the experimental standalone pricer.
---

# Experimental rough-logSV pricing

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-18](https://github.com/ArturSepp/StochVolModels/commit/96fea18acc102468d0f80318529eca96aa3de106)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Rough-logSV is a stochastic volatility model with multiplicative volatility shocks,
fractional memory and quadratic mean reversion. Its **experimental** standalone interface
is `stochvolmodels.pricers.rough_logsv`, using `RoughLogSvParams` and `RoughLogSVPricer`.
The classical `LogSVPricer` has no rough-mode switch. This guide documents the development
implementation and the analytics of [Sepp and Rakhmonov's 18 November 2026 working
draft](#references); the draft is not presented as a published article.

Parameters, kernel fitting, history simulation and numerical controls have their canonical
home in `stochvolmodels.models.rough_logsv`. The compatibility facade above also exposes the
concrete pricer; the model package itself does not import it. Existing experimental paths
remain available without warnings, and numerical conventions are unchanged by this move.

## Overview

The model separates three questions: whether volatility is well defined, whether its
second moment stays controlled at long horizons, and whether the price is a true
martingale. The paper gives exact moment equations for **linear drift**, a time-averaged
second-moment bound for **quadratic drift**, and a sufficient martingale condition that
allows positive spot-volatility correlation.

These results have different implementation counterparts. Deterministic moment routines
check kernel approximations and simulation error. General nonlinear option prices use
history-preserving conditional simulation, with the full nonlinear Fourier/backward
partial differential equation (PDE) at $H=1/2$ as a reference. There is no claimed
closed-form transform or closed nonlinear moment system for the general rough model.

## Inputs, notation, and assumptions

### Model and conventions

Time is in years and volatility is an annualised decimal. Write $\alpha=H+1/2$.
The paper's core parameter class has $\sigma_0,\theta,\varepsilon>0$,
$\kappa_1,\kappa_2\ge0$ and $\beta\in\mathbb R$; the implementation also accepts
specified zero-noise and zero-level limits.

| Paper symbol | `RoughLogSvParams` field | Meaning and units |
|---|---|---|
| $\sigma_0$ | `sigma0` | Initial volatility, $\mathrm{year}^{-1/2}$ |
| $\theta$ | `theta` | Drift equilibrium level, $\mathrm{year}^{-1/2}$; not generally the nonlinear stationary mean |
| $\kappa_1$ | `kappa1` | Linear mean-reversion coefficient, $\mathrm{year}^{-\alpha}$ |
| $\kappa_2$ | `kappa2` | Quadratic damping coefficient, $\mathrm{year}^{-H}$ |
| $\beta$ | `beta` | Volatility shock loading on the price Brownian motion, $\mathrm{year}^{-H}$ |
| $\varepsilon$ | `epsilon` | Orthogonal volatility shock loading, $\mathrm{year}^{-H}$ |
| $H$ | `H` | Kernel roughness, $0<H\le1/2$ |

The derived properties are `vartheta` $=\sqrt{\beta^2+\varepsilon^2}$,
`rho` $=\beta/\vartheta$ when $\vartheta>0$, and
`martingale_margin` $=\kappa_2-\beta$. Fixing the same numerical volatility-of-volatility
at different $H$ does not hold its units or short-horizon effect fixed.

The paper uses a zero-rate price, or equivalently its stated forward-price convention,
and a unit price-volatility backbone. The package accepts European `C` and `P` quotes
under the money-market measure, strictly increasing integer-day ACT/365 maturities,
and positive forwards, strikes and discount factors. It rejects fractional days rather
than rounding them. The deterministic backbone extension discussed in the draft is
not part of this standalone interface.

## Methodology

### Volterra dynamics and the memory state

With independent Brownian motions $W^{(0)},W^{(1)}$, define
$\vartheta dW^\sigma=\beta dW^{(0)}+\varepsilon dW^{(1)}$ and
$b(x)=(\kappa_1+\kappa_2x)(\theta-x)$. The paper's model is

$$
\frac{dS_t}{S_t}=\sigma_t dW^{(0)}_t,\qquad
K(t)=\frac{t^{H-1/2}}{\Gamma(H+1/2)},
$$

$$
\sigma_t=\sigma_0+\int_0^t K(t-s)
\big[b(\sigma_s) ds+\vartheta\sigma_s dW^\sigma_s\big].
$$

The term "log-normal" describes the multiplicative volatility noise; it does not mean
that rough-logSV volatility is the exponential of a Gaussian Volterra process.
The memory available at time $t$ is the entire curve

$$
\Theta^t_s=\sigma_0+\int_0^t K(s-r)b(\sigma_r) dr
+\int_0^t K(s-r)\vartheta\sigma_r dW^\sigma_r,\qquad s\ge t.
$$

Although $\Theta^t_t=\sigma_t$, the future curve is generally neither
$\mathbb E[\sigma_s\mid\mathcal F_t]$ nor forward variance
$\mathbb E[\sigma_s^2\mid\mathcal F_t]$. Replacing it by the current scalar volatility
loses the rough model's history. See the draft's "Forward volatility curve" and
"Factor representation" results.

### Exact moment equations with linear drift

Set $\kappa_2=0$. The mean-reversion resolvent satisfies
$R=\kappa_1K-\kappa_1K*R$, where $*$ denotes time convolution. Define
$G=R/\kappa_1$ for $\kappa_1>0$, and $G=K$ when $\kappa_1=0$.
For the fractional kernel,

$$
G(t)=t^{\alpha-1}E_{\alpha,\alpha}(-\kappa_1t^\alpha),\qquad
E_{a,b}(z)=\sum_{j=0}^{\infty}\frac{z^j}{\Gamma(aj+b)}.
$$

Write $\psi_t(s)=\Theta^t_s-\theta$. The draft's "Conditional mean" and
"Conditional second moment" propositions give

$$
m_t(s)=\mathbb E[\sigma_s\mid\mathcal F_t]
=\theta+\psi_t(s)-\int_t^s R(s-r)\psi_t(r) dr,
$$

$$
v_t(s)=\mathbb E[\sigma_s^2\mid\mathcal F_t]
=m_t(s)^2+\vartheta^2\int_t^s G(s-r)^2v_t(r) dr.
$$

The second equation is deterministic conditional on the observed memory curve. At
time zero, $m_0(s)=\theta+(\sigma_0-\theta)E_{\alpha,1}(-\kappa_1s^\alpha)$.
These exact identities still require numerical evaluation of a resolvent or integral
equation; they are not zero-error numerical outputs. For $\kappa_2>0$, the first two
moments no longer close under these equations.

For $I_T=\int_0^T\sigma_s^2 ds$, the continuously sampled variance-swap strike is
$T^{-1}\mathbb E[I_T]=T^{-1}\int_0^T v_0(s) ds$. In the draft's decimal VIX convention,

$$
\mathrm{VIX}_T^2=\frac1\Delta\int_T^{T+\Delta}v_T(s) ds,
\qquad \Delta=30/365.
$$

With an exponential kernel approximation and linear drift, this conditional squared
VIX is a quadratic form in the complete factor state. A VIX future still requires
an outer expectation of its square root. The package moment targets below do not
constitute a standalone VIX futures or options pricer.

### Linear-drift stability and growth

For $\kappa_2=0$, $\kappa_1>0$ and $\sigma_0,\theta>0$, the draft's
"Long-horizon second-moment stability" proposition uses

$$
\gamma=\vartheta^2\int_0^\infty G(u)^2 du
=\vartheta^2 C(\alpha)\kappa_1^{-2H/\alpha},
$$

$$
C(\alpha)=-\frac{\cot(\pi\alpha/2)}{\alpha\sin(\pi/\alpha)},
\quad 1/2<\alpha<1,\qquad C(1)=1/2.
$$

The second moment is uniformly bounded in time exactly when $\gamma<1$, and its
limit is $\theta^2/(1-\gamma)$. It diverges when $\gamma\ge1$. This is a
second-moment criterion, not a proof of a stationary rough-volatility distribution.
When $\gamma>1$, the exponential growth rate $\lambda_*>0$ is the unique root of
$\vartheta^2\int_0^\infty e^{-\lambda_*u}G(u)^2 du=1$.

When $\kappa_1=\kappa_2=0$, the "Mittag-Leffler growth" corollary gives an independent
finite-horizon reference:

$$
\mathbb E[\sigma_t^2]=\sigma_0^2 E_{2H,1}(a_Ht^{2H}),\qquad
\mathbb E[I_T]=\sigma_0^2 T E_{2H,2}(a_HT^{2H}),\qquad
a_H=\frac{\vartheta^2\Gamma(2H)}{\Gamma(H+1/2)^2}.
$$

Large moments can be dominated by rare paths. An apparently stable Monte Carlo mean
does not replace these independent checks.

### Quadratic damping and the martingale condition

The draft's "Global well-posedness with quadratic drift" theorem covers the fractional
kernel and positive exponential mixtures with drift $c_0+c_1\sigma-d\sigma^2$,
$c_0,d\ge0$. It establishes a non-negative, non-explosive strong solution and finite
moments on finite horizons. **Strict positivity for singular fractional kernels
remains open.** Finite-horizon moments do not imply uniform long-horizon bounds.

For $\kappa_2>0$, let $A_T=\mathbb E[I_T]/T$ and $x^+=\max(x,0)$. The draft's
"Time-averaged second moment under quadratic drift" proposition gives, for the
fractional kernel,

$$
\kappa_2 A_T\le\kappa_1\theta
+(\kappa_2\theta-\kappa_1)^+\sqrt{A_T}
+\frac{\sigma_0 T^{-\alpha}}{\Gamma(2-\alpha)},
$$

$$
\limsup_{T\to\infty}A_T\le\theta\max\big(\theta,\kappa_1/\kappa_2\big).
$$

This controls the expected variance swap at long horizons for every $\vartheta$.
It does not establish a pointwise bound on $\sup_t\mathbb E[\sigma_t^2]$ for
$H<1/2$. A finite exponential kernel has its own bound involving its resolvent of
the first kind; the fractional bound must not be substituted for that finite-kernel
bound. At $H=1/2$, the quadratic-drift diffusion has a generalised inverse Gaussian
stationary density and all stationary moments when $\kappa_1,\kappa_2,\theta,\vartheta>0$.

Under the stopped share measure, the volatility drift becomes

$$
\kappa_1\theta+(\kappa_2\theta-\kappa_1)\sigma
-(\kappa_2-\beta)\sigma^2.
$$

Consequently $\kappa_2\ge\beta$ is **sufficient** for the price to be a true
martingale on every finite horizon, including when $\beta>0$ (the draft's
"Martingale property" theorem). This is distinct from the linear-drift moment
condition $\gamma<1$. The general converse remains open; failure of this sufficient
condition is not itself a proof of a martingale defect. The draft identifies the
special finite-kernel, linear-drift positive-correlation case covered by prior results.

## Worked example

The constant-kernel limit checks the analytics without a market dataset. Set
$H=1/2$, $\kappa_2=0$, $\sigma_0=\theta$ and $\kappa_1>0$. Then
$K(t)=1$, $G(t)=e^{-\kappa_1t}$ and $m_0(t)=\theta$. With
$a=\vartheta^2-2\kappa_1$, the second-moment equation reduces to
$v'(t)=2\kappa_1\theta^2+a v(t)$, giving

$$
v(t)=\theta^2\left(1+\vartheta^2\frac{e^{at}-1}{a}\right),
$$

$$
\frac{\mathbb E[I_T]}T=
\theta^2\left(1+\frac{\vartheta^2}{a}
\left(\frac{e^{aT}-1}{aT}-1\right)\right).
$$

At $a=0$, use the limits $v(t)=\theta^2(1+\vartheta^2t)$ and
$\mathbb E[I_T]/T=\theta^2(1+\vartheta^2T/2)$ rather than dividing by zero.
For $a<0$ the limiting second moment agrees with
$\theta^2/(1-\vartheta^2/(2\kappa_1))$; at and above the boundary it grows.

`markov_moment_targets` with the exact node `0.0` and weight `1.0` computes the
same continuous-time mean and variance-swap volatility by a matrix exponential.
`exact_linear_discrete_moments` instead computes the moments of the chosen splitting
scheme. Comparing the latter with the continuous-time result as the time step shrinks
tests time bias separately from Monte Carlo error. The classical limit does not by
itself qualify a rough kernel or a nonlinear option price.

## Implementation in stochvolmodels

### Analytics entry points

These are experimental direct-module APIs, not stable package-root exports. The
[API reference](api.md#research-rough-logsv-pricing) documents their signatures.

| Module and function | Computation | Qualification |
|---|---|---|
| `rough_logsv.kernel_fit.fractional_moment_targets` | Linear-drift mean, variance-swap volatility and leverage functional at supplied maturities | Product integration on three grids, with extrapolation only when differences converge monotonically; numerical targets, not certified error bounds |
| `rough_logsv.kernel_fit.markov_moment_targets` | The same continuous-time functionals for supplied exponential nodes and weights | Matrix-exponential solution of the finite-kernel moment equations; no fractional-kernel or time-step error is included |
| `rough_logsv.kernel_fit.fit_kernel_to_moments` | Fits nodes and weights to linear-drift targets | Kernel approximation, not calibration to market options; does not qualify nonlinear pricing |
| `rough_logsv.kernel_fit.kernel_mass_errors` | Relative errors in percent in the integrated kernel and squared kernel | Kernel diagnostics only |
| `rough_logsv.history.prepare_history` | Prepares a fixed kernel, time step and measure tilt | Requires the solver's positive linear-flow spectrum |
| `rough_logsv.history.simulate_history` | Evolves the complete factor state through all requested dates | Keeps history and uses adapted, beginning-of-step price integrals |
| `rough_logsv.history.exact_linear_discrete_moments` | First/second moments and integrated variance of that discrete scheme | Linear drift and ordinary measure only; not continuous-time moments |
| `rough_logsv.history.conditional_payoffs` | Bounded conditional option estimator in forward units | Requires half-tilted states and the pricer's sufficient martingale condition |
| `rough_logsv.history.raw_forward` | Unnormalised ordinary-measure conditional forward | Separate ordinary-measure states, with no empirical correction |

All module paths in this table start with `stochvolmodels.pricers.`.
The moment-target dictionaries use `MomentKey.MEAN`, `MomentKey.LEVEL` and
`MomentKey.LEVERAGE`, each aligned to the supplied maturities. `LEVEL` is
$\sqrt{\mathbb E[I_T]/T}$, not total variance or terminal volatility. `LEVERAGE` is
$\Lambda(T)=\int_0^T G(T-s)v_0(s) ds$; it is neither $\rho$ nor option skew.
In the linear-drift model,
$\operatorname{Cov}(\log(S_T/S_0)+I_T/2,\sigma_T)=\beta\Lambda(T)$.

The stability, growth-rate, nonlinear-bound and conditional-VIX formulas above are
paper results. This interface does not expose dedicated standalone solvers for all
of them. In particular, no nonlinear closed moment expansion is implemented.

### Pricing interface

For $0<H<1/2$, `RoughKernel` takes explicit nonnegative nodes and positive weights for
$K^N(t)=\sum_i w_i e^{-x_i t}$. Nodes have inverse-year units. Its target $H$ and
smoothing provenance do not certify approximation accuracy. The pricer does not
silently choose a historical one-, two- or three-factor rule. At $H=1/2$,
`RoughKernel.classical()` is exactly one zero node with unit weight; the default
pricer uses the full nonlinear PDE configured by `ClassicalPdeConfig`.

For rough pricing, supply both the pricing and comparison kernels, a `RoughMCConfig`
and a `DiscretePdeControl`. The control binds its independently validated mean to the
control parameters, discrete time step and exact quote coordinates. A continuous PDE
mean cannot replace the mean of the discrete control. Construction checks the supplied
validation errors against their allowances; callers retain the independent evidence.

The following fragment assumes these validated inputs and an `OptionChain` already exist.

<!-- fragment -->
```python
from stochvolmodels.pricers.rough_logsv import RoughLogSvParams, RoughLogSVPricer

params = RoughLogSvParams(H=0.30)
pricer = RoughLogSVPricer(
    kernel=kernel,
    comparison_kernel=comparison_kernel,
    mc_config=allocation,
    control=validated_control,
)
result = pricer.price_chain_result(chain, params, full_spreads=currency_spreads)
prices = result.prices
checks = result.diagnostics["checks"]
```

The fragment's parameters are illustrative, not fitted or qualified. `price_chain`
returns premiums grouped by expiry; `price_chain_result` returns a `RoughPricingResult`
with standard errors and diagnostics. `model_mc_price_chain` returns rough premiums
and standard errors without a routing flag. Use `price_chain_result` for the PDE.

Prices, errors and `full_spreads` are currency premiums including discounting.
`normalized_summary` uses undiscounted premiums divided by the forward. Divide a
currency premium by both the forward and discount factor to recover that convention.
The archived BTC/ETH examples use unit discounting. A coin-denominated quote spread
cannot be used directly as a currency spread.

### Sampling and diagnostics

`RoughMCConfig` freezes grids, antithetic-pair counts, independent scramble counts,
seed entropy, phase, chunk size, a time cap and a tape-workspace limit. Its defaults
are a small research allocation, not a qualification claim. Execution is serial;
the cap is checked between chunks, and the workspace limit is not a total host-RAM
guarantee. Use separate phases for exploratory and confirmation work.

Option estimates use half-tilted streams. With ordinary-measure
$I=\int\sigma_s^2 ds$ and $J=\int\sigma_s dW^\sigma_s$, the raw-forward
estimator is $\exp(\rho J-\rho^2 I/2)$. It is evaluated on separate streams and is
never used to renormalise option prices. The solver does not silently reset memory,
floor volatility or discard failed paths.

Uncertainty is computed across independent scrambled nets, with the control-mean
allowance added to the price interval. The pointwise 95% halfwidth must be at most
0.1 full spread; the absolute last time-step and paired smoothing changes plus their
halfwidths must each be at most 0.1 spread. The raw-forward halfwidth must be at most
0.0005, and its absolute deviation from one plus halfwidth at most 0.001. These
numerical checks qualify the evaluated point and coordinates, not an optimizer domain.

### Replaying selected BTC/ETH evidence

The draft's fixed $H$ illustrations use 45 BTC quotes on 30 June 2023 and 39 ETH
quotes on 26 October 2021, both at 08:00 UTC, across three maturities per asset.
The saved fits are at $H=0.30$ and $H=0.50$. These are retrospectively selected
positive-skew illustrations; $H$ is fixed rather than estimated.

The repository-only replay runner takes explicit local prerequisites and a fresh output root:

```console
python -B validation/rough_logsv/benchmark.py --paper-root <paper-directory> --evidence-root <evidence-directory> --output <new-output-directory>
```

It reconstructs saved prices and intervals, checks checkpoint hashes and raw forwards,
replays selected tapes through the package solver and archived reference, and recomputes
the nonlinear $H=1/2$ PDE. The paper's BTC/ETH figure producers use the standalone PDE
and revalue the saved rough checkpoints through the package estimator. These commands
do not start a new calibration or stochastic sampling allocation. The local paper and
evidence bundles are not included in the wheel, and the installed package does not
import paper scripts or private evidence.

## Interpretation and limitations

- The moment closure and $\gamma$ threshold apply to linear drift. Quadratic drift has
  different bounds; the classical stationary law is not a rough stationary-law formula.
- The draft proves non-negativity; strict positivity for singular kernels and the
  general martingale converse remain open. Numerical raw-forward evidence cannot
  settle either question.
- The price backbone is one. A deterministic multiplier in the price equation is
  not itself forward variance: the latter also contains $\mathbb E[\sigma_t^2]$.
  The draft's extension requires a time-dependent tilted-drift argument that it does
  not prove, and it is not implemented here.
- Positive kernel coefficients and the solver's positive-spectrum restriction are
  implementation requirements, not sufficient accuracy guarantees. Check kernel,
  smoothing, time-step, sampling and raw-forward errors separately.
- The BTC/ETH fits demonstrate selected risk-neutral positive-skew fits. They do not
  identify physical leverage, estimate $H$, establish global optimality or validate a
  wider parameter search. $H=0.20$ and wider calibration remain gated:
  `calibration_ready` and `parameter_domain_qualified` remain false, and
  `calibrate_model_params_to_chain` raises an explicit gate error.

## See also

### Migration and related interfaces

Remove `use_rough_mc` and `rough_scheme` from classical pricer calls. Both keywords now
raise a migration error, including `use_rough_mc=False`. The former
`CalibrationEngine.ROUGH_MC` name is retained only to explain the migration and gate.
Historical split-step functions and `LogSvParams.H/nodes/weights` remain for reproduction.

- [Experimental rough-logSV API](api.md#research-rough-logsv-pricing)
- [Classical log-normal SV analytics](logsv_model.md)
- [Classical moment and stationary-distribution analytics](volatility_distribution_and_moments.md)
- [Classical Monte Carlo schemes](monte_carlo_simulation.md)
- [Option-chain conventions](option_chains_and_conventions.md)
- [Classical calibration](calibration.md)
- [Paper provenance and replication](reproducing_the_papers.md#rough-logsv-authors-working-draft)

## References

- Sepp, A. and Rakhmonov, P. (2026). *Rough Log-normal Stochastic Volatility with
  Quadratic Drift: Second-moment stability and the martingale property under positive
  correlation*. Authors' working draft, **18 November 2026**. Source used for this
  guide: `papers/rough_volterra_lognormal_sv_model_2026/paper/rough_volterra_lognormal.tex`
  in the authors' local research checkout. Sections "Model", "Moments and the stability
  parameter", "Martingale property under positive correlation", and "Numerical
  illustrations". References above use named results from that draft, not equation
  numbers from the published classical paper. The draft is not bundled with the
  package; no public paper URL or DOI is asserted here.
- Sepp, A. and Rakhmonov, P. (2023). *Log-normal stochastic volatility model with
  quadratic drift*. International Journal of Theoretical and Applied Finance 26(8),
  2450003. [DOI: 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
  Published classical $H=1/2$ reference.
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
