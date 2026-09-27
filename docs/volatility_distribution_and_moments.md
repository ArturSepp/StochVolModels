---
myst:
  html_meta:
    description: >-
      The steady-state distribution of volatility in the log-normal stochastic volatility model
      with quadratic drift: moments, skewness, kurtosis of returns, the truncated moment system
      over time and the expected quadratic variance, with stochvolmodels code and regenerated
      paper figures.
---

# Steady-state distribution, moments and expected quadratic variance

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

In the log-normal stochastic volatility (SV) model with quadratic drift, volatility has a
stationary distribution in closed form: a generalised inverse Gaussian law whose right tail is
controlled by the quadratic mean-reversion rate $\kappa_2$. Its moments give the skewness of
volatility and the kurtosis of short-horizon returns. Moments at a finite horizon, and the expected
quadratic variance (QV) that prices a variance swap, follow from a linear system of ordinary
differential equations that the package solves by matrix exponentiation after truncation. This
article derives these results following Sepp and Rakhmonov (2023, Sections 3.5 to 3.7) and checks
each of them against an independent computation.

## Overview

The quadratic drift keeps the model a martingale when the volatility beta $\beta$ is positive (see
the [conventions page](option_chains_and_conventions.md#measures-mma-and-inverse)). It also changes
the distribution of volatility. The term $-\kappa_2 \sigma^2$ in the drift pulls high volatility
back faster than the linear term does, so large volatilities become rarer and returns less
fat-tailed. This article measures the effect three ways:

1. the stationary density of volatility and its moments, which set the skewness of volatility and
   the excess kurtosis of short-horizon returns;
2. the moments of volatility at a finite horizon, from a truncated moment system;
3. the expected QV, which is the fair strike of a continuously monitored variance swap.

These results help to choose $\kappa_2$ against the observed tail of implied or realised
volatility, to check that a calibrated parameter set has finite moments, and to price variance
swaps without simulation.

## Inputs, notation, and assumptions

| Symbol | Meaning | Convention |
|---|---|---|
| $\sigma_t$ | Volatility | Annualised decimal |
| $\theta$ | Mean volatility, `theta` | $\theta > 0$ |
| $\kappa_1$, $\kappa_2$ | Linear and quadratic mean-reversion rates, `kappa1`, `kappa2` | Per year, non-negative |
| $\vartheta^2$ | Total vol-of-vol, $\beta^2 + \varepsilon^2$ | Per year |
| $Y_t$ | Mean-adjusted volatility, $\sigma_t - \theta$ | Annualised decimal |
| $\kappa$ | Effective mean-reversion rate, $\kappa_1 + \kappa_2 \theta$ | Per year |
| $k^{\ast}$ | Truncation order of the moment system, `n_terms` | Integer |
| $I_\tau$ | Quadratic variance, $\int_0^\tau \sigma_t^2 dt$ | Variance times years |
| $\widehat{I}_\tau$ | Annualised expected QV, $E[I_\tau] / \tau$ | Annualised variance |

The dynamics are those of Eq. (3.12) under the money-market-account (MMA) measure. The price shock
enters the volatility through $\beta$, but the law of volatility alone depends on $\beta$ and the
residual vol-of-vol $\varepsilon$ only through $\vartheta^2$. The parameters satisfy
Assumption 3.1 of the paper, $\kappa_1 \ge 0$, $\kappa_2 \ge 0$ and $\theta > 0$, under which
volatility stays positive and finite (Theorem 3.2). Units and the mapping of symbols to code are
on the [conventions page](option_chains_and_conventions.md#notation-paper-and-code).

## Methodology

### The steady-state density

Under Eq. (3.12) volatility solves

$$
d\sigma_t = (\kappa_1 + \kappa_2 \sigma_t)(\theta - \sigma_t) dt + \vartheta \sigma_t dW_t ,
$$

where $W$ is a Brownian motion that combines the two shocks of the model. Its stationary density
$G$ solves the Fokker-Planck equation, Eq. (3.37):

$$
\frac{1}{2} \vartheta^2 \frac{d^2}{d\sigma^2} \left( \sigma^2 G \right) - \frac{d}{d\sigma} \left( (\kappa_1 + \kappa_2 \sigma)(\theta - \sigma) G \right) = 0 .
$$

The solution, Eq. (3.38), is the generalised inverse Gaussian density (Jørgensen, 1982):

$$
G(\sigma) = c \sigma^{\eta - 1} \exp\left( -\frac{q}{\sigma} - b \sigma \right), \quad \sigma > 0,
$$

$$
q = \frac{2 \kappa_1 \theta}{\vartheta^2}, \quad b = \frac{2 \kappa_2}{\vartheta^2}, \quad \eta = \frac{2 (\kappa_2 \theta - \kappa_1)}{\vartheta^2} - 1, \quad c = \frac{(b / q)^{\eta / 2}}{2 K_\eta \left( 2 \sqrt{q b} \right)} ,
$$

where $K_\eta$ is the modified Bessel function of the second kind. The quadratic rate enters only
through $b$, and the factor $e^{-b \sigma}$ gives an exponential right tail. With a linear drift,
$\kappa_2 = 0$, the density is inverse gamma with shape $\alpha = -\eta = 1 + 2 \kappa_1 / \vartheta^2$,
scale $q$ and normalising constant $c = q^{\alpha} / \Gamma(\alpha)$. Its right tail decays only as
the power $\sigma^{-\alpha - 1}$.

### Moments, skewness and the kurtosis of returns

The moments of the steady state are, Eq. (3.39),

$$
m(r) = E[\sigma^r] = \left( \frac{q}{b} \right)^{r / 2} \frac{K_{\eta + r} \left( 2 \sqrt{q b} \right)}{K_\eta \left( 2 \sqrt{q b} \right)}, \quad b > 0 .
$$

For $\kappa_2 = 0$ they are $m(r) = q^r \Gamma(\alpha - r) / \Gamma(\alpha)$ for $r < \alpha$, and
infinite otherwise: with a linear drift only moments of order below $1 + 2 \kappa_1 / \vartheta^2$
exist. The skewness of volatility is

$$
\frac{m(3) - 3 m(1) m(2) + 2 m(1)^3}{\left( m(2) - m(1)^2 \right)^{3 / 2}} .
$$

Over a short period $\delta t$, returns are Gaussian conditional on volatility, with variance
$\sigma^2 \delta t$ (Barndorff-Nielsen and Shiryaev, 2015). Their unconditional second and fourth
moments are $\delta t m(2)$ and $3 \delta t^2 m(4)$, so the excess kurtosis of returns is,
Eq. (3.44),

$$
k = \frac{3 m(4)}{m(2)^2} - 3 = \frac{3 K_{\eta + 4} \left( 2 \sqrt{q b} \right) K_\eta \left( 2 \sqrt{q b} \right)}{K_{\eta + 2} \left( 2 \sqrt{q b} \right)^2} - 3, \quad b > 0 .
$$

For $\kappa_2 = 0$ the inverse gamma moments give

$$
k = \frac{3 (\alpha - 1)(\alpha - 2)}{(\alpha - 3)(\alpha - 4)} - 3 ,
$$

which is finite only when $\alpha > 4$, that is when $\kappa_1 / \vartheta^2 > 3/2$. As $\kappa_2$
grows, both the skewness of volatility and the kurtosis of returns fall: $\kappa_2$ acts as a
dampening parameter on the tails.

The printed normalising constant of Eq. (3.38) for $\kappa_2 = 0$ and the printed $\kappa_2 = 0$
branch of Eq. (3.44) differ from the forms above, which follow from the inverse gamma density. The
canonical script checks the forms given here against numerical integration. The finiteness
condition $\kappa_1 / \vartheta^2 > 3/2$ agrees with the paper, and the published figures use only
the $\kappa_2 > 0$ branch.

### Moments at a finite horizon

The mean-adjusted volatility follows, Eq. (3.33),

$$
dY_t = -\left( \kappa Y_t + \kappa_2 Y_t^2 \right) dt + \vartheta (Y_t + \theta) dW_t .
$$

By Itô's lemma, its moments $\bar{m}^{(n)}(\tau) = E[Y_\tau^n]$ satisfy, for $n \ge 1$,

$$
\partial_\tau \bar{m}^{(n)} = c(n) \theta^2 \bar{m}^{(n-2)} + 2 c(n) \theta \bar{m}^{(n-1)} + \left( c(n) - n \kappa \right) \bar{m}^{(n)} - n \kappa_2 \bar{m}^{(n+1)} ,
$$

with $c(n) = \vartheta^2 n (n - 1) / 2$, $\bar{m}^{(0)} = 1$ and $\bar{m}^{(n)}(0) = Y_0^n$. The quadratic drift couples each moment to
the next one, so the system is infinite (Eqs. (3.45) to (3.47)). Truncating at order $k^{\ast}$
and freezing $\bar{m}^{(k^{\ast} + 1)}$ at its initial value $Y_0^{k^{\ast} + 1}$ gives the finite
linear system of Eq. (3.48),

$$
\partial_\tau M = \Lambda M + C, \quad M(0) = \left( Y_0, Y_0^2, \ldots, Y_0^{k^{\ast}} \right)^\top ,
$$

where the rows of the $k^{\ast} \times k^{\ast}$ matrix $\Lambda$ are the coefficients of the
recursion and $C = \left( 0, c(2) \theta^2, 0, \ldots, 0, -k^{\ast} \kappa_2 Y_0^{k^{\ast} + 1} \right)^\top$.
Its solution, Eq. (3.49), is

$$
M(\tau) = e^{\Lambda \tau} M(0) + \Lambda^{-1} \left( e^{\Lambda \tau} - I \right) C .
$$

The truncation is admissible when every eigenvalue of $\Lambda$ has a negative real part. With a
linear drift, $\kappa_2 = 0$, the matrix is lower triangular and the solution is exact for every
moment up to $k^{\ast}$ (Remark 3.1 of the paper).

### Expected quadratic variance

The annualised expected QV is, Eqs. (3.52) and (3.53),

$$
\widehat{I}_\tau = \frac{1}{\tau} E\left[ \int_0^\tau \sigma_t^2 dt \right] = \frac{1}{\tau} \left( \widehat{M}_2(\tau) + 2 \theta \widehat{M}_1(\tau) \right) + \theta^2 ,
$$

where $\widehat{M}(\tau) = \int_0^\tau M(s) ds$ holds the integrated moments of Eq. (3.54),

$$
\widehat{M}(\tau) = \Lambda^{-1} \left( e^{\Lambda \tau} - I \right) M(0) + \Lambda^{-1} \left( \Lambda^{-1} \left( e^{\Lambda \tau} - I \right) - \tau I \right) C .
$$

$\widehat{I}_\tau$ is the model's fair strike of a continuously monitored variance swap. As the
horizon grows, it converges to the stationary second moment $m(2)$.

## Worked example

The example uses the parameters of the paper's Fig. 1: $\theta = 1$, $\vartheta = 1.5$ and
$\kappa_1 = 4$, with $\kappa_2$ equal to 0, 4 or 8. Every block below is an excerpt of
[`examples/docs/volatility_distribution_and_moments.py`](../examples/docs/volatility_distribution_and_moments.py),
which asserts every number quoted here.

```python
def gig_parameters(params: svm.LogSvParams) -> tuple:
    """Return (q, b, eta) of the generalised inverse Gaussian steady state, Eq. (3.38)."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    q = 2.0 * params.kappa1 * params.theta / vartheta2
    b = 2.0 * params.kappa2 / vartheta2
    eta = 2.0 * (params.kappa2 * params.theta - params.kappa1) / vartheta2 - 1.0
    return q, b, eta


def steady_state_density(sigma: np.ndarray, params: svm.LogSvParams) -> np.ndarray:
    """Steady-state density G(sigma); inverse gamma with shape -eta and scale q when kappa2 = 0."""
    q, b, eta = gig_parameters(params)
    if b > 0.0:
        c = (b / q) ** (eta / 2.0) / (2.0 * sps.kv(eta, 2.0 * np.sqrt(q * b)))
    else:
        c = q ** (-eta) / sps.gamma(-eta)
    return c * sigma ** (eta - 1.0) * np.exp(-(q / sigma + b * sigma))
```

```python
def steady_state_moment(r: float, params: svm.LogSvParams) -> float:
    """Moment E[sigma^r] under the steady state, Eq. (3.39); infinite when it does not exist."""
    q, b, eta = gig_parameters(params)
    if b > 0.0:
        z = 2.0 * np.sqrt(q * b)
        return (q / b) ** (r / 2.0) * sps.kv(eta + r, z) / sps.kv(eta, z)
    return q ** r * sps.gamma(-eta - r) / sps.gamma(-eta) if r < -eta else np.inf
```

Each density integrates to one within $10^{-9}$, and the four Bessel moments agree with numerical
integration of the density to a relative $10^{-8}$. The statistics of the steady state are:

| $\kappa_2$ | Mean of $\sigma$ | Standard deviation | Skewness of $\sigma$ | Excess kurtosis of returns |
|---|---|---|---|---|
| 0 | 1.00 | 0.626 | 4.11 | 28.5 |
| 4 | 0.936 | 0.351 | 1.15 | 2.05 |
| 8 | 0.942 | 0.290 | 0.837 | 1.26 |

With a linear drift the mean is exactly $\theta$, but the inverse gamma tail, of shape
$\alpha = 4.56$, gives a volatility skewness of 4.11 and an excess kurtosis of returns of 28.5.
A quadratic rate of $\kappa_2 = 4$ cuts the kurtosis to 2.05. It also pulls the mean slightly
below $\theta$, because the reverting force $(\kappa_1 + \kappa_2 \sigma)(\theta - \sigma)$ grows
with $\sigma$ and so acts more strongly above the mean than below it.

[![Steady-state densities of volatility for three quadratic mean-reversion rates, and the skewness of volatility and excess kurtosis of returns as functions of the quadratic rate.](images/steady_state_density.png)](images/steady_state_density.png)

*IJTAF Fig. 1, regenerated. (A) The steady-state density, Eq. (3.38), for $\kappa_1 = 4$ and
$\kappa_2 = 0, 4, 8$. (B) The skewness of volatility and (C) the excess kurtosis of returns,
Eq. (3.44), as functions of $\kappa_2$ for $\kappa_1 = 1, 4, 8$. $\theta = 1$ and
$\vartheta = 1.5$. Drawn by the paper module `steady_state_pdf.py`; panels arranged for the web.*

The long-horizon limit of the truncated moment system must reproduce the stationary moments. This
is an independent check, because the two results come from different equations:

```python
def truncated_vs_stationary(kappa2: float, n_terms: int, horizon: float = 20.0) -> dict:
    """Moments of Y = sigma - theta from the truncated system (3.49) at a long horizon."""
    params = svm.LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=kappa2,
                             beta=0.0, volvol=1.0)
    ode = compute_analytic_vol_moments(params=params, t=horizon, n_terms=n_terms)
    m1, m2 = steady_state_moment(1, params), steady_state_moment(2, params)
    exact = (m1 - params.theta, m2 - 2.0 * params.theta * m1 + params.theta ** 2)
    return {"ode": ode[:2], "exact": np.array(exact)}
```

With $\kappa_2 = 0$ the truncated system is exact, and it matches the stationary moments to
$10^{-10}$. With $\kappa_2 = 4$ and $\vartheta = 1$, the parameters of the paper's Fig. 2:

| Moment of $Y$ | Stationary, Eq. (3.39) | $k^{\ast} = 4$ | $k^{\ast} = 8$ |
|---|---|---|---|
| $E[Y]$ | -0.0299 | -0.0282 | -0.0299 |
| $E[Y^2]$ | 0.0597 | 0.0563 | 0.0598 |

Truncation at order four understates both moments in absolute value by between 5% and 6%; order
eight is within 0.2%. The paper compares both orders with Monte Carlo simulation in its Fig. 2,
regenerated below, and describes order four as consistent with the simulation for the first two
moments and order eight for the first four. With 100,000 paths the Monte Carlo intervals are
narrow enough to detect the order-four bias in the second moment that the comparison above
quantifies; the order-eight curves follow the simulation more closely.

[![Four moments of mean-adjusted volatility over one and a half years from the truncated system of order four and order eight, against Monte Carlo estimates with confidence intervals.](images/vol_moments_vs_mc.png)](images/vol_moments_vs_mc.png)

*IJTAF Fig. 2, regenerated. Moments of $Y = \sigma - \theta$ from the truncated system,
Eq. (3.49), with $k^{\ast} = 4$ (top) and $k^{\ast} = 8$ (bottom), against Monte Carlo estimates
with 95% intervals (dots). $\sigma_0 = 1.5$, $\theta = 1$, $\kappa_1 = \kappa_2 = 4$, $\beta = 0$
and $\varepsilon = 1$. These are the parameters of the paper's code, and they reproduce the
published figure; the published caption states $\vartheta = 1.5$. 100,000 paths and 540 time
steps per year, as the paper module runs them; seed 37 for both NumPy's and numba's generators.*

The expected QV starts from $\sigma_0 = 1.5$, above the mean, and decays towards the stationary
second moment:

```python
def expected_qvar(ttm: float) -> dict:
    """Annualised expected quadratic variance, Eq. (3.53), for the three cases of Fig. 3."""
    return {kappa2: svm.compute_analytic_qvar(
                params=svm.LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=kappa2,
                                       beta=0.0, volvol=1.5),
                ttm=ttm, n_terms=4)
            for kappa2 in FIG1_KAPPA2S}
```

At two years, $\widehat{I}_2$ is 1.55, 1.07 and 1.02 for $\kappa_2$ equal to 0, 4 and 8, against
stationary values $m(2)$ of 1.39, 1.00 and 0.971.

[![Annualised expected quadratic variance over two years for three quadratic mean-reversion rates and two initial volatilities, against Monte Carlo estimates.](images/expected_qvar_vs_mc.png)](images/expected_qvar_vs_mc.png)

*IJTAF Fig. 3, regenerated. The annualised expected QV, Eq. (3.53) with $k^{\ast} = 4$, for
$\kappa_1 = 4$ and $\kappa_2 = 0, 4, 8$, starting from $\sigma_0 = 1.5$ (upper curves) and
$\sigma_0 = 0.5$ (lower curves), with $\theta = 1$ and $\vartheta = 1.5$, against Monte Carlo
estimates (dots). The error bars span $\pm 2 \times 1.96$ standard errors, as the paper module draws
them; the published caption calls them 95% intervals. 25,000 paths instead of the paper's 100,000,
which leaves the analytic curves unchanged and widens the bars; 720 time steps per year, as the
paper module runs a two-year horizon; seed 37.* The script also simulates 20,000 volatility paths
with daily steps and a fixed seed: the first two moments of $Y$ at one year and the one-year
expected QV each lie within three standard errors of the analytic values.

## Implementation in stochvolmodels

| Name | Role |
|---|---|
| `compute_analytic_qvar` | Annualised expected QV, Eq. (3.53); a stable export |
| `stochvolmodels.pricers.logsv.vol_moments_ode.compute_analytic_vol_moments` | Truncated moment system, Eq. (3.49), or the integrated moments of Eq. (3.54) with `is_qvar=True`; an internal module function |
| `LogSvParams.get_vol_moments_lambda` | The matrix $\Lambda$ of Eq. (3.48) |
| `LogSvParams.assert_vol_moments_stability` | Prints whether every eigenvalue of $\Lambda$ has a negative real part; despite its name it does not raise |
| `LogSVPricer.simulate_vol_paths` | Volatility paths for Monte Carlo checks |

The package has no function for the steady-state density; the canonical script evaluates the
closed form with SciPy. Two details of `simulate_vol_paths` matter for reproducible checks:

- `nb_steps` is a number of steps per year. Pass it explicitly: when it is omitted, the method uses
  $\lceil 360 \tau \rceil$ as the per-year count, so a one-and-a-half-year simulation takes 540
  steps per year.
- It draws its increments with NumPy's global generator, which
  `stochvolmodels.utils.funcs.set_seed` does not reach. Pass `brownians` drawn from a local
  generator, as the canonical script does, or seed NumPy.

Run the example from a checkout with `python examples/docs/volatility_distribution_and_moments.py`.
The figures are regenerated by `scripts/docs_analytics/logsv_model.py`, which calls the paper
module under
[`papers/logsv_model_with_quadratic_drift`](https://github.com/ArturSepp/StochVolModels/tree/main/papers/logsv_model_with_quadratic_drift);
the [analytics gallery](analytics_gallery.md) records their parameters and provenance.

## Interpretation and limitations

- **Measure.** The stationary law is that of the MMA measure. Under the inverse measure the
  quadratic rate becomes $\kappa_2 - \beta$ (Section 3.3 of the paper), and under the physical
  measure it is shifted by the volatility risk premia (Section 3.1). A fit to observed volatility
  uses the physical parameters.
- **What the kurtosis measures.** It is the kurtosis of short-horizon returns when volatility is
  drawn from its stationary law. It ignores the dependence between returns and volatility changes
  and says nothing about the conditional distribution at a given date.
- **Truncation error.** It depends on $k^{\ast}$, the horizon and the distance of $\sigma_0$
  from $\theta$, because the closure freezes the moment of order $k^{\ast} + 1$ at its initial
  value. Check the eigenvalues of $\Lambda$, and compare two truncation orders before relying on a
  higher moment.
- **Heavy tails without the quadratic drift.** With $\kappa_2 = 0$, moments of order $\alpha$ and
  above are infinite. The expected QV needs $\alpha > 2$, and the kurtosis of returns
  $\alpha > 4$.
- **Printed equations.** Use the $\kappa_2 = 0$ forms given here, as explained above. Equation
  numbers refer to the published PDF.

## See also

- [Option chains, notation and conventions](option_chains_and_conventions.md)
- [Log-normal stochastic volatility model with quadratic drift](logsv_model.md)
- [Analytic versus Monte Carlo validation](analytic_vs_monte_carlo.md)
- [Bitcoin options: MMA, inverse and QV valuation](app_bitcoin_options.md)
- [API reference](api.md)

## References

- Barndorff-Nielsen, O. E. and Shiryaev, A. N. (2015). *Change of Time and Change of Measure*.
  World Scientific.
- Jørgensen, B. (1982). *Statistical Properties of the Generalized Inverse Gaussian Distribution*.
  Springer.
- Sepp, A. and Rakhmonov, P. (2023). Log-normal stochastic volatility model with quadratic drift.
  *International Journal of Theoretical and Applied Finance* 26(8), 2450003.
  [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
