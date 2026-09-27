---
myst:
  html_meta:
    description: >-
      Case study: the Nelson-Siegel HJM model with log-normal stochastic volatility on USD
      swaptions of August 2023 and 3M SOFR futures options, with Figures 5 to 9 of Sepp and
      Rakhmonov (2025) regenerated from the repository's modules, the accuracy of the affine
      expansion against Monte Carlo, and where the modules differ from the published article.
---

# USD swaptions and SOFR futures options

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Sepp and Rakhmonov (2025, Sections 7.5 to 7.7) calibrate the factor HJM model with a log-normal
stochastic volatility driver to USD swaptions and to options on 3M SOFR futures, and check the
first-order affine expansion against Monte Carlo simulation. The market data and the parameters are
hard-coded in the modules that accompany the article, `papers/sv_for_factor_hjm/calibration_fig_5_6_7.py`
and `calibration_fig_8_9.py`. This case study regenerates Figures 5 to 9 of the article from those
modules and quotes the results by section. The parameters the modules record differ from the
tables of the published article; the page says which results each set reproduces.

```{note}
`stochvolmodels.pricers.factor_hjm` is an experimental research surface (see
[stability tiers](option_chains_and_conventions.md#stability-tiers)); the model and the pricer are
explained in [stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md).
```

## Overview

The study asks three questions. Does one set of piecewise-constant parameters per expiry fit the
smiles of swaptions on 2y, 5y and 10y swaps at once? Is the drift-frozen, first-order expansion
accurate against a simulation of the full dynamics, at long expiries and under parameter shocks?
And does the same model fit the steep, convex smiles of short-dated SOFR options?

## Study design and data

**Swaptions.** Normal implied volatilities of USD swaptions of 18 August 2023 (the article says
mid-August 2023), for tenors 2y, 5y and 10y. The study uses the expiries 1y, 2y, 3y and 5y and the
five strikes around the money of each slice, 60 options. The module moves every slice to the
forward swap rate of its flat 4.3% zero curve, 4.3938%, keeping the distance of each strike to the
forward, and records one volatility per option: the surface has no bid-ask spreads.

**SOFR options.** Normal volatilities of 3M SOFR futures options with 75 and 103 days to expiry,
17 and 15 strikes, with a futures rate of 432.32 bp for both. The module fits a normal SABR smile to
each slice and re-strikes it at five deltas, from the 25-delta put to the 25-delta call; Figure 8
compares the model with these five refitted volatilities. The article dates the data October 2023
and names the expiries 84 and 175 days in the text of Section 7.7; its Table 3, the caption of
Figure 8 and the module use 75 and 103 days.

**Monte Carlo.** The article simulates 200,000 paths with daily steps. The swaption comparisons here
use 200,000 paths at 360 steps a year, in 20 batches of 10,000 with seeds 1 to 20, because the
module's `calc_mc_vols` holds every increment in memory; one batch with its fixed seed reproduces it
exactly. The SOFR comparison uses the module's own simulation, $2^{17}$ paths with seed 20.

## Configuration

Both modules use the Nelson-Siegel basis with $\lambda = 0.55$, key tenors 2y, 5y and 10y, the
historical correlation of the 2y, 5y and 10y yields, and $\sigma_0 = \theta = 1$.

| Item | Published article | Module |
|---|---|---|
| Swaption mean reversion | $\kappa_1 = 0.25$, $\kappa_2 = 0.5$ (Section 7.5) | $\kappa_1 = \kappa_2 = 0.25$ |
| Swaption parameters per expiry | Table 1 | different in every row, for example 1y betas $(0.0152, 0.1063, 0.6667)$ against $(0.0705, -0.3236, 0.7477)$ |
| SOFR mean reversion | $\kappa_1 = 0.5$, $\kappa_2 = 1.0$ | the same |
| SOFR parameters per expiry | Table 3 | different, for example the 75d beta $-0.567$ against $-1.995$ |
| Figure 6 expiry | 10y in the text, 5y in the caption | 5y |
| Figure 7 expiry | 5y in the caption | 2y |
| Monte Carlo paths | 200,000 | 50,000 (Figures 6 and 7), $2^{17}$ (Figure 9) |

The strike axes of the published Figures 6 and 7 match the module's 5y and 2y expiries. The
companion script
[`examples/docs/app_swaptions_and_sofr_options.py`](../examples/docs/app_swaptions_and_sofr_options.py)
asserts every number quoted below; it builds either parameter set:

```python
def swaption_params(published: bool = False):
    """Parameters recorded in the module, or those of Table 1 of the article."""
    params = swaption_module.getCalibRateLogSVParams()["USD"]
    if published:
        for idx, expiry in enumerate(EXPIRY_IDS):
            a, beta, beta0 = TABLE_1[expiry]
            params.update_params(idx=idx, A_idx=np.array(a), beta_idx=np.array(beta),
                                 volvol_idx=beta0, kappa2=0.5)
    return params
```

and compares the expansion with the simulation on 21 strikes across each tenor's market range:

```python
def expansion_vs_monte_carlo(params, expiry: float, nb_batch: int = 20) -> dict:
    """The expansion against pooled Monte Carlo on 21 strikes spanning each tenor's market range."""
    chain = swaption_chain()
    idx = int(np.argmin(np.abs(chain.ttms - expiry)))
    strikes = [np.linspace(k[idx][0], k[idx][-1], 21) for k in chain.strikes_ttms]
    mc = pooled_mc_vols(params, expiry, chain.tenors, strikes, nb_batch=nb_batch)
    _, vols = logsv_chain_de_pricer(
        params=params, t_grid=generate_ttms_grid(np.array([expiry])), ttms=np.array([expiry]),
        forwards=[np.array([f]) for f in mc["forward"]], strikes_ttms=[[k] for k in strikes],
        optiontypes_ttms=[np.repeat("C", 21)], expansion_order=ExpansionOrder.FIRST)
    mc["strikes"], mc["expansion"] = strikes, [np.asarray(v[0]) for v in vols]
    mc["inside"] = [int(np.sum((e >= d) & (e <= u)))
                    for e, u, d in zip(mc["expansion"], mc["up"], mc["down"])]
    return mc
```

## Results

**Swaption fit (Section 7.5, Figure 5).** With the module's parameters the root-mean-square error
of the model against the market is 1.15, 0.48 and 0.70 bp for the 2y, 5y and 10y tenors, the largest
error 3.35 bp at the lowest strike of the 2y x 2y swaption. With the parameters of Table 1 and
$\kappa_2 = 0.5$ it is 2.29, 1.38 and 1.95 bp. Condition (33) on the quadratic mean reversion holds
for all twelve expiry and tenor pairs.

[![Model normal volatilities against USD swaption quotes for twelve expiry and tenor pairs.](images/rdr_fig5_swaption_fit.png)](images/rdr_fig5_swaption_fit.png)

*Paper replication: RDR Figure 5 regenerated from `calibration_fig_5_6_7.py` with the module's
parameters, which differ from Table 1. Rows are the 2y, 5y and 10y tenors, columns the 1y, 2y, 3y
and 5y expiries; market quotes of 18 August 2023 (dots) and the first-order expansion (line).*

**Expansion against Monte Carlo at 5y (Figure 6).** The expansion lies inside the 95% interval of
the simulation at 19, 15 and 0 of 21 strikes for the 2y, 5y and 10y tenors. For the 10y tenor it is
0.55 to 1.46 bp above the simulated volatility at every strike, where the article describes the two
as consistent. Part of the gap may come from the simulation: its forward swap rates are 0.66 to
0.72 bp below the model's, 1.3 to 2.1 standard errors.

[![First-order expansion against Monte Carlo 95% intervals for 5y-expiry swaptions on 2y, 5y and 10y swaps.](images/rdr_fig6_swaption_mc.png)](images/rdr_fig6_swaption_mc.png)

*Paper replication: RDR Figure 6 regenerated with the module's parameters, 5y expiry; 200,000 paths
at 360 steps a year, seeds 1 to 20.*

**Accuracy under shocks (Section 7.6, Table 2, Figure 7).** From the base scenario of Table 2, the
four scenarios keep it, add 0.02 to the yield volatilities, multiply the residual vol-of-vol by
four, and multiply the betas by $-2$. At the 2y expiry the expansion lies inside the interval at all 63
strikes of the first two scenarios, and at 9, 11 and 10 and at 9, 7 and 7 of 21 per tenor in the
last two, where its average distance above the simulation reaches 0.94 and 0.71 bp. This agrees
with the article: the expansion is less accurate when the vol-of-vol is high or the betas large.

[![First-order expansion against Monte Carlo intervals for 2y-expiry swaptions under four parameter scenarios.](images/rdr_fig7_shock_scenarios.png)](images/rdr_fig7_shock_scenarios.png)

*Paper replication: RDR Figure 7 regenerated from `get_scenarios` of the module, 2y expiry as in the
module and the published strike axes; 200,000 paths per scenario, seeds 1 to 20.*

**SOFR fit (Section 7.7, Figure 8).** With the module's parameters the root-mean-square error of the
model against the SABR refit is 0.03 and 0.48 bp at 75 and 103 days, the model lies inside a
one-tick price band at all five strikes of both slices, and its error against the raw quotes is
1.46 and 2.07 bp. With the parameters of Table 3 the errors against the refit are 18.84 and
12.33 bp, so the module's values, not the table's, produce the fit the article reports.

[![Model normal volatilities against the SABR refit and the quotes of 3M SOFR options at 75 and 103 days.](images/rdr_fig8_sofr_fit.png)](images/rdr_fig8_sofr_fit.png)

*Paper replication: RDR Figure 8 regenerated from `calibration_fig_8_9.py` with the module's
parameters; quotes (grey), the SABR refit at five deltas with one-tick bands (orange) and the
first-order expansion (line), against strike.*

**SOFR expansion against Monte Carlo (Figure 9).** The module centres the strikes on its model
futures rate, 566.59 bp at 75 days, 134 bp above the futures rate of the data. Both valuations
use the same model, so the comparison tests the expansion, but at strikes away from the market's.
At 75 days the first- and second-order expansions lie inside the 95% interval at all 21 strikes;
at 103 days, 15 and 21 of 21. The interval half-width is 0.91 and 0.92 bp.

[![First- and second-order expansions against Monte Carlo 95% intervals for SOFR options at 75 and 103 days.](images/rdr_fig9_sofr_mc.png)](images/rdr_fig9_sofr_mc.png)

*Paper replication: RDR Figure 9 regenerated from `calibration_fig_8_9.py` with the module's
parameters and simulation ($2^{17}$ paths, seed 20). The module computes the 75-day panel only; the
103-day panel, shown in the article, uses the same code with that expiry's futures rate.*

## What the study does and does not show

- **It shows** that one volatility driver with piecewise-constant loadings fits the swaption
  surface across three tenors to about 1 bp, and the two SOFR slices to within a tick of their
  refit, with the parameters the modules record.
- **It shows** where the first-order expansion is accurate: at 2y for the base and the
  level-shifted scenarios, and at 75 days for SOFR options. At 5y for the 10y tenor, under a high
  vol-of-vol or large betas, and at 103 days to first order, it leaves the Monte Carlo interval.
- **It does not show** the published Tables 1 and 3: their values give larger errors than the
  modules' on the same data, and the article's footnote 5, that the truncated moment system of
  the SOFR parameters is stable, fails for the module's values (largest real parts of the
  eigenvalues 3.798 and 0.030).
- **It does not show** out-of-sample performance or hedging: each market is one snapshot.

## Reproduce

Run the companion script from a checkout:

```console
python examples/docs/app_swaptions_and_sofr_options.py
```

The swaption fit and the two swaption Monte Carlo cases take one to two minutes each and are listed
in `SLOW_CASES`; the other cases take seconds. The script imports the paper modules from the
repository, so it does not run from an installed wheel.

## See also

- [Stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md)
- [Calibration to implied volatilities](calibration.md)
- [Research papers and replication](reproducing_the_papers.md)

## References

- Sepp, A. and Rakhmonov, P. (2025). Stochastic volatility for factor Heath-Jarrow-Morton
  framework. *Review of Derivatives Research* 28, article 12.
  [DOI 10.1007/s11147-025-09217-4](https://doi.org/10.1007/s11147-025-09217-4).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
