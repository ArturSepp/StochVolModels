# SPX latent volatility and the volatility risk premium

The historical fit ends on 14 July 2026. The pricing exercise uses the 15 July 2026 VIX close and the paper's frozen SPXW option surface. The four maturities have approximately 12, 30, 65 and 93 calendar days to expiry.

## Inputs

| Observations | Source and coverage used |
|---|---|
| SPX close-to-close log returns and VIX closes | Frozen Bloomberg SPX Index and VIX Index series. The historical sample has 5,738 sessions from 22 September 2003 to 14 July 2026. The VIX close on 15 July is 15.67 points. |
| Daily intraday realised variance | Oxford–Man five-minute S&P 500 realised variance, distributed with `rumidas`. The original dated series supplies 4,153 measurements during the historical sample, ending 31 March 2020. Later sessions use returns and VIX alone. [Series description](https://cran.r-universe.dev/rumidas/doc/page/rv5) |
| SPXW options | 103 selected bid–ask IV pairs from the [paper's dated snapshot](../calibration_fits_20261029/spxw_20260715_selected_quotes.csv). The [unrestricted PDE fit](../calibration_pde_20261029/spxw_pde_fit.json) records SHA-256 `4523c6952ea17a6a90b656af67f51379267708886319dba3cef9e6ffde2ce741` for the quote file. |

The replication uses the local [Bloomberg input manifest](../empirical_bloomberg_20261029/bloomberg_request.json) and checks the two vendor files against their recorded hashes. Historical estimates use observation dates before the option snapshot. The vendor files are later-vintage downloads. To restore the public RV input, save the [original R data object](https://raw.githubusercontent.com/cran/rumidas/master/data/rv5.rda) as `inputs/rv5.rda`. Its SHA-256 is `113a746716b2ebf541bf72bb08b8bc6b2ac254043f14fe7d658ed5c373825b01`. The reader verifies this hash and extracts the original date index.

## Historical estimation

The physical volatility drift is $(\kappa_{1,P}+\kappa_{2,P}\sigma)(\theta_P-\sigma)=A+B\sigma-C_P\sigma^2$. A Gaussian filter approximates the distribution of daily log volatility using seven-point Gauss–Hermite quadrature. Each session carries equal weight. The demeaned SPX return gives the common shock and updates the previous close's state. Log realised variance measures twice that log state plus the log session length and an estimated offset. Daily returns and realised variance cover overlapping intervals. The log VIX close measures the 30-day pricing map at the new state, with its own noise scale. We estimate a constant historical pricing coefficient along with the physical parameters. Gaussian closure, the beginning-of-session approximation for integrated variance, and conditional measurement independence enter the quasi-likelihood.

Two starting values gave the same physical fit to the reported precision. Estimates are $(\theta_P,\kappa_{1,P},\kappa_{2,P})=(0.17934,5.61545,11.13329)$, with signed return loading $\beta=-1.31000$, orthogonal loading $\varepsilon=0.85081$ and $\rho=-0.83865$. The constant historical VIX coefficient reaches its admissible lower bound of zero. Re-estimating with physical $\kappa_1$ fixed at zero or one reduces the average daily quasi-log-score by 0.0572 and 0.0392, respectively. These comparisons describe in-sample fit. The filtered median on 14 July is 13.96%.

## Pricing and comparison

In the sequential fit, the 15 July return and VIX update the closing state using the historical coefficients. Its median is 13.057%. We hold that value fixed and estimate one premium from the options. The fit gives a premium of −2.01744 and pricing parameters $(\theta_Q,\kappa_{1,Q},\kappa_{2,Q})=(0.18864,5.33847,9.11585)$. The resulting 30-day model VIX is 14.88 points and pooled IV RMSE is 2.431 percentage points. The historical VIX measurement relation and the July option-implied drift use different pricing coefficients.

The proportional risk premium preserves $A=1.00706$, $B=-3.61883$ and both Brownian loadings. Only $C_Q=C_P+\delta$ changes. In the VIX-matched comparison, each trial coefficient gives a spot volatility by inversion of the 15 July VIX close. The SPXW fit gives $\delta=-2.06352$, $C_Q=9.06977$ and spot volatility of 14.04%. The corresponding pricing parameters are $(\theta_Q,\kappa_{1,Q},\kappa_{2,Q})=(0.18888,5.33189,9.06977)$. The two individual Brownian risk prices enter this exercise through their weighted sum.

The objective assigns each maturity one quarter of the total weight and each quote within a maturity equal weight. The premium and spot comparison fits these two quantities to options. The free drift and spot comparison fits all three pricing drift coefficients and the initial state with the two historical diffusion loadings. The unrestricted comparison reprices the archived six-parameter calibration. Each comparison uses the same forward, discount, strike set, midpoint IVs and full-state PDE. Final scores use a 1,601-node spatial grid. The [manuscript table](../../paper/sections/spx_physical_pricing.tex) contains the per-maturity scores and inside-spread percentages. The [smile comparison](../../paper/figures/spx_physical_pricing_20261124/spx_smiles.png) shows the primary restricted and unrestricted fits against the market. All pricing errors are in sample for one surface.

## Fit to the 15 July surface

| SPXW fit | Spot volatility | IV RMSE, percentage points | Quotes inside bid–ask |
|---|---:|---:|---:|
| Premium with filtered spot | 13.06% | 2.431 | 3/103 |
| Premium with VIX-inverted spot | 14.04% | 2.599 | 2/103 |
| Premium and freely fitted spot | 12.31% | 2.401 | 4/103 |
| Free pricing drift and spot, historical diffusion | 12.94% | 2.096 | 6/103 |
| Unrestricted pricing fit | 7.22% | 0.447 | 21/103 |

The historical-diffusion fit with free drift reaches the lower search bound for pricing $\kappa_1$ at 0.05. The six-parameter fit reaches the upper bound at 30 and gives $(\beta,\varepsilon)=(-4.176,4.724)$. Both diffusion loadings are invariant under an equivalent change of measure. The search results describe the July surface conditional on the model and the parameter bounds.

## Reproduction

Run from the repository root with the frozen local inputs in place.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$py = 'C:\Python\StochVolModels312\Scripts\python.exe'
$module = 'papers.robust_volatility_models_2026.replication.spx_physical_pricing_20261124'
foreach ($step in 'physical','historical_checks','pricing','filtered_pricing','drift_check','verify_results','make_section') {
    & $py -B -m "$module.$step"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $py -B papers/robust_volatility_models_2026/replication/stage_manuscript.py --sync
```

`physical.json` contains input hashes, search bounds, optimiser starts and the grid sensitivity. `filtered_state.csv` holds daily posterior log-state moments and quasi-log-scores. `historical_checks.json` stores the fixed-speed fits and an independent check of the historical VIX map against the exact linear-drift moment equations. `pricing.json`, `quote_fits.csv`, `filtered_pricing.json` and `free_drift.json` hold the pricing parameters, final model IVs, grid differences and fit statistics. The verifier recomputes all five pricing scores from individual quotes and checks the measure-change restrictions.
