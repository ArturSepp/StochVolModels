# Historical volatility and option-implied risk premia across markets

This study applies the SPX historical filter and measure-change restriction to gold, long US Treasuries, oil, bitcoin and ether. TLT has two separately estimated histories. The 2024 date was chosen for its positive option skew in the paper's original experiment. Each risk premium is an in-sample estimate for one surface.

## Observations and units

The [quarterly panel](quarterly_panel/asset_summary.csv) extends the experiment to SPXW, GLD, TLT, USO, BTC and ETH from 2022Q2 to 2024Q1. The [date-level fits](quarterly_panel/date_results.csv) contain the physical estimates, both premia, all five option fits, bid–ask counts and paired error differences.

| Market | Option data | Historical measurements |
|---|---|---|
| GLD, TLT and TLT+ | Frozen rows from the original OCA option table. The verifier rejoins the actual ThetaData cache on timestamp and contract and checks both IVs, forward and discount. | Cash-adjusted ETF log returns, Cboe GVZ or VXTLT, and the Chicago Booth Risk Lab's five-minute trade-based realised volatility for the same ETF. |
| USO | The 15 July 2026 partition in the OCA `uso_400d` ThetaData backtest dataset. | USO returns, Cboe OVX and the USO Risk Lab series. This is the oil fund that underlies OVX. |
| BTC and ETH | OCA's `tardis/btc_options_oca.parquet` and `eth_options_oca.parquet`. The latest common eligible timestamp in these frozen archives was 5 May 2024 at 08:00 UTC. | Deribit USD index returns and realised variance from the 24 one-hour index returns ending at 08:00 UTC. The historical DVOL measurement is the close of the 07:00–08:00 UTC candle. |

The Bloomberg files are later-vintage downloads. Estimation stops at the preceding close. We update the state at the valuation time with the historical parameters held fixed. The ETF clock has 252 sessions per year. We convert the Risk Lab annualised volatility to session variance by squaring and dividing by 252. The crypto clock has 365 days per year. Each day requires 24 hourly returns for its realised-variance observation. All measurement equations include their own estimated noise scale and the realised-variance offset absorbs differences in coverage. [Risk Lab data and measurement methods](https://dachxiu.chicagobooth.edu/) · [Cboe volatility-index methodologies](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf)

We use the Risk Lab five-minute trade series in the sixth whitespace-delimited field (zero-based index 5). To restore the dated research inputs, download the original [GLD](https://dachxiu.chicagobooth.edu/data.php?ticker=90448), [TLT](https://dachxiu.chicagobooth.edu/data.php?ticker=89468) and [USO](https://dachxiu.chicagobooth.edu/data.php?ticker=91208) responses into `inputs/{gld,tlt,uso}_risklab.txt`. The data source updates over time. The historical program checks the frozen hashes and fund identifiers. OCA market inputs and the underlying Tardis index files are local licensed data.

## Option quotes

The source-cache check matched every original GLD and TLT quote, forward and discount at its exact timestamp. The separately processed `tlt_400d` archive gave bid or ask IV differences up to 0.14 points in July 2026 and 0.36 points in August 2024 on the same contracts. We used the original `thetadata_options/tlt` cache for both comparisons with the published benchmark.

Each surface gives four maturities equal weight and each retained strike within its maturity equal weight. We used the original exercise's frozen quotes wherever the paper already had a benchmark. New quotes must be OTM, have positive ordered premiums and implied volatilities, and lie within 15% absolute log-moneyness. We require both wings and at least six eligible strikes within a maturity. Among distinct eligible expiries from 7 to 120 days we select the nearest available in succession to 12, 30, 60 and 90 days. Each expiry contributes up to 25 evenly spaced strikes plus its nearest-to-forward eligible strike.

For USO the 3- and 5-percentage-point spread caps failed to yield four eligible two-sided slices. The 8-point cap retained 60 quotes. BTC and ETH retained 59 and 52 quotes under the 3-point cap. The option-markets table gives within-surface comparisons. Quote counts inside the bid–ask band depend on the spread screen. The diagnostics also rescore each fit on the retained subset with spreads at most 3 points.

The ETF comparison uses the cached European-equivalent Black IVs of American options. The crypto comparison uses European inverse contracts, provider-reported bid and ask IVs based on the expiry forward, and OCA's unit discount. All fits target IVs. The inverse coin premium and the normalised forward Black price use the OCA conversion convention. [Deribit inverse-option specification](https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options)

## Replication

We fitted the primary filtered-state premium on the original 401/256 and 801/512 space/time grids. The other searches used 401/32 and 801/64 and kept the same spatial domain and Fourier resolution. GLD and the two TLT unrestricted parameter sets were repriced from their original calibrations. Every reported IV and score uses the 1,601-node, 1,024-step reference grid. Each fit also records a price check with the 801-node, 512-step grid. The optimiser searches a bounded parameter region. Convergence from the chosen starts does not establish uniqueness.

Use the repository Python from its root. Install its parquet reader `pyarrow` in that environment. `ANALYTICS_RESOURCES` may point to the local analytics resources directory that contains `options_data` and `tardis`. The original historical Bloomberg and SPX replication bundles must also be in place.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$py = 'C:\Python\StochVolModels312\Scripts\python.exe'
$study = 'papers.robust_volatility_models_2026.replication.cross_asset_physical_pricing_20261124'
& $py -B -m "$study.oca_inputs"
& $py -B -m "$study.crypto_history" --download
foreach ($market in 'gld','tlt','tlt_positive','uso','btc','eth') {
    & $py -B -m "$study.physical" $market
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $py -B -m "$study.pricing" $market filtered inverted spot drift unrestricted
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $py -B -m "$study.historical_checks"
& $py -B -m "$study.boundary_check"
& $py -B -m "$study.check_new_prices"
& $py -B -m "$study.exhibits"
& $py -B papers/robust_volatility_models_2026/replication/physical_pricing_section_20261127/make_section.py
& $py -B papers/robust_volatility_models_2026/replication/stage_manuscript.py --sync
```

The dated option files record the OCA source, hash, expiry set, spread cap and exact timestamp. Each historical JSON contains the start values, objective, fitted parameters, sample counts and refined-filter check. The option JSON files contain every fitted IV, optimiser results, pooled and maturity-specific errors and bid–ask counts. `quote_diagnostics.json` collects these results across the five pricing specifications. The [manuscript exhibit](../../paper/sections/cross_asset_physical_pricing.tex) and [smiles](../../paper/figures/cross_asset_physical_pricing_20261124/cross_asset_smiles.png) are generated from these files.

The [calibration parameters](calibration_parameters.csv) contain the physical and pricing coefficients for the 30 original cross-asset fits. For the July TLT affine limit, the polynomial drift coordinates remain finite. The file leaves the equilibrium and its associated mean-reversion parameter empty at that limit. On the shortest USO, BTC and ETH expiries, we checked the two wings and the nearest-to-forward strike against conditional Monte Carlo using four independent scrambled Sobol draws of 8,192 paths. The largest absolute price difference across the nine contracts was 0.045 basis points of forward and the largest difference as a share of a contract's bid–ask spread was 0.30%. The individual price differences and sampling errors are in `independent_short_prices.json`.

## Quarterly replication

For each asset we start at the last common trading day of the quarter and work backwards within the final calendar month until the asset has four eligible expiries. OCA provides the canonical GLD and TLT month files, the USO 400-day cache and the Tardis BTC and ETH archives. SPXW uses OCA's continuous end-of-day series, which joins the Cboe and ThetaData partitions in June 2023. Each asset uses its own observation time. The additional wing screen requires absolute log-moneyness no greater than 2.25 times the midpoint IV times the square root of time to expiry. The selected quotes, source path, timestamp, hashes and skipped dates are recorded in `quarterly_panel/`.

Each historical optimisation uses two starts and data through the preceding close. The three primary option fits use the same dated strikes and maturity weights as the two free-spot and free-drift controls. The search uses coarser PDE grids with final scoring at 801 spatial nodes and 128 time steps. The scalar fits also compare the admissible endpoints and zero premium, with further refinement near zero. The verification program recomputes each error from the individual strikes. For each asset it reprices the first date, the closest-to-zero premium and the smallest gap to free Q on a finer grid. It compares three short-expiry prices on the first date with independent conditional Monte Carlo. Counts are descriptive. A positive gap is the restricted minus the unrestricted RMSE in IV percentage points.

```powershell
& $py -B -m "$study.panel_inputs" select
foreach ($asset in 'spx','gld','tlt','uso','btc','eth') {
    & $py -B -m "$study.panel_inputs" $asset
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $py -B -m "$study.panel_pricing" $asset filtered inverted unrestricted spot drift
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $py -B -m "$study.panel_checks" $asset
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $py -B -m "$study.panel_exhibit"
& $py -B papers/robust_volatility_models_2026/replication/stage_manuscript.py --sync
```
