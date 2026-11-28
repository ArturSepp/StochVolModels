# What Is a Robust Stochastic Volatility Model?

*Authors: [Artur Sepp](https://github.com/ArturSepp) and Parviz Rakhmonov / First recorded: [2026-11-24](https://github.com/ArturSepp/StochVolModels/commit/ec1bd08e395a56a1ce14ca8a32ed9cd4dfb9ebe2)*

Research in [StochVolModels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The paper gives a criterion for when a stochastic volatility model retains its
form under changes of measure and derives the stock martingale condition for
quadratic-drift lognormal volatility. The empirical study compares volatility
indices across assets, calibrates Heston and lognormal stochastic volatility
(LogSV) to option surfaces, and examines how the choice of spot volatility
affects the inferred volatility risk premium.

## Layout

| Directory | Contents |
|---|---|
| [paper](paper/) | Manuscript [PDF](paper/robust_volatility_models.pdf), [LaTeX source](paper/robust_volatility_models.tex), and figures |
| [replication](replication/) | Scripts for the empirical analysis, option calibration, and manuscript exhibits |

## Build the manuscript

From the StochVolModels repository root, run the following with a LaTeX
distribution that provides `pdflatex`. The checkout contains all four figures
and all LaTeX needed for the 36-page manuscript.

```powershell
Push-Location papers/robust_volatility_models_2026/paper
try {
    1..3 | ForEach-Object {
        pdflatex -interaction=nonstopmode -halt-on-error robust_volatility_models.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $_ failed" }
    }
} finally {
    Pop-Location
}
```

## Regenerate the exhibits from saved results

The checkout provides the scripts and the material needed to compile the
manuscript. Exhibit regeneration also needs the saved fit results and market-data
snapshots listed below. Preserve the original filenames and bytes where the
scripts verify input hashes.

### Data supplied separately

Place these files under `papers/robust_volatility_models_2026/replication/`.
In the table, braces indicate lists of filenames or filename components.

| Directory | Required files |
|---|---|
| `empirical_bloomberg_20261029/` | `panel_results.json`, `bloomberg_request.json`, `inputs/bloomberg_volatility.csv`, `inputs/bloomberg_reference_cash_adjusted.csv`, and `tables/{VIX,OVX,MOVE}_beta_span{65,260}.csv` |
| `empirical_bloomberg_20261030/` | `descriptive_results.json` |
| `dvol_diagnostics_20261030/` | `descriptive_results.json` and `tables/{btc,eth}_beta_{65,260}.csv` |
| `vix_fit_20261123/` | `synthetic_results.json` |
| `calibration_fits_20261029/` | `{spxw,gld,tlt}_20260715_selected_quotes.csv` |
| `calibration_pde_20261029/` | `{comparison_table,maturity_comparison,numerical_checks}.csv`, `{spxw,gld,tlt}_pde_{fit,checks}.json`, and `{spxw,gld,tlt}_quote_results.csv` |
| `calibration_positive_skew_20261031/` | `{comparison_table,maturity_comparison,numerical_checks,selected_surface_summary}.csv`, `scan_summary.json`, `tlt_20240802_selected_quotes.csv`, `tlt_pde_{fit,checks}.json`, and `tlt_quote_results.csv` |
| `spx_physical_pricing_20261124/` | `{physical,pricing,filtered_pricing,free_drift}.json` and `quote_fits.csv` |
| `cross_asset_physical_pricing_20261124/` | `history_checks.json` and `{gld,tlt,tlt_positive,uso,btc,eth}_{physical,filtered,inverted,spot,drift,unrestricted}.json`. Under `inputs/`, supply `{gld,tlt,uso}_risklab.txt`, `{btc,eth}_history.{csv,json}`, and `{uso,btc,eth}_quotes.{csv,json}`. |
| `cross_asset_physical_pricing_20261124/quarterly_panel/` | `dates.json`, each asset's `results.csv` and `checks.json`, and the files for each accepted asset and date in `dates.json`. In every `<asset>/<date>/` directory, supply `quotes.csv`, `source.json`, `history.json`, and `<asset>_{filtered,inverted,spot,drift,unrestricted}.json`. This covers 47 surfaces across SPXW, GLD, TLT, USO, BTC, and ETH. |

Point `ANALYTICS_RESOURCES` at the market-data root. The cross-asset exhibit
checks selected GLD and TLT quotes against the three original
Option Chain Analytics (OCA) partitions. The quarterly exhibit checks BTC and
ETH spot prices against the hourly Tardis index series. Supply these five files
under that root with the original directory structure.

```text
options_data/thetadata_options/gld/options/2026-07.parquet
options_data/thetadata_options/tlt/options/2026-07.parquet
options_data/thetadata_options/tlt/options/2024-08.parquet
tardis/BTC_perp_freq_H.feather
tardis/ETH_perp_freq_H.feather
```

### Commands

Use Python 3.12 in your external project environment. From the repository root,
install the research dependencies and the Parquet and Feather reader.

```powershell
python -m pip install -e ".[research]" pyarrow
```

Run the exhibits in this order. Replace `C:\path\to\analytics_resources` with
the root holding the five market-data files above.

```powershell
$ErrorActionPreference = 'Stop'
$env:ANALYTICS_RESOURCES = 'C:\path\to\analytics_resources'
$env:PYTHONPATH = (Resolve-Path src).Path
$rep = 'papers/robust_volatility_models_2026/replication'
$cross = 'papers.robust_volatility_models_2026.replication.cross_asset_physical_pricing_20261124'

python -B "$rep/empirical_bloomberg_20261030/make_empirical_section.py"
python -B "$rep/empirical_bloomberg_20261030/make_beta_panel.py"
python -B "$rep/calibration_section_20261030/make_calibration_section.py"
python -B -m "$cross.exhibits"
python -B "$rep/physical_pricing_section_20261127/make_section.py"
python -B -m "$cross.panel_exhibit"
python -B "$rep/stage_manuscript.py" --sync
python -B "$rep/stage_manuscript.py" --check
```

The staging check reports one main file and 11 verified dependencies.
`--sync` updates the seven embedded blocks and four PNGs from their generated
sources. PNG bytes can vary with the plotting environment. Rebuild the
manuscript with the LaTeX command above.

## Check the option fits from frozen quotes

Run from the repository root using `$rep` and the Python environment above.
The four Heston fits use the selected quote CSVs from the table. The commands
write Heston fit JSONs into the corresponding replication folders.

```powershell
python -B "$rep/calibration_fits_20261029/run_calibrations.py" --model Heston
python -B "$rep/calibration_positive_skew_20261031/run_calibrations.py" --ticker TLT --date 2024-08-02 --model Heston
```

The independent forward-density replay reprices SPXW, GLD, TLT, and TLT+
at the reported LogSV parameters. In addition to the quote CSVs and
`*_pde_fit.json` files above, supply
`calibration_pde_20261029/{spxw,gld,tlt}_pde_mc.json` and
`calibration_pde_20261029/runtime_provenance.json`, plus
`calibration_positive_skew_20261031/tlt_pde_mc.json` and
`calibration_positive_skew_20261031/BUNDLE_MANIFEST.json`.

```powershell
python -B -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.review_density --review "$rep" --case all --output review_density.json
```

## Rerun analyses from market snapshots

The exhibit commands reuse saved results. To refit the models, supply the
dated volatility histories, realised-variance series, and selected option
quotes used by the analysis scripts. The calibration runners also use saved
starting fits, and the validation steps use saved fit and Monte Carlo
records. The SPXW and cross-asset guides give the market sources, file
locations, and commands for the physical and pricing fits. [Index panel
runner](replication/empirical_bloomberg_20261029/run_expanded_panel.py) ·
[SPXW workflow](replication/spx_physical_pricing_20261124/README.md) ·
[Cross-asset and quarterly workflow](replication/cross_asset_physical_pricing_20261124/README.md) ·
[PDE calibration runner](replication/calibration_pde_20261029/run_pde.py) ·
[Positive-skew TLT runner](replication/calibration_positive_skew_20261031/run_pde.py)
