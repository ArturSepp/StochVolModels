# Stochastic volatility for factor Heath–Jarrow–Morton models

Artur Sepp and Parviz Rakhmonov. *Review of Derivatives Research* 28, 12 (2025).
[Published article](https://doi.org/10.1007/s11147-025-09217-4)

The [manuscript sources](paper/README.md) include the figures and an inline
bibliography. Build the author source from this project's `paper/` directory
with a TeX distribution that provides `enotez` and the preamble packages.

```console
pdflatex -interaction=nonstopmode -halt-on-error factor_sv_model.tex
pdflatex -interaction=nonstopmode -halt-on-error factor_sv_model.tex
pdflatex -interaction=nonstopmode -halt-on-error factor_sv_model.tex
```

From the StochVolModels repository root, use an external Python environment.

```console
python -m pip install -e ".[research]"
python examples/docs/app_swaptions_and_sofr_options.py
python -m papers.sv_for_factor_hjm_rdr_2025.calibration_fig_5_6_7
python -m papers.sv_for_factor_hjm_rdr_2025.calibration_fig_8_9
```

The example verifies the swaption and SOFR pricing workflows on the quotes
stored in the figure scripts. The original drivers choose their cases via
`Locals`. They retain recorded parameter choices. Their default cases provide pricing examples
and Monte Carlo comparisons. The source build uses the supplied figure PDFs. ## Historical curve-factor analysis

Supply the daily U.S. Treasury par-yield panel at
`empirics/xml/ust_rates.csv`, with dates in the first column and the tenor
names from [the loader](empirics/ust_rates_data.py). The manuscript uses
January 2007 through December 2023. The first command below downloads
the Treasury XML and creates the CSV. If you have the CSV, start with the
second command. Run from the repository root.

```console
python -m pip install beautifulsoup4 lxml requests
python -m papers.sv_for_factor_hjm_rdr_2025.empirics.ust_rates_data
python -m papers.sv_for_factor_hjm_rdr_2025.empirics.ts_analysis
```
