---
myst:
  html_meta:
    description: >-
      The research papers behind stochvolmodels: the citation ledger, the documentation page each
      paper supports, which published figures the site regenerates or reproduces, the commands that
      replicate them, the classification of the paper directories and their known issues.
---

# Research papers and replication

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Part of the [stochvolmodels](https://github.com/ArturSepp/StochVolModels) documentation.
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The package implements the models of two published articles, whose PDF and LaTeX source are in the
repository, and supports four further papers. This page lists them with the documentation pages
that rest on each, shows which published figures the site regenerates from repository code, gives
the commands that replicate them, and records the known issues of the paper directories.

## The papers

Every reference below was checked against its publisher, Crossref or arXiv record; the working
papers are cited as such.

| Paper | Status | In the repository | Documentation pages |
|---|---|---|---|
| Sepp and Rakhmonov (2023), Log-normal stochastic volatility model with quadratic drift, *International Journal of Theoretical and Applied Finance* 26(8), 2450003, [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031) | Published, open access (CC BY 4.0) | PDF and LaTeX in `papers/logsv_model_with_quadratic_drift/paper/`; figure code in that directory | [Log-normal SV model](logsv_model.md), [steady state and moments](volatility_distribution_and_moments.md), [martingale conditions](martingale_conditions_and_skews.md), [affine expansion](affine_expansion.md), [Fourier pricing](european_option_pricing.md), [inverse options](inverse_options.md), [QV options](quadratic_variance_options.md), [Monte Carlo schemes](monte_carlo_simulation.md), [analytic versus Monte Carlo](analytic_vs_monte_carlo.md), [calibration](calibration.md), [Bitcoin options](app_bitcoin_options.md) |
| Sepp and Rakhmonov (2025), Stochastic volatility for factor Heath-Jarrow-Morton framework, *Review of Derivatives Research* 28, article 12, [DOI 10.1007/s11147-025-09217-4](https://doi.org/10.1007/s11147-025-09217-4) | Published, exclusive licence to Springer | PDF and LaTeX in `papers/sv_for_factor_hjm/paper/`; figure code in that directory | [Factor HJM rates](factor_hjm_stochastic_volatility.md), [USD swaptions and SOFR options](app_swaptions_and_sofr_options.md) |
| Lucic and Sepp (2024), Valuation and hedging of cryptocurrency inverse options, *Quantitative Finance* 24(7), 851-869, [DOI 10.1080/14697688.2024.2364804](https://doi.org/10.1080/14697688.2024.2364804) | Published | Code in `papers/inverse_options/` | [Inverse options](inverse_options.md) |
| Lipton, Lucic and Sepp (2025), Unified approach for hedging impermanent loss of liquidity provision, *Digital Finance* 7(3), 429-477, [DOI 10.1007/s42521-025-00144-5](https://doi.org/10.1007/s42521-025-00144-5) | Published | Code in `papers/il_hedging/` | [Impermanent loss hedging](app_impermanent_loss_hedging.md) |
| Liu, Packham and Sepp (2025), Jump risk premia in the presence of clustered jumps, arXiv 2510.21297, first posted on SSRN (4735365) in 2024, [DOI 10.48550/arXiv.2510.21297](https://doi.org/10.48550/arXiv.2510.21297) | Working paper | Development code in `papers/jump_risk_premia_clustered_jumps/` | [Clustered jumps](hawkes_jump_diffusion.md) |
| Sepp and Rakhmonov (2023), What is a robust stochastic volatility model, SSRN 4647027, [DOI 10.2139/ssrn.4647027](https://doi.org/10.2139/ssrn.4647027) | Working paper; text not in the repository | Code and recorded parameters in `papers/volatility_models/` | [Robust SV models](app_robust_volatility_models.md) |

Articles cite the two published papers by section and by the equation numbers of their PDF; the
LaTeX sources number some equations differently, as each `paper/README.md` explains.

## Published figures on this site

The [analytics gallery](analytics_gallery.md) lists every exhibit with its class and producer. For
the two published articles:

| Figure | On this site | Exhibit | Page |
|---|---|---|---|
| IJTAF Fig. 1 | Regenerated from `steady_state_pdf.py` | `steady_state_density.png` | [Steady state and moments](volatility_distribution_and_moments.md) |
| IJTAF Figs. 2 and 3 | Regenerated from `moments_vol_qvar.py` | `vol_moments_vs_mc.png`, `expected_qvar_vs_mc.png` | [Steady state and moments](volatility_distribution_and_moments.md) |
| IJTAF Figs. 4 and 5 | Regenerated from `ode_sol_in_time.py` | `first_order_odes.png`, `second_order_odes.png` | [Affine expansion](affine_expansion.md) |
| IJTAF Fig. 6 | Recomputed with the package, the figure code needing `qis` | `expansion_pdfs_vs_mc.png` | [Affine expansion](affine_expansion.md) |
| IJTAF Figs. 7 and 8 | Reproduced from the PDF under CC BY 4.0; the Deribit data are not distributable | `ijtaf_fig7_btc_calibrations.png`, `ijtaf_fig8_btc_fit.png` | [Bitcoin options](app_bitcoin_options.md) |
| IJTAF Fig. 9 | Not shown; an analogue on the bundled Bitcoin chain | `btc_case_measures.png` | [Bitcoin options](app_bitcoin_options.md) |
| IJTAF Fig. 10 | Recomputed with the package, the figure code needing `qis` | `qvar_option_smiles.png` | [QV options](quadratic_variance_options.md) |
| RDR Figs. 1 to 4 | Not shown: no figure code for Figs. 1 to 3; Fig. 4 needs online US Treasury data | | |
| RDR Figs. 5 to 9 | Regenerated from `calibration_fig_5_6_7.py` and `calibration_fig_8_9.py` with the modules' parameters, never cropped | `rdr_fig5_swaption_fit.png` to `rdr_fig9_sofr_mc.png` | [USD swaptions and SOFR options](app_swaptions_and_sofr_options.md) |

## Replication commands

The package itself does not import the paper code. Install the research extra for the paper
directories, from a source checkout:

```console
python -m pip install -e ".[research]"
```

The registered exhibits are regenerated into a fresh directory outside the checkout and checked
there; the committed previews are compared with the published manifest:

```console
python -m scripts.docs_analytics.run --all --output-root <fresh directory>
python -m scripts.docs_analytics.validate --run-root <fresh directory>
python -m scripts.docs_analytics.run --verify
```

`--only <exhibit>` regenerates one exhibit. Each documentation page's numbers are asserted by its
canonical script, which runs offline, for example `python examples/docs/app_swaptions_and_sofr_options.py`;
the [examples page](examples.md) lists them all. The narrow offline test lane of the IJTAF results
runs with `pytest -m paper_replication`. The paper directories' own entry points run from the
repository root, for example `python -m papers.logsv_model_with_quadratic_drift.article_figures`;
read `papers/README.md` and the README of the directory first.

## Reproducibility record

Record the Git commit, package and dependency versions, script entry point, selected case, input
data provenance, local settings, output path, seed, path and step counts, and hardware. Keep
generated figures and private or local datasets outside Git, apart from the reviewed previews in
`docs/images/`. `src/stochvolmodels/settings.yaml.example` documents the shared path configuration;
copy it to the ignored `src/stochvolmodels/settings.yaml` and do not commit that machine-local file.

The repository CI performs import and characterized numerical smoke checks, not every long paper
calibration or figure. A full replication can take substantially longer and can depend on inputs
that are not distributable. Reproducing a plot is not evidence that a model is appropriate for a
new dataset; validate market conventions and residuals separately.

## Classification of the paper directories

| Directory | Classification | Inputs |
|---|---|---|
| `logsv_model_with_quadratic_drift` | principal published-paper implementation with article source and PDF | generated model grids, bundled option chains, and documented local option data for the time-series calibrations |
| `sv_for_factor_hjm` | published-paper implementation of an experimental package surface | hard-coded swaption and SOFR option data; online US Treasury data for Fig. 4 |
| `inverse_options`, `il_hedging` | code of published papers | generated inputs |
| `volatility_models` | code of a working paper, with recorded parameters | public volatility indices and private Bitcoin data, downloaded or local |
| `jump_risk_premia_clustered_jumps` | development code related to a working paper, not an exact replication | local cryptocurrency data and optional research dependencies |
| `risk_premia_gmm`, `t_distribution`, `forward_var`, `barriers` | exploratory, no publication mapping asserted; not cited on this site | generated and local research inputs, depending on the script |

No JOSS acceptance claim depends on licensed data or a full paper rerun. The offline quickstart,
synthetic option-chain examples, package tests, and built documentation are the reviewer gates.

## Known issues of the paper directories

These are recorded in the documentation audits and have not been changed; the pages that use the
code say how they work around them.

- **`logsv_model_with_quadratic_drift`.** `article_figures.py` draws Figs. 6 and 10 through `qis`.
  In `steady_state_pdf.py`, `vol_moment` inverts the scale factor, so absolute moments are wrong;
  Fig. 1, which shows skewness and kurtosis, is not affected. The docstring of `calibrations.py`
  refers to figures and a table that the published article does not contain; the
  [skews case study](app_positive_and_negative_skews.md) uses its recorded parameters as a package
  illustration.
- **`sv_for_factor_hjm`.** The recorded parameters differ from Tables 1 and 3 of the article, and
  some expiries and path counts differ from its text; the
  [case study](app_swaptions_and_sofr_options.md) sets both out. `plot_mkt_model_joint_smile_MF`
  calls `np.in1d`, which NumPy has removed, and `plot_mkt_model_joint_fut_smile_MF` fails with
  `add_up_down=False`.
- **`volatility_models`.** Every module imports `qis`. `autocorr_fit.py` calls
  `LogSVPricer.simulate_vol_paths` with a total step count, which the package reads as steps per
  year, so its simulated autocorrelation cannot be reproduced as written; two empirical plots use a
  pandas frequency that pandas 3 rejects.
- **`inverse_options`.** `compare_net_delta.py` imports `qis`, and its output could not be matched
  to a numbered figure of the article.
- **`il_hedging`.** The README lists the authors in another order than the published article.
- **`jump_risk_premia_clustered_jumps`.** Most scripts need Tardis or Deribit data and `qis`;
  `hawkes_estimator.py` uses `beta1_m` where the negative intensity needs `beta2_m`.

## See also

- [Analytics gallery](analytics_gallery.md)
- [Examples and recipes](examples.md)
- [Documentation standard](documentation_standard.md)
- [Testing and coverage](testing_and_coverage.md)
