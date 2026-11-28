---
myst:
  html_meta:
    description: >-
      The research papers behind stochvolmodels: the citation ledger, the documentation page each
      paper supports, which published figures the site regenerates or reproduces, the commands that
      replicate them, the classification of the paper directories and their known issues.
---

# Research papers and replication

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The repository includes article manuscripts and research code for published and accepted papers,
including the bivariate Hawkes model in the accepted SIFIN article. Its experimental rough-logSV implementation
also accompanies an authors' working draft, identified separately below. This page maps the
papers to documentation, describes figure replication, and records data prerequisites
and the known issues of the paper directories.

## The papers

The table links publisher, Crossref or arXiv records. The authors confirmed the SIFIN acceptance on 23 November 2026. The local authors' draft below has a separate
provenance record.

| Paper | Status | In the repository | Documentation pages |
|---|---|---|---|
| Sepp and Rakhmonov (2023), Log-normal stochastic volatility model with quadratic drift, *International Journal of Theoretical and Applied Finance* 26(8), 2450003, [DOI 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031) | Published, open access (CC BY 4.0) | PDF and LaTeX in `papers/logsv_model_with_quadratic_drift_ijtaf_2023/paper/`; figure code in `replication/figures/`, with theory, calibration, empirical and verification modules under `replication/` | [Log-normal SV model](logsv_model.md), [steady state and moments](volatility_distribution_and_moments.md), [martingale conditions](martingale_conditions_and_skews.md), [affine expansion](affine_expansion.md), [Fourier pricing](european_option_pricing.md), [inverse options](inverse_options.md), [QV options](quadratic_variance_options.md), [Monte Carlo schemes](monte_carlo_simulation.md), [analytic versus Monte Carlo](analytic_vs_monte_carlo.md), [calibration](calibration.md), [Bitcoin options](app_bitcoin_options.md) |
| Sepp and Rakhmonov (2025), Stochastic volatility for factor Heath-Jarrow-Morton framework, *Review of Derivatives Research* 28, article 12, [DOI 10.1007/s11147-025-09217-4](https://doi.org/10.1007/s11147-025-09217-4) | Published, exclusive licence to Springer | PDF and LaTeX in `papers/sv_for_factor_hjm_rdr_2025/paper/`; figure code in that directory | [Factor HJM rates](factor_hjm_stochastic_volatility.md), [USD swaptions and SOFR options](app_swaptions_and_sofr_options.md) |
| Lucic and Sepp (2024), Valuation and hedging of cryptocurrency inverse options, *Quantitative Finance* 24(7), 851-869, [DOI 10.1080/14697688.2024.2364804](https://doi.org/10.1080/14697688.2024.2364804) | Published | Code in `../papers/inverse_options_qf_2023/` | [Inverse options](inverse_options.md) |
| Lipton, Lucic and Sepp (2025), Unified approach for hedging impermanent loss of liquidity provision, *Digital Finance* 7(3), 429-477, [DOI 10.1007/s42521-025-00144-5](https://doi.org/10.1007/s42521-025-00144-5) | Published | Code in `papers/unified_approach_for_hedging_il_df_2025/` | [Impermanent loss hedging](app_impermanent_loss_hedging.md) |
| Liu, Packham and Sepp, *Jump risk premia in the presence of clustered jumps*, accepted in *SIAM Journal on Financial Mathematics*. [2025 arXiv preprint](https://arxiv.org/abs/2510.21297) | Accepted for publication | [Author manuscript and replication scope](https://github.com/ArturSepp/StochVolModels/blob/main/papers/jump_risk_premia_clustered_jumps_sifin_2027/README.md) | [Clustered jumps](hawkes_jump_diffusion.md) |
| Sepp and Rakhmonov (2023), What is a robust stochastic volatility model, SSRN 4647027, [DOI 10.2139/ssrn.4647027](https://doi.org/10.2139/ssrn.4647027) | SSRN working paper, revised 29 February 2024 | [Original SSRN code and recorded parameters](https://github.com/ArturSepp/StochVolModels/tree/588193611f4fbf5595d60b198e34cb5c24826f8a/papers/volatility_models) | [Robust SV models](app_robust_volatility_models.md) |

Lipton and Sepp (2022), *Toward an efficient hybrid method for pricing barrier options on assets with stochastic volatility* (*Wilmott*), combine conditional heat potentials and Monte Carlo. The tracked checkout carries the author manuscript, figure runner and generated-input comparisons. [Public preprint](https://arxiv.org/abs/2202.07849)

The separate local revision by Sepp and Rakhmonov, *What Is a Robust Stochastic Volatility Model? Measure Changes, Martingality and Cross-Asset Applications*, is dated **17 November 2026** on its title page. Its local `replication/stage_manuscript.py` assembles exhibits for the revised study. The public snapshot above preserves the original SSRN scripts. Run their offline recorded-parameter example with `python -m examples.docs.app_robust_volatility_models`. Empirical comparisons require separately sourced volatility histories.

Articles cite the two published papers by section and by the equation numbers of their PDF; the
LaTeX sources number some equations differently, as each `paper/README.md` explains.
In the IJTAF journal PDF, (2.1)–(2.2) appear on page 5 and the affine-expansion systems
(4.17) and (4.25) on pages 29 and 32. In the RDR journal PDF, factor dynamics (9),
drift freezing (37), the first-order expansion (108) and the simulation scheme (124)
appear on pages 6, 12, 30 and 36, respectively.

## Rough-logSV authors' working draft

Sepp and Rakhmonov submitted *Rough log-normal stochastic volatility with quadratic drift:
second-moment stability and the martingale property under positive correlation*
to *Quantitative Finance* and arXiv on **24 November 2026**.
The [paper folder](https://github.com/ArturSepp/StochVolModels/tree/main/papers/rough_volterra_lognormal_sv_model_2026)
links the reference figures and the [experimental rough-logSV guide](rough_logsv.md)
explains the computational methods.

The guide separates the draft's exact linear-drift moment equations, stability
criterion, nonlinear time-averaged bound and sufficient martingale condition from
the package's numerical algorithms. Strict positivity for singular kernels and the
general martingale converse remain open. The classical published IJTAF paper is a
separate reference; its equation numbers must not be used for the rough draft.

The [local figure runner](https://github.com/ArturSepp/StochVolModels/blob/main/papers/rough_volterra_lognormal_sv_model_2026/replication/README.md)
recomputes Figure 2 from the model parameters. Figure 1 replays four fixed-H fits
on 45 BTC and 39 ETH quotes using the authors’ quote panel and checkpoints.
Figure 3 renders the two Monte Carlo summaries included in the checkout.

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
repository root, for example `python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.figures.article_figures`;
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
| `logsv_model_with_quadratic_drift_ijtaf_2023` | principal published-paper implementation with article source and PDF | generated model grids, bundled option chains, and documented local option data for the time-series calibrations |
| `sv_for_factor_hjm_rdr_2025` | published-paper implementation of an experimental package surface | hard-coded swaption and SOFR option data; online US Treasury data for Fig. 4 |
| `inverse_options_qf_2023`, `unified_approach_for_hedging_il_df_2025` | code of published papers | generated inputs |
| `robust_volatility_models_2026` | manuscript, exhibit staging, and empirical replication code | volatility-index histories and local option and market-data snapshots |
| `jump_risk_premia_clustered_jumps_sifin_2027` | Accepted manuscript, Hawkes option pricer and selected estimation and calibration workflows | historical BTC data and separate co-author code for empirical results |
| `barrier_options_sv_wilmott_2022` | Published conditional barrier pricing. Run `replication/article_figures.py` from its documented module path | generated model paths, author manuscript and archived figures |
| `t_distribution`, `forward_var` | Exploratory terminal-law illustrations and forward-variance calibration. Entry points are `illustrations.py` and `calibrate_forward_var.py` | generated or recorded inputs according to the runner |

No JOSS acceptance claim depends on licensed data or a full paper rerun. The offline quickstart,
synthetic option-chain examples, package tests, and built documentation are the reviewer gates.

## Known issues of the paper directories

The following notes distinguish corrected implementation defects from remaining limitations;
the pages that use the code describe their verification and workarounds.

- **`logsv_model_with_quadratic_drift_ijtaf_2023`.** `replication/figures/article_figures.py` draws Figs. 6 and 10 through `qis`.
  The stationary-moment scale in `replication/theory/steady_state_pdf.py` is corrected to
  $(q/b)^{r/2}$, including the inverse-gamma limit at zero quadratic mean reversion.
  The `--quick` verification entry point checks orders zero through four against independent
  density quadrature at non-unit scales and checks divergent inverse-gamma moments.
  The full verification also reports expansion prices, PDE refinement and Monte Carlo
  uncertainty for four vanilla cases. The [Fourier comparison](numerical_accuracy_and_performance.md#logsv-expansion-against-the-full-state-model)
  reproduces the report and compares the public pricer at the same inputs.
  This scale correction does not change the standardized skewness and kurtosis of Fig. 1.
  The docstring of `replication/calibration/calibrations.py`
  refers to figures and a table that the published article does not contain; the
  [skews case study](app_positive_and_negative_skews.md) uses its recorded parameters as a package
  illustration.
- **`sv_for_factor_hjm_rdr_2025`.** The recorded parameters differ from Tables 1 and 3 of the article, and
  some expiries and path counts differ from its text. The
  [case study](app_swaptions_and_sofr_options.md) sets both out.
- **Original SSRN scripts.** The [archived revision](https://github.com/ArturSepp/StochVolModels/tree/588193611f4fbf5595d60b198e34cb5c24826f8a/papers/volatility_models) uses `qis`. Its `autocorr_fit.py` calls
  `LogSVPricer.simulate_vol_paths` with a total step count, which the package reads as steps per
  year, so its simulated autocorrelation cannot be reproduced as written; two empirical plots use a
  pandas frequency that pandas 3 rejects.
- **`inverse_options_qf_2023`.** `compare_net_delta.py` imports `qis`, and its output could not be matched
  to a numbered figure of the article.
- **`unified_approach_for_hedging_il_df_2025`.** The corrected author manuscript and theoretical
  replication include independent payoff, quadrature and delta checks. See the paper README for
  their scope; these checks do not revalidate historical empirical results.
- **`jump_risk_premia_clustered_jumps_sifin_2027`.** Most scripts need Tardis or Deribit data and `qis`;
  `hawkes_estimator.py` uses `beta1_m` where the negative intensity needs `beta2_m`.

## See also

- [Analytics gallery](analytics_gallery.md)
- [Examples and recipes](examples.md)
- [Documentation standard](documentation_standard.md)
- [Testing and coverage](testing_and_coverage.md)
