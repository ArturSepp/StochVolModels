# Review of volatility models

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Research in [StochVolModels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Local working module for the volatility-model review by Artur Sepp and Parviz
Rakhmonov. Local and licensed input snapshots are not redistribution permissions.

The current manuscript is
[robust_volatility_models.tex](paper/robust_volatility_models.tex)
(38 pages). The [pre-30 October revision](drafts/review_volatility_models_revised_20261028.tex) and the
[original manuscript](drafts/review_volatility_models.tex) are preserved unchanged
in `drafts/`, with their PDFs and compilation dependencies. These are the two
substantial historical versions retained as browsable drafts. Three intermediate
checkpoints are preserved in `drafts/older_audit_snapshots.zip`.
The 30 October changes are listed in
[agents/2026-10-30_REVISION_LOG.md](agents/2026-10-30_REVISION_LOG.md); the review
that motivated them is
[agents/review-volatility-models-revision-ijtaf-2026-10-29.md](agents/review-volatility-models-revision-ijtaf-2026-10-29.md).
Existing PDF files are retained exports; `paper/*_20261030.pdf` were compiled on
30 October 2026 from the staged sources.

## Layout

| Directory | Contents |
|---|---|
| [paper](paper/) | Current manuscript, PDF, generated TeX sections and figures |
| [replication](replication/) | All Python scripts, dated experiment bundles, frozen data and results; formerly `replication_code/` |
| [drafts](drafts/) | Original manuscript and pre-30 October revision with build dependencies; ZIP of intermediate checkpoints |
| [agents](agents/) | Historical revision plans, reviews, audits and provenance records |
| [private](private/) | Historical console outputs and LaTeX build artifacts |

Experiment bundles retain their original names and internal structure. Frozen
data, results and historical provenance records are unchanged. The [original SSRN replication](replication/ssrn_2023/) contains the earlier paper's
recorded parameters, empirical fits and figure scripts.

## Build and reproduce

Open the current manuscript in the LaTeX editor, or compile it with `paper/` as
the source directory (three `pdflatex` passes).

The active evidence bundles are:

- The [SPX physical and pricing calibration](replication/spx_physical_pricing_20261124/README.md) estimates latent volatility from historical observations and tests the measure-change restriction on the July SPXW quotes.

- The [cross-asset dynamics comparison](replication/cross_asset_dynamics_20261124/REPORT.md) reports daily, EWMA and persistence fits for 18 indices and identifies the separate pricing calibrations.
- [`vix_fit_20261123`](replication/vix_fit_20261123/README.md) contains the four-parameter
  Bloomberg VIX fit, the comparison with the archived estimates, and the regenerated
  observation-map and persistence experiments.
- [`empirical_bloomberg_20261030`](replication/empirical_bloomberg_20261030/README.md):
  descriptive index analysis with the 3/2 marginal family and the main/notes
  cohort split, plus `make_beta_panel.py` for the five-panel beta figure; reads the frozen inputs of
  [`empirical_bloomberg_20261029`](replication/empirical_bloomberg_20261029/README.md),
  whose expanded Cboe panel is unchanged.
- [`calibration_section_20261030`](replication/calibration_section_20261030/README.md):
  calibration section built from the frozen PDE outputs of
  [`calibration_pde_20261029`](replication/calibration_pde_20261029/README.md),
  with the per-maturity table, information criterion and half-lives. No recalibration.

Both generators also write `*_macros.tex`, which define every number quoted in the
manuscript's abstract, introduction, selection guide and discussion. Sibling bundles
retain preliminary results and diagnostics. The LogSV PDE remains in
[`../logsv_model_with_quadratic_drift_ijtaf_2023/replication`](../logsv_model_with_quadratic_drift_ijtaf_2023/replication/).

Use `C:\Python\StochVolModels312\Scripts\python.exe`. Never create an environment
under OneDrive. From this module directory, after regenerating the experiment
figures or TeX snippets, stage the manuscript dependencies with:

```powershell
& 'C:\Python\StochVolModels312\Scripts\python.exe' -B replication/stage_manuscript.py --sync
& 'C:\Python\StochVolModels312\Scripts\python.exe' -B replication/stage_manuscript.py --check
```

The staging check follows `paper/robust_volatility_models.tex` and verifies its four generated TeX files and five figures. The `--sync` option copies
graphics into `paper/figures/` and generated TeX into `paper/sections/`, rewriting
dependency paths in those staged snippets only. Edit generated text through its
producer in `replication/`. Staging does not run calibrations or change equations.
The `--sync` option writes the staging manifest to
[exhibit_manifest.json](../../agents/robust_volatility_review_20261030/exhibit_manifest.json).
Compile historical drafts from `drafts/`; they do not participate in current staging.

The original installation scripts and audit rewrite scripts are historical
one-time operations, not current build commands. The historical mathematical-audit
Python files now live in `replication/historical_mathematical_audit_20261029/`.
Do not rerun these rewrite/install scripts on the revised manuscript. Historical
provenance records retain the paths used when the experiments ran. The
[cleanup inventory and verification](../../agents/robust_volatility_review_20261030/)
record this reorganization. `private/cleanup_20261030/paper_before.zip` preserves
the complete pre-cleanup paper folder, including redundant staged files.

## Fixed-parameter LogSV density comparison

Run from the StochVolModels checkout with its external interpreter. The four case
keys are `SPXW`, `GLD`, `TLT` and `TLT+`. The last key selects the separate TLT
snapshot from 2 August 2024. The default replays all four frozen fits.

```console
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.review_density --review papers/robust_volatility_models_2026/replication --output agents/robust_review_density.json
```

The runner recomputes the archived fine-grid transform PDE and compares each
retained discounted call or put with a two-dimensional forward density at the
same parameters. The local JSON records raw nodal premiums, the refined price,
actual bid–ask spreads, archived Monte Carlo errors, and the equal-maturity
midpoint-IV RMSE. Prices are divided by the original forward. GLD and TLT use
the frozen European-equivalent quotes and effective discounts.

The report compares raw grid premiums and prices obtained by interpolating
nodal call and put premiums separately in strike. The reported comparison uses
second-order Richardson extrapolation of these prices across nested spatial
and time meshes. It checks both review limits on every quote: 0.005 IV
percentage points and 2% of the observed premium spread. The JSON retains
per-mesh density mass, forward error and negative mass at each expiry.

## Working plan

See the [revision log of 30 October 2026](agents/2026-10-30_REVISION_LOG.md) for the
open author items (crux sentence, expansion regime, European-equivalent option
conversion, affiliations and bibliography details) and the
[revision plan and outline of 28 October](agents/plans/2026-10-28_REVISION_PLAN_AND_UPDATED_OUTLINE.md)
for the scope decisions it builds on.

The [equation review](../../agents/robust_volatility_review_20261030/equation_review.md)
records the mathematical and interpretive findings. Its accepted recommendations
are implemented in the current manuscript and section generators; the
[implementation record](../../agents/robust_volatility_review_20261030/implementation/README.md)
lists the changes and validation. Saved data, calibration results and figures are unchanged.
The positive-skew calibration and DVOL integration are already complete. Round 3 of the
revision log records the length cut (exhibits moved to the package, float settings,
unreferenced staged figures moved to `private/unstaged_figures_20261030_r3/`).
