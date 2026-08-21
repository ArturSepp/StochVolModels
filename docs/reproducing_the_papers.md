# Reproducing the papers

## Small workflow versus full replication

Install the optional research dependency only for repository paper scripts:

```console
python -m pip install -e ".[research]"
```

The library and quickstart do not import `qis`. Paper code lives under `papers/`, is excluded from
the wheel, and may require local data or output paths. Read `papers/README.md` and the README inside
the selected paper directory before running a script.

The two principal source bundles are:

- `papers/logsv_model_with_quadratic_drift/`, with the article source/PDF and equation-mapped LogSV
  analyses;
- `papers/sv_for_factor_hjm/`, with the experimental Factor HJM article source/PDF and figure
  workflows.

Additional directories cover inverse options, forward variance, impermanent-loss hedging,
risk-premia mixtures, Student-t work, and volatility-model studies. These are research workflows,
not a single stable command-line application. The forward-variance, Gaussian-mixture, and
Student-t directories are explicitly exploratory analyses with no publication mapping asserted.

`papers/jump_risk_premia_clustered_jumps/` is deliberately classified as development code related
to Liu, Packham, and Sepp's *Jump risk premia in the presence of clustered jumps*, not as an exact
replication package. It records Hawkes estimation, option calibration, and Monte Carlo experiments
that rely on local cryptocurrency data and historical research APIs; its README documents the
scope and prerequisites.

## Reproducibility record

Record the Git commit, package/dependency versions, script entry point, selected enum/test case,
input data provenance, local settings, output path, seed, path/step counts, and hardware. Keep
generated figures and private/local datasets outside Git.
`src/stochvolmodels/settings.yaml.example` documents the shared path configuration; copy it to the
ignored `src/stochvolmodels/settings.yaml` and do not commit that machine-local file.

The repository CI performs import and characterized numerical smoke checks, not every long paper
calibration or figure. A full replication can take substantially longer and can depend on inputs
that are not distributable. Reproducing a plot is not evidence that a model is appropriate for a
new dataset; validate market conventions and residuals separately.

## Reproduction classification

| Directory | Classification | Inputs |
|---|---|---|
| `logsv_model_with_quadratic_drift` | principal published-paper implementation with article source/PDF | generated model grids plus documented local option data for calibration cases |
| `sv_for_factor_hjm` | published-paper implementation of an experimental package surface | paper source/PDF and local calibration inputs |
| `jump_risk_premia_clustered_jumps` | development code related to the cited draft, not exact replication | local cryptocurrency data and optional research dependencies |
| `inverse_options`, `volatility_models`, `il_hedging` | supporting published/working-paper illustrations | mixture of generated and documented local/public inputs |
| `risk_premia_gmm`, `t_distribution`, `forward_var` | exploratory, no publication mapping asserted | generated and local research inputs depending on the script |

No JOSS acceptance claim depends on licensed data or a full paper rerun. The offline quickstart,
synthetic option-chain examples, package tests, and built documentation are the reviewer gates.
