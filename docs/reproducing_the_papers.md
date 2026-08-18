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
not a single stable command-line application.

`papers/jump_risk_premia_clustered_jumps/` is deliberately classified as development code related
to Liu, Packham, and Sepp's *Jump risk premia in the presence of clustered jumps*, not as an exact
replication package. It records Hawkes estimation, option calibration, and Monte Carlo experiments
that rely on local cryptocurrency data and historical research APIs; its README documents the
scope and prerequisites.

## Reproducibility record

Record the Git commit, package/dependency versions, script entry point, selected enum/test case,
input data provenance, local settings, output path, seed, path/step counts, and hardware. Keep
generated figures and private/local datasets outside Git. `papers/settings.yaml.example` documents
the optional path configuration; do not commit `papers/settings.yaml`.

The repository CI performs import and characterized numerical smoke checks, not every long paper
calibration or figure. A full replication can take substantially longer and can depend on inputs
that are not distributable. Reproducing a plot is not evidence that a model is appropriate for a
new dataset; validate market conventions and residuals separately.
