# Contributing to StochVolModels

<!-- docs-start -->

Thank you for your interest in `stochvolmodels`. The package is a reference implementation used by
published research, so numerical conventions and established model parameterisations constrain
what can change.

## Scope

In scope:

- bug fixes in European-option pricing, implied-volatility inversion, simulation, or calibration;
- numerical robustness improvements with a reproducer and an independent reference check;
- tests, documentation, and offline examples using generated or openly redistributable data;
- improvements to the shared `OptionChain` conventions or stable LogSV/Heston workflows; and
- characterized research contributions to an existing advanced model, after discussion in an issue.

Please open an issue before implementing any of the following:

- a public signature, default, model parameterisation, transform, or simulation-dynamics change;
- a new model or dependency;
- a change to paper replication code or a stored numerical regression baseline; or
- a feature that belongs in a sibling package such as `vanilla-option-pricers`, `qis`, or
  `optimalportfolios`.

American, exotic, and path-dependent payoffs are outside this package's intended scope. Examples
that require a paid data subscription cannot be reviewer or CI gates.

## Reporting a bug

Open a [bug report](https://github.com/ArturSepp/StochVolModels/issues/new?template=bug_report.yml)
with the package and Python versions, operating system, exact install command and extras, a minimal
self-contained reproducer, and the full traceback or incorrect values. Use the bundled synthetic
chain or generated data where possible; maintainers cannot reproduce proprietary datasets.

For a numerical problem, state the convention being used (forward/spot, discount factor/rate,
option type, volatility quotation, maturity units) and provide the independent result against
which the output was checked.

## Questions and support

Open a public [GitHub issue](https://github.com/ArturSepp/StochVolModels/issues) and describe the
workflow you are trying to implement. Methodology questions are welcome; cite the paper and
equation or section when relevant. The project is maintained on a best-effort basis and does not
offer private support or a guaranteed response time.

## Development setup

```console
git clone https://github.com/ArturSepp/StochVolModels.git
cd StochVolModels
python -m pip install -e ".[dev,docs]"
python -m pytest -m "not slow"
python -m pytest -m slow
ruff check --select E9,F63,F7,F82,F811 src
python -m sphinx -W --keep-going -b html docs docs/_build/html
python examples/getting_started/quickstart.py
```

Tests live in `src/stochvolmodels/tests/`. Unmarked fast tests cover core contracts and
deterministic numerical properties; `slow` covers simulation/regressions,
`optional_integration` covers optional providers, `repository_only` covers JOSS/docs/examples, and
`paper_replication` is reserved for long paper gates. Build and inspect an installed artifact with:

```console
python -m pip install build twine
python -m build
python -m twine check dist/*
python scripts/check_wheel_contents.py dist/*.whl
```

Run installed-wheel tests from a directory outside the checkout so local source cannot mask a
packaging error:

```console
python -m pytest --pyargs stochvolmodels.tests -m "not slow"
```

## Numerical changes

- Add a test that fails before the fix and passes after it.
- Cross-check pricing, transforms, calibration, or simulation through a genuinely different route:
  analytic versus Monte Carlo, a limiting case, a closed-form identity, or a trusted external
  reference.
- Keep stochastic tests statistical: state the seed, standard error, and acceptance rationale.
- Do not regenerate a regression `.npz` merely because a test failed. Explain and obtain approval
  for any intentional baseline change.
- Re-run the affected paper workflows when a transform, parameterisation, or simulation path could
  alter published output.

## Pull requests

- Keep one topic per pull request and describe the user-visible effect.
- Add or update tests and documentation for changed behavior.
- Run the command set above and report the results.
- Preserve the stable root API. Identify any signature or default change explicitly.
- Do not bump versions, alter release metadata, or commit generated figures/calibration output as
  part of an ordinary contribution.
- Keep paper and example dependencies optional; `qis` and data-provider packages are not core
  runtime dependencies.

## Conduct

Be civil, assume good faith, and keep technical disagreement focused on evidence and the code.
The JOSS review, if active, is additionally governed by the JOSS code of conduct.

## Licence

The project is MIT licensed. By contributing, you agree that your contribution is distributed
under the same licence.
