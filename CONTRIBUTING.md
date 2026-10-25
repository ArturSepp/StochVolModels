# Contributing to StochVolModels

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-21](https://github.com/ArturSepp/StochVolModels/commit/83e164c17fcfaa4ef8cd08cbe099bf89681efeb4)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

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

New American or general exotic/path-dependent product families are outside this package's intended
scope. The narrowly characterized book-analytics layer includes European and integrated-variance
payoffs. Examples that require a paid data subscription cannot be reviewer or CI gates.

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

## Starter tasks

These are bounded proposals, not assigned issues or confirmed defects. Recheck source and open
work before agreeing one task. Use the existing [feature form](https://github.com/ArturSepp/StochVolModels/issues/new?template=feature_request.yml),
Area **Documentation or example**. Include the task ID, owner, exact paths, reproduction, acceptance and focused
check under Proposed behavior; retain required boundary checkboxes. Use the bug form only for
observed incorrect behaviour. Keep numerical code, public signatures, dependencies, fixtures and
canonical examples unchanged for these documentation tasks.

### SVM-1: clarify LogSV first output

Owner: StochVolModels. Edit only `docs/getting_started.md` to clarify one observed setup or output-reading ambiguity. Acceptance: locate the vanilla price/IV and two-maturity chain dimensions while preserving every asserted number, canonical excerpt and notebook.

```console
python examples/getting_started/quickstart.py
python scripts/check_quickstart_notebook.py
python scripts/check_docs.py --files docs/getting_started.md
```

### SVM-2: clarify chain conventions

Owner: StochVolModels. Edit only `docs/option_chains_and_conventions.md` to clarify one forward, discount-factor, maturity or shape ambiguity. Acceptance: keep the stable/advanced distinction and numerical conventions unchanged; do not alter a pricer or parameter.

```console
python -m pytest src/stochvolmodels/tests/test_option_chain_characterization.py -k "discount or uniform" -q
```

### SVM-3: check release navigation

Owner: StochVolModels. Edit only `docs/getting_started.md` to clarify one stable/main or optional-research boundary. Acceptance: retain the pinned canonical example and existing page anchors; keep research/provider packages optional.

```console
python -m pytest src/stochvolmodels/tests/test_docs_seo.py -q
```

Run the package's existing strict HTML command after a documentation edit and inspect the
changed passage. On the maintainer's OneDrive checkout, dot-source the shared
`Enter-AgentRepo.ps1` first and use `C:\Python\StochVolModels312\Scripts\python.exe`.
Keep exports, caches, builds and run output on C. The `uv` setup below is for ordinary contributor
checkouts outside OneDrive; never create `.venv` in the synchronized checkout.

Follow the [shared documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md) for verification and review.
Record the revision, actual versions, exact checks and skips, rendered review and material
automation assistance. Preserve MIT licensing and attribution. Automated checks do not
establish independent reader success. Review time and acceptance are not promised.

## Development setup

```console
git clone https://github.com/ArturSepp/StochVolModels.git
cd StochVolModels
uv sync --locked --group test --group lint --extra docs
uv run --no-sync pytest -m "not slow"
uv run --no-sync pytest -m slow
uv run --no-sync ruff check --select F,TID251,TID253,ICN src/stochvolmodels
uv run --no-sync python scripts/check_fitter_contracts.py
uv run --no-sync python -m sphinx -W --keep-going -b html docs docs/_build/html
uv run --no-sync python examples/getting_started/quickstart.py
```

Tests live in `src/stochvolmodels/tests/`. Unmarked fast tests cover core contracts and
deterministic numerical properties; `slow` covers simulation/regressions,
`optional_integration` covers optional providers, `repository_only` covers JOSS/docs/examples, and
`paper_replication` is reserved for long paper gates.

The authoritative Python 3.12 source CI also installs the `research` extra. Include that extra
when exercising optional provider contracts locally; a core-only run may skip them. Repository-only
synthetic research tests, including `test_research_study_tools.py`, do not require the private
data or output folders from an earlier study. Native quote-scale tests in
`test_native_quote_diagnostics.py` exercise the core package without an optional data provider.

Component development diagnostics live beside their implementation under the nearest
`run_local/` package and end in `*_run.py`. Each exposes `Locals` and
`run_local(local=...)`; invoke it explicitly, for example with
`python -m stochvolmodels.pricers.run_local.heston_pricer_run`. Pytest modules remain in
`src/stochvolmodels/tests/`, and distributions exclude development runners.

Build and inspect an installed artifact with:

```console
python -m pip install build twine
python -m build
python -m twine check dist/*
python scripts/check_wheel_contents.py dist/*.whl
```

Run installed-wheel tests from a directory outside the checkout so local source cannot mask a
packaging error:

```console
python -m pytest --pyargs stochvolmodels -m "not slow"
```

The wheel ships automated test modules but excludes repository examples, documentation, paper
sources and development runners. Tests that need these assets skip when they are absent; this is
why an installed-wheel pass complements the source run. Report executed commands, skips and
failures separately, including whether the slow and optional-provider lanes were actually run.

## Numerical imports

Use `numpy` as `np` for elementary operations and constants, including scalar calculations:
`np.sqrt`, `np.log`, `np.exp`, `np.pi` and `np.isfinite`. Use `scipy.special` for special
functions such as `erf`, `erfc`, `gamma` and `comb`; request `comb(..., exact=True)` when coefficients
must remain exact integers. Preserve complex input types, integer conversions, tolerances
and overflow/domain handling when replacing an implementation.

Ruff's `TID251` rule rejects `math` and `cmath` imports throughout `src/stochvolmodels`,
including tests and development runners. An exception must be an explicit symbol import
with a line-level `noqa: TID251` and a concrete numerical or compilation reason. Do not
add blanket file or directory exclusions. The current exceptions are:

- `math.erf` in `utils/mc_payoffs.py`: the scalar normal-CDF helper is compiled by Numba.
  Ordinary Python and array code uses `scipy.special.erf`; calling it from this compiled
  kernel would require optional integration that the package does not declare.
- `math.fsum` in `valuation.py`: ordinary NumPy/SciPy reductions do not guarantee the same
  accuracy for sums with severe cancellation.

Keep exceptions narrow and verify them through the compiled kernel or an independent
numerical reference. Adding optional compilation support is a dependency change, not an
implicit part of replacing an import.

## Fitter annotations and module headers

Every function under `src/stochvolmodels/fitters/` requires argument and return annotations.
This includes private helpers, nested optimizer callbacks, adapters and `run_local/` dispatchers.
Conventional instance/class receivers may omit `self`/`cls` annotations; collection and callable
annotations must include their contained types or signatures.

Each module starts with a docstring containing a summary and an explanatory paragraph. State the
module's responsibility, important input/output units or shapes, and relevant ownership boundaries
or limitations. Keep detailed function documentation in NumPy-style docstrings.

`python scripts/check_fitter_contracts.py` checks these contracts from syntax trees without
importing optional providers. The static CI workflow runs it alongside the Ruff import rules and
docstring-coverage check. `test_fitter_contracts.py` verifies that incomplete headers, missing
annotations and bare collection annotations are rejected.

## Numerical changes

- Add a test that fails before the fix and passes after it.
- Cross-check pricing, transforms, calibration, or simulation through a different route:
  analytic versus Monte Carlo, a limiting case, a closed-form identity, or a trusted external
  reference.
- Confirm that the regression rejects its intended defect, for example a wrong tick scale,
  acceptance of a price at an open bound, or a missing saved draw. Restore the correct
  implementation afterward and record the failing and passing checks. A mutation check is
  separate evidence from an ordinary passing test run.
- Preserve quote identities, units, missing-data reasons and all requested observations or draws.
  Native admissibility and synthetic sensitivity checks do not establish quote freshness, wing
  accuracy or a production warning policy.
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
