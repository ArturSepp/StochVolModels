---
myst:
  html_meta:
    description: >-
      How stochvolmodels is tested: source and installed-wheel suites, numerical import and
      fitter contracts, native-unit diagnostics, synthetic research evidence, coverage scopes,
      and independent numerical and documentation checks.
---

# Testing and coverage

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-22](https://github.com/ArturSepp/StochVolModels/commit/db6ea3ec2e724ae8d08e19bc9a8e41054b6f4dd7)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The automated checks separate scientific behavior from packaging and documentation so a failure
identifies the contract that changed. The fast source suite covers deterministic pricing,
calibration, conventions, and repository contracts. Slow tests preserve numerical regression and
simulation results. A clean core-only wheel is tested outside the checkout over the complete
shipped package. The commands below select different checks; passing a focused fast run does not
establish that the slow, optional-provider, installed-wheel or documentation lanes passed.

```bash
pytest -m "not slow"
pytest -m slow
pytest -m paper_replication
python -m build
python scripts/check_wheel_contents.py dist/*.whl
python -m pytest --pyargs stochvolmodels -m "not slow"
```

Use the interpreter and generated-output policy in
[`AGENTS.md`](https://github.com/ArturSepp/StochVolModels/blob/main/AGENTS.md). In the maintainer's
Windows OneDrive checkout, use the external environment under `C:\Python\StochVolModels312`
through the repository launcher; never create a virtual environment under OneDrive.

The [source CI](https://github.com/ArturSepp/StochVolModels/blob/main/.github/workflows/ci.yml)
installs the `research` extra for the authoritative Python 3.12 source run. This exercises available
optional-provider contracts as well as repository examples. Other core environments may skip
tests marked `optional_integration`. The separate
[synthetic OCA lane](https://github.com/ArturSepp/StochVolModels/blob/main/.github/workflows/examples.yml)
checks the optional adapter without market credentials.

The wheel includes the package's automated test modules, including native quote diagnostics.
Repository examples, documentation, paper sources and component `run_local/` runners are excluded.
Tests marked `repository_only` skip when their required repository assets are absent; the marker
does not remove those test modules from the wheel. The wheel-content check compares the shipped
test-module inventory with the source inventory, and the installed test run happens outside the
checkout so local files cannot conceal a packaging error.

## Source contracts

The [static workflow](https://github.com/ArturSepp/StochVolModels/blob/main/.github/workflows/static.yml)
runs Ruff with `F,TID251,TID253,ICN`, checks stable docstring coverage, and runs
`scripts/check_fitter_contracts.py`. These checks complement numerical tests:

- Elementary functions and constants use NumPy; special functions use `scipy.special`.
  Exact integer combinations use `scipy.special.comb(..., exact=True)`. Ruff rejects `math` and
  `cmath` imports except for documented, symbol-specific import-line exceptions. The current
  exceptions are the compiled `math.erf` helper and cancellation-sensitive `math.fsum` sums;
  see the [numerical import convention](https://github.com/ArturSepp/StochVolModels/blob/main/CONTRIBUTING.md#numerical-imports).
- Every function under `fitters/` has argument and return annotations, including private helpers,
  nested callbacks and development dispatchers. Conventional instance/class receivers may omit
  `self`/`cls` annotations; collection annotations specify their element types.
- Every fitter module has a summary and an explanatory paragraph covering its role and contracts.
  The syntax-tree checker does not import providers or run calibration. Its own regression tests
  reject missing annotations, bare collection types and inadequate headers.

```console
python -m ruff check --select F,TID251,TID253,ICN src/stochvolmodels
python scripts/check_fitter_contracts.py
python -m pytest src/stochvolmodels/tests/test_fitter_contracts.py
```

## Coverage and scientific scope

Coverage reports two scopes. The **whole package** includes advanced and experimental research
code and is ratcheted at 44.50%. The **stable JOSS scope** excludes Factor HJM, rough LogSV,
Hawkes, and presentation/rate support modules and is ratcheted at 86.50%. The exact definition is
maintained in `scripts/coverage_scopes.json`; CI evaluates it from `coverage.json`. Experimental
code is therefore visible in the honest whole-package number but cannot dilute the stable scope.

The stable suite checks transforms and integration weights through both Python and Numba routes;
Fourier prices against delegated Black pricing; digital and density identities; affine ODEs
against an adaptive SciPy solver; analytic moments and prices against fixed-random Monte Carlo;
and successful synthetic GMM and Student-t calibration recovery. These independent numerical
contracts are the reason for the ratchet—the percentage is a summary, not a substitute for them.

Stable maintained docstrings are gated at 100%. Factor HJM, rough LogSV, and the private PDE
worktree remain explicitly experimental and are reported separately rather than presented as
stable API coverage.

The `paper_replication` lane is intentionally narrow and offline. It verifies the published LogSV
MGF normalization, moment-stability conditions, constant-volatility limit, and analytic-versus-
Monte-Carlo agreement. Full figures and private option-data calibrations remain documented manual
paper workflows.

## Diagnostic and research regressions

These maintained tests use deterministic synthetic inputs or small saved-report fixtures. They do
not require private historical output folders, and they do not rerun a market study or establish
a production warning threshold.

| Contract | Maintained tests | Independent evidence or rejected defect |
|---|---|---|
| Native quote units and scales | `test_native_quote_diagnostics.py`, `test_fit_diagnostics.py` | Hand-computed tick floors and open shock intervals, unit-rescaling invariance, scalar/vector agreement, and explicit missing scales; locked quotes and boundary equality remain visible |
| Synthetic research fixtures and tolerance ranks | `test_research_study_tools.py` | Matrix-based correlation reference, exact integer binomial enumeration, unchanged observations across hidden-truth variants, and unavailable cutoffs when required scores are missing |
| Saved research evidence | `test_research_study_tools.py` | SHA256 checks reject changed/missing artifacts and path escapes; score reconstruction uses every saved draw and actual training support, rejecting missing, duplicated or corrupted draw evidence |
| Model, pricer and volatility-data alignment | `test_heston_consistency.py`, `test_vol_data_consistency.py` | Pricing-variable and initial-state forwarding, price/volatility round trips, quote-preserving normalization and ragged expiry alignment |
| Historical and temporal diagnostics | `test_historical_surface_validation.py`, `test_longitudinal_replay.py` | Held-out inputs cannot enter calibration; missing observations, source reuse, expiry censoring and calibration/validation boundaries cannot become recovery evidence |

For a numerical change, first show that its regression fails on the defect, then restore the
implementation and show that it passes. A deliberate faulty scale, relaxed boundary or corrupted
report is useful evidence only when the intended assertion rejects it. Record such mutation checks
separately from ordinary suite results; a test's presence alone does not prove that it detects the
intended error. Keep statistical assumptions, full requested denominators and incomplete evidence
visible when interpreting research results.

The [surface diagnostics guide](surface_diagnostics.md) documents the native-scale example and
the boundary between quote admissibility, fit residuals, sensitivity and model accuracy.

## Documentation gates

Canonical scripts assert their documented numerical examples, and registered figures carry
provenance. The checks are marked `repository_only`: their scripts, pages and figure registry are
absent from the wheel, while the shipped test modules skip when those assets are unavailable.

| Gate | Command | What it checks |
|---|---|---|
| Canonical scripts | `pytest src/stochvolmodels/tests/test_docs_examples.py` | Every case of every script under `examples/docs/` that `scripts/docs_inventory.json` names, run offline with network access refused, asserting the numbers its page quotes; the cases a script lists in `SLOW_CASES` run in the slow lane |
| Page standard | `python scripts/check_docs.py` | Page forms and required headings, bylines and descriptions; Python blocks that are verbatim excerpts of the page's script; one owning page for every stable and advanced export and every parameter, and its section in the [API reference](api.md); GitHub math pitfalls; local links; retired citation strings |
| Exhibits | `python -m scripts.docs_analytics.run --list` and `--verify` | Every image under `docs/images/` is registered, displayed by the pages that claim it, and matches the committed manifest; `python -m scripts.docs_analytics.validate --run-root <directory>` checks a regenerated bundle and its recorded checks |
| Tooling | `pytest src/stochvolmodels/tests/test_docs_tooling.py` | The checks above pass on the checkout and fail on seeded defects |
| Sphinx | `python -m sphinx -E -W -b html docs <directory>`, with `-b doctest` and `-b linkcheck` | Warnings are errors; the doctest block below runs; external links resolve (link checking runs on demand in CI) |

`check_docs.py --all` additionally requires every page to be adopted after a review of its rendering
on GitHub, in Sphinx and in an editor. The [documentation standard](documentation_standard.md)
defines the forms and the exhibit classes, and the [analytics gallery](analytics_gallery.md) lists
every registered exhibit.

## Numerical verification map

| Claim | Primary route | Independent route |
|---|---|---|
| European option prices | Fourier/MGF analytic pricer | Monte Carlo with sampling error |
| Heston implementation | Heston transform | limiting cases and Monte Carlo |
| LogSV implementation | affine expansion | moment ODE and Monte Carlo |
| Implied volatility | model price inversion | delegated vanilla-pricer round trip |
| Calibration | optimizer objective | repricing residuals on synthetic chains |

The public option-type enum is available on a core install:

```{doctest}
>>> from stochvolmodels import OptionType
>>> OptionType.CALL.name
'CALL'
```
