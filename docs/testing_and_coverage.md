---
myst:
  html_meta:
    description: >-
      How stochvolmodels is tested: fast and slow suites, paper-replication checks, coverage scopes,
      the documentation gates that assert every quoted number and figure, and the numerical
      verification map of analytic against independent routes.
---

# Testing and coverage

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-22](https://github.com/ArturSepp/StochVolModels/commit/db6ea3ec2e724ae8d08e19bc9a8e41054b6f4dd7)*

Part of the [stochvolmodels](https://github.com/ArturSepp/StochVolModels) documentation.
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The automated checks separate scientific behavior from packaging and documentation so a failure
identifies the contract that changed. The fast source suite covers deterministic pricing,
calibration, conventions, and repository contracts. Slow tests preserve numerical regression and
simulation results. A clean core-only wheel is tested outside the checkout over the complete
shipped package.

```bash
pytest -m "not slow"
pytest -m slow
pytest -m paper_replication
python -m build
python scripts/check_wheel_contents.py dist/*.whl
python -m pytest --pyargs stochvolmodels -m "not slow"
```

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

## Documentation gates

Every number the site quotes is asserted by a test, and every figure carries its provenance. These
checks are repository-only (marker `repository_only`) and are absent from the wheel.

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
