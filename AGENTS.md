# AGENTS.md

Guidance for AI coding agents working in the **StochVolModels** repository.

## Project overview

`stochvolmodels` implements pricing analytics and Monte Carlo simulation for stochastic
volatility models — the Karasinski-Sepp log-normal stochastic volatility model (with
quadratic drift) and the Heston model as benchmark — for European calls and puts and
their implied volatilities. The design is a single generic model interface: a
closed-form moment generating function for Fourier-transform pricing on one side, and
Monte Carlo dynamics on the other, so analytic and simulated prices are directly
comparable.

This is the reference implementation maintained by one of the model's originators.
`my_papers/` reproduces the computations and figures of five published papers.
Distribution and import name `stochvolmodels`. Licensed MIT (`LICENSE.txt`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
stochvolmodels/
  pricers/    model implementations (log-normal SV, Heston, rough log-SV)
  data/       option chain containers and market data
  utils/      numerical utilities (Fourier transforms, quadrature)
  examples/   runnable examples
  tests/      test modules (test_*.py) — inside the package
my_papers/    replication code for five published papers
docs/         documentation sources
```

There is no top-level `tests/` directory; tests live in `stochvolmodels/tests/`.

## Commands

```bash
pip install -e ".[dev]"
pytest stochvolmodels/tests/ -v        # as CI runs it (see Known issues)
ruff check stochvolmodels/             # lint
```

Optional extras: `research` (pulls in `qis`), `visualization`, `numerical`, `jupyter`,
`dev` (includes `pytest-regressions`), `all`. Supported Python is >= 3.9; CI runs
3.10 - 3.12.

## Conventions

- Test files are named `test_*.py` and live in `stochvolmodels/tests/` (plus a few
  alongside the pricers they cover).
- Line length 100 (`ruff`, rules `E`, `F`, `W`, `I`).
- Pricing kernels are `numba`-compiled (28 modules use numba): keep them array-based,
  avoid pandas inside compiled code, and preserve the existing signature style.
- Each model provides both an analytic transform-based pricer and a Monte Carlo
  simulator; new models are expected to provide both so they can be cross-validated.
- Dataclasses carry model parameters; enums carry model and option type selection.
- Regression tests use `pytest-regressions`; when output legitimately changes, update
  the stored regression files deliberately and say so.

## Constraints — do not do these

- Do not change model parameterisations or the moment generating function without
  re-running `my_papers/` — five published papers depend on this code.
- Do not make `qis` a hard dependency: it is an optional `research` extra used by the
  paper code, not by the pricing library.
- Do not add exotic or path-dependent payoffs; this package covers European vanillas
  under stochastic volatility by design.
- Do not silently regenerate `pytest-regressions` baselines to make a failing test pass.
- Do not commit calibration output or figures.

## Replication contract

`my_papers/` reproduces the results of five papers, including the log-normal stochastic
volatility model with quadratic drift (IJTAF 2023) and cryptocurrency inverse options
(Quantitative Finance 2024). Changes to pricers, transforms, or simulation must be
verified against these before being proposed.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.

## Known issues

`[tool.pytest.ini_options] testpaths = ["tests"]` points at a directory that does not
exist at the repository root, so a bare `pytest` collects nothing — this is why CI
passes an explicit path. Setting `testpaths = ["stochvolmodels"]` would fix it if the
maintainer asks.
