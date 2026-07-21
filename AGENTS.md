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
`papers/` reproduces the computations and figures of the published papers.
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
  pricers/           log-normal SV, Heston, Hawkes jump-diffusion, Gaussian mixture,
                     Student-t; subpackages analytic/, logsv/, factor_hjm/, rough_logsv/
  data/              option chain containers and market data
  utils/             numerical utilities (Fourier transforms, quadrature, plotting)
  examples/          runnable examples
  tests/             test modules (test_*.py) — inside the package
papers/              replication code, 8 directories
  <paper>/paper/     article PDF and LaTeX source, where available
docs/                two figures used by README.md
CHANGELOG.md         every public change is recorded here
```

There is no top-level `tests/` directory; tests live in `stochvolmodels/tests/`.

## Commands

```bash
pip install -e ".[dev]"
pytest                                 # testpaths points at stochvolmodels/tests
pytest stochvolmodels/tests/ -v        # what CI runs
ruff check stochvolmodels/             # lint
```

Optional extras: `research` (pulls in `qis`), `visualization`, `numerical`, `jupyter`,
`dev` (includes `pytest-regressions`), `all`. Supported Python is >= 3.10; CI runs
3.10 - 3.12.

## Conventions

- Test files are named `test_*.py` and live in `stochvolmodels/tests/`. Nothing named
  `test_*.py` sits anywhere else in the package.
- Line length 100 (`ruff`, rules `E`, `F`, `W`, `I`).
- Pricing kernels are `numba`-compiled (14 modules import numba): keep them array-based,
  avoid pandas inside compiled code, and preserve the existing signature style. A
  mutable default such as `arr: np.ndarray = None` cannot be typed in nopython mode.
- Each model provides both an analytic transform-based pricer and a Monte Carlo
  simulator; new models are expected to provide both so they can be cross-validated.
- Dataclasses carry model parameters; enums carry model and option type selection.
- Runnable examples sit behind an enum of cases plus a dispatcher called under
  `if __name__ == '__main__':`. The package uses `LocalTests` / `run_local_test`;
  `papers/` uses `UnitTests` / `run_unit_test`.
- Regression tests use `pytest-regressions`; when output legitimately changes, update
  the stored regression files deliberately and say so.
- Docstrings are NumPy-style. Where a function implements a published result, the
  docstring cites the equation number — see "Papers and equation numbering".

## Paths

Nothing in `papers/` hardcodes a filesystem path. Output and input directories resolve
through `papers/local_path.py`:

```python
from papers import local_path as lp

qis.save_fig(fig, file_name='fig_1', local_path=lp.get_output_path())
```

Resolution order: `papers/settings.yaml` if present, otherwise `docs/figures` and
`resources` under the repository root. Both defaults are gitignored, so a fresh clone
runs with no configuration. `papers/settings.yaml` is gitignored; the committed
template is `papers/settings.yaml.example`. Do not reintroduce absolute paths.

## Papers and equation numbering

Two directories carry the article and its LaTeX source under `paper/`. Docstrings that
cite equations cite the **published PDF**, not the `.tex`, because the numbering
differs in both cases:

- Log-normal SV: the PDF numbers by section, the source sequentially. Source (1), (2),
  (3) are the PDF's (2.1), (2.2), (2.3).
- Factor HJM: both number sequentially, but the source predates the revision that added
  the auxiliary factor, so numbering diverges after equation (2).

Each `paper/README.md` records this. Do not "correct" an equation reference against the
LaTeX source.

## Constraints — do not do these

- Do not change model parameterisations or the moment generating function without
  re-running `papers/` — published papers depend on this code.
- Do not make `qis` a hard dependency: it is an optional `research` extra used by the
  paper code, not by the pricing library. `yfinance` is likewise test and example only.
- Do not add exotic or path-dependent payoffs; this package covers European vanillas
  under stochastic volatility by design.
- Do not silently regenerate `pytest-regressions` baselines to make a failing test pass.
- Do not commit calibration output or figures.
- Do not add a dependency without asking. PyYAML is imported lazily in
  `papers/local_path.py` precisely to avoid becoming one.

## Replication contract

`papers/` reproduces the results of the published papers, including the log-normal
stochastic volatility model with quadratic drift (IJTAF 2024, 26(8) 2450003), stochastic
volatility for the factor HJM framework (Review of Derivatives Research 2025, 28:12), and
cryptocurrency inverse options. Changes to pricers, transforms, or simulation must be
verified against these before being proposed.

## Release checklist

1. `version` in `pyproject.toml`
2. an entry in `CHANGELOG.md`: version, date, and the change classified as
   added / changed / fixed / removed, naming the public symbol that changed
3. the software BibTeX entry in `README.md` (if it pins a version)

[TODO: `CITATION.cff` is referenced by some tooling conventions but does not exist in
this repository. Either add one and include it in this list, or leave it out.]

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.

## Known issues

- `stochvolmodels/pricers/logsv/logsv_params.py` imports from the package root
  (`from stochvolmodels import ...`), the one remaining module-level import cycle.
- `pricers/rough_logsv/` is work in progress and is deliberately less documented than
  the rest of the package. Leave it alone unless asked.
- `utils/mc_payoffs.py:compute_mc_vars_payoff` returns a zero payoff for an unrecognised
  option type code rather than raising.
- `utils/mgf_pricer.py:compute_integration_weights` applies Simpson's rule without
  checking that the grid has an odd number of points.
- The `rtol=1e-7` regression tolerance has roughly 7x headroom over the observed
  Linux-versus-Windows deviation.
- `pyproject.toml` classifiers claim Python 3.13 but the CI matrix stops at 3.12.
