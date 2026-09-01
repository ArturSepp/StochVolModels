## Python environment (mandatory)

- Never create, use, or install packages into a Python virtual environment anywhere under `C:\Users\artur\OneDrive`.
- Keep this repository's environment outside OneDrive at `C:\Python\StochVolModels312`.
- Use `C:\Python\StochVolModels312\Scripts\python.exe` for Python, tests, linters, and package installation.
- If it is missing, create it with `py -3.12 -m venv C:\Python\StochVolModels312`.
- Never run plain `uv sync` or plain `uv run` from this checkout: uv otherwise creates `<repo>\.venv` even when uv was launched through a Python executable under `C:\Python`.
- If a uv project operation is required, first set `UV_PROJECT_ENVIRONMENT=C:\Python\StochVolModels312`; for pip-style operations prefer `uv pip ... --python C:\Python\StochVolModels312\Scripts\python.exe`.
- If any OneDrive-local environment already exists, do not use it; report it for removal.

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

This package is one of nine open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `privateassets` | PrivateAssets | Private-assets analytics |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` and `privateassets` depend on `qis`; `stochvolmodels`
depends on `vanilla-option-pricers` and has an optional `research` extra that pulls in
`qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
src/stochvolmodels/
  pricers/           log-normal SV, Heston, Hawkes jump-diffusion, Gaussian mixture,
                     Student-t; subpackages logsv/, factor_hjm/, rough_logsv/;
                     source-adjacent run_local/ packages hold development runners
  data/              option-chain containers; run_local/ holds development diagnostics
  fitters/           approximate LogSV smile fitter; run_local/ holds development diagnostics
  utils/             numerical utilities (Fourier transforms, quadrature, plotting)
  tests/             test modules (test_*.py) — inside the package
examples/            repository-only runnable examples, grouped by task
papers/              replication code, 9 directories
  <paper>/paper/     article PDF and LaTeX source, where available
docs/                Sphinx/Furo user guide, API reference, and adoption documentation
CHANGELOG.md         every public change is recorded here
```

There is no top-level `tests/` directory. Automated tests live in
`src/stochvolmodels/tests/`. Component development runners live beside their implementation in
the nearest `run_local/<subject>_run.py`, expose `Locals` and `run_local(local=...)`, and are
excluded from built distributions.

## Commands

```bash
uv sync --group test --locked
uv run pytest -m "not slow"                       # core-fast source suite
uv run pytest -m slow                             # numerical regression/simulation suite
uv run pytest --cov=stochvolmodels --cov-report=json
uv run python scripts/check_coverage_scopes.py coverage.json
uv run --only-group lint ruff check --select F,TID251,TID253,ICN src/stochvolmodels
uv run --only-group lint interrogate -v src/stochvolmodels
python -m pytest --pyargs stochvolmodels          # complete installed-wheel collection
```

Optional extras: `research` (pulls in `qis` and OCA), `visualization`, `numerical`, `jupyter`,
`docs`, `all`. The `test` dependency group includes `pytest-regressions`. Supported Python is
>= 3.10. CI runs Linux 3.10 - 3.14, Windows 3.12 numerical regressions, macOS 3.12 smoke tests, a core-only installed
wheel, cross-platform quickstarts, documentation/doctests/link checks, and scheduled audits.

## Conventions

- Test files are named `test_*.py` and live in `src/stochvolmodels/tests/`. Nothing named
  `test_*.py` sits anywhere else in the package.
- Following OptimalPortfolios, component development dispatchers live in the nearest
  `run_local/<module>_run.py` and use `Locals` plus `run_local(local=...)`. Production modules do
  not import them. Every pytest-shaped file is an automated test and collects at least one test.
- Line length 100 (`ruff`, configured rules `E`, `F`, `W`; CI gates correctness and import
  boundaries through `F`, `TID251`, `TID253`, and `ICN`).
- Pricing kernels are `numba`-compiled (14 modules import numba): keep them array-based,
  avoid pandas inside compiled code, and preserve the existing signature style. A
  mutable default such as `arr: np.ndarray = None` cannot be typed in nopython mode.
- Each model provides both an analytic transform-based pricer and a Monte Carlo
  simulator; new models are expected to provide both so they can be cross-validated.
- Dataclasses carry model parameters; enums carry model and option type selection.
- Runnable examples sit behind an enum of cases plus a dispatcher called under
  `if __name__ == '__main__':`. Package development runners use `Locals` / `run_local`;
  repository examples retain their existing case enums, and `papers/` uses
  `UnitTests` / `run_unit_test`.
- Runnable examples live under root `examples/`, are repository-only, and are excluded from the
  wheel. Stable user examples use the public API; advanced examples may use internals when labelled.
- Regression tests use `pytest-regressions`; when output legitimately changes, update
  the stored regression files deliberately and say so.
- Docstrings are NumPy-style. Where a function implements a published result, the
  docstring cites the equation number — see "Papers and equation numbering".

## Paths

Nothing in `papers/` or `examples/` hardcodes a developer filesystem path. Output and input
directories resolve through the package-wide module:

```python
from stochvolmodels import local_path as lp

qis.save_fig(fig, file_name='fig_1', local_path=lp.get_output_path())
bbg_vols_path = f"{lp.get_resource_path()}bbg_vols\\"
```

`src/stochvolmodels/settings.yaml` is the ignored machine-local configuration; copy the committed
`settings.yaml.example` beside it. Missing keys default to the ignored `resources/` and `outputs/`
directories under the repository root. The getters return absolute strings with a trailing path
separator, matching the QIS ecosystem convention. Convert to `Path` only when a consumer needs
path operations. Do not reintroduce absolute paths outside the ignored YAML.

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
  `stochvolmodels.local_path` precisely to avoid becoming one.

## Repository-specific agent artifacts

By maintainer direction, all StochVolModels roadmaps, execution plans, audits, and reports live in
the ignored `agents/` directory. This repository-specific rule overrides the generic roadmap
location inside the generated shared-agent block below; do not edit that generated block directly.

<!-- ===== SHARED AGENT CORE (standalone variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-08-08, agent core v1.2. -->

## Domain invariants

- **No look-ahead in any rolling or expanding estimation.** Estimation is point-in-time; a
  full-sample statistic inside a rolling path is forward-looking and wrong even when it runs
  clean.
- Conventions are stated, never implied: volatility quotation, rate and dividend conventions,
  annualisation. One convention per concept across the stack — if this package and a sibling
  disagree, that is a bug to report, not a difference to accommodate.

## Dependency surface

This package depends on the lower-level `vanilla-option-pricers` package for Black-Scholes-Merton
and Bachelier analytics. Nothing else from the stack is a runtime dependency — `qis` enters only
via the optional `research` extra used by `papers/`. Ask before adding any dependency.

**Never invent a symbol.** If a function, class, or keyword argument is not in the export
surface of this package or of a dependency, it does not exist. Check in one line —
`python -c "import stochvolmodels; print([n for n in dir(stochvolmodels) if not n.startswith('_')])"`
— and say a symbol is missing rather than producing code that calls it.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean: calibration objectives, Fourier/characteristic-function integration, anything
  simulated. Verify against a reference computed a different way — here the analytic pricer and
  the Monte Carlo simulator cross-validate each other — and say which.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Roadmap execution

Feature roadmaps live at the repository root as `ROADMAP_<feature>.md`. An execution request
names the file and the stage. A stage is complete when its stated verification command passes;
its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->

## Replication contract

`papers/` reproduces the results of the published papers, including the log-normal
stochastic volatility model with quadratic drift (IJTAF 2023, 26(8) 2450003), stochastic
volatility for the factor HJM framework (Review of Derivatives Research 2025, 28:12), and
cryptocurrency inverse options. Changes to pricers, transforms, or simulation must be
verified against these before being proposed.

## Release checklist

1. `version` in `pyproject.toml`
2. an entry in `CHANGELOG.md`: version, date, and the change classified as
   added / changed / fixed / removed, naming the public symbol that changed
3. the software BibTeX entry in `README.md` (if it pins a version)
4. `CITATION.cff` version, release date, and preferred-citation metadata

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.

## Known issues

- `src/stochvolmodels/pricers/rough_logsv/` and `pricers/factor_hjm/` are experimental public
  module paths. Preserve compatibility, and do not refactor their numerical kernels unless asked.
- The legacy rough Gaussian quadrature compatibility function deliberately raises `ImportError`;
  `orthopy` and `quadpy` are not supported dependencies.
- Current Fourier pricers preserve their historical even-grid integration weights behind a private
  compatibility helper. Changing that numerical path requires separate baseline approval.
- The `rtol=1e-7` regression tolerance has roughly 7x headroom over the observed
  Linux-versus-Windows deviation.
