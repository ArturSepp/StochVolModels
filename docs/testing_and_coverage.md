# Testing and coverage

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
