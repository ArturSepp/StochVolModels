# stochvolmodels

`stochvolmodels` provides Fourier-transform pricing, Monte Carlo validation, and calibration of
European options under stochastic-volatility models in Python.

The package connects one option-chain representation to Heston and Karasinski-Sepp log-normal SV
pricing, implied-volatility inversion, constrained calibration, Monte Carlo checks, and the
research implementations associated with the project. It is intentionally narrower than a
general derivatives platform.

Start with [installation and first result](getting_started.md). The complete offline script runs
against the installed wheel without data downloads, credentials, plots, or optional research
dependencies.

## Choose a workflow

- [Price European options](european_option_pricing.md)
- [Build and interpret option chains](option_chains_and_conventions.md)
- [Use the LogSV model](logsv_model.md)
- [Use the Heston model](heston_model.md)
- [Calibrate to market implied volatilities](calibration.md)
- [Validate analytic prices against Monte Carlo](analytic_vs_monte_carlo.md)
- [Choose this package or an alternative](package_comparison.md)

## Project links

- [PyPI](https://pypi.org/project/stochvolmodels/)
- [Source repository](https://github.com/ArturSepp/StochVolModels)
- [Issue tracker](https://github.com/ArturSepp/StochVolModels/issues)
- [Contributing and support](contributing.md)
- [JOSS paper draft](https://github.com/ArturSepp/StochVolModels/blob/main/paper.md)
- [Changelog](https://github.com/ArturSepp/StochVolModels/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff)

```{toctree}
:maxdepth: 2
:caption: User guide

getting_started
european_option_pricing
option_chains_and_conventions
logsv_model
heston_model
calibration
analytic_vs_monte_carlo
numerical_accuracy_and_performance
testing_and_coverage
reproducing_the_papers
package_comparison
api
contributing
```
