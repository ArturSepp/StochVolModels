# stochvolmodels.utils

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../CITATION.cff).

Numerical transforms, integration, payoff calculations, rates, plotting and array utilities shared by more than one model.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Numerical transforms, integration, payoff calculations, rates, plotting and array utilities shared by more than one model. | — | [quickstart.py](../../../examples/getting_started/quickstart.py) | — |
| [`_arrays.py`](_arrays.py) | Shared immutable NumPy array construction for models and fitters. | — | [test_dupire.py](../tests/test_dupire.py) | — |
| [`_fourier_quadrature.py`](_fourier_quadrature.py) | Build Gauss–Kronrod vanilla integration plans and estimate resolution and tail errors in premium units. | [`utils.mgf_pricer`](mgf_pricer.py) | [test_fourier_quadrature.py](../tests/test_fourier_quadrature.py) | — |
| [`config.py`](config.py) | Define payoff codes, state variables and coordinate-system conventions. | — | [run_heston_pricer.py](../../../examples/pricers/run_heston_pricer.py) | — |
| [`funcs.py`](funcs.py) | Numerical and timing utilities shared across the package. | — | [logsv_model.py](../../../examples/docs/logsv_model.py) | — |
| [`mc_payoffs.py`](mc_payoffs.py) | Monte Carlo payoff evaluation for vanilla, inverse and quadratic-variance options. | [`utils.config`](config.py) | [test_rough_logsv_mc_conventions.py](../tests/test_rough_logsv_mc_conventions.py) | — |
| [`mgf_pricer.py`](mgf_pricer.py) | Integrate Fourier payoff transforms for vanilla and quadratic-variation contracts. | [`utils.config`](config.py) | [run_bsm_mgf_pricer.py](../../../examples/pricers/run_bsm_mgf_pricer.py) | — |
| [`plots.py`](plots.py) | Plot volatility smiles, distributions, model comparisons and diagnostics. | [`local_path`](../local_path.py) | [run_gmm_pricer.py](../../../examples/pricers/run_gmm_pricer.py) | — |
| [`rate_core.py`](rate_core.py) | Interest rate conventions: discount factors, bonds, swap and Libor rates. | — | [run_rate_logsv_pricer.py](../../../examples/pricers/run_rate_logsv_pricer.py) | — |
| [`var_swap_pricer.py`](var_swap_pricer.py) | Model-free variance swap strike from a slice of option prices. | — | [quadratic_variance_options.py](../../../examples/docs/quadratic_variance_options.py) | — |
