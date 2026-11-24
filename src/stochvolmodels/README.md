# stochvolmodels

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../CITATION.cff).

The package root exposes stable names lazily. The two standalone modules value simulated paths or densities and resolve user-configured storage paths.

See the [software_design.md](../../docs/software_design.md) for the public API tiers and the [run_add_model.py](../../examples/pricers/run_add_model.py) for an executable extension.

- [data](data/README.md) — Represent market observations, simulated states and densities.
- [estimation](estimation/README.md) — Estimate historical volatility and construct realized-variance targets from observed prices.
- [fitters](fitters/README.md) — Estimate parameters from market quotes, infer local volatility or leverage, and measure residuals and surface consistency.
- [models](models/README.md) — Declare model capabilities and parameter ownership.
- [pde_solvers](pde_solvers/README.md) — Discretise forward and backward equations on one-dimensional and tensor grids.
- [pricers](pricers/README.md) — Combine a model with an instrument and valuation method.
- [products](products/README.md) — Define cash flows from simulated observations, including European options and integrated variance.
- [tests](tests/README.md) — Regression tests and fixtures for the installed package and checkout workflows.
- [utils](utils/README.md) — Numerical transforms, integration, payoff calculations, rates, plotting and array utilities shared by more than one model.

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Resolve stable, advanced and compatibility exports on first access. | Lazy [`data`](data/__init__.py), [`data.option_chain`](data/option_chain.py), [`data.sample_option_chains`](data/sample_option_chains.py), [`models.tdist.distribution`](models/tdist/distribution.py), [`pricers`](pricers/__init__.py), [`pricers.gmm_pricer`](pricers/gmm_pricer.py), [`pricers.hawkes_jd_pricer`](pricers/hawkes_jd_pricer.py), [`pricers.heston_pricer`](pricers/heston_pricer.py), [`pricers.logsv.affine_expansion`](pricers/logsv/affine_expansion.py), [`pricers.logsv.logsv_params`](pricers/logsv/logsv_params.py), [`pricers.logsv.vol_moments_ode`](pricers/logsv/vol_moments_ode.py), [`pricers.logsv_pricer`](pricers/logsv_pricer.py), [`pricers.model_pricer`](pricers/model_pricer.py), [`pricers.tdist.analytics`](pricers/tdist/analytics.py), [`pricers.tdist_pricer`](pricers/tdist_pricer.py), [`utils`](utils/__init__.py), [`utils.config`](utils/config.py), [`utils.funcs`](utils/funcs.py), [`utils.mc_payoffs`](utils/mc_payoffs.py), [`utils.mgf_pricer`](utils/mgf_pricer.py), [`utils.plots`](utils/plots.py) | [run_add_model.py](../../examples/pricers/run_add_model.py) | — |
| [`local_path.py`](local_path.py) | Resolve machine-local resource and output directories. | — | [test_local_path.py](tests/test_local_path.py) | `yaml` |
| [`valuation.py`](valuation.py) | Path and density valuation with explicit measure and quadrature conventions. | [`data.model_paths`](data/model_paths.py), [`products`](products/__init__.py), [`products.payoffs`](products/payoffs.py), [`utils._arrays`](utils/_arrays.py), [`utils.config`](utils/config.py)<br>Lazy [`data.density`](data/density.py) | [test_path_valuation.py](tests/test_path_valuation.py) | — |
