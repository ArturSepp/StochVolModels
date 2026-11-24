# stochvolmodels.data

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../CITATION.cff).

Represent market observations, simulated states and densities. Use the bundled sample chains for offline examples. Provider loaders read separately installed or locally configured data.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Represent market observations, simulated states and densities. | — | [quickstart.py](../../../examples/getting_started/quickstart.py) | — |
| [`density.py`](density.py) | Immutable one- and two-dimensional densities with grid and coordinate metadata. | [`pde_solvers.grids`](../pde_solvers/grids.py), [`utils._arrays`](../utils/_arrays.py), [`utils.config`](../utils/config.py) | [run_density_pricer.py](../../../examples/pricers/run_density_pricer.py) | — |
| [`fetch_option_chain.py`](fetch_option_chain.py) | Load normalised panels or selected provider expiries from optional OCA sources. | [`data.option_chain`](option_chain.py), [`fitters.adapters.oca`](../fitters/adapters/oca.py), [`local_path`](../local_path.py) | [load_cboe_option_chain.py](../../../examples/data/load_cboe_option_chain.py) | `option_chain_analytics` |
| [`model_paths.py`](model_paths.py) | Canonical path payload shared by dynamic models and pathwise payoffs. | — | [test_path_valuation.py](../tests/test_path_valuation.py) | — |
| [`option_chain.py`](option_chain.py) | Carry forwards, discounts, strikes and quotes by expiry and invert model premiums to Black IV. | [`utils.rate_core`](../utils/rate_core.py), [`utils.var_swap_pricer`](../utils/var_swap_pricer.py) | [test_option_chain_characterization.py](../tests/test_option_chain_characterization.py) | — |
| [`sample_option_chains.py`](sample_option_chains.py) | Bundled option chains for deterministic examples and paper illustrations. | [`data.option_chain`](option_chain.py) | [run_sample_option_chains.py](../../../examples/data/run_sample_option_chains.py) | — |
