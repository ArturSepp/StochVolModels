# stochvolmodels.fitters.adapters

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Convert native quotes or optional OptionChainAnalytics observations into aligned calibration and diagnostic inputs. diagnostic_chain_from_oca returns the selected chain and a separate report of quote IDs, scales and source quality.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Convert native quotes or optional OptionChainAnalytics observations into aligned calibration and diagnostic inputs. | [`fitters.adapters.native_quotes`](native_quotes.py), [`fitters.adapters.oca`](oca.py) | [surface_diagnostics.py](../../../../examples/docs/surface_diagnostics.py) | — |
| [`native_quotes.py`](native_quotes.py) | Native quote scales and conditional additive-price admissibility diagnostics. | [`fitters.diagnostics`](../diagnostics.py) | [test_native_quote_diagnostics.py](../../tests/test_native_quote_diagnostics.py) | — |
| [`oca.py`](oca.py) | Convert point-in-time OptionChainAnalytics objects into SVM calibration inputs. | [`data.option_chain`](../../data/option_chain.py) | [test_oca_option_chain_adapter.py](../../tests/test_oca_option_chain_adapter.py) | `option_chain_analytics` |
| [`oca_diagnostics.py`](oca_diagnostics.py) | Return an SVM chain plus provenance for selected OCA contracts, units and quality. | [`data.option_chain`](../../data/option_chain.py), [`fitters.adapters.oca`](oca.py), [`fitters.diagnostics`](../diagnostics.py) | [oca_surface_diagnostics.py](../../../../examples/fitters/oca_surface_diagnostics.py) | `option_chain_analytics` |
