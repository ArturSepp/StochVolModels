# stochvolmodels.pricers.logsv

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Historical import paths for the LogSV parameters, affine expansion, moments and backbone fitter. Follow each row to its current implementation.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Historical import paths for the LogSV parameters, affine expansion, moments and backbone fitter. | — | [test_logsv_expansion_errata.py](../../tests/test_logsv_expansion_errata.py) | — |
| [`affine_expansion.py`](affine_expansion.py) | Compatibility import for `stochvolmodels.models.logsv.affine_expansion`. | [`models.logsv.affine_expansion`](../../models/logsv/affine_expansion.py) | [test_logsv_expansion_errata.py](../../tests/test_logsv_expansion_errata.py) | — |
| [`logsv_params.py`](logsv_params.py) | Compatibility imports for `stochvolmodels.models.logsv.params`. | [`models.logsv.kernel`](../../models/logsv/kernel.py), [`models.logsv.moments`](../../models/logsv/moments.py), [`models.logsv.params`](../../models/logsv/params.py) | [test_pde_logsv.py](../../tests/test_pde_logsv.py) | — |
| [`vol_moments_ode.py`](vol_moments_ode.py) | Compatibility entry points for LogSV moments and backbone calibration. | [`fitters.logsv_backbone`](../../fitters/logsv_backbone.py), [`models.logsv.moments`](../../models/logsv/moments.py), [`models.logsv.params`](../../models/logsv/params.py) | [volatility_distribution_and_moments.py](../../../../examples/docs/volatility_distribution_and_moments.py) | — |
