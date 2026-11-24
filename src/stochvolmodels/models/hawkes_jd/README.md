# stochvolmodels.models.hawkes_jd

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Specify two interacting jump intensities and solve the associated affine transform for clustered positive and negative jumps.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Specify two interacting jump intensities and solve the associated affine transform for clustered positive and negative jumps. | Lazy [`models.hawkes_jd.params`](params.py), [`models.hawkes_jd.transforms`](transforms.py) | [run_hawkes_jd_pricer.py](../../../../examples/pricers/run_hawkes_jd_pricer.py) | — |
| [`params.py`](params.py) | Specify mean reversion, jump-size laws and self- and cross-excitation parameters. | [`models.base`](../base.py) | [run_hawkes_jd_pricer.py](../../../../examples/pricers/run_hawkes_jd_pricer.py) | — |
| [`transforms.py`](transforms.py) | Propagate the affine transform for the bivariate Hawkes jump-diffusion. | [`models.hawkes_jd.params`](params.py) | [run_hawkes_jd_pricer.py](../../../../examples/pricers/run_hawkes_jd_pricer.py) | — |
