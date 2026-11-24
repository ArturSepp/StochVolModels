# stochvolmodels.models.rough_logsv

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Experimental rough LogSV kernels, Markovian lifts, splitting schemes, variance reduction and named reference controls.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Experimental rough LogSV kernels, Markovian lifts, splitting schemes, variance reduction and named reference controls. | Lazy [`models.rough_logsv.multilevel`](multilevel.py), [`models.rough_logsv.params`](params.py) | [test_public_api_contracts.py](../../tests/test_public_api_contracts.py) | — |
| [`_classical_pde.py`](_classical_pde.py) | Research-only Fourier/backward-PDE reference for a scalar volatility diffusion. | — | [monte_carlo_simulation.py](../../../../examples/docs/monte_carlo_simulation.py) | — |
| [`_constrained_kernel.py`](_constrained_kernel.py) | Private experimental positive kernel fits with hard geometric constraints. | [`models.rough_logsv.kernel_fit`](kernel_fit.py) | [test_rough_logsv_constrained_kernel.py](../../tests/test_rough_logsv_constrained_kernel.py) | — |
| [`_history.py`](_history.py) | Full-state weak splitting for finite exponential rough-logSV lifts. | — | [test_rough_logsv_standalone.py](../../tests/test_rough_logsv_standalone.py) | — |
| [`_history_fast.py`](_history_fast.py) | Algebraically fused implementation of the audited full-history split. | — | [test_rough_logsv_standalone.py](../../tests/test_rough_logsv_standalone.py) | — |
| [`expm.py`](expm.py) | Compute batched matrix exponentials and rank-one inverses for finite-factor lifts. | — | [test_rough_expm.py](../../tests/test_rough_expm.py) | — |
| [`history.py`](history.py) | Validated full-history solver and bounded conditional payoff interface. | [`models.rough_logsv._history`](_history.py), [`models.rough_logsv._history_fast`](_history_fast.py), [`models.rough_logsv.params`](params.py) | [test_rough_logsv_standalone.py](../../tests/test_rough_logsv_standalone.py) | — |
| [`kernel_fit.py`](kernel_fit.py) | Sum-of-exponentials kernels fitted to the moments of the rough log-normal Volterra model. | [`models.rough_logsv.rough_kernel`](rough_kernel.py) | [test_rough_logsv_kernel_fit.py](../../tests/test_rough_logsv_kernel_fit.py) | — |
| [`multilevel.py`](multilevel.py) | Bounded sampling, discrete classical controls and independent-scramble intervals. | [`models.rough_logsv.history`](history.py), [`models.rough_logsv.params`](params.py) | [test_rough_logsv_standalone.py](../../tests/test_rough_logsv_standalone.py) | — |
| [`params.py`](params.py) | Independent parameter and kernel objects for experimental rough log-normal SV. | [`models.base`](../base.py) | [run_rough_logsv_pricer.py](../../../../examples/pricers/run_rough_logsv_pricer.py) | — |
| [`rough_kernel.py`](rough_kernel.py) | Fractional kernel and its Markovian (sum-of-exponentials) quadrature approximations. | — | [test_rough_logsv_kernel_fit.py](../../tests/test_rough_logsv_kernel_fit.py) | — |
| [`split_simulation.py`](split_simulation.py) | Split-step Monte Carlo scheme for the rough log-normal Volterra stochastic volatility model. | [`models.rough_logsv.expm`](expm.py) | [test_rough_logsv_split_step.py](../../tests/test_rough_logsv_split_step.py) | — |
