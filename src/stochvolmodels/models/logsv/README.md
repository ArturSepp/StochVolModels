# stochvolmodels.models.logsv

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Own the quadratic-drift log-normal volatility parameters, simulation, affine expansion, moments and PDE coefficients.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Own the quadratic-drift log-normal volatility parameters, simulation, affine expansion, moments and PDE coefficients. | Lazy [`data.model_paths`](../../data/model_paths.py), [`models.logsv.model`](model.py), [`models.logsv.params`](params.py) | [test_logsv_paths.py](../../tests/test_logsv_paths.py) | — |
| [`_calibration_sensitivities.py`](_calibration_sensitivities.py) | Propagate the classical PARAMS4 and PARAMS5 vanilla calibration sensitivities on the accepted coefficient time steps. | [`models.logsv.affine_expansion`](affine_expansion.py), [`models.logsv.params`](params.py) | [test_fourier_calibration.py](../../tests/test_fourier_calibration.py) | — |
| [`affine_expansion.py`](affine_expansion.py) | Affine expansion of the moment generating function of the log-normal SV model. | [`utils.config`](../../utils/config.py) | [affine_expansion.py](../../../../examples/docs/affine_expansion.py) | — |
| [`coefficients.py`](coefficients.py) | LogSV generators in normalized log-forward and volatility coordinates. | [`models.logsv`](__init__.py), [`models.logsv.params`](params.py), [`pde_solvers`](../../pde_solvers/__init__.py) | [test_pde_logsv.py](../../tests/test_pde_logsv.py) | — |
| [`grids.py`](grids.py) | Build parameter-dependent volatility, log-return and quadratic-variation grids. | [`utils.config`](../../utils/config.py)<br>Typing [`models.logsv.params`](params.py) | [run_logsv_pde_pricer.py](../../../../examples/pricers/run_logsv_pde_pricer.py) | — |
| [`kernel.py`](kernel.py) | Select exponential nodes and weights for a Markovian approximation of the rough kernel. | [`models.rough_logsv.kernel_fit`](../rough_logsv/kernel_fit.py), [`models.rough_logsv.rough_kernel`](../rough_logsv/rough_kernel.py)<br>Typing [`models.logsv.params`](params.py) | [test_rough_logsv_kernel_fit.py](../../tests/test_rough_logsv_kernel_fit.py) | — |
| [`model.py`](model.py) | Path adapter for the maintained continuous equity LogSV simulation. | [`data.model_paths`](../../data/model_paths.py)<br>Lazy [`models.logsv.params`](params.py), [`models.logsv.simulation`](simulation.py) | [test_logsv_paths.py](../../tests/test_logsv_paths.py) | — |
| [`moments.py`](moments.py) | Moments of the volatility process and the expected quadratic variance. | [`models.logsv.params`](params.py) | [run_vol_moments_ode.py](../../../../examples/models/run_vol_moments_ode.py) | — |
| [`params.py`](params.py) | Parameters of the log-normal beta SV model with quadratic drift. | [`models.base`](../base.py), [`models.rough_logsv.kernel_fit`](../rough_logsv/kernel_fit.py), [`utils.config`](../../utils/config.py), [`utils.funcs`](../../utils/funcs.py)<br>Lazy [`models.logsv.grids`](grids.py), [`models.logsv.kernel`](kernel.py), [`models.logsv.moments`](moments.py) | [run_logsv_pde_pricer.py](../../../../examples/pricers/run_logsv_pde_pricer.py) | — |
| [`simulation.py`](simulation.py) | LogSV terminal-state propagation and reproducible chain random tapes. | [`utils.funcs`](../../utils/funcs.py) | [run_rough_logsv_perf.py](../../../../examples/models/run_rough_logsv_perf.py) | — |
