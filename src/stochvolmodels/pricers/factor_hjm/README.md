# stochvolmodels.pricers.factor_hjm

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Experimental factor HJM rate curves, affine expansions and Monte Carlo. The swaption and futures-option chain adapters quote annual normal implied volatility. Consult the rates guide for display in basis points.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Experimental factor HJM rate curves, affine expansions and Monte Carlo. | — | [factor_hjm_stochastic_volatility.py](../../../../examples/docs/factor_hjm_stochastic_volatility.py) | — |
| [`double_exp_pricer.py`](double_exp_pricer.py) | Double-exponential series pricer for swaptions and options on rate futures. | — | [factor_hjm_stochastic_volatility.py](../../../../examples/docs/factor_hjm_stochastic_volatility.py) | — |
| [`factor_hjm_pricer.py`](factor_hjm_pricer.py) | Monte Carlo driver for the factor HJM model with a stochastic volatility driver. | [`pricers.factor_hjm.rate_logsv_params`](rate_logsv_params.py), [`pricers.factor_hjm.rate_logsv_pricer`](rate_logsv_pricer.py), [`utils.rate_core`](../../utils/rate_core.py) | [factor_hjm_stochastic_volatility.py](../../../../examples/docs/factor_hjm_stochastic_volatility.py) | — |
| [`rate_affine_expansion.py`](rate_affine_expansion.py) | Affine expansion of the MGF for the factor HJM model with a log-normal SV driver. | [`models.logsv.affine_expansion`](../../models/logsv/affine_expansion.py) | [test_factor_hjm_characterization.py](../../tests/test_factor_hjm_characterization.py) | — |
| [`rate_evaluate.py`](rate_evaluate.py) | Curve evaluation helpers: discount factors, bonds, annuities and par rates. | [`utils.rate_core`](../../utils/rate_core.py) | [factor_hjm_stochastic_volatility.py](../../../../examples/docs/factor_hjm_stochastic_volatility.py) | — |
| [`rate_factor_basis.py`](rate_factor_basis.py) | Yield curve basis functions for the factor HJM framework. | [`utils.rate_core`](../../utils/rate_core.py) | [factor_hjm_stochastic_volatility.py](../../../../examples/docs/factor_hjm_stochastic_volatility.py) | — |
| [`rate_logsv_ivols.py`](rate_logsv_ivols.py) | Implied volatility inversion and fitting for rates options. | — | [test_factor_hjm_characterization.py](../../tests/test_factor_hjm_characterization.py) | — |
| [`rate_logsv_params.py`](rate_logsv_params.py) | Parameters of the factor HJM model with a log-normal stochastic volatility driver. | [`pricers.factor_hjm.rate_factor_basis`](rate_factor_basis.py), [`pricers.model_pricer`](../model_pricer.py), [`utils.rate_core`](../../utils/rate_core.py) | [factor_hjm_stochastic_volatility.py](../../../../examples/docs/factor_hjm_stochastic_volatility.py) | — |
| [`rate_logsv_pricer.py`](rate_logsv_pricer.py) | Pricer for the factor HJM model with a log-normal stochastic volatility driver. | [`data.option_chain`](../../data/option_chain.py), [`models.logsv.affine_expansion`](../../models/logsv/affine_expansion.py), [`pricers.factor_hjm.double_exp_pricer`](double_exp_pricer.py), [`pricers.factor_hjm.rate_affine_expansion`](rate_affine_expansion.py), [`pricers.factor_hjm.rate_evaluate`](rate_evaluate.py), [`pricers.factor_hjm.rate_factor_basis`](rate_factor_basis.py), [`pricers.factor_hjm.rate_logsv_params`](rate_logsv_params.py), [`pricers.model_pricer`](../model_pricer.py), [`utils.funcs`](../../utils/funcs.py), [`utils.rate_core`](../../utils/rate_core.py) | [run_rate_logsv_pricer.py](../../../../examples/pricers/run_rate_logsv_pricer.py) | — |
