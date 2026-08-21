# Stable API reference

Names listed by `stochvolmodels.__all__` are the stable high-level surface. Historical root exports
remain lazy and import-compatible but are not the recommended starting point. Factor HJM and rough
LogSV deep-module paths are experimental research APIs.

## Data and model classes

```{eval-rst}
.. autoclass:: stochvolmodels.OptionSlice
   :members:

.. autoclass:: stochvolmodels.OptionChain
   :members: get_uniform_chain, get_slice, compute_model_ivols_from_chain_data

.. autoclass:: stochvolmodels.LogSvParams
   :members:

.. autoclass:: stochvolmodels.LogSVPricer
   :members: price_vanilla, price_slice, price_chain, compute_chain_prices_with_vols, calibrate_model_params_to_chain

.. autoclass:: stochvolmodels.HestonParams
   :members:

.. autoclass:: stochvolmodels.HestonPricer
   :members: price_vanilla, price_slice, price_chain, compute_chain_prices_with_vols, calibrate_model_params_to_chain

.. autoclass:: stochvolmodels.GmmParams
   :members:

.. autoclass:: stochvolmodels.GmmPricer
   :members:

.. autoclass:: stochvolmodels.TdistParams
   :members:

.. autoclass:: stochvolmodels.TdistPricer
   :members:
```

## Enums and errors

```{eval-rst}
.. autoclass:: stochvolmodels.OptionType
   :members:

.. autoclass:: stochvolmodels.VariableType
   :members:

.. autoclass:: stochvolmodels.LogsvModelCalibrationType
   :members:

.. autoclass:: stochvolmodels.ConstraintsType
   :members:

.. autoclass:: stochvolmodels.CalibrationEngine
   :members:

.. autoexception:: stochvolmodels.CalibrationError
```

Analytic Black-Scholes-Merton, absolute-normal Bachelier, implied-volatility, and
quadratic-variance functions are also stable and listed in `stochvolmodels.__all__`. The Black and
Bachelier functions are the same callable objects exported by `vanilla_option_pricers`; SVM does
not maintain duplicate implementations. The removed
`stochvolmodels.pricers.analytic.bsm` and `stochvolmodels.pricers.analytic.bachelier` paths are not
compatibility facades in 2.0.

## Local resource and output paths

`stochvolmodels.local_path` reads the ignored package-adjacent `settings.yaml`. Its getters return
absolute, separator-terminated strings for compatibility with the wider `qis` ecosystem.

```{eval-rst}
.. automodule:: stochvolmodels.local_path
   :members: get_resource_path, get_local_resource_path, get_output_path
```

## Approximate LogSV smile utilities

The provider-independent utilities under `stochvolmodels.fitters` support initialization,
diagnostics, and synthetic grids. They are separate from the full transform-based
`LogSVPricer.calibrate_model_params_to_chain` calibration.

```{eval-rst}
.. automodule:: stochvolmodels.fitters
   :members:
```
