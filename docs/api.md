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

Analytic Black-Scholes-Merton, Bachelier, implied-volatility, and quadratic-variance functions are
also stable and listed in `stochvolmodels.__all__`; inspect `dir(stochvolmodels)` or the package
initializer for the exact release-specific manifest.
