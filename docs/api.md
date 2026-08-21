# Stable API reference

Names listed by `stochvolmodels.__all__` are the stable high-level surface. Historical root exports
remain lazy and import-compatible but are not the recommended starting point. Factor HJM and rough
LogSV deep-module paths are experimental research APIs.

The installed release string is `stochvolmodels.__version__`. The API contract test resolves every
name below and fails if `__all__`, the documented maturity boundary, or a stable docstring drifts.

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

## Black-Scholes-Merton analytics

These are the same callable objects exported by `vanilla_option_pricers`; SVM does not maintain a
duplicate implementation.

```{eval-rst}
.. autofunction:: stochvolmodels.compute_bsm_vanilla_price
.. autofunction:: stochvolmodels.compute_bsm_vanilla_slice_prices
.. autofunction:: stochvolmodels.compute_bsm_vanilla_delta
.. autofunction:: stochvolmodels.compute_bsm_vanilla_vega
.. autofunction:: stochvolmodels.compute_bsm_vanilla_gamma
.. autofunction:: stochvolmodels.compute_bsm_vanilla_theta
.. autofunction:: stochvolmodels.compute_bsm_strike_from_delta
.. autofunction:: stochvolmodels.infer_bsm_implied_vol
.. autofunction:: stochvolmodels.infer_bsm_ivols_from_slice_prices
```

## Absolute-normal Bachelier analytics

```{eval-rst}
.. autofunction:: stochvolmodels.compute_normal_price
.. autofunction:: stochvolmodels.compute_normal_slice_prices
.. autofunction:: stochvolmodels.compute_normal_delta
.. autofunction:: stochvolmodels.compute_normal_delta_to_strike
.. autofunction:: stochvolmodels.compute_normal_slice_vegas
.. autofunction:: stochvolmodels.infer_normal_implied_vol
.. autofunction:: stochvolmodels.infer_normal_ivols_from_slice_prices
```

## Quadratic-variance analytics

```{eval-rst}
.. autofunction:: stochvolmodels.compute_analytic_qvar
```

The removed
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
