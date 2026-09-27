---
myst:
  html_meta:
    description: >-
      stochvolmodels API reference: the stable and advanced public names grouped by the page that
      explains them, the provisional and experimental surfaces, and parameter maps for LogSvParams,
      HestonParams and LogSV calibration.
---

# API reference

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Reference for [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Names listed by `stochvolmodels.__all__` are the stable high-level surface. Advanced names are
reachable at the package root outside `__all__` and are documented here for research use. Other
historical root names remain lazy and import-compatible but are not the recommended starting point.
The tiers are defined on the
[conventions page](option_chains_and_conventions.md#stability-tiers).

Each section below is named after the page that explains its names. A page that is not written
yet is named in italics. The API contract test resolves every stable name and fails if `__all__` or
a stable docstring drifts; `scripts/check_docs.py` checks that each name sits under its owning page.

## Installation and first result

Explained in [Installation and first result](getting_started.md). The installed release string is
`stochvolmodels.__version__`.

## Option chains, notation and conventions

Explained in [Option chains, notation and conventions](option_chains_and_conventions.md).

```{eval-rst}
.. autoclass:: stochvolmodels.OptionSlice
   :members:

.. autoclass:: stochvolmodels.OptionChain
   :members: get_uniform_chain, get_slice, get_mid_vols, to_forward_normalised_strikes, compute_model_ivols_from_chain_data

.. autoclass:: stochvolmodels.OptionType
   :members:

.. autoclass:: stochvolmodels.VariableType
   :members:
```

## Log-normal stochastic volatility model with quadratic drift

Explained in [Log-normal stochastic volatility model with quadratic drift](logsv_model.md).

```{eval-rst}
.. autoclass:: stochvolmodels.LogSvParams
   :members:

.. autoclass:: stochvolmodels.LogSVPricer
   :members: price_vanilla, price_slice, price_chain, compute_chain_prices_with_vols, model_mc_price_chain, calibrate_model_params_to_chain
```

## Steady-state distribution, moments and expected quadratic variance

Explained in
[Steady-state distribution, moments and expected quadratic variance](volatility_distribution_and_moments.md).

```{eval-rst}
.. autofunction:: stochvolmodels.compute_analytic_qvar
```

## Martingale conditions, measures and positive skews

Explained in [Martingale conditions, measures and positive skews](martingale_conditions_and_skews.md).

```{eval-rst}
.. autoclass:: stochvolmodels.ConstraintsType
   :members:
```

## The moment generating function and its affine expansion

Explained in
[The moment generating function and its affine expansion](affine_expansion.md). These are
advanced names.

```{eval-rst}
.. autoclass:: stochvolmodels.ExpansionOrder
   :members:

.. autofunction:: stochvolmodels.compute_logsv_a_mgf_grid
.. autofunction:: stochvolmodels.func_a_ode_quadratic_terms
.. autofunction:: stochvolmodels.func_rhs
.. autofunction:: stochvolmodels.func_rhs_jac
.. autofunction:: stochvolmodels.get_expansion_n
.. autofunction:: stochvolmodels.get_init_conditions_a
.. autofunction:: stochvolmodels.solve_a_ode_grid
.. autofunction:: stochvolmodels.solve_analytic_ode_for_a
.. autofunction:: stochvolmodels.solve_analytic_ode_for_a0
.. autofunction:: stochvolmodels.solve_analytic_ode_grid_phi
.. autofunction:: stochvolmodels.solve_ode_for_a
```

## Fourier pricing of European options

Explained in [Fourier pricing of European options](european_option_pricing.md). The model
pricers' `price_vanilla`, `price_slice` and `price_chain` are documented with each pricer; inverse
options are explained in [Inverse options and the inverse measure](inverse_options.md), and options
on quadratic variance in
[Expected quadratic variance, variance swaps and QV options](quadratic_variance_options.md). The
functions below are the same callable
objects exported by `vanilla_option_pricers`; stochvolmodels does not maintain a duplicate
implementation.

### Black-Scholes-Merton analytics

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

### Absolute-normal Bachelier analytics

```{eval-rst}
.. autofunction:: stochvolmodels.compute_normal_price
.. autofunction:: stochvolmodels.compute_normal_slice_prices
.. autofunction:: stochvolmodels.compute_normal_delta
.. autofunction:: stochvolmodels.compute_normal_delta_to_strike
.. autofunction:: stochvolmodels.compute_normal_slice_vegas
.. autofunction:: stochvolmodels.infer_normal_implied_vol
.. autofunction:: stochvolmodels.infer_normal_ivols_from_slice_prices
```

## Monte Carlo simulation schemes

Explained in [Monte Carlo simulation schemes](monte_carlo_simulation.md). These are advanced
names: they
draw the random numbers once from a seeded local generator and reuse them across valuations.

```{eval-rst}
.. autofunction:: stochvolmodels.get_randoms_for_chain_valuation
.. autofunction:: stochvolmodels.get_randoms_for_rough_vol_chain_valuation
.. autofunction:: stochvolmodels.logsv_mc_chain_pricer_fixed_randoms
.. autofunction:: stochvolmodels.rough_logsv_mc_chain_pricer_fixed_randoms
```

## Calibration to implied volatilities

Explained in [Calibration to implied volatilities](calibration.md). `ConstraintsType` is
explained with the martingale conditions above.

```{eval-rst}
.. autoclass:: stochvolmodels.LogsvModelCalibrationType
   :members:

.. autoclass:: stochvolmodels.CalibrationEngine
   :members:

.. autoexception:: stochvolmodels.CalibrationError
```

## The Heston model as a benchmark

Explained in [The Heston model as a benchmark](heston_model.md).

```{eval-rst}
.. autoclass:: stochvolmodels.HestonParams
   :members:

.. autoclass:: stochvolmodels.HestonPricer
   :members: price_vanilla, price_slice, price_chain, compute_chain_prices_with_vols, model_mc_price_chain, calibrate_model_params_to_chain
```

## Jump-diffusion with clustered jumps

Explained in [Jump-diffusion with clustered jumps](hawkes_jump_diffusion.md). These are advanced names.

```{eval-rst}
.. autoclass:: stochvolmodels.HawkesJDParams
   :members:

.. autoclass:: stochvolmodels.HawkesJDPricer
   :members: price_chain, model_mc_price_chain, calibrate_model_params_to_chain
```

## Gaussian-mixture and Student-t smiles

Explained in [Gaussian-mixture and Student-t smiles](terminal_distribution_models.md).

```{eval-rst}
.. autoclass:: stochvolmodels.GmmParams
   :members:

.. autoclass:: stochvolmodels.GmmPricer
   :members:

.. autoclass:: stochvolmodels.TdistParams
   :members:

.. autoclass:: stochvolmodels.TdistPricer
   :members:
```

## Stochastic volatility for factor HJM rates

Explained in [Stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md), with
the case study [USD swaptions and SOFR futures options](app_swaptions_and_sofr_options.md). These
names are reached by deep imports from the experimental `stochvolmodels.pricers.factor_hjm` package.

```{eval-rst}
.. autoclass:: stochvolmodels.pricers.factor_hjm.rate_factor_basis.NelsonSiegel
   :members: swap_rate, annuity, bond

.. autoclass:: stochvolmodels.pricers.factor_hjm.rate_logsv_params.MultiFactRateLogSvParams
   :members: transform_QA_params, check_QA_kappa2, check_QT_kappa2, update_params, reduce

.. autofunction:: stochvolmodels.pricers.factor_hjm.rate_logsv_pricer.logsv_chain_de_pricer

.. autofunction:: stochvolmodels.pricers.factor_hjm.factor_hjm_pricer.calc_mc_vols
```

## Provisional and experimental surfaces

The [software design](software_design.md#models-without-an-article) page says which of these
surfaces have no article yet, and why.

The following direct-module APIs ship in the wheel but are intentionally absent from the stable
package-root `__all__`. They support the repository's volatility-book analytics and may be refined
after validation across additional model implementations:

- `stochvolmodels.data.model_paths.ModelPaths` is the validated path payload.

- `stochvolmodels.models.PathModel` and `TransformModel` describe independent dynamic
  capabilities. `stochvolmodels.models.logsv.LogSvModel` and
  `stochvolmodels.models.tgarch.TgarchModel` are the first path implementations.

- `stochvolmodels.models.TerminalDistributionModel` and `TerminalSmileModel` describe
  one-maturity laws and smiles separately from path models. Implementations are
  `stochvolmodels.pricers.tdist_pricer.TdistTerminalModel`,
  `stochvolmodels.pricers.gmm_pricer.GmmTerminalModel`, and
  `stochvolmodels.models.inverse_gamma_normal.InverseGammaNormalTerminalModel`.

- `stochvolmodels.products.payoffs` provides `EuropeanOptionPayoff` and
  `IntegratedVarianceOptionPayoff`. `stochvolmodels.valuation` provides raw and self-normalized
  path valuation with explicit measure, likelihood-weight, recentering, standard-error, and ESS
  policies.

- `stochvolmodels.models.regime_logsv`, `stochvolmodels.models.regime_logsv_simulation`, and
  `stochvolmodels.pricers.regime_switch_logsv_pricer` provide the provisional regime-switching
  LogSV equilibrium, transform, Fourier, and independent Monte Carlo stack.

Student-t and GMM validate the terminal boundary without being made into artificial path models.
Root aliases and protocol stabilization are deferred until the same contracts have been exercised
by another dynamic model.

The deep modules `stochvolmodels.pricers.rough_logsv` and `stochvolmodels.pricers.factor_hjm` are
experimental research surfaces that may evolve between minor releases; the second is explained in
[Stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md). The removed
`stochvolmodels.pricers.analytic.bsm` and `stochvolmodels.pricers.analytic.bachelier` paths are not
compatibility facades in 2.0.

### OHLC volatility estimation

OHLC estimators operate only on price-bar data and have no market-data-provider dependency.

```{eval-rst}
.. automodule:: stochvolmodels.estimation
   :members:
```

### Approximate LogSV smile utilities

Explained in [Approximate log-normal SV smile fitter](logsv_smile_fitter.md). The
provider-independent utilities under `stochvolmodels.fitters` support initialization,
diagnostics, and synthetic grids. They are separate from the full transform-based
`LogSVPricer.calibrate_model_params_to_chain` calibration.

```{eval-rst}
.. automodule:: stochvolmodels.fitters
   :members:
```

### Local resource and output paths

`stochvolmodels.local_path` reads the ignored package-adjacent `settings.yaml`. Its getters return
absolute, separator-terminated strings for compatibility with the wider `qis` ecosystem.

```{eval-rst}
.. automodule:: stochvolmodels.local_path
   :members: get_resource_path, get_local_resource_path, get_output_path
```

## Parameter maps

Each field of the parameter classes, and each keyword of the LogSV calibration, is explained on
one page.

### `LogSvParams` fields

| Field | Default | Meaning | Explained in |
|---|---|---|---|
| `sigma0` | `0.2` | Initial volatility | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) |
| `theta` | `0.2` | Mean volatility | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) |
| `kappa1` | `1.0` | Linear mean-reversion rate | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) |
| `kappa2` | `2.5` | Quadratic mean-reversion rate; `None` sets `kappa1 / theta` | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) |
| `beta` | `-1.0` | Volatility beta | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) |
| `volvol` | `1.0` | Volatility of residual volatility | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) |
| `vol_backbone` | `None` | Term structure of multiplicative scalings of `theta` | [Calibration to implied volatilities](calibration.md) |
| `H` | `0.5` | Hurst exponent; values below 0.5 select the experimental rough extension | [Monte Carlo simulation schemes](monte_carlo_simulation.md) |
| `weights` | `None` | Rough-kernel quadrature weights, set by `approximate_kernel` | [Monte Carlo simulation schemes](monte_carlo_simulation.md) |
| `nodes` | `None` | Rough-kernel quadrature nodes, set by `approximate_kernel` | [Monte Carlo simulation schemes](monte_carlo_simulation.md) |

### `HestonParams` fields

| Field | Default | Meaning | Explained in |
|---|---|---|---|
| `v0` | `0.04` | Initial variance | [The Heston model as a benchmark](heston_model.md) |
| `theta` | `0.04` | Long-run variance | [The Heston model as a benchmark](heston_model.md) |
| `kappa` | `4.0` | Mean-reversion rate of the variance | [The Heston model as a benchmark](heston_model.md) |
| `rho` | `-0.5` | Correlation of the variance and price shocks | [The Heston model as a benchmark](heston_model.md) |
| `volvol` | `0.4` | Volatility of the variance | [The Heston model as a benchmark](heston_model.md) |

### `LogSVPricer.calibrate_model_params_to_chain` keywords

| Keyword | Default | Meaning | Explained in |
|---|---|---|---|
| `option_chain` | required | Market chain with bid and ask implied volatilities | [Calibration to implied volatilities](calibration.md) |
| `params0` | required | Starting parameters; fixed fields stay at these values | [Calibration to implied volatilities](calibration.md) |
| `params_min` | lower box bounds | Lower bounds of the fitted parameters | [Calibration to implied volatilities](calibration.md) |
| `params_max` | upper box bounds | Upper bounds of the fitted parameters | [Calibration to implied volatilities](calibration.md) |
| `is_vega_weighted` | `True` | Weight implied-volatility residuals by vega | [Calibration to implied volatilities](calibration.md) |
| `is_unit_ttm_vega` | `False` | Compute the vegas at unit maturity | [Calibration to implied volatilities](calibration.md) |
| `model_calibration_type` | `PARAMS5` | Which parameters are free | [Calibration to implied volatilities](calibration.md) |
| `constraints_type` | `UNCONSTRAINT` | Martingale and moment constraints | [Calibration to implied volatilities](calibration.md) |
| `calibration_engine` | `ANALYTIC` | Transform pricer or Monte Carlo inside the objective | [Calibration to implied volatilities](calibration.md) |
| `nb_path` | `100000` | Monte Carlo paths, for the Monte Carlo engines | [Calibration to implied volatilities](calibration.md) |
| `nb_steps` | `360` | Monte Carlo time steps per year, for the Monte Carlo engines | [Calibration to implied volatilities](calibration.md) |
| `seed` | `10` | Seed of the fixed random numbers, for the Monte Carlo engines | [Calibration to implied volatilities](calibration.md) |
