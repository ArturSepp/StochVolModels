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

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Names listed by `stochvolmodels.__all__` are the stable high-level surface. Advanced names are
reachable at the package root outside `__all__` and are documented here for research use. Other
historical root names remain lazy and import-compatible but are not the recommended starting point.
The tiers are defined on the
[conventions page](option_chains_and_conventions.md#stability-tiers).

Each section below is named after the page that explains its names. A page that is not written
yet is named in italics. The API contract test resolves every stable name and fails if `__all__` or
a stable docstring drifts; `scripts/check_docs.py` checks that each name sits under its owning page.

Canonical parameter and numerical modules live under `stochvolmodels.models`;
concrete option-chain adapters live under `stochvolmodels.pricers`. LSV projection is
`stochvolmodels.fitters.lsv_projection`. See [software design](software_design.md#layers) for the ownership map,
retained calibration methods and compatibility limits.

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
The analytic vanilla and implied-volatility methods accept
[frozen Fourier grid settings](logsv_model.md#frozen-fourier-grid-for-a-chain).

```{eval-rst}
.. autoclass:: stochvolmodels.LogSvParams
   :members:

.. autoclass:: stochvolmodels.LogSVPricer
   :members: price_vanilla, price_slice, price_chain, prepare_fourier, compute_chain_prices_with_vols, compute_model_ivols_for_chain, model_mc_price_chain, calibrate_model_params_to_chain

.. autoclass:: stochvolmodels.PreparedLogSVGrid
   :members: price_chain, sigma0_sensitivity, forward_greeks, refined_for
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
The analytic vanilla and implied-volatility methods accept the
[controlled Fourier pricing settings](heston_model.md#controlled-fourier-pricing).

```{eval-rst}
.. autoclass:: stochvolmodels.HestonParams
   :members:

.. autoclass:: stochvolmodels.HestonPricer
   :members: price_vanilla, price_slice, price_chain, prepare_fourier, compute_chain_prices_with_vols, compute_model_ivols_for_chain, model_mc_price_chain, calibrate_model_params_to_chain

.. autoclass:: stochvolmodels.PreparedHestonGrid
   :members: price_chain, v0_sensitivity, forward_greeks, refined_for
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

## Finite-difference PDE solvers

The provisional types and functions below are explained in [Finite-difference PDE solvers](pde_solvers.md).

```{eval-rst}
.. autoclass:: stochvolmodels.pde_solvers.ADIConfig
   :members:

.. autoclass:: stochvolmodels.pde_solvers.ADIScheme
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundaryCondition1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundaryKind
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundaryMass1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundaryMass2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundaryPair1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundarySet2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundarySide
   :members:

.. autoclass:: stochvolmodels.pde_solvers.BoundaryTarget
   :members:

.. autoclass:: stochvolmodels.pde_solvers.CoefficientEvaluation
   :members:

.. autoclass:: stochvolmodels.pde_solvers.CoefficientProvider1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.CoefficientProvider2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.DensityPolicy
   :members:

.. autoclass:: stochvolmodels.pde_solvers.DriftDiscretization
   :members:

.. autoclass:: stochvolmodels.pde_solvers.Grid1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.JumpCoordinate
   :members:

.. autoclass:: stochvolmodels.pde_solvers.JumpOperator1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.JumpOperator2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.JumpStepDiagnostics
   :members:

.. autoclass:: stochvolmodels.pde_solvers.JumpStepKind
   :members:

.. autoclass:: stochvolmodels.pde_solvers.JumpStepResult
   :members:

.. autoclass:: stochvolmodels.pde_solvers.TranslationJump
   :members:

.. autofunction:: stochvolmodels.pde_solvers.log_forward_jump_compensator

.. autofunction:: stochvolmodels.pde_solvers.strang_split_step

.. autoclass:: stochvolmodels.pde_solvers.PDE1DCoefficients
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDE1DSolver
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDE2DADISolver
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDE2DCoefficients
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDEDiagnostics
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDEDiagnostics2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDEDirection
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDESolution1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDESolution2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDEState1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.PDEState2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.StaticCoefficientProvider1D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.StaticCoefficientProvider2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.TensorGrid2D
   :members:

.. autoclass:: stochvolmodels.pde_solvers.TimeScheme
   :members:

.. autoclass:: stochvolmodels.pde_solvers.TimeStepConfig
   :members:

```

### Compound-Poisson exponential jumps

Import these provisional names from `stochvolmodels.pde_solvers`. Prepare one kernel
per active coordinate and pass the process as `jump=` to `PDE1DSolver.solve` or
`PDE2DADISolver.solve`. [Jump operator and boundary conventions](pde_solvers.md#exponential-jump-marks) ·
[Two-name survival example](pde_solvers.md#two-name-survival-with-continuous-barriers)

```{eval-rst}
.. autoclass:: stochvolmodels.pde_solvers.ExponentialJumpKernel
   :members:

.. autofunction:: stochvolmodels.pde_solvers.apply_common_exponential_jump

.. autoclass:: stochvolmodels.pde_solvers.ExponentialJumpProcess
   :members: poisson_step

.. autoclass:: stochvolmodels.pde_solvers.ExponentialJumpStep
```

For a standalone `poisson_step`, read `values` for the propagated array and
`poisson_terms` for the number of retained count terms, including zero arrivals.
`truncated_probability` is the omitted Poisson count probability. In a forward step,
`killed_mass` is the integrated density lost through a zero exterior. A backward
step returns `None` for this field.

## Local volatility and local stochastic volatility

These provisional models, configurations, pricers, fitters and result records are explained in [Local volatility and local stochastic volatility](local_volatility_and_lsv.md).

### `models.lsv.local_vol`

```{eval-rst}
.. automodule:: stochvolmodels.models.lsv.local_vol
   :members: CoordinateSystem, LocalVolFamily, ParametricLocalVolSlice, ParametricLocalVolFormula, LocalVolSurface, LocalVolCoefficientProvider1D
   :member-order: bysource
```

### `models.logsv.coefficients`

```{eval-rst}
.. automodule:: stochvolmodels.models.logsv.coefficients
   :members: LogSvPdeCoefficients, LogSvCoefficientProvider
   :member-order: bysource
```

### `models.lsv`

```{eval-rst}
.. automodule:: stochvolmodels.models.lsv
   :members: LeverageExtrapolation, LeverageSurface, FixedLogForwardJump, LSVModel
   :member-order: bysource
```

### `models.lsv.simulation`

```{eval-rst}
.. automodule:: stochvolmodels.models.lsv.simulation
   :members: LSVMonteCarloConfig, LSVMonteCarloEngine
   :member-order: bysource
```

### Density states and valuation

```{eval-rst}
.. automodule:: stochvolmodels.data.density
   :members: DensityState1D, DensityState2D
   :member-order: bysource

.. autoclass:: stochvolmodels.valuation.DensityPricingResult
   :members:

.. autoclass:: stochvolmodels.valuation.DensityPricer
   :members:
```

Coordinate metadata uses `stochvolmodels.utils.config.CoordinateSystem`.
The former `pricers.density_pricer` names remain aliases to these objects.

### `pricers.local_vol_pde_pricer`

```{eval-rst}
.. automodule:: stochvolmodels.pricers.local_vol_pde_pricer
   :members: LocalVolPdeExpiry, LocalVolPdeResult, LocalVolPdePricer
   :member-order: bysource
```

### `pricers.logsv_pde_pricer`

```{eval-rst}
.. automodule:: stochvolmodels.pricers.logsv_pde_pricer
   :members: LogSvPdeExpiry, LogSvPdeResult, LogSvPdePricer
   :member-order: bysource
```

### `fitters.lsv_projection`

```{eval-rst}
.. automodule:: stochvolmodels.fitters.lsv_projection
   :members: LowDensityPolicy, LSVProjectionConfig, ConditionalMomentResult, project_conditional_variance, LeverageUpdateResult, recover_leverage_slice, LSVProjectionSlice, LSVRepricingSlice, SequentialLSVProjectionResult, SequentialLSVProjector
   :member-order: bysource
```

### `pricers.lsv_mc_pricer`

```{eval-rst}
.. automodule:: stochvolmodels.pricers.lsv_mc_pricer
   :members: LSVMonteCarloExpiry, LSVMonteCarloResult, LSVMonteCarloPricer
   :member-order: bysource
```

### `fitters.dupire`

```{eval-rst}
.. automodule:: stochvolmodels.fitters.dupire
   :members: CallSurface, DupireRepairPolicy, DupireJumpAdjustment, DupireResult, DupireRecoverer
   :member-order: bysource
```

### `fitters.lsv_fixed_point`

```{eval-rst}
.. automodule:: stochvolmodels.fitters.lsv_fixed_point
   :members: LSVFixedPointConfig, LSVFixedPointIteration, LSVFixedPointSlice, LSVFixedPointResult, LSVConvergenceError, LSVFixedPointCalibrator
   :member-order: bysource
```

### `fitters.lsv_parametric`

```{eval-rst}
.. automodule:: stochvolmodels.fitters.lsv_parametric
   :members: CalibrationWeighting, ImpliedVolCalibrationTarget, calibration_weights, LSVParametricCalibrationConfig, LSVRejectedTrial, LSVCalibrationError, LSVCalibrationEvaluation, LSVParametricSlice, LSVParametricCalibrationResult, LSVParametricCalibrator
   :member-order: bysource
```

### `fitters.lsv_market`

```{eval-rst}
.. automodule:: stochvolmodels.fitters.lsv_market
   :members: LSVQuoteSelection, MarketQuoteSlice, LSVCalibrationInput, LSVOptionChainAdapter
   :member-order: bysource
```

### `fitters.lsv_market_pipeline`

```{eval-rst}
.. automodule:: stochvolmodels.fitters.lsv_market_pipeline
   :members: LSVMarketAcceptanceCriteria, LSVMarketAcceptance, LSVMarketPriceSlice, LSVMarketPipelineResult, LSVMarketPipeline
   :member-order: bysource
```

## Provisional and experimental surfaces

The provisional [finite-difference solvers](software_design.md#generic-finite-difference-solvers)
are available as `stochvolmodels.pde_solvers.PDE1DSolver` and `PDE2DADISolver`.
The same solver package exposes `TranslationJump` for additive or level-multiplicative
jumps, including forward and backward translations, compound-Poisson steps and Strang splitting.
Both solvers also accept exponential marks with constant arrival intensities.
[Compound-Poisson exponential jumps](#compound-poisson-exponential-jumps)

`stochvolmodels.valuation.DensityPricer` prices calls and puts from a
`DensityState1D`. Use `DensityState2D.marginal_x_state()` for a joint density.
`stochvolmodels.models.logsv.coefficients.LogSvCoefficientProvider` supplies the LogSV generator
in either the money-market or inverse measure and accepts a volatility backbone.
The provisional `stochvolmodels.pricers.logsv_pde_pricer.LogSvPdePricer` uses the
2D solver for ordinary European LogSV chains under the money-market measure.
The detailed 1D and 2D pricer results expose `expiry.boundary_mass` for each maturity.
`Grid1D.boundary_mass(density)` and `TensorGrid2D.boundary_mass(density)` return the
same `BoundaryMass1D` and `BoundaryMass2D` diagnostics from supplied densities.
The provisional `stochvolmodels.models.lsv.local_vol.LocalVolSurface` supplies deterministic
volatility to `stochvolmodels.pricers.local_vol_pde_pricer.LocalVolPdePricer`.
The pricer uses a Rannacher start followed by Crank–Nicolson in each interval. A numeric `theta`
selects a fixed scheme, and `price_chain_detailed(...).steps` reports every substep.
See [local-volatility pricing](software_design.md#deterministic-local-volatility) for
bucket conventions, replay settings and measured accuracy. The provisional `stochvolmodels.fitters.dupire` module exposes
`CallSurface`, `DupireRecoverer`, `DupireResult` and `DupireRepairPolicy` for
[continuous Dupire inversion](software_design.md#continuous-dupire-inversion).
`DupireRecoverer.jump_adjusted` also returns a `DupireJumpAdjustment` on the result.
See [fixed-jump Dupire inversion](software_design.md#fixed-jump-dupire-inversion).
The provisional `stochvolmodels.models.lsv.LSVModel` binds LogSV dynamics to a
`LeverageSurface`. The `stochvolmodels.fitters.lsv_projection` module provides
`project_conditional_variance`, `recover_leverage_slice` and `SequentialLSVProjector`.
Both leverage fitters accept `leverage_times` and `quote_chain`. Each `LSVRepricingSlice`
in `result.repricing` contains the target and fitted premiums, Black IVs and signed errors
in the original strike order. The returned model can be priced by `LogSvPdePricer`.
See [conditional variance and leverage](software_design.md#conditional-variance-and-leverage).
The provisional `stochvolmodels.fitters.lsv_fixed_point.LSVFixedPointCalibrator` fits
the leverage and exposes the convergence history for every maturity.
See [fixed-point calibration](software_design.md#fixed-point-leverage-calibration).

The provisional `stochvolmodels.fitters.lsv_parametric.LSVParametricCalibrator` fits
a bounded leverage formula to Black implied volatilities at each maturity. The result
contains a priceable model, the parameter matrix and per-expiry fit diagnostics.
See [parametric calibration](software_design.md#parametric-leverage-calibration).

The provisional `stochvolmodels.models.lsv.simulation.LSVMonteCarloEngine` generates
`ModelPaths` from a fitted `LSVModel`. The corresponding
`stochvolmodels.pricers.lsv_mc_pricer.LSVMonteCarloPricer` prices ragged European
chains and reports sampling uncertainty. See [LSV Monte Carlo](software_design.md#lsv-monte-carlo).
The same `LSVModel` with `FixedLogForwardJump` can be priced by `LogSvPdePricer`.
See [jump-aware LSV PDE pricing](software_design.md#jump-aware-lsv-pde-pricing).
The provisional `stochvolmodels.fitters.lsv_market.LSVOptionChainAdapter` converts an
ordinary bid/ask chain into an `LSVCalibrationInput` for fitting and Dupire recovery.
See the [OptionChain market adapter](software_design.md#optionchain-market-adapter).
The provisional `stochvolmodels.fitters.lsv_market_pipeline.LSVMarketPipeline` fits
and validates a complete chain under fixed LogSV dynamics. The measured limits and
outcomes are described in [LSV market calibration](software_design.md#lsv-market-calibration).

The [software design](software_design.md#models-without-an-article) page says which of these
surfaces have no article yet, and why.

The following direct-module APIs ship in the wheel but are intentionally absent from the stable
package-root `__all__`. They support the repository's volatility-book analytics and may be refined
after validation across additional model implementations:

- `stochvolmodels.data.model_paths.ModelPaths` is the validated path payload.

- `stochvolmodels.models.PathModel` and `TransformModel` describe independent dynamic
  capabilities. `stochvolmodels.models.logsv.LogSvModel` and
  `stochvolmodels.models.tgarch.TgarchModel` provide paths;
  `stochvolmodels.models.heston.HestonModel` provides both paths and a joint log-return /
  integrated-variance transform, reusing the maintained Heston kernels.

- `stochvolmodels.models.TerminalDistributionModel` and `TerminalSmileModel` describe
  one-maturity laws and smiles separately from path models. Implementations are
  `stochvolmodels.pricers.tdist_pricer.TdistTerminalModel`,
  `stochvolmodels.pricers.gmm_pricer.GmmTerminalModel`, and
  `stochvolmodels.models.inverse_gamma.InverseGammaNormalTerminalModel`.
  For an option chain, import `InverseGammaNormalPricer` from
  `stochvolmodels.pricers.inverse_gamma_pricer` and pass `InverseGammaNormalParams`
  from `stochvolmodels.models.inverse_gamma`. The chain must have one expiry equal to
  `params.ttm`. The pricer's `compute_chain_prices_with_vols` method returns discounted European
  call and put prices in quote currency and annualized Black implied volatilities.

- `stochvolmodels.products.payoffs` provides `EuropeanOptionPayoff` and
  `IntegratedVarianceOptionPayoff`. `stochvolmodels.valuation` provides raw and self-normalized
  path valuation with explicit measure, likelihood-weight, recentering, standard-error, and ESS
  policies.

- `stochvolmodels.models.regime_logsv`, `stochvolmodels.models.regime_logsv.simulation`, and
  `stochvolmodels.pricers.regime_switch_logsv_pricer` provide the provisional regime-switching
  LogSV equilibrium, transform, Fourier, and independent Monte Carlo stack.

Student-t and GMM validate the terminal boundary without being made into artificial path models.
Root aliases and protocol stabilization are deferred until the same contracts have been exercised
across the supported measures and pricing methods of the dynamic models.

The deep modules `stochvolmodels.pricers.rough_logsv` and `stochvolmodels.pricers.factor_hjm` are
experimental research surfaces that may evolve between minor releases. The first has the
standalone interface documented below; the second is explained in
[Stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md). The removed
`stochvolmodels.pricers.analytic.bsm` and `stochvolmodels.pricers.analytic.bachelier` paths are not
compatibility facades in 2.0.

### Research rough-logSV pricing

The [rough-logSV guide](rough_logsv.md) explains the model, the working paper's
moment/stability and martingale results, explicit kernel/control inputs, and the
calibration gate. These are experimental subpackage and direct-module APIs, not
stable package-root exports. Linear continuous-time moment targets, exact discrete
moments and nonlinear option estimates are distinct computations.

#### Model and pricing configuration

```{eval-rst}
.. autoclass:: stochvolmodels.models.rough_logsv.params.RoughLogSvParams
   :members:

.. autoclass:: stochvolmodels.models.rough_logsv.params.RoughKernel
   :members:

.. autoclass:: stochvolmodels.models.rough_logsv.multilevel.RoughMCConfig
   :members:

.. autoclass:: stochvolmodels.models.rough_logsv.multilevel.DiscretePdeControl
   :members:

.. autoclass:: stochvolmodels.pricers.rough_logsv_pricer.ClassicalPdeConfig
   :members:

.. autoclass:: stochvolmodels.pricers.rough_logsv_pricer.RoughPricingResult
   :members:

.. autoclass:: stochvolmodels.pricers.rough_logsv_pricer.RoughLogSVPricer
   :members: price_chain, price_chain_result, model_mc_price_chain, calibrate_model_params_to_chain
```

#### Linear-drift moment and kernel analytics

The moment helpers apply only to linear drift. Fractional targets are numerical
product-integration estimates; finite-kernel targets solve the continuous-time moment
system by a matrix exponential. Fitting a kernel to these targets is not market
calibration. `stochvolmodels.models.rough_logsv.kernel_fit.fit_kernel_to_moments`
returns sorted nodes and matching weights; inspect achieved moment errors before
accepting a fitted kernel. See [the analytics map](rough_logsv.md#analytics-entry-points) for units
and scope, and [the paper reference](rough_logsv.md#references) for the current title.

```{eval-rst}
.. autoclass:: stochvolmodels.models.rough_logsv.kernel_fit.MomentKey
   :members:

.. autofunction:: stochvolmodels.models.rough_logsv.kernel_fit.fractional_moment_targets

.. autofunction:: stochvolmodels.models.rough_logsv.kernel_fit.markov_moment_targets

.. autofunction:: stochvolmodels.models.rough_logsv.kernel_fit.kernel_mass_errors
```

#### Full-history and conditional estimators

`exact_linear_discrete_moments` checks the chosen discrete scheme under the ordinary
measure; it is not a fractional continuous-time reference. Half-tilted option states
must not be passed to the ordinary-measure raw-forward estimator.

```{eval-rst}
.. autofunction:: stochvolmodels.models.rough_logsv.history.prepare_history

.. autofunction:: stochvolmodels.models.rough_logsv.history.simulate_history

.. autofunction:: stochvolmodels.models.rough_logsv.history.exact_linear_discrete_moments

.. autofunction:: stochvolmodels.models.rough_logsv.history.conditional_payoffs

.. autofunction:: stochvolmodels.models.rough_logsv.history.raw_forward
```

### Heston path and transform adapter

The [Heston model guide](heston_model.md) explains the provisional direct-module adapter,
its named states and transform convention. It uses the same `HestonParams` type as the
stable `HestonPricer`; path evolution, option-chain pricing and payoff valuation remain
separate capabilities.

```{eval-rst}
.. autoclass:: stochvolmodels.models.heston.HestonModel
   :members: params, log_mgf_grid, simulate_paths
```

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
   :members: calc_logsv_atm_fit, fit_logsv_ivols, calc_logsv_ivols, calc_logsv_ivols_partials, calc_logsv_pdf, calc_logsv_pdf_core, infer_strikes_from_deltas, get_vols_delta_space, get_pdf_delta_space, generate_grid_option_prices_from_slice
```

### Fit and surface diagnostics

Explained in [Fit and surface diagnostics](surface_diagnostics.md). Core diagnostics are
available from `stochvolmodels.fitters`; the OCA provenance adapter is optional.

```{eval-rst}
.. automodule:: stochvolmodels.fitters.diagnostics
   :members: DiagnosticReport, diagnose_fit

.. automodule:: stochvolmodels.fitters.surface_checks
   :members: SurfaceEvaluator, check_call_slice, check_calendar, diagnose_surface, compare_surface_reports

.. automodule:: stochvolmodels.fitters.adapters.oca_diagnostics
   :members: diagnostic_chain_from_oca

.. automodule:: stochvolmodels.fitters.adapters.native_quotes
   :members: diagnose_native_quotes, normalized_vanilla_price_bounds
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
| `vol_backbone` | `None` | Return-volatility loadings on right-ended calendar intervals | [Calibration to implied volatilities](calibration.md) |
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
| `vol_scaler` | `None` | Fixed analytic cutoff scaler, inferred from the first market ATM slice by default | [Calibration to implied volatilities](calibration.md) |
| `max_phi` | `None` | Number of analytic frequency nodes, fixed throughout the fit | [Calibration to implied volatilities](calibration.md) |
| `historical_backbone_order` | `False` | Use the forward-variance paper’s increasing-expiry transform order | [Calibration to implied volatilities](calibration.md) |
| `fourier` | `None` | A settings mapping selects a fixed optimizer grid and independent fitted-premium validation | [Calibration to implied volatilities](calibration.md) |
| `use_analytic_gradient` | `None` | Automatically supply the weighted IV-loss gradient for controlled analytic `PARAMS4` and `PARAMS5` fits. Pass `False` for finite differences. | [Calibration to implied volatilities](calibration.md) |

## Barrier reverse convertibles

The direct-module APIs for five-strike calibration, correlated basket simulation,
par coupons, fixed-coupon marks and shared-path batch valuation are documented in
the [single-payment OTC guide](software_design.md#single-payment-otc-valuation).

`BrcQuote.status` reports the underlying valuation status. Pass `model=None` to
`value_brc` for a fixed or paid contract with explicit historical evidence.

## Accumulators and decumulators

These provisional public interfaces use direct-module imports. Construct
`AccumulatorTerms` and `AccumulatorState` from `stochvolmodels.products.accumulator`.
Import `quote_accumulator_strike`, `value_accumulator` and `AccumulatorPdeConfig`
from `stochvolmodels.pricers.accumulator`. The quote returns the zero-premium strike as
a fraction of original spot. The mark reports future-fixing PV, fixed unpaid PV and
their sum in settlement currency. Both entry points take the same dated marginal model.

Supply the original ACT/365 fixing and distinct payment times, and one zero-based payment
index per fixing. State is after all fixings and payments through its elapsed time from
issue. Times within $10^{-12}$ years of that cutoff are treated as observed. Start with
`AccumulatorState.at_issue`. Its `advance` method takes newly observed native closes
and the locked strike and returns the new state and the net cash paid during the interval.
At a midlife mark use current native forwards starting at today's spot and payment discounts
ordered by `state.outstanding_payments(terms)`. The original notional, spot, strike and
number of fixings continue to govern settlement.

Check `status` before using a result. A live certified valuation returns `ok`. Earned
cash on a stopped trade can be valued with `model=None` and returns `fixed`, or
`settled` when the remaining balance is zero. Incomplete evidence returns
`unresolved_history`. Failure to meet numerical budgets returns `uncertified` and a NaN
stochastic price. A zero-exposure par quote has no unique strike. The model supplies
`marginal_pde_coefficient_provider(asset_id)` and reproducible antithetic blocks via
`iter_path_blocks` for independent path comparisons.
[Worked synthetic example](../examples/accumulator_local_vol.py) ·
[PDE and payoff conventions](software_design.md#scheduled-accumulators-and-decumulators)
