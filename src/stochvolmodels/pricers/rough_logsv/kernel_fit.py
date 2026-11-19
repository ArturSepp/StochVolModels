"""Compatibility imports for ``stochvolmodels.models.rough_logsv.kernel_fit``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv.kernel_fit import (
    Enum as Enum,
    lru_cache as lru_cache,
    Dict as Dict,
    List as List,
    Optional as Optional,
    Tuple as Tuple,
    warnings as warnings,
    np as np,
    expm as expm,
    differential_evolution as differential_evolution,
    minimize as minimize,
    gamma as gamma,
    european_rule as european_rule,
    MATURITY_FRACTIONS as MATURITY_FRACTIONS,
    DEFAULT_FIT_PARAMS as DEFAULT_FIT_PARAMS,
    STEP_RULE_CONSTANT as STEP_RULE_CONSTANT,
    ParamSet as ParamSet,
    MomentKey as MomentKey,
    KernelRule as KernelRule,
    _validate_maturities as _validate_maturities,
    _validate_model_params as _validate_model_params,
    _validate_kernel as _validate_kernel,
    _validate_positive as _validate_positive,
    _validate_fit_domain as _validate_fit_domain,
    _fractional_moments_on_grid as _fractional_moments_on_grid,
    fractional_moment_targets as fractional_moment_targets,
    markov_moment_targets as markov_moment_targets,
    fit_kernel_to_moments as fit_kernel_to_moments,
    _cached_fit as _cached_fit,
    moment_fitted_kernel as moment_fitted_kernel,
    kernel_mass_errors as kernel_mass_errors,
    max_time_step as max_time_step,
    max_weight_sum as max_weight_sum,
)
