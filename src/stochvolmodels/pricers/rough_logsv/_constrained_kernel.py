"""Compatibility imports for ``stochvolmodels.models.rough_logsv._constrained_kernel``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv._constrained_kernel import (
    perf_counter as perf_counter,
    warnings as warnings,
    np as np,
    least_squares as least_squares,
    expit as expit,
    softmax as softmax,
    DEFAULT_FIT_PARAMS as DEFAULT_FIT_PARAMS,
    MATURITY_FRACTIONS as MATURITY_FRACTIONS,
    MomentKey as MomentKey,
    fractional_moment_targets as fractional_moment_targets,
    markov_moment_targets as markov_moment_targets,
    _PARAMETER_BOUND as _PARAMETER_BOUND,
    _CAP_RESERVE as _CAP_RESERVE,
    _decode_kernel as _decode_kernel,
    _encode_kernel as _encode_kernel,
    _validate_achieved_kernel as _validate_achieved_kernel,
    _spectral_seed as _spectral_seed,
    fit_positive_kernel as fit_positive_kernel,
)
