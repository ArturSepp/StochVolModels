"""Compatibility imports for ``stochvolmodels.models.rough_logsv.history``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv.history import (
    np as np,
    log_ndtr as log_ndtr,
    _history as _history,
    _history_fast as _history_fast,
    RoughKernel as RoughKernel,
    RoughLogSvParams as RoughLogSvParams,
    prepare_history as prepare_history,
    simulate_history as simulate_history,
    conditional_payoffs as conditional_payoffs,
    raw_forward as raw_forward,
    exact_linear_discrete_moments as exact_linear_discrete_moments,
)
