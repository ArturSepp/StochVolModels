"""Compatibility imports for ``stochvolmodels.models.rough_logsv._history``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv._history import (
    dataclass as dataclass,
    np as np,
    eigh as eigh,
    log_ndtr as log_ndtr,
    roots_hermitenorm as roots_hermitenorm,
    njit as njit,
    Model as Model,
    model as model,
    prepare as prepare,
    linear_flow as linear_flow,
    simulate as simulate,
    paths as paths,
    half_payoffs as half_payoffs,
    covariance_a as covariance_a,
    linear_moments as linear_moments,
    exact_discrete as exact_discrete,
    scalar_maps as scalar_maps,
    scalar_frequency as scalar_frequency,
    scalar_transform as scalar_transform,
)
