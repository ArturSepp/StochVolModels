"""Experimental rough-logSV parameters, kernels and numerical controls.

Imports are lazy so historical kernel utilities keep their original dependency
surface. No paper, agent or private-data module is imported by this package.
"""

from importlib import import_module

__all__ = (
    "RoughLogSvParams",
    "RoughKernel",
    "RoughMCConfig",
    "DiscretePdeControl",
)

_MODULES = {
    "RoughLogSvParams": "stochvolmodels.models.rough_logsv.params",
    "RoughKernel": "stochvolmodels.models.rough_logsv.params",
    "RoughMCConfig": "stochvolmodels.models.rough_logsv.multilevel",
    "DiscretePdeControl": "stochvolmodels.models.rough_logsv.multilevel",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name]), name)
    globals()[name] = value
    return value
