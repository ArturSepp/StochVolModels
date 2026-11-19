"""Public definitions for the lsv model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = ("LeverageExtrapolation", "LeverageSurface", "FixedLogForwardJump", "LSVModel")

_EXPORTS = {
    # Retain first-party names available from existing module imports.
    "Grid1D": "stochvolmodels.pde_solvers",
    "LogSvModel": "stochvolmodels.models.logsv",
    "LogSvParams": "stochvolmodels.models.logsv.params",
    "PDE2DCoefficients": "stochvolmodels.pde_solvers",
    "TensorGrid2D": "stochvolmodels.pde_solvers",
    "LeverageExtrapolation": ".model",
    "LeverageSurface": ".model",
    "FixedLogForwardJump": ".model",
    "LSVModel": ".model",
}


def __getattr__(name: str) -> Any:
    """Resolve a public definition or an existing compatibility import lazily."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy definitions in interactive discovery."""
    return sorted(set(globals()) | set(_EXPORTS))
