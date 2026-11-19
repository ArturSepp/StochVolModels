"""Public definitions for the heston model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = ("HestonParams", "HestonModel")

_EXPORTS = {
    # Retain first-party names available from existing module imports.
    "ModelParams": "stochvolmodels.models.base",
    "ModelPaths": "stochvolmodels.data.model_paths",
    "HestonParams": ".params",
    "_real": ".model",
    "_integer": ".model",
    "_readonly": ".model",
    "HestonModel": ".model",
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
