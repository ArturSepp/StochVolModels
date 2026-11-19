"""Public definitions for the logsv model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = ("LogSvParams", "LogSvModel", "LogSvMeasure")

_EXPORTS = {
    # Retain first-party names available from existing module imports.
    "ModelPaths": "stochvolmodels.data.model_paths",
    "_finite_float": ".model",
    "_positive_float": ".model",
    "_nonnegative_float": ".model",
    "_positive_int": ".model",
    "_validated_seed": ".model",
    "_as_measure": ".model",
    "_observation_grid": ".model",
    "_readonly": ".model",

    "LogSvParams": ".params",
    "LogSvModel": ".model",
    "LogSvMeasure": ".model",
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
