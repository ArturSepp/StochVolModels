"""Public definitions for the tgarch model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = (
    "TgarchLimitParams",
    "TgarchMeasure",
    "TgarchModel",
    "TgarchParams",
    "derive_tgarch_limit_params",
)

_EXPORTS = {
    # Retain first-party names available from existing module imports.
    "ModelPaths": "stochvolmodels.data.model_paths",
    "TgarchMeasure": ".params",
    "_finite_float": ".params",
    "_positive_float": ".params",
    "_positive_int": ".params",
    "_validated_seed": ".params",
    "_as_measure": ".params",
    "TgarchParams": ".params",
    "TgarchLimitParams": ".params",
    "derive_tgarch_limit_params": ".params",
    "_step_grid": ".simulation",
    "_resolve_limit_params": ".simulation",
    "_exact_q_law": ".simulation",
    "_antithetic_normal_block": ".simulation",
    "_effective_sample_size": ".simulation",
    "_TerminalState": ".simulation",
    "_simulate_terminal": ".simulation",
    "_readonly": ".model",
    "TgarchModel": ".model",
    "M1": ".params",
    "S1": ".params",
    "SIGMA_FLOOR": ".params",
    "FloatArray": ".params",
    "_DEFAULT_CHUNK_STEPS": ".simulation",
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
