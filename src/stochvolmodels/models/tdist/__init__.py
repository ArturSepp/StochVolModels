"""Public definitions for the tdist model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = (
    "TdistParams",
    "compute_upsilon",
    "pdf_tdist",
    "cdf_tdist",
    "cum_mean_tdist",
    "imply_drift_tdist",
    "compute_default_prob_tdist",
    "compute_forward_tdist",
)

_EXPORTS = {
    "TdistParams": ".params",
    "compute_upsilon": ".distribution",
    "pdf_tdist": ".distribution",
    "cdf_tdist": ".distribution",
    "cum_mean_tdist": ".distribution",
    "imply_drift_tdist": ".distribution",
    "compute_default_prob_tdist": ".distribution",
    "compute_forward_tdist": ".distribution",
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
