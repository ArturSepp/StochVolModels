"""Public definitions for the hawkes jd model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = ("HawkesJDParams", "compute_hawkes_a_mgf_grid", "solve_a_ode_grid", "solve_ode_for_a")

_EXPORTS = {
    "HawkesJDParams": ".params",
    "compute_hawkes_a_mgf_grid": ".transforms",
    "solve_a_ode_grid": ".transforms",
    "solve_ode_for_a": ".transforms",
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
