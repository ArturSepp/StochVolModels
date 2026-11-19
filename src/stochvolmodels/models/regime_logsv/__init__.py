"""Public definitions for the regime logsv model family.

Implementations are imported on demand so parameters do not load simulation or
pricing workflows. Existing family imports and serialized model names remain valid.
"""

from importlib import import_module
from typing import Any

__all__ = (
    "Regime",
    "EquilibriumClosure",
    "RegimeRiskPremiaScales",
    "RegimeLogSvDynamics",
    "RegimeTransition",
    "CrraRiskPremia",
    "RegimeSwitchLogSvParams",
    "EquilibriumSolution",
    "solve_regime_switch_equilibrium",
    "RiskNeutralState",
    "evaluate_risk_neutral_state",
)

_EXPORTS = {
    "_finite_float": ".params",
    "Regime": ".params",
    "_as_regime": ".params",
    "EquilibriumClosure": ".params",
    "RegimeRiskPremiaScales": ".params",
    "RegimeLogSvDynamics": ".params",
    "RegimeTransition": ".params",
    "CrraRiskPremia": ".params",
    "RegimeSwitchLogSvParams": ".params",
    "_poly_pad": ".equilibrium",
    "_poly_mul": ".equilibrium",
    "_poly_derivative": ".equilibrium",
    "_poly_shift": ".equilibrium",
    "_poly_exp": ".equilibrium",
    "_poly_value": ".equilibrium",
    "_sigma2_polynomial": ".equilibrium",
    "_physical_drift_polynomial": ".equilibrium",
    "_regime_difference": ".equilibrium",
    "_equilibrium_rhs": ".equilibrium",
    "EquilibriumSolution": ".equilibrium",
    "solve_regime_switch_equilibrium": ".equilibrium",
    "_risk_neutral_drift_polynomial": ".equilibrium",
    "_validate_continuous_boundary_admissibility": ".equilibrium",
    "RiskNeutralState": ".equilibrium",
    "evaluate_risk_neutral_state": ".equilibrium",
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
