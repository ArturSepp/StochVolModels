"""
config file
"""

from enum import Enum


class OptionType(str, Enum):
    """Option payoff codes accepted by SVM and VanillaOptionPricers."""

    CALL = "C"
    PUT = "P"
    INVERSE_CALL = "IC"
    INVERSE_PUT = "IP"


class VariableType(Enum):
    """
    state variables for log SV model
    """
    LOG_RETURN = 1  # with transform var PHI
    Q_VAR = 2  # with transform var PSI
    SIGMA = 3  # with trasform for THETA


class CoordinateSystem(str, Enum):
    """Surface axis: log(S/F) or absolute zero-carry asset level."""

    LEVEL = "level"
    LOG_FORWARD = "log_forward"
