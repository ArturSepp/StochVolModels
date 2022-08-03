from enum import Enum


class VariableType(Enum):
    LOG_RETURN = 1  # with transform var PHI
    Q_VAR = 2  # with transform var PSI
    SIGMA = 3  # with trasform for THETA
