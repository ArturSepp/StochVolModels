"""Compatibility imports for ``stochvolmodels.models.rough_logsv._classical_pde``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv._classical_pde import (
    dataclass as dataclass,
    np as np,
    njit as njit,
    roots_legendre as roots_legendre,
    ndtr as ndtr,
    simpson as simpson,
    Model as Model,
    from_row as from_row,
    grid as grid,
    fitted_rates as fitted_rates,
    operator as operator,
    factorize as factorize,
    advance as advance,
    evolve as evolve,
    transforms as transforms,
    contour as contour,
    calls as calls,
    black as black,
)
