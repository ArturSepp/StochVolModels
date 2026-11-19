"""Compatibility imports for ``stochvolmodels.models.rough_logsv._history_fast``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv._history_fast import (
    np as np,
    njit as njit,
    simulate as simulate,
    paths as paths,
)
