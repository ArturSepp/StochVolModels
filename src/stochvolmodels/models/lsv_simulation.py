"""Compatibility import for ``stochvolmodels.models.lsv.simulation``.

The module object is shared so existing function patches and imports reach the
canonical implementation.
"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module('stochvolmodels.models.lsv.simulation')
