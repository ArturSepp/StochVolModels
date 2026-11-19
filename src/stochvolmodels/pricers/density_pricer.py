"""Compatibility imports for density states and vanilla density valuation.

Canonical payloads live in data.density and their payoff reduction lives in
valuation. These aliases preserve existing imports and old serialized objects.
"""

from stochvolmodels.data.density import (
    DensityState1D as DensityState1D, DensityState2D as DensityState2D,
    CoordinateSystem as CoordinateSystem, Grid1D as Grid1D, TensorGrid2D as TensorGrid2D,
    FloatArray as FloatArray, _readonly_array as _readonly_array,
)
from stochvolmodels.valuation import (
    DensityPricer as DensityPricer, DensityPricingResult as DensityPricingResult,
    _validated_strikes as _validated_strikes,
)
