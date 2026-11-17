"""Shared immutable NumPy array construction for models and fitters."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def readonly_float_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
    require_finite: bool = True,
) -> NDArray[np.float64]:
    """Validate array and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or (require_finite and not np.all(np.isfinite(array))):
        qualifier = "finite " if require_finite else ""
        raise ValueError(f"{name} must be a {qualifier}{ndim}-dimensional array")
    array.setflags(write=False)
    return array
