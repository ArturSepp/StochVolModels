"""Compatibility imports for ``stochvolmodels.models.rough_logsv.expm``.

The canonical implementation owns the definitions and numerical behavior.
Existing import paths retain the same objects without deprecation warnings.
"""

from stochvolmodels.models.rough_logsv.expm import (
    np as np,
    njit as njit,
    prange as prange,
    _cabs as _cabs,
    _csqrt as _csqrt,
    _ccbrt as _ccbrt,
    _cubic_roots_monic as _cubic_roots_monic,
    _solve_3x3_complex as _solve_3x3_complex,
    expA_n3_numba as expA_n3_numba,
    expA_n1_numba as expA_n1_numba,
    _sinhc as _sinhc,
    expA_n2_numba as expA_n2_numba,
    batch_expA_n3 as batch_expA_n3,
    batch_expA_n2 as batch_expA_n2,
    batch_expA_n1 as batch_expA_n1,
    batch_expA as batch_expA,
    invA_rank1_numba_general as invA_rank1_numba_general,
    batch_invA as batch_invA,
)
