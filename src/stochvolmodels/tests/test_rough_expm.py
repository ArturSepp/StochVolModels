"""Independent matrix-exponential references for the rough-kernel two-factor solver."""

import numpy as np
import pytest
from scipy.linalg import expm

from stochvolmodels.pricers.rough_logsv.expm import _sinhc, expA_n2_numba


@pytest.mark.parametrize("z", [0.0, 1.0e-14, -1.0e-14, 1.0e-14j, 1.0e-14 * (1.0 + 1.0j)])
def test_sinhc_has_unit_limit_at_zero(z: complex) -> None:
    """The removable singularity has unit limit along real and complex directions."""
    expected = 1.0 if z == 0.0 else np.sinh(z) / z
    np.testing.assert_allclose(_sinhc(z), expected, rtol=2.0e-15, atol=0.0)


@pytest.mark.parametrize(
    ("kappa", "lambdas", "weights"),
    [
        pytest.param(0.0, [0.0, 0.0], [0.0, 0.0], id="zero"),
        pytest.param(1.0, [2.0, 1.0], [0.0, 1.0], id="repeated-eigenvalue"),
        pytest.param(1.0, [0.0, 0.0], [1.0, -1.0], id="nilpotent"),
        pytest.param(1.0e-13, [0.0, 0.0], [0.3, 0.7], id="tiny"),
        pytest.param(0.7, [0.1, 2.5], [0.3, 0.7], id="distinct-real-eigenvalues"),
        pytest.param(1.0, [3.0, 1.0], [-1.0, 1.0], id="complex-eigenvalues"),
    ],
)
def test_two_factor_exponential_matches_scipy(
    kappa: float, lambdas: list[float], weights: list[float]
) -> None:
    """Compare closed-form propagation with SciPy's independent matrix algorithm."""
    rates = np.asarray(lambdas, dtype=np.float64)
    loadings = np.asarray(weights, dtype=np.float64)
    generator = -np.diag(rates) - kappa * np.outer(np.ones(2), loadings)
    expected = expm(generator)
    actual = expA_n2_numba(kappa, rates, loadings)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-16)
