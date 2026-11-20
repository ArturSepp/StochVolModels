"""Independent finite-state checks of the experimental upper-volatility closure."""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from stochvolmodels.pde_solvers import (
    BoundaryPair1D, BoundarySet2D, Grid1D, PDE2DCoefficients, PDEDirection, TensorGrid2D,
)
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D

pytestmark = pytest.mark.repository_only


@pytest.mark.parametrize("covariance", [-.06, .06])
@pytest.mark.parametrize("upper_drift", [-.3, .1])
def test_return_law_including_mixed_face_and_corners(monkeypatch, covariance, upper_drift):
    root = Path(__file__).resolve().parents[3]
    if not (root/"validation").is_dir():
        pytest.skip("requires the repository research backend")
    monkeypatch.syspath_prepend(str(root))
    from validation.pde_monotone import Monotone2DReference, MonotoneConfig

    grid = TensorGrid2D(Grid1D([-.5, 0., .5]), Grid1D([0., .5, 1.]))
    ones = np.ones(grid.shape)
    coeff = PDE2DCoefficients(variance_x=.05*ones, drift_x=0.*ones,
                              variance_y=ones*[0., .05, .2],
                              drift_y=ones*[.1, 0., upper_drift],
                              covariance_xy=ones*[0., covariance, covariance], discount=0.*ones)
    pair = BoundaryPair1D.natural()
    split = SplitOperator2D.build(grid, coeff, BoundarySet2D(pair, pair))
    upper_return = 1.6 + 2*max(-upper_drift, 0.)
    # Independent row generators: constant tangential diffusion plus a birth/death chain.
    qx = np.array([[0., 0., 0.], [.2, -.4, .2], [0., 0., 0.]])
    qs = np.array([[-.2, .2, 0.], [.2, -.4, .2], [0., upper_return, -upper_return]])
    dense = np.kron(qx, np.eye(3)) + np.kron(np.eye(3), qs)
    # At the only fully interior node, transfer |c| of each axial variance to
    # two diagonal jumps of spacing 0.5. One mixed jump arrives at an upper corner.
    rate = abs(covariance)/(2*.5**2)
    dense[4, [1, 3, 5, 7]] -= rate
    dense[4, [0, 8] if covariance > 0 else [2, 6]] += rate
    dense[4, 4] += 2*rate
    initial = np.zeros(9)
    initial[[2, 5, 8]] = [.25, .5, .25]
    weights = grid.quadrature_weights.ravel()
    density = (initial/weights).reshape(grid.shape)
    new = Monotone2DReference.from_split(split, spatial=MonotoneConfig(
        fit_log_forward=False, reflect_upper_vol=True))
    np.testing.assert_allclose(new.backward.toarray(), dense, atol=1e-14)
    np.testing.assert_allclose(weights[:, None]*new.forward.toarray(),
                               dense.T*weights[None, :], atol=1e-15)
    for method, expected in (
        ("expm", expm(.125*dense.T) @ initial),
        ("be", np.linalg.solve(np.eye(9)-.125*dense.T, initial)),
    ):
        observed = new.step(density, PDEDirection.FORWARD, .125, method).values.ravel()*weights
        np.testing.assert_allclose(observed, expected, atol=2e-14)
        assert observed.reshape(3, 3)[:, 1].min() > 0 and observed.min() >= 0
    np.testing.assert_allclose(new.moment_errors["covariance_xy"][:, -1], -covariance)
    np.testing.assert_allclose(new.moment_errors["drift_y"][:, -1], -.5*upper_return-upper_drift)
    np.testing.assert_allclose(new.moment_errors["variance_y"][:, -1], .25*upper_return-.4)
    np.testing.assert_allclose(new.moment_errors["drift_y"][:, 0], 0., atol=1e-15)
    np.testing.assert_allclose(new.moment_errors["variance_y"][:, 0], .1*.5)
    # Switching off the return transition must be detectable from every initial upper node.
    control = Monotone2DReference.from_split(split, spatial=MonotoneConfig(fit_log_forward=False))
    trapped = control.step(density, PDEDirection.FORWARD, .125, "expm").values
    np.testing.assert_allclose(trapped[:, :-1], 0., atol=1e-15)
    assert (trapped.ravel()*weights-expm(.125*dense.T)@initial).max() > .01
