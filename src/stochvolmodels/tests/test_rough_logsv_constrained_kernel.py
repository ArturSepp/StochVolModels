"""Hard-constraint and independent scalar-moment checks for experimental kernels."""
import numpy as np
import pytest

from stochvolmodels.pricers.rough_logsv import _constrained_kernel as constrained
from stochvolmodels.pricers.rough_logsv.kernel_fit import MomentKey


@pytest.mark.parametrize('n,cap', [(3, 10.0), (6, 14.0), (9, 18.0)])
@pytest.mark.parametrize('value', [-20.0, 0.0, 20.0])
def test_parameter_map_obeys_actual_caps(n, cap, value):
    """Extreme coordinates cannot exceed caps or create signed weights."""
    z = np.full(2 * n, value)
    nodes, weights = constrained._decode_kernel(z, n, .001, 360., .001, cap)
    assert np.all(nodes >= .001)
    assert np.all(nodes <= 360.)
    assert np.all(weights >= .001)
    assert weights.sum() < cap


def test_achieved_cap_overshoot_is_rejected():
    """Reject the old soft-penalty defect even for a small achieved overshoot."""
    with pytest.raises(ValueError, match='hard'):
        constrained._validate_achieved_kernel(
            np.array([.5]), np.array([10.000000001]), .001, 360., .001, 10.)


@pytest.mark.parametrize('nodes,weights', [([361.], [1.]), ([1.], [-.1]), ([np.nan], [1.])])
def test_achieved_invalid_geometry_is_rejected(nodes, weights):
    """Do not admit oversized nodes, signed weights or nonfinite parameters."""
    with pytest.raises(ValueError, match='hard'):
        constrained._validate_achieved_kernel(nodes, weights, .001, 360., .001, 10.)


def test_invalid_seed_is_not_clipped_or_rescaled():
    """A seed with excess total weight must fail before any optimizer is called."""
    with pytest.raises(ValueError, match='slack'):
        constrained._encode_kernel([1., 5.], [6., 6.], .001, 360., .001, 10.)
    with pytest.raises(ValueError, match='inside'):
        constrained._encode_kernel([1., 361.], [1., 1.], .001, 360., .001, 10.)


def test_classical_endpoint_is_exact_and_requires_feasible_cap():
    """The constant-kernel endpoint bypasses optimization with dimension one."""
    got = constrained.fit_positive_kernel(.5, 9, weight_cap=18.)
    assert got['nodes'] == [0.]
    assert got['weights'] == [1.]
    assert got['effective_N'] == 1
    assert got['status'] == 'exact_classical'
    with pytest.raises(ValueError, match='requires'):
        constrained.fit_positive_kernel(.5, 3, weight_cap=.5)


def test_classical_endpoint_respects_effective_factor_floor():
    """Check the actual one-factor geometry, including cap/floor equality."""
    with pytest.raises(ValueError, match='floor'):
        constrained.fit_positive_kernel(.5, 3, weight_cap=10., weight_floor=2.)
    for floor in (.2, 1.):
        got = constrained.fit_positive_kernel(.5, 9, weight_cap=1., weight_floor=floor)
        assert got['weights'] == [1.]
        assert got['feasible']


@pytest.mark.parametrize('n,node_cap', [(1, .01), (3, .01), (1, .001001), (3, .001001)])
def test_spectral_seed_handles_small_and_narrow_intervals(n, node_cap):
    """A valid small node interval must produce an ordered, interior seed."""
    nodes, weights = constrained._spectral_seed(.2, n, .001, node_cap, .001, 10.)
    assert np.all(nodes > .001)
    assert np.all(nodes < node_cap)
    assert np.all(np.diff(nodes) > 0.)
    coordinates = constrained._encode_kernel(nodes, weights, .001, node_cap, .001, 10.)
    got_nodes, got_weights = constrained._decode_kernel(coordinates, n, .001, node_cap, .001, 10.)
    np.testing.assert_allclose(got_nodes, nodes, rtol=1e-12)
    np.testing.assert_allclose(got_weights, weights, rtol=1e-12)


def test_fit_recovers_independently_computed_scalar_moments():
    """Recover one-factor geometry from closed scalar SDE moments, not package targets."""
    node, weight, level, kappa, nu = 1.2, 1.4, .5, 3., .7
    times = np.array([.03, .12, .4, 1.])
    a = node + weight * kappa
    rate = 2 * a - (weight * nu) ** 2
    stationary_v = 2 * a * level ** 2 / rate
    integrated_v = stationary_v * times + (level ** 2 - stationary_v) * (-np.expm1(-rate * times)) / rate
    leverage = weight * (stationary_v * (-np.expm1(-a * times)) / a
                         + (level ** 2 - stationary_v)
                         * (np.exp(-rate * times) - np.exp(-a * times)) / (a - rate))
    target = {MomentKey.MEAN: np.full_like(times, level),
              MomentKey.LEVEL: np.sqrt(integrated_v / times), MomentKey.LEVERAGE: leverage}
    got = constrained.fit_positive_kernel(.2, 1, weight_cap=3., node_cap=10.,
        maturities=times, fit_params=((level, level, kappa, nu),), targets=[target],
        initial_kernel=([2.], [1.]), max_nfev=80, tolerance=1e-11)
    assert got['success'], got
    np.testing.assert_allclose(got['nodes'], [node], rtol=2e-6)
    np.testing.assert_allclose(got['weights'], [weight], rtol=2e-6)
    assert got['objective'] < 1e-16


def test_evaluation_budget_is_not_labeled_convergence():
    """A feasible initial point after one evaluation does not qualify a fit."""
    target = {key: np.array([.4]) for key in MomentKey}
    got = constrained.fit_positive_kernel(.2, 1, weight_cap=3., node_cap=10.,
        maturities=(.2,), fit_params=((.5, .5, 3., .7),), targets=[target],
        initial_kernel=([2.], [1.]), max_nfev=1)
    assert not got['success']
    assert got['status'] == 'budget_exhausted_feasible'
    assert got['feasible']
