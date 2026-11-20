"""Independent regression controls for integrated bins and antithetic ratio uncertainty."""

from pathlib import Path

import numpy as np
import pytest

from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D

pytestmark = pytest.mark.repository_only


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root/"validation").is_dir():
        pytest.skip("requires the repository qualification runner")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_projection_report, pde_projection_study

    return pde_projection_study, pde_projection_report


def test_bin_integrals_including_partial_cells_and_tails(modules):
    study, _ = modules
    grid = TensorGrid2D(Grid1D([0., 1., 3.]), Grid1D([0., 1., 2.]))
    mass = np.array([[.02, .04, .04], [.05, .10, .15], [.15, .25, .20]])
    obs = study.bin_density(grid, mass/grid.quadrature_weights, [.25, 1., 2.6])
    # Intersect x cells [0,.5], [.5,2], [2,3] with each physical bin.
    parts = np.array([[.5, 0., 0.], [.5, 1/3, 0.], [0., 2/3, .6], [0., 0., .4]])
    prob = parts @ [.1, .3, .6]
    second = parts @ [.2, .7, 1.05]
    for k in ("prob", "package_probability"):
        np.testing.assert_allclose(obs[k], prob, atol=1e-15)
    for k in ("numerator", "package_numerator"):
        np.testing.assert_allclose(obs[k], second, atol=1e-15)
    np.testing.assert_allclose(obs["conditional"], second/prob)
    np.testing.assert_allclose(obs["share"].sum(), 1.)
    np.testing.assert_allclose(obs["direct_second"], 1.95)


def test_pair_ratio_influence_matches_manual_independent_pairs(modules):
    _, report = modules
    ids = np.array([[0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1]])
    sigma2 = np.arange(1., 13.).reshape(2, 6)
    result, influence = report.bin_estimate(ids, sigma2, 0)
    # List the independent pairs' hit counts and sums, in original partner order.
    counts = np.array([1., 1., 2., 1., 2., 0.])
    totals = np.array([1., 5., 9., 7., 19., 0.])
    ratio = totals.sum()/counts.sum()
    scores = (totals-counts*ratio)/counts.mean()
    np.testing.assert_allclose(result["conditional"], ratio)
    np.testing.assert_allclose(result["se"], scores.std(ddof=1)/np.sqrt(6))
    np.testing.assert_allclose(influence.ravel(), scores)
    np.testing.assert_allclose(result["seed_estimates"], [15/4, 26/3])
    assert (result["paths_hit"], result["pairs_hit"]) == (7, 5)
    np.testing.assert_allclose(report.standard_error(influence-influence), 0.)


def test_unoccupied_bin_has_no_conditional_estimate(modules):
    _, report = modules
    ids = np.zeros((2, 6), dtype=int)
    result, influence = report.bin_estimate(ids, np.ones_like(ids), 1)
    assert result["prob"] == result["numerator"] == result["pairs_hit"] == 0
    assert np.isnan(result["conditional"]) and np.isnan(result["se"])
    assert np.isnan(influence).all()
