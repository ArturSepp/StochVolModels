"""Small-grid controls for the Stage 2 boundary experiment and weak moment budgets."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.repository_only


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root/"validation/pde_attribution_report.py").exists():
        pytest.skip("requires the volatility qualification checkout")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_attribution_report, pde_moment_attribution
    return pde_moment_attribution, pde_attribution_report


def test_nested_extension_does_not_insert_inside_old_domain(modules):
    experiment, _ = modules
    old = np.array([-1., 0., .5, 1.])
    wide = np.array([-2., -.9, 0., .5, .97, 1.4, 2.])
    np.testing.assert_array_equal(experiment._extend_from_archive(old, wide),
                                  [-2., -1., 0., .5, 1., 1.4, 2.])


def test_joint_be_budget_matches_dense_probability_evolution(modules):
    from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D

    experiment, reporting = modules
    x, s = np.array([-1., 0., .7, 2.]), np.array([0., .2, .5])
    grid = TensorGrid2D(Grid1D(x), Grid1D(s))
    params = SimpleNamespace(sigma0=.2, kappa1=.5, kappa2=.5, theta=.2, beta=-1., volvol=1.5)
    n = x.size*s.size
    a = np.zeros((n, n))
    # An independent chain visits both horizontal faces and the four corners.
    for i in range(x.size):
        for j in range(s.size):
            k = i*s.size+j
            for ni, nj, q in ((i-1, j, .8), (i+1, j, .4), (i, j-1, .55), (i, j+1, .7)):
                if 0 <= ni < x.size and 0 <= nj < s.size:
                    a[k, ni*s.size+nj] = q
            a[k, k] = -a[k].sum()
    budget = experiment.MomentBudget(SimpleNamespace(grid=grid, backward=a), params, 8)
    p = np.zeros(n)
    p[4] = 1.
    step = .25
    region_total = np.zeros((4, 3))
    abs_total = region_total.copy()
    for _ in range(8):
        p = np.linalg.solve(np.eye(n)-step*a.T, p)
        budget.observe(p.reshape(grid.shape)/grid.quadrature_weights, step)
        for k, prob in enumerate(p):
            i, j = divmod(k, s.size)
            region = 0 if j == 0 else 1 if j == 2 else 2 if i in (0, 3) else 3
            b = (.5+.5*s[j])*(.2-s[j])
            v = 3.25*s[j]**2
            physical = [0., b, 2*s[j]*b+v]
            for power in range(3):
                generator = sum(rate*s[col % s.size]**power for col, rate in enumerate(a[k]))
                gap = generator - physical[power]
                region_total[region, power] += step*prob*gap
                abs_total[region, power] += step*prob*abs(gap)
    row = budget.checkpoints[-1]
    np.testing.assert_allclose(row["integrated_signed"], region_total, atol=2e-15)
    np.testing.assert_allclose(row["integrated_absolute"], abs_total, atol=2e-15)
    np.testing.assert_allclose(budget.peak_reconciliation, 0., atol=2e-15)
    np.testing.assert_allclose(row["sigma_cell_mass"], p.reshape(4, 3).sum(axis=0))
    assert row["occupancy"][1] > 0 and row["occupancy"][2] > 0
    reporting.verify_marginals(dict(grid=dict(sigma=s), initial=budget.initial,
                                    checkpoints=budget.checkpoints))


def test_signed_factorial_interaction(modules):
    _, reporting = modules
    records = {}
    for key, value in (("d16", 3.), ("x_only", 5.), ("sigma_only", 8.), ("both", 9.5)):
        r = dict(mean=value, second_moment=2*value, variance=.1*value, cdf=[0., value/20])
        records[key] = dict(checkpoints=[r]*4, prices=[[value, 2*value]])
    obs, price = reporting.factorial(records)
    assert all(r["mean"] == -.5 for r in obs)
    np.testing.assert_allclose(obs[-1]["cdf"], [0., -.025])
    np.testing.assert_allclose(price, [[-.5, -1.]])
