"""Known-law basket path, quanto, correlation and Brownian coupling checks."""

from dataclasses import replace

import numpy as np
import pytest
from stochvolmodels.models.basket import CorrelatedLocalVolModel
from stochvolmodels.models.basket.simulation import simulate_block
from stochvolmodels.models.lsv.local_vol import LocalVolSurface
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import Grid1D


def market(rho=0.6):
    surface = LocalVolSurface(
        np.array([0.5]), Grid1D(np.array([-1.0, 0.0, 1.0])), np.full((1, 3), 0.25)
    )
    return CorrelatedLocalVolModel(
        asof="issue",
        asset_ids=("A", "B"),
        currencies=("USD", "EUR"),
        settlement_currency="USD",
        surfaces=(surface, surface),
        curve_times=[0.0, 1.0],
        forward_levels=[[100, 200], [100, 200]],
        correlation=[[1, rho], [rho, 1]],
        fx_vols=[0, 0.1],
        equity_fx_correlations=[0, 0.5],
    )


def test_constant_quanto_and_coupling():
    m = market()
    cfg = LSVMonteCarloConfig(paths=8192, seed=41)
    ts = np.array([0.0, 0.25, 0.5, 1.0])
    a = m.simulate_paths(observation_times=ts, config=cfg)
    b = m.simulate_paths(observation_times=ts, config=cfg, refinement=1)
    np.testing.assert_array_equal(
        a.provenance["independent_group_ids"], np.tile(np.arange(cfg.paths // 2), 2)
    )
    assert not a.assets.flags.writeable
    np.testing.assert_allclose(a.assets, b.assets[:, ::2], rtol=2e-14)
    np.testing.assert_allclose(a.states["quadratic_variance"][0, :, 0], ts * 0.25**2)
    y = np.log(a.assets[:, -1] / m.spots)
    np.testing.assert_allclose(
        y.mean(axis=0), [-(0.25**2) / 2, -(0.25**2) / 2 - 0.25 * 0.5 * 0.1], atol=1e-12
    )
    assert np.corrcoef(y.T)[0, 1] == pytest.approx(0.6, abs=0.035)

    def block(i):
        return simulate_block(m, ts, cfg, block_id=i, size=128).assets

    forward = [block(i) for i in (0, 1)]
    reverse = [block(i) for i in (1, 0)]
    np.testing.assert_array_equal(forward[0], reverse[1])
    assert not np.array_equal(forward[0], forward[1])
    with pytest.raises(ValueError, match="semidefinite"):
        market(1.2)
    singular = replace(m, correlation=[[1, 1], [1, 1]], fx_vols=[0, 0])
    z = singular.simulate_paths(observation_times=ts, config=cfg)
    np.testing.assert_allclose(z.assets[:, :, 0] / 100, z.assets[:, :, 1] / 200, rtol=1e-14)
