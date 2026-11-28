"""Sparse PDE fit, arbitrage checks and independent numerical refinement."""

import numpy as np
import pytest
import vanilla_option_pricers as vop

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.sparse_local_vol import fit_sparse_local_vol
from stochvolmodels.models.lsv import LeverageSurface, LSVModel
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig, LSVMonteCarloEngine
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
from stochvolmodels.tests.test_lsv_mc import _constant


def chain(vols):
    return OptionChain(
        ttms=np.array([180 / 365]),
        forwards=np.ones(1),
        discfactors=np.array([0.98]),
        ids=np.array(["6M"]),
        strikes_ttms=[np.array([0.8, 0.9, 1.0, 1.1, 1.2])],
        optiontypes_ttms=[np.array(["P", "P", "C", "C", "C"])],
        mid_ivs=[np.asarray(vols)],
    )


@pytest.mark.parametrize("vols", [[0.25] * 5, [0.25, 0.235, 0.22, 0.212, 0.204]])
def test_fit(vols):
    result = fit_sparse_local_vol(chain(vols), spot=1.0)
    assert result.status == "ok"
    assert max(abs(result.residual_iv_bp)) < 1
    assert max(abs(result.refined_iv_bp)) < 5
    assert result.surface.values.shape == (1, 5)
    assert result.time_grid[-2:] == pytest.approx([175 / 365, 180 / 365])
    assert result.density_diagnostics.mass == pytest.approx(1.0, abs=1e-8)
    assert result.density_diagnostics.normalized_forward == pytest.approx(1.0, abs=5e-5)
    if np.ptp(vols) == 0:
        assert result.surface.values[0] == pytest.approx(vols, abs=0.002)


def test_arbitrage_and_missing_strike():
    with pytest.raises(ValueError, match="arbitrage"):
        fit_sparse_local_vol(chain([0.20, 0.21, 0.80, 0.19, 0.18]), spot=1.0)
    with pytest.raises(ValueError, match="strikes"):
        fit_sparse_local_vol(chain([0.2] * 5), spot=2.0)


def test_skew_repricing_with_wider_domain_and_standalone_lsv_mc():
    iv = np.array([0.25, 0.235, 0.22, 0.212, 0.204])
    quotes = chain(iv)
    t = quotes.ttms[0]
    fit = fit_sparse_local_vol(quotes, spot=1.0)
    pricer = LocalVolPdePricer(grid=Grid1D.uniform(-3.0, 3.0, 1201))
    wide = pricer.price_chain_detailed(
        quotes, fit.surface, time_grid=np.r_[np.arange(0, t, 7 / 1460), t]
    )
    assert np.max(np.abs(wide.implied_vols[0] - iv)) < 5e-4
    assert wide.expiries[0].max_parity_error < 5e-5
    lev = LeverageSurface(fit.surface.times, fit.surface.grid, fit.surface.values)
    model = LSVModel.from_params(_constant(sigma0=1.0, theta=1.0), lev)
    cfg = LSVMonteCarloConfig(paths=32768, steps_per_year=252, seed=78)
    terminal = (
        LSVMonteCarloEngine(model, cfg)
        .simulate_paths(observation_times=np.array([0.0, t]))
        .assets[:, -1, 0]
    )
    strikes = quotes.strikes_ttms[0]
    kinds = quotes.optiontypes_ttms[0]
    target = vop.compute_bsm_vanilla_slice_prices(t, 1.0, strikes, iv, kinds)
    amounts = np.maximum((terminal[:, None] - strikes) * np.where(kinds == "C", 1.0, -1.0), 0.0)
    groups = amounts.reshape(2, -1, 5).mean(axis=0)
    assert np.all(
        np.abs(groups.mean(axis=0) - target) < 4 * groups.std(axis=0, ddof=1) / np.sqrt(len(groups))
    )
