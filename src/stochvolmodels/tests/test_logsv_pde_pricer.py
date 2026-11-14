"""Independent contracts for the opt-in real-space LogSV vanilla pricer."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numba.typed import List
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import LogSvMeasure, LogSvModel
from stochvolmodels.models.logsv_pde import LogSvPdeCoefficients
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.utils.config import VariableType


def chain():
    return OptionChain(
        ttms=np.array([0.07, 0.251]),
        forwards=np.array([134.0, 82.0]),
        discfactors=np.array([0.97, 1.012]),
        ids=None,
        strikes_ttms=List([np.array([134.0, 129.0, 134.0]), np.array([86.0, 78.0, 86.0, 82.0])]),
        optiontypes_ttms=List([np.array(["P", "C", "C"]), np.array(["C", "P", "P", "C"])]),
    )


@pytest.mark.parametrize("beta", [-1.5, 0.7])
def test_generator_matches_mma_dynamics_and_snapshots_params(beta):
    params = LogSvParams(beta=beta, kappa2=1.5)
    provider = LogSvPdeCoefficients(LogSvModel(params))
    params.beta = 123.0
    grid = TensorGrid2D(Grid1D.uniform(-1, 1, 5), Grid1D.uniform(0.0, 1.0, 7))
    c = provider.evaluate(0.1, grid)
    sigma = grid.y_mesh
    np.testing.assert_allclose(c.variance_x, sigma**2 / 2)
    np.testing.assert_allclose(c.drift_x, -(sigma**2) / 2)
    np.testing.assert_allclose(
        c.drift_y, (params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma)
    )
    np.testing.assert_allclose(c.variance_y, (params.volvol**2 + beta**2) * sigma**2 / 2)
    np.testing.assert_allclose(c.covariance_xy, beta * sigma**2)
    np.testing.assert_array_equal(c.discount, 0.0)


def test_ragged_black_prices_market_parity_and_chain_ivols():
    data = chain()
    params = LogSvParams(sigma0=0.25, theta=0.25, beta=0.0, volvol=0.0)
    pricer = LogSvPdePricer()
    result = pricer.price_chain_detailed(data, params)
    assert [d.ttm for d in result.expiries] == list(data.ttms)
    assert len(result.steps) == 2 * pricer.steps_per_interval
    assert result.peak_negative_mass < 1.0e-12
    for i, (t, f, d, k, kinds) in enumerate(
        zip(data.ttms, data.forwards, data.discfactors, data.strikes_ttms, data.optiontypes_ttms)
    ):
        sd = params.sigma0 * np.sqrt(t)
        d1 = np.log(f / k) / sd + sd / 2
        calls = f * ndtr(d1) - k * ndtr(d1 - sd)
        oracle = d * np.where(kinds == "C", calls, calls - f + k)
        np.testing.assert_allclose(
            result.prices[i] / (d * f), oracle / (d * f), atol=7.0e-5, rtol=0.0
        )
        np.testing.assert_allclose(result.implied_vols[i], 0.25, atol=0.001, rtol=0.0)
        assert not result.invalid_iv[i].any()
        expiry = result.expiries[i]
        assert abs(expiry.mass - 1.0) < 1.0e-11
        assert abs(expiry.normalized_forward - 1.0) < 5.0e-6
        assert expiry.max_parity_error / (d * f) < 5.0e-6
        call, put = ((2, 0), (0, 2))[i]
        assert (
            abs(result.prices[i][call] - result.prices[i][put] - d * (f - k[call])) < d * f * 5.0e-6
        )
    plain, iv = pricer.compute_chain_prices_with_vols(data, params)
    for i in range(2):
        np.testing.assert_array_equal(plain[i], result.prices[i])
        np.testing.assert_array_equal(iv[i], result.implied_vols[i])


@pytest.mark.parametrize(
    "updates",
    [
        {"H": 0.2},
        {"vol_backbone": pd.Series([1.0], index=[0.5])},
        {"beta": 2.0, "kappa2": 1.0},
        {"nodes": np.array([1.0])},
    ],
)
def test_out_of_scope_dynamics(updates):
    params = LogSvParams()
    for key, value in updates.items():
        setattr(params, key, value)
    with pytest.raises(ValueError):
        LogSvPdePricer().price_chain(chain(), params)


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"measure": LogSvMeasure.INVERSE},
        {"is_spot_measure": False},
        {"variable_type": VariableType.Q_VAR},
        {"variable_type": VariableType.SIGMA},
    ],
    ids=["inverse", "legacy-inverse", "qvar", "sigma"],
)
def test_out_of_scope_measure_or_variable(request_kwargs):
    with pytest.raises(ValueError):
        LogSvPdePricer().price_chain(chain(), LogSvParams(), **request_kwargs)


def test_integer_strikes_keep_fractional_vols_and_failed_inversion_mask():
    grid = TensorGrid2D(Grid1D.uniform(-0.5, 0.5, 15), Grid1D.uniform(0.0, 0.5, 5))
    pricer = LogSvPdePricer(grid=grid, steps_per_interval=4)
    data = OptionChain.slice_to_chain(0.2, 100.0, np.array([100, 300]), np.array(["C", "C"]), 0.96)
    result = pricer.price_chain_detailed(
        data, LogSvParams(sigma0=0.25, theta=0.25, beta=0.0, volvol=0.0)
    )
    assert 0.0 < result.implied_vols[0][0] < 1.0
    assert result.prices[0][1] == 0.0
    assert np.isnan(result.implied_vols[0][1])
    np.testing.assert_array_equal(result.invalid_iv[0], [False, True])


def test_exact_adjacent_float_expiries_are_distinct():
    near = np.nextafter(0.125, 1.0)
    data = replace(chain(), ttms=np.array([0.125, near]))
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 13), Grid1D.uniform(0.0, 1.0, 5))
    result = LogSvPdePricer(grid=grid, steps_per_interval=4).price_chain_detailed(
        data, LogSvParams(sigma0=0.25, theta=0.25, beta=0.0, volvol=0.0)
    )
    assert [snapshot.ttm for snapshot in result.expiries] == [0.125, near]


def test_inverse_payoff_and_invalid_grid_rejected():
    data = chain()
    types = (np.array(["IP", "C", "C"]), data.optiontypes_ttms[1])
    with pytest.raises(ValueError, match="ordinary"):
        LogSvPdePricer().price_chain(replace(data, optiontypes_ttms=types), LogSvParams())
    grid = TensorGrid2D(Grid1D.uniform(0.1, 1.0, 5), Grid1D.uniform(0.01, 1.0, 5))
    with pytest.raises(ValueError):
        LogSvPdePricer(grid=grid).price_chain(data, LogSvParams())


@pytest.mark.repository_only
@pytest.mark.slow
def test_ijtaf_fixed_parameter_density_comparison(monkeypatch):
    """Replay the tracked paper cross-check on its original strike arrays."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "papers/logsv_model_with_quadratic_drift_ijtaf_2023").is_dir():
        pytest.skip("The IJTAF replication is available in the source checkout")
    monkeypatch.syspath_prepend(str(root))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        run_checks,
    )

    spy, btc = run_checks.run_local(run_checks.Locals.DENSITY)
    assert spy["case"] == "published_spy_12d"
    assert sum(len(q["strikes"]) for q in btc["quotes"]) == 49
    assert all("difference_in_full_price_spreads" in q for q in btc["quotes"])
