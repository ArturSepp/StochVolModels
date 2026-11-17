"""Repository-only regressions for the stochastic-volatility paper PDE replays."""

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.slow,
    pytest.mark.skipif(not (ROOT / "papers").is_dir(), reason="requires the paper checkout"),
]


def test_btc_final_fit_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        final_fit_density,
    )

    result = final_fit_density.replay()
    assert result["quote_count"] == 49
    assert result["reference_objective"] == pytest.approx(0.000496641409, abs=2e-8)
    assert result["density_objective"] == pytest.approx(0.0004732673, abs=2e-8)
    assert result["params"]["sigma0"] == pytest.approx(0.8616628659443291, abs=1e-14, rel=0)
    assert [len(row["strikes"]) for row in result["slices"]] == [12, 13, 15, 9]
    assert np.isfinite(result["peak_negative_mass"])


def test_figure8_density_replay(monkeypatch):
    monkeypatch.syspath_prepend(
        str(ROOT / "papers/unified_approach_for_hedging_il_df_2025/replication")
    )
    from figure8_density import compare_figure8

    from verify_replication import quadrature

    report = compare_figure8(quadrature)
    assert len(report["log_range"]) == 19 and len(report["strikes_over_forward"]) == 31
    assert [case["params"]["volvol"] for case in report["cases"]] == [1.0, 2.0, 3.0]
    for case in report["cases"]:
        for name in ("coarse", "fine", "time_refined", "wide_domain"):
            assert len(case["grids"][name]["protection_apr"]) == 19
            assert len(case["grids"][name]["implied_vols"]) == 31
        assert np.all(np.asarray(case["grids"]["fine"]["protection_apr"]) > 0.0)
    json.dumps(report, allow_nan=False)


def test_ijtaf_observables(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        figure_density as figures,
    )
    from stochvolmodels.pde_solvers import Grid1D

    axis = Grid1D([0.0, 1.0, 3.0])
    # Control cells have widths 0.5, 1.5, 1.0. Both histogram tails retain mass.
    actual = figures.histogram_probabilities(axis, [0.2, 0.4, 0.3], [0.25, 1.25, 2.5])
    np.testing.assert_allclose(actual, [0.05, 0.35, 0.45, 0.15])
    # Unequal time steps and a jump in eta distinguish integral and discrete means.
    actual = figures.annualized_variance(
        np.array([0.0, 0.25, 1.0]), np.array([1.0, 3.0, 9.0]), np.array([2.0, 0.5])
    )
    np.testing.assert_allclose(actual, [4.0, 8.0, 3.125])


def test_ijtaf_figure3_paper_fixings(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.theory import (
        moments_vol_qvar as paper,
    )
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        figure_density as figures,
    )

    # Drive the unmodified paper plotting routine with known positive volatility paths.
    times = np.linspace(0, 2, 8)
    paths = np.random.default_rng(37).uniform(0.2, 2, (len(times), 7))
    cases = list(figures.figure3_cases().items())[:2]
    monkeypatch.setattr(paper.LogSVPricer, "simulate_vol_paths", lambda *a, **kw: (paths, times))
    fig, ax = plt.subplots()
    try:
        paper.plot_qvar_vs_mc(dict(cases), is_vol=False, nb_path=7, grid_step=2, ax=ax)
        marker, _, (bars,) = ax.containers[0].lines
        means, se = figures.mc_fixings(paths, set(range(0, len(times), 2)))
        np.testing.assert_allclose(marker.get_xdata(), times[::2])
        np.testing.assert_allclose(marker.get_ydata(), means)
        np.testing.assert_allclose(np.asarray(bars.get_segments())[:, 1, 1], means + 3.92 * se)
    finally:
        plt.close(fig)


def test_ijtaf_figure6_terminal_marginals(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        figure_density as figures,
    )

    report = figures.figure6_comparison()
    figures.verify_figure6(report)
    assert set(report["panels"]) == {"log_return", "terminal_volatility"}
    for row in report["panels"].values():
        assert len(row["edges"]) == 200 and len(row["mc"]) == 201
        np.testing.assert_allclose(np.sum(row["fine"]), 1.0, rtol=0, atol=1e-10)


def test_ijtaf_figure3_expected_variance(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        figure_density as figures,
    )

    report = figures.figure3_comparison()
    figures.verify_figure3(report)
    assert [(c["sigma0"], c["kappa2"]) for c in report["cases"]] == [
        (s, k) for s in (1.5, 0.5) for k in (0, 4, 8)
    ]
    for case in report["cases"]:
        assert len(case["ode"]) == len(case["curve_times"]) == 1442
        assert case["grid_indices"] == [*range(0, 1442, 15), 1441]
        np.testing.assert_allclose([case["times"][0], case["times"][-1]], [0.0, 2.0])


def test_barrier_paper_partition(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.barriers.verify_pde import check_independent, check_partition

    example = check_partition()
    assert example["survivor"] == pytest.approx(0.5736505196, abs=1e-9)
    assert example["lower_exit"] == pytest.approx(0.4247287729, abs=1e-9)
    assert example["upper_exit"] == pytest.approx(0.0016207075, abs=1e-9)
    assert example["max_partition_gap"] == pytest.approx(0.01883667357, abs=1e-9)
    comparisons = check_independent()
    for case in ("constant", "time_varying"):
        rows = [r for r in comparisons if r["case"] == case]
        assert [r["nt"] for r in rows] == [10, 20, 40]
        assert rows[-1]["max_partition_gap"] < 0.3 * rows[0]["max_partition_gap"]


def test_barrier_endpoints_and_single_interval(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.barriers.pde_adapter import BarrierPde1D

    x = np.linspace(-0.05, 0.05, 101)
    a = BarrierPde1D(x)
    drift, var = np.zeros(2), np.full(2, 0.04)
    for x0, expected in [(x[0], [1.0, 0.0]), (x[-1], [0.0, 1.0])]:
        law = a.green(0.04, 199, drift, var, x0=x0)
        np.testing.assert_array_equal(law.density, 0.0)
        np.testing.assert_allclose([law.lower_exit[0], law.upper_exit[0]], expected, atol=1e-13)
    # Brownian motion killed at ±a, started at the midpoint (odd sine modes).
    odd = np.arange(1, 100, 2)
    exact = np.sum(
        4
        * (-1.0) ** ((odd - 1) // 2)
        / (np.pi * odd)
        * np.exp(-0.5 * var[1] * 0.04 * (np.pi * odd / (x[-1] - x[0])) ** 2)
    )
    law = a.green(0.04, 199, drift, var)
    np.testing.assert_allclose(law.surviving_mass, exact, rtol=0, atol=1e-4)
    np.testing.assert_allclose([law.lower_exit, law.upper_exit], (1 - exact) / 2, rtol=0, atol=5e-5)
    historical = a.survival(0.04, 199, drift, var, partition="historical")[:, 0]
    np.testing.assert_array_equal(historical, np.r_[0.0, np.ones(100)])
    full = a.survival(0.04, 199, drift, var)[:, 0]
    np.testing.assert_allclose(full[[0, -1]], [0.0, 1.0])
    np.testing.assert_allclose(full[50], (1 + exact) / 2, atol=5e-5, rtol=0)


def test_forward_variance_backbone_black(monkeypatch):
    """An unquoted knot and two off-knot expiries exercise loading lookup and state carry."""
    from numba.typed import List
    import pandas as pd
    from scipy.special import ndtr
    from stochvolmodels import LogSvParams, OptionChain
    from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D

    monkeypatch.syspath_prepend(str(ROOT))
    from papers.forward_var.verify_pde import price_backbone_chain

    ts = np.array([0.04, 0.15, 0.3])
    chain = OptionChain(
        ttms=ts,
        forwards=np.array([106.0, 81.0, 123.0]),
        discfactors=np.array([0.99, 0.97, 0.95]),
        ids=None,
        strikes_ttms=List(
            [np.array([101.0, 106.0]), np.array([80.0, 85.0]), np.array([120.0, 123.0])]
        ),
        optiontypes_ttms=List([np.array(["C", "P"]) for _ in ts]),
    )
    params = LogSvParams(
        sigma0=0.5,
        theta=0.5,
        beta=0.0,
        volvol=0.0,
        kappa1=0.0,
        kappa2=0.0,
        vol_backbone=pd.Series([0.7, 1.8, 0.9], index=[0.04, 0.1, 0.3]),
    )
    grid = TensorGrid2D(Grid1D.uniform(-2.0, 2.0, 401), Grid1D.uniform(0.0, 1.0, 5))
    prices, intervals = price_backbone_chain(chain, params, grid, steps=96)
    np.testing.assert_array_equal([s["end"] for s in intervals], [0.04, 0.1, 0.15, 0.3])
    for i, variance_time in enumerate(
        [
            0.04 * 0.7**2,
            0.04 * 0.7**2 + 0.06 * 1.8**2 + 0.05 * 0.9**2,
            0.04 * 0.7**2 + 0.06 * 1.8**2 + 0.2 * 0.9**2,
        ]
    ):
        f, d, k = chain.forwards[i], chain.discfactors[i], chain.strikes_ttms[i]
        sd = params.sigma0 * np.sqrt(variance_time)
        d1 = np.log(f / k) / sd + 0.5 * sd
        calls = f * ndtr(d1) - k * ndtr(d1 - sd)
        expected = np.where(chain.optiontypes_ttms[i] == "C", calls, calls - f + k)
        np.testing.assert_allclose(prices[i] / (f * d), expected / f, rtol=0.0, atol=2e-4)

    params.beta = 1.0
    params.kappa2 = 1.5
    with pytest.raises(ValueError, match="eta \\* beta"):
        price_backbone_chain(chain, params, grid, steps=2)


def test_forward_variance_backbone_generator():
    from stochvolmodels import LogSvParams
    from stochvolmodels.models.logsv_pde import _mma_coefficients
    from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D

    grid = TensorGrid2D(Grid1D.uniform(-1, 1, 3), Grid1D.uniform(0.0, 1.0, 5))
    params = LogSvParams(beta=0.4, volvol=0.9)
    eta, sigma = 1.6, grid.y_mesh
    c = _mma_coefficients(params, grid, eta)
    np.testing.assert_allclose(c.variance_x, (eta * sigma) ** 2 / 2)
    np.testing.assert_allclose(c.drift_x, -((eta * sigma) ** 2) / 2)
    np.testing.assert_allclose(c.covariance_xy, eta * params.beta * sigma**2)
    np.testing.assert_allclose(c.variance_y, params.vartheta2 * sigma**2 / 2)
    np.testing.assert_allclose(
        c.drift_y, (params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma)
    )


def test_forward_variance_paper_smile(monkeypatch):
    pytest.importorskip("qis")
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.forward_var import calibrate_forward_var as paper
    from papers.forward_var.verify_pde import compare_case, verify_case

    chain, params = paper.mc_smile_inputs(paper.sv.get_btc_test_chain_data())
    case = compare_case(chain, params, diagnostics=False)
    verify_case(case)
    assert [row["id"] for row in case["slices"]] == ["2w", "1m", "2m", "3m"]
    assert [len(row["strikes"]) for row in case["slices"]] == [31] * 4
    assert case["historical_mc_steps_per_year"] == 156
    assert case["default_mc_steps_per_year"] == 360
    # Seed-24 prices at the middle display strike from the archived forward-variance replay.
    np.testing.assert_allclose(
        [row["mc_prices"][15] for row in case["slices"]],
        [2575.376965, 3583.801353, 4734.336843, 4921.496570],
        rtol=1e-7,
    )
    for row in case["slices"]:
        assert np.all(np.isfinite(row["default_mc_prices"]))
        assert np.all(np.asarray(row["default_mc_se"]) > 0)
        assert row["default_mc_prices"] != row["mc_prices"]
    np.testing.assert_allclose(
        [r["eta"] for r in case["slices"]],
        [1.02500436, 1.11975866, 1.23471716, 1.25955165],
        rtol=0,
        atol=2e-7,
    )


def test_review_density_projection_and_objective(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        review_density as review,
    )

    # On a uniform spot grid with equal interior weights, node premiums are quadratic.
    spot = np.linspace(0.5, 1.5, 21)
    weights = np.full(21, 0.05)
    weights[[0, -1]] = 0.025
    strikes = np.array([0.823, 0.98, 1.141, 1.26])
    kinds = np.array(["P", "C", "P", "C"])
    raw, smooth = review._project(spot, weights, strikes, kinds)
    direct = [sum(weights * np.maximum((spot - k) * (1 if side == "C" else -1), 0))
              for k, side in zip(strikes, kinds)]
    np.testing.assert_allclose(raw, direct, rtol=0, atol=1e-15)
    np.testing.assert_allclose(smooth, 0.5 * np.where(kinds == "C", 1.5 - strikes, strikes - 0.5)**2,
                               rtol=0, atol=1e-14)
    bid = [np.array([0.19]), np.full(3, 0.33)]
    ask = [b + 0.02 for b in bid]
    chain = SimpleNamespace(strikes_ttms=[np.ones(1), np.ones(3)], bid_ivs=bid, ask_ivs=ask,
                            get_mid_vols=lambda: [0.5 * (b + a) for b, a in zip(bid, ask)])
    score = review._score([np.array([0.23]), np.full(3, 0.34)], chain)
    assert score["rmse_pp"] == pytest.approx(3.0 / np.sqrt(2))
    assert score["within_bidask_percent"] == 75.0


@pytest.mark.parametrize("case", ["SPXW", "GLD", "TLT", "TLT+"])
def test_local_review_fixed_parameters(monkeypatch, case):
    review_root = ROOT / "papers" / "robust_volatility_models" / "replication"
    if not review_root.is_dir():
        pytest.skip("requires the local review's frozen surfaces")
    monkeypatch.syspath_prepend(str(ROOT))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        review_density as review,
    )

    report = review.replay(review_root, case)
    count = review.CASES[case][2]
    assert len(report["quotes"]) == count == report["comparison"]["within_review_limits"]
    assert abs(report["reference_score"]["rmse_pp"] - report["density_score"]["rmse_pp"]) < 0.005
    assert all(q["objective_weight"] > 0 for q in report["quotes"])
    assert sum(q["objective_weight"] for q in report["quotes"]) == pytest.approx(1)
