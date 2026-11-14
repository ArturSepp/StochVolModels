"""Repository-only regressions for the BTC, impermanent-loss and IJTAF PDE replays."""

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
