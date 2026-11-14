"""Repository-only regressions for the fixed-fit BTC and Figure 8 PDE comparisons."""

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
