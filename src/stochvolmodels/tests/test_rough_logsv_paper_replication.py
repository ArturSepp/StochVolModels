"""Repository-only contracts for replaying saved rough-logSV figures without calibration."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.pricers.rough_logsv import RoughLogSvParams, RoughMCConfig


ROOT = Path(__file__).resolve().parents[3]
REPLICATION = ROOT / "papers/rough_volterra_lognormal_sv_model_2026/replication"
pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.skipif(not (REPLICATION / "rough_logsv_saved_fits.py").is_file(),
                       reason="paper replay adapter requires the source checkout"),
]


@pytest.fixture
def replay():
    path = REPLICATION / "rough_logsv_saved_fits.py"
    spec = importlib.util.spec_from_file_location("rough_logsv_paper_replay_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved():
    folder = REPLICATION / "results/btc_fixed_h_2026-11-18"
    files = (REPLICATION / "inputs/core_option_panel.csv",
             folder / "classical_full_fit.json", folder / "classical_full_fit_quotes.csv")
    if not all(path.is_file() for path in files):
        pytest.skip("frozen local paper fixtures are not distributed")
    return pd.read_csv(files[0]), files[1], files[2]


def _standalone_stub(monkeypatch, replay, quote_file, currency=True):
    table = pd.read_csv(quote_file)
    called = []

    def price(_self, chain, params):
        assert params.H == .5
        assert np.array_equal(chain.ttms * 365, [7, 28, 91])
        called.append(True)
        return SimpleNamespace(prices=[
            (g.price * g.common_forward if currency else g.price).to_numpy()
            for _, g in table.groupby("days_to_expiry", sort=True)
        ])

    monkeypatch.setattr(replay.RoughLogSVPricer, "price_chain_result", price)
    return called


def test_classical_replay_uses_standalone_currency_premiums(replay, saved, monkeypatch):
    panel, fit, quotes = saved
    called = _standalone_stub(monkeypatch, replay, quotes)
    metadata, table, audit = replay.replay_fit("BTC", panel, fit, quotes)
    assert called == [True]
    np.testing.assert_allclose(table.price, pd.read_csv(quotes).price, rtol=0, atol=1e-16)
    assert metadata["parameters"]["H"] == .5
    assert audit["wider_calibration_gated"] and not audit["calibration_ready"]
    assert not audit["optimizer_run"] and not audit["fresh_sampling"]


def test_wrong_premium_units_cannot_silently_change_figure(replay, saved, monkeypatch):
    panel, fit, quotes = saved
    _standalone_stub(monkeypatch, replay, quotes, currency=False)
    with pytest.raises(ValueError, match="differs from the saved fit"):
        replay.replay_fit("BTC", panel, fit, quotes)


@pytest.mark.parametrize("column", ["common_forward", "bid_price", "ttm_act365", "optiontype"])
def test_changed_quote_conventions_fail_before_pricing(replay, saved, column, monkeypatch):
    panel, fit, quotes = saved
    changed = panel.copy()
    changed.loc[0, column] = "C" if column == "optiontype" else changed.loc[0, column] * 1.1
    called = _standalone_stub(monkeypatch, replay, quotes)
    with pytest.raises(ValueError, match="quote binding"):
        replay.replay_fit("BTC", changed, fit, quotes)
    assert not called


def test_reordered_quotes_do_not_reassign_control_means(replay, saved):
    panel, _, quotes = saved
    frame = pd.read_csv(quotes)
    with pytest.raises(ValueError, match="quote binding"):
        replay._same_quotes(frame.iloc[::-1], panel.loc[panel.asset.eq("BTC")])


def test_h020_remains_gated(replay, saved, tmp_path, monkeypatch):
    panel, fit, quotes = saved
    changed = json.loads(fit.read_text())
    changed["parameters"]["H"] = .2
    candidate = tmp_path / "fit.json"
    candidate.write_text(json.dumps(changed))
    called = _standalone_stub(monkeypatch, replay, quotes)
    with pytest.raises(ValueError, match="wider calibration"):
        replay.replay_fit("BTC", panel, candidate, quotes)
    assert not called


def test_classical_qualification_binds_parameter_vector(replay, saved, tmp_path, monkeypatch):
    panel, fit, quotes = saved
    changed = json.loads(fit.read_text())
    changed["parameters"]["theta"] += .001
    candidate = tmp_path / "fit.json"
    candidate.write_text(json.dumps(changed))
    called = _standalone_stub(monkeypatch, replay, quotes)
    with pytest.raises(ValueError, match="bind the saved parameters"):
        replay.replay_fit("BTC", panel, candidate, quotes)
    assert not called


def test_confirmation_selection_requires_exact_parameters(replay, tmp_path):
    p = RoughLogSvParams()
    folder = tmp_path / "btc-h030-precision-20261118/final_confirmation"
    folder.mkdir(parents=True)
    names = [name for name in asdict(p) if name != "H"]
    (folder / "config.json").write_text(json.dumps({
        "asset": "BTC", "H": .3, "parameter_names": names,
        "parameters": [getattr(p, name) for name in names],
    }))
    selected, _ = replay._confirmation("BTC", p, tmp_path, {})
    assert selected == folder
    p.theta += 1e-12
    with pytest.raises(ValueError, match="No saved confirmation"):
        replay._confirmation("BTC", p, tmp_path, {})


@pytest.mark.parametrize("defect", ["missing", "duplicate", "count"])
def test_checkpoint_inventory_cannot_silently_drop_or_relabel_samples(replay, defect):
    config = RoughMCConfig(pairs=(8, 4), raw_pairs=4, replicates=2, chunk_pairs=2)
    rows = [
        {"file": f"L{level}_R{rep}.npz", "level": level, "rep": rep, "pairs": pairs}
        for level, pairs in enumerate(config.pairs + (config.raw_pairs,))
        for rep in range(config.replicates)
    ]
    assert len(replay._checkpoint_catalog({"status": "completed", "completed": rows}, config)) == 6
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["pairs"] *= 2
    with pytest.raises(ValueError, match="Checkpoint"):
        replay._checkpoint_catalog({"status": "completed", "completed": rows}, config)
