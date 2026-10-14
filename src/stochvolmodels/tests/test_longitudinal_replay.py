"""Hand-enumerated event histories, temporal boundaries and exact OCA identities."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.fitters.diagnostics import DiagnosticReport


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def replay():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/replay_surface_diagnostics.py").is_file()), None)
    if root is None:
        pytest.skip("longitudinal replay is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("replay_surface_diagnostics")


@pytest.fixture
def inputs(replay):
    return replay.synthetic_inputs()


def combined(replay):
    """Independent hand prices, checks and parity; the put is excluded from fitting."""
    when, expiry = "2024-04-17T00:00:00+00:00", "2024-06-01T00:00:00+00:00"
    quotes = pd.DataFrame([dict(source_row_id=cp, contract="option_"+cp, strike=100.,
        optiontype=cp, expiry=expiry, exchange_time=when, bid_price=9., ask_price=11.,
        bid_iv=.19, ask_iv=.21, log_moneyness=0., quote_age_seconds=0., forward_price=100.,
        discount=1., ttm=.125) for cp in ("C", "P")])
    checks = pd.DataFrame([dict(source_row_id=cp, check=check, status="pass", reason="")
                          for cp in ("C", "P") for check in sorted(replay.REQUIRED_QUALITY)])
    qm = dict(kind="quote_quality", config=dict(price_unit="USD per unit"),
              requested_time=when, source_time=when)
    quality = DiagnosticReport.from_tables(dict(quotes=quotes, checks=checks,
        summary=pd.DataFrame([dict(snapshot_age_status="pass")])), qm)
    parity = DiagnosticReport.from_tables(dict(pairs=pd.DataFrame([dict(strike=100.,
        source_row_ids=["C", "P"], status="pass", reason="", distance_outside=0.)])),
        dict(qm, kind="parity"))
    return DiagnosticReport.from_tables(dict(quotes=pd.DataFrame([dict(quote_id="C",
        strike=100., optiontype="C", model_price=11.5, model_iv=.20, forward=100., discount=1.,
        ttm=.125, bid_price=9., ask_price=11., bid_iv=.19, ask_iv=.21)])),
        dict(kind="combined_research", price_unit="USD per unit", price_tolerance=.01,
             iv_tolerance=.0001, oca_quality=quality.to_dict(), oca_parity=parity.to_dict()))


def normalize(replay, report):
    tracks, observations = replay.combined_observations(report, snapshot_id="day_0",
        underlying_id="test asset", model_id="saved_model")
    return pd.DataFrame(observations).merge(pd.DataFrame(tracks), on="track_id")


def test_hand_history_preserves_gaps_recovery_and_cohort_censoring(replay, inputs):
    manifest, observations = inputs
    report = replay.replay_diagnostics(manifest, observations)
    rows, events = report.table("observations"), report.table("episodes")
    assert rows.transition.tolist() == ["onset", "persistent_violation", "unavailable",
        "recovered_after_gap", "onset", "clear", "onset"]
    assert len(events) == 3
    first = events.iloc[0]
    assert first.violation_observations == 2 and first.unavailable_observations == 1
    assert first.consecutive_violation_pairs == 1
    assert first.recovery_interval_start == manifest["snapshots"][1]["requested_time"]
    assert first.recovery_interval_end == manifest["snapshots"][3]["requested_time"]
    assert pd.isna(events.iloc[0].censor_reason)
    assert events.censor_reason.tolist()[1:] == ["split_end", "panel_end"]
    assert rows.iloc[4].event_id != rows.iloc[6].event_id
    assert pd.isna(rows.iloc[5].event_id)
    assert report.metadata["continuous_persistence"] is False


def test_missing_slot_and_unselected_contract_cannot_resolve_an_event(replay, inputs):
    manifest, observations = inputs
    del observations[1:4]
    report = replay.replay_diagnostics(manifest, observations)
    rows = report.table("observations")
    assert rows.state.tolist()[:5] == ["violation", "unavailable", "unavailable",
                                       "unavailable", "violation"]
    assert rows.iloc[4].transition == "violation_after_gap"
    assert rows.iloc[0].event_id == rows.iloc[4].event_id
    assert report.table("episodes").iloc[0].recovery_time is None
    assert report.table("summary").expected_slots.sum() == 7


def test_schedule_gap_blocks_consecutive_persistence(replay, inputs):
    manifest, observations = inputs
    manifest["max_gap_seconds"] = 3600
    report = replay.replay_diagnostics(manifest, observations)
    assert report.table("observations").iloc[1].transition == "violation_after_gap"
    assert report.table("episodes").consecutive_violation_pairs.sum() == 0
    assert report.table("episodes").iloc[0].schedule_gaps == 3


def test_expiry_is_unavailable_and_censors_instead_of_recovering(replay, inputs):
    manifest, observations = inputs
    track = manifest["tracks"][0]
    old_id = track["track_id"]
    track.update(replay.track_identity(**{name: track[name] if name != "expiry"
        else manifest["snapshots"][3]["requested_time"] for name in replay.IDENTITY}))
    for row in observations:
        assert row["track_id"] == old_id
        row["track_id"] = track["track_id"]
    report = replay.replay_diagnostics(manifest, observations)
    rows = report.table("observations")
    assert rows.iloc[3:].state.eq("unavailable").all()
    assert all("expired" in reasons for reasons in rows.iloc[3:].reasons)
    events = report.table("episodes")
    assert len(events) == 1 and events.iloc[0].censor_reason == "expiry"
    assert events.iloc[0].recovery_time is None


def test_validation_changes_cannot_change_calibration_events_or_counts(replay, inputs):
    manifest, observations = inputs
    before = replay.replay_diagnostics(manifest, observations)
    for row in observations[5:]:
        row.update(state="violation", severity=999., source_ids=["different_validation"])
    after = replay.replay_diagnostics(manifest, observations)
    for table in ("observations", "episodes", "summary"):
        a, b = before.table(table), after.table(table)
        pd.testing.assert_frame_equal(a.loc[a.split.eq("calibration")].reset_index(drop=True),
                                      b.loc[b.split.eq("calibration")].reset_index(drop=True))


def test_future_append_does_not_rewrite_prior_observation_states(replay, inputs):
    manifest, observations = inputs
    full = replay.replay_diagnostics(manifest, observations).table("observations")
    manifest["snapshots"] = manifest["snapshots"][:3]
    manifest["tracks"][0]["expected_snapshot_ids"] = ["day_0", "day_1", "day_2"]
    prefix = replay.replay_diagnostics(manifest, observations[:3]).table("observations")
    pd.testing.assert_frame_equal(full.iloc[:3].reset_index(drop=True), prefix)


def test_missing_time_or_source_identity_abstains(replay, inputs):
    manifest, observations = inputs
    observations[0]["source_time"] = None
    observations[1]["source_ids"] = []
    rows = replay.replay_diagnostics(manifest, observations).table("observations")
    assert rows.iloc[:3].state.eq("unavailable").all()
    assert rows.iloc[3].transition == "clear"


@pytest.mark.parametrize("defect", ["future", "naive", "duplicate", "unexpected", "split",
                                    "nan", "negative", "bool", "zero_violation", "unknown_state",
                                    "snapshot_duplicate", "snapshot_split", "track_identity",
                                    "slot_duplicate", "unknown_slot", "gap_zero"])
def test_invalid_frozen_evidence_is_rejected(replay, inputs, defect):
    manifest, observations = inputs
    row = observations[0]
    if defect == "future":
        row["source_time"] = "2025-01-01T00:00:00Z"
    elif defect == "naive":
        row["source_time"] = "2024-04-17"
    elif defect == "duplicate":
        observations.append(deepcopy(row))
    elif defect == "unexpected":
        row["track_id"] = "unknown"
    elif defect == "split":
        row["split"] = "validation"
    elif defect in ("nan", "negative", "bool", "zero_violation"):
        row["severity"] = {"nan": np.nan, "negative": -1., "bool": True,
                           "zero_violation": 0.}[defect]
    elif defect == "unknown_state":
        row["state"] = "probably_fine"
    elif defect == "snapshot_duplicate":
        manifest["snapshots"].append(deepcopy(manifest["snapshots"][0]))
    elif defect == "snapshot_split":
        manifest["snapshots"][0]["split"] = "validation"
    elif defect == "track_identity":
        manifest["tracks"][0]["instrument_id"] = "changed"
    elif defect == "slot_duplicate":
        manifest["tracks"][0]["expected_snapshot_ids"].append("day_0")
    elif defect == "unknown_slot":
        manifest["tracks"][0]["expected_snapshot_ids"].append("missing")
    else:
        manifest["max_gap_seconds"] = 0
    with pytest.raises(ValueError):
        replay.replay_diagnostics(manifest, observations)


def test_normalization_uses_original_contracts_and_hand_interval_distance(replay):
    report = combined(replay)
    before = report.to_dict()
    frame = normalize(replay, report)
    rows = frame.set_index(["instrument_id", "diagnostic"])
    assert rows.loc[("option_C", "observed_price"), "severity"] == .5
    assert rows.loc[("option_C", "observed_iv"), "state"] == "clear"
    assert rows.loc[("option_P", "observed_price"), "state"] == "unavailable"
    assert rows.loc[("option_P", "observed_iv"), "state"] == "unavailable"
    assert rows.loc[("strike:100", "parity"), "state"] == "clear"
    assert all(frame.expiry.eq("2024-06-01T00:00:00+00:00"))
    assert report.to_dict() == before


def test_known_quote_failure_remains_visible_despite_unchecked_age(replay):
    raw = combined(replay).to_dict()
    checks = raw["metadata"]["oca_quality"]["tables"]["checks"]
    for row in checks["data"]:
        if row[1] == "quote_age":
            row[2], row[3] = "not_checked", "age_limit_missing"
        if row[0] == "C" and row[1] == "uncrossed_price":
            row[2], row[3] = "fail", "crossed_price"
    frame = normalize(replay, raw).set_index(["instrument_id", "diagnostic"])
    assert frame.loc[("option_C", "quote_quality"), "state"] == "violation"
    assert frame.loc[("option_P", "quote_quality"), "state"] == "unavailable"


@pytest.mark.parametrize("defect", ["duplicate_contract", "model_identity", "model_strike",
                                    "parity_clock", "parity_legs", "units", "forward"])
def test_mismatched_sidecars_cannot_grant_recovery(replay, defect):
    raw = combined(replay).to_dict()
    quality = raw["metadata"]["oca_quality"]
    if defect == "duplicate_contract":
        quality["tables"]["quotes"]["data"][1][1] = "option_C"
    elif defect == "model_identity":
        raw["tables"]["quotes"]["data"][0][0] = "other_source"
    elif defect == "model_strike":
        raw["tables"]["quotes"]["data"][0][1] = 101.
    elif defect == "parity_clock":
        raw["metadata"]["oca_parity"]["metadata"]["source_time"] = "2024-04-16T00:00:00Z"
    elif defect == "parity_legs":
        quality["tables"]["quotes"]["data"][1][3] = "C"
    elif defect == "units":
        quality["metadata"]["config"]["price_unit"] = "coin"
    else:
        raw["tables"]["quotes"]["data"][0][5] = 101.
    with pytest.raises(ValueError):
        normalize(replay, raw)


def test_immutable_strict_roundtrip_and_input_independence(replay, inputs):
    manifest, observations = inputs
    report = replay.replay_diagnostics(manifest, observations)
    before = json.dumps(report.to_dict(), allow_nan=False, sort_keys=True)
    manifest["tracks"][0]["instrument_id"] = "changed"
    observations[0]["source_ids"].append("changed")
    changed = report.table("observations")
    changed.iloc[0, 0] = "changed"
    assert json.dumps(report.to_dict(), allow_nan=False, sort_keys=True) == before
    assert DiagnosticReport.from_dict(json.loads(before)) == report


def test_saved_cli_rejects_changed_observations_digest(replay, inputs, tmp_path, monkeypatch):
    manifest, observations = inputs
    raw = json.dumps(observations).encode()
    (tmp_path / "observations.json").write_bytes(raw)
    manifest.update(observations_file="observations.json", observations_sha256="0"*64)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["replay", "--manifest", str(path)])
    with pytest.raises(ValueError, match="digest mismatch"):
        replay.run_local(replay.Locals.SAVED)


def test_snapshot_mixing_is_rejected(replay, inputs):
    manifest, observations = inputs
    observations[0]["requested_time"] = observations[1]["requested_time"]
    with pytest.raises(ValueError, match="declared snapshot"):
        replay.replay_diagnostics(manifest, observations)


def test_changed_policy_cannot_reuse_a_track_to_claim_recovery(replay):
    original = combined(replay).to_dict()
    changed = deepcopy(original)
    changed["metadata"]["price_tolerance"] = 1.
    before, after = normalize(replay, original), normalize(replay, changed)

    def selected(frame):
        return frame.loc[frame.diagnostic.eq("observed_price") & frame.instrument_id.eq("option_C")]

    assert selected(before).state.iloc[0] == "violation"
    assert selected(after).state.iloc[0] == "clear"
    assert selected(before).track_id.iloc[0] != selected(after).track_id.iloc[0]


def test_saved_cli_success_matches_direct_replay(replay, inputs, tmp_path, monkeypatch, capsys):
    manifest, observations = inputs
    raw = json.dumps(observations).encode()
    (tmp_path / "observations.json").write_bytes(raw)
    manifest.update(observations_file="observations.json",
                    observations_sha256=replay.sha256(raw).hexdigest())
    path, output = tmp_path / "manifest.json", tmp_path / "result"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["replay", "--manifest", str(path), "--output", str(output)])
    replay.run_local(replay.Locals.SAVED)
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert saved == replay.replay_diagnostics(manifest, observations).to_dict()
    assert (output / "episodes.csv").exists()
    assert "calibration" in capsys.readouterr().out


@pytest.mark.parametrize("defect", ["cached", "regressed", "calibration_source_in_validation"])
def test_reused_or_out_of_order_sources_are_unavailable(replay, inputs, defect):
    manifest, observations = inputs
    if defect == "cached":
        observations[1]["source_ids"] = observations[0]["source_ids"][:]
        observations[1]["source_time"] = observations[0]["source_time"]
        index, reason = 1, "reused_source_observation"
    elif defect == "regressed":
        observations[3]["source_time"] = "2024-04-16T00:00:00Z"
        index, reason = 3, "source_time_regressed"
    else:
        observations[5]["source_time"] = observations[4]["source_time"]
        index, reason = 5, "source_before_validation_split"
    report = replay.replay_diagnostics(manifest, observations)
    row = report.table("observations").iloc[index]
    assert row.state == "unavailable"
    assert reason in row.reasons


def test_validation_inventory_cannot_create_calibration_missing_slots(replay, inputs):
    manifest, observations = inputs
    inventory = [dict(track_id=r["track_id"], snapshot_id=r["snapshot_id"]) for r in observations]
    base = replay.replay_diagnostics(replay.freeze_causal_schedule(manifest, inventory),
                                     observations)
    track = deepcopy(manifest["tracks"][0])
    track.update(replay.track_identity(**{k: track[k] if k != "instrument_id" else "new_contract"
                                        for k in replay.IDENTITY}))
    manifest["tracks"].append(track)
    new = dict(observations[-1], track_id=track["track_id"])
    observations.append(new)
    inventory.append(dict(track_id=track["track_id"], snapshot_id=new["snapshot_id"]))
    frozen = replay.freeze_causal_schedule(manifest, inventory)
    assert frozen["tracks"][-1]["expected_snapshot_ids"] == ["day_6"]
    extended = replay.replay_diagnostics(frozen, observations)
    for name in ("observations", "episodes", "summary"):
        a, b = base.table(name), extended.table(name)
        pd.testing.assert_frame_equal(a.loc[a.split.eq("calibration")].reset_index(drop=True),
                                      b.loc[b.split.eq("calibration")].reset_index(drop=True))
    frozen["tracks"][-1]["expected_snapshot_ids"].append("day_0")
    with pytest.raises(ValueError, match="precedes known inventory"):
        replay.replay_diagnostics(frozen, observations)
