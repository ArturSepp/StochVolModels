"""Retrospective diagnostic episodes on an explicitly frozen snapshot schedule.

Repository-only research: no fitting, threshold selection or live-data acquisition.
Run the synthetic example, or supply a manifest binding a normalized observations file
by SHA256. OCA combined sidecars can be normalized with ``combined_observations``.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from stochvolmodels import local_path
from stochvolmodels.fitters.diagnostics import DiagnosticReport


IDENTITY = ("underlying_id", "model_id", "expiry", "instrument_id", "diagnostic", "severity_unit",
            "rule_id")
REQUIRED_QUALITY = {"strike", "forward_price", "discount", "ttm", "optiontype",
                    "two_sided_price", "nonnegative_price", "uncrossed_price", "iv",
                    "exchange_time", "expiry", "price_bounds", "quote_age"}


def _time(value):
    """Normalize an explicitly aware timestamp; never infer a timezone."""
    if value is None:
        raise ValueError("An aware timestamp is required")
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("An aware timestamp is required")
    return result.tz_convert("UTC")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _restore(value):
    """Validate an already serialized sidecar without reconverting every primitive."""
    return value if isinstance(value, DiagnosticReport) else DiagnosticReport(
        json.dumps(value, sort_keys=True, allow_nan=False))


def track_identity(**fields):
    """Identify a fixed diagnostic scope, independent of time and observed outcomes."""
    fields.setdefault("rule_id", "caller_frozen")
    fields = {name: fields[name] for name in IDENTITY}
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise ValueError("Track identities must be nonempty strings")
    fields["expiry"] = _time(fields["expiry"]).isoformat()
    return dict(track_id=_digest(fields), **fields)


def freeze_causal_schedule(manifest, inventory):
    """Activate tracks when first known from inventory, without reading diagnostic states.

    Inventory records contain only ``track_id`` and ``snapshot_id``. A new contract
    first listed in validation cannot create missing calibration observations.
    Once activated, a track retains subsequent expected slots even when it disappears.
    """
    frozen = json.loads(json.dumps(manifest, allow_nan=False))
    times = {s["snapshot_id"]: _time(s["requested_time"]) for s in frozen["snapshots"]}
    tracks = {t["track_id"]: t for t in frozen["tracks"]}
    known = {}
    for row in inventory:
        tid, sid = row["track_id"], row["snapshot_id"]
        if tid not in tracks or sid not in tracks[tid]["expected_snapshot_ids"] or sid not in times:
            raise ValueError("Inventory is outside the declared track schedule")
        known[tid] = min(known.get(tid, times[sid]), times[sid])
    if set(known) != set(tracks):
        raise ValueError("Every track needs an inventory activation time")
    for tid, track in tracks.items():
        track["known_from"] = known[tid].isoformat()
        track["expected_snapshot_ids"] = [s for s in track["expected_snapshot_ids"]
                                          if times[s] >= known[tid]]
    frozen["schedule_policy"] = "first known inventory, then retain all subsequent expected slots"
    return frozen


def replay_diagnostics(manifest, observations):
    """Replay frozen expected slots, retaining unknowns and resetting at the split.

    A persistent violation means two adjacent, available scheduled observations of
    the same track within ``max_gap_seconds``. An episode can span missing evidence;
    its elapsed time is never interpreted as continuously observed persistence.
    """
    if manifest.get("schema_version") != 1 or not manifest.get("panel_id"):
        raise ValueError("A version-1 frozen panel manifest is required")
    split = _time(manifest["split_utc"])
    gap_limit = manifest["max_gap_seconds"]
    if (isinstance(gap_limit, bool) or not isinstance(gap_limit, Real)
            or not np.isfinite(gap_limit) or gap_limit <= 0):
        raise ValueError("max_gap_seconds must be finite and positive")
    snapshots = {}
    for item in manifest["snapshots"]:
        sid, when = item["snapshot_id"], _time(item["requested_time"])
        cohort = "calibration" if when < split else "validation"
        if not isinstance(sid, str) or not sid or sid in snapshots or item["split"] != cohort:
            raise ValueError("Duplicate snapshot or inconsistent split")
        snapshots[sid] = dict(snapshot_id=sid, requested_time=when.isoformat(), split=cohort)
    if not snapshots or len({s["requested_time"] for s in snapshots.values()}) != len(snapshots):
        raise ValueError("Snapshot times must be nonempty and unique")
    tracks, expected = {}, set()
    for item in manifest["tracks"]:
        track = track_identity(**{name: item[name] for name in IDENTITY})
        tid, slots = item["track_id"], item["expected_snapshot_ids"]
        if tid != track["track_id"] or tid in tracks or not slots or len(set(slots)) != len(slots):
            raise ValueError("Invalid track identity or duplicate expected slot")
        if not set(slots).issubset(snapshots):
            raise ValueError("Unknown expected snapshot")
        known = _time(item["known_from"]) if "known_from" in item else None
        if known is not None and any(_time(snapshots[s]["requested_time"]) < known for s in slots):
            raise ValueError("Expected observation precedes known inventory")
        tracks[tid] = dict(track, expected_snapshot_ids=list(slots),
                          known_from=known.isoformat() if known is not None else None)
        expected.update((tid, sid) for sid in slots)
    evidence = {}
    for item in observations:
        key = item["track_id"], item["snapshot_id"]
        if key not in expected or key in evidence:
            raise ValueError("Unexpected or duplicate observation")
        if item["state"] not in ("clear", "violation", "unavailable"):
            raise ValueError("Unknown observation state")
        snap = snapshots[key[1]]
        if _time(item["requested_time"]) != _time(snap["requested_time"]):
            raise ValueError("Observation is not from the declared snapshot")
        if item.get("split", snap["split"]) != snap["split"]:
            raise ValueError("Observation crosses the frozen split")
        source = item.get("source_time")
        if source is not None and _time(source) > _time(snap["requested_time"]):
            raise ValueError("Future source observation")
        value = item.get("severity")
        if item["state"] != "unavailable" and (
                isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
                or value < 0 or (item["state"] == "clear" and value != 0)
                or (item["state"] == "violation" and value == 0)):
            raise ValueError("Available evidence requires a finite consistent severity")
        if (not isinstance(item.get("reasons", []), list)
                or not isinstance(item.get("source_ids", []), list)):
            raise ValueError("Reasons and source identities must be lists")
        if (any(not isinstance(s, str) or not s for s in item.get("source_ids", []))
                or len(set(item.get("source_ids", []))) != len(item.get("source_ids", []))
                or any(not isinstance(s, str) for s in item.get("reasons", []))):
            raise ValueError("Source identities must be unique strings and reasons must be strings")
        evidence[key] = item

    rows, episodes = [], []
    for tid, track in sorted(tracks.items()):
        slots = sorted((snapshots[s] for s in track["expected_snapshot_ids"]),
                       key=lambda s: s["requested_time"])
        for cohort in ("calibration", "validation"):
            local = [s for s in slots if s["split"] == cohort]
            episode, previous = None, None
            seen_sources, latest_source = set(), None
            for snap in local:
                when = _time(snap["requested_time"])
                supplied = evidence.get((tid, snap["snapshot_id"]))
                item = supplied or dict(state="unavailable", reasons=["missing_observation"])
                state, reasons = item["state"], list(item.get("reasons", []))
                expired = when >= _time(track["expiry"])
                if expired:
                    state, reasons = "unavailable", reasons + ["expired"]
                    if episode:
                        episode["censor_reason"] = "expiry"
                        episodes.append(episode)
                        episode = None
                elif state != "unavailable" and (
                        item.get("source_time") is None or not item.get("source_ids")):
                    state, reasons = "unavailable", reasons + ["missing_source_time_or_identity"]
                if not expired and item.get("source_time") is not None and item.get("source_ids"):
                    source = _time(item["source_time"])
                    identity = source.isoformat(), tuple(sorted(item["source_ids"]))
                    temporal_reason = None
                    if cohort == "validation" and source < split:
                        temporal_reason = "source_before_validation_split"
                    elif identity in seen_sources:
                        temporal_reason = "reused_source_observation"
                    elif latest_source is not None and source < latest_source:
                        temporal_reason = "source_time_regressed"
                    if temporal_reason:
                        state, reasons = "unavailable", reasons + [temporal_reason]
                    seen_sources.add(identity)
                    latest_source = source if latest_source is None else max(source, latest_source)
                schedule_gap = previous is not None and (
                    when - _time(previous["requested_time"])).total_seconds() > gap_limit
                if schedule_gap and episode:
                    episode["schedule_gaps"] += 1
                event_id = episode["event_id"] if episode else None
                transition = "unavailable" if state == "unavailable" else "clear"
                if state == "violation":
                    if episode is None:
                        event_id = _digest([manifest["panel_id"], tid, cohort, snap["snapshot_id"]])
                        episode = dict(event_id=event_id, track_id=tid, split=cohort,
                            first_violation_time=snap["requested_time"],
                            last_violation_time=snap["requested_time"], recovery_time=None,
                            recovery_interval_start=None, recovery_interval_end=None,
                            violation_observations=0, unavailable_observations=0,
                            consecutive_violation_pairs=0, schedule_gaps=0,
                            peak_severity=0., censor_reason=None)
                        transition = "onset"
                    elif previous["state"] == "violation" and not schedule_gap:
                        transition = "persistent_violation"
                        episode["consecutive_violation_pairs"] += 1
                    else:
                        transition = "violation_after_gap"
                    episode["violation_observations"] += 1
                    episode["last_violation_time"] = snap["requested_time"]
                    episode["peak_severity"] = max(episode["peak_severity"], item["severity"])
                elif state == "clear" and episode:
                    interrupted = episode["unavailable_observations"] or episode["schedule_gaps"]
                    transition = "recovered_after_gap" if interrupted else "recovered"
                    episode.update(recovery_time=snap["requested_time"],
                        recovery_interval_start=episode["last_violation_time"],
                        recovery_interval_end=snap["requested_time"])
                    episodes.append(episode)
                    episode = None
                elif episode:
                    episode["unavailable_observations"] += 1
                row = dict(track_id=tid, **snap, state=state, transition=transition,
                    reported_state=item["state"], reported_severity=item.get("severity"),
                    event_id=event_id,
                    severity=item.get("severity") if state != "unavailable" else None,
                    source_time=item.get("source_time"), source_ids=item.get("source_ids", []),
                    reasons=sorted(set(reasons)), details=item.get("details", {}))
                rows.append(row)
                previous = row
            if episode:
                episode["censor_reason"] = "split_end" if cohort == "calibration" else "panel_end"
                episodes.append(episode)
    row_columns = ("track_id", "snapshot_id", "requested_time", "split", "state", "transition",
                   "reported_state", "reported_severity", "event_id", "severity", "source_time",
                   "source_ids", "reasons", "details")
    frame = pd.DataFrame(rows, columns=row_columns)
    episode_columns = ("event_id", "track_id", "split", "first_violation_time",
        "last_violation_time", "recovery_time", "recovery_interval_start", "recovery_interval_end",
        "violation_observations",
        "unavailable_observations", "consecutive_violation_pairs", "schedule_gaps", "peak_severity",
        "censor_reason")
    events = pd.DataFrame(episodes, columns=episode_columns)
    definitions = pd.DataFrame(list(tracks.values()), columns=("track_id", *IDENTITY,
                                                             "expected_snapshot_ids", "known_from"))
    summary = []
    for (cohort, diagnostic), group in frame.merge(definitions, on="track_id").groupby(
            ["split", "diagnostic"], sort=True):
        ev = events[events.split.eq(cohort) & events.track_id.isin(group.track_id)]
        summary.append(dict(split=cohort, diagnostic=diagnostic, expected_slots=len(group),
            clear=int(group.state.eq("clear").sum()),
            violation=int(group.state.eq("violation").sum()),
            unavailable=int(group.state.eq("unavailable").sum()), onsets=len(ev),
            recovered=int(ev.recovery_time.notna().sum()),
            censored=int(ev.censor_reason.notna().sum()),
            consecutive_violation_pairs=int(group.transition.eq("persistent_violation").sum())))
    return DiagnosticReport.from_tables(dict(observations=frame, episodes=events,
        tracks=definitions, summary=pd.DataFrame(summary)), dict(kind="longitudinal_replay",
        manifest=manifest, manifest_sha256=_digest(manifest),
        expected_slots=len(expected), split_reset=True, refits=0, threshold_selection=False,
        persistence_scope="consecutive scheduled observations of an identical diagnostic track",
        recovery_scope="first observed clear after violation; times are observation brackets",
        receive_availability="not established by replay", continuous_persistence=False,
        accuracy_certificate=False, executable_arbitrage=False))


def combined_observations(report, *, snapshot_id, underlying_id, model_id):
    """Normalize one saved expiry report into contract and expiry/strike tracks.

    Retain original OCA checks, pair identities and model residuals separately. Known
    quote failures remain violations even when age checks are unavailable. Contracts
    excluded from the saved fit have unavailable residuals, never a clean fit result.
    """
    report = _restore(report)
    metadata = report.metadata
    if metadata.get("kind") != "combined_research":
        raise ValueError("Expected a combined research report")
    quality = _restore(metadata["oca_quality"])
    qm = quality.metadata
    requested, source = _time(qm["requested_time"]), qm.get("source_time")
    if source is not None and _time(source) > requested:
        raise ValueError("Future OCA source observation")
    quotes, checks = quality.table("quotes"), quality.table("checks")
    if quotes.source_row_id.duplicated().any() or quotes.contract.duplicated().any():
        raise ValueError("Ambiguous source contract identities")
    if len(set(quotes.expiry)) != 1:
        raise ValueError("Normalize one actual expiry at a time")
    expiry = _time(quotes.expiry.iloc[0]).isoformat()
    unit = metadata["price_unit"]
    if qm["config"]["price_unit"] != unit:
        raise ValueError("Quote and fit price units differ")
    tolerances = [metadata[name+"_tolerance"] for name in ("price", "iv")]
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0
           for v in tolerances):
        raise ValueError("Finite nonnegative interval tolerances are required")
    policy = dict(config=qm["config"], price_basis=qm.get("price_basis"))
    tracks, result = {}, []

    def add(instrument, diagnostic, state, severity, ids, reasons, details, *, owner, units,
            rule=None):
        track = track_identity(underlying_id=underlying_id, model_id=owner, expiry=expiry,
            instrument_id=instrument, diagnostic=diagnostic, severity_unit=units,
            rule_id=_digest(dict(policy=policy, rule=rule)))
        tracks[track["track_id"]] = track
        result.append(dict(track_id=track["track_id"], snapshot_id=snapshot_id,
            requested_time=requested.isoformat(), source_time=source, state=state,
            severity=severity, source_ids=list(ids),
            reasons=sorted(set(reasons)), details=details))

    indexed = quotes.set_index("source_row_id")
    model = report.table("quotes")
    if model.quote_id.duplicated().any() or not set(model.quote_id).issubset(indexed.index):
        raise ValueError("Unmatched or duplicate model quote identity")
    model = model.set_index("quote_id")
    snapshot_status = quality.table("summary").iloc[0].snapshot_age_status
    check_groups = dict(tuple(checks.groupby("source_row_id", sort=False)))
    empty_checks = checks.iloc[:0]
    for q in quotes.to_dict("records"):
        local = check_groups.get(q["source_row_id"], empty_checks)
        if local.check.duplicated().any():
            raise ValueError("Ambiguous quote checks")
        failed = local[local.status.eq("fail")]
        missing = REQUIRED_QUALITY - set(local.loc[local.status.isin(["pass", "fail"]), "check"])
        reasons = failed.reason.tolist() + ["unchecked:" + name for name in sorted(missing)]
        n_fail = len(failed) + int(snapshot_status == "fail")
        if snapshot_status != "pass":
            reasons.append("snapshot_age:" + snapshot_status)
        state = "violation" if n_fail else ("unavailable" if missing or snapshot_status != "pass"
                                            else "clear")
        details = {name: q.get(name) for name in ("strike", "optiontype", "log_moneyness",
                                                  "exchange_time", "quote_age_seconds")}
        add(q["contract"], "quote_quality", state, n_fail if state != "unavailable" else None,
            [q["source_row_id"]], reasons, details, owner="oca", units="failed_checks")
        for kind, units in (("price", unit), ("iv", "decimal IV")):
            state, severity, reasons = "unavailable", None, ["not_selected_for_saved_fit"]
            if q["source_row_id"] in model.index:
                row = model.loc[q["source_row_id"]]
                if row.strike != q["strike"] or row.optiontype != q["optiontype"]:
                    raise ValueError("Model and OCA contract observations differ")
                fields = dict(forward="forward_price", discount="discount", ttm="ttm",
                              bid_price="bid_price", ask_price="ask_price",
                              bid_iv="bid_iv", ask_iv="ask_iv")
                for left, right in fields.items():
                    a, b = row[left], q[right]
                    if not (pd.isna(a) and pd.isna(b)) and a != b:
                        raise ValueError("Model and OCA quote values differ")
                values = [row["model_" + ("price" if kind == "price" else "iv")],
                          q["bid_"+kind], q["ask_"+kind]]
                valid = all(v is not None and np.isfinite(v) for v in values)
                if valid and values[1] <= values[2] and min(values) >= 0 and (
                        kind == "price" or min(values) > 0):
                    distance = max(values[1]-values[0], values[0]-values[2], 0.)
                    tolerance = metadata[kind+"_tolerance"]
                    state = "violation" if distance > tolerance else "clear"
                    severity, reasons = (distance if state == "violation" else 0.), []
                else:
                    reasons = ["invalid_"+kind+"_evidence"]
            add(q["contract"], "observed_"+kind, state, severity, [q["source_row_id"]],
                reasons, details, owner=model_id, units=units,
                rule=dict(interval_tolerance=metadata[kind+"_tolerance"]))
    if metadata.get("oca_parity") is not None:
        parity = _restore(metadata["oca_parity"])
        pm = parity.metadata
        if (pm["config"] != qm["config"] or _time(pm["requested_time"]) != requested
                or pm.get("source_time") != source):
            raise ValueError("Parity and quote source provenance differ")
        for strike, group in parity.table("pairs").groupby("strike", sort=True):
            ids = sorted({sid for pair in group.source_row_ids for sid in pair})
            if not set(ids).issubset(indexed.index):
                raise ValueError("Unmatched parity source identity")
            state, severity = "unavailable", None
            reasons = ["ambiguous_or_unavailable_parity"]
            if len(group) == 1:
                pair = group.iloc[0]
                reasons = [pair.reason] if pair.reason else []
                if pair.status in ("pass", "fail"):
                    legs = indexed.loc[ids]
                    if (len(ids) != 2 or set(legs.optiontype) != {"C", "P"}
                            or legs.strike.ne(strike).any() or legs.expiry.nunique() != 1
                            or legs.exchange_time.nunique() != 1):
                        raise ValueError("Parity legs are not the same source pair")
                    if pair.status == "pass":
                        state, severity = "clear", 0.
                    elif pair.distance_outside is not None and np.isfinite(pair.distance_outside):
                        if pair.distance_outside > 0:
                            state, severity = "violation", float(pair.distance_outside)
            add(f"strike:{strike:.17g}", "parity", state, severity, ids, reasons,
                dict(strike=float(strike)), owner="oca", units=unit,
                rule={k: v for k, v in pm.items() if k not in ("source_time", "requested_time")})
    return list(tracks.values()), result


def synthetic_inputs():
    """A frozen seven-day example with a gap, recovery and a split reset."""
    snapshots = [dict(snapshot_id=f"day_{i}", requested_time=t.isoformat(),
        split="calibration" if i < 5 else "validation")
        for i, t in enumerate(pd.date_range("2024-04-17", periods=7, tz="UTC"))]
    track = track_identity(underlying_id="synthetic", model_id="saved_model",
        expiry="2024-06-01T00:00:00Z", instrument_id="synthetic-call-100",
        diagnostic="observed_price", severity_unit="USD per unit")
    track["expected_snapshot_ids"] = [s["snapshot_id"] for s in snapshots]
    observations = [dict(track_id=track["track_id"], snapshot_id=s["snapshot_id"],
        requested_time=s["requested_time"], source_time=s["requested_time"],
        source_ids=["source_"+str(i)], state=state,
        severity=1. if state == "violation" else 0. if state == "clear" else None,
        reasons=["quote_missing"] if state == "unavailable" else [])
        for i, (s, state) in enumerate(zip(snapshots,
            ("violation", "violation", "unavailable", "clear", "violation", "clear", "violation")))]
    return dict(schema_version=1, panel_id="synthetic_replay_v1",
        split_utc="2024-04-22T00:00:00Z", max_gap_seconds=86400,
        snapshots=snapshots, tracks=[track]), observations


class Locals(Enum):
    """Offline synthetic replay and caller-frozen saved observations."""

    SYNTHETIC = 1
    SAVED = 2


def run_local(local: Locals = Locals.SYNTHETIC):
    """Write immutable observations, episodes and cohort summaries to a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.manifest:
        local = Locals.SAVED
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        raw = (args.manifest.parent / manifest["observations_file"]).read_bytes()
        if sha256(raw).hexdigest() != manifest["observations_sha256"]:
            raise ValueError("Frozen observations digest mismatch")
        observations = json.loads(raw)
    elif local == Locals.SYNTHETIC:
        manifest, observations = synthetic_inputs()
    else:
        parser.error("Saved replay requires --manifest")
    report = replay_diagnostics(manifest, observations)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(local_path.get_output_path()) / "longitudinal_replay" / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report.to_dict(), indent=2, allow_nan=False),
                                       encoding="utf-8")
    for name in ("observations", "episodes", "tracks", "summary"):
        report.table(name).to_csv(output / (name+".csv"), index=False)
    print(report.table("summary").to_string(index=False), output)


if __name__ == "__main__":
    run_local()
