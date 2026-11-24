"""Replay the paper's selected fits with the standalone rough-logSV implementation.

H=0.50 is repriced by RoughLogSVPricer. H=0.30 revalues every frozen path checkpoint
with the package's conditional estimator and interval reconstruction. This replay
does not generate paths, run an optimizer, or import numerical code from agents/.
Historical fit flags describe their original bounded experiments, not permission
to calibrate a new parameter domain.
"""

from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models import rough_logsv
from stochvolmodels.pricers.rough_logsv import (
    DiscretePdeControl, RoughLogSVPricer, RoughLogSvParams, RoughMCConfig,
)
from stochvolmodels.models.rough_logsv.history import raw_forward
from stochvolmodels.models.rough_logsv.multilevel import (
    numerical_checks, prices_from_states, summarize_replicates,
)
from stochvolmodels.pricers import rough_logsv_pricer


DEFAULT_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "agents"
REPLAY_TOLERANCE_SPREADS = 1e-7
CASES = {
    "BTC": ("2023-06-30T08:00:00Z", (7, 28, 91), 45),
    "ETH": ("2021-10-26T08:00:00Z", (10, 31, 66), 39),
}
QUOTE_KEYS = (
    "contract", "asset", "observation_time_utc", "expiry", "days_to_expiry",
    "strike", "optiontype", "common_forward", "bid_price", "ask_price",
    "ttm_act365", "common_bid_iv", "common_ask_iv", "x",
)


def _read(path, hashes):
    """Read an immutable input and retain the digest of the bytes consumed."""
    path = Path(path)
    content = path.read_bytes()
    hashes[str(path.resolve())] = hashlib.sha256(content).hexdigest()
    return content


def _json(path, hashes):
    return json.loads(_read(path, hashes).decode("utf-8-sig"))


def _csv(path, hashes):
    return pd.read_csv(BytesIO(_read(path, hashes)))


def _same_quotes(actual, expected):
    """Require exact quote identities, ordering and premium conventions."""
    if (len(actual) != len(expected) or not actual.contract.is_unique
            or not expected.contract.is_unique):
        raise ValueError("Saved fit must retain every selected quote exactly once")
    for key in QUOTE_KEYS:
        if not np.array_equal(actual[key].to_numpy(), expected[key].to_numpy()):
            raise ValueError(f"Saved fit quote binding differs: {key}")


def _coordinates(frame):
    days = np.sort(frame.days_to_expiry.unique()).astype(np.int64)
    return (days, np.searchsorted(days, frame.days_to_expiry),
            (frame.strike / frame.common_forward).to_numpy(),
            frame.optiontype.eq("C").to_numpy())


def _chain(frame):
    groups = list(frame.groupby("days_to_expiry", sort=True))
    return OptionChain(
        ttms=np.array([float(day) / 365 for day, _ in groups]),
        forwards=np.array([float(g.common_forward.iloc[0]) for _, g in groups]),
        discfactors=np.ones(len(groups)),
        ids=np.array([str(int(day)) for day, _ in groups]),
        strikes_ttms=[g.strike.to_numpy() for _, g in groups],
        optiontypes_ttms=[g.optiontype.to_numpy() for _, g in groups],
    )


def _confirmation(asset, params, root, hashes):
    """Select only an existing confirmation with the identical parameter vector."""
    candidates = (
        ["btc-h030-precision-20261118/final_confirmation"] if asset == "BTC" else [
            "calibration-robustness-20261118/ETH/H030/confirmation",
            "eth-fixed-h-20261118/confirmation",
        ]
    )
    for relative in candidates:
        folder = root / relative
        if not (folder / "config.json").is_file():
            continue
        config = _json(folder / "config.json", hashes)
        values = dict(zip(config["parameter_names"], config["parameters"], strict=True))
        if (config["asset"] == asset and config["H"] == params.H
                and set(values) == set(asdict(params)) - {"H"}
                and all(values[name] == getattr(params, name) for name in values)):
            return folder, config
    raise ValueError("No saved confirmation matches these parameters; wider calibration is gated")


def _control(asset, frame, home, config, hashes):
    """Bind the validated discrete mean to the original global quote indices."""
    panel = _csv(home / "inputs/core_option_panel.csv", hashes)
    selected = panel.loc[panel.asset.eq(asset)]
    _same_quotes(frame, selected)
    path = home / "inputs/control_means.csv"
    table = _csv(path, hashes)
    expected_hashes = {
        value for name, value in config.get("dependencies", {}).items()
        if name.replace("\\", "/").endswith("/inputs/control_means.csv")
    }
    if expected_hashes != {hashes[str(path.resolve())]}:
        raise ValueError("Control-mean evidence differs from the frozen confirmation")
    if not table.quote_id.is_unique:
        raise ValueError("Duplicate control-mean quote identifiers")
    table = table.set_index("quote_id").loc[selected.index]
    if not table.asset.eq(asset).all() or not table.passed.eq(True).all():
        raise ValueError("Saved discrete control mean is not qualified for these quotes")
    matrix = _csv(home / "inputs/qualification_matrix.csv", hashes)
    rows = matrix.loc[matrix.asset.eq(asset) & matrix.H.eq(.5) & matrix.model.eq("quadratic")]
    if len(rows) != 1:
        raise ValueError("One frozen classical control parameter vector required")
    params = RoughLogSvParams(**{
        key: float(rows.iloc[0][key]) for key in asdict(RoughLogSvParams())
    })
    days, index, strikes, calls = _coordinates(frame)
    return DiscretePdeControl(
        params, tuple(days), tuple(index), tuple(strikes), tuple(calls),
        tuple(table["mean"]), tuple(table.allowance),
        tuple(map(tuple, table[["space", "quadrature", "domain", "contour"]].to_numpy())),
    )


def _checkpoint_catalog(status, config):
    """Require exactly the frozen level/replicate allocation, without missing rows."""
    if status["status"] != "completed":
        raise ValueError("A completed independent confirmation is required")
    expected = {
        f"L{level}_R{rep}.npz": (level, rep, pairs)
        for level, pairs in enumerate(config.pairs + (config.raw_pairs,))
        for rep in range(config.replicates)
    }
    catalog = {row["file"]: row for row in status["completed"]}
    if len(catalog) != len(status["completed"]) or set(catalog) != set(expected):
        raise ValueError("Checkpoint coverage differs from the frozen allocation")
    for name, identity in expected.items():
        row = catalog[name]
        if (row["level"], row["rep"], row["pairs"]) != identity:
            raise ValueError("Checkpoint identity or sample count differs")
    return catalog


def _rough(asset, params, frame, root, hashes):
    folder, conf = _confirmation(asset, params, root, hashes)
    days, index, strikes, calls = _coordinates(frame)
    if (conf["days"] != days.tolist() or conf["quotes"] != len(frame)
            or pd.Timestamp(conf["date"]) != pd.Timestamp(CASES[asset][0])):
        raise ValueError("Confirmation belongs to a different selected snapshot")
    config = RoughMCConfig(
        grids=tuple(conf["grids"]), pairs=tuple(conf["pairs"]), replicates=conf["replicates"],
        raw_pairs=conf["raw_pairs"], chunk_pairs=conf["chunk_pairs"],
        entropy=tuple(conf["entropy"]), phase=conf["phase"],
    )
    home = root / ("btc-fixed-h-20261118" if asset == "BTC" else "eth-fixed-h-20261118")
    cv = _control(asset, frame, home, conf, hashes)
    cv.validate_for(days, index, strikes, calls, config)
    status = _json(folder / "status.json", hashes)
    catalog = _checkpoint_catalog(status, config)
    levels = np.empty((len(config.grids), config.replicates, len(frame)))
    smoothing = np.empty((config.replicates, len(frame)))
    forwards = np.empty((config.replicates, len(days)))
    for level, count in enumerate(config.pairs + (config.raw_pairs,)):
        for rep in range(config.replicates):
            name = f"L{level}_R{rep}.npz"
            content = _read(folder / name, hashes)
            if hashlib.sha256(content).hexdigest() != catalog[name]["sha256"]:
                raise ValueError(f"Checkpoint digest mismatch: {name}")
            shape = (4, len(days), 2 * count)
            with np.load(BytesIO(content), allow_pickle=False) as data:
                keys = (["ordinary"] if level == len(config.grids) else
                        ["fine", "coarse" if level else "control"] +
                        (["smoothing"] if level == len(config.grids) - 1 else []))
                if any(data[key].shape != shape or not np.isfinite(data[key]).all() for key in keys):
                    raise ValueError(f"Checkpoint states/counts invalid: {name}")
                if level == len(config.grids):
                    raw = raw_forward(data["ordinary"][0], data["ordinary"][1], params.rho)
                    np.testing.assert_allclose(raw, data["raw_forward"], rtol=2e-13, atol=1e-13)
                    forwards[rep] = raw.mean(axis=1)
                    continue
                fine = prices_from_states(data["fine"], params.rho, index, strikes, calls)
                other = prices_from_states(
                    data["coarse" if level else "control"],
                    params.rho if level else cv.params.rho, index, strikes, calls,
                )
                difference = fine - other
                if level == 0:
                    difference += np.asarray(cv.means)[:, None]
                levels[level, rep] = difference.mean(axis=1)
                if level == len(config.grids) - 1:
                    smoothing[rep] = (fine - prices_from_states(
                        data["smoothing"], params.rho, index, strikes, calls)).mean(axis=1)
    summary = summarize_replicates(levels, smoothing, forwards, cv.allowance)
    reference = _csv(folder / "forward_results.csv", hashes)
    if not np.array_equal(reference.days, days):
        raise ValueError("Raw-forward maturity ordering differs")
    for name, saved in (("raw_forward", "raw_mean"), ("raw_forward_halfwidth", "halfwidth")):
        np.testing.assert_allclose(summary[name], reference[saved], rtol=0, atol=2e-13)
    spread = (frame.ask_price - frame.bid_price).to_numpy()
    checks = numerical_checks(summary, spread)
    if not checks["pointwise_passed"] or checks["calibration_ready"]:
        raise ValueError("Selected-point checks must pass while wider calibration stays gated")
    return summary, {
        "method": "standalone_conditional_estimator_on_frozen_checkpoints",
        "confirmation": str(folder.resolve()), "checkpoints_verified": len(catalog),
        "price_passed": int(checks["price"].sum()), "time_passed": int(checks["time"].sum()),
        "smoothing_passed": int(checks["smoothing"].sum()),
        "raw_forward_passed": int(checks["raw_forward"].sum()),
        "raw_forward": summary["raw_forward"].tolist(),
        "raw_forward_halfwidth": summary["raw_forward_halfwidth"].tolist(),
        "control_mean_validated_before_replay": True,
    }


def replay_fit(asset, panel, fit_file, quote_file, evidence_root=None):
    """Return saved metadata, independently replayed prices and audit provenance.

    Files supplied as fit_file and quote_file are read-only. The returned table
    replaces prices/errors with reconstructed values and retains its original
    column order. CSV rounding is tolerated at 1e-7 of a full bid-ask spread.
    """
    if asset not in CASES:
        raise ValueError("Only the selected BTC/ETH cases are supported")
    hashes = {}
    fit, frame = _json(fit_file, hashes), _csv(quote_file, hashes)
    params = RoughLogSvParams(**fit["parameters"])
    if not fit.get("calibrated") or params.H not in (.3, .5):
        raise ValueError("Only saved qualified H=0.30/0.50 fits; wider calibration stays gated")
    expected = panel.loc[panel.asset.eq(asset)]
    date, days, count = CASES[asset]
    _same_quotes(frame, expected)
    if (len(frame) != count or tuple(sorted(frame.days_to_expiry.unique())) != days
            or not pd.to_datetime(frame.observation_time_utc, utc=True).eq(pd.Timestamp(date)).all()):
        raise ValueError("New dates or maturities are outside the selected-case replay")
    # Reuse the standalone chain validator even for replay, including ACT/365 and units.
    chain = _chain(frame)
    from stochvolmodels.pricers.rough_logsv_pricer import chain_coordinates
    chain_coordinates(chain)
    spread = (frame.ask_price - frame.bid_price).to_numpy()
    if not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError("Positive full bid-ask spreads required")
    if params.martingale_margin < 0:
        raise ValueError("The selected fits must satisfy kappa2 >= beta")
    if params.H == .5:
        if (not fit["qualification"].get("passed")
                or fit["qualification"]["parameters"] != fit["parameters"]):
            raise ValueError("Classical qualification does not bind the saved parameters")
        result = RoughLogSVPricer().price_chain_result(chain, params)
        summary = {"price": np.concatenate(result.prices) / frame.common_forward.to_numpy()}
        audit = {"method": "RoughLogSVPricer.nonlinear_classical_pde",
                 "saved_pde_qualification_passed": True}
    else:
        if not fit.get("independent_confirmation"):
            raise ValueError("Independent rough-price confirmation required")
        root = DEFAULT_EVIDENCE_ROOT if evidence_root is None else Path(evidence_root)
        summary, audit = _rough(asset, params, frame, root, hashes)
    errors = {}
    for name in ("price", "price_halfwidth", "time_change", "time_halfwidth",
                 "smoothing_change", "smoothing_halfwidth"):
        if name not in summary:
            continue
        error = float(np.max(np.abs(summary[name] - frame[name].to_numpy()) / spread))
        if not np.isfinite(error) or error > REPLAY_TOLERANCE_SPREADS:
            raise ValueError(f"Standalone {name} differs from the saved fit by {error} spreads")
        errors[name] = error
    inside = ((summary["price"] >= frame.bid_price) & (summary["price"] <= frame.ask_price))
    if not np.array_equal(inside, frame.inside_bid_ask):
        raise ValueError("Repricing changes the saved bid-ask inclusion counts")
    replayed = frame.copy()
    for name, values in summary.items():
        if name in replayed and len(values) == len(replayed):
            replayed[name] = values
    replayed["inside_bid_ask"] = inside
    replayed["error_spreads"] = (replayed.price - (replayed.bid_price + replayed.ask_price) / 2) / spread
    if params.H == .3:
        replayed["price_halfwidth_spreads"] = replayed.price_halfwidth / spread
        replayed["time_bound_spreads"] = (abs(replayed.time_change) + replayed.time_halfwidth) / spread
        replayed["smoothing_bound_spreads"] = (
            abs(replayed.smoothing_change) + replayed.smoothing_halfwidth) / spread
    model_dir = Path(rough_logsv.__file__).resolve().parent
    source_files = [Path(__file__), Path(rough_logsv_pricer.__file__)] + [
        model_dir / name
        for name in ("params.py", "history.py", "multilevel.py", "_classical_pde.py")
    ]
    for source in source_files:
        _read(source, hashes)
    audit.update(asset=asset, H=params.H, quotes=len(frame), numerical_differences_spreads=errors,
                 inside_bid_ask=int(inside.sum()), parameters=asdict(params),
                 fresh_sampling=False, optimizer_run=False, calibration_ready=False,
                 wider_calibration_gated=True, inputs_sha256=hashes)
    return fit, replayed, audit
