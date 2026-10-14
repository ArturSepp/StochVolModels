"""Caller-supplied finite-scope wing cutoffs; repository-only research reporting."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


FORMAT = "svm-wing-research-policy-v1"
SCOPE = {
    "model": "raw_svi_constrained", "draws": 128, "grid_points": 81,
    "ttms": [0.1, 0.75], "half_widths": [0.2, 0.4], "training_points": 31,
    "iv_half_spread": 0.005, "shock_stds": [0.00025, 0.001],
    "noise_assumption": "known_iid_gaussian_iv",
}
NUMERICAL_FILES = ("_svi_benchmark.py", "_svi_wing_benchmark.py", "compare_smile_fitters.py")


def numerical_hashes():
    """Identify the fitted model, optimizer and prediction code used by the policy."""
    directory = Path(__file__).resolve().parent
    return {name: sha256((directory / name).read_bytes()).hexdigest()
            for name in NUMERICAL_FILES}


def load_policy(path):
    """Read a caller-frozen JSON artifact; reject missing/ambiguous scope and cutoffs.

    No cutoff is installed with the example. A valid artifact has ``format``,
    ``policy_id``, ``scope`` equal to SCOPE, two ``cutoffs`` keyed by half-width,
    ``numerical_sha256`` and a ``calibration_sha256`` identifying source evidence.
    This checks its contract, not the truth of its calibration claims.
    """
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    validate_policy(value)
    return value, sha256(raw).hexdigest()


def validate_policy(policy):
    """Reject malformed policy artifacts before running expensive refits."""
    if not isinstance(policy, dict) or policy.get("format") != FORMAT:
        raise ValueError("Unsupported wing research policy format")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"].strip():
        raise ValueError("Research policy requires an identity")
    if policy.get("scope") != SCOPE:
        raise ValueError("Research policy scope differs from the implemented broad design")
    cutoffs = policy.get("cutoffs")
    if not isinstance(cutoffs, dict) or set(cutoffs) != {"0.2", "0.4"}:
        raise ValueError("Both exact half-width cutoffs are required")
    for cutoff in cutoffs.values():
        if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                or not np.isfinite(cutoff) or cutoff < 0):
            raise ValueError("Cutoffs must be finite and nonnegative")
    hashes = policy.get("numerical_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(NUMERICAL_FILES):
        raise ValueError("Numerical source hashes are required")
    for digest in [*hashes.values(), policy.get("calibration_sha256")]:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Policy provenance requires SHA256 digests")


def assess_research_policy(report, policy, *, noise_assumption=None, quality_reasons=()):
    """Return a scoped cutoff exceedance or explicit unavailable classification.

    ``noise_assumption`` is a caller declaration, never inferred from spreads.
    Nonempty upstream quote/parity reasons suppress classification. Coverage
    remains conditional on the calibration population, even for eligible inputs.
    An assessment does not change the fitted curve or emit automatic warnings.
    """
    validate_policy(policy)
    result = dict(status="unavailable", reasons=[], score=None, cutoff=None,
                  policy_id=policy["policy_id"], automatic_warning=False,
                  calibration_sha256=policy["calibration_sha256"],
                  interpretation="research cutoff exceedance; not an accuracy certificate")
    reasons = result["reasons"]
    if policy["numerical_sha256"] != numerical_hashes():
        reasons.append("numerical_source_mismatch")
    if noise_assumption != SCOPE["noise_assumption"]:
        reasons.append("noise_assumption_outside_calibration")
    if quality_reasons:
        reasons.append("upstream_quality_unavailable")
        result["quality_reasons"] = list(quality_reasons)
    try:
        metadata = report["metadata"]
        if metadata["model"] != SCOPE["model"]:
            reasons.append("model_outside_calibration")
        if report["requested"] != 128 or metadata["grid_points"] != 81:
            reasons.append("draws_or_grid_outside_calibration")
        if metadata["shock_std"] not in SCOPE["shock_stds"]:
            reasons.append("shock_scale_outside_calibration")
        training = report["training"]
        if len(training["ttms"]) != 1 or len(report["slices"]) != 1:
            reasons.append("require_one_calibrated_slice")
            return result
        t = float(training["ttms"][0])
        if not any(abs(t - time) <= 1e-12 for time in SCOPE["ttms"]):
            reasons.append("tenor_outside_calibration")
        forward = float(training["forwards"][0])
        k = np.log(np.asarray(training["strikes_ttms"][0], dtype=float) / forward)
        width = next((w for w in SCOPE["half_widths"]
                      if k.shape == (31,) and np.allclose(k, np.linspace(-w, w, 31),
                                                       rtol=0, atol=1e-12)), None)
        if width is None:
            reasons.append("geometry_outside_calibration")
        bid, ask = (np.asarray(training[name][0], dtype=float)
                    for name in ("bid_ivs", "ask_ivs"))
        if (bid.shape != k.shape or ask.shape != k.shape
                or not np.all(np.isfinite(bid) & (bid > 0))
                or not np.all(np.isfinite(ask) & (ask >= bid))
                or not np.allclose(ask - bid, .01, rtol=0, atol=1e-12)):
            reasons.append("quotes_outside_calibration")
        attempts = report["attempts"]
        if (report["status"] != "complete" or report["successful"] != 128
                or report["failed"] != 0 or len(attempts) != 128 or "error" in report
                or any(a["status"] != "completed" or a["index"] != i
                       for i, a in enumerate(attempts))):
            reasons.append("incomplete_diagnostic")
        grid = report["slices"][0]["grid"]
        points = np.asarray(grid["k"], dtype=float)
        base = np.asarray(grid["base_iv"], dtype=float)
        if width is not None and (points.shape != (81,)
                or not np.allclose(points, np.linspace(-2 * width, 2 * width, 81),
                                   rtol=0, atol=1e-12)):
            reasons.append("prediction_grid_outside_calibration")
        vols = np.asarray([a["ivs"][0] for a in attempts], dtype=float)
        if (base.shape != (81,) or vols.shape != (128, 81)
                or not np.all(np.isfinite(base) & (base > 0))
                or not np.all(np.isfinite(vols) & (vols > 0))):
            reasons.append("nonfinite_or_missing_predictions")
        if reasons:
            return result
        wings = (points < k[0] - 1e-12) | (points > k[-1] + 1e-12)
        score = float(np.max(np.sqrt(np.mean((vols - base)**2, axis=0))[wings])
                      / metadata["shock_std"])
        cutoff = float(policy["cutoffs"][str(width)])
        result.update(status="flag" if score > cutoff else "clear", score=score,
                      cutoff=cutoff, half_width=width)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        reasons.append("malformed_or_missing_diagnostic")
    return result
