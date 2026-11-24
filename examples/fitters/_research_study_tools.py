"""Read-only provenance and statistical helpers for frozen research studies.

Extracted from the October 2026 agent studies ``svi_128_cutoff_stats.py``,
``svi_repeatability_stats.py`` and their saved-output auditors. These helpers
replace machine-specific artifact paths and hardcoded cohort/cutoff choices.
They are repository-only: they neither fit models nor install warning policies.

Tolerance limits use the first order-statistic rank r satisfying
``P[Binomial(n, coverage) <= r - 1] >= 1 - alpha``. Coverage and error probability
are explicit caller inputs. The caller must supply an independent calibration
cohort and account for multiple comparisons; this module cannot infer either.
Missing scores are retained as unavailable rather than silently discarded.

File hashes establish correspondence with a supplied manifest, not authenticity
of that manifest or validity of a study's conclusions. Saved wing-score checks
reconstruct the statistic from all refit predictions; they do not certify the
optimizer, pricing accuracy, noise assumptions or independence of refits.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class ToleranceEstimate:
    """An observed order-statistic limit or an explicit unavailable result."""

    status: Literal["estimated", "unavailable_scores", "insufficient_sample"]
    sample_size: int
    missing: int
    rank: int | None
    confidence: float | None
    cutoff: float | None
    selected_id: str | None


def tolerance_rank(sample_size: int, *, coverage: float, alpha: float
                   ) -> tuple[int | None, float | None]:
    """Find the minimum exact binomial tolerance rank without interpolation.

    Return ``(None, None)`` when even the sample maximum cannot attain the
    requested confidence. ``alpha`` is the one-sided error probability. Compare
    the survival probability directly: subtracting a tiny alpha from one can
    round away the distinction between feasible and infeasible confidence.
    """
    if (isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer))
            or sample_size < 0 or isinstance(coverage, bool) or isinstance(alpha, bool)
            or not np.isfinite(coverage) or not np.isfinite(alpha)
            or not 0 < coverage < 1 or not 0 < alpha < 1):
        raise ValueError("Require nonnegative sample size and coverage/alpha in (0, 1)")
    for rank in range(1, sample_size + 1):
        if binom.sf(rank - 1, sample_size, coverage) <= alpha:
            return rank, float(binom.cdf(rank - 1, sample_size, coverage))
    return None, None


def estimate_tolerance_cutoff(scores: Mapping[str, float | None], *, coverage: float,
                              alpha: float) -> ToleranceEstimate:
    """Select a finite nonnegative score by rank, resolving ties by dataset ID.

    Each key denotes one independent calibration dataset, not a refit or repeated
    batch of that dataset. ``None`` and nonfinite scores make the whole estimate
    unavailable; the original denominator is preserved. No default cutoff,
    coverage level or number of refits is supplied.
    """
    rank, confidence = tolerance_rank(len(scores), coverage=coverage, alpha=alpha)
    finite: list[tuple[float, str]] = []
    missing = 0
    for identity, score in scores.items():
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Every dataset requires a nonempty string identity")
        if score is None:
            missing += 1
        elif isinstance(score, bool) or not isinstance(score, (int, float, np.number)):
            raise ValueError("Scores must be real numbers or None")
        elif not np.isrealobj(score):
            raise ValueError("Scores must be real numbers or None")
        elif not np.isfinite(score):
            missing += 1
        elif score < 0:
            raise ValueError("Scores must be nonnegative")
        else:
            finite.append((float(score), identity))
    if missing or rank is None:
        status = "unavailable_scores" if missing else "insufficient_sample"
        return ToleranceEstimate(status, len(scores), missing, rank, confidence, None, None)
    cutoff, selected = sorted(finite)[rank - 1]
    return ToleranceEstimate("estimated", len(scores), 0, rank, confidence, cutoff, selected)


def verify_artifact_hashes(root: Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Verify all supplied relative POSIX paths against lowercase SHA256 digests.

    Reject empty manifests, missing files, path escapes (including symlinks),
    absolute Windows paths, alternate data streams and malformed digests. Read
    only the listed files; do not create or rewrite files or inspect unlisted
    artifacts. Return verified digests, or raise ``ValueError`` on a mismatch.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or not expected:
        raise ValueError("Require an artifact directory and a nonempty manifest")
    verified: dict[str, str] = {}
    for name, digest in expected.items():
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
                or ".." in PurePosixPath(name).parts or PurePosixPath(name).as_posix() != name):
            raise ValueError(f"Artifact path must be a contained relative POSIX path: {name!r}")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Invalid SHA256 digest for {name}")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Artifact is missing or outside the supplied root: {name}")
        with path.open("rb") as stream:
            actual = file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHA256 mismatch for {name}")
        verified[name] = actual
    return verified


def verify_saved_wing_score(report: Mapping[str, Any]) -> float | None:
    """Audit one saved wing-reliability slice using all requested refit draws.

    The wing mask is reconstructed from the training strikes and forward rather
    than trusting saved region labels. The saved combined-wing score must match
    RMS displacement from the unperturbed IV curve divided by the shock scale.
    Incomplete reports may carry only an unavailable score. Malformed complete
    reports and changed scores raise ``ValueError``; no refits are performed.
    """
    try:
        slices = report["slices"]
        if len(slices) > 1 or (report["status"] == "complete" and not slices):
            raise ValueError("Saved-score audit requires exactly one completed slice")
        if not slices:
            return None
        saved = slices[0]["regions"]["combined_wings"]["max_rms_amplification"]
        if report["status"] != "complete":
            if saved is not None:
                raise ValueError("Incomplete reports cannot supply a wing score")
            return None
        count = report["requested"]
        attempts = report["attempts"]
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2
                or len(attempts) != count or report["successful"] != count or report["failed"] != 0
                or any(a["status"] != "completed" or a["index"] != i
                       for i, a in enumerate(attempts))):
            raise ValueError("Every requested refit must be present, ordered and completed")
        grid = slices[0]["grid"]
        k, base = (np.asarray(grid[name], dtype=float) for name in ("k", "base_iv"))
        training = report["training"]
        if len(training["forwards"]) != 1 or len(training["strikes_ttms"]) != 1:
            raise ValueError("Saved-score audit requires one training slice")
        forward = float(training["forwards"][0])
        strikes = np.asarray(training["strikes_ttms"][0], dtype=float)
        shock = report["metadata"]["shock_std"]
        vols = np.asarray([a["ivs"][0] for a in attempts], dtype=float)
        if (k.ndim != 1 or len(k) < 2 or base.shape != k.shape
                or vols.shape != (count, len(k)) or not np.all(np.isfinite(k))
                or not np.all(np.diff(k) > 0) or not np.all(np.isfinite(base) & (base > 0))
                or not np.all(np.isfinite(vols) & (vols > 0))
                or strikes.ndim != 1 or len(strikes) < 2
                or not np.all(np.isfinite(strikes) & (strikes > 0))
                or not np.isfinite(forward) or forward <= 0 or isinstance(shock, bool)
                or not np.isfinite(shock) or shock <= 0):
            raise ValueError("Invalid saved prediction arrays, training inputs or shock scale")
        observed = np.log(strikes / forward)
        wings = (k < np.min(observed) - 1e-12) | (k > np.max(observed) + 1e-12)
        score = (float(np.max(np.sqrt(np.sum((vols - base)**2, axis=0) / count)[wings])
                       / shock) if np.any(wings) else None)
        if score is None:
            if saved is not None:
                raise ValueError("A grid without wings cannot supply a wing score")
        elif (saved is None or isinstance(saved, bool) or not np.isfinite(saved)
              or not np.isclose(saved, score, rtol=1e-10, atol=1e-12)):
            raise ValueError("Saved wing score differs from all-draw RMS reconstruction")
        return score
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed saved wing-reliability report") from error
