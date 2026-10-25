"""Independent synthetic-law, exact-statistics and saved-evidence regression checks.

These repository-only tests exercise the maintained extraction of the October
2026 research helpers. They require neither private market data nor historical
output folders and perform no model calibration or policy installation.
"""

from collections.abc import Generator
from copy import deepcopy
from hashlib import sha256
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy.special import comb


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def research() -> Generator[ModuleType, None, None]:
    """Import the repository-only helpers without relying on the working directory."""
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/_research_study_tools.py").is_file()), None)
    if root is None:
        pytest.skip("research study tools are repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("_research_study_tools")


@pytest.fixture(scope="module")
def challenges(research: ModuleType) -> ModuleType:
    """Load the fixed panel in the same temporary example import path."""
    return importlib.import_module("_svi_research_fixtures")


def test_panel_preserves_unique_observations_and_hidden_truth_pairing(
        challenges: ModuleType) -> None:
    """Hidden truths must not create additional observations or change training quotes."""
    panel = challenges.synthetic_challenge_panel()
    assert len(panel) == 60
    assert len({case.case_id for case in panel}) == 60
    assert len({case.observations_id for case in panel}) == 48
    assert len(challenges.synthetic_challenge_panel(include_hidden=False)) == 48
    controls = {case.case_id: case for case in panel if not case.hidden_wing}
    for case in panel:
        assert np.all(case.bid_iv > 0) and np.all(case.ask_iv > case.bid_iv)
        assert case.k.shape == (31,) and case.evaluation_k.shape == (81,)
        with pytest.raises(ValueError):
            case.bid_iv[0] = 0
        if not case.hidden_wing:
            continue
        control = controls[case.observations_id]
        np.testing.assert_array_equal(case.bid_iv, control.bid_iv)
        np.testing.assert_array_equal(case.ask_iv, control.ask_iv)
        inside = np.abs(case.evaluation_k) <= case.half_width
        np.testing.assert_array_equal(case.evaluation_truth_iv[inside],
                                      control.evaluation_truth_iv[inside])
        np.testing.assert_allclose(case.evaluation_truth_iv[[0, -1]]
                                   - control.evaluation_truth_iv[[0, -1]], .03)
    second = challenges.synthetic_challenge_panel()
    for first, repeated in zip(panel, second):
        np.testing.assert_array_equal(first.bid_iv, repeated.bid_iv)
        np.testing.assert_array_equal(first.evaluation_truth_iv, repeated.evaluation_truth_iv)


def test_noise_laws_against_independent_matrix_and_curve_references(
        challenges: ModuleType) -> None:
    """Use an eigenvalue matrix square root rather than the generator's scalar formula."""
    cases = {case.regime: case for case in challenges.synthetic_challenge_panel()
             if case.case_id.startswith("g00_") and not case.hidden_wing}
    control = cases["gaussian_control"]
    k = control.k
    # Historical first profile, independently expressed in annualized IV units.
    truth = np.sqrt((.004 + .018 * (-.4 * (k - .01) + np.hypot(k - .01, .08))) / .25)
    driver = np.random.default_rng(15000000).normal(0, .001, len(k))
    np.testing.assert_allclose((control.bid_iv + control.ask_iv) / 2, truth + driver,
                               rtol=0, atol=1e-15)
    uniform = cases["spread_uniform"]
    half_spread = (uniform.ask_iv - uniform.bid_iv) / 2
    np.testing.assert_allclose(half_spread, .005 * (1 + 2 * (k / .2)**2), atol=1e-15)
    assert np.all(np.abs((uniform.bid_iv + uniform.ask_iv) / 2 - truth) <= half_spread)
    correlation = .3 * np.eye(len(k)) + .7 * np.ones((len(k), len(k)))
    eigenvalues, vectors = np.linalg.eigh(correlation)
    symmetric_root = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    reference = half_spread / np.sqrt(3) * (symmetric_root @ (driver / .001))
    correlated = cases["spread_correlated"]
    np.testing.assert_allclose((correlated.bid_iv + correlated.ask_iv) / 2 - truth,
                               reference, rtol=1e-12, atol=1e-15)
    bump = cases["localized_bump"]
    np.testing.assert_allclose((bump.bid_iv + bump.ask_iv - control.bid_iv - control.ask_iv) / 2,
                               .03 * np.exp(-.5 * ((k - .12) / .024)**2), atol=1e-15)


@pytest.mark.parametrize("n,coverage,alpha", [(0, .9, .05), (3, .9, .05),
                                             (8, .5, .1), (20, .8, .1), (64, .9, .05 / 3)])
def test_tolerance_rank_matches_exact_enumeration(
        research: ModuleType, n: int, coverage: float, alpha: float) -> None:
    """Enumerate exact integer binomial coefficients rather than using a CDF routine."""
    probabilities = [comb(n, j, exact=True) * coverage**j * (1 - coverage)**(n - j)
                     for j in range(n + 1)]
    expected = next((rank for rank in range(1, n + 1)
                     if sum(probabilities[:rank]) >= 1 - alpha), None)
    rank, confidence = research.tolerance_rank(n, coverage=coverage, alpha=alpha)
    assert rank == expected
    if rank is None:
        assert confidence is None
    else:
        assert confidence == pytest.approx(sum(probabilities[:rank]), abs=2e-14)


def test_tolerance_rank_compares_small_tail_probabilities_without_subtraction(
        research: ModuleType) -> None:
    """Rounded unit CDF values must not admit an unattainable confidence level."""
    assert .5**55 > 1e-18 and .5**60 < 1e-18
    assert research.tolerance_rank(55, coverage=.5, alpha=1e-18) == (None, None)
    assert research.tolerance_rank(60, coverage=.5, alpha=1e-18)[0] == 60


def test_cutoff_is_an_observed_order_statistic_and_retains_missing_denominator(
        research: ModuleType) -> None:
    """A missing member must not silently reduce the sample or allow interpolation."""
    scores = {f"case_{i:02d}": float(i) for i in range(20)}
    estimate = research.estimate_tolerance_cutoff(scores, coverage=.8, alpha=.1)
    assert estimate.status == "estimated" and estimate.rank == 19
    assert estimate.cutoff == 18 and estimate.selected_id == "case_18"
    scores["case_00"] = None
    scores["case_01"] = np.nan
    scores["case_02"] = np.inf
    missing = research.estimate_tolerance_cutoff(scores, coverage=.8, alpha=.1)
    assert missing.status == "unavailable_scores" and missing.sample_size == 20
    assert missing.missing == 3 and missing.cutoff is None and missing.rank == estimate.rank
    empty = research.estimate_tolerance_cutoff({}, coverage=.8, alpha=.1)
    assert empty.status == "insufficient_sample" and empty.cutoff is None
    ties = research.estimate_tolerance_cutoff({f"case_{i:02d}": 1. for i in range(20)},
                                               coverage=.8, alpha=.1)
    assert ties.selected_id == "case_18" and ties.cutoff == 1.


@pytest.mark.parametrize("scores", [{"": 1.}, {"a": -1.}, {"a": True}, {"a": 1j}])
def test_invalid_calibration_inputs_are_not_accepted(
        research: ModuleType, scores: dict[str, Any]) -> None:
    """Invalid identities and values cannot masquerade as an unavailable estimate."""
    with pytest.raises(ValueError):
        research.estimate_tolerance_cutoff(scores, coverage=.8, alpha=.1)


@pytest.mark.parametrize("kwargs", [dict(sample_size=-1, coverage=.9, alpha=.05),
                                   dict(sample_size=True, coverage=.9, alpha=.05),
                                   dict(sample_size=20, coverage=1., alpha=.05),
                                   dict(sample_size=20, coverage=.9, alpha=np.nan)])
def test_invalid_tolerance_design_is_rejected(research: ModuleType,
                                              kwargs: dict[str, Any]) -> None:
    """Require a valid sample count and explicit confidence design."""
    with pytest.raises(ValueError):
        research.tolerance_rank(**kwargs)


def test_artifact_hashes_detect_tampering_missing_files_and_preserve_inputs(
        research: ModuleType, tmp_path: Path) -> None:
    """Hash actual bytes and fail after an artifact changes or disappears."""
    folder = tmp_path / "reports"
    folder.mkdir()
    path = folder / "one.json"
    original = b'{"score": 2.0}\n'
    path.write_bytes(original)
    expected = {"reports/one.json": sha256(original).hexdigest()}
    assert research.verify_artifact_hashes(tmp_path, expected) == expected
    assert path.read_bytes() == original
    path.write_bytes(b'{"score": 3.0}\n')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        research.verify_artifact_hashes(tmp_path, expected)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        research.verify_artifact_hashes(tmp_path, expected)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "C:/outside.json",
                                  "reports/../../outside.json", "one.json:stream",
                                  "reports\\one.json", "./one.json"])
def test_manifest_paths_cannot_escape_the_artifact_root(
        research: ModuleType, tmp_path: Path, name: str) -> None:
    """Reject platform-specific path escapes before attempting to read an artifact."""
    with pytest.raises(ValueError, match="relative POSIX"):
        research.verify_artifact_hashes(tmp_path, {name: "0" * 64})
    with pytest.raises(ValueError, match="nonempty"):
        research.verify_artifact_hashes(tmp_path, {})
    with pytest.raises(ValueError, match="Invalid SHA256"):
        research.verify_artifact_hashes(tmp_path, {"one.json": "invalid"})


@pytest.fixture
def saved_report() -> dict[str, Any]:
    """Provide hand-computable refit displacements and deliberately false region labels."""
    base = np.array([.2, .25, .3])
    return dict(status="complete", requested=2, successful=2, failed=0,
                training=dict(forwards=[100.], strikes_ttms=[[100 * np.exp(-.1),
                                                            100 * np.exp(.1)]]),
                metadata=dict(shock_std=.01),
                slices=[dict(grid=dict(k=[-.2, 0., .2], base_iv=base,
                                       region=["training_range"] * 3),
                             regions=dict(combined_wings=dict(
                                 max_rms_amplification=np.sqrt(10))))],
                attempts=[dict(index=0, status="completed", ivs=[base + [.01, .1, .02]]),
                          dict(index=1, status="completed", ivs=[base + [.03, .1, .04]])])


def test_saved_wing_score_uses_all_draws_and_actual_training_support(
        research: ModuleType, saved_report: dict[str, Any]) -> None:
    """Exclude the larger central move, ignore stale region labels and retain the last draw."""
    before = deepcopy(saved_report)
    assert research.verify_saved_wing_score(saved_report) == pytest.approx(np.sqrt(10))
    np.testing.assert_array_equal(saved_report["attempts"][1]["ivs"][0],
                                  before["attempts"][1]["ivs"][0])
    saved_report["attempts"][-1]["ivs"][0][-1] += .1
    with pytest.raises(ValueError, match="all-draw RMS"):
        research.verify_saved_wing_score(saved_report)


@pytest.mark.parametrize("defect", ["score", "missing_draw", "failed_draw", "duplicate_index",
                                    "nan", "shock", "training", "empty_slice"])
def test_saved_score_audit_rejects_corrupt_complete_reports(
        research: ModuleType, saved_report: dict[str, Any], defect: str) -> None:
    """Each mutation invalidates a necessary independent input to the audit."""
    if defect == "score":
        saved_report["slices"][0]["regions"]["combined_wings"]["max_rms_amplification"] = 0.
    elif defect == "missing_draw":
        saved_report["attempts"].pop()
    elif defect == "failed_draw":
        saved_report["attempts"][1]["status"] = "failed"
    elif defect == "duplicate_index":
        saved_report["attempts"][1]["index"] = 0
    elif defect == "nan":
        saved_report["attempts"][1]["ivs"][0][0] = np.nan
    elif defect == "shock":
        saved_report["metadata"]["shock_std"] = 0
    elif defect == "training":
        saved_report["training"]["strikes_ttms"] = [[-1, 100]]
    else:
        saved_report["slices"] = []
    with pytest.raises(ValueError):
        research.verify_saved_wing_score(saved_report)


def test_incomplete_runs_abstain_and_cannot_retain_a_claimed_score(
        research: ModuleType, saved_report: dict[str, Any]) -> None:
    """Missing evidence remains unavailable instead of appearing to pass a cutoff."""
    saved_report["status"] = "incomplete"
    with pytest.raises(ValueError, match="Incomplete"):
        research.verify_saved_wing_score(saved_report)
    saved_report["slices"][0]["regions"]["combined_wings"]["max_rms_amplification"] = None
    assert research.verify_saved_wing_score(saved_report) is None
    assert research.verify_saved_wing_score({"status": "failed", "slices": []}) is None
