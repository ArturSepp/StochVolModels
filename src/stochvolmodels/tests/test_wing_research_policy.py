"""Scope abstention and independent all-draw RMS checks for research cutoffs."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def policy_module():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/_wing_research_policy.py").is_file()), None)
    if root is None:
        pytest.skip("wing research policy is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("_wing_research_policy")


@pytest.fixture
def policy(policy_module):
    return dict(format=policy_module.FORMAT, policy_id="test-frozen-policy",
                scope=deepcopy(policy_module.SCOPE), cutoffs={"0.2": 4.0, "0.4": 2.0},
                numerical_sha256=policy_module.numerical_hashes(), calibration_sha256="a" * 64)


@pytest.fixture
def evidence():
    k = np.linspace(-.4, .4, 81)
    base = np.full(81, .25)
    vols = [.25 + .0001 * ((i % 4) + 1) * (1 + k) for i in range(128)]
    return dict(status="complete", requested=128, successful=128, failed=0,
                metadata=dict(model="raw_svi_constrained", grid_points=81, shock_std=.001),
                training=dict(ttms=[.1], forwards=[100.],
                              strikes_ttms=[100 * np.exp(np.linspace(-.2, .2, 31))],
                              bid_ivs=[np.full(31, .245)], ask_ivs=[np.full(31, .255)]),
                slices=[dict(grid=dict(k=k, base_iv=base),
                             regions=dict(combined_wings=dict(max_rms_amplification=999.)))],
                attempts=[dict(index=i, status="completed", ivs=[v])
                          for i, v in enumerate(vols)])


def test_independent_rms_ignores_stale_summary_and_uses_every_draw(
        policy_module, policy, evidence):
    out = policy_module.assess_research_policy(
        evidence, policy, noise_assumption="known_iid_gaussian_iv")
    squares = [(.0001 * ((i % 4) + 1) * 1.4)**2 for i in range(128)]
    reference = (sum(squares) / 128)**.5 / .001
    assert out["score"] == pytest.approx(reference, abs=1e-13)
    assert out["status"] == "clear" and out["cutoff"] == 4.
    assert out["automatic_warning"] is False
    changed = deepcopy(evidence)
    changed["attempts"][-1]["ivs"][0][-1] += .2
    assert policy_module.assess_research_policy(
        changed, policy, noise_assumption="known_iid_gaussian_iv")["status"] == "flag"


def test_tie_is_clear(policy_module, policy, evidence):
    score = policy_module.assess_research_policy(
        evidence, policy, noise_assumption="known_iid_gaussian_iv")["score"]
    policy["cutoffs"]["0.2"] = score
    assert policy_module.assess_research_policy(
        evidence, policy, noise_assumption="known_iid_gaussian_iv")["status"] == "clear"


@pytest.mark.parametrize("defect", ["narrow", "unequal_grid", "tenor", "multiple",
                                  "draws", "grid", "shared", "shock", "spread",
                                  "failure", "failed_status", "missing", "nan", "source",
                                  "noise", "quality", "base_failure"])
def test_unsupported_or_incomplete_reports_abstain(policy_module, policy, evidence, defect):
    kwargs = dict(noise_assumption="known_iid_gaussian_iv")
    if defect == "narrow":
        evidence["training"]["strikes_ttms"] = [100 * np.exp(np.linspace(-.25, -.01, 31))]
    elif defect == "unequal_grid":
        evidence["training"]["strikes_ttms"][0][4] += .01
    elif defect == "tenor":
        evidence["training"]["ttms"] = [.25]
    elif defect == "multiple":
        evidence["training"]["ttms"] = [.1, .75]
    elif defect == "draws":
        evidence["requested"] = 64
    elif defect == "grid":
        evidence["slices"][0]["grid"]["k"] = np.linspace(-.41, .4, 81)
    elif defect == "shared":
        evidence["metadata"]["model"] = "raw_svi_shared_rho"
    elif defect == "shock":
        evidence["metadata"]["shock_std"] = .005 / np.sqrt(3)
    elif defect == "spread":
        evidence["training"]["ask_ivs"][0][4] += .001
    elif defect == "failure":
        evidence["attempts"][3]["status"] = "failed"
    elif defect == "failed_status":
        evidence["status"] = "incomplete"
    elif defect == "missing":
        del evidence["attempts"][2]["ivs"]
    elif defect == "nan":
        evidence["attempts"][9]["ivs"][0][1] = np.nan
    elif defect == "source":
        policy["numerical_sha256"]["_svi_wing_benchmark.py"] = "b" * 64
    elif defect == "noise":
        kwargs["noise_assumption"] = "spread_uniform"
    elif defect == "quality":
        kwargs["quality_reasons"] = ["stale_quote", "outside_parity_interval"]
    else:
        evidence["error"] = "base fit failure"
    out = policy_module.assess_research_policy(evidence, policy, **kwargs)
    assert out["status"] == "unavailable" and out["reasons"]
    assert out["score"] is None and out["automatic_warning"] is False


@pytest.mark.parametrize("defect", ["format", "scope", "missing_cutoff", "nan", "bool", "hash"])
def test_invalid_artifact_rejected(policy_module, policy, tmp_path, defect):
    if defect == "format":
        policy["format"] = "unknown"
    elif defect == "scope":
        policy["scope"]["draws"] = 64
    elif defect == "missing_cutoff":
        del policy["cutoffs"]["0.4"]
    elif defect in ("nan", "bool"):
        policy["cutoffs"]["0.2"] = np.nan if defect == "nan" else True
    else:
        policy["calibration_sha256"] = "not-a-hash"
    path = tmp_path / "cutoff.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        policy_module.load_policy(path)


@pytest.mark.parametrize("opt_in", [False, True])
def test_cli_archives_explicit_artifact_and_preserves_defaults(
        policy_module, policy, evidence, tmp_path, monkeypatch, opt_in):
    import sys
    diagnostic = importlib.import_module("diagnose_wing_reliability")
    artifact = tmp_path / "input_policy.json"
    artifact.write_text(json.dumps(policy))
    output = tmp_path / "output"
    args = ["diagnose_wing_reliability.py", "--output", str(output)]
    if opt_in:
        args += ["--research-policy", str(artifact), "--research-noise-assumption",
                 "known_iid_gaussian_iv", "--draws", "128", "--shock-std", ".001"]
    seen = []

    def fake_diagnostic(training, **kwargs):
        seen.append(kwargs)
        return deepcopy(evidence)

    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(diagnostic, "source_identity", lambda: {"sha256": {}})
    monkeypatch.setattr(diagnostic, "diagnose_wing_reliability", fake_diagnostic)
    diagnostic.run_local()
    saved = json.loads((output / "report.json").read_text())
    assert seen[0]["draws"] == (128 if opt_in else 16)
    assert seen[0]["shock_std"] == (.001 if opt_in else .0005)
    assert ("research_assessment" in saved["result"]) == opt_in
    if opt_in:
        from hashlib import sha256
        assert (output / "research_policy.json").read_bytes() == artifact.read_bytes()
        assert saved["metadata"]["research_policy"]["sha256"] == sha256(
            artifact.read_bytes()).hexdigest()
        assert saved["result"]["research_assessment"]["status"] == "clear"
