"""Decision controls for the repository-only volatility-moment audit."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.repository_only


@pytest.fixture
def audit(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root/"validation").is_dir():
        pytest.skip("repository-only qualification")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_moment_audit_report
    return pde_moment_audit_report


def test_worst_quote_and_quote_specific_allowance(audit):
    mc = dict(mean=[.1, .2], se=[.0001, .00001])
    quote = audit.mc_check([.10045, .19984], mc)
    np.testing.assert_allclose(quote["gap_bp"], [4.5, -1.6])
    np.testing.assert_allclose(quote["allowance_bp"], [5., 1.4])
    np.testing.assert_array_equal(quote["passed"], [True, False])
    pair = audit.price_pair([.1, .19989], [.1, .2], ["short call", "long put"])
    assert not pair["passed"] and pair["worst_quote"] == "long put"
    assert pair["max_bp"] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        audit.mc_check([.1], mc)


def test_conditional_qualification_requires_precision_and_all_bins(audit):
    rows = []
    for t in (.25, .5, 1., 2.):
        for i in range(6):
            precise = t == .25 and i == 2
            est = dict(conditional=1., se=.005 if precise else .015)
            flags = dict(grid=True, mc_step=True)
            grids = {"return_22": dict(conditional=1.008, share=.2),
                     "joint_fine": dict(conditional=1., share=.2)}
            for k in ("2880", "5760"):
                flags["precision_"+k] = precise
                for name in grids:
                    flags[f"agreement_{name}_{k}"] = True
            rows.append(dict(time=t, bin=i, mc={"2880": est, "5760": est}, pde=grids,
                             grid_change=.008, checks=flags, qualified=precise))
    decisions = audit.joint_decisions(dict(rows=rows))
    assert [(r["time"], r["bin"]) for r in decisions if r["qualified"]] == [(.25, 2)]
    with pytest.raises(ValueError, match="coverage"):
        audit.joint_decisions(dict(rows=[*rows, rows[0]]))
    rows[-1]["qualified"] = True
    with pytest.raises(ValueError, match="decision"):
        audit.joint_decisions(dict(rows=rows))
