"""Fast controls for the public, archive-independent numerical qualification."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.repository_only


@pytest.fixture
def portable(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root/"validation").is_dir():
        pytest.skip("requires the repository qualification runner")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_portable

    return pde_portable, json.loads(pde_portable.FIXTURE.read_bytes())


def test_frozen_chain_grids_and_per_quote_decisions(portable):
    p, fixture = portable
    spec = fixture["btc"]
    case = p.restore_case(spec["snapshot"])
    assert p._hash(p.input_snapshot(case)) == spec["input_sha256"]
    grids = [p.btc_variant(case, k) for k in (4, 5)]
    assert [v.grid.shape for v in grids] == [(599, 353), (733, 427)]
    assert grids[0].grid.x.bounds == grids[1].grid.x.bounds
    assert grids[0].grid.y.bounds == grids[1].grid.y.bounds
    assert grids[0].steps == grids[1].steps == (320, 320, 400, 640)
    lengths = [len(s) for s in case.chain.strikes_ttms]
    split = np.cumsum(lengths)[:-1]
    a, b = (np.split([r[k] for r in spec["quotes"]], split) for k in ("price4", "price5"))
    rows = p.quote_decisions(case.chain, a, b, spec)
    assert len(rows) == 49 and all(r["reproduced"] and r["spatial_pass"] for r in rows)
    b[-1][-1] += .001*case.chain.forwards[-1]
    row = p.quote_decisions(case.chain, a, b, spec)[-1]
    assert not row["reproduced"] and not row["spatial_pass"]
    spec["quotes"].reverse()
    with pytest.raises(ValueError, match="coordinate"):
        p.quote_decisions(case.chain, a, b, spec)


def test_frozen_bin_gates_and_paired_error(portable):
    p, fixture = portable
    spec = fixture["stress"]
    assert len(spec["bins"]) == 24
    for ref in spec["bins"]:
        row = p.bin_decision(ref["time"], ref["bin"], ref["conditional"], ref["mc"],
                             ref["mc_se"], ref["coupled_step_se"], ref, spec["limits"])
        assert row["qualified"] and row["reproduced"]
    se = [*ref["mc_se"]]
    se[1] = .02*ref["mc"][1]
    row = p.bin_decision(ref["time"], ref["bin"], ref["conditional"], ref["mc"], se,
                         ref["coupled_step_se"], ref, spec["limits"])
    assert not row["checks"]["precision_fine"] and not row["reproduced"]
    row = p.bin_decision(ref["time"], ref["bin"], ref["conditional"], ref["mc"], ref["mc_se"],
                         2*ref["coupled_step_se"], ref, spec["limits"])
    assert row["qualified"] and not row["reproduced"]


def test_coupled_paths_match_package_at_both_resolutions(portable):
    p, fixture = portable
    from stochvolmodels.models.lsv import simulation
    from validation.pde_coupled_mc import sample

    spec = fixture["stress"]
    model = p.stress_case(spec).model
    seed, pairs, rate = 73191, 13, 96
    times = spec["times"]
    actual = sample(model, seed, pairs, times, rate)

    def paths(n):
        cfg = simulation.LSVMonteCarloConfig(paths=2*pairs, seed=seed, steps_per_year=n)
        return simulation.LSVMonteCarloEngine(model, cfg).simulate_paths(observation_times=np.r_[0., times])

    increments = np.random.default_rng(seed).standard_normal((2*rate, 2, 2, pairs))
    coarse = increments.sum(axis=1)/np.sqrt(2.)
    stream = iter(np.concatenate((coarse, -coarse), axis=-1).reshape(-1, 2*pairs))
    with patch.object(simulation, "_normals", side_effect=lambda *args: next(stream)):
        coarse_paths = paths(rate)
    assert next(stream, None) is None
    for i, ref in enumerate((coarse_paths, paths(2*rate))):
        for f, field in enumerate(("log_zero_drift_return", "sigma")):
            expected = ref.states[field][:, 1:].T
            np.testing.assert_allclose(actual[i, :, f].reshape(expected.shape), expected,
                                       rtol=2e-12, atol=2e-13)


def test_small_joint_density_and_independent_pair_covariance(portable):
    p, fixture = portable
    from validation.pde_coupled_mc import coupled_statistics, sample

    spec = fixture["stress"]
    model_case = p.stress_case(spec)
    grid = p.TensorGrid2D(p.Grid1D(np.linspace(-2., 2., 41)), p.Grid1D(np.linspace(0., 2., 21)))
    controls = dict(prices=[.23113129423579334, .15594715657939315, .10333094470165521,
                            .0668734322187285, .02960268554332544],
                    second_moments=[.06976017361398269, .0835783145817294,
                                    .08071540105362976, .06264148203470482])
    run = p.projection(model_case, "small", grid, 16, np.log(spec["strikes"]), controls)
    assert len(run["prices"][0]) == 5
    for obs in run["checkpoints"]:
        assert len(obs["conditional"]) == 6
        np.testing.assert_allclose(sum(obs["prob"]), 1., atol=1e-12)
    sampling = dict(rates=[16, 32], batches=2, pairs_per_batch=128, chunk_pairs=64, master_seed=42)
    means, cov = coupled_statistics(model_case.model, spec["times"], spec["strikes"], sampling, 2)
    all_paths = np.concatenate([
        sample(model_case.model, np.random.SeedSequence(42, spawn_key=(b, c)),
               64, spec["times"], 16) for b in range(2) for c in range(2)], axis=-1)
    x, s = all_paths[:, -1, 0], all_paths[:, -1, 1]
    hit = x < np.log(spec["strikes"][0])
    den, num = hit.mean(axis=1), (s**2*hit).mean(axis=1)
    draws = np.array([den[0], num[0], den[1], num[1]])
    np.testing.assert_allclose(means[18], draws.mean(axis=1))
    np.testing.assert_allclose(cov[18], np.cov(draws)/draws.shape[1], atol=1e-16)
    estimates, errors, paired = p.ratio_statistics(means[18], cov[18])
    influence = (num-estimates[:, None]*den)/den.mean(axis=1)[:, None]
    np.testing.assert_allclose(errors, influence.std(axis=1, ddof=1)/16)
    np.testing.assert_allclose(paired, np.diff(influence, axis=0).std(ddof=1)/16)
