"""Check the daily simulation grid, the archived inputs and the reported comparisons."""

import json

import numpy as np
import pandas as pd

from ..vix_fit_20261123.fit_vix import LAST_LAG, model_acf, simulate_daily
from .compare_assets import BLOOMBERG, OUT, PRIMARY, series_for


def check_daily_grid():
    params = dict(theta=0.24, kappa1=2.0, kappa2=8.0, v=0.85)
    rng = np.random.default_rng(23)
    for days in (260, 365):
        increments = rng.normal(size=(160, 3)) / np.sqrt(days)
        states = simulate_daily(*params.values(), increments, days)
        assert states.shape == (len(increments) + 1, increments.shape[1])
        np.testing.assert_array_equal(states[0], np.full(3, params["theta"]))
        # Independent one-step update, including the last supplied shock.
        level = states[:-1]
        drift = (params["kappa1"] + params["kappa2"] * level) * (params["theta"] - level)
        predicted = level * np.exp(
            (drift / level - params["v"] ** 2 / 2) / days + params["v"] * increments
        )
        np.testing.assert_allclose(states[1:], predicted, rtol=1e-13)
        assert np.all(np.isfinite(states) & (states > 0))
        direct = [
            [np.corrcoef(path[:-lag], path[lag:])[0, 1] for path in states.T]
            for lag in range(1, LAST_LAG + 1)
        ]
        np.testing.assert_allclose(
            model_acf(params, increments, days), np.mean(direct, axis=1), atol=1e-12
        )


def main():
    check_daily_grid()
    panel = json.loads((BLOOMBERG / "panel_results.json").read_text())["established_panel"]
    names = list(dict.fromkeys((*PRIMARY, *panel)))
    csv = pd.read_csv(OUT / "fits.csv")
    assert set(csv["asset"]) == set(names) and len(csv) == 6 * len(names)
    cols = ["theta", "kappa1", "kappa2", "v", "local_speed"]
    for name in names:
        result = json.loads((OUT / "results" / (name.replace(" ", "_") + ".json")).read_text())
        levels, days, digest = series_for(name)
        assert result["input_sha256"] == digest and result["n"] == len(levels)
        assert (result["start"], result["end"]) == tuple(
            str(i.date()) for i in levels.index[[0, -1]]
        )
        assert result["observation_clock"] == days
        fit = result["persistence"]
        transition = result["transition"]
        subset = csv.loc[csv["asset"] == name]
        rows = [fit["parameters"]] + [r["parameters"] for r in transition["fits"].values()]
        np.testing.assert_allclose(subset[cols], [[p[k] for k in cols] for p in rows], rtol=1e-8)
        for par in rows:
            assert par["theta"] > 0 and par["v"] > 0 and min(par["kappa1"], par["kappa2"]) >= 0
            np.testing.assert_allclose(
                par["local_speed"], par["kappa1"] + par["kappa2"] * par["theta"]
            )
        assert transition["validation"]["n"] == len(levels) - 1 - int(0.7 * (len(levels) - 1))
        scores = transition["fits"]
        for s in scores.values():
            np.testing.assert_allclose(
                s["gain_over_equal"],
                s["mean_oos_logscore"] - scores["equal"]["mean_oos_logscore"],
                atol=1e-12,
            )
        observed = np.array([levels.autocorr(lag) for lag in range(1, LAST_LAG + 1)])
        np.testing.assert_allclose(observed, result["acf"]["empirical"], atol=1e-12)
        for method, sse in (
            ("persistence", fit["acf_sse"]),
            ("daily", result["transition_acf_sse"]),
        ):
            residuals = observed - result["acf"][method]
            np.testing.assert_allclose(residuals @ residuals, sse, atol=1e-12)
        if "profile_checks" in fit:
            assert fit["acf_sse"] <= min(c["acf_sse"] for c in fit["profile_checks"]) + 1e-10
        if name == "VIX":
            vix = OUT.parent / "vix_fit_20261123"
            standalone = json.loads((vix / "transition_weight_check.json").read_text())
            profile = json.loads((vix / "vix_profile.json").read_text())
            np.testing.assert_allclose(
                result["acf"]["persistence"], profile["full"]["best"]["fitted_acf"]
            )
            np.testing.assert_allclose(
                result["transition_acf_sse"],
                standalone["in_sample_acf_check"]["equal_transition_sse"],
            )
    print(
        f"Checked both daily grids, {len(names)} source histories, "
        f"{len(csv)} fits and 119 correlations per fitted curve."
    )


if __name__ == "__main__":
    main()
