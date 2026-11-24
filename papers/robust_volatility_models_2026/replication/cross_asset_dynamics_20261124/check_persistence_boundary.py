"""Audit the two boundary fits and run an inverse-Gamma positive control."""

import json

import numpy as np
from scipy.stats import invgamma

from ..vix_fit_20261123.fit_vix import LAST_LAG, N_PATHS, SEED, YEARS, model_acf
from .compare_assets import OUT, persistence_fit, shocks


def main():
    vix = json.loads((OUT.parent / "vix_fit_20261123/vix_profile.json").read_text())
    comparisons = []
    for path in sorted((OUT / "results").glob("*.json")):
        result = json.loads(path.read_text())
        acf = result["acf"]["empirical"]
        profile = result["persistence"].get("profile_checks", vix["full"]["profile"])
        errors = [next(r["acf_sse"] for r in profile if r["share"] == share) for share in (0, 1)]
        comparisons.append(
            dict(
                asset=result["asset"],
                first_20_log_decay=-np.log(acf[19]) / 20,
                last_59_log_decay=-np.log(acf[-1] / acf[59]) / 59,
                gamma_rmse=np.sqrt(errors[0] / LAST_LAG),
                inverse_gamma_rmse=np.sqrt(errors[1] / LAST_LAG),
            )
        )
    params = dict(theta=0.2, kappa1=2.0, kappa2=0.0, v=0.75)
    alpha = 1 + 2 * params["kappa1"] / params["v"] ** 2
    quantiles = (np.arange(100_000) + 0.5) / 100_000
    levels = invgamma.ppf(quantiles, alpha, scale=(alpha - 1) * params["theta"])
    # Independent ten-year paths for the target and the fitter, both starting at theta.
    days = 260
    target_noise = np.random.RandomState(SEED + 1).normal(size=(YEARS * days, N_PATHS)) / np.sqrt(
        days
    )
    target = model_acf(params, target_noise)
    del target_noise
    fitted, _ = persistence_fit(levels, days, shocks(days), target)
    result = dict(
        panel=comparisons,
        positive_control=dict(
            true=params,
            histogram_quantiles=len(levels),
            target_seed=SEED + 1,
            fit_seed=SEED,
            paths=N_PATHS,
            fit=fitted,
        ),
    )
    (OUT / "persistence_boundary_check.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print("Inverse-Gamma positive control", fitted, flush=True)


if __name__ == "__main__":
    main()
