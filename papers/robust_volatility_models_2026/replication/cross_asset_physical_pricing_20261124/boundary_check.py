"""Verify the exact affine pricing boundary for the July TLT filtered-state fit."""

import json

import numpy as np

from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.fourier import (
    calls_from_transform,
    gaussian_grid,
)
from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.logvol_pde import (
    _evolve,
)
from .physical import OUT
from .pricing import SEARCH_MEDIUM, evaluate_iv, quotes, quote_metrics, restricted


def main() -> None:
    h = json.loads((OUT / "tlt_physical.json").read_text())
    info = json.loads((OUT / "tlt_filtered.json").read_text())
    df, chain, w, _ = quotes("tlt")
    p, spot = h["parameters"], h["spot"]
    assert p["B"] > 0 and p["beta"] <= 0 and info["parameters"]["kappa2"] < 1e-4
    grid = SEARCH_MEDIUM
    phi, weights = gaussian_grid(grid.quadrature_order, grid.cutoff)
    y = np.linspace(
        np.log(spot) - grid.logvol_halfwidth, np.log(spot) + grid.logvol_halfwidth, grid.space_nodes
    )
    prices = []
    for t, k, types, discount in zip(
        chain.ttms, chain.strikes_ttms, chain.optiontypes_ttms, chain.discfactors
    ):
        # The kernel uses k1*theta/s + k2*theta-k1-k2*s. The algebraic pair
        # below supplies A/s + B exactly; neither number is a fitted equilibrium.
        tr = _evolve(
            phi,
            y,
            spot,
            -p["A"] / p["B"],
            -p["B"],
            0.0,
            p["beta"],
            p["epsilon"],
            t,
            grid.time_steps,
        )
        calls = calls_from_transform(tr, phi, weights, t, spot, k)
        prices.append(discount * (calls - np.where(types == "P", 1 - k, 0.0)))
    exact = np.concatenate(chain.compute_model_ivols_from_chain_data(prices))
    near = evaluate_iv(chain, restricted(p, info["parameters"]["kappa2"], spot), grid)
    np.testing.assert_allclose(exact, near, atol=2e-6, rtol=0)
    mid = np.concatenate(chain.get_mid_vols())
    scores = {
        str(c): quote_metrics(df, w, mid, evaluate_iv(chain, restricted(p, c, spot), grid))[
            "rmse_pp"
        ]
        for c in (0.10, 1.0, 5.0)
    }
    rmse = quote_metrics(df, w, mid, exact)["rmse_pp"]
    assert all(v > rmse for v in scores.values())
    record = dict(
        rmse_zero=rmse,
        near_zero_max_iv_diff_pp=float(100 * max(abs(exact - near))),
        positive_coefficient_rmse=scores,
    )
    (OUT / "tlt_boundary_check.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record, flush=True)


if __name__ == "__main__":
    main()
