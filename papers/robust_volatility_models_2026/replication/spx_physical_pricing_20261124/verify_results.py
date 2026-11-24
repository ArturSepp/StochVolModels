"""Check the restricted fits against the archived SPX quotes and plot the smiles."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .physical import CUTOFF, DATE, OUT, source_data
from .pricing import market, quote_metrics


def main():
    df, _, weights, ref = market()
    hist = json.loads((OUT / "physical.json").read_text())
    opts = json.loads((OUT / "pricing.json").read_text())
    drift = json.loads((OUT / "free_drift.json").read_text())
    fixed = json.loads((OUT / "filtered_pricing.json").read_text())
    quotes = pd.read_csv(OUT / "quote_fits.csv")
    states = pd.read_csv(OUT / "filtered_state.csv", index_col="date")
    data, hashes = source_data()
    sample = data.loc[:CUTOFF]
    assert opts["valuation_date"] == DATE and hist["end"] < DATE
    assert list(states.index) == list(sample.index.strftime("%Y-%m-%d"))
    assert (states.log_state_sd > 0).all() and np.isfinite(states).all().all()
    assert hashes == hist["hashes"] and sample.rv.count() == hist["rv_n"]
    assert quotes.contract.tolist() == df.contract.tolist()
    assert opts["quote_sha256"] == drift["quote_sha256"] == ref["quote_sha256"]
    assert fixed["quote_sha256"] == ref["quote_sha256"] and fixed["valuation_date"] == DATE
    model_rows = dict(opts["fits"], free_drift=drift, filtered_spot=fixed)
    mid = ((df.bid_iv + df.ask_iv) / 2).to_numpy()
    for name, row in model_rows.items():
        iv = row["fine_iv"] if "fine_iv" in row else quotes[name + "_iv"].to_numpy()
        vals = quote_metrics(df, weights, mid, np.asarray(iv))
        for key in vals:
            np.testing.assert_allclose(vals[key], row[key], atol=1e-10)
    p = hist["parameters"]
    for row in (opts["fits"]["premium_only"], opts["fits"]["premium_and_state"], fixed):
        q = row["parameters"]
        np.testing.assert_allclose(
            [
                q["theta"] * q["kappa1"],
                q["kappa2"] * q["theta"] - q["kappa1"],
                q["beta"],
                q["volvol"],
                q["kappa2"] - row["delta"],
            ],
            [p["A"], p["B"], p["beta"], p["epsilon"], p["C"]],
            atol=1e-10,
        )
        assert q["kappa2"] >= max(0.0, q["beta"])
    np.testing.assert_allclose(opts["fits"]["premium_only"]["model_vix"], data.loc[DATE, "vix"])
    np.testing.assert_allclose(
        quotes.unrestricted_iv, np.concatenate(ref["final"]["ivols"]), atol=2e-6
    )
    plot_smiles(df, mid, quotes, np.asarray(fixed["fine_iv"]))
    print(
        f"Verified {len(sample)} sessions, {sample.rv.count()} RV observations "
        f"and five fits on {len(df)} quotes."
    )


def plot_smiles(df, mid, quotes, filtered_iv):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True, layout="constrained")
    for ax, (t, g) in zip(axes.flat, df.groupby("ttm")):
        x = 100 * np.log(g.strike / g.forward_price)
        ax.fill_between(
            x, 100 * g.bid_iv, 100 * g.ask_iv, color="#b9c3cb", alpha=0.55, label="Bid–ask"
        )
        ax.plot(
            x, 100 * mid[g.index], "o", color="#253340", markersize=2.7, label="Market midpoint"
        )
        ax.plot(
            x,
            100 * quotes.premium_only_iv[g.index],
            color="#ad5924",
            lw=2,
            label="VIX-inverted spot + premium",
        )
        ax.plot(
            x,
            100 * filtered_iv[g.index],
            color="#71465c",
            ls="--",
            lw=1.7,
            label="Filtered spot + premium",
        )
        ax.plot(
            x,
            100 * quotes.unrestricted_iv[g.index],
            color="#266ba2",
            lw=1.8,
            label="Unrestricted Q",
        )
        expiry = pd.to_datetime(g.mat_id.iloc[0], format="%d%b%Y").strftime("%d %b")
        ax.set_title(f"{expiry} ({365 * t:.0f} days)", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.22)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.supxlabel("Log-moneyness (%)")
    fig.supylabel("Black implied volatility (%)")
    fig.savefig(OUT / "spx_smiles.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
