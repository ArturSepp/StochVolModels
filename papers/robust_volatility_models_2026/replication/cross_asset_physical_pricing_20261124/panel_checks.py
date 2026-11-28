"""Verify paired quarterly fits and reprice the smallest gaps and least certain signs per asset."""

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd
from stochvolmodels import LogSvParams

from ..spx_physical_pricing_20261124.physical import vix_map
from ..spx_physical_pricing_20261124.pricing import FINE, chain_from_quotes, evaluate_iv
from .check_new_prices import compare_short
from .oca_inputs import RESOURCES
from .panel_inputs import PANEL, SOURCES, save, sha
from .pricing import MODELS, restricted


def read(path):
    return json.loads(path.read_text())


def data_for(asset, date):
    folder = PANEL / asset / date
    meta = read(folder / "source.json")
    assert sha(folder / "quotes.csv") == meta["quote_sha256"]
    frame, chain, _, _ = chain_from_quotes(folder / "quotes.csv", meta)
    assert (
        frame.exchange_time.nunique() == 1
        and str(pd.Timestamp(frame.exchange_time[0]).date()) == date
    )
    assert frame.ttm.nunique() == 4 and frame.contract.is_unique
    return folder, frame, chain, meta


def rmse(frame, iv):
    e = 100 * (iv - (frame.bid_iv + frame.ask_iv).to_numpy() / 2)
    per_maturity = pd.Series(e**2).groupby(frame.ttm).mean()
    return float(np.sqrt(per_maturity.mean()))


def results(asset):
    rows = []
    spot_index = None
    if asset in ("btc", "eth"):
        path = RESOURCES / "tardis" / f"{asset.upper()}_perp_freq_H.feather"
        spot_index = (
            pd.read_feather(path, columns=["timestamp", "index_price"])
            .set_index("timestamp")
            .index_price
        )
    for entry in read(PANEL / "dates.json"):
        date = entry["dates"].get(asset)
        if date is None:
            continue
        folder, frame, _, meta = data_for(asset, date)
        if spot_index is not None:
            stamp = pd.Timestamp(meta["timestamp"])
            assert stamp.hour == 8 and stamp.utcoffset().total_seconds() == 0
            np.testing.assert_allclose(frame.spot_price, spot_index.loc[stamp], rtol=1e-7)
        h = read(folder / "history.json")
        assert h["end"] < date == h["date"] and any(x["success"] for x in h["runs"])
        ph = h["parameters"]
        row = dict(
            asset=asset.upper(),
            quarter=entry["quarter"],
            date=date,
            quotes=len(frame),
            halfspread_rms_pp=rmse(frame, frame.ask_iv.to_numpy()),
            history_end=h["end"],
            sessions=h["n"],
            spot_filtered=h["spot"],
            observed_index=h["index"],
            theta_P=ph["theta"],
            kappa1_P=ph["kappa1"],
            kappa2_P=ph["C"],
            beta_P=ph["beta"],
            epsilon_P=ph["epsilon"],
        )
        for mode in MODELS:
            m = read(folder / f"{asset}_{mode}.json")
            assert m["quote_sha256"] == meta["quote_sha256"] and len(frame) == m["n"]
            assert m["historical_date"] == date and m["runs"][-1]["success"]
            iv = np.array(m["fine_iv"])
            q = m["parameters"]
            assert (
                iv.shape == (len(frame),)
                and np.all(iv > 0)
                and q["kappa2"] >= max(0, q["beta"]) - 1e-8
            )
            row[f"rmse_{mode}"] = rmse(frame, iv)
            row[f"inside_{mode}"] = int(((frame.bid_iv <= iv) & (iv <= frame.ask_iv)).sum())
            np.testing.assert_allclose(row[f"rmse_{mode}"], m["rmse_pp"], rtol=1e-10)
            assert row[f"inside_{mode}"] == m["inside_count"]
            for field, value in q.items():
                row[f"{field}_{mode}"] = value
            row[f"A_{mode}"] = q["theta"] * q["kappa1"]
            row[f"B_{mode}"] = q["theta"] * q["kappa2"] - q["kappa1"]
            if q["kappa2"] < 1e-4 and row[f"B_{mode}"] > 0:
                row[f"theta_{mode}"] = np.nan
                row[f"kappa1_{mode}"] = np.nan
            if mode in ("filtered", "inverted", "spot"):
                row[f"delta_{mode}"] = q["kappa2"] - ph["C"]
                np.testing.assert_allclose(m["delta"], row[f"delta_{mode}"], atol=1e-9)
                np.testing.assert_allclose(
                    [ph["A"], ph["B"], ph["beta"], ph["epsilon"]],
                    [
                        q["theta"] * q["kappa1"],
                        q["theta"] * q["kappa2"] - q["kappa1"],
                        q["beta"],
                        q["volvol"],
                    ],
                )
            if mode == "inverted":
                np.testing.assert_allclose(m["model_index"], h["index"], atol=1e-8)
            if mode == "filtered":
                np.testing.assert_allclose(q["sigma0"], h["spot"], atol=1e-10)
        row["flip"] = row["delta_filtered"] * row["delta_inverted"] < 0
        for mode in ("filtered", "inverted", "spot", "drift"):
            row[f"gap_{mode}"] = row[f"rmse_{mode}"] - row["rmse_unrestricted"]
        rows.append(row)
    return pd.DataFrame(rows)


def reprice(asset, date, first):
    first = bool(first)
    folder, frame, chain, _ = data_for(asset, date)
    modes = ("filtered", "inverted", "unrestricted")
    hashes = {mode: sha(folder / f"{asset}_{mode}.json") for mode in modes}
    cache = folder / "fine_check.json"
    if cache.is_file():
        old = read(cache)
        needs_profile = any(
            abs(read(folder / f"{asset}_{mode}.json")["delta"]) <= 0.25
            for mode in ("filtered", "inverted")
        )
        if (
            old["hashes"] == hashes
            and old["first"] == first
            and (old.get("profile_version") == 1 or not needs_profile)
        ):
            return old["checked"]
    grid = replace(FINE, space_nodes=1201, time_steps=256)
    checked = []
    for mode in modes:
        m = read(folder / f"{asset}_{mode}.json")
        iv = evaluate_iv(chain, LogSvParams(**m["parameters"]), grid)
        item = dict(
            date=date,
            mode=mode,
            rmse=rmse(frame, iv),
            base=m["rmse_pp"],
            max_iv_diff_pp=float(100 * np.max(abs(iv - np.array(m["fine_iv"])))),
        )
        if mode in ("filtered", "inverted") and abs(m["delta"]) <= 0.25:
            # The fine-grid slope at zero provides a separate check of a small fitted sign.
            h = read(folder / "history.json")
            p = h["parameters"]
            residuals = []
            for shift in (-0.05, 0.0, 0.05):
                cq = p["C"] + shift
                assert cq >= max(0.0, p["beta"])
                spot = h["spot"]
                if mode == "inverted":
                    y, mapped = vix_map(p, cq, 1001, 192)
                    assert mapped[0] < np.log(h["index"]) < mapped[-1] and np.all(
                        np.diff(mapped) > 0
                    )
                    spot = np.exp(np.interp(np.log(h["index"]), mapped, y))
                residuals.append(
                    rmse(frame, evaluate_iv(chain, restricted(p, cq, spot), grid)) ** 2
                )
            left, center, right = residuals
            curvature = left - 2 * center + right
            assert curvature > 0
            item["local_delta_at_zero"] = 0.05 * (left - right) / (2 * curvature)
            item["local_sign_matches"] = bool(
                np.sign(item["local_delta_at_zero"]) == np.sign(m["delta"])
            )
        if first and mode == "unrestricted":
            item["mc"] = compare_short(asset, m, frame)
        checked.append(item)
    save(cache, dict(hashes=hashes, first=first, checked=checked, profile_version=1))
    return checked


def audit(asset):
    out = results(asset)
    near = out[["delta_filtered", "delta_inverted"]].abs().min(axis=1)
    target = {
        out.index[0],
        near.idxmin(),
        out[["gap_filtered", "gap_inverted"]].min(axis=1).idxmin(),
    }
    target.update(near.index[near <= 0.25])
    checked = []
    for idx in sorted(target):
        checked.extend(reprice(asset, out.loc[idx, "date"], idx == out.index[0]))
    save(PANEL / asset / "checks.json", checked)
    out.to_csv(PANEL / asset / "results.csv", index=False)
    print(asset, len(out), len(checked), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=SOURCES)
    parser.add_argument("--first", action="store_true")
    args = parser.parse_args()
    if args.first:
        date = next(
            e["dates"][args.asset] for e in read(PANEL / "dates.json") if args.asset in e["dates"]
        )
        reprice(args.asset, date, True)
    else:
        audit(args.asset)
