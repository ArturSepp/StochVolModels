"""Cross-check three short-dated prices in each new unrestricted OCA calibration."""

import json
import sys

import numpy as np
from scipy.special import ndtr
from stochvolmodels import LogSvParams, OptionChain

from .physical import OUT
from .pricing import quotes


def compare_short(case, fit, df):
    # The original research module imports its sibling pricing_engine by name.
    sys.path.insert(0, str(OUT.parent / "calibration_fits_20261029"))
    from mc_engine import MonteCarlo

    short = df.loc[df.ttm == df.ttm.min()].copy()
    atm = np.abs(np.log(short.strike / short.forward_price)).idxmin()
    g = short.loc[np.unique([short.index[0], atm, short.index[-1]])]
    t = float(g.ttm.iloc[0])
    k = (g.strike / g.forward_price).to_numpy()
    signs = np.where(g.optiontype == "C", 1.0, -1.0)
    discount = g.discount.iloc[0]
    iv = np.array(fit["fine_iv"])[g.index]

    def black(vol):
        sd = vol * np.sqrt(t)
        d1 = -np.log(k) / sd + 0.5 * sd
        return discount * signs * (ndtr(signs * d1) - k * ndtr(signs * (d1 - sd)))

    ref = black(iv)
    spread = black(g.ask_iv.to_numpy()) - black(g.bid_iv.to_numpy())
    chain = OptionChain(
        ttms=np.array([t]),
        forwards=np.ones(1),
        discfactors=np.array([discount]),
        strikes_ttms=[k],
        optiontypes_ttms=[g.optiontype.to_numpy(dtype="U1")],
        ids=np.array([g.mat_id.iloc[0]]),
        ticker=case.upper(),
    )
    p = LogSvParams(**fit["parameters"])
    runs = []
    coarse = []
    for seed in [241101, 241102, 241103, 241104]:
        engine = MonteCarlo(chain, power=13, steps=256, seed=seed)
        runs.append(engine.prices(p)[0])
        coarse.append(engine.coarsen().prices(p)[0])
    vals = np.asarray(runs)
    gap = vals.mean(axis=0) - ref
    rec = dict(
        case=case,
        contracts=g.contract.to_list(),
        seeds=[241101, 241102, 241103, 241104],
        paths=8192,
        steps=256,
        price_gap_bp_forward=(gap * 1e4).tolist(),
        standard_error_bp_forward=(vals.std(axis=0, ddof=1) * 5000).tolist(),
        paired_time_change_bp_forward=(np.mean(vals - np.asarray(coarse), axis=0) * 1e4).tolist(),
        fraction_of_iv_spread_in_price=(gap / spread).tolist(),
    )
    return rec


def main() -> None:
    rows = []
    for case in ("uso", "btc", "eth"):
        fit = json.loads((OUT / f"{case}_unrestricted.json").read_text())
        df, _, _, _ = quotes(case)
        rec = compare_short(case, fit, df)
        rows.append(rec)
        print(case, rec, flush=True)
    (OUT / "independent_short_prices.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
