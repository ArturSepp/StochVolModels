"""Held-out Bloomberg surface wings and exact fixed-distribution rate sensitivity.

The 125%, 130%, 140% and 150% nodes are never used to select or calibrate a prior.
They are vendor-calculated surface values, not independent executable option quotes.
"""

import argparse
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class Locals(Enum):
    """Provider retrieval and offline evaluation."""

    FETCH = "fetch"
    EVALUATE = "evaluate"


def fetch_wings(data_path, wings_path):
    """Read documented maturity/moneyness overrides without changing the fit inputs."""
    import bbg_fetch as bbg

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    start, end = data.index.min(), data.dropna().index.max()
    series = []
    for tenor, days in [("6m", 180), ("12m", 360)]:
        for moneyness in [125, 130, 140, 150]:
            raw = bbg.bdh(
                "SPX Index",
                "IVOL_MONEYNESS",
                start_date=start,
                end_date=end,
                IVOL_MATURITY=f"MATURITY_{days}D",
                IVOL_MONEYNESS_LEVEL=f"MONEY_LVL_{moneyness}_0",
            )
            if raw.empty:
                raise RuntimeError(f"No held-out wing data: {tenor}, {moneyness}")
            column = raw.iloc[:, 0].rename(f"{tenor}_{moneyness}") * 0.01
            series.append(column)
            print(f"{tenor}_{moneyness}: {column.notna().sum()} observations", flush=True)
    pd.concat(series, axis=1, sort=True).to_csv(wings_path)
    bbg.disconnect()


def evaluate(data_path, forwards_path, wings_path, output):
    """Score frozen fits on withheld nodes and isolate rate effects algebraically."""
    from papers.risk_premia_gmm.replication import spx_feasibility as base

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    forwards = pd.read_csv(forwards_path, index_col=0, parse_dates=True)
    wings = pd.read_csv(wings_path, index_col=0, parse_dates=True)
    fits = pd.read_csv(output / "baseline_fits.csv")
    records = []
    for _, row in fits[fits.accepted & fits.tenor.isin(["6m", "12m"])].iterrows():
        date = pd.Timestamp(row.date)
        if date not in wings.index:
            continue
        ratios = np.array([1.25, 1.3, 1.4, 1.5])
        strikes = ratios * data.loc[date, "spot_price"] / forwards.loc[date, f"forward_{row.tenor}"]
        theta = np.array(json.loads(row.theta))
        model = base.mixture_prices(theta, int(row.components), row.ttm, strikes, np.ones(4))
        model_vols = base.implied_vols(model, strikes, row.ttm, np.ones(4))
        for i, moneyness in enumerate([125, 130, 140, 150]):
            observed = wings.loc[date, f"{row.tenor}_{moneyness}"]
            if not np.isfinite(observed) or observed <= 0:
                continue
            records.append(
                {
                    "date": row.date,
                    "tenor": row.tenor,
                    "case": row.case,
                    "moneyness": moneyness,
                    "observed_iv": observed,
                    "model_iv": model_vols[i],
                    "error_vol_bp": (model_vols[i] - observed) * 1e4,
                }
            )
    scored = pd.DataFrame(records)
    scored.to_csv(output / "heldout_wing_errors.csv", index=False)
    scored.groupby(["tenor", "case"]).error_vol_bp.agg(
        valid_nodes="count",
        rmse_vol_bp=lambda x: np.sqrt(np.mean(x * x)),
        median_abs_vol_bp=lambda x: x.abs().median(),
        p95_abs_vol_bp=lambda x: x.abs().quantile(0.95),
    ).to_csv(output / "heldout_wing_summary.csv")
    valid = scored.dropna(subset=["error_vol_bp"])
    keys = ["date", "tenor", "moneyness"]
    common_counts = valid.groupby(keys)["case"].transform("nunique")
    common = valid[common_counts == 5]
    common.groupby(["tenor", "case"]).error_vol_bp.agg(
        valid_nodes="count",
        rmse_vol_bp=lambda x: np.sqrt(np.mean(x * x)),
        median_abs_vol_bp=lambda x: x.abs().median(),
        p95_abs_vol_bp=lambda x: x.abs().quantile(0.95),
    ).to_csv(output / "heldout_wing_matched_summary.csv")
    rate = fits[fits.accepted][["date", "tenor", "case", "ttm", "erp_lambda3"]].copy()
    for shift, label in [(0.01, "plus"), (-0.01, "minus")]:
        rate[f"rate_{label}_100bp_change_pp"] = 100 * rate.erp_lambda3 * np.expm1(shift * rate.ttm)
    rate.to_csv(output / "fixed_distribution_rate_sensitivity.csv", index=False)
    rate.groupby(["tenor", "case"]).rate_plus_100bp_change_pp.agg(
        median_pp="median", p95_pp=lambda x: x.quantile(0.95), max_pp="max"
    ).to_csv(output / "fixed_distribution_rate_summary.csv")
    manifest = {
        "wings_path": str(wings_path),
        "wings_sha256": hashlib.sha256(wings_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calibration_use": "None: prior strengths frozen before retrieval",
        "source_type": "Bloomberg LIVE calculated surface; not raw bid/ask quotes",
        "first_scored_date": str(scored.date.min()),
        "last_scored_date": str(scored.date.max()),
        "matched_model_node_rows": len(common),
        "heldout_rows": len(scored),
        "valid_model_nodes": int(scored.error_vol_bp.notna().sum()),
        "rate_effect": "Same fitted distribution and forward; ERP(r+dr)/ERP(r)=exp(dr*T)",
    }
    (output / "heldout_validation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print((output / "heldout_wing_summary.csv").read_text())


def run_local(
    local: Locals,
    *,
    data_path: Path,
    wings_path: Path,
    forwards_path: Path | None = None,
    output: Path | None = None,
) -> None:
    """Run the selected fetch or evaluation using explicit data paths."""
    if local is Locals.FETCH:
        fetch_wings(data_path, wings_path)
    elif local is Locals.EVALUATE:
        if forwards_path is None or output is None:
            raise ValueError("evaluate requires --forwards and --output")
        evaluate(data_path, forwards_path, wings_path, output)
    else:
        raise ValueError(f"Unknown local operation: {local!r}")


def _parse_args() -> argparse.Namespace:
    """Parse the existing fetch/evaluate commands and validate required paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=[x.value for x in Locals])
    for name in ("data", "forwards", "wings", "output"):
        parser.add_argument("--" + name, type=Path, required=name in ("data", "wings"))
    args = parser.parse_args()
    if Locals(args.mode) is Locals.EVALUATE:
        if args.forwards is None or args.output is None:
            parser.error("evaluate requires --forwards and --output")
    return args


if __name__ == "__main__":
    args = _parse_args()
    run_local(
        local=Locals(args.mode),
        data_path=args.data,
        wings_path=args.wings,
        forwards_path=args.forwards,
        output=args.output,
    )
