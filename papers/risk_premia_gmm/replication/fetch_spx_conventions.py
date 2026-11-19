"""Retrieve Bloomberg LIVE conventions and historical implied forwards for SPX.

Uses the external Bloomberg Python environment. The historical end date is fixed
to the existing IV cache so the tail study compares an unchanged sample.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import pandas as pd

MATURITIES = {"30d": 30, "60d": 60, "3m": 90, "6m": 180, "12m": 360}


class Locals(Enum):
    """Available provider operations."""

    FETCH = "fetch"


def fetch(data_path, forwards_path, output):
    """Persist provider metadata, convention checks and unfilled forward histories."""
    import bbg_fetch as bbg
    import blpapi

    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    end = data.dropna().index.max()
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "IVOL_MONEYNESS_LEVEL",
        "IVOL_MATURITY",
        "IVOL_IMPLIED_FORWARD",
        "IVOL_MONEYNESS",
        "EQY_DVD_YLD_12M",
        "IMP_FORWARD_PRICE",
        "END_OF_DAY_IMPLIED_FORWARD_PRICE",
        "IVOL_SURFACE_MONEYNESS",
    ]
    options = blpapi.SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    session = blpapi.Session(options)
    metadata = []
    try:
        if not session.start() or not session.openService("//blp/apiflds"):
            raise RuntimeError("Bloomberg field metadata service unavailable")
        req = session.getService("//blp/apiflds").createRequest("FieldInfoRequest")
        for field in fields:
            req.getElement("id").appendValue(field)
        req.set("returnFieldDocumentation", True)
        session.sendRequest(req)
        for _ in range(20):
            event = session.nextEvent(3000)
            for msg in event:
                if msg.hasElement("fieldData"):
                    entries = msg.getElement("fieldData")
                    for i in range(entries.numValues()):
                        entry = entries.getValueAsElement(i)
                        if entry.hasElement("fieldInfo"):
                            info = entry.getElement("fieldInfo")
                            metadata.append(
                                {
                                    k: info.getElementAsString(k)
                                    for k in ("mnemonic", "description", "documentation")
                                    if info.hasElement(k)
                                }
                            )
            if event.eventType() == blpapi.Event.RESPONSE:
                break
    finally:
        session.stop()
    (output / "bloomberg_field_definitions.json").write_text(json.dumps(metadata, indent=2))
    definitions = {m["mnemonic"]: m["documentation"] for m in metadata}
    if "strike divided by underlying price" not in definitions.get("IVOL_MONEYNESS_LEVEL", ""):
        raise RuntimeError(
            "Moneyness definition changed or not retrieved; inspect provider metadata"
        )
    frames, tenor_checks = [], []
    fixed_fields = [
        bbg.IMPVOL_FIELDS_MNY_30DAY,
        bbg.IMPVOL_FIELDS_MNY_60DAY,
        bbg.IMPVOL_FIELDS_MNY_3MTH,
        bbg.IMPVOL_FIELDS_MNY_6MTH,
        bbg.IMPVOL_FIELDS_MNY_12M,
    ]
    for (tenor, days), mapping in zip(MATURITIES.items(), fixed_fields):
        maturity = f"MATURITY_{days}D"
        raw = bbg.bdh(
            "SPX Index",
            "IVOL_IMPLIED_FORWARD",
            start_date=data.index.min(),
            end_date=end,
            IVOL_MATURITY=maturity,
        )
        series = raw.iloc[:, 0].rename(f"forward_{tenor}")
        if series.dropna().empty or (series.dropna() <= 0).any():
            raise RuntimeError(f"No valid positive implied forwards: {tenor}")
        frames.append(series)
        atm_field = next(k for k in mapping if "100.0%" in k)
        check = bbg.bdh(
            "SPX Index",
            ["IVOL_MONEYNESS", atm_field],
            start_date=end - pd.Timedelta(days=10),
            end_date=end,
            IVOL_MATURITY=maturity,
            IVOL_MONEYNESS_LEVEL="MONEY_LVL_100_0",
        )
        check.columns = check.columns.droplevel(0)
        # Generic history is rounded to 0.01 volatility percentage point.
        error = float((check.IVOL_MONEYNESS - check[atm_field]).abs().max())
        tenor_checks.append(
            {
                "tenor": tenor,
                "calendar_days": days,
                "override": maturity,
                "max_atm_vol_difference_percent": error,
                "matched_within_provider_rounding": bool(error <= 0.00501),
            }
        )
        check.to_csv(output / f"tenor_check_{tenor}.csv")
        print(
            f"{tenor}: {series.notna().sum()} forwards, ATM convention error {error:.6g}",
            flush=True,
        )
    forward_data = pd.concat(frames, axis=1, sort=True).sort_index()
    forwards_path.parent.mkdir(parents=True, exist_ok=True)
    forward_data.to_csv(forwards_path)
    surface = bbg.bds("SPX Index", "IVOL_SURFACE_MONEYNESS", IVOL_TIME_RANGE="1/1/12")
    surface.to_csv(output / "live_surface_moneyness_evidence.csv", index=False)
    ratio_error = float((surface.strike / surface.underlying_price - surface.moneyness).abs().max())
    bulk_status = {}
    for name, overrides in [
        ("IMP_FORWARD_PRICE", {"IVOL_SURFACE_SOURCE": "LIVE"}),
        ("END_OF_DAY_IMPLIED_FORWARD_PRICE", {}),
    ]:
        bulk = bbg.bds("SPX Index", name, **overrides)
        bulk.to_csv(output / f"{name}.csv", index=False)
        bulk_status[name] = {"rows": len(bulk), "columns": list(bulk.columns)}
    joined = data.join(forward_data)
    missing = joined.loc[data.dropna().index, forward_data.columns].isna().sum().to_dict()
    record = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "forwards_path": str(forwards_path),
        "forwards_sha256": hashlib.sha256(forwards_path.read_bytes()).hexdigest(),
        "end_date": str(end.date()),
        "tenor_checks": tenor_checks,
        "moneyness": "K/S; provider IVOL_MONEYNESS_LEVEL documentation",
        "max_strike_spot_mapping_error": ratio_error,
        "missing_forwards_on_complete_iv_dates": {k: int(v) for k, v in missing.items()},
        "discount_factor_bulk_status": bulk_status,
        "rate_status": (
            "No historical tenor discount curve recovered. USGG3M remains an explicitly "
            "labelled rate proxy; stress +/-100bp with forward held fixed."
        ),
        "year_fraction": (
            "Documented fixed-day tenors divided by 365 (ACT/365F research convention)."
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "conventions_manifest.json").write_text(json.dumps(record, indent=2))
    bbg.disconnect()
    print(json.dumps(record, indent=2))


def run_local(local: Locals = Locals.FETCH) -> None:
    """Fetch the evidence and forward history to explicit private paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--forwards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if local is Locals.FETCH:
        fetch(args.data, args.forwards, args.output)


if __name__ == "__main__":
    run_local()
