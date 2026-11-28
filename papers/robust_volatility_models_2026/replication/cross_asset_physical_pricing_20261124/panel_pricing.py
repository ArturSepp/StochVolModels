"""Run the five pricing specifications on the quarterly OCA samples."""

import argparse
import json
import time

from ..spx_physical_pricing_20261124.pricing import chain_from_quotes
from .panel_inputs import PANEL, SOURCES, sha
from .pricing import MODELS, fit_surface


def fit_dates(asset: str, modes: list[str], wait: bool, quarter: str | None = None) -> None:
    entries = json.loads((PANEL / "dates.json").read_text())
    for entry in entries:
        date = entry["dates"].get(asset)
        if date is None or (quarter is not None and entry["quarter"] != quarter):
            continue
        folder = PANEL / asset / date
        history = folder / "history.json"
        while wait and not history.is_file():
            time.sleep(3)
        hist = json.loads(history.read_text())
        reference = json.loads((folder / "source.json").read_text())
        assert hist["end"] < date == hist["date"] == reference["date"]
        assert sha(folder / "quotes.csv") == reference["quote_sha256"]
        data = chain_from_quotes(folder / "quotes.csv", reference)
        for mode in modes:
            result = folder / f"{asset}_{mode}.json"
            if result.is_file():
                prior = json.loads(result.read_text())
                assert (
                    prior["historical_date"] == date
                    and prior["quote_sha256"] == reference["quote_sha256"]
                )
                continue
            print(asset, date, mode, flush=True)
            fit_surface(asset, mode, hist, data, folder, panel=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=SOURCES)
    parser.add_argument("modes", nargs="+", choices=MODELS)
    parser.add_argument("--wait-history", action="store_true")
    parser.add_argument("--quarter", help="Run one quarter, for example 2024Q1")
    args = parser.parse_args()
    fit_dates(args.asset, args.modes, args.wait_history, args.quarter)
