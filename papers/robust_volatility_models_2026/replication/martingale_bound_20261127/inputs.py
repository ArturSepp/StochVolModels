"""Select the buffered positive-loading surfaces and re-read the hashed source quotes.

select() applies C-beta <= 0.06 to both archived experiments. load() returns
the original chain, weights and optimizer record. All study outputs are local.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..cross_asset_physical_pricing_20261124.panel_inputs import PANEL, sha
from ..cross_asset_physical_pricing_20261124.pricing import OUT as ARCHIVE, quotes
from ..spx_physical_pricing_20261124.pricing import chain_from_quotes

OUT = Path(__file__).resolve().parent
EXPECTED = {
    ("gld", "2022Q4"),
    ("gld", "2023Q1"),
    ("gld", "2023Q2"),
    ("tlt", "2023Q1"),
    ("tlt", "2023Q4"),
    ("btc", "2023Q2"),
    ("btc", "2023Q4"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


@dataclass(frozen=True)
class Surface:
    asset: str
    date: str
    quarter: str = ""

    @property
    def key(self):
        return f"{self.asset}_{self.date}"


def select():
    panel = pd.read_csv(PANEL / "date_results.csv")
    assert len(panel) == 47 and not panel.duplicated(["asset", "quarter"]).any()
    positive = panel.loc[panel.beta_unrestricted > 0]
    margin = positive.kappa2_unrestricted - positive.beta_unrestricted
    near = positive.loc[margin <= 0.06]
    panel_cases = [Surface(x.asset.lower(), x.date, x.quarter) for x in near.itertuples()]
    assert len(positive) == 12 and len(panel_cases) == 7
    assert {(x.asset, x.quarter) for x in panel_cases} == EXPECTED
    dated = pd.read_csv(ARCHIVE / "calibration_parameters.csv").query(
        "specification == 'unrestricted'"
    )
    kept = dated.loc[(dated.beta_Q > 0) & (dated.C_Q - dated.beta_Q <= 0.06)]
    dated_cases = [Surface(x.market, x.date) for x in kept.itertuples()]
    assert dated_cases == [Surface("uso", "2026-07-15")]
    return dated_cases + panel_cases, dict(
        quarterly=len(panel),
        positive=len(positive),
        buffered=len(near),
        dated=len(dated),
        dated_buffered=len(kept),
        next_margin=float(margin[margin > 0.06].min()),
    )


def load(case):
    if case.quarter:
        folder = PANEL / case.asset / case.date
        ref = read(folder / "source.json")
        path = folder / "quotes.csv"
        assert sha(path) == ref["quote_sha256"]
        market = chain_from_quotes(path, ref)
    else:
        folder = ARCHIVE
        market = quotes(case.asset)
    fit = read(folder / f"{case.asset}_unrestricted.json")
    frame, chain, _, ref = market
    assert fit["quote_sha256"] == ref["quote_sha256"] and ref["date"] == case.date
    assert fit["case"] == case.asset and fit["mode"] == "unrestricted"
    assert len(chain.ttms) == 4 and len(frame) == fit["n"] and np.all(np.diff(chain.ttms) > 0)
    return market, fit
