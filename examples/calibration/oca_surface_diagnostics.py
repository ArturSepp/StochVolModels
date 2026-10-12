"""Advanced offline OCA-to-SVM diagnostics; requires the research extra and OCA 5.3."""

from enum import Enum

import numpy as np
import pandas as pd
import vanilla_option_pricers as bsm

from stochvolmodels import fitters
from stochvolmodels.fitters.adapters.oca_diagnostics import diagnostic_chain_from_oca


def integrated_report(multiplier: float = 1.0):
    """Reconstruct flawed synthetic observations, fit LogSV and retain all evidence."""
    from option_chain_analytics import OptionsDataDFs, generate_simulated_options_data
    from option_chain_analytics.diagnostics import (
        QuoteDiagnosticConfig,
        reconstruct_with_diagnostics,
    )

    observations = generate_simulated_options_data(contract_size=0.1)
    frame = observations.chain_ts.copy()
    source_time = observations.get_timeindex()[0]
    rows = frame.index[frame.exchange_time.eq(source_time)]
    frame.loc[rows[4], "bid_price"] = frame.loc[rows[4], "ask_price"] + 0.5
    frame = pd.concat([frame, frame.loc[[rows[6]]]], ignore_index=True)
    frame[["bid_price", "ask_price", "mark_price"]] /= multiplier
    frame["usd_multiplier"] = multiplier
    observations = OptionsDataDFs(frame, observations.spot_data, observations.ticker)
    requested_time = source_time + pd.Timedelta(seconds=60)
    config = QuoteDiagnosticConfig(
        "USD per unit",
        price_conversion="usd_per_unit",
        payoff="standard_european",
        settlement="cash",
        max_quote_age_seconds=30,
        max_snapshot_age_seconds=30,
        year_basis="ACT/365",
        delta_convention="discounted forward delta",
    )
    oca_chain, quality = reconstruct_with_diagnostics(
        observations,
        requested_time,
        config=config,
        spread_floor=0.01,
        time_selection="previous",
        refit_parity=True,
    )
    chain, provenance = diagnostic_chain_from_oca(
        oca_chain,
        value_time=requested_time,
        days_map={"1w": 7, "2w": 13, "1m": 30},
        delta_bounds=(None, None),
        price_conversion="usd_per_unit",
        price_unit="USD per unit",
        quality_report=quality,
    )
    ivols, prices, parameters = [], [], []
    for i, ttm in enumerate(chain.ttms):
        k = np.log(chain.strikes_ttms[i] / chain.forwards[i])
        params = fitters.fit_logsv_ivols(k, 0.5 * (chain.bid_ivs[i] + chain.ask_ivs[i]), ttm)
        parameters.append(params)
        iv = fitters.calc_logsv_ivols(k, **params)
        ivols.append(iv)
        prices.append(
            np.array(
                [
                    bsm.compute_bsm_vanilla_price(
                        forward=chain.forwards[i],
                        strike=strike,
                        ttm=ttm,
                        vol=vol,
                        optiontype=cp,
                        discfactor=chain.discfactors[i],
                    )
                    for strike, cp, vol in zip(chain.strikes_ttms[i], chain.optiontypes_ttms[i], iv)
                ]
            )
        )
    report = fitters.diagnose_fit(
        chain,
        model_prices=prices,
        model_ivols=ivols,
        price_unit="USD per unit",
        spread_floor=0.01,
        metadata={"oca": provenance.to_dict()},
    )
    bad_report = fitters.diagnose_fit(
        chain,
        model_prices=[p + 1 for p in prices],
        model_ivols=[v + 0.1 for v in ivols],
        price_unit="USD per unit",
        spread_floor=0.01,
    )

    def slice_iv(strikes, ttm):
        index = int(np.flatnonzero(np.isclose(chain.ttms, ttm, rtol=0, atol=1e-12))[0])
        return fitters.calc_logsv_ivols(
            np.log(strikes / chain.forwards[index]), **parameters[index]
        )

    surface = fitters.SurfaceEvaluator(
        iv=slice_iv,
        forward=lambda t: float(np.interp(t, chain.ttms, chain.forwards)),
        discount=lambda t: float(np.interp(t, chain.ttms, chain.discfactors)),
        ttm_domain=(float(chain.ttms[0]), float(chain.ttms[-1])),
        k_domain=(-0.15, 0.15),
        underlying_id="SYNTH",
        carry_model="proportional",
        interpolation="none",
    )
    surface_report = fitters.diagnose_surface(
        surface,
        chain.ttms,
        np.linspace(-0.15, 0.15, 31),
        price_unit="USD per unit",
        metadata={"oca_source_time": quality.metadata["source_time"]},
    )
    assert provenance.table("counts").svm_reconciled.all()
    assert len(provenance.table("source_selection")) == 31
    assert report.table("summary").iv_rmse.iloc[0] < 0.005
    assert bad_report.table("summary").price_inside_bid_ask_fraction.iloc[0] == 0
    assert quality.table("summary").snapshot_age_status.iloc[0] == "fail"
    return chain, quality, provenance, report, bad_report, surface_report


class Locals(Enum):
    """Ordinary and non-unit quotation examples."""

    REPORT = 1
    SCALED = 2


def run_local(local: Locals):
    """Print a compact report from the reproducible synthetic workflow."""
    if local not in (Locals.REPORT, Locals.SCALED):
        raise ValueError(local)
    result = integrated_report(100.0 if local == Locals.SCALED else 1.0)
    print(result[2].table("counts").to_string(index=False))
    print(result[3].table("summary").to_string(index=False))
    print(result[5].table("checks").to_string(index=False))
    return result


if __name__ == "__main__":
    run_local(local=Locals.REPORT)
