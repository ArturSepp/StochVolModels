"""Time one synthetic snapshot and its 1, 12 or 48 live BRC marks in a fresh process."""

import argparse
from dataclasses import replace
from time import perf_counter
from unittest.mock import patch

import numpy as np

import brc_local_vol as example
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pricers import barrier_reverse_convertible as pricing
from stochvolmodels.products.barrier_reverse_convertible import (
    BrcState,
    BrcTerms,
    Monitoring,
    TouchState,
)


def benchmark(count):
    """Count numerical work at every resolution, including any tolerance retries."""
    start = perf_counter()
    with patch.object(example, "fit_sparse_local_vol", wraps=example.fit_sparse_local_vol) as fit:
        model = example.synthetic_market()
    calibrated = perf_counter()
    state = BrcState(asof=model.asof, touch=TouchState.CLEAR)
    config = pricing.BrcConfig(mc=LSVMonteCarloConfig(paths=8192, steps_per_year=252, seed=481))
    requests = []
    for i in range(count):
        vintage, tier = divmod(i, 4)
        expiry = 1.0 - vintage / 12.0
        original = model.spots * np.exp(vintage * 0.003)
        terms = BrcTerms(
            asset_ids=model.asset_ids,
            fixings=original,
            barriers=original * (0.6, 0.7, 0.8, 0.7)[tier],
            currency="USD",
            fixing_time=expiry,
            payment_time=expiry + 0.01,
            face=100.0,
        )
        if tier == 3:
            terms = replace(terms, monitoring=Monitoring.MATURITY)
        requests.append(
            pricing.BrcRequest(
                terms=terms,
                state=state,
                coupon=0.09,
                discount_factor=float(np.exp(-0.035 * terms.payment_time)),
                discount_currency="USD",
            )
        )
    counts = {}
    simulate = pricing.simulate_block

    def measured(*args, **kwargs):
        paths = simulate(*args, **kwargs)
        level = kwargs["refinement"]
        blocks, steps = counts.get(level, (0, 0))
        n, t, d = paths.assets.shape
        counts[level] = (blocks + 1, steps + n * (t - 1) * d)
        return paths

    with patch.object(pricing, "simulate_block", measured):
        values = pricing.value_brc_batch(requests, model, config=config)
    assert fit.call_count == 3 and counts[0][0] == 1 and all(v.status == "ok" for v in values)
    print(
        f"notes={count}, fits={fit.call_count}, fit_seconds={calibrated - start:.2f}, "
        f"total_seconds={perf_counter() - start:.2f}, (blocks,asset_steps)_by_level={counts}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", choices=(1, 12, 48), type=int)
    benchmark(parser.parse_args().notes)
