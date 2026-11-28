"""Reprice all stored constrained IVs and check the auxiliary forward before fitting.

Grid errors in IV are absolute decimals. Forward discrepancies are basis
points. The regression uses the stated USO test maturity in years.
"""

from dataclasses import replace

import numpy as np
from stochvolmodels import LogSvParams

from ..spx_physical_pricing_20261124.pricing import FINE, MEDIUM
from .inputs import OUT, load, save, select
from .pricer import evaluate_iv, forward_ratio


def scoring_grid(case):
    return replace(MEDIUM, time_steps=128) if case.quarter else FINE


def steps(t):
    return max(512, int(np.ceil(2048 * t)))


def defects(params, horizons):
    return [1e4 * (1 - forward_ratio(params, float(t), time_steps=steps(t))) for t in horizons]


def main():
    cases, counts = select()
    controls = {}
    for case in cases:
        (_, chain, _, _), fit = load(case)
        params = LogSvParams(**fit["parameters"])
        iv = evaluate_iv(chain, params, scoring_grid(case))
        np.testing.assert_allclose(iv, fit["fine_iv"], rtol=0, atol=2e-6)
        diff = max(abs(iv - np.array(fit["fine_iv"])))
        forward = defects(params, [*chain.ttms, 1.0])
        assert max(map(abs, forward)) <= 0.001, (case, forward)
        controls[case.key] = dict(
            max_abs_iv=float(diff), defect_bp=forward, quote_sha256=fit["quote_sha256"]
        )
        print(case.key, "K1", diff, "K2 (bp)", max(map(abs, forward)), flush=True)
    _, uso = load(cases[0])
    test = LogSvParams(**{**uso["parameters"], "kappa2": 0.05})
    widths = [6, 8, 10, 12]
    prototype = [11.265, 10.956, 10.916, 10.910]
    obtained = [
        1e4 * (1 - forward_ratio(test, 0.254636, h, n, 512))
        for h, n in zip(widths, [801, 1069, 1335, 1601])
    ]
    np.testing.assert_allclose(obtained, prototype, atol=0.005, rtol=0)
    save(
        OUT / "controls.json",
        dict(
            selection=counts,
            constrained=controls,
            prototype=dict(widths=widths, expected=prototype, obtained=obtained),
        ),
    )
    print("K3", obtained, flush=True)


if __name__ == "__main__":
    main()
