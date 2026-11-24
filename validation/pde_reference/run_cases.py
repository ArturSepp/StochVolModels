"""Run historical local-volatility and LSV comparisons from a source checkout."""

import argparse
from enum import Enum
import json

import numpy as np

from validation.pde_reference.cases import _pde_call_surface
from validation.pde_reference.dupire import DupireRecoverer
from validation.pde_reference.stress import (
    run_difficult_btc_initialization_stress,
    run_production_lsv_stress_matrix,
    run_realistic_btc_calibration_benchmark,
)


class Locals(str, Enum):
    """Historical comparisons selectable from the command line."""

    LOCAL_VOL = "local-vol"
    BTC_BASELINE = "btc-baseline"
    BTC_ADVERSE = "btc-adverse"
    STRESS = "stress"


def local_vol_recovery() -> dict:
    """Recover the known local variance from synthetic calls near the forward."""
    calls, surface = _pde_call_surface(spatial_size=201, time_steps=200)
    result = DupireRecoverer().continuous(calls)
    # Strike curvature is well resolved near the forward and away from the time edges.
    rows = (calls.times >= 0.15) & (calls.times <= 0.4)
    columns = (calls.strikes >= 0.95) & (calls.strikes <= 1.05)
    interior = np.ix_(rows, columns)
    truth = np.vstack(
        [surface.variance(t, np.log(calls.strikes[columns])) for t in calls.times[rows]]
    )
    error = float(np.max(np.abs(result.local_variance[interior] - truth)))
    assert result.valid_mask[interior].all()
    np.testing.assert_allclose(result.local_variance[interior], truth, rtol=0, atol=0.0025)
    return {"checked_nodes": int(truth.size), "max_annual_variance_error": error}


def run_local(local: Locals) -> None:
    """Report each actual decision and fail if it differs from the archived expectation."""
    if local == Locals.LOCAL_VOL:
        print(json.dumps(local_vol_recovery(), allow_nan=False))
    elif local == Locals.STRESS:
        results = run_production_lsv_stress_matrix()
        for result in results:
            print(
                json.dumps(
                    {
                        "scenario": result.scenario.name,
                        "accepted": result.passed,
                        "failed_checks": result.failed_checks,
                        "max_price_error": result.max_price_error,
                    },
                    allow_nan=False,
                )
            )
        if not all(result.expectation_satisfied for result in results):
            raise RuntimeError("Historical stress outcomes differ from the expected decisions")
    elif local in (Locals.BTC_BASELINE, Locals.BTC_ADVERSE):
        result = (
            run_realistic_btc_calibration_benchmark()
            if local == Locals.BTC_BASELINE
            else run_difficult_btc_initialization_stress()
        )
        print(
            json.dumps(
                {
                    "case": local.value,
                    **result.audit_summary(),
                    "failed_checks": result.pipeline_result.acceptance.failed_checks,
                },
                allow_nan=False,
            )
        )
        if not result.expectation_satisfied:
            raise RuntimeError(f"Unexpected acceptance decision for {local.value}")
    else:
        raise ValueError(local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=[case.value for case in Locals], default=Locals.LOCAL_VOL.value
    )
    run_local(Locals(parser.parse_args().case))
