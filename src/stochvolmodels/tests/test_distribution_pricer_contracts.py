"""Independent contracts for the stable Gaussian-mixture and Student-t pricers."""

from types import SimpleNamespace

import numpy as np
import pytest
from numba.typed import List

import stochvolmodels.pricers.gmm_pricer as gmm_module
import stochvolmodels.pricers.tdist_pricer as tdist_module
from stochvolmodels import (
    CalibrationError,
    GmmParams,
    GmmPricer,
    OptionChain,
    TdistParams,
    TdistPricer,
    compute_bsm_vanilla_price,
)
from stochvolmodels.fitters.tdist import imply_drift_tdist


def _quoted_slice() -> OptionChain:
    """Return a small deterministic one-maturity chain suitable for calibration tests."""
    return OptionChain(
        ttms=np.array([0.5]),
        forwards=np.array([1.0]),
        strikes_ttms=List([np.array([0.9, 1.0, 1.1])]),
        optiontypes_ttms=List([np.array(["P", "C", "C"])]),
        ids=np.array(["6m"]),
        discount_rates=np.array([0.02]),
        bid_ivs=List([np.array([0.24, 0.22, 0.23])]),
        ask_ivs=List([np.array([0.26, 0.24, 0.25])]),
    )


def test_one_state_gmm_reduces_to_black_scholes() -> None:
    """A martingale one-state Gaussian mixture is exactly Black--Scholes."""
    vol = 0.35
    ttm = 0.75
    forward = 1.05
    strike = 1.1
    discfactor = 0.97
    params = GmmParams(
        gmm_weights=np.array([1.0]),
        gmm_mus=np.array([-0.5 * vol**2]),
        gmm_vols=np.array([vol]),
        ttm=ttm,
    )

    actual, _ = GmmPricer().price_vanilla(
        params=params,
        ttm=ttm,
        forward=forward,
        strike=strike,
        optiontype="C",
        discfactor=discfactor,
    )
    expected = compute_bsm_vanilla_price(
        forward=forward,
        strike=strike,
        ttm=ttm,
        vol=vol,
        optiontype="C",
        discfactor=discfactor,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)


def test_student_t_prices_satisfy_discounted_put_call_parity() -> None:
    """Student-t call and put prices preserve the forward martingale identity."""
    ttm = 0.5
    forward = 1.03
    strike = 1.0
    discfactor = 0.98
    rf_rate = -np.log(discfactor) / ttm
    params = TdistParams(
        drift=imply_drift_tdist(rf_rate=rf_rate, vol=0.3, nu=5.0, ttm=ttm),
        vol=0.3,
        nu=5.0,
        ttm=ttm,
    )
    pricer = TdistPricer()

    call, _ = pricer.price_vanilla(
        params=params,
        ttm=ttm,
        forward=forward,
        strike=strike,
        optiontype="C",
        discfactor=discfactor,
    )
    put, _ = pricer.price_vanilla(
        params=params,
        ttm=ttm,
        forward=forward,
        strike=strike,
        optiontype="P",
        discfactor=discfactor,
    )

    np.testing.assert_allclose(
        call - put,
        discfactor * (forward - strike),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("module", "pricer", "kwargs", "size"),
    [
        (gmm_module, GmmPricer(), {"n_mixtures": 2}, 6),
        (tdist_module, TdistPricer(), {}, 2),
    ],
)
def test_distribution_calibrators_reject_failed_optimizer_results(
    monkeypatch: pytest.MonkeyPatch,
    module,
    pricer,
    kwargs: dict,
    size: int,
) -> None:
    """A failed optimizer must not be converted silently into public model parameters."""

    def failed_minimize(*args, **optimizer_kwargs):
        return SimpleNamespace(
            success=False,
            message="forced optimizer failure",
            x=np.zeros(size),
        )

    monkeypatch.setattr(module, "minimize", failed_minimize)

    with pytest.raises(CalibrationError, match="forced optimizer failure"):
        pricer.calibrate_model_params_to_chain_slice(
            option_chain=_quoted_slice(),
            is_vega_weighted=False,
            **kwargs,
        )
