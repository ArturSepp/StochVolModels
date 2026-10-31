"""Calibration must reject numerical trials without hiding unusable final fits."""
from types import SimpleNamespace

import numpy as np
import pytest

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers import logsv_pricer as module
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.model_pricer import CalibrationError


def _inputs(monkeypatch, valuation):
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([.25]), ids=np.array(['3m']), forwards=np.array([1.]),
        strikes=np.array([.9, 1., 1.1]), flat_vol=.2,
    )
    pricer = module.LogSVPricer()
    monkeypatch.setattr(pricer, 'compute_model_ivols_for_chain', valuation)
    params = LogSvParams(sigma0=.2, theta=.22, kappa1=3., kappa2=12., beta=-.3, volvol=.4)
    return chain, pricer, params


def _calibrate(chain, pricer, params):
    return pricer.calibrate_model_params_to_chain(
        chain, params0=params, is_vega_weighted=False,
        model_calibration_type=module.LogsvModelCalibrationType.PARAMS4,
    )


def test_invalid_intermediate_candidate_does_not_abort_and_is_reported(monkeypatch):
    calls = []

    def valuation(**kwargs):
        sigma = kwargs['params'].sigma0
        calls.append(sigma)
        if sigma > .3:
            raise FloatingPointError('forced incomplete coefficient integration')
        return [np.full(3, .21)]

    chain, pricer, params = _inputs(monkeypatch, valuation)

    def minimize(fun, x0, **kwargs):
        bad = x0.copy()
        bad[0] = .4
        penalty = fun(bad, None)
        valid = fun(x0, None)
        assert np.isfinite(penalty) and penalty > valid
        assert valid == pytest.approx(3e-4)
        return SimpleNamespace(x=x0, success=True, message='converged')

    monkeypatch.setattr(module, 'minimize', minimize)
    with pytest.warns(RuntimeWarning, match='rejected 1 of 3'):
        fitted = _calibrate(chain, pricer, params)
    assert fitted.sigma0 == params.sigma0
    assert calls == [.4, .2, .2]  # The returned parameters were independently repriced.


@pytest.mark.parametrize('failure', ['ode', 'nan', 'partial_nan', 'infinity'])
def test_invalid_final_candidate_cannot_be_reported_as_success(monkeypatch, failure):
    def valuation(**kwargs):
        if failure == 'ode':
            raise FloatingPointError('forced integration failure')
        values = {'nan': [np.nan]*3, 'partial_nan': [.2, np.nan, .2],
                  'infinity': [.2, np.inf, .2]}
        return [np.array(values[failure])]

    chain, pricer, params = _inputs(monkeypatch, valuation)

    def minimize(fun, x0, **kwargs):
        assert np.isfinite(fun(x0, None))
        return SimpleNamespace(x=x0, success=True, message='false success')

    monkeypatch.setattr(module, 'minimize', minimize)
    with pytest.warns(RuntimeWarning, match='rejected 2 of 2'):
        with pytest.raises(CalibrationError, match='final.*numerical'):
            _calibrate(chain, pricer, params)


def test_non_numerical_configuration_error_is_not_swallowed(monkeypatch):
    def valuation(**kwargs):
        raise ValueError('bad input shape')

    chain, pricer, params = _inputs(monkeypatch, valuation)
    monkeypatch.setattr(module, 'minimize', lambda fun, x0, **kwargs: fun(x0, None))
    with pytest.raises(ValueError, match='bad input shape'):
        _calibrate(chain, pricer, params)


def test_missing_market_quote_does_not_reject_valid_remaining_quotes(monkeypatch):
    chain, pricer, params = _inputs(monkeypatch, lambda **kwargs: [np.array([.21, np.nan, .21])])
    chain.bid_ivs[0][1] = np.nan
    chain.ask_ivs[0][1] = np.nan
    # Keep the scaler independent of the deliberately missing ATM quote.
    monkeypatch.setattr(pricer, 'set_vol_scaler', lambda **kwargs: .1)

    def minimize(fun, x0, **kwargs):
        assert fun(x0, None) == pytest.approx(2e-4)
        return SimpleNamespace(x=x0, success=True, message='converged')

    monkeypatch.setattr(module, 'minimize', minimize)
    assert _calibrate(chain, pricer, params).sigma0 == params.sigma0


def test_no_usable_market_quotes_is_an_error(monkeypatch):
    chain, pricer, params = _inputs(monkeypatch, lambda **kwargs: [np.full(3, .2)])
    chain.bid_ivs[0][:] = np.nan
    chain.ask_ivs[0][:] = np.nan
    monkeypatch.setattr(pricer, 'set_vol_scaler', lambda **kwargs: .1)
    monkeypatch.setattr(module, 'minimize', lambda fun, x0, **kwargs: fun(x0, None))
    with pytest.raises(CalibrationError, match='no finite nonzero-weight'):
        _calibrate(chain, pricer, params)


def test_zero_weight_quotes_are_omitted_and_vega_weights_are_preserved(monkeypatch):
    chain, pricer, params = _inputs(monkeypatch, lambda **kwargs: [np.array([.21, np.nan, .23])])
    monkeypatch.setattr(chain, 'get_chain_vegas', lambda **kwargs: [np.array([1., 0., 3.])])

    def minimize(fun, x0, **kwargs):
        assert fun(x0, None) == pytest.approx(.25*.01**2 + .75*.03**2)
        return SimpleNamespace(x=x0, success=True, message='converged')

    monkeypatch.setattr(module, 'minimize', minimize)
    fitted = pricer.calibrate_model_params_to_chain(
        chain, params0=params,
        model_calibration_type=module.LogsvModelCalibrationType.PARAMS4,
    )
    assert fitted.sigma0 == params.sigma0
