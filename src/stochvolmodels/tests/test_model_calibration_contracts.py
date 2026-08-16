from types import SimpleNamespace

import numpy as np
import pytest

import stochvolmodels.pricers.heston_pricer as heston_module
import stochvolmodels.pricers.logsv_pricer as logsv_module
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import (
    CalibrationEngine,
    ConstraintsType,
    LogsvModelCalibrationType,
    LogSVPricer,
)
from stochvolmodels.pricers.model_pricer import (
    CalibrationError,
    validate_optimization_result,
)


def _calibration_chain() -> OptionChain:
    return OptionChain.get_uniform_chain(
        ttms=np.array([0.25]),
        ids=np.array(["3m"]),
        forwards=np.array([1.0]),
        strikes=np.array([0.9, 1.0, 1.1]),
        flat_vol=0.2,
    )


def _constraint_items(constraints):
    if constraints is None:
        return ()
    if isinstance(constraints, dict):
        return (constraints,)
    return constraints


def test_heston_calibration_builds_weighted_objective_and_feller_constraint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chain = _calibration_chain()
    params0 = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    captured = {}
    pricer = HestonPricer()
    monkeypatch.setattr(
        pricer,
        "compute_model_ivols_for_chain",
        lambda **kwargs: [chain.get_mid_vols()[0] + 0.01],
    )

    def fake_minimize(fun, x0, args, method, constraints, bounds, options):
        captured["objective"] = fun(np.asarray(x0), args)
        captured["constraint_values"] = [
            item["fun"](np.asarray(x0)) for item in _constraint_items(constraints)
        ]
        captured["method"] = method
        captured["bounds"] = bounds
        captured["options"] = options
        return SimpleNamespace(x=np.asarray(x0), success=True, message="converged")

    monkeypatch.setattr(heston_module, "minimize", fake_minimize)
    fitted = pricer.calibrate_model_params_to_chain(
        chain, params0=params0, is_vega_weighted=False
    )

    np.testing.assert_allclose(captured["objective"], 3.0e-4, atol=1.0e-15)
    assert captured["constraint_values"] == pytest.approx([0.04])
    assert captured["method"] == "SLSQP"
    assert len(captured["bounds"]) == 5
    assert captured["options"]["ftol"] == 1.0e-8
    assert fitted == params0
    assert capsys.readouterr().out == ""


def test_logsv_calibration_builds_objective_and_requested_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params0 = LogSvParams(
        sigma0=0.2,
        theta=0.22,
        kappa1=3.0,
        kappa2=12.0,
        beta=-0.3,
        volvol=0.4,
    )
    captured = {}
    pricer = LogSVPricer()
    monkeypatch.setattr(
        pricer,
        "compute_model_ivols_for_chain",
        lambda **kwargs: [chain.get_mid_vols()[0] + 0.01],
    )

    def fake_minimize(fun, x0, args, method, constraints, bounds, options):
        captured["objective"] = fun(np.asarray(x0), args)
        captured["constraint_values"] = [
            item["fun"](np.asarray(x0)) for item in _constraint_items(constraints)
        ]
        captured["method"] = method
        captured["bounds"] = bounds
        return SimpleNamespace(x=np.asarray(x0), success=True, message="converged")

    monkeypatch.setattr(logsv_module, "minimize", fake_minimize)
    fitted = pricer.calibrate_model_params_to_chain(
        chain,
        params0=params0,
        is_vega_weighted=False,
        model_calibration_type=LogsvModelCalibrationType.PARAMS4,
        constraints_type=ConstraintsType.INVERSE_MARTINGALE_MOMENT4,
        calibration_engine=CalibrationEngine.ANALYTIC,
    )

    np.testing.assert_allclose(captured["objective"], 3.0e-4, atol=1.0e-15)
    assert captured["constraint_values"] == pytest.approx([12.6, 5.265])
    assert captured["method"] == "SLSQP"
    assert len(captured["bounds"]) == 4
    assert fitted == params0


def test_logsv_parameter_codec_preserves_all_supported_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params0 = LogSvParams(
        sigma0=0.2,
        theta=0.25,
        kappa1=3.0,
        kappa2=4.0,
        beta=-0.3,
        volvol=0.5,
        H=0.45,
        nodes=np.array([0.1, 0.2]),
        weights=np.array([0.6, 0.4]),
    )
    params_min = LogSvParams(
        sigma0=0.1,
        theta=0.1,
        kappa1=0.25,
        kappa2=0.25,
        beta=-3.0,
        volvol=0.2,
    )
    params_max = LogSvParams(
        sigma0=1.5,
        theta=1.5,
        kappa1=10.0,
        kappa2=10.0,
        beta=3.0,
        volvol=3.0,
    )

    codec4 = logsv_module._LogSvParameterCodec(
        params0,
        params_min,
        params_max,
        LogsvModelCalibrationType.PARAMS4,
        None,
    )
    p4, bounds4 = codec4.initial_and_bounds()
    parsed4 = codec4.parse(p4)
    np.testing.assert_allclose(p4, np.array([0.2, 0.25, -0.3, 0.5]))
    assert len(bounds4) == 4
    assert parsed4.kappa1 == params0.kappa1
    assert parsed4.kappa2 == params0.kappa2
    assert parsed4.nodes is params0.nodes
    assert parsed4.weights is params0.weights

    codec5 = logsv_module._LogSvParameterCodec(
        params0,
        params_min,
        params_max,
        LogsvModelCalibrationType.PARAMS5,
        None,
    )
    p5, bounds5 = codec5.initial_and_bounds()
    parsed5 = codec5.parse(p5)
    np.testing.assert_allclose(p5, np.array([0.2, 0.25, 3.0, -0.3, 0.5]))
    assert len(bounds5) == 5
    assert parsed5.kappa2 == pytest.approx(parsed5.kappa1 / parsed5.theta)

    monkeypatch.setattr(
        logsv_module,
        "fit_model_vol_backbone_to_varswaps",
        lambda **kwargs: "backbone",
    )
    codec_varswap = logsv_module._LogSvParameterCodec(
        params0,
        params_min,
        params_max,
        LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT,
        np.array([0.04]),
    )
    p_varswap, bounds_varswap = codec_varswap.initial_and_bounds()
    parsed_varswap = codec_varswap.parse(p_varswap)
    np.testing.assert_allclose(p_varswap, np.array([-0.3, 0.5]))
    assert len(bounds_varswap) == 2
    assert parsed_varswap.vol_backbone == "backbone"


def test_logsv_calibration_components_preserve_weights_and_constraints() -> None:
    chain = _calibration_chain()
    market_vols = np.concatenate(chain.get_mid_vols())
    unweighted = logsv_module._compute_logsv_calibration_weights(
        chain,
        market_vols,
        is_vega_weighted=False,
        is_unit_ttm_vega=False,
    )
    weighted = logsv_module._compute_logsv_calibration_weights(
        chain,
        market_vols,
        is_vega_weighted=True,
        is_unit_ttm_vega=False,
    )
    np.testing.assert_array_equal(unweighted, np.ones_like(market_vols))
    assert np.all(weighted > 0.0)
    np.testing.assert_allclose(np.sum(weighted), 1.0, atol=1.0e-15)

    params0 = LogSvParams(theta=0.22, kappa1=3.0, kappa2=12.0, beta=-0.3, volvol=0.4)
    codec = logsv_module._LogSvParameterCodec(
        params0,
        LogSvParams(),
        LogSvParams(sigma0=2.0, theta=2.0, kappa1=20.0, kappa2=20.0, beta=3.0, volvol=3.0),
        LogsvModelCalibrationType.PARAMS4,
        None,
    )
    pars, _ = codec.initial_and_bounds()

    assert logsv_module._build_logsv_constraints(
        codec, ConstraintsType.UNCONSTRAINT
    ) is None
    mma = _constraint_items(
        logsv_module._build_logsv_constraints(codec, ConstraintsType.MMA_MARTINGALE)
    )
    inverse_moment = _constraint_items(
        logsv_module._build_logsv_constraints(
            codec, ConstraintsType.INVERSE_MARTINGALE_MOMENT4
        )
    )
    assert [item["fun"](pars) for item in mma] == pytest.approx([12.3])
    assert [item["fun"](pars) for item in inverse_moment] == pytest.approx(
        [12.6, 5.265]
    )


def test_heston_calibration_rejects_failed_optimizer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params0 = HestonParams()

    def failed_minimize(fun, x0, **kwargs):
        return SimpleNamespace(x=np.asarray(x0), success=False, message="iteration limit")

    monkeypatch.setattr(heston_module, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="iteration limit"):
        HestonPricer().calibrate_model_params_to_chain(chain, params0=params0)


def test_logsv_calibration_rejects_failed_optimizer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _calibration_chain()
    params0 = LogSvParams()

    def failed_minimize(fun, x0, **kwargs):
        return SimpleNamespace(x=np.asarray(x0), success=False, message="iteration limit")

    monkeypatch.setattr(logsv_module, "minimize", failed_minimize)
    with pytest.raises(RuntimeError, match="iteration limit"):
        LogSVPricer().calibrate_model_params_to_chain(chain, params0=params0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.array([np.nan]), "non-finite"),
        (np.array([-0.1]), "below bounds"),
        (np.array([1.1]), "above bounds"),
    ],
)
def test_optimizer_result_validation_rejects_invalid_success_vectors(
    values: np.ndarray, message: str
) -> None:
    result = SimpleNamespace(x=values, success=True, message="reported success")

    with pytest.raises(CalibrationError, match=message):
        validate_optimization_result(result, bounds=((0.0, 1.0),))
